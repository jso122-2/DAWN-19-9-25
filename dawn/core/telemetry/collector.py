#!/usr/bin/env python3
"""
DAWN Telemetry Collector
========================

Centralized telemetry collection and aggregation system for DAWN.
Collects telemetry from all subsystems and provides unified access
to telemetry data with real-time aggregation and analysis.
"""

import time
import threading
import logging
from typing import Dict, Any, List, Optional, Callable, Set
from collections import defaultdict, deque
from dataclasses import dataclass
from datetime import datetime, timedelta
import statistics
import uuid

from .logger import DAWNTelemetryLogger, TelemetryEvent, TelemetryLevel
from .config import TelemetryConfig

logger = logging.getLogger(__name__)

@dataclass
class SubsystemMetrics:
    """Aggregated metrics for a DAWN subsystem."""
    subsystem: str
    total_events: int
    events_by_level: Dict[str, int]
    events_by_component: Dict[str, int]
    events_by_type: Dict[str, int]
    avg_events_per_minute: float
    last_event_time: Optional[datetime]
    health_score: float

@dataclass
class TelemetryAggregation:
    """Aggregated telemetry data over a time window."""
    time_window_minutes: int
    total_events: int
    events_by_subsystem: Dict[str, SubsystemMetrics]
    events_by_level: Dict[str, int]
    top_event_types: List[tuple]  # (event_type, count)
    top_components: List[tuple]   # (component, count)
    error_rate: float
    performance_metrics: Dict[str, Any]
    system_health_score: float

class TelemetryCollector:
    """
    Centralized telemetry collector for DAWN consciousness system.
    
    Features:
    - Real-time telemetry aggregation
    - Subsystem health monitoring  
    - Performance analytics
    - Alert generation
    - Historical trend analysis
    """
    
    def __init__(self, config: TelemetryConfig, telemetry_logger: DAWNTelemetryLogger):
        """
        Initialize telemetry collector.
        
        Args:
            config: Telemetry configuration
            telemetry_logger: Main telemetry logger
        """
        self.config = config
        self.telemetry_logger = telemetry_logger
        self.collector_id = str(uuid.uuid4())
        self.start_time = time.time()
        
        # Collection state
        self.running = False
        self.collection_thread = None
        self.aggregation_interval = 60.0  # seconds
        
        # Data storage
        self.event_history = deque(maxlen=100000)  # Store raw events
        self.aggregations_history = deque(maxlen=1440)  # 24 hours of minute aggregations
        self.subsystem_registry = set()
        self.component_registry = set()
        
        # Real-time metrics
        self.current_aggregation = None
        self.last_aggregation_time = time.time()
        
        # Alert system
        self.alert_handlers = []
        self.alert_thresholds = {
            'error_rate_threshold': 0.1,  # 10% error rate
            'high_event_rate_threshold': 1000,  # events per minute
            'low_health_score_threshold': 0.7,
            'subsystem_silence_threshold': 300  # seconds without events
        }
        
        # Performance tracking
        self.collector_metrics = {
            'events_collected': 0,
            'aggregations_performed': 0,
            'alerts_generated': 0,
            'avg_processing_time_ms': 0.0
        }
        
        # Thread safety
        self.lock = threading.RLock()
        
        logger.info(f"📊 Telemetry Collector initialized: {self.collector_id[:8]}")
    
    def start(self) -> None:
        """Start telemetry collection."""
        if self.running:
            return
        
        self.running = True
        
        # Start collection thread
        self.collection_thread = threading.Thread(
            target=self._collection_loop,
            name="telemetry_collector",
            daemon=True
        )
        self.collection_thread.start()
        
        logger.info("📊 Telemetry collector started")
    
    def stop(self) -> None:
        """Stop telemetry collection."""
        if not self.running:
            return
        
        self.running = False
        
        if self.collection_thread and self.collection_thread.is_alive():
            self.collection_thread.join(timeout=2.0)
        
        # Perform final aggregation
        self._perform_aggregation()
        
        logger.info("📊 Telemetry collector stopped")
    
    def _collection_loop(self) -> None:
        """Main collection loop."""
        while self.running:
            try:
                # Collect events from telemetry logger
                self._collect_events()
                
                # Perform aggregation if needed
                if time.time() - self.last_aggregation_time >= self.aggregation_interval:
                    self._perform_aggregation()
                    self.last_aggregation_time = time.time()
                
                # Check for alerts
                self._check_alerts()
                
                time.sleep(1.0)  # Collection frequency
                
            except Exception as e:
                logger.error(f"Error in telemetry collection loop: {e}")
                time.sleep(5.0)
    
    def _collect_events(self) -> None:
        """Collect new events from telemetry logger."""
        try:
            # Get recent events from logger
            recent_events = self.telemetry_logger.get_recent_events(1000)
            
            # Filter for new events (simple approach - could be optimized)
            new_events = []
            if recent_events:
                # Convert to TelemetryEvent objects for processing
                for event_dict in recent_events:
                    try:
                        event = TelemetryEvent(**event_dict)
                        
                        # Check if this is a new event (not in our history)
                        if not self._is_duplicate_event(event):
                            new_events.append(event)
                            self.event_history.append(event)
                            
                            # Register subsystem and component
                            self.subsystem_registry.add(event.subsystem)
                            self.component_registry.add(f"{event.subsystem}.{event.component}")
                    except Exception as e:
                        logger.warning(f"Error processing event: {e}")
                        continue
            
            # Update collector metrics
            with self.lock:
                self.collector_metrics['events_collected'] += len(new_events)
            
        except Exception as e:
            logger.error(f"Error collecting events: {e}")
    
    def _is_duplicate_event(self, event: TelemetryEvent) -> bool:
        """Check if event is already in our history (simple deduplication)."""
        # Simple approach - check last few events for exact timestamp match
        for hist_event in list(self.event_history)[-10:]:
            if (hist_event.timestamp == event.timestamp and 
                hist_event.subsystem == event.subsystem and
                hist_event.component == event.component and
                hist_event.event_type == event.event_type):
                return True
        return False
    
    def _perform_aggregation(self) -> None:
        """Perform telemetry data aggregation."""
        start_time = time.perf_counter()
        
        try:
            # Get events from the last aggregation window
            cutoff_time = datetime.now() - timedelta(minutes=1)
            recent_events = [
                event for event in self.event_history
                if self._parse_timestamp(event.timestamp) >= cutoff_time
            ]
            
            if not recent_events:
                return
            
            # Create aggregation
            aggregation = self._create_aggregation(recent_events, 1)
            
            # Store aggregation
            self.aggregations_history.append(aggregation)
            self.current_aggregation = aggregation
            
            # Update collector metrics
            processing_time_ms = (time.perf_counter() - start_time) * 1000
            with self.lock:
                self.collector_metrics['aggregations_performed'] += 1
                # Update rolling average
                current_avg = self.collector_metrics['avg_processing_time_ms']
                count = self.collector_metrics['aggregations_performed']
                self.collector_metrics['avg_processing_time_ms'] = (
                    (current_avg * (count - 1) + processing_time_ms) / count
                )
            
            logger.debug(f"Aggregated {len(recent_events)} events in {processing_time_ms:.2f}ms")
            
        except Exception as e:
            logger.error(f"Error performing aggregation: {e}")
    
    def _create_aggregation(self, events: List[TelemetryEvent], window_minutes: int) -> TelemetryAggregation:
        """Create aggregation from events."""
        # Initialize counters
        total_events = len(events)
        events_by_level = defaultdict(int)
        events_by_subsystem = defaultdict(lambda: {
            'total': 0,
            'by_level': defaultdict(int),
            'by_component': defaultdict(int),
            'by_type': defaultdict(int),
            'last_event': None
        })
        events_by_type = defaultdict(int)
        events_by_component = defaultdict(int)
        error_count = 0
        performance_metrics = {
            'total_duration_ms': 0,
            'operation_count': 0,
            'success_rate': 0.0
        }
        
        # Process events
        for event in events:
            # Level counts
            events_by_level[event.level] += 1
            
            # Subsystem stats
            subsystem_stats = events_by_subsystem[event.subsystem]
            subsystem_stats['total'] += 1
            subsystem_stats['by_level'][event.level] += 1
            subsystem_stats['by_component'][event.component] += 1
            subsystem_stats['by_type'][event.event_type] += 1
            subsystem_stats['last_event'] = event.timestamp
            
            # Event type and component counts
            events_by_type[event.event_type] += 1
            events_by_component[f"{event.subsystem}.{event.component}"] += 1
            
            # Error tracking
            if event.level in ['ERROR', 'CRITICAL']:
                error_count += 1
            
            # Performance metrics
            if event.event_type == 'performance_metric' and 'duration_ms' in event.data:
                performance_metrics['total_duration_ms'] += event.data.get('duration_ms', 0)
                performance_metrics['operation_count'] += 1
                if event.data.get('success', True):
                    performance_metrics['success_rate'] += 1
        
        # Calculate derived metrics
        error_rate = error_count / total_events if total_events > 0 else 0.0
        
        if performance_metrics['operation_count'] > 0:
            performance_metrics['avg_duration_ms'] = (
                performance_metrics['total_duration_ms'] / performance_metrics['operation_count']
            )
            performance_metrics['success_rate'] = (
                performance_metrics['success_rate'] / performance_metrics['operation_count']
            )
        
        # Create subsystem metrics
        subsystem_metrics = {}
        for subsystem, stats in events_by_subsystem.items():
            avg_events_per_minute = stats['total'] / window_minutes
            
            # Calculate health score based on error rate and activity
            subsystem_error_rate = stats['by_level'].get('ERROR', 0) + stats['by_level'].get('CRITICAL', 0)
            subsystem_error_rate = subsystem_error_rate / stats['total'] if stats['total'] > 0 else 0
            
            health_score = max(0.0, 1.0 - (subsystem_error_rate * 2))  # Errors heavily impact health
            
            subsystem_metrics[subsystem] = SubsystemMetrics(
                subsystem=subsystem,
                total_events=stats['total'],
                events_by_level=dict(stats['by_level']),
                events_by_component=dict(stats['by_component']),
                events_by_type=dict(stats['by_type']),
                avg_events_per_minute=avg_events_per_minute,
                last_event_time=self._parse_timestamp(stats['last_event']) if stats['last_event'] else None,
                health_score=health_score
            )
        
        # Calculate system health score
        if subsystem_metrics:
            system_health_score = statistics.mean([sm.health_score for sm in subsystem_metrics.values()])
        else:
            system_health_score = 1.0
        
        # Get top event types and components
        top_event_types = sorted(events_by_type.items(), key=lambda x: x[1], reverse=True)[:10]
        top_components = sorted(events_by_component.items(), key=lambda x: x[1], reverse=True)[:10]
        
        return TelemetryAggregation(
            time_window_minutes=window_minutes,
            total_events=total_events,
            events_by_subsystem=subsystem_metrics,
            events_by_level=dict(events_by_level),
            top_event_types=top_event_types,
            top_components=top_components,
            error_rate=error_rate,
            performance_metrics=performance_metrics,
            system_health_score=system_health_score
        )
    
    def _parse_timestamp(self, timestamp_str: str) -> datetime:
        """Parse timestamp string to datetime object."""
        try:
            # Handle ISO format with timezone
            if timestamp_str.endswith('Z'):
                return datetime.fromisoformat(timestamp_str[:-1] + '+00:00')
            return datetime.fromisoformat(timestamp_str)
        except:
            # Fallback to current time
            return datetime.now()
    
    def _check_alerts(self) -> None:
        """Check for alert conditions."""
        if not self.current_aggregation:
            return
        
        alerts_generated = 0
        
        try:
            # Check error rate
            if self.current_aggregation.error_rate > self.alert_thresholds['error_rate_threshold']:
                self._generate_alert(
                    'high_error_rate',
                    f"High error rate detected: {self.current_aggregation.error_rate:.1%}",
                    {'error_rate': self.current_aggregation.error_rate}
                )
                alerts_generated += 1
            
            # Check event rate
            if self.current_aggregation.total_events > self.alert_thresholds['high_event_rate_threshold']:
                self._generate_alert(
                    'high_event_rate',
                    f"High event rate detected: {self.current_aggregation.total_events} events/min",
                    {'event_rate': self.current_aggregation.total_events}
                )
                alerts_generated += 1
            
            # Check system health
            if self.current_aggregation.system_health_score < self.alert_thresholds['low_health_score_threshold']:
                self._generate_alert(
                    'low_system_health',
                    f"Low system health score: {self.current_aggregation.system_health_score:.2f}",
                    {'health_score': self.current_aggregation.system_health_score}
                )
                alerts_generated += 1
            
            # Check subsystem silence
            current_time = datetime.now()
            silence_threshold = timedelta(seconds=self.alert_thresholds['subsystem_silence_threshold'])
            
            for subsystem, metrics in self.current_aggregation.events_by_subsystem.items():
                if metrics.last_event_time:
                    time_since_last = current_time - metrics.last_event_time
                    if time_since_last > silence_threshold:
                        self._generate_alert(
                            'subsystem_silence',
                            f"Subsystem {subsystem} has been silent for {time_since_last.total_seconds():.0f}s",
                            {'subsystem': subsystem, 'silence_duration': time_since_last.total_seconds()}
                        )
                        alerts_generated += 1
            
            # Update metrics
            if alerts_generated > 0:
                with self.lock:
                    self.collector_metrics['alerts_generated'] += alerts_generated
            
        except Exception as e:
            logger.error(f"Error checking alerts: {e}")
    
    def _generate_alert(self, alert_type: str, message: str, data: Dict[str, Any]) -> None:
        """Generate an alert."""
        alert = {
            'timestamp': datetime.now().isoformat(),
            'type': alert_type,
            'message': message,
            'data': data,
            'collector_id': self.collector_id
        }
        
        # Send to alert handlers
        for handler in self.alert_handlers:
            try:
                handler(alert)
            except Exception as e:
                logger.error(f"Error in alert handler: {e}")
        
        # Log the alert
        logger.warning(f"TELEMETRY ALERT [{alert_type}]: {message}")
    
    def add_alert_handler(self, handler: Callable[[Dict[str, Any]], None]) -> None:
        """Add an alert handler function."""
        self.alert_handlers.append(handler)
        logger.info(f"Added telemetry alert handler: {handler}")
    
    def remove_alert_handler(self, handler: Callable[[Dict[str, Any]], None]) -> bool:
        """Remove an alert handler."""
        if handler in self.alert_handlers:
            self.alert_handlers.remove(handler)
            logger.info(f"Removed telemetry alert handler: {handler}")
            return True
        return False
    
    def get_current_aggregation(self) -> Optional[TelemetryAggregation]:
        """Get the current telemetry aggregation."""
        return self.current_aggregation
    
    def get_historical_aggregations(self, hours: int = 1) -> List[TelemetryAggregation]:
        """Get historical aggregations for the specified number of hours."""
        max_aggregations = hours * 60  # 1 aggregation per minute
        return list(self.aggregations_history)[-max_aggregations:]
    
    def get_subsystem_health(self, subsystem: str = None) -> Dict[str, Any]:
        """Get health information for a specific subsystem or all subsystems."""
        if not self.current_aggregation:
            return {}
        
        if subsystem:
            if subsystem in self.current_aggregation.events_by_subsystem:
                metrics = self.current_aggregation.events_by_subsystem[subsystem]
                return {
                    'subsystem': subsystem,
                    'health_score': metrics.health_score,
                    'total_events': metrics.total_events,
                    'avg_events_per_minute': metrics.avg_events_per_minute,
                    'last_event_time': metrics.last_event_time.isoformat() if metrics.last_event_time else None,
                    'events_by_level': metrics.events_by_level
                }
            return {}
        
        # Return all subsystems
        health_data = {}
        for subsystem, metrics in self.current_aggregation.events_by_subsystem.items():
            health_data[subsystem] = {
                'health_score': metrics.health_score,
                'total_events': metrics.total_events,
                'avg_events_per_minute': metrics.avg_events_per_minute,
                'last_event_time': metrics.last_event_time.isoformat() if metrics.last_event_time else None,
                'events_by_level': metrics.events_by_level
            }
        
        return health_data
    
    def get_performance_summary(self) -> Dict[str, Any]:
        """Get performance summary from recent aggregations."""
        if not self.current_aggregation:
            return {}
        
        perf_metrics = self.current_aggregation.performance_metrics
        
        return {
            'avg_operation_duration_ms': perf_metrics.get('avg_duration_ms', 0.0),
            'operation_success_rate': perf_metrics.get('success_rate', 0.0),
            'total_operations': perf_metrics.get('operation_count', 0),
            'system_health_score': self.current_aggregation.system_health_score,
            'error_rate': self.current_aggregation.error_rate,
            'total_events_per_minute': self.current_aggregation.total_events
        }
    
    def get_collector_metrics(self) -> Dict[str, Any]:
        """Get telemetry collector metrics."""
        with self.lock:
            metrics = self.collector_metrics.copy()
        
        return {
            **metrics,
            'collector_id': self.collector_id,
            'uptime_seconds': time.time() - self.start_time,
            'running': self.running,
            'registered_subsystems': len(self.subsystem_registry),
            'registered_components': len(self.component_registry),
            'event_history_size': len(self.event_history),
            'aggregations_history_size': len(self.aggregations_history),
            'alert_handlers': len(self.alert_handlers)
        }
    
    def export_aggregation_data(self, hours: int = 1) -> Dict[str, Any]:
        """Export aggregation data for external analysis."""
        aggregations = self.get_historical_aggregations(hours)
        
        export_data = {
            'export_timestamp': datetime.now().isoformat(),
            'time_range_hours': hours,
            'total_aggregations': len(aggregations),
            'aggregations': []
        }
        
        for agg in aggregations:
            agg_data = {
                'time_window_minutes': agg.time_window_minutes,
                'total_events': agg.total_events,
                'events_by_level': agg.events_by_level,
                'error_rate': agg.error_rate,
                'system_health_score': agg.system_health_score,
                'top_event_types': agg.top_event_types[:5],  # Top 5
                'top_components': agg.top_components[:5],    # Top 5
                'performance_metrics': agg.performance_metrics,
                'subsystems': {}
            }
            
            # Add subsystem data
            for subsystem, metrics in agg.events_by_subsystem.items():
                agg_data['subsystems'][subsystem] = {
                    'total_events': metrics.total_events,
                    'health_score': metrics.health_score,
                    'avg_events_per_minute': metrics.avg_events_per_minute,
                    'events_by_level': metrics.events_by_level
                }
            
            export_data['aggregations'].append(agg_data)
        
        return export_data
