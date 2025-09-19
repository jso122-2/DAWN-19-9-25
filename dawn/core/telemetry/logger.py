#!/usr/bin/env python3
"""
DAWN Telemetry Logger
====================

Core telemetry logging system for DAWN consciousness system.
Provides structured, high-performance telemetry collection with configurable
output formats and filtering.
"""

import time
import json
import threading
import uuid
import logging
from datetime import datetime, timezone
from typing import Dict, Any, Optional, List, Union
from dataclasses import dataclass, asdict
from enum import Enum
from collections import deque
import os

# Optional psutil import for system metrics
try:
    import psutil
    PSUTIL_AVAILABLE = True
except ImportError:
    PSUTIL_AVAILABLE = False

logger = logging.getLogger(__name__)

class TelemetryLevel(Enum):
    """Telemetry logging levels."""
    DEBUG = "DEBUG"
    INFO = "INFO" 
    WARN = "WARN"
    ERROR = "ERROR"
    CRITICAL = "CRITICAL"

@dataclass
class TelemetryEvent:
    """Structured telemetry event."""
    timestamp: str
    level: str
    subsystem: str
    component: str
    event_type: str
    tick_id: Optional[int]
    session_id: str
    data: Dict[str, Any]
    metadata: Dict[str, Any]
    
    def to_dict(self) -> Dict[str, Any]:
        """Convert to dictionary for serialization."""
        return asdict(self)
    
    def to_json(self) -> str:
        """Convert to JSON string."""
        return json.dumps(self.to_dict(), separators=(',', ':'))

class TelemetryBuffer:
    """Thread-safe circular buffer for telemetry events."""
    
    def __init__(self, max_size: int = 10000):
        self.max_size = max_size
        self.buffer = deque(maxlen=max_size)
        self.lock = threading.RLock()
        self.total_events = 0
        self.dropped_events = 0
    
    def add_event(self, event: TelemetryEvent) -> None:
        """Add an event to the buffer."""
        with self.lock:
            if len(self.buffer) >= self.max_size:
                self.dropped_events += 1
            self.buffer.append(event)
            self.total_events += 1
    
    def get_events(self, count: Optional[int] = None) -> List[TelemetryEvent]:
        """Get events from buffer."""
        with self.lock:
            if count is None:
                return list(self.buffer)
            return list(self.buffer)[-count:] if count > 0 else []
    
    def clear(self) -> int:
        """Clear buffer and return number of events cleared."""
        with self.lock:
            count = len(self.buffer)
            self.buffer.clear()
            return count
    
    def get_stats(self) -> Dict[str, int]:
        """Get buffer statistics."""
        with self.lock:
            return {
                'current_size': len(self.buffer),
                'max_size': self.max_size,
                'total_events': self.total_events,
                'dropped_events': self.dropped_events
            }

class DAWNTelemetryLogger:
    """
    Main telemetry logger for DAWN consciousness system.
    
    Provides structured, high-performance telemetry collection with:
    - Thread-safe event buffering
    - Configurable filtering and levels
    - Multiple output formats
    - Performance monitoring
    - Automatic system metrics collection
    """
    
    def __init__(self, config: Optional[Dict[str, Any]] = None):
        """
        Initialize telemetry logger.
        
        Args:
            config: Configuration dictionary
        """
        self.config = config or {}
        self.session_id = str(uuid.uuid4())
        self.start_time = time.time()
        self.enabled = self.config.get('enabled', True)
        
        # Buffer configuration
        buffer_size = self.config.get('buffer_size', 10000)
        self.buffer = TelemetryBuffer(buffer_size)
        
        # Filtering configuration
        self.min_level = TelemetryLevel(self.config.get('min_level', 'INFO'))
        self.subsystem_filters = self.config.get('subsystem_filters', {})
        self.component_filters = self.config.get('component_filters', {})
        
        # Output configuration
        self.output_handlers = []
        self.auto_flush_interval = self.config.get('auto_flush_interval', 30.0)
        self.auto_flush_enabled = self.config.get('auto_flush_enabled', True)
        
        # Performance tracking
        self.metrics = {
            'events_logged': 0,
            'events_filtered': 0,
            'events_dropped': 0,
            'avg_log_time_ms': 0.0,
            'last_flush_time': time.time(),
            'flush_count': 0
        }
        
        # Threading
        self.lock = threading.RLock()
        self.flush_thread = None
        self.running = False
        
        # System monitoring
        self.collect_system_metrics = self.config.get('collect_system_metrics', True)
        self.system_metrics_interval = self.config.get('system_metrics_interval', 10.0)
        self.system_metrics_thread = None
        
        logger.info(f"🔍 DAWN Telemetry Logger initialized (session: {self.session_id[:8]})")
        
        # Start background processes
        if self.enabled:
            self.start()
    
    def start(self) -> None:
        """Start background telemetry processes."""
        if self.running:
            return
        
        self.running = True
        
        # Start auto-flush thread
        if self.auto_flush_enabled:
            self.flush_thread = threading.Thread(
                target=self._auto_flush_loop,
                name="telemetry_flush",
                daemon=True
            )
            self.flush_thread.start()
        
        # Start system metrics collection
        if self.collect_system_metrics:
            self.system_metrics_thread = threading.Thread(
                target=self._system_metrics_loop,
                name="telemetry_system_metrics",
                daemon=True
            )
            self.system_metrics_thread.start()
        
        logger.info("🔍 Telemetry background processes started")
    
    def stop(self) -> None:
        """Stop background telemetry processes."""
        if not self.running:
            return
        
        self.running = False
        
        # Stop threads
        for thread in [self.flush_thread, self.system_metrics_thread]:
            if thread and thread.is_alive():
                thread.join(timeout=2.0)
        
        # Final flush
        self.flush()
        
        logger.info("🔍 Telemetry logger stopped")
    
    def log_event(self, subsystem: str, component: str, event_type: str,
                  level: TelemetryLevel = TelemetryLevel.INFO,
                  data: Optional[Dict[str, Any]] = None,
                  metadata: Optional[Dict[str, Any]] = None,
                  tick_id: Optional[int] = None) -> None:
        """
        Log a telemetry event.
        
        Args:
            subsystem: DAWN subsystem name
            component: Component within subsystem
            event_type: Type of event
            level: Telemetry level
            data: Event-specific data
            metadata: Additional metadata
            tick_id: Current DAWN tick ID
        """
        if not self.enabled:
            return
        
        start_time = time.perf_counter()
        
        # Apply filters
        if not self._should_log_event(subsystem, component, level):
            with self.lock:
                self.metrics['events_filtered'] += 1
            return
        
        # Create event
        event = TelemetryEvent(
            timestamp=datetime.now(timezone.utc).isoformat(),
            level=level.value,
            subsystem=subsystem,
            component=component,
            event_type=event_type,
            tick_id=tick_id,
            session_id=self.session_id,
            data=data or {},
            metadata=self._enhance_metadata(metadata or {})
        )
        
        # Add to buffer
        self.buffer.add_event(event)
        
        # Update metrics
        log_time_ms = (time.perf_counter() - start_time) * 1000
        with self.lock:
            self.metrics['events_logged'] += 1
            # Update rolling average
            current_avg = self.metrics['avg_log_time_ms']
            count = self.metrics['events_logged']
            self.metrics['avg_log_time_ms'] = (current_avg * (count - 1) + log_time_ms) / count
    
    def _should_log_event(self, subsystem: str, component: str, level: TelemetryLevel) -> bool:
        """Check if event should be logged based on filters."""
        # Level filter
        level_values = {
            TelemetryLevel.DEBUG: 0,
            TelemetryLevel.INFO: 1,
            TelemetryLevel.WARN: 2,
            TelemetryLevel.ERROR: 3,
            TelemetryLevel.CRITICAL: 4
        }
        
        if level_values[level] < level_values[self.min_level]:
            return False
        
        # Subsystem filter
        if subsystem in self.subsystem_filters:
            subsystem_config = self.subsystem_filters[subsystem]
            if not subsystem_config.get('enabled', True):
                return False
            
            # Component filter within subsystem
            if component in subsystem_config.get('component_filters', {}):
                component_config = subsystem_config['component_filters'][component]
                if not component_config.get('enabled', True):
                    return False
        
        # Global component filter
        if component in self.component_filters:
            if not self.component_filters[component].get('enabled', True):
                return False
        
        return True
    
    def _enhance_metadata(self, metadata: Dict[str, Any]) -> Dict[str, Any]:
        """Enhance metadata with system information."""
        enhanced = metadata.copy()
        
        # Add system information if not present
        if 'process_id' not in enhanced:
            enhanced['process_id'] = os.getpid()
        
        if 'thread_id' not in enhanced:
            enhanced['thread_id'] = threading.get_ident()
        
        if 'uptime_seconds' not in enhanced:
            enhanced['uptime_seconds'] = time.time() - self.start_time
        
        return enhanced
    
    def _auto_flush_loop(self) -> None:
        """Background loop for automatic flushing."""
        while self.running:
            try:
                time.sleep(self.auto_flush_interval)
                if self.running:
                    self.flush()
            except Exception as e:
                logger.error(f"Error in auto-flush loop: {e}")
    
    def _system_metrics_loop(self) -> None:
        """Background loop for system metrics collection."""
        while self.running:
            try:
                self._collect_system_metrics()
                time.sleep(self.system_metrics_interval)
            except Exception as e:
                logger.error(f"Error in system metrics collection: {e}")
                time.sleep(5.0)
    
    def _collect_system_metrics(self) -> None:
        """Collect system performance metrics."""
        if not PSUTIL_AVAILABLE:
            # Log a warning once and disable system metrics collection
            if not hasattr(self, '_psutil_warning_logged'):
                logger.warning("psutil not available, system metrics collection disabled")
                self._psutil_warning_logged = True
                self.collect_system_metrics = False
            return
        
        try:
            # CPU metrics
            cpu_percent = psutil.cpu_percent(interval=0.1)
            self.log_event(
                'system', 'cpu', 'usage_metric',
                TelemetryLevel.DEBUG,
                {'cpu_percent': cpu_percent}
            )
            
            # Memory metrics
            memory = psutil.virtual_memory()
            self.log_event(
                'system', 'memory', 'usage_metric',
                TelemetryLevel.DEBUG,
                {
                    'memory_percent': memory.percent,
                    'memory_available_mb': memory.available / (1024 * 1024),
                    'memory_used_mb': memory.used / (1024 * 1024)
                }
            )
            
            # Process metrics
            process = psutil.Process()
            process_memory = process.memory_info()
            self.log_event(
                'system', 'process', 'usage_metric',
                TelemetryLevel.DEBUG,
                {
                    'process_memory_mb': process_memory.rss / (1024 * 1024),
                    'process_cpu_percent': process.cpu_percent(),
                    'num_threads': process.num_threads()
                }
            )
            
        except Exception as e:
            logger.error(f"Error collecting system metrics: {e}")
    
    def flush(self) -> int:
        """Flush buffered events to output handlers."""
        if not self.output_handlers:
            return 0
        
        events = self.buffer.get_events()
        if not events:
            return 0
        
        flushed_count = 0
        for handler in self.output_handlers:
            try:
                handler.write_events(events)
                flushed_count = len(events)
            except Exception as e:
                logger.error(f"Error flushing to handler {handler}: {e}")
        
        # Clear buffer after successful flush
        if flushed_count > 0:
            cleared_count = self.buffer.clear()
            with self.lock:
                self.metrics['last_flush_time'] = time.time()
                self.metrics['flush_count'] += 1
            
            logger.debug(f"Flushed {flushed_count} telemetry events")
        
        return flushed_count
    
    def add_output_handler(self, handler) -> None:
        """Add an output handler for telemetry events."""
        self.output_handlers.append(handler)
        logger.info(f"Added telemetry output handler: {handler}")
    
    def remove_output_handler(self, handler) -> bool:
        """Remove an output handler."""
        if handler in self.output_handlers:
            self.output_handlers.remove(handler)
            logger.info(f"Removed telemetry output handler: {handler}")
            return True
        return False
    
    def get_metrics(self) -> Dict[str, Any]:
        """Get telemetry logger metrics."""
        buffer_stats = self.buffer.get_stats()
        
        with self.lock:
            metrics = self.metrics.copy()
        
        return {
            **metrics,
            'buffer_stats': buffer_stats,
            'session_id': self.session_id,
            'uptime_seconds': time.time() - self.start_time,
            'running': self.running,
            'output_handlers': len(self.output_handlers)
        }
    
    def get_recent_events(self, count: int = 100) -> List[Dict[str, Any]]:
        """Get recent telemetry events."""
        events = self.buffer.get_events(count)
        return [event.to_dict() for event in events]
    
    def configure_filtering(self, subsystem: str = None, component: str = None,
                          enabled: bool = True, min_level: str = None) -> None:
        """Configure filtering for subsystem or component."""
        if subsystem:
            if subsystem not in self.subsystem_filters:
                self.subsystem_filters[subsystem] = {}
            
            self.subsystem_filters[subsystem]['enabled'] = enabled
            
            if min_level:
                self.subsystem_filters[subsystem]['min_level'] = min_level
            
            if component:
                if 'component_filters' not in self.subsystem_filters[subsystem]:
                    self.subsystem_filters[subsystem]['component_filters'] = {}
                
                self.subsystem_filters[subsystem]['component_filters'][component] = {
                    'enabled': enabled
                }
        elif component:
            self.component_filters[component] = {'enabled': enabled}
        
        logger.info(f"Updated telemetry filtering: {subsystem}.{component} enabled={enabled}")

# Convenience functions for common telemetry patterns
def create_performance_context(logger: DAWNTelemetryLogger, subsystem: str, 
                             component: str, operation: str, tick_id: int = None):
    """Create a context manager for performance telemetry."""
    class PerformanceContext:
        def __init__(self):
            self.start_time = None
            self.success = True
            self.metadata = {}
        
        def __enter__(self):
            self.start_time = time.perf_counter()
            return self
        
        def __exit__(self, exc_type, exc_val, exc_tb):
            duration_ms = (time.perf_counter() - self.start_time) * 1000
            self.success = exc_type is None
            
            data = {
                'operation': operation,
                'duration_ms': duration_ms,
                'success': self.success
            }
            
            if not self.success:
                data['error_type'] = exc_type.__name__ if exc_type else None
                data['error_message'] = str(exc_val) if exc_val else None
            
            logger.log_event(
                subsystem, component, 'performance_metric',
                TelemetryLevel.INFO, data, self.metadata, tick_id
            )
        
        def add_metadata(self, key: str, value: Any):
            self.metadata[key] = value
    
    return PerformanceContext()
