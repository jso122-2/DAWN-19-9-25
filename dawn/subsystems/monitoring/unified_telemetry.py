#!/usr/bin/env python3
"""
📊 Unified Telemetry & Logging System
═══════════════════════════════════

Comprehensive telemetry collection and structured logging for all DAWN systems.
Provides unified observability across fractal memory, bloom systems, tracers,
mycelial layer, and all other subsystems.

"Logs include bloom count, Juliet rebloom triggers, entropy values, visual intensity,
and comprehensive bloom summary records with tick-scoped snapshots."

Based on documentation: Fractal Memory/Failure Modes & Safeguards + Logs & Telemetry.rtf
"""

import logging
import time
import threading
import json
import uuid
import numpy as np
from typing import Dict, List, Any, Optional, Tuple, Set, Union, Callable
from dataclasses import dataclass, field, asdict
from enum import Enum
from collections import defaultdict, deque
from datetime import datetime, timedelta
import sqlite3
import os
from pathlib import Path

logger = logging.getLogger(__name__)

class TelemetryLevel(Enum):
    """Telemetry collection levels"""
    MINIMAL = "minimal"       # Only critical metrics
    STANDARD = "standard"     # Standard operational metrics
    DETAILED = "detailed"     # Detailed system metrics
    DEBUG = "debug"          # Full debug telemetry

class EventType(Enum):
    """Types of telemetry events"""
    # Bloom system events
    BLOOM_CREATED = "bloom_created"
    BLOOM_DECAYED = "bloom_decayed"
    JULIET_REBLOOM = "juliet_rebloom"
    REBLOOM_TRIGGER = "rebloom_trigger"
    
    # Memory system events
    MEMORY_ENCODED = "memory_encoded"
    MEMORY_ACCESSED = "memory_accessed"
    FRACTAL_GENERATED = "fractal_generated"
    GHOST_TRACE_CREATED = "ghost_trace_created"
    
    # Shimmer system events
    SHIMMER_DECAY = "shimmer_decay"
    SHIMMER_BOOST = "shimmer_boost"
    SHIMMER_COLLAPSE = "shimmer_collapse"
    
    # Ash/Soot events
    ASH_CREATED = "ash_created"
    SOOT_CREATED = "soot_created"
    RESIDUE_CRYSTALLIZED = "residue_crystallized"
    VOLCANIC_ASH_EVENT = "volcanic_ash_event"
    
    # Tracer events
    TRACER_SPAWNED = "tracer_spawned"
    TRACER_RETIRED = "tracer_retired"
    TRACER_ALERT = "tracer_alert"
    TRACER_OBSERVATION = "tracer_observation"
    
    # Mycelial events
    NODE_CREATED = "node_created"
    NODE_PRUNED = "node_pruned"
    EDGE_FORMED = "edge_formed"
    NUTRIENT_FLOW = "nutrient_flow"
    AUTOPHAGY_EVENT = "autophagy_event"
    
    # Cache events
    CACHE_HIT = "cache_hit"
    CACHE_MISS = "cache_miss"
    FLOW_STATE_CHANGE = "flow_state_change"
    RIDER_STEERING = "rider_steering"
    
    # System events
    TICK_COMPLETED = "tick_completed"
    SYSTEM_STATE_CHANGE = "system_state_change"
    FAILURE_DETECTED = "failure_detected"
    SAFEGUARD_TRIGGERED = "safeguard_triggered"

@dataclass
class TelemetryEvent:
    """Individual telemetry event"""
    event_id: str = field(default_factory=lambda: str(uuid.uuid4()))
    event_type: EventType = EventType.SYSTEM_STATE_CHANGE
    timestamp: float = field(default_factory=time.time)
    tick_id: Optional[str] = None
    system: str = ""
    subsystem: str = ""
    severity: str = "info"
    message: str = ""
    metrics: Dict[str, Any] = field(default_factory=dict)
    metadata: Dict[str, Any] = field(default_factory=dict)
    
    def to_dict(self) -> Dict[str, Any]:
        """Convert to dictionary for serialization"""
        return asdict(self)

@dataclass
class BloomSummaryRecord:
    """Tick-scoped bloom summary as specified in documentation"""
    tick_id: str
    timestamp: float
    total_blooms: int
    active_juliet_flowers: int
    bloom_intensity_avg: float
    bloom_intensity_max: float
    entropy_distribution: List[float]
    mood_state_distribution: Dict[str, int]
    rebloom_depth_avg: float
    bloom_types: Dict[str, int]
    visual_intensity_metrics: Dict[str, float]
    system_pressure: float
    shimmer_health: float

@dataclass
class SystemSnapshot:
    """Complete system state snapshot"""
    snapshot_id: str = field(default_factory=lambda: str(uuid.uuid4()))
    timestamp: float = field(default_factory=time.time)
    tick_id: Optional[str] = None
    
    # Memory system metrics
    memory_metrics: Dict[str, Any] = field(default_factory=dict)
    fractal_metrics: Dict[str, Any] = field(default_factory=dict)
    rebloom_metrics: Dict[str, Any] = field(default_factory=dict)
    shimmer_metrics: Dict[str, Any] = field(default_factory=dict)
    
    # Residue system metrics
    ash_soot_metrics: Dict[str, Any] = field(default_factory=dict)
    ghost_trace_metrics: Dict[str, Any] = field(default_factory=dict)
    
    # Network system metrics
    mycelial_metrics: Dict[str, Any] = field(default_factory=dict)
    cache_metrics: Dict[str, Any] = field(default_factory=dict)
    
    # Cognitive system metrics
    tracer_metrics: Dict[str, Any] = field(default_factory=dict)
    schema_metrics: Dict[str, Any] = field(default_factory=dict)
    
    # System health
    failure_metrics: Dict[str, Any] = field(default_factory=dict)
    performance_metrics: Dict[str, Any] = field(default_factory=dict)

class UnifiedTelemetrySystem:
    """
    Unified telemetry collection system that gathers structured logs and metrics
    from all DAWN subsystems for comprehensive observability.
    """
    
    def __init__(self,
                 telemetry_level: TelemetryLevel = TelemetryLevel.STANDARD,
                 storage_path: str = "dawn_telemetry",
                 retention_days: int = 30,
                 snapshot_interval: float = 10.0):
        
        self.telemetry_level = telemetry_level
        self.storage_path = Path(storage_path)
        self.retention_days = retention_days
        self.snapshot_interval = snapshot_interval
        
        # Create storage directory
        self.storage_path.mkdir(parents=True, exist_ok=True)
        
        # Event storage
        self.event_buffer: deque = deque(maxlen=100000)
        self.bloom_summaries: deque = deque(maxlen=10000)
        self.system_snapshots: deque = deque(maxlen=1000)
        
        # Real-time metrics
        self.current_metrics: Dict[str, Dict[str, Any]] = defaultdict(dict)
        self.metric_history: Dict[str, deque] = defaultdict(lambda: deque(maxlen=1000))
        
        # Event subscriptions
        self.event_subscribers: Dict[EventType, List[Callable]] = defaultdict(list)
        self.metric_subscribers: Dict[str, List[Callable]] = defaultdict(list)
        
        # Statistics
        self.stats = {
            'events_collected': 0,
            'events_by_type': defaultdict(int),
            'snapshots_taken': 0,
            'bloom_summaries_created': 0,
            'storage_operations': 0,
            'start_time': time.time()
        }
        
        # Thread safety
        self._lock = threading.RLock()
        
        # Background processing
        self._processing_active = True
        self._snapshot_thread = threading.Thread(target=self._snapshot_loop, daemon=True)
        self._storage_thread = threading.Thread(target=self._storage_loop, daemon=True)
        self._snapshot_thread.start()
        self._storage_thread.start()
        
        # Initialize database
        self._init_database()
        
        logger.info(f"📊 UnifiedTelemetrySystem initialized - level: {telemetry_level.value}")
    
    def _init_database(self):
        """Initialize SQLite database for persistent storage"""
        db_path = self.storage_path / "telemetry.db"
        
        with sqlite3.connect(str(db_path)) as conn:
            # Events table
            conn.execute("""
                CREATE TABLE IF NOT EXISTS events (
                    event_id TEXT PRIMARY KEY,
                    event_type TEXT,
                    timestamp REAL,
                    tick_id TEXT,
                    system TEXT,
                    subsystem TEXT,
                    severity TEXT,
                    message TEXT,
                    metrics TEXT,
                    metadata TEXT
                )
            """)
            
            # Bloom summaries table
            conn.execute("""
                CREATE TABLE IF NOT EXISTS bloom_summaries (
                    tick_id TEXT PRIMARY KEY,
                    timestamp REAL,
                    total_blooms INTEGER,
                    active_juliet_flowers INTEGER,
                    bloom_intensity_avg REAL,
                    bloom_intensity_max REAL,
                    entropy_distribution TEXT,
                    mood_state_distribution TEXT,
                    rebloom_depth_avg REAL,
                    bloom_types TEXT,
                    visual_intensity_metrics TEXT,
                    system_pressure REAL,
                    shimmer_health REAL
                )
            """)
            
            # System snapshots table
            conn.execute("""
                CREATE TABLE IF NOT EXISTS system_snapshots (
                    snapshot_id TEXT PRIMARY KEY,
                    timestamp REAL,
                    tick_id TEXT,
                    snapshot_data TEXT
                )
            """)
            
            # Create indexes
            conn.execute("CREATE INDEX IF NOT EXISTS idx_events_timestamp ON events(timestamp)")
            conn.execute("CREATE INDEX IF NOT EXISTS idx_events_type ON events(event_type)")
            conn.execute("CREATE INDEX IF NOT EXISTS idx_bloom_timestamp ON bloom_summaries(timestamp)")
            conn.execute("CREATE INDEX IF NOT EXISTS idx_snapshots_timestamp ON system_snapshots(timestamp)")
            
            conn.commit()
    
    def log_event(self,
                  event_type: EventType,
                  system: str,
                  message: str = "",
                  metrics: Optional[Dict[str, Any]] = None,
                  metadata: Optional[Dict[str, Any]] = None,
                  severity: str = "info",
                  tick_id: Optional[str] = None,
                  subsystem: str = "") -> TelemetryEvent:
        """Log a telemetry event"""
        
        event = TelemetryEvent(
            event_type=event_type,
            system=system,
            subsystem=subsystem,
            severity=severity,
            message=message,
            metrics=metrics or {},
            metadata=metadata or {},
            tick_id=tick_id
        )
        
        with self._lock:
            self.event_buffer.append(event)
            self.stats['events_collected'] += 1
            self.stats['events_by_type'][event_type.value] += 1
        
        # Notify subscribers
        self._notify_event_subscribers(event)
        
        logger.debug(f"📊 Event logged: {event_type.value} from {system}")
        return event
    
    def update_metrics(self,
                      system: str,
                      metrics: Dict[str, Any],
                      subsystem: str = "",
                      tick_id: Optional[str] = None):
        """Update real-time metrics for a system"""
        
        with self._lock:
            system_key = f"{system}.{subsystem}" if subsystem else system
            
            # Update current metrics
            self.current_metrics[system_key].update(metrics)
            self.current_metrics[system_key]['last_update'] = time.time()
            
            # Add to history
            metric_entry = {
                'timestamp': time.time(),
                'tick_id': tick_id,
                **metrics
            }
            self.metric_history[system_key].append(metric_entry)
        
        # Notify subscribers
        self._notify_metric_subscribers(system_key, metrics)
    
    def create_bloom_summary(self,
                           tick_id: str,
                           bloom_data: Dict[str, Any]) -> BloomSummaryRecord:
        """Create a bloom summary record as specified in documentation"""
        
        summary = BloomSummaryRecord(
            tick_id=tick_id,
            timestamp=time.time(),
            total_blooms=bloom_data.get('total_blooms', 0),
            active_juliet_flowers=bloom_data.get('active_juliet_flowers', 0),
            bloom_intensity_avg=bloom_data.get('bloom_intensity_avg', 0.0),
            bloom_intensity_max=bloom_data.get('bloom_intensity_max', 0.0),
            entropy_distribution=bloom_data.get('entropy_distribution', []),
            mood_state_distribution=bloom_data.get('mood_state_distribution', {}),
            rebloom_depth_avg=bloom_data.get('rebloom_depth_avg', 0.0),
            bloom_types=bloom_data.get('bloom_types', {}),
            visual_intensity_metrics=bloom_data.get('visual_intensity_metrics', {}),
            system_pressure=bloom_data.get('system_pressure', 0.0),
            shimmer_health=bloom_data.get('shimmer_health', 0.0)
        )
        
        with self._lock:
            self.bloom_summaries.append(summary)
            self.stats['bloom_summaries_created'] += 1
        
        # Log the summary creation
        self.log_event(
            EventType.TICK_COMPLETED,
            "bloom_system",
            f"Bloom summary created for tick {tick_id}",
            metrics=asdict(summary),
            tick_id=tick_id
        )
        
        return summary
    
    def take_system_snapshot(self, tick_id: Optional[str] = None) -> SystemSnapshot:
        """Take a complete system state snapshot"""
        
        snapshot = SystemSnapshot(tick_id=tick_id)
        
        with self._lock:
            # Collect metrics from all systems
            for system_key, metrics in self.current_metrics.items():
                if 'memory' in system_key.lower():
                    snapshot.memory_metrics[system_key] = metrics.copy()
                elif 'fractal' in system_key.lower():
                    snapshot.fractal_metrics[system_key] = metrics.copy()
                elif 'rebloom' in system_key.lower():
                    snapshot.rebloom_metrics[system_key] = metrics.copy()
                elif 'shimmer' in system_key.lower():
                    snapshot.shimmer_metrics[system_key] = metrics.copy()
                elif 'ash' in system_key.lower() or 'soot' in system_key.lower():
                    snapshot.ash_soot_metrics[system_key] = metrics.copy()
                elif 'ghost' in system_key.lower():
                    snapshot.ghost_trace_metrics[system_key] = metrics.copy()
                elif 'mycelial' in system_key.lower():
                    snapshot.mycelial_metrics[system_key] = metrics.copy()
                elif 'cache' in system_key.lower() or 'carrin' in system_key.lower():
                    snapshot.cache_metrics[system_key] = metrics.copy()
                elif 'tracer' in system_key.lower():
                    snapshot.tracer_metrics[system_key] = metrics.copy()
                elif 'schema' in system_key.lower():
                    snapshot.schema_metrics[system_key] = metrics.copy()
                elif 'failure' in system_key.lower() or 'monitor' in system_key.lower():
                    snapshot.failure_metrics[system_key] = metrics.copy()
                else:
                    snapshot.performance_metrics[system_key] = metrics.copy()
            
            self.system_snapshots.append(snapshot)
            self.stats['snapshots_taken'] += 1
        
        logger.debug(f"📊 System snapshot taken: {snapshot.snapshot_id}")
        return snapshot
    
    def subscribe_to_events(self, event_type: EventType, callback: Callable[[TelemetryEvent], None]):
        """Subscribe to specific event types"""
        self.event_subscribers[event_type].append(callback)
    
    def subscribe_to_metrics(self, system: str, callback: Callable[[str, Dict[str, Any]], None]):
        """Subscribe to metric updates for a system"""
        self.metric_subscribers[system].append(callback)
    
    def _notify_event_subscribers(self, event: TelemetryEvent):
        """Notify event subscribers"""
        subscribers = self.event_subscribers.get(event.event_type, [])
        for callback in subscribers:
            try:
                callback(event)
            except Exception as e:
                logger.error(f"Error notifying event subscriber: {e}")
    
    def _notify_metric_subscribers(self, system: str, metrics: Dict[str, Any]):
        """Notify metric subscribers"""
        subscribers = self.metric_subscribers.get(system, [])
        for callback in subscribers:
            try:
                callback(system, metrics)
            except Exception as e:
                logger.error(f"Error notifying metric subscriber: {e}")
    
    def _snapshot_loop(self):
        """Background thread for taking regular snapshots"""
        while self._processing_active:
            try:
                self.take_system_snapshot()
                time.sleep(self.snapshot_interval)
            except Exception as e:
                logger.error(f"Error in snapshot loop: {e}")
                time.sleep(1.0)
    
    def _storage_loop(self):
        """Background thread for persistent storage"""
        while self._processing_active:
            try:
                self._store_events_batch()
                self._store_bloom_summaries_batch()
                self._store_snapshots_batch()
                self._cleanup_old_data()
                time.sleep(5.0)  # Store every 5 seconds
            except Exception as e:
                logger.error(f"Error in storage loop: {e}")
                time.sleep(1.0)
    
    def _store_events_batch(self):
        """Store batched events to database"""
        if not self.event_buffer:
            return
        
        db_path = self.storage_path / "telemetry.db"
        events_to_store = []
        
        with self._lock:
            # Take up to 1000 events from buffer
            for _ in range(min(1000, len(self.event_buffer))):
                if self.event_buffer:
                    events_to_store.append(self.event_buffer.popleft())
        
        if events_to_store:
            with sqlite3.connect(str(db_path)) as conn:
                for event in events_to_store:
                    conn.execute("""
                        INSERT OR REPLACE INTO events 
                        (event_id, event_type, timestamp, tick_id, system, subsystem, 
                         severity, message, metrics, metadata)
                        VALUES (?, ?, ?, ?, ?, ?, ?, ?, ?, ?)
                    """, (
                        event.event_id,
                        event.event_type.value,
                        event.timestamp,
                        event.tick_id,
                        event.system,
                        event.subsystem,
                        event.severity,
                        event.message,
                        json.dumps(event.metrics),
                        json.dumps(event.metadata)
                    ))
                conn.commit()
                self.stats['storage_operations'] += len(events_to_store)
    
    def _store_bloom_summaries_batch(self):
        """Store bloom summaries to database"""
        if not self.bloom_summaries:
            return
        
        db_path = self.storage_path / "telemetry.db"
        summaries_to_store = []
        
        with self._lock:
            # Take up to 100 summaries
            for _ in range(min(100, len(self.bloom_summaries))):
                if self.bloom_summaries:
                    summaries_to_store.append(self.bloom_summaries.popleft())
        
        if summaries_to_store:
            with sqlite3.connect(str(db_path)) as conn:
                for summary in summaries_to_store:
                    conn.execute("""
                        INSERT OR REPLACE INTO bloom_summaries
                        (tick_id, timestamp, total_blooms, active_juliet_flowers,
                         bloom_intensity_avg, bloom_intensity_max, entropy_distribution,
                         mood_state_distribution, rebloom_depth_avg, bloom_types,
                         visual_intensity_metrics, system_pressure, shimmer_health)
                        VALUES (?, ?, ?, ?, ?, ?, ?, ?, ?, ?, ?, ?, ?)
                    """, (
                        summary.tick_id,
                        summary.timestamp,
                        summary.total_blooms,
                        summary.active_juliet_flowers,
                        summary.bloom_intensity_avg,
                        summary.bloom_intensity_max,
                        json.dumps(summary.entropy_distribution),
                        json.dumps(summary.mood_state_distribution),
                        summary.rebloom_depth_avg,
                        json.dumps(summary.bloom_types),
                        json.dumps(summary.visual_intensity_metrics),
                        summary.system_pressure,
                        summary.shimmer_health
                    ))
                conn.commit()
    
    def _store_snapshots_batch(self):
        """Store system snapshots to database"""
        if not self.system_snapshots:
            return
        
        db_path = self.storage_path / "telemetry.db"
        snapshots_to_store = []
        
        with self._lock:
            # Take up to 10 snapshots
            for _ in range(min(10, len(self.system_snapshots))):
                if self.system_snapshots:
                    snapshots_to_store.append(self.system_snapshots.popleft())
        
        if snapshots_to_store:
            with sqlite3.connect(str(db_path)) as conn:
                for snapshot in snapshots_to_store:
                    conn.execute("""
                        INSERT OR REPLACE INTO system_snapshots
                        (snapshot_id, timestamp, tick_id, snapshot_data)
                        VALUES (?, ?, ?, ?)
                    """, (
                        snapshot.snapshot_id,
                        snapshot.timestamp,
                        snapshot.tick_id,
                        json.dumps(asdict(snapshot))
                    ))
                conn.commit()
    
    def _cleanup_old_data(self):
        """Clean up old data based on retention policy"""
        cutoff_time = time.time() - (self.retention_days * 24 * 3600)
        db_path = self.storage_path / "telemetry.db"
        
        with sqlite3.connect(str(db_path)) as conn:
            # Clean up old events
            conn.execute("DELETE FROM events WHERE timestamp < ?", (cutoff_time,))
            conn.execute("DELETE FROM bloom_summaries WHERE timestamp < ?", (cutoff_time,))
            conn.execute("DELETE FROM system_snapshots WHERE timestamp < ?", (cutoff_time,))
            conn.commit()
    
    def query_events(self,
                    event_type: Optional[EventType] = None,
                    system: Optional[str] = None,
                    start_time: Optional[float] = None,
                    end_time: Optional[float] = None,
                    limit: int = 1000) -> List[TelemetryEvent]:
        """Query events from storage"""
        
        db_path = self.storage_path / "telemetry.db"
        query = "SELECT * FROM events WHERE 1=1"
        params = []
        
        if event_type:
            query += " AND event_type = ?"
            params.append(event_type.value)
        
        if system:
            query += " AND system = ?"
            params.append(system)
        
        if start_time:
            query += " AND timestamp >= ?"
            params.append(start_time)
        
        if end_time:
            query += " AND timestamp <= ?"
            params.append(end_time)
        
        query += " ORDER BY timestamp DESC LIMIT ?"
        params.append(limit)
        
        events = []
        with sqlite3.connect(str(db_path)) as conn:
            cursor = conn.execute(query, params)
            for row in cursor.fetchall():
                event = TelemetryEvent(
                    event_id=row[0],
                    event_type=EventType(row[1]),
                    timestamp=row[2],
                    tick_id=row[3],
                    system=row[4],
                    subsystem=row[5],
                    severity=row[6],
                    message=row[7],
                    metrics=json.loads(row[8]) if row[8] else {},
                    metadata=json.loads(row[9]) if row[9] else {}
                )
                events.append(event)
        
        return events
    
    def get_system_dashboard(self) -> Dict[str, Any]:
        """Get comprehensive system dashboard data"""
        with self._lock:
            # Calculate uptime
            uptime = time.time() - self.stats['start_time']
            
            # Get recent event counts
            recent_events = defaultdict(int)
            cutoff = time.time() - 3600  # Last hour
            
            for event in list(self.event_buffer):
                if event.timestamp > cutoff:
                    recent_events[event.event_type.value] += 1
            
            # System health indicators
            health_indicators = {}
            for system, metrics in self.current_metrics.items():
                last_update = metrics.get('last_update', 0)
                health_indicators[system] = {
                    'online': time.time() - last_update < 60,  # Updated within 1 minute
                    'last_seen': last_update,
                    'key_metrics': {k: v for k, v in metrics.items() 
                                  if k != 'last_update' and isinstance(v, (int, float))}
                }
            
            return {
                'system_uptime_seconds': uptime,
                'telemetry_stats': dict(self.stats),
                'recent_event_counts': dict(recent_events),
                'system_health': health_indicators,
                'active_systems': len(self.current_metrics),
                'buffer_status': {
                    'events_buffered': len(self.event_buffer),
                    'bloom_summaries_buffered': len(self.bloom_summaries),
                    'snapshots_buffered': len(self.system_snapshots)
                },
                'telemetry_level': self.telemetry_level.value,
                'storage_path': str(self.storage_path)
            }
    
    def shutdown(self):
        """Shutdown the telemetry system"""
        self._processing_active = False
        
        # Wait for threads to complete
        if self._snapshot_thread.is_alive():
            self._snapshot_thread.join(timeout=2.0)
        if self._storage_thread.is_alive():
            self._storage_thread.join(timeout=5.0)
        
        # Final storage flush
        self._store_events_batch()
        self._store_bloom_summaries_batch()
        self._store_snapshots_batch()
        
        logger.info("📊 UnifiedTelemetrySystem shutdown complete")


# Global telemetry system instance
_telemetry_system = None

def get_telemetry_system(config: Optional[Dict[str, Any]] = None) -> UnifiedTelemetrySystem:
    """Get the global telemetry system instance"""
    global _telemetry_system
    if _telemetry_system is None:
        config = config or {}
        _telemetry_system = UnifiedTelemetrySystem(
            telemetry_level=TelemetryLevel(config.get('telemetry_level', 'standard')),
            storage_path=config.get('storage_path', 'dawn_telemetry'),
            retention_days=config.get('retention_days', 30),
            snapshot_interval=config.get('snapshot_interval', 10.0)
        )
    return _telemetry_system


# Example usage and testing
if __name__ == "__main__":
    logging.basicConfig(level=logging.INFO)
    
    # Initialize telemetry system
    telemetry = UnifiedTelemetrySystem()
    
    # Log some example events
    telemetry.log_event(
        EventType.JULIET_REBLOOM,
        "memory_system",
        "Memory successfully rebloomed into Juliet flower",
        metrics={'enhancement_level': 0.85, 'access_count': 15},
        metadata={'memory_id': 'test_memory_123'}
    )
    
    telemetry.log_event(
        EventType.TRACER_SPAWNED,
        "tracer_manager",
        "Crow tracer spawned for anomaly detection",
        metrics={'spawn_reason': 'entropy_spike', 'target_entropy': 0.7}
    )
    
    # Update some metrics
    telemetry.update_metrics(
        "rebloom_engine",
        {
            'active_juliet_flowers': 42,
            'rebloom_success_rate': 0.73,
            'average_enhancement_level': 0.68
        }
    )
    
    # Create bloom summary
    telemetry.create_bloom_summary(
        "tick_12345",
        {
            'total_blooms': 156,
            'active_juliet_flowers': 42,
            'bloom_intensity_avg': 0.65,
            'bloom_intensity_max': 0.92,
            'entropy_distribution': [0.1, 0.3, 0.4, 0.2],
            'mood_state_distribution': {'curious': 25, 'focused': 17},
            'system_pressure': 0.72,
            'shimmer_health': 0.81
        }
    )
    
    # Let it run for a bit
    time.sleep(5)
    
    # Get dashboard
    dashboard = telemetry.get_system_dashboard()
    print(f"System dashboard: {json.dumps(dashboard, indent=2)}")
    
    # Shutdown
    telemetry.shutdown()
