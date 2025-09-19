#!/usr/bin/env python3
"""
🔍 Universal JSON Logger - Complete State Logging for Every DAWN Process
========================================================================

This universal logging system ensures that ABSOLUTELY EVERY process and module
in the DAWN system writes its state to disk in JSON/JSONL format. It provides:

- Automatic state detection and serialization for any Python object
- JSONL streaming for high-frequency updates
- Per-process and per-module state files
- Automatic discovery and registration of all DAWN components
- State diffing and change detection
- Performance-optimized batch writing
- Thread-safe operations across all processes

Every class that inherits from any base class or uses DAWN components will
automatically get comprehensive JSON state logging.
"""

import json
import time
import threading
import inspect
import sys
import os
import gc
from typing import Dict, List, Any, Optional, Union, Type, Set, Callable
from dataclasses import dataclass, field, asdict, is_dataclass
from datetime import datetime, timezone
from pathlib import Path
from collections import defaultdict, deque
from enum import Enum
import uuid
import traceback
# Optional psutil import for system metrics
try:
    import psutil
    PSUTIL_AVAILABLE = True
except ImportError:
    PSUTIL_AVAILABLE = False
    # Mock psutil functionality
    class MockProcess:
        def cpu_percent(self): return 0.0
        def memory_info(self): 
            class MockMemInfo:
                rss = 0
            return MockMemInfo()
        def num_threads(self): return 1
        def open_files(self): return []
    
    class MockPsutil:
        @staticmethod
        def Process(): return MockProcess()
    
    psutil = MockPsutil()
import pickle
import gzip

# Import existing enhanced logger to build upon it
try:
    from dawn.core.telemetry.enhanced_module_logger import (
        EnhancedModuleLogger, get_enhanced_logger, StateChangeType, DeltaChange
    )
    ENHANCED_LOGGING_AVAILABLE = True
except ImportError:
    ENHANCED_LOGGING_AVAILABLE = False

# Import existing telemetry
try:
    from dawn.core.telemetry.system import log_event, create_performance_context
    from dawn.core.telemetry.logger import TelemetryLevel
    TELEMETRY_AVAILABLE = True
except ImportError:
    TELEMETRY_AVAILABLE = False
    # Mock implementations
    def log_event(*args, **kwargs): pass
    def create_performance_context(*args, **kwargs):
        class MockContext:
            def __enter__(self): return self
            def __exit__(self, *args): pass
            def add_metadata(self, key, value): pass
        return MockContext()
    class TelemetryLevel:
        DEBUG = "debug"
        INFO = "info"

import logging
logger = logging.getLogger(__name__)

class LogFormat(Enum):
    """Supported logging formats"""
    JSON = "json"           # Single JSON file per module
    JSONL = "jsonl"         # JSON Lines streaming format
    COMPRESSED = "gz"       # Gzipped JSON for space efficiency
    PICKLE = "pickle"       # Python pickle for complex objects

class StateScope(Enum):
    """Scope of state logging"""
    FULL = "full"           # Complete object state
    CHANGES_ONLY = "changes" # Only changed fields
    SUMMARY = "summary"     # High-level summary only
    MINIMAL = "minimal"     # Critical fields only

@dataclass
class LoggingConfig:
    """Configuration for universal JSON logging"""
    base_path: str = "logs/universal_json"
    format: LogFormat = LogFormat.JSONL
    scope: StateScope = StateScope.FULL
    max_file_size_mb: int = 100
    max_files_per_module: int = 10
    flush_interval_seconds: float = 1.0
    enable_compression: bool = True
    enable_state_diffing: bool = True
    enable_auto_discovery: bool = True
    include_system_metrics: bool = True
    include_stack_traces: bool = False
    batch_size: int = 100

@dataclass 
class StateSnapshot:
    """Complete state snapshot of an object"""
    object_id: str
    class_name: str
    module_name: str
    timestamp: float
    tick_number: Optional[int]
    state_data: Dict[str, Any]
    state_hash: str
    change_summary: Dict[str, Any]
    system_metrics: Dict[str, Any] = field(default_factory=dict)
    stack_trace: Optional[str] = None

class StateSerializer:
    """Advanced state serializer that can handle any Python object"""
    
    @staticmethod
    def serialize_value(value: Any, max_depth: int = 10, current_depth: int = 0, 
                       seen_objects: Optional[Set[int]] = None) -> Any:
        """Recursively serialize any Python value to JSON-compatible format"""
        if seen_objects is None:
            seen_objects = set()
            
        if current_depth > max_depth:
            return f"<MAX_DEPTH_EXCEEDED: {type(value).__name__}>"
        
        # Check for circular references
        obj_id = id(value)
        if obj_id in seen_objects:
            return f"<CIRCULAR_REF: {type(value).__name__}>"
        
        # Handle None, primitives (no need to track these)
        if value is None or isinstance(value, (bool, int, float, str)):
            return value
        
        # Handle enums
        if isinstance(value, Enum):
            return {"__enum__": True, "name": value.name, "value": value.value}
        
        # Handle datetime
        if isinstance(value, datetime):
            return {"__datetime__": True, "isoformat": value.isoformat()}
        
        # Add to seen objects for complex types
        seen_objects.add(obj_id)
        
        try:
            # Skip logging objects to prevent self-referential loops
            if (hasattr(value, '__class__') and 
                ('Logger' in value.__class__.__name__ or 
                 'Snapshot' in value.__class__.__name__ or
                 'Universal' in value.__class__.__name__ or
                 'Serializer' in value.__class__.__name__)):
                return f"<LOGGING_OBJ_SKIPPED: {type(value).__name__}>"
            
            # Handle dataclasses
            if is_dataclass(value) and not isinstance(value, type):
                return {
                    "__dataclass__": True,
                    "class": value.__class__.__name__,
                    "data": StateSerializer.serialize_value(asdict(value), max_depth, current_depth + 1, seen_objects)
                }
        
            # Handle lists, tuples
            if isinstance(value, (list, tuple)):
                return [StateSerializer.serialize_value(item, max_depth, current_depth + 1, seen_objects) 
                       for item in value[:100]]  # Limit list size
        
            # Handle sets
            if isinstance(value, set):
                return {
                    "__set__": True,
                    "items": [StateSerializer.serialize_value(item, max_depth, current_depth + 1, seen_objects) 
                             for item in list(value)[:100]]
                }
        
            # Handle dictionaries
            if isinstance(value, dict):
                result = {}
                for k, v in list(value.items())[:100]:  # Limit dict size
                    try:
                        key_str = str(k) if not isinstance(k, str) else k
                        result[key_str] = StateSerializer.serialize_value(v, max_depth, current_depth + 1, seen_objects)
                    except Exception as e:
                        result[f"<SERIALIZATION_ERROR_{k}>"] = str(e)
                return result
        
            # Handle deques
            if isinstance(value, deque):
                return {
                    "__deque__": True,
                    "maxlen": value.maxlen,
                    "items": [StateSerializer.serialize_value(item, max_depth, current_depth + 1, seen_objects) 
                             for item in list(value)[:100]]
                }
        
            # Handle complex objects with __dict__
            if hasattr(value, '__dict__'):
                try:
                    obj_dict = {}
                    for attr_name, attr_value in value.__dict__.items():
                        if not attr_name.startswith('_'):  # Skip private attributes
                            try:
                                obj_dict[attr_name] = StateSerializer.serialize_value(
                                    attr_value, max_depth, current_depth + 1, seen_objects
                                )
                            except Exception as e:
                                obj_dict[f"<ERROR_{attr_name}>"] = str(e)
                    
                    return {
                        "__object__": True,
                        "class": value.__class__.__name__,
                        "module": value.__class__.__module__,
                        "attributes": obj_dict
                    }
                except Exception as e:
                    return f"<OBJECT_SERIALIZATION_ERROR: {e}>"
            
            # Handle callable objects
            if callable(value):
                return {
                    "__callable__": True,
                    "name": getattr(value, '__name__', 'unknown'),
                    "module": getattr(value, '__module__', 'unknown')
                }
            
            # Fallback: convert to string
            try:
                return str(value)
            except Exception as e:
                return f"<FALLBACK_ERROR: {e}>"
        
        finally:
            # Remove from seen objects when done with this branch
            seen_objects.discard(obj_id)

class UniversalObjectTracker:
    """Tracks all objects in the system for automatic state logging"""
    
    def __init__(self):
        self.tracked_objects: Dict[str, Any] = {}
        self.object_metadata: Dict[str, Dict[str, Any]] = {}
        self.discovery_patterns = [
            "dawn.consciousness.*",
            "dawn.processing.*",
            "dawn.subsystems.*",
            "dawn.core.*",
            "*Engine*",
            "*Manager*",
            "*System*",
            "*Controller*",
            "*Orchestrator*"
        ]
        self._lock = threading.RLock()
    
    def register_object(self, obj: Any, name: Optional[str] = None) -> Optional[str]:
        """Register an object for tracking"""
        with self._lock:
            try:
                # Safety check - skip logging objects
                if hasattr(obj, '__class__'):
                    class_name = obj.__class__.__name__
                    module_name = getattr(obj.__class__, '__module__', '')
                    
                    if (any(keyword in class_name for keyword in 
                           ['Logger', 'Snapshot', 'Universal', 'Serializer', 'LogEntry', 'Repository', 'Telemetry']) or
                        any(keyword in module_name for keyword in 
                           ['logging', 'telemetry', 'centralized_repo'])):
                        return None  # Don't register logging objects
                
                object_id = f"{obj.__class__.__module__}.{obj.__class__.__name__}_{id(obj)}"
                if name:
                    object_id = f"{name}_{id(obj)}"
                
                # Check if already registered
                if object_id in self.tracked_objects:
                    return object_id
                
                self.tracked_objects[object_id] = obj
                self.object_metadata[object_id] = {
                    'registered_at': time.time(),
                    'class_name': obj.__class__.__name__,
                    'module_name': obj.__class__.__module__,
                    'custom_name': name,
                    'last_logged': 0.0,
                    'log_count': 0
                }
                
                return object_id
            except Exception as e:
                # If registration fails, return None
                return None
    
    def discover_dawn_objects(self) -> List[str]:
        """Automatically discover all DAWN objects in memory"""
        discovered = []
        
        with self._lock:
            # Get all objects from garbage collector
            for obj in gc.get_objects():
                try:
                    if hasattr(obj, '__class__') and hasattr(obj, '__module__'):
                        module_name = obj.__class__.__module__ or ""
                        class_name = obj.__class__.__name__
                        
                        # Skip logging objects to prevent feedback loops
                        if (any(keyword in class_name for keyword in 
                               ['Logger', 'Snapshot', 'Universal', 'Serializer', 'LogEntry', 'Repository', 'Telemetry']) or
                            any(keyword in module_name for keyword in 
                               ['logging', 'telemetry', 'centralized_repo'])):
                            continue
                        
                        # Check if it matches DAWN patterns
                        if any([
                            module_name.startswith('dawn.'),
                            'Engine' in class_name,
                            'Manager' in class_name,
                            'System' in class_name,
                            'Controller' in class_name,
                            'Orchestrator' in class_name
                        ]):
                            object_id = self.register_object(obj)
                            if object_id:  # Only add if registration was successful
                                discovered.append(object_id)
                            
                except Exception as e:
                    # Skip objects that can't be introspected
                    continue
        
        logger.info(f"🔍 Discovered {len(discovered)} DAWN objects for logging")
        return discovered
    
    def get_object_state(self, object_id: str) -> Optional[Dict[str, Any]]:
        """Get current state of a tracked object"""
        with self._lock:
            if object_id not in self.tracked_objects:
                return None
            
            obj = self.tracked_objects[object_id]
            try:
                return StateSerializer.serialize_value(obj)
            except Exception as e:
                logger.warning(f"Failed to serialize {object_id}: {e}")
                return {"error": str(e), "class": obj.__class__.__name__}

class UniversalJsonLogger:
    """Universal JSON logger that captures state from every DAWN process and module"""
    
    def __init__(self, config: Optional[LoggingConfig] = None):
        self.config = config or LoggingConfig()
        self.base_path = Path(self.config.base_path)
        self.base_path.mkdir(parents=True, exist_ok=True)
        
        # Core components
        self.object_tracker = UniversalObjectTracker()
        self.file_handles: Dict[str, Any] = {}
        self.state_cache: Dict[str, Dict[str, Any]] = {}
        self.write_queues: Dict[str, deque] = defaultdict(lambda: deque())
        
        # Threading
        self._lock = threading.RLock()
        self._running = True
        self._flush_thread = threading.Thread(target=self._flush_loop, daemon=True)
        self._discovery_thread = threading.Thread(target=self._discovery_loop, daemon=True)
        
        # Statistics
        self.stats = {
            'objects_tracked': 0,
            'states_logged': 0,
            'files_created': 0,
            'bytes_written': 0,
            'errors': 0,
            'start_time': time.time()
        }
        
        # Start background threads
        self._flush_thread.start()
        if self.config.enable_auto_discovery:
            self._discovery_thread.start()
        
        # Integration with existing enhanced logging
        if ENHANCED_LOGGING_AVAILABLE:
            self._integrate_with_enhanced_logging()
        
        # Initialize centralized repository integration
        self.centralized_repo = None
        self._init_centralized_integration()
        
        logger.info(f"🔍 UniversalJsonLogger initialized at {self.base_path}")
    
    def _integrate_with_enhanced_logging(self):
        """Integrate with existing enhanced module logger"""
        try:
            # Hook into existing enhanced loggers to also write universal JSON
            original_log_tick = EnhancedModuleLogger.log_tick_update
            
            def enhanced_log_tick_with_universal(self_logger, state_data, tick_number=None, export_json=True):
                # Call original method
                result = original_log_tick(self_logger, state_data, tick_number, export_json)
                
                # Also log to universal system
                try:
                    object_id = f"{self_logger.subsystem}.{self_logger.module_name}"
                    universal_logger = get_universal_logger()
                    universal_logger.log_object_state(
                        object_id, state_data, tick_number=tick_number
                    )
                except Exception as e:
                    logger.warning(f"Failed to integrate with universal logger: {e}")
                
                return result
            
            # Monkey patch the method
            EnhancedModuleLogger.log_tick_update = enhanced_log_tick_with_universal
            logger.info("✅ Integrated with existing enhanced module logging")
            
        except Exception as e:
            logger.warning(f"Could not integrate with enhanced logging: {e}")
    
    def _init_centralized_integration(self):
        """Initialize integration with centralized repository"""
        try:
            # Import here to avoid circular imports
            from .centralized_repo import get_centralized_repository
            
            # Get centralized repository
            self.centralized_repo = get_centralized_repository()
            
            # Hook into our queue method to also send to centralized repo
            original_queue = self._queue_snapshot
            
            def centralized_queue(snapshot: StateSnapshot):
                # Call original method
                original_queue(snapshot)
                
                # Also send to centralized repository
                if self.centralized_repo:
                    try:
                        log_data = asdict(snapshot)
                        self.centralized_repo.add_log_entry(
                            system=snapshot.module_name or "unknown",
                            subsystem=snapshot.class_name or "unknown",
                            module=snapshot.object_id.split('_')[0] if '_' in snapshot.object_id else "unknown",
                            log_data=log_data,
                            log_type="state"
                        )
                    except Exception as e:
                        logger.debug(f"Failed to send to centralized repo: {e}")
            
            # Replace the method
            self._queue_snapshot = centralized_queue
            
            logger.info("🗂️ Integrated with centralized deep logging repository")
            
        except Exception as e:
            logger.warning(f"Could not integrate with centralized repository: {e}")
    
    def log_object_state(self, object_id: str, state_data: Optional[Dict[str, Any]] = None, 
                        tick_number: Optional[int] = None, custom_metadata: Optional[Dict[str, Any]] = None) -> bool:
        """Log the state of an object"""
        try:
            with self._lock:
                # Get state data
                if state_data is None:
                    state_data = self.object_tracker.get_object_state(object_id)
                    if state_data is None:
                        return False
                
                # Create state snapshot
                snapshot = StateSnapshot(
                    object_id=object_id,
                    class_name=object_id.split('_')[0].split('.')[-1] if '.' in object_id else 'Unknown',
                    module_name=object_id.split('.')[0] if '.' in object_id else 'Unknown',
                    timestamp=time.time(),
                    tick_number=tick_number,
                    state_data=state_data,
                    state_hash=self._calculate_state_hash(state_data),
                    change_summary=self._calculate_changes(object_id, state_data),
                    system_metrics=self._get_system_metrics() if self.config.include_system_metrics else {},
                    stack_trace=self._get_stack_trace() if self.config.include_stack_traces else None
                )
                
                # Add custom metadata
                if custom_metadata:
                    snapshot.state_data.update(custom_metadata)
                
                # Queue for writing
                self._queue_snapshot(snapshot)
                
                # Update statistics
                self.stats['states_logged'] += 1
                if object_id in self.object_tracker.object_metadata:
                    self.object_tracker.object_metadata[object_id]['last_logged'] = time.time()
                    self.object_tracker.object_metadata[object_id]['log_count'] += 1
                
                return True
                
        except Exception as e:
            self.stats['errors'] += 1
            logger.error(f"Failed to log state for {object_id}: {e}")
            return False
    
    def log_all_tracked_objects(self, tick_number: Optional[int] = None) -> int:
        """Log state of all tracked objects"""
        logged_count = 0
        
        with self._lock:
            for object_id in list(self.object_tracker.tracked_objects.keys()):
                if self.log_object_state(object_id, tick_number=tick_number):
                    logged_count += 1
        
        return logged_count
    
    def register_object(self, obj: Any, name: Optional[str] = None) -> Optional[str]:
        """Register an object for automatic state logging"""
        object_id = self.object_tracker.register_object(obj, name)
        
        if object_id:  # Only proceed if registration was successful
            self.stats['objects_tracked'] += 1
            
            # Log initial state
            self.log_object_state(object_id)
        
        return object_id
    
    def _queue_snapshot(self, snapshot: StateSnapshot):
        """Queue a snapshot for writing"""
        # Determine target file
        file_key = f"{snapshot.module_name}_{snapshot.class_name}"
        
        # Add to write queue
        self.write_queues[file_key].append(snapshot)
        
        # Immediate flush if queue is large
        if len(self.write_queues[file_key]) >= self.config.batch_size:
            self._flush_queue(file_key)
    
    def _flush_queue(self, file_key: str):
        """Flush write queue for a specific file"""
        if not self.write_queues[file_key]:
            return
        
        try:
            # Get file path
            file_path = self._get_file_path(file_key)
            
            # Prepare data for writing
            snapshots_to_write = []
            while self.write_queues[file_key] and len(snapshots_to_write) < self.config.batch_size:
                snapshots_to_write.append(self.write_queues[file_key].popleft())
            
            # Write based on format
            if self.config.format == LogFormat.JSONL:
                self._write_jsonl(file_path, snapshots_to_write)
            elif self.config.format == LogFormat.JSON:
                self._write_json(file_path, snapshots_to_write)
            elif self.config.format == LogFormat.COMPRESSED:
                self._write_compressed(file_path, snapshots_to_write)
            
            self.stats['bytes_written'] += file_path.stat().st_size if file_path.exists() else 0
            
        except Exception as e:
            self.stats['errors'] += 1
            logger.error(f"Failed to flush queue for {file_key}: {e}")
    
    def _write_jsonl(self, file_path: Path, snapshots: List[StateSnapshot]):
        """Write snapshots in JSONL format"""
        with open(file_path, 'a', encoding='utf-8') as f:
            for snapshot in snapshots:
                json_line = json.dumps(asdict(snapshot), default=str, separators=(',', ':'))
                f.write(json_line + '\n')
    
    def _write_json(self, file_path: Path, snapshots: List[StateSnapshot]):
        """Write snapshots in JSON format"""
        # Load existing data
        existing_data = []
        if file_path.exists():
            try:
                with open(file_path, 'r', encoding='utf-8') as f:
                    existing_data = json.load(f)
            except:
                existing_data = []
        
        # Add new snapshots
        for snapshot in snapshots:
            existing_data.append(asdict(snapshot))
        
        # Keep only recent snapshots to prevent files from growing too large
        if len(existing_data) > 1000:
            existing_data = existing_data[-1000:]
        
        # Write back
        with open(file_path, 'w', encoding='utf-8') as f:
            json.dump(existing_data, f, indent=2, default=str)
    
    def _write_compressed(self, file_path: Path, snapshots: List[StateSnapshot]):
        """Write snapshots in compressed format"""
        file_path = file_path.with_suffix(file_path.suffix + '.gz')
        
        with gzip.open(file_path, 'at', encoding='utf-8') as f:
            for snapshot in snapshots:
                json_line = json.dumps(asdict(snapshot), default=str, separators=(',', ':'))
                f.write(json_line + '\n')
    
    def _get_file_path(self, file_key: str) -> Path:
        """Get file path for a given file key"""
        timestamp = datetime.now().strftime("%Y%m%d")
        
        if self.config.format == LogFormat.JSONL:
            filename = f"{file_key}_{timestamp}.jsonl"
        elif self.config.format == LogFormat.JSON:
            filename = f"{file_key}_{timestamp}.json"
        else:
            filename = f"{file_key}_{timestamp}.log"
        
        return self.base_path / filename
    
    def _calculate_state_hash(self, state_data: Dict[str, Any]) -> str:
        """Calculate hash of state data for change detection"""
        try:
            state_str = json.dumps(state_data, sort_keys=True, default=str)
            return str(hash(state_str))
        except:
            return str(hash(str(state_data)))
    
    def _calculate_changes(self, object_id: str, new_state: Dict[str, Any]) -> Dict[str, Any]:
        """Calculate changes from previous state"""
        if object_id not in self.state_cache:
            self.state_cache[object_id] = new_state
            return {"change_type": "initial_state", "changes": 0}
        
        old_state = self.state_cache[object_id]
        changes = []
        
        # Simple change detection
        for key, new_value in new_state.items():
            if key not in old_state:
                changes.append({"field": key, "change": "added", "new_value": new_value})
            elif old_state[key] != new_value:
                changes.append({"field": key, "change": "modified", 
                              "old_value": old_state[key], "new_value": new_value})
        
        for key in old_state:
            if key not in new_state:
                changes.append({"field": key, "change": "removed", "old_value": old_state[key]})
        
        # Update cache
        self.state_cache[object_id] = new_state.copy()
        
        return {"changes": len(changes), "change_list": changes[:10]}  # Limit change list size
    
    def _get_system_metrics(self) -> Dict[str, Any]:
        """Get current system metrics"""
        try:
            process = psutil.Process()
            return {
                'cpu_percent': process.cpu_percent(),
                'memory_mb': process.memory_info().rss / 1024 / 1024,
                'threads': process.num_threads(),
                'open_files': len(process.open_files()),
                'system_time': time.time()
            }
        except:
            return {}
    
    def _get_stack_trace(self) -> str:
        """Get current stack trace"""
        return traceback.format_stack()[-5:]  # Last 5 frames
    
    def _flush_loop(self):
        """Background thread for periodic flushing"""
        while self._running:
            try:
                with self._lock:
                    for file_key in list(self.write_queues.keys()):
                        if self.write_queues[file_key]:
                            self._flush_queue(file_key)
                
                time.sleep(self.config.flush_interval_seconds)
                
            except Exception as e:
                logger.error(f"Error in flush loop: {e}")
                time.sleep(1.0)
    
    def _discovery_loop(self):
        """Background thread for automatic object discovery"""
        while self._running:
            try:
                discovered = self.object_tracker.discover_dawn_objects()
                if discovered and len(discovered) > 0:
                    # Only log if we found genuinely new objects
                    if not hasattr(self, '_last_discovery_count'):
                        self._last_discovery_count = 0
                    if len(discovered) != self._last_discovery_count:
                        logger.debug(f"🔍 Auto-discovered {len(discovered)} new objects")
                        self._last_discovery_count = len(discovered)
                
                time.sleep(30.0)  # Discover every 30 seconds
                
            except Exception as e:
                logger.error(f"Error in discovery loop: {e}")
                time.sleep(10.0)
    
    def get_stats(self) -> Dict[str, Any]:
        """Get logging statistics"""
        with self._lock:
            runtime = time.time() - self.stats['start_time']
            return {
                **self.stats,
                'runtime_seconds': runtime,
                'objects_tracked': len(self.object_tracker.tracked_objects),
                'queued_snapshots': sum(len(q) for q in self.write_queues.values()),
                'states_per_second': self.stats['states_logged'] / max(runtime, 1),
                'files_active': len(self.write_queues)
            }
    
    def shutdown(self):
        """Shutdown the logger"""
        logger.info("🔍 Shutting down UniversalJsonLogger...")
        
        self._running = False
        
        # Wait for threads
        if self._flush_thread.is_alive():
            self._flush_thread.join(timeout=5.0)
        if self._discovery_thread.is_alive():
            self._discovery_thread.join(timeout=2.0)
        
        # Final flush
        with self._lock:
            for file_key in list(self.write_queues.keys()):
                self._flush_queue(file_key)
        
        # Close file handles
        for handle in self.file_handles.values():
            try:
                handle.close()
            except:
                pass
        
        logger.info("✅ UniversalJsonLogger shutdown complete")

# Global singleton instance
_universal_logger: Optional[UniversalJsonLogger] = None
_logger_lock = threading.Lock()

def get_universal_logger(config: Optional[LoggingConfig] = None) -> UniversalJsonLogger:
    """Get the global universal JSON logger instance"""
    global _universal_logger
    
    with _logger_lock:
        if _universal_logger is None:
            _universal_logger = UniversalJsonLogger(config)
        return _universal_logger

def log_object_state(obj: Any, name: Optional[str] = None, 
                    custom_metadata: Optional[Dict[str, Any]] = None,
                    tick_number: Optional[int] = None) -> bool:
    """Convenience function to log any object's state"""
    logger_instance = get_universal_logger()
    object_id = logger_instance.register_object(obj, name)
    return logger_instance.log_object_state(object_id, custom_metadata=custom_metadata, tick_number=tick_number)

def register_for_logging(obj: Any, name: Optional[str] = None) -> str:
    """Register an object for automatic logging"""
    return get_universal_logger().register_object(obj, name)

# Automatic registration decorator
def auto_log_state(name: Optional[str] = None, log_on_methods: Optional[List[str]] = None):
    """Decorator to automatically log object state"""
    def decorator(cls):
        original_init = cls.__init__
        
        def new_init(self, *args, **kwargs):
            original_init(self, *args, **kwargs)
            # Register for logging
            register_for_logging(self, name or cls.__name__)
            
            # Hook specified methods to log state
            if log_on_methods:
                for method_name in log_on_methods:
                    if hasattr(self, method_name):
                        original_method = getattr(self, method_name)
                        
                        def logged_method(*method_args, **method_kwargs):
                            result = original_method(*method_args, **method_kwargs)
                            log_object_state(self)
                            return result
                        
                        setattr(self, method_name, logged_method)
        
        cls.__init__ = new_init
        return cls
    
    return decorator

# Context manager for batch logging
class BatchLogging:
    """Context manager for efficient batch logging"""
    
    def __init__(self, tick_number: Optional[int] = None):
        self.tick_number = tick_number
        self.logger = get_universal_logger()
        self.objects_to_log = []
    
    def add_object(self, obj: Any, name: Optional[str] = None):
        """Add object to batch"""
        object_id = self.logger.register_object(obj, name)
        self.objects_to_log.append(object_id)
    
    def __enter__(self):
        return self
    
    def __exit__(self, exc_type, exc_val, exc_tb):
        # Log all objects in batch
        for object_id in self.objects_to_log:
            self.logger.log_object_state(object_id, tick_number=self.tick_number)

if __name__ == "__main__":
    # Example usage and testing
    logging.basicConfig(level=logging.INFO)
    
    # Create test objects
    class TestEngine:
        def __init__(self, name):
            self.name = name
            self.status = "running"
            self.metrics = {"processed": 0, "errors": 0}
        
        def process(self):
            self.metrics["processed"] += 1
            self.status = "processing"
    
    # Initialize universal logger
    logger_instance = get_universal_logger(LoggingConfig(
        format=LogFormat.JSONL,
        scope=StateScope.FULL,
        flush_interval_seconds=0.5
    ))
    
    # Create and register test objects
    engine1 = TestEngine("consciousness_engine")
    engine2 = TestEngine("processing_engine")
    
    register_for_logging(engine1, "consciousness_engine")
    register_for_logging(engine2, "processing_engine")
    
    # Simulate some activity
    for i in range(10):
        engine1.process()
        engine2.process()
        
        # Log states
        log_object_state(engine1, tick_number=i)
        log_object_state(engine2, tick_number=i)
        
        time.sleep(0.1)
    
    # Get statistics
    stats = logger_instance.get_stats()
    print(f"Universal JSON Logging Stats: {json.dumps(stats, indent=2)}")
    
    # Shutdown
    logger_instance.shutdown()
