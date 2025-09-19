#!/usr/bin/env python3
"""
🔍 Enhanced Module Logger - Stable State & Delta Tracking
========================================================

This enhanced logging system provides every DAWN module with:
- Stable state management with persistent logging
- Per-tick JSON dictionary export 
- Movement and delta update tracking
- Comprehensive state change detection
- Historical state comparison and analysis

Every module gets its own stable logging state that tracks all movements,
changes, and delta updates with full JSON export capabilities.
"""

import time
import json
import threading
import copy
from typing import Dict, List, Any, Optional, Tuple, Set, Union
from dataclasses import dataclass, field, asdict
from datetime import datetime
from collections import deque, defaultdict
from enum import Enum
from pathlib import Path
import numpy as np

# Import base telemetry system
try:
    from .system import log_event, log_performance, create_performance_context
    from .logger import TelemetryLevel
    TELEMETRY_AVAILABLE = True
except ImportError:
    TELEMETRY_AVAILABLE = False
    
    # Mock functions
    def log_event(*args, **kwargs): pass
    def log_performance(*args, **kwargs): pass
    def create_performance_context(*args, **kwargs):
        class MockContext:
            def __enter__(self): return self
            def __exit__(self, *args): pass
            def add_metadata(self, key, value): pass
        return MockContext()
    
    class TelemetryLevel:
        DEBUG = "debug"
        INFO = "info"
        WARN = "warn"
        ERROR = "error"
        CRITICAL = "critical"

class StateChangeType(Enum):
    """Types of state changes detected"""
    CREATED = "created"
    UPDATED = "updated"
    DELETED = "deleted"
    MOVED = "moved"
    SCALED = "scaled"
    ROTATED = "rotated"
    TRANSFORMED = "transformed"
    CONNECTED = "connected"
    DISCONNECTED = "disconnected"

@dataclass
class DeltaChange:
    """Represents a delta change in module state"""
    field_path: str
    old_value: Any
    new_value: Any
    change_type: StateChangeType
    timestamp: float
    magnitude: float = 0.0  # Magnitude of change
    
    def to_dict(self) -> Dict[str, Any]:
        return {
            'field_path': self.field_path,
            'old_value': self._serialize_value(self.old_value),
            'new_value': self._serialize_value(self.new_value),
            'change_type': self.change_type.value,
            'timestamp': self.timestamp,
            'magnitude': self.magnitude
        }
    
    def _serialize_value(self, value: Any) -> Any:
        """Serialize value for JSON export"""
        if isinstance(value, (int, float, str, bool, type(None))):
            return value
        elif isinstance(value, (list, tuple)):
            return list(value)
        elif isinstance(value, dict):
            return value
        elif hasattr(value, '__dict__'):
            return str(value)
        else:
            return str(value)

@dataclass
class MovementVector:
    """Represents movement in multidimensional space"""
    dimensions: Dict[str, float] = field(default_factory=dict)
    velocity: Dict[str, float] = field(default_factory=dict)
    acceleration: Dict[str, float] = field(default_factory=dict)
    timestamp: float = field(default_factory=time.time)
    
    def calculate_magnitude(self) -> float:
        """Calculate movement magnitude across all dimensions"""
        return np.sqrt(sum(v**2 for v in self.dimensions.values()))
    
    def to_dict(self) -> Dict[str, Any]:
        return {
            'dimensions': self.dimensions,
            'velocity': self.velocity,
            'acceleration': self.acceleration,
            'magnitude': self.calculate_magnitude(),
            'timestamp': self.timestamp
        }

@dataclass
class StableModuleState:
    """Stable state container for a module with full tracking"""
    module_name: str
    subsystem: str
    current_state: Dict[str, Any] = field(default_factory=dict)
    previous_state: Dict[str, Any] = field(default_factory=dict)
    state_history: deque = field(default_factory=lambda: deque(maxlen=100))
    delta_changes: deque = field(default_factory=lambda: deque(maxlen=1000))
    movement_history: deque = field(default_factory=lambda: deque(maxlen=100))
    
    # Tracking metrics
    tick_count: int = 0
    last_update_time: float = field(default_factory=time.time)
    total_changes: int = 0
    significant_changes: int = 0
    
    # Position tracking
    position: Dict[str, float] = field(default_factory=dict)
    last_position: Dict[str, float] = field(default_factory=dict)
    
    # Performance metrics
    update_times: deque = field(default_factory=lambda: deque(maxlen=100))
    avg_update_time: float = 0.0
    
    def update_state(self, new_state: Dict[str, Any], tick_number: Optional[int] = None) -> List[DeltaChange]:
        """Update state and track all changes"""
        update_start = time.time()
        
        # Store previous state
        self.previous_state = copy.deepcopy(self.current_state)
        
        # Detect changes
        changes = self._detect_changes(self.current_state, new_state)
        
        # Update current state
        self.current_state = copy.deepcopy(new_state)
        
        # Update tracking metrics
        if tick_number is not None:
            self.tick_count = tick_number
        else:
            self.tick_count += 1
            
        self.last_update_time = time.time()
        self.total_changes += len(changes)
        self.significant_changes += len([c for c in changes if c.magnitude > 0.1])
        
        # Store in history
        self.state_history.append({
            'tick': self.tick_count,
            'timestamp': self.last_update_time,
            'state': copy.deepcopy(new_state),
            'changes_count': len(changes)
        })
        
        # Store delta changes
        for change in changes:
            self.delta_changes.append(change)
        
        # Track movement if position data available
        self._track_movement()
        
        # Update performance metrics
        update_time = time.time() - update_start
        self.update_times.append(update_time)
        self.avg_update_time = np.mean(self.update_times) if self.update_times else 0.0
        
        return changes
    
    def _detect_changes(self, old_state: Dict[str, Any], new_state: Dict[str, Any], 
                       path: str = "") -> List[DeltaChange]:
        """Detect all changes between states"""
        changes = []
        current_time = time.time()
        
        # Check for new or changed fields
        for key, new_value in new_state.items():
            current_path = f"{path}.{key}" if path else key
            
            if key not in old_state:
                # New field created
                changes.append(DeltaChange(
                    field_path=current_path,
                    old_value=None,
                    new_value=new_value,
                    change_type=StateChangeType.CREATED,
                    timestamp=current_time,
                    magnitude=self._calculate_change_magnitude(None, new_value)
                ))
            elif old_state[key] != new_value:
                # Field updated
                if isinstance(new_value, dict) and isinstance(old_state[key], dict):
                    # Recursively check nested dictionaries
                    changes.extend(self._detect_changes(old_state[key], new_value, current_path))
                else:
                    changes.append(DeltaChange(
                        field_path=current_path,
                        old_value=old_state[key],
                        new_value=new_value,
                        change_type=self._determine_change_type(old_state[key], new_value),
                        timestamp=current_time,
                        magnitude=self._calculate_change_magnitude(old_state[key], new_value)
                    ))
        
        # Check for deleted fields
        for key, old_value in old_state.items():
            if key not in new_state:
                current_path = f"{path}.{key}" if path else key
                changes.append(DeltaChange(
                    field_path=current_path,
                    old_value=old_value,
                    new_value=None,
                    change_type=StateChangeType.DELETED,
                    timestamp=current_time,
                    magnitude=self._calculate_change_magnitude(old_value, None)
                ))
        
        return changes
    
    def _determine_change_type(self, old_value: Any, new_value: Any) -> StateChangeType:
        """Determine the type of change based on values"""
        if isinstance(old_value, (int, float)) and isinstance(new_value, (int, float)):
            if abs(new_value - old_value) > abs(old_value) * 0.5:  # Large scale change
                return StateChangeType.SCALED
            else:
                return StateChangeType.UPDATED
        elif isinstance(old_value, dict) and isinstance(new_value, dict):
            if 'position' in str(old_value) or 'position' in str(new_value):
                return StateChangeType.MOVED
            elif 'connection' in str(old_value) or 'connection' in str(new_value):
                return StateChangeType.CONNECTED
            else:
                return StateChangeType.TRANSFORMED
        else:
            return StateChangeType.UPDATED
    
    def _calculate_change_magnitude(self, old_value: Any, new_value: Any) -> float:
        """Calculate magnitude of change"""
        if old_value is None and new_value is not None:
            return 1.0  # Creation
        elif old_value is not None and new_value is None:
            return 1.0  # Deletion
        elif isinstance(old_value, (int, float)) and isinstance(new_value, (int, float)):
            if old_value == 0:
                return abs(new_value)
            else:
                return abs((new_value - old_value) / old_value)
        elif isinstance(old_value, str) and isinstance(new_value, str):
            return 1.0 if old_value != new_value else 0.0
        else:
            return 0.5  # Default moderate change
    
    def _track_movement(self):
        """Track movement if position data is available"""
        position_fields = ['x', 'y', 'z', 'position', 'location', 'coordinates']
        
        current_position = {}
        for field in position_fields:
            if field in self.current_state:
                value = self.current_state[field]
                if isinstance(value, (int, float)):
                    current_position[field] = float(value)
                elif isinstance(value, (list, tuple)) and len(value) >= 2:
                    for i, coord in enumerate(value[:3]):
                        current_position[f'{field}_{i}'] = float(coord)
        
        if current_position:
            # Calculate movement vector
            movement = MovementVector()
            
            if self.last_position:
                for dim, pos in current_position.items():
                    if dim in self.last_position:
                        movement.dimensions[dim] = pos - self.last_position[dim]
                        # Simple velocity calculation
                        if len(self.movement_history) > 0:
                            last_movement = self.movement_history[-1]
                            time_delta = movement.timestamp - last_movement.timestamp
                            if time_delta > 0:
                                movement.velocity[dim] = movement.dimensions[dim] / time_delta
            
            self.movement_history.append(movement)
            self.last_position = current_position
    
    def get_tick_json(self) -> Dict[str, Any]:
        """Generate comprehensive JSON for current tick"""
        recent_changes = [c.to_dict() for c in list(self.delta_changes)[-10:]]  # Last 10 changes
        recent_movement = self.movement_history[-1].to_dict() if self.movement_history else None
        
        return {
            'module_info': {
                'name': self.module_name,
                'subsystem': self.subsystem,
                'tick': self.tick_count,
                'timestamp': self.last_update_time
            },
            'current_state': copy.deepcopy(self.current_state),
            'state_summary': {
                'total_fields': len(self.current_state),
                'total_changes': self.total_changes,
                'significant_changes': self.significant_changes,
                'change_rate': self.total_changes / max(self.tick_count, 1)
            },
            'recent_changes': recent_changes,
            'movement': recent_movement,
            'performance': {
                'avg_update_time_ms': self.avg_update_time * 1000,
                'state_history_size': len(self.state_history),
                'delta_changes_tracked': len(self.delta_changes)
            },
            'position_tracking': {
                'current_position': self.position,
                'last_position': self.last_position,
                'movement_history_size': len(self.movement_history)
            }
        }
    
    def get_delta_summary(self, last_n_ticks: int = 10) -> Dict[str, Any]:
        """Get summary of recent delta changes"""
        recent_changes = list(self.delta_changes)[-last_n_ticks * 10:]  # Approximate
        
        change_types = defaultdict(int)
        field_changes = defaultdict(int)
        total_magnitude = 0.0
        
        for change in recent_changes:
            change_types[change.change_type.value] += 1
            field_changes[change.field_path] += 1
            total_magnitude += change.magnitude
        
        return {
            'total_changes': len(recent_changes),
            'change_types': dict(change_types),
            'most_changed_fields': dict(sorted(field_changes.items(), 
                                             key=lambda x: x[1], reverse=True)[:5]),
            'average_magnitude': total_magnitude / len(recent_changes) if recent_changes else 0.0,
            'time_range': {
                'start': recent_changes[0].timestamp if recent_changes else None,
                'end': recent_changes[-1].timestamp if recent_changes else None
            }
        }

class EnhancedModuleLogger:
    """Enhanced module logger with stable state management"""
    
    def __init__(self, module_name: str, subsystem: str, 
                 json_export_path: Optional[str] = None):
        self.module_name = module_name
        self.subsystem = subsystem
        self.json_export_path = json_export_path or f"logs/{subsystem}_{module_name}_states.json"
        
        # Stable state management
        self.stable_state = StableModuleState(module_name, subsystem)
        
        # Thread safety
        self._lock = threading.RLock()
        
        # Export configuration
        self.auto_export = True
        self.export_every_n_ticks = 10
        
        # Ensure export directory exists
        Path(self.json_export_path).parent.mkdir(parents=True, exist_ok=True)
        
        # Initialize with telemetry
        if TELEMETRY_AVAILABLE:
            log_event(subsystem, module_name, 'enhanced_logger_init', 
                     TelemetryLevel.INFO, {
                         'module_name': module_name,
                         'subsystem': subsystem,
                         'export_path': self.json_export_path
                     })
    
    def log_tick_update(self, state_data: Dict[str, Any], tick_number: Optional[int] = None,
                       export_json: bool = True) -> Dict[str, Any]:
        """Log a complete tick update with delta tracking"""
        with self._lock:
            with create_performance_context(self.subsystem, self.module_name, 'tick_update') as perf_ctx:
                # Update stable state and get changes
                changes = self.stable_state.update_state(state_data, tick_number)
                
                perf_ctx.add_metadata('changes_detected', len(changes))
                perf_ctx.add_metadata('tick_number', self.stable_state.tick_count)
                
                # Generate tick JSON
                tick_json = self.stable_state.get_tick_json()
                
                # Log to telemetry system
                if TELEMETRY_AVAILABLE:
                    log_event(self.subsystem, self.module_name, 'tick_update', 
                             TelemetryLevel.DEBUG, {
                                 'tick': self.stable_state.tick_count,
                                 'changes_count': len(changes),
                                 'significant_changes': len([c for c in changes if c.magnitude > 0.1]),
                                 'state_fields': len(state_data),
                                 'movement_detected': len(self.stable_state.movement_history) > 0
                             })
                    
                    # Log significant changes
                    for change in changes:
                        if change.magnitude > 0.1:  # Only log significant changes
                            log_event(self.subsystem, self.module_name, 'significant_change',
                                     TelemetryLevel.INFO, {
                                         'field': change.field_path,
                                         'change_type': change.change_type.value,
                                         'magnitude': change.magnitude,
                                         'old_value': change._serialize_value(change.old_value),
                                         'new_value': change._serialize_value(change.new_value)
                                     })
                
                # Export JSON if requested
                if export_json and self.auto_export:
                    if self.stable_state.tick_count % self.export_every_n_ticks == 0:
                        self._export_tick_json(tick_json)
                
                return tick_json
    
    def log_movement(self, position_data: Dict[str, float], 
                    movement_type: str = "position_update"):
        """Log movement with delta tracking"""
        with self._lock:
            # Update position in stable state
            self.stable_state.position.update(position_data)
            
            # Log movement event
            if TELEMETRY_AVAILABLE:
                log_event(self.subsystem, self.module_name, 'movement', 
                         TelemetryLevel.DEBUG, {
                             'movement_type': movement_type,
                             'position': position_data,
                             'tick': self.stable_state.tick_count
                         })
    
    def log_delta_change(self, field_path: str, old_value: Any, new_value: Any,
                        change_type: StateChangeType = StateChangeType.UPDATED):
        """Manually log a delta change"""
        with self._lock:
            change = DeltaChange(
                field_path=field_path,
                old_value=old_value,
                new_value=new_value,
                change_type=change_type,
                timestamp=time.time(),
                magnitude=self.stable_state._calculate_change_magnitude(old_value, new_value)
            )
            
            self.stable_state.delta_changes.append(change)
            
            if TELEMETRY_AVAILABLE:
                log_event(self.subsystem, self.module_name, 'manual_delta_change',
                         TelemetryLevel.DEBUG, change.to_dict())
    
    def get_current_json(self) -> Dict[str, Any]:
        """Get current state as JSON"""
        with self._lock:
            return self.stable_state.get_tick_json()
    
    def get_delta_summary(self, last_n_ticks: int = 10) -> Dict[str, Any]:
        """Get delta change summary"""
        with self._lock:
            return self.stable_state.get_delta_summary(last_n_ticks)
    
    def export_full_state(self, filename: Optional[str] = None) -> str:
        """Export complete state history to JSON file"""
        with self._lock:
            export_path = filename or f"{self.json_export_path.replace('.json', '_full.json')}"
            
            export_data = {
                'module_info': {
                    'name': self.module_name,
                    'subsystem': self.subsystem,
                    'export_timestamp': time.time()
                },
                'current_state': self.stable_state.get_tick_json(),
                'state_history': list(self.stable_state.state_history),
                'delta_changes': [c.to_dict() for c in self.stable_state.delta_changes],
                'movement_history': [m.to_dict() for m in self.stable_state.movement_history],
                'performance_metrics': {
                    'total_ticks': self.stable_state.tick_count,
                    'total_changes': self.stable_state.total_changes,
                    'significant_changes': self.stable_state.significant_changes,
                    'avg_update_time': self.stable_state.avg_update_time
                }
            }
            
            try:
                with open(export_path, 'w') as f:
                    json.dump(export_data, f, indent=2, default=str)
                
                if TELEMETRY_AVAILABLE:
                    log_event(self.subsystem, self.module_name, 'full_state_export',
                             TelemetryLevel.INFO, {
                                 'export_path': export_path,
                                 'total_ticks': self.stable_state.tick_count,
                                 'total_changes': self.stable_state.total_changes
                             })
                
                return export_path
            except Exception as e:
                if TELEMETRY_AVAILABLE:
                    log_event(self.subsystem, self.module_name, 'export_error',
                             TelemetryLevel.ERROR, {'error': str(e)})
                raise
    
    def _export_tick_json(self, tick_json: Dict[str, Any]):
        """Export tick JSON to file"""
        try:
            # Load existing data
            existing_data = []
            if Path(self.json_export_path).exists():
                with open(self.json_export_path, 'r') as f:
                    existing_data = json.load(f)
            
            # Append new tick data
            existing_data.append(tick_json)
            
            # Keep only last 1000 ticks
            if len(existing_data) > 1000:
                existing_data = existing_data[-1000:]
            
            # Save updated data
            with open(self.json_export_path, 'w') as f:
                json.dump(existing_data, f, indent=2, default=str)
                
        except Exception as e:
            if TELEMETRY_AVAILABLE:
                log_event(self.subsystem, self.module_name, 'json_export_error',
                         TelemetryLevel.WARN, {'error': str(e)})

# Global registry of enhanced loggers
_module_loggers: Dict[str, EnhancedModuleLogger] = {}
_logger_lock = threading.Lock()

def get_enhanced_logger(module_name: str, subsystem: str, 
                       json_export_path: Optional[str] = None) -> EnhancedModuleLogger:
    """Get or create enhanced logger for a module"""
    logger_key = f"{subsystem}.{module_name}"
    
    with _logger_lock:
        if logger_key not in _module_loggers:
            _module_loggers[logger_key] = EnhancedModuleLogger(
                module_name, subsystem, json_export_path
            )
        
        return _module_loggers[logger_key]

def log_module_tick(module_name: str, subsystem: str, state_data: Dict[str, Any],
                   tick_number: Optional[int] = None) -> Dict[str, Any]:
    """Convenience function to log module tick update"""
    logger = get_enhanced_logger(module_name, subsystem)
    return logger.log_tick_update(state_data, tick_number)

def export_all_module_states(output_dir: str = "logs/module_exports") -> List[str]:
    """Export all module states to files"""
    Path(output_dir).mkdir(parents=True, exist_ok=True)
    exported_files = []
    
    with _logger_lock:
        for logger_key, logger in _module_loggers.items():
            filename = f"{output_dir}/{logger_key}_complete_state.json"
            try:
                exported_file = logger.export_full_state(filename)
                exported_files.append(exported_file)
            except Exception as e:
                print(f"Failed to export {logger_key}: {e}")
    
    return exported_files
