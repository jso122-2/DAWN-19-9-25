"""
DAWN Core Schema System Implementation
=====================================
Comprehensive schema management and health monitoring system for DAWN consciousness.

This module provides the central SchemaSystem class that coordinates:
- Schema state management and parsing
- SHI (Schema Health Index) calculation
- SCUP (Semantic Coherence Under Pressure) tracking
- Real-time validation and monitoring
- Integration with other DAWN subsystems

Author: DAWN Development Team
Generated: 2025-09-18
"""

import time
import math
import json
import threading
from typing import Dict, List, Optional, Tuple, Any, Callable
from dataclasses import dataclass, field
from datetime import datetime, timedelta
from enum import Enum
from pathlib import Path
from collections import deque
import numpy as np

# Telemetry system imports
try:
    from dawn.core.telemetry.system import (
        log_event, log_performance, log_error, create_performance_context
    )
    from dawn.core.telemetry.logger import TelemetryLevel
    TELEMETRY_AVAILABLE = True
except ImportError:
    TELEMETRY_AVAILABLE = False
    
    # Mock telemetry functions
    def log_event(*args, **kwargs): pass
    def log_performance(*args, **kwargs): pass
    def log_error(*args, **kwargs): pass
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

# Import torch with fallback for systems without PyTorch
try:
    import torch
    import torch.nn as nn
    import torch.nn.functional as F
    TORCH_AVAILABLE = True
except ImportError:
    TORCH_AVAILABLE = False
    print("[SchemaSystem] Warning: PyTorch not available, using numpy fallback")

# DAWN imports
try:
    from ..mycelial.core import MycelialLayer
    from ..thermal.pulse.unified_pulse_heat import UnifiedPulseHeat
    from .shi_calculator import SHICalculator, HealthStatus, HealthComponent
    from .scup_tracker import SCUPTracker, SCUPState
    from .schema_validator import SchemaValidator
    from .schema_monitor import SchemaMonitor
except ImportError:
    # Fallback for standalone testing
    pass


class SchemaMode(Enum):
    """Schema operational modes"""
    NORMAL = "normal"
    MYTHIC = "mythic"
    RECOVERY = "recovery"
    EMERGENCY = "emergency"


class SchemaFlags:
    """Schema state flags and overrides"""
    
    def __init__(self):
        self.suppression_active = False
        self.override_trigger = None
        self.mythic_mode = False
        self.emergency_brake = False
        self.recovery_active = False
        
    def reset(self):
        """Reset all flags to default state"""
        self.suppression_active = False
        self.override_trigger = None
        self.mythic_mode = False
        self.emergency_brake = False
        self.recovery_active = False
        
    def get_active_flags(self) -> Dict[str, bool]:
        """Get all active flags"""
        return {
            'suppression': self.suppression_active,
            'override': self.override_trigger is not None,
            'mythic': self.mythic_mode,
            'emergency': self.emergency_brake,
            'recovery': self.recovery_active
        }


@dataclass
class SchemaNode:
    """Individual schema node representation"""
    id: str
    health: float = 0.5
    tint: List[float] = field(default_factory=lambda: [0.33, 0.33, 0.34])
    energy: float = 0.5
    connections: List[str] = field(default_factory=list)
    last_accessed: float = field(default_factory=time.time)
    
    def to_dict(self) -> Dict[str, Any]:
        return {
            "id": self.id,
            "health": self.health,
            "tint": self.tint,
            "energy": self.energy,
            "connections": len(self.connections),
            "last_accessed": self.last_accessed
        }


@dataclass
class SchemaEdge:
    """Schema edge representation"""
    id: str
    source: str
    target: str
    weight: float = 0.5
    tension: float = 0.0
    entropy: float = 0.0
    conductivity: float = 0.5
    last_flow: float = 0.0
    
    def to_dict(self) -> Dict[str, Any]:
        return {
            "id": self.id,
            "source": self.source,
            "target": self.target,
            "weight": self.weight,
            "tension": self.tension,
            "entropy": self.entropy,
            "conductivity": self.conductivity
        }


@dataclass
class ResidueMetrics:
    """Residue balance tracking"""
    soot_ratio: float = 0.0
    ash_bias: List[float] = field(default_factory=lambda: [0.33, 0.33, 0.34])
    total_residue: float = 0.0
    conversion_rate: float = 0.0
    
    def get_health_impact(self) -> float:
        """Calculate residue impact on schema health"""
        # High soot = negative impact, balanced ash = positive
        soot_penalty = self.soot_ratio * 0.3
        ash_balance = 1.0 - np.std(self.ash_bias) if len(self.ash_bias) > 0 else 0.5
        ash_bonus = ash_balance * 0.2
        
        return max(0.0, min(1.0, ash_bonus - soot_penalty))


@dataclass
class TracerCounts:
    """Tracer activity tracking"""
    crow: int = 0
    spider: int = 0
    ant: int = 0
    whale: int = 0
    owl: int = 0
    bee: int = 0
    beetle: int = 0
    
    def get_divergence_score(self) -> float:
        """Calculate tracer divergence for SHI"""
        total = sum([self.crow, self.spider, self.ant, self.whale, self.owl, self.bee, self.beetle])
        if total == 0:
            return 0.0
            
        # Calculate variance in tracer activity
        counts = [self.crow, self.spider, self.ant, self.whale, self.owl, self.bee, self.beetle]
        mean_activity = total / 7
        variance = np.var(counts) / (mean_activity + 0.001)  # Avoid division by zero
        
        return min(1.0, variance / 10.0)  # Normalize to [0,1]


@dataclass
class SchemaSnapshot:
    """Complete schema state snapshot"""
    tick: int
    timestamp: float
    nodes: List[SchemaNode]
    edges: List[SchemaEdge]
    blooms: Dict[str, Any]
    residue: ResidueMetrics
    tracers: TracerCounts
    shi: float
    scup: float
    signals: Dict[str, float]
    flags: Dict[str, bool]
    
    def to_json(self) -> Dict[str, Any]:
        """Export schema snapshot for GUI/monitoring"""
        return {
            "tick": self.tick,
            "timestamp": self.timestamp,
            "nodes": [node.to_dict() for node in self.nodes],
            "edges": [edge.to_dict() for edge in self.edges],
            "blooms": self.blooms,
            "residue": {
                "soot_ratio": self.residue.soot_ratio,
                "ash_bias": self.residue.ash_bias,
                "total_residue": self.residue.total_residue
            },
            "tracers": {
                "crow": self.tracers.crow,
                "spider": self.tracers.spider,
                "ant": self.tracers.ant,
                "whale": self.tracers.whale,
                "owl": self.tracers.owl,
                "bee": self.tracers.bee,
                "beetle": self.tracers.beetle
            },
            "shi": self.shi,
            "scup": self.scup,
            "signals": self.signals,
            "flags": self.flags
        }


class SchemaState:
    """Central schema state management"""
    
    def __init__(self):
        self.nodes: Dict[str, SchemaNode] = {}
        self.edges: Dict[str, SchemaEdge] = {}
        self.mode = SchemaMode.NORMAL
        self.flags = SchemaFlags()
        
        # History tracking
        self.snapshot_history = deque(maxlen=1000)
        self.health_history = deque(maxlen=1000)
        self.event_log = deque(maxlen=500)
        
        # Thread safety
        self.lock = threading.RLock()
        
    def add_node(self, node_id: str, health: float = 0.5, 
                 tint: Optional[List[float]] = None, energy: float = 0.5) -> SchemaNode:
        """Add a new schema node"""
        with self.lock:
            if tint is None:
                tint = [0.33, 0.33, 0.34]
                
            node = SchemaNode(
                id=node_id,
                health=health,
                tint=tint,
                energy=energy
            )
            
            self.nodes[node_id] = node
            self._log_event("node_added", {"node_id": node_id, "health": health})
            return node
    
    def add_edge(self, edge_id: str, source: str, target: str, 
                 weight: float = 0.5, tension: float = 0.0) -> SchemaEdge:
        """Add a new schema edge"""
        with self.lock:
            edge = SchemaEdge(
                id=edge_id,
                source=source,
                target=target,
                weight=weight,
                tension=tension
            )
            
            self.edges[edge_id] = edge
            
            # Update node connections
            if source in self.nodes:
                self.nodes[source].connections.append(target)
            if target in self.nodes:
                self.nodes[target].connections.append(source)
                
            self._log_event("edge_added", {"edge_id": edge_id, "source": source, "target": target})
            return edge
    
    def update_node_health(self, node_id: str, health: float):
        """Update node health value"""
        with self.lock:
            if node_id in self.nodes:
                self.nodes[node_id].health = max(0.0, min(1.0, health))
                self.nodes[node_id].last_accessed = time.time()
    
    def update_edge_tension(self, edge_id: str, tension: float):
        """Update edge tension value"""
        with self.lock:
            if edge_id in self.edges:
                self.edges[edge_id].tension = max(0.0, min(1.0, tension))
    
    def get_average_health(self) -> float:
        """Calculate average node health"""
        with self.lock:
            if not self.nodes:
                return 0.5
            return np.mean([node.health for node in self.nodes.values()])
    
    def get_average_tension(self) -> float:
        """Calculate average edge tension"""
        with self.lock:
            if not self.edges:
                return 0.0
            return np.mean([edge.tension for edge in self.edges.values()])
    
    def _log_event(self, event_type: str, data: Dict[str, Any]):
        """Log schema events"""
        event = {
            "timestamp": time.time(),
            "type": event_type,
            "data": data
        }
        self.event_log.append(event)


class SchemaSystem:
    """
    Core Schema System for DAWN consciousness
    
    Provides comprehensive schema management including:
    - State parsing and coordination
    - SHI calculation and health monitoring  
    - SCUP tracking and pressure management
    - Real-time validation and anomaly detection
    - Integration with other DAWN subsystems
    """
    
    def __init__(self, 
                 enable_shi_calculation: bool = True,
                 enable_scup_tracking: bool = True,
                 enable_validation: bool = True,
                 enable_monitoring: bool = True,
                 vault_path: Optional[str] = None):
        
        # Log schema system initialization
        if TELEMETRY_AVAILABLE:
            log_event('schema_system', 'initialization', 'system_init_start', 
                     TelemetryLevel.INFO, {
                         'enable_shi_calculation': enable_shi_calculation,
                         'enable_scup_tracking': enable_scup_tracking,
                         'enable_validation': enable_validation,
                         'enable_monitoring': enable_monitoring,
                         'vault_path': vault_path
                     })
        
        # Core components
        self.state = SchemaState()
        self.shi_calculator = SHICalculator() if enable_shi_calculation else None
        self.scup_tracker = SCUPTracker() if enable_scup_tracking else None
        self.validator = SchemaValidator() if enable_validation else None
        self.monitor = SchemaMonitor() if enable_monitoring else None
        
        # Configuration
        self.vault_path = vault_path
        self.tick_count = 0
        self.last_shi = 0.5
        self.last_scup = 0.5
        
        # Performance tracking
        self.processing_times = deque(maxlen=100)
        self.error_count = 0
        
        # Thread safety
        self.lock = threading.RLock()
        
        # Log successful initialization
        if TELEMETRY_AVAILABLE:
            log_event('schema_system', 'initialization', 'system_init_complete', 
                     TelemetryLevel.INFO, {
                         'components_initialized': {
                             'shi_calculator': self.shi_calculator is not None,
                             'scup_tracker': self.scup_tracker is not None,
                             'validator': self.validator is not None,
                             'monitor': self.monitor is not None
                         },
                         'initial_shi': self.last_shi,
                         'initial_scup': self.last_scup
                     })
        
        print("[SchemaSystem] 🧠 Core schema system initialized")
        
    def tick_update(self, 
                   tick_data: Dict[str, Any],
                   external_pressures: Optional[Dict[str, float]] = None,
                   consciousness_state: Optional[Dict[str, float]] = None) -> SchemaSnapshot:
        """
        Main tick update processing
        
        Args:
            tick_data: Raw tick data from DAWN systems
            external_pressures: External pressure sources
            consciousness_state: Current consciousness state
            
        Returns:
            SchemaSnapshot: Complete schema state snapshot
        """
        start_time = time.time()
        
        # Log tick update start
        if TELEMETRY_AVAILABLE:
            log_event('schema_system', 'tick_update', 'tick_start', 
                     TelemetryLevel.DEBUG, {
                         'tick_count': self.tick_count + 1,
                         'tick_data_keys': list(tick_data.keys()) if tick_data else [],
                         'external_pressures': external_pressures,
                         'consciousness_state_keys': list(consciousness_state.keys()) if consciousness_state else [],
                         'current_shi': self.last_shi,
                         'current_scup': self.last_scup
                     })
        
        try:
            with self.lock:
                with create_performance_context('schema_system', 'tick_update', 'schema_processing') as perf_ctx:
                    self.tick_count += 1
                    perf_ctx.add_metadata('tick_count', self.tick_count)
                    perf_ctx.add_metadata('nodes_count', len(self.state.nodes))
                    perf_ctx.add_metadata('edges_count', len(self.state.edges))
                    
                    # Parse tick data into structured format
                    parsed_state = self._parse_tick_data(tick_data)
                    perf_ctx.add_metadata('parsed_signals', list(parsed_state['signals'].keys()))
                    
                    # Update schema state
                    self._update_schema_state(parsed_state)
                    
                    # Calculate SHI if enabled
                    shi = self._calculate_shi(parsed_state) if self.shi_calculator else self.last_shi
                    perf_ctx.add_metadata('shi_calculated', shi)
                    
                    # Update SCUP if enabled
                    scup_result = self._update_scup(parsed_state, external_pressures) if self.scup_tracker else {'scup': self.last_scup}
                    perf_ctx.add_metadata('scup_calculated', scup_result['scup'])
                    
                    # Validate schema if enabled
                    validation_results = self._validate_schema(parsed_state) if self.validator else None
                    if validation_results:
                        perf_ctx.add_metadata('validation_passed', validation_results.get('valid', True))
                    
                    # Monitor for anomalies if enabled
                    monitoring_results = self._monitor_schema(parsed_state) if self.monitor else None
                    if monitoring_results:
                        perf_ctx.add_metadata('anomalies_detected', monitoring_results.get('anomaly_count', 0))
                    
                    # Generate schema snapshot
                    snapshot = self._generate_schema_snapshot(
                        parsed_state, shi, scup_result['scup']
                    )
                    
                    # Update histories
                    self.state.snapshot_history.append(snapshot)
                    self.state.health_history.append(shi)
                    
                    # Store last values
                    self.last_shi = shi
                    self.last_scup = scup_result['scup']
                    
                    # Track processing time
                    processing_time = time.time() - start_time
                    self.processing_times.append(processing_time)
                    
                    # Log successful tick completion
                    if TELEMETRY_AVAILABLE:
                        log_event('schema_system', 'tick_update', 'tick_complete', 
                                 TelemetryLevel.DEBUG, {
                                     'tick_count': self.tick_count,
                                     'processing_time_ms': processing_time * 1000,
                                     'shi_value': shi,
                                     'scup_value': scup_result['scup'],
                                     'nodes_processed': len(snapshot.nodes),
                                     'edges_processed': len(snapshot.edges),
                                     'signals_processed': len(snapshot.signals),
                                     'validation_enabled': self.validator is not None,
                                     'monitoring_enabled': self.monitor is not None
                                 })
                    
                    return snapshot
                
        except Exception as e:
            self.error_count += 1
            processing_time = time.time() - start_time
            
            # Log schema processing error
            if TELEMETRY_AVAILABLE:
                log_error('schema_system', 'tick_update', e, {
                    'tick_count': self.tick_count,
                    'processing_time_ms': processing_time * 1000,
                    'error_count': self.error_count,
                    'nodes_count': len(self.state.nodes),
                    'edges_count': len(self.state.edges)
                })
            
            print(f"[SchemaSystem] ❌ Error in tick_update: {str(e)}")
            
            # Return emergency snapshot
            return self._generate_emergency_snapshot()
    
    def _parse_tick_data(self, tick_data: Dict[str, Any]) -> Dict[str, Any]:
        """Parse raw tick data into structured schema format"""
        parsed = {
            'timestamp': time.time(),
            'tick': self.tick_count,
            'signals': {},
            'blooms': {},
            'tracers': TracerCounts(),
            'residue': ResidueMetrics(),
            'pressures': {},
            'entropy_data': {}
        }
        
        # Extract signals
        if 'signals' in tick_data:
            parsed['signals'] = tick_data['signals']
        else:
            # Default signals
            parsed['signals'] = {
                'pressure': 0.5,
                'drift': 0.3,
                'entropy': 0.4
            }
        
        # Extract bloom data
        if 'blooms' in tick_data:
            parsed['blooms'] = tick_data['blooms']
        
        # Extract tracer counts
        if 'tracers' in tick_data:
            tracer_data = tick_data['tracers']
            parsed['tracers'] = TracerCounts(
                crow=tracer_data.get('crow', 0),
                spider=tracer_data.get('spider', 0),
                ant=tracer_data.get('ant', 0),
                whale=tracer_data.get('whale', 0),
                owl=tracer_data.get('owl', 0),
                bee=tracer_data.get('bee', 0),
                beetle=tracer_data.get('beetle', 0)
            )
        
        # Extract residue data
        if 'residue' in tick_data:
            residue_data = tick_data['residue']
            parsed['residue'] = ResidueMetrics(
                soot_ratio=residue_data.get('soot_ratio', 0.0),
                ash_bias=residue_data.get('ash_bias', [0.33, 0.33, 0.34]),
                total_residue=residue_data.get('total_residue', 0.0)
            )
        
        return parsed
    
    def _update_schema_state(self, parsed_state: Dict[str, Any]):
        """Update internal schema state based on parsed data"""
        # Update node healths based on signals
        for node_id, node in self.state.nodes.items():
            # Decay health slightly over time
            health_decay = 0.001
            new_health = max(0.0, node.health - health_decay)
            
            # Apply signal-based adjustments
            if 'pressure' in parsed_state['signals']:
                pressure = parsed_state['signals']['pressure']
                # High pressure reduces health
                pressure_impact = -pressure * 0.01
                new_health += pressure_impact
            
            self.state.update_node_health(node_id, new_health)
        
        # Update edge tensions based on drift
        for edge_id, edge in self.state.edges.items():
            # Base tension from drift
            drift = parsed_state['signals'].get('drift', 0.0)
            tension_adjustment = drift * 0.02
            
            new_tension = max(0.0, min(1.0, edge.tension + tension_adjustment))
            self.state.update_edge_tension(edge_id, new_tension)
    
    def _calculate_shi(self, parsed_state: Dict[str, Any]) -> float:
        """Calculate Schema Health Index"""
        if not self.shi_calculator:
            return self.last_shi
        
        # Extract components for SHI calculation
        sigil_entropy = parsed_state['signals'].get('entropy', 0.0)
        edge_volatility = self.state.get_average_tension()
        tracer_divergence = parsed_state['tracers'].get_divergence_score()
        scup_value = self.last_scup
        residue_balance = parsed_state['residue'].get_health_impact()
        
        # Calculate SHI
        shi = self.shi_calculator.calculate_shi(
            sigil_entropy=sigil_entropy,
            edge_volatility=edge_volatility,
            tracer_divergence=tracer_divergence,
            scup_value=scup_value,
            residue_balance=residue_balance
        )
        
        return shi
    
    def _update_scup(self, parsed_state: Dict[str, Any], 
                    external_pressures: Optional[Dict[str, float]]) -> Dict[str, Any]:
        """Update SCUP tracking"""
        if not self.scup_tracker:
            return {'scup': self.last_scup}
        
        # Extract SCUP inputs
        alignment = 1.0 - parsed_state['signals'].get('drift', 0.0)
        entropy = parsed_state['signals'].get('entropy', 0.0)
        pressure = parsed_state['signals'].get('pressure', 0.0)
        
        # Add external pressures
        if external_pressures:
            total_pressure = sum(external_pressures.values())
            pressure = min(1.0, pressure + total_pressure * 0.1)
        
        # Compute SCUP
        scup_result = self.scup_tracker.compute_scup(
            alignment=alignment,
            entropy=entropy,
            pressure=pressure,
            method="auto"
        )
        
        return scup_result
    
    def _validate_schema(self, parsed_state: Dict[str, Any]) -> Dict[str, Any]:
        """Validate schema state"""
        if not self.validator:
            return {}
        
        # Create validation snapshot
        validation_data = {
            'nodes': list(self.state.nodes.values()),
            'edges': list(self.state.edges.values()),
            'signals': parsed_state['signals'],
            'shi': self.last_shi,
            'scup': self.last_scup
        }
        
        return self.validator.validate_schema_snapshot(validation_data)
    
    def _monitor_schema(self, parsed_state: Dict[str, Any]) -> Dict[str, Any]:
        """Monitor schema for anomalies"""
        if not self.monitor:
            return {}
        
        # Create monitoring data
        monitoring_data = {
            'schema_state': self.state,
            'parsed_state': parsed_state,
            'shi': self.last_shi,
            'scup': self.last_scup,
            'tick': self.tick_count
        }
        
        return self.monitor.monitor_real_time(monitoring_data)
    
    def _generate_schema_snapshot(self, parsed_state: Dict[str, Any], 
                                shi: float, scup: float) -> SchemaSnapshot:
        """Generate complete schema snapshot"""
        return SchemaSnapshot(
            tick=self.tick_count,
            timestamp=time.time(),
            nodes=list(self.state.nodes.values()),
            edges=list(self.state.edges.values()),
            blooms=parsed_state['blooms'],
            residue=parsed_state['residue'],
            tracers=parsed_state['tracers'],
            shi=shi,
            scup=scup,
            signals=parsed_state['signals'],
            flags=self.state.flags.get_active_flags()
        )
    
    def _generate_emergency_snapshot(self) -> SchemaSnapshot:
        """Generate emergency schema snapshot during errors"""
        return SchemaSnapshot(
            tick=self.tick_count,
            timestamp=time.time(),
            nodes=[],
            edges=[],
            blooms={},
            residue=ResidueMetrics(),
            tracers=TracerCounts(),
            shi=0.0,
            scup=0.0,
            signals={'emergency': 1.0},
            flags={'emergency': True}
        )
    
    def add_schema_node(self, node_id: str, health: float = 0.5, 
                       tint: Optional[List[float]] = None, energy: float = 0.5) -> SchemaNode:
        """Add a new schema node"""
        return self.state.add_node(node_id, health, tint, energy)
    
    def add_schema_edge(self, edge_id: str, source: str, target: str, 
                       weight: float = 0.5, tension: float = 0.0) -> SchemaEdge:
        """Add a new schema edge"""
        return self.state.add_edge(edge_id, source, target, weight, tension)
    
    def get_system_status(self) -> Dict[str, Any]:
        """Get comprehensive system status"""
        with self.lock:
            avg_processing_time = np.mean(self.processing_times) if self.processing_times else 0.0
            
            return {
                'tick_count': self.tick_count,
                'node_count': len(self.state.nodes),
                'edge_count': len(self.state.edges),
                'current_shi': self.last_shi,
                'current_scup': self.last_scup,
                'average_health': self.state.get_average_health(),
                'average_tension': self.state.get_average_tension(),
                'mode': self.state.mode.value,
                'flags': self.state.flags.get_active_flags(),
                'performance': {
                    'avg_processing_time': avg_processing_time,
                    'error_count': self.error_count,
                    'history_size': len(self.state.snapshot_history)
                }
            }
    
    def get_health_trend(self, window: int = 50) -> Dict[str, float]:
        """Get health trend analysis"""
        with self.lock:
            if len(self.state.health_history) < 2:
                return {'trend': 0.0, 'stability': 1.0, 'direction': 'stable'}
            
            recent_history = list(self.state.health_history)[-window:]
            
            # Calculate trend
            if len(recent_history) >= 2:
                trend = recent_history[-1] - recent_history[0]
                stability = 1.0 - np.std(recent_history)
                
                if trend > 0.05:
                    direction = 'improving'
                elif trend < -0.05:
                    direction = 'degrading'
                else:
                    direction = 'stable'
            else:
                trend = 0.0
                stability = 1.0
                direction = 'stable'
            
            return {
                'trend': trend,
                'stability': max(0.0, stability),
                'direction': direction
            }


# Factory function for easy instantiation
def create_schema_system(config: Optional[Dict[str, Any]] = None) -> SchemaSystem:
    """
    Create a configured SchemaSystem instance
    
    Args:
        config: Optional configuration dictionary
        
    Returns:
        SchemaSystem: Configured schema system instance
    """
    if config is None:
        config = {}
    
    return SchemaSystem(
        enable_shi_calculation=config.get('enable_shi', True),
        enable_scup_tracking=config.get('enable_scup', True),
        enable_validation=config.get('enable_validation', True),
        enable_monitoring=config.get('enable_monitoring', True),
        vault_path=config.get('vault_path')
    )


if __name__ == "__main__":
    # Example usage
    print("DAWN Core Schema System - Example Usage")
    
    # Create schema system
    schema_system = create_schema_system()
    
    # Add some nodes and edges
    schema_system.add_schema_node("concept_1", health=0.8, energy=0.7)
    schema_system.add_schema_node("concept_2", health=0.6, energy=0.5)
    schema_system.add_schema_edge("edge_1", "concept_1", "concept_2", weight=0.7)
    
    # Simulate tick updates
    for tick in range(10):
        tick_data = {
            'signals': {
                'pressure': 0.3 + 0.1 * np.sin(tick * 0.5),
                'drift': 0.2 + 0.05 * np.cos(tick * 0.3),
                'entropy': 0.4 + 0.1 * np.random.normal(0, 0.1)
            },
            'tracers': {
                'crow': np.random.randint(0, 5),
                'spider': np.random.randint(0, 3),
                'ant': np.random.randint(0, 10)
            },
            'residue': {
                'soot_ratio': 0.1 + 0.05 * np.random.normal(0, 0.1),
                'ash_bias': [0.33, 0.33, 0.34]
            }
        }
        
        snapshot = schema_system.tick_update(tick_data)
        
        print(f"Tick {tick}: SHI={snapshot.shi:.3f}, SCUP={snapshot.scup:.3f}, "
              f"Nodes={len(snapshot.nodes)}, Edges={len(snapshot.edges)}")
    
    # Get system status
    status = schema_system.get_system_status()
    print(f"\nSystem Status: {status}")
    
    # Get health trend
    trend = schema_system.get_health_trend()
    print(f"Health Trend: {trend}")
