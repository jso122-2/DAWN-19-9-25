"""
Base tracer framework for DAWN consciousness system.

This module provides the foundational classes and interfaces for DAWN's
distributed cognitive monitoring network. Each tracer embodies a specific
biological metaphor and cognitive function.
"""

from abc import ABC, abstractmethod
from dataclasses import dataclass, field
from typing import Dict, Any, List, Optional, Union
from enum import Enum
import uuid
import time
import logging

# DAWN singleton integration
from dawn.core.singleton import get_dawn

logger = logging.getLogger(__name__)


class TracerType(Enum):
    """Enumeration of all tracer types in the DAWN ecosystem"""
    CROW = "crow"
    WHALE = "whale" 
    ANT = "ant"
    SPIDER = "spider"
    BEETLE = "beetle"
    BEE = "bee"
    OWL = "owl"
    MEDIEVAL_BEE = "medieval_bee"


class TracerStatus(Enum):
    """Lifecycle status of a tracer"""
    SPAWNING = "spawning"
    ACTIVE = "active"
    RETIRING = "retiring"
    RETIRED = "retired"


class AlertSeverity(Enum):
    """Alert severity levels for tracer reports"""
    INFO = "info"
    WARN = "warn"
    CRITICAL = "critical"


@dataclass
class TracerReport:
    """Base structure for all tracer reports"""
    tracer_id: str
    tracer_type: TracerType
    tick_id: int
    timestamp: float
    severity: AlertSeverity = AlertSeverity.INFO
    report_type: str = "observation"
    metadata: Dict[str, Any] = field(default_factory=dict)
    
    def to_dict(self) -> Dict[str, Any]:
        """Convert report to dictionary for serialization"""
        return {
            "tracer_id": self.tracer_id,
            "tracer_type": self.tracer_type.value,
            "tick_id": self.tick_id,
            "timestamp": self.timestamp,
            "severity": self.severity.value,
            "report_type": self.report_type,
            "metadata": self.metadata
        }


class BaseTracer(ABC):
    """
    Abstract base class for all DAWN tracers.
    
    Each tracer embodies a specific biological metaphor and implements
    specialized cognitive monitoring functions with distinct lifecycles,
    costs, and behaviors.
    
    Now includes DAWN singleton integration for unified system access.
    """
    
    def __init__(self, tracer_id: str = None):
        self.tracer_id = tracer_id or str(uuid.uuid4())
        self.status = TracerStatus.SPAWNING
        self.spawn_tick = None
        self.retire_tick = None
        self.current_nutrient_cost = 0.0
        self.total_nutrient_consumed = 0.0
        self.reports: List[TracerReport] = []
        self.spawn_context = {}
        
        # DAWN singleton integration
        self._dawn = None
        self._consciousness_bus = None
        self._telemetry_system = None
        self._initialize_dawn_integration()
        
    @property
    @abstractmethod
    def tracer_type(self) -> TracerType:
        """Return the tracer type"""
        pass
    
    @property
    @abstractmethod
    def base_lifespan(self) -> int:
        """Base lifespan in ticks"""
        pass
    
    @property
    @abstractmethod
    def base_nutrient_cost(self) -> float:
        """Base nutrient cost per tick"""
        pass
    
    @property
    @abstractmethod
    def archetype_description(self) -> str:
        """Human-readable description of the tracer's archetype"""
        pass
    
    @abstractmethod
    def spawn_conditions_met(self, context: Dict[str, Any]) -> bool:
        """Check if conditions are met for spawning this tracer"""
        pass
    
    @abstractmethod
    def observe(self, context: Dict[str, Any]) -> List[TracerReport]:
        """Core observation logic - return reports"""
        pass
    
    @abstractmethod
    def should_retire(self, context: Dict[str, Any]) -> bool:
        """Check if tracer should retire"""
        pass
    
    def spawn(self, tick_id: int, context: Dict[str, Any]) -> None:
        """Initialize tracer for active duty"""
        self.spawn_tick = tick_id
        self.status = TracerStatus.ACTIVE
        self.spawn_context = context.copy()
        self.current_nutrient_cost = self.base_nutrient_cost
        
        logger.info(f"Spawned {self.tracer_type.value} tracer {self.tracer_id} at tick {tick_id}")
        
    def retire(self, tick_id: int) -> float:
        """Retire tracer and return any energy to recycle"""
        self.retire_tick = tick_id
        self.status = TracerStatus.RETIRED
        
        # Calculate recycled energy (10% of total consumed)
        recycled_energy = self.total_nutrient_consumed * 0.1
        
        lifespan = tick_id - self.spawn_tick if self.spawn_tick else 0
        logger.info(f"Retired {self.tracer_type.value} tracer {self.tracer_id} after {lifespan} ticks, "
                   f"recycling {recycled_energy:.3f} energy")
        
        return recycled_energy
    
    def tick(self, tick_id: int, context: Dict[str, Any]) -> List[TracerReport]:
        """Main execution cycle"""
        if self.status != TracerStatus.ACTIVE:
            return []
            
        # Consume nutrients for this tick
        self.total_nutrient_consumed += self.current_nutrient_cost
        
        # Check retirement conditions first
        if self.should_retire(context):
            self.status = TracerStatus.RETIRING
            return []
            
        # Perform observations
        try:
            reports = self.observe(context)
            self.reports.extend(reports)
            return reports
        except Exception as e:
            logger.error(f"Error in {self.tracer_type.value} tracer {self.tracer_id}: {e}")
            # Force retirement on error
            self.status = TracerStatus.RETIRING
            return []
    
    def get_age(self, current_tick: int) -> int:
        """Get the current age of the tracer in ticks"""
        if self.spawn_tick is None:
            return 0
        return current_tick - self.spawn_tick
    
    def _initialize_dawn_integration(self):
        """Initialize integration with DAWN singleton"""
        try:
            self._dawn = get_dawn()
            
            if self._dawn.is_initialized:
                self._consciousness_bus = self._dawn.consciousness_bus
                self._telemetry_system = self._dawn.telemetry_system
                
                if self._consciousness_bus:
                    # Register tracer with consciousness bus
                    self._consciousness_bus.register_module(
                        f'tracer_{self.tracer_id}',
                        self,
                        capabilities=['cognitive_monitoring', 'anomaly_detection']
                    )
                
                if self._telemetry_system:
                    # Log tracer creation
                    self._telemetry_system.log_event(
                        'tracer_created',
                        {
                            'tracer_id': self.tracer_id,
                            'tracer_type': self.tracer_type.value if hasattr(self, 'tracer_type') else 'unknown',
                            'archetype': self.archetype_description if hasattr(self, 'archetype_description') else 'unknown'
                        }
                    )
                    
        except Exception as e:
            logger.debug(f"Could not initialize DAWN integration for tracer {self.tracer_id}: {e}")
    
    @property
    def dawn(self):
        """Get DAWN singleton instance"""
        if self._dawn is None:
            self._dawn = get_dawn()
        return self._dawn
    
    @property
    def consciousness_bus(self):
        """Get consciousness bus instance"""
        if self._consciousness_bus is None and self.dawn.is_initialized:
            self._consciousness_bus = self.dawn.consciousness_bus
        return self._consciousness_bus
    
    @property
    def telemetry_system(self):
        """Get telemetry system instance"""
        if self._telemetry_system is None and self.dawn.is_initialized:
            self._telemetry_system = self.dawn.telemetry_system
        return self._telemetry_system
    
    def log_to_telemetry(self, event_type: str, data: Dict[str, Any]):
        """Log event to telemetry system if available"""
        if self.telemetry_system:
            try:
                self.telemetry_system.log_event(event_type, {
                    'tracer_id': self.tracer_id,
                    'tracer_type': self.tracer_type.value,
                    **data
                })
            except Exception as e:
                logger.debug(f"Failed to log telemetry for tracer {self.tracer_id}: {e}")
    
    def broadcast_to_consciousness(self, message_type: str, data: Dict[str, Any]):
        """Broadcast message to consciousness bus if available"""
        if self.consciousness_bus:
            try:
                self.consciousness_bus.broadcast_message(message_type, {
                    'source_tracer': self.tracer_id,
                    'tracer_type': self.tracer_type.value,
                    **data
                })
            except Exception as e:
                logger.debug(f"Failed to broadcast to consciousness for tracer {self.tracer_id}: {e}")
    
    def get_status_info(self) -> Dict[str, Any]:
        """Get comprehensive status information"""
        return {
            "tracer_id": self.tracer_id,
            "tracer_type": self.tracer_type.value,
            "status": self.status.value,
            "spawn_tick": self.spawn_tick,
            "retire_tick": self.retire_tick,
            "current_nutrient_cost": self.current_nutrient_cost,
            "total_nutrient_consumed": self.total_nutrient_consumed,
            "report_count": len(self.reports),
            "archetype": self.archetype_description,
            "dawn_integration": {
                "dawn_connected": self._dawn is not None,
                "consciousness_bus_connected": self._consciousness_bus is not None,
                "telemetry_connected": self._telemetry_system is not None
            }
        }


class TracerSpawnConditions:
    """Utility class for common spawn condition checks"""
    
    @staticmethod
    def entropy_spike(context: Dict[str, Any], threshold: float = 0.7) -> bool:
        """Check for entropy spike above threshold"""
        return context.get('entropy', 0.0) > threshold
    
    @staticmethod
    def sustained_entropy(context: Dict[str, Any], threshold: float = 0.6, window: int = 5) -> bool:
        """Check for sustained high entropy over time window"""
        entropy_history = context.get('entropy_history', [])
        if len(entropy_history) < window:
            return False
        
        recent_avg = sum(entropy_history[-window:]) / window
        return recent_avg > threshold
    
    @staticmethod
    def drift_misalignment(context: Dict[str, Any], threshold: float = 0.5) -> bool:
        """Check for drift misalignment above threshold"""
        return context.get('drift_magnitude', 0.0) > threshold
    
    @staticmethod
    def pressure_anomaly(context: Dict[str, Any], threshold: float = 0.3) -> bool:
        """Check for sudden pressure changes"""
        pressure = context.get('pressure', 0.0)
        pressure_history = context.get('pressure_history', [])
        
        if len(pressure_history) < 1:
            return False
            
        pressure_change = abs(pressure - pressure_history[-1])
        return pressure_change > threshold
    
    @staticmethod
    def soot_accumulation(context: Dict[str, Any], threshold: float = 0.4) -> bool:
        """Check for excessive soot accumulation"""
        return context.get('soot_ratio', 0.0) > threshold
    
    @staticmethod
    def cluster_isolation(context: Dict[str, Any], ratio_threshold: float = 0.3) -> bool:
        """Check for isolated schema clusters"""
        clusters = context.get('schema_clusters', [])
        
        for cluster in clusters:
            cross_links = cluster.get('cross_links', 0)
            internal_links = cluster.get('internal_links', 1)
            
            if cross_links / internal_links < ratio_threshold:
                return True
                
        return False
    
    @staticmethod
    def schema_tension(context: Dict[str, Any], threshold: float = 0.6) -> bool:
        """Check for high schema edge tension"""
        schema_edges = context.get('schema_edges', [])
        
        for edge in schema_edges:
            if edge.get('tension', 0.0) > threshold:
                return True
                
        return False
