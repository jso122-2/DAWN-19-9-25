"""
DAWN Tracer System - Distributed Cognitive Monitoring Network

This module provides the complete tracer ecosystem for DAWN consciousness,
implementing biological metaphors for specialized cognitive monitoring and
maintenance functions.

The tracer system includes:
- Base architecture (BaseTracer, TracerManager)
- Fast tracers (Crow, Ant, Bee) 
- Medium tracers (Spider, Beetle)
- Heavy tracers (Whale, Owl, Medieval Bee)

Each tracer embodies a specific biological archetype and cognitive function,
working together to maintain DAWN's cognitive health across all timescales.
"""

from .base_tracer import (
    BaseTracer,
    TracerType, 
    TracerStatus,
    AlertSeverity,
    TracerReport,
    TracerSpawnConditions
)

from .tracer_manager import (
    TracerManager,
    TracerEcosystemMetrics
)

# Fast Tracers - lightweight, high-frequency monitoring
from .crow_tracer import CrowTracer
from .ant_tracer import AntTracer  
from .bee_tracer import BeeTracer

# Medium Tracers - structural monitoring and maintenance
from .spider_tracer import SpiderTracer
from .beetle_tracer import BeetleTracer

# Heavy Tracers - deep analysis and archival
from .whale_tracer import WhaleTracer
from .owl_tracer import OwlTracer
from .medieval_bee_tracer import MedievalBeeTracer

# Tracer registry for easy access
TRACER_CLASSES = {
    TracerType.CROW: CrowTracer,
    TracerType.ANT: AntTracer,
    TracerType.BEE: BeeTracer,
    TracerType.SPIDER: SpiderTracer,
    TracerType.BEETLE: BeetleTracer,
    TracerType.WHALE: WhaleTracer,
    TracerType.OWL: OwlTracer,
    TracerType.MEDIEVAL_BEE: MedievalBeeTracer
}

def create_tracer_ecosystem(nutrient_budget: float = 100.0) -> TracerManager:
    """
    Create a complete tracer ecosystem with all tracer types registered.
    
    Args:
        nutrient_budget: Total nutrient budget for the ecosystem
        
    Returns:
        TracerManager: Configured tracer manager with all tracers registered
    """
    manager = TracerManager(nutrient_budget=nutrient_budget)
    
    # Register all tracer classes
    for tracer_type, tracer_class in TRACER_CLASSES.items():
        manager.register_tracer_class(tracer_type, tracer_class)
    
    return manager

def get_tracer_archetypes() -> dict:
    """
    Get a summary of all tracer archetypes and their descriptions.
    
    Returns:
        dict: Mapping of tracer type to archetype description
    """
    archetypes = {}
    
    for tracer_type, tracer_class in TRACER_CLASSES.items():
        # Create temporary instance to get archetype description
        temp_instance = tracer_class()
        archetypes[tracer_type.value] = temp_instance.archetype_description
    
    return archetypes

__all__ = [
    # Base classes
    'BaseTracer',
    'TracerType',
    'TracerStatus', 
    'AlertSeverity',
    'TracerReport',
    'TracerSpawnConditions',
    'TracerManager',
    'TracerEcosystemMetrics',
    
    # Fast tracers
    'CrowTracer',
    'AntTracer',
    'BeeTracer',
    
    # Medium tracers
    'SpiderTracer',
    'BeetleTracer',
    
    # Heavy tracers
    'WhaleTracer',
    'OwlTracer',
    'MedievalBeeTracer',
    
    # Utilities
    'TRACER_CLASSES',
    'create_tracer_ecosystem',
    'get_tracer_archetypes'
]
