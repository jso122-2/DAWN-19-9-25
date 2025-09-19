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

# CUDA-powered tracer modeling and visualization
try:
    from .cuda_tracer_engine import (
        CUDATracerModelingEngine,
        CUDATracerModelConfig,
        get_cuda_tracer_engine,
        reset_cuda_tracer_engine
    )
    from .cuda_tracer_visualization import (
        CUDATracerVisualizationEngine,
        TracerVisualizationConfig,
        get_cuda_tracer_visualization_engine,
        reset_cuda_tracer_visualization_engine
    )
    CUDA_TRACER_AVAILABLE = True
except ImportError:
    CUDA_TRACER_AVAILABLE = False

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

def create_tracer_ecosystem(nutrient_budget: float = 100.0, enable_cuda: bool = True) -> TracerManager:
    """
    Create a complete tracer ecosystem with all registered tracer types.
    
    Args:
        nutrient_budget: Initial nutrient budget for the ecosystem
        enable_cuda: Whether to enable CUDA acceleration if available
        
    Returns:
        TracerManager: Configured tracer ecosystem
    """
    manager = TracerManager(nutrient_budget)
    
    # Register all standard tracer classes
    manager.register_tracer_class(TracerType.CROW, CrowTracer)
    manager.register_tracer_class(TracerType.ANT, AntTracer)
    manager.register_tracer_class(TracerType.BEE, BeeTracer)
    manager.register_tracer_class(TracerType.SPIDER, SpiderTracer)
    manager.register_tracer_class(TracerType.BEETLE, BeetleTracer)
    manager.register_tracer_class(TracerType.WHALE, WhaleTracer)
    manager.register_tracer_class(TracerType.OWL, OwlTracer)
    manager.register_tracer_class(TracerType.MEDIEVAL_BEE, MedievalBeeTracer)
    
    # Initialize CUDA acceleration if requested and available
    if enable_cuda and CUDA_TRACER_AVAILABLE:
        try:
            cuda_engine = get_cuda_tracer_engine()
            if cuda_engine.cuda_available:
                logger.info("✅ CUDA tracer acceleration enabled")
            else:
                logger.info("⚠️  CUDA libraries available but no GPU detected")
        except Exception as e:
            logger.warning(f"Could not initialize CUDA tracer acceleration: {e}")
    
    return manager


def create_cuda_tracer_ecosystem(
    nutrient_budget: float = 100.0,
    cuda_config: Optional[CUDATracerModelConfig] = None,
    viz_config: Optional[TracerVisualizationConfig] = None,
    enable_visualization: bool = True
) -> Dict[str, Any]:
    """
    Create a complete CUDA-accelerated tracer ecosystem with visualization.
    
    Args:
        nutrient_budget: Initial nutrient budget for the ecosystem
        cuda_config: CUDA modeling configuration
        viz_config: Visualization configuration
        enable_visualization: Whether to enable real-time visualization
        
    Returns:
        Dict containing manager, cuda_engine, and viz_engine
    """
    if not CUDA_TRACER_AVAILABLE:
        raise ImportError("CUDA tracer components not available")
    
    # Create standard ecosystem
    manager = create_tracer_ecosystem(nutrient_budget, enable_cuda=True)
    
    # Get CUDA engines
    cuda_engine = get_cuda_tracer_engine(cuda_config)
    
    ecosystem = {
        'manager': manager,
        'cuda_engine': cuda_engine,
        'viz_engine': None
    }
    
    if enable_visualization:
        try:
            viz_engine = get_cuda_tracer_visualization_engine(viz_config)
            viz_engine.connect_to_tracer_manager(manager)
            ecosystem['viz_engine'] = viz_engine
            logger.info("✅ CUDA tracer visualization enabled")
        except Exception as e:
            logger.warning(f"Could not initialize CUDA tracer visualization: {e}")
    
    return ecosystem

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

# Add CUDA components to exports if available
if CUDA_TRACER_AVAILABLE:
    __all__.extend([
        'CUDATracerModelingEngine',
        'CUDATracerModelConfig', 
        'get_cuda_tracer_engine',
        'reset_cuda_tracer_engine',
        'CUDATracerVisualizationEngine',
        'TracerVisualizationConfig',
        'get_cuda_tracer_visualization_engine',
        'reset_cuda_tracer_visualization_engine',
        'create_cuda_tracer_ecosystem',
        'CUDA_TRACER_AVAILABLE'
    ])
