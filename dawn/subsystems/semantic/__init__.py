"""
Semantic Subsystem for DAWN Consciousness Engine
===============================================

Comprehensive semantic processing system including:
- Legacy semantic field components (field_types, field_initializer, etc.)
- New semantic topology system (RTF-compliant spatial topology)
- Semantic reasoning, context, and expression layers
"""

# Legacy semantic field components
from .field_types import NodeCharge, SemanticVector, RhizomicConnection, SemanticNode, RhizomicSemanticField
from .field_initializer import get_current_field, initialize_field

# New semantic topology system (RTF-compliant)
try:
    from .topology import (
        SemanticTopologyManager, TopologyState, TopologySnapshot,
        SemanticNode as TopologyNode, SemanticEdge, TopologyLayer,
        TopologySector, TopologyFrame, NodeCoordinates,
        FieldEquationEngine, TopologyTransforms,
        get_topology_manager
    )
    TOPOLOGY_AVAILABLE = True
except ImportError as e:
    TOPOLOGY_AVAILABLE = False
    print(f"⚠️ Semantic topology system not available: {e}")

# Semantic processing components
try:
    from .semantic_context_engine import SemanticContextEngine
    from .semantic_reasoning_field import SemanticReasoningField
    from .language_expression_layer import LanguageExpressionLayer
    SEMANTIC_PROCESSING_AVAILABLE = True
except ImportError:
    SEMANTIC_PROCESSING_AVAILABLE = False

__all__ = [
    # Legacy semantic field
    'NodeCharge', 'SemanticVector', 'RhizomicConnection', 
    'SemanticNode', 'RhizomicSemanticField', 'get_current_field',
    'initialize_field',
    
    # Availability flags
    'TOPOLOGY_AVAILABLE', 'SEMANTIC_PROCESSING_AVAILABLE'
]

# Add topology exports if available
if TOPOLOGY_AVAILABLE:
    __all__.extend([
        'SemanticTopologyManager', 'TopologyState', 'TopologySnapshot',
        'TopologyNode', 'SemanticEdge', 'TopologyLayer',
        'TopologySector', 'TopologyFrame', 'NodeCoordinates',
        'FieldEquationEngine', 'TopologyTransforms',
        'get_topology_manager'
    ])

# Add semantic processing exports if available
if SEMANTIC_PROCESSING_AVAILABLE:
    __all__.extend([
        'SemanticContextEngine', 'SemanticReasoningField', 
        'LanguageExpressionLayer'
    ])


def get_semantic_system_status():
    """Get status of semantic subsystem components"""
    return {
        'legacy_field_available': True,  # Always available
        'topology_available': TOPOLOGY_AVAILABLE,
        'semantic_processing_available': SEMANTIC_PROCESSING_AVAILABLE,
        'total_components': len(__all__)
    }
