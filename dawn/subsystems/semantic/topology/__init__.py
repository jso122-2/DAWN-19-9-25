"""
DAWN Semantic Topology System
============================

Implementation of DAWN's spatial semantic topology system based on RTF specifications.
Provides the spatial shape of meaning - arrangement of concepts, memories, and residues
for coherent attention routing, energy distribution, and cognitive updates.

Core Components:
- Topology primitives (Node, Edge, Layer, Sector, Frame)
- Field equations (coherence, tension, pigment diffusion, residue pressure)  
- Transform operators (weave, prune, fuse, fission, lift, sink, reproject)
- Topology manager for integration with DAWN consciousness system

Based on DAWN-docs/Semantic Topology/ specifications.
"""

from .primitives import (
    SemanticNode,
    SemanticEdge, 
    TopologyLayer,
    TopologySector,
    TopologyFrame,
    NodeCoordinates,
    TopologyMetrics
)

from .field_equations import (
    FieldEquationEngine,
    LocalCoherenceCalculator,
    TensionUpdater,
    PigmentDiffuser,
    ResiduePressureCalculator
)

from .transforms import (
    TopologyTransforms,
    WeaveOperator,
    PruneOperator, 
    FuseOperator,
    FissionOperator,
    LiftOperator,
    SinkOperator,
    ReprojectOperator
)

from .topology_manager import (
    SemanticTopologyManager,
    TopologyState,
    TopologySnapshot,
    get_topology_manager
)

__all__ = [
    # Primitives
    'SemanticNode',
    'SemanticEdge',
    'TopologyLayer', 
    'TopologySector',
    'TopologyFrame',
    'NodeCoordinates',
    'TopologyMetrics',
    
    # Field Equations
    'FieldEquationEngine',
    'LocalCoherenceCalculator',
    'TensionUpdater', 
    'PigmentDiffuser',
    'ResiduePressureCalculator',
    
    # Transforms
    'TopologyTransforms',
    'WeaveOperator',
    'PruneOperator',
    'FuseOperator', 
    'FissionOperator',
    'LiftOperator',
    'SinkOperator',
    'ReprojectOperator',
    
    # Manager
    'SemanticTopologyManager',
    'TopologyState',
    'TopologySnapshot',
    'get_topology_manager'
]
