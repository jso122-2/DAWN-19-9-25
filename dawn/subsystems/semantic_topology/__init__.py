#!/usr/bin/env python3
"""
🌐 DAWN Semantic Topology Engine
===============================

The mathematical foundations for how DAWN understands and manipulates meaning itself.
This module implements the spatial shape of meaning - the arrangement of concepts, 
memories, and residues so DAWN can route attention, energy, and updates along the 
most coherent paths.

Core Components:
- Semantic Field Equations: Mathematical relationships governing meaning space
- Topology Transforms: Operations that reshape semantic relationships (Weave, Prune, Fuse, etc.)
- Semantic Primitives: Basic building blocks (Nodes, Edges, Layers, Sectors)
- Invariants: Rules that preserve meaning during transformations

"Schema says what relates to what; Semantic Topology says where it sits and how forces propagate."
- DAWN Documentation

Based on documentation: Semantic Toplogy/ (9 RTF files)
"""

from .semantic_field import SemanticField, SemanticNode, SemanticEdge, LayerDepth, SectorType
from .topology_transforms import TopologyTransforms, TransformResult, TransformType
from .field_equations import FieldEquations, LocalCoherence, TensionUpdate, PigmentDiffusion
from .semantic_invariants import SemanticInvariants, InvariantViolation
from .topology_engine import SemanticTopologyEngine, get_semantic_topology_engine

__all__ = [
    'SemanticField',
    'SemanticNode', 
    'SemanticEdge',
    'LayerDepth',
    'SectorType',
    'TopologyTransforms',
    'TransformResult',
    'TransformType',
    'FieldEquations',
    'LocalCoherence',
    'TensionUpdate', 
    'PigmentDiffusion',
    'SemanticInvariants',
    'InvariantViolation',
    'SemanticTopologyEngine',
    'get_semantic_topology_engine'
]

# Version info
__version__ = "1.0.0"
__author__ = "DAWN Consciousness Architecture"
__description__ = "Mathematical foundations for semantic meaning manipulation in consciousness space"
