#!/usr/bin/env python3
"""
🌐 Semantic Field - The Shape of Meaning Space
==============================================

Core data structures and mathematical foundations for DAWN's semantic topology.
Implements nodes, edges, layers, sectors, and the semantic field that contains them all.

This is where meaning gets its spatial structure - concepts have positions, 
relationships have geometry, and consciousness can navigate through semantic space.

Based on documentation: Semantic Toplogy/Primitives.rtf + Semantic Topology.rtf
"""

import numpy as np
import uuid
import time
import logging
from typing import Dict, List, Optional, Tuple, Set, Any, Union
from dataclasses import dataclass, field
from enum import Enum
from collections import defaultdict, deque

logger = logging.getLogger(__name__)

class LayerDepth(Enum):
    """Depth planes in semantic space"""
    SURFACE = 0      # Surface consciousness
    SHALLOW = 1      # Near-surface processing  
    DEEP = 2         # Deep semantic processing
    PROFOUND = 3     # Profound understanding layers
    TRANSCENDENT = 4 # Transcendent meaning layers

class SectorType(Enum):
    """Regional partitions of semantic space"""
    CORE = "core"                   # Central, stable concepts
    PERIPHERAL = "peripheral"       # Edge concepts, experimental
    TRANSITIONAL = "transitional"   # Concepts in flux
    DEEP = "deep"                   # Profound, rarely accessed

@dataclass
class SemanticNode:
    """
    A semantic unit - concept, memory, or cluster in meaning space.
    
    Represents a point in the topology where meaning concentrates.
    Has both projected coordinates (for visualization/topology ops) 
    and latent embeddings (for semantic similarity).
    """
    
    # Identity
    id: str = field(default_factory=lambda: str(uuid.uuid4()))
    
    # Spatial properties  
    position: np.ndarray = field(default_factory=lambda: np.zeros(3))  # Projected coords (x,y,z)
    embedding: np.ndarray = field(default_factory=lambda: np.zeros(512))  # Latent semantics
    
    # Visual properties
    tint: np.ndarray = field(default_factory=lambda: np.array([0.5, 0.5, 0.5]))  # [R,G,B] pigment bias
    
    # Health and energy
    health: float = 1.0          # Structural vitality [0,1]
    energy: float = 1.0          # Mycelial budget [0,1]
    
    # Topology properties
    layer: LayerDepth = LayerDepth.SURFACE
    sector: SectorType = SectorType.CORE
    
    # Temporal tracking
    last_touch: int = field(default_factory=lambda: int(time.time()))
    creation_time: float = field(default_factory=time.time)
    
    # Semantic properties
    concept_strength: float = 0.5    # How well-defined this concept is
    access_frequency: float = 0.0    # How often accessed
    contextual_relevance: float = 0.5 # Current contextual importance
    
    def __post_init__(self):
        """Validate and normalize fields after creation"""
        # Ensure position is 3D numpy array
        if not isinstance(self.position, np.ndarray):
            self.position = np.array(self.position)
        if self.position.shape != (3,):
            self.position = np.pad(self.position, (0, max(0, 3 - len(self.position))))[:3]
            
        # Ensure embedding is numpy array
        if not isinstance(self.embedding, np.ndarray):
            self.embedding = np.array(self.embedding)
            
        # Ensure tint is valid RGB
        if not isinstance(self.tint, np.ndarray):
            self.tint = np.array(self.tint)
        self.tint = np.clip(self.tint, 0.0, 1.0)
        
        # Clamp health and energy
        self.health = max(0.0, min(1.0, self.health))
        self.energy = max(0.0, min(1.0, self.energy))
        
    def semantic_distance_to(self, other: 'SemanticNode') -> float:
        """Calculate semantic distance using embeddings"""
        if self.embedding.shape != other.embedding.shape:
            return float('inf')
        return float(np.linalg.norm(self.embedding - other.embedding))
    
    def spatial_distance_to(self, other: 'SemanticNode') -> float:
        """Calculate spatial distance using projected positions"""
        return float(np.linalg.norm(self.position - other.position))
    
    def update_position_from_embedding(self, projection_matrix: Optional[np.ndarray] = None):
        """Update projected position from high-dimensional embedding"""
        if projection_matrix is not None:
            # Use provided projection matrix
            self.position = (projection_matrix @ self.embedding)[:3]
        else:
            # Simple PCA-style projection (first 3 dimensions)
            self.position = self.embedding[:3]
            
    def calculate_radial_position(self, alpha: float = 1.0, beta: float = 1.0, gamma: float = 1.0) -> float:
        """
        Calculate radial position in semantic field using the documented formula:
        R_node = 1 / ((A_count * α) + (C_relevance * β) + (S_feedback * γ))
        """
        access_term = self.access_frequency * alpha
        relevance_term = self.contextual_relevance * beta  
        feedback_term = self.concept_strength * gamma
        
        denominator = access_term + relevance_term + feedback_term
        if denominator == 0:
            return float('inf')  # Infinite distance for unused concepts
            
        return 1.0 / denominator
    
    def to_dict(self) -> Dict[str, Any]:
        """Convert to dictionary for serialization"""
        return {
            'id': self.id,
            'position': self.position.tolist(),
            'embedding': self.embedding.tolist(),
            'tint': self.tint.tolist(),
            'health': self.health,
            'energy': self.energy,
            'layer': self.layer.value,
            'sector': self.sector.value,
            'last_touch': self.last_touch,
            'creation_time': self.creation_time,
            'concept_strength': self.concept_strength,
            'access_frequency': self.access_frequency,
            'contextual_relevance': self.contextual_relevance
        }

@dataclass  
class SemanticEdge:
    """
    A relation between semantic nodes - directed or undirected.
    
    Represents the connections in meaning space - how concepts relate,
    influence each other, and allow consciousness to navigate between them.
    """
    
    # Connection (required fields first)
    node_a: str  # Node ID
    node_b: str  # Node ID
    
    # Identity
    id: str = field(default_factory=lambda: str(uuid.uuid4()))
    directed: bool = False  # Whether edge has direction (a -> b)
    
    # Strength and reliability
    weight: float = 1.0          # Connection strength [0, inf)
    tension: float = 0.0         # Strain/stress [0, inf) - Spider tracer source
    volatility: float = 0.1      # Likelihood of change [0,1]
    reliability: float = 0.9     # Historical stability [0,1]
    
    # Temporal tracking
    last_update: float = field(default_factory=time.time)
    creation_time: float = field(default_factory=time.time)
    
    # Usage statistics
    traversal_count: int = 0     # How often this edge is used
    strength_history: deque = field(default_factory=lambda: deque(maxlen=100))
    
    def __post_init__(self):
        """Validate fields after creation"""
        # Generate ID from nodes if not provided
        if not hasattr(self, 'id') or not self.id:
            direction = "->" if self.directed else "<->"
            self.id = f"{self.node_a}{direction}{self.node_b}"
            
        # Clamp values to valid ranges
        self.weight = max(0.0, self.weight)
        self.tension = max(0.0, self.tension)
        self.volatility = max(0.0, min(1.0, self.volatility))
        self.reliability = max(0.0, min(1.0, self.reliability))
        
        # Initialize strength history
        if not self.strength_history:
            self.strength_history = deque([self.weight], maxlen=100)
    
    def update_tension(self, expected_weight: float, gamma: float = 0.8):
        """
        Update tension using the documented formula:
        τ_ij ← γ·τ_ij + (1-γ)·|w_ij - f(dist_x(i,j))|
        """
        tension_delta = abs(self.weight - expected_weight)
        self.tension = gamma * self.tension + (1 - gamma) * tension_delta
        self.last_update = time.time()
        
    def update_weight(self, new_weight: float):
        """Update edge weight and track history"""
        self.weight = max(0.0, new_weight)
        self.strength_history.append(self.weight)
        self.last_update = time.time()
        
    def traverse(self):
        """Record a traversal of this edge"""
        self.traversal_count += 1
        self.last_update = time.time()
        
    def calculate_reliability(self) -> float:
        """Calculate reliability based on weight stability"""
        if len(self.strength_history) < 2:
            return self.reliability
            
        # Calculate variance in recent weights
        weights = np.array(list(self.strength_history))
        variance = np.var(weights)
        
        # Lower variance = higher reliability
        stability_factor = 1.0 / (1.0 + variance)
        
        # Update reliability with exponential moving average
        self.reliability = 0.9 * self.reliability + 0.1 * stability_factor
        return self.reliability
        
    def to_dict(self) -> Dict[str, Any]:
        """Convert to dictionary for serialization"""
        return {
            'id': self.id,
            'node_a': self.node_a,
            'node_b': self.node_b,
            'directed': self.directed,
            'weight': self.weight,
            'tension': self.tension,
            'volatility': self.volatility,
            'reliability': self.reliability,
            'last_update': self.last_update,
            'creation_time': self.creation_time,
            'traversal_count': self.traversal_count,
            'strength_history': list(self.strength_history)
        }

class SemanticField:
    """
    The complete semantic topology - the shape of meaning space in DAWN.
    
    Contains all nodes, edges, and provides operations for navigating,
    querying, and transforming the semantic landscape.
    """
    
    def __init__(self, dimensions: int = 3, embedding_dim: int = 512):
        self.dimensions = dimensions
        self.embedding_dim = embedding_dim
        
        # Core data structures
        self.nodes: Dict[str, SemanticNode] = {}
        self.edges: Dict[str, SemanticEdge] = {}
        
        # Spatial indexing
        self.layer_index: Dict[LayerDepth, Set[str]] = defaultdict(set)
        self.sector_index: Dict[SectorType, Set[str]] = defaultdict(set)
        
        # Adjacency tracking
        self.adjacency: Dict[str, Set[str]] = defaultdict(set)
        
        # Field properties
        self.total_energy = 0.0
        self.total_health = 0.0
        self.coherence_threshold = 0.7
        self.tension_threshold = 0.5
        
        # Statistics
        self.stats = {
            'nodes_created': 0,
            'edges_created': 0,
            'transforms_applied': 0,
            'field_updates': 0,
            'last_update': time.time()
        }
        
        logger.info(f"🌐 SemanticField initialized - {dimensions}D space, {embedding_dim}D embeddings")
    
    def add_node(self, node: SemanticNode) -> bool:
        """Add a node to the semantic field"""
        if node.id in self.nodes:
            logger.warning(f"Node {node.id} already exists in field")
            return False
            
        self.nodes[node.id] = node
        self.layer_index[node.layer].add(node.id)
        self.sector_index[node.sector].add(node.id)
        
        self.total_energy += node.energy
        self.total_health += node.health
        self.stats['nodes_created'] += 1
        
        logger.debug(f"Added semantic node: {node.id} at layer {node.layer.value}")
        return True
        
    def add_edge(self, edge: SemanticEdge) -> bool:
        """Add an edge to the semantic field"""
        if edge.node_a not in self.nodes or edge.node_b not in self.nodes:
            logger.error(f"Cannot add edge {edge.id}: missing nodes")
            return False
            
        if edge.id in self.edges:
            logger.warning(f"Edge {edge.id} already exists")
            return False
            
        self.edges[edge.id] = edge
        self.adjacency[edge.node_a].add(edge.node_b)
        if not edge.directed:
            self.adjacency[edge.node_b].add(edge.node_a)
            
        self.stats['edges_created'] += 1
        
        logger.debug(f"Added semantic edge: {edge.id} ({edge.weight:.3f})")
        return True
        
    def get_neighbors(self, node_id: str) -> Set[str]:
        """Get neighbor nodes of a given node"""
        return self.adjacency.get(node_id, set())
        
    def get_nodes_in_layer(self, layer: LayerDepth) -> List[SemanticNode]:
        """Get all nodes in a specific layer"""
        node_ids = self.layer_index.get(layer, set())
        return [self.nodes[nid] for nid in node_ids if nid in self.nodes]
        
    def get_nodes_in_sector(self, sector: SectorType) -> List[SemanticNode]:
        """Get all nodes in a specific sector"""
        node_ids = self.sector_index.get(sector, set())
        return [self.nodes[nid] for nid in node_ids if nid in self.nodes]
        
    def calculate_semantic_distance(self, node_a_id: str, node_b_id: str) -> float:
        """Calculate semantic distance between two nodes"""
        if node_a_id not in self.nodes or node_b_id not in self.nodes:
            return float('inf')
        return self.nodes[node_a_id].semantic_distance_to(self.nodes[node_b_id])
        
    def calculate_spatial_distance(self, node_a_id: str, node_b_id: str) -> float:
        """Calculate spatial distance between two nodes"""
        if node_a_id not in self.nodes or node_b_id not in self.nodes:
            return float('inf')
        return self.nodes[node_a_id].spatial_distance_to(self.nodes[node_b_id])
        
    def find_path(self, start_id: str, end_id: str, max_hops: int = 10) -> Optional[List[str]]:
        """Find shortest path between two nodes using BFS"""
        if start_id not in self.nodes or end_id not in self.nodes:
            return None
            
        if start_id == end_id:
            return [start_id]
            
        queue = deque([(start_id, [start_id])])
        visited = {start_id}
        
        while queue:
            current, path = queue.popleft()
            
            if len(path) > max_hops:
                continue
                
            for neighbor in self.get_neighbors(current):
                if neighbor == end_id:
                    return path + [neighbor]
                    
                if neighbor not in visited:
                    visited.add(neighbor)
                    queue.append((neighbor, path + [neighbor]))
                    
        return None
        
    def get_field_statistics(self) -> Dict[str, Any]:
        """Get comprehensive field statistics"""
        return {
            'total_nodes': len(self.nodes),
            'total_edges': len(self.edges),
            'total_energy': self.total_energy,
            'total_health': self.total_health,
            'average_energy': self.total_energy / max(1, len(self.nodes)),
            'average_health': self.total_health / max(1, len(self.nodes)),
            'nodes_by_layer': {layer.value: len(nodes) for layer, nodes in self.layer_index.items()},
            'nodes_by_sector': {sector.value: len(nodes) for sector, nodes in self.sector_index.items()},
            'creation_stats': dict(self.stats),
            'dimensions': self.dimensions,
            'embedding_dim': self.embedding_dim
        }
        
    def update_field_energy(self):
        """Recalculate total field energy and health"""
        self.total_energy = sum(node.energy for node in self.nodes.values())
        self.total_health = sum(node.health for node in self.nodes.values())
        self.stats['field_updates'] += 1
        self.stats['last_update'] = time.time()
        
    def to_dict(self) -> Dict[str, Any]:
        """Serialize entire field to dictionary"""
        return {
            'dimensions': self.dimensions,
            'embedding_dim': self.embedding_dim,
            'nodes': {nid: node.to_dict() for nid, node in self.nodes.items()},
            'edges': {eid: edge.to_dict() for eid, edge in self.edges.items()},
            'statistics': self.get_field_statistics(),
            'field_energy': self.total_energy,
            'field_health': self.total_health
        }


# Global semantic field instance
_semantic_field = None

def get_semantic_field(dimensions: int = 3, embedding_dim: int = 512) -> SemanticField:
    """Get the global semantic field instance"""
    global _semantic_field
    if _semantic_field is None:
        _semantic_field = SemanticField(dimensions, embedding_dim)
    return _semantic_field
