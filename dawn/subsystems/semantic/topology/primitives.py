"""
Semantic Topology Primitives
============================

Core primitives for DAWN's semantic topology system based on RTF specifications.
Implements nodes, edges, layers, sectors, frames and coordinate systems.
"""

import uuid
import numpy as np
import time
from typing import Dict, List, Set, Optional, Tuple, Any, Union
from dataclasses import dataclass, field
from enum import Enum
from datetime import datetime
import logging

logger = logging.getLogger(__name__)


class TopologySector(Enum):
    """Regional partition of semantic space"""
    CORE = "core"                    # Central high-energy concepts
    PERIPHERAL = "peripheral"        # Edge concepts and transitions  
    TRANSITIONAL = "transitional"    # Bridge regions between core/peripheral
    DEEP = "deep"                   # Underlayer archived concepts


@dataclass
class NodeCoordinates:
    """Coordinate system for semantic topology"""
    position: np.ndarray           # Projected coords (default 3D for GUI)
    embedding: np.ndarray          # High-dimensional latent semantics
    layer: int = 0                 # Depth plane (0=surface, 1+=deeper)
    sector: TopologySector = TopologySector.PERIPHERAL
    
    def __post_init__(self):
        """Normalize and validate coordinates"""
        if self.position.size == 0:
            self.position = np.zeros(3)
        
        # Ensure position is 3D for GUI compatibility
        if len(self.position) < 3:
            self.position = np.pad(self.position, (0, 3 - len(self.position)))
        elif len(self.position) > 3:
            self.position = self.position[:3]
            
        # Normalize embedding if provided
        if self.embedding.size > 0 and np.linalg.norm(self.embedding) > 0:
            self.embedding = self.embedding / np.linalg.norm(self.embedding)
    
    def semantic_distance(self, other: 'NodeCoordinates') -> float:
        """Calculate semantic distance in embedding space"""
        if self.embedding.size == 0 or other.embedding.size == 0:
            return float('inf')
        return np.linalg.norm(self.embedding - other.embedding)
    
    def spatial_distance(self, other: 'NodeCoordinates') -> float:
        """Calculate spatial distance in projected space"""
        return np.linalg.norm(self.position - other.position)


@dataclass 
class SemanticNode:
    """Semantic unit in topology (concept/memory/cluster)"""
    id: str
    coordinates: NodeCoordinates
    tint: np.ndarray = field(default_factory=lambda: np.array([0.5, 0.5, 0.5]))  # RGB pigment bias
    health: float = 1.0            # Structural vitality [0,1]
    energy: float = 0.5            # Mycelial budget at node [0,1]
    last_touch: int = 0            # Tick when last updated
    
    # Topology-specific fields
    local_coherence: float = 0.5   # Structural harmony around node
    residue_pressure: float = 0.0  # Local soot/ash/entropy pressure
    content: str = ""              # Semantic content description
    
    def __post_init__(self):
        """Initialize and validate node"""
        if not self.id:
            self.id = str(uuid.uuid4())
        
        # Clamp tint to [0,1] RGB
        self.tint = np.clip(self.tint, 0.0, 1.0)
        
        # Clamp health and energy
        self.health = max(0.0, min(1.0, self.health))
        self.energy = max(0.0, min(1.0, self.energy))
        
        if self.last_touch == 0:
            self.last_touch = int(time.time())
    
    def calculate_similarity(self, other: 'SemanticNode') -> float:
        """Calculate semantic similarity with other node"""
        return 1.0 / (1.0 + self.coordinates.semantic_distance(other.coordinates))
    
    def is_healthy(self) -> bool:
        """Check if node is in healthy state"""
        return self.health > 0.3 and self.energy > 0.1
    
    def get_pigment_intensity(self) -> float:
        """Get overall pigment intensity"""
        return np.linalg.norm(self.tint)


@dataclass
class SemanticEdge:
    """Relation between semantic nodes (directed or undirected)"""
    id: str
    source_id: str
    target_id: str
    weight: float = 0.5            # Connection strength [0,1]
    tension: float = 0.0           # Strain from geometric mismatch [0,∞)
    volatility: float = 0.1        # Likelihood of change [0,1]
    reliability: float = 0.8       # Historical stability [0,1]
    directed: bool = False         # Whether edge is directional
    
    # Topology-specific fields
    nutrient_flow: float = 0.0     # Energy flow along edge
    rattling_phase: float = 0.0    # Quantum-like oscillation
    
    def __post_init__(self):
        """Initialize and validate edge"""
        if not self.id:
            self.id = f"{self.source_id}-{self.target_id}"
        
        # Clamp values to valid ranges
        self.weight = max(0.0, min(1.0, self.weight))
        self.tension = max(0.0, self.tension)
        self.volatility = max(0.0, min(1.0, self.volatility))
        self.reliability = max(0.0, min(1.0, self.reliability))
        self.nutrient_flow = max(0.0, min(1.0, self.nutrient_flow))
    
    def get_effective_weight(self, direction: str = "forward") -> float:
        """Get connection strength with directional rattling"""
        rattle_factor = np.sin(self.rattling_phase) * 0.2 + 1.0
        if direction == "reverse":
            rattle_factor = np.cos(self.rattling_phase) * 0.2 + 1.0
        return self.weight * rattle_factor
    
    def is_stable(self) -> bool:
        """Check if edge is in stable state"""
        return self.tension < 0.5 and self.reliability > 0.6
    
    def update_rattling(self, delta_time: float = 1.0):
        """Update quantum rattling phase"""
        self.rattling_phase += delta_time * 0.1  # Slow oscillation
        self.rattling_phase %= 2 * np.pi


@dataclass
class TopologyLayer:
    """Depth plane in semantic topology"""
    depth: int                     # Layer depth (0=surface, 1+=deeper)
    node_ids: Set[str] = field(default_factory=set)
    capacity: int = 1000          # Max nodes per layer
    decay_rate: float = 0.01      # Natural decay toward deeper layers
    rebloom_threshold: float = 0.7 # Energy threshold for lift operations
    
    def add_node(self, node_id: str) -> bool:
        """Add node to layer if capacity allows"""
        if len(self.node_ids) >= self.capacity:
            return False
        self.node_ids.add(node_id)
        return True
    
    def remove_node(self, node_id: str) -> bool:
        """Remove node from layer"""
        if node_id in self.node_ids:
            self.node_ids.remove(node_id)
            return True
        return False
    
    def get_occupancy_ratio(self) -> float:
        """Get layer occupancy as ratio of capacity"""
        return len(self.node_ids) / max(1, self.capacity)


@dataclass
class TopologyFrame:
    """Moving reference window for local topology operations"""
    center: np.ndarray             # Center position in topology space
    radius: float = 1.0           # Spatial radius for operations
    layer: int = 0                # Primary layer focus
    filters: Dict[str, Any] = field(default_factory=dict)
    
    # Operation budgets
    motion_budget: float = 0.1     # Available movement per tick
    energy_budget: float = 0.1     # Available energy for operations
    
    def __post_init__(self):
        """Initialize frame with defaults"""
        if self.center.size == 0:
            self.center = np.zeros(3)
        
        # Ensure center is 3D
        if len(self.center) < 3:
            self.center = np.pad(self.center, (0, 3 - len(self.center)))
        elif len(self.center) > 3:
            self.center = self.center[:3]
    
    def contains_position(self, position: np.ndarray) -> bool:
        """Check if position is within frame"""
        distance = np.linalg.norm(position - self.center)
        return distance <= self.radius
    
    def contains_node(self, node: SemanticNode) -> bool:
        """Check if node is within frame"""
        # Check spatial containment
        if not self.contains_position(node.coordinates.position):
            return False
        
        # Check layer filter
        if node.coordinates.layer != self.layer:
            return False
        
        # Check additional filters
        if 'sector' in self.filters:
            allowed_sectors = self.filters['sector']
            if isinstance(allowed_sectors, list):
                if node.coordinates.sector.value not in allowed_sectors:
                    return False
            elif node.coordinates.sector.value != allowed_sectors:
                return False
        
        return True
    
    def move_toward(self, target: np.ndarray, speed: float = 0.1):
        """Move frame center toward target position"""
        direction = target - self.center
        distance = np.linalg.norm(direction)
        
        if distance > 0:
            # Normalize direction and apply speed
            movement = (direction / distance) * min(speed, distance)
            
            # Apply motion budget constraint
            movement_magnitude = np.linalg.norm(movement)
            if movement_magnitude > self.motion_budget:
                movement = movement * (self.motion_budget / movement_magnitude)
            
            self.center += movement
            self.motion_budget -= movement_magnitude


@dataclass
class TopologyMetrics:
    """Metrics for topology health and performance"""
    total_nodes: int = 0
    total_edges: int = 0
    average_coherence: float = 0.0
    average_tension: float = 0.0
    pigment_entropy: float = 0.0
    residue_pressure_mean: float = 0.0
    
    # Layer distribution
    layer_occupancy: Dict[int, int] = field(default_factory=dict)
    
    # Sector distribution  
    sector_distribution: Dict[str, int] = field(default_factory=dict)
    
    # Performance metrics
    update_time_ms: float = 0.0
    memory_usage_mb: float = 0.0
    
    def calculate_topology_health(self) -> float:
        """Calculate overall topology health score [0,1]"""
        health_factors = []
        
        # Coherence factor (higher is better)
        if self.average_coherence > 0:
            health_factors.append(min(1.0, self.average_coherence))
        
        # Tension factor (lower is better)
        tension_health = max(0.0, 1.0 - self.average_tension)
        health_factors.append(tension_health)
        
        # Pigment entropy factor (moderate entropy is healthy)
        if self.pigment_entropy > 0:
            # Optimal entropy around 0.5
            entropy_health = 1.0 - abs(self.pigment_entropy - 0.5) * 2
            health_factors.append(max(0.0, entropy_health))
        
        # Residue pressure factor (lower is better)
        pressure_health = max(0.0, 1.0 - self.residue_pressure_mean)
        health_factors.append(pressure_health)
        
        # Return weighted average
        if health_factors:
            return sum(health_factors) / len(health_factors)
        return 0.5
    
    def to_dict(self) -> Dict[str, Any]:
        """Convert metrics to dictionary for serialization"""
        return {
            'total_nodes': self.total_nodes,
            'total_edges': self.total_edges,
            'average_coherence': self.average_coherence,
            'average_tension': self.average_tension,
            'pigment_entropy': self.pigment_entropy,
            'residue_pressure_mean': self.residue_pressure_mean,
            'layer_occupancy': self.layer_occupancy,
            'sector_distribution': self.sector_distribution,
            'topology_health': self.calculate_topology_health(),
            'update_time_ms': self.update_time_ms,
            'memory_usage_mb': self.memory_usage_mb
        }


def create_semantic_node(content: str, embedding: np.ndarray, 
                        position: Optional[np.ndarray] = None,
                        sector: TopologySector = TopologySector.PERIPHERAL,
                        layer: int = 0) -> SemanticNode:
    """Factory function to create semantic node with proper initialization"""
    if position is None:
        position = np.random.randn(3) * 0.1  # Small random position
    
    coordinates = NodeCoordinates(
        position=position,
        embedding=embedding,
        layer=layer,
        sector=sector
    )
    
    return SemanticNode(
        id=str(uuid.uuid4()),
        coordinates=coordinates,
        content=content,
        tint=np.random.rand(3),  # Random initial pigment
        last_touch=int(time.time())
    )


def create_semantic_edge(source_id: str, target_id: str,
                        weight: float = 0.5,
                        directed: bool = False) -> SemanticEdge:
    """Factory function to create semantic edge with proper initialization"""
    return SemanticEdge(
        id=f"{source_id}-{target_id}",
        source_id=source_id,
        target_id=target_id,
        weight=weight,
        directed=directed,
        rattling_phase=np.random.rand() * 2 * np.pi
    )
