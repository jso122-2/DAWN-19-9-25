"""
🍄 Mycelial Layer Core Implementation
====================================

Core classes for the mycelial layer - a living substrate for cognition that grows,
prunes, and redistributes resources in response to pressure, drift, and entropy.

Based on DAWN-docs/Fractal Memory/Mycelial Layer.rtf specifications.
"""

import torch
import torch.nn as nn
import torch.nn.functional as F
import numpy as np
import time
import math
from typing import Dict, List, Optional, Set, Tuple, Any, Union
from dataclasses import dataclass, field
from enum import Enum
from collections import defaultdict, deque
import threading
import logging

logger = logging.getLogger(__name__)

class NodeState(Enum):
    """States of mycelial nodes"""
    HEALTHY = "healthy"      # Normal operation
    STARVING = "starving"    # Low energy, needs nutrients
    BLOOMING = "blooming"    # High energy, can share resources
    AUTOPHAGY = "autophagy"  # Self-digesting to create metabolites
    DORMANT = "dormant"      # Inactive but stable

class EdgeType(Enum):
    """Types of mycelial connections"""
    SEMANTIC = "semantic"    # Meaning-based connection
    TEMPORAL = "temporal"    # Time-based connection
    CAUSAL = "causal"       # Cause-effect relationship
    ASSOCIATIVE = "associative"  # Association-based
    METABOLIC = "metabolic"  # Resource flow channel

@dataclass
class MycelialNode:
    """
    A node in the mycelial layer with biological-inspired properties.
    
    Each node has its own internal health, demand, and energy state.
    Nodes can grow, maintain connections, or undergo autophagy.
    """
    
    # Identity
    id: str
    created_at: float = field(default_factory=time.time)
    
    # State variables
    health: float = 1.0              # [0,1] structural vitality
    energy: float = 0.5              # [0,1] current energy level
    demand: float = 0.0              # Current nutrient demand
    
    # Biological properties
    basal_cost: float = 0.01         # Energy cost per tick
    max_energy: float = 1.0          # Maximum energy capacity
    metabolic_rate: float = 0.1      # Rate of energy conversion
    
    # Network properties
    position: torch.Tensor = field(default_factory=lambda: torch.zeros(3))
    connections: Dict[str, 'MycelialEdge'] = field(default_factory=dict)
    cluster_id: Optional[str] = None
    
    # Pressure and drift alignment
    pressure: float = 0.0            # Current cognitive pressure
    drift_alignment: float = 0.0     # Alignment with system drift
    recency: float = 1.0            # How recently accessed
    entropy: float = 0.0            # Local entropy level
    
    # State tracking
    state: NodeState = NodeState.HEALTHY
    last_update: float = field(default_factory=time.time)
    tick_count: int = 0
    
    # Autophagy tracking
    energy_low_ticks: int = 0        # Ticks below energy threshold
    autophagy_threshold: float = 0.1  # Energy level that triggers autophagy
    autophagy_time_threshold: int = 5 # Ticks before autophagy triggers
    
    # Metabolite production
    metabolites_produced: List[Dict[str, Any]] = field(default_factory=list)
    
    def __post_init__(self):
        """Initialize computed properties"""
        if isinstance(self.position, (list, tuple)):
            self.position = torch.tensor(self.position, dtype=torch.float32)
    
    def compute_demand(self, 
                      pressure_weight: float = 1.0,
                      drift_weight: float = 0.8,
                      recency_weight: float = 0.5,
                      entropy_weight: float = -0.3) -> float:
        """
        Compute nutrient demand based on DAWN formula:
        D_i = wP*P_i + wΔ*drift_align_i + wR*recency_i - wσ*σ_i
        """
        demand = (pressure_weight * self.pressure +
                 drift_weight * self.drift_alignment +
                 recency_weight * self.recency +
                 entropy_weight * self.entropy)
        
        self.demand = max(0.0, demand)
        return self.demand
    
    def update_energy(self, nutrients_received: float, efficiency: float = 0.9) -> float:
        """
        Update energy based on nutrients received:
        energy_i = clamp(energy_i + η*nutrients_i - basal_cost_i, 0, E_max)
        """
        energy_gain = efficiency * nutrients_received
        new_energy = self.energy + energy_gain - self.basal_cost
        
        old_energy = self.energy
        self.energy = max(0.0, min(new_energy, self.max_energy))
        
        # Update state based on energy level
        self._update_state()
        
        return self.energy - old_energy
    
    def _update_state(self):
        """Update node state based on current conditions"""
        if self.energy < self.autophagy_threshold:
            self.energy_low_ticks += 1
            if self.energy_low_ticks >= self.autophagy_time_threshold:
                self.state = NodeState.AUTOPHAGY
            else:
                self.state = NodeState.STARVING
        elif self.energy > 0.8:
            self.state = NodeState.BLOOMING
            self.energy_low_ticks = 0
        elif self.energy > 0.3:
            self.state = NodeState.HEALTHY
            self.energy_low_ticks = 0
        else:
            self.state = NodeState.STARVING
            self.energy_low_ticks = max(0, self.energy_low_ticks - 1)
    
    def can_grow_connection(self, target_node: 'MycelialNode',
                          similarity_threshold: float = 0.5,
                          energy_threshold: float = 0.6) -> bool:
        """
        Check if this node can grow a connection to target node.
        Implements growth gate logic.
        """
        # Energy requirement
        if self.energy < energy_threshold:
            return False
        
        # Don't connect to self
        if self.id == target_node.id:
            return False
        
        # Already connected
        if target_node.id in self.connections:
            return False
        
        # Similarity check (placeholder - implement based on semantic content)
        similarity = self._compute_similarity(target_node)
        if similarity < similarity_threshold:
            return False
        
        # Temporal proximity (placeholder)
        temporal_ok = abs(self.last_update - target_node.last_update) < 10.0
        
        # Mood/pressure compatibility
        pressure_compatible = abs(self.pressure - target_node.pressure) < 0.5
        
        return temporal_ok and pressure_compatible
    
    def _compute_similarity(self, other_node: 'MycelialNode') -> float:
        """Compute similarity with another node (placeholder)"""
        # Implement based on semantic content, position, etc.
        position_similarity = 1.0 / (1.0 + torch.norm(self.position - other_node.position).item())
        pressure_similarity = 1.0 - abs(self.pressure - other_node.pressure)
        return (position_similarity + pressure_similarity) / 2.0
    
    def trigger_autophagy(self) -> List[Dict[str, Any]]:
        """
        Trigger autophagy - convert node's history to metabolites.
        Returns list of metabolites for neighboring nodes to absorb.
        """
        metabolites = []
        
        if self.state == NodeState.AUTOPHAGY:
            # Create metabolites from node's properties
            metabolite = {
                'type': 'semantic_trace',
                'source_id': self.id,
                'energy_content': self.energy * 0.8,  # 80% recovery
                'semantic_content': {
                    'pressure': self.pressure,
                    'drift_alignment': self.drift_alignment,
                    'position': self.position.clone(),
                    'connections': list(self.connections.keys())
                },
                'timestamp': time.time()
            }
            metabolites.append(metabolite)
            self.metabolites_produced.append(metabolite)
            
            # Reduce own energy to near zero
            self.energy = 0.05
            self.state = NodeState.DORMANT
        
        return metabolites
    
    def absorb_metabolite(self, metabolite: Dict[str, Any]) -> bool:
        """Absorb a metabolite from a decomposed neighbor"""
        if self.energy >= self.max_energy:
            return False
        
        # Gain energy from metabolite
        energy_gain = metabolite.get('energy_content', 0.0) * 0.6  # 60% efficiency
        self.energy = min(self.max_energy, self.energy + energy_gain)
        
        # Bias toward recovering patterns from metabolite
        semantic_content = metabolite.get('semantic_content', {})
        if semantic_content:
            # Slightly bias pressure and drift toward absorbed values
            bias_strength = 0.1
            self.pressure += bias_strength * (semantic_content.get('pressure', 0) - self.pressure)
            self.drift_alignment += bias_strength * (semantic_content.get('drift_alignment', 0) - self.drift_alignment)
        
        return True
    
    def tick_update(self, delta_time: float = 1.0):
        """Update node state for one tick"""
        self.tick_count += 1
        self.last_update = time.time()
        
        # Decay recency
        self.recency *= 0.99
        
        # Update demand
        self.compute_demand()
        
        # Update state
        self._update_state()

@dataclass
class MycelialEdge:
    """
    An edge in the mycelial layer representing a conductive channel
    for both information and resources.
    """
    
    # Identity
    id: str
    source_id: str
    target_id: str
    edge_type: EdgeType = EdgeType.SEMANTIC
    
    # Connection properties
    weight: float = 1.0              # Connection strength
    conductivity: float = 0.9        # Energy flow efficiency
    reliability: float = 0.8         # Historical stability
    volatility: float = 0.1          # Likelihood of change
    
    # Flow properties
    passive_flow_rate: float = 0.0   # Current passive flow
    active_flow_rate: float = 0.0    # Current active flow
    flow_history: deque = field(default_factory=lambda: deque(maxlen=50))
    
    # Decay properties
    shimmer_decay_rate: float = 0.01 # Rate of weight decay
    time_decay_rate: float = 0.005   # Time-based decay
    entropy_decay_rate: float = 0.02 # Entropy-based decay
    
    # State tracking
    created_at: float = field(default_factory=time.time)
    last_used: float = field(default_factory=time.time)
    usage_count: int = 0
    
    def update_weight(self, 
                     similarity: float,
                     energy_source: float,
                     energy_target: float,
                     delta_time: float = 1.0) -> float:
        """
        Update edge weight based on DAWN formula:
        Δw_ij = α * similarity(i,j) * reliability_ij * f(energy_i,energy_j)
               - β * time_decay_ij - χ * mean_entropy_ij
        """
        # Positive reinforcement term
        energy_factor = math.sqrt(energy_source * energy_target)  # f(energy_i, energy_j)
        reinforcement = 0.1 * similarity * self.reliability * energy_factor
        
        # Decay terms
        time_decay = self.time_decay_rate * delta_time
        entropy_decay = self.entropy_decay_rate * self.volatility
        
        weight_change = reinforcement - time_decay - entropy_decay
        old_weight = self.weight
        self.weight = max(0.0, self.weight + weight_change)
        
        # Update reliability based on weight stability
        weight_stability = 1.0 - abs(weight_change) / max(old_weight, 0.1)
        self.reliability = 0.9 * self.reliability + 0.1 * weight_stability
        
        return weight_change
    
    def compute_conductivity(self) -> float:
        """Compute edge conductivity: g_ij = sigmoid(κ * w_ij)"""
        kappa = 2.0  # Scaling parameter
        self.conductivity = torch.sigmoid(torch.tensor(kappa * self.weight)).item()
        return self.conductivity
    
    def compute_passive_flow(self, energy_source: float, energy_target: float) -> float:
        """
        Compute passive energy flow: F_passive_ij = g_ij * (energy_i - energy_j)
        """
        self.compute_conductivity()
        self.passive_flow_rate = self.conductivity * (energy_source - energy_target)
        return self.passive_flow_rate
    
    def compute_active_flow(self, 
                           energy_source: float,
                           bloom_source: float = 0.0,
                           starve_target: float = 0.0,
                           gamma: float = 0.2) -> float:
        """
        Compute active energy flow: F_active_ij = γ * g_ij * (bloom_i + starve_j) * energy_i
        """
        self.compute_conductivity()
        self.active_flow_rate = (gamma * self.conductivity * 
                               (bloom_source + starve_target) * energy_source)
        return self.active_flow_rate
    
    def apply_shimmer_decay(self, delta_time: float = 1.0):
        """Apply shimmer decay to reduce weight over time"""
        decay_amount = self.shimmer_decay_rate * delta_time
        self.weight = max(0.0, self.weight - decay_amount)
        
        # Increase volatility as weight decreases
        if self.weight < 0.1:
            self.volatility = min(1.0, self.volatility + 0.1 * delta_time)
    
    def record_usage(self):
        """Record usage of this edge"""
        self.last_used = time.time()
        self.usage_count += 1
        self.flow_history.append({
            'timestamp': self.last_used,
            'passive_flow': self.passive_flow_rate,
            'active_flow': self.active_flow_rate,
            'weight': self.weight
        })

class MycelialLayer:
    """
    The main mycelial layer class that manages the living substrate for cognition.
    
    A nervous system that grows, prunes, and redistributes resources in response
    to pressure, drift, and entropy.
    """
    
    def __init__(self, 
                 max_nodes: int = 10000,
                 max_edges_per_node: int = 50,
                 tick_rate: float = 1.0):
        
        # Core storage
        self.nodes: Dict[str, MycelialNode] = {}
        self.edges: Dict[str, MycelialEdge] = {}
        self.node_edges: Dict[str, Set[str]] = defaultdict(set)
        
        # Configuration
        self.max_nodes = max_nodes
        self.max_edges_per_node = max_edges_per_node
        self.tick_rate = tick_rate
        
        # Global state
        self.total_energy = 0.0
        self.global_pressure = 0.0
        self.global_drift = 0.0
        self.tick_count = 0
        
        # Nutrient budget (managed by NutrientEconomy)
        self.global_budget = 100.0
        self.budget_per_tick = 10.0
        
        # Growth gates
        self.growth_threshold_energy = 0.6
        self.growth_threshold_similarity = 0.5
        
        # Statistics
        self.stats = {
            'nodes_created': 0,
            'nodes_pruned': 0,
            'edges_created': 0,
            'edges_pruned': 0,
            'metabolites_created': 0,
            'energy_flows': 0
        }
        
        # Thread safety
        self._lock = threading.RLock()
        
        logger.info(f"MycelialLayer initialized with max_nodes={max_nodes}")
    
    def add_node(self, node_id: str, **kwargs) -> MycelialNode:
        """Add a new node to the mycelial layer"""
        with self._lock:
            if node_id in self.nodes:
                return self.nodes[node_id]
            
            if len(self.nodes) >= self.max_nodes:
                self._prune_weakest_nodes(1)
            
            node = MycelialNode(id=node_id, **kwargs)
            self.nodes[node_id] = node
            self.node_edges[node_id] = set()
            self.stats['nodes_created'] += 1
            
            logger.debug(f"Added node {node_id}")
            return node
    
    def add_edge(self, source_id: str, target_id: str, 
                edge_type: EdgeType = EdgeType.SEMANTIC, **kwargs) -> Optional[MycelialEdge]:
        """Add a new edge between nodes"""
        with self._lock:
            if source_id not in self.nodes or target_id not in self.nodes:
                return None
            
            edge_id = f"{source_id}->{target_id}"
            if edge_id in self.edges:
                return self.edges[edge_id]
            
            # Check growth gate
            source_node = self.nodes[source_id]
            target_node = self.nodes[target_id]
            
            if not source_node.can_grow_connection(target_node):
                return None
            
            # Check edge limits
            if len(self.node_edges[source_id]) >= self.max_edges_per_node:
                self._prune_weakest_edges(source_id, 1)
            
            edge = MycelialEdge(
                id=edge_id,
                source_id=source_id,
                target_id=target_id,
                edge_type=edge_type,
                **kwargs
            )
            
            self.edges[edge_id] = edge
            self.node_edges[source_id].add(edge_id)
            self.node_edges[target_id].add(edge_id)
            
            source_node.connections[target_id] = edge
            target_node.connections[source_id] = edge
            
            self.stats['edges_created'] += 1
            
            logger.debug(f"Added edge {edge_id}")
            return edge
    
    def tick_update(self, delta_time: Optional[float] = None):
        """Perform one tick update of the mycelial layer"""
        if delta_time is None:
            delta_time = 1.0 / self.tick_rate
        
        with self._lock:
            self.tick_count += 1
            
            # Update all nodes
            for node in self.nodes.values():
                node.tick_update(delta_time)
            
            # Update energy flows
            self._update_energy_flows(delta_time)
            
            # Apply decay to edges
            self._apply_edge_decay(delta_time)
            
            # Handle autophagy
            self._process_autophagy()
            
            # Prune dead edges and nodes
            self._prune_dead_connections()
            
            # Update global statistics
            self._update_global_stats()
    
    def _update_energy_flows(self, delta_time: float):
        """Update energy flows between connected nodes"""
        for edge in self.edges.values():
            source = self.nodes.get(edge.source_id)
            target = self.nodes.get(edge.target_id)
            
            if not source or not target:
                continue
            
            # Compute flows
            passive_flow = edge.compute_passive_flow(source.energy, target.energy)
            
            bloom_source = 1.0 if source.state == NodeState.BLOOMING else 0.0
            starve_target = 1.0 if target.state == NodeState.STARVING else 0.0
            active_flow = edge.compute_active_flow(source.energy, bloom_source, starve_target)
            
            # Apply flows
            total_flow = (passive_flow + active_flow) * delta_time * edge.conductivity
            
            if total_flow > 0 and source.energy > total_flow:
                source.energy -= total_flow
                target.energy = min(target.max_energy, target.energy + total_flow * 0.9)  # 90% efficiency
                edge.record_usage()
                self.stats['energy_flows'] += 1
    
    def _apply_edge_decay(self, delta_time: float):
        """Apply shimmer decay to all edges"""
        for edge in self.edges.values():
            edge.apply_shimmer_decay(delta_time)
    
    def _process_autophagy(self):
        """Process autophagy for starving nodes"""
        metabolites_by_location = defaultdict(list)
        
        for node in list(self.nodes.values()):
            if node.state == NodeState.AUTOPHAGY:
                metabolites = node.trigger_autophagy()
                
                # Distribute metabolites to neighbors
                for neighbor_id in node.connections.keys():
                    metabolites_by_location[neighbor_id].extend(metabolites)
                
                self.stats['metabolites_created'] += len(metabolites)
        
        # Let neighbors absorb metabolites
        for neighbor_id, metabolites in metabolites_by_location.items():
            neighbor = self.nodes.get(neighbor_id)
            if neighbor:
                for metabolite in metabolites:
                    neighbor.absorb_metabolite(metabolite)
    
    def _prune_dead_connections(self):
        """Remove edges and nodes that have become too weak"""
        # Remove edges with weight below threshold
        dead_edges = []
        for edge_id, edge in self.edges.items():
            if edge.weight < 0.01:  # Threshold for edge death
                dead_edges.append(edge_id)
        
        for edge_id in dead_edges:
            self._remove_edge(edge_id)
        
        # Remove nodes with energy too low for too long
        dead_nodes = []
        for node_id, node in self.nodes.items():
            if node.energy < 0.01 and node.energy_low_ticks > 20:
                dead_nodes.append(node_id)
        
        for node_id in dead_nodes:
            self._remove_node(node_id)
    
    def _remove_edge(self, edge_id: str):
        """Remove an edge from the layer"""
        if edge_id not in self.edges:
            return
        
        edge = self.edges[edge_id]
        
        # Remove from node connections
        if edge.source_id in self.nodes:
            self.nodes[edge.source_id].connections.pop(edge.target_id, None)
        if edge.target_id in self.nodes:
            self.nodes[edge.target_id].connections.pop(edge.source_id, None)
        
        # Remove from node edges tracking
        self.node_edges[edge.source_id].discard(edge_id)
        self.node_edges[edge.target_id].discard(edge_id)
        
        del self.edges[edge_id]
        self.stats['edges_pruned'] += 1
    
    def _remove_node(self, node_id: str):
        """Remove a node and all its edges"""
        if node_id not in self.nodes:
            return
        
        # Remove all edges connected to this node
        connected_edges = list(self.node_edges[node_id])
        for edge_id in connected_edges:
            self._remove_edge(edge_id)
        
        del self.nodes[node_id]
        del self.node_edges[node_id]
        self.stats['nodes_pruned'] += 1
    
    def _prune_weakest_nodes(self, count: int):
        """Remove the weakest nodes to make room"""
        if len(self.nodes) < count:
            return
        
        # Sort by energy level
        sorted_nodes = sorted(self.nodes.items(), key=lambda x: x[1].energy)
        
        for i in range(min(count, len(sorted_nodes))):
            node_id = sorted_nodes[i][0]
            self._remove_node(node_id)
    
    def _prune_weakest_edges(self, node_id: str, count: int):
        """Remove the weakest edges from a node"""
        node_edge_ids = list(self.node_edges[node_id])
        if len(node_edge_ids) < count:
            return
        
        # Sort by weight
        sorted_edges = sorted(
            [(eid, self.edges[eid]) for eid in node_edge_ids],
            key=lambda x: x[1].weight
        )
        
        for i in range(min(count, len(sorted_edges))):
            edge_id = sorted_edges[i][0]
            self._remove_edge(edge_id)
    
    def _update_global_stats(self):
        """Update global statistics"""
        self.total_energy = sum(node.energy for node in self.nodes.values())
        
        if self.nodes:
            self.global_pressure = sum(node.pressure for node in self.nodes.values()) / len(self.nodes)
            self.global_drift = sum(node.drift_alignment for node in self.nodes.values()) / len(self.nodes)
    
    def get_stats(self) -> Dict[str, Any]:
        """Get current layer statistics"""
        return {
            'tick_count': self.tick_count,
            'node_count': len(self.nodes),
            'edge_count': len(self.edges),
            'total_energy': self.total_energy,
            'global_pressure': self.global_pressure,
            'global_drift': self.global_drift,
            'avg_energy': self.total_energy / max(1, len(self.nodes)),
            **self.stats
        }
