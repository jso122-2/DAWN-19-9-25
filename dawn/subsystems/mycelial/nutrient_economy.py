"""
🌱 Nutrient Economy Implementation
=================================

Implements the global nutrient budget and allocation system for the mycelial layer.
Each tick computes demand per node and allocates nutrients via global budget.

Based on DAWN formula:
- Demand: D_i = wP*P_i + wΔ*drift_align_i + wR*recency_i - wσ*σ_i  
- Allocation: a_i = softmax(D)_i * B_t
- Energy: energy_i = clamp(energy_i + η*nutrients_i - basal_cost_i, 0, E_max)
"""

import torch
import torch.nn.functional as F
import numpy as np
import time
import math
from typing import Dict, List, Optional, Tuple, Any
from dataclasses import dataclass, field
from enum import Enum
from collections import defaultdict, deque
import threading
import logging

logger = logging.getLogger(__name__)

class NutrientType(Enum):
    """Types of nutrients in the mycelial economy"""
    ENERGY = "energy"                # Basic energy for maintenance
    GROWTH = "growth"               # Resources for new connections
    MAINTENANCE = "maintenance"      # Resources for connection upkeep
    METABOLIC = "metabolic"         # Resources from decomposition

@dataclass
class NutrientPacket:
    """A packet of nutrients being distributed"""
    nutrient_type: NutrientType
    amount: float
    target_node_id: str
    source: str = "global_budget"
    efficiency: float = 0.9
    timestamp: float = field(default_factory=time.time)
    metadata: Dict[str, Any] = field(default_factory=dict)

@dataclass
class GlobalNutrientBudget:
    """Global nutrient budget management"""
    
    # Budget parameters
    total_budget: float = 100.0
    budget_per_tick: float = 10.0
    emergency_reserve: float = 20.0
    
    # Allocation weights  
    pressure_weight: float = 1.0      # wP
    drift_weight: float = 0.8         # wΔ  
    recency_weight: float = 0.5       # wR
    entropy_weight: float = -0.3      # wσ (negative because entropy reduces demand)
    
    # Efficiency parameters
    base_efficiency: float = 0.9      # η - metabolic conversion efficiency
    transport_loss: float = 0.05      # Loss during nutrient transport
    
    # Tracking
    allocated_this_tick: float = 0.0
    total_allocated: float = 0.0
    allocation_history: deque = field(default_factory=lambda: deque(maxlen=1000))
    
    # Adaptive parameters
    pressure_adaptation: float = 0.01  # How much to adapt weights based on results
    demand_smoothing: float = 0.1     # Smoothing factor for demand calculations

class NutrientEconomy:
    """
    Manages the complete nutrient economy for the mycelial layer.
    
    Computes demand, allocates nutrients via global budget, and tracks
    the flow of resources throughout the system.
    """
    
    def __init__(self, mycelial_layer):
        self.mycelial_layer = mycelial_layer
        self.budget = GlobalNutrientBudget()
        
        # Demand tracking
        self.node_demands: Dict[str, float] = {}
        self.normalized_demands: Dict[str, float] = {}
        self.allocations: Dict[str, float] = {}
        
        # Distribution tracking
        self.nutrient_packets: List[NutrientPacket] = []
        self.distribution_efficiency: float = 0.95
        
        # Statistics
        self.total_demand: float = 0.0
        self.satisfied_demand_ratio: float = 0.0
        self.energy_generation_rate: float = 0.0
        
        # History tracking
        self.demand_history: deque = deque(maxlen=1000)
        self.allocation_history: deque = deque(maxlen=1000)
        self.efficiency_history: deque = deque(maxlen=1000)
        
        # Thread safety
        self._lock = threading.RLock()
        
        logger.info("NutrientEconomy initialized")
    
    def compute_all_demands(self) -> Dict[str, float]:
        """
        Compute nutrient demand for all nodes in the mycelial layer.
        
        Uses DAWN formula: D_i = wP*P_i + wΔ*drift_align_i + wR*recency_i - wσ*σ_i
        """
        with self._lock:
            demands = {}
            
            for node_id, node in self.mycelial_layer.nodes.items():
                demand = self._compute_node_demand(node)
                demands[node_id] = demand
                node.demand = demand
            
            self.node_demands = demands
            self.total_demand = sum(demands.values())
            
            # Normalize demands using softmax for allocation
            if demands:
                demand_tensor = torch.tensor(list(demands.values()), dtype=torch.float32)
                # Apply temperature to softmax for smoother distribution
                temperature = 2.0
                normalized = F.softmax(demand_tensor / temperature, dim=0)
                
                self.normalized_demands = {
                    node_id: normalized[i].item() 
                    for i, node_id in enumerate(demands.keys())
                }
            else:
                self.normalized_demands = {}
            
            return demands
    
    def _compute_node_demand(self, node) -> float:
        """
        Compute demand for a single node using the DAWN formula.
        """
        # Base demand calculation
        demand = (
            self.budget.pressure_weight * node.pressure +
            self.budget.drift_weight * node.drift_alignment +
            self.budget.recency_weight * node.recency +
            self.budget.entropy_weight * node.entropy
        )
        
        # Additional factors
        
        # Energy deficit increases demand
        energy_deficit = max(0.0, node.max_energy - node.energy)
        demand += energy_deficit * 0.5
        
        # Starving nodes get priority
        if hasattr(node, 'state') and str(node.state) == 'NodeState.STARVING':
            demand *= 1.5
        
        # Nodes in autophagy get minimal resources (they're decomposing)
        if hasattr(node, 'state') and str(node.state) == 'NodeState.AUTOPHAGY':
            demand *= 0.1
        
        # Connection count affects demand (more connections = more maintenance)
        connection_factor = 1.0 + len(node.connections) * 0.1
        demand *= connection_factor
        
        return max(0.0, demand)
    
    def allocate_nutrients(self) -> Dict[str, float]:
        """
        Allocate nutrients based on computed demands and global budget.
        
        Uses formula: a_i = softmax(D)_i * B_t
        """
        with self._lock:
            if not self.normalized_demands:
                return {}
            
            # Available budget for this tick
            available_budget = min(
                self.budget.budget_per_tick,
                self.budget.total_budget - self.budget.emergency_reserve
            )
            
            # Allocate based on normalized demands
            allocations = {}
            total_allocated = 0.0
            
            for node_id, normalized_demand in self.normalized_demands.items():
                allocation = normalized_demand * available_budget
                allocations[node_id] = allocation
                total_allocated += allocation
            
            # Update budget tracking
            self.budget.allocated_this_tick = total_allocated
            self.budget.total_allocated += total_allocated
            self.budget.total_budget = max(0.0, self.budget.total_budget - total_allocated)
            
            # Track allocation efficiency
            if self.total_demand > 0:
                self.satisfied_demand_ratio = total_allocated / self.total_demand
            else:
                self.satisfied_demand_ratio = 1.0
            
            self.allocations = allocations
            
            # Record in history
            self.allocation_history.append({
                'timestamp': time.time(),
                'total_allocated': total_allocated,
                'total_demand': self.total_demand,
                'satisfaction_ratio': self.satisfied_demand_ratio,
                'nodes_served': len(allocations)
            })
            
            return allocations
    
    def distribute_nutrients(self) -> List[NutrientPacket]:
        """
        Create and distribute nutrient packets to nodes based on allocations.
        """
        with self._lock:
            packets = []
            
            for node_id, allocation in self.allocations.items():
                if allocation > 0:
                    packet = NutrientPacket(
                        nutrient_type=NutrientType.ENERGY,
                        amount=allocation,
                        target_node_id=node_id,
                        efficiency=self.budget.base_efficiency,
                        metadata={
                            'demand': self.node_demands.get(node_id, 0.0),
                            'normalized_demand': self.normalized_demands.get(node_id, 0.0)
                        }
                    )
                    packets.append(packet)
            
            self.nutrient_packets.extend(packets)
            return packets
    
    def apply_nutrients_to_nodes(self, packets: List[NutrientPacket]) -> Dict[str, float]:
        """
        Apply nutrient packets to their target nodes and update energy levels.
        
        Uses formula: energy_i = clamp(energy_i + η*nutrients_i - basal_cost_i, 0, E_max)
        """
        with self._lock:
            energy_changes = {}
            
            for packet in packets:
                node = self.mycelial_layer.nodes.get(packet.target_node_id)
                if not node:
                    continue
                
                # Apply nutrients with transport loss
                effective_nutrients = packet.amount * (1.0 - self.budget.transport_loss)
                
                # Update node energy using its own method
                energy_change = node.update_energy(effective_nutrients, packet.efficiency)
                energy_changes[packet.target_node_id] = energy_change
                
                # Track energy generation
                self.energy_generation_rate += effective_nutrients * packet.efficiency
            
            return energy_changes
    
    def replenish_budget(self, amount: float = None):
        """Replenish the global budget (e.g., from external sources or recycling)"""
        if amount is None:
            # Default replenishment based on system size and activity
            base_replenishment = len(self.mycelial_layer.nodes) * 0.1
            activity_bonus = self.satisfied_demand_ratio * 2.0
            amount = base_replenishment + activity_bonus
        
        self.budget.total_budget += amount
        logger.debug(f"Budget replenished by {amount:.2f}, total now {self.budget.total_budget:.2f}")
    
    def process_metabolite_nutrients(self, metabolites: List[Dict[str, Any]]) -> float:
        """
        Process metabolites from autophagy and convert them to budget nutrients.
        """
        total_recovered = 0.0
        
        for metabolite in metabolites:
            energy_content = metabolite.get('energy_content', 0.0)
            # Convert 70% of metabolite energy back to global budget
            recovered = energy_content * 0.7
            total_recovered += recovered
        
        if total_recovered > 0:
            self.replenish_budget(total_recovered)
            logger.debug(f"Recovered {total_recovered:.2f} nutrients from {len(metabolites)} metabolites")
        
        return total_recovered
    
    def tick_update(self, delta_time: float = 1.0):
        """
        Perform one tick of the nutrient economy:
        1. Compute demands
        2. Allocate nutrients  
        3. Distribute packets
        4. Apply to nodes
        5. Replenish budget
        """
        with self._lock:
            # 1. Compute all node demands
            demands = self.compute_all_demands()
            
            # 2. Allocate nutrients based on demands
            allocations = self.allocate_nutrients()
            
            # 3. Create and distribute nutrient packets
            packets = self.distribute_nutrients()
            
            # 4. Apply nutrients to nodes
            energy_changes = self.apply_nutrients_to_nodes(packets)
            
            # 5. Natural budget replenishment 
            natural_replenishment = self.budget.budget_per_tick * 0.5 * delta_time
            self.replenish_budget(natural_replenishment)
            
            # 6. Update efficiency tracking
            self._update_efficiency_metrics()
            
            # 7. Adaptive parameter adjustment
            self._adapt_parameters()
            
            # Clear processed packets
            self.nutrient_packets.clear()
            
            return {
                'demands_computed': len(demands),
                'nutrients_allocated': sum(allocations.values()),
                'packets_distributed': len(packets),
                'energy_changes': energy_changes,
                'budget_remaining': self.budget.total_budget
            }
    
    def _update_efficiency_metrics(self):
        """Update efficiency and performance metrics"""
        # Calculate distribution efficiency
        if self.total_demand > 0:
            efficiency = min(1.0, self.budget.allocated_this_tick / self.total_demand)
        else:
            efficiency = 1.0
        
        self.distribution_efficiency = (
            0.9 * self.distribution_efficiency + 0.1 * efficiency
        )
        
        # Record efficiency history
        self.efficiency_history.append({
            'timestamp': time.time(),
            'distribution_efficiency': self.distribution_efficiency,
            'satisfaction_ratio': self.satisfied_demand_ratio,
            'energy_generation_rate': self.energy_generation_rate
        })
        
        # Reset per-tick counters
        self.budget.allocated_this_tick = 0.0
        self.energy_generation_rate = 0.0
    
    def _adapt_parameters(self):
        """Adaptively adjust allocation parameters based on performance"""
        # Simple adaptation: if satisfaction ratio is low, slightly increase pressure weight
        if self.satisfied_demand_ratio < 0.7:
            self.budget.pressure_weight += self.budget.pressure_adaptation
        elif self.satisfied_demand_ratio > 0.95:
            self.budget.pressure_weight -= self.budget.pressure_adaptation * 0.5
        
        # Keep weights in reasonable bounds
        self.budget.pressure_weight = max(0.1, min(2.0, self.budget.pressure_weight))
    
    def get_economy_stats(self) -> Dict[str, Any]:
        """Get comprehensive statistics about the nutrient economy"""
        return {
            'total_budget': self.budget.total_budget,
            'budget_per_tick': self.budget.budget_per_tick,
            'emergency_reserve': self.budget.emergency_reserve,
            'total_demand': self.total_demand,
            'satisfied_demand_ratio': self.satisfied_demand_ratio,
            'distribution_efficiency': self.distribution_efficiency,
            'energy_generation_rate': self.energy_generation_rate,
            'nodes_with_demand': len(self.node_demands),
            'active_allocations': len(self.allocations),
            'allocation_weights': {
                'pressure': self.budget.pressure_weight,
                'drift': self.budget.drift_weight,
                'recency': self.budget.recency_weight,
                'entropy': self.budget.entropy_weight
            },
            'efficiency_metrics': {
                'base_efficiency': self.budget.base_efficiency,
                'transport_loss': self.budget.transport_loss,
                'distribution_efficiency': self.distribution_efficiency
            }
        }
    
    def get_node_nutrient_status(self, node_id: str) -> Optional[Dict[str, Any]]:
        """Get detailed nutrient status for a specific node"""
        if node_id not in self.mycelial_layer.nodes:
            return None
        
        node = self.mycelial_layer.nodes[node_id]
        
        return {
            'node_id': node_id,
            'current_energy': node.energy,
            'max_energy': node.max_energy,
            'energy_ratio': node.energy / node.max_energy,
            'demand': self.node_demands.get(node_id, 0.0),
            'normalized_demand': self.normalized_demands.get(node_id, 0.0),
            'allocation': self.allocations.get(node_id, 0.0),
            'state': str(node.state) if hasattr(node, 'state') else 'unknown',
            'pressure': node.pressure,
            'drift_alignment': node.drift_alignment,
            'recency': node.recency,
            'entropy': node.entropy,
            'connections': len(node.connections),
            'basal_cost': node.basal_cost
        }

class NutrientFlowVisualizer:
    """Utility class for visualizing nutrient flows and allocations"""
    
    def __init__(self, nutrient_economy: NutrientEconomy):
        self.economy = nutrient_economy
    
    def generate_flow_map(self) -> Dict[str, Any]:
        """Generate a visual map of nutrient flows"""
        flow_map = {
            'nodes': {},
            'global_stats': self.economy.get_economy_stats(),
            'timestamp': time.time()
        }
        
        # Add node information
        for node_id in self.economy.mycelial_layer.nodes.keys():
            status = self.economy.get_node_nutrient_status(node_id)
            if status:
                flow_map['nodes'][node_id] = status
        
        return flow_map
    
    def get_allocation_efficiency_trend(self, window_size: int = 50) -> List[float]:
        """Get recent allocation efficiency trend"""
        if len(self.economy.efficiency_history) < window_size:
            return [h['distribution_efficiency'] for h in self.economy.efficiency_history]
        else:
            recent = list(self.economy.efficiency_history)[-window_size:]
            return [h['distribution_efficiency'] for h in recent]
