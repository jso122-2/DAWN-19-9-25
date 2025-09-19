"""
⚡ Energy Flow Management System
===============================

Implements the dual energy flow mechanisms for the mycelial layer:
1. Passive Diffusion: Energy equalizes between connected nodes
2. Active Transport: Blooms push energy outwards, starved nodes pull energy inwards

Based on DAWN formulas:
- Passive flow: F_passive_ij = g_ij * (energy_i - energy_j)  
- Active flow: F_active_ij = γ * g_ij * (bloom_i + starve_j) * energy_i
- Conductivity: g_ij = sigmoid(κ * w_ij)
"""

import torch
import torch.nn.functional as F
import numpy as np
import time
import math
from typing import Dict, List, Optional, Tuple, Any, Set
from dataclasses import dataclass, field
from enum import Enum
from collections import defaultdict, deque
import threading
import logging

logger = logging.getLogger(__name__)

class FlowType(Enum):
    """Types of energy flows"""
    PASSIVE = "passive"      # Diffusion-based equilibration
    ACTIVE = "active"        # Directed transport (bloom/starve)
    METABOLIC = "metabolic"  # From metabolite absorption
    EXTERNAL = "external"    # From nutrient economy

@dataclass
class EnergyFlow:
    """Represents an energy flow between two nodes"""
    source_id: str
    target_id: str
    flow_type: FlowType
    flow_rate: float         # Energy units per tick
    efficiency: float = 0.95 # Transport efficiency (loss factor)
    conductivity: float = 1.0 # Edge conductivity
    
    # Flow dynamics
    resistance: float = 0.0   # Flow resistance
    momentum: float = 0.0     # Flow momentum for smoothing
    
    # Tracking
    total_transferred: float = 0.0
    transfer_count: int = 0
    last_transfer: float = 0.0
    created_at: float = field(default_factory=time.time)
    
    def apply_flow(self, delta_time: float = 1.0) -> Tuple[float, float]:
        """
        Apply this flow, returning (energy_removed_from_source, energy_added_to_target)
        """
        actual_flow = self.flow_rate * delta_time
        
        # Apply efficiency and resistance
        transferred = actual_flow * self.efficiency * (1.0 - self.resistance)
        loss = actual_flow - transferred
        
        # Update tracking
        self.total_transferred += transferred
        self.transfer_count += 1
        self.last_transfer = transferred
        
        return actual_flow, transferred

class PassiveDiffusion:
    """
    Implements passive energy diffusion between connected nodes.
    Energy naturally flows from high to low concentration.
    """
    
    def __init__(self, diffusion_rate: float = 0.1, min_flow_threshold: float = 0.001):
        self.diffusion_rate = diffusion_rate
        self.min_flow_threshold = min_flow_threshold
        
        # Physics parameters
        self.viscosity = 0.1           # Flow resistance
        self.thermal_noise = 0.001     # Random fluctuations
        self.equilibrium_threshold = 0.01  # Energy difference threshold
        
        # Tracking
        self.active_flows: Dict[str, EnergyFlow] = {}
        self.flow_history: deque = deque(maxlen=1000)
        
    def compute_passive_flows(self, mycelial_layer) -> List[EnergyFlow]:
        """
        Compute passive diffusion flows for all edges in the layer.
        
        Uses formula: F_passive_ij = g_ij * (energy_i - energy_j)
        """
        flows = []
        
        for edge in mycelial_layer.edges.values():
            source_node = mycelial_layer.nodes.get(edge.source_id)
            target_node = mycelial_layer.nodes.get(edge.target_id)
            
            if not source_node or not target_node:
                continue
            
            # Compute conductivity: g_ij = sigmoid(κ * w_ij)
            kappa = 2.0  # Scaling parameter
            conductivity = torch.sigmoid(torch.tensor(kappa * edge.weight)).item()
            
            # Energy difference
            energy_diff = source_node.energy - target_node.energy
            
            # Skip if difference is too small
            if abs(energy_diff) < self.equilibrium_threshold:
                continue
            
            # Compute passive flow rate
            base_flow_rate = conductivity * energy_diff * self.diffusion_rate
            
            # Apply thermal noise for realistic fluctuations
            noise = np.random.normal(0, self.thermal_noise)
            flow_rate = base_flow_rate + noise
            
            # Skip tiny flows
            if abs(flow_rate) < self.min_flow_threshold:
                continue
            
            # Determine flow direction
            if flow_rate > 0:
                # Flow from source to target
                flow = EnergyFlow(
                    source_id=edge.source_id,
                    target_id=edge.target_id,
                    flow_type=FlowType.PASSIVE,
                    flow_rate=flow_rate,
                    conductivity=conductivity,
                    resistance=self.viscosity
                )
            else:
                # Flow from target to source
                flow = EnergyFlow(
                    source_id=edge.target_id,
                    target_id=edge.source_id,
                    flow_type=FlowType.PASSIVE,
                    flow_rate=abs(flow_rate),
                    conductivity=conductivity,
                    resistance=self.viscosity
                )
            
            flows.append(flow)
            
        return flows
    
    def apply_passive_flows(self, flows: List[EnergyFlow], mycelial_layer, delta_time: float) -> Dict[str, float]:
        """Apply computed passive flows to the mycelial layer"""
        energy_changes = defaultdict(float)
        
        for flow in flows:
            source_node = mycelial_layer.nodes.get(flow.source_id)
            target_node = mycelial_layer.nodes.get(flow.target_id)
            
            if not source_node or not target_node:
                continue
            
            # Check if source has enough energy
            required_energy, transferred_energy = flow.apply_flow(delta_time)
            
            if source_node.energy >= required_energy:
                # Apply the flow
                source_node.energy -= required_energy
                target_node.energy = min(target_node.max_energy, 
                                       target_node.energy + transferred_energy)
                
                energy_changes[flow.source_id] -= required_energy
                energy_changes[flow.target_id] += transferred_energy
                
                # Record flow in history
                self.flow_history.append({
                    'timestamp': time.time(),
                    'source_id': flow.source_id,
                    'target_id': flow.target_id,
                    'flow_rate': flow.flow_rate,
                    'transferred': transferred_energy,
                    'conductivity': flow.conductivity
                })
        
        return dict(energy_changes)

class ActiveTransport:
    """
    Implements active energy transport based on node states.
    Blooming nodes push energy outwards, starving nodes pull energy inwards.
    """
    
    def __init__(self, transport_strength: float = 0.2, bloom_threshold: float = 0.8, starve_threshold: float = 0.3):
        self.transport_strength = transport_strength  # γ in formula
        self.bloom_threshold = bloom_threshold
        self.starve_threshold = starve_threshold
        
        # Active transport parameters
        self.bloom_push_multiplier = 2.0    # How much blooms push
        self.starve_pull_multiplier = 1.5   # How much starving nodes pull
        self.max_active_flow = 0.5          # Maximum active flow per tick
        
        # Tracking
        self.active_transports: List[EnergyFlow] = []
        self.transport_history: deque = deque(maxlen=1000)
        
    def compute_active_flows(self, mycelial_layer) -> List[EnergyFlow]:
        """
        Compute active transport flows based on node states.
        
        Uses formula: F_active_ij = γ * g_ij * (bloom_i + starve_j) * energy_i
        """
        flows = []
        
        for edge in mycelial_layer.edges.values():
            source_node = mycelial_layer.nodes.get(edge.source_id)
            target_node = mycelial_layer.nodes.get(edge.target_id)
            
            if not source_node or not target_node:
                continue
            
            # Compute conductivity
            kappa = 2.0
            conductivity = torch.sigmoid(torch.tensor(kappa * edge.weight)).item()
            
            # Determine bloom and starve factors
            bloom_source = self._compute_bloom_factor(source_node)
            starve_target = self._compute_starve_factor(target_node)
            
            # Also check reverse direction
            bloom_target = self._compute_bloom_factor(target_node)
            starve_source = self._compute_starve_factor(source_node)
            
            # Compute active flow: source -> target
            if bloom_source > 0 or starve_target > 0:
                flow_rate = (self.transport_strength * conductivity * 
                           (bloom_source + starve_target) * source_node.energy)
                
                if flow_rate > self.min_flow_threshold:
                    flow = EnergyFlow(
                        source_id=edge.source_id,
                        target_id=edge.target_id,
                        flow_type=FlowType.ACTIVE,
                        flow_rate=min(flow_rate, self.max_active_flow),
                        conductivity=conductivity,
                        efficiency=0.85  # Active transport is less efficient
                    )
                    flows.append(flow)
            
            # Compute active flow: target -> source  
            if bloom_target > 0 or starve_source > 0:
                flow_rate = (self.transport_strength * conductivity * 
                           (bloom_target + starve_source) * target_node.energy)
                
                if flow_rate > self.min_flow_threshold:
                    flow = EnergyFlow(
                        source_id=edge.target_id,
                        target_id=edge.source_id,
                        flow_type=FlowType.ACTIVE,
                        flow_rate=min(flow_rate, self.max_active_flow),
                        conductivity=conductivity,
                        efficiency=0.85
                    )
                    flows.append(flow)
        
        return flows
    
    def _compute_bloom_factor(self, node) -> float:
        """Compute how much a node is blooming (wants to share energy)"""
        if node.energy > self.bloom_threshold:
            excess = node.energy - self.bloom_threshold
            return min(1.0, excess * self.bloom_push_multiplier)
        return 0.0
    
    def _compute_starve_factor(self, node) -> float:
        """Compute how much a node is starving (wants to pull energy)"""
        if node.energy < self.starve_threshold:
            deficit = self.starve_threshold - node.energy
            return min(1.0, deficit * self.starve_pull_multiplier)
        return 0.0
    
    def apply_active_flows(self, flows: List[EnergyFlow], mycelial_layer, delta_time: float) -> Dict[str, float]:
        """Apply computed active transport flows"""
        energy_changes = defaultdict(float)
        
        for flow in flows:
            source_node = mycelial_layer.nodes.get(flow.source_id)
            target_node = mycelial_layer.nodes.get(flow.target_id)
            
            if not source_node or not target_node:
                continue
            
            # Check energy availability and capacity
            required_energy, transferred_energy = flow.apply_flow(delta_time)
            
            if (source_node.energy >= required_energy and 
                target_node.energy < target_node.max_energy):
                
                # Apply the flow
                actual_target_gain = min(transferred_energy, 
                                       target_node.max_energy - target_node.energy)
                
                source_node.energy -= required_energy
                target_node.energy += actual_target_gain
                
                energy_changes[flow.source_id] -= required_energy
                energy_changes[flow.target_id] += actual_target_gain
                
                # Record transport
                self.transport_history.append({
                    'timestamp': time.time(),
                    'source_id': flow.source_id,
                    'target_id': flow.target_id,
                    'flow_type': flow.flow_type.value,
                    'flow_rate': flow.flow_rate,
                    'transferred': actual_target_gain,
                    'source_energy': source_node.energy + required_energy,  # Before transfer
                    'target_energy': target_node.energy - actual_target_gain  # Before transfer
                })
        
        return dict(energy_changes)

class EnergyFlowManager:
    """
    Main manager for all energy flow types in the mycelial layer.
    Coordinates passive diffusion and active transport.
    """
    
    def __init__(self, mycelial_layer):
        self.mycelial_layer = mycelial_layer
        
        # Flow systems
        self.passive_diffusion = PassiveDiffusion()
        self.active_transport = ActiveTransport()
        
        # Flow coordination
        self.flow_priority = [FlowType.ACTIVE, FlowType.PASSIVE]  # Active transport first
        self.max_flows_per_tick = 1000
        self.energy_conservation_check = True
        
        # Performance tracking
        self.total_energy_transferred = 0.0
        self.flow_efficiency = 0.95
        self.energy_loss_rate = 0.05
        
        # Statistics
        self.stats = {
            'passive_flows': 0,
            'active_flows': 0,
            'total_transfers': 0,
            'energy_conserved': True,
            'average_flow_rate': 0.0
        }
        
        # Thread safety
        self._lock = threading.RLock()
        
        logger.info("EnergyFlowManager initialized")
    
    def compute_all_flows(self) -> Tuple[List[EnergyFlow], List[EnergyFlow]]:
        """Compute both passive and active flows"""
        with self._lock:
            # Compute passive diffusion flows
            passive_flows = self.passive_diffusion.compute_passive_flows(self.mycelial_layer)
            
            # Compute active transport flows
            active_flows = self.active_transport.compute_active_flows(self.mycelial_layer)
            
            return passive_flows, active_flows
    
    def apply_all_flows(self, passive_flows: List[EnergyFlow], 
                       active_flows: List[EnergyFlow], 
                       delta_time: float = 1.0) -> Dict[str, Any]:
        """Apply all energy flows in priority order"""
        with self._lock:
            total_energy_before = self._calculate_total_energy()
            all_energy_changes = defaultdict(float)
            
            # Apply active flows first (higher priority)
            active_changes = self.active_transport.apply_active_flows(
                active_flows, self.mycelial_layer, delta_time
            )
            for node_id, change in active_changes.items():
                all_energy_changes[node_id] += change
            
            # Then apply passive flows
            passive_changes = self.passive_diffusion.apply_passive_flows(
                passive_flows, self.mycelial_layer, delta_time
            )
            for node_id, change in passive_changes.items():
                all_energy_changes[node_id] += change
            
            # Energy conservation check
            total_energy_after = self._calculate_total_energy()
            energy_conserved = abs(total_energy_before - total_energy_after) < 0.001
            
            # Update statistics
            self.stats.update({
                'passive_flows': len(passive_flows),
                'active_flows': len(active_flows),
                'total_transfers': len(passive_flows) + len(active_flows),
                'energy_conserved': energy_conserved,
                'average_flow_rate': self._calculate_average_flow_rate(passive_flows + active_flows)
            })
            
            if not energy_conserved:
                logger.warning(f"Energy not conserved: {total_energy_before:.4f} -> {total_energy_after:.4f}")
            
            return {
                'energy_changes': dict(all_energy_changes),
                'passive_flows_applied': len(passive_flows),
                'active_flows_applied': len(active_flows),
                'total_energy_before': total_energy_before,
                'total_energy_after': total_energy_after,
                'energy_conserved': energy_conserved
            }
    
    def tick_update(self, delta_time: float = 1.0) -> Dict[str, Any]:
        """Perform one tick of energy flow updates"""
        with self._lock:
            # Compute all flows
            passive_flows, active_flows = self.compute_all_flows()
            
            # Apply flows
            results = self.apply_all_flows(passive_flows, active_flows, delta_time)
            
            # Update edge usage tracking
            self._update_edge_usage(passive_flows + active_flows)
            
            return results
    
    def _calculate_total_energy(self) -> float:
        """Calculate total energy in the system"""
        return sum(node.energy for node in self.mycelial_layer.nodes.values())
    
    def _calculate_average_flow_rate(self, flows: List[EnergyFlow]) -> float:
        """Calculate average flow rate"""
        if not flows:
            return 0.0
        return sum(flow.flow_rate for flow in flows) / len(flows)
    
    def _update_edge_usage(self, flows: List[EnergyFlow]):
        """Update edge usage statistics based on flows"""
        edge_usage = defaultdict(int)
        
        for flow in flows:
            # Find corresponding edge
            edge_id = f"{flow.source_id}->{flow.target_id}"
            reverse_edge_id = f"{flow.target_id}->{flow.source_id}"
            
            if edge_id in self.mycelial_layer.edges:
                edge_usage[edge_id] += 1
                self.mycelial_layer.edges[edge_id].record_usage()
            elif reverse_edge_id in self.mycelial_layer.edges:
                edge_usage[reverse_edge_id] += 1
                self.mycelial_layer.edges[reverse_edge_id].record_usage()
    
    def get_flow_statistics(self) -> Dict[str, Any]:
        """Get comprehensive flow statistics"""
        return {
            'current_stats': self.stats.copy(),
            'passive_diffusion': {
                'flow_count': len(self.passive_diffusion.active_flows),
                'diffusion_rate': self.passive_diffusion.diffusion_rate,
                'viscosity': self.passive_diffusion.viscosity,
                'recent_flows': len(self.passive_diffusion.flow_history)
            },
            'active_transport': {
                'transport_strength': self.active_transport.transport_strength,
                'bloom_threshold': self.active_transport.bloom_threshold,
                'starve_threshold': self.active_transport.starve_threshold,
                'recent_transports': len(self.active_transport.transport_history)
            },
            'system': {
                'total_energy': self._calculate_total_energy(),
                'node_count': len(self.mycelial_layer.nodes),
                'edge_count': len(self.mycelial_layer.edges),
                'flow_efficiency': self.flow_efficiency
            }
        }
    
    def get_node_flow_status(self, node_id: str) -> Optional[Dict[str, Any]]:
        """Get flow status for a specific node"""
        if node_id not in self.mycelial_layer.nodes:
            return None
        
        node = self.mycelial_layer.nodes[node_id]
        
        # Count flows involving this node
        incoming_flows = 0
        outgoing_flows = 0
        
        for flow in (list(self.passive_diffusion.flow_history) + 
                    list(self.active_transport.transport_history))[-50:]:  # Recent flows
            if flow['target_id'] == node_id:
                incoming_flows += 1
            elif flow['source_id'] == node_id:
                outgoing_flows += 1
        
        return {
            'node_id': node_id,
            'current_energy': node.energy,
            'max_energy': node.max_energy,
            'energy_ratio': node.energy / node.max_energy,
            'bloom_factor': self.active_transport._compute_bloom_factor(node),
            'starve_factor': self.active_transport._compute_starve_factor(node),
            'connection_count': len(node.connections),
            'recent_incoming_flows': incoming_flows,
            'recent_outgoing_flows': outgoing_flows,
            'net_flow_balance': incoming_flows - outgoing_flows
        }
