"""
🧬 Metabolite Management System
==============================

Implements the metabolite production, distribution, and absorption mechanisms.
When nodes undergo autophagy, they produce metabolites - semantic traces containing
part of their old meaning that neighboring nodes can absorb for recovery.

Key Features:
- Metabolite production from decomposing nodes
- Semantic trace preservation and encoding
- Neighbor-based distribution algorithms
- Absorption efficiency based on compatibility
- Metabolite decay and recycling
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
import json

logger = logging.getLogger(__name__)

class MetaboliteType(Enum):
    """Types of metabolites produced during autophagy"""
    SEMANTIC_TRACE = "semantic_trace"      # General semantic information
    ENERGY_RESIDUE = "energy_residue"      # Pure energy content
    CONNECTION_MAP = "connection_map"       # Network connection information
    PRESSURE_SIGNATURE = "pressure_signature"  # Pressure/mood information
    FINAL_ESSENCE = "final_essence"        # Complete node essence at death

class MetaboliteState(Enum):
    """Lifecycle states of metabolites"""
    FRESH = "fresh"          # Just produced, high potency
    ACTIVE = "active"        # Available for absorption
    DEGRADED = "degraded"    # Partially decayed, reduced potency
    INERT = "inert"         # Nearly inactive, will be recycled soon
    ABSORBED = "absorbed"    # Successfully absorbed by a node

@dataclass
class MetaboliteTrace:
    """
    A metabolite - semantic trace containing part of a decomposed node's meaning.
    Neighboring nodes can absorb these to bias themselves toward recovering
    or recombining the lost pattern.
    """
    
    # Identity and origin
    id: str
    source_node_id: str
    metabolite_type: MetaboliteType
    created_at: float = field(default_factory=time.time)
    
    # Content and potency
    energy_content: float = 0.0
    semantic_content: Dict[str, Any] = field(default_factory=dict)
    potency: float = 1.0          # Effectiveness multiplier [0,1]
    
    # Lifecycle management
    state: MetaboliteState = MetaboliteState.FRESH
    decay_rate: float = 0.05      # Rate of potency decay per tick
    max_lifetime: float = 100.0   # Maximum ticks before forced recycling
    
    # Distribution tracking
    eligible_recipients: Set[str] = field(default_factory=set)
    distribution_attempts: int = 0
    absorption_attempts: int = 0
    
    # Compatibility factors
    pressure_signature: float = 0.0
    drift_signature: float = 0.0
    entropy_signature: float = 0.0
    
    # Spatial information
    origin_position: torch.Tensor = field(default_factory=lambda: torch.zeros(3))
    distribution_radius: float = 2.0
    
    def __post_init__(self):
        """Initialize computed properties"""
        if isinstance(self.origin_position, (list, tuple)):
            self.origin_position = torch.tensor(self.origin_position, dtype=torch.float32)
        
        # Extract signatures from semantic content if available
        if 'pressure' in self.semantic_content:
            self.pressure_signature = self.semantic_content['pressure']
        if 'drift_alignment' in self.semantic_content:
            self.drift_signature = self.semantic_content['drift_alignment']
        if 'entropy' in self.semantic_content:
            self.entropy_signature = self.semantic_content['entropy']
    
    def decay(self, delta_time: float = 1.0) -> bool:
        """
        Apply decay to the metabolite. Returns True if still viable.
        """
        # Apply decay based on state
        if self.state == MetaboliteState.FRESH:
            # Fresh metabolites decay slowly
            self.potency -= self.decay_rate * 0.5 * delta_time
            if self.potency < 0.8:
                self.state = MetaboliteState.ACTIVE
        
        elif self.state == MetaboliteState.ACTIVE:
            # Active metabolites decay normally
            self.potency -= self.decay_rate * delta_time
            if self.potency < 0.3:
                self.state = MetaboliteState.DEGRADED
        
        elif self.state == MetaboliteState.DEGRADED:
            # Degraded metabolites decay faster
            self.potency -= self.decay_rate * 2.0 * delta_time
            if self.potency < 0.1:
                self.state = MetaboliteState.INERT
        
        # Check lifetime
        age = time.time() - self.created_at
        if age > self.max_lifetime:
            self.state = MetaboliteState.INERT
        
        return self.state != MetaboliteState.INERT and self.potency > 0.0
    
    def compute_compatibility(self, target_node) -> float:
        """
        Compute how compatible this metabolite is with a target node.
        Higher compatibility = better absorption efficiency.
        """
        compatibility = 0.0
        
        # Pressure compatibility
        pressure_diff = abs(self.pressure_signature - target_node.pressure)
        pressure_compat = max(0.0, 1.0 - pressure_diff)
        compatibility += 0.3 * pressure_compat
        
        # Drift compatibility  
        drift_diff = abs(self.drift_signature - target_node.drift_alignment)
        drift_compat = max(0.0, 1.0 - drift_diff)
        compatibility += 0.3 * drift_compat
        
        # Energy compatibility (can the node use the energy?)
        energy_need = max(0.0, target_node.max_energy - target_node.energy)
        energy_compat = min(1.0, self.energy_content / max(0.1, energy_need))
        compatibility += 0.2 * energy_compat
        
        # Spatial compatibility
        if hasattr(target_node, 'position'):
            spatial_distance = torch.norm(self.origin_position - target_node.position).item()
            spatial_compat = max(0.0, 1.0 - spatial_distance / self.distribution_radius)
            compatibility += 0.2 * spatial_compat
        
        return min(1.0, compatibility)
    
    def create_absorption_effect(self, compatibility: float) -> Dict[str, Any]:
        """
        Create the effect that will be applied when this metabolite is absorbed.
        """
        base_strength = self.potency * compatibility
        
        effect = {
            'energy_gain': self.energy_content * base_strength * 0.6,  # 60% efficiency
            'bias_effects': {},
            'semantic_integration': {},
            'metabolite_id': self.id,
            'source_node_id': self.source_node_id,
            'compatibility': compatibility,
            'potency': self.potency
        }
        
        # Bias effects based on metabolite type
        if self.metabolite_type == MetaboliteType.SEMANTIC_TRACE:
            effect['bias_effects'] = {
                'pressure_bias': (self.pressure_signature - 0) * base_strength * 0.1,
                'drift_bias': (self.drift_signature - 0) * base_strength * 0.1,
                'entropy_bias': (self.entropy_signature - 0) * base_strength * 0.05
            }
        
        elif self.metabolite_type == MetaboliteType.PRESSURE_SIGNATURE:
            effect['bias_effects'] = {
                'pressure_bias': (self.pressure_signature - 0) * base_strength * 0.2,
                'mood_stabilization': base_strength * 0.1
            }
        
        elif self.metabolite_type == MetaboliteType.CONNECTION_MAP:
            effect['semantic_integration'] = {
                'connection_memories': self.semantic_content.get('connections', []),
                'network_bias_strength': base_strength * 0.15
            }
        
        elif self.metabolite_type == MetaboliteType.FINAL_ESSENCE:
            # Final essence provides comprehensive benefits
            effect['bias_effects'] = {
                'pressure_bias': (self.pressure_signature - 0) * base_strength * 0.15,
                'drift_bias': (self.drift_signature - 0) * base_strength * 0.15,
                'vitality_boost': base_strength * 0.2
            }
            effect['semantic_integration'] = {
                'essence_memories': self.semantic_content,
                'deep_integration_strength': base_strength * 0.3
            }
        
        return effect

class MetaboliteManager:
    """
    Manages the complete lifecycle of metabolites in the mycelial layer.
    Handles production, distribution, absorption, and recycling.
    """
    
    def __init__(self, mycelial_layer):
        self.mycelial_layer = mycelial_layer
        
        # Metabolite storage
        self.active_metabolites: Dict[str, MetaboliteTrace] = {}
        self.metabolite_counter = 0
        
        # Distribution system
        self.distribution_queue: deque = deque()
        self.absorption_queue: deque = deque()
        
        # Configuration
        self.max_metabolites = 1000
        self.distribution_batch_size = 50
        self.absorption_efficiency_base = 0.7
        
        # Tracking and statistics
        self.total_produced = 0
        self.total_absorbed = 0
        self.total_recycled = 0
        self.energy_conserved = 0.0
        
        # History for analysis
        self.production_history: deque = deque(maxlen=1000)
        self.absorption_history: deque = deque(maxlen=1000)
        self.recycling_history: deque = deque(maxlen=500)
        
        # Thread safety
        self._lock = threading.RLock()
        
        logger.info("MetaboliteManager initialized")
    
    def produce_metabolite(self, source_node, metabolite_type: MetaboliteType, 
                          energy_content: float, semantic_content: Dict[str, Any]) -> MetaboliteTrace:
        """
        Produce a new metabolite from a decomposing node.
        """
        with self._lock:
            self.metabolite_counter += 1
            metabolite_id = f"metabolite_{self.metabolite_counter}_{time.time():.3f}"
            
            metabolite = MetaboliteTrace(
                id=metabolite_id,
                source_node_id=source_node.id,
                metabolite_type=metabolite_type,
                energy_content=energy_content,
                semantic_content=semantic_content.copy(),
                origin_position=source_node.position.clone() if hasattr(source_node, 'position') else torch.zeros(3)
            )
            
            # Find eligible recipients (neighbors within radius)
            metabolite.eligible_recipients = self._find_eligible_recipients(source_node)
            
            # Store the metabolite
            self.active_metabolites[metabolite_id] = metabolite
            self.total_produced += 1
            
            # Queue for distribution
            self.distribution_queue.append(metabolite_id)
            
            # Record production
            self.production_history.append({
                'timestamp': time.time(),
                'metabolite_id': metabolite_id,
                'source_node_id': source_node.id,
                'metabolite_type': metabolite_type.value,
                'energy_content': energy_content,
                'eligible_recipients': len(metabolite.eligible_recipients)
            })
            
            logger.debug(f"Produced metabolite {metabolite_id} from node {source_node.id}")
            return metabolite
    
    def _find_eligible_recipients(self, source_node) -> Set[str]:
        """Find nodes that can potentially absorb metabolites from the source"""
        recipients = set()
        
        # Direct neighbors
        recipients.update(source_node.connections.keys())
        
        # Neighbors of neighbors (2-hop)
        for neighbor_id in source_node.connections.keys():
            neighbor = self.mycelial_layer.nodes.get(neighbor_id)
            if neighbor:
                recipients.update(neighbor.connections.keys())
        
        # Remove the source itself
        recipients.discard(source_node.id)
        
        return recipients
    
    def distribute_metabolites(self, max_distributions: int = None) -> int:
        """
        Distribute queued metabolites to eligible recipients.
        """
        if max_distributions is None:
            max_distributions = self.distribution_batch_size
        
        distributions = 0
        
        with self._lock:
            while self.distribution_queue and distributions < max_distributions:
                metabolite_id = self.distribution_queue.popleft()
                metabolite = self.active_metabolites.get(metabolite_id)
                
                if not metabolite or metabolite.state == MetaboliteState.INERT:
                    continue
                
                # Attempt distribution to eligible recipients
                distributed = self._distribute_single_metabolite(metabolite)
                if distributed:
                    distributions += 1
                
                metabolite.distribution_attempts += 1
        
        return distributions
    
    def _distribute_single_metabolite(self, metabolite: MetaboliteTrace) -> bool:
        """Distribute a single metabolite to its best recipient"""
        best_recipient = None
        best_compatibility = 0.0
        
        # Evaluate all eligible recipients
        for recipient_id in metabolite.eligible_recipients:
            recipient_node = self.mycelial_layer.nodes.get(recipient_id)
            if not recipient_node:
                continue
            
            # Skip if recipient is at max energy
            if recipient_node.energy >= recipient_node.max_energy:
                continue
            
            # Compute compatibility
            compatibility = metabolite.compute_compatibility(recipient_node)
            
            if compatibility > best_compatibility:
                best_compatibility = compatibility
                best_recipient = recipient_id
        
        # Queue for absorption if good recipient found
        if best_recipient and best_compatibility > 0.3:  # Minimum compatibility threshold
            absorption_task = {
                'metabolite_id': metabolite.id,
                'recipient_id': best_recipient,
                'compatibility': best_compatibility,
                'timestamp': time.time()
            }
            self.absorption_queue.append(absorption_task)
            return True
        
        return False
    
    def process_absorptions(self, max_absorptions: int = None) -> int:
        """
        Process queued metabolite absorptions.
        """
        if max_absorptions is None:
            max_absorptions = self.distribution_batch_size
        
        absorptions = 0
        
        with self._lock:
            while self.absorption_queue and absorptions < max_absorptions:
                task = self.absorption_queue.popleft()
                
                metabolite = self.active_metabolites.get(task['metabolite_id'])
                recipient = self.mycelial_layer.nodes.get(task['recipient_id'])
                
                if not metabolite or not recipient:
                    continue
                
                # Attempt absorption
                absorbed = self._absorb_metabolite(metabolite, recipient, task['compatibility'])
                if absorbed:
                    absorptions += 1
                    self.total_absorbed += 1
                
                metabolite.absorption_attempts += 1
        
        return absorptions
    
    def _absorb_metabolite(self, metabolite: MetaboliteTrace, recipient_node, compatibility: float) -> bool:
        """Absorb a metabolite into a recipient node"""
        
        # Check if recipient can still benefit
        if recipient_node.energy >= recipient_node.max_energy:
            return False
        
        # Create absorption effect
        effect = metabolite.create_absorption_effect(compatibility)
        
        # Apply energy gain
        energy_gain = effect['energy_gain']
        recipient_node.energy = min(recipient_node.max_energy, 
                                   recipient_node.energy + energy_gain)
        
        # Apply bias effects
        bias_strength = 0.1  # How much metabolites can bias node properties
        for bias_type, bias_value in effect.get('bias_effects', {}).items():
            if bias_type == 'pressure_bias':
                recipient_node.pressure += bias_value * bias_strength
            elif bias_type == 'drift_bias':
                recipient_node.drift_alignment += bias_value * bias_strength
            elif bias_type == 'entropy_bias':
                recipient_node.entropy = max(0.0, recipient_node.entropy + bias_value * bias_strength)
            elif bias_type == 'vitality_boost':
                if hasattr(recipient_node, 'health'):
                    recipient_node.health = min(1.0, recipient_node.health + bias_value)
        
        # Apply semantic integration (placeholder for future semantic systems)
        semantic_integration = effect.get('semantic_integration', {})
        if semantic_integration:
            # Could integrate with semantic memory systems
            pass
        
        # Mark metabolite as absorbed
        metabolite.state = MetaboliteState.ABSORBED
        
        # Record absorption
        self.absorption_history.append({
            'timestamp': time.time(),
            'metabolite_id': metabolite.id,
            'recipient_id': recipient_node.id,
            'energy_transferred': energy_gain,
            'compatibility': compatibility,
            'metabolite_type': metabolite.metabolite_type.value,
            'effect_summary': effect
        })
        
        logger.debug(f"Node {recipient_node.id} absorbed metabolite {metabolite.id}")
        return True
    
    def tick_update(self, delta_time: float = 1.0) -> Dict[str, Any]:
        """
        Perform one tick of metabolite management:
        1. Decay existing metabolites
        2. Distribute queued metabolites  
        3. Process absorptions
        4. Recycle inert metabolites
        """
        with self._lock:
            # 1. Decay and clean up metabolites
            decayed_count = self._decay_metabolites(delta_time)
            
            # 2. Distribute metabolites
            distributed_count = self.distribute_metabolites()
            
            # 3. Process absorptions
            absorbed_count = self.process_absorptions()
            
            # 4. Recycle inert metabolites
            recycled_count = self._recycle_inert_metabolites()
            
            # 5. Limit total metabolites
            if len(self.active_metabolites) > self.max_metabolites:
                pruned_count = self._prune_excess_metabolites()
            else:
                pruned_count = 0
            
            return {
                'active_metabolites': len(self.active_metabolites),
                'decayed': decayed_count,
                'distributed': distributed_count,
                'absorbed': absorbed_count,
                'recycled': recycled_count,
                'pruned': pruned_count,
                'queued_distributions': len(self.distribution_queue),
                'queued_absorptions': len(self.absorption_queue)
            }
    
    def _decay_metabolites(self, delta_time: float) -> int:
        """Apply decay to all active metabolites"""
        decayed_count = 0
        to_remove = []
        
        for metabolite_id, metabolite in self.active_metabolites.items():
            if not metabolite.decay(delta_time):
                to_remove.append(metabolite_id)
                decayed_count += 1
        
        # Remove fully decayed metabolites
        for metabolite_id in to_remove:
            del self.active_metabolites[metabolite_id]
        
        return decayed_count
    
    def _recycle_inert_metabolites(self) -> int:
        """Recycle inert metabolites back into global energy budget"""
        recycled_count = 0
        to_remove = []
        
        for metabolite_id, metabolite in self.active_metabolites.items():
            if metabolite.state == MetaboliteState.INERT:
                # Recycle energy back to global budget (if nutrient economy exists)
                if hasattr(self.mycelial_layer, 'nutrient_economy'):
                    recycled_energy = metabolite.energy_content * 0.5  # 50% recovery
                    self.mycelial_layer.nutrient_economy.replenish_budget(recycled_energy)
                    self.energy_conserved += recycled_energy
                
                to_remove.append(metabolite_id)
                recycled_count += 1
                
                # Record recycling
                self.recycling_history.append({
                    'timestamp': time.time(),
                    'metabolite_id': metabolite_id,
                    'energy_recycled': metabolite.energy_content * 0.5,
                    'age': time.time() - metabolite.created_at,
                    'final_potency': metabolite.potency
                })
        
        # Remove recycled metabolites
        for metabolite_id in to_remove:
            del self.active_metabolites[metabolite_id]
        
        self.total_recycled += recycled_count
        return recycled_count
    
    def _prune_excess_metabolites(self) -> int:
        """Remove excess metabolites when over limit"""
        excess = len(self.active_metabolites) - self.max_metabolites
        if excess <= 0:
            return 0
        
        # Sort by potency (remove weakest first)
        sorted_metabolites = sorted(
            self.active_metabolites.items(),
            key=lambda x: x[1].potency
        )
        
        pruned_count = 0
        for i in range(min(excess, len(sorted_metabolites))):
            metabolite_id = sorted_metabolites[i][0]
            del self.active_metabolites[metabolite_id]
            pruned_count += 1
        
        return pruned_count
    
    def get_metabolite_statistics(self) -> Dict[str, Any]:
        """Get comprehensive metabolite system statistics"""
        # Analyze current metabolites by type and state
        type_counts = defaultdict(int)
        state_counts = defaultdict(int)
        total_energy = 0.0
        avg_potency = 0.0
        
        for metabolite in self.active_metabolites.values():
            type_counts[metabolite.metabolite_type.value] += 1
            state_counts[metabolite.state.value] += 1
            total_energy += metabolite.energy_content
            avg_potency += metabolite.potency
        
        if self.active_metabolites:
            avg_potency /= len(self.active_metabolites)
        
        return {
            'totals': {
                'active_metabolites': len(self.active_metabolites),
                'total_produced': self.total_produced,
                'total_absorbed': self.total_absorbed,
                'total_recycled': self.total_recycled,
                'energy_conserved': self.energy_conserved
            },
            'current_distribution': {
                'by_type': dict(type_counts),
                'by_state': dict(state_counts),
                'total_energy': total_energy,
                'average_potency': avg_potency
            },
            'queues': {
                'pending_distributions': len(self.distribution_queue),
                'pending_absorptions': len(self.absorption_queue)
            },
            'efficiency': {
                'absorption_rate': self.total_absorbed / max(1, self.total_produced),
                'recycling_rate': self.total_recycled / max(1, self.total_produced),
                'energy_conservation': self.energy_conserved
            },
            'config': {
                'max_metabolites': self.max_metabolites,
                'distribution_batch_size': self.distribution_batch_size,
                'absorption_efficiency_base': self.absorption_efficiency_base
            }
        }
    
    def get_node_metabolite_status(self, node_id: str) -> Dict[str, Any]:
        """Get metabolite-related status for a specific node"""
        # Count metabolites involving this node
        produced_by_node = 0
        absorbed_by_node = 0
        pending_for_node = 0
        
        for metabolite in self.active_metabolites.values():
            if metabolite.source_node_id == node_id:
                produced_by_node += 1
            if node_id in metabolite.eligible_recipients:
                pending_for_node += 1
        
        # Count recent absorptions
        recent_absorptions = sum(1 for h in self.absorption_history 
                               if h['recipient_id'] == node_id and 
                               time.time() - h['timestamp'] < 60.0)  # Last 60 seconds
        
        return {
            'node_id': node_id,
            'metabolites_produced': produced_by_node,
            'recent_absorptions': recent_absorptions,
            'pending_metabolites': pending_for_node,
            'absorption_capacity': self._compute_absorption_capacity(node_id),
            'compatibility_score': self._compute_avg_compatibility(node_id)
        }
    
    def _compute_absorption_capacity(self, node_id: str) -> float:
        """Compute how much absorption capacity a node has"""
        node = self.mycelial_layer.nodes.get(node_id)
        if not node:
            return 0.0
        
        energy_capacity = node.max_energy - node.energy
        return energy_capacity / node.max_energy
    
    def _compute_avg_compatibility(self, node_id: str) -> float:
        """Compute average compatibility of pending metabolites with a node"""
        node = self.mycelial_layer.nodes.get(node_id)
        if not node:
            return 0.0
        
        compatibilities = []
        for metabolite in self.active_metabolites.values():
            if node_id in metabolite.eligible_recipients:
                compatibility = metabolite.compute_compatibility(node)
                compatibilities.append(compatibility)
        
        return sum(compatibilities) / max(1, len(compatibilities))
