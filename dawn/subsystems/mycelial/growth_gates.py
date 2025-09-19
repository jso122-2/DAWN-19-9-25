"""
🌱 Growth Gates and Autophagy Management
=======================================

Implements the biological-inspired growth and pruning mechanisms:
1. Growth Gates: Selective connection formation based on similarity, energy, and compatibility
2. Autophagy: Self-digestion of starved nodes to create metabolites for neighbors

Based on DAWN specifications:
- Growth Gate: energy_i > θ_grow AND similarity > θ_sim AND temporal_proximity_ok AND mood_compatible
- Autophagy: Trigger if energy_i < θ_prune for τ ticks
"""

import torch
import torch.nn.functional as F
import numpy as np
import time
import math
from typing import Dict, List, Optional, Tuple, Any, Set, Callable
from dataclasses import dataclass, field
from enum import Enum
from collections import defaultdict, deque
import threading
import logging

logger = logging.getLogger(__name__)

class GrowthDecision(Enum):
    """Possible outcomes of growth gate evaluation"""
    APPROVED = "approved"
    REJECTED_ENERGY = "rejected_energy"
    REJECTED_SIMILARITY = "rejected_similarity"
    REJECTED_TEMPORAL = "rejected_temporal"
    REJECTED_MOOD = "rejected_mood"
    REJECTED_CAPACITY = "rejected_capacity"
    REJECTED_EXISTING = "rejected_existing"

class AutophagyTrigger(Enum):
    """Reasons for autophagy activation"""
    ENERGY_STARVATION = "energy_starvation"
    MANUAL_TRIGGER = "manual_trigger"
    ISOLATION = "isolation"
    ENTROPY_OVERLOAD = "entropy_overload"

@dataclass
class GrowthGateConfig:
    """Configuration for growth gate thresholds and parameters"""
    
    # Energy requirements
    energy_threshold: float = 0.6        # θ_grow - minimum energy to grow
    target_energy_threshold: float = 0.3  # Target must have some energy
    
    # Similarity requirements  
    similarity_threshold: float = 0.5     # θ_sim - minimum similarity
    similarity_method: str = "cosine"     # Method for computing similarity
    
    # Temporal proximity
    temporal_window: float = 10.0         # Seconds for temporal proximity
    max_age_difference: float = 100.0     # Maximum age difference
    
    # Mood/pressure compatibility
    pressure_tolerance: float = 0.5       # Maximum pressure difference
    drift_tolerance: float = 0.3          # Maximum drift difference
    
    # Capacity limits
    max_connections_per_node: int = 50    # Connection limit per node
    max_connections_total: int = 10000    # Global connection limit
    
    # Advanced parameters
    redundancy_penalty: float = 0.8       # Penalty for redundant connections
    diversity_bonus: float = 1.2          # Bonus for diverse connections

@dataclass  
class AutophagyConfig:
    """Configuration for autophagy mechanisms"""
    
    # Energy-based autophagy
    energy_threshold: float = 0.1         # θ_prune - energy threshold
    starvation_time: int = 5              # τ - ticks before autophagy
    
    # Alternative triggers
    isolation_threshold: int = 1          # Minimum connections to avoid isolation
    entropy_threshold: float = 0.9        # Maximum entropy before autophagy
    
    # Metabolite production
    energy_recovery_rate: float = 0.8     # How much energy becomes metabolites
    metabolite_distribution_radius: int = 2  # Hops for metabolite distribution
    
    # Process parameters
    autophagy_duration: int = 3           # Ticks for autophagy process
    revival_possibility: float = 0.1      # Chance of revival during autophagy

class SimilarityComputer:
    """Computes various types of similarity between nodes"""
    
    def __init__(self):
        self.methods = {
            'cosine': self._cosine_similarity,
            'euclidean': self._euclidean_similarity,
            'pressure': self._pressure_similarity,
            'semantic': self._semantic_similarity,
            'hybrid': self._hybrid_similarity
        }
    
    def compute_similarity(self, node1, node2, method: str = "hybrid") -> float:
        """Compute similarity between two nodes using specified method"""
        if method not in self.methods:
            method = "hybrid"
        
        return self.methods[method](node1, node2)
    
    def _cosine_similarity(self, node1, node2) -> float:
        """Cosine similarity based on position vectors"""
        pos1 = node1.position.flatten()
        pos2 = node2.position.flatten()
        
        dot_product = torch.dot(pos1, pos2)
        norm1 = torch.norm(pos1)
        norm2 = torch.norm(pos2)
        
        if norm1 == 0 or norm2 == 0:
            return 0.0
        
        return (dot_product / (norm1 * norm2)).item()
    
    def _euclidean_similarity(self, node1, node2) -> float:
        """Euclidean distance-based similarity"""
        distance = torch.norm(node1.position - node2.position).item()
        # Convert distance to similarity (0-1 range)
        return 1.0 / (1.0 + distance)
    
    def _pressure_similarity(self, node1, node2) -> float:
        """Similarity based on pressure and state alignment"""
        pressure_sim = 1.0 - abs(node1.pressure - node2.pressure)
        drift_sim = 1.0 - abs(node1.drift_alignment - node2.drift_alignment)
        entropy_sim = 1.0 - abs(node1.entropy - node2.entropy)
        
        return (pressure_sim + drift_sim + entropy_sim) / 3.0
    
    def _semantic_similarity(self, node1, node2) -> float:
        """Semantic similarity (placeholder for future semantic content)"""
        # For now, use a combination of recency and connection overlap
        recency_sim = 1.0 - abs(node1.recency - node2.recency)
        
        # Connection overlap
        connections1 = set(node1.connections.keys())
        connections2 = set(node2.connections.keys())
        
        if not connections1 and not connections2:
            overlap = 1.0
        elif not connections1 or not connections2:
            overlap = 0.0
        else:
            intersection = len(connections1.intersection(connections2))
            union = len(connections1.union(connections2))
            overlap = intersection / union if union > 0 else 0.0
        
        return (recency_sim + overlap) / 2.0
    
    def _hybrid_similarity(self, node1, node2) -> float:
        """Hybrid similarity combining multiple methods"""
        cosine_sim = self._cosine_similarity(node1, node2)
        pressure_sim = self._pressure_similarity(node1, node2)
        semantic_sim = self._semantic_similarity(node1, node2)
        
        # Weighted combination
        return (0.4 * cosine_sim + 0.4 * pressure_sim + 0.2 * semantic_sim)

class GrowthGate:
    """
    Implements selective connection formation with biological-inspired constraints.
    
    New connections don't form blindly - enforces thresholds of similarity,
    temporal proximity, and mood/pressure compatibility.
    """
    
    def __init__(self, config: Optional[GrowthGateConfig] = None):
        self.config = config or GrowthGateConfig()
        self.similarity_computer = SimilarityComputer()
        
        # Decision tracking
        self.growth_decisions: deque = deque(maxlen=1000)
        self.growth_attempts: int = 0
        self.growth_approvals: int = 0
        
        # Performance monitoring
        self.decision_times: deque = deque(maxlen=100)
        self.similarity_cache: Dict[Tuple[str, str], Tuple[float, float]] = {}  # (sim, timestamp)
        
        # Statistics
        self.rejection_reasons: Dict[GrowthDecision, int] = defaultdict(int)
        
        logger.info("GrowthGate initialized")
    
    def evaluate_growth_proposal(self, source_node, target_node, mycelial_layer) -> Tuple[GrowthDecision, Dict[str, Any]]:
        """
        Evaluate whether a connection should be formed between two nodes.
        
        Implements the growth gate logic:
        energy_i > θ_grow AND similarity > θ_sim AND temporal_proximity_ok AND mood_compatible
        """
        start_time = time.time()
        self.growth_attempts += 1
        
        evaluation = {
            'source_id': source_node.id,
            'target_id': target_node.id,
            'timestamp': start_time,
            'checks': {}
        }
        
        # Check 1: Energy requirement
        if source_node.energy < self.config.energy_threshold:
            decision = GrowthDecision.REJECTED_ENERGY
            evaluation['checks']['energy'] = {
                'passed': False,
                'source_energy': source_node.energy,
                'threshold': self.config.energy_threshold
            }
            self._record_decision(decision, evaluation)
            return decision, evaluation
        
        if target_node.energy < self.config.target_energy_threshold:
            decision = GrowthDecision.REJECTED_ENERGY
            evaluation['checks']['target_energy'] = {
                'passed': False,
                'target_energy': target_node.energy,
                'threshold': self.config.target_energy_threshold
            }
            self._record_decision(decision, evaluation)
            return decision, evaluation
        
        evaluation['checks']['energy'] = {'passed': True}
        
        # Check 2: Existing connection
        if target_node.id in source_node.connections:
            decision = GrowthDecision.REJECTED_EXISTING
            evaluation['checks']['existing'] = {'passed': False}
            self._record_decision(decision, evaluation)
            return decision, evaluation
        
        evaluation['checks']['existing'] = {'passed': True}
        
        # Check 3: Capacity limits
        if len(source_node.connections) >= self.config.max_connections_per_node:
            decision = GrowthDecision.REJECTED_CAPACITY
            evaluation['checks']['capacity'] = {
                'passed': False,
                'current_connections': len(source_node.connections),
                'max_connections': self.config.max_connections_per_node
            }
            self._record_decision(decision, evaluation)
            return decision, evaluation
        
        if len(mycelial_layer.edges) >= self.config.max_connections_total:
            decision = GrowthDecision.REJECTED_CAPACITY
            evaluation['checks']['global_capacity'] = {
                'passed': False,
                'current_edges': len(mycelial_layer.edges),
                'max_edges': self.config.max_connections_total
            }
            self._record_decision(decision, evaluation)
            return decision, evaluation
        
        evaluation['checks']['capacity'] = {'passed': True}
        
        # Check 4: Similarity requirement
        similarity = self._get_cached_similarity(source_node, target_node)
        if similarity < self.config.similarity_threshold:
            decision = GrowthDecision.REJECTED_SIMILARITY
            evaluation['checks']['similarity'] = {
                'passed': False,
                'similarity': similarity,
                'threshold': self.config.similarity_threshold
            }
            self._record_decision(decision, evaluation)
            return decision, evaluation
        
        evaluation['checks']['similarity'] = {
            'passed': True,
            'similarity': similarity
        }
        
        # Check 5: Temporal proximity
        temporal_ok = self._check_temporal_proximity(source_node, target_node)
        if not temporal_ok:
            decision = GrowthDecision.REJECTED_TEMPORAL
            evaluation['checks']['temporal'] = {'passed': False}
            self._record_decision(decision, evaluation)
            return decision, evaluation
        
        evaluation['checks']['temporal'] = {'passed': True}
        
        # Check 6: Mood/pressure compatibility
        mood_compatible = self._check_mood_compatibility(source_node, target_node)
        if not mood_compatible:
            decision = GrowthDecision.REJECTED_MOOD
            evaluation['checks']['mood'] = {'passed': False}
            self._record_decision(decision, evaluation)
            return decision, evaluation
        
        evaluation['checks']['mood'] = {'passed': True}
        
        # All checks passed!
        decision = GrowthDecision.APPROVED
        self.growth_approvals += 1
        
        evaluation['approved'] = True
        self._record_decision(decision, evaluation)
        
        # Record timing
        self.decision_times.append(time.time() - start_time)
        
        return decision, evaluation
    
    def _get_cached_similarity(self, node1, node2) -> float:
        """Get similarity with caching for performance"""
        cache_key = (node1.id, node2.id) if node1.id < node2.id else (node2.id, node1.id)
        current_time = time.time()
        
        # Check cache
        if cache_key in self.similarity_cache:
            similarity, timestamp = self.similarity_cache[cache_key]
            if current_time - timestamp < 5.0:  # Cache for 5 seconds
                return similarity
        
        # Compute new similarity
        similarity = self.similarity_computer.compute_similarity(
            node1, node2, self.config.similarity_method
        )
        
        # Cache result
        self.similarity_cache[cache_key] = (similarity, current_time)
        
        # Clean old cache entries periodically
        if len(self.similarity_cache) > 1000:
            self._clean_similarity_cache(current_time)
        
        return similarity
    
    def _clean_similarity_cache(self, current_time: float):
        """Remove old entries from similarity cache"""
        to_remove = []
        for key, (similarity, timestamp) in self.similarity_cache.items():
            if current_time - timestamp > 10.0:  # Remove entries older than 10 seconds
                to_remove.append(key)
        
        for key in to_remove:
            del self.similarity_cache[key]
    
    def _check_temporal_proximity(self, node1, node2) -> bool:
        """Check if nodes are temporally compatible"""
        # Time since last update
        time_diff = abs(node1.last_update - node2.last_update)
        if time_diff > self.config.temporal_window:
            return False
        
        # Age difference
        age_diff = abs(node1.created_at - node2.created_at)
        if age_diff > self.config.max_age_difference:
            return False
        
        return True
    
    def _check_mood_compatibility(self, node1, node2) -> bool:
        """Check if nodes have compatible mood/pressure states"""
        pressure_diff = abs(node1.pressure - node2.pressure)
        if pressure_diff > self.config.pressure_tolerance:
            return False
        
        drift_diff = abs(node1.drift_alignment - node2.drift_alignment)
        if drift_diff > self.config.drift_tolerance:
            return False
        
        return True
    
    def _record_decision(self, decision: GrowthDecision, evaluation: Dict[str, Any]):
        """Record a growth decision for analysis"""
        self.rejection_reasons[decision] += 1
        
        evaluation['decision'] = decision.value
        self.growth_decisions.append(evaluation)
    
    def get_growth_statistics(self) -> Dict[str, Any]:
        """Get statistics about growth gate decisions"""
        approval_rate = self.growth_approvals / max(1, self.growth_attempts)
        
        avg_decision_time = (sum(self.decision_times) / max(1, len(self.decision_times))) if self.decision_times else 0.0
        
        return {
            'total_attempts': self.growth_attempts,
            'total_approvals': self.growth_approvals,
            'approval_rate': approval_rate,
            'rejection_reasons': dict(self.rejection_reasons),
            'average_decision_time': avg_decision_time,
            'similarity_cache_size': len(self.similarity_cache),
            'config': {
                'energy_threshold': self.config.energy_threshold,
                'similarity_threshold': self.config.similarity_threshold,
                'temporal_window': self.config.temporal_window,
                'pressure_tolerance': self.config.pressure_tolerance
            }
        }

class AutophagyManager:
    """
    Manages autophagy (self-digestion) of starved or isolated nodes.
    
    Converts failing nodes into metabolites for neighboring nodes to absorb.
    """
    
    def __init__(self, config: Optional[AutophagyConfig] = None):
        self.config = config or AutophagyConfig()
        
        # Autophagy tracking
        self.nodes_in_autophagy: Dict[str, Dict[str, Any]] = {}
        self.autophagy_history: deque = deque(maxlen=1000)
        self.metabolites_produced: List[Dict[str, Any]] = []
        
        # Statistics
        self.total_autophagies: int = 0
        self.metabolites_created: int = 0
        self.energy_recycled: float = 0.0
        
        logger.info("AutophagyManager initialized")
    
    def evaluate_autophagy_candidates(self, mycelial_layer) -> List[Tuple[str, AutophagyTrigger]]:
        """Find nodes that should undergo autophagy"""
        candidates = []
        
        for node_id, node in mycelial_layer.nodes.items():
            trigger = self._evaluate_single_node(node)
            if trigger:
                candidates.append((node_id, trigger))
        
        return candidates
    
    def _evaluate_single_node(self, node) -> Optional[AutophagyTrigger]:
        """Evaluate if a single node should undergo autophagy"""
        
        # Energy starvation check
        if (node.energy < self.config.energy_threshold and 
            node.energy_low_ticks >= self.config.starvation_time):
            return AutophagyTrigger.ENERGY_STARVATION
        
        # Isolation check  
        if len(node.connections) <= self.config.isolation_threshold:
            # Only trigger if isolated for a while
            if hasattr(node, 'isolation_ticks'):
                if node.isolation_ticks > self.config.starvation_time:
                    return AutophagyTrigger.ISOLATION
            else:
                node.isolation_ticks = 0
            node.isolation_ticks += 1
        else:
            if hasattr(node, 'isolation_ticks'):
                node.isolation_ticks = 0
        
        # Entropy overload check
        if node.entropy > self.config.entropy_threshold:
            if hasattr(node, 'entropy_overload_ticks'):
                if node.entropy_overload_ticks > self.config.starvation_time:
                    return AutophagyTrigger.ENTROPY_OVERLOAD
            else:
                node.entropy_overload_ticks = 0
            node.entropy_overload_ticks += 1
        else:
            if hasattr(node, 'entropy_overload_ticks'):
                node.entropy_overload_ticks = 0
        
        return None
    
    def initiate_autophagy(self, node_id: str, trigger: AutophagyTrigger, mycelial_layer) -> bool:
        """Initiate autophagy process for a node"""
        if node_id not in mycelial_layer.nodes:
            return False
        
        if node_id in self.nodes_in_autophagy:
            return False  # Already in autophagy
        
        node = mycelial_layer.nodes[node_id]
        
        # Check for revival possibility
        if np.random.random() < self.config.revival_possibility:
            logger.info(f"Node {node_id} avoided autophagy through revival")
            return False
        
        # Start autophagy process
        autophagy_state = {
            'node_id': node_id,
            'trigger': trigger,
            'start_time': time.time(),
            'start_tick': mycelial_layer.tick_count,
            'duration_remaining': self.config.autophagy_duration,
            'original_energy': node.energy,
            'metabolites_to_produce': [],
            'neighbors': list(node.connections.keys())
        }
        
        self.nodes_in_autophagy[node_id] = autophagy_state
        
        # Update node state
        if hasattr(node, 'state'):
            node.state = 'NodeState.AUTOPHAGY'
        
        logger.info(f"Initiated autophagy for node {node_id} due to {trigger.value}")
        return True
    
    def process_autophagy(self, mycelial_layer, delta_time: float = 1.0) -> List[Dict[str, Any]]:
        """Process ongoing autophagy and produce metabolites"""
        all_metabolites = []
        completed_autophagies = []
        
        for node_id, autophagy_state in self.nodes_in_autophagy.items():
            node = mycelial_layer.nodes.get(node_id)
            if not node:
                completed_autophagies.append(node_id)
                continue
            
            # Update autophagy progress
            autophagy_state['duration_remaining'] -= delta_time
            
            # Produce metabolites during autophagy
            metabolites = self._produce_metabolites(node, autophagy_state)
            all_metabolites.extend(metabolites)
            
            # Check if autophagy is complete
            if autophagy_state['duration_remaining'] <= 0:
                final_metabolites = self._complete_autophagy(node, autophagy_state, mycelial_layer)
                all_metabolites.extend(final_metabolites)
                completed_autophagies.append(node_id)
        
        # Remove completed autophagies
        for node_id in completed_autophagies:
            if node_id in self.nodes_in_autophagy:
                del self.nodes_in_autophagy[node_id]
        
        # Distribute metabolites to neighbors
        if all_metabolites:
            self._distribute_metabolites(all_metabolites, mycelial_layer)
        
        return all_metabolites
    
    def _produce_metabolites(self, node, autophagy_state) -> List[Dict[str, Any]]:
        """Produce metabolites during autophagy process"""
        metabolites = []
        
        # Gradual energy conversion to metabolites
        energy_to_convert = node.energy * 0.1  # Convert 10% per tick
        
        if energy_to_convert > 0.01:  # Minimum threshold
            metabolite = {
                'type': 'semantic_trace',
                'source_id': node.id,
                'energy_content': energy_to_convert * self.config.energy_recovery_rate,
                'semantic_content': {
                    'pressure': node.pressure,
                    'drift_alignment': node.drift_alignment,
                    'position': node.position.clone().detach(),
                    'connections': list(node.connections.keys()),
                    'recency': node.recency,
                    'entropy': node.entropy
                },
                'production_stage': 'ongoing',
                'timestamp': time.time(),
                'autophagy_trigger': autophagy_state['trigger'].value
            }
            
            metabolites.append(metabolite)
            
            # Reduce node energy
            node.energy = max(0.0, node.energy - energy_to_convert)
            
            self.metabolites_created += 1
            self.energy_recycled += metabolite['energy_content']
        
        return metabolites
    
    def _complete_autophagy(self, node, autophagy_state, mycelial_layer) -> List[Dict[str, Any]]:
        """Complete autophagy process and produce final metabolites"""
        final_metabolites = []
        
        # Convert remaining energy to final metabolite
        if node.energy > 0:
            final_metabolite = {
                'type': 'final_essence',
                'source_id': node.id,
                'energy_content': node.energy * self.config.energy_recovery_rate,
                'semantic_content': {
                    'pressure': node.pressure,
                    'drift_alignment': node.drift_alignment,
                    'position': node.position.clone().detach(),
                    'final_connections': list(node.connections.keys()),
                    'total_lifetime': time.time() - node.created_at,
                    'final_state': {
                        'energy': node.energy,
                        'health': node.health,
                        'entropy': node.entropy
                    }
                },
                'production_stage': 'final',
                'timestamp': time.time(),
                'autophagy_trigger': autophagy_state['trigger'].value
            }
            
            final_metabolites.append(final_metabolite)
            self.metabolites_created += 1
            self.energy_recycled += final_metabolite['energy_content']
        
        # Record autophagy completion
        self.autophagy_history.append({
            'node_id': node.id,
            'trigger': autophagy_state['trigger'].value,
            'start_time': autophagy_state['start_time'],
            'completion_time': time.time(),
            'duration': time.time() - autophagy_state['start_time'],
            'original_energy': autophagy_state['original_energy'],
            'final_energy': node.energy,
            'metabolites_produced': len(autophagy_state['metabolites_to_produce']) + len(final_metabolites),
            'energy_recycled': sum(m['energy_content'] for m in final_metabolites)
        })
        
        self.total_autophagies += 1
        
        # Set node to dormant state
        node.energy = 0.01  # Minimal energy to prevent immediate removal
        if hasattr(node, 'state'):
            node.state = 'NodeState.DORMANT'
        
        logger.info(f"Completed autophagy for node {node.id}, produced {len(final_metabolites)} final metabolites")
        
        return final_metabolites
    
    def _distribute_metabolites(self, metabolites: List[Dict[str, Any]], mycelial_layer):
        """Distribute metabolites to neighboring nodes"""
        for metabolite in metabolites:
            source_id = metabolite['source_id']
            source_node = mycelial_layer.nodes.get(source_id)
            
            if not source_node:
                continue
            
            # Find neighbors within distribution radius
            target_nodes = self._find_distribution_targets(source_id, mycelial_layer)
            
            # Distribute to each target
            for target_id in target_nodes:
                target_node = mycelial_layer.nodes.get(target_id)
                if target_node and target_node.energy < target_node.max_energy:
                    absorbed = target_node.absorb_metabolite(metabolite)
                    if absorbed:
                        logger.debug(f"Node {target_id} absorbed metabolite from {source_id}")
    
    def _find_distribution_targets(self, source_id: str, mycelial_layer) -> List[str]:
        """Find nodes within distribution radius for metabolite sharing"""
        targets = []
        visited = set()
        queue = deque([(source_id, 0)])  # (node_id, distance)
        
        while queue:
            current_id, distance = queue.popleft()
            
            if current_id in visited:
                continue
            
            visited.add(current_id)
            
            if distance > 0:  # Don't include source
                targets.append(current_id)
            
            if distance < self.config.metabolite_distribution_radius:
                current_node = mycelial_layer.nodes.get(current_id)
                if current_node:
                    for neighbor_id in current_node.connections.keys():
                        if neighbor_id not in visited:
                            queue.append((neighbor_id, distance + 1))
        
        return targets
    
    def get_autophagy_statistics(self) -> Dict[str, Any]:
        """Get comprehensive autophagy statistics"""
        active_count = len(self.nodes_in_autophagy)
        
        # Calculate average autophagy duration
        if self.autophagy_history:
            avg_duration = sum(h['duration'] for h in self.autophagy_history) / len(self.autophagy_history)
        else:
            avg_duration = 0.0
        
        # Count triggers
        trigger_counts = defaultdict(int)
        for history in self.autophagy_history:
            trigger_counts[history['trigger']] += 1
        
        return {
            'total_autophagies': self.total_autophagies,
            'active_autophagies': active_count,
            'metabolites_created': self.metabolites_created,
            'energy_recycled': self.energy_recycled,
            'average_duration': avg_duration,
            'trigger_distribution': dict(trigger_counts),
            'config': {
                'energy_threshold': self.config.energy_threshold,
                'starvation_time': self.config.starvation_time,
                'energy_recovery_rate': self.config.energy_recovery_rate,
                'distribution_radius': self.config.metabolite_distribution_radius
            },
            'active_nodes': list(self.nodes_in_autophagy.keys())
        }
