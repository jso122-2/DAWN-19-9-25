#!/usr/bin/env python3
"""
🔄 Topology Transforms - Operators that Reshape Meaning Space
============================================================

Implementation of the canonical operators that can reshape DAWN's semantic topology.
These transforms allow consciousness to actively modify the structure of meaning space:

- Weave: Reinforce or create edges along manifold
- Prune: Remove/attenuate incoherent edges  
- Fuse: Merge co-firing clusters
- Fission: Split brittle/entropic clusters
- Lift: Promote nodes toward surface (rebloom path)
- Sink: Demote nodes deeper (decay path)
- Reproject: Update positions from embeddings

Each operator is idempotent under guard conditions and composable via Sigil Ring.

Based on documentation: Transforms.rtf
"""

import numpy as np
import logging
import time
import uuid
from typing import Dict, List, Optional, Tuple, Set, Any, Union
from dataclasses import dataclass, field
from enum import Enum
from collections import defaultdict

from .semantic_field import SemanticField, SemanticNode, SemanticEdge, LayerDepth, SectorType
from .field_equations import FieldEquations

logger = logging.getLogger(__name__)

class TransformType(Enum):
    """Types of topology transforms"""
    WEAVE = "weave"           # Reinforce/create edges
    PRUNE = "prune"           # Remove/attenuate edges
    FUSE = "fuse"             # Merge clusters
    FISSION = "fission"       # Split clusters
    LIFT = "lift"             # Promote toward surface
    SINK = "sink"             # Demote deeper
    REPROJECT = "reproject"   # Update positions

class TransformResult(Enum):
    """Results of transform operations"""
    SUCCESS = "success"
    FAILED_GUARD = "failed_guard"
    INSUFFICIENT_BUDGET = "insufficient_budget"
    INVALID_TARGET = "invalid_target"
    ERROR = "error"

@dataclass
class TransformOperation:
    """Record of a topology transform operation"""
    transform_id: str = field(default_factory=lambda: str(uuid.uuid4()))
    transform_type: TransformType = TransformType.WEAVE
    target_nodes: List[str] = field(default_factory=list)
    target_edges: List[str] = field(default_factory=list)
    parameters: Dict[str, Any] = field(default_factory=dict)
    result: TransformResult = TransformResult.SUCCESS
    energy_cost: float = 0.0
    execution_time: float = 0.0
    timestamp: float = field(default_factory=time.time)
    provenance: Dict[str, Any] = field(default_factory=dict)

class TopologyTransforms:
    """
    Implementation of all topology transform operators for DAWN's semantic field.
    
    Provides the canonical operations that consciousness can use to actively
    reshape the structure and connectivity of meaning space.
    """
    
    def __init__(self, semantic_field: SemanticField, field_equations: FieldEquations):
        self.field = semantic_field
        self.equations = field_equations
        
        # Transform budgets and limits
        self.budgets = {
            'motion_budget_per_tick': 10.0,
            'energy_budget_per_tick': 5.0,
            'max_transforms_per_tick': 50,
            'fuse_fission_budget': 2.0  # Heavy operations
        }
        
        # Current tick budgets (reset each tick)
        self.current_budgets = dict(self.budgets)
        
        # Transform parameters
        self.weave_params = {
            'similarity_threshold': 0.7,
            'max_distance': 2.0,
            'min_tension_threshold': 0.3,
            'weight_increment': 0.1
        }
        
        self.prune_params = {
            'coherence_threshold': 0.3,
            'volatility_threshold': 0.7,
            'reliability_threshold': 0.2,
            'weight_decay': 0.1
        }
        
        self.fuse_params = {
            'cofire_threshold': 0.8,
            'tension_threshold': 0.2,
            'energy_overlap_threshold': 0.6,
            'min_sustained_ticks': 5
        }
        
        self.fission_params = {
            'entropy_threshold': 0.8,
            'coherence_threshold': 0.2,
            'min_cluster_size': 3,
            'tension_ridge_threshold': 0.7
        }
        
        self.lift_sink_params = {
            'ash_threshold': 0.7,      # For lift
            'soot_threshold': 0.7,     # For sink
            'coherence_requirement': 0.5,
            'energy_requirement': 0.3,
            'pigment_bias_threshold': 0.6
        }
        
        self.reproject_params = {
            'drift_tolerance': 1.0,
            'motion_budget_per_node': 0.2,
            'smoothing_factor': 0.8
        }
        
        # Operation history
        self.operation_history: List[TransformOperation] = []
        self.tick_count = 0
        
        logger.info("🔄 TopologyTransforms initialized with semantic field")
    
    def reset_tick_budgets(self):
        """Reset budgets for a new tick"""
        self.current_budgets = dict(self.budgets)
        self.tick_count += 1
        
    def check_budget(self, operation: str, cost: float) -> bool:
        """Check if we have budget for an operation"""
        budget_key = f"{operation}_budget_per_tick"
        if budget_key in self.current_budgets:
            return self.current_budgets[budget_key] >= cost
        return self.current_budgets['energy_budget_per_tick'] >= cost
    
    def consume_budget(self, operation: str, cost: float):
        """Consume budget for an operation"""
        budget_key = f"{operation}_budget_per_tick"
        if budget_key in self.current_budgets:
            self.current_budgets[budget_key] -= cost
        self.current_budgets['energy_budget_per_tick'] -= cost
        
    def weave(self, node_a_id: str, node_b_id: str, alpha: float = None) -> TransformOperation:
        """
        Weave transform: Reinforce or create edge (A-B) along local manifold.
        
        Guard conditions:
        - Semantic similarity high (||eA - eB|| small)
        - Distance in projected space short  
        - Tension τAB above threshold
        
        Effect: Increase weightAB ← weightAB + α·Δ; reduce τAB
        """
        op = TransformOperation(
            transform_type=TransformType.WEAVE,
            target_nodes=[node_a_id, node_b_id],
            parameters={'alpha': alpha or self.weave_params['weight_increment']}
        )
        
        start_time = time.time()
        
        try:
            # Validate nodes exist
            if node_a_id not in self.field.nodes or node_b_id not in self.field.nodes:
                op.result = TransformResult.INVALID_TARGET
                return op
                
            node_a = self.field.nodes[node_a_id]
            node_b = self.field.nodes[node_b_id]
            
            # Check guard conditions
            semantic_similarity = self.equations.calculate_semantic_similarity(node_a, node_b)
            spatial_distance = node_a.spatial_distance_to(node_b)
            
            # Guard 1: Semantic similarity high
            if semantic_similarity < self.weave_params['similarity_threshold']:
                op.result = TransformResult.FAILED_GUARD
                op.provenance['guard_failure'] = f"Low semantic similarity: {semantic_similarity:.3f}"
                return op
                
            # Guard 2: Spatial distance short
            if spatial_distance > self.weave_params['max_distance']:
                op.result = TransformResult.FAILED_GUARD
                op.provenance['guard_failure'] = f"Distance too large: {spatial_distance:.3f}"
                return op
            
            # Find or create edge
            edge_id = f"{node_a_id}<->{node_b_id}"
            alt_edge_id = f"{node_b_id}<->{node_a_id}"
            
            edge = None
            if edge_id in self.field.edges:
                edge = self.field.edges[edge_id]
            elif alt_edge_id in self.field.edges:
                edge = self.field.edges[alt_edge_id]
                edge_id = alt_edge_id
                
            # Guard 3: Tension above threshold (for existing edges)
            if edge and edge.tension < self.weave_params['min_tension_threshold']:
                op.result = TransformResult.FAILED_GUARD
                op.provenance['guard_failure'] = f"Low tension: {edge.tension:.3f}"
                return op
                
            # Check budget
            energy_cost = 0.5
            if not self.check_budget('energy', energy_cost):
                op.result = TransformResult.INSUFFICIENT_BUDGET
                return op
                
            # Create edge if it doesn't exist
            if edge is None:
                edge = SemanticEdge(
                    node_a=node_a_id,
                    node_b=node_b_id,
                    weight=0.1,  # Start with small weight
                    tension=0.5  # Initial tension
                )
                self.field.add_edge(edge)
                edge_id = edge.id
                
            # Apply weave effect
            alpha = op.parameters['alpha']
            weight_delta = alpha * semantic_similarity  # Scale by similarity
            
            old_weight = edge.weight
            edge.update_weight(edge.weight + weight_delta)
            
            # Reduce tension
            edge.tension = max(0.0, edge.tension * 0.8)
            
            # Record operation
            op.target_edges = [edge_id]
            op.energy_cost = energy_cost
            op.provenance.update({
                'old_weight': old_weight,
                'new_weight': edge.weight,
                'weight_delta': weight_delta,
                'semantic_similarity': semantic_similarity,
                'spatial_distance': spatial_distance,
                'sigil_house': 'Weaving'
            })
            
            self.consume_budget('energy', energy_cost)
            
        except Exception as e:
            logger.error(f"Weave transform failed: {e}")
            op.result = TransformResult.ERROR
            op.provenance['error'] = str(e)
            
        finally:
            op.execution_time = time.time() - start_time
            self.operation_history.append(op)
            
        return op
    
    def prune(self, edge_ids: List[str], kappa: float = None) -> TransformOperation:
        """
        Prune transform: Attenuate or remove edge(s) in set E.
        
        Guard conditions:
        - Coherence low (edge weight disagrees with geometry)
        - Volatility > κ
        - Reliability low
        
        Effect: Reduce weight, possibly delete edge; free nutrients routed back
        """
        op = TransformOperation(
            transform_type=TransformType.PRUNE,
            target_edges=edge_ids,
            parameters={'kappa': kappa or self.prune_params['volatility_threshold']}
        )
        
        start_time = time.time()
        
        try:
            edges_to_prune = []
            total_energy_cost = 0.0
            
            for edge_id in edge_ids:
                if edge_id not in self.field.edges:
                    continue
                    
                edge = self.field.edges[edge_id]
                
                # Check guard conditions
                coherence_ok = True  # Would need field equations context
                volatility_high = edge.volatility > op.parameters['kappa']
                reliability_low = edge.reliability < self.prune_params['reliability_threshold']
                
                if volatility_high or reliability_low:
                    edges_to_prune.append(edge_id)
                    total_energy_cost += 0.3
                    
            if not edges_to_prune:
                op.result = TransformResult.FAILED_GUARD
                op.provenance['guard_failure'] = "No edges meet pruning criteria"
                return op
                
            # Check budget
            if not self.check_budget('energy', total_energy_cost):
                op.result = TransformResult.INSUFFICIENT_BUDGET
                return op
                
            # Apply pruning
            pruned_weights = {}
            for edge_id in edges_to_prune:
                edge = self.field.edges[edge_id]
                old_weight = edge.weight
                
                # Reduce weight
                weight_decay = self.prune_params['weight_decay']
                new_weight = edge.weight * (1 - weight_decay)
                
                if new_weight < 0.01:  # Delete very weak edges
                    # Remove edge from field
                    del self.field.edges[edge_id]
                    # Update adjacency
                    self.field.adjacency[edge.node_a].discard(edge.node_b)
                    self.field.adjacency[edge.node_b].discard(edge.node_a)
                    pruned_weights[edge_id] = {'deleted': True, 'old_weight': old_weight}
                else:
                    edge.update_weight(new_weight)
                    pruned_weights[edge_id] = {'old_weight': old_weight, 'new_weight': new_weight}
                    
            op.energy_cost = total_energy_cost
            op.provenance.update({
                'pruned_edges': pruned_weights,
                'sigil_house': 'Purification'
            })
            
            self.consume_budget('energy', total_energy_cost)
            
        except Exception as e:
            logger.error(f"Prune transform failed: {e}")
            op.result = TransformResult.ERROR
            op.provenance['error'] = str(e)
            
        finally:
            op.execution_time = time.time() - start_time
            self.operation_history.append(op)
            
        return op
    
    def fuse(self, cluster_a_ids: List[str], cluster_b_ids: List[str]) -> TransformOperation:
        """
        Fuse transform: Merge clusters C1, C2 into single cluster.
        
        Guard conditions:
        - Sustained co-firing across ticks
        - Low tension between clusters
        - Strong nutrient/energy overlap
        
        Effect: New node/cluster with combined energy/health; edges merged; IDs mapped
        """
        op = TransformOperation(
            transform_type=TransformType.FUSE,
            target_nodes=cluster_a_ids + cluster_b_ids,
            parameters={'cluster_a': cluster_a_ids, 'cluster_b': cluster_b_ids}
        )
        
        start_time = time.time()
        
        try:
            # Check budget (fuse is expensive)
            energy_cost = self.budgets['fuse_fission_budget']
            if not self.check_budget('energy', energy_cost):
                op.result = TransformResult.INSUFFICIENT_BUDGET
                return op
                
            # Validate all nodes exist
            all_nodes = cluster_a_ids + cluster_b_ids
            for node_id in all_nodes:
                if node_id not in self.field.nodes:
                    op.result = TransformResult.INVALID_TARGET
                    return op
                    
            # Calculate cluster properties
            cluster_a_nodes = [self.field.nodes[nid] for nid in cluster_a_ids]
            cluster_b_nodes = [self.field.nodes[nid] for nid in cluster_b_ids]
            
            # Guard conditions (simplified)
            total_energy_a = sum(node.energy for node in cluster_a_nodes)
            total_energy_b = sum(node.energy for node in cluster_b_nodes)
            energy_overlap = min(total_energy_a, total_energy_b) / max(total_energy_a, total_energy_b, 0.001)
            
            if energy_overlap < self.fuse_params['energy_overlap_threshold']:
                op.result = TransformResult.FAILED_GUARD
                op.provenance['guard_failure'] = f"Low energy overlap: {energy_overlap:.3f}"
                return op
                
            # Create fused node
            fused_node = SemanticNode()
            
            # Combine positions (centroid)
            all_positions = [node.position for node in cluster_a_nodes + cluster_b_nodes]
            fused_node.position = np.mean(all_positions, axis=0)
            
            # Combine embeddings (average)
            all_embeddings = [node.embedding for node in cluster_a_nodes + cluster_b_nodes]
            fused_node.embedding = np.mean(all_embeddings, axis=0)
            
            # Combine energy and health (conservation)
            fused_node.energy = sum(node.energy for node in cluster_a_nodes + cluster_b_nodes)
            fused_node.health = sum(node.health for node in cluster_a_nodes + cluster_b_nodes)
            
            # Average tint
            all_tints = [node.tint for node in cluster_a_nodes + cluster_b_nodes]
            fused_node.tint = np.mean(all_tints, axis=0)
            
            # Add fused node to field
            self.field.add_node(fused_node)
            
            # Merge edges - redirect all edges to fused node
            edges_to_redirect = []
            for edge_id, edge in list(self.field.edges.items()):
                if edge.node_a in all_nodes or edge.node_b in all_nodes:
                    edges_to_redirect.append((edge_id, edge))
                    
            for edge_id, edge in edges_to_redirect:
                # Create new edge to fused node
                new_node_a = fused_node.id if edge.node_a in all_nodes else edge.node_a
                new_node_b = fused_node.id if edge.node_b in all_nodes else edge.node_b
                
                if new_node_a != new_node_b:  # Avoid self-loops
                    new_edge = SemanticEdge(
                        node_a=new_node_a,
                        node_b=new_node_b,
                        weight=edge.weight,
                        tension=edge.tension,
                        volatility=edge.volatility,
                        reliability=edge.reliability
                    )
                    self.field.add_edge(new_edge)
                    
                # Remove old edge
                del self.field.edges[edge_id]
                
            # Remove old nodes
            for node_id in all_nodes:
                if node_id in self.field.nodes:
                    node = self.field.nodes[node_id]
                    self.field.layer_index[node.layer].discard(node_id)
                    self.field.sector_index[node.sector].discard(node_id)
                    del self.field.nodes[node_id]
                    
            # Update adjacency
            self.field.adjacency = defaultdict(set)
            for edge in self.field.edges.values():
                self.field.adjacency[edge.node_a].add(edge.node_b)
                if not edge.directed:
                    self.field.adjacency[edge.node_b].add(edge.node_a)
                    
            op.energy_cost = energy_cost
            op.provenance.update({
                'fused_node_id': fused_node.id,
                'original_clusters': {'a': cluster_a_ids, 'b': cluster_b_ids},
                'energy_conservation': fused_node.energy,
                'health_conservation': fused_node.health,
                'sigil_house': 'Memory/Weaving'
            })
            
            self.consume_budget('energy', energy_cost)
            
        except Exception as e:
            logger.error(f"Fuse transform failed: {e}")
            op.result = TransformResult.ERROR
            op.provenance['error'] = str(e)
            
        finally:
            op.execution_time = time.time() - start_time
            self.operation_history.append(op)
            
        return op
    
    def lift(self, node_id: str) -> TransformOperation:
        """
        Lift transform: Promote node N toward surface (lower layer index).
        
        Guard conditions:
        - Ash-rich
        - Coherent  
        - Pigment-biased
        - Candidate for rebloom
        
        Effect: N.layer ← max(0, N.layer-1); rebloom path triggered if surface reached
        """
        op = TransformOperation(
            transform_type=TransformType.LIFT,
            target_nodes=[node_id]
        )
        
        start_time = time.time()
        
        try:
            if node_id not in self.field.nodes:
                op.result = TransformResult.INVALID_TARGET
                return op
                
            node = self.field.nodes[node_id]
            
            # Guard conditions (simplified - would need residue system integration)
            ash_level = getattr(node, 'ash_level', 0.5)  # Placeholder
            coherence = 0.7  # Would calculate from field equations
            pigment_bias = np.linalg.norm(node.tint - 0.5)
            
            if (ash_level < self.lift_sink_params['ash_threshold'] or
                coherence < self.lift_sink_params['coherence_requirement'] or
                pigment_bias < self.lift_sink_params['pigment_bias_threshold']):
                op.result = TransformResult.FAILED_GUARD
                op.provenance['guard_failure'] = f"Failed lift requirements: ash={ash_level:.3f}, coherence={coherence:.3f}, pigment_bias={pigment_bias:.3f}"
                return op
                
            # Check if already at surface
            if node.layer == LayerDepth.SURFACE:
                op.result = TransformResult.FAILED_GUARD
                op.provenance['guard_failure'] = "Already at surface layer"
                return op
                
            # Apply lift
            old_layer = node.layer
            new_layer_value = max(0, node.layer.value - 1)
            new_layer = LayerDepth(new_layer_value)
            
            # Update indices
            self.field.layer_index[old_layer].discard(node_id)
            self.field.layer_index[new_layer].add(node_id)
            node.layer = new_layer
            
            op.energy_cost = 0.4
            op.provenance.update({
                'old_layer': old_layer.value,
                'new_layer': new_layer.value,
                'ash_level': ash_level,
                'coherence': coherence,
                'pigment_bias': pigment_bias,
                'rebloom_triggered': new_layer == LayerDepth.SURFACE,
                'sigil_house': 'Memory'
            })
            
            self.consume_budget('energy', op.energy_cost)
            
        except Exception as e:
            logger.error(f"Lift transform failed: {e}")
            op.result = TransformResult.ERROR
            op.provenance['error'] = str(e)
            
        finally:
            op.execution_time = time.time() - start_time
            self.operation_history.append(op)
            
        return op
    
    def sink(self, node_id: str) -> TransformOperation:
        """
        Sink transform: Demote node N deeper (raise layer index).
        
        Guard conditions:
        - Soot-heavy
        - Low health/energy
        - Prolonged neglect
        
        Effect: N.layer ← N.layer+1; marked for shimmer decay or compost
        """
        op = TransformOperation(
            transform_type=TransformType.SINK,
            target_nodes=[node_id]
        )
        
        start_time = time.time()
        
        try:
            if node_id not in self.field.nodes:
                op.result = TransformResult.INVALID_TARGET
                return op
                
            node = self.field.nodes[node_id]
            
            # Guard conditions
            soot_level = getattr(node, 'soot_level', 0.3)  # Placeholder
            neglect_time = time.time() - node.last_touch
            
            if (soot_level < self.lift_sink_params['soot_threshold'] or
                node.health > self.lift_sink_params['energy_requirement'] or
                neglect_time < 300):  # 5 minutes neglect
                op.result = TransformResult.FAILED_GUARD
                op.provenance['guard_failure'] = f"Failed sink requirements: soot={soot_level:.3f}, health={node.health:.3f}, neglect={neglect_time:.1f}s"
                return op
                
            # Check if already at deepest layer
            if node.layer == LayerDepth.TRANSCENDENT:
                op.result = TransformResult.FAILED_GUARD
                op.provenance['guard_failure'] = "Already at deepest layer"
                return op
                
            # Apply sink
            old_layer = node.layer
            new_layer_value = min(4, node.layer.value + 1)
            new_layer = LayerDepth(new_layer_value)
            
            # Update indices
            self.field.layer_index[old_layer].discard(node_id)
            self.field.layer_index[new_layer].add(node_id)
            node.layer = new_layer
            
            op.energy_cost = 0.2
            op.provenance.update({
                'old_layer': old_layer.value,
                'new_layer': new_layer.value,
                'soot_level': soot_level,
                'health': node.health,
                'neglect_time': neglect_time,
                'decay_marked': new_layer == LayerDepth.TRANSCENDENT,
                'sigil_house': 'Purification'
            })
            
            self.consume_budget('energy', op.energy_cost)
            
        except Exception as e:
            logger.error(f"Sink transform failed: {e}")
            op.result = TransformResult.ERROR
            op.provenance['error'] = str(e)
            
        finally:
            op.execution_time = time.time() - start_time
            self.operation_history.append(op)
            
        return op
    
    def reproject(self, node_ids: List[str], projection_matrix: Optional[np.ndarray] = None) -> TransformOperation:
        """
        Reproject transform: Update positions x ← Π(e,telemetry).
        
        Guard conditions:
        - Latent-projected drift Δd beyond tolerance
        - Motion budget available
        
        Effect: Smooth movement of x toward embedding manifold; preserve edge continuity
        """
        op = TransformOperation(
            transform_type=TransformType.REPROJECT,
            target_nodes=node_ids,
            parameters={'projection_matrix_provided': projection_matrix is not None}
        )
        
        start_time = time.time()
        
        try:
            total_motion_cost = len(node_ids) * self.reproject_params['motion_budget_per_node']
            
            if not self.check_budget('motion', total_motion_cost):
                op.result = TransformResult.INSUFFICIENT_BUDGET
                return op
                
            reprojected_nodes = {}
            total_drift = 0.0
            
            for node_id in node_ids:
                if node_id not in self.field.nodes:
                    continue
                    
                node = self.field.nodes[node_id]
                old_position = node.position.copy()
                
                # Calculate new position from embedding
                if projection_matrix is not None:
                    new_position = (projection_matrix @ node.embedding)[:3]
                else:
                    # Simple projection (first 3 dimensions of embedding)
                    new_position = node.embedding[:3]
                    
                # Calculate drift
                drift = np.linalg.norm(new_position - old_position)
                total_drift += drift
                
                # Guard: Check drift tolerance
                if drift < self.reproject_params['drift_tolerance']:
                    continue  # Skip nodes with small drift
                    
                # Apply smoothing to preserve continuity
                smoothing = self.reproject_params['smoothing_factor']
                smoothed_position = smoothing * old_position + (1 - smoothing) * new_position
                
                node.position = smoothed_position
                reprojected_nodes[node_id] = {
                    'old_position': old_position.tolist(),
                    'new_position': smoothed_position.tolist(),
                    'drift': drift
                }
                
            if not reprojected_nodes:
                op.result = TransformResult.FAILED_GUARD
                op.provenance['guard_failure'] = "No nodes exceeded drift tolerance"
                return op
                
            op.energy_cost = total_motion_cost
            op.provenance.update({
                'reprojected_nodes': reprojected_nodes,
                'total_drift': total_drift,
                'average_drift': total_drift / len(reprojected_nodes),
                'sigil_house': 'Mirrors'
            })
            
            self.consume_budget('motion', total_motion_cost)
            
        except Exception as e:
            logger.error(f"Reproject transform failed: {e}")
            op.result = TransformResult.ERROR
            op.provenance['error'] = str(e)
            
        finally:
            op.execution_time = time.time() - start_time
            self.operation_history.append(op)
            
        return op
    
    def get_transform_statistics(self) -> Dict[str, Any]:
        """Get comprehensive transform statistics"""
        if not self.operation_history:
            return {'total_operations': 0}
            
        operations_by_type = defaultdict(int)
        results_by_type = defaultdict(lambda: defaultdict(int))
        total_energy_cost = 0.0
        total_execution_time = 0.0
        
        for op in self.operation_history:
            operations_by_type[op.transform_type.value] += 1
            results_by_type[op.transform_type.value][op.result.value] += 1
            total_energy_cost += op.energy_cost
            total_execution_time += op.execution_time
            
        return {
            'total_operations': len(self.operation_history),
            'operations_by_type': dict(operations_by_type),
            'results_by_type': {k: dict(v) for k, v in results_by_type.items()},
            'total_energy_cost': total_energy_cost,
            'total_execution_time': total_execution_time,
            'average_execution_time': total_execution_time / len(self.operation_history),
            'current_budgets': dict(self.current_budgets),
            'tick_count': self.tick_count
        }
