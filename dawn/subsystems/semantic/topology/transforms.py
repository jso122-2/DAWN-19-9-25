"""
Semantic Topology Transform Operators
=====================================

Implementation of topology transform operators based on RTF specifications.
Provides weave, prune, fuse, fission, lift, sink, and reproject operations
for semantic topology manipulation.
"""

import numpy as np
import time
import uuid
from typing import Dict, List, Set, Optional, Tuple, Any, Union
from dataclasses import dataclass
from enum import Enum
import logging

from .primitives import SemanticNode, SemanticEdge, TopologyLayer, TopologyFrame, NodeCoordinates

logger = logging.getLogger(__name__)


class TransformResult(Enum):
    """Result status for transform operations"""
    SUCCESS = "success"
    FAILED_PRECONDITION = "failed_precondition"
    FAILED_GUARD = "failed_guard"
    INSUFFICIENT_BUDGET = "insufficient_budget"
    ERROR = "error"


@dataclass
class TransformOperation:
    """Record of a topology transform operation"""
    operation_type: str
    timestamp: float
    node_ids: List[str]
    edge_ids: List[str] 
    parameters: Dict[str, Any]
    result: TransformResult
    energy_cost: float
    description: str


class WeaveOperator:
    """Reinforce or create edges along existing manifold"""
    
    def __init__(self, energy_budget: float = 1.0):
        self.energy_budget = energy_budget
        self.operations: List[TransformOperation] = []
    
    def weave(self, node_a: SemanticNode, node_b: SemanticNode,
              alpha: float, existing_edges: Dict[str, SemanticEdge],
              preserve_geometry: bool = True) -> Tuple[TransformResult, Optional[SemanticEdge]]:
        """
        Weave operation: reinforce or create edge along manifold
        
        Args:
            node_a, node_b: Nodes to connect
            alpha: Weave strength [0,1]
            existing_edges: Current edge dictionary
            preserve_geometry: Whether to preserve local topology
        """
        edge_id = f"{node_a.id}-{node_b.id}"
        reverse_id = f"{node_b.id}-{node_a.id}"
        
        # Calculate energy cost
        distance = node_a.coordinates.spatial_distance(node_b.coordinates)
        energy_cost = alpha * (0.1 + 0.05 * distance)  # Base cost + distance penalty
        
        if energy_cost > self.energy_budget:
            return TransformResult.INSUFFICIENT_BUDGET, None
        
        # Check if edge already exists
        existing_edge = existing_edges.get(edge_id) or existing_edges.get(reverse_id)
        
        try:
            if existing_edge:
                # Reinforce existing edge
                new_weight = min(1.0, existing_edge.weight + alpha * 0.2)
                existing_edge.weight = new_weight
                existing_edge.reliability = min(1.0, existing_edge.reliability + alpha * 0.1)
                
                # Reduce volatility through reinforcement
                existing_edge.volatility = max(0.0, existing_edge.volatility - alpha * 0.1)
                
                result_edge = existing_edge
                operation_type = "weave_reinforce"
                
            else:
                # Create new edge
                similarity = node_a.calculate_similarity(node_b)
                base_weight = alpha * similarity
                
                new_edge = SemanticEdge(
                    id=edge_id,
                    source_id=node_a.id,
                    target_id=node_b.id,
                    weight=base_weight,
                    tension=0.0,  # New edges start with no tension
                    volatility=0.3,  # Moderate initial volatility
                    reliability=0.5 + alpha * 0.3,  # Higher alpha = more reliable
                    directed=False
                )
                
                existing_edges[edge_id] = new_edge
                result_edge = new_edge
                operation_type = "weave_create"
            
            # Apply geometry preservation if requested
            if preserve_geometry and result_edge:
                self._preserve_local_geometry(node_a, node_b, result_edge, existing_edges)
            
            # Record operation
            operation = TransformOperation(
                operation_type=operation_type,
                timestamp=time.time(),
                node_ids=[node_a.id, node_b.id],
                edge_ids=[result_edge.id] if result_edge else [],
                parameters={'alpha': alpha, 'preserve_geometry': preserve_geometry},
                result=TransformResult.SUCCESS,
                energy_cost=energy_cost,
                description=f"Weaved connection between {node_a.id} and {node_b.id}"
            )
            
            self.operations.append(operation)
            self.energy_budget -= energy_cost
            
            logger.debug(f"Weave successful: {operation.description}")
            return TransformResult.SUCCESS, result_edge
            
        except Exception as e:
            logger.error(f"Weave operation failed: {e}")
            return TransformResult.ERROR, None
    
    def _preserve_local_geometry(self, node_a: SemanticNode, node_b: SemanticNode,
                                edge: SemanticEdge, existing_edges: Dict[str, SemanticEdge]):
        """Adjust edge properties to preserve local geometric structure"""
        # Calculate expected weight based on distance
        distance = node_a.coordinates.spatial_distance(node_b.coordinates)
        expected_weight = np.exp(-distance)  # Exponential decay with distance
        
        # Adjust weight to be closer to geometric expectation
        geometry_factor = 0.3  # How much to adjust toward geometric expectation
        edge.weight = (1 - geometry_factor) * edge.weight + geometry_factor * expected_weight


class PruneOperator:
    """Remove or attenuate edges that violate coherence"""
    
    def __init__(self, energy_budget: float = 1.0):
        self.energy_budget = energy_budget
        self.operations: List[TransformOperation] = []
        
        # Pruning thresholds
        self.high_volatility_threshold = 0.8
        self.low_reliability_threshold = 0.3
        self.high_tension_threshold = 0.7
    
    def prune(self, edge: SemanticEdge, kappa: float,
              existing_edges: Dict[str, SemanticEdge],
              remove_completely: bool = False) -> TransformResult:
        """
        Prune operation: remove or attenuate problematic edges
        
        Args:
            edge: Edge to prune
            kappa: Pruning strength [0,1]
            existing_edges: Current edge dictionary
            remove_completely: Whether to remove edge entirely
        """
        energy_cost = kappa * 0.05  # Pruning is generally low cost
        
        if energy_cost > self.energy_budget:
            return TransformResult.INSUFFICIENT_BUDGET
        
        try:
            if remove_completely:
                # Remove edge entirely
                if edge.id in existing_edges:
                    del existing_edges[edge.id]
                operation_type = "prune_remove"
                description = f"Removed edge {edge.id}"
                
            else:
                # Attenuate edge properties
                edge.weight = max(0.0, edge.weight - kappa * 0.3)
                edge.reliability = max(0.0, edge.reliability - kappa * 0.2)
                edge.volatility = min(1.0, edge.volatility + kappa * 0.1)
                
                # If weight becomes too low, mark for removal
                if edge.weight < 0.05:
                    if edge.id in existing_edges:
                        del existing_edges[edge.id]
                    operation_type = "prune_attenuate_remove"
                    description = f"Attenuated and removed weak edge {edge.id}"
                else:
                    operation_type = "prune_attenuate"
                    description = f"Attenuated edge {edge.id}"
            
            # Record operation
            operation = TransformOperation(
                operation_type=operation_type,
                timestamp=time.time(),
                node_ids=[edge.source_id, edge.target_id],
                edge_ids=[edge.id],
                parameters={'kappa': kappa, 'remove_completely': remove_completely},
                result=TransformResult.SUCCESS,
                energy_cost=energy_cost,
                description=description
            )
            
            self.operations.append(operation)
            self.energy_budget -= energy_cost
            
            logger.debug(f"Prune successful: {operation.description}")
            return TransformResult.SUCCESS
            
        except Exception as e:
            logger.error(f"Prune operation failed: {e}")
            return TransformResult.ERROR
    
    def auto_prune_problematic_edges(self, edges: Dict[str, SemanticEdge]) -> List[str]:
        """Automatically prune edges that violate coherence criteria"""
        pruned_edges = []
        
        for edge_id, edge in list(edges.items()):  # Use list() to avoid modification during iteration
            should_prune = False
            prune_strength = 0.5
            
            # Check for high volatility
            if edge.volatility > self.high_volatility_threshold:
                should_prune = True
                prune_strength = max(prune_strength, edge.volatility)
            
            # Check for low reliability
            if edge.reliability < self.low_reliability_threshold:
                should_prune = True
                prune_strength = max(prune_strength, 1.0 - edge.reliability)
            
            # Check for high tension
            if edge.tension > self.high_tension_threshold:
                should_prune = True
                prune_strength = max(prune_strength, edge.tension)
            
            if should_prune and self.energy_budget > 0:
                result = self.prune(edge, prune_strength, edges, remove_completely=True)
                if result == TransformResult.SUCCESS:
                    pruned_edges.append(edge_id)
        
        return pruned_edges


class FuseOperator:
    """Merge neighboring clusters with sustained co-firing"""
    
    def __init__(self, energy_budget: float = 1.0):
        self.energy_budget = energy_budget
        self.operations: List[TransformOperation] = []
    
    def fuse(self, cluster_nodes_1: List[SemanticNode], 
             cluster_nodes_2: List[SemanticNode],
             existing_edges: Dict[str, SemanticEdge],
             preserve_energy: bool = True) -> Tuple[TransformResult, Optional[SemanticNode]]:
        """
        Fuse operation: merge two neighboring clusters
        
        Args:
            cluster_nodes_1, cluster_nodes_2: Node clusters to merge
            existing_edges: Current edge dictionary
            preserve_energy: Whether to conserve total energy
        """
        if not cluster_nodes_1 or not cluster_nodes_2:
            return TransformResult.FAILED_PRECONDITION, None
        
        # Calculate energy cost (proportional to cluster sizes)
        energy_cost = 0.2 * (len(cluster_nodes_1) + len(cluster_nodes_2))
        
        if energy_cost > self.energy_budget:
            return TransformResult.INSUFFICIENT_BUDGET, None
        
        try:
            # Calculate centroid position and properties
            all_nodes = cluster_nodes_1 + cluster_nodes_2
            
            # Position: weighted average of all node positions
            positions = np.array([node.coordinates.position for node in all_nodes])
            weights = np.array([node.energy for node in all_nodes])
            
            if np.sum(weights) > 0:
                centroid_position = np.average(positions, axis=0, weights=weights)
            else:
                centroid_position = np.mean(positions, axis=0)
            
            # Embedding: weighted average of embeddings
            embeddings = np.array([node.coordinates.embedding for node in all_nodes])
            if np.sum(weights) > 0:
                centroid_embedding = np.average(embeddings, axis=0, weights=weights)
            else:
                centroid_embedding = np.mean(embeddings, axis=0)
            
            # Normalize embedding
            if np.linalg.norm(centroid_embedding) > 0:
                centroid_embedding = centroid_embedding / np.linalg.norm(centroid_embedding)
            
            # Create fused node
            fused_coordinates = NodeCoordinates(
                position=centroid_position,
                embedding=centroid_embedding,
                layer=min(node.coordinates.layer for node in all_nodes),  # Use shallowest layer
                sector=cluster_nodes_1[0].coordinates.sector  # Use first cluster's sector
            )
            
            # Aggregate properties
            total_energy = sum(node.energy for node in all_nodes) if preserve_energy else 0.5
            total_health = np.mean([node.health for node in all_nodes])
            
            # Average pigments
            pigments = np.array([node.tint for node in all_nodes])
            average_pigment = np.mean(pigments, axis=0)
            
            # Create content summary
            contents = [node.content for node in all_nodes if node.content]
            fused_content = f"FUSED: {'; '.join(contents[:3])}"  # Limit to first 3
            
            fused_node = SemanticNode(
                id=str(uuid.uuid4()),
                coordinates=fused_coordinates,
                tint=average_pigment,
                health=total_health,
                energy=total_energy,
                content=fused_content,
                last_touch=int(time.time())
            )
            
            # Record operation
            node_ids = [node.id for node in all_nodes]
            operation = TransformOperation(
                operation_type="fuse",
                timestamp=time.time(),
                node_ids=node_ids,
                edge_ids=[],
                parameters={'preserve_energy': preserve_energy, 'cluster_sizes': [len(cluster_nodes_1), len(cluster_nodes_2)]},
                result=TransformResult.SUCCESS,
                energy_cost=energy_cost,
                description=f"Fused {len(all_nodes)} nodes into single node"
            )
            
            self.operations.append(operation)
            self.energy_budget -= energy_cost
            
            logger.debug(f"Fuse successful: {operation.description}")
            return TransformResult.SUCCESS, fused_node
            
        except Exception as e:
            logger.error(f"Fuse operation failed: {e}")
            return TransformResult.ERROR, None


class FissionOperator:
    """Split brittle/high-entropy clusters along natural cuts"""
    
    def __init__(self, energy_budget: float = 1.0):
        self.energy_budget = energy_budget
        self.operations: List[TransformOperation] = []
        
        # Fission criteria
        self.entropy_threshold = 0.7
        self.min_cluster_size = 4  # Minimum size for fission
    
    def fission(self, cluster_nodes: List[SemanticNode],
                psi: float, existing_edges: Dict[str, SemanticEdge],
                preserve_energy: bool = True) -> Tuple[TransformResult, List[List[SemanticNode]]]:
        """
        Fission operation: split cluster along natural boundaries
        
        Args:
            cluster_nodes: Nodes in cluster to split
            psi: Fission strength [0,1]
            existing_edges: Current edge dictionary
            preserve_energy: Whether to conserve total energy
        """
        if len(cluster_nodes) < self.min_cluster_size:
            return TransformResult.FAILED_PRECONDITION, []
        
        energy_cost = 0.15 * len(cluster_nodes) * psi
        
        if energy_cost > self.energy_budget:
            return TransformResult.INSUFFICIENT_BUDGET, []
        
        try:
            # Find natural cut using embedding similarity
            split_clusters = self._find_natural_cut(cluster_nodes, existing_edges)
            
            if len(split_clusters) < 2:
                return TransformResult.FAILED_GUARD, []
            
            # Redistribute energy if preserving
            if preserve_energy:
                total_energy = sum(node.energy for node in cluster_nodes)
                for i, cluster in enumerate(split_clusters):
                    cluster_energy = total_energy * (len(cluster) / len(cluster_nodes))
                    energy_per_node = cluster_energy / len(cluster)
                    
                    for node in cluster:
                        node.energy = energy_per_node
            
            # Record operation
            operation = TransformOperation(
                operation_type="fission",
                timestamp=time.time(),
                node_ids=[node.id for node in cluster_nodes],
                edge_ids=[],
                parameters={'psi': psi, 'preserve_energy': preserve_energy, 'split_count': len(split_clusters)},
                result=TransformResult.SUCCESS,
                energy_cost=energy_cost,
                description=f"Split cluster of {len(cluster_nodes)} nodes into {len(split_clusters)} clusters"
            )
            
            self.operations.append(operation)
            self.energy_budget -= energy_cost
            
            logger.debug(f"Fission successful: {operation.description}")
            return TransformResult.SUCCESS, split_clusters
            
        except Exception as e:
            logger.error(f"Fission operation failed: {e}")
            return TransformResult.ERROR, []
    
    def _find_natural_cut(self, nodes: List[SemanticNode],
                         edges: Dict[str, SemanticEdge]) -> List[List[SemanticNode]]:
        """Find natural cut points in cluster using embedding similarity"""
        if len(nodes) < 2:
            return [nodes]
        
        # Calculate pairwise similarities
        similarities = {}
        for i, node_a in enumerate(nodes):
            for j, node_b in enumerate(nodes[i+1:], i+1):
                sim = node_a.calculate_similarity(node_b)
                similarities[(i, j)] = sim
        
        # Find the weakest connections (lowest similarities)
        sorted_pairs = sorted(similarities.items(), key=lambda x: x[1])
        
        # Use k-means-like clustering on embeddings
        embeddings = np.array([node.coordinates.embedding for node in nodes])
        
        try:
            from sklearn.cluster import KMeans
            kmeans = KMeans(n_clusters=2, random_state=42)
            labels = kmeans.fit_predict(embeddings)
            
            # Group nodes by cluster labels
            cluster_0 = [nodes[i] for i in range(len(nodes)) if labels[i] == 0]
            cluster_1 = [nodes[i] for i in range(len(nodes)) if labels[i] == 1]
            
            return [cluster_0, cluster_1]
            
        except ImportError:
            # Fallback: simple distance-based splitting
            center_embedding = np.mean(embeddings, axis=0)
            
            cluster_0 = []
            cluster_1 = []
            
            for node in nodes:
                distance_to_center = np.linalg.norm(node.coordinates.embedding - center_embedding)
                if distance_to_center < np.median([np.linalg.norm(n.coordinates.embedding - center_embedding) for n in nodes]):
                    cluster_0.append(node)
                else:
                    cluster_1.append(node)
            
            return [cluster_0, cluster_1] if cluster_0 and cluster_1 else [nodes]


class LiftOperator:
    """Promote deep nodes toward surface (rebloom path)"""
    
    def __init__(self, energy_budget: float = 1.0):
        self.energy_budget = energy_budget
        self.operations: List[TransformOperation] = []
        
        self.rebloom_energy_threshold = 0.6
    
    def lift(self, node: SemanticNode, layers: Dict[int, TopologyLayer],
             target_layer: Optional[int] = None) -> TransformResult:
        """
        Lift operation: move node toward surface layers
        
        Args:
            node: Node to lift
            layers: Layer dictionary
            target_layer: Specific target layer (None for one level up)
        """
        current_layer = node.coordinates.layer
        
        if target_layer is None:
            target_layer = max(0, current_layer - 1)  # Move one layer up
        
        if target_layer >= current_layer:
            return TransformResult.FAILED_PRECONDITION  # Can't lift to same or deeper layer
        
        # Check rebloom criteria
        if node.energy < self.rebloom_energy_threshold:
            return TransformResult.FAILED_GUARD
        
        energy_cost = 0.1 * (current_layer - target_layer)  # Cost proportional to lift distance
        
        if energy_cost > self.energy_budget:
            return TransformResult.INSUFFICIENT_BUDGET
        
        try:
            # Remove from current layer
            if current_layer in layers:
                layers[current_layer].remove_node(node.id)
            
            # Add to target layer
            if target_layer not in layers:
                layers[target_layer] = TopologyLayer(depth=target_layer)
            
            if not layers[target_layer].add_node(node.id):
                # Target layer is full, try next available
                for layer_depth in range(target_layer, current_layer):
                    if layer_depth not in layers:
                        layers[layer_depth] = TopologyLayer(depth=layer_depth)
                    if layers[layer_depth].add_node(node.id):
                        target_layer = layer_depth
                        break
                else:
                    return TransformResult.FAILED_GUARD  # No available layer
            
            # Update node layer
            node.coordinates.layer = target_layer
            
            # Boost node properties for successful rebloom
            node.health = min(1.0, node.health + 0.1)
            node.energy = max(0.0, node.energy - 0.1)  # Lifting costs energy
            
            # Record operation
            operation = TransformOperation(
                operation_type="lift",
                timestamp=time.time(),
                node_ids=[node.id],
                edge_ids=[],
                parameters={'from_layer': current_layer, 'to_layer': target_layer},
                result=TransformResult.SUCCESS,
                energy_cost=energy_cost,
                description=f"Lifted node {node.id} from layer {current_layer} to {target_layer}"
            )
            
            self.operations.append(operation)
            self.energy_budget -= energy_cost
            
            logger.debug(f"Lift successful: {operation.description}")
            return TransformResult.SUCCESS
            
        except Exception as e:
            logger.error(f"Lift operation failed: {e}")
            return TransformResult.ERROR


class SinkOperator:
    """Demote surface nodes into underlayers (decay path)"""
    
    def __init__(self, energy_budget: float = 1.0):
        self.energy_budget = energy_budget
        self.operations: List[TransformOperation] = []
        
        self.decay_health_threshold = 0.3
    
    def sink(self, node: SemanticNode, layers: Dict[int, TopologyLayer],
             target_layer: Optional[int] = None) -> TransformResult:
        """
        Sink operation: move node toward deeper layers
        
        Args:
            node: Node to sink
            layers: Layer dictionary  
            target_layer: Specific target layer (None for one level down)
        """
        current_layer = node.coordinates.layer
        
        if target_layer is None:
            target_layer = current_layer + 1  # Move one layer down
        
        if target_layer <= current_layer:
            return TransformResult.FAILED_PRECONDITION  # Can't sink to same or shallower layer
        
        energy_cost = 0.05 * (target_layer - current_layer)  # Sinking is cheaper than lifting
        
        if energy_cost > self.energy_budget:
            return TransformResult.INSUFFICIENT_BUDGET
        
        try:
            # Remove from current layer
            if current_layer in layers:
                layers[current_layer].remove_node(node.id)
            
            # Add to target layer
            if target_layer not in layers:
                layers[target_layer] = TopologyLayer(depth=target_layer)
            
            if not layers[target_layer].add_node(node.id):
                # Target layer is full, try next deeper layer
                layer_depth = target_layer + 1
                while layer_depth < target_layer + 5:  # Try up to 5 layers deeper
                    if layer_depth not in layers:
                        layers[layer_depth] = TopologyLayer(depth=layer_depth)
                    if layers[layer_depth].add_node(node.id):
                        target_layer = layer_depth
                        break
                    layer_depth += 1
                else:
                    return TransformResult.FAILED_GUARD  # No available layer
            
            # Update node layer
            node.coordinates.layer = target_layer
            
            # Apply decay effects
            node.health = max(0.0, node.health - 0.05)  # Slight health decay
            node.energy = max(0.0, node.energy - 0.02)  # Slight energy decay
            
            # Record operation
            operation = TransformOperation(
                operation_type="sink",
                timestamp=time.time(),
                node_ids=[node.id],
                edge_ids=[],
                parameters={'from_layer': current_layer, 'to_layer': target_layer},
                result=TransformResult.SUCCESS,
                energy_cost=energy_cost,
                description=f"Sunk node {node.id} from layer {current_layer} to {target_layer}"
            )
            
            self.operations.append(operation)
            self.energy_budget -= energy_cost
            
            logger.debug(f"Sink successful: {operation.description}")
            return TransformResult.SUCCESS
            
        except Exception as e:
            logger.error(f"Sink operation failed: {e}")
            return TransformResult.ERROR


class ReprojectOperator:
    """Update spatial positions from embeddings without tearing layout"""
    
    def __init__(self, energy_budget: float = 1.0):
        self.energy_budget = energy_budget
        self.operations: List[TransformOperation] = []
        
        self.max_movement_per_tick = 0.2  # Maximum position change per operation
    
    def reproject(self, nodes: Dict[str, SemanticNode],
                  edges: Dict[str, SemanticEdge],
                  projection_method: str = "pca") -> TransformResult:
        """
        Reproject operation: update positions from embeddings with continuity
        
        Args:
            nodes: Node dictionary
            edges: Edge dictionary for continuity constraints
            projection_method: Method for dimensionality reduction
        """
        if not nodes:
            return TransformResult.FAILED_PRECONDITION
        
        energy_cost = 0.1 * len(nodes)
        
        if energy_cost > self.energy_budget:
            return TransformResult.INSUFFICIENT_BUDGET
        
        try:
            # Extract embeddings
            node_list = list(nodes.values())
            embeddings = np.array([node.coordinates.embedding for node in node_list])
            current_positions = np.array([node.coordinates.position for node in node_list])
            
            if embeddings.shape[0] < 2:
                return TransformResult.FAILED_PRECONDITION
            
            # Apply dimensionality reduction
            if projection_method == "pca":
                new_positions = self._pca_projection(embeddings)
            elif projection_method == "tsne":
                new_positions = self._tsne_projection(embeddings)
            else:
                new_positions = self._simple_projection(embeddings)
            
            # Apply continuity constraints
            constrained_positions = self._apply_continuity_constraints(
                current_positions, new_positions, edges, node_list
            )
            
            # Update node positions
            for i, node in enumerate(node_list):
                node.coordinates.position = constrained_positions[i]
            
            # Record operation
            operation = TransformOperation(
                operation_type="reproject",
                timestamp=time.time(),
                node_ids=[node.id for node in node_list],
                edge_ids=[],
                parameters={'method': projection_method, 'node_count': len(node_list)},
                result=TransformResult.SUCCESS,
                energy_cost=energy_cost,
                description=f"Reprojected {len(node_list)} nodes using {projection_method}"
            )
            
            self.operations.append(operation)
            self.energy_budget -= energy_cost
            
            logger.debug(f"Reproject successful: {operation.description}")
            return TransformResult.SUCCESS
            
        except Exception as e:
            logger.error(f"Reproject operation failed: {e}")
            return TransformResult.ERROR
    
    def _pca_projection(self, embeddings: np.ndarray) -> np.ndarray:
        """Project embeddings to 3D using PCA"""
        try:
            from sklearn.decomposition import PCA
            pca = PCA(n_components=3)
            return pca.fit_transform(embeddings)
        except ImportError:
            return self._simple_projection(embeddings)
    
    def _tsne_projection(self, embeddings: np.ndarray) -> np.ndarray:
        """Project embeddings to 3D using t-SNE"""
        try:
            from sklearn.manifold import TSNE
            tsne = TSNE(n_components=3, random_state=42)
            return tsne.fit_transform(embeddings)
        except ImportError:
            return self._simple_projection(embeddings)
    
    def _simple_projection(self, embeddings: np.ndarray) -> np.ndarray:
        """Simple projection using first 3 dimensions"""
        if embeddings.shape[1] >= 3:
            return embeddings[:, :3]
        else:
            # Pad with zeros if embedding is less than 3D
            padded = np.zeros((embeddings.shape[0], 3))
            padded[:, :embeddings.shape[1]] = embeddings
            return padded
    
    def _apply_continuity_constraints(self, current_positions: np.ndarray,
                                    new_positions: np.ndarray,
                                    edges: Dict[str, SemanticEdge],
                                    nodes: List[SemanticNode]) -> np.ndarray:
        """Apply continuity constraints to prevent tearing reliable edges"""
        constrained_positions = new_positions.copy()
        
        # Limit movement per node
        movements = new_positions - current_positions
        movement_magnitudes = np.linalg.norm(movements, axis=1)
        
        # Scale down excessive movements
        for i, magnitude in enumerate(movement_magnitudes):
            if magnitude > self.max_movement_per_tick:
                scale_factor = self.max_movement_per_tick / magnitude
                movements[i] *= scale_factor
        
        constrained_positions = current_positions + movements
        
        # Additional constraint: preserve high-reliability edges
        node_id_to_index = {node.id: i for i, node in enumerate(nodes)}
        
        for edge in edges.values():
            if edge.reliability > 0.8:  # High reliability edges
                if edge.source_id in node_id_to_index and edge.target_id in node_id_to_index:
                    src_idx = node_id_to_index[edge.source_id]
                    tgt_idx = node_id_to_index[edge.target_id]
                    
                    # Calculate current and new distances
                    current_dist = np.linalg.norm(current_positions[src_idx] - current_positions[tgt_idx])
                    new_dist = np.linalg.norm(constrained_positions[src_idx] - constrained_positions[tgt_idx])
                    
                    # If distance change is too large, adjust positions
                    if abs(new_dist - current_dist) > current_dist * 0.5:  # 50% change threshold
                        # Move both nodes toward their midpoint
                        midpoint = (constrained_positions[src_idx] + constrained_positions[tgt_idx]) / 2
                        
                        # Adjust positions to maintain approximate original distance
                        direction = constrained_positions[tgt_idx] - constrained_positions[src_idx]
                        if np.linalg.norm(direction) > 0:
                            direction = direction / np.linalg.norm(direction)
                            half_dist = current_dist / 2
                            
                            constrained_positions[src_idx] = midpoint - direction * half_dist
                            constrained_positions[tgt_idx] = midpoint + direction * half_dist
        
        return constrained_positions


class TopologyTransforms:
    """Unified interface for all topology transform operations"""
    
    def __init__(self, total_energy_budget: float = 5.0):
        self.total_energy_budget = total_energy_budget
        self.current_energy = total_energy_budget
        
        # Initialize operators with shared budget
        budget_per_operator = total_energy_budget / 7
        self.weave = WeaveOperator(budget_per_operator)
        self.prune = PruneOperator(budget_per_operator) 
        self.fuse = FuseOperator(budget_per_operator)
        self.fission = FissionOperator(budget_per_operator)
        self.lift = LiftOperator(budget_per_operator)
        self.sink = SinkOperator(budget_per_operator)
        self.reproject = ReprojectOperator(budget_per_operator)
        
        self.all_operations: List[TransformOperation] = []
    
    def get_operation_history(self) -> List[TransformOperation]:
        """Get complete history of all transform operations"""
        all_ops = []
        all_ops.extend(self.weave.operations)
        all_ops.extend(self.prune.operations)
        all_ops.extend(self.fuse.operations)
        all_ops.extend(self.fission.operations)
        all_ops.extend(self.lift.operations)
        all_ops.extend(self.sink.operations)
        all_ops.extend(self.reproject.operations)
        
        # Sort by timestamp
        all_ops.sort(key=lambda op: op.timestamp)
        return all_ops
    
    def get_energy_status(self) -> Dict[str, float]:
        """Get energy budget status for all operators"""
        return {
            'total_budget': self.total_energy_budget,
            'weave_remaining': self.weave.energy_budget,
            'prune_remaining': self.prune.energy_budget,
            'fuse_remaining': self.fuse.energy_budget,
            'fission_remaining': self.fission.energy_budget,
            'lift_remaining': self.lift.energy_budget,
            'sink_remaining': self.sink.energy_budget,
            'reproject_remaining': self.reproject.energy_budget,
            'total_remaining': (
                self.weave.energy_budget + self.prune.energy_budget +
                self.fuse.energy_budget + self.fission.energy_budget +
                self.lift.energy_budget + self.sink.energy_budget +
                self.reproject.energy_budget
            )
        }
    
    def recharge_energy(self, amount: float = None):
        """Recharge energy budgets for all operators"""
        if amount is None:
            amount = self.total_energy_budget
        
        budget_per_operator = amount / 7
        
        self.weave.energy_budget = budget_per_operator
        self.prune.energy_budget = budget_per_operator
        self.fuse.energy_budget = budget_per_operator
        self.fission.energy_budget = budget_per_operator
        self.lift.energy_budget = budget_per_operator
        self.sink.energy_budget = budget_per_operator
        self.reproject.energy_budget = budget_per_operator
    
    def get_transform_statistics(self) -> Dict[str, Any]:
        """Get statistics about transform operations"""
        operations = self.get_operation_history()
        
        if not operations:
            return {'total_operations': 0}
        
        # Count by type
        type_counts = {}
        total_energy_used = 0
        success_count = 0
        
        for op in operations:
            type_counts[op.operation_type] = type_counts.get(op.operation_type, 0) + 1
            total_energy_used += op.energy_cost
            if op.result == TransformResult.SUCCESS:
                success_count += 1
        
        return {
            'total_operations': len(operations),
            'success_rate': success_count / len(operations),
            'total_energy_used': total_energy_used,
            'operations_by_type': type_counts,
            'latest_operation': operations[-1].operation_type if operations else None,
            'average_energy_per_operation': total_energy_used / len(operations)
        }
