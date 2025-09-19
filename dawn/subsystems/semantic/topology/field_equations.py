"""
Semantic Topology Field Equations
=================================

Implementation of mathematical field equations for semantic topology based on RTF specifications.
Provides local coherence, tension updates, pigment diffusion, and residue pressure calculations.
"""

import numpy as np
import time
from typing import Dict, List, Set, Optional, Tuple, Any
from dataclasses import dataclass
from collections import defaultdict
import logging

from .primitives import SemanticNode, SemanticEdge, NodeCoordinates

logger = logging.getLogger(__name__)


@dataclass
class FieldParameters:
    """Parameters for field equation calculations"""
    # Tension update parameters
    gamma: float = 0.8              # EMA factor for tension [0.7, 0.95]
    lambda_decay: float = 1.0       # Distance decay in f(d) = exp(-λd)
    
    # Pigment diffusion parameters  
    eta: float = 0.05               # Diffusion rate [0.01, 0.1]
    pigment_saturation_cap: float = 2.0  # Maximum pigment magnitude
    
    # Coherence calculation parameters
    similarity_weight: float = 0.6   # Weight for semantic similarity
    distance_weight: float = 0.4     # Weight for spatial distance
    
    # Residue pressure parameters
    soot_weight: float = 0.4        # Weight for soot in pressure calculation
    ash_weight: float = 0.3         # Weight for ash in pressure calculation  
    entropy_weight: float = 0.3     # Weight for entropy in pressure calculation
    
    # Neighborhood parameters
    k_neighbors: int = 8            # K for k-NN neighborhood
    max_neighbor_distance: float = 2.0  # Maximum distance for neighbors


class LocalCoherenceCalculator:
    """Calculate local coherence around semantic nodes"""
    
    def __init__(self, params: FieldParameters):
        self.params = params
    
    def calculate_coherence(self, node: SemanticNode, 
                          neighbors: List[SemanticNode],
                          edges: Dict[str, SemanticEdge]) -> float:
        """
        Calculate local coherence around a node using RTF formula:
        C_i = (1/|N(i)|) * Σ[sim(e_i,e_j) * w_ij / (1 + dist_x(i,j))]
        """
        if not neighbors:
            return 0.0
        
        coherence_sum = 0.0
        valid_neighbors = 0
        
        for neighbor in neighbors:
            # Calculate semantic similarity
            similarity = node.calculate_similarity(neighbor)
            
            # Get edge weight (default to small value if no edge exists)
            edge_key = f"{node.id}-{neighbor.id}"
            reverse_key = f"{neighbor.id}-{node.id}"
            
            edge_weight = 0.1  # Default weak connection
            if edge_key in edges:
                edge_weight = edges[edge_key].weight
            elif reverse_key in edges:
                edge_weight = edges[reverse_key].weight
            
            # Calculate spatial distance
            spatial_dist = node.coordinates.spatial_distance(neighbor.coordinates)
            
            # Apply coherence formula
            coherence_contribution = (similarity * edge_weight) / (1.0 + spatial_dist)
            coherence_sum += coherence_contribution
            valid_neighbors += 1
        
        return coherence_sum / valid_neighbors if valid_neighbors > 0 else 0.0
    
    def find_neighbors(self, node: SemanticNode, 
                      all_nodes: List[SemanticNode]) -> List[SemanticNode]:
        """Find k-nearest neighbors for a node"""
        if len(all_nodes) <= 1:
            return []
        
        # Calculate distances to all other nodes
        distances = []
        for other_node in all_nodes:
            if other_node.id != node.id:
                # Use combined semantic and spatial distance
                semantic_dist = node.coordinates.semantic_distance(other_node.coordinates)
                spatial_dist = node.coordinates.spatial_distance(other_node.coordinates)
                
                # Weighted combination
                combined_dist = (self.params.similarity_weight * semantic_dist + 
                               self.params.distance_weight * spatial_dist)
                
                if combined_dist <= self.params.max_neighbor_distance:
                    distances.append((combined_dist, other_node))
        
        # Sort by distance and take k nearest
        distances.sort(key=lambda x: x[0])
        k = min(self.params.k_neighbors, len(distances))
        
        return [node for _, node in distances[:k]]


class TensionUpdater:
    """Update tension values for semantic edges"""
    
    def __init__(self, params: FieldParameters):
        self.params = params
    
    def update_tension(self, edge: SemanticEdge,
                      source_node: SemanticNode,
                      target_node: SemanticNode) -> float:
        """
        Update edge tension using RTF formula:
        τ_ij ← γ·τ_ij + (1-γ)·|w_ij - f(dist_x(i,j))|
        """
        # Calculate spatial distance
        spatial_dist = source_node.coordinates.spatial_distance(target_node.coordinates)
        
        # Calculate expected weight from distance: f(d) = exp(-λd)
        expected_weight = np.exp(-self.params.lambda_decay * spatial_dist)
        
        # Calculate tension as deviation from expected
        weight_deviation = abs(edge.weight - expected_weight)
        
        # Apply exponential moving average
        new_tension = (self.params.gamma * edge.tension + 
                      (1.0 - self.params.gamma) * weight_deviation)
        
        return max(0.0, new_tension)  # Tension cannot be negative
    
    def batch_update_tensions(self, edges: Dict[str, SemanticEdge],
                             nodes: Dict[str, SemanticNode]) -> Dict[str, float]:
        """Update tensions for all edges in batch"""
        updated_tensions = {}
        
        for edge_id, edge in edges.items():
            if edge.source_id in nodes and edge.target_id in nodes:
                source_node = nodes[edge.source_id]
                target_node = nodes[edge.target_id]
                
                new_tension = self.update_tension(edge, source_node, target_node)
                updated_tensions[edge_id] = new_tension
                
                # Update edge tension in place
                edge.tension = new_tension
            else:
                logger.warning(f"Missing nodes for edge {edge_id}")
        
        return updated_tensions


class PigmentDiffuser:
    """Handle pigment diffusion across semantic topology"""
    
    def __init__(self, params: FieldParameters):
        self.params = params
    
    def diffuse_pigments(self, node: SemanticNode,
                        neighbors: List[SemanticNode],
                        edges: Dict[str, SemanticEdge]) -> np.ndarray:
        """
        Apply pigment diffusion using RTF formula:
        P_i ← P_i + η * Σ[(P_j - P_i) * w_ij]
        """
        if not neighbors:
            return node.tint.copy()
        
        diffusion_sum = np.zeros(3)  # RGB
        
        for neighbor in neighbors:
            # Get edge weight
            edge_key = f"{node.id}-{neighbor.id}"
            reverse_key = f"{neighbor.id}-{node.id}"
            
            edge_weight = 0.1  # Default weak connection
            if edge_key in edges:
                edge_weight = edges[edge_key].weight
            elif reverse_key in edges:
                edge_weight = edges[reverse_key].weight
            
            # Calculate pigment difference
            pigment_diff = neighbor.tint - node.tint
            
            # Apply diffusion
            diffusion_sum += pigment_diff * edge_weight
        
        # Apply diffusion rate
        new_pigment = node.tint + self.params.eta * diffusion_sum
        
        # Apply mythic schema tinting effects
        new_pigment = self._apply_mythic_tinting_effects(new_pigment, node)
        
        # Apply saturation cap
        pigment_magnitude = np.linalg.norm(new_pigment)
        if pigment_magnitude > self.params.pigment_saturation_cap:
            new_pigment = new_pigment * (self.params.pigment_saturation_cap / pigment_magnitude)
        
        # Clamp to [0,1] range
        return np.clip(new_pigment, 0.0, 1.0)
    
    def batch_diffuse_pigments(self, nodes: Dict[str, SemanticNode],
                              edges: Dict[str, SemanticEdge],
                              coherence_calculator: LocalCoherenceCalculator) -> Dict[str, np.ndarray]:
        """Apply pigment diffusion to all nodes"""
        new_pigments = {}
        
        for node_id, node in nodes.items():
            # Find neighbors for this node
            all_nodes = list(nodes.values())
            neighbors = coherence_calculator.find_neighbors(node, all_nodes)
            
            # Calculate new pigment
            new_pigment = self.diffuse_pigments(node, neighbors, edges)
            new_pigments[node_id] = new_pigment
        
        # Apply new pigments
        for node_id, new_pigment in new_pigments.items():
            nodes[node_id].tint = new_pigment
        
        return new_pigments
    
    def _apply_mythic_tinting_effects(self, pigment: np.ndarray, node) -> np.ndarray:
        """Apply mythic schema tinting effects based on pigment channels"""
        r, g, b = pigment
        
        # Apply channel-specific effects as per mythic documentation
        # Red tint = urgency, vitality, passion
        # Green tint = balance, nurture, sustainability  
        # Blue tint = reflection, distance, abstraction
        
        # Enhance dominant channels slightly (positive feedback)
        max_channel_idx = np.argmax(pigment)
        enhancement_factor = 0.05
        
        enhanced_pigment = pigment.copy()
        enhanced_pigment[max_channel_idx] += enhancement_factor * pigment[max_channel_idx]
        
        return enhanced_pigment
    
    def apply_tinting_to_edge_weight(self, base_weight: float, source_pigment: np.ndarray, 
                                   target_pigment: np.ndarray, alpha: float = 0.1) -> float:
        """
        Apply mythic tinting formula to edge weights:
        tinted_weight_ij = w_ij * (1 + α * P_bias)
        
        Where P_bias is the dominant pigment channel bias
        """
        # Calculate average pigment bias between source and target
        avg_pigment = (source_pigment + target_pigment) / 2.0
        r, g, b = avg_pigment
        
        # Determine dominant bias
        dominant_bias = max(r, g, b)
        
        # Apply tinting formula from mythic documentation
        tinted_weight = base_weight * (1.0 + alpha * dominant_bias)
        
        return min(1.0, tinted_weight)  # Cap at 1.0
    
    def calculate_pigment_edge_effects(self, edge_weight: float, source_pigment: np.ndarray, 
                                     target_pigment: np.ndarray) -> Dict[str, float]:
        """
        Calculate mythic pigment effects on edge properties:
        - Red tint: Increases edge activation priority
        - Green tint: Increases edge stability (slower decay)  
        - Blue tint: Biases toward long-distance connections and conceptual drift
        """
        avg_pigment = (source_pigment + target_pigment) / 2.0
        r, g, b = avg_pigment
        
        effects = {
            'activation_priority': 1.0 + r * 0.3,      # Red increases activation priority
            'stability_factor': 1.0 + g * 0.25,       # Green increases stability
            'connection_range': 1.0 + b * 0.2,        # Blue promotes long-range connections
            'decay_resistance': 1.0 + g * 0.15,       # Green resists decay
            'conceptual_drift': b * 0.4                # Blue promotes drift
        }
        
        return effects
    
    def get_pigment_bias_type(self, pigment: np.ndarray) -> str:
        """Determine the dominant pigment bias type"""
        r, g, b = pigment
        max_val = max(r, g, b)
        
        if max_val < 0.1:
            return "neutral"
        elif r == max_val:
            return "urgency"
        elif g == max_val:
            return "balance"
        else:
            return "abstraction"


class ResiduePressureCalculator:
    """Calculate residue pressure fields around nodes"""
    
    def __init__(self, params: FieldParameters):
        self.params = params
    
    def calculate_residue_pressure(self, node: SemanticNode,
                                  neighbors: List[SemanticNode],
                                  residue_data: Dict[str, Dict[str, float]]) -> float:
        """
        Calculate residue pressure using RTF formula:
        ρ_i = softmax_local(soot_avg, ash_avg, entropy_avg)
        """
        if not neighbors:
            return 0.0
        
        # Collect residue data from neighborhood window
        soot_values = []
        ash_values = []
        entropy_values = []
        
        # Include the node itself
        neighborhood = neighbors + [node]
        
        for neighbor in neighborhood:
            node_residue = residue_data.get(neighbor.id, {})
            
            soot_values.append(node_residue.get('soot', 0.0))
            ash_values.append(node_residue.get('ash', 0.0))
            entropy_values.append(node_residue.get('entropy', 0.0))
        
        if not soot_values:
            return 0.0
        
        # Calculate local averages
        soot_avg = np.mean(soot_values)
        ash_avg = np.mean(ash_values)
        entropy_avg = np.mean(entropy_values)
        
        # Apply softmax normalization locally
        inputs = np.array([
            self.params.soot_weight * soot_avg,
            self.params.ash_weight * ash_avg,
            self.params.entropy_weight * entropy_avg
        ])
        
        # Softmax with temperature for smoother distribution
        temperature = 2.0
        exp_inputs = np.exp(inputs / temperature)
        softmax_probs = exp_inputs / np.sum(exp_inputs)
        
        # Return weighted pressure (higher values indicate more pressure)
        pressure = np.dot(softmax_probs, inputs)
        
        return max(0.0, min(1.0, pressure))  # Clamp to [0,1]
    
    def batch_calculate_pressure(self, nodes: Dict[str, SemanticNode],
                               residue_data: Dict[str, Dict[str, float]],
                               coherence_calculator: LocalCoherenceCalculator) -> Dict[str, float]:
        """Calculate residue pressure for all nodes"""
        pressures = {}
        
        for node_id, node in nodes.items():
            # Find neighbors
            all_nodes = list(nodes.values())
            neighbors = coherence_calculator.find_neighbors(node, all_nodes)
            
            # Calculate pressure
            pressure = self.calculate_residue_pressure(node, neighbors, residue_data)
            pressures[node_id] = pressure
            
            # Update node pressure
            node.residue_pressure = pressure
        
        return pressures


class FieldEquationEngine:
    """Main engine for running all field equations"""
    
    def __init__(self, params: Optional[FieldParameters] = None):
        self.params = params or FieldParameters()
        
        # Initialize calculators
        self.coherence_calculator = LocalCoherenceCalculator(self.params)
        self.tension_updater = TensionUpdater(self.params)
        self.pigment_diffuser = PigmentDiffuser(self.params)
        self.pressure_calculator = ResiduePressureCalculator(self.params)
        
        # Performance tracking
        self.last_update_time = 0.0
        self.update_count = 0
    
    def update_all_fields(self, nodes: Dict[str, SemanticNode],
                         edges: Dict[str, SemanticEdge],
                         residue_data: Dict[str, Dict[str, float]]) -> Dict[str, Any]:
        """Run complete field equation update cycle"""
        start_time = time.time()
        
        results = {
            'coherence_updates': {},
            'tension_updates': {},
            'pigment_updates': {},
            'pressure_updates': {},
            'average_coherence': 0.0,
            'average_tension': 0.0,
            'update_time_ms': 0.0
        }
        
        try:
            # 1. Update local coherence for all nodes
            coherence_sum = 0.0
            for node_id, node in nodes.items():
                all_nodes = list(nodes.values())
                neighbors = self.coherence_calculator.find_neighbors(node, all_nodes)
                
                new_coherence = self.coherence_calculator.calculate_coherence(
                    node, neighbors, edges
                )
                
                node.local_coherence = new_coherence
                results['coherence_updates'][node_id] = new_coherence
                coherence_sum += new_coherence
            
            results['average_coherence'] = coherence_sum / len(nodes) if nodes else 0.0
            
            # 2. Update edge tensions
            tension_updates = self.tension_updater.batch_update_tensions(edges, nodes)
            results['tension_updates'] = tension_updates
            
            if tension_updates:
                results['average_tension'] = np.mean(list(tension_updates.values()))
            
            # 3. Apply pigment diffusion
            pigment_updates = self.pigment_diffuser.batch_diffuse_pigments(
                nodes, edges, self.coherence_calculator
            )
            results['pigment_updates'] = {k: v.tolist() for k, v in pigment_updates.items()}
            
            # 4. Calculate residue pressures
            pressure_updates = self.pressure_calculator.batch_calculate_pressure(
                nodes, residue_data, self.coherence_calculator
            )
            results['pressure_updates'] = pressure_updates
            
            # Update performance tracking
            self.update_count += 1
            self.last_update_time = time.time() - start_time
            results['update_time_ms'] = self.last_update_time * 1000
            
            logger.debug(f"Field equations updated in {results['update_time_ms']:.2f}ms")
            
        except Exception as e:
            logger.error(f"Error in field equation update: {e}")
            results['error'] = str(e)
        
        return results
    
    def get_field_statistics(self, nodes: Dict[str, SemanticNode],
                           edges: Dict[str, SemanticEdge]) -> Dict[str, Any]:
        """Get statistical summary of field state"""
        if not nodes:
            return {'error': 'No nodes in topology'}
        
        # Coherence statistics
        coherences = [node.local_coherence for node in nodes.values()]
        coherence_stats = {
            'mean': np.mean(coherences),
            'std': np.std(coherences),
            'min': np.min(coherences),
            'max': np.max(coherences)
        }
        
        # Tension statistics
        if edges:
            tensions = [edge.tension for edge in edges.values()]
            tension_stats = {
                'mean': np.mean(tensions),
                'std': np.std(tensions),
                'min': np.min(tensions),
                'max': np.max(tensions)
            }
        else:
            tension_stats = {'mean': 0, 'std': 0, 'min': 0, 'max': 0}
        
        # Pigment statistics
        pigments = np.array([node.tint for node in nodes.values()])
        pigment_stats = {
            'mean_intensity': np.mean([np.linalg.norm(p) for p in pigments]),
            'entropy': -np.sum(pigments * np.log(pigments + 1e-8)) / len(pigments)
        }
        
        # Pressure statistics
        pressures = [node.residue_pressure for node in nodes.values()]
        pressure_stats = {
            'mean': np.mean(pressures),
            'std': np.std(pressures),
            'min': np.min(pressures),
            'max': np.max(pressures)
        }
        
        return {
            'coherence': coherence_stats,
            'tension': tension_stats,
            'pigment': pigment_stats,
            'pressure': pressure_stats,
            'update_count': self.update_count,
            'last_update_time_ms': self.last_update_time * 1000
        }
