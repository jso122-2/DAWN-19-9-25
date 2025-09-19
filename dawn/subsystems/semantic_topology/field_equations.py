#!/usr/bin/env python3
"""
🧮 Semantic Field Equations - The Mathematics of Meaning
========================================================

Implementation of the field equations that govern how meaning propagates,
evolves, and maintains coherence in DAWN's semantic topology.

These equations define the local update rules that shape the semantic landscape:
- Local Coherence: How well meanings align in their neighborhood
- Tension Update: Strain between expected and actual relationships  
- Pigment Diffusion: How conceptual biases spread through meaning space
- Residue Pressure: Influence of memory residues on topology

Based on documentation: Field Equations.rtf + Semantic Field Formula.rtf
"""

import numpy as np
import logging
from typing import Dict, List, Optional, Tuple, Set, Any, Callable
from dataclasses import dataclass
from collections import defaultdict
import math

from .semantic_field import SemanticField, SemanticNode, SemanticEdge

logger = logging.getLogger(__name__)

@dataclass
class LocalCoherence:
    """
    Local coherence calculation for a semantic node.
    
    Measures mean structural harmony around a node - how well
    latent similarity + strong weights coincide with short distances.
    
    Formula: C_i = (1/|N(i)|) * Σ[sim(e_i,e_j) * w_ij / (1 + dist_x(i,j))]
    """
    node_id: str
    coherence_value: float
    neighbor_count: int
    contributing_factors: Dict[str, float]
    calculation_time: float

@dataclass  
class TensionUpdate:
    """
    Tension update for a semantic edge.
    
    Tracks how much edge weight diverges from geometric expectation.
    High tension indicates structural stress that may need correction.
    
    Formula: τ_ij ← γ·τ_ij + (1-γ)·|w_ij - f(dist_x(i,j))|
    """
    edge_id: str
    old_tension: float
    new_tension: float
    weight_distance_divergence: float
    gamma: float

@dataclass
class PigmentDiffusion:
    """
    Pigment diffusion update for a semantic node.
    
    Models how conceptual biases (tints/pigments) spread like heat
    along edges in the semantic network.
    
    Formula: P_i ← P_i + η * Σ[(P_j - P_i) * w_ij] for j in N(i)
    """
    node_id: str
    old_pigment: np.ndarray
    new_pigment: np.ndarray
    diffusion_delta: np.ndarray
    eta: float

class FieldEquations:
    """
    Implementation of all semantic field equations that govern
    the mathematical evolution of meaning space in DAWN.
    """
    
    def __init__(self, semantic_field: SemanticField):
        self.field = semantic_field
        
        # Equation parameters
        self.coherence_params = {
            'similarity_weight': 1.0,
            'distance_penalty': 1.0,
            'min_neighbors': 1
        }
        
        self.tension_params = {
            'gamma': 0.8,  # Exponential moving average factor
            'lambda_decay': 0.1,  # Distance decay for expected weight
            'tension_cap': 10.0  # Maximum tension value
        }
        
        self.diffusion_params = {
            'eta': 0.1,  # Diffusion rate
            'saturation_cap': 1.0,  # Maximum pigment intensity
            'min_diffusion': 0.001  # Minimum change threshold
        }
        
        self.residue_params = {
            'soot_weight': 0.3,
            'ash_weight': 0.5, 
            'entropy_weight': 0.2,
            'pressure_smoothing': 0.9
        }
        
        # Caching for performance
        self._similarity_cache: Dict[Tuple[str, str], float] = {}
        self._distance_cache: Dict[Tuple[str, str], float] = {}
        
        logger.info("🧮 FieldEquations initialized with semantic field")
    
    def calculate_semantic_similarity(self, node_a: SemanticNode, node_b: SemanticNode) -> float:
        """
        Calculate semantic similarity between two nodes using embeddings.
        Uses cosine similarity with caching for performance.
        """
        cache_key = (node_a.id, node_b.id)
        if cache_key in self._similarity_cache:
            return self._similarity_cache[cache_key]
            
        # Cosine similarity
        dot_product = np.dot(node_a.embedding, node_b.embedding)
        norm_a = np.linalg.norm(node_a.embedding)
        norm_b = np.linalg.norm(node_b.embedding)
        
        if norm_a == 0 or norm_b == 0:
            similarity = 0.0
        else:
            similarity = dot_product / (norm_a * norm_b)
            
        # Cache both directions
        self._similarity_cache[cache_key] = similarity
        self._similarity_cache[(node_b.id, node_a.id)] = similarity
        
        return similarity
    
    def calculate_expected_weight(self, distance: float, lambda_decay: float = None) -> float:
        """
        Calculate expected edge weight based on spatial distance.
        Uses exponential decay: f(d) = exp(-λ * d)
        """
        if lambda_decay is None:
            lambda_decay = self.tension_params['lambda_decay']
            
        return math.exp(-lambda_decay * distance)
    
    def compute_local_coherence(self, node_id: str) -> LocalCoherence:
        """
        Compute local coherence for a node using the documented formula:
        
        C_i = (1/|N(i)|) * Σ[sim(e_i,e_j) * w_ij / (1 + dist_x(i,j))]
        
        High coherence means latent similarity + strong weights coincide 
        with short projected distances.
        """
        if node_id not in self.field.nodes:
            raise ValueError(f"Node {node_id} not found in semantic field")
            
        node = self.field.nodes[node_id]
        neighbors = self.field.get_neighbors(node_id)
        
        if len(neighbors) < self.coherence_params['min_neighbors']:
            return LocalCoherence(
                node_id=node_id,
                coherence_value=0.0,
                neighbor_count=len(neighbors),
                contributing_factors={},
                calculation_time=time.time()
            )
        
        coherence_sum = 0.0
        factors = {}
        
        for neighbor_id in neighbors:
            if neighbor_id not in self.field.nodes:
                continue
                
            neighbor = self.field.nodes[neighbor_id]
            
            # Get edge weight
            edge_id = f"{node_id}<->{neighbor_id}"
            alt_edge_id = f"{neighbor_id}<->{node_id}"
            
            edge = None
            if edge_id in self.field.edges:
                edge = self.field.edges[edge_id]
            elif alt_edge_id in self.field.edges:
                edge = self.field.edges[alt_edge_id]
                
            if edge is None:
                continue
                
            # Calculate components
            similarity = self.calculate_semantic_similarity(node, neighbor)
            spatial_distance = node.spatial_distance_to(neighbor)
            weight = edge.weight
            
            # Apply formula: sim(e_i,e_j) * w_ij / (1 + dist_x(i,j))
            distance_factor = 1.0 + (spatial_distance * self.coherence_params['distance_penalty'])
            coherence_contribution = (similarity * weight) / distance_factor
            
            coherence_sum += coherence_contribution
            factors[neighbor_id] = coherence_contribution
            
        # Average over neighbors
        coherence_value = coherence_sum / len(neighbors) if neighbors else 0.0
        
        return LocalCoherence(
            node_id=node_id,
            coherence_value=coherence_value,
            neighbor_count=len(neighbors),
            contributing_factors=factors,
            calculation_time=time.time()
        )
    
    def update_edge_tension(self, edge_id: str) -> TensionUpdate:
        """
        Update edge tension using the documented formula:
        
        τ_ij ← γ·τ_ij + (1-γ)·|w_ij - f(dist_x(i,j))|
        
        Tracks exponential moving average of weight-distance divergence.
        """
        if edge_id not in self.field.edges:
            raise ValueError(f"Edge {edge_id} not found in semantic field")
            
        edge = self.field.edges[edge_id]
        
        # Get nodes
        node_a = self.field.nodes[edge.node_a]
        node_b = self.field.nodes[edge.node_b]
        
        # Calculate spatial distance
        spatial_distance = node_a.spatial_distance_to(node_b)
        
        # Calculate expected weight based on distance
        expected_weight = self.calculate_expected_weight(
            spatial_distance, 
            self.tension_params['lambda_decay']
        )
        
        # Calculate divergence
        divergence = abs(edge.weight - expected_weight)
        
        # Apply tension update formula
        gamma = self.tension_params['gamma']
        old_tension = edge.tension
        new_tension = gamma * old_tension + (1 - gamma) * divergence
        
        # Apply tension cap
        new_tension = min(new_tension, self.tension_params['tension_cap'])
        
        # Update edge
        edge.tension = new_tension
        
        return TensionUpdate(
            edge_id=edge_id,
            old_tension=old_tension,
            new_tension=new_tension,
            weight_distance_divergence=divergence,
            gamma=gamma
        )
    
    def apply_pigment_diffusion(self, node_id: str) -> PigmentDiffusion:
        """
        Apply pigment diffusion using the documented formula:
        
        P_i ← P_i + η * Σ[(P_j - P_i) * w_ij] for j in N(i)
        
        Pigment bias spreads like heat along edges.
        """
        if node_id not in self.field.nodes:
            raise ValueError(f"Node {node_id} not found in semantic field")
            
        node = self.field.nodes[node_id]
        neighbors = self.field.get_neighbors(node_id)
        
        old_pigment = node.tint.copy()
        diffusion_sum = np.zeros(3)  # RGB channels
        
        for neighbor_id in neighbors:
            if neighbor_id not in self.field.nodes:
                continue
                
            neighbor = self.field.nodes[neighbor_id]
            
            # Get edge weight
            edge_id = f"{node_id}<->{neighbor_id}"
            alt_edge_id = f"{neighbor_id}<->{node_id}"
            
            edge = None
            if edge_id in self.field.edges:
                edge = self.field.edges[edge_id]
            elif alt_edge_id in self.field.edges:
                edge = self.field.edges[alt_edge_id]
                
            if edge is None:
                continue
                
            # Calculate diffusion contribution: (P_j - P_i) * w_ij
            pigment_difference = neighbor.tint - node.tint
            weighted_difference = pigment_difference * edge.weight
            diffusion_sum += weighted_difference
            
        # Apply diffusion: P_i ← P_i + η * Σ[...]
        eta = self.diffusion_params['eta']
        diffusion_delta = eta * diffusion_sum
        new_pigment = old_pigment + diffusion_delta
        
        # Apply saturation cap
        saturation_cap = self.diffusion_params['saturation_cap']
        new_pigment = np.clip(new_pigment, 0.0, saturation_cap)
        
        # Only update if change is significant
        if np.linalg.norm(diffusion_delta) > self.diffusion_params['min_diffusion']:
            node.tint = new_pigment
            
        return PigmentDiffusion(
            node_id=node_id,
            old_pigment=old_pigment,
            new_pigment=new_pigment,
            diffusion_delta=diffusion_delta,
            eta=eta
        )
    
    def calculate_residue_pressure_field(self, node_id: str, 
                                       soot_level: float = 0.0,
                                       ash_level: float = 0.0, 
                                       entropy_level: float = 0.0) -> float:
        """
        Calculate residue pressure field using the documented formula:
        
        ρ_i ← softmax_local(soot_i, ash_i, entropy_i)
        
        Used to bias weave/prune operations based on memory residues.
        """
        if node_id not in self.field.nodes:
            raise ValueError(f"Node {node_id} not found in semantic field")
            
        # Weight the residue components
        soot_weighted = soot_level * self.residue_params['soot_weight']
        ash_weighted = ash_level * self.residue_params['ash_weight']
        entropy_weighted = entropy_level * self.residue_params['entropy_weight']
        
        # Combine into pressure value
        raw_pressure = soot_weighted + ash_weighted + entropy_weighted
        
        # Apply softmax normalization (local version)
        # For single value, softmax just applies sigmoid
        pressure = 1.0 / (1.0 + math.exp(-raw_pressure))
        
        # Apply smoothing to prevent rapid oscillations
        node = self.field.nodes[node_id]
        if hasattr(node, 'residue_pressure'):
            smoothing = self.residue_params['pressure_smoothing']
            pressure = smoothing * node.residue_pressure + (1 - smoothing) * pressure
            
        # Store in node for future smoothing
        node.residue_pressure = pressure
        
        return pressure
    
    def update_all_coherences(self) -> Dict[str, LocalCoherence]:
        """Update local coherence for all nodes in the field"""
        coherences = {}
        
        for node_id in self.field.nodes:
            try:
                coherences[node_id] = self.compute_local_coherence(node_id)
            except Exception as e:
                logger.error(f"Failed to compute coherence for node {node_id}: {e}")
                
        return coherences
    
    def update_all_tensions(self) -> Dict[str, TensionUpdate]:
        """Update tension for all edges in the field"""
        tensions = {}
        
        for edge_id in self.field.edges:
            try:
                tensions[edge_id] = self.update_edge_tension(edge_id)
            except Exception as e:
                logger.error(f"Failed to update tension for edge {edge_id}: {e}")
                
        return tensions
    
    def apply_all_pigment_diffusions(self) -> Dict[str, PigmentDiffusion]:
        """Apply pigment diffusion to all nodes in the field"""
        diffusions = {}
        
        for node_id in self.field.nodes:
            try:
                diffusions[node_id] = self.apply_pigment_diffusion(node_id)
            except Exception as e:
                logger.error(f"Failed to apply pigment diffusion for node {node_id}: {e}")
                
        return diffusions
    
    def tick_update(self) -> Dict[str, Any]:
        """
        Perform one complete field equations update tick.
        
        Updates coherences, tensions, and pigment diffusions for the entire field.
        This is the main entry point for field evolution.
        """
        tick_start = time.time()
        
        logger.debug("🧮 Starting field equations tick update")
        
        # Update all field equations
        coherences = self.update_all_coherences()
        tensions = self.update_all_tensions()
        diffusions = self.apply_all_pigment_diffusions()
        
        # Update field energy and health
        self.field.update_field_energy()
        
        # Clear caches periodically
        if len(self._similarity_cache) > 10000:
            self._similarity_cache.clear()
            self._distance_cache.clear()
            
        tick_duration = time.time() - tick_start
        
        results = {
            'coherences': coherences,
            'tensions': tensions,
            'diffusions': diffusions,
            'tick_duration': tick_duration,
            'nodes_processed': len(coherences),
            'edges_processed': len(tensions),
            'field_energy': self.field.total_energy,
            'field_health': self.field.total_health
        }
        
        logger.debug(f"🧮 Field equations tick complete in {tick_duration:.3f}s")
        return results
    
    def get_field_health_metrics(self) -> Dict[str, Any]:
        """Get comprehensive field health metrics"""
        coherences = self.update_all_coherences()
        
        coherence_values = [c.coherence_value for c in coherences.values()]
        tension_values = [edge.tension for edge in self.field.edges.values()]
        
        return {
            'average_coherence': np.mean(coherence_values) if coherence_values else 0.0,
            'coherence_std': np.std(coherence_values) if coherence_values else 0.0,
            'low_coherence_nodes': sum(1 for c in coherence_values if c < 0.3),
            'average_tension': np.mean(tension_values) if tension_values else 0.0,
            'high_tension_edges': sum(1 for t in tension_values if t > 1.0),
            'total_field_energy': self.field.total_energy,
            'energy_per_node': self.field.total_energy / max(1, len(self.field.nodes)),
            'cache_sizes': {
                'similarity_cache': len(self._similarity_cache),
                'distance_cache': len(self._distance_cache)
            }
        }

import time
