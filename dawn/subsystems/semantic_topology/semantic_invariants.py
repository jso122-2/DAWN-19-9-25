#!/usr/bin/env python3
"""
🛡️ Semantic Invariants - Preserving Meaning During Transformations
==================================================================

Implementation of invariants that ensure semantic topology transformations
preserve the essential structure and meaning relationships in DAWN's consciousness.

These invariants act as guardrails to prevent topology operations from
breaking the coherence of meaning space or violating fundamental constraints.

Based on documentation: Invariants.rtf
"""

import numpy as np
import logging
import time
from typing import Dict, List, Optional, Tuple, Set, Any, Union
from dataclasses import dataclass, field
from enum import Enum
from collections import defaultdict

from .semantic_field import SemanticField, SemanticNode, SemanticEdge, LayerDepth
from .field_equations import FieldEquations

logger = logging.getLogger(__name__)

class InvariantType(Enum):
    """Types of semantic invariants"""
    TOPOLOGY_SCHEMA_CONSISTENCY = "topology_schema_consistency"
    NO_FREE_TEARS = "no_free_tears"
    LAYER_ORDERING = "layer_ordering"
    ENERGY_CONSERVATION = "energy_conservation"
    PIGMENT_INTEGRITY = "pigment_integrity"
    MOTION_BUDGET = "motion_budget"

class ViolationSeverity(Enum):
    """Severity levels for invariant violations"""
    INFO = "info"           # Informational, no action needed
    WARNING = "warning"     # Should be addressed but not critical
    ERROR = "error"         # Serious violation, needs correction
    CRITICAL = "critical"   # System integrity at risk

@dataclass
class InvariantViolation:
    """Record of an invariant violation"""
    invariant_type: InvariantType
    severity: ViolationSeverity
    description: str
    affected_nodes: List[str] = field(default_factory=list)
    affected_edges: List[str] = field(default_factory=list)
    metrics: Dict[str, float] = field(default_factory=dict)
    timestamp: float = field(default_factory=time.time)
    suggested_correction: Optional[str] = None

class SemanticInvariants:
    """
    Guardian system that monitors and enforces semantic invariants
    to preserve meaning integrity during topology transformations.
    """
    
    def __init__(self, semantic_field: SemanticField, field_equations: FieldEquations):
        self.field = semantic_field
        self.equations = field_equations
        
        # Invariant thresholds and parameters
        self.thresholds = {
            # Topology-Schema Consistency
            'max_distance_weight_ratio': 2.0,     # δ threshold
            'min_weight_for_consistency': 0.5,     # κ threshold  
            'consistency_tolerance_ticks': 10,     # N ticks before correction
            
            # No Free Tears
            'max_displacement_per_tick': 1.0,      # ε threshold
            'min_reliability_for_tear': 0.7,       # r* threshold
            'tear_displacement_threshold': 2.0,     # Major displacement
            
            # Layer Ordering
            'min_ash_for_lift': 0.7,
            'min_pigment_bias_for_lift': 0.5,
            'max_energy_for_sink': 0.3,
            'min_soot_for_sink': 0.7,
            
            # Energy Conservation  
            'energy_conservation_tolerance': 0.1,   # ±ε for diffusion
            'total_energy_drift_threshold': 0.05,  # 5% total energy drift
            
            # Pigment Integrity
            'pigment_saturation_cap': 1.0,
            'max_single_channel_dominance': 0.8,   # SHI threshold
            'pigment_balance_threshold': 0.3,       # Minimum balance across channels
            
            # Motion Budget
            'motion_budget_per_tick': 10.0,
            'max_node_displacement': 0.5
        }
        
        # Violation tracking
        self.violation_history: List[InvariantViolation] = []
        self.persistent_violations: Dict[str, InvariantViolation] = {}
        
        # Statistics
        self.stats = {
            'total_checks': 0,
            'violations_detected': 0,
            'violations_by_type': defaultdict(int),
            'violations_by_severity': defaultdict(int)
        }
        
        logger.info("🛡️ SemanticInvariants guardian initialized")
    
    def check_topology_schema_consistency(self) -> List[InvariantViolation]:
        """
        Invariant 1: Topology ↔ Schema Consistency
        
        Strong edges (high weight, high reliability) must correspond to short projected distances.
        Tolerance band: if d_x > δ while w_ij > κ, flag as incoherent.
        """
        violations = []
        
        for edge_id, edge in self.field.edges.items():
            if edge.weight < self.thresholds['min_weight_for_consistency']:
                continue  # Only check strong edges
                
            if edge.node_a not in self.field.nodes or edge.node_b not in self.field.nodes:
                continue
                
            node_a = self.field.nodes[edge.node_a]
            node_b = self.field.nodes[edge.node_b]
            
            spatial_distance = node_a.spatial_distance_to(node_b)
            distance_weight_ratio = spatial_distance / edge.weight
            
            if distance_weight_ratio > self.thresholds['max_distance_weight_ratio']:
                violation = InvariantViolation(
                    invariant_type=InvariantType.TOPOLOGY_SCHEMA_CONSISTENCY,
                    severity=ViolationSeverity.WARNING,
                    description=f"Strong edge {edge_id} spans large distance: ratio {distance_weight_ratio:.2f}",
                    affected_edges=[edge_id],
                    metrics={
                        'spatial_distance': spatial_distance,
                        'edge_weight': edge.weight,
                        'distance_weight_ratio': distance_weight_ratio,
                        'reliability': edge.reliability
                    },
                    suggested_correction="Consider Prune/Weave/Reproject operations"
                )
                violations.append(violation)
                
        return violations
    
    def check_no_free_tears(self, recent_displacements: Dict[str, float] = None) -> List[InvariantViolation]:
        """
        Invariant 2: No Free Tears
        
        Topology updates must not strand high-reliability edges across large distances.
        All large displacements require explicit operators and consume motion budget.
        """
        violations = []
        
        if recent_displacements is None:
            recent_displacements = {}
            
        for edge_id, edge in self.field.edges.items():
            if edge.reliability < self.thresholds['min_reliability_for_tear']:
                continue  # Only check high-reliability edges
                
            if edge.node_a not in self.field.nodes or edge.node_b not in self.field.nodes:
                continue
                
            node_a = self.field.nodes[edge.node_a]
            node_b = self.field.nodes[edge.node_b]
            
            # Check for recent large displacements
            displacement_a = recent_displacements.get(edge.node_a, 0.0)
            displacement_b = recent_displacements.get(edge.node_b, 0.0)
            max_displacement = max(displacement_a, displacement_b)
            
            if max_displacement > self.thresholds['tear_displacement_threshold']:
                current_distance = node_a.spatial_distance_to(node_b)
                
                violation = InvariantViolation(
                    invariant_type=InvariantType.NO_FREE_TEARS,
                    severity=ViolationSeverity.ERROR,
                    description=f"High-reliability edge {edge_id} torn by large displacement",
                    affected_edges=[edge_id],
                    affected_nodes=[edge.node_a, edge.node_b],
                    metrics={
                        'max_displacement': max_displacement,
                        'current_distance': current_distance,
                        'edge_reliability': edge.reliability,
                        'displacement_a': displacement_a,
                        'displacement_b': displacement_b
                    },
                    suggested_correction="Use explicit Weave/Reproject operators with motion budget"
                )
                violations.append(violation)
                
        return violations
    
    def check_layer_ordering(self) -> List[InvariantViolation]:
        """
        Invariant 3: Layer Ordering
        
        Vertical moves (Sink/Lift) must respect rebloom/decay rules.
        No bypassing: nodes may not jump multiple layers in one tick.
        """
        violations = []
        
        for node_id, node in self.field.nodes.items():
            # Check lift requirements
            if hasattr(node, 'recent_layer_change') and node.recent_layer_change == 'lift':
                ash_level = getattr(node, 'ash_level', 0.0)
                pigment_bias = np.linalg.norm(node.tint - 0.5)
                
                if (ash_level < self.thresholds['min_ash_for_lift'] or
                    pigment_bias < self.thresholds['min_pigment_bias_for_lift']):
                    violation = InvariantViolation(
                        invariant_type=InvariantType.LAYER_ORDERING,
                        severity=ViolationSeverity.ERROR,
                        description=f"Node {node_id} lifted without meeting requirements",
                        affected_nodes=[node_id],
                        metrics={
                            'ash_level': ash_level,
                            'pigment_bias': pigment_bias,
                            'current_layer': node.layer.value
                        },
                        suggested_correction="Verify Ash accumulation and pigment bias before lift"
                    )
                    violations.append(violation)
                    
            # Check sink requirements  
            if hasattr(node, 'recent_layer_change') and node.recent_layer_change == 'sink':
                soot_level = getattr(node, 'soot_level', 0.0)
                
                if (node.energy > self.thresholds['max_energy_for_sink'] or
                    soot_level < self.thresholds['min_soot_for_sink']):
                    violation = InvariantViolation(
                        invariant_type=InvariantType.LAYER_ORDERING,
                        severity=ViolationSeverity.ERROR,
                        description=f"Node {node_id} sunk without meeting requirements",
                        affected_nodes=[node_id],
                        metrics={
                            'energy_level': node.energy,
                            'soot_level': soot_level,
                            'current_layer': node.layer.value
                        },
                        suggested_correction="Verify low energy and high soot before sink"
                    )
                    violations.append(violation)
                    
        return violations
    
    def check_energy_conservation(self, previous_total_energy: float = None) -> List[InvariantViolation]:
        """
        Invariant 4: Energy Conservation
        
        Cluster Fuse and Fission redistribute energy and health; they cannot create or destroy it.
        Σ energy_before ≈ Σ energy_after (±ε for diffusion).
        """
        violations = []
        
        current_total_energy = self.field.total_energy
        
        if previous_total_energy is not None:
            energy_drift = abs(current_total_energy - previous_total_energy) / max(previous_total_energy, 0.001)
            
            if energy_drift > self.thresholds['total_energy_drift_threshold']:
                violation = InvariantViolation(
                    invariant_type=InvariantType.ENERGY_CONSERVATION,
                    severity=ViolationSeverity.WARNING,
                    description=f"Total field energy drift detected: {energy_drift:.1%}",
                    metrics={
                        'previous_energy': previous_total_energy,
                        'current_energy': current_total_energy,
                        'energy_drift': energy_drift,
                        'drift_threshold': self.thresholds['total_energy_drift_threshold']
                    },
                    suggested_correction="Review recent Fuse/Fission operations for energy leaks"
                )
                violations.append(violation)
                
        # Check individual node energy bounds
        for node_id, node in self.field.nodes.items():
            if node.energy < 0 or node.energy > 2.0:  # Allow some overflow for temporary states
                violation = InvariantViolation(
                    invariant_type=InvariantType.ENERGY_CONSERVATION,
                    severity=ViolationSeverity.ERROR,
                    description=f"Node {node_id} has invalid energy: {node.energy:.3f}",
                    affected_nodes=[node_id],
                    metrics={'node_energy': node.energy},
                    suggested_correction="Clamp energy to valid range [0, 1]"
                )
                violations.append(violation)
                
        return violations
    
    def check_pigment_integrity(self) -> List[InvariantViolation]:
        """
        Invariant 5: Pigment Integrity
        
        Pigment diffusion cannot exceed saturation caps.
        Global hue balance must remain within SHI thresholds (no total dominance).
        """
        violations = []
        
        # Check individual node pigment saturation
        for node_id, node in self.field.nodes.items():
            if np.any(node.tint > self.thresholds['pigment_saturation_cap']):
                violation = InvariantViolation(
                    invariant_type=InvariantType.PIGMENT_INTEGRITY,
                    severity=ViolationSeverity.WARNING,
                    description=f"Node {node_id} exceeds pigment saturation",
                    affected_nodes=[node_id],
                    metrics={
                        'tint_values': node.tint.tolist(),
                        'max_tint': float(np.max(node.tint)),
                        'saturation_cap': self.thresholds['pigment_saturation_cap']
                    },
                    suggested_correction="Apply saturation clamping"
                )
                violations.append(violation)
                
        # Check global hue balance
        if self.field.nodes:
            all_tints = np.array([node.tint for node in self.field.nodes.values()])
            global_tint_mean = np.mean(all_tints, axis=0)
            
            # Check for single channel dominance
            max_channel = np.max(global_tint_mean)
            if max_channel > self.thresholds['max_single_channel_dominance']:
                dominant_channel = ['R', 'G', 'B'][np.argmax(global_tint_mean)]
                
                violation = InvariantViolation(
                    invariant_type=InvariantType.PIGMENT_INTEGRITY,
                    severity=ViolationSeverity.WARNING,
                    description=f"Global pigment imbalance: {dominant_channel} channel dominance",
                    metrics={
                        'global_tint_mean': global_tint_mean.tolist(),
                        'dominant_channel': dominant_channel,
                        'dominance_level': float(max_channel)
                    },
                    suggested_correction="Deploy Bee/Medieval Bee tracers for pigment redistribution"
                )
                violations.append(violation)
                
        return violations
    
    def check_motion_budget(self, recent_displacements: Dict[str, float] = None) -> List[InvariantViolation]:
        """
        Invariant 6: Motion Budget
        
        Per tick, total displacement of nodes/edges is capped (Σ||Δx|| ≤ budget).
        Excess movement is queued into subsequent ticks to avoid discontinuities.
        """
        violations = []
        
        if recent_displacements is None:
            return violations
            
        total_displacement = sum(recent_displacements.values())
        
        if total_displacement > self.thresholds['motion_budget_per_tick']:
            violation = InvariantViolation(
                invariant_type=InvariantType.MOTION_BUDGET,
                severity=ViolationSeverity.WARNING,
                description=f"Motion budget exceeded: {total_displacement:.2f} > {self.thresholds['motion_budget_per_tick']}",
                affected_nodes=list(recent_displacements.keys()),
                metrics={
                    'total_displacement': total_displacement,
                    'motion_budget': self.thresholds['motion_budget_per_tick'],
                    'excess_motion': total_displacement - self.thresholds['motion_budget_per_tick']
                },
                suggested_correction="Queue excess movement to subsequent ticks"
            )
            violations.append(violation)
            
        # Check individual node displacements
        for node_id, displacement in recent_displacements.items():
            if displacement > self.thresholds['max_node_displacement']:
                violation = InvariantViolation(
                    invariant_type=InvariantType.MOTION_BUDGET,
                    severity=ViolationSeverity.ERROR,
                    description=f"Node {node_id} displacement too large: {displacement:.3f}",
                    affected_nodes=[node_id],
                    metrics={'node_displacement': displacement},
                    suggested_correction="Smooth displacement over multiple ticks"
                )
                violations.append(violation)
                
        return violations
    
    def check_all_invariants(self, 
                           previous_total_energy: float = None,
                           recent_displacements: Dict[str, float] = None) -> List[InvariantViolation]:
        """
        Comprehensive invariant check - runs all invariant validations.
        
        This is the main entry point for invariant checking during topology operations.
        """
        all_violations = []
        
        self.stats['total_checks'] += 1
        
        # Run all invariant checks
        invariant_checks = [
            self.check_topology_schema_consistency,
            lambda: self.check_no_free_tears(recent_displacements),
            self.check_layer_ordering,
            lambda: self.check_energy_conservation(previous_total_energy),
            self.check_pigment_integrity,
            lambda: self.check_motion_budget(recent_displacements)
        ]
        
        for check in invariant_checks:
            try:
                violations = check()
                all_violations.extend(violations)
            except Exception as e:
                logger.error(f"Invariant check failed: {e}")
                
        # Update statistics
        for violation in all_violations:
            self.stats['violations_detected'] += 1
            self.stats['violations_by_type'][violation.invariant_type.value] += 1
            self.stats['violations_by_severity'][violation.severity.value] += 1
            
        # Store violations
        self.violation_history.extend(all_violations)
        
        # Keep violation history manageable
        if len(self.violation_history) > 1000:
            self.violation_history = self.violation_history[-500:]
            
        return all_violations
    
    def get_violation_summary(self) -> Dict[str, Any]:
        """Get summary of recent violations and system health"""
        recent_violations = [v for v in self.violation_history if time.time() - v.timestamp < 300]  # Last 5 minutes
        
        severity_counts = defaultdict(int)
        type_counts = defaultdict(int)
        
        for violation in recent_violations:
            severity_counts[violation.severity.value] += 1
            type_counts[violation.invariant_type.value] += 1
            
        return {
            'total_recent_violations': len(recent_violations),
            'violations_by_severity': dict(severity_counts),
            'violations_by_type': dict(type_counts),
            'critical_violations': [v for v in recent_violations if v.severity == ViolationSeverity.CRITICAL],
            'system_health_score': self._calculate_health_score(recent_violations),
            'statistics': dict(self.stats),
            'persistent_violations': len(self.persistent_violations)
        }
    
    def _calculate_health_score(self, recent_violations: List[InvariantViolation]) -> float:
        """Calculate overall system health score based on violations"""
        if not recent_violations:
            return 1.0
            
        # Weight violations by severity
        severity_weights = {
            ViolationSeverity.INFO: 0.0,
            ViolationSeverity.WARNING: 0.1,
            ViolationSeverity.ERROR: 0.3,
            ViolationSeverity.CRITICAL: 0.8
        }
        
        total_penalty = sum(severity_weights.get(v.severity, 0.0) for v in recent_violations)
        max_possible_penalty = len(recent_violations) * 0.8  # All critical
        
        if max_possible_penalty == 0:
            return 1.0
            
        health_score = 1.0 - (total_penalty / max_possible_penalty)
        return max(0.0, health_score)
    
    def suggest_corrections(self, violations: List[InvariantViolation]) -> List[str]:
        """Generate actionable correction suggestions for violations"""
        corrections = []
        
        # Group by invariant type for consolidated suggestions
        by_type = defaultdict(list)
        for violation in violations:
            by_type[violation.invariant_type].append(violation)
            
        for invariant_type, type_violations in by_type.items():
            if invariant_type == InvariantType.TOPOLOGY_SCHEMA_CONSISTENCY:
                corrections.append(f"Run Prune operation on {len(type_violations)} incoherent edges")
                corrections.append("Consider Reproject to align positions with embeddings")
                
            elif invariant_type == InvariantType.NO_FREE_TEARS:
                corrections.append(f"Apply Weave operations to {len(type_violations)} torn edges")
                corrections.append("Increase motion budget for smoother repositioning")
                
            elif invariant_type == InvariantType.ENERGY_CONSERVATION:
                corrections.append("Audit recent Fuse/Fission operations for energy leaks")
                corrections.append("Clamp node energies to valid range [0, 1]")
                
            elif invariant_type == InvariantType.PIGMENT_INTEGRITY:
                corrections.append("Deploy Bee tracers for pigment redistribution")
                corrections.append("Apply saturation clamping to oversaturated nodes")
                
        return corrections
