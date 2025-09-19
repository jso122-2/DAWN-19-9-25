#!/usr/bin/env python3
"""
DAWN Stable State Detection - Core Classes
==========================================
"""

import time
import logging
from datetime import datetime
from typing import Dict, List, Optional, Any
from dataclasses import dataclass, field
from enum import Enum

logger = logging.getLogger(__name__)

class StabilityLevel(Enum):
    """Stability classification levels."""
    CRITICAL = 0    # System failure imminent
    UNSTABLE = 1    # Significant degradation  
    DEGRADED = 2    # Minor issues detected
    STABLE = 3      # Normal operation
    OPTIMAL = 4     # Peak performance

class RecoveryAction(Enum):
    """Types of recovery actions available."""
    MONITOR = "monitor"
    SOFT_RESET = "soft_reset"
    AUTO_ROLLBACK = "auto_rollback"
    EMERGENCY_STABILIZE = "emergency_stabilize"
    GRACEFUL_DEGRADATION = "graceful_degradation"
    SELECTIVE_RESTART = "selective_restart"

@dataclass
class StabilityMetrics:
    """Complete stability assessment of DAWN system."""
    timestamp: datetime
    entropy_stability: float
    memory_coherence: float
    sigil_cascade_health: float
    recursive_depth_safe: float
    symbolic_organ_synergy: float
    unified_field_coherence: float
    overall_stability: float
    stability_level: StabilityLevel
    failing_systems: List[str] = field(default_factory=list)
    warning_systems: List[str] = field(default_factory=list)
    degradation_rate: float = 0.0
    prediction_horizon: float = 0.0

@dataclass
class StableSnapshot:
    """Snapshot of a known-good system state."""
    snapshot_id: str
    timestamp: datetime
    stability_score: float
    system_state: Dict[str, Any]
    module_states: Dict[str, Any]
    configuration: Dict[str, Any]
    state_hash: str
    description: str = ""

@dataclass
class StabilityEvent:
    """Event record for stability monitoring."""
    event_id: str
    timestamp: datetime
    event_type: str
    stability_score: float
    failing_systems: List[str]
    degradation_rate: float
    recovery_action: RecoveryAction
    rollback_target: Optional[str] = None
    details: Dict[str, Any] = field(default_factory=dict)
    success: bool = True

def calculate_stability_score(module_health: Dict[str, Any], 
                            safe_limits: Dict[str, float],
                            stability_weights: Dict[str, float]) -> StabilityMetrics:
    """
    Calculate comprehensive stability score for the DAWN system.
    
    Args:
        module_health: Health data from all modules
        safe_limits: Safe operational limits
        stability_weights: Component weights for overall score
        
    Returns:
        Complete stability metrics
    """
    current_time = datetime.now()
    
    # Initialize metrics
    entropy_stability = 0.0
    memory_coherence = 0.0  
    sigil_cascade_health = 0.0
    recursive_depth_safe = 0.0
    symbolic_organ_synergy = 0.0
    unified_field_coherence = 0.0
    
    failing_systems = []
    warning_systems = []
    
    # Calculate entropy stability
    entropy_values = []
    for module_name, health in module_health.items():
        if 'entropy' in health:
            entropy_values.append(health['entropy'])
        elif 'current_entropy' in health:
            entropy_values.append(health['current_entropy'])
            
    if entropy_values:
        import numpy as np
        entropy_variance = np.var(entropy_values) if len(entropy_values) > 1 else 0.0
        entropy_stability = max(0.0, 1.0 - (entropy_variance / safe_limits['max_entropy_variance']))
        
        if entropy_variance > safe_limits['max_entropy_variance']:
            warning_systems.append('entropy_regulation')
    else:
        entropy_stability = 0.5  # Unknown state
        
    # Calculate memory coherence
    total_reblooms = 0
    successful_reblooms = 0
    
    for module_name, health in module_health.items():
        if 'total_reblooms' in health and 'successful_reblooms' in health:
            total_reblooms += health['total_reblooms'] 
            successful_reblooms += health['successful_reblooms']
        elif 'total_reblooms' in health:
            total_reblooms += health['total_reblooms']
            successful_reblooms += health['total_reblooms']  # Assume success
            
    if total_reblooms > 0:
        memory_coherence = successful_reblooms / total_reblooms
        if memory_coherence < safe_limits['min_memory_coherence']:
            failing_systems.append('memory_coherence')
    else:
        memory_coherence = 1.0  # No reblooms means perfect coherence
        
    # Calculate sigil cascade health
    cascade_depths = []
    infinite_loops_detected = False
    
    for module_name, health in module_health.items():
        if 'cascade_depth' in health:
            cascade_depths.append(health['cascade_depth'])
            if health['cascade_depth'] > safe_limits['max_cascade_depth']:
                failing_systems.append(f'sigil_cascade_{module_name}')
                
        if 'infinite_loops' in health and health['infinite_loops']:
            infinite_loops_detected = True
            failing_systems.append(f'infinite_loops_{module_name}')
            
    if cascade_depths:
        max_cascade = max(cascade_depths)
        sigil_cascade_health = max(0.0, 1.0 - (max_cascade / (safe_limits['max_cascade_depth'] * 2)))
    else:
        sigil_cascade_health = 1.0
        
    if infinite_loops_detected:
        sigil_cascade_health *= 0.5
        
    # Calculate recursive depth safety
    recursive_depths = []
    
    for module_name, health in module_health.items():
        if 'current_depth' in health:
            recursive_depths.append(health['current_depth'])
        elif 'recursion_depth' in health:
            recursive_depths.append(health['recursion_depth'])
            
    if recursive_depths:
        max_depth = max(recursive_depths)
        recursive_depth_safe = max(0.0, 1.0 - (max_depth / safe_limits['max_recursive_depth']))
        
        if max_depth > safe_limits['max_recursive_depth']:
            failing_systems.append('recursive_depth')
    else:
        recursive_depth_safe = 1.0
        
    # Calculate symbolic organ synergy
    organ_synergies = []
    
    for module_name, health in module_health.items():
        if 'organ_synergy' in health:
            organ_synergies.append(health['organ_synergy'])
        elif 'embodied_coherence' in health:
            organ_synergies.append(health['embodied_coherence'])
            
    if organ_synergies:
        symbolic_organ_synergy = sum(organ_synergies) / len(organ_synergies)
        
        if symbolic_organ_synergy < safe_limits['min_organ_synergy']:
            warning_systems.append('symbolic_organs')
    else:
        symbolic_organ_synergy = 0.8  # Default reasonable value
        
    # Calculate unified field coherence
    field_coherences = []
    
    for module_name, health in module_health.items():
        if 'field_coherence' in health:
            field_coherences.append(health['field_coherence'])
        elif 'communion_active' in health and health['communion_active']:
            field_coherences.append(0.9)  # High coherence if communion active
            
    if field_coherences:
        unified_field_coherence = sum(field_coherences) / len(field_coherences)
        
        if unified_field_coherence < safe_limits['min_field_coherence']:
            warning_systems.append('unified_field')
    else:
        unified_field_coherence = 0.7  # Default
        
    # Calculate overall stability score
    component_scores = {
        'entropy_stability': entropy_stability,
        'memory_coherence': memory_coherence,
        'sigil_cascade_health': sigil_cascade_health,
        'recursive_depth_safe': recursive_depth_safe,
        'symbolic_organ_synergy': symbolic_organ_synergy,
        'unified_field_coherence': unified_field_coherence
    }
    
    overall_stability = sum(
        score * stability_weights[component]
        for component, score in component_scores.items()
    )
    
    # Determine stability level
    if overall_stability >= 0.9:
        stability_level = StabilityLevel.OPTIMAL
    elif overall_stability >= 0.7:
        stability_level = StabilityLevel.STABLE
    elif overall_stability >= 0.5:
        stability_level = StabilityLevel.DEGRADED
    elif overall_stability >= 0.3:
        stability_level = StabilityLevel.UNSTABLE
    else:
        stability_level = StabilityLevel.CRITICAL
        
    # Create stability metrics
    metrics = StabilityMetrics(
        timestamp=current_time,
        entropy_stability=entropy_stability,
        memory_coherence=memory_coherence,
        sigil_cascade_health=sigil_cascade_health,
        recursive_depth_safe=recursive_depth_safe,
        symbolic_organ_synergy=symbolic_organ_synergy,
        unified_field_coherence=unified_field_coherence,
        overall_stability=overall_stability,
        stability_level=stability_level,
        failing_systems=failing_systems,
        warning_systems=warning_systems
    )
    
    return metrics
