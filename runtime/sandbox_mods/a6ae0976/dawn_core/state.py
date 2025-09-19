"""
DAWN Core State Management
=========================

Legacy compatibility state management for dawn_core path structure.
This provides the same functionality as dawn.core.foundation.state but under
the old dawn_core module structure for backward compatibility.

This module maintains consciousness state for DAWN's recursive self-modification
system and provides all necessary state management functions.
"""

from dataclasses import dataclass, asdict
from time import time

@dataclass
class ConsciousnessState:
    """
    Core consciousness state representation for DAWN system.
    
    This class tracks all aspects of DAWN's consciousness including unity,
    awareness, momentum, and various derived metrics used throughout the system.
    """
    unity: float = 0.0
    awareness: float = 0.0
    momentum: float = 0.0
    level: str = "fragmented"  # "fragmented"|"coherent"|"meta_aware"|"transcendent"
    ticks: int = 0
    peak_unity: float = 0.0
    updated_at: float = 0.0
    
    # Advanced consciousness metrics
    coherence: float = 0.0
    integration_quality: float = 0.0
    stability_coherence: float = 0.0
    visual_coherence: float = 0.0
    artistic_coherence: float = 0.0
    meta_cognitive_activity: float = 0.0
    cycle_count: int = 0
    growth_rate: float = 0.0
    seed: int = 0
    session_id: str = ""
    demo_name: str = ""
    sync_status: str = "initializing"  # "initializing"|"syncing"|"synchronized"|"fragmented"
    
    # SCUP (System Consciousness Unity Protocol) metrics
    entropy_drift: float = 0.0    # Measure of consciousness entropy changes
    pressure_value: float = 0.0   # Consciousness pressure/tension
    scup_coherence: float = 0.0   # SCUP-specific coherence measure
    
    def to_dict(self):
        """Convert state to dictionary for serialization."""
        return asdict(self)

# Global state instance
STATE = ConsciousnessState()

def clamp(x, lo=0.0, hi=1.0):
    """Clamp value between lo and hi bounds."""
    return max(lo, min(hi, x))

def label_for(u, a):
    """
    Determine consciousness level label based on unity and awareness.
    
    Args:
        u: Unity value (0.0-1.0)
        a: Awareness value (0.0-1.0)
        
    Returns:
        String label for consciousness level
    """
    if u >= .90 and a >= .90: return "transcendent"
    if u >= .80 and a >= .80: return "meta_aware"
    if u >= .60 and a >= .60: return "coherent"
    return "fragmented"

def set_state(**kw):
    """
    Update consciousness state with provided keyword arguments.
    
    This function updates the global state and recalculates derived metrics
    automatically to maintain consistency.
    """
    for k, v in kw.items():
        setattr(STATE, k, v)
    
    # Update derived metrics
    STATE.peak_unity = max(STATE.peak_unity, STATE.unity)
    STATE.level = label_for(STATE.unity, STATE.awareness)
    
    # Calculate integration_quality as combination of unity and awareness
    STATE.integration_quality = (STATE.unity + STATE.awareness) / 2.0
    
    # Calculate stability_coherence based on coherence and integration
    STATE.stability_coherence = (STATE.coherence + STATE.integration_quality) / 2.0
    
    # Calculate visual and artistic coherence based on integration quality
    STATE.visual_coherence = STATE.integration_quality * 0.9  # Slightly lower than integration
    STATE.artistic_coherence = STATE.integration_quality * 0.8  # Even lower for artistic expression
    
    # Calculate meta-cognitive activity based on awareness level
    STATE.meta_cognitive_activity = STATE.awareness * 0.95
    
    # Calculate growth potential based on consciousness
    STATE.growth_rate = (STATE.unity + STATE.awareness) * 0.1
    
    # Update timestamp
    STATE.updated_at = time()

def get_state():
    """Get the current consciousness state."""
    return STATE

# Legacy compatibility functions for delta updates
def update_unity_delta(delta, reason=""):
    """
    Update unity by a delta amount with optional reason.
    
    Args:
        delta: Amount to change unity by
        reason: Optional reason for the change (for logging)
    """
    new_unity = clamp(STATE.unity + delta)
    set_state(unity=new_unity)

def update_awareness_delta(delta, reason=""):
    """
    Update awareness by a delta amount with optional reason.
    
    Args:
        delta: Amount to change awareness by
        reason: Optional reason for the change (for logging)
    """
    new_awareness = clamp(STATE.awareness + delta)
    set_state(awareness=new_awareness)

def reset_state():
    """Reset the consciousness state to default values."""
    global STATE
    STATE = ConsciousnessState()

# Helper functions for consciousness level checking
def is_transcendent():
    """Check if current state is transcendent."""
    return get_state().level == "transcendent"

def is_meta_aware():
    """Check if current state is meta-aware or higher."""
    level = get_state().level
    return level in ["meta_aware", "transcendent"]

def is_coherent():
    """Check if current state is coherent or higher."""
    level = get_state().level
    return level in ["coherent", "meta_aware", "transcendent"]

def get_state_summary():
    """Get a formatted summary of current state."""
    state = get_state()
    return f"Unity: {state.unity:.3f} | Awareness: {state.awareness:.3f} | Level: {state.level} | Ticks: {state.ticks}"

# Additional helper functions for compatibility
def evolve_consciousness(ticks=1):
    """
    Evolve consciousness by specified number of ticks.
    
    Args:
        ticks: Number of ticks to evolve by
    """
    current_ticks = STATE.ticks + ticks
    set_state(ticks=current_ticks)

def is_ready_for_level(target_level):
    """
    Check if consciousness is ready for a target level.
    
    Args:
        target_level: Target consciousness level to check
        
    Returns:
        Boolean indicating readiness
    """
    current_level = get_state().level
    level_hierarchy = ["fragmented", "coherent", "meta_aware", "transcendent"]
    
    try:
        current_idx = level_hierarchy.index(current_level)
        target_idx = level_hierarchy.index(target_level)
        return current_idx >= target_idx
    except ValueError:
        return False

def set_session_info(session_id, demo_name=""):
    """
    Set session information for tracking.
    
    Args:
        session_id: Unique session identifier
        demo_name: Optional demo name
    """
    set_state(session_id=session_id, demo_name=demo_name)

def label_from_metrics(unity, awareness):
    """
    Get consciousness level label from specific metrics.
    
    Args:
        unity: Unity value
        awareness: Awareness value
        
    Returns:
        Consciousness level label
    """
    return label_for(unity, awareness)
