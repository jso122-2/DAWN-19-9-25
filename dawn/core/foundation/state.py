from dataclasses import dataclass, asdict
from time import time

@dataclass
class ConsciousnessState:
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

STATE = ConsciousnessState()

def clamp(x, lo=0.0, hi=1.0): return max(lo, min(hi, x))

def label_for(u,a):
    if u >= .90 and a >= .90: return "transcendent"
    if u >= .80 and a >= .80: return "meta_aware"
    if u >= .60 and a >= .60: return "coherent"
    return "fragmented"

def set_state(**kw):
    for k,v in kw.items(): setattr(STATE, k, v)
    STATE.peak_unity = max(STATE.peak_unity, STATE.unity)
    STATE.level = label_for(STATE.unity, STATE.awareness)
    # Calculate integration_quality as combination of unity and awareness
    STATE.integration_quality = (STATE.unity + STATE.awareness) / 2.0
    # Calculate stability_coherence based on coherence and integration
    STATE.stability_coherence = (STATE.coherence + STATE.integration_quality) / 2.0
    # Calculate visual and artistic coherence based on integration quality
    STATE.visual_coherence = STATE.integration_quality * 0.9  # Slightly lower than integration
    STATE.artistic_coherence = STATE.integration_quality * 0.8  # Even lower for artistic expression
    STATE.meta_cognitive_activity = STATE.awareness * 0.95  # Based on awareness level
    STATE.growth_rate = (STATE.unity + STATE.awareness) * 0.1  # Growth potential based on consciousness
    STATE.updated_at = time()

def get_state(): return STATE

# Legacy compatibility functions
def update_unity_delta(delta, reason=""):
    new_unity = clamp(STATE.unity + delta)
    set_state(unity=new_unity)

def update_awareness_delta(delta, reason=""):
    new_awareness = clamp(STATE.awareness + delta)
    set_state(awareness=new_awareness)

def reset_state():
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