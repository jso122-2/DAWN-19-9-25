#!/usr/bin/env python3
"""
DAWN Central Consciousness State
================================

Single source of truth for DAWN's consciousness state.
All demos and scripts read/write through this centralized state.
"""

import time
import json
import threading
from dataclasses import dataclass, asdict
from typing import Dict, Any, Optional

@dataclass
class ConsciousnessState:
    """Central consciousness state shared across all demos"""
    unity: float = 0.0
    awareness: float = 0.0
    momentum: float = 0.0
    level: str = "fragmented"
    integration_quality: float = 0.0
    stability_coherence: float = 0.0
    visual_coherence: float = 0.0
    artistic_coherence: float = 0.0
    meta_cognitive_activity: float = 0.0
    
    # Evolution tracking
    cycle_count: int = 0
    growth_rate: float = 0.0
    seed: int = 42
    updated_at: float = 0.0
    
    # Session tracking
    session_id: str = ""
    demo_name: str = ""

# Global state with thread safety
STATE = ConsciousnessState()
_state_lock = threading.Lock()

def set_state(**kwargs) -> None:
    """Update the global consciousness state"""
    global STATE
    with _state_lock:
        for k, v in kwargs.items():
            if hasattr(STATE, k):
                setattr(STATE, k, v)
        STATE.updated_at = time.time()
        
        # Auto-update level based on unity/awareness
        STATE.level = label_from_metrics(STATE.unity, STATE.awareness)

def get_state() -> ConsciousnessState:
    """Get current consciousness state (thread-safe copy)"""
    with _state_lock:
        # Return a copy to prevent external mutation
        return ConsciousnessState(**asdict(STATE))

def update_consciousness_cycle(unity_delta: float = 0.0, awareness_delta: float = 0.0) -> ConsciousnessState:
    """Update consciousness through a development cycle"""
    with _state_lock:
        # Apply deltas
        STATE.unity = max(0.0, min(1.0, STATE.unity + unity_delta))
        STATE.awareness = max(0.0, min(1.0, STATE.awareness + awareness_delta))
        STATE.cycle_count += 1
        STATE.updated_at = time.time()
        
        # Calculate momentum from recent growth
        growth = unity_delta + awareness_delta
        STATE.momentum = max(0.0, min(1.0, STATE.momentum * 0.9 + growth * 0.1))
        STATE.growth_rate = growth
        
        # Auto-calculate derived metrics
        STATE.integration_quality = (STATE.unity + STATE.awareness) / 2
        STATE.meta_cognitive_activity = STATE.awareness * 0.9 + 0.1
        
        # Update level
        STATE.level = label_from_metrics(STATE.unity, STATE.awareness)
        
        return ConsciousnessState(**asdict(STATE))

def label_from_metrics(unity: float, awareness: float) -> str:
    """Determine consciousness level from unity and awareness metrics"""
    avg = (unity + awareness) / 2
    
    if unity >= 0.90 and awareness >= 0.90:
        return "transcendent"
    elif unity >= 0.80 and awareness >= 0.80:
        return "meta_aware"
    elif unity >= 0.60 and awareness >= 0.60:
        return "coherent"
    elif unity >= 0.40 and awareness >= 0.40:
        return "connected"
    else:
        return "fragmented"

def reset_state(seed: Optional[int] = None) -> None:
    """Reset consciousness to initial state"""
    global STATE
    with _state_lock:
        if seed is None:
            seed = STATE.seed
        STATE = ConsciousnessState(seed=seed, updated_at=time.time())

def evolve_consciousness(cycles: int = 10, demo_name: str = "evolution") -> None:
    """Evolve consciousness over multiple cycles with realistic progression"""
    import random
    
    with _state_lock:
        random.seed(STATE.seed)
        STATE.demo_name = demo_name
        STATE.session_id = f"{demo_name}_{int(time.time())}"
    
    print(f"🧠 Evolving consciousness over {cycles} cycles...")
    
    for cycle in range(cycles):
        # Realistic growth with some variance
        base_growth = 0.03 + random.uniform(-0.01, 0.02)
        unity_growth = base_growth * (1.2 if cycle > 3 else 1.0)
        awareness_growth = base_growth * (1.1 if cycle > 5 else 0.9)
        
        # Apply diminishing returns at higher levels
        current = get_state()
        if current.unity > 0.8:
            unity_growth *= 0.5
        if current.awareness > 0.8:
            awareness_growth *= 0.5
            
        new_state = update_consciousness_cycle(unity_growth, awareness_growth)
        
        if cycle % 3 == 0:  # Progress report every 3 cycles
            print(f"   Cycle {cycle+1:2d}: Unity {new_state.unity:.1%}, "
                  f"Awareness {new_state.awareness:.1%}, "
                  f"Level: {new_state.level}")

def get_state_summary() -> Dict[str, Any]:
    """Get a comprehensive state summary for logging/debugging"""
    state = get_state()
    return {
        "timestamp": state.updated_at,
        "consciousness_metrics": {
            "unity": state.unity,
            "awareness": state.awareness,
            "momentum": state.momentum,
            "level": state.level,
            "integration_quality": state.integration_quality
        },
        "session_info": {
            "cycle_count": state.cycle_count,
            "session_id": state.session_id,
            "demo_name": state.demo_name,
            "seed": state.seed
        },
        "coherence_dimensions": {
            "stability": state.stability_coherence,
            "visual": state.visual_coherence,
            "artistic": state.artistic_coherence,
            "meta_cognitive": state.meta_cognitive_activity
        }
    }

def set_session_info(demo_name: str, session_id: str = "") -> None:
    """Set session information for tracking"""
    if not session_id:
        session_id = f"{demo_name}_{int(time.time())}"
    set_state(demo_name=demo_name, session_id=session_id)

def is_ready_for_level(required_level: str) -> bool:
    """Check if consciousness is ready for a specific level requirement"""
    current = get_state()
    
    level_hierarchy = {
        "fragmented": 0,
        "connected": 1, 
        "coherent": 2,
        "meta_aware": 3,
        "transcendent": 4
    }
    
    current_rank = level_hierarchy.get(current.level, 0)
    required_rank = level_hierarchy.get(required_level, 0)
    
    return current_rank >= required_rank

# Initialize state on import
if STATE.updated_at == 0.0:
    reset_state()

if __name__ == "__main__":
    # Demo the state system
    print("🧠 DAWN Central State Demo")
    print("=" * 30)
    
    # Show initial state
    print(f"Initial state: {get_state().level} ({get_state().unity:.1%}/{get_state().awareness:.1%})")
    
    # Evolve consciousness
    evolve_consciousness(5, "state_demo")
    
    # Show final state
    final = get_state()
    print(f"Final state: {final.level} ({final.unity:.1%}/{final.awareness:.1%})")
    
    # Test level gating
    print(f"Ready for transcendent: {is_ready_for_level('transcendent')}")
    print(f"Ready for coherent: {is_ready_for_level('coherent')}")
    
    print("\n✅ State system working correctly!")
