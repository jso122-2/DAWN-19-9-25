#!/usr/bin/env python3
"""
DAWN Determinism Control
========================

Seed management for deterministic behavior across demos.
"""

import random
import os
import numpy as np
from dawn_core.state import set_state

def apply_seed(seed: int = None, preserve_state: bool = False) -> int:
    """Apply seed for deterministic behavior"""
    if seed is None:
        seed = int(os.environ.get("DAWN_SEED", "42"))
    
    # Set seeds for all random sources
    random.seed(seed)
    np.random.seed(seed)
    
    # Update consciousness state with seed (preserving other values if requested)
    if preserve_state:
        from dawn_core.state import get_state
        current = get_state()
        set_state(seed=seed, unity=current.unity, awareness=current.awareness, 
                 momentum=current.momentum, level=current.level,
                 integration_quality=current.integration_quality)
    else:
        set_state(seed=seed)
    
    return seed

def get_deterministic_flag() -> bool:
    """Check if deterministic mode is enabled"""
    return os.environ.get("DAWN_DETERMINISTIC", "true").lower() in ("true", "1", "yes")

def set_deterministic(enabled: bool = True) -> None:
    """Enable/disable deterministic mode"""
    os.environ["DAWN_DETERMINISTIC"] = "true" if enabled else "false"

if __name__ == "__main__":
    print(f"Applied seed: {apply_seed()}")
    print(f"Deterministic mode: {get_deterministic_flag()}")
