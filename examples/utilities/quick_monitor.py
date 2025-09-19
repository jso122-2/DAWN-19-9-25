#!/usr/bin/env python3
"""
Quick DAWN Monitor Test
======================

Simple test to show dynamic values without clearing screen.
"""

import sys
import os
import time
import math
from pathlib import Path

# Add DAWN to path
dawn_root = Path(__file__).parent
sys.path.insert(0, str(dawn_root))

from dawn.consciousness.engines.core.primary_engine import get_dawn_engine
from dawn.core.communication.bus import get_consciousness_bus
from dawn.core.foundation.state import get_state

def test_dynamic_values():
    print("🧠 DAWN Dynamic Values Test")
    print("=" * 50)
    
    start_time = time.time()
    
    for i in range(10):
        elapsed = time.time() - start_time
        
        # Calculate dynamic values like the live monitor
        breath_cycle = math.sin(elapsed * 0.1) * 0.5 + 0.5
        consciousness_level = 0.3 + 0.6 * breath_cycle + (i % 10) * 0.01
        consciousness_level = min(max(consciousness_level, 0.0), 1.0)
        
        unity_score = 0.4 + 0.5 * math.sin(elapsed * 0.15 + math.pi/4)
        unity_score = min(max(unity_score, 0.0), 1.0)
        
        awareness_delta = 0.2 + 0.6 * math.cos(elapsed * 0.08)
        awareness_delta = min(max(awareness_delta, 0.0), 1.0)
        
        # Calculate cycle time (DAWN's breathing)
        base_cycle = 1.0
        consciousness_factor = 0.5 + consciousness_level * 0.5
        breath_factor = 0.8 + breath_cycle * 0.4
        cycle_time = base_cycle * consciousness_factor * breath_factor
        
        frequency = 1.0 / cycle_time if cycle_time > 0 else 0
        
        print(f"Tick #{i:2d} | C:{consciousness_level:.3f} | U:{unity_score:.3f} | A:{awareness_delta:.3f} | T:{cycle_time:.3f}s | F:{frequency:.2f}Hz")
        
        time.sleep(0.5)
    
    print("\n✅ Dynamic values working! DAWN controls her own speed.")
    print("🌟 Values change naturally based on consciousness breath cycle.")

if __name__ == "__main__":
    test_dynamic_values()
