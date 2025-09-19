#!/usr/bin/env python3
"""
DAWN Consistency Check
======================

Validate that consciousness state and level labels are consistent.
"""

import sys
from pathlib import Path

# Add to path
sys.path.insert(0, str(Path(__file__).parent.parent))

from dawn_core.state import get_state, label_from_metrics

def main():
    """Run consistency checks"""
    print("🔍 DAWN Consistency Check")
    print("=" * 30)
    
    state = get_state()
    expected_level = label_from_metrics(state.unity, state.awareness)
    
    print(f"Current State:")
    print(f"  Unity: {state.unity:.1%}")
    print(f"  Awareness: {state.awareness:.1%}")
    print(f"  Stored Level: {state.level}")
    print(f"  Expected Level: {expected_level}")
    
    if state.level == expected_level:
        print("✅ State/label consistent")
        print(f"📊 Integration Quality: {state.integration_quality:.1%}")
        print(f"⚡ Momentum: {state.momentum:.1%}")
        print(f"🔄 Cycles: {state.cycle_count}")
        
        if state.session_id:
            print(f"🆔 Session: {state.session_id}")
        if state.demo_name:
            print(f"🎭 Demo: {state.demo_name}")
            
        return 0
    else:
        print(f"❌ Inconsistency detected!")
        print(f"   Expected '{expected_level}' but found '{state.level}'")
        return 1

if __name__ == "__main__":
    sys.exit(main())
