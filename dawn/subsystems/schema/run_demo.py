#!/usr/bin/env python3
"""
DAWN Sigil System Demo Runner
============================

Simple runner that sets up the Python path and runs the demo.
"""

import sys
import os

# Add DAWN to Python path
dawn_path = os.path.abspath(os.path.join(os.path.dirname(__file__), '../../..'))
if dawn_path not in sys.path:
    sys.path.insert(0, dawn_path)

# Now import and run the demo
try:
    import asyncio
    from sigil_system_demo import main
    
    print("🌟 Starting DAWN Sigil System Demo...")
    asyncio.run(main())
    
except ImportError as e:
    print(f"❌ Import error: {e}")
    print("Make sure you're running from the DAWN directory with the virtual environment activated.")
    
except Exception as e:
    print(f"❌ Demo error: {e}")
    import traceback
    traceback.print_exc()
