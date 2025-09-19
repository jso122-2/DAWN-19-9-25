#!/usr/bin/env python3
"""
Quick launcher for DAWN Tick State Reader
========================================

Simple wrapper for easy access to the tick state monitoring tool.

Usage:
    python tick_reader.py              # Live monitoring mode
    python tick_reader.py --snapshot   # Single snapshot
    python tick_reader.py --analyze    # Historical analysis
"""

import sys
from pathlib import Path

# Add DAWN to path
dawn_root = Path(__file__).parent
sys.path.insert(0, str(dawn_root))

# Import and run the tick reader
from dawn.tools.monitoring.tick_state_reader import main

if __name__ == "__main__":
    sys.exit(main())
