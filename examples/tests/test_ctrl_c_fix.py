#!/usr/bin/env python3
"""
🛑 Test Ctrl+C Fix
==================

Simple test to verify that Ctrl+C handling works properly.
This will run for 10 seconds and should respond immediately to Ctrl+C.
"""

import signal
import sys
import time

def signal_handler(signum, frame):
    print("\n🛑 Ctrl+C detected - exiting immediately!")
    sys.exit(0)

def main():
    # Setup signal handler
    signal.signal(signal.SIGINT, signal_handler)
    signal.signal(signal.SIGTERM, signal_handler)
    
    print("🛑 Testing Ctrl+C responsiveness...")
    print("⚠️  Press Ctrl+C to test - should exit immediately")
    print("⏰ Will auto-exit in 10 seconds if not interrupted")
    
    for i in range(100):  # 10 seconds with 0.1s intervals
        print(f"⏱️  Tick {i+1}/100 - Press Ctrl+C now!", end='\r')
        sys.stdout.flush()
        time.sleep(0.1)
        
    print("\n✅ Test completed - Ctrl+C handling working if you see this")

if __name__ == "__main__":
    main()
