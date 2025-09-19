#!/usr/bin/env python3
"""
Shared State Reader for DAWN Tick Monitoring
============================================

Connects to actual running DAWN instances through shared memory/files
instead of creating new singleton instances.
"""

import sys
import os
import time
import json
import mmap
import struct
import signal
import argparse
from pathlib import Path
from datetime import datetime
from dataclasses import dataclass, asdict
from typing import Dict, Any, List, Optional
import threading

# Add DAWN to path
dawn_root = Path(__file__).parent.parent.parent.parent
sys.path.insert(0, str(dawn_root))

@dataclass
class SharedTickState:
    """Shared tick state structure"""
    timestamp: float
    tick_count: int
    current_phase: str
    phase_duration: float
    cycle_time: float
    consciousness_level: float
    unity_score: float
    awareness_delta: float
    processing_load: float
    active_modules: int
    error_count: int
    scup_value: float
    heat_level: float
    engine_status: str
    
    def to_dict(self) -> Dict[str, Any]:
        return asdict(self)
    
    @classmethod
    def from_dict(cls, data: Dict[str, Any]) -> 'SharedTickState':
        return cls(**data)

class SharedStateManager:
    """Manages shared state between DAWN processes"""
    
    def __init__(self, state_file: str = "/tmp/dawn_shared_state.json"):
        self.state_file = Path(state_file)
        self.lock_file = Path(f"{state_file}.lock")
        
    def write_state(self, state: SharedTickState) -> bool:
        """Write state to shared file with locking"""
        try:
            # Simple file-based locking
            if self.lock_file.exists():
                return False  # Another process is writing
                
            # Create lock
            self.lock_file.touch()
            
            # Write state
            with open(self.state_file, 'w') as f:
                json.dump(state.to_dict(), f)
            
            # Remove lock
            self.lock_file.unlink()
            return True
            
        except Exception as e:
            # Clean up lock on error
            if self.lock_file.exists():
                self.lock_file.unlink()
            return False
    
    def read_state(self) -> Optional[SharedTickState]:
        """Read state from shared file"""
        try:
            if not self.state_file.exists():
                return None
                
            with open(self.state_file, 'r') as f:
                data = json.load(f)
                
            return SharedTickState.from_dict(data)
            
        except Exception as e:
            return None
    
    def cleanup(self):
        """Clean up shared state files"""
        for file_path in [self.state_file, self.lock_file]:
            if file_path.exists():
                file_path.unlink()

class LiveTickReader:
    """Reads live tick data from shared state"""
    
    def __init__(self):
        self.state_manager = SharedStateManager()
        self.running = False
        
    def start_monitoring(self):
        """Start live monitoring"""
        self.running = True
        print("🔗 Connected to shared DAWN state")
        
    def stop_monitoring(self):
        """Stop monitoring"""
        self.running = False
        
    def get_live_state(self) -> Optional[SharedTickState]:
        """Get current live state"""
        return self.state_manager.read_state()
    
    def display_live(self, interval: float = 1.0):
        """Display live updates"""
        self.start_monitoring()
        
        try:
            print("🚀 DAWN Live Tick Monitor - Shared State Mode")
            print("=" * 80)
            print("Press Ctrl+C or Ctrl+D to stop\n")
            
            last_tick = -1
            no_data_count = 0
            
            while self.running:
                try:
                    state = self.get_live_state()
                    
                    if state is None:
                        no_data_count += 1
                        if no_data_count > 5:
                            print(f"⏳ No DAWN process detected - waiting for connection...")
                            no_data_count = 0
                        time.sleep(interval)
                        continue
                    
                    # Check if data is fresh
                    data_age = time.time() - state.timestamp
                    if data_age > 10:  # Data older than 10 seconds
                        print(f"⚠️  Stale data (age: {data_age:.1f}s) - DAWN process may have stopped")
                        time.sleep(interval)
                        continue
                    
                    # Only update display if tick changed
                    if state.tick_count != last_tick:
                        last_tick = state.tick_count
                        no_data_count = 0
                        
                        # Clear screen and display
                        print("\033[2J\033[H")  # Clear screen
                        self._display_state(state, data_age)
                    
                    time.sleep(interval)
                    
                except EOFError:
                    print("\n👋 EOF detected - stopping monitor...")
                    break
                    
        except KeyboardInterrupt:
            print("\n👋 Interrupt detected - stopping monitor...")
        except EOFError:
            print("\n👋 EOF detected - stopping monitor...")
        finally:
            self.stop_monitoring()
    
    def _display_state(self, state: SharedTickState, data_age: float):
        """Display current state"""
        # Get current time for display
        current_time = datetime.now().strftime("%H:%M:%S.%f")[:-3]
        
        print(f"🕐 {current_time} | Tick #{state.tick_count} | Age: {data_age:.1f}s")
        print("-" * 80)
        print(f"🔄 Current Phase: {state.current_phase}")
        print(f"⏱️  Phase Duration: {state.phase_duration:.3f}s")
        print(f"🔁 Cycle Time: {state.cycle_time:.3f}s")
        print()
        print("🧠 Consciousness Metrics:")
        print(f"   Level: {state.consciousness_level:.3f}")
        print(f"   Unity: {state.unity_score:.3f}")
        print(f"   Awareness Δ: +{state.awareness_delta:.3f}")
        if state.scup_value > 0:
            print(f"   SCUP: {state.scup_value:.3f}")
        print()
        print("💻 System Health:")
        print(f"   Processing Load: {state.processing_load:.1f}%")
        print(f"   Active Modules: {state.active_modules}")
        print(f"   Engine Status: {state.engine_status}")
        print(f"   Errors: {state.error_count}")
        if state.heat_level > 0:
            print(f"   Heat Level: {state.heat_level:.1f}")
        print()

def main():
    """Main entry point"""
    parser = argparse.ArgumentParser(description="DAWN Shared State Tick Reader")
    parser.add_argument("--interval", type=float, default=0.5,
                       help="Update interval for live mode")
    parser.add_argument("--cleanup", action="store_true",
                       help="Clean up shared state files and exit")
    
    args = parser.parse_args()
    
    if args.cleanup:
        manager = SharedStateManager()
        manager.cleanup()
        print("🧹 Shared state files cleaned up")
        return 0
    
    # Create reader
    reader = LiveTickReader()
    
    try:
        reader.display_live(args.interval)
    except KeyboardInterrupt:
        print("\n👋 Keyboard interrupt - exiting gracefully...")
        return 0
    except EOFError:
        print("\n👋 EOF detected - exiting gracefully...")
        return 0
    except Exception as e:
        print(f"❌ Error: {e}")
        return 1
    
    return 0

if __name__ == "__main__":
    sys.exit(main())
