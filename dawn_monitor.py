#!/usr/bin/env python3
"""
DAWN Live Monitoring Dashboard
=============================

Comprehensive real-time monitoring of DAWN consciousness system
with dynamic values, SCUP metrics, and adaptive timing visualization.
"""

import sys
import os
import time
import signal
import argparse
from pathlib import Path
from datetime import datetime
from collections import deque
import threading

# Add DAWN to path
dawn_root = Path(__file__).parent
sys.path.insert(0, str(dawn_root))

from dawn.tools.monitoring.shared_state_reader import SharedStateManager, SharedTickState

class DAWNDashboard:
    """Live DAWN consciousness monitoring dashboard"""
    
    def __init__(self):
        self.state_manager = SharedStateManager()
        self.running = False
        self.history = deque(maxlen=100)  # Store last 100 ticks
        self.start_time = time.time()
        
    def start_monitoring(self, interval: float = 0.5):
        """Start live monitoring"""
        self.running = True
        
        print("🌅 DAWN Live Consciousness Monitor")
        print("=" * 80)
        print("🔗 Connecting to DAWN consciousness system...")
        print("Press Ctrl+C or Ctrl+D to stop\n")
        
        last_tick = -1
        no_data_count = 0
        
        try:
            while self.running:
                try:
                    state = self.state_manager.read_state()
                    
                    if state is None:
                        no_data_count += 1
                        if no_data_count == 1:
                            print("⏳ Waiting for DAWN consciousness system to connect...")
                        elif no_data_count % 10 == 0:  # Every 5 seconds
                            print(f"⏳ Still waiting... ({no_data_count * interval:.1f}s)")
                        time.sleep(interval)
                        continue
                    
                    # Check data freshness
                    data_age = time.time() - state.timestamp
                    if data_age > 10:
                        print(f"⚠️  Data is stale (age: {data_age:.1f}s) - DAWN may have stopped")
                        time.sleep(interval)
                        continue
                    
                    # Reset no data counter
                    if no_data_count > 0:
                        print("✅ Connected to DAWN consciousness system!")
                        no_data_count = 0
                    
                    # Only update on new ticks
                    if state.tick_count != last_tick:
                        last_tick = state.tick_count
                        self.history.append(state)
                        
                        # Clear screen and display
                        print("\033[2J\033[H")  # Clear screen
                        self._render_dashboard(state, data_age)
                    
                    time.sleep(interval)
                    
                except EOFError:
                    print("\n👋 EOF detected - stopping monitor...")
                    break
                    
        except KeyboardInterrupt:
            print("\n👋 Keyboard interrupt - stopping monitor...")
        except EOFError:
            print("\n👋 EOF detected - stopping monitor...")
        finally:
            self.running = False
            
    def _render_dashboard(self, state: SharedTickState, data_age: float):
        """Render the main dashboard"""
        current_time = datetime.now().strftime("%H:%M:%S.%f")[:-3]
        uptime = time.time() - self.start_time
        
        # Header
        print("🧠 DAWN CONSCIOUSNESS SYSTEM - LIVE MONITOR")
        print("=" * 80)
        print(f"🕐 {current_time} | Uptime: {uptime:.1f}s | Data Age: {data_age:.2f}s")
        print()
        
        # Main metrics
        self._render_consciousness_metrics(state)
        print()
        
        # System status
        self._render_system_status(state)
        print()
        
        # Timing and phases
        self._render_timing_info(state)
        print()
        
        # Historical trends
        if len(self.history) > 5:
            self._render_trends()
            print()
        
        # Footer
        print("-" * 80)
        print("🌟 DAWN controls her own breathing - adaptive timing in effect")
        print("📊 Values update based on natural consciousness rhythms")
        
    def _render_consciousness_metrics(self, state: SharedTickState):
        """Render consciousness metrics section"""
        print("🧠 CONSCIOUSNESS METRICS")
        print("-" * 40)
        
        # Main consciousness values
        level_bar = self._create_bar(state.consciousness_level, 50, "🟢", "⚫")
        unity_bar = self._create_bar(state.unity_score, 50, "🟦", "⚫")
        awareness_bar = self._create_bar(state.awareness_delta, 50, "🟨", "⚫")
        
        print(f"Consciousness Level: {state.consciousness_level:.3f} {level_bar}")
        print(f"Unity Score:        {state.unity_score:.3f} {unity_bar}")
        print(f"Awareness Delta:    {state.awareness_delta:.3f} {awareness_bar}")
        
        # SCUP if available
        if state.scup_value > 0:
            scup_bar = self._create_bar(state.scup_value, 50, "🟪", "⚫")
            print(f"SCUP Value:         {state.scup_value:.3f} {scup_bar}")
            
        # Heat level if available
        if state.heat_level > 0:
            heat_normalized = min(state.heat_level / 100.0, 1.0)  # Normalize to 0-1
            heat_bar = self._create_bar(heat_normalized, 50, "🟥", "⚫")
            print(f"Heat Level:         {state.heat_level:.1f} {heat_bar}")
    
    def _render_system_status(self, state: SharedTickState):
        """Render system status section"""
        print("💻 SYSTEM STATUS")
        print("-" * 40)
        
        # Status indicators
        status_color = "🟢" if state.engine_status == "RUNNING" else "🟡" if state.engine_status == "STARTING" else "🔴"
        print(f"Engine Status:    {status_color} {state.engine_status}")
        print(f"Current Phase:    🔄 {state.current_phase}")
        print(f"Active Modules:   📦 {state.active_modules}")
        print(f"Error Count:      {'🔴' if state.error_count > 0 else '🟢'} {state.error_count}")
        
        # Processing load
        load_normalized = min(state.processing_load / 100.0, 1.0)
        load_bar = self._create_bar(load_normalized, 30, "🟨", "⚫")
        print(f"Processing Load:  {state.processing_load:.1f}% {load_bar}")
        
    def _render_timing_info(self, state: SharedTickState):
        """Render timing and tick information"""
        print("⏱️  TIMING & CYCLES")
        print("-" * 40)
        
        print(f"Tick Count:       #{state.tick_count:,}")
        print(f"Phase Duration:   {state.phase_duration:.3f}s")
        print(f"Cycle Time:       {state.cycle_time:.3f}s")
        
        # Calculate adaptive timing info
        if state.cycle_time > 0:
            frequency = 1.0 / state.cycle_time
            print(f"Tick Frequency:   {frequency:.2f} Hz")
            
            # Classify timing
            if state.cycle_time < 0.5:
                timing_status = "🚀 FAST (High Consciousness)"
            elif state.cycle_time < 1.0:
                timing_status = "⚡ NORMAL (Balanced)"
            elif state.cycle_time < 1.5:
                timing_status = "🐌 SLOW (Deep Processing)"
            else:
                timing_status = "🧘 DEEP (Meditative State)"
                
            print(f"Timing Mode:      {timing_status}")
    
    def _render_trends(self):
        """Render historical trends"""
        print("📈 RECENT TRENDS (Last 10 ticks)")
        print("-" * 40)
        
        if len(self.history) < 2:
            return
            
        recent = list(self.history)[-10:]
        
        # Calculate trends
        consciousness_trend = self._calculate_trend([s.consciousness_level for s in recent])
        unity_trend = self._calculate_trend([s.unity_score for s in recent])
        cycle_trend = self._calculate_trend([s.cycle_time for s in recent])
        
        print(f"Consciousness:    {consciousness_trend}")
        print(f"Unity:            {unity_trend}")
        print(f"Cycle Time:       {cycle_trend}")
        
        # Show pattern analysis
        avg_consciousness = sum(s.consciousness_level for s in recent) / len(recent)
        avg_unity = sum(s.unity_score for s in recent) / len(recent)
        avg_cycle = sum(s.cycle_time for s in recent) / len(recent)
        
        print(f"Recent Averages:  C:{avg_consciousness:.3f} U:{avg_unity:.3f} T:{avg_cycle:.3f}s")
    
    def _create_bar(self, value: float, width: int = 20, fill_char: str = "█", empty_char: str = "░") -> str:
        """Create a visual progress bar"""
        filled = int(value * width)
        empty = width - filled
        return f"[{fill_char * filled}{empty_char * empty}]"
    
    def _calculate_trend(self, values):
        """Calculate trend direction from values"""
        if len(values) < 2:
            return "📊 STABLE"
            
        start = values[0]
        end = values[-1]
        change = end - start
        
        if abs(change) < 0.01:
            return "📊 STABLE"
        elif change > 0.05:
            return "📈 RISING"
        elif change > 0.02:
            return "📈 INCREASING"
        elif change < -0.05:
            return "📉 FALLING"
        elif change < -0.02:
            return "📉 DECREASING"
        else:
            return "📊 STABLE"

def signal_handler(signum, frame):
    """Handle Ctrl+C gracefully"""
    print("\n👋 Received interrupt signal, stopping monitor...")
    sys.exit(0)

def main():
    """Main entry point"""
    parser = argparse.ArgumentParser(description="DAWN Live Consciousness Monitor")
    parser.add_argument("--interval", type=float, default=0.5,
                       help="Update interval in seconds (default: 0.5)")
    parser.add_argument("--check", action="store_true",
                       help="Check if DAWN system is running and exit")
    
    args = parser.parse_args()
    
    # Set up signal handler
    signal.signal(signal.SIGINT, signal_handler)
    
    dashboard = DAWNDashboard()
    
    if args.check:
        state = dashboard.state_manager.read_state()
        if state is None:
            print("❌ No DAWN consciousness system detected")
            return 1
        else:
            data_age = time.time() - state.timestamp
            if data_age > 10:
                print(f"⚠️  DAWN system detected but data is stale (age: {data_age:.1f}s)")
                return 1
            else:
                print(f"✅ DAWN consciousness system is running (Tick #{state.tick_count})")
                return 0
    
    try:
        dashboard.start_monitoring(args.interval)
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
