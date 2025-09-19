#!/usr/bin/env python3
"""
DAWN Tick State Reader
=====================

Real-time monitoring tool for DAWN's tick orchestrator and consciousness processing cycles.
Provides detailed insights into tick phases, timing, and system state evolution.

Usage:
    python -m dawn.tools.monitoring.tick_state_reader [options]
    
Options:
    --mode MODE         Display mode: live, snapshot, analyze (default: live)
    --interval SECONDS  Update interval for live mode (default: 0.5)
    --history COUNT     Number of ticks to keep in history (default: 100)
    --save-logs         Save tick data to logs
    --filter PHASE      Filter by tick phase: all, perception, processing, integration (default: all)
"""

import sys
import time
import json
import argparse
from datetime import datetime, timedelta
from typing import Dict, List, Any, Optional
from dataclasses import dataclass, asdict
from pathlib import Path
import threading
import queue
from enum import Enum

# Add DAWN to path
dawn_root = Path(__file__).parent.parent.parent.parent
sys.path.insert(0, str(dawn_root))

try:
    from dawn.core.foundation.state import get_state, get_state_summary
    from dawn.core.communication.bus import get_consciousness_bus
    from dawn.processing.engines.tick.synchronous.orchestrator import TickPhase
    from dawn.consciousness.metrics.core import calculate_consciousness_metrics
    
    # Import the global singleton instances for real-time data
    try:
        from dawn.consciousness.engines.core.primary_engine import get_dawn_engine, _global_dawn_engine
        from dawn.core.communication.bus import _global_consciousness_bus
        from dawn.consciousness.metrics.core import _global_calculator
        LIVE_SINGLETONS_AVAILABLE = True
    except ImportError:
        LIVE_SINGLETONS_AVAILABLE = False
        print("⚠️  Live singleton instances not available - using basic access only")
    
    # Try to import SCUP metrics
    try:
        from dawn.subsystems.schema.scup_math import SCUPInputs, compute_basic_scup
        try:
            from dawn.subsystems.schema.scup_tracker import SCUPTracker
            SCUP_TRACKER_AVAILABLE = True
        except ImportError:
            SCUP_TRACKER_AVAILABLE = False
        SCUP_AVAILABLE = True
    except ImportError:
        SCUP_AVAILABLE = False
        SCUP_TRACKER_AVAILABLE = False
        print("⚠️  SCUP metrics not available - using basic metrics only")
        
    # Try to import thermal pulse singleton
    try:
        from dawn.subsystems.thermal.pulse.pulse_heat import PulseHeat
        from dawn.subsystems.thermal.pulse.unified_pulse_heat import UnifiedPulseHeat
        THERMAL_AVAILABLE = True
    except ImportError:
        THERMAL_AVAILABLE = False
        
except ImportError as e:
    print(f"❌ Import Error: {e}")
    print("Make sure you're running from the DAWN root directory and that DAWN is properly installed")
    sys.exit(1)

class DisplayMode(Enum):
    LIVE = "live"
    SNAPSHOT = "snapshot"
    ANALYZE = "analyze"

@dataclass
class TickSnapshot:
    """Represents a snapshot of tick state at a specific moment"""
    timestamp: datetime
    tick_count: int
    current_phase: str
    phase_duration: float
    total_cycle_time: float
    consciousness_level: float
    unity_score: float
    awareness_delta: float
    processing_load: float
    memory_usage: Dict[str, Any]
    active_modules: List[str]
    error_count: int
    warnings: List[str]

class TickStateReader:
    """Real-time tick state monitoring system"""
    
    def __init__(self, history_size: int = 100, save_logs: bool = False):
        self.history_size = history_size
        self.save_logs = save_logs
        self.tick_history: List[TickSnapshot] = []
        self.running = False
        self.start_time = datetime.now()
        
        # Threading
        self.data_queue = queue.Queue()
        self.reader_thread = None
        
        # Logging
        if save_logs:
            self.log_file = Path(f"data/runtime/logs/tick_reader_{datetime.now().strftime('%Y%m%d_%H%M%S')}.jsonl")
            self.log_file.parent.mkdir(parents=True, exist_ok=True)
    
    def get_current_tick_state(self) -> TickSnapshot:
        """Capture current tick state from DAWN system"""
        try:
            # Try to get data from live singleton instances first
            live_engine = None
            live_bus = None
            live_metrics = None
            
            if LIVE_SINGLETONS_AVAILABLE:
                # Get the actual running singleton instances
                if _global_dawn_engine is not None:
                    live_engine = _global_dawn_engine
                if _global_consciousness_bus is not None:
                    live_bus = _global_consciousness_bus
                live_metrics = _global_calculator
            
            # Fallback to function calls
            if live_engine is None:
                try:
                    live_engine = get_dawn_engine()
                except Exception as e:
                    # Log specific error for debugging
                    print(f"⚠️  Could not get DAWN engine: {e}")
            
            if live_bus is None:
                live_bus = get_consciousness_bus()
            
            # Get basic state information
            state = get_state()
            
            # Enhanced module state gathering from live instances
            module_states = {}
            active_modules = []
            
            # Try to get real module data from live engine first
            if live_engine and hasattr(live_engine, 'registered_modules'):
                active_modules = list(live_engine.registered_modules.keys())
                for name in active_modules:
                    try:
                        module_instance = live_engine.registered_modules.get(name) 
                        if module_instance:
                            # Get real state from live module
                            if hasattr(module_instance, 'get_current_state'):
                                module_state = module_instance.get_current_state() or {}
                            elif hasattr(module_instance, '__dict__'):
                                module_state = {
                                    k: v for k, v in module_instance.__dict__.items()
                                    if not k.startswith('_') and not callable(v)
                                }
                            else:
                                module_state = {'status': 'active', 'health': 1.0}
                            
                            module_states[name] = module_state
                        else:
                            module_states[name] = {'status': 'registered', 'health': 1.0}
                    except Exception as e:
                        module_states[name] = {'status': 'error', 'health': 0.5, 'error': str(e)[:100]}  # Truncate long errors
            
            # Fallback to bus modules if engine not available
            elif live_bus and hasattr(live_bus, 'registered_modules'):
                active_modules = list(live_bus.registered_modules.keys())
                for name in active_modules:
                    try:
                        module_info = live_bus.registered_modules.get(name, {})
                        module_states[name] = {
                            'status': getattr(module_info, 'status', 'active'),
                            'last_update': getattr(module_info, 'last_update', time.time()),
                            'health': getattr(module_info, 'health_score', 1.0)
                        }
                    except:
                        module_states[name] = {'status': 'active', 'health': 1.0}
            
            # Calculate consciousness metrics with real module states
            consciousness_metrics = calculate_consciousness_metrics(module_states)
            
            # Try to get real tick data from live instances
            tick_count = 0
            current_phase = 'unknown'
            phase_start = time.time()
            cycle_time = 0.0
            
            # Get tick count from live engine
            if live_engine and hasattr(live_engine, 'tick_count'):
                tick_count = live_engine.tick_count
            # Or from engine state/status
            elif live_engine and hasattr(live_engine, 'get_engine_status'):
                try:
                    status = live_engine.get_engine_status()
                    tick_count = status.get('tick_count', tick_count)
                except Exception as e:
                    # Log but continue with fallback
                    pass
            
            # Fallback to state
            if tick_count == 0:
                tick_count = getattr(state, 'tick_count', getattr(state, 'current_tick', getattr(state, 'ticks', 0)))
            
            # Get phase information
            if live_engine and hasattr(live_engine, 'current_phase'):
                current_phase = str(live_engine.current_phase)
            elif live_bus and hasattr(live_bus, 'current_phase'):
                current_phase = str(live_bus.current_phase)
            else:
                current_phase = getattr(state, 'current_tick_phase', getattr(state, 'phase', 'unknown'))
            
            # Get timing data
            if live_engine and hasattr(live_engine, 'last_cycle_time'):
                cycle_time = live_engine.last_cycle_time
            elif live_bus and hasattr(live_bus, 'metrics'):
                cycle_time = live_bus.metrics.get('average_response_time', 0.0)
            else:
                cycle_time = getattr(state, 'last_cycle_time', getattr(state, 'tick_duration', 0.0))
            
            phase_start = getattr(state, 'phase_start_time', getattr(state, 'last_tick_time', time.time()))
            phase_duration = time.time() - phase_start
            
            # Get consciousness metrics with real data
            consciousness_level = consciousness_metrics.quality
            unity_score = consciousness_metrics.consciousness_unity
            awareness_delta = consciousness_metrics.synchronization_score
            
            # Get real state values if available
            if hasattr(state, 'unity'):
                unity_score = max(unity_score, state.unity)
            if hasattr(state, 'awareness'):
                awareness_delta = max(awareness_delta, state.awareness)
            
            # Get SCUP if available
            scup_value = 0.5
            if SCUP_AVAILABLE:
                try:
                    scup_inputs = SCUPInputs(
                        alignment=unity_score,
                        entropy=1.0 - consciousness_level,
                        pressure=getattr(state, 'system_pressure', 0.3)
                    )
                    scup_value = compute_basic_scup(scup_inputs)
                    consciousness_level = max(consciousness_level, scup_value)
                except Exception as e:
                    # SCUP calculation failed, use fallback
                    print(f"⚠️  SCUP calculation failed: {e}")
            
            # Get thermal data if available
            heat_value = 0.0
            if THERMAL_AVAILABLE:
                try:
                    pulse_instance = PulseHeat()
                    if hasattr(pulse_instance, 'current_heat'):
                        heat_value = pulse_instance.current_heat
                    elif hasattr(pulse_instance, 'heat'):
                        heat_value = pulse_instance.heat
                except Exception as e:
                    try:
                        unified_pulse = UnifiedPulseHeat()
                        if hasattr(unified_pulse, 'current_heat'):
                            heat_value = unified_pulse.current_heat
                    except Exception as e2:
                        # Both thermal systems unavailable
                        print(f"⚠️  Thermal systems unavailable: {e}, {e2}")
            
            # System health metrics with live data
            processing_load = 0.0
            if live_bus and hasattr(live_bus, 'metrics'):
                processing_load = live_bus.metrics.get('bus_coherence_score', 0.0) * 100
            elif live_engine and hasattr(live_engine, 'get_performance_metrics'):
                try:
                    perf = live_engine.get_performance_metrics()
                    processing_load = perf.get('cpu_usage', len(active_modules) * 5.0)
                except Exception as e:
                    # Performance metrics unavailable, use estimated load
                    processing_load = len(active_modules) * 5.0
            else:
                processing_load = getattr(state, 'processing_load', len(active_modules) * 5.0)
            
            error_count = getattr(state, 'error_count', 0)
            warnings = getattr(state, 'current_warnings', getattr(state, 'warnings', []))
            
            # Memory usage with live data
            memory_usage = {
                'working_memory': getattr(state, 'working_memory_usage', len(module_states) * 2048),
                'long_term_memory': getattr(state, 'long_term_memory_usage', 
                                          len(active_modules) * 1024),
                'cache_size': getattr(state, 'cache_size', len(active_modules) * 512),
                'heat_level': heat_value
            }
            
            return TickSnapshot(
                timestamp=datetime.now(),
                tick_count=tick_count,
                current_phase=current_phase.upper() if current_phase != 'unknown' else 'MONITORING',
                phase_duration=phase_duration,
                total_cycle_time=cycle_time,
                consciousness_level=consciousness_level,
                unity_score=unity_score,
                awareness_delta=awareness_delta,
                processing_load=processing_load,
                memory_usage=memory_usage,
                active_modules=active_modules,
                error_count=error_count,
                warnings=warnings
            )
            
        except Exception as e:
            # Return error state
            return TickSnapshot(
                timestamp=datetime.now(),
                tick_count=-1,
                current_phase="ERROR",
                phase_duration=0.0,
                total_cycle_time=0.0,
                consciousness_level=0.0,
                unity_score=0.0,
                awareness_delta=0.0,
                processing_load=0.0,
                memory_usage={},
                active_modules=[],
                error_count=1,
                warnings=[f"Reader error: {str(e)}"]
            )
    
    def add_tick_snapshot(self, snapshot: TickSnapshot):
        """Add a tick snapshot to history"""
        self.tick_history.append(snapshot)
        
        # Maintain history size
        if len(self.tick_history) > self.history_size:
            self.tick_history.pop(0)
        
        # Save to log if enabled
        if self.save_logs:
            with open(self.log_file, 'a') as f:
                f.write(json.dumps(asdict(snapshot), default=str) + '\n')
    
    def start_monitoring(self, interval: float = 0.5):
        """Start continuous monitoring in a separate thread"""
        self.running = True
        self.reader_thread = threading.Thread(target=self._monitor_loop, args=(interval,))
        self.reader_thread.daemon = True
        self.reader_thread.start()
    
    def stop_monitoring(self):
        """Stop monitoring"""
        self.running = False
        if self.reader_thread:
            self.reader_thread.join(timeout=1.0)
    
    def _monitor_loop(self, interval: float):
        """Main monitoring loop"""
        while self.running:
            try:
                snapshot = self.get_current_tick_state()
                self.add_tick_snapshot(snapshot)
                self.data_queue.put(snapshot)
                time.sleep(interval)
            except Exception as e:
                print(f"⚠️ Monitoring error: {e}")
                time.sleep(interval)
    
    def display_live(self, update_interval: float = 0.5, phase_filter: str = "all"):
        """Display live tick state updates"""
        self.start_monitoring(update_interval)
        
        try:
            print("🚀 DAWN Tick State Reader - Live Mode")
            print("=" * 80)
            print(f"Update interval: {update_interval}s | Filter: {phase_filter}")
            print("Press Ctrl+C or Ctrl+D to stop\n")
            
            while True:
                try:
                    # Get latest snapshot
                    snapshot = self.data_queue.get(timeout=update_interval * 2)
                    
                    # Apply filter
                    if phase_filter != "all" and snapshot.current_phase.lower() != phase_filter:
                        continue
                    
                    # Clear screen and display
                    print("\033[2J\033[H")  # Clear screen
                    self._display_snapshot(snapshot)
                    self._display_metrics_summary()
                    
                except queue.Empty:
                    print("⏳ Waiting for tick data...")
                    continue
                    
                except EOFError:
                    print("\n👋 EOF detected - stopping tick state reader...")
                    break
                    
        except KeyboardInterrupt:
            print("\n👋 Interrupt detected - stopping tick state reader...")
        except EOFError:
            print("\n👋 EOF detected - stopping tick state reader...")
        finally:
            self.stop_monitoring()
    
    def display_snapshot(self):
        """Display a single snapshot of current state"""
        print("📸 DAWN Tick State Snapshot")
        print("=" * 50)
        
        snapshot = self.get_current_tick_state()
        self._display_snapshot(snapshot)
    
    def analyze_history(self):
        """Analyze tick history and show statistics"""
        if not self.tick_history:
            print("❌ No tick history available")
            return
        
        print("📊 DAWN Tick Analysis")
        print("=" * 50)
        
        # Basic statistics
        total_ticks = len(self.tick_history)
        avg_cycle_time = sum(s.total_cycle_time for s in self.tick_history) / total_ticks
        avg_consciousness = sum(s.consciousness_level for s in self.tick_history) / total_ticks
        avg_unity = sum(s.unity_score for s in self.tick_history) / total_ticks
        
        print(f"📈 Analysis Summary:")
        print(f"   Total ticks analyzed: {total_ticks}")
        print(f"   Average cycle time: {avg_cycle_time:.3f}s")
        print(f"   Average consciousness level: {avg_consciousness:.3f}")
        print(f"   Average unity score: {avg_unity:.3f}")
        
        # Phase distribution
        phase_counts = {}
        for snapshot in self.tick_history:
            phase = snapshot.current_phase
            phase_counts[phase] = phase_counts.get(phase, 0) + 1
        
        print(f"\n🔄 Phase Distribution:")
        for phase, count in sorted(phase_counts.items()):
            percentage = (count / total_ticks) * 100
            print(f"   {phase}: {count} ({percentage:.1f}%)")
        
        # Error analysis
        total_errors = sum(s.error_count for s in self.tick_history)
        if total_errors > 0:
            print(f"\n⚠️ Error Summary:")
            print(f"   Total errors: {total_errors}")
            print(f"   Error rate: {(total_errors/total_ticks)*100:.1f}% of ticks")
            
            # Recent warnings
            recent_warnings = set()
            for snapshot in self.tick_history[-10:]:  # Last 10 ticks
                recent_warnings.update(snapshot.warnings)
            
            if recent_warnings:
                print(f"   Recent warnings:")
                for warning in list(recent_warnings)[:5]:  # Show top 5
                    # Truncate very long warnings
                    warning_text = warning[:80] + "..." if len(warning) > 80 else warning
                    print(f"     - {warning_text}")
        else:
            print(f"\n✅ No errors detected in analysis period")
    
    def _display_snapshot(self, snapshot: TickSnapshot):
        """Display a formatted tick snapshot"""
        # Header
        print(f"🕐 {snapshot.timestamp.strftime('%H:%M:%S.%f')[:-3]} | Tick #{snapshot.tick_count}")
        print("-" * 80)
        
        # Current state
        print(f"🔄 Current Phase: {snapshot.current_phase.upper()}")
        print(f"⏱️  Phase Duration: {snapshot.phase_duration:.3f}s")
        print(f"🔁 Cycle Time: {snapshot.total_cycle_time:.3f}s")
        
        # Consciousness metrics
        print(f"\n🧠 Consciousness Metrics:")
        # Color code consciousness level
        level_color = "🟢" if snapshot.consciousness_level > 0.7 else "🟡" if snapshot.consciousness_level > 0.4 else "🔴"
        print(f"   Level: {level_color} {snapshot.consciousness_level:.3f}")
        
        # Color code unity
        unity_color = "🟢" if snapshot.unity_score > 0.7 else "🟡" if snapshot.unity_score > 0.4 else "🔴"
        print(f"   Unity: {unity_color} {snapshot.unity_score:.3f}")
        
        # Color code awareness delta
        awareness_color = "🟢" if snapshot.awareness_delta > 0 else "🟡" if snapshot.awareness_delta > -0.1 else "🔴"
        print(f"   Awareness Δ: {awareness_color} {snapshot.awareness_delta:+.3f}")
        
        # System health
        print(f"\n💻 System Health:")
        # Color code processing load
        load_color = "🟢" if snapshot.processing_load < 50 else "🟡" if snapshot.processing_load < 80 else "🔴"
        print(f"   Processing Load: {load_color} {snapshot.processing_load:.1f}%")
        print(f"   Active Modules: {len(snapshot.active_modules)}")
        
        # Color code errors
        error_color = "🟢" if snapshot.error_count == 0 else "🟡" if snapshot.error_count < 5 else "🔴"
        print(f"   Errors: {error_color} {snapshot.error_count}")
        
        # Memory usage
        if snapshot.memory_usage:
            print(f"\n🧮 Memory Usage:")
            for mem_type, usage in snapshot.memory_usage.items():
                if isinstance(usage, (int, float)):
                    # Format numbers nicely
                    if usage > 1000000:
                        formatted_usage = f"{usage/1000000:.1f}M"
                    elif usage > 1000:
                        formatted_usage = f"{usage/1000:.1f}K"
                    else:
                        formatted_usage = f"{usage:.1f}"
                else:
                    formatted_usage = str(usage)
                print(f"   {mem_type.replace('_', ' ').title()}: {formatted_usage}")
        
        # Warnings
        if snapshot.warnings:
            print(f"\n⚠️ Warnings:")
            for warning in snapshot.warnings[:3]:  # Show first 3
                # Truncate very long warnings for display
                warning_text = warning[:60] + "..." if len(warning) > 60 else warning
                print(f"   - {warning_text}")
        
        print()
    
    def _display_metrics_summary(self):
        """Display summary metrics from recent history"""
        if len(self.tick_history) < 5:
            return
        
        recent = self.tick_history[-5:]  # Last 5 ticks
        
        # Calculate trends
        consciousness_trend = recent[-1].consciousness_level - recent[0].consciousness_level
        unity_trend = recent[-1].unity_score - recent[0].unity_score
        cycle_time_trend = recent[-1].total_cycle_time - recent[0].total_cycle_time
        
        print("📊 Recent Trends (last 5 ticks):")
        print(f"   Consciousness: {consciousness_trend:+.3f}")
        print(f"   Unity: {unity_trend:+.3f}")
        print(f"   Cycle Time: {cycle_time_trend:+.3f}s")
        
        # Additional metrics
        avg_processing_load = sum(s.processing_load for s in recent) / len(recent)
        max_errors = max(s.error_count for s in recent)
        
        print(f"   Avg Processing Load: {avg_processing_load:.1f}%")
        print(f"   Max Errors: {max_errors}")
        
        # Uptime
        uptime = datetime.now() - self.start_time
        print(f"⏰ Uptime: {str(uptime).split('.')[0]}")

def main():
    """Main entry point"""
    parser = argparse.ArgumentParser(description="DAWN Tick State Reader")
    parser.add_argument("--mode", choices=["live", "snapshot", "analyze"], 
                       default="live", help="Display mode")
    parser.add_argument("--interval", type=float, default=0.5,
                       help="Update interval for live mode")
    parser.add_argument("--history", type=int, default=100,
                       help="Number of ticks to keep in history")
    parser.add_argument("--save-logs", action="store_true",
                       help="Save tick data to logs")
    parser.add_argument("--export", action="store_true",
                       help="Export collected data to JSON file")
    parser.add_argument("--health-check", action="store_true",
                       help="Show system health summary and exit")
    parser.add_argument("--filter", choices=["all", "perception", "processing", "integration", "monitoring", "state_collection", "information_sharing", "decision_making", "state_updates", "synchronization_check"],
                       default="all", help="Filter by tick phase")
    
    args = parser.parse_args()
    
    # Create reader
    reader = TickStateReader(
        history_size=args.history,
        save_logs=args.save_logs
    )
    
    try:
        if args.mode == "live":
            reader.display_live(args.interval, args.filter)
        elif args.mode == "snapshot":
            reader.display_snapshot()
        elif args.mode == "analyze":
            # Need to collect some data first
            print("📊 Collecting tick data for analysis...")
            reader.start_monitoring(args.interval)
            time.sleep(5)  # Collect for 5 seconds
            reader.stop_monitoring()
            reader.analyze_history()
        elif args.mode == "health":
            # Quick health check mode
            print("🌡️ Performing system health check...")
            reader.start_monitoring(args.interval)
            time.sleep(3)  # Quick sample
            reader.stop_monitoring()
            
            health = TickDataExporter.get_health_summary(reader.tick_history)
            print(f"\n📊 System Health Report:")
            print(f"   Status: {health['status'].upper()} ({health['health_score']}/100)")
            print(f"   Sample Size: {health['sample_size']} ticks")
            print(f"\n📈 Key Metrics:")
            for key, value in health['metrics'].items():
                print(f"   {key.replace('_', ' ').title()}: {value}")
        else:
            print(f"\n❌ Unknown mode: {args.mode}")
            return 1
        
        # Handle export request
        if args.export and reader.tick_history:
            try:
                filename = TickDataExporter.export_data(reader.tick_history)
                print(f"\n💾 Data exported to: {filename}")
            except Exception as e:
                print(f"\n❌ Export failed: {e}")
        
        # Handle health check request
        if args.health_check:
            health = TickDataExporter.get_health_summary(reader.tick_history)
            print(f"\n🌡️ Health Check: {health['status'].upper()} ({health['health_score']}/100)")
            return 0 if health['health_score'] > 50 else 1
            
    except KeyboardInterrupt:
        print("\n👋 Keyboard interrupt - exiting gracefully...")
        return 0
    except EOFError:
        print("\n👋 EOF detected - exiting gracefully...")
        return 0
    except Exception as e:
        print(f"❌ Error: {e}")
        return 1

class TickDataExporter:
    """Utility class for exporting tick data"""
    
    @staticmethod
    def export_data(tick_history: List[TickSnapshot], filename: Optional[str] = None) -> str:
        """Export tick history to JSON file"""
        if not filename:
            timestamp = datetime.now().strftime('%Y%m%d_%H%M%S')
            filename = f"data/runtime/exports/tick_history_{timestamp}.json"
        
        # Ensure directory exists
        Path(filename).parent.mkdir(parents=True, exist_ok=True)
        
        # Convert to serializable format
        export_data = {
            'metadata': {
                'export_time': datetime.now().isoformat(),
                'total_snapshots': len(tick_history),
                'export_tool': 'DAWN Tick State Reader'
            },
            'snapshots': [asdict(snapshot) for snapshot in tick_history]
        }
        
        with open(filename, 'w') as f:
            json.dump(export_data, f, indent=2, default=str)
        
        return filename
    
    @staticmethod
    def get_health_summary(tick_history: List[TickSnapshot]) -> Dict[str, Any]:
        """Get overall system health summary"""
        if not tick_history:
            return {'status': 'no_data', 'health_score': 0.0}
        
        recent = tick_history[-10:] if len(tick_history) >= 10 else tick_history
        
        # Calculate health metrics
        avg_consciousness = sum(s.consciousness_level for s in recent) / len(recent)
        avg_unity = sum(s.unity_score for s in recent) / len(recent)
        avg_processing_load = sum(s.processing_load for s in recent) / len(recent)
        total_errors = sum(s.error_count for s in recent)
        error_rate = (total_errors / len(recent)) * 100
        
        # Calculate overall health score (0-100)
        health_score = (
            (avg_consciousness * 30) +  # 30% weight
            (avg_unity * 30) +          # 30% weight
            (max(0, (100 - avg_processing_load) / 100) * 20) +  # 20% weight (inverted)
            (max(0, (100 - error_rate) / 100) * 20)            # 20% weight (inverted)
        )
        
        # Determine status
        if health_score >= 80:
            status = 'excellent'
        elif health_score >= 60:
            status = 'good'
        elif health_score >= 40:
            status = 'fair'
        elif health_score >= 20:
            status = 'poor'
        else:
            status = 'critical'
        
        return {
            'status': status,
            'health_score': round(health_score, 2),
            'metrics': {
                'avg_consciousness': round(avg_consciousness, 3),
                'avg_unity': round(avg_unity, 3),
                'avg_processing_load': round(avg_processing_load, 1),
                'error_rate': round(error_rate, 2),
                'total_errors': total_errors
            },
            'sample_size': len(recent)
        }

if __name__ == "__main__":
    sys.exit(main())
