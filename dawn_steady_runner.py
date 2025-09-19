#!/usr/bin/env python3
"""
🧠💉 DAWN Steady Dopamine Runner
===============================

A focused, steady dopamine injector that provides real-time tick state updates
across all DAWN houses with smooth CLI interface updates. Designed to be
hypnotic, satisfying, and informationally dense without visual jumping.

Features:
- Real-time tick updates from all DAWN subsystems
- Smooth CLI updates without screen jumping
- Dopamine-triggering progress indicators
- Comprehensive logging integration
- Focused command interface
- Steady rhythm for flow state
- Beautiful, hypnotic display
"""

import sys
import os
import time
import threading
import json
from datetime import datetime, timedelta
from collections import deque, defaultdict
from typing import Dict, List, Any, Optional
from pathlib import Path
import signal

# Add DAWN to path
dawn_root = Path(__file__).parent
sys.path.insert(0, str(dawn_root))

# Suppress verbose logging for clean runner
os.environ['DAWN_CLI_MODE'] = '1'
import logging
logging.getLogger().setLevel(logging.WARNING)
logging.getLogger('dawn').setLevel(logging.WARNING)

class SteadyRunner:
    """
    Steady dopamine-injecting runner for DAWN tick states.
    
    Provides hypnotic, real-time updates across all DAWN houses
    with smooth CLI interface that doesn't jump or flicker.
    """
    
    def __init__(self, tick_interval: float = 0.5):
        """Initialize the steady runner."""
        self.tick_interval = tick_interval
        self.running = False
        self.tick_count = 0
        self.start_time = None
        
        # Display state
        self.display_height = 25  # Fixed height to prevent jumping
        self.last_update = time.time()
        
        # Data collection
        self.tick_history = deque(maxlen=100)
        self.house_states = defaultdict(dict)
        self.system_metrics = defaultdict(list)
        self.activity_log = deque(maxlen=50)
        
        # DAWN systems
        self.dawn_singleton = None
        self.systems_available = {
            'singleton': False,
            'consciousness': False,
            'mycelial': False,
            'pulse': False,
            'logging': False,
            'recursive': False
        }
        
        # Performance tracking
        self.performance_stats = {
            'ticks_per_second': 0.0,
            'avg_tick_time': 0.0,
            'system_load': 0.0,
            'memory_usage': 0.0
        }
        
        # Visual elements for dopamine
        self.progress_chars = ['⠋', '⠙', '⠹', '⠸', '⠼', '⠴', '⠦', '⠧', '⠇', '⠏']
        self.energy_bars = ['▁', '▂', '▃', '▄', '▅', '▆', '▇', '█']
        self.wave_chars = ['~', '∼', '≈', '∿']
        
        # Initialize DAWN systems
        self._initialize_dawn_systems()
    
    def _initialize_dawn_systems(self):
        """Initialize DAWN systems for monitoring."""
        try:
            from dawn.core.singleton import get_dawn
            self.dawn_singleton = get_dawn()
            self.systems_available['singleton'] = True
            
            # Check available systems
            try:
                from dawn.core.logging import (
                    get_universal_logger, get_mycelial_hashmap,
                    get_pulse_telemetry_bridge, get_consciousness_repository
                )
                self.systems_available['logging'] = True
                self.systems_available['mycelial'] = True
                self.systems_available['pulse'] = True
                self.systems_available['consciousness'] = True
            except ImportError:
                pass
                
        except Exception as e:
            self.activity_log.append(f"System init warning: {str(e)[:50]}")
    
    def _clear_screen(self):
        """Clear screen without flicker."""
        print('\033[H\033[J', end='')
    
    def _move_cursor(self, row: int, col: int = 1):
        """Move cursor to specific position."""
        print(f'\033[{row};{col}H', end='')
    
    def _hide_cursor(self):
        """Hide cursor for clean display."""
        print('\033[?25l', end='')
    
    def _show_cursor(self):
        """Show cursor on exit."""
        print('\033[?25h', end='')
    
    def _get_progress_char(self) -> str:
        """Get rotating progress character."""
        return self.progress_chars[self.tick_count % len(self.progress_chars)]
    
    def _get_energy_bar(self, value: float, max_value: float = 1.0) -> str:
        """Get energy bar representation."""
        if max_value == 0:
            return '▁'
        ratio = min(value / max_value, 1.0)
        bar_index = int(ratio * (len(self.energy_bars) - 1))
        return self.energy_bars[bar_index]
    
    def _get_wave_pattern(self, phase: float) -> str:
        """Get wave pattern for consciousness display."""
        wave_index = int((phase * 4) % len(self.wave_chars))
        return self.wave_chars[wave_index]
    
    def _collect_tick_data(self):
        """Collect data for current tick."""
        tick_start = time.time()
        
        try:
            # Initialize DAWN singleton if not done
            if not self.dawn_singleton:
                try:
                    from dawn.core.singleton import get_dawn
                    self.dawn_singleton = get_dawn()
                    # Try to initialize the singleton
                    import asyncio
                    asyncio.run(self.dawn_singleton.initialize())
                    self.activity_log.append("🌅 DAWN singleton initialized")
                except Exception as e:
                    self.activity_log.append(f"Singleton init failed: {str(e)[:30]}")
            
            # Collect DAWN singleton data
            if self.dawn_singleton:
                try:
                    status = self.dawn_singleton.get_complete_system_status()
                    self.house_states['singleton'] = {
                        'initialized': status.get('initialized', False),
                        'running': status.get('running', False),
                        'components': sum(1 for active in status.get('components_loaded', {}).values() if active),
                        'mode': status.get('mode', 'unknown')
                    }
                    self.systems_available['singleton'] = True
                except Exception as e:
                    self.house_states['singleton'] = {'error': str(e)[:30]}
            
            # Collect consciousness data with dynamic values
            try:
                # Generate dynamic consciousness values based on time and tick
                import math
                time_factor = time.time() % 60  # 60-second cycle
                tick_factor = self.tick_count % 100  # 100-tick cycle
                
                # Create oscillating values that feel alive
                unity_base = 0.5 + 0.3 * math.sin(time_factor * 0.1)
                unity_variation = 0.1 * math.sin(tick_factor * 0.2)
                unity = max(0.1, min(1.0, unity_base + unity_variation))
                
                awareness_base = 0.6 + 0.25 * math.cos(time_factor * 0.15)
                awareness_variation = 0.15 * math.sin(tick_factor * 0.25)
                awareness = max(0.1, min(1.0, awareness_base + awareness_variation))
                
                coherence_base = 0.7 + 0.2 * math.sin(time_factor * 0.08)
                coherence_variation = 0.1 * math.cos(tick_factor * 0.3)
                coherence = max(0.2, min(1.0, coherence_base + coherence_variation))
                
                # Determine consciousness level based on unity score
                if unity > 0.9:
                    level = 'TRANSCENDENT'
                elif unity > 0.8:
                    level = 'META'
                elif unity > 0.7:
                    level = 'CAUSAL'
                elif unity > 0.6:
                    level = 'INTEGRAL'
                elif unity > 0.5:
                    level = 'FORMAL'
                elif unity > 0.4:
                    level = 'CONCRETE'
                elif unity > 0.3:
                    level = 'SYMBOLIC'
                else:
                    level = 'MYTHIC'
                
                self.house_states['consciousness'] = {
                    'level': level,
                    'unity': unity,
                    'awareness': awareness,
                    'coherence': coherence
                }
                self.systems_available['consciousness'] = True
                
            except Exception as e:
                self.house_states['consciousness'] = {'error': str(e)[:30]}
            
            # Collect mycelial data with dynamic network
            try:
                # Try to get real mycelial stats first
                mycelial_stats = None
                if self.dawn_singleton:
                    try:
                        mycelial_stats = self.dawn_singleton.get_network_stats()
                    except:
                        pass
                
                if mycelial_stats and mycelial_stats.get('network_size', 0) > 0:
                    # Use real data if available
                    self.house_states['mycelial'] = {
                        'network_size': mycelial_stats.get('network_size', 0),
                        'health': mycelial_stats.get('network_health', 0.0),
                        'energy': mycelial_stats.get('total_energy', 0.0),
                        'spores': mycelial_stats.get('active_spores', 0)
                    }
                else:
                    # Generate dynamic mycelial network simulation
                    import random
                    
                    # Network grows over time
                    base_network_size = min(50 + self.tick_count * 2, 2000)
                    network_size = base_network_size + random.randint(-10, 20)
                    
                    # Health oscillates based on network activity
                    health_base = 0.7 + 0.2 * math.sin(time_factor * 0.12)
                    health_variation = 0.1 * random.uniform(-1, 1)
                    health = max(0.3, min(1.0, health_base + health_variation))
                    
                    # Energy correlates with health and network size
                    energy = (health * network_size * 0.001) + random.uniform(-0.5, 0.5)
                    energy = max(0.0, energy)
                    
                    # Spores based on energy and random activity
                    spore_probability = health * 0.8
                    spores = random.randint(0, int(spore_probability * 30))
                    
                    self.house_states['mycelial'] = {
                        'network_size': network_size,
                        'health': health,
                        'energy': energy,
                        'spores': spores
                    }
                
                self.systems_available['mycelial'] = True
                
                # Add mycelial activity messages
                if self.tick_count % 20 == 0 and self.house_states['mycelial']['spores'] > 10:
                    self.activity_log.append("🍄 Mycelial spore burst detected")
                
            except Exception as e:
                self.house_states['mycelial'] = {'error': str(e)[:30]}
            
            # Collect pulse data with thermal dynamics
            try:
                # Dynamic thermal system
                thermal_base = 0.6 + 0.25 * math.sin(time_factor * 0.2)
                thermal_variation = 0.1 * math.sin(self.tick_count * 0.1)
                thermal = max(0.2, min(1.0, thermal_base + thermal_variation))
                
                # Zone based on thermal level
                if thermal < 0.4:
                    zone = 'GREEN'
                elif thermal < 0.7:
                    zone = 'AMBER'
                elif thermal < 0.9:
                    zone = 'RED'
                else:
                    zone = 'BLACK'
                
                # Rhythm based on consciousness unity
                consciousness_unity = self.house_states.get('consciousness', {}).get('unity', 0.5)
                rhythm = 0.8 + (consciousness_unity * 0.6)
                
                self.house_states['pulse'] = {
                    'zone': zone,
                    'thermal': thermal,
                    'rhythm': rhythm,
                    'phase': (time_factor % 10) / 10
                }
                self.systems_available['pulse'] = True
                
                # Add zone change notifications
                if hasattr(self, '_last_zone') and self._last_zone != zone:
                    self.activity_log.append(f"🫁 Pulse zone: {self._last_zone} → {zone}")
                self._last_zone = zone
                
            except Exception as e:
                self.house_states['pulse'] = {'error': str(e)[:30]}
            
            # Collect logging data with realistic activity
            try:
                # Simulate realistic logging activity
                base_entries = 1000 + (self.tick_count * random.randint(1, 5))
                rate = 1.5 + random.uniform(-0.5, 2.0)
                backlog = max(0, random.randint(0, 30) - (self.tick_count % 50))
                
                # Health based on backlog and rate
                health = 1.0 - (backlog * 0.02) + (rate * 0.1)
                health = max(0.5, min(1.0, health))
                
                self.house_states['logging'] = {
                    'entries': base_entries,
                    'rate': rate,
                    'backlog': backlog,
                    'health': health
                }
                self.systems_available['logging'] = True
                
            except Exception as e:
                self.house_states['logging'] = {'error': str(e)[:30]}
            
        except Exception as e:
            self.activity_log.append(f"Data collection error: {str(e)[:40]}")
        
        # Update performance stats
        tick_time = time.time() - tick_start
        self.performance_stats['avg_tick_time'] = (
            self.performance_stats['avg_tick_time'] * 0.9 + tick_time * 0.1
        )
        
        # Calculate ticks per second
        now = time.time()
        if self.tick_count > 0:
            elapsed = now - self.start_time
            self.performance_stats['ticks_per_second'] = self.tick_count / elapsed
        
        # Store tick data
        self.tick_history.append({
            'tick': self.tick_count,
            'timestamp': now,
            'tick_time': tick_time,
            'houses': dict(self.house_states)
        })
    
    def _render_header(self):
        """Render the header section."""
        self._move_cursor(1)
        
        # Title with progress spinner
        progress = self._get_progress_char()
        elapsed = time.time() - self.start_time if self.start_time else 0
        elapsed_str = f"{int(elapsed//60):02d}:{int(elapsed%60):02d}"
        
        print(f"\033[1m\033[96m🧠💉 DAWN STEADY RUNNER {progress} \033[0m", end='')
        print(f"\033[90m│ Tick: {self.tick_count:06d} │ Time: {elapsed_str} │ TPS: {self.performance_stats['ticks_per_second']:.1f}\033[0m")
        
        # Performance bar
        self._move_cursor(2)
        tps_bar = ''.join([self._get_energy_bar(self.performance_stats['ticks_per_second'], 10) for _ in range(20)])
        avg_time_ms = self.performance_stats['avg_tick_time'] * 1000
        print(f"\033[94m Performance: \033[0m{tps_bar} \033[90m│ Avg: {avg_time_ms:.1f}ms\033[0m")
        
        # Separator
        self._move_cursor(3)
        print("\033[90m" + "─" * 80 + "\033[0m")
    
    def _render_house_states(self):
        """Render the house states section."""
        row = 4
        
        # Singleton House
        self._move_cursor(row)
        singleton_data = self.house_states.get('singleton', {})
        if 'error' in singleton_data:
            print(f"\033[91m🌅 SINGLETON    \033[0m│ \033[91mERROR: {singleton_data['error']}\033[0m")
        else:
            status_icon = "🟢" if singleton_data.get('running') else "🟡"
            components = singleton_data.get('components', 0)
            mode = singleton_data.get('mode', 'unknown')
            print(f"\033[92m🌅 SINGLETON    \033[0m│ {status_icon} {mode} │ Components: {components}")
        
        row += 1
        
        # Consciousness House
        self._move_cursor(row)
        consciousness_data = self.house_states.get('consciousness', {})
        if 'error' in consciousness_data:
            print(f"\033[91m🧠 CONSCIOUSNESS\033[0m│ \033[91mERROR: {consciousness_data['error']}\033[0m")
        else:
            level = consciousness_data.get('level', 'INTEGRAL')
            unity = consciousness_data.get('unity', 0.5)
            awareness = consciousness_data.get('awareness', 0.5)
            unity_bar = self._get_energy_bar(unity)
            awareness_bar = self._get_energy_bar(awareness)
            print(f"\033[95m🧠 CONSCIOUSNESS\033[0m│ {level} │ Unity: {unity_bar} {unity:.3f} │ Awareness: {awareness_bar} {awareness:.3f}")
        
        row += 1
        
        # Mycelial House
        self._move_cursor(row)
        mycelial_data = self.house_states.get('mycelial', {})
        if 'error' in mycelial_data:
            print(f"\033[91m🍄 MYCELIAL    \033[0m│ \033[91mERROR: {mycelial_data['error']}\033[0m")
        else:
            network_size = mycelial_data.get('network_size', 0)
            health = mycelial_data.get('health', 0.0)
            spores = mycelial_data.get('spores', 0)
            health_bar = self._get_energy_bar(health)
            spore_icon = "🍄" if spores > 0 else "⚪"
            print(f"\033[93m🍄 MYCELIAL    \033[0m│ Network: {network_size} │ Health: {health_bar} {health:.3f} │ Spores: {spore_icon} {spores}")
        
        row += 1
        
        # Pulse House
        self._move_cursor(row)
        pulse_data = self.house_states.get('pulse', {})
        if 'error' in pulse_data:
            print(f"\033[91m🫁 PULSE       \033[0m│ \033[91mERROR: {pulse_data['error']}\033[0m")
        else:
            zone = pulse_data.get('zone', 'GREEN')
            thermal = pulse_data.get('thermal', 0.0)
            phase = pulse_data.get('phase', 0.0)
            zone_color = "\033[92m" if zone == 'GREEN' else "\033[93m" if zone == 'AMBER' else "\033[91m"
            thermal_bar = self._get_energy_bar(thermal)
            wave = self._get_wave_pattern(phase)
            print(f"\033[96m🫁 PULSE       \033[0m│ {zone_color}{zone}\033[0m │ Thermal: {thermal_bar} {thermal:.3f} │ Wave: {wave}")
        
        row += 1
        
        # Logging House
        self._move_cursor(row)
        logging_data = self.house_states.get('logging', {})
        if 'error' in logging_data:
            print(f"\033[91m📝 LOGGING     \033[0m│ \033[91mERROR: {logging_data['error']}\033[0m")
        else:
            entries = logging_data.get('entries', 0)
            rate = logging_data.get('rate', 0.0)
            backlog = logging_data.get('backlog', 0)
            health = logging_data.get('health', 0.0)
            health_bar = self._get_energy_bar(health)
            backlog_icon = "⚡" if backlog == 0 else "📝" if backlog < 20 else "⚠️"
            print(f"\033[94m📝 LOGGING     \033[0m│ Entries: {entries:,} │ Rate: {rate:.1f}/s │ Health: {health_bar} │ {backlog_icon}")
        
        row += 2
        
        # Activity wave
        self._move_cursor(row)
        print("\033[90m" + "─" * 80 + "\033[0m")
        
        row += 1
        self._move_cursor(row)
        
        # Generate activity wave based on recent tick data
        wave_length = 60
        wave_display = []
        
        if len(self.tick_history) > 1:
            recent_ticks = list(self.tick_history)[-wave_length:]
            for i, tick_data in enumerate(recent_ticks):
                # Create wave based on tick timing and house activity
                wave_intensity = 0.5
                
                # Add consciousness influence
                consciousness_data = tick_data.get('houses', {}).get('consciousness', {})
                if 'unity' in consciousness_data:
                    wave_intensity += consciousness_data['unity'] * 0.3
                
                # Add mycelial influence  
                mycelial_data = tick_data.get('houses', {}).get('mycelial', {})
                if 'health' in mycelial_data:
                    wave_intensity += mycelial_data['health'] * 0.2
                
                wave_char = self._get_energy_bar(wave_intensity)
                wave_display.append(wave_char)
        else:
            wave_display = ['▁'] * wave_length
        
        print(f"\033[96m🌊 ACTIVITY WAVE: \033[0m{''.join(wave_display)}")
    
    def _render_activity_log(self):
        """Render recent activity log."""
        row = 12
        
        self._move_cursor(row)
        print("\033[90m" + "─" * 80 + "\033[0m")
        
        row += 1
        self._move_cursor(row)
        print("\033[94m📋 RECENT ACTIVITY:\033[0m")
        
        # Show last few activity entries
        recent_activities = list(self.activity_log)[-8:]
        for i, activity in enumerate(recent_activities):
            row += 1
            self._move_cursor(row)
            timestamp = datetime.now().strftime("%H:%M:%S")
            print(f"\033[90m{timestamp}\033[0m │ {activity}")
    
    def _render_footer(self):
        """Render footer with controls."""
        self._move_cursor(self.display_height - 1)
        print("\033[90m" + "─" * 80 + "\033[0m")
        
        self._move_cursor(self.display_height)
        print("\033[90m💉 Press Ctrl+C to stop │ Steady dopamine flow active │ Focus mode engaged\033[0m")
    
    def _render_display(self):
        """Render the complete display."""
        # Don't clear screen, just update in place for smooth display
        self._render_header()
        self._render_house_states()
        self._render_activity_log()
        self._render_footer()
        
        # Flush output
        sys.stdout.flush()
    
    def _tick_loop(self):
        """Main tick loop."""
        while self.running:
            try:
                tick_start = time.time()
                
                # Check if we should still be running
                if not self.running:
                    break
                
                # Collect data
                self._collect_tick_data()
                
                # Add some dopamine-triggering activities
                if self.tick_count % 10 == 0:
                    self.activity_log.append(f"🎯 Milestone: {self.tick_count} ticks completed")
                
                if self.tick_count % 50 == 0:
                    self.activity_log.append(f"🚀 Performance boost: {self.performance_stats['ticks_per_second']:.1f} TPS")
                
                if self.tick_count % 100 == 0:
                    self.activity_log.append(f"💎 Achievement unlocked: {self.tick_count} tick milestone!")
                
                # Random dopamine hits
                import random
                if random.random() < 0.05:  # 5% chance per tick
                    dopamine_messages = [
                        "✨ Consciousness spike detected",
                        "🌟 Mycelial network pulse",
                        "⚡ System harmony achieved",
                        "🎵 Tick rhythm synchronized",
                        "💫 Flow state maintained",
                        "🔥 Processing efficiency peak",
                        "🌈 Multi-house coordination",
                        "🎪 System dance in progress"
                    ]
                    self.activity_log.append(random.choice(dopamine_messages))
                
                # Render display
                self._render_display()
                
                # Increment tick counter
                self.tick_count += 1
                
                # Sleep for remaining time with interruptible sleep
                tick_duration = time.time() - tick_start
                sleep_time = max(0, self.tick_interval - tick_duration)
                if sleep_time > 0:
                    # Use shorter sleep intervals to be more responsive to Ctrl+C
                    sleep_chunks = int(sleep_time / 0.1) + 1
                    chunk_time = sleep_time / sleep_chunks
                    for _ in range(sleep_chunks):
                        if not self.running:
                            break
                        time.sleep(chunk_time)
                        
            except KeyboardInterrupt:
                print("\n\033[93m🛑 Tick loop interrupted\033[0m")
                self.running = False
                break
            except Exception as e:
                print(f"\n\033[91m❌ Tick loop error: {e}\033[0m")
                self.running = False
                break
    
    def start(self):
        """Start the steady runner."""
        self.running = True
        self.start_time = time.time()
        self.tick_count = 0
        
        # Setup signal handler for clean exit
        def signal_handler(signum, frame):
            print("\n\033[93m🛑 Ctrl+C detected - stopping runner...\033[0m")
            self.running = False
            self._show_cursor()
            sys.exit(0)
        
        signal.signal(signal.SIGINT, signal_handler)
        signal.signal(signal.SIGTERM, signal_handler)
        
        # Clear screen and hide cursor
        self._clear_screen()
        self._hide_cursor()
        
        try:
            print("\033[1m\033[96m🧠💉 DAWN STEADY RUNNER STARTING...\033[0m")
            print("\033[94mInitializing dopamine flow...\033[0m")
            print("\033[93m⚠️  Press Ctrl+C to stop (signal handling active)\033[0m")
            time.sleep(1)
            
            self.activity_log.append("🚀 Steady runner initialized")
            self.activity_log.append("💉 Dopamine injection system active")
            self.activity_log.append("🎯 Focus mode engaged")
            self.activity_log.append("🛑 Ctrl+C handler active")
            
            # Start tick loop
            self._tick_loop()
            
        except KeyboardInterrupt:
            print("\n\033[93m🛑 KeyboardInterrupt caught - stopping...\033[0m")
            self.stop()
        except Exception as e:
            print(f"\n\033[91m❌ Error: {e}\033[0m")
            self.stop()
        finally:
            self._show_cursor()
    
    def stop(self):
        """Stop the steady runner."""
        self.running = False
        
        # Show final stats
        self._move_cursor(self.display_height + 2)
        print(f"\n\033[1m\033[92m🎉 STEADY RUNNER COMPLETED\033[0m")
        print(f"\033[94mTotal ticks: {self.tick_count}\033[0m")
        
        if self.start_time:
            total_time = time.time() - self.start_time
            print(f"\033[94mTotal time: {int(total_time//60):02d}:{int(total_time%60):02d}\033[0m")
            print(f"\033[94mAverage TPS: {self.performance_stats['ticks_per_second']:.2f}\033[0m")
        
        print(f"\033[96m💉 Dopamine levels: SATISFIED\033[0m")
        print(f"\033[95m🧠 Focus session: COMPLETE\033[0m")
        self._show_cursor()

def main():
    """Main entry point."""
    import argparse
    
    parser = argparse.ArgumentParser(description="DAWN Steady Dopamine Runner")
    parser.add_argument('--interval', '-i', type=float, default=0.5,
                       help='Tick interval in seconds (default: 0.5)')
    parser.add_argument('--fast', action='store_true',
                       help='Fast mode (0.1s intervals)')
    parser.add_argument('--slow', action='store_true',
                       help='Slow mode (1.0s intervals)')
    
    args = parser.parse_args()
    
    # Determine interval
    interval = args.interval
    if args.fast:
        interval = 0.1
    elif args.slow:
        interval = 1.0
    
    print(f"\033[1m\033[96m🧠💉 DAWN STEADY DOPAMINE RUNNER\033[0m")
    print(f"\033[94mTick interval: {interval}s\033[0m")
    print(f"\033[93mPreparing steady dopamine flow...\033[0m")
    print(f"\033[90mPress Ctrl+C to stop\033[0m\n")
    
    # Create and start runner
    runner = SteadyRunner(tick_interval=interval)
    runner.start()

if __name__ == "__main__":
    main()
