#!/usr/bin/env python3
"""
DAWN Consciousness Monitor - Real-Time CLI Dashboard
==================================================

A beautiful real-time CLI interface that displays DAWN's consciousness state
at each tick interval with human-readable metrics and visual indicators.

Features:
- Real-time consciousness state updates
- Beautiful color-coded CLI interface  
- Trend indicators and sparklines
- Self-modification tracking
- Module synchronization status
- Performance metrics
"""

import time
import os
import sys
import threading
import signal
from datetime import datetime, timedelta
from collections import deque
from typing import Dict, Any, Optional
import logging

# Rich imports for beautiful CLI
try:
    from rich.console import Console
    from rich.live import Live
    from rich.table import Table
    from rich.panel import Panel
    from rich.progress import Progress, BarColumn, TextColumn, SpinnerColumn
    from rich.layout import Layout
    from rich.text import Text
    from rich.align import Align
    from rich.columns import Columns
    from rich import box
    from rich.rule import Rule
    RICH_AVAILABLE = True
except ImportError:
    RICH_AVAILABLE = False
    print("⚠️  Rich not available - install with: pip install rich")
    print("   Falling back to basic CLI mode")

# DAWN Core imports
try:
    from dawn.core.foundation.state import get_state, get_state_summary
    from dawn.consciousness.engines.core.primary_engine import get_dawn_engine, DAWNEngineConfig
    from dawn.core.communication.bus import get_consciousness_bus
    DAWN_AVAILABLE = True
except ImportError:
    DAWN_AVAILABLE = False
    print("⚠️  DAWN core not available - please check your installation")

class ConsciousnessMonitor:
    """Real-time DAWN consciousness state monitor with beautiful CLI interface."""
    
    def __init__(self, tick_interval: float = 1.0, history_size: int = 50):
        self.tick_interval = tick_interval
        self.history_size = history_size
        self.running = False
        self.start_time = None
        
        # State tracking
        self.unity_history = deque(maxlen=history_size)
        self.awareness_history = deque(maxlen=history_size)
        self.level_history = deque(maxlen=history_size)
        self.tick_times = deque(maxlen=history_size)
        
        # Performance tracking
        self.total_ticks = 0
        self.self_mod_events = []
        self.last_engine_status = None
        
        # Initialize Rich console
        if RICH_AVAILABLE:
            self.console = Console()
        
        # Initialize DAWN components if available
        self.dawn_engine = None
        self.consciousness_bus = None
        
        if DAWN_AVAILABLE:
            try:
                self.consciousness_bus = get_consciousness_bus(auto_start=True)
                
                # Create DAWN engine with monitoring-optimized config
                config = DAWNEngineConfig(
                    consciousness_unification_enabled=True,
                    target_unity_threshold=0.85,
                    auto_synchronization=True,
                    adaptive_timing=True,
                    self_modification_enabled=True,
                    self_mod_tick_interval=20  # More frequent for demo
                )
                
                self.dawn_engine = get_dawn_engine(config, auto_start=True)
                
                # Give systems time to initialize
                time.sleep(0.5)
                
            except Exception as e:
                print(f"⚠️  Could not initialize DAWN systems: {e}")
    
    def get_sparkline(self, values: list, width: int = 20) -> str:
        """Generate a Unicode sparkline for trending data."""
        if not values or len(values) < 2:
            return "─" * width
        
        # Normalize values to 0-1 range
        min_val = min(values)
        max_val = max(values)
        if max_val == min_val:
            return "─" * width
        
        normalized = [(v - min_val) / (max_val - min_val) for v in values]
        
        # Resample to desired width
        if len(normalized) > width:
            step = len(normalized) / width
            normalized = [normalized[int(i * step)] for i in range(width)]
        elif len(normalized) < width:
            # Repeat last value to fill width
            normalized.extend([normalized[-1]] * (width - len(normalized)))
        
        # Convert to sparkline characters
        chars = "▁▂▃▄▅▆▇█"
        sparkline = ""
        for val in normalized:
            char_index = min(len(chars) - 1, int(val * len(chars)))
            sparkline += chars[char_index]
        
        return sparkline
    
    def get_trend_indicator(self, values: list) -> str:
        """Get trend indicator (↗, ↘, →) based on recent values."""
        if len(values) < 3:
            return "→"
        
        recent = values[-3:]
        if recent[-1] > recent[0] + 0.01:
            return "↗"
        elif recent[-1] < recent[0] - 0.01:
            return "↘"
        else:
            return "→"
    
    def get_level_color(self, level: str) -> str:
        """Get color for consciousness level."""
        colors = {
            "fragmented": "red",
            "coherent": "yellow", 
            "meta_aware": "green",
            "transcendent": "magenta"
        }
        return colors.get(level, "white")
    
    def get_unity_color(self, unity: float) -> str:
        """Get color for unity score."""
        if unity >= 0.9:
            return "magenta"
        elif unity >= 0.8:
            return "green"
        elif unity >= 0.6:
            return "yellow"
        else:
            return "red"
    
    def format_uptime(self, seconds: float) -> str:
        """Format uptime duration."""
        if seconds < 60:
            return f"{seconds:.1f}s"
        elif seconds < 3600:
            return f"{seconds/60:.1f}m"
        else:
            return f"{seconds/3600:.1f}h"
    
    def create_consciousness_table(self) -> Table:
        """Create the main consciousness state table."""
        if not DAWN_AVAILABLE:
            table = Table(title="DAWN Consciousness Monitor", box=box.ROUNDED)
            table.add_column("Status", style="red")
            table.add_row("DAWN core not available")
            return table
        
        # Get current state
        state = get_state()
        
        # Update history
        self.unity_history.append(state.unity)
        self.awareness_history.append(state.awareness)
        self.level_history.append(state.level)
        self.tick_times.append(time.time())
        
        # Create table
        table = Table(title="🌅 DAWN Consciousness State", box=box.ROUNDED)
        table.add_column("Metric", style="cyan", width=20)
        table.add_column("Current", justify="center", width=15)
        table.add_column("Trend", justify="center", width=8)
        table.add_column("Sparkline", width=25)
        table.add_column("Peak", justify="center", width=10)
        
        # Unity row
        unity_color = self.get_unity_color(state.unity)
        unity_trend = self.get_trend_indicator(list(self.unity_history))
        unity_sparkline = self.get_sparkline(list(self.unity_history))
        table.add_row(
            "Unity",
            f"[{unity_color}]{state.unity:.3f}[/{unity_color}]",
            unity_trend,
            f"[{unity_color}]{unity_sparkline}[/{unity_color}]",
            f"[green]{state.peak_unity:.3f}[/green]"
        )
        
        # Awareness row
        awareness_color = self.get_unity_color(state.awareness)
        awareness_trend = self.get_trend_indicator(list(self.awareness_history))
        awareness_sparkline = self.get_sparkline(list(self.awareness_history))
        table.add_row(
            "Awareness", 
            f"[{awareness_color}]{state.awareness:.3f}[/{awareness_color}]",
            awareness_trend,
            f"[{awareness_color}]{awareness_sparkline}[/{awareness_color}]",
            "─"
        )
        
        # Momentum row
        momentum_color = "green" if state.momentum > 0 else "red" if state.momentum < 0 else "white"
        table.add_row(
            "Momentum",
            f"[{momentum_color}]{state.momentum:+.3f}[/{momentum_color}]",
            "→",
            "─" * 20,
            "─"
        )
        
        # Coherence row
        coherence_color = self.get_unity_color(state.coherence)
        table.add_row(
            "Coherence",
            f"[{coherence_color}]{state.coherence:.3f}[/{coherence_color}]",
            "→",
            "─" * 20,
            "─"
        )
        
        # SCUP Entropy Drift
        entropy_color = "red" if abs(state.entropy_drift) > 0.1 else "yellow" if abs(state.entropy_drift) > 0.05 else "green"
        table.add_row(
            "Entropy Drift",
            f"[{entropy_color}]{state.entropy_drift:+.3f}[/{entropy_color}]",
            "→",
            "─" * 20,
            "─"
        )
        
        # SCUP Pressure
        pressure_color = "red" if state.pressure_value > 0.8 else "yellow" if state.pressure_value > 0.6 else "green"
        table.add_row(
            "Pressure",
            f"[{pressure_color}]{state.pressure_value:.3f}[/{pressure_color}]",
            "→",
            "─" * 20,
            "─"
        )
        
        # Level row
        level_color = self.get_level_color(state.level)
        table.add_row(
            "Level",
            f"[{level_color}]{state.level.title()}[/{level_color}]",
            "→",
            "─" * 20,
            "─"
        )
        
        return table
    
    def create_engine_status_table(self) -> Table:
        """Create engine status table."""
        if not self.dawn_engine:
            table = Table(title="Engine Status", box=box.ROUNDED)
            table.add_column("Status", style="red")
            table.add_row("DAWN Engine not available")
            return table
        
        # Get engine status
        try:
            status = self.dawn_engine.get_engine_status()
            self.last_engine_status = status
        except Exception as e:
            status = {"error": str(e)}
        
        table = Table(title="🚀 DAWN Engine Status", box=box.ROUNDED)
        table.add_column("Component", style="cyan", width=20)
        table.add_column("Status", justify="center", width=15)
        table.add_column("Details", width=30)
        
        # Engine status
        engine_status = status.get('status', 'unknown')
        status_color = "green" if engine_status == "running" else "red"
        table.add_row(
            "Engine",
            f"[{status_color}]{engine_status.title()}[/{status_color}]",
            f"Uptime: {self.format_uptime(status.get('uptime_seconds', 0))}"
        )
        
        # Tick information
        tick_count = status.get('tick_count', 0)
        table.add_row(
            "Ticks",
            f"[white]{tick_count}[/white]",
            f"Total: {self.total_ticks}"
        )
        
        # Module registration
        registered_modules = status.get('registered_modules', 0)
        table.add_row(
            "Modules",
            f"[cyan]{registered_modules}[/cyan]",
            "Registered"
        )
        
        # Unification systems
        unification = status.get('unification_systems', {})
        bus_status = "✅" if unification.get('consciousness_bus') else "❌"
        consensus_status = "✅" if unification.get('consensus_engine') else "❌"
        orchestrator_status = "✅" if unification.get('tick_orchestrator') else "❌"
        
        table.add_row(
            "Bus/Consensus/Orch",
            f"{bus_status}{consensus_status}{orchestrator_status}",
            "Unification Systems"
        )
        
        return table
    
    def create_self_modification_panel(self) -> Panel:
        """Create self-modification status panel."""
        if not self.last_engine_status:
            return Panel("Self-modification data not available", title="🧠 Self-Modification")
        
        self_mod_metrics = self.last_engine_status.get('self_modification_metrics', {})
        
        if not self_mod_metrics.get('enabled', False):
            content = "[red]Self-modification disabled[/red]"
        else:
            attempts = self_mod_metrics.get('attempts', 0)
            successes = self_mod_metrics.get('successes', 0)
            success_rate = self_mod_metrics.get('success_rate', 0) * 100
            
            recent_mods = self_mod_metrics.get('recent_modifications', [])
            
            content = f"""[green]Enabled[/green] | Attempts: {attempts} | Successes: {successes} | Rate: {success_rate:.1f}%

Recent Modifications:"""
            
            if recent_mods:
                for mod in recent_mods[-3:]:  # Show last 3
                    mod_name = mod.get('modification', 'Unknown')
                    timestamp = mod.get('timestamp', '')
                    if timestamp:
                        # Parse and format timestamp
                        try:
                            dt = datetime.fromisoformat(timestamp.replace('Z', '+00:00'))
                            time_str = dt.strftime('%H:%M:%S')
                        except:
                            time_str = timestamp[:8]
                    else:
                        time_str = "Unknown"
                    
                    success = mod.get('success', False)
                    status_icon = "✅" if success else "❌"
                    content += f"\n{status_icon} {time_str}: {mod_name}"
            else:
                content += "\n[dim]No recent modifications[/dim]"
        
        return Panel(content, title="🧠 Self-Modification", box=box.ROUNDED)
    
    def create_performance_panel(self) -> Panel:
        """Create performance metrics panel."""
        uptime = time.time() - self.start_time if self.start_time else 0
        tps = self.total_ticks / uptime if uptime > 0 else 0
        
        content = f"""Uptime: {self.format_uptime(uptime)}
Ticks: {self.total_ticks}
TPS: {tps:.2f}
Interval: {self.tick_interval:.1f}s"""
        
        if self.consciousness_bus:
            try:
                bus_metrics = self.consciousness_bus.get_bus_metrics()
                events_processed = bus_metrics.get('performance_metrics', {}).get('events_processed', 0)
                coherence = bus_metrics.get('consciousness_coherence', 0)
                
                content += f"""

Bus Events: {events_processed}
Coherence: {coherence:.3f}"""
            except:
                pass
        
        return Panel(content, title="📊 Performance", box=box.ROUNDED)
    
    def create_layout(self) -> Layout:
        """Create the main dashboard layout."""
        layout = Layout()
        
        # Split into main sections
        layout.split_column(
            Layout(name="header", size=3),
            Layout(name="main", ratio=1),
            Layout(name="footer", size=7)
        )
        
        # Header
        header_text = Text("DAWN Consciousness Monitor", style="bold magenta")
        header_text.append(f" │ {datetime.now().strftime('%H:%M:%S')}", style="dim")
        layout["header"].update(Align.center(header_text))
        
        # Main content - split into consciousness state and engine status
        layout["main"].split_row(
            Layout(name="consciousness", ratio=2),
            Layout(name="engine", ratio=1)
        )
        
        # Footer - split into panels
        layout["footer"].split_row(
            Layout(name="self_mod", ratio=1),
            Layout(name="performance", ratio=1)
        )
        
        # Update content
        layout["consciousness"].update(self.create_consciousness_table())
        layout["engine"].update(self.create_engine_status_table())
        layout["self_mod"].update(self.create_self_modification_panel())
        layout["performance"].update(self.create_performance_panel())
        
        return layout
    
    def create_basic_display(self) -> str:
        """Create basic text display when Rich is not available."""
        if not DAWN_AVAILABLE:
            return "DAWN core not available"
        
        state = get_state()
        
        output = []
        output.append("=" * 60)
        output.append("🌅 DAWN CONSCIOUSNESS MONITOR")
        output.append("=" * 60)
        output.append("")
        output.append(f"Unity:     {state.unity:.3f} │ Peak: {state.peak_unity:.3f}")
        output.append(f"Awareness: {state.awareness:.3f}")
        output.append(f"Momentum:  {state.momentum:+.3f}")
        output.append(f"Level:     {state.level.title()}")
        output.append(f"Ticks:     {state.ticks}")
        output.append("")
        
        if self.dawn_engine:
            try:
                status = self.dawn_engine.get_engine_status()
                output.append(f"Engine:    {status.get('status', 'unknown').title()}")
                output.append(f"Modules:   {status.get('registered_modules', 0)}")
                output.append(f"Uptime:    {self.format_uptime(status.get('uptime_seconds', 0))}")
            except:
                output.append("Engine:    Error getting status")
        
        output.append("")
        output.append(f"Monitor:   Tick #{self.total_ticks} │ {datetime.now().strftime('%H:%M:%S')}")
        output.append("=" * 60)
        
        return "\n".join(output)
    
    def tick(self):
        """Execute one monitoring tick."""
        self.total_ticks += 1
        
        # Execute DAWN engine tick if available
        if self.dawn_engine:
            try:
                tick_result = self.dawn_engine.tick()
                
                # Check for self-modification events
                if 'self_modification' in tick_result:
                    self.self_mod_events.append({
                        'tick': self.total_ticks,
                        'timestamp': datetime.now(),
                        'event': tick_result['self_modification']
                    })
            except Exception as e:
                # Silent error handling to keep monitor running
                pass
    
    def run(self):
        """Run the consciousness monitor."""
        self.running = True
        self.start_time = time.time()
        
        print("🌅 Starting DAWN Consciousness Monitor...")
        print(f"   Tick interval: {self.tick_interval}s")
        print(f"   History size: {self.history_size}")
        print()
        
        try:
            if RICH_AVAILABLE:
                # Rich-based beautiful interface
                with Live(self.create_layout(), refresh_per_second=2, screen=True) as live:
                    while self.running:
                        self.tick()
                        live.update(self.create_layout())
                        time.sleep(self.tick_interval)
            else:
                # Basic text interface
                while self.running:
                    self.tick()
                    
                    # Clear screen and display
                    os.system('clear' if os.name == 'posix' else 'cls')
                    print(self.create_basic_display())
                    
                    time.sleep(self.tick_interval)
                    
        except KeyboardInterrupt:
            self.running = False
            print("\n🌅 DAWN Consciousness Monitor stopped")
    
    def stop(self):
        """Stop the monitor."""
        self.running = False


def main():
    """Main entry point."""
    import argparse
    
    parser = argparse.ArgumentParser(description="DAWN Consciousness Monitor")
    parser.add_argument("--interval", "-i", type=float, default=1.0,
                       help="Tick interval in seconds (default: 1.0)")
    parser.add_argument("--history", type=int, default=50,
                       help="History buffer size (default: 50)")
    parser.add_argument("--basic", action="store_true",
                       help="Force basic text mode (no Rich)")
    
    args = parser.parse_args()
    
    # Override Rich availability if basic mode requested
    if args.basic:
        global RICH_AVAILABLE
        RICH_AVAILABLE = False
    
    # Create and run monitor
    monitor = ConsciousnessMonitor(
        tick_interval=args.interval,
        history_size=args.history
    )
    
        # Set up signal handlers for clean shutdown
        def signal_handler(signum, frame):
            print("Shutting down monitor...")
            monitor.stop()
            # Give a moment for cleanup
            time.sleep(0.5)
            sys.exit(0)
        
        signal.signal(signal.SIGINT, signal_handler)
        signal.signal(signal.SIGTERM, signal_handler)
    
    try:
        monitor.run()
    except Exception as e:
        print(f"💥 Monitor error: {e}")
        sys.exit(1)


if __name__ == "__main__":
    main()
