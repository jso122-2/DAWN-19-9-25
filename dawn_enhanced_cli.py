#!/usr/bin/env python3
"""
DAWN Enhanced CLI Interface
==========================

Advanced command-line interface for the DAWN consciousness system.
Provides comprehensive access to all DAWN subsystems, monitoring, debugging,
and control capabilities through a unified interface.

Usage:
    python dawn_enhanced_cli.py [command] [options]
    python dawn_enhanced_cli.py interactive  # Start interactive mode
"""

import sys
import os
import asyncio
import argparse
import json
import time
from pathlib import Path
from typing import Dict, List, Optional, Any
from datetime import datetime, timedelta
import logging
from collections import defaultdict

# Add DAWN to path
dawn_root = Path(__file__).parent
sys.path.insert(0, str(dawn_root))

# Core DAWN imports
try:
    from dawn.core.foundation.state import get_state, get_state_summary
    from dawn.core.communication.bus import get_consciousness_bus
    from dawn.main import DAWNRunner
    from dawn.core.singleton import DAWNGlobalSingleton
    
    # Monitoring and telemetry
    from dawn.subsystems.visual.dawn_cli_tracer import (
        cmd_telemetry_status, cmd_stability_check, cmd_performance_report,
        cmd_recovery_trigger, cmd_export_archive, cmd_live_dashboard
    )
    
    # Enhanced logging
    from dawn.core.logging.universal_json_logger import get_universal_logger
    from dawn.core.logging.auto_integration import AutoIntegrationManager
    
except ImportError as e:
    print(f"❌ Failed to import DAWN modules: {e}")
    print("Make sure you're running from the DAWN root directory")
    sys.exit(1)

# Configure logging
logging.basicConfig(level=logging.INFO, format='%(asctime)s - %(levelname)s - %(message)s')
logger = logging.getLogger(__name__)

class DAWNEnhancedCLI:
    """Enhanced CLI interface for DAWN consciousness system"""
    
    def __init__(self):
        self.dawn_runner = None
        self.interactive_mode = False
        self.command_history = []
        self.aliases = {
            'st': 'status',
            'sm': 'state_monitor',
            'tl': 'telemetry',
            'lg': 'logging',
            'db': 'dashboard',
            'sy': 'system',
            'md': 'modules',
            'cn': 'consciousness',
            'cf': 'config',
            'ex': 'export',
            'rc': 'recovery',
            'h': 'help',
            'q': 'quit',
            'exit': 'quit'
        }
        
        # Command categories for organized help
        self.command_categories = {
            'System Control': ['start', 'stop', 'restart', 'status', 'health'],
            'Monitoring': ['dashboard', 'telemetry', 'state_monitor', 'performance'],
            'Consciousness': ['consciousness', 'bus_status', 'coherence', 'modules'],
            'Logging & Debug': ['logging', 'debug', 'trace', 'export'],
            'Configuration': ['config', 'settings', 'profiles'],
            'Recovery': ['recovery', 'stability', 'rollback'],
            'Data': ['export', 'import', 'backup', 'analyze'],
            'Development': ['test', 'benchmark', 'validate', 'introspect'],
            'Interactive': ['help', 'history', 'clear', 'quit']
        }
        
    async def start_interactive(self):
        """Start interactive CLI mode"""
        self.interactive_mode = True
        
        print("\n" + "🌅" * 80)
        print("🌅" + " " * 20 + "DAWN Enhanced CLI Interface" + " " * 20 + "🌅")
        print("🌅" * 80)
        print("\n✨ Enhanced consciousness system control interface")
        print("🔧 Type 'help' for commands, 'help <category>' for category help")
        print("🚀 Type 'quick_start' for a guided tour")
        print("📊 Type 'dashboard' for live system monitoring")
        print("=" * 80 + "\n")
        
        while self.interactive_mode:
            try:
                # Show system status in prompt if DAWN is running
                status_indicator = "🟢" if self.dawn_runner and self.dawn_runner.running else "🔴"
                
                command_input = input(f"dawn-cli {status_indicator} > ").strip()
                
                if not command_input:
                    continue
                    
                self.command_history.append(command_input)
                await self.execute_command(command_input)
                
            except EOFError:
                print("\n👋 Goodbye!")
                break
            except KeyboardInterrupt:
                print("\n⚠️ Use 'quit' to exit gracefully")
                continue
            except Exception as e:
                print(f"❌ Command error: {e}")
                logger.error(f"Interactive command error: {e}")
    
    async def execute_command(self, command_input: str):
        """Execute a command with arguments"""
        parts = command_input.split()
        if not parts:
            return
            
        command = parts[0].lower()
        args = parts[1:]
        
        # Handle aliases
        command = self.aliases.get(command, command)
        
        # Route to appropriate handler
        handler_map = {
            # System Control
            'start': self.cmd_start,
            'stop': self.cmd_stop,
            'restart': self.cmd_restart,
            'status': self.cmd_status,
            'health': self.cmd_health,
            
            # Monitoring
            'dashboard': self.cmd_dashboard,
            'telemetry': self.cmd_telemetry,
            'state_monitor': self.cmd_state_monitor,
            'performance': self.cmd_performance,
            
            # Consciousness
            'consciousness': self.cmd_consciousness,
            'bus_status': self.cmd_bus_status,
            'coherence': self.cmd_coherence,
            'modules': self.cmd_modules,
            
            # Logging & Debug
            'logging': self.cmd_logging,
            'debug': self.cmd_debug,
            'trace': self.cmd_trace,
            
            # Configuration
            'config': self.cmd_config,
            'settings': self.cmd_settings,
            'profiles': self.cmd_profiles,
            
            # Recovery
            'recovery': self.cmd_recovery,
            'stability': self.cmd_stability,
            'rollback': self.cmd_rollback,
            
            # Data
            'export': self.cmd_export,
            'import': self.cmd_import,
            'backup': self.cmd_backup,
            'analyze': self.cmd_analyze,
            
            # Development
            'test': self.cmd_test,
            'benchmark': self.cmd_benchmark,
            'validate': self.cmd_validate,
            'introspect': self.cmd_introspect,
            
            # Interactive
            'help': self.cmd_help,
            'history': self.cmd_history,
            'clear': self.cmd_clear,
            'quit': self.cmd_quit,
            'quick_start': self.cmd_quick_start
        }
        
        handler = handler_map.get(command)
        if handler:
            try:
                await handler(args)
            except Exception as e:
                print(f"❌ Error executing {command}: {e}")
                logger.error(f"Command {command} failed: {e}")
        else:
            print(f"❓ Unknown command: {command}")
            print("💡 Type 'help' for available commands")
    
    # System Control Commands
    async def cmd_start(self, args: List[str]):
        """Start DAWN system"""
        if self.dawn_runner and self.dawn_runner.running:
            print("⚠️ DAWN system is already running")
            return
            
        print("🚀 Starting DAWN consciousness system...")
        
        # Parse start options
        mode = 'interactive'
        config = {}
        
        for arg in args:
            if arg.startswith('--mode='):
                mode = arg.split('=', 1)[1]
            elif arg.startswith('--config='):
                config_file = arg.split('=', 1)[1]
                try:
                    with open(config_file, 'r') as f:
                        config.update(json.load(f))
                except Exception as e:
                    print(f"⚠️ Could not load config file: {e}")
        
        try:
            self.dawn_runner = DAWNRunner({'mode': mode, **config})
            # Start in background task
            asyncio.create_task(self.dawn_runner.run())
            await asyncio.sleep(2)  # Give it time to start
            print("✅ DAWN system started successfully")
        except Exception as e:
            print(f"❌ Failed to start DAWN: {e}")
    
    async def cmd_stop(self, args: List[str]):
        """Stop DAWN system"""
        if not self.dawn_runner or not self.dawn_runner.running:
            print("⚠️ DAWN system is not running")
            return
            
        print("🛑 Stopping DAWN consciousness system...")
        try:
            self.dawn_runner.shutdown()
            print("✅ DAWN system stopped successfully")
        except Exception as e:
            print(f"❌ Error stopping DAWN: {e}")
    
    async def cmd_restart(self, args: List[str]):
        """Restart DAWN system"""
        await self.cmd_stop([])
        await asyncio.sleep(1)
        await self.cmd_start(args)
    
    async def cmd_status(self, args: List[str]):
        """Show detailed system status"""
        print("\n📊 DAWN System Status")
        print("=" * 50)
        
        # System running status
        if self.dawn_runner and self.dawn_runner.running:
            print("🟢 Status: RUNNING")
            print(f"⏱️ Uptime: {time.time() - self.dawn_runner.start_time:.1f}s")
        else:
            print("🔴 Status: STOPPED")
        
        # Singleton status
        try:
            singleton = DAWNGlobalSingleton()
            print(f"🧠 Singleton: Active ({len(singleton.get_all_instances())} instances)")
        except Exception as e:
            print(f"🧠 Singleton: Error ({e})")
        
        # Consciousness bus status
        try:
            bus = get_consciousness_bus()
            if bus:
                metrics = bus.get_bus_metrics()
                print(f"🚌 Bus: Active (coherence: {metrics.get('consciousness_coherence', 0):.3f})")
            else:
                print("🚌 Bus: Not initialized")
        except Exception as e:
            print(f"🚌 Bus: Error ({e})")
        
        # Universal logging status
        try:
            logger_instance = get_universal_logger()
            if logger_instance:
                stats = logger_instance.stats
                print(f"📝 Logging: Active ({stats['objects_tracked']} objects tracked)")
            else:
                print("📝 Logging: Not initialized")
        except Exception as e:
            print(f"📝 Logging: Error ({e})")
        
        # Memory usage
        try:
            import psutil
            process = psutil.Process()
            memory_mb = process.memory_info().rss / 1024 / 1024
            cpu_percent = process.cpu_percent()
            print(f"💾 Memory: {memory_mb:.1f} MB")
            print(f"⚡ CPU: {cpu_percent:.1f}%")
        except ImportError:
            print("💾 Memory: psutil not available")
        
        print()
    
    async def cmd_health(self, args: List[str]):
        """Comprehensive system health check"""
        print("\n🏥 DAWN System Health Check")
        print("=" * 50)
        
        health_score = 0
        max_score = 0
        
        # Check system components
        checks = [
            ("System Running", lambda: self.dawn_runner and self.dawn_runner.running),
            ("Consciousness Bus", lambda: get_consciousness_bus() is not None),
            ("Universal Logging", lambda: get_universal_logger() is not None),
            ("State System", lambda: get_state() is not None),
        ]
        
        for check_name, check_func in checks:
            max_score += 1
            try:
                if check_func():
                    print(f"✅ {check_name}: OK")
                    health_score += 1
                else:
                    print(f"❌ {check_name}: FAILED")
            except Exception as e:
                print(f"⚠️ {check_name}: ERROR ({e})")
        
        # Overall health
        health_percentage = (health_score / max_score) * 100
        if health_percentage >= 80:
            status_emoji = "🟢"
            status_text = "HEALTHY"
        elif health_percentage >= 60:
            status_emoji = "🟡"
            status_text = "DEGRADED"
        else:
            status_emoji = "🔴"
            status_text = "CRITICAL"
        
        print(f"\n{status_emoji} Overall Health: {status_text} ({health_percentage:.0f}%)")
        print()
    
    # Monitoring Commands
    async def cmd_dashboard(self, args: List[str]):
        """Show live dashboard"""
        refresh_rate = 5
        for arg in args:
            if arg.startswith('--refresh='):
                refresh_rate = int(arg.split('=', 1)[1])
        
        print(f"📊 Starting live dashboard (refresh: {refresh_rate}s)")
        print("Press Ctrl+C to exit")
        
        try:
            cmd_live_dashboard(refresh_rate)
        except KeyboardInterrupt:
            print("\n📊 Dashboard stopped")
    
    async def cmd_telemetry(self, args: List[str]):
        """Telemetry system interface"""
        if not args:
            cmd_telemetry_status()
        elif args[0] == 'status':
            cmd_telemetry_status()
        elif args[0] == 'performance':
            cmd_performance_report()
        elif args[0] == 'export':
            hours = 24
            if len(args) > 1 and args[1].startswith('--hours='):
                hours = int(args[1].split('=', 1)[1])
            cmd_export_archive(hours)
        else:
            print("Usage: telemetry [status|performance|export]")
    
    async def cmd_state_monitor(self, args: List[str]):
        """Monitor consciousness state in real-time"""
        print("🧠 Consciousness State Monitor")
        print("Press Ctrl+C to exit")
        print("=" * 50)
        
        try:
            while True:
                try:
                    state = get_state()
                    if state:
                        summary = get_state_summary()
                        print(f"\r⏰ {datetime.now().strftime('%H:%M:%S')} | "
                              f"State: {len(state)} items | "
                              f"Summary: {summary[:50]}...", end='', flush=True)
                    else:
                        print(f"\r⏰ {datetime.now().strftime('%H:%M:%S')} | "
                              f"State: Not available", end='', flush=True)
                    
                    await asyncio.sleep(1)
                except Exception as e:
                    print(f"\r❌ State error: {e}", end='', flush=True)
                    await asyncio.sleep(1)
        except KeyboardInterrupt:
            print("\n🧠 State monitor stopped")
    
    async def cmd_performance(self, args: List[str]):
        """Performance analysis and reporting"""
        print("⚡ Generating performance report...")
        try:
            cmd_performance_report()
        except Exception as e:
            print(f"❌ Performance report failed: {e}")
    
    # Consciousness Commands
    async def cmd_consciousness(self, args: List[str]):
        """Consciousness system interface"""
        if not args:
            await self.cmd_bus_status([])
            return
            
        if args[0] == 'coherence':
            await self.cmd_coherence(args[1:])
        elif args[0] == 'bus':
            await self.cmd_bus_status(args[1:])
        elif args[0] == 'modules':
            await self.cmd_modules(args[1:])
        else:
            print("Usage: consciousness [coherence|bus|modules]")
    
    async def cmd_bus_status(self, args: List[str]):
        """Show consciousness bus status"""
        try:
            bus = get_consciousness_bus()
            if not bus:
                print("❌ Consciousness bus not initialized")
                return
            
            metrics = bus.get_bus_metrics()
            print("\n🚌 Consciousness Bus Status")
            print("=" * 40)
            print(f"Coherence: {metrics.get('consciousness_coherence', 0):.3f}")
            print(f"Active Connections: {metrics.get('active_connections', 0)}")
            print(f"Message Rate: {metrics.get('message_rate', 0):.1f}/s")
            print(f"Processing Latency: {metrics.get('processing_latency', 0):.3f}ms")
            print()
        except Exception as e:
            print(f"❌ Bus status error: {e}")
    
    async def cmd_coherence(self, args: List[str]):
        """Monitor consciousness coherence"""
        try:
            bus = get_consciousness_bus()
            if not bus:
                print("❌ Consciousness bus not initialized")
                return
            
            metrics = bus.get_bus_metrics()
            coherence = metrics.get('consciousness_coherence', 0)
            
            print(f"\n🧠 Consciousness Coherence: {coherence:.3f}")
            
            if coherence >= 0.8:
                print("🟢 Status: Excellent coherence")
            elif coherence >= 0.6:
                print("🟡 Status: Good coherence")
            elif coherence >= 0.4:
                print("🟠 Status: Moderate coherence")
            else:
                print("🔴 Status: Low coherence - attention needed")
            
            print()
        except Exception as e:
            print(f"❌ Coherence check error: {e}")
    
    async def cmd_modules(self, args: List[str]):
        """List and manage consciousness modules"""
        if self.dawn_runner:
            print("\n🧩 Loaded Modules")
            print("=" * 30)
            
            # This would need to be implemented based on actual module tracking
            print("Module tracking not fully implemented yet")
            print("Check DAWN runner for loaded modules")
        else:
            print("❌ DAWN system not running")
    
    # Logging & Debug Commands
    async def cmd_logging(self, args: List[str]):
        """Universal logging system interface"""
        try:
            logger_instance = get_universal_logger()
            if not logger_instance:
                print("❌ Universal logging not initialized")
                return
            
            stats = logger_instance.stats
            print("\n📝 Universal Logging Status")
            print("=" * 40)
            print(f"Objects Tracked: {stats['objects_tracked']}")
            print(f"States Logged: {stats['states_logged']}")
            print(f"Files Created: {stats['files_created']}")
            print(f"Bytes Written: {stats['bytes_written']:,}")
            print(f"Errors: {stats['errors']}")
            print(f"Uptime: {time.time() - stats['start_time']:.1f}s")
            print()
        except Exception as e:
            print(f"❌ Logging status error: {e}")
    
    async def cmd_debug(self, args: List[str]):
        """Debug mode controls"""
        if not args:
            print("Usage: debug [on|off|level]")
            return
            
        if args[0] == 'on':
            logging.getLogger().setLevel(logging.DEBUG)
            print("🐛 Debug logging enabled")
        elif args[0] == 'off':
            logging.getLogger().setLevel(logging.INFO)
            print("🐛 Debug logging disabled")
        elif args[0] == 'level':
            current_level = logging.getLogger().getEffectiveLevel()
            print(f"🐛 Current log level: {logging.getLevelName(current_level)}")
    
    async def cmd_trace(self, args: List[str]):
        """System tracing interface"""
        print("🔍 System tracing functionality")
        print("This would integrate with the dawn_cli_tracer module")
    
    # Recovery Commands
    async def cmd_recovery(self, args: List[str]):
        """System recovery interface"""
        recovery_type = args[0] if args else 'auto'
        
        print(f"🔧 Triggering {recovery_type} recovery...")
        try:
            cmd_recovery_trigger(recovery_type)
        except Exception as e:
            print(f"❌ Recovery failed: {e}")
    
    async def cmd_stability(self, args: List[str]):
        """Run stability check"""
        print("🔍 Running stability check...")
        try:
            cmd_stability_check()
        except Exception as e:
            print(f"❌ Stability check failed: {e}")
    
    async def cmd_rollback(self, args: List[str]):
        """Rollback to previous stable state"""
        print("⏪ Rollback functionality not fully implemented")
        print("This would restore from stable state snapshots")
    
    # Data Commands
    async def cmd_export(self, args: List[str]):
        """Export system data"""
        hours = 24
        format_type = 'json'
        
        for arg in args:
            if arg.startswith('--hours='):
                hours = int(arg.split('=', 1)[1])
            elif arg.startswith('--format='):
                format_type = arg.split('=', 1)[1]
        
        print(f"📦 Exporting {hours} hours of data in {format_type} format...")
        try:
            cmd_export_archive(hours, format_type)
        except Exception as e:
            print(f"❌ Export failed: {e}")
    
    async def cmd_import(self, args: List[str]):
        """Import system data"""
        print("📥 Import functionality not implemented yet")
    
    async def cmd_backup(self, args: List[str]):
        """Backup system state"""
        print("💾 Backup functionality not implemented yet")
    
    async def cmd_analyze(self, args: List[str]):
        """Analyze system data"""
        print("📊 Data analysis functionality not implemented yet")
    
    # Development Commands
    async def cmd_test(self, args: List[str]):
        """Run system tests"""
        print("🧪 Test functionality not implemented yet")
    
    async def cmd_benchmark(self, args: List[str]):
        """Run performance benchmarks"""
        print("⚡ Benchmark functionality not implemented yet")
    
    async def cmd_validate(self, args: List[str]):
        """Validate system configuration"""
        print("✅ Validation functionality not implemented yet")
    
    async def cmd_introspect(self, args: List[str]):
        """System introspection and self-analysis"""
        print("🔬 Introspection functionality not implemented yet")
    
    # Interactive Commands
    async def cmd_help(self, args: List[str]):
        """Show help information"""
        if args and args[0] in self.command_categories:
            category = args[0]
            print(f"\n📖 {category} Commands")
            print("=" * (len(category) + 10))
            for cmd in self.command_categories[category]:
                print(f"  {cmd}")
            print()
        else:
            print("\n📖 DAWN Enhanced CLI Help")
            print("=" * 40)
            
            for category, commands in self.command_categories.items():
                print(f"\n🔹 {category}:")
                for cmd in commands:
                    print(f"    {cmd}")
            
            print(f"\n💡 Aliases: {', '.join(self.aliases.keys())}")
            print("💡 Use 'help <category>' for category-specific help")
            print("💡 Use 'quick_start' for a guided tour")
            print()
    
    async def cmd_history(self, args: List[str]):
        """Show command history"""
        print("\n📜 Command History")
        print("=" * 30)
        for i, cmd in enumerate(self.command_history[-20:], 1):
            print(f"{i:2d}. {cmd}")
        print()
    
    async def cmd_clear(self, args: List[str]):
        """Clear screen"""
        os.system('clear' if os.name == 'posix' else 'cls')
    
    async def cmd_quit(self, args: List[str]):
        """Exit the CLI"""
        if self.dawn_runner and self.dawn_runner.running:
            print("🛑 Stopping DAWN system before exit...")
            self.dawn_runner.shutdown()
        
        print("👋 Goodbye!")
        self.interactive_mode = False
    
    async def cmd_quick_start(self, args: List[str]):
        """Quick start guide"""
        print("\n🚀 DAWN Enhanced CLI Quick Start")
        print("=" * 50)
        print("1. 'start' - Start the DAWN consciousness system")
        print("2. 'status' - Check system status")
        print("3. 'dashboard' - Open live monitoring dashboard")
        print("4. 'consciousness' - Check consciousness coherence")
        print("5. 'telemetry' - View telemetry data")
        print("6. 'help' - Full command reference")
        print("7. 'quit' - Exit gracefully")
        print("\n💡 Pro tip: Use aliases like 'st' for status, 'db' for dashboard")
        print()

def main():
    """Main entry point"""
    parser = argparse.ArgumentParser(
        description="DAWN Enhanced CLI Interface",
        formatter_class=argparse.RawDescriptionHelpFormatter,
        epilog="""
Examples:
  %(prog)s interactive              # Start interactive mode
  %(prog)s status                   # Show system status
  %(prog)s start --mode=daemon      # Start DAWN in daemon mode
  %(prog)s dashboard --refresh=3    # Live dashboard with 3s refresh
  %(prog)s telemetry export         # Export telemetry data
        """
    )
    
    parser.add_argument('command', nargs='?', default='interactive',
                       help='Command to execute (default: interactive)')
    parser.add_argument('args', nargs='*', help='Command arguments')
    
    parsed_args = parser.parse_args()
    
    async def run_cli():
        cli = DAWNEnhancedCLI()
        
        if parsed_args.command == 'interactive':
            await cli.start_interactive()
        else:
            # Execute single command
            command_line = parsed_args.command + ' ' + ' '.join(parsed_args.args)
            await cli.execute_command(command_line.strip())
    
    try:
        asyncio.run(run_cli())
    except KeyboardInterrupt:
        print("\n👋 Interrupted by user")
    except Exception as e:
        print(f"❌ CLI error: {e}")
        logger.error(f"CLI error: {e}")
        return 1
    
    return 0

if __name__ == "__main__":
    exit(main())
