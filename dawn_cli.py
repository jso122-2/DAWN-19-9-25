#!/usr/bin/env python3
"""
🌅💻 DAWN Command Line Interface
===============================

Comprehensive CLI for all DAWN systems implemented in this chat session:
- Universal JSON logging system
- Centralized deep repository
- Consciousness-depth logging (8-level hierarchy)
- Sigil consciousness logging
- Pulse-telemetry unification
- Mycelial semantic hash map with spore propagation
- Recursive self-writing capabilities
- Live monitoring and visualization
- Complete DAWN singleton integration

Usage:
    python3 dawn_cli.py <command> [options]
    
Commands:
    init            - Initialize all DAWN systems
    status          - Show comprehensive system status
    logging         - Universal logging operations
    consciousness   - Consciousness-depth logging operations
    sigil           - Sigil consciousness operations
    pulse           - Pulse-telemetry operations
    mycelial        - Mycelial network operations
    recursive       - Recursive self-writing operations
    monitor         - Live monitoring operations
    singleton       - DAWN singleton operations
    help            - Show detailed help
"""

import sys
import os
import argparse
import json
import time
from pathlib import Path
from datetime import datetime
from typing import Dict, List, Any, Optional

# Add DAWN to path
dawn_root = Path(__file__).parent
sys.path.insert(0, str(dawn_root))

# Color codes for CLI output
class Colors:
    HEADER = '\033[95m'
    BLUE = '\033[94m'
    CYAN = '\033[96m'
    GREEN = '\033[92m'
    YELLOW = '\033[93m'
    RED = '\033[91m'
    BOLD = '\033[1m'
    UNDERLINE = '\033[4m'
    END = '\033[0m'

def colored(text: str, color: str) -> str:
    """Apply color to text."""
    return f"{color}{text}{Colors.END}"

def print_header(text: str):
    """Print colored header."""
    print(colored(f"🌅 {text}", Colors.BOLD + Colors.CYAN))

def print_success(text: str):
    """Print success message."""
    print(colored(f"✅ {text}", Colors.GREEN))

def print_warning(text: str):
    """Print warning message."""
    print(colored(f"⚠️  {text}", Colors.YELLOW))

def print_error(text: str):
    """Print error message."""
    print(colored(f"❌ {text}", Colors.RED))

def print_info(text: str):
    """Print info message."""
    print(colored(f"ℹ️  {text}", Colors.BLUE))

class DAWNCLIError(Exception):
    """Custom exception for CLI errors."""
    pass

class DAWNCLI:
    """Main DAWN CLI controller."""
    
    def __init__(self):
        """Initialize DAWN CLI."""
        self.dawn_singleton = None
        self.systems_available = {
            'singleton': False,
            'logging': False,
            'consciousness': False,
            'sigil': False,
            'pulse': False,
            'mycelial': False,
            'recursive': False,
            'monitor': False
        }
        
        # Initialize DAWN systems
        self._initialize_dawn_systems()
    
    def _initialize_dawn_systems(self):
        """Initialize DAWN systems for CLI access."""
        try:
            # Set quiet mode for CLI usage
            import os
            os.environ['DAWN_CLI_MODE'] = '1'
            
            # Import and initialize DAWN singleton
            from dawn.core.singleton import get_dawn
            self.dawn_singleton = get_dawn()
            self.systems_available['singleton'] = True
            
            # Check logging systems
            try:
                from dawn.core.logging import (
                    get_universal_logger, get_centralized_repository,
                    get_consciousness_repository, get_sigil_consciousness_logger,
                    get_pulse_telemetry_bridge, get_mycelial_hashmap,
                    get_recursive_self_writing_integrator
                )
                self.systems_available['logging'] = True
                self.systems_available['consciousness'] = True
                self.systems_available['sigil'] = True
                self.systems_available['pulse'] = True
                self.systems_available['mycelial'] = True
                self.systems_available['recursive'] = True
            except ImportError:
                pass
            
            # Check monitor system
            try:
                from live_monitor import LiveDAWNMonitor
                self.systems_available['monitor'] = True
            except ImportError:
                pass
                
        except Exception as e:
            print_warning(f"DAWN systems initialization incomplete: {e}")
    
    def cmd_init(self, args):
        """Initialize all DAWN systems."""
        print_header("DAWN SYSTEM INITIALIZATION")
        
        try:
            if not self.dawn_singleton:
                raise DAWNCLIError("DAWN singleton not available")
            
            print_info("Initializing DAWN singleton...")
            
            # Initialize singleton (this initializes all systems)
            import asyncio
            
            async def init_dawn():
                return await self.dawn_singleton.initialize()
            
            success = asyncio.run(init_dawn())
            
            if success:
                print_success("DAWN singleton initialized")
                
                # Get system status
                status = self.dawn_singleton.get_complete_system_status()
                
                print_info("System Status:")
                print(f"  Initialized: {status.get('initialized', False)}")
                print(f"  Complete logging: {status.get('complete_logging_initialized', False)}")
                print(f"  Mycelial integration: {status.get('mycelial_integration_active', False)}")
                print(f"  All systems integrated: {status.get('all_systems_integrated', False)}")
                
                # Show available systems
                logging_systems = status.get('logging_systems', {})
                if logging_systems:
                    print_info("Logging Systems:")
                    for system, active in logging_systems.items():
                        status_icon = "✅" if active else "⚪"
                        print(f"  {status_icon} {system}")
                
                mycelial_systems = status.get('mycelial_systems', {})
                if mycelial_systems:
                    print_info("Mycelial Systems:")
                    print(f"  Hash Map: {'✅' if mycelial_systems.get('hashmap_active') else '⚪'}")
                    print(f"  Integration: {'✅' if mycelial_systems.get('integration_active') else '⚪'}")
                
                print_success("DAWN initialization complete")
            else:
                print_error("DAWN initialization failed")
                
        except Exception as e:
            print_error(f"Initialization failed: {e}")
    
    def cmd_status(self, args):
        """Show comprehensive system status."""
        print_header("DAWN SYSTEM STATUS")
        
        try:
            if not self.dawn_singleton:
                print_error("DAWN singleton not available")
                return
            
            # Get complete system status
            status = self.dawn_singleton.get_complete_system_status()
            
            # Core status
            print_info("Core System:")
            print(f"  Initialized: {'✅' if status.get('initialized') else '❌'}")
            print(f"  Running: {'✅' if status.get('running') else '❌'}")
            print(f"  Mode: {status.get('mode', 'unknown')}")
            
            startup_time = status.get('startup_time')
            if startup_time:
                print(f"  Startup: {startup_time}")
            
            # Components status
            components = status.get('components_loaded', {})
            if components:
                active_components = sum(1 for active in components.values() if active)
                print(f"  Components: {active_components}/{len(components)} loaded")
            
            # Logging systems
            logging_systems = status.get('logging_systems', {})
            if logging_systems:
                print_info("Logging Systems:")
                for system, active in logging_systems.items():
                    status_icon = "✅" if active else "❌"
                    print(f"  {status_icon} {system.replace('_', ' ').title()}")
            
            # Mycelial systems
            mycelial_systems = status.get('mycelial_systems', {})
            if mycelial_systems:
                print_info("Mycelial Network:")
                print(f"  Hash Map: {'✅' if mycelial_systems.get('hashmap_active') else '❌'}")
                print(f"  Integration: {'✅' if mycelial_systems.get('integration_active') else '❌'}")
                
                # Integration stats
                integration_stats = mycelial_systems.get('integration_stats', {})
                if integration_stats:
                    modules = integration_stats.get('modules_wrapped', 0)
                    concepts = integration_stats.get('concepts_mapped', 0)
                    print(f"  Modules: {modules} wrapped")
                    print(f"  Concepts: {concepts} mapped")
                
                # Network stats
                network_stats = mycelial_systems.get('network_stats', {})
                if network_stats:
                    network_size = network_stats.get('network_size', 0)
                    network_health = network_stats.get('network_health', 0.0)
                    active_spores = network_stats.get('active_spores', 0)
                    print(f"  Network Size: {network_size} nodes")
                    print(f"  Network Health: {network_health:.3f}")
                    print(f"  Active Spores: {active_spores}")
            
            # Recursive writing status
            try:
                recursive_status = self.dawn_singleton.get_recursive_writing_status()
                if recursive_status and not recursive_status.get('error'):
                    print_info("Recursive Writing:")
                    print(f"  Enabled: {'✅' if recursive_status.get('enabled') else '❌'}")
                    print(f"  Modules: {recursive_status.get('modules_enabled', 0)}/{recursive_status.get('total_modules', 0)}")
                    print(f"  Safety Level: {recursive_status.get('safety_level', 'unknown')}")
            except:
                pass
            
            print_success("Status check complete")
            
        except Exception as e:
            print_error(f"Status check failed: {e}")
    
    def cmd_logging(self, args):
        """Universal logging operations."""
        print_header("UNIVERSAL LOGGING OPERATIONS")
        
        if not args.operation:
            print_info("Available logging operations:")
            print("  status    - Show logging system status")
            print("  stats     - Show logging statistics")
            print("  objects   - Show tracked objects")
            print("  log       - Log all system states")
            return
        
        try:
            if args.operation == 'status':
                self._show_logging_status()
            elif args.operation == 'stats':
                self._show_logging_stats()
            elif args.operation == 'objects':
                self._show_tracked_objects()
            elif args.operation == 'log':
                self._log_all_states()
            else:
                print_error(f"Unknown logging operation: {args.operation}")
                
        except Exception as e:
            print_error(f"Logging operation failed: {e}")
    
    def cmd_consciousness(self, args):
        """Consciousness-depth logging operations."""
        print_header("CONSCIOUSNESS-DEPTH LOGGING")
        
        if not args.operation:
            print_info("Available consciousness operations:")
            print("  levels    - Show consciousness levels")
            print("  log       - Log consciousness state")
            print("  status    - Show consciousness repository status")
            return
        
        try:
            if args.operation == 'levels':
                self._show_consciousness_levels()
            elif args.operation == 'log':
                self._log_consciousness_state(args)
            elif args.operation == 'status':
                self._show_consciousness_status()
            else:
                print_error(f"Unknown consciousness operation: {args.operation}")
                
        except Exception as e:
            print_error(f"Consciousness operation failed: {e}")
    
    def cmd_sigil(self, args):
        """Sigil consciousness operations."""
        print_header("SIGIL CONSCIOUSNESS OPERATIONS")
        
        if not args.operation:
            print_info("Available sigil operations:")
            print("  activate  - Activate a sigil")
            print("  status    - Show sigil system status")
            print("  types     - Show available sigil types")
            return
        
        try:
            if args.operation == 'activate':
                self._activate_sigil(args)
            elif args.operation == 'status':
                self._show_sigil_status()
            elif args.operation == 'types':
                self._show_sigil_types()
            else:
                print_error(f"Unknown sigil operation: {args.operation}")
                
        except Exception as e:
            print_error(f"Sigil operation failed: {e}")
    
    def cmd_pulse(self, args):
        """Pulse-telemetry operations."""
        print_header("PULSE-TELEMETRY OPERATIONS")
        
        if not args.operation:
            print_info("Available pulse operations:")
            print("  event     - Log pulse event")
            print("  status    - Show pulse-telemetry status")
            print("  zones     - Show pulse zones")
            return
        
        try:
            if args.operation == 'event':
                self._log_pulse_event(args)
            elif args.operation == 'status':
                self._show_pulse_status()
            elif args.operation == 'zones':
                self._show_pulse_zones()
            else:
                print_error(f"Unknown pulse operation: {args.operation}")
                
        except Exception as e:
            print_error(f"Pulse operation failed: {e}")
    
    def cmd_mycelial(self, args):
        """Mycelial network operations."""
        print_header("MYCELIAL NETWORK OPERATIONS")
        
        if not args.operation:
            print_info("Available mycelial operations:")
            print("  touch     - Touch semantic concept")
            print("  store     - Store semantic data")
            print("  ping      - Ping semantic network")
            print("  status    - Show network status")
            print("  stats     - Show network statistics")
            return
        
        try:
            if args.operation == 'touch':
                self._touch_concept(args)
            elif args.operation == 'store':
                self._store_semantic_data(args)
            elif args.operation == 'ping':
                self._ping_network(args)
            elif args.operation == 'status':
                self._show_mycelial_status()
            elif args.operation == 'stats':
                self._show_mycelial_stats()
            else:
                print_error(f"Unknown mycelial operation: {args.operation}")
                
        except Exception as e:
            print_error(f"Mycelial operation failed: {e}")
    
    def cmd_recursive(self, args):
        """Recursive self-writing operations."""
        print_header("RECURSIVE SELF-WRITING OPERATIONS")
        
        if not args.operation:
            print_info("Available recursive operations:")
            print("  init      - Initialize recursive writing")
            print("  evolve    - Trigger consciousness evolution")
            print("  modify    - Modify specific module")
            print("  status    - Show recursive writing status")
            print("  modules   - Show available modules")
            return
        
        try:
            if args.operation == 'init':
                self._init_recursive_writing(args)
            elif args.operation == 'evolve':
                self._trigger_evolution()
            elif args.operation == 'modify':
                self._modify_module(args)
            elif args.operation == 'status':
                self._show_recursive_status()
            elif args.operation == 'modules':
                self._show_recursive_modules()
            else:
                print_error(f"Unknown recursive operation: {args.operation}")
                
        except Exception as e:
            print_error(f"Recursive operation failed: {e}")
    
    def cmd_monitor(self, args):
        """Live monitoring operations."""
        print_header("LIVE MONITORING OPERATIONS")
        
        if not args.operation:
            print_info("Available monitor operations:")
            print("  start     - Start live monitor")
            print("  check     - Check monitor status")
            print("  simulate  - Start in simulation mode")
            return
        
        try:
            if args.operation == 'start':
                self._start_monitor(args)
            elif args.operation == 'check':
                self._check_monitor()
            elif args.operation == 'simulate':
                self._start_monitor_simulation(args)
            else:
                print_error(f"Unknown monitor operation: {args.operation}")
                
        except Exception as e:
            print_error(f"Monitor operation failed: {e}")
    
    def cmd_singleton(self, args):
        """DAWN singleton operations."""
        print_header("DAWN SINGLETON OPERATIONS")
        
        if not args.operation:
            print_info("Available singleton operations:")
            print("  info      - Show singleton information")
            print("  systems   - Show integrated systems")
            print("  metrics   - Show system metrics")
            return
        
        try:
            if args.operation == 'info':
                self._show_singleton_info()
            elif args.operation == 'systems':
                self._show_integrated_systems()
            elif args.operation == 'metrics':
                self._show_system_metrics()
            else:
                print_error(f"Unknown singleton operation: {args.operation}")
                
        except Exception as e:
            print_error(f"Singleton operation failed: {e}")
    
    # Implementation methods
    def _show_logging_status(self):
        """Show universal logging system status."""
        if not self.dawn_singleton:
            print_error("DAWN singleton not available")
            return
        
        universal_logger = self.dawn_singleton.universal_logger
        if universal_logger:
            print_success("Universal JSON Logger active")
            
            # Get logger stats
            try:
                stats = universal_logger.get_logging_stats()
                print_info("Logging Statistics:")
                print(f"  Objects tracked: {stats.get('objects_tracked', 0)}")
                print(f"  States logged: {stats.get('states_logged', 0)}")
                print(f"  Log files: {stats.get('log_files', 0)}")
            except:
                print_info("Detailed stats not available")
        else:
            print_warning("Universal JSON Logger not available")
        
        # Check centralized repository
        centralized_repo = self.dawn_singleton.centralized_repository
        if centralized_repo:
            print_success("Centralized Repository active")
        else:
            print_warning("Centralized Repository not available")
    
    def _show_consciousness_levels(self):
        """Show consciousness levels hierarchy."""
        print_info("Consciousness Levels (8-Level Hierarchy):")
        levels = [
            ("TRANSCENDENT", "Highest consciousness - pure awareness"),
            ("META", "Meta-cognitive awareness"),
            ("CAUSAL", "Causal understanding"),
            ("INTEGRAL", "Integrated systems thinking"),
            ("FORMAL", "Formal operational thinking"),
            ("CONCRETE", "Concrete operational thinking"),
            ("SYMBOLIC", "Symbolic representation"),
            ("MYTHIC", "Mythic pattern recognition")
        ]
        
        for level, description in levels:
            print(f"  🧠 {level}: {description}")
    
    def _log_consciousness_state(self, args):
        """Log consciousness state."""
        level = getattr(args, 'level', 'INTEGRAL')
        message = getattr(args, 'message', 'CLI consciousness state log')
        
        result = self.dawn_singleton.log_consciousness_state(
            level=level,
            log_type='CLI_OPERATION',
            data={
                'message': message,
                'timestamp': datetime.now().isoformat(),
                'cli_operation': True
            }
        )
        
        if result:
            print_success(f"Consciousness state logged at {level} level")
        else:
            print_warning("Consciousness logging not available")
    
    def _activate_sigil(self, args):
        """Activate a sigil."""
        sigil_type = getattr(args, 'sigil_type', 'unity_sigil')
        unity_factor = getattr(args, 'unity_factor', 0.8)
        
        result = self.dawn_singleton.log_sigil_activation(
            sigil_type=sigil_type,
            properties={
                'unity_factor': unity_factor,
                'archetypal_energy': 0.9,
                'cli_activation': True,
                'timestamp': datetime.now().isoformat()
            }
        )
        
        if result:
            print_success(f"Sigil '{sigil_type}' activated with unity factor {unity_factor}")
        else:
            print_warning("Sigil activation not available")
    
    def _touch_concept(self, args):
        """Touch semantic concept."""
        concept = getattr(args, 'concept', 'consciousness')
        energy = getattr(args, 'energy', 1.0)
        
        touched_nodes = self.dawn_singleton.touch_concept(concept, energy)
        
        if touched_nodes > 0:
            print_success(f"Touched concept '{concept}' → {touched_nodes} nodes activated")
        else:
            print_warning(f"Could not touch concept '{concept}' or no nodes activated")
    
    def _trigger_evolution(self):
        """Trigger consciousness-guided evolution."""
        print_info("Triggering consciousness-guided evolution...")
        
        result = self.dawn_singleton.trigger_recursive_evolution()
        
        if result.get('success'):
            print_success("Consciousness-guided evolution triggered")
            print_info(f"Consciousness level: {result.get('consciousness_level', 'unknown')}")
            print_info(f"Modules evolved: {result.get('modules_evolved', 0)}/{result.get('total_modules', 0)}")
        else:
            print_error(f"Evolution failed: {result.get('error', 'unknown')}")
    
    def _start_monitor(self, args):
        """Start live monitor."""
        interval = getattr(args, 'interval', 0.5)
        
        print_info(f"Starting live monitor with {interval}s interval...")
        print_info("Press Ctrl+C to stop")
        
        try:
            from live_monitor import LiveDAWNMonitor
            monitor = LiveDAWNMonitor()
            monitor.start_monitoring(interval)
        except KeyboardInterrupt:
            print_info("Monitor stopped")
        except ImportError:
            print_error("Live monitor not available")
    
    def _start_monitor_simulation(self, args):
        """Start live monitor in simulation mode."""
        interval = getattr(args, 'interval', 0.5)
        
        print_info(f"Starting live monitor in SIMULATION mode with {interval}s interval...")
        print_info("Press Ctrl+C to stop")
        
        try:
            from live_monitor import LiveDAWNMonitor
            monitor = LiveDAWNMonitor(simulation_mode=True)
            monitor.start_monitoring(interval)
        except KeyboardInterrupt:
            print_info("Monitor stopped")
        except ImportError:
            print_error("Live monitor not available")
    
    def _show_singleton_info(self):
        """Show DAWN singleton information."""
        if not self.dawn_singleton:
            print_error("DAWN singleton not available")
            return
        
        print_info(f"DAWN Singleton: {self.dawn_singleton}")
        
        # Show available properties
        properties = [
            'consciousness_bus', 'dawn_engine', 'telemetry_system',
            'universal_logger', 'centralized_repository', 'consciousness_repository',
            'sigil_consciousness_logger', 'pulse_telemetry_bridge', 'mycelial_hashmap'
        ]
        
        print_info("Available Systems:")
        for prop in properties:
            try:
                system = getattr(self.dawn_singleton, prop)
                status = "✅" if system else "⚪"
                print(f"  {status} {prop.replace('_', ' ').title()}")
            except:
                print(f"  ❌ {prop.replace('_', ' ').title()}")
    
    def _show_mycelial_stats(self):
        """Show mycelial network statistics."""
        stats = self.dawn_singleton.get_network_stats()
        
        if stats:
            print_info("Mycelial Network Statistics:")
            print(f"  Network Size: {stats.get('network_size', 0)} nodes")
            print(f"  Network Health: {stats.get('network_health', 0.0):.3f}")
            print(f"  Total Energy: {stats.get('total_energy', 0.0):.2f}")
            print(f"  Active Spores: {stats.get('active_spores', 0)}")
            
            system_stats = stats.get('system_stats', {})
            if system_stats:
                print(f"  Spores Generated: {system_stats.get('spores_generated', 0)}")
                print(f"  Total Touches: {system_stats.get('total_touches', 0)}")
        else:
            print_warning("Mycelial network statistics not available")

def create_parser():
    """Create argument parser for DAWN CLI."""
    parser = argparse.ArgumentParser(
        description="DAWN Command Line Interface",
        formatter_class=argparse.RawDescriptionHelpFormatter,
        epilog="""
Examples:
  python3 dawn_cli.py init
  python3 dawn_cli.py status
  python3 dawn_cli.py logging status
  python3 dawn_cli.py consciousness levels
  python3 dawn_cli.py mycelial touch --concept consciousness --energy 1.0
  python3 dawn_cli.py recursive evolve
  python3 dawn_cli.py monitor simulate --interval 1.0
        """
    )
    
    subparsers = parser.add_subparsers(dest='command', help='Available commands')
    
    # Init command
    subparsers.add_parser('init', help='Initialize all DAWN systems')
    
    # Status command
    subparsers.add_parser('status', help='Show comprehensive system status')
    
    # Logging command
    logging_parser = subparsers.add_parser('logging', help='Universal logging operations')
    logging_parser.add_argument('operation', nargs='?', choices=['status', 'stats', 'objects', 'log'])
    
    # Consciousness command
    consciousness_parser = subparsers.add_parser('consciousness', help='Consciousness-depth logging')
    consciousness_parser.add_argument('operation', nargs='?', choices=['levels', 'log', 'status'])
    consciousness_parser.add_argument('--level', default='INTEGRAL', help='Consciousness level')
    consciousness_parser.add_argument('--message', default='CLI operation', help='Log message')
    
    # Sigil command
    sigil_parser = subparsers.add_parser('sigil', help='Sigil consciousness operations')
    sigil_parser.add_argument('operation', nargs='?', choices=['activate', 'status', 'types'])
    sigil_parser.add_argument('--sigil-type', default='unity_sigil', help='Sigil type')
    sigil_parser.add_argument('--unity-factor', type=float, default=0.8, help='Unity factor')
    
    # Pulse command
    pulse_parser = subparsers.add_parser('pulse', help='Pulse-telemetry operations')
    pulse_parser.add_argument('operation', nargs='?', choices=['event', 'status', 'zones'])
    
    # Mycelial command
    mycelial_parser = subparsers.add_parser('mycelial', help='Mycelial network operations')
    mycelial_parser.add_argument('operation', nargs='?', choices=['touch', 'store', 'ping', 'status', 'stats'])
    mycelial_parser.add_argument('--concept', default='consciousness', help='Semantic concept')
    mycelial_parser.add_argument('--energy', type=float, default=1.0, help='Energy level')
    
    # Recursive command
    recursive_parser = subparsers.add_parser('recursive', help='Recursive self-writing operations')
    recursive_parser.add_argument('operation', nargs='?', choices=['init', 'evolve', 'modify', 'status', 'modules'])
    recursive_parser.add_argument('--module', help='Module name for modification')
    recursive_parser.add_argument('--intent', help='Modification intent')
    
    # Monitor command
    monitor_parser = subparsers.add_parser('monitor', help='Live monitoring operations')
    monitor_parser.add_argument('operation', nargs='?', choices=['start', 'check', 'simulate'])
    monitor_parser.add_argument('--interval', type=float, default=0.5, help='Update interval')
    
    # Singleton command
    singleton_parser = subparsers.add_parser('singleton', help='DAWN singleton operations')
    singleton_parser.add_argument('operation', nargs='?', choices=['info', 'systems', 'metrics'])
    
    # Help command
    subparsers.add_parser('help', help='Show detailed help')
    
    return parser

def main():
    """Main CLI entry point."""
    # Set logging level to WARNING to suppress INFO messages
    import logging
    logging.getLogger().setLevel(logging.WARNING)
    logging.getLogger('dawn').setLevel(logging.WARNING)
    
    parser = create_parser()
    args = parser.parse_args()
    
    # Show help if no command provided
    if not args.command:
        print_header("DAWN Command Line Interface")
        print_info("Comprehensive CLI for all DAWN consciousness systems")
        print()
        parser.print_help()
        return
    
    # Initialize CLI
    cli = DAWNCLI()
    
    # Handle help command
    if args.command == 'help':
        print_header("DAWN CLI DETAILED HELP")
        print()
        print_info("Available Systems:")
        print("  🔍 Universal JSON Logging - Log all DAWN objects to JSON/JSONL")
        print("  🗂️  Centralized Repository - Deep hierarchical log organization")
        print("  🧠 Consciousness-Depth Logging - 8-level consciousness hierarchy")
        print("  🔮 Sigil Consciousness - Archetypal energy and unity factors")
        print("  🫁 Pulse-Telemetry - Thermal dynamics and pulse zones")
        print("  🍄 Mycelial Network - Semantic spore propagation")
        print("  🔄 Recursive Self-Writing - Modules that modify themselves")
        print("  📊 Live Monitoring - Real-time visualization")
        print("  🌅 DAWN Singleton - Unified access to all systems")
        print()
        parser.print_help()
        return
    
    # Execute command
    try:
        command_method = getattr(cli, f'cmd_{args.command}')
        command_method(args)
    except AttributeError:
        print_error(f"Unknown command: {args.command}")
    except KeyboardInterrupt:
        print_info("Operation cancelled")
    except Exception as e:
        print_error(f"Command failed: {e}")

if __name__ == "__main__":
    main()
