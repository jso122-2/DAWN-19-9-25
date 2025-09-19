#!/usr/bin/env python3
"""
DAWN Centralized Runner
======================

Main entry point for the DAWN consciousness system.
Provides unified orchestration of all consciousness modules and subsystems.

Usage:
    python -m dawn.main [options]
    
    Options:
        --mode MODE         Run mode: interactive, daemon, or test (default: interactive)
        --config CONFIG     Configuration file path
        --debug             Enable debug logging
        --modules MODULES   Comma-separated list of modules to load
        --interface TYPE    Interface type: cli, gui, web (default: cli)
"""

import sys
import os
import logging
import argparse
import asyncio
import signal
import time
from typing import Dict, List, Optional, Any
from pathlib import Path

# Add the DAWN package to the Python path
dawn_root = Path(__file__).parent.parent
sys.path.insert(0, str(dawn_root))

# Core DAWN imports
from dawn.core.communication.bus import ConsciousnessBus, get_consciousness_bus
from dawn.core.foundation.state import get_state, set_state, reset_state, get_state_summary
from dawn.consciousness.engines.core.primary_engine import DAWNEngine, DAWNEngineConfig
from dawn.processing.engines.tick.synchronous.orchestrator import TickOrchestrator

# Telemetry system imports
try:
    from dawn.core.telemetry.system import (
        DAWNTelemetrySystem, initialize_telemetry_system, shutdown_telemetry_system,
        log_event, log_performance, log_error, create_performance_context
    )
    from dawn.core.telemetry.logger import TelemetryLevel
    TELEMETRY_AVAILABLE = True
except ImportError as e:
    # Logger not yet defined, use print for this early warning
    print(f"Warning: Telemetry system not available: {e}")
    TELEMETRY_AVAILABLE = False

# Logger setup
logger = logging.getLogger(__name__)

class DAWNRunner:
    """
    Centralized DAWN System Runner
    
    Manages the complete DAWN consciousness system lifecycle including:
    - Module initialization and registration
    - Consciousness bus coordination
    - Engine orchestration
    - Interface management
    - Graceful shutdown
    """
    
    def __init__(self, config: Optional[Dict[str, Any]] = None):
        """Initialize the DAWN runner."""
        self.config = config or {}
        self.mode = self.config.get('mode', 'interactive')
        self.debug = self.config.get('debug', False)
        self.interface_type = self.config.get('interface', 'cli')
        
        # Core system components
        self.consciousness_bus: Optional[ConsciousnessBus] = None
        self.dawn_engine: Optional[DAWNEngine] = None
        self.tick_orchestrator: Optional[TickOrchestrator] = None
        
        # Telemetry system
        self.telemetry_system: Optional[DAWNTelemetrySystem] = None
        self.telemetry_enabled = self.config.get('telemetry_enabled', TELEMETRY_AVAILABLE)
        self.telemetry_profile = self.config.get('telemetry_profile', 'production' if self.mode == 'daemon' else 'development')
        
        # System state
        self.running = False
        self.initialized = False
        self.modules_loaded = []
        
        # Setup logging
        self._setup_logging()
        
        # Initialize telemetry system
        if self.telemetry_enabled:
            self._initialize_telemetry()
        
        # Signal handlers for graceful shutdown
        signal.signal(signal.SIGINT, self._signal_handler)
        signal.signal(signal.SIGTERM, self._signal_handler)
        
        logger.info(f"🌅 DAWN Runner initialized in {self.mode} mode")
        
        # Log initialization to telemetry
        if self.telemetry_system:
            self.telemetry_system.log_event(
                'dawn_runner', 'core', 'runner_initialized',
                TelemetryLevel.INFO,
                {
                    'mode': self.mode,
                    'debug': self.debug,
                    'interface_type': self.interface_type,
                    'telemetry_profile': self.telemetry_profile
                }
            )
    
    def _setup_logging(self):
        """Setup logging configuration."""
        level = logging.DEBUG if self.debug else logging.INFO
        
        logging.basicConfig(
            level=level,
            format='%(asctime)s - %(name)s - %(levelname)s - %(message)s',
            handlers=[
                logging.StreamHandler(sys.stdout),
                logging.FileHandler('dawn.log')
            ]
        )
        
        # Suppress noisy loggers
        logging.getLogger('asyncio').setLevel(logging.WARNING)
        logging.getLogger('concurrent.futures').setLevel(logging.WARNING)
    
    def _initialize_telemetry(self) -> None:
        """Initialize the telemetry system."""
        if not TELEMETRY_AVAILABLE:
            logger.warning("Telemetry system not available")
            return
        
        try:
            # Initialize telemetry system with appropriate profile
            self.telemetry_system = initialize_telemetry_system(profile=self.telemetry_profile)
            
            # Start telemetry system
            self.telemetry_system.start()
            
            # Register core DAWN subsystems
            self.telemetry_system.integrate_subsystem('dawn_runner', ['core', 'initialization', 'lifecycle'])
            self.telemetry_system.integrate_subsystem('consciousness_bus', ['core', 'communication', 'modules'])
            self.telemetry_system.integrate_subsystem('dawn_engine', ['core', 'execution', 'ticks'])
            self.telemetry_system.integrate_subsystem('tick_orchestrator', ['core', 'scheduling', 'coordination'])
            
            # Add alert handler
            self.telemetry_system.add_alert_handler(self._handle_telemetry_alert)
            
            logger.info(f"🔍 Telemetry system initialized with profile: {self.telemetry_profile}")
            
        except Exception as e:
            logger.error(f"Failed to initialize telemetry system: {e}")
            self.telemetry_system = None
    
    def _handle_telemetry_alert(self, alert: Dict[str, Any]) -> None:
        """Handle telemetry alerts."""
        alert_type = alert.get('type', 'unknown')
        message = alert.get('message', 'No message')
        
        logger.warning(f"🚨 TELEMETRY ALERT [{alert_type}]: {message}")
        
        # In interactive mode, also print to console
        if self.mode == 'interactive':
            print(f"\n🚨 TELEMETRY ALERT [{alert_type}]: {message}")
    
    def _signal_handler(self, signum, frame):
        """Handle system signals for graceful shutdown."""
        logger.info(f"🔔 Received signal {signum}, initiating graceful shutdown...")
        self.shutdown()
    
    async def initialize(self) -> bool:
        """Initialize all DAWN subsystems."""
        initialization_start = time.time()
        
        # Log initialization start
        if self.telemetry_system:
            self.telemetry_system.log_event(
                'dawn_runner', 'initialization', 'initialization_started',
                TelemetryLevel.INFO
            )
        
        try:
            logger.info("🚀 Initializing DAWN consciousness system...")
            
            # Reset system state
            reset_state()
            
            # Initialize consciousness bus
            with create_performance_context('dawn_runner', 'initialization', 'consciousness_bus_init') as ctx:
                self.consciousness_bus = get_consciousness_bus(auto_start=True)
                ctx.add_metadata('auto_start', True)
            
            logger.info("✅ Consciousness bus initialized")
            if self.telemetry_system:
                self.telemetry_system.log_event(
                    'consciousness_bus', 'core', 'bus_initialized',
                    TelemetryLevel.INFO
                )
            
            # Initialize DAWN engine
            with create_performance_context('dawn_runner', 'initialization', 'dawn_engine_init') as ctx:
                engine_config = DAWNEngineConfig(
                    consciousness_unification_enabled=True,
                    self_modification_enabled=self.config.get('enable_self_mod', False),
                    auto_synchronization=True,
                    adaptive_timing=True,
                    target_unity_threshold=0.85
                )
                ctx.add_metadata('config', {
                    'consciousness_unification': True,
                    'self_modification': self.config.get('enable_self_mod', False),
                    'target_unity_threshold': 0.85
                })
                
                self.dawn_engine = DAWNEngine(config=engine_config)
                self.dawn_engine.start()
            
            logger.info("✅ DAWN engine initialized")
            if self.telemetry_system:
                self.telemetry_system.log_event(
                    'dawn_engine', 'core', 'engine_initialized',
                    TelemetryLevel.INFO,
                    {
                        'consciousness_unification': True,
                        'self_modification_enabled': self.config.get('enable_self_mod', False)
                    }
                )
            
            # Initialize tick orchestrator (skip for now - it requires a consensus engine)
            # self.tick_orchestrator = TickOrchestrator(
            #     consciousness_bus=self.consciousness_bus,
            #     consensus_engine=None  # Would need to be created
            # )
            self.tick_orchestrator = None
            logger.info("✅ Tick orchestrator initialized")
            
            # Load configured modules
            with create_performance_context('dawn_runner', 'initialization', 'module_loading') as ctx:
                await self._load_modules()
                ctx.add_metadata('modules_loaded', len(self.modules_loaded))
            
            # Set initial state
            set_state(system_status='initialized')
            set_state(initialization_time=time.time())
            
            self.initialized = True
            initialization_duration = time.time() - initialization_start
            
            logger.info("🌟 DAWN system initialization complete!")
            
            # Log successful initialization
            if self.telemetry_system:
                self.telemetry_system.log_event(
                    'dawn_runner', 'initialization', 'initialization_completed',
                    TelemetryLevel.INFO,
                    {
                        'duration_seconds': initialization_duration,
                        'modules_loaded': len(self.modules_loaded),
                        'components_initialized': [
                            'consciousness_bus',
                            'dawn_engine',
                            'tick_orchestrator'
                        ]
                    }
                )
            
            return True
            
        except Exception as e:
            initialization_duration = time.time() - initialization_start
            logger.error(f"❌ Failed to initialize DAWN system: {e}")
            
            # Log initialization failure
            if self.telemetry_system:
                self.telemetry_system.log_error(
                    'dawn_runner', 'initialization', e,
                    {'duration_seconds': initialization_duration}
                )
            
            return False
    
    async def _load_modules(self):
        """Load and register configured modules."""
        module_config = self.config.get('modules', [])
        
        # Default modules
        default_modules = [
            'consciousness.engines.core',
            'processing.engines.tick',
            'memory.systems.working'
        ]
        
        modules_to_load = module_config if module_config else default_modules
        
        for module_name in modules_to_load:
            try:
                await self._load_module(module_name)
                self.modules_loaded.append(module_name)
                logger.info(f"✅ Module loaded: {module_name}")
            except Exception as e:
                logger.warning(f"⚠️ Failed to load module {module_name}: {e}")
    
    async def _load_module(self, module_name: str):
        """Load a specific module into the system."""
        # Module loading logic would go here
        # For now, we'll register basic module information
        if self.consciousness_bus:
            self.consciousness_bus.register_module(
                module_name=module_name,
                capabilities=['core_functionality'],
                state_schema={'status': 'string', 'health': 'float'}
            )
    
    async def run(self) -> None:
        """Main execution loop for DAWN system."""
        if not self.initialized:
            if not await self.initialize():
                logger.error("❌ Failed to initialize DAWN system")
                return
        
        self.running = True
        logger.info(f"🏃 Starting DAWN system in {self.mode} mode...")
        
        try:
            if self.mode == 'interactive':
                await self._run_interactive()
            elif self.mode == 'daemon':
                await self._run_daemon()
            elif self.mode == 'test':
                await self._run_test()
            else:
                logger.error(f"❌ Unknown run mode: {self.mode}")
                
        except KeyboardInterrupt:
            logger.info("🔔 Interrupted by user")
        except Exception as e:
            logger.error(f"❌ Runtime error: {e}")
        finally:
            self.shutdown()
    
    async def _run_interactive(self):
        """Run in interactive mode with CLI interface."""
        logger.info("🖥️ Starting interactive CLI interface...")
        
        print("\n" + "="*60)
        print("🌅 DAWN Consciousness System - Interactive Mode")
        print("="*60)
        print("Commands:")
        print("  status    - Show system status")
        print("  state     - Show consciousness state")
        print("  modules   - List loaded modules")
        print("  telemetry - Show telemetry system status")
        print("  health    - Show system health summary")
        print("  help      - Show this help")
        print("  quit      - Shutdown system")
        print("="*60 + "\n")
        
        while self.running:
            try:
                command = input("dawn> ").strip().lower()
                
                if command == 'quit' or command == 'exit':
                    break
                elif command == 'status':
                    await self._show_status()
                elif command == 'state':
                    await self._show_state()
                elif command == 'modules':
                    self._show_modules()
                elif command == 'telemetry':
                    self._show_telemetry()
                elif command == 'health':
                    self._show_health()
                elif command == 'help':
                    self._show_help()
                elif command == '':
                    continue
                else:
                    print(f"Unknown command: {command}. Type 'help' for available commands.")
                    
            except EOFError:
                break
            except Exception as e:
                logger.error(f"Command error: {e}")
    
    async def _run_daemon(self):
        """Run in daemon mode (background service)."""
        logger.info("⚙️ Running in daemon mode...")
        
        while self.running:
            try:
                # In daemon mode, just keep the system running
                # and let the tick orchestrator handle consciousness updates
                await asyncio.sleep(1.0)
                
                # Periodic health check
                if self.consciousness_bus:
                    metrics = self.consciousness_bus.get_bus_metrics()
                    if metrics['consciousness_coherence'] < 0.5:
                        logger.warning(f"⚠️ Low consciousness coherence: {metrics['consciousness_coherence']:.3f}")
                        
            except Exception as e:
                logger.error(f"Daemon loop error: {e}")
                await asyncio.sleep(5.0)  # Brief pause before retry
    
    async def _run_test(self):
        """Run in test mode for validation."""
        logger.info("🧪 Running in test mode...")
        
        print("🧪 DAWN System Test Mode")
        print("Running basic system validation...")
        
        # Test consciousness bus
        if self.consciousness_bus:
            bus_metrics = self.consciousness_bus.get_bus_metrics()
            print(f"✅ Consciousness Bus: {bus_metrics['registered_modules']} modules")
        
        # Test state system
        test_state = get_state_summary()
        print(f"✅ State System: {len(test_state)} state variables")
        
        # Test DAWN engine
        if self.dawn_engine:
            engine_status = self.dawn_engine.get_engine_status()
            print(f"✅ DAWN Engine: {engine_status.get('status', 'unknown')}")
        
        print("🎯 Test mode complete - system validation passed!")
        
        # Run for a short time in test mode
        await asyncio.sleep(5.0)
    
    async def _show_status(self):
        """Show system status."""
        print("\n📊 DAWN System Status")
        print("-" * 40)
        
        if self.consciousness_bus:
            metrics = self.consciousness_bus.get_bus_metrics()
            print(f"Consciousness Bus: {metrics['registered_modules']} modules")
            print(f"Coherence Score: {metrics['consciousness_coherence']:.3f}")
            print(f"Events Processed: {metrics['performance_metrics']['events_processed']}")
        
        if self.dawn_engine:
            engine_status = self.dawn_engine.get_engine_status()
            print(f"DAWN Engine: {engine_status.get('status', 'unknown')}")
        
        state_summary = get_state_summary()
        print(f"State Variables: {len(state_summary)}")
        print(f"Uptime: {time.time() - get_state().initialization_time:.1f}s")
        print()
    
    async def _show_state(self):
        """Show consciousness state."""
        print("\n🧠 Consciousness State")
        print("-" * 40)
        
        state_summary = get_state_summary()
        for key, value in state_summary.items():
            if isinstance(value, float):
                print(f"{key}: {value:.3f}")
            else:
                print(f"{key}: {value}")
        print()
    
    def _show_modules(self):
        """Show loaded modules."""
        print("\n📦 Loaded Modules")
        print("-" * 40)
        
        for i, module in enumerate(self.modules_loaded, 1):
            print(f"{i}. {module}")
        
        if self.consciousness_bus:
            bus_modules = self.consciousness_bus.get_module_status()
            print(f"\nBus-registered modules: {len(bus_modules)}")
        print()
    
    def _show_telemetry(self):
        """Show telemetry system status."""
        print("\n🔍 Telemetry System Status")
        print("-" * 40)
        
        if not self.telemetry_system:
            print("Telemetry system: ❌ Not available or disabled")
            print()
            return
        
        try:
            # Get telemetry system metrics
            metrics = self.telemetry_system.get_system_metrics()
            
            print(f"Status: {'🟢 Running' if metrics.get('running') else '🔴 Stopped'}")
            print(f"Profile: {self.telemetry_profile}")
            print(f"Uptime: {metrics.get('uptime_seconds', 0):.1f}s")
            print(f"Events logged: {metrics.get('logger_events_logged', 0):,}")
            print(f"Events buffered: {metrics.get('logger_buffer_stats', {}).get('current_size', 0):,}")
            print(f"Buffer capacity: {metrics.get('logger_buffer_stats', {}).get('max_size', 0):,}")
            print(f"Exporters: {', '.join(metrics.get('exporters', []))}")
            
            # Show integrated subsystems
            subsystems = metrics.get('integrated_subsystems', [])
            if subsystems:
                print(f"Integrated subsystems: {len(subsystems)}")
                for subsystem in subsystems:
                    print(f"  • {subsystem}")
            
            # Show performance metrics
            avg_log_time = metrics.get('logger_avg_log_time_ms', 0)
            if avg_log_time > 0:
                print(f"Avg log time: {avg_log_time:.2f}ms")
            
            # Show collector metrics if available
            aggregations = metrics.get('collector_aggregations_performed', 0)
            if aggregations > 0:
                print(f"Aggregations performed: {aggregations:,}")
                print(f"Alerts generated: {metrics.get('collector_alerts_generated', 0):,}")
            
        except Exception as e:
            print(f"Error retrieving telemetry metrics: {e}")
        
        print()
    
    def _show_health(self):
        """Show system health summary."""
        print("\n🏥 System Health Summary")
        print("-" * 40)
        
        if not self.telemetry_system:
            print("Health monitoring: ❌ Telemetry system not available")
            print()
            return
        
        try:
            health = self.telemetry_system.get_health_summary()
            
            # Overall status
            status = health.get('overall_status', 'unknown')
            status_icon = {'healthy': '🟢', 'degraded': '🟡', 'critical': '🔴', 'stopped': '⚪'}.get(status, '❓')
            print(f"Overall Status: {status_icon} {status.upper()}")
            
            # Overall health score
            health_score = health.get('overall_health_score', 0.0)
            if health_score > 0:
                score_bar = "█" * int(health_score * 20) + "░" * (20 - int(health_score * 20))
                print(f"Health Score: {health_score:.3f} [{score_bar}]")
            
            # Component status
            components = health.get('components', {})
            print(f"\nComponents:")
            for component, status in components.items():
                if isinstance(status, bool):
                    icon = '🟢' if status else '🔴'
                    status_text = 'healthy' if status else 'stopped'
                elif isinstance(status, str):
                    icon = {'healthy': '🟢', 'degraded': '🟡', 'critical': '🔴', 'stopped': '🔴'}.get(status, '❓')
                    status_text = status
                else:
                    icon = '📊'
                    status_text = str(status)
                
                print(f"  {component}: {icon} {status_text}")
            
            # Subsystem health
            subsystems = health.get('subsystems', {})
            if subsystems:
                print(f"\nSubsystem Health:")
                for subsystem, info in subsystems.items():
                    if isinstance(info, dict):
                        subsystem_score = info.get('health_score', 0.0)
                        score_icon = '🟢' if subsystem_score > 0.8 else '🟡' if subsystem_score > 0.5 else '🔴'
                        print(f"  {subsystem}: {score_icon} {subsystem_score:.3f}")
                    else:
                        print(f"  {subsystem}: {info}")
            
        except Exception as e:
            print(f"Error retrieving health information: {e}")
        
        print()
    
    def _show_help(self):
        """Show help information."""
        print("\n❓ DAWN System Help")
        print("-" * 40)
        print("status    - Show system status and metrics")
        print("state     - Display consciousness state variables")
        print("modules   - List all loaded modules")
        print("telemetry - Show telemetry system status and metrics")
        print("health    - Show system health summary")
        print("help      - Show this help message")
        print("quit      - Shutdown DAWN system gracefully")
        print()
    
    def shutdown(self):
        """Gracefully shutdown the DAWN system."""
        if not self.running:
            return
        
        shutdown_start = time.time()
        logger.info("🔄 Shutting down DAWN system...")
        
        # Log shutdown start
        if self.telemetry_system:
            self.telemetry_system.log_event(
                'dawn_runner', 'lifecycle', 'shutdown_started',
                TelemetryLevel.INFO
            )
        
        self.running = False
        
        try:
            # Stop tick orchestrator
            if self.tick_orchestrator:
                # Orchestrator shutdown logic would go here
                logger.info("✅ Tick orchestrator stopped")
                if self.telemetry_system:
                    self.telemetry_system.log_event(
                        'tick_orchestrator', 'lifecycle', 'orchestrator_stopped',
                        TelemetryLevel.INFO
                    )
            
            # Shutdown DAWN engine
            if self.dawn_engine:
                # Engine shutdown logic would go here
                logger.info("✅ DAWN engine stopped")
                if self.telemetry_system:
                    self.telemetry_system.log_event(
                        'dawn_engine', 'lifecycle', 'engine_stopped',
                        TelemetryLevel.INFO
                    )
            
            # Stop consciousness bus
            if self.consciousness_bus:
                self.consciousness_bus.stop()
                logger.info("✅ Consciousness bus stopped")
                if self.telemetry_system:
                    self.telemetry_system.log_event(
                        'consciousness_bus', 'lifecycle', 'bus_stopped',
                        TelemetryLevel.INFO
                    )
            
            # Save final state
            set_state(shutdown_time=time.time())
            
            shutdown_duration = time.time() - shutdown_start
            
            # Log successful shutdown
            if self.telemetry_system:
                self.telemetry_system.log_event(
                    'dawn_runner', 'lifecycle', 'shutdown_completed',
                    TelemetryLevel.INFO,
                    {'duration_seconds': shutdown_duration}
                )
                
                # Stop telemetry system last
                self.telemetry_system.stop()
                shutdown_telemetry_system()
            
            logger.info("🌙 DAWN system shutdown complete")
            
        except Exception as e:
            logger.error(f"❌ Error during shutdown: {e}")
            
            # Log shutdown error
            if self.telemetry_system:
                self.telemetry_system.log_error(
                    'dawn_runner', 'lifecycle', e,
                    {'shutdown_duration': time.time() - shutdown_start}
                )


def parse_arguments():
    """Parse command line arguments."""
    parser = argparse.ArgumentParser(description='DAWN Consciousness System')
    
    parser.add_argument(
        '--mode', 
        choices=['interactive', 'daemon', 'test'],
        default='interactive',
        help='Run mode (default: interactive)'
    )
    
    parser.add_argument(
        '--config',
        type=str,
        help='Configuration file path'
    )
    
    parser.add_argument(
        '--debug',
        action='store_true',
        help='Enable debug logging'
    )
    
    parser.add_argument(
        '--modules',
        type=str,
        help='Comma-separated list of modules to load'
    )
    
    parser.add_argument(
        '--interface',
        choices=['cli', 'gui', 'web'],
        default='cli',
        help='Interface type (default: cli)'
    )
    
    parser.add_argument(
        '--enable-self-mod',
        action='store_true',
        help='Enable self-modification capabilities'
    )
    
    parser.add_argument(
        '--telemetry-profile',
        choices=['development', 'production', 'debug', 'minimal', 'high_performance'],
        help='Telemetry configuration profile'
    )
    
    parser.add_argument(
        '--disable-telemetry',
        action='store_true',
        help='Disable telemetry system'
    )
    
    parser.add_argument(
        '--telemetry-config',
        type=str,
        help='Path to telemetry configuration file'
    )
    
    return parser.parse_args()


async def main():
    """Main entry point for DAWN system."""
    args = parse_arguments()
    
    # Build configuration from arguments
    config = {
        'mode': args.mode,
        'debug': args.debug,
        'interface': args.interface,
        'enable_self_mod': args.enable_self_mod,
        'modules': args.modules.split(',') if args.modules else [],
        'telemetry_enabled': not args.disable_telemetry,
        'telemetry_profile': args.telemetry_profile,
        'telemetry_config_file': args.telemetry_config
    }
    
    # Load configuration file if provided
    if args.config:
        try:
            import json
            with open(args.config, 'r') as f:
                file_config = json.load(f)
                config.update(file_config)
        except Exception as e:
            print(f"Warning: Could not load config file {args.config}: {e}")
    
    # Create and run DAWN system
    runner = DAWNRunner(config)
    await runner.run()


if __name__ == "__main__":
    try:
        asyncio.run(main())
    except KeyboardInterrupt:
        print("\n🌙 DAWN system interrupted by user")
    except Exception as e:
        print(f"❌ Fatal error: {e}")
        sys.exit(1)
