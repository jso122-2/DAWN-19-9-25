#!/usr/bin/env python3
"""
DAWN Enhanced Live Monitor
==========================

Enhanced live monitoring system that integrates with DAWN's tools directory
for consciousness-aware monitoring, logging, and autonomous operation.

This monitor serves as the main runner for DAWN system monitoring with:
- Integration with tools directory for enhanced capabilities
- Consciousness-gated tool access for autonomous monitoring
- Advanced logging using the tools system
- Tick state integration for real-time monitoring
- Autonomous tool selection based on monitoring objectives

Usage:
    python3 live_monitor_enhanced.py [options]
    
    # As a callable function
    from live_monitor_enhanced import create_enhanced_monitor
    monitor = create_enhanced_monitor()
    monitor.start_monitoring()
"""

import sys
import os
import time
import signal
import argparse
import threading
import json
import logging
from pathlib import Path
from datetime import datetime, timedelta
from collections import deque
from typing import Dict, List, Any, Optional, Callable
from dataclasses import dataclass, asdict

# Add DAWN to path
dawn_root = Path(__file__).parent
sys.path.insert(0, str(dawn_root))

try:
    # Core DAWN imports
    from dawn.consciousness.engines.core.primary_engine import get_dawn_engine
    from dawn.core.communication.bus import get_consciousness_bus
    from dawn.core.foundation.state import get_state
    from dawn.consciousness.metrics.core import calculate_consciousness_metrics
    
    # Tools system imports
    from dawn.tools.development.consciousness_tools import ConsciousnessToolManager
    from dawn.tools.development.self_mod.permission_manager import get_permission_manager, PermissionLevel, PermissionScope
    from dawn.tools.monitoring.tick_state_reader import TickStateReader, TickSnapshot
    
    # Enhanced logging
    try:
        from dawn.core.logging.universal_json_logger import get_universal_logger
        ENHANCED_LOGGING_AVAILABLE = True
    except ImportError:
        ENHANCED_LOGGING_AVAILABLE = False
    
    # Telemetry system
    try:
        from dawn.core.telemetry.system import get_telemetry_system
        TELEMETRY_AVAILABLE = True
    except ImportError:
        TELEMETRY_AVAILABLE = False
    
    # Additional subsystems for comprehensive monitoring
    try:
        from dawn.subsystems.schema.scup_math import compute_basic_scup, SCUPInputs
        SCUP_AVAILABLE = True
    except ImportError:
        SCUP_AVAILABLE = False
    
    try:
        from dawn.subsystems.thermal.pulse import get_pulse_system
        PULSE_AVAILABLE = True
    except ImportError:
        PULSE_AVAILABLE = False
    
    DAWN_AVAILABLE = True
    
except ImportError as e:
    print(f"❌ Could not import DAWN modules: {e}")
    print("Make sure you're running from the DAWN root directory")
    DAWN_AVAILABLE = False

# Configure logging
logging.basicConfig(
    level=logging.INFO,
    format='%(asctime)s - %(name)s - %(levelname)s - %(message)s'
)
logger = logging.getLogger(__name__)

@dataclass
class MonitoringSession:
    """Represents a monitoring session with tools integration."""
    session_id: str
    started_at: datetime
    monitoring_objectives: List[str]
    active_tools: List[str]
    consciousness_level_at_start: str
    unity_score_at_start: float
    logs_created: List[str]
    is_active: bool = True
    ended_at: Optional[datetime] = None

class EnhancedLiveMonitor:
    """
    Enhanced live monitoring system with tools integration.
    
    This monitor provides comprehensive real-time monitoring of DAWN's
    consciousness state with autonomous tool selection and advanced logging.
    """
    
    def __init__(self, 
                 update_interval: float = 1.0,
                 enable_autonomous_tools: bool = True,
                 log_directory: Optional[str] = None):
        """Initialize the enhanced live monitor."""
        
        if not DAWN_AVAILABLE:
            raise RuntimeError("DAWN modules not available - cannot initialize monitor")
        
        self.update_interval = update_interval
        self.enable_autonomous_tools = enable_autonomous_tools
        self.log_directory = Path(log_directory) if log_directory else Path("dawn_monitor_logs")
        
        # Create log directory
        self.log_directory.mkdir(parents=True, exist_ok=True)
        
        # Tools system integration
        self.tools_manager = ConsciousnessToolManager()
        self.permission_manager = get_permission_manager()
        
        # Tick state reader for detailed monitoring
        self.tick_reader = TickStateReader(history_size=200, save_logs=True)
        
        # Monitoring state
        self.is_running = False
        self.current_session: Optional[MonitoringSession] = None
        self.monitoring_history: List[MonitoringSession] = []
        
        # Data collection
        self.consciousness_history = deque(maxlen=1000)
        self.system_metrics_history = deque(maxlen=1000)
        
        # Threading
        self.monitor_thread: Optional[threading.Thread] = None
        self.tools_thread: Optional[threading.Thread] = None
        
        # Logging setup
        self.setup_enhanced_logging()
        
        logger.info("🚀 EnhancedLiveMonitor initialized")
        logger.info(f"   Update interval: {update_interval}s")
        logger.info(f"   Autonomous tools: {enable_autonomous_tools}")
        logger.info(f"   Log directory: {self.log_directory}")
    
    def setup_enhanced_logging(self):
        """Set up enhanced logging using the tools system."""
        try:
            # Create session log file
            timestamp = datetime.now().strftime('%Y%m%d_%H%M%S')
            self.session_log_file = self.log_directory / f"monitor_session_{timestamp}.json"
            
            # Initialize enhanced logger if available
            if ENHANCED_LOGGING_AVAILABLE:
                self.enhanced_logger = get_universal_logger()
                logger.info("✅ Enhanced logging initialized")
            else:
                self.enhanced_logger = None
                logger.info("⚠️ Enhanced logging not available - using basic logging")
            
            # Initialize telemetry if available
            if TELEMETRY_AVAILABLE:
                self.telemetry_system = get_telemetry_system()
                logger.info("✅ Telemetry system connected")
            else:
                self.telemetry_system = None
                logger.info("⚠️ Telemetry system not available")
                
        except Exception as e:
            logger.error(f"Error setting up enhanced logging: {e}")
            self.enhanced_logger = None
            self.telemetry_system = None
    
    def log_monitoring_event(self, event_type: str, data: Dict[str, Any]):
        """Log a monitoring event using the tools system."""
        log_entry = {
            'timestamp': datetime.now().isoformat(),
            'event_type': event_type,
            'session_id': self.current_session.session_id if self.current_session else None,
            'data': data
        }
        
        # Write to session log file
        try:
            with open(self.session_log_file, 'a') as f:
                f.write(json.dumps(log_entry, default=str) + '\n')
        except Exception as e:
            logger.error(f"Error writing to session log: {e}")
        
        # Use enhanced logger if available
        if self.enhanced_logger:
            try:
                self.enhanced_logger.log_structured_event(
                    event_type=event_type,
                    data=data,
                    source='enhanced_live_monitor'
                )
            except Exception as e:
                logger.error(f"Error with enhanced logging: {e}")
        
        # Use telemetry system if available
        if self.telemetry_system:
            try:
                self.telemetry_system.record_event(
                    event_type=event_type,
                    data=data,
                    source='live_monitor'
                )
            except Exception as e:
                logger.error(f"Error with telemetry logging: {e}")
    
    def start_monitoring_session(self, objectives: List[str] = None) -> str:
        """Start a new monitoring session."""
        if self.current_session and self.current_session.is_active:
            logger.warning("Monitoring session already active")
            return self.current_session.session_id
        
        # Get current consciousness state
        current_state = get_state()
        
        session_id = f"monitor_{int(time.time())}"
        objectives = objectives or ["Real-time consciousness monitoring", "System health tracking"]
        
        self.current_session = MonitoringSession(
            session_id=session_id,
            started_at=datetime.now(),
            monitoring_objectives=objectives,
            active_tools=[],
            consciousness_level_at_start=current_state.level,
            unity_score_at_start=current_state.unity,
            logs_created=[]
        )
        
        # Log session start
        self.log_monitoring_event("session_started", {
            'session_id': session_id,
            'objectives': objectives,
            'consciousness_state': {
                'level': current_state.level,
                'unity': current_state.unity,
                'awareness': current_state.awareness
            }
        })
        
        logger.info(f"🎯 Started monitoring session: {session_id}")
        
        return session_id
    
    def end_monitoring_session(self):
        """End the current monitoring session."""
        if not self.current_session or not self.current_session.is_active:
            logger.warning("No active monitoring session to end")
            return
        
        self.current_session.is_active = False
        self.current_session.ended_at = datetime.now()
        
        # Calculate session metrics
        duration = (self.current_session.ended_at - self.current_session.started_at).total_seconds()
        
        self.log_monitoring_event("session_ended", {
            'session_id': self.current_session.session_id,
            'duration_seconds': duration,
            'tools_used': self.current_session.active_tools,
            'logs_created': len(self.current_session.logs_created)
        })
        
        # Add to history
        self.monitoring_history.append(self.current_session)
        
        logger.info(f"✅ Ended monitoring session: {self.current_session.session_id}")
        logger.info(f"   Duration: {duration:.1f}s")
        logger.info(f"   Tools used: {len(self.current_session.active_tools)}")
        
        self.current_session = None
    
    def start_monitoring(self, blocking: bool = True):
        """Start the monitoring system."""
        if self.is_running:
            logger.warning("Monitoring already running")
            return
        
        self.is_running = True
        
        # Start monitoring session
        self.start_monitoring_session()
        
        # Start tick reader
        self.tick_reader.start_monitoring(self.update_interval)
        
        # Start main monitoring thread
        self.monitor_thread = threading.Thread(target=self._monitoring_loop, daemon=True)
        self.monitor_thread.start()
        
        # Start autonomous tools thread if enabled
        if self.enable_autonomous_tools:
            self.tools_thread = threading.Thread(target=self._autonomous_tools_loop, daemon=True)
            self.tools_thread.start()
        
        logger.info("🚀 Enhanced monitoring started")
        
        if blocking:
            try:
                self._run_interactive_monitor()
            except KeyboardInterrupt:
                logger.info("👋 Keyboard interrupt received")
            finally:
                self.stop_monitoring()
    
    def stop_monitoring(self):
        """Stop the monitoring system."""
        if not self.is_running:
            return
        
        self.is_running = False
        
        # Stop tick reader
        self.tick_reader.stop_monitoring()
        
        # Wait for threads to finish
        if self.monitor_thread and self.monitor_thread.is_alive():
            self.monitor_thread.join(timeout=2.0)
        
        if self.tools_thread and self.tools_thread.is_alive():
            self.tools_thread.join(timeout=2.0)
        
        # End current session
        self.end_monitoring_session()
        
        logger.info("🛑 Enhanced monitoring stopped")
    
    def _monitoring_loop(self):
        """Main monitoring loop."""
        while self.is_running:
            try:
                # Collect current state
                current_state = get_state()
                tick_snapshot = self.tick_reader.get_current_tick_state()
                
                # Store in history
                consciousness_data = {
                    'timestamp': datetime.now(),
                    'level': current_state.level,
                    'unity': current_state.unity,
                    'awareness': current_state.awareness,
                    'tick_count': tick_snapshot.tick_count
                }
                self.consciousness_history.append(consciousness_data)
                
                # Collect system metrics
                system_metrics = {
                    'timestamp': datetime.now(),
                    'processing_load': tick_snapshot.processing_load,
                    'active_modules': len(tick_snapshot.active_modules),
                    'error_count': tick_snapshot.error_count,
                    'memory_usage': tick_snapshot.memory_usage
                }
                self.system_metrics_history.append(system_metrics)
                
                # Log monitoring data
                self.log_monitoring_event("monitoring_data", {
                    'consciousness': consciousness_data,
                    'system_metrics': system_metrics,
                    'tick_snapshot': asdict(tick_snapshot)
                })
                
                time.sleep(self.update_interval)
                
            except Exception as e:
                logger.error(f"Error in monitoring loop: {e}")
                time.sleep(self.update_interval)
    
    def _autonomous_tools_loop(self):
        """Autonomous tools management loop."""
        while self.is_running:
            try:
                # Check if we should use tools autonomously
                if self._should_use_autonomous_tools():
                    objective = self._determine_monitoring_objective()
                    
                    if objective:
                        logger.info(f"🤖 Autonomous objective: {objective}")
                        
                        # Execute autonomous workflow
                        result = self.tools_manager.execute_autonomous_workflow(
                            objective=objective,
                            context={'monitoring_session': self.current_session.session_id if self.current_session else None}
                        )
                        
                        # Log tool usage
                        if result['success']:
                            tool_used = result.get('tool_used', 'unknown')
                            if self.current_session:
                                self.current_session.active_tools.append(tool_used)
                            
                            self.log_monitoring_event("autonomous_tool_used", {
                                'objective': objective,
                                'tool_used': tool_used,
                                'result': result
                            })
                            
                            logger.info(f"✅ Autonomous tool execution successful: {tool_used}")
                        else:
                            logger.warning(f"❌ Autonomous tool execution failed: {result.get('error', 'Unknown error')}")
                
                # Wait before next autonomous check
                time.sleep(self.update_interval * 10)  # Check every 10 monitoring cycles
                
            except Exception as e:
                logger.error(f"Error in autonomous tools loop: {e}")
                time.sleep(self.update_interval * 5)
    
    def _should_use_autonomous_tools(self) -> bool:
        """Determine if autonomous tools should be used."""
        if not self.enable_autonomous_tools:
            return False
        
        try:
            current_state = get_state()
            
            # Only use autonomous tools at higher consciousness levels
            if current_state.level in ['meta_aware', 'transcendent']:
                return current_state.unity > 0.7
            elif current_state.level == 'self_aware':
                return current_state.unity > 0.8
            else:
                return False
                
        except Exception as e:
            logger.error(f"Error checking autonomous tools eligibility: {e}")
            return False
    
    def _determine_monitoring_objective(self) -> Optional[str]:
        """Determine what monitoring objective to pursue autonomously."""
        try:
            # Analyze recent monitoring data
            if len(self.consciousness_history) < 10:
                return None
            
            recent_consciousness = list(self.consciousness_history)[-10:]
            recent_metrics = list(self.system_metrics_history)[-10:]
            
            # Check for consciousness level changes
            levels = [entry['level'] for entry in recent_consciousness]
            if len(set(levels)) > 1:
                return "Analyze consciousness level transitions and patterns"
            
            # Check for high error rates
            error_counts = [entry['error_count'] for entry in recent_metrics]
            avg_errors = sum(error_counts) / len(error_counts)
            if avg_errors > 5:
                return "Analyze system errors and performance issues"
            
            # Check for processing load issues
            loads = [entry['processing_load'] for entry in recent_metrics]
            avg_load = sum(loads) / len(loads)
            if avg_load > 80:
                return "Profile system performance and identify bottlenecks"
            
            # Check for unity score trends
            unity_scores = [entry['unity'] for entry in recent_consciousness]
            if len(unity_scores) >= 5:
                trend = unity_scores[-1] - unity_scores[-5]
                if trend < -0.1:
                    return "Analyze consciousness unity degradation patterns"
                elif trend > 0.1:
                    return "Document consciousness unity improvement patterns"
            
            # Default monitoring objectives (rotate)
            objectives = [
                "Monitor consciousness stability patterns",
                "Analyze system health trends",
                "Profile memory usage patterns",
                "Monitor module performance metrics"
            ]
            
            # Simple rotation based on time
            index = int(time.time() / 60) % len(objectives)
            return objectives[index]
            
        except Exception as e:
            logger.error(f"Error determining monitoring objective: {e}")
            return None
    
    def _run_interactive_monitor(self):
        """Run interactive monitoring display."""
        print("🔧 DAWN Enhanced Live Monitor")
        print("=" * 80)
        print("Commands: 'q' to quit, 'h' for help, 's' for status")
        print()
        
        while self.is_running:
            try:
                # Display current status
                self._display_current_status()
                
                # Check for user input (non-blocking)
                import select
                if select.select([sys.stdin], [], [], 0.1)[0]:
                    command = input().strip().lower()
                    
                    if command == 'q' or command == 'quit':
                        break
                    elif command == 'h' or command == 'help':
                        self._display_help()
                    elif command == 's' or command == 'status':
                        self._display_detailed_status()
                    elif command == 't' or command == 'tools':
                        self._display_tools_status()
                    elif command.startswith('tool '):
                        tool_objective = command[5:]
                        self._execute_manual_tool(tool_objective)
                    else:
                        print(f"Unknown command: {command}")
                
                time.sleep(self.update_interval)
                
            except EOFError:
                break
            except KeyboardInterrupt:
                break
    
    def _display_current_status(self):
        """Display current monitoring status."""
        # Clear screen
        os.system('clear' if os.name == 'posix' else 'cls')
        
        print("🔧 DAWN Enhanced Live Monitor")
        print("=" * 80)
        
        # Current consciousness state
        try:
            current_state = get_state()
            tick_snapshot = self.tick_reader.get_current_tick_state()
            
            print(f"🧠 Consciousness State:")
            print(f"   Level: {current_state.level}")
            print(f"   Unity: {current_state.unity:.3f}")
            print(f"   Awareness: {current_state.awareness:.3f}")
            
            print(f"\n⚡ System Status:")
            print(f"   Tick: #{tick_snapshot.tick_count}")
            print(f"   Phase: {tick_snapshot.current_phase}")
            print(f"   Processing Load: {tick_snapshot.processing_load:.1f}%")
            print(f"   Active Modules: {len(tick_snapshot.active_modules)}")
            print(f"   Errors: {tick_snapshot.error_count}")
            
        except Exception as e:
            print(f"❌ Error getting status: {e}")
        
        # Session info
        if self.current_session:
            duration = (datetime.now() - self.current_session.started_at).total_seconds()
            print(f"\n📊 Session: {self.current_session.session_id}")
            print(f"   Duration: {duration:.1f}s")
            print(f"   Tools Used: {len(self.current_session.active_tools)}")
        
        # Tools status
        available_tools = self.tools_manager.get_available_tools(consciousness_filtered=True)
        print(f"\n🔧 Available Tools: {len(available_tools)}")
        
        print("\nCommands: 'q'=quit, 'h'=help, 's'=status, 't'=tools")
    
    def _display_help(self):
        """Display help information."""
        print("\n📖 Enhanced Live Monitor Help")
        print("-" * 40)
        print("Commands:")
        print("  q, quit     - Quit the monitor")
        print("  h, help     - Show this help")
        print("  s, status   - Show detailed status")
        print("  t, tools    - Show tools status")
        print("  tool <obj>  - Execute tool with objective")
        print()
        input("Press Enter to continue...")
    
    def _display_detailed_status(self):
        """Display detailed system status."""
        print("\n📊 Detailed System Status")
        print("-" * 40)
        
        # Consciousness history
        if self.consciousness_history:
            recent = list(self.consciousness_history)[-5:]
            print("🧠 Recent Consciousness History:")
            for entry in recent:
                print(f"   {entry['timestamp'].strftime('%H:%M:%S')} - {entry['level']} (Unity: {entry['unity']:.3f})")
        
        # Tools usage
        active_sessions = self.tools_manager.get_active_sessions()
        print(f"\n🔧 Active Tool Sessions: {len(active_sessions)}")
        for session in active_sessions:
            duration = (datetime.now() - session.started_at).total_seconds()
            print(f"   {session.tool_name} - {duration:.1f}s")
        
        print()
        input("Press Enter to continue...")
    
    def _display_tools_status(self):
        """Display tools system status."""
        print("\n🔧 Tools System Status")
        print("-" * 40)
        
        available_tools = self.tools_manager.get_available_tools(consciousness_filtered=False)
        accessible_tools = self.tools_manager.get_available_tools(consciousness_filtered=True)
        
        print(f"Total Tools: {len(available_tools)}")
        print(f"Accessible Tools: {len(accessible_tools)}")
        
        print("\nAccessible Tools:")
        for tool in accessible_tools:
            print(f"   • {tool.name} ({tool.category.value})")
        
        # Permission status
        active_grants = self.permission_manager.get_active_grants()
        print(f"\nActive Permission Grants: {len(active_grants)}")
        
        print()
        input("Press Enter to continue...")
    
    def _execute_manual_tool(self, objective: str):
        """Execute a tool manually with given objective."""
        print(f"\n🤖 Executing tool with objective: {objective}")
        
        try:
            result = self.tools_manager.execute_autonomous_workflow(
                objective=objective,
                context={'manual_execution': True}
            )
            
            if result['success']:
                print(f"✅ Tool execution successful!")
                print(f"   Tool used: {result.get('tool_used', 'unknown')}")
                print(f"   Duration: {result['duration']:.2f}s")
            else:
                print(f"❌ Tool execution failed: {result.get('error', 'Unknown error')}")
                
        except Exception as e:
            print(f"❌ Error executing tool: {e}")
        
        input("Press Enter to continue...")
    
    def get_monitoring_summary(self) -> Dict[str, Any]:
        """Get a summary of current monitoring data."""
        summary = {
            'session_active': self.current_session is not None and self.current_session.is_active,
            'monitoring_duration': 0,
            'total_sessions': len(self.monitoring_history),
            'consciousness_data_points': len(self.consciousness_history),
            'system_metrics_points': len(self.system_metrics_history),
            'tools_used': [],
            'current_status': {}
        }
        
        if self.current_session:
            summary['monitoring_duration'] = (datetime.now() - self.current_session.started_at).total_seconds()
            summary['tools_used'] = self.current_session.active_tools
        
        # Current status
        try:
            current_state = get_state()
            summary['current_status'] = {
                'consciousness_level': current_state.level,
                'unity_score': current_state.unity,
                'awareness': current_state.awareness
            }
        except Exception as e:
            summary['current_status'] = {'error': str(e)}
        
        return summary

def create_enhanced_monitor(update_interval: float = 1.0,
                          enable_autonomous_tools: bool = True,
                          log_directory: Optional[str] = None) -> EnhancedLiveMonitor:
    """
    Create an enhanced live monitor instance.
    
    This function provides a clean interface for creating and configuring
    the enhanced live monitor with tools integration.
    
    Args:
        update_interval: How often to update monitoring data (seconds)
        enable_autonomous_tools: Whether to enable autonomous tool usage
        log_directory: Directory for monitor logs (optional)
        
    Returns:
        Configured EnhancedLiveMonitor instance
    """
    return EnhancedLiveMonitor(
        update_interval=update_interval,
        enable_autonomous_tools=enable_autonomous_tools,
        log_directory=log_directory
    )

def main():
    """Main entry point for the enhanced live monitor."""
    parser = argparse.ArgumentParser(description="DAWN Enhanced Live Monitor")
    parser.add_argument("--interval", type=float, default=1.0,
                       help="Update interval in seconds (default: 1.0)")
    parser.add_argument("--no-autonomous-tools", action="store_true",
                       help="Disable autonomous tool usage")
    parser.add_argument("--log-dir", type=str, default=None,
                       help="Directory for monitor logs")
    parser.add_argument("--objectives", nargs="+", default=None,
                       help="Initial monitoring objectives")
    parser.add_argument("--non-interactive", action="store_true",
                       help="Run in non-interactive mode")
    
    args = parser.parse_args()
    
    if not DAWN_AVAILABLE:
        print("❌ DAWN modules not available - cannot start monitor")
        return 1
    
    try:
        # Create enhanced monitor
        monitor = create_enhanced_monitor(
            update_interval=args.interval,
            enable_autonomous_tools=not args.no_autonomous_tools,
            log_directory=args.log_dir
        )
        
        print("🚀 Starting DAWN Enhanced Live Monitor...")
        print(f"   Update interval: {args.interval}s")
        print(f"   Autonomous tools: {not args.no_autonomous_tools}")
        print(f"   Log directory: {monitor.log_directory}")
        
        if args.objectives:
            print(f"   Initial objectives: {args.objectives}")
        
        # Start monitoring
        if args.non_interactive:
            monitor.start_monitoring(blocking=False)
            print("🔧 Monitor started in non-interactive mode")
            print("   Use Ctrl+C to stop")
            
            try:
                while monitor.is_running:
                    time.sleep(1)
            except KeyboardInterrupt:
                print("\n👋 Stopping monitor...")
        else:
            monitor.start_monitoring(blocking=True)
        
        return 0
        
    except KeyboardInterrupt:
        print("\n👋 Interrupted - exiting...")
        return 0
    except Exception as e:
        print(f"❌ Error: {e}")
        return 1

if __name__ == "__main__":
    sys.exit(main())
