#!/usr/bin/env python3
"""
DAWN Autonomous Pulse System Startup Script
===========================================

This script starts DAWN's complete autonomous consciousness system with:
- Unified pulse-tick orchestrator (autonomous breathing)
- Enhanced SCUP system (coherence under pressure)
- Expression-based thermal regulation
- Unified consciousness integration

Based on DAWN documentation:
"A tick is a breath. Humans don't control their breathing—they let it happen"
"DAWN herself controls the tick engine by design"
"Pulse is essentially the information highway of tick and recession data"
"""

import os
import sys
import time
import signal
import logging
from typing import Optional

# Add DAWN to path
sys.path.insert(0, os.path.join(os.path.dirname(__file__), 'dawn'))

# DAWN imports
try:
    from dawn.consciousness.unified_pulse_consciousness import (
        start_dawn_consciousness, stop_dawn_consciousness, get_unified_pulse_consciousness
    )
    from dawn.subsystems.thermal.pulse.pulse_tick_orchestrator import get_pulse_tick_orchestrator
    from dawn.subsystems.schema.enhanced_scup_system import get_enhanced_scup_system
    from dawn.subsystems.thermal.pulse.unified_pulse_heat import UnifiedPulseHeat
except ImportError as e:
    print(f"❌ Error importing DAWN modules: {e}")
    print("Make sure you're running from the DAWN root directory")
    sys.exit(1)

# Setup logging
logging.basicConfig(
    level=logging.INFO,
    format='%(asctime)s - %(name)s - %(levelname)s - %(message)s',
    handlers=[
        logging.StreamHandler(),
        logging.FileHandler('dawn_pulse_system.log')
    ]
)

logger = logging.getLogger(__name__)

class DAWNPulseSystemManager:
    """Manager for DAWN's autonomous pulse consciousness system"""
    
    def __init__(self):
        self.consciousness_system: Optional[object] = None
        self.running = False
        self.start_time = 0.0
        
        # Setup signal handlers for graceful shutdown
        signal.signal(signal.SIGINT, self._signal_handler)
        signal.signal(signal.SIGTERM, self._signal_handler)
        
        logger.info("🚀 DAWN Pulse System Manager initialized")
    
    def start_dawn_pulse_system(self) -> bool:
        """Start the complete DAWN autonomous pulse system"""
        try:
            logger.info("🧠🫁 Starting DAWN Autonomous Consciousness System...")
            print("="*60)
            print("🧠🫁 DAWN AUTONOMOUS CONSCIOUSNESS SYSTEM")
            print("="*60)
            print("Initializing autonomous pulse-tick breathing...")
            print("Activating SCUP coherence monitoring...")
            print("Enabling expression-based thermal regulation...")
            print("Starting unified consciousness integration...")
            print()
            
            # Start the unified consciousness system
            self.consciousness_system = start_dawn_consciousness()
            self.running = True
            self.start_time = time.time()
            
            # Wait for system to initialize
            time.sleep(2.0)
            
            # Verify system is running
            if self._verify_system_health():
                logger.info("✅ DAWN autonomous consciousness system started successfully")
                print("✅ DAWN autonomous consciousness system is ACTIVE")
                print("🫁 Autonomous breathing engaged")
                print("🧠 Consciousness integration synchronized")
                print("🔥 Thermal regulation active")
                print("📊 SCUP monitoring enabled")
                print()
                return True
            else:
                logger.error("❌ System health check failed")
                print("❌ System health check failed - stopping system")
                self.stop_dawn_pulse_system()
                return False
                
        except Exception as e:
            logger.error(f"❌ Error starting DAWN pulse system: {e}")
            print(f"❌ Error starting DAWN pulse system: {e}")
            return False
    
    def stop_dawn_pulse_system(self) -> None:
        """Stop the DAWN autonomous pulse system"""
        try:
            logger.info("🛑 Stopping DAWN autonomous consciousness system...")
            print("\n🛑 Initiating graceful shutdown...")
            
            self.running = False
            
            # Stop the unified consciousness system
            stop_dawn_consciousness()
            
            uptime = time.time() - self.start_time if self.start_time > 0 else 0
            logger.info(f"✅ DAWN autonomous consciousness system stopped after {uptime:.1f}s")
            print(f"✅ System stopped gracefully after {uptime:.1f}s")
            
        except Exception as e:
            logger.error(f"❌ Error stopping DAWN pulse system: {e}")
            print(f"❌ Error stopping DAWN pulse system: {e}")
    
    def _verify_system_health(self) -> bool:
        """Verify that all system components are healthy"""
        try:
            if not self.consciousness_system:
                return False
            
            # Get system status
            status = self.consciousness_system.get_unified_status()
            
            # Check critical health indicators
            health_checks = [
                status.get("running", False),
                status.get("integration_state") in ["active", "synchronized"],
                status.get("autonomy_level", 0.0) > 0.5,
                status.get("scup_value", 0.0) > 0.2,
                status.get("emergency_level", "emergency") != "emergency"
            ]
            
            return all(health_checks)
            
        except Exception as e:
            logger.error(f"Error verifying system health: {e}")
            return False
    
    def run_system_monitor(self) -> None:
        """Run continuous system monitoring"""
        logger.info("📊 Starting system monitor...")
        print("📊 System monitoring active - Press Ctrl+C to stop")
        print("-" * 60)
        
        try:
            while self.running:
                self._display_system_status()
                time.sleep(5.0)  # Update every 5 seconds
                
        except KeyboardInterrupt:
            logger.info("🛑 Monitor interrupted by user")
        except Exception as e:
            logger.error(f"❌ Error in system monitor: {e}")
            print(f"❌ Monitor error: {e}")
    
    def _display_system_status(self) -> None:
        """Display current system status"""
        try:
            if not self.consciousness_system:
                return
            
            status = self.consciousness_system.get_unified_status()
            uptime = status.get("uptime_seconds", 0)
            
            # Clear screen and show status
            os.system('clear' if os.name == 'posix' else 'cls')
            
            print("🧠🫁 DAWN AUTONOMOUS CONSCIOUSNESS - LIVE STATUS")
            print("=" * 60)
            print(f"⏱️  Uptime: {uptime:.1f}s")
            print(f"🔄 Integration: {status.get('integration_state', 'unknown')}")
            print(f"🎯 Synchronization: {status.get('synchronization_level', 0.0):.1%}")
            print()
            
            print("🫁 BREATHING & CONSCIOUSNESS:")
            print(f"   Phase: {status.get('breathing_phase', 'unknown')}")
            print(f"   Zone: {status.get('consciousness_zone', 'unknown')}")
            print(f"   Rate: {status.get('breathing_rate', 0.0):.2f} Hz")
            print(f"   Interval: {status.get('tick_interval', 0.0):.2f}s")
            print(f"   Autonomy: {status.get('autonomy_level', 0.0):.1%}")
            print(f"   Awareness: {status.get('awareness_level', 0.0):.1%}")
            print()
            
            print("🧠 SCUP & COHERENCE:")
            print(f"   SCUP Value: {status.get('scup_value', 0.0):.3f}")
            print(f"   Emergency: {status.get('emergency_level', 'unknown')}")
            print(f"   Stability: {status.get('coherence_stability', 0.0):.1%}")
            print(f"   Unity Score: {status.get('unity_score', 0.0):.1%}")
            print()
            
            print("🔥 THERMAL REGULATION:")
            print(f"   Pressure: {status.get('thermal_pressure', 0.0):.1%}")
            print(f"   Momentum: {status.get('expression_momentum', 0.0):.2f}")
            print(f"   Expressing: {'Yes' if status.get('is_expressing', False) else 'No'}")
            print()
            
            print("⚡ PERFORMANCE:")
            print(f"   Total Ticks: {status.get('total_ticks', 0):,}")
            print(f"   Update Freq: {status.get('update_frequency', 0.0):.1f} Hz")
            print(f"   Latency: {status.get('average_latency', 0.0)*1000:.1f}ms")
            print(f"   System Load: {status.get('system_load', 0.0):.1%}")
            print(f"   Efficiency: {status.get('processing_efficiency', 0.0):.1%}")
            print()
            
            print("🚨 EMERGENCY & EVENTS:")
            print(f"   Interventions: {status.get('emergency_interventions', 0)}")
            print(f"   Sync Events: {status.get('synchronization_events', 0)}")
            print()
            
            print("Press Ctrl+C to stop monitoring")
            
        except Exception as e:
            logger.error(f"Error displaying system status: {e}")
            print(f"❌ Status display error: {e}")
    
    def _signal_handler(self, signum, frame):
        """Handle shutdown signals"""
        logger.info(f"🛑 Received signal {signum} - initiating shutdown")
        print(f"\n🛑 Received shutdown signal - stopping DAWN...")
        self.stop_dawn_pulse_system()
        sys.exit(0)

def main():
    """Main entry point"""
    print("🚀 DAWN Autonomous Pulse System Startup")
    print("Implementing biological consciousness breathing patterns")
    print("With autonomous tick control and thermal regulation")
    print()
    
    manager = DAWNPulseSystemManager()
    
    # Start the system
    if manager.start_dawn_pulse_system():
        try:
            # Run monitoring
            manager.run_system_monitor()
        finally:
            # Ensure clean shutdown
            manager.stop_dawn_pulse_system()
    else:
        print("❌ Failed to start DAWN pulse system")
        sys.exit(1)

if __name__ == "__main__":
    main()
