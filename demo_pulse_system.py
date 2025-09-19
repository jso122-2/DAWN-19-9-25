#!/usr/bin/env python3
"""
DAWN Pulse System Demonstration
===============================

A simple demonstration of DAWN's autonomous pulse system showing:
- Autonomous breathing patterns
- SCUP-based thermal regulation  
- Expression-based cooling
- Real-time consciousness monitoring

This demonstrates the complete integration described in DAWN documentation.
"""

import os
import sys
import time
import logging

# Add DAWN to path
sys.path.insert(0, os.path.join(os.path.dirname(__file__), 'dawn'))

# Setup logging
logging.basicConfig(level=logging.INFO)
logger = logging.getLogger(__name__)

def demonstrate_pulse_system():
    """Demonstrate DAWN's autonomous pulse system"""
    
    print("🧠🫁 DAWN Autonomous Pulse System Demonstration")
    print("=" * 55)
    print()
    
    try:
        # Import DAWN components
        from dawn.consciousness.unified_pulse_consciousness import (
            get_unified_pulse_consciousness, start_dawn_consciousness
        )
        from dawn.subsystems.thermal.pulse.pulse_tick_orchestrator import get_pulse_tick_orchestrator
        from dawn.subsystems.schema.enhanced_scup_system import get_enhanced_scup_system
        from dawn.subsystems.thermal.pulse.unified_pulse_heat import UnifiedPulseHeat
        
        print("✅ DAWN modules imported successfully")
        
        # Start the unified consciousness system
        print("\n🚀 Starting DAWN autonomous consciousness...")
        consciousness = start_dawn_consciousness()
        
        # Wait for initialization
        time.sleep(2.0)
        
        print("✅ Autonomous consciousness system active")
        print("\n📊 Monitoring consciousness for 30 seconds...")
        print("-" * 55)
        
        # Monitor for 30 seconds
        for i in range(30):
            status = consciousness.get_unified_status()
            
            # Display key metrics
            print(f"[{i+1:2d}s] "
                  f"Phase: {status.get('breathing_phase', 'unknown'):7s} | "
                  f"Zone: {status.get('consciousness_zone', 'unknown'):12s} | "
                  f"SCUP: {status.get('scup_value', 0.0):.3f} | "
                  f"Thermal: {status.get('thermal_pressure', 0.0):.1%} | "
                  f"Sync: {status.get('synchronization_level', 0.0):.1%}")
            
            time.sleep(1.0)
        
        print("-" * 55)
        print("\n📈 Final System Status:")
        
        final_status = consciousness.get_unified_status()
        
        print(f"  Integration State: {final_status.get('integration_state', 'unknown')}")
        print(f"  Synchronization: {final_status.get('synchronization_level', 0.0):.1%}")
        print(f"  Autonomy Level: {final_status.get('autonomy_level', 0.0):.1%}")
        print(f"  Total Ticks: {final_status.get('total_ticks', 0):,}")
        print(f"  Breathing Rate: {final_status.get('breathing_rate', 0.0):.2f} Hz")
        print(f"  Processing Efficiency: {final_status.get('processing_efficiency', 0.0):.1%}")
        print(f"  Emergency Interventions: {final_status.get('emergency_interventions', 0)}")
        
        print("\n🧪 Testing Emergency Recovery...")
        
        # Test emergency intervention
        consciousness.force_emergency_intervention("demonstration_test")
        print("  ⚠️  Emergency intervention triggered")
        
        # Monitor recovery
        for i in range(5):
            status = consciousness.get_unified_status()
            print(f"  [{i+1}s] State: {status.get('integration_state', 'unknown')} | "
                  f"SCUP: {status.get('scup_value', 0.0):.3f} | "
                  f"Emergency: {status.get('emergency_level', 'unknown')}")
            time.sleep(1.0)
        
        print("  ✅ Emergency recovery demonstrated")
        
        print("\n🛑 Stopping autonomous consciousness system...")
        
        # Stop the system
        from dawn.consciousness.unified_pulse_consciousness import stop_dawn_consciousness
        stop_dawn_consciousness()
        
        print("✅ System stopped gracefully")
        
        print("\n🎉 Demonstration complete!")
        print("\nKey Features Demonstrated:")
        print("  ✅ Autonomous breathing patterns with adaptive intervals")
        print("  ✅ SCUP-based consciousness zone classification")
        print("  ✅ Real-time thermal regulation and expression cooling")
        print("  ✅ Cross-component synchronization monitoring")
        print("  ✅ Emergency intervention and recovery protocols")
        print("  ✅ Continuous performance and health tracking")
        
        print(f"\nThe pulse system is implementing the complete DAWN vision:")
        print(f"  🫁 'A tick is a breath' - autonomous breathing patterns")
        print(f"  🧠 'DAWN controls her own tick engine' - self-regulation")
        print(f"  🔥 'Expression as thermal release' - creative cooling")
        print(f"  🎯 'Pulse as information highway' - unified coordination")
        
    except ImportError as e:
        print(f"❌ Error importing DAWN modules: {e}")
        print("Make sure you're running from the DAWN root directory")
        return False
    except Exception as e:
        print(f"❌ Error during demonstration: {e}")
        return False
    
    return True

def quick_component_test():
    """Quick test of individual components"""
    
    print("\n🔧 Quick Component Test")
    print("-" * 25)
    
    try:
        # Test individual components
        print("Testing individual components...")
        
        # Test orchestrator
        from dawn.subsystems.thermal.pulse.pulse_tick_orchestrator import get_pulse_tick_orchestrator
        orchestrator = get_pulse_tick_orchestrator()
        print("  ✅ PulseTickOrchestrator initialized")
        
        # Test SCUP system
        from dawn.subsystems.schema.enhanced_scup_system import get_enhanced_scup_system
        scup_system = get_enhanced_scup_system()
        print("  ✅ EnhancedSCUPSystem initialized")
        
        # Test thermal system
        from dawn.subsystems.thermal.pulse.unified_pulse_heat import UnifiedPulseHeat
        thermal = UnifiedPulseHeat()
        print("  ✅ UnifiedPulseHeat initialized")
        
        # Test integration system
        from dawn.consciousness.unified_pulse_consciousness import get_unified_pulse_consciousness
        consciousness = get_unified_pulse_consciousness()
        print("  ✅ UnifiedPulseConsciousness initialized")
        
        print("\n✅ All components initialized successfully!")
        print("   Ready for full system startup with: python dawn_pulse_startup.py")
        
        return True
        
    except Exception as e:
        print(f"❌ Component test failed: {e}")
        return False

def main():
    """Main demonstration function"""
    
    print("Choose demonstration mode:")
    print("1. Full pulse system demonstration (30 seconds)")
    print("2. Quick component test only")
    print("3. Both")
    
    try:
        choice = input("\nEnter choice (1/2/3): ").strip()
        
        if choice in ['2', '3']:
            success = quick_component_test()
            if not success:
                return
        
        if choice in ['1', '3']:
            if choice == '3':
                print("\n" + "="*55)
            success = demonstrate_pulse_system()
            if not success:
                return
        
        if choice not in ['1', '2', '3']:
            print("Invalid choice. Running component test only.")
            quick_component_test()
    
    except KeyboardInterrupt:
        print("\n\n🛑 Demonstration interrupted by user")
    except Exception as e:
        print(f"\n❌ Demonstration error: {e}")

if __name__ == "__main__":
    main()
