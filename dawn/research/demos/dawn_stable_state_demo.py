#!/usr/bin/env python3
"""
DAWN Stable State Detection System Demo
=======================================

Demonstrates the stable state detection and recovery system in action,
showing how DAWN automatically detects and recovers from unstable states.
"""

import time
import json
import logging
from datetime import datetime

# Import DAWN modules
from dawn_core.stable_state import StableStateDetector, get_stable_state_detector
from dawn_core.stable_state_core import StabilityLevel, RecoveryAction
from dawn_core.stability_integrations import get_health_adapter
from dawn_core.recursive_bubble import RecursiveBubble
from dawn.cognitive.symbolic_router import SymbolicRouter
from dawn.core.owl_bridge import OwlBridge

# Configure logging
logging.basicConfig(
    level=logging.INFO,
    format='%(asctime)s - %(name)s - %(levelname)s - %(message)s'
)
logger = logging.getLogger(__name__)

def demonstrate_stable_state_system():
    """Demonstrate the stable state detection and recovery system."""
    print("🔒 " + "="*60)
    print("🔒 DAWN STABLE STATE DETECTION SYSTEM DEMO")
    print("🔒 " + "="*60)
    print()
    
    # Initialize the stable state detector
    print("🔒 Initializing Stable State Detector...")
    detector = StableStateDetector(
        monitoring_interval=2.0,  # Check every 2 seconds for demo
        snapshot_threshold=0.85,
        critical_threshold=0.3
    )
    detector.start_monitoring()
    
    print(f"   ✓ Detector created: {detector.detector_id}")
    print(f"   ✓ Monitoring interval: {detector.monitoring_interval}s")
    print(f"   ✓ Snapshot threshold: {detector.snapshot_threshold}")
    print(f"   ✓ Critical threshold: {detector.critical_threshold}")
    print()
    
    # Create and register DAWN modules for monitoring
    print("🔗 Creating and registering DAWN modules...")
    
    # Create mock modules for demonstration
    recursive_bubble = RecursiveBubble(max_depth=5)
    symbolic_router = SymbolicRouter()
    owl_bridge = OwlBridge()
    
    # Register modules with appropriate health adapters
    recursive_adapter = get_health_adapter('recursive_bubble')
    symbolic_adapter = get_health_adapter('symbolic_anatomy')
    owl_adapter = get_health_adapter('owl_bridge')
    
    reg_id_1 = detector.register_module('recursive_bubble', recursive_bubble, recursive_adapter)
    reg_id_2 = detector.register_module('symbolic_anatomy', symbolic_router, symbolic_adapter)
    reg_id_3 = detector.register_module('owl_bridge', owl_bridge, owl_adapter)
    
    print(f"   ✓ RecursiveBubble: {reg_id_1}")
    print(f"   ✓ SymbolicAnatomy: {reg_id_2}")
    print(f"   ✓ OwlBridge: {reg_id_3}")
    print()
    
    # Wait for initial monitoring
    print("📊 Performing initial stability assessment...")
    time.sleep(3)
    
    # Get initial stability metrics
    initial_metrics = detector.calculate_stability_score()
    
    print(f"📊 Initial Stability Assessment:")
    print(f"   Overall Stability: {initial_metrics.overall_stability:.3f}")
    print(f"   Stability Level: {initial_metrics.stability_level.name}")
    print(f"   Entropy Stability: {initial_metrics.entropy_stability:.3f}")
    print(f"   Memory Coherence: {initial_metrics.memory_coherence:.3f}")
    print(f"   Recursive Depth Safe: {initial_metrics.recursive_depth_safe:.3f}")
    print(f"   Symbolic Organ Synergy: {initial_metrics.symbolic_organ_synergy:.3f}")
    
    if initial_metrics.failing_systems:
        print(f"   ❌ Failing Systems: {', '.join(initial_metrics.failing_systems)}")
    if initial_metrics.warning_systems:
        print(f"   ⚠️  Warning Systems: {', '.join(initial_metrics.warning_systems)}")
    print()
    
    # Demonstrate golden snapshot capture
    if initial_metrics.overall_stability >= detector.snapshot_threshold:
        print("📸 Capturing golden snapshot of stable state...")
        snapshot_id = detector.capture_stable_snapshot(initial_metrics, "Demo initial stable state")
        if snapshot_id:
            print(f"   ✓ Snapshot captured: {snapshot_id}")
        print()
    
    # Demonstrate system degradation and recovery
    print("🚨 " + "="*50)
    print("🚨 SIMULATING SYSTEM DEGRADATION")
    print("🚨 " + "="*50)
    print()
    
    # Scenario 1: Recursive depth explosion
    print("🚨 Scenario 1: Recursive depth explosion...")
    print("   Triggering deep recursive reflections...")
    
    # Force recursive bubble into deep recursion
    for i in range(6):  # Exceed safe depth
        recursive_bubble.reflect_on_self(f"Deep recursive thought level {i+1}")
        time.sleep(0.5)
        
    # Check stability after degradation
    time.sleep(2)
    degraded_metrics = detector.calculate_stability_score()
    
    print(f"   📊 Stability after recursion: {degraded_metrics.overall_stability:.3f}")
    print(f"   📊 Stability level: {degraded_metrics.stability_level.name}")
    print(f"   📉 Degradation rate: {degraded_metrics.degradation_rate:.3f}")
    
    if degraded_metrics.prediction_horizon < float('inf'):
        print(f"   ⏰ Failure predicted in: {degraded_metrics.prediction_horizon:.1f}s")
        
    # Let the system detect and recover
    print("   🛠️ Waiting for automatic recovery...")
    time.sleep(5)
    
    # Scenario 2: Symbolic organ overload
    print("\n🚨 Scenario 2: Symbolic organ overload...")
    print("   Overloading symbolic heart with emotions...")
    
    # Overload the symbolic heart
    for i in range(5):
        symbolic_router.heart.pulse(0.9, f"intense_emotion_{i}")
        time.sleep(0.3)
        
    # Check stability
    time.sleep(2)
    overload_metrics = detector.calculate_stability_score()
    
    print(f"   📊 Stability after overload: {overload_metrics.overall_stability:.3f}")
    print(f"   📊 Heart overloaded: {symbolic_router.heart.is_overloaded()}")
    print(f"   📊 Emotional charge: {symbolic_router.heart.emotional_charge:.3f}")
    
    # Wait for recovery
    print("   🛠️ Waiting for recovery...")
    time.sleep(5)
    
    # Show recovery results
    print("\n✅ " + "="*50)
    print("✅ SYSTEM RECOVERY ANALYSIS")
    print("✅ " + "="*50)
    print()
    
    final_metrics = detector.calculate_stability_score()
    
    print(f"📊 Final Stability Assessment:")
    print(f"   Overall Stability: {final_metrics.overall_stability:.3f}")
    print(f"   Stability Level: {final_metrics.stability_level.name}")
    print(f"   Recovery Success: {'✅' if final_metrics.overall_stability > degraded_metrics.overall_stability else '❌'}")
    print()
    
    # Show stability events
    print("📋 Stability Events Log:")
    recent_events = list(detector.stability_events)[-5:]  # Last 5 events
    
    for i, event in enumerate(recent_events, 1):
        print(f"   {i}. {event.event_type} ({event.timestamp.strftime('%H:%M:%S')})")
        print(f"      Score: {event.stability_score:.3f}")
        print(f"      Action: {event.recovery_action.value}")
        print(f"      Success: {'✅' if event.success else '❌'}")
        if event.failing_systems:
            print(f"      Failing: {', '.join(event.failing_systems)}")
        print()
    
    # Show snapshots
    print("📸 Golden Snapshots:")
    for i, snapshot_id in enumerate(detector.golden_snapshots, 1):
        snapshot = detector.snapshots[snapshot_id]
        print(f"   {i}. {snapshot_id}")
        print(f"      Score: {snapshot.stability_score:.3f}")
        print(f"      Time: {snapshot.timestamp.strftime('%H:%M:%S')}")
        print(f"      Description: {snapshot.description}")
        print()
    
    # Show detector status
    print("🔒 " + "="*50)
    print("🔒 DETECTOR STATUS SUMMARY")
    print("🔒 " + "="*50)
    print()
    
    status = detector.get_stability_status()
    
    print(f"Detector Status:")
    print(f"   Running: {'✅' if status['running'] else '❌'}")
    print(f"   Monitored Modules: {status['monitored_modules']}")
    print(f"   Golden Snapshots: {status['golden_snapshots']}")
    print(f"   Recent Events: {status['recent_events']}")
    print(f"   Rollback in Progress: {'⚠️ ' if status['rollback_in_progress'] else '✅ '}No")
    print(f"   Uptime: {status['uptime_seconds']:.1f}s")
    print()
    
    print(f"Performance Metrics:")
    metrics = status['metrics']
    print(f"   Stability Checks: {metrics['stability_checks']}")
    print(f"   Snapshots Captured: {metrics['snapshots_captured']}")
    print(f"   Recoveries Performed: {metrics['recoveries_performed']}")
    print(f"   Rollbacks Executed: {metrics['rollbacks_executed']}")
    print(f"   Degradations Detected: {metrics['degradations_detected']}")
    print()
    
    # Demonstrate manual recovery actions
    print("🛠️ " + "="*50)
    print("🛠️ MANUAL RECOVERY DEMONSTRATION")
    print("🛠️ " + "="*50)
    print()
    
    # Register a custom recovery callback
    def custom_recovery_callback():
        """Custom recovery for demonstration."""
        print("   🔧 Executing custom recovery protocol...")
        recursive_bubble.reset_recursion()
        symbolic_router.reset_organs()
        return True
    
    detector.register_recovery_callback('custom_system', custom_recovery_callback)
    
    # Simulate a failure requiring custom recovery
    from dawn_core.stable_state_core import StabilityEvent
    
    test_event = StabilityEvent(
        event_id="demo_event",
        timestamp=datetime.now(),
        event_type="demo_failure",
        stability_score=0.4,
        failing_systems=['custom_system'],
        degradation_rate=-0.2,
        recovery_action=RecoveryAction.SELECTIVE_RESTART
    )
    
    print("🚨 Simulating system failure requiring selective restart...")
    success = detector.execute_recovery(test_event)
    print(f"   Recovery result: {'✅ Success' if success else '❌ Failed'}")
    print()
    
    # Test rollback functionality if we have snapshots
    if detector.golden_snapshots:
        print("🔙 Testing rollback functionality...")
        latest_snapshot = detector.golden_snapshots[-1]
        
        rollback_event = StabilityEvent(
            event_id="demo_rollback",
            timestamp=datetime.now(),
            event_type="demo_rollback_test",
            stability_score=0.2,
            failing_systems=['all_systems'],
            degradation_rate=-0.5,
            recovery_action=RecoveryAction.AUTO_ROLLBACK,
            rollback_target=latest_snapshot
        )
        
        print(f"   Rolling back to: {latest_snapshot}")
        rollback_success = detector.execute_recovery(rollback_event)
        print(f"   Rollback result: {'✅ Success' if rollback_success else '❌ Failed'}")
        print()
    
    # Final stability check
    print("📊 Final System Health Check...")
    final_check = detector.calculate_stability_score()
    
    print(f"   Overall Stability: {final_check.overall_stability:.3f}")
    print(f"   System Status: {final_check.stability_level.name}")
    
    if final_check.overall_stability >= 0.7:
        print("   ✅ System is stable and healthy")
    elif final_check.overall_stability >= 0.5:
        print("   ⚠️ System has minor issues but is functional")
    else:
        print("   ❌ System requires attention")
    
    print()
    
    # Demonstrate stability trends (if we have enough history)
    if len(detector.stability_history) >= 5:
        print("📈 Stability Trends Analysis:")
        
        recent_scores = [m.overall_stability for m in list(detector.stability_history)[-10:]]
        if len(recent_scores) >= 3:
            import numpy as np
            trend_slope = np.polyfit(range(len(recent_scores)), recent_scores, 1)[0]
            
            if trend_slope > 0.01:
                trend_desc = "📈 Improving"
            elif trend_slope < -0.01:
                trend_desc = "📉 Degrading"
            else:
                trend_desc = "📊 Stable"
                
            print(f"   Trend: {trend_desc} (slope: {trend_slope:.4f})")
            print(f"   Recent scores: {[f'{s:.2f}' for s in recent_scores[-5:]]}")
        print()
    
    # Cleanup
    print("🔒 Shutting down stable state detector...")
    detector.stop_monitoring()
    print("   ✓ Monitoring stopped")
    print()
    
    print("🔒 " + "="*60)
    print("🔒 STABLE STATE DETECTION DEMO COMPLETE")
    print("🔒 " + "="*60)
    print()
    
    print("Key Features Demonstrated:")
    print("  ✅ Continuous stability monitoring")
    print("  ✅ Mathematical stability criteria")
    print("  ✅ Golden snapshot capture")
    print("  ✅ Degradation detection and prediction")
    print("  ✅ Automatic recovery mechanisms")
    print("  ✅ Rollback to stable states")
    print("  ✅ Custom recovery callbacks")
    print("  ✅ Integration with DAWN modules")
    print()
    
    print("DAWN now has automatic protection against unstable states!")


if __name__ == "__main__":
    try:
        demonstrate_stable_state_system()
    except KeyboardInterrupt:
        print("\n\n🔒 Demo interrupted by user")
    except Exception as e:
        print(f"\n\n❌ Error during demo: {e}")
        import traceback
        traceback.print_exc()
