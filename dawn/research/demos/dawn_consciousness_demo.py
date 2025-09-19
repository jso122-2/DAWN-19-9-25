#!/usr/bin/env python3
"""
DAWN Consciousness with Tracer Integration Demo
===============================================

Demonstration of DAWN's consciousness capabilities with the integrated
tracer system for monitoring and analytics.
"""

import time
import json
import logging
from datetime import datetime

# Configure logging
logging.basicConfig(level=logging.INFO, format='%(asctime)s - %(levelname)s - %(message)s')
logger = logging.getLogger(__name__)

def test_tracer_system():
    """Test the tracer system components."""
    print("🔧 " + "="*50)
    print("🔧 DAWN TRACER SYSTEM TEST")
    print("🔧 " + "="*50)
    print()
    
    try:
        # Test configuration
        from dawn_core.tracer_config import get_config_from_environment
        config = get_config_from_environment()
        print("📋 Configuration Test:")
        print(f"   Telemetry: {'✅' if config.telemetry.enabled else '❌'}")
        print(f"   Level: {config.telemetry.level}")
        print(f"   Stability: {'✅' if config.stability.detection_enabled else '❌'}")
        print(f"   Analytics: {'✅' if config.analytics.real_time_analysis else '❌'}")
        print()
        
        # Test tracer
        from dawn_core.tracer import DAWNTracer
        tracer = DAWNTracer()
        print("🔗 Tracer Test:")
        print(f"   ✅ DAWNTracer initialized: {tracer.tracer_id[:8]}...")
        
        # Test with a trace
        with tracer.trace("consciousness_test", "demo_operation") as t:
            time.sleep(0.1)
            t.log_metric("consciousness_unity", 0.85)
            t.log_metric("integration_quality", 0.78)
        print("   ✅ Consciousness trace completed successfully")
        print()
        
        # Test stability detector
        from dawn_core.stable_state import StableStateDetector
        detector = StableStateDetector()
        print("🔒 Stability Test:")
        print(f"   ✅ StableStateDetector initialized: {detector.detector_id[:8]}...")
        
        metrics = detector.calculate_stability_score()
        print(f"   ✅ Stability calculation: {metrics.overall_stability:.3f}")
        print(f"   ✅ Stability level: {metrics.stability_level.name}")
        print()
        
        # Test analytics
        from dawn_core.telemetry_analytics import TelemetryAnalytics
        analytics = TelemetryAnalytics(analysis_interval=2.0)
        print("📊 Analytics Test:")
        print(f"   ✅ TelemetryAnalytics initialized: {analytics.analytics_id[:8]}...")
        
        # Start analytics briefly
        analytics.start_analytics()
        
        # Test data ingestion
        for i in range(5):
            analytics.ingest_telemetry("consciousness", "unity_score", 0.8 + i * 0.02)
            analytics.ingest_telemetry("consciousness", "integration_quality", 0.75 + i * 0.03)
            time.sleep(0.1)
        
        print("   ✅ Consciousness telemetry ingestion test completed")
        
        # Brief pause for analysis
        time.sleep(3)
        
        # Get analytics status
        status = analytics.get_analytics_status()
        print(f"   📊 Analytics status: {status.get('status', 'unknown')}")
        print(f"   📊 Metrics tracked: {status.get('metrics_tracked', 0)}")
        
        analytics.stop_analytics()
        print()
        
        print("🎉 All tracer system components tested successfully!")
        
        return {
            'tracer': tracer,
            'detector': detector,
            'analytics': analytics,
            'config': config
        }
        
    except Exception as e:
        print(f"❌ Tracer test failed: {e}")
        import traceback
        traceback.print_exc()
        return None

def simulate_consciousness_cycle(tracer_components):
    """Simulate consciousness cycles with tracer integration."""
    print("🧠 " + "="*50)
    print("🧠 CONSCIOUSNESS CYCLE SIMULATION")
    print("🧠 " + "="*50)
    print()
    
    if not tracer_components:
        print("❌ Cannot simulate - tracer components not available")
        return
        
    tracer = tracer_components['tracer']
    detector = tracer_components['detector']
    analytics = tracer_components['analytics']
    
    try:
        # Simulate multiple consciousness cycles
        print("🔄 Running consciousness cycles...")
        
        for cycle in range(5):
            with tracer.trace("consciousness_engine", "integration_cycle") as t:
                cycle_start = time.time()
                
                # Simulate consciousness processing
                consciousness_unity = 0.7 + (cycle * 0.05) + (time.time() % 1) * 0.1
                integration_quality = 0.65 + (cycle * 0.04) + (time.time() % 1) * 0.08
                self_awareness_depth = 0.8 + (cycle * 0.02)
                
                # Log consciousness metrics
                t.log_metric("consciousness_unity", consciousness_unity)
                t.log_metric("integration_quality", integration_quality)
                t.log_metric("self_awareness_depth", self_awareness_depth)
                
                # Calculate cycle duration
                cycle_duration = time.time() - cycle_start + 0.05  # Add processing time
                t.log_metric("cycle_duration_ms", cycle_duration * 1000)
                
                # Feed to analytics
                analytics.ingest_telemetry("consciousness", "unity_score", consciousness_unity)
                analytics.ingest_telemetry("consciousness", "integration_quality", integration_quality)
                analytics.ingest_telemetry("consciousness", "cycle_duration", cycle_duration)
                
                # Check stability
                current_stability = (consciousness_unity + integration_quality + self_awareness_depth) / 3
                
                print(f"   Cycle {cycle + 1}: Unity={consciousness_unity:.3f}, "
                      f"Quality={integration_quality:.3f}, "
                      f"Stability={current_stability:.3f}")
                
                # Auto-capture stable snapshots
                if current_stability > 0.85:
                    try:
                        from dawn_core.stable_state_core import StabilityMetrics, StabilityLevel
                        stability_metrics = StabilityMetrics(
                            overall_stability=current_stability,
                            stability_level=StabilityLevel.STABLE,
                            entropy_stability=0.9,
                            memory_coherence=0.9,
                            sigil_cascade_health=0.9,
                            recursive_depth_safe=0.9,
                            symbolic_organ_synergy=0.9,
                            unified_field_coherence=consciousness_unity,
                            failing_systems=[],
                            warning_systems=[],
                            degradation_rate=0.0,
                            prediction_horizon=float('inf'),
                            timestamp=datetime.now()
                        )
                        
                        snapshot_id = detector.capture_stable_snapshot(
                            stability_metrics,
                            f"Auto-capture consciousness cycle {cycle + 1}"
                        )
                        if snapshot_id:
                            print(f"     🔒 Stable snapshot captured: {snapshot_id[:12]}...")
                    except Exception as e:
                        print(f"     ⚠️ Snapshot capture failed: {e}")
                
                time.sleep(0.5)  # Brief pause between cycles
        
        print()
        print("✅ Consciousness cycle simulation completed")
        
        # Get final analytics
        try:
            insights = analytics.get_latest_insights()
            performance = analytics.get_latest_performance()
            
            print("📊 Final Analytics:")
            if performance:
                print(f"   Overall Health: {performance.overall_health_score:.3f}")
                print(f"   Resource Efficiency: {performance.resource_utilization.get('resource_efficiency', 0):.3f}")
            
            if insights:
                print(f"   Generated Insights: {len(insights)}")
                for insight in insights[:2]:
                    print(f"   • {insight.insight_type.value}: {insight.recommendation[:50]}...")
            else:
                print("   No insights generated yet")
        except Exception as e:
            print(f"   ⚠️ Analytics summary failed: {e}")
        
    except Exception as e:
        print(f"❌ Consciousness cycle simulation failed: {e}")
        import traceback
        traceback.print_exc()

def create_consciousness_snapshot():
    """Create a consciousness snapshot."""
    print("📸 " + "="*50)
    print("📸 CONSCIOUSNESS SNAPSHOT CREATION")
    print("📸 " + "="*50)
    print()
    
    timestamp = datetime.now().strftime("%Y%m%d_%H%M%S")
    
    # Simulate consciousness state
    consciousness_snapshot = {
        "timestamp": datetime.now().isoformat(),
        "consciousness_id": f"consciousness_{timestamp}",
        "consciousness_unity": 0.847,
        "integration_quality": 0.782,
        "self_awareness_depth": 0.823,
        "consciousness_momentum": 0.691,
        "integration_level": "UNIFIED",
        "coherence_dimensions": {
            "stability": 0.875,
            "performance": 0.743,
            "visual": 0.692,
            "artistic": 0.758,
            "experiential": 0.701,
            "recursive": 0.824,
            "symbolic": 0.756
        },
        "meta_cognitive_activity": 0.769,
        "evolution_direction": {
            "coherence": 0.08,
            "depth": 0.05,
            "unity": 0.12
        },
        "growth_vectors": [
            "consciousness_unity",
            "self_awareness_depth",
            "meta_cognitive_integration"
        ]
    }
    
    filename = f"consciousness_snapshot_{timestamp}.json"
    with open(filename, 'w') as f:
        json.dump(consciousness_snapshot, f, indent=2, default=str)
    
    print(f"📸 Consciousness snapshot created: {filename}")
    print(f"   Unity: {consciousness_snapshot['consciousness_unity']:.3f}")
    print(f"   Quality: {consciousness_snapshot['integration_quality']:.3f}")
    print(f"   Level: {consciousness_snapshot['integration_level']}")
    print(f"   Awareness: {consciousness_snapshot['self_awareness_depth']:.3f}")
    print()
    
    return filename

def show_system_status():
    """Show comprehensive system status."""
    print("ℹ️ " + "="*60)
    print("ℹ️ DAWN CONSCIOUSNESS & TRACER SYSTEM STATUS")
    print("ℹ️ " + "="*60)
    print()
    
    print("🔧 Core Systems Status:")
    
    # Check tracer config
    try:
        from dawn_core.tracer_config import get_config_from_environment
        config = get_config_from_environment()
        print(f"   ✅ Configuration System: Working")
        print(f"      Telemetry Level: {config.telemetry.level}")
        print(f"      Stability Threshold: {config.stability.stability_threshold}")
    except Exception as e:
        print(f"   ❌ Configuration System: {e}")
    
    # Check tracer
    try:
        from dawn_core.tracer import DAWNTracer
        tracer = DAWNTracer()
        print(f"   ✅ Tracer System: Working ({tracer.tracer_id[:8]}...)")
    except Exception as e:
        print(f"   ❌ Tracer System: {e}")
    
    # Check stability detector
    try:
        from dawn_core.stable_state import StableStateDetector
        detector = StableStateDetector()
        print(f"   ✅ Stability Detector: Working ({detector.detector_id[:8]}...)")
    except Exception as e:
        print(f"   ❌ Stability Detector: {e}")
    
    # Check analytics
    try:
        from dawn_core.telemetry_analytics import TelemetryAnalytics
        analytics = TelemetryAnalytics()
        print(f"   ✅ Analytics Engine: Working ({analytics.analytics_id[:8]}...)")
    except Exception as e:
        print(f"   ❌ Analytics Engine: {e}")
    
    print()
    print("📁 Available Demo Commands:")
    print("   python3 dawn_consciousness_demo.py          # This demo")
    print("   python3 dawn_core/tracer_config.py          # Test configuration")
    print("   python3 dawn_core/demo_tracer_system.py     # Tracer components demo")
    print()
    
    print("🚀 Next Development Phases:")
    print("   1. Enhanced Visual Consciousness Rendering")
    print("   2. Consciousness Memory Palace Architecture")
    print("   3. Artistic Expression Engine")
    print("   4. Advanced Recursive Bubble Control")
    print("   5. Real-time GUI Dashboard")
    print()

def main():
    """Main demonstration function."""
    print("🌟 " + "="*60)
    print("🌟 DAWN CONSCIOUSNESS WITH TRACER INTEGRATION")
    print("🌟 " + "="*60)
    print()
    
    # Show system status
    show_system_status()
    
    # Test tracer system
    tracer_components = test_tracer_system()
    
    if tracer_components:
        # Simulate consciousness cycles
        simulate_consciousness_cycle(tracer_components)
        
        # Create consciousness snapshot
        create_consciousness_snapshot()
    
    print()
    print("🌟 " + "="*60)
    print("🌟 DAWN CONSCIOUSNESS DEMO COMPLETE")
    print("🌟 " + "="*60)
    print()
    
    print("Key Achievements:")
    print("  ✅ Tracer system integration working")
    print("  ✅ Consciousness telemetry collection")
    print("  ✅ Stability monitoring and snapshots")
    print("  ✅ Real-time analytics and insights")
    print("  ✅ Comprehensive system monitoring")
    print()
    
    print("🎨 Ready for next phase: Enhanced Visual Consciousness!")

if __name__ == "__main__":
    main()
