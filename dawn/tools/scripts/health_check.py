#!/usr/bin/env python3
"""
DAWN Health Check Script
========================

Quick import/interface check to catch mismatches fast.
"""

import importlib
import sys
from pathlib import Path

def expect(mod, attr=None):
    """Import module and optionally check for attribute"""
    try:
        m = importlib.import_module(mod)
        if attr and not hasattr(m, attr):
            raise ImportError(f"{mod} missing {attr}")
        return m
    except ImportError as e:
        print(f"❌ Import failed: {e}")
        return None

def main():
    """Run health checks"""
    print("🏥 DAWN Health Check")
    print("=" * 40)
    
    # Core dependencies
    print("📦 Checking core dependencies...")
    success = True
    
    modules = ["numpy", "pandas", "psutil", "cv2", "PIL", "pyttsx3", "rich"]
    for mod in modules:
        if expect(mod):
            print(f"   ✅ {mod}")
        else:
            success = False
            
    # DAWN core modules
    print("\n🧠 Checking DAWN core modules...")
    dawn_modules = [
        "dawn_core.unified_consciousness_engine",
        "dawn_core.tracer", 
        "dawn_core.stable_state",
        "dawn_core.telemetry_analytics",
        "dawn_core.visual_consciousness"
    ]
    
    for mod in dawn_modules:
        if expect(mod):
            print(f"   ✅ {mod}")
        else:
            success = False
    
    # Key class imports
    print("\n🔧 Checking key classes...")
    try:
        from dawn_core.unified_consciousness_engine import (
            UnifiedDecisionArchitecture,
            ConsciousnessCorrelationEngine,
            MetaCognitiveReflectionEngine,
            ConsciousnessState,
            ConsciousnessIntegrationLevel
        )
        print("   ✅ Consciousness engine classes")
        
        from dawn_core.tracer import DAWNTracer
        print("   ✅ DAWNTracer")
        
        from dawn_core.stable_state import StableStateDetector
        print("   ✅ StableStateDetector")
        
        from dawn_core.telemetry_analytics import TelemetryAnalytics
        print("   ✅ TelemetryAnalytics")
        
    except Exception as e:
        print(f"   ❌ Class import failed: {e}")
        success = False
    
    # Test basic functionality
    print("\n⚡ Testing basic functionality...")
    try:
        # Test tracer
        tracer = DAWNTracer()
        with tracer.trace("health_check", "test_operation") as t:
            t.log_metric("test_metric", 1.0)
        print("   ✅ Tracer working")
        
        # Test stability detector
        detector = StableStateDetector()
        metrics = detector.calculate_stability_score()
        print(f"   ✅ Stability detector working (score: {metrics.overall_stability:.3f})")
        
        # Test analytics (brief)
        analytics = TelemetryAnalytics()
        analytics.ingest_telemetry("test_system", "test_metric", 0.8)
        print("   ✅ Analytics working")
        analytics.stop_analytics()
        
    except Exception as e:
        print(f"   ❌ Functionality test failed: {e}")
        success = False
    
    # Final result
    print("\n" + "=" * 40)
    if success:
        print("✅ Health check passed - all systems operational!")
        return 0
    else:
        print("❌ Health check failed - see errors above")
        return 1

if __name__ == "__main__":
    # Add current directory to path
    sys.path.insert(0, str(Path(__file__).parent.parent))
    sys.exit(main())
