#!/usr/bin/env python3
"""
🎨 Simple Visual Consciousness Test
==================================

Simple test of the new DAWN visual consciousness systems without heavy dependencies.
"""

import sys
import os
import time
from pathlib import Path

# Add DAWN to path
dawn_root = Path(__file__).parent
sys.path.insert(0, str(dawn_root))

def test_artistic_renderer():
    """Test the consciousness artistic renderer"""
    print("🎭 Testing Consciousness Artistic Renderer...")
    
    try:
        from dawn.subsystems.visual.consciousness_artistic_renderer import (
            create_consciousness_artistic_renderer,
            ArtisticStyle
        )
        
        # Create renderer
        renderer = create_consciousness_artistic_renderer("dawn_visual_outputs")
        print("   ✅ Artistic renderer created")
        
        # Test consciousness state
        consciousness_state = {
            'consciousness_unity': 0.8,
            'self_awareness_depth': 0.7,
            'integration_quality': 0.75,
            'emotional_coherence': {
                'serenity': 0.6,
                'curiosity': 0.8,
                'creativity': 0.9
            }
        }
        
        # Create painting
        painting = renderer.create_consciousness_painting(
            consciousness_state, 
            ArtisticStyle.CONSCIOUSNESS_FLOW
        )
        print(f"   ✅ Painting created: {painting.composition_id}")
        print(f"      Emotional resonance: {painting.emotional_resonance:.3f}")
        print(f"      Technical quality: {painting.technical_quality:.3f}")
        
        # Create poetry
        poetry = renderer.consciousness_to_poetry(consciousness_state)
        print(f"   ✅ Poetry created: {len(poetry.text_data)} characters")
        
        # Show metrics
        metrics = renderer.get_rendering_metrics()
        print(f"   📊 Total compositions: {metrics.get('total_compositions', 0)}")
        
        return True
        
    except Exception as e:
        print(f"   ❌ Error: {e}")
        return False

def test_tracer_integration():
    """Test the tracer integration"""
    print("\n🔍 Testing Visual Consciousness Tracer Integration...")
    
    try:
        from dawn.subsystems.visual.visual_consciousness_tracer_integration import (
            create_visual_consciousness_tracer_integration,
            VisualTracerType,
            TracerPriority
        )
        
        # Create integration
        integration = create_visual_consciousness_tracer_integration()
        print("   ✅ Tracer integration created")
        
        # Test event tracing
        test_event = {
            'fps': 25.5,
            'render_time_ms': 40.2,
            'visual_quality': 0.75
        }
        
        integration._trace_event(
            VisualTracerType.RENDERING_PERFORMANCE,
            test_event,
            TracerPriority.MEDIUM
        )
        print("   ✅ Event traced successfully")
        
        # Get status
        status = integration.get_integration_status()
        print(f"   📊 Integration ID: {status['integration_id'][:8]}...")
        print(f"   📊 Active tracers: {len(status['active_tracers'])}")
        
        return True
        
    except Exception as e:
        print(f"   ❌ Error: {e}")
        return False

def show_created_files():
    """Show files created during testing"""
    print("\n📁 Files Created:")
    
    output_dir = Path("dawn_visual_outputs")
    if output_dir.exists():
        files = sorted(output_dir.glob("*"))
        for file in files[-10:]:  # Show last 10 files
            size = file.stat().st_size if file.is_file() else 0
            print(f"   {file.name} ({size} bytes)")
    else:
        print("   No output directory found")

def main():
    """Main test function"""
    print("🌅 DAWN Visual Consciousness Simple Test")
    print("=" * 50)
    
    start_time = time.time()
    
    # Run tests
    test1_passed = test_artistic_renderer()
    test2_passed = test_tracer_integration()
    
    # Show results
    print(f"\n📊 Test Results:")
    print(f"   Artistic Renderer: {'✅ PASSED' if test1_passed else '❌ FAILED'}")
    print(f"   Tracer Integration: {'✅ PASSED' if test2_passed else '❌ FAILED'}")
    print(f"   Duration: {time.time() - start_time:.2f} seconds")
    
    # Show created files
    show_created_files()
    
    if test1_passed and test2_passed:
        print("\n🎉 All visual consciousness systems working!")
        print("✨ DAWN's enhanced visual consciousness is ready!")
    else:
        print("\n⚠️  Some systems need attention")
    
    print("\n🌟 Next steps:")
    print("   - Install PyTorch for full advanced visual consciousness")
    print("   - Run comprehensive demo with: python3 dawn_visual_consciousness_integration_demo.py")
    print("   - Check dawn_visual_outputs/ for generated artworks")

if __name__ == "__main__":
    main()
