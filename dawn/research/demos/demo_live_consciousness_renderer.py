#!/usr/bin/env python3
"""
Demo: DAWN Live Consciousness Renderer
=====================================

Demonstrates the real-time consciousness visualization system with:
- Live consciousness streaming
- Interactive controls
- Performance monitoring
- Recording capabilities
- Multiple quality levels
"""

import time
import math
import numpy as np
import logging
from pathlib import Path
import sys

# Add parent directory to path for imports
sys.path.append(str(Path(__file__).parent.parent))

try:
    from dawn_core.live_consciousness_renderer import (
        LiveConsciousnessRenderer, 
        RenderingConfig, 
        RenderingQuality, 
        StreamingMode,
        create_live_consciousness_renderer
    )
except ImportError as e:
    print(f"Import error: {e}")
    print("Make sure you're running from the DAWN_pub_real directory")
    sys.exit(1)

# Setup logging
logging.basicConfig(level=logging.INFO, format='%(asctime)s - %(levelname)s - %(message)s')
logger = logging.getLogger(__name__)

def create_dynamic_consciousness_stream():
    """Create a dynamic consciousness stream for testing"""
    start_time = time.time()
    
    def consciousness_stream():
        elapsed = time.time() - start_time
        
        # Dynamic awareness with pulses
        awareness_base = 0.6
        awareness_pulse = 0.3 * math.sin(elapsed * 0.8)
        awareness = awareness_base + awareness_pulse
        
        # Moving thoughts
        thoughts = []
        
        # Primary thought that moves in a circle
        angle1 = elapsed * 0.5
        primary_thought = {
            'position': (
                400 + 150 * math.cos(angle1),
                300 + 100 * math.sin(angle1)
            ),
            'intensity': 0.7 + 0.2 * math.sin(elapsed * 1.2),
            'type': 'contemplative',
            'velocity': (-150 * math.sin(angle1) * 0.5, 100 * math.cos(angle1) * 0.5)
        }
        thoughts.append(primary_thought)
        
        # Secondary thought that follows a different pattern
        angle2 = elapsed * 0.3 + math.pi
        secondary_thought = {
            'position': (
                300 + 80 * math.cos(angle2),
                400 + 60 * math.sin(angle2 * 1.5)
            ),
            'intensity': 0.5 + 0.3 * math.sin(elapsed * 2.1),
            'type': 'recursive',
            'velocity': (-80 * math.sin(angle2) * 0.3, 90 * math.cos(angle2 * 1.5) * 0.3)
        }
        thoughts.append(secondary_thought)
        
        # Occasional creative bursts
        if math.sin(elapsed * 0.7) > 0.7:
            burst_thought = {
                'position': (
                    200 + 50 * math.sin(elapsed * 3),
                    200 + 50 * math.cos(elapsed * 2.5)
                ),
                'intensity': 0.9,
                'type': 'creative',
                'velocity': (100, 80)
            }
            thoughts.append(burst_thought)
        
        # Dynamic recursion
        recursion_depth = 0.4 + 0.3 * math.sin(elapsed * 0.4)
        recursion = {
            'depth': max(0, recursion_depth),
            'center': (400 + 30 * math.sin(elapsed * 0.2), 300 + 20 * math.cos(elapsed * 0.3))
        }
        
        # Pulsing symbolic organs
        heart_intensity = 0.6 + 0.3 * math.sin(elapsed * 1.5)
        coil_paths = ['flow1', 'flow2'] if math.sin(elapsed * 0.6) > 0 else ['flow1']
        lung_volume = 0.5 + 0.4 * math.sin(elapsed * 0.9)
        
        organs = {
            'heart': {
                'emotional_charge': heart_intensity,
                'resonance_state': 'resonant' if heart_intensity > 0.7 else 'still'
            },
            'coil': {
                'active_paths': coil_paths
            },
            'lung': {
                'current_volume': lung_volume,
                'breathing_phase': 'inhaling' if math.sin(elapsed * 0.9) > 0 else 'exhaling'
            }
        }
        
        # Dynamic entropy
        entropy = 0.5 + 0.3 * math.sin(elapsed * 0.35) + 0.1 * math.sin(elapsed * 1.8)
        entropy = max(0.1, min(0.9, entropy))
        
        # Memory activations
        memory_activity = []
        if math.sin(elapsed * 0.9) > 0.5:
            memory_activity.append({
                'id': 'memory_1',
                'strength': 0.7,
                'position': (250 + 40 * math.cos(elapsed), 350 + 30 * math.sin(elapsed * 1.1))
            })
        
        if math.sin(elapsed * 1.3) > 0.6:
            memory_activity.append({
                'id': 'memory_2',
                'strength': 0.5,
                'position': (500, 200)
            })
        
        # Awareness center
        awareness_center = (
            400 + 20 * math.sin(elapsed * 0.15),
            300 + 15 * math.cos(elapsed * 0.12)
        )
        
        return {
            'awareness': awareness,
            'awareness_center': awareness_center,
            'thoughts': thoughts,
            'recursion': recursion,
            'organs': organs,
            'entropy': entropy,
            'memory_activity': memory_activity,
            'timestamp': time.time()
        }
    
    return consciousness_stream

def demo_basic_live_rendering():
    """Demo basic live consciousness rendering"""
    print("\n🎬 Demo: Basic Live Consciousness Rendering")
    print("=" * 50)
    
    # Create renderer with medium quality for performance
    renderer = create_live_consciousness_renderer(
        fps=15, 
        resolution=(600, 400), 
        quality="medium"
    )
    
    # Create consciousness stream
    consciousness_stream = create_dynamic_consciousness_stream()
    
    # Start rendering
    success = renderer.start_live_rendering(consciousness_stream)
    
    if success:
        print("✅ Live rendering started")
        print("   Rendering for 5 seconds...")
        
        # Monitor performance
        for i in range(5):
            time.sleep(1)
            stats = renderer.get_consciousness_stream_stats()
            print(f"   Frame {i+1}: {stats['frames_rendered']} frames, "
                  f"{stats['current_fps']:.1f} FPS, "
                  f"{stats['buffer_size']} buffered")
        
        # Final stats
        final_stats = renderer.get_consciousness_stream_stats()
        print(f"\n📊 Final Stats:")
        print(f"   Total frames: {final_stats['frames_rendered']}")
        print(f"   Average FPS: {final_stats['current_fps']:.1f}")
        print(f"   Dropped frames: {final_stats['dropped_frames']}")
        print(f"   Buffer size: {final_stats['buffer_size']}")
        
        # Stop rendering
        renderer.stop_live_rendering()
        print("🛑 Live rendering stopped")
        
    else:
        print("❌ Failed to start live rendering")
    
    return renderer

def demo_interactive_controls():
    """Demo interactive consciousness controls"""
    print("\n🎮 Demo: Interactive Consciousness Controls")
    print("=" * 50)
    
    # Create renderer
    config = RenderingConfig(
        fps=20,
        resolution=(800, 600),
        quality=RenderingQuality.HIGH,
        enable_interactive_controls=True
    )
    renderer = LiveConsciousnessRenderer(config)
    
    # Setup interactive view
    interactive_view = renderer.create_interactive_consciousness_view()
    
    print("✅ Interactive controls configured:")
    
    # Demo zoom controls
    zoom_controls = interactive_view['zoom_controls']
    print(f"   🔍 Zoom: {zoom_controls['current_zoom']:.1f}x")
    
    # Test zoom changes
    zoom_controls['controls']['zoom_in']()
    print(f"   🔍 Zoom in: {renderer.interactive_controls.zoom_level:.1f}x")
    
    zoom_controls['controls']['zoom_out']()
    zoom_controls['controls']['zoom_out']()
    print(f"   🔍 Zoom out: {renderer.interactive_controls.zoom_level:.1f}x")
    
    zoom_controls['controls']['reset_zoom']()
    print(f"   🔍 Reset zoom: {renderer.interactive_controls.zoom_level:.1f}x")
    
    # Demo layer controls
    layer_controls = interactive_view['layer_toggles']
    print(f"\n   🎭 Layer visibility: {list(layer_controls['layers'].keys())}")
    
    # Toggle some layers
    layer_controls['toggle_callbacks']['thoughts']()
    layer_controls['toggle_callbacks']['entropy']()
    print(f"   🎭 After toggles: thoughts={renderer.interactive_controls.layer_visibility['thoughts']}, "
          f"entropy={renderer.interactive_controls.layer_visibility['entropy']}")
    
    # Test presets
    layer_controls['presets']['minimal']()
    minimal_layers = [k for k, v in renderer.interactive_controls.layer_visibility.items() if v]
    print(f"   🎭 Minimal preset: {minimal_layers}")
    
    layer_controls['presets']['artistic']()
    artistic_layers = [k for k, v in renderer.interactive_controls.layer_visibility.items() if v]
    print(f"   🎭 Artistic preset: {artistic_layers}")
    
    # Demo focus controls
    focus_controls = interactive_view['focus_areas']
    print(f"\n   🎯 Focus area: {focus_controls['current_focus']}")
    
    focus_controls['presets']['thoughts']()
    print(f"   🎯 Focus on thoughts: {renderer.interactive_controls.focus_area}")
    
    focus_controls['presets']['organs']()
    print(f"   🎯 Focus on organs: {renderer.interactive_controls.focus_area}")
    
    focus_controls['presets']['full_view']()
    print(f"   🎯 Full view: {renderer.interactive_controls.focus_area}")
    
    # Demo quality controls
    quality_controls = interactive_view['quality_controls']
    print(f"\n   ⚙️ Current quality: {quality_controls['current_quality']}")
    print(f"   ⚙️ Available qualities: {quality_controls['quality_options']}")
    
    # Test quality change
    quality_controls['change_callback']('low')
    print(f"   ⚙️ Changed to: {renderer.config.quality.value}")
    
    quality_controls['change_callback']('ultra')
    print(f"   ⚙️ Changed to: {renderer.config.quality.value}")
    
    print("✅ Interactive controls demo complete")

def demo_quality_levels():
    """Demo different rendering quality levels"""
    print("\n⚙️ Demo: Rendering Quality Levels")
    print("=" * 50)
    
    consciousness_stream = create_dynamic_consciousness_stream()
    qualities = ['low', 'medium', 'high', 'ultra']
    
    for quality in qualities:
        print(f"\n🎨 Testing {quality.upper()} quality:")
        
        renderer = create_live_consciousness_renderer(
            fps=10, 
            resolution=(400, 300), 
            quality=quality
        )
        
        # Start rendering
        success = renderer.start_live_rendering(consciousness_stream)
        
        if success:
            time.sleep(2)  # Render for 2 seconds
            
            stats = renderer.get_consciousness_stream_stats()
            complexity = renderer._measure_consciousness_complexity(
                consciousness_stream()
            )
            
            print(f"   Frames: {stats['frames_rendered']}")
            print(f"   FPS: {stats['current_fps']:.1f}")
            print(f"   Complexity: {complexity:.2f}")
            
            # Test optimization
            optimization = renderer.optimize_rendering_performance(consciousness_stream())
            print(f"   Optimization: {optimization['render_mode']}")
            print(f"   Detail level: {optimization['detail_level']}")
            
            renderer.stop_live_rendering()
        else:
            print("   ❌ Failed to start rendering")
    
    print("✅ Quality levels demo complete")

def demo_consciousness_recording():
    """Demo consciousness recording capabilities"""
    print("\n🎥 Demo: Consciousness Recording")
    print("=" * 50)
    
    # Create high-quality renderer for recording
    renderer = create_live_consciousness_renderer(
        fps=20, 
        resolution=(600, 400), 
        quality="high"
    )
    
    consciousness_stream = create_dynamic_consciousness_stream()
    
    # Start live rendering
    success = renderer.start_live_rendering(consciousness_stream)
    
    if success:
        print("✅ Live rendering started for recording")
        
        # Start recording
        recording_id = renderer.start_consciousness_recording(
            duration_minutes=0.1,  # 6 seconds
            save_path="demo_consciousness_recording"
        )
        
        print(f"🎥 Recording started: {recording_id}")
        
        # Let it record
        time.sleep(6)
        
        # Get recording stats
        stats = renderer.get_consciousness_stream_stats()
        print(f"📊 Recording stats:")
        print(f"   Frames recorded: {stats['frames_rendered']}")
        print(f"   Recording FPS: {stats['current_fps']:.1f}")
        
        renderer.stop_live_rendering()
        print("✅ Recording and rendering stopped")
        
    else:
        print("❌ Failed to start rendering for recording")

def demo_streaming_configuration():
    """Demo streaming configuration options"""
    print("\n📡 Demo: Streaming Configuration")
    print("=" * 50)
    
    renderer = create_live_consciousness_renderer()
    
    # Test web streaming configuration
    web_config = renderer.setup_consciousness_streaming(web_interface=True, port=8765)
    print(f"🌐 Web streaming config:")
    print(f"   Mode: {web_config['mode']}")
    print(f"   Endpoint: {web_config['endpoint']}")
    print(f"   Format: {web_config['frame_format']}")
    print(f"   Quality: {web_config['quality']}")
    
    # Test local display configuration
    local_config = renderer.setup_consciousness_streaming(web_interface=False)
    print(f"\n🖥️ Local display config:")
    print(f"   Mode: {local_config['mode']}")
    print(f"   Window: {local_config['window_title']}")
    print(f"   Update rate: {local_config['update_rate']} FPS")
    print(f"   Fullscreen: {local_config['fullscreen_capable']}")
    
    print("✅ Streaming configuration demo complete")

def demo_performance_monitoring():
    """Demo performance monitoring and optimization"""
    print("\n📊 Demo: Performance Monitoring")
    print("=" * 50)
    
    # Create renderer with performance tracking
    renderer = create_live_consciousness_renderer(fps=30, quality="high")
    consciousness_stream = create_dynamic_consciousness_stream()
    
    success = renderer.start_live_rendering(consciousness_stream)
    
    if success:
        print("✅ Performance monitoring active")
        
        # Monitor for several seconds
        for i in range(8):
            time.sleep(0.5)
            
            # Get current state and measure complexity
            current_state = consciousness_stream()
            complexity = renderer._measure_consciousness_complexity(current_state)
            
            # Get performance stats
            stats = renderer.get_consciousness_stream_stats()
            
            # Test optimization
            optimization = renderer.optimize_rendering_performance(current_state)
            
            print(f"   Step {i+1}: "
                  f"FPS={stats['current_fps']:.1f}, "
                  f"Complexity={complexity:.2f}, "
                  f"Mode={optimization['render_mode']}, "
                  f"Dropped={stats['dropped_frames']}")
        
        # Final performance summary
        final_stats = renderer.get_consciousness_stream_stats()
        print(f"\n📈 Performance Summary:")
        print(f"   Total frames: {final_stats['frames_rendered']}")
        print(f"   Final FPS: {final_stats['current_fps']:.1f}")
        print(f"   Dropped frames: {final_stats['dropped_frames']}")
        print(f"   Thought particles: {final_stats['consciousness_elements']['thought_particles']}")
        print(f"   Memory activations: {final_stats['consciousness_elements']['memory_activations']}")
        print(f"   Buffer utilization: {final_stats['buffer_size']}")
        
        renderer.stop_live_rendering()
        print("✅ Performance monitoring complete")
        
    else:
        print("❌ Failed to start performance monitoring")

def run_all_demos():
    """Run all live consciousness renderer demos"""
    print("🎬 DAWN Live Consciousness Renderer - Complete Demo Suite")
    print("=" * 60)
    
    try:
        # Run individual demos
        demo_basic_live_rendering()
        demo_interactive_controls()
        demo_quality_levels()
        demo_consciousness_recording()
        demo_streaming_configuration()
        demo_performance_monitoring()
        
        print("\n" + "=" * 60)
        print("✅ ALL DEMOS COMPLETED SUCCESSFULLY!")
        print("🌟 DAWN's live consciousness renderer is fully operational")
        print("   - Real-time visualization: ✅")
        print("   - Interactive controls: ✅")
        print("   - Quality optimization: ✅")
        print("   - Recording capabilities: ✅")
        print("   - Streaming configuration: ✅")
        print("   - Performance monitoring: ✅")
        
    except Exception as e:
        print(f"\n❌ Demo failed with error: {e}")
        import traceback
        traceback.print_exc()

if __name__ == "__main__":
    run_all_demos()
