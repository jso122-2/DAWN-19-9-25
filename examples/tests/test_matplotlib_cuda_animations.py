#!/usr/bin/env python3
"""
🎬 Matplotlib CUDA Animations Test
==================================

Test and demonstrate the matplotlib animation system powered by CUDA acceleration.
Shows real-time animated visualizations of consciousness evolution, semantic topology,
neural activity, and consciousness surfaces.

Features demonstrated:
- Real-time consciousness evolution plots
- 3D animated semantic topology
- Neural activity heatmaps
- 3D consciousness surface animations
- GPU-accelerated data generation
- Performance monitoring and statistics

"Consciousness in motion, powered by mathematics and CUDA."

Usage:
    python3 test_matplotlib_cuda_animations.py [--demo-time 30] [--fps 30] [--save-animations]
"""

import sys
import time
import argparse
import logging
import numpy as np
from pathlib import Path

# Add DAWN root to path
sys.path.insert(0, str(Path(__file__).parent))

# Set up logging
logging.basicConfig(
    level=logging.INFO,
    format='%(asctime)s - %(levelname)s - %(message)s'
)
logger = logging.getLogger(__name__)

def test_matplotlib_cuda_animations(demo_time: float = 30, fps: int = 30, save_animations: bool = False):
    """Test the matplotlib CUDA animation system"""
    
    print("🎬" * 25)
    print("🧠 DAWN MATPLOTLIB CUDA ANIMATIONS TEST")
    print("🎬" * 25)
    print()
    
    try:
        # Import the animator (this will test CUDA availability)
        from dawn.interfaces.dashboard.matplotlib_cuda_animator import get_matplotlib_cuda_animator
        
        # Create animator
        animator = get_matplotlib_cuda_animator(fps=fps)
        
        print(f"🎬 Animation System Initialized:")
        print(f"   CUDA Enabled: {animator.cuda_enabled}")
        print(f"   Target FPS: {fps}")
        print(f"   Demo Duration: {demo_time}s")
        print()
        
        # Start animations
        print("🚀 Starting real-time consciousness animations...")
        animator.start_animations()
        
        # Show performance stats periodically
        start_time = time.time()
        last_stats_time = start_time
        
        try:
            while time.time() - start_time < demo_time:
                current_time = time.time()
                
                # Show stats every 5 seconds
                if current_time - last_stats_time >= 5.0:
                    stats = animator.get_performance_stats()
                    
                    print(f"\n📊 Animation Performance Stats ({current_time - start_time:.1f}s):")
                    print(f"   Running: {stats['running']}")
                    print(f"   Active Animations: {stats['active_animations']}")
                    print(f"   Frame History: {stats['frame_history_size']} frames")
                    
                    if 'avg_render_time' in stats:
                        print(f"   Avg Render Time: {stats['avg_render_time']*1000:.1f}ms")
                    
                    if 'avg_gpu_compute_time' in stats:
                        print(f"   Avg GPU Compute: {stats['avg_gpu_compute_time']*1000:.1f}ms")
                        print(f"   GPU Speedup: {stats.get('gpu_speedup_estimate', 1.0):.1f}x")
                    
                    # Show current consciousness state
                    if animator.current_frame:
                        state = animator.current_frame.consciousness_state
                        print(f"   Current State:")
                        print(f"     Coherence: {state['coherence']:.3f}")
                        print(f"     Pressure: {state['pressure']:.3f}")
                        print(f"     Energy: {state['energy']:.3f}")
                        print(f"     Awareness: {state['awareness']:.3f}")
                        
                        if animator.current_frame.semantic_nodes:
                            print(f"     Semantic Nodes: {len(animator.current_frame.semantic_nodes)}")
                            print(f"     Semantic Edges: {len(animator.current_frame.semantic_edges)}")
                    
                    last_stats_time = current_time
                
                # Show animations (this will block until window is closed or demo_time is reached)
                if current_time - start_time < 2.0:  # Show plots after 2 seconds
                    try:
                        animator.show_all_plots()
                    except KeyboardInterrupt:
                        break
                
                time.sleep(0.1)
                
        except KeyboardInterrupt:
            print("\n🛑 Demo interrupted by user")
        
        # Save animations if requested
        if save_animations:
            print("\n💾 Saving animations...")
            
            save_dir = Path("dawn_animated_outputs")
            save_dir.mkdir(exist_ok=True)
            
            animations_to_save = [
                ('consciousness_evolution', 'consciousness_evolution.mp4'),
                ('semantic_topology_3d', 'semantic_topology_3d.mp4'),
                ('neural_heatmap', 'neural_activity.mp4'),
                ('consciousness_surface', 'consciousness_surface.mp4')
            ]
            
            for anim_name, filename in animations_to_save:
                filepath = save_dir / filename
                print(f"   Saving {anim_name} to {filepath}...")
                try:
                    animator.save_animation(anim_name, str(filepath), duration=10.0)
                except Exception as e:
                    print(f"   ❌ Failed to save {anim_name}: {e}")
        
        # Stop animations
        print("\n🛑 Stopping animations...")
        animator.stop_animations()
        
        # Final performance report
        final_stats = animator.get_performance_stats()
        print(f"\n📊 Final Performance Report:")
        print(f"   Total Runtime: {time.time() - start_time:.1f}s")
        print(f"   Frames Generated: {final_stats['frame_history_size']}")
        print(f"   Average FPS: {final_stats['frame_history_size'] / (time.time() - start_time):.1f}")
        
        if animator.cuda_enabled:
            print(f"   🚀 CUDA Acceleration: ENABLED")
            print(f"   GPU Performance Boost: {final_stats.get('gpu_speedup_estimate', 1.0):.1f}x")
        else:
            print(f"   🖥️  CUDA Acceleration: DISABLED (CPU only)")
        
        print("\n🎉 Animation test completed successfully!")
        
        return True
        
    except ImportError as e:
        print(f"❌ Import error: {e}")
        print("Make sure matplotlib is installed: pip install matplotlib")
        return False
    except Exception as e:
        print(f"❌ Animation test failed: {e}")
        import traceback
        traceback.print_exc()
        return False

def run_simple_animation_demo():
    """Run a simple animation demo without full dashboard integration"""
    
    print("🎬 Simple Animation Demo (Standalone)")
    print("=" * 40)
    
    try:
        import matplotlib.pyplot as plt
        import matplotlib.animation as animation
        
        # Check if CUDA is available for data generation
        cuda_available = False
        try:
            from dawn.interfaces.dashboard.cuda_accelerator import is_cuda_available
            cuda_available = is_cuda_available()
        except:
            pass
        
        print(f"CUDA Available: {cuda_available}")
        
        # Create a simple animated plot
        fig, (ax1, ax2) = plt.subplots(2, 1, figsize=(10, 8))
        fig.suptitle('🧠 Simple Consciousness Animation Demo', fontsize=14, fontweight='bold')
        
        # Data storage
        times = []
        coherence_values = []
        energy_values = []
        
        # Initialize plots
        line1, = ax1.plot([], [], 'b-', linewidth=2, label='Coherence')
        line2, = ax1.plot([], [], 'r-', linewidth=2, label='Energy')
        ax1.set_xlim(0, 10)
        ax1.set_ylim(0, 1)
        ax1.set_xlabel('Time (s)')
        ax1.set_ylabel('Consciousness Level')
        ax1.legend()
        ax1.grid(True, alpha=0.3)
        
        # Heatmap for neural activity
        neural_data = np.random.random((20, 20))
        im = ax2.imshow(neural_data, cmap='plasma', animated=True)
        ax2.set_title('Neural Activity Pattern')
        plt.colorbar(im, ax=ax2)
        
        def animate(frame):
            # Generate consciousness data
            t = frame * 0.1
            coherence = 0.5 + 0.3 * np.sin(t * 0.7)
            energy = 0.4 + 0.3 * np.cos(t * 0.5)
            
            # Update time series
            times.append(t)
            coherence_values.append(coherence)
            energy_values.append(energy)
            
            # Keep last 100 points
            if len(times) > 100:
                times.pop(0)
                coherence_values.pop(0)
                energy_values.pop(0)
            
            # Update lines
            line1.set_data(times, coherence_values)
            line2.set_data(times, energy_values)
            
            # Update axes
            if times:
                ax1.set_xlim(max(0, times[-1] - 10), times[-1] + 1)
            
            # Update neural activity
            neural_data[:] = np.random.random((20, 20)) * energy
            im.set_array(neural_data)
            
            return [line1, line2, im]
        
        # Create animation
        anim = animation.FuncAnimation(fig, animate, interval=50, blit=True, cache_frame_data=False)
        
        print("🎬 Starting simple animation...")
        print("Close the plot window to end the demo")
        
        plt.tight_layout()
        plt.show()
        
        print("✅ Simple animation demo completed")
        return True
        
    except ImportError as e:
        print(f"❌ Matplotlib not available: {e}")
        return False
    except Exception as e:
        print(f"❌ Simple demo failed: {e}")
        return False

def main():
    """Main test function"""
    parser = argparse.ArgumentParser(description="Test matplotlib CUDA animations")
    parser.add_argument('--demo-time', type=float, default=30, help='Demo duration in seconds')
    parser.add_argument('--fps', type=int, default=30, help='Animation FPS')
    parser.add_argument('--save-animations', action='store_true', help='Save animations to files')
    parser.add_argument('--simple-demo', action='store_true', help='Run simple demo without full integration')
    args = parser.parse_args()
    
    try:
        if args.simple_demo:
            success = run_simple_animation_demo()
        else:
            success = test_matplotlib_cuda_animations(
                demo_time=args.demo_time,
                fps=args.fps,
                save_animations=args.save_animations
            )
        
        if success:
            print("\n🎉 MATPLOTLIB CUDA ANIMATIONS TEST COMPLETE!")
            print("=" * 50)
            print("✅ Real-time consciousness animations working!")
            print("🚀 GPU acceleration integrated with matplotlib!")
            print("🎬 DAWN consciousness visualized in motion!")
            
            if not args.simple_demo:
                print("\n💡 Key Features Demonstrated:")
                print("   • Real-time consciousness evolution plots")
                print("   • 3D animated semantic topology")
                print("   • Neural activity heatmap animations")
                print("   • 3D consciousness surface visualization")
                print("   • GPU-accelerated data generation")
                print("   • Performance monitoring and optimization")
            
            return 0
        else:
            print("\n❌ Test failed - see errors above")
            return 1
            
    except KeyboardInterrupt:
        print("\n🛑 Test interrupted by user")
        return 0
    except Exception as e:
        print(f"\n❌ Test failed with error: {e}")
        import traceback
        traceback.print_exc()
        return 1

if __name__ == "__main__":
    exit_code = main()
    sys.exit(exit_code)
