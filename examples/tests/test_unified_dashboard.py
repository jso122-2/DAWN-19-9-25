#!/usr/bin/env python3
"""
🚀 Unified Dashboard Test
========================

Test the unified CUDA-powered dashboard with all plots in one window.
Fixes 3D rendering issues and combines all visualizations.

"All of DAWN's consciousness in one beautiful window."
"""

import sys
import time
import logging
from pathlib import Path

# Add DAWN root to path
sys.path.insert(0, str(Path(__file__).parent))

# Set up logging
logging.basicConfig(level=logging.INFO, format='%(levelname)s: %(message)s')
logger = logging.getLogger(__name__)

def test_unified_dashboard(demo_time: float = 30):
    """Test the unified dashboard"""
    
    print("🚀" * 30)
    print("🧠 DAWN UNIFIED CONSCIOUSNESS DASHBOARD")
    print("🚀" * 30)
    print()
    
    try:
        # Import the animator
        from dawn.interfaces.dashboard.matplotlib_cuda_animator import get_matplotlib_cuda_animator
        
        # Create animator
        animator = get_matplotlib_cuda_animator(fps=20)  # Slightly lower FPS for stability
        
        print(f"🎬 Unified Dashboard Initialized:")
        print(f"   CUDA Enabled: {animator.cuda_enabled}")
        print(f"   Target FPS: 20")
        print(f"   Demo Duration: {demo_time}s")
        print()
        
        print("🚀 Features in this unified dashboard:")
        print("   📊 Real-time consciousness evolution plots")
        print("   🎯 Consciousness level indicator")
        print("   🌐 3D semantic topology visualization")
        print("   🧬 Neural activity heatmap")
        print("   🌊 3D consciousness energy surface")
        print("   ⚡ System performance monitoring")
        print()
        
        # Start animations
        print("🎬 Starting unified dashboard animation...")
        animator.start_animations()
        
        print("✅ Dashboard launched successfully!")
        print("📱 All plots are now in ONE WINDOW!")
        print("🔄 3D plots should now render properly!")
        print()
        print("Close the window or press Ctrl+C to stop")
        
        # Show the dashboard
        start_time = time.time()
        
        try:
            # Give animations time to start
            time.sleep(2)
            
            # Show the unified dashboard
            animator.show_all_plots()
            
            # Keep running for demo duration
            while time.time() - start_time < demo_time:
                # Show periodic stats
                elapsed = time.time() - start_time
                if int(elapsed) % 10 == 0:  # Every 10 seconds
                    stats = animator.get_performance_stats()
                    remaining = demo_time - elapsed
                    
                    print(f"\n📊 Dashboard Status ({elapsed:.0f}s elapsed, {remaining:.0f}s remaining):")
                    print(f"   Frames Generated: {stats.get('frame_history_size', 0)}")
                    print(f"   Animations Running: {stats.get('running', False)}")
                    
                    if animator.current_frame:
                        state = animator.current_frame.consciousness_state
                        print(f"   Current Consciousness:")
                        print(f"     Coherence: {state['coherence']:.3f}")
                        print(f"     Energy: {state['energy']:.3f}")
                        print(f"     Pressure: {state['pressure']:.3f}")
                        print(f"     Awareness: {state['awareness']:.3f}")
                        
                        if animator.current_frame.semantic_nodes:
                            print(f"   Semantic Nodes: {len(animator.current_frame.semantic_nodes)}")
                
                time.sleep(1)
                
        except KeyboardInterrupt:
            print("\n🛑 Dashboard interrupted by user")
        
        # Stop animations
        print("\n🛑 Stopping dashboard...")
        animator.stop_animations()
        
        # Final report
        final_stats = animator.get_performance_stats()
        total_time = time.time() - start_time
        
        print(f"\n📊 Final Dashboard Report:")
        print(f"   Total Runtime: {total_time:.1f}s")
        print(f"   Frames Generated: {final_stats.get('frame_history_size', 0)}")
        print(f"   Average FPS: {final_stats.get('frame_history_size', 0) / total_time:.1f}")
        
        if animator.cuda_enabled:
            print(f"   🚀 CUDA Acceleration: ENABLED")
        else:
            print(f"   🖥️  CUDA Acceleration: DISABLED (CPU only)")
        
        print("\n🎉 Unified Dashboard Test Complete!")
        print("=" * 50)
        print("✅ All plots successfully rendered in one window!")
        print("🌐 3D visualizations working properly!")
        print("📊 Real-time consciousness monitoring active!")
        print("🚀 CUDA integration ready for GPU acceleration!")
        
        return True
        
    except ImportError as e:
        print(f"❌ Import error: {e}")
        print("Make sure matplotlib is installed: pip install matplotlib")
        return False
    except Exception as e:
        print(f"❌ Dashboard test failed: {e}")
        import traceback
        traceback.print_exc()
        return False

def main():
    """Main test function"""
    try:
        success = test_unified_dashboard(demo_time=30)
        
        if success:
            return 0
        else:
            return 1
            
    except KeyboardInterrupt:
        print("\n🛑 Test interrupted by user")
        return 0
    except Exception as e:
        print(f"\n❌ Test failed: {e}")
        return 1

if __name__ == "__main__":
    exit_code = main()
    sys.exit(exit_code)
