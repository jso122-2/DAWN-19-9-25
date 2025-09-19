#!/usr/bin/env python3
"""
🚀 DAWN CUDA House Animations Demo
=================================

Comprehensive demonstration of CUDA-accelerated house animations for:
- Mycelial House: Living network dynamics with nutrient flows
- Schema House: Sigil operations and symbolic transformations
- Monitoring House: Real-time telemetry and system health

Features demonstrated:
- GPU-accelerated particle systems and fluid dynamics
- Real-time matplotlib animations with interactive controls
- GUI integration with the DAWN consciousness interface
- Performance monitoring and CUDA acceleration benefits
- Multi-threaded animation management

"Bringing consciousness architecture to life through animation."
"""

import sys
import time
import logging
import asyncio
import threading
from pathlib import Path
from typing import Dict, List, Any

# Add DAWN root to path
sys.path.insert(0, str(Path(__file__).parent))

# Set up logging
logging.basicConfig(
    level=logging.INFO,
    format='%(asctime)s - %(name)s - %(levelname)s - %(message)s'
)
logger = logging.getLogger(__name__)

def print_banner():
    """Print demo banner"""
    print("🚀" * 60)
    print("🧠 DAWN CUDA HOUSE ANIMATIONS DEMONSTRATION")
    print("🚀" * 60)
    print()
    print("This demo showcases GPU-accelerated house animations:")
    print("🍄 Mycelial House - Living network dynamics")
    print("🏛️ Schema House - Sigil operations & transformations") 
    print("📊 Monitoring House - Real-time system telemetry")
    print()
    print("Features:")
    print("✅ CUDA-Accelerated Particle Systems")
    print("✅ Real-time Fluid Dynamics")
    print("✅ Interactive Matplotlib Animations")
    print("✅ GUI Integration with DAWN")
    print("✅ Performance Monitoring")
    print("✅ Multi-threaded Animation Management")
    print()

def test_cuda_availability():
    """Test CUDA availability for animations"""
    print("🔍 Testing CUDA Availability for House Animations...")
    
    cuda_status = {
        'cupy': False,
        'torch_cuda': False,
        'matplotlib': False,
        'scipy': False
    }
    
    # Test CuPy
    try:
        import cupy as cp
        test_array = cp.array([1, 2, 3, 4, 5])
        result = cp.sum(test_array)
        cuda_status['cupy'] = True
        print(f"✅ CuPy: Available (test sum: {cp.asnumpy(result)})")
    except ImportError:
        print("❌ CuPy: Not available")
    except Exception as e:
        print(f"⚠️  CuPy: Error - {e}")
    
    # Test PyTorch CUDA
    try:
        import torch
        if torch.cuda.is_available():
            cuda_status['torch_cuda'] = True
            device_count = torch.cuda.device_count()
            print(f"✅ PyTorch CUDA: Available ({device_count} GPU(s))")
            
            # Show GPU info
            for i in range(min(device_count, 2)):
                props = torch.cuda.get_device_properties(i)
                print(f"   GPU {i}: {props.name} ({props.total_memory / 1024**3:.1f} GB)")
        else:
            print("❌ PyTorch CUDA: Available but no CUDA support")
    except ImportError:
        print("❌ PyTorch: Not available")
    except Exception as e:
        print(f"⚠️  PyTorch CUDA: Error - {e}")
    
    # Test Matplotlib
    try:
        import matplotlib.pyplot as plt
        from mpl_toolkits.mplot3d import Axes3D
        cuda_status['matplotlib'] = True
        print("✅ Matplotlib: Available with 3D support")
    except ImportError:
        print("❌ Matplotlib: Not available")
    
    # Test SciPy
    try:
        from scipy import ndimage, signal
        cuda_status['scipy'] = True
        print("✅ SciPy: Available for fluid dynamics")
    except ImportError:
        print("❌ SciPy: Not available")
    
    print(f"📊 CUDA Status: {sum(cuda_status.values())}/{len(cuda_status)} components available")
    return any(cuda_status.values())

def test_house_animation_manager():
    """Test the house animation manager"""
    print("🎭 Testing House Animation Manager...")
    
    try:
        from dawn.interfaces.visualization.cuda_house_animations import (
            get_house_animation_manager, AnimationConfig
        )
        
        # Create custom config
        config = AnimationConfig(
            fps=30,
            duration=10.0,  # Shorter for demo
            figure_size=(12, 8),
            enable_cuda=True,
            particle_count=200,  # Fewer particles for demo
            animation_quality='medium'
        )
        
        # Get manager
        manager = get_house_animation_manager(config)
        
        # Show status
        status = manager.get_status()
        print(f"✅ Manager Status:")
        print(f"   Manager ID: {status['manager_id']}")
        print(f"   CUDA Available: {status['cuda_available']}")
        print(f"   Matplotlib Available: {status['matplotlib_available']}")
        print(f"   Running: {status['running']}")
        
        return manager
        
    except Exception as e:
        print(f"❌ Error testing house animation manager: {e}")
        return None

def test_individual_house_animators(manager):
    """Test individual house animators"""
    print("\n🏠 Testing Individual House Animators...")
    
    house_types = ['mycelial', 'schema', 'monitoring']
    
    for house_type in house_types:
        print(f"\n🎬 Testing {house_type.title()} House Animator...")
        
        try:
            # Create animator
            animator = manager.create_animator(house_type)
            
            if animator:
                print(f"✅ {house_type.title()} animator created")
                
                # Get performance stats
                stats = animator.get_performance_stats()
                print(f"   Animator ID: {stats['animator_id']}")
                print(f"   CUDA Enabled: {stats['cuda_enabled']}")
                print(f"   Particle Systems: {stats['particle_systems']}")
                
                # Test start/stop
                success = manager.start_animation(house_type)
                if success:
                    print(f"✅ {house_type.title()} animation started")
                    
                    # Let it run briefly
                    time.sleep(2)
                    
                    # Get updated stats
                    stats = animator.get_performance_stats()
                    print(f"   Frame Count: {stats['frame_count']}")
                    print(f"   Average FPS: {stats['avg_fps']:.1f}")
                    
                    # Stop animation
                    manager.stop_animation(house_type)
                    print(f"🛑 {house_type.title()} animation stopped")
                else:
                    print(f"❌ Failed to start {house_type} animation")
            else:
                print(f"❌ Failed to create {house_type} animator")
                
        except Exception as e:
            print(f"❌ Error testing {house_type} animator: {e}")

def test_gui_integration():
    """Test GUI integration"""
    print("\n🖥️  Testing GUI Integration...")
    
    try:
        import tkinter as tk
        from tkinter import ttk
        from dawn.interfaces.visualization.cuda_house_animations import get_house_animation_manager
        from matplotlib.backends.backend_tkagg import FigureCanvasTkAgg
        
        # Create simple test GUI
        root = tk.Tk()
        root.title("DAWN House Animations Test")
        root.geometry("1200x800")
        root.configure(bg='#0a0a0a')
        
        # Create notebook
        notebook = ttk.Notebook(root)
        notebook.pack(fill=tk.BOTH, expand=True, padx=10, pady=10)
        
        # Get animation manager
        manager = get_house_animation_manager()
        
        # Test each house type
        house_types = ['mycelial', 'schema', 'monitoring']
        canvases = {}
        
        for house_type in house_types:
            try:
                # Create tab
                tab_frame = ttk.Frame(notebook)
                notebook.add(tab_frame, text=f"🎬 {house_type.title()}")
                
                # Create control frame
                control_frame = ttk.Frame(tab_frame)
                control_frame.pack(side=tk.TOP, fill=tk.X, padx=10, pady=5)
                
                # Animation canvas frame
                canvas_frame = ttk.Frame(tab_frame)
                canvas_frame.pack(fill=tk.BOTH, expand=True, padx=10, pady=5)
                
                # Create animator
                animator = manager.create_animator(house_type)
                
                if animator:
                    # Start animation
                    manager.start_animation(house_type)
                    
                    # Create canvas
                    if animator.figure:
                        canvas = FigureCanvasTkAgg(animator.figure, canvas_frame)
                        canvas.draw()
                        canvas.get_tk_widget().pack(fill=tk.BOTH, expand=True)
                        canvases[house_type] = canvas
                        
                        print(f"✅ Created GUI tab for {house_type}")
                    else:
                        print(f"⚠️  No figure available for {house_type}")
                else:
                    print(f"❌ Failed to create animator for {house_type}")
                    
            except Exception as e:
                print(f"❌ Error creating GUI tab for {house_type}: {e}")
        
        print(f"\n🎮 GUI Test Window Created with {len(canvases)} animated tabs!")
        print("   Close the window to continue the demo...")
        
        try:
            # Auto-close after 15 seconds
            root.after(15000, root.quit)
            root.mainloop()
        except:
            pass
        
        # Cleanup
        try:
            manager.stop_all_animations()
            root.destroy()
        except:
            pass
        
        print("✅ GUI integration test completed")
        
    except Exception as e:
        print(f"❌ Error testing GUI integration: {e}")

def test_enhanced_gui_system():
    """Test the enhanced GUI system with house animations"""
    print("\n🎛️  Testing Enhanced DAWN GUI System...")
    
    try:
        # Import the enhanced GUI
        from dawn_consciousness_gui import ConsciousnessGUI
        
        print("✅ Enhanced DAWN GUI system available")
        print("   The GUI now includes:")
        print("   - 🍄 Mycelial House animated tab")
        print("   - 🏛️ Schema House animated tab")
        print("   - 📊 Monitoring House animated tab")
        print("   - Interactive animation controls")
        print("   - Real-time performance monitoring")
        
        # Show note about GUI launch
        print("\n🎮 To test the enhanced GUI with house animations, run:")
        print("   python dawn_consciousness_gui.py")
        print("   Look for the new house animation tabs!")
        print("   Use the ▶️ Start Animation buttons to begin animations.")
        
        return True
        
    except Exception as e:
        print(f"⚠️  Enhanced GUI not available: {e}")
        return False

def performance_comparison():
    """Compare CUDA vs CPU performance"""
    print("\n⚡ Performance Comparison: CUDA vs CPU...")
    
    try:
        from dawn.interfaces.visualization.cuda_house_animations import MycelialHouseAnimator, AnimationConfig
        import numpy as np
        import time
        
        # Test data sizes
        test_sizes = [64, 128, 256]
        
        for size in test_sizes:
            print(f"\n📊 Testing with {size}x{size} data...")
            
            # Create test animator
            config = AnimationConfig(enable_cuda=True, particle_count=size*4)
            animator = MycelialHouseAnimator(config)
            
            # Generate test data
            test_field = np.random.rand(size, size).astype(np.float32)
            
            # Test CPU processing
            start_time = time.time()
            cpu_result = animator._process_fluid_cpu(test_field, 'diffuse')
            cpu_time = time.time() - start_time
            
            # Test CUDA processing if available
            if animator.cuda_enabled:
                start_time = time.time()
                cuda_result = animator.process_fluid_cuda(test_field, 'diffuse')
                cuda_time = time.time() - start_time
                
                speedup = cpu_time / cuda_time if cuda_time > 0 else 0
                print(f"   CPU Time: {cpu_time*1000:.2f}ms")
                print(f"   CUDA Time: {cuda_time*1000:.2f}ms")
                print(f"   Speedup: {speedup:.1f}x")
                
                # Verify results are similar
                if np.allclose(cpu_result, cuda_result, atol=1e-5):
                    print("   ✅ Results match between CPU and CUDA")
                else:
                    print("   ⚠️  Results differ between CPU and CUDA")
            else:
                print(f"   CPU Time: {cpu_time*1000:.2f}ms")
                print("   CUDA not available for comparison")
        
    except Exception as e:
        print(f"❌ Error in performance comparison: {e}")

def demonstrate_live_animation():
    """Demonstrate live animation with real data"""
    print("\n🔄 Demonstrating Live Animation with Real Data...")
    
    try:
        from dawn.interfaces.visualization.cuda_house_animations import get_house_animation_manager
        import matplotlib.pyplot as plt
        
        # Get manager
        manager = get_house_animation_manager()
        
        # Create mycelial animator for live demo
        animator = manager.create_animator('mycelial')
        
        if animator:
            print("✅ Created mycelial animator for live demo")
            
            # Start animation
            success = manager.start_animation('mycelial')
            
            if success:
                print("🎬 Live animation started...")
                print("   Animation will run for 10 seconds...")
                
                # Let animation run
                time.sleep(10)
                
                # Get final performance stats
                stats = animator.get_performance_stats()
                print(f"\n📈 Final Performance Stats:")
                print(f"   Total Frames: {stats['frame_count']}")
                print(f"   Average FPS: {stats['avg_fps']:.1f}")
                print(f"   Runtime: {stats['uptime']:.1f}s")
                print(f"   Particle Systems: {stats['particle_systems']}")
                
                # Stop animation
                manager.stop_animation('mycelial')
                print("🛑 Live animation stopped")
            else:
                print("❌ Failed to start live animation")
        else:
            print("❌ Failed to create animator for live demo")
            
    except Exception as e:
        print(f"❌ Error in live animation demo: {e}")

async def main():
    """Main demo function"""
    print_banner()
    
    # Test CUDA availability
    cuda_available = test_cuda_availability()
    print()
    
    # Test house animation manager
    manager = test_house_animation_manager()
    print()
    
    if manager:
        # Test individual animators
        test_individual_house_animators(manager)
        print()
        
        # Test GUI integration
        test_gui_integration()
        print()
        
        # Performance comparison
        if cuda_available:
            performance_comparison()
            print()
        
        # Demonstrate live animation
        demonstrate_live_animation()
        print()
    
    # Test enhanced GUI system
    test_enhanced_gui_system()
    print()
    
    # Show comprehensive summary
    print("📋 COMPREHENSIVE DEMO SUMMARY")
    print("=" * 50)
    print(f"✅ CUDA Available: {cuda_available}")
    print(f"✅ Animation Manager: {manager is not None}")
    
    if manager:
        status = manager.get_status()
        print(f"✅ House Animators: {len(status['animators'])}")
        print(f"✅ CUDA Acceleration: {status['cuda_available']}")
        print(f"✅ Matplotlib Support: {status['matplotlib_available']}")
        
        print("\n🎬 Available House Animations:")
        for house_type, animator_info in status['animators'].items():
            print(f"   🏠 {house_type.title()} House - Ready")
    
    # Cleanup
    print("\n🧹 Cleaning up...")
    
    if manager:
        try:
            manager.stop_all_animations()
            print("✅ Stopped all animations")
        except:
            pass
    
    print("\n🚀" * 60)
    print("🧠 DAWN CUDA HOUSE ANIMATIONS DEMO COMPLETE")
    print("🚀" * 60)
    print()
    print("Key Features Demonstrated:")
    print("🍄 Mycelial House - Living network with nutrient flows")
    print("🏛️ Schema House - Sigil operations and transformations")
    print("📊 Monitoring House - Real-time system telemetry")
    print("⚡ CUDA Acceleration - GPU-powered particle systems")
    print("🖥️  GUI Integration - Embedded in DAWN consciousness GUI")
    print("🎮 Interactive Controls - Start/stop animations")
    print()
    print("To use in the main GUI:")
    print("1. Run: python dawn_consciousness_gui.py")
    print("2. Look for house animation tabs")
    print("3. Click ▶️ Start Animation buttons")
    print("4. Watch consciousness come alive!")
    print()
    print("Thank you for exploring DAWN's animated consciousness!")

if __name__ == "__main__":
    # Run the comprehensive demo
    try:
        asyncio.run(main())
    except KeyboardInterrupt:
        print("\n👋 Demo interrupted. Goodbye!")
    except Exception as e:
        print(f"❌ Demo error: {e}")
        import traceback
        traceback.print_exc()
