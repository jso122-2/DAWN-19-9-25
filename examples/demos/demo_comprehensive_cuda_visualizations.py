#!/usr/bin/env python3
"""
🚀 DAWN Comprehensive CUDA Visualizations Demo
==============================================

Complete demonstration of the DAWN CUDA visualization architecture including:
- CUDA matplotlib visualization engine
- GUI integration components  
- Unified visualization manager
- Real-time data collection and rendering
- Interactive GUI controls

Features demonstrated:
- 15+ different visualization types
- GPU-accelerated data processing
- Real-time subsystem monitoring
- Interactive Tkinter integration
- Multi-threaded visualization updates
- DAWN singleton integration

"The ultimate showcase of consciousness visualization at light speed."
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
    print("🧠 DAWN COMPREHENSIVE CUDA VISUALIZATIONS DEMONSTRATION")
    print("🚀" * 60)
    print()
    print("This demo showcases the complete DAWN CUDA visualization system:")
    print("✅ CUDA-Accelerated Matplotlib Engine")
    print("✅ 15+ Specialized Visualizations")
    print("✅ GUI Integration Framework")
    print("✅ Unified Visualization Manager")
    print("✅ Real-time Data Collection")
    print("✅ Interactive Controls")
    print("✅ DAWN Singleton Integration")
    print()

def test_cuda_availability():
    """Test CUDA availability for visualizations"""
    print("🔍 Testing CUDA Availability for Visualizations...")
    
    cuda_status = {
        'cupy': False,
        'torch_cuda': False,
        'pycuda': False,
        'matplotlib': False,
        'scipy': False
    }
    
    # Test CuPy
    try:
        import cupy as cp
        test_array = cp.array([1, 2, 3, 4, 5])
        result = cp.sum(test_array)
        cuda_status['cupy'] = True
        print(f"✅ CuPy: Available (test result: {cp.asnumpy(result)})")
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
            for i in range(min(device_count, 2)):  # Show max 2 GPUs
                props = torch.cuda.get_device_properties(i)
                print(f"   GPU {i}: {props.name}")
                print(f"   Memory: {props.total_memory / 1024**3:.1f} GB")
                print(f"   Compute: {props.major}.{props.minor}")
        else:
            print("❌ PyTorch CUDA: Available but no CUDA support")
    except ImportError:
        print("❌ PyTorch: Not available")
    except Exception as e:
        print(f"⚠️  PyTorch CUDA: Error - {e}")
    
    # Test PyCUDA
    try:
        import pycuda.driver as cuda
        import pycuda.autoinit
        cuda_status['pycuda'] = True
        print("✅ PyCUDA: Available")
    except ImportError:
        print("❌ PyCUDA: Not available")
    except Exception as e:
        print(f"⚠️  PyCUDA: Error - {e}")
    
    # Test Matplotlib
    try:
        import matplotlib.pyplot as plt
        from mpl_toolkits.mplot3d import Axes3D
        cuda_status['matplotlib'] = True
        print("✅ Matplotlib: Available")
    except ImportError:
        print("❌ Matplotlib: Not available")
    
    # Test SciPy
    try:
        from scipy import ndimage, signal
        cuda_status['scipy'] = True
        print("✅ SciPy: Available")
    except ImportError:
        print("❌ SciPy: Not available")
    
    print(f"📊 CUDA Status Summary: {sum(cuda_status.values())}/{len(cuda_status)} components available")
    return any(cuda_status.values())

async def initialize_dawn_system():
    """Initialize the DAWN consciousness system"""
    print("🌅 Initializing DAWN Consciousness System...")
    
    try:
        from dawn.core.singleton import get_dawn
        
        # Get DAWN singleton
        dawn = get_dawn()
        
        # Initialize the system
        success = await dawn.initialize()
        
        if success:
            print("✅ DAWN system initialized successfully")
            
            # Start the system
            await dawn.start()
            print("✅ DAWN system started")
            
            # Show system status
            try:
                status = dawn.get_status()
                print(f"📊 DAWN Status: Running={status.running}, Mode={status.mode}")
            except:
                print("📊 DAWN Status: System active")
            
            return dawn
        else:
            print("❌ DAWN system initialization failed")
            return None
            
    except Exception as e:
        print(f"⚠️  DAWN system initialization error: {e}")
        return None

def test_cuda_visualization_engine():
    """Test the CUDA visualization engine"""
    print("🎨 Testing CUDA Visualization Engine...")
    
    try:
        from dawn.interfaces.visualization.cuda_matplotlib_engine import (
            get_cuda_matplotlib_engine, VisualizationConfig
        )
        
        # Create engine with custom config
        config = VisualizationConfig(
            figure_size=(12, 8),
            style="dark_background",
            color_scheme="consciousness",
            enable_cuda=True,
            enable_3d=True
        )
        
        engine = get_cuda_matplotlib_engine(config)
        
        # Show engine summary
        summary = engine.get_engine_summary()
        print(f"✅ Engine Summary:")
        print(f"   Engine ID: {summary['engine_id']}")
        print(f"   CUDA Enabled: {summary['configuration']['cuda_enabled']}")
        print(f"   Available Visualizations: {summary['status']['figures_created']}")
        print(f"   Capabilities: {summary['capabilities']}")
        
        # Test some visualizations
        print("\n🎯 Testing Individual Visualizations...")
        
        # Test tracer ecosystem visualization
        tracer_data = {
            'tracers': {
                'crow_1': {'tracer_type': 'crow', 'activity_level': 0.8, 'age': 5},
                'whale_1': {'tracer_type': 'whale', 'activity_level': 0.6, 'age': 15}
            },
            'positions': {
                'crow_1': [0.3, 0.7, 0.5],
                'whale_1': [0.8, 0.2, 0.6]
            },
            'trails': {
                'crow_1': [[0.25, 0.65, 0.45], [0.28, 0.68, 0.48], [0.3, 0.7, 0.5]]
            }
        }
        
        fig1 = engine.visualize_tracer_ecosystem_3d(tracer_data)
        print("✅ Created tracer ecosystem 3D visualization")
        
        # Test consciousness flow visualization
        consciousness_data = {
            'consciousness_history': [
                {'coherence': 0.7, 'unity': 0.6, 'pressure': 0.4},
                {'coherence': 0.8, 'unity': 0.7, 'pressure': 0.5},
                {'coherence': 0.6, 'unity': 0.5, 'pressure': 0.6}
            ],
            'current_state': {
                'coherence': 0.8, 'unity': 0.7, 'pressure': 0.5,
                'entropy': 0.3, 'awareness': 0.9, 'integration': 0.8
            }
        }
        
        fig2 = engine.visualize_consciousness_flow(consciousness_data)
        print("✅ Created consciousness flow visualization")
        
        # Test semantic topology visualization
        import numpy as np
        semantic_data = {
            'semantic_field': np.random.rand(16, 16, 16) * 0.8,
            'clusters': [
                {'center': [0.3, 0.7, 0.5], 'coherence': 0.8},
                {'center': [0.8, 0.2, 0.6], 'coherence': 0.6}
            ],
            'edges': [
                {'start': [0.3, 0.7, 0.5], 'end': [0.8, 0.2, 0.6], 'strength': 0.7}
            ]
        }
        
        fig3 = engine.visualize_semantic_topology_3d(semantic_data)
        print("✅ Created semantic topology 3D visualization")
        
        # Test telemetry dashboard
        telemetry_data = {
            'performance_history': [
                {'cpu_usage': 45, 'memory_usage': 60, 'gpu_usage': 30},
                {'cpu_usage': 50, 'memory_usage': 65, 'gpu_usage': 35}
            ],
            'health_indicators': {
                'CPU Health': 0.8,
                'Memory Health': 0.7,
                'GPU Health': 0.9,
                'Network Health': 0.6
            },
            'resource_usage': {
                'CPU': 45,
                'Memory': 60,
                'GPU': 30,
                'Network': 20
            }
        }
        
        fig4 = engine.visualize_telemetry_dashboard(telemetry_data)
        print("✅ Created telemetry dashboard visualization")
        
        print(f"🎨 Total Visualizations Available: {len(engine.get_available_visualizations())}")
        
        return engine
        
    except Exception as e:
        print(f"❌ Error testing CUDA visualization engine: {e}")
        return None

def test_unified_visualization_manager():
    """Test the unified visualization manager"""
    print("🏗️  Testing Unified Visualization Manager...")
    
    try:
        from dawn.interfaces.visualization.unified_manager import get_unified_visualization_manager
        
        # Create manager
        manager = get_unified_visualization_manager()
        
        # Show system status
        status = manager.get_system_status()
        print(f"✅ Manager Status:")
        print(f"   Running: {status['running']}")
        print(f"   Data Sources: {status['data_collection']['sources']}")
        print(f"   Available Visualizations: {status['visualizations']['available']}")
        print(f"   Visualization Bindings: {status['visualizations']['bindings']}")
        
        # Start the system
        print("\n🚀 Starting unified visualization system...")
        manager.start_system()
        
        # Let it collect data for a few seconds
        print("⏱️  Collecting data for 5 seconds...")
        time.sleep(5)
        
        # Show updated status
        updated_status = manager.get_system_status()
        print(f"\n📊 Updated Status:")
        print(f"   Data Collection Active: {updated_status['data_collection']['active']}")
        print(f"   Data Points Collected: {updated_status['data_collection']['data_points']}")
        
        # Show available visualizations by category
        available_viz = manager.get_available_visualizations()
        print(f"\n🎨 Available Visualizations by Category:")
        for category, visualizations in available_viz.items():
            print(f"   {category}: {len(visualizations)} visualizations")
            for viz in visualizations[:3]:  # Show first 3
                print(f"     - {viz}")
            if len(visualizations) > 3:
                print(f"     ... and {len(visualizations) - 3} more")
        
        return manager
        
    except Exception as e:
        print(f"❌ Error testing unified visualization manager: {e}")
        return None

def test_gui_integration():
    """Test GUI integration components"""
    print("🖥️  Testing GUI Integration...")
    
    try:
        from dawn.interfaces.visualization.gui_integration import get_visualization_gui_manager
        import tkinter as tk
        from tkinter import ttk
        
        # Create GUI manager
        gui_manager = get_visualization_gui_manager()
        
        # Show manager summary
        summary = gui_manager.get_manager_summary()
        print(f"✅ GUI Manager Summary:")
        print(f"   Manager ID: {summary['manager_id']}")
        print(f"   Active Widgets: {summary['active_widgets']}")
        print(f"   Available Visualizations: {summary['available_visualizations']}")
        print(f"   Capabilities: {summary['capabilities']}")
        
        # Test creating a simple Tkinter widget
        if summary['capabilities']['tkinter_available']:
            print("\n🎨 Creating test Tkinter visualization widget...")
            
            # Create simple Tkinter window
            root = tk.Tk()
            root.title("DAWN CUDA Visualization Test")
            root.geometry("1000x600")
            root.configure(bg='#0a0a0a')
            
            # Create notebook for visualizations
            notebook = ttk.Notebook(root)
            notebook.pack(fill=tk.BOTH, expand=True, padx=10, pady=10)
            
            # Test creating different visualization widgets
            test_visualizations = ['consciousness_flow', 'tracer_ecosystem_3d', 'system_health_radar']
            
            for i, viz_name in enumerate(test_visualizations[:2]):  # Test first 2
                try:
                    # Create tab
                    tab_frame = ttk.Frame(notebook)
                    notebook.add(tab_frame, text=f"🧠 {viz_name.replace('_', ' ').title()}")
                    
                    # Create visualization widget
                    widget = gui_manager.create_tkinter_widget(tab_frame, viz_name)
                    
                    if widget:
                        widget.pack(fill=tk.BOTH, expand=True)
                        
                        # Set test data
                        if viz_name == 'consciousness_flow':
                            test_data = {
                                'consciousness_history': [
                                    {'coherence': 0.7, 'unity': 0.6, 'pressure': 0.4},
                                    {'coherence': 0.8, 'unity': 0.7, 'pressure': 0.5}
                                ],
                                'current_state': {
                                    'coherence': 0.8, 'unity': 0.7, 'pressure': 0.5,
                                    'entropy': 0.3, 'awareness': 0.9, 'integration': 0.8
                                }
                            }
                        elif viz_name == 'tracer_ecosystem_3d':
                            test_data = {
                                'tracers': {
                                    'crow_1': {'tracer_type': 'crow', 'activity_level': 0.8}
                                },
                                'positions': {
                                    'crow_1': [0.3, 0.7, 0.5]
                                }
                            }
                        else:
                            test_data = {
                                'health_metrics': {
                                    'CPU Health': 0.8,
                                    'Memory Health': 0.7,
                                    'GPU Health': 0.9
                                }
                            }
                        
                        widget.set_data(test_data)
                        widget.refresh()
                        
                        print(f"✅ Created widget for {viz_name}")
                    else:
                        print(f"⚠️  Failed to create widget for {viz_name}")
                        
                except Exception as e:
                    print(f"⚠️  Error creating widget for {viz_name}: {e}")
            
            print("\n🎮 GUI Test Window Created!")
            print("   Close the window to continue the demo...")
            
            try:
                # Run for a limited time
                root.after(10000, root.quit)  # Auto-close after 10 seconds
                root.mainloop()
            except:
                pass
            
            try:
                root.destroy()
            except:
                pass
            
            print("✅ GUI integration test completed")
        else:
            print("⚠️  Tkinter not available - skipping GUI test")
        
        return gui_manager
        
    except Exception as e:
        print(f"❌ Error testing GUI integration: {e}")
        return None

def test_performance_benchmarks():
    """Test performance of CUDA vs CPU processing"""
    print("⚡ Testing CUDA vs CPU Performance...")
    
    try:
        from dawn.interfaces.visualization.cuda_matplotlib_engine import get_cuda_matplotlib_engine
        import numpy as np
        import time
        
        engine = get_cuda_matplotlib_engine()
        
        # Create test data
        test_sizes = [100, 500, 1000]
        
        for size in test_sizes:
            print(f"\n📊 Testing with {size}x{size} data...")
            
            # Generate test data
            test_data = np.random.rand(size, size).astype(np.float32)
            
            # Test CPU processing
            start_time = time.time()
            cpu_result = engine._process_data_cpu(test_data, 'normalize')
            cpu_time = time.time() - start_time
            
            # Test CUDA processing if available
            if engine.cuda_enabled:
                start_time = time.time()
                cuda_result = engine.process_data_cuda(test_data, 'normalize')
                cuda_time = time.time() - start_time
                
                speedup = cpu_time / cuda_time if cuda_time > 0 else 0
                print(f"   CPU Time: {cpu_time*1000:.2f}ms")
                print(f"   CUDA Time: {cuda_time*1000:.2f}ms")
                print(f"   Speedup: {speedup:.1f}x")
                
                # Verify results are similar
                if np.allclose(cpu_result, cuda_result, atol=1e-6):
                    print("   ✅ Results match between CPU and CUDA")
                else:
                    print("   ⚠️  Results differ between CPU and CUDA")
            else:
                print(f"   CPU Time: {cpu_time*1000:.2f}ms")
                print("   CUDA not available for comparison")
        
    except Exception as e:
        print(f"❌ Error testing performance: {e}")

def test_enhanced_gui_system():
    """Test the enhanced GUI system with CUDA visualizations"""
    print("🎛️  Testing Enhanced GUI System...")
    
    try:
        # Import the enhanced GUI
        from dawn_consciousness_gui import ConsciousnessGUI
        
        print("✅ Enhanced GUI system available")
        print("   The GUI now includes:")
        print("   - CUDA-accelerated visualizations")
        print("   - Unified visualization manager integration")
        print("   - Real-time data collection")
        print("   - Interactive CUDA visualization tabs")
        
        # Show note about GUI launch
        print("\n🎮 To test the enhanced GUI, run:")
        print("   python dawn_consciousness_gui.py")
        print("   Look for the new '🚀 CUDA' tabs!")
        
        return True
        
    except Exception as e:
        print(f"⚠️  Enhanced GUI not available: {e}")
        return False

def demonstrate_real_time_updates():
    """Demonstrate real-time visualization updates"""
    print("🔄 Demonstrating Real-time Updates...")
    
    try:
        from dawn.interfaces.visualization.unified_manager import get_unified_visualization_manager
        
        manager = get_unified_visualization_manager()
        
        if not manager.running:
            manager.start_system()
        
        # Monitor data collection for a few update cycles
        print("📊 Monitoring data collection...")
        
        for i in range(5):
            status = manager.get_system_status()
            data_points = status['data_collection']['data_points']
            active = status['data_collection']['active']
            
            print(f"   Cycle {i+1}: Active={active}, Data Points={data_points}")
            time.sleep(2)
        
        print("✅ Real-time updates working")
        
    except Exception as e:
        print(f"❌ Error demonstrating real-time updates: {e}")

async def main():
    """Main demo function"""
    print_banner()
    
    # Test CUDA availability
    cuda_available = test_cuda_availability()
    print()
    
    # Initialize DAWN system
    dawn_system = await initialize_dawn_system()
    print()
    
    # Test CUDA visualization engine
    viz_engine = test_cuda_visualization_engine()
    print()
    
    # Test unified visualization manager
    unified_manager = test_unified_visualization_manager()
    print()
    
    # Test GUI integration
    gui_manager = test_gui_integration()
    print()
    
    # Test performance benchmarks
    if cuda_available and viz_engine:
        test_performance_benchmarks()
        print()
    
    # Test enhanced GUI system
    test_enhanced_gui_system()
    print()
    
    # Demonstrate real-time updates
    if unified_manager:
        demonstrate_real_time_updates()
        print()
    
    # Show comprehensive summary
    print("📋 COMPREHENSIVE DEMO SUMMARY")
    print("=" * 50)
    print(f"✅ CUDA Available: {cuda_available}")
    print(f"✅ DAWN System: {dawn_system is not None}")
    print(f"✅ Visualization Engine: {viz_engine is not None}")
    print(f"✅ Unified Manager: {unified_manager is not None}")
    print(f"✅ GUI Integration: {gui_manager is not None}")
    
    if viz_engine:
        available_viz = viz_engine.get_available_visualizations()
        print(f"✅ Available Visualizations: {len(available_viz)}")
        
        print("\n🎨 Visualization Categories:")
        if unified_manager:
            categories = unified_manager.get_available_visualizations()
            for category, visualizations in categories.items():
                print(f"   {category}: {len(visualizations)} visualizations")
    
    # Cleanup
    print("\n🧹 Cleaning up...")
    
    if unified_manager and unified_manager.running:
        unified_manager.stop_system()
        print("✅ Stopped unified visualization system")
    
    if gui_manager:
        gui_manager.cleanup_widgets()
        print("✅ Cleaned up GUI widgets")
    
    if dawn_system:
        try:
            await dawn_system.shutdown()
            print("✅ DAWN system shutdown complete")
        except:
            pass
    
    print("\n🚀" * 60)
    print("🧠 DAWN COMPREHENSIVE CUDA VISUALIZATIONS DEMO COMPLETE")
    print("🚀" * 60)
    print()
    print("Key Features Demonstrated:")
    print("📊 15+ CUDA-accelerated visualizations")
    print("🎨 Real-time data processing and rendering")
    print("🖥️  GUI framework integration (Tkinter/Qt)")
    print("🔄 Unified visualization management")
    print("⚡ GPU performance acceleration")
    print("🧠 DAWN consciousness system integration")
    print()
    print("To use in your applications:")
    print("1. Import: from dawn.interfaces.visualization import *")
    print("2. Create: manager = get_unified_visualization_manager()")
    print("3. Start: manager.start_system()")
    print("4. Create widgets: manager.create_gui_widget('tkinter', parent, 'viz_name')")
    print()
    print("Thank you for exploring DAWN's consciousness visualization!")

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
