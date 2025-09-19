#!/usr/bin/env python3
"""
DAWN Visual System - Quick Test
==============================

Simple, fast test runner for DAWN matplotlib/seaborn visualizations.
Perfect for quick validation after changes.

Usage:
    python quick_visual_test.py
"""

import torch
import numpy as np
import matplotlib.pyplot as plt
import time
import sys
from pathlib import Path

# Ensure we can import DAWN modules
sys.path.insert(0, str(Path(__file__).parent))

def test_imports():
    """Test that all visual modules can be imported"""
    print("🔍 Testing imports...")
    
    global device  # Make device available globally
    
    try:
        from dawn.subsystems.visual.dawn_visual_base import DAWNVisualBase, device
        print(f"   ✅ DAWNVisualBase imported - Device: {device}")
        
        from dawn.subsystems.visual.advanced_visual_consciousness import create_advanced_visual_consciousness
        print("   ✅ Advanced Visual Consciousness")
        
        from dawn.subsystems.visual.tick_pulse_matplotlib import create_tick_pulse_visualizer
        print("   ✅ Tick Pulse Visualizer")
        
        from dawn.subsystems.visual.entropy_flow_matplotlib import create_entropy_flow_visualizer
        print("   ✅ Entropy Flow Visualizer")
        
        from dawn.subsystems.visual.heat_monitor_matplotlib import create_heat_monitor_visualizer
        print("   ✅ Heat Monitor Visualizer")
        
        from dawn.subsystems.visual.seaborn_consciousness_analytics import create_consciousness_analytics
        print("   ✅ Consciousness Analytics")
        
        from dawn.subsystems.visual.seaborn_mood_tracker import create_mood_tracker
        print("   ✅ Mood Tracker")
        
        return True
        
    except ImportError as e:
        print(f"   ❌ Import failed: {e}")
        return False

def quick_tensor_test():
    """Quick test of device-agnostic tensor operations"""
    print("\n🧮 Testing tensor operations...")
    
    try:
        # Create test data
        data = torch.randn(10, 4).to(device)
        print(f"   ✅ Created tensor: {data.shape} on {device}")
        
        # Test operations
        processed = torch.softmax(data, dim=-1)
        mean_val = processed.mean()
        print(f"   ✅ Tensor operations work - Mean: {mean_val:.3f}")
        
        # Test NaN handling
        data_with_nan = data.clone()
        data_with_nan[0, 0] = float('nan')
        has_nan = torch.isnan(data_with_nan).any()
        print(f"   ✅ NaN detection works: {has_nan}")
        
        return True
        
    except Exception as e:
        print(f"   ❌ Tensor test failed: {e}")
        return False

def quick_visualization_test():
    """Quick test of core visualizers"""
    print("\n🎨 Testing core visualizers...")
    
    output_dir = Path("./quick_test_output")
    output_dir.mkdir(exist_ok=True)
    
    tests_passed = 0
    total_tests = 0
    
    # Test 1: Advanced Visual Consciousness
    total_tests += 1
    try:
        print("   🔬 Testing Advanced Visual Consciousness...")
        from dawn.subsystems.visual.advanced_visual_consciousness import create_advanced_visual_consciousness
        
        visualizer = create_advanced_visual_consciousness()
        
        consciousness_state = {
            'schema': 0.7,
            'coherence': 0.8,
            'utility': 0.6,
            'processing_intensity': 0.5
        }
        
        artwork = visualizer.create_consciousness_artwork(consciousness_state, style='CONSCIOUSNESS_FLOW')
        print(f"      ✅ Created consciousness artwork")
        tests_passed += 1
        
    except Exception as e:
        print(f"      ❌ Failed: {e}")
    
    # Test 2: Tick Pulse Visualizer
    total_tests += 1
    try:
        print("   🔬 Testing Tick Pulse Visualizer...")
        from dawn.subsystems.visual.tick_pulse_matplotlib import create_tick_pulse_visualizer
        
        visualizer = create_tick_pulse_visualizer()
        
        # Add some pulse data using tensor
        current_time = time.time()
        pulse_tensor = torch.tensor([[
            current_time + i,
            0.5 + 0.3 * np.sin(i),
            1.0,
            0.8
        ] for i in range(5)], dtype=torch.float32).to(device)
        
        visualizer._update_tick_data_from_tensor(pulse_tensor)
        
        # Render frame
        test_data = torch.rand(5, 4).to(device)
        fig = visualizer.render_frame(test_data)
        
        output_path = output_dir / "tick_pulse_test.png"
        visualizer.save_consciousness_frame(str(output_path))
        plt.close(fig)
        
        print(f"      ✅ Tick pulse visualization saved to {output_path}")
        tests_passed += 1
        
    except Exception as e:
        print(f"      ❌ Failed: {e}")
    
    # Test 3: Heat Monitor
    total_tests += 1
    try:
        print("   🔬 Testing Heat Monitor...")
        from dawn.subsystems.visual.heat_monitor_matplotlib import create_heat_monitor_visualizer
        
        visualizer = create_heat_monitor_visualizer()
        
        # Add heat data using tensor
        current_time = time.time()
        heat_tensor = torch.tensor([[
            0.3 + 0.4 * np.sin(i * 0.5),
            0.6,  # load estimate
            0.1,  # complexity
            current_time + i
        ] for i in range(5)], dtype=torch.float32).to(device)
        
        visualizer._update_heat_data_from_tensor(heat_tensor)
        
        # Render frame
        test_data = torch.rand(1, 4).to(device)
        fig = visualizer.render_frame(test_data)
        
        output_path = output_dir / "heat_monitor_test.png"
        visualizer.save_consciousness_frame(str(output_path))
        plt.close(fig)
        
        print(f"      ✅ Heat monitor visualization saved to {output_path}")
        tests_passed += 1
        
    except Exception as e:
        print(f"      ❌ Failed: {e}")
    
    # Test 4: Simple matplotlib test
    total_tests += 1
    try:
        print("   🔬 Testing Simple Matplotlib...")
        
        # Create a simple plot using matplotlib with consciousness colors
        from dawn.subsystems.visual.dawn_visual_base import DAWNVisualBase, DAWNVisualConfig
        
        class SimpleTest(DAWNVisualBase):
            def __init__(self):
                super().__init__(DAWNVisualConfig())
            
            def render_frame(self, data):
                return self.create_figure((1, 1))
            
            def update_visualization(self, data):
                pass
                
        test_vis = SimpleTest()
        fig = test_vis.create_figure((1, 1))
        ax = test_vis.axes[0] if hasattr(test_vis, 'axes') and test_vis.axes else fig.add_subplot(111)
        
        # Simple consciousness-themed plot
        x = np.linspace(0, 10, 100)
        y = np.sin(x) * np.exp(-x/10)
        ax.plot(x, y, color=test_vis.consciousness_colors['awareness'], linewidth=2)
        ax.set_title('Consciousness Wave', color='white')
        ax.set_facecolor('#0a0a0a')
        
        output_path = output_dir / "simple_matplotlib_test.png"
        test_vis.save_consciousness_frame(str(output_path))
        plt.close(fig)
        
        print(f"      ✅ Simple matplotlib test saved to {output_path}")
        tests_passed += 1
        
    except Exception as e:
        print(f"      ❌ Failed: {e}")
    
    print(f"\n   📊 Visualization tests: {tests_passed}/{total_tests} passed")
    return tests_passed == total_tests

def main():
    """Run quick visual system test"""
    print("🚀 DAWN Visual System - Quick Test")
    print("=" * 40)
    
    start_time = time.time()
    
    # Run tests
    import_success = test_imports()
    tensor_success = quick_tensor_test() if import_success else False
    visual_success = quick_visualization_test() if import_success and tensor_success else False
    
    duration = time.time() - start_time
    
    # Summary
    print("\n" + "=" * 40)
    print("📋 Quick Test Summary")
    print("=" * 40)
    
    print(f"✅ Imports: {'PASS' if import_success else 'FAIL'}")
    print(f"✅ Tensors: {'PASS' if tensor_success else 'FAIL'}")
    print(f"✅ Visuals: {'PASS' if visual_success else 'FAIL'}")
    
    print(f"\n⏱️ Test Duration: {duration:.1f}s")
    try:
        print(f"🖥️ Device: {device if import_success else 'Unknown'}")
    except NameError:
        print(f"🖥️ Device: Unknown")
    print(f"🔥 PyTorch: {torch.__version__}")
    
    if import_success and tensor_success and visual_success:
        print("\n🎉 ALL TESTS PASSED! Visual system is working correctly!")
        print("📁 Test outputs saved to: ./quick_test_output/")
        print("\n💡 To run comprehensive tests, use:")
        print("   python run_all_dawn_visuals.py --mode full")
    else:
        print("\n⚠️ Some tests failed. Check the errors above.")
        print("💡 Make sure you're in the DAWN root directory and all dependencies are installed.")
        
        if not import_success:
            print("\n🔧 Import issues - try:")
            print("   pip install torch matplotlib seaborn pandas scikit-learn")
    
    return import_success and tensor_success and visual_success

if __name__ == "__main__":
    success = main()
    sys.exit(0 if success else 1)
