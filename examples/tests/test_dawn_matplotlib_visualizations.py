#!/usr/bin/env python3
"""
DAWN Matplotlib/Seaborn Visualization Test Suite
===============================================

Comprehensive test script to validate all new matplotlib/seaborn-based DAWN
consciousness visualizations. Tests the unified visual base and all converted
visualization modules for functionality, performance, and integration.

PyTorch Best Practices:
- Device-agnostic tensor operations
- Memory-efficient gradient handling  
- Type hints for all functions
- Proper error handling with NaN checks
"""

import torch
import numpy as np
import matplotlib.pyplot as plt
import seaborn as sns
import time
import sys
import logging
from pathlib import Path
from typing import Dict, List, Optional, Any
import traceback

# Add DAWN visual modules to path
sys.path.insert(0, str(Path(__file__).parent / "dawn" / "subsystems" / "visual"))

# Import DAWN visual modules
try:
    from dawn_visual_base import (
        DAWNVisualBase, 
        DAWNVisualConfig, 
        ConsciousnessColorPalette,
        device,
        ExampleConsciousnessVisualizer
    )
    from advanced_visual_consciousness import (
        AdvancedVisualConsciousness,
        ArtisticRenderingConfig,
        create_advanced_visual_consciousness
    )
    from tick_pulse_matplotlib import (
        TickPulseVisualizer,
        TickPulseConfig,
        create_tick_pulse_visualizer
    )
    from entropy_flow_matplotlib import (
        EntropyFlowVisualizer,
        EntropyFlowConfig,
        create_entropy_flow_visualizer
    )
    from heat_monitor_matplotlib import (
        HeatMonitorVisualizer,
        HeatMonitorConfig,
        create_heat_monitor_visualizer
    )
    VISUALIZERS_AVAILABLE = True
except ImportError as e:
    print(f"⚠️ Warning: Could not import all visualizers: {e}")
    VISUALIZERS_AVAILABLE = False

# Setup logging
logging.basicConfig(level=logging.INFO, format='%(asctime)s - %(levelname)s - %(message)s')
logger = logging.getLogger(__name__)

class DAWNVisualizationTester:
    """Comprehensive test suite for DAWN matplotlib/seaborn visualizations"""
    
    def __init__(self):
        """Initialize the test suite"""
        self.test_results: Dict[str, Dict[str, Any]] = {}
        self.test_data_dir = Path("./test_visual_output")
        self.test_data_dir.mkdir(exist_ok=True)
        
        # Test configurations
        self.visual_config = DAWNVisualConfig(
            figure_size=(10, 8),
            animation_fps=30,
            enable_real_time=False,  # Disable for testing
            save_frames=True,
            output_directory=str(self.test_data_dir),
            memory_efficient=True
        )
        
        logger.info(f"🧪 DAWN Visualization Test Suite initialized")
        logger.info(f"   Device: {device}")
        logger.info(f"   Output directory: {self.test_data_dir}")
        logger.info(f"   PyTorch version: {torch.__version__}")
    
    def run_all_tests(self) -> Dict[str, Dict[str, Any]]:
        """Run comprehensive test suite"""
        print("🚀 Starting DAWN Visualization Test Suite")
        print("=" * 60)
        
        if not VISUALIZERS_AVAILABLE:
            print("❌ Cannot run tests - visualizers not available")
            return {}
        
        # Test sequence
        test_methods = [
            ("Base Visual System", self.test_dawn_visual_base),
            ("Advanced Visual Consciousness", self.test_advanced_visual_consciousness),
            ("Tick Pulse Visualizer", self.test_tick_pulse_visualizer),
            ("Entropy Flow Visualizer", self.test_entropy_flow_visualizer),
            ("Heat Monitor Visualizer", self.test_heat_monitor_visualizer),
            ("Performance Benchmarks", self.test_performance_benchmarks),
            ("Memory Efficiency", self.test_memory_efficiency),
            ("Device Compatibility", self.test_device_compatibility)
        ]
        
        for test_name, test_method in test_methods:
            print(f"\n🔬 Testing: {test_name}")
            print("-" * 40)
            
            try:
                start_time = time.time()
                result = test_method()
                end_time = time.time()
                
                self.test_results[test_name] = {
                    "status": "PASSED" if result else "FAILED",
                    "duration_ms": (end_time - start_time) * 1000,
                    "details": result if isinstance(result, dict) else {}
                }
                
                status_emoji = "✅" if result else "❌"
                print(f"{status_emoji} {test_name}: {self.test_results[test_name]['status']} ({self.test_results[test_name]['duration_ms']:.1f}ms)")
                
            except Exception as e:
                self.test_results[test_name] = {
                    "status": "ERROR",
                    "error": str(e),
                    "traceback": traceback.format_exc()
                }
                print(f"❌ {test_name}: ERROR - {e}")
        
        # Generate summary
        self.print_test_summary()
        
        return self.test_results
    
    def test_dawn_visual_base(self) -> bool:
        """Test the base visual system"""
        try:
            # Test base class instantiation
            visualizer = ExampleConsciousnessVisualizer(self.visual_config)
            
            # Test tensor processing
            test_data = torch.randn(50, 3).to(device)
            processed = visualizer.process_consciousness_tensor(test_data)
            
            assert processed.shape == test_data.shape, f"Shape mismatch: {processed.shape} vs {test_data.shape}"
            assert not torch.isnan(processed).any(), "Processed tensor contains NaN values"
            
            # Test figure creation
            fig = visualizer.create_figure((2, 2))
            assert fig is not None, "Figure creation failed"
            assert len(visualizer.axes) == 4, f"Expected 4 axes, got {len(visualizer.axes)}"
            
            # Test consciousness plotting methods
            visualizer.plot_consciousness_trajectory(test_data, visualizer.axes[0])
            visualizer.plot_consciousness_heatmap(test_data[:10, :10], visualizer.axes[1])
            visualizer.plot_consciousness_distribution(test_data.flatten(), visualizer.axes[2])
            
            # Test frame rendering
            rendered_fig = visualizer.render_frame(test_data)
            assert rendered_fig is not None, "Frame rendering failed"
            
            # Test frame saving
            saved_path = visualizer.save_consciousness_frame("test_base_visual.png")
            assert Path(saved_path).exists(), f"Frame not saved to {saved_path}"
            
            # Test performance metrics
            metrics = visualizer.get_performance_stats()
            assert isinstance(metrics, dict), "Performance stats should be dict"
            assert 'fps' in metrics, "FPS should be in performance stats"
            
            plt.close('all')  # Cleanup
            
            print("  ✓ Base class instantiation")
            print("  ✓ Tensor processing")
            print("  ✓ Figure creation")
            print("  ✓ Consciousness plotting")
            print("  ✓ Frame rendering")
            print("  ✓ Frame saving")
            print("  ✓ Performance metrics")
            
            return True
            
        except Exception as e:
            print(f"  ❌ Base visual test failed: {e}")
            return False
    
    def test_advanced_visual_consciousness(self) -> bool:
        """Test advanced visual consciousness system"""
        try:
            # Create configurations
            artistic_config = ArtisticRenderingConfig(
                canvas_size=(400, 300),
                target_fps=30,
                consciousness_particle_density=100
            )
            
            # Test visualizer creation
            visualizer = create_advanced_visual_consciousness(
                artistic_config=artistic_config,
                visual_config=self.visual_config
            )
            
            assert visualizer is not None, "Visualizer creation failed"
            assert hasattr(visualizer, 'artistic_styles'), "Missing artistic styles"
            
            # Test consciousness data processing
            test_consciousness = torch.randn(4).to(device)  # SCUP values
            
            # Test frame rendering
            fig = visualizer.render_frame(test_consciousness)
            assert fig is not None, "Frame rendering failed"
            
            # Test artwork creation
            test_state = visualizer._generate_simulated_consciousness_state()
            artwork = visualizer.create_consciousness_artwork(test_state)
            
            assert artwork is not None, "Artwork creation failed"
            assert artwork.artwork_id is not None, "Artwork missing ID"
            assert artwork.style_category in [style.value for style in visualizer.artistic_styles.keys()], "Invalid style category"
            
            # Test artwork saving
            artwork_path = self.test_data_dir / "test_advanced_artwork.png"
            saved = visualizer.save_artwork(artwork, str(artwork_path))
            assert saved, "Artwork saving failed"
            
            # Test metrics
            metrics = visualizer.get_visual_consciousness_metrics()
            assert metrics.total_artworks_created >= 1, "Artwork count not updated"
            
            plt.close('all')  # Cleanup
            
            print("  ✓ Visualizer creation")
            print("  ✓ Consciousness data processing")
            print("  ✓ Frame rendering")
            print("  ✓ Artwork creation")
            print("  ✓ Artwork saving")
            print("  ✓ Metrics tracking")
            
            return True
            
        except Exception as e:
            print(f"  ❌ Advanced visual consciousness test failed: {e}")
            return False
    
    def test_tick_pulse_visualizer(self) -> bool:
        """Test tick pulse visualizer"""
        try:
            # Create configurations
            tick_config = TickPulseConfig(
                buffer_size=50,
                save_frames=True,
                output_dir=str(self.test_data_dir)
            )
            
            # Test visualizer creation
            visualizer = create_tick_pulse_visualizer(tick_config, self.visual_config)
            
            assert visualizer is not None, "Tick pulse visualizer creation failed"
            
            # Test data processing
            test_data = torch.tensor([0.5], dtype=torch.float32).to(device)
            
            # Test frame rendering
            fig = visualizer.render_frame(test_data)
            assert fig is not None, "Tick pulse frame rendering failed"
            
            # Test multiple data points
            for i in range(10):
                pulse_value = 0.5 + 0.3 * np.sin(i * 0.5)
                test_tensor = torch.tensor([pulse_value], dtype=torch.float32).to(device)
                visualizer._update_tick_data_from_tensor(visualizer.process_consciousness_tensor(test_tensor))
            
            # Test metrics
            metrics = visualizer.get_tick_metrics()
            assert 'current_tick' in metrics, "Missing current tick in metrics"
            assert 'tick_rate_hz' in metrics, "Missing tick rate in metrics"
            assert 'pulse_strength' in metrics, "Missing pulse strength in metrics"
            
            # Test frame saving
            saved_path = visualizer.save_consciousness_frame("test_tick_pulse.png")
            assert Path(saved_path).exists(), f"Tick pulse frame not saved"
            
            plt.close('all')  # Cleanup
            
            print("  ✓ Visualizer creation")
            print("  ✓ Data processing")
            print("  ✓ Frame rendering")
            print("  ✓ Multi-point updates")
            print("  ✓ Metrics collection")
            print("  ✓ Frame saving")
            
            return True
            
        except Exception as e:
            print(f"  ❌ Tick pulse visualizer test failed: {e}")
            return False
    
    def test_entropy_flow_visualizer(self) -> bool:
        """Test entropy flow visualizer"""
        try:
            # Create configurations
            entropy_config = EntropyFlowConfig(
                grid_size=8,  # Smaller for testing
                buffer_size=20,
                save_frames=True,
                output_dir=str(self.test_data_dir)
            )
            
            # Test visualizer creation
            visualizer = create_entropy_flow_visualizer(entropy_config, self.visual_config)
            
            assert visualizer is not None, "Entropy flow visualizer creation failed"
            assert visualizer.entropy_processor is not None, "Missing entropy processor"
            
            # Test data processing
            test_data = torch.tensor([0.6, 0.4, 0.3], dtype=torch.float32).to(device)
            
            # Test frame rendering
            fig = visualizer.render_frame(test_data)
            assert fig is not None, "Entropy flow frame rendering failed"
            
            # Test neural processor
            with torch.no_grad():
                visualizer.entropy_processor.eval()
                test_state = torch.randn(1, 4).to(device)
                U, V = visualizer.entropy_processor(test_state)
                assert U.shape == (1, entropy_config.grid_size, entropy_config.grid_size), f"Unexpected U shape: {U.shape}"
                assert V.shape == (1, entropy_config.grid_size, entropy_config.grid_size), f"Unexpected V shape: {V.shape}"
                assert not torch.isnan(U).any(), "U contains NaN values"
                assert not torch.isnan(V).any(), "V contains NaN values"
            
            # Test metrics
            metrics = visualizer.get_entropy_metrics()
            assert 'current_entropy' in metrics, "Missing current entropy in metrics"
            assert 'total_flow_magnitude' in metrics, "Missing flow magnitude in metrics"
            
            # Test frame saving
            saved_path = visualizer.save_consciousness_frame("test_entropy_flow.png")
            assert Path(saved_path).exists(), f"Entropy flow frame not saved"
            
            plt.close('all')  # Cleanup
            
            print("  ✓ Visualizer creation")
            print("  ✓ Data processing")
            print("  ✓ Frame rendering")
            print("  ✓ Neural processor")
            print("  ✓ Metrics collection")
            print("  ✓ Frame saving")
            
            return True
            
        except Exception as e:
            print(f"  ❌ Entropy flow visualizer test failed: {e}")
            return False
    
    def test_heat_monitor_visualizer(self) -> bool:
        """Test heat monitor visualizer"""
        try:
            # Create configurations
            heat_config = HeatMonitorConfig(
                buffer_size=30,
                save_frames=True,
                output_dir=str(self.test_data_dir)
            )
            
            # Test visualizer creation
            visualizer = create_heat_monitor_visualizer(heat_config, self.visual_config)
            
            assert visualizer is not None, "Heat monitor visualizer creation failed"
            assert len(visualizer.heat_zones) > 0, "Missing heat zones"
            
            # Test data processing
            test_data = torch.tensor([0.7, 0.5], dtype=torch.float32).to(device)
            
            # Test frame rendering
            fig = visualizer.render_frame(test_data)
            assert fig is not None, "Heat monitor frame rendering failed"
            
            # Test heat processing
            for heat_level in [0.1, 0.3, 0.6, 0.9]:
                test_tensor = torch.tensor([heat_level], dtype=torch.float32).to(device)
                visualizer._update_heat_data_from_tensor(visualizer.process_consciousness_tensor(test_tensor))
                
                # Check heat zone detection
                current_zone = visualizer._get_current_heat_zone()
                assert current_zone is not None, f"No heat zone found for level {heat_level}"
                assert current_zone.min_heat <= heat_level <= current_zone.max_heat, f"Heat level {heat_level} not in zone {current_zone.name}"
            
            # Test metrics
            metrics = visualizer.get_heat_metrics()
            assert 'current_heat' in metrics, "Missing current heat in metrics"
            assert 'peak_heat' in metrics, "Missing peak heat in metrics"
            assert 'current_zone' in metrics, "Missing current zone in metrics"
            
            # Test frame saving
            saved_path = visualizer.save_consciousness_frame("test_heat_monitor.png")
            assert Path(saved_path).exists(), f"Heat monitor frame not saved"
            
            plt.close('all')  # Cleanup
            
            print("  ✓ Visualizer creation")
            print("  ✓ Data processing")
            print("  ✓ Frame rendering")
            print("  ✓ Heat zone detection")
            print("  ✓ Metrics collection")
            print("  ✓ Frame saving")
            
            return True
            
        except Exception as e:
            print(f"  ❌ Heat monitor visualizer test failed: {e}")
            return False
    
    def test_performance_benchmarks(self) -> Dict[str, float]:
        """Test performance benchmarks for all visualizers"""
        try:
            benchmarks = {}
            
            # Test base visualizer performance
            visualizer = ExampleConsciousnessVisualizer(self.visual_config)
            test_data = torch.randn(100, 3).to(device)
            
            # Benchmark frame rendering
            start_time = time.time()
            for _ in range(10):
                fig = visualizer.render_frame(test_data)
                plt.close(fig)
            base_render_time = (time.time() - start_time) / 10 * 1000  # ms per frame
            
            benchmarks['base_render_ms'] = base_render_time
            
            # Benchmark tensor processing
            start_time = time.time()
            for _ in range(100):
                processed = visualizer.process_consciousness_tensor(test_data)
            tensor_process_time = (time.time() - start_time) / 100 * 1000  # ms per operation
            
            benchmarks['tensor_process_ms'] = tensor_process_time
            
            print(f"  ✓ Base render time: {base_render_time:.2f}ms/frame")
            print(f"  ✓ Tensor processing: {tensor_process_time:.4f}ms/op")
            print(f"  ✓ Estimated max FPS: {1000/base_render_time:.1f}")
            
            return benchmarks
            
        except Exception as e:
            print(f"  ❌ Performance benchmark test failed: {e}")
            return {}
    
    def test_memory_efficiency(self) -> Dict[str, float]:
        """Test memory efficiency of visualizations"""
        try:
            memory_stats = {}
            
            if torch.cuda.is_available():
                # Clear GPU memory
                torch.cuda.empty_cache()
                initial_memory = torch.cuda.memory_allocated(device) / 1024**2  # MB
                
                # Create multiple visualizers
                visualizers = []
                for i in range(5):
                    vis = ExampleConsciousnessVisualizer(self.visual_config)
                    test_data = torch.randn(50, 3).to(device)
                    fig = vis.render_frame(test_data)
                    visualizers.append((vis, fig))
                
                peak_memory = torch.cuda.memory_allocated(device) / 1024**2  # MB
                memory_used = peak_memory - initial_memory
                
                # Cleanup
                for vis, fig in visualizers:
                    plt.close(fig)
                del visualizers
                torch.cuda.empty_cache()
                
                final_memory = torch.cuda.memory_allocated(device) / 1024**2  # MB
                
                memory_stats['initial_memory_mb'] = initial_memory
                memory_stats['peak_memory_mb'] = peak_memory
                memory_stats['memory_used_mb'] = memory_used
                memory_stats['final_memory_mb'] = final_memory
                memory_stats['memory_per_visualizer_mb'] = memory_used / 5
                
                print(f"  ✓ Initial GPU memory: {initial_memory:.1f}MB")
                print(f"  ✓ Peak GPU memory: {peak_memory:.1f}MB")
                print(f"  ✓ Memory per visualizer: {memory_used/5:.1f}MB")
                print(f"  ✓ Memory cleanup: {(peak_memory - final_memory):.1f}MB freed")
            else:
                print("  ⚠️ CUDA not available, skipping GPU memory test")
                memory_stats['gpu_available'] = False
            
            return memory_stats
            
        except Exception as e:
            print(f"  ❌ Memory efficiency test failed: {e}")
            return {}
    
    def test_device_compatibility(self) -> Dict[str, Any]:
        """Test device compatibility (CPU/GPU)"""
        try:
            compatibility = {}
            
            # Test CPU compatibility
            original_device = device
            cpu_device = torch.device('cpu')
            
            # Force CPU mode
            test_data_cpu = torch.randn(20, 3, device=cpu_device)
            visualizer = ExampleConsciousnessVisualizer(self.visual_config)
            
            # Test tensor processing on CPU
            processed_cpu = visualizer.process_consciousness_tensor(test_data_cpu)
            assert processed_cpu.device == cpu_device, "CPU processing failed"
            
            compatibility['cpu_compatible'] = True
            
            # Test GPU compatibility if available
            if torch.cuda.is_available():
                gpu_device = torch.device('cuda')
                test_data_gpu = torch.randn(20, 3, device=gpu_device)
                
                processed_gpu = visualizer.process_consciousness_tensor(test_data_gpu)
                assert processed_gpu.device == cpu_device, "GPU tensor not properly moved to CPU for visualization"
                
                compatibility['gpu_compatible'] = True
                compatibility['gpu_device_name'] = torch.cuda.get_device_name(0)
            else:
                compatibility['gpu_compatible'] = False
                compatibility['gpu_device_name'] = "N/A"
            
            compatibility['current_device'] = str(original_device)
            
            print(f"  ✓ CPU compatibility: {compatibility['cpu_compatible']}")
            print(f"  ✓ GPU compatibility: {compatibility['gpu_compatible']}")
            print(f"  ✓ Current device: {compatibility['current_device']}")
            if compatibility['gpu_compatible']:
                print(f"  ✓ GPU device: {compatibility['gpu_device_name']}")
            
            return compatibility
            
        except Exception as e:
            print(f"  ❌ Device compatibility test failed: {e}")
            return {}
    
    def print_test_summary(self) -> None:
        """Print comprehensive test summary"""
        print("\n" + "=" * 60)
        print("🏁 DAWN Visualization Test Summary")
        print("=" * 60)
        
        total_tests = len(self.test_results)
        passed_tests = sum(1 for result in self.test_results.values() if result.get('status') == 'PASSED')
        failed_tests = sum(1 for result in self.test_results.values() if result.get('status') == 'FAILED')
        error_tests = sum(1 for result in self.test_results.values() if result.get('status') == 'ERROR')
        
        print(f"Total Tests: {total_tests}")
        print(f"✅ Passed: {passed_tests}")
        print(f"❌ Failed: {failed_tests}")
        print(f"💥 Errors: {error_tests}")
        print(f"Success Rate: {(passed_tests/total_tests)*100:.1f}%")
        
        # Performance summary
        total_time = sum(result.get('duration_ms', 0) for result in self.test_results.values())
        print(f"Total Test Time: {total_time:.1f}ms")
        
        # Detailed results
        print("\nDetailed Results:")
        print("-" * 30)
        for test_name, result in self.test_results.items():
            status_emoji = {"PASSED": "✅", "FAILED": "❌", "ERROR": "💥"}.get(result['status'], "❓")
            duration = result.get('duration_ms', 0)
            print(f"{status_emoji} {test_name}: {result['status']} ({duration:.1f}ms)")
            
            if result['status'] == 'ERROR' and 'error' in result:
                print(f"    Error: {result['error']}")
        
        # System info
        print(f"\nSystem Information:")
        print(f"Device: {device}")
        print(f"PyTorch: {torch.__version__}")
        print(f"CUDA Available: {torch.cuda.is_available()}")
        if torch.cuda.is_available():
            print(f"GPU: {torch.cuda.get_device_name(0)}")
            print(f"GPU Memory: {torch.cuda.get_device_properties(0).total_memory / 1024**3:.1f}GB")
        
        print(f"\nTest Output Directory: {self.test_data_dir}")
        saved_files = list(self.test_data_dir.glob("*.png"))
        print(f"Generated Visualizations: {len(saved_files)}")
        
        if passed_tests == total_tests:
            print("\n🎉 All tests passed! DAWN matplotlib/seaborn integration is working correctly.")
        else:
            print(f"\n⚠️ {failed_tests + error_tests} tests failed. Check logs for details.")


def main():
    """Main test function"""
    print("🎨 DAWN Matplotlib/Seaborn Visualization Test Suite")
    print("=" * 60)
    print(f"PyTorch Version: {torch.__version__}")
    print(f"Device: {device}")
    print(f"CUDA Available: {torch.cuda.is_available()}")
    
    if not VISUALIZERS_AVAILABLE:
        print("❌ Cannot run tests - required visualizers not available")
        print("Please ensure all visualization modules are properly installed")
        return 1
    
    # Create and run test suite
    tester = DAWNVisualizationTester()
    results = tester.run_all_tests()
    
    # Check overall success
    total_tests = len(results)
    passed_tests = sum(1 for result in results.values() if result.get('status') == 'PASSED')
    
    if passed_tests == total_tests:
        print("\n🎉 All visualization tests completed successfully!")
        return 0
    else:
        print(f"\n⚠️ {total_tests - passed_tests} tests failed")
        return 1


if __name__ == "__main__":
    exit_code = main()
    sys.exit(exit_code)
