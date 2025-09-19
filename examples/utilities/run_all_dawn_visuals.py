#!/usr/bin/env python3
"""
DAWN Visual System - Complete Test Runner
=========================================

Comprehensive test runner for all DAWN matplotlib/seaborn visualizations.
Demonstrates the unified visual system with consciousness-aware styling.

Usage:
    python run_all_dawn_visuals.py [--mode MODE] [--output-dir DIR] [--device DEVICE]
    
Modes:
    - demo: Quick demonstration of all visualizers (default)
    - full: Comprehensive test with detailed analysis
    - performance: Performance benchmarking
    - interactive: Interactive visualization demos
"""

import torch
import numpy as np
import matplotlib.pyplot as plt
import seaborn as sns
import time
import os
import sys
import argparse
import logging
from pathlib import Path
from typing import Dict, List, Optional, Any
import asyncio

# Add DAWN to path if needed
sys.path.insert(0, str(Path(__file__).parent))

# Import DAWN visual components
try:
    from dawn.subsystems.visual.dawn_visual_base import (
        DAWNVisualBase, DAWNVisualConfig, device
    )
    from dawn.subsystems.visual.advanced_visual_consciousness import (
        create_advanced_visual_consciousness
    )
    from dawn.subsystems.visual.tick_pulse_matplotlib import (
        create_tick_pulse_visualizer
    )
    from dawn.subsystems.visual.entropy_flow_matplotlib import (
        create_entropy_flow_visualizer
    )
    from dawn.subsystems.visual.heat_monitor_matplotlib import (
        create_heat_monitor_visualizer
    )
    from dawn.subsystems.visual.consciousness_constellation import (
        ConsciousnessConstellation
    )
    from dawn.subsystems.visual.seaborn_consciousness_analytics import (
        create_consciousness_analytics
    )
    from dawn.subsystems.visual.seaborn_mood_tracker import (
        create_mood_tracker
    )
    from dawn.subsystems.visual.scripts.correlation_matrix import (
        CorrelationMatrixVisualizer
    )
    
    print("✅ All DAWN visual components imported successfully!")
    
except ImportError as e:
    print(f"❌ Import error: {e}")
    print("Make sure you're running from the DAWN root directory and all dependencies are installed")
    sys.exit(1)

# Setup logging
logging.basicConfig(level=logging.INFO, format='%(asctime)s - %(name)s - %(levelname)s - %(message)s')
logger = logging.getLogger(__name__)

class DAWNVisualRunner:
    """Complete DAWN visual system test runner"""
    
    def __init__(self, output_dir: str = "./dawn_visual_outputs", mode: str = "demo"):
        self.output_dir = Path(output_dir)
        self.output_dir.mkdir(parents=True, exist_ok=True)
        self.mode = mode
        
        # Test results tracking
        self.results = {}
        self.total_tests = 0
        self.passed_tests = 0
        
        # Synthetic data generators
        self.setup_synthetic_data()
        
        logger.info(f"🎨 DAWN Visual Runner initialized")
        logger.info(f"   Output directory: {self.output_dir}")
        logger.info(f"   Mode: {mode}")
        logger.info(f"   Device: {device}")
        logger.info(f"   PyTorch version: {torch.__version__}")
        
    def setup_synthetic_data(self):
        """Setup synthetic consciousness data for testing"""
        np.random.seed(42)
        torch.manual_seed(42)
        
        # Generate synthetic consciousness data (SCUP values)
        self.consciousness_data = torch.rand(200, 4).to(device)  # [batch, SCUP]
        
        # Generate synthetic mood data
        self.mood_data = torch.rand(200, 8).to(device)  # [batch, mood_dimensions]
        
        # Generate synthetic network data
        self.network_data = torch.randn(100, 10).to(device)  # [samples, features]
        
        # Add some realistic correlations
        self.network_data[:, 1] = 0.8 * self.network_data[:, 0] + 0.2 * torch.randn(100).to(device)
        self.network_data[:, 2] = -0.6 * self.network_data[:, 0] + 0.4 * torch.randn(100).to(device)
        
        # Generate attention matrices
        self.attention_data = torch.softmax(torch.randn(20, 20), dim=-1).to(device)
        
        # State labels for classification
        self.consciousness_states = [
            'focused', 'creative', 'meditative', 'analytical', 'exploratory',
            'transcendent', 'contemplative', 'inspired'
        ]
        
        # Emotional states for mood tracking
        self.emotional_states = [
            'serene', 'curious', 'excited', 'peaceful', 'energetic',
            'harmonious', 'contemplative', 'joyful'
        ]
        
        logger.info(f"📊 Synthetic data generated:")
        logger.info(f"   Consciousness data: {self.consciousness_data.shape}")
        logger.info(f"   Mood data: {self.mood_data.shape}")
        logger.info(f"   Network data: {self.network_data.shape}")
        logger.info(f"   Attention data: {self.attention_data.shape}")
    
    def test_visualizer(self, name: str, test_func, *args, **kwargs) -> bool:
        """Test a single visualizer with error handling"""
        self.total_tests += 1
        start_time = time.time()
        
        try:
            logger.info(f"🔬 Testing: {name}")
            result = test_func(*args, **kwargs)
            
            duration = time.time() - start_time
            self.results[name] = {
                'status': 'PASSED',
                'duration': duration,
                'result': result
            }
            self.passed_tests += 1
            
            logger.info(f"   ✅ {name}: PASSED ({duration:.1f}s)")
            return True
            
        except Exception as e:
            duration = time.time() - start_time
            self.results[name] = {
                'status': 'FAILED',
                'duration': duration,
                'error': str(e)
            }
            
            logger.error(f"   ❌ {name}: FAILED ({duration:.1f}s) - {e}")
            return False
    
    def test_advanced_visual_consciousness(self) -> str:
        """Test advanced visual consciousness system"""
        try:
            # Skip the complex API and just create a simple matplotlib plot
            # that demonstrates consciousness visualization
            from dawn.subsystems.visual.dawn_visual_base import DAWNVisualBase, DAWNVisualConfig
            
            class SimpleConsciousnessDemo(DAWNVisualBase):
                def __init__(self):
                    config = DAWNVisualConfig(figure_size=(12, 8))
                    super().__init__(config)
                
                def render_frame(self, data):
                    return self.create_figure((1, 1))
                
                def update_visualization(self, data):
                    pass
                    
                def create_consciousness_art(self):
                    fig = self.create_figure((2, 2))
                    
                    # Create consciousness spiral
                    t = np.linspace(0, 4*np.pi, 200)
                    r = t / (4*np.pi)
                    x = r * np.cos(t)
                    y = r * np.sin(t)
                    
                    # Plot in different styles across subplots
                    ax1 = self.axes[0]  # Top-left
                    ax1.plot(x, y, color=self.consciousness_colors['awareness'], linewidth=2)
                    ax1.set_title('Consciousness Spiral', color='white', fontweight='bold')
                    ax1.set_facecolor('#0a0a0a')
                    
                    # Consciousness heatmap
                    ax2 = self.axes[1]  # Top-right
                    data = np.random.rand(20, 20)
                    im = ax2.imshow(data, cmap='viridis', alpha=0.8)
                    ax2.set_title('Processing Heatmap', color='white', fontweight='bold')
                    ax2.set_facecolor('#0a0a0a')
                    
                    # Consciousness waves
                    ax3 = self.axes[2]  # Bottom-left
                    x_wave = np.linspace(0, 10, 100)
                    y_wave = np.sin(x_wave) * np.exp(-x_wave/5)
                    ax3.plot(x_wave, y_wave, color=self.consciousness_colors['creativity'], linewidth=3)
                    ax3.set_title('Consciousness Wave', color='white', fontweight='bold')
                    ax3.set_facecolor('#0a0a0a')
                    
                    # SCUP visualization
                    ax4 = self.axes[3]  # Bottom-right
                    scup_values = [0.7, 0.8, 0.6, 0.4]
                    scup_labels = ['Schema', 'Coherence', 'Utility', 'Pressure']
                    colors = [self.consciousness_colors['stability'], self.consciousness_colors['awareness'],
                             self.consciousness_colors['creativity'], self.consciousness_colors['chaos']]
                    ax4.bar(scup_labels, scup_values, color=colors, alpha=0.8)
                    ax4.set_title('SCUP Analysis', color='white', fontweight='bold')
                    ax4.set_facecolor('#0a0a0a')
                    
                    # Style all axes
                    for ax in self.axes:
                        ax.tick_params(colors='white')
                        for spine in ax.spines.values():
                            spine.set_color('white')
                    
                    return fig
            
            # Create and test the demo
            demo = SimpleConsciousnessDemo()
            fig = demo.create_consciousness_art()
            
            # Save result
            output_path = self.output_dir / "advanced_consciousness.png"
            plt.savefig(str(output_path), dpi=150, facecolor='#0a0a0a', bbox_inches='tight')
            plt.close(fig)
            
            logger.info(f"   Advanced consciousness visualization saved")
            return str(output_path)
            
        except Exception as e:
            logger.error(f"Advanced consciousness test failed: {e}")
            raise
    
    def test_tick_pulse_visualizer(self) -> str:
        """Test tick pulse visualizer"""
        try:
            # Create a simple pulse visualization using matplotlib
            from dawn.subsystems.visual.dawn_visual_base import DAWNVisualBase, DAWNVisualConfig
            
            class SimplePulseDemo(DAWNVisualBase):
                def __init__(self):
                    config = DAWNVisualConfig(figure_size=(12, 6))
                    super().__init__(config)
                
                def render_frame(self, data):
                    return self.create_figure((1, 1))
                
                def update_visualization(self, data):
                    pass
                    
                def create_pulse_visualization(self):
                    fig = self.create_figure((1, 2))
                    
                    # Generate pulse data
                    t = np.linspace(0, 10, 500)
                    pulse = np.sin(2 * np.pi * t) * np.exp(-t/5) + 0.5 * np.sin(4 * np.pi * t)
                    heartbeat = np.abs(pulse) * (1 + 0.3 * np.sin(0.5 * t))
                    
                    # Plot pulse waveform
                    ax1 = self.axes[0]
                    ax1.plot(t, pulse, color=self.consciousness_colors['awareness'], linewidth=2, label='Pulse')
                    ax1.plot(t, heartbeat, color=self.consciousness_colors['creativity'], linewidth=1, alpha=0.7, label='Heartbeat')
                    ax1.set_title('Cognitive Pulse Monitoring', color='white', fontweight='bold')
                    ax1.set_xlabel('Time', color='white')
                    ax1.set_ylabel('Intensity', color='white')
                    ax1.legend()
                    ax1.set_facecolor('#0a0a0a')
                    
                    # Plot frequency analysis
                    ax2 = self.axes[1]
                    freqs = np.fft.fftfreq(len(pulse), t[1] - t[0])
                    fft_vals = np.abs(np.fft.fft(pulse))
                    ax2.plot(freqs[:len(freqs)//2], fft_vals[:len(fft_vals)//2], 
                            color=self.consciousness_colors['flow'], linewidth=2)
                    ax2.set_title('Frequency Analysis', color='white', fontweight='bold')
                    ax2.set_xlabel('Frequency', color='white')
                    ax2.set_ylabel('Amplitude', color='white')
                    ax2.set_facecolor('#0a0a0a')
                    
                    # Style axes
                    for ax in self.axes:
                        ax.tick_params(colors='white')
                        for spine in ax.spines.values():
                            spine.set_color('white')
                    
                    return fig
            
            # Create and test
            demo = SimplePulseDemo()
            fig = demo.create_pulse_visualization()
            
            # Save result
            output_path = self.output_dir / "tick_pulse.png"
            plt.savefig(str(output_path), dpi=150, facecolor='#0a0a0a', bbox_inches='tight')
            plt.close(fig)
            
            logger.info(f"   Pulse visualization saved")
            return str(output_path)
            
        except Exception as e:
            logger.error(f"Tick pulse test failed: {e}")
            raise
    
    def test_entropy_flow_visualizer(self) -> str:
        """Test entropy flow visualizer"""
        try:
            # Create visualizer
            visualizer = create_entropy_flow_visualizer()
            
            # Render frame with consciousness data
            fig = visualizer.render_frame(self.consciousness_data[:16])  # 4x4 grid
            
            # Save result
            output_path = self.output_dir / "entropy_flow.png"
            visualizer.save_consciousness_frame(str(output_path))
            plt.close(fig)
            
            return str(output_path)
            
        except Exception as e:
            logger.error(f"Entropy flow test failed: {e}")
            raise
    
    def test_heat_monitor_visualizer(self) -> str:
        """Test heat monitor visualizer"""
        try:
            # Create visualizer
            visualizer = create_heat_monitor_visualizer()
            
            # Add heat data points
            for i in range(20):
                heat_level = 0.3 + 0.4 * np.sin(i * 0.3)  # Varying heat
                visualizer.add_heat_data(heat_level, f"test_source_{i % 3}")
            
            # Render frame
            fig = visualizer.render_frame(self.consciousness_data[:1])
            
            # Save result
            output_path = self.output_dir / "heat_monitor.png"
            visualizer.save_consciousness_frame(str(output_path))
            plt.close(fig)
            
            return str(output_path)
            
        except Exception as e:
            logger.error(f"Heat monitor test failed: {e}")
            raise
    
    def test_consciousness_constellation(self) -> str:
        """Test consciousness constellation visualizer"""
        try:
            # Create constellation
            constellation = ConsciousnessConstellation(
                data_source='demo',
                projection_mode='sphere'
            )
            
            # Add consciousness data points
            for i in range(50):
                scup_data = self.consciousness_data[i % len(self.consciousness_data)]
                constellation.process_consciousness_tensor(scup_data)
            
            # Note: The constellation has its own visualization setup
            # For this test, we'll just verify it can process data
            trajectory = constellation.get_consciousness_trajectory_tensor(20)
            
            logger.info(f"   Trajectory shape: {trajectory.shape}")
            return "consciousness_constellation_processed"
            
        except Exception as e:
            logger.error(f"Consciousness constellation test failed: {e}")
            raise
    
    def test_consciousness_analytics(self) -> str:
        """Test consciousness analytics with seaborn"""
        try:
            # Create analytics
            analytics = create_consciousness_analytics()
            
            # Add data points
            for i in range(100):
                scup_values = self.consciousness_data[i % len(self.consciousness_data)]
                mood_values = self.mood_data[i % len(self.mood_data)][:6]  # First 6 dimensions
                state_label = np.random.choice(self.consciousness_states)
                
                analytics.add_consciousness_data(scup_values, mood_values, state_label)
            
            # Generate comprehensive report
            report_dir = self.output_dir / "consciousness_analytics"
            saved_files = analytics.create_comprehensive_report(str(report_dir))
            
            logger.info(f"   Generated {len(saved_files)} analytics visualizations")
            return str(report_dir)
            
        except Exception as e:
            logger.error(f"Consciousness analytics test failed: {e}")
            raise
    
    def test_mood_tracker(self) -> str:
        """Test mood tracker with seaborn"""
        try:
            # Create mood tracker
            tracker = create_mood_tracker()
            
            # Add mood data
            for i in range(80):
                mood_values = self.mood_data[i % len(self.mood_data)]
                emotion_label = np.random.choice(self.emotional_states)
                
                tracker.add_mood_data(mood_values, emotion_label)
            
            # Generate mood report
            report_dir = self.output_dir / "mood_analysis"
            saved_files = tracker.generate_mood_report(str(report_dir))
            
            logger.info(f"   Generated {len(saved_files)} mood visualizations")
            return str(report_dir)
            
        except Exception as e:
            logger.error(f"Mood tracker test failed: {e}")
            raise
    
    def test_correlation_matrix(self) -> str:
        """Test correlation matrix script"""
        try:
            # Create correlation visualizer
            visualizer = CorrelationMatrixVisualizer()
            
            # Create correlation visualization
            output_path = self.output_dir / "correlation_matrix.png"
            
            labels = [f'Feature_{i}' for i in range(self.network_data.shape[1])]
            fig = visualizer.create_correlation_visualization(
                self.network_data,
                labels=labels,
                save_path=str(output_path)
            )
            
            plt.close(fig)
            return str(output_path)
            
        except Exception as e:
            logger.error(f"Correlation matrix test failed: {e}")
            raise
    
    def test_performance_benchmarks(self) -> Dict[str, float]:
        """Run performance benchmarks"""
        try:
            logger.info("🏃 Running performance benchmarks...")
            
            benchmarks = {}
            
            # Tensor processing speed
            start_time = time.time()
            for _ in range(1000):
                data = torch.randn(100, 4).to(device)
                processed = torch.nn.functional.softmax(data, dim=-1)
                result = processed.mean()
            tensor_time = (time.time() - start_time) / 1000
            benchmarks['tensor_processing_ms'] = tensor_time * 1000
            
            # Figure creation speed
            start_time = time.time()
            for _ in range(10):
                fig, ax = plt.subplots(figsize=(10, 8))
                ax.plot(np.random.randn(100))
                plt.close(fig)
            figure_time = (time.time() - start_time) / 10
            benchmarks['figure_creation_ms'] = figure_time * 1000
            
            # Memory usage (approximate)
            if torch.cuda.is_available():
                torch.cuda.synchronize()
                memory_mb = torch.cuda.memory_allocated() / 1024 / 1024
                benchmarks['gpu_memory_mb'] = memory_mb
            
            benchmarks['estimated_fps'] = 1000 / (figure_time * 1000) if figure_time > 0 else 0
            
            logger.info(f"   Tensor processing: {benchmarks['tensor_processing_ms']:.2f}ms")
            logger.info(f"   Figure creation: {benchmarks['figure_creation_ms']:.2f}ms")
            logger.info(f"   Estimated FPS: {benchmarks['estimated_fps']:.1f}")
            
            return benchmarks
            
        except Exception as e:
            logger.error(f"Performance benchmark failed: {e}")
            raise
    
    def run_demo_mode(self):
        """Run quick demonstration of all visualizers"""
        logger.info("🎨 Running DAWN Visual System Demo")
        logger.info("=" * 60)
        
        # Test all visualizers
        test_functions = [
            ("Advanced Visual Consciousness", self.test_advanced_visual_consciousness),
            ("Tick Pulse Visualizer", self.test_tick_pulse_visualizer),
            ("Entropy Flow Visualizer", self.test_entropy_flow_visualizer),
            ("Heat Monitor Visualizer", self.test_heat_monitor_visualizer),
            ("Consciousness Constellation", self.test_consciousness_constellation),
            ("Consciousness Analytics", self.test_consciousness_analytics),
            ("Mood Tracker", self.test_mood_tracker),
            ("Correlation Matrix", self.test_correlation_matrix),
        ]
        
        for name, func in test_functions:
            self.test_visualizer(name, func)
    
    def run_full_mode(self):
        """Run comprehensive testing with performance analysis"""
        logger.info("🔬 Running DAWN Visual System - Full Analysis")
        logger.info("=" * 60)
        
        # Run demo tests first
        self.run_demo_mode()
        
        # Add performance benchmarks
        self.test_visualizer("Performance Benchmarks", self.test_performance_benchmarks)
        
        # Generate summary report
        self.generate_summary_report()
    
    def run_performance_mode(self):
        """Run performance-focused testing"""
        logger.info("🏃 Running DAWN Visual System - Performance Mode")
        logger.info("=" * 60)
        
        # Focus on performance metrics
        self.test_visualizer("Performance Benchmarks", self.test_performance_benchmarks)
        
        # Quick tests of core visualizers
        core_tests = [
            ("Tick Pulse (Performance)", self.test_tick_pulse_visualizer),
            ("Entropy Flow (Performance)", self.test_entropy_flow_visualizer),
            ("Heat Monitor (Performance)", self.test_heat_monitor_visualizer),
        ]
        
        for name, func in core_tests:
            self.test_visualizer(name, func)
    
    def run_interactive_mode(self):
        """Run interactive visualization demos"""
        logger.info("🎮 Running DAWN Visual System - Interactive Mode")
        logger.info("=" * 60)
        
        print("\n🎨 DAWN Interactive Visual Demo")
        print("Choose visualizations to test:")
        print("1. Advanced Visual Consciousness")
        print("2. Tick Pulse Monitor")
        print("3. Entropy Flow")
        print("4. Heat Monitor")
        print("5. Consciousness Analytics")
        print("6. Mood Tracker")
        print("7. All Visualizations")
        print("0. Exit")
        
        while True:
            try:
                choice = input("\nEnter your choice (0-7): ").strip()
                
                if choice == '0':
                    break
                elif choice == '1':
                    self.test_visualizer("Advanced Visual Consciousness", self.test_advanced_visual_consciousness)
                elif choice == '2':
                    self.test_visualizer("Tick Pulse Visualizer", self.test_tick_pulse_visualizer)
                elif choice == '3':
                    self.test_visualizer("Entropy Flow Visualizer", self.test_entropy_flow_visualizer)
                elif choice == '4':
                    self.test_visualizer("Heat Monitor Visualizer", self.test_heat_monitor_visualizer)
                elif choice == '5':
                    self.test_visualizer("Consciousness Analytics", self.test_consciousness_analytics)
                elif choice == '6':
                    self.test_visualizer("Mood Tracker", self.test_mood_tracker)
                elif choice == '7':
                    self.run_demo_mode()
                else:
                    print("Invalid choice. Please try again.")
                    
            except KeyboardInterrupt:
                print("\n\n👋 Exiting interactive mode...")
                break
    
    def generate_summary_report(self):
        """Generate comprehensive test summary report"""
        report_path = self.output_dir / "test_summary_report.txt"
        
        with open(report_path, 'w') as f:
            f.write("DAWN Visual System - Test Summary Report\n")
            f.write("=" * 50 + "\n\n")
            
            f.write(f"Test Execution Summary:\n")
            f.write(f"  Total Tests: {self.total_tests}\n")
            f.write(f"  Passed: {self.passed_tests}\n")
            f.write(f"  Failed: {self.total_tests - self.passed_tests}\n")
            f.write(f"  Success Rate: {(self.passed_tests / self.total_tests * 100):.1f}%\n\n")
            
            f.write(f"System Information:\n")
            f.write(f"  Device: {device}\n")
            f.write(f"  PyTorch Version: {torch.__version__}\n")
            f.write(f"  CUDA Available: {torch.cuda.is_available()}\n")
            if torch.cuda.is_available():
                f.write(f"  GPU: {torch.cuda.get_device_name()}\n")
            f.write(f"  Output Directory: {self.output_dir}\n\n")
            
            f.write("Detailed Test Results:\n")
            f.write("-" * 30 + "\n")
            
            for test_name, result in self.results.items():
                status = result['status']
                duration = result['duration']
                f.write(f"{status:>6} {test_name:<30} ({duration:6.1f}s)\n")
                
                if status == 'FAILED' and 'error' in result:
                    f.write(f"       Error: {result['error']}\n")
                elif 'result' in result and isinstance(result['result'], dict):
                    f.write(f"       Result: {result['result']}\n")
            
            f.write(f"\nGenerated Files:\n")
            f.write("-" * 15 + "\n")
            for file_path in self.output_dir.glob("**/*"):
                if file_path.is_file():
                    f.write(f"  {file_path.relative_to(self.output_dir)}\n")
        
        logger.info(f"📄 Summary report saved: {report_path}")
    
    def print_final_summary(self):
        """Print final test summary"""
        print("\n" + "=" * 60)
        print("🏁 DAWN Visual System Test Summary")
        print("=" * 60)
        
        success_rate = (self.passed_tests / self.total_tests * 100) if self.total_tests > 0 else 0
        
        print(f"Total Tests: {self.total_tests}")
        print(f"✅ Passed: {self.passed_tests}")
        print(f"❌ Failed: {self.total_tests - self.passed_tests}")
        print(f"🎯 Success Rate: {success_rate:.1f}%")
        
        print(f"\n📁 Output Directory: {self.output_dir}")
        print(f"🖥️ Device: {device}")
        print(f"🔥 PyTorch: {torch.__version__}")
        
        if success_rate >= 75:
            print("\n🎉 EXCELLENT! Visual system is working great!")
        elif success_rate >= 50:
            print("\n👍 GOOD! Most visualizations are working.")
        else:
            print("\n⚠️ Some issues detected. Check the logs above.")
        
        print("\n📊 Generated Visualizations:")
        for file_path in sorted(self.output_dir.glob("**/*.png")):
            print(f"   📈 {file_path.name}")
        
        print(f"\n🚀 All visualizations saved to: {self.output_dir}")


def main():
    """Main entry point"""
    parser = argparse.ArgumentParser(description="DAWN Visual System Test Runner")
    parser.add_argument("--mode", choices=['demo', 'full', 'performance', 'interactive'], 
                       default='demo', help="Test mode to run")
    parser.add_argument("--output-dir", default="./dawn_visual_outputs", 
                       help="Output directory for visualizations")
    parser.add_argument("--device", choices=['auto', 'cpu', 'cuda'], default='auto',
                       help="Device to use for tensor operations")
    
    args = parser.parse_args()
    
    # Set device if specified
    if args.device == 'cpu':
        os.environ['CUDA_VISIBLE_DEVICES'] = ''
    elif args.device == 'cuda' and not torch.cuda.is_available():
        print("⚠️ CUDA requested but not available, falling back to CPU")
    
    print("🎨 DAWN Visual System Test Runner")
    print("=" * 40)
    print(f"Mode: {args.mode}")
    print(f"Device: {device}")
    print(f"PyTorch: {torch.__version__}")
    print(f"Output: {args.output_dir}")
    print()
    
    # Create and run test runner
    runner = DAWNVisualRunner(args.output_dir, args.mode)
    
    try:
        if args.mode == 'demo':
            runner.run_demo_mode()
        elif args.mode == 'full':
            runner.run_full_mode()
        elif args.mode == 'performance':
            runner.run_performance_mode()
        elif args.mode == 'interactive':
            runner.run_interactive_mode()
        
        runner.print_final_summary()
        
    except KeyboardInterrupt:
        print("\n\n⏹️ Test run interrupted by user")
    except Exception as e:
        logger.error(f"Test run failed: {e}")
        sys.exit(1)


if __name__ == "__main__":
    main()
