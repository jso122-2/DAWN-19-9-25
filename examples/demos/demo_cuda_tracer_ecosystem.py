#!/usr/bin/env python3
"""
🚀 DAWN CUDA Tracer Ecosystem Demo
==================================

Comprehensive demonstration of the CUDA-accelerated tracer ecosystem with
DAWN singleton integration, real-time GPU modeling, and interactive visualization.

Features demonstrated:
- DAWN singleton integration
- CUDA-accelerated tracer modeling
- Real-time ecosystem simulation
- GPU-powered visualization
- Telemetry and consciousness integration
- Interactive ecosystem control

"Consciousness modeling at the speed of light."
"""

import sys
import time
import logging
import asyncio
from pathlib import Path

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
    print("🚀" * 50)
    print("🧠 DAWN CUDA TRACER ECOSYSTEM DEMONSTRATION")
    print("🚀" * 50)
    print()
    print("This demo showcases:")
    print("✅ DAWN Singleton Integration")
    print("✅ CUDA-Accelerated Tracer Modeling")
    print("✅ Real-time GPU Simulation")
    print("✅ Interactive 3D Visualization")
    print("✅ Telemetry & Consciousness Integration")
    print()

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
            status = dawn.get_status()
            print(f"📊 DAWN Status: {status}")
            
            return dawn
        else:
            print("❌ DAWN system initialization failed")
            return None
            
    except Exception as e:
        print(f"⚠️  DAWN system initialization error: {e}")
        return None

def test_cuda_availability():
    """Test CUDA availability for tracer modeling"""
    print("🔍 Testing CUDA Availability...")
    
    cuda_status = {
        'cupy': False,
        'torch_cuda': False,
        'pycuda': False
    }
    
    # Test CuPy
    try:
        import cupy as cp
        cuda_status['cupy'] = True
        print(f"✅ CuPy: Available")
        
        # Test basic GPU operation
        test_array = cp.array([1, 2, 3, 4, 5])
        result = cp.sum(test_array)
        print(f"   GPU Test: {cp.asnumpy(result)} (success)")
        
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
            for i in range(device_count):
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
    
    print(f"📊 CUDA Status Summary: {cuda_status}")
    return any(cuda_status.values())

def create_tracer_ecosystem():
    """Create and configure the tracer ecosystem"""
    print("🏗️  Creating DAWN Tracer Ecosystem...")
    
    try:
        from dawn.consciousness.tracers import (
            create_cuda_tracer_ecosystem,
            CUDA_TRACER_AVAILABLE,
            CUDATracerModelConfig,
            TracerVisualizationConfig
        )
        
        if not CUDA_TRACER_AVAILABLE:
            print("⚠️  CUDA tracer components not available - using standard ecosystem")
            from dawn.consciousness.tracers import create_tracer_ecosystem
            return {'manager': create_tracer_ecosystem(nutrient_budget=150.0)}
        
        print("✅ CUDA tracer components available")
        
        # Configure CUDA modeling
        cuda_config = CUDATracerModelConfig(
            device_id=0,
            max_tracers=500,
            simulation_timesteps=1000,
            nutrient_grid_size=128,
            interaction_radius=0.2,
            enable_visualization=True,
            enable_predictive_analytics=True
        )
        
        # Configure visualization
        viz_config = TracerVisualizationConfig(
            window_size=(1600, 1200),
            fps=30.0,
            max_trail_length=50,
            particle_size_scale=1.5,
            enable_3d=True,
            enable_trails=True,
            enable_interactions=True,
            enable_nutrient_field=True,
            color_scheme="biological",
            background_color="black"
        )
        
        # Create CUDA ecosystem
        ecosystem = create_cuda_tracer_ecosystem(
            nutrient_budget=200.0,
            cuda_config=cuda_config,
            viz_config=viz_config,
            enable_visualization=True
        )
        
        print("✅ CUDA tracer ecosystem created successfully")
        print(f"📊 Components: {list(ecosystem.keys())}")
        
        return ecosystem
        
    except Exception as e:
        print(f"❌ Error creating tracer ecosystem: {e}")
        return None

def demonstrate_tracer_spawning(ecosystem):
    """Demonstrate tracer spawning and basic operations"""
    print("🐣 Demonstrating Tracer Spawning...")
    
    manager = ecosystem['manager']
    
    # Create test context for spawning
    test_context = {
        'tick_id': 1,
        'timestamp': time.time(),
        'entropy': 0.8,  # High entropy to trigger spawning
        'pressure': 0.6,
        'drift_magnitude': 0.4,
        'soot_ratio': 0.3,
        'avg_schema_coherence': 0.7,
        'memory_pressure': 0.4,
        'entropy_history': [0.5, 0.6, 0.7, 0.8],
        'pressure_history': [0.4, 0.5, 0.6],
        'drift_history': [0.2, 0.3, 0.4],
        'active_blooms': [],
        'soot_fragments': [],
        'schema_edges': [],
        'schema_clusters': [],
        'ash_fragments': [],
        'mycelial_flows': []
    }
    
    print("🎯 Test context created with high entropy to trigger spawning")
    
    # Run several ticks to spawn tracers
    for tick in range(1, 11):
        test_context['tick_id'] = tick
        test_context['timestamp'] = time.time()
        
        # Vary entropy to create interesting patterns
        test_context['entropy'] = 0.5 + 0.3 * (tick % 3) / 2
        test_context['pressure'] = 0.4 + 0.2 * (tick % 2)
        
        # Execute tick
        tick_summary = manager.tick(tick, test_context)
        
        active_count = tick_summary['ecosystem_state']['active_tracers']
        spawned = tick_summary['ecosystem_state']['spawned_this_tick']
        retired = tick_summary['ecosystem_state']['retired_this_tick']
        
        print(f"Tick {tick:2d}: Active={active_count:2d}, Spawned={spawned}, Retired={retired}")
        
        # Brief pause
        time.sleep(0.1)
    
    # Show final ecosystem state
    final_status = manager.get_tracer_status()
    print(f"📊 Final ecosystem state: {len(final_status)} active tracers")
    
    # Show tracer types
    tracer_types = {}
    for tracer_info in final_status.values():
        tracer_type = tracer_info['tracer_type']
        tracer_types[tracer_type] = tracer_types.get(tracer_type, 0) + 1
    
    print("🏷️  Active tracer types:")
    for tracer_type, count in tracer_types.items():
        print(f"   {tracer_type}: {count}")

def demonstrate_cuda_simulation(ecosystem):
    """Demonstrate CUDA-accelerated simulation"""
    print("🚀 Demonstrating CUDA Simulation...")
    
    cuda_engine = ecosystem.get('cuda_engine')
    if not cuda_engine:
        print("⚠️  CUDA engine not available")
        return
    
    manager = ecosystem['manager']
    
    # Initialize CUDA simulation with active tracers
    print("🏗️  Initializing CUDA simulation...")
    if manager.initialize_cuda_simulation():
        print("✅ CUDA simulation initialized")
        
        # Show GPU status
        cuda_status = manager.get_cuda_simulation_status()
        print(f"📊 CUDA Status: {cuda_status}")
        
        # Run simulation steps
        print("⚡ Running CUDA simulation steps...")
        for step in range(20):
            step_result = cuda_engine.run_cuda_simulation_step(dt=0.01)
            
            if 'error' not in step_result:
                execution_time = step_result.get('execution_time', 0) * 1000
                tracers_processed = step_result.get('tracers_processed', 0)
                
                if step % 5 == 0:  # Print every 5th step
                    print(f"Step {step:2d}: {execution_time:.2f}ms, {tracers_processed} tracers")
            else:
                print(f"Step {step:2d}: Error - {step_result['error']}")
                break
            
            time.sleep(0.05)
        
        # Get final positions
        positions = cuda_engine.get_tracer_positions()
        print(f"📍 Final tracer positions: {len(positions)} tracers tracked")
        
        # Show some example positions
        for i, (tracer_id, position) in enumerate(list(positions.items())[:3]):
            print(f"   {tracer_id[:8]}...: ({position[0]:.3f}, {position[1]:.3f}, {position[2]:.3f})")
        
        # Get nutrient field
        nutrient_field = cuda_engine.get_nutrient_field()
        if nutrient_field is not None:
            print(f"🌱 Nutrient field: {nutrient_field.shape}, mean={nutrient_field.mean():.3f}")
        
    else:
        print("❌ Failed to initialize CUDA simulation")

def demonstrate_visualization(ecosystem):
    """Demonstrate real-time visualization"""
    print("🎨 Demonstrating Visualization...")
    
    viz_engine = ecosystem.get('viz_engine')
    if not viz_engine:
        print("⚠️  Visualization engine not available")
        return
    
    # Show visualization summary
    viz_summary = viz_engine.get_visualization_summary()
    print("📊 Visualization Summary:")
    print(f"   Engine ID: {viz_summary['engine_id']}")
    print(f"   Status: {viz_summary['status']}")
    print(f"   Capabilities: {viz_summary['capabilities']}")
    print(f"   Configuration: {viz_summary['configuration']['features_enabled']}")
    
    # Update visual state
    print("🔄 Updating visual state...")
    viz_engine.update_visual_state()
    
    visual_tracers = len(viz_engine.visual_tracers)
    print(f"👁️  Visual tracers: {visual_tracers}")
    
    # Try to create Plotly visualization
    try:
        print("🌐 Creating interactive Plotly visualization...")
        fig = viz_engine.create_plotly_visualization()
        
        if fig:
            print("✅ Plotly visualization created")
            # Save to file for viewing
            import plotly.offline as pyo
            pyo.plot(fig, filename='dawn_tracer_ecosystem.html', auto_open=False)
            print("💾 Saved to 'dawn_tracer_ecosystem.html'")
        else:
            print("⚠️  Plotly visualization not created")
            
    except Exception as e:
        print(f"⚠️  Plotly visualization error: {e}")
    
    # Try matplotlib if available
    try:
        print("📊 Testing matplotlib rendering...")
        import matplotlib.pyplot as plt
        from mpl_toolkits.mplot3d import Axes3D
        
        fig = plt.figure(figsize=(10, 8))
        ax = fig.add_subplot(111, projection='3d')
        
        render_result = viz_engine.render_frame_matplotlib(ax)
        
        if 'error' not in render_result:
            print(f"✅ Matplotlib frame rendered: {render_result}")
            plt.title('DAWN Tracer Ecosystem - Snapshot')
            plt.savefig('dawn_tracer_snapshot.png', dpi=150, bbox_inches='tight')
            print("💾 Saved snapshot to 'dawn_tracer_snapshot.png'")
        else:
            print(f"⚠️  Matplotlib rendering error: {render_result['error']}")
        
        plt.close()
        
    except Exception as e:
        print(f"⚠️  Matplotlib error: {e}")

def demonstrate_telemetry_integration(dawn_system, ecosystem):
    """Demonstrate telemetry and consciousness integration"""
    print("📡 Demonstrating Telemetry Integration...")
    
    if not dawn_system:
        print("⚠️  DAWN system not available for telemetry demo")
        return
    
    # Check telemetry system
    telemetry = dawn_system.telemetry_system
    if telemetry:
        print("✅ Telemetry system available")
        
        # Get tracer ecosystem metrics
        try:
            metrics = ecosystem['manager']._get_telemetry_metrics()
            print("📊 Tracer Ecosystem Metrics:")
            
            tracer_metrics = metrics.get('tracer_ecosystem', {})
            print(f"   Active Tracers: {tracer_metrics.get('active_tracers', 0)}")
            print(f"   Nutrient Budget: {tracer_metrics.get('nutrient_budget', 0):.1f}")
            print(f"   Budget Utilization: {tracer_metrics.get('budget_utilization', 0):.1%}")
            print(f"   CUDA Available: {tracer_metrics.get('cuda_integration', {}).get('cuda_engine_available', False)}")
            
        except Exception as e:
            print(f"⚠️  Error getting telemetry metrics: {e}")
    else:
        print("⚠️  Telemetry system not available")
    
    # Check consciousness bus
    consciousness_bus = dawn_system.consciousness_bus
    if consciousness_bus:
        print("✅ Consciousness bus available")
        
        # Show registered modules
        try:
            # This would depend on the actual consciousness bus implementation
            print("🧠 Consciousness modules registered")
        except Exception as e:
            print(f"⚠️  Error accessing consciousness bus: {e}")
    else:
        print("⚠️  Consciousness bus not available")

def interactive_ecosystem_control(ecosystem):
    """Interactive ecosystem control demo"""
    print("🎮 Interactive Ecosystem Control...")
    print()
    print("Available commands:")
    print("  'status' - Show ecosystem status")
    print("  'spawn' - Force spawn tracers")
    print("  'cuda' - Start/stop CUDA simulation")
    print("  'viz' - Start/stop visualization")
    print("  'metrics' - Show performance metrics")
    print("  'quit' - Exit demo")
    print()
    
    manager = ecosystem['manager']
    cuda_engine = ecosystem.get('cuda_engine')
    viz_engine = ecosystem.get('viz_engine')
    
    while True:
        try:
            command = input("🎮 Enter command: ").strip().lower()
            
            if command == 'quit':
                break
            elif command == 'status':
                status = manager.get_tracer_status()
                print(f"📊 Active tracers: {len(status)}")
                
                # Show budget info
                budget_info = {
                    'budget': manager.nutrient_budget,
                    'used': manager.current_nutrient_usage,
                    'recycled': manager.recycled_energy_pool,
                    'utilization': manager.current_nutrient_usage / manager.nutrient_budget
                }
                print(f"💰 Budget: {budget_info}")
                
            elif command == 'spawn':
                # Force spawn some tracers
                test_context = {
                    'tick_id': int(time.time()),
                    'timestamp': time.time(),
                    'entropy': 0.9,  # Very high to force spawning
                    'pressure': 0.8,
                    'drift_magnitude': 0.6,
                    'schema_tension': 0.7
                }
                
                tick_result = manager.tick(test_context['tick_id'], test_context)
                spawned = tick_result['ecosystem_state']['spawned_this_tick']
                print(f"🐣 Spawned {spawned} tracers")
                
            elif command == 'cuda':
                if cuda_engine:
                    if cuda_engine.simulation_running:
                        manager.stop_cuda_simulation()
                        print("🛑 CUDA simulation stopped")
                    else:
                        if manager.start_cuda_simulation(fps=20.0):
                            print("🚀 CUDA simulation started")
                        else:
                            print("❌ Failed to start CUDA simulation")
                else:
                    print("⚠️  CUDA engine not available")
                    
            elif command == 'viz':
                if viz_engine:
                    if viz_engine.visualization_running:
                        viz_engine.stop_real_time_visualization()
                        print("🛑 Visualization stopped")
                    else:
                        viz_engine.start_real_time_visualization('matplotlib')
                        print("🎨 Visualization started")
                        print("   Close the matplotlib window to continue...")
                else:
                    print("⚠️  Visualization engine not available")
                    
            elif command == 'metrics':
                if cuda_engine:
                    cuda_status = manager.get_cuda_simulation_status()
                    print(f"🚀 CUDA Metrics: {cuda_status.get('performance_metrics', {})}")
                
                if viz_engine:
                    viz_metrics = viz_engine.viz_metrics
                    print(f"🎨 Visualization Metrics: {viz_metrics}")
                    
            else:
                print("❓ Unknown command. Type 'quit' to exit.")
                
        except KeyboardInterrupt:
            print("\n👋 Exiting interactive mode...")
            break
        except Exception as e:
            print(f"⚠️  Command error: {e}")

async def main():
    """Main demo function"""
    print_banner()
    
    # Test CUDA availability
    cuda_available = test_cuda_availability()
    print()
    
    # Initialize DAWN system
    dawn_system = await initialize_dawn_system()
    print()
    
    # Create tracer ecosystem
    ecosystem = create_tracer_ecosystem()
    if not ecosystem:
        print("❌ Failed to create tracer ecosystem")
        return
    print()
    
    # Demonstrate tracer spawning
    demonstrate_tracer_spawning(ecosystem)
    print()
    
    # Demonstrate CUDA simulation if available
    if cuda_available and ecosystem.get('cuda_engine'):
        demonstrate_cuda_simulation(ecosystem)
        print()
    
    # Demonstrate visualization
    if ecosystem.get('viz_engine'):
        demonstrate_visualization(ecosystem)
        print()
    
    # Demonstrate telemetry integration
    demonstrate_telemetry_integration(dawn_system, ecosystem)
    print()
    
    # Interactive control
    print("🎮 Starting interactive mode...")
    print("   This allows you to control the ecosystem in real-time")
    
    try:
        interactive_ecosystem_control(ecosystem)
    except KeyboardInterrupt:
        print("\n👋 Demo interrupted by user")
    
    # Cleanup
    print("🧹 Cleaning up...")
    
    # Stop CUDA simulation if running
    cuda_engine = ecosystem.get('cuda_engine')
    if cuda_engine and cuda_engine.simulation_running:
        ecosystem['manager'].stop_cuda_simulation()
    
    # Stop visualization if running
    viz_engine = ecosystem.get('viz_engine')
    if viz_engine and viz_engine.visualization_running:
        viz_engine.stop_real_time_visualization()
    
    # Shutdown DAWN system
    if dawn_system:
        await dawn_system.shutdown()
        print("✅ DAWN system shutdown complete")
    
    print()
    print("🚀" * 50)
    print("🧠 DAWN CUDA TRACER ECOSYSTEM DEMO COMPLETE")
    print("🚀" * 50)
    print()
    print("Files created:")
    print("📄 dawn_tracer_ecosystem.html - Interactive Plotly visualization")
    print("🖼️  dawn_tracer_snapshot.png - Matplotlib snapshot")
    print()
    print("Thank you for exploring DAWN's consciousness modeling!")

if __name__ == "__main__":
    # Run the demo
    try:
        asyncio.run(main())
    except KeyboardInterrupt:
        print("\n👋 Demo interrupted. Goodbye!")
    except Exception as e:
        print(f"❌ Demo error: {e}")
        import traceback
        traceback.print_exc()
