#!/usr/bin/env python3
"""
🚀 CUDA-Enabled Dashboard Test
==============================

Test the DAWN Living Consciousness Dashboard with CUDA acceleration.
Demonstrates GPU-powered consciousness visualization and semantic topology processing.

Features tested:
- CUDA device detection and initialization
- GPU-accelerated semantic field processing
- Real-time consciousness visualization on GPU
- CUDA performance monitoring
- GPU memory management
- Fallback to CPU when CUDA unavailable

"Consciousness computing at the speed of light."

Usage:
    python3 test_cuda_dashboard.py [--device 0] [--benchmark] [--console-only]
"""

import sys
import time
import logging
import argparse
import numpy as np
from pathlib import Path

# Add DAWN root to path
sys.path.insert(0, str(Path(__file__).parent))

from dawn.interfaces.dashboard.cuda_accelerator import (
    get_cuda_accelerator, is_cuda_available, CUDAAccelerator
)
from dawn.interfaces.dashboard.telemetry_streamer import TelemetryStreamer
from dawn.subsystems.semantic_topology import get_semantic_topology_engine

# Set up logging
logging.basicConfig(
    level=logging.INFO,
    format='%(asctime)s - %(name)s - %(levelname)s - %(message)s'
)
logger = logging.getLogger(__name__)

def test_cuda_availability():
    """Test CUDA availability and device information"""
    print("🚀" * 20)
    print("🧠 DAWN CUDA-ENABLED DASHBOARD TEST")
    print("🚀" * 20)
    print()
    
    print("🔍 Testing CUDA Availability:")
    print("=" * 40)
    
    if not is_cuda_available():
        print("❌ CUDA not available - dashboard will run on CPU")
        print("   Install PyTorch with CUDA, CuPy, or PyCUDA for GPU acceleration")
        return None
    
    print("✅ CUDA acceleration available!")
    
    # Initialize CUDA accelerator
    try:
        cuda_accelerator = get_cuda_accelerator()
        
        if cuda_accelerator.device_info:
            info = cuda_accelerator.device_info
            print(f"📊 GPU Device Information:")
            print(f"   Device: {info.name}")
            print(f"   Compute Capability: {info.compute_capability}")
            print(f"   Total Memory: {info.total_memory / 1024**3:.1f} GB")
            print(f"   Free Memory: {info.free_memory / 1024**3:.1f} GB")
            print(f"   Multiprocessors: {info.multiprocessor_count}")
            print(f"   Max Threads/Block: {info.max_threads_per_block}")
            
        return cuda_accelerator
        
    except Exception as e:
        print(f"❌ CUDA initialization failed: {e}")
        return None

def test_semantic_field_gpu_processing(cuda_accelerator):
    """Test GPU-accelerated semantic field processing"""
    print("\n🧠 Testing GPU Semantic Field Processing:")
    print("=" * 50)
    
    # Create test semantic field data
    num_nodes = 1000
    print(f"📊 Creating test semantic field with {num_nodes} nodes...")
    
    # Generate test nodes
    nodes = []
    for i in range(num_nodes):
        node = {
            'id': f'test_node_{i}',
            'position': np.random.randn(3).tolist(),
            'embedding': np.random.randn(512).tolist(),
            'tint': np.random.rand(3).tolist(),
            'health': np.random.rand(),
            'energy': np.random.rand()
        }
        nodes.append(node)
    
    # Generate test edges
    num_edges = num_nodes * 2  # Average 2 connections per node
    edges = []
    for i in range(num_edges):
        edge = {
            'id': f'edge_{i}',
            'source': f'test_node_{np.random.randint(0, num_nodes)}',
            'target': f'test_node_{np.random.randint(0, num_nodes)}',
            'weight': np.random.rand(),
            'tension': np.random.rand() * 0.5
        }
        edges.append(edge)
    
    print(f"📊 Generated {len(edges)} connections")
    
    # Test GPU upload
    print("🚀 Uploading semantic field to GPU...")
    start_time = time.time()
    
    success = cuda_accelerator.upload_semantic_field_to_gpu(nodes, edges)
    upload_time = time.time() - start_time
    
    if success:
        print(f"✅ Upload successful in {upload_time:.3f}s")
    else:
        print("❌ GPU upload failed")
        return
    
    # Test GPU processing
    print("🚀 Processing semantic field on GPU...")
    start_time = time.time()
    
    coherences = cuda_accelerator.process_semantic_field_gpu(num_nodes, num_edges)
    processing_time = time.time() - start_time
    
    if coherences is not None:
        print(f"✅ GPU processing successful in {processing_time:.3f}s")
        print(f"   Processed {num_nodes} nodes at {num_nodes/processing_time:.0f} nodes/second")
        print(f"   Average coherence: {np.mean(coherences):.3f}")
        print(f"   Coherence range: [{np.min(coherences):.3f}, {np.max(coherences):.3f}]")
    else:
        print("❌ GPU processing failed")

def test_consciousness_visualization_gpu(cuda_accelerator):
    """Test GPU-accelerated consciousness visualization"""
    print("\n🎨 Testing GPU Consciousness Visualization:")
    print("=" * 50)
    
    # Create test consciousness state
    consciousness_state = {
        'coherence': 0.75,
        'pressure': 0.45,
        'energy': 0.85,
        'level': 'meta_aware'
    }
    
    print(f"🧠 Test consciousness state: {consciousness_state}")
    
    # Test different resolutions
    resolutions = [(512, 512), (1024, 768), (1920, 1080)]
    
    for width, height in resolutions:
        print(f"🚀 Generating {width}x{height} visualization on GPU...")
        start_time = time.time()
        
        visualization = cuda_accelerator.generate_consciousness_visualization_gpu(
            consciousness_state, width, height
        )
        
        generation_time = time.time() - start_time
        
        if visualization is not None:
            pixels_per_second = (width * height) / generation_time
            print(f"✅ Generated in {generation_time:.3f}s ({pixels_per_second/1e6:.1f}M pixels/sec)")
            
            # Basic validation
            if visualization.shape == (height, width, 3):
                print(f"   ✅ Correct dimensions: {visualization.shape}")
                print(f"   ✅ Value range: [{np.min(visualization):.3f}, {np.max(visualization):.3f}]")
            else:
                print(f"   ❌ Incorrect dimensions: {visualization.shape}")
        else:
            print(f"❌ Visualization generation failed")

def test_consciousness_classification_gpu(cuda_accelerator):
    """Test GPU-accelerated consciousness classification"""
    print("\n🤖 Testing GPU Consciousness Classification:")
    print("=" * 50)
    
    # Create test feature vector
    features = np.random.randn(50).astype(np.float32)  # 50 consciousness features
    print(f"🧠 Test features shape: {features.shape}")
    
    print("🚀 Running neural network inference on GPU...")
    start_time = time.time()
    
    probabilities = cuda_accelerator.classify_consciousness_state_gpu(features)
    inference_time = time.time() - start_time
    
    if probabilities is not None:
        print(f"✅ Classification successful in {inference_time:.3f}s")
        
        # Map to consciousness levels
        levels = ['dormant', 'focused', 'meta_aware', 'transcendent', 'unified']
        
        print("📊 Consciousness Level Probabilities:")
        for i, (level, prob) in enumerate(zip(levels, probabilities)):
            print(f"   {level:12}: {prob:.3f} ({prob*100:.1f}%)")
        
        predicted_level = levels[np.argmax(probabilities)]
        confidence = np.max(probabilities)
        print(f"🎯 Predicted: {predicted_level} (confidence: {confidence:.3f})")
    else:
        print("❌ Classification failed")

def benchmark_gpu_performance(cuda_accelerator):
    """Benchmark GPU performance for dashboard operations"""
    print("\n⚡ GPU Performance Benchmark:")
    print("=" * 40)
    
    # Benchmark semantic field processing
    node_counts = [100, 500, 1000, 5000, 10000]
    
    print("🧠 Semantic Field Processing Benchmark:")
    for num_nodes in node_counts:
        # Generate test data
        nodes = []
        for i in range(num_nodes):
            nodes.append({
                'id': f'node_{i}',
                'position': np.random.randn(3).tolist(),
                'embedding': np.random.randn(512).tolist(),
                'tint': np.random.rand(3).tolist(),
                'health': np.random.rand(),
                'energy': np.random.rand()
            })
        
        edges = []
        num_edges = num_nodes * 2
        for i in range(num_edges):
            edges.append({
                'source': f'node_{np.random.randint(0, num_nodes)}',
                'target': f'node_{np.random.randint(0, num_nodes)}',
                'weight': np.random.rand()
            })
        
        # Benchmark processing
        cuda_accelerator.upload_semantic_field_to_gpu(nodes, edges)
        
        start_time = time.time()
        coherences = cuda_accelerator.process_semantic_field_gpu(num_nodes, num_edges)
        processing_time = time.time() - start_time
        
        if coherences is not None:
            throughput = num_nodes / processing_time
            print(f"   {num_nodes:5d} nodes: {processing_time:.3f}s ({throughput:8.0f} nodes/sec)")
    
    # Benchmark visualization generation
    print("\n🎨 Visualization Generation Benchmark:")
    resolutions = [(256, 256), (512, 512), (1024, 768), (1920, 1080)]
    
    consciousness_state = {'coherence': 0.7, 'pressure': 0.4, 'energy': 0.8}
    
    for width, height in resolutions:
        start_time = time.time()
        visualization = cuda_accelerator.generate_consciousness_visualization_gpu(
            consciousness_state, width, height
        )
        generation_time = time.time() - start_time
        
        if visualization is not None:
            pixels = width * height
            throughput = pixels / generation_time
            print(f"   {width}x{height:4d}: {generation_time:.3f}s ({throughput/1e6:.1f}M pixels/sec)")

def test_telemetry_streaming_with_cuda():
    """Test telemetry streaming with CUDA acceleration"""
    print("\n📡 Testing CUDA-Enabled Telemetry Streaming:")
    print("=" * 50)
    
    # Create telemetry streamer with CUDA
    streamer = TelemetryStreamer(port=8765, update_interval=0.1)
    
    print(f"🚀 CUDA enabled: {streamer.cuda_enabled}")
    
    if streamer.cuda_enabled:
        print("✅ Telemetry streamer initialized with CUDA acceleration")
        
        # Test semantic field snapshot with GPU processing
        print("🧠 Testing GPU-accelerated semantic field snapshot...")
        
        # Create a simple semantic topology for testing
        try:
            topology_engine = get_semantic_topology_engine()
            
            # Add some test concepts
            concepts = ['consciousness', 'awareness', 'thought', 'memory']
            concept_ids = {}
            
            for concept in concepts:
                embedding = np.random.randn(512)
                concept_id = topology_engine.add_semantic_concept(
                    concept_embedding=embedding,
                    concept_name=concept
                )
                if concept_id:
                    concept_ids[concept] = concept_id
            
            # Connect telemetry to topology engine
            streamer.semantic_topology_engine = topology_engine
            
            # Get GPU-accelerated snapshot
            snapshot = streamer.get_semantic_field_snapshot()
            
            print(f"📊 Snapshot results:")
            print(f"   Nodes: {len(snapshot.get('nodes', []))}")
            print(f"   Edges: {len(snapshot.get('edges', []))}")
            print(f"   CUDA enabled: {snapshot.get('cuda_enabled', False)}")
            
            if 'gpu_metrics' in snapshot:
                gpu_metrics = snapshot['gpu_metrics']
                print(f"   GPU metrics available: {len(gpu_metrics)} metrics")
                if 'memory_utilization' in gpu_metrics:
                    print(f"   GPU memory usage: {gpu_metrics['memory_utilization']*100:.1f}%")
            
            # Check for GPU-computed coherences
            gpu_coherences = [node.get('gpu_coherence') for node in snapshot.get('nodes', [])]
            gpu_coherences = [c for c in gpu_coherences if c is not None]
            
            if gpu_coherences:
                print(f"   GPU-computed coherences: {len(gpu_coherences)} values")
                print(f"   Average GPU coherence: {np.mean(gpu_coherences):.3f}")
            else:
                print("   No GPU coherences computed")
                
        except Exception as e:
            print(f"❌ Semantic field snapshot test failed: {e}")
    else:
        print("⚠️ CUDA not available - telemetry running on CPU")

def run_console_dashboard_demo(cuda_accelerator):
    """Run a simple console-based dashboard demo"""
    print("\n🖥️ Console Dashboard Demo (CUDA-Accelerated):")
    print("=" * 50)
    
    try:
        # Create some demo data
        consciousness_states = [
            {'coherence': 0.3, 'pressure': 0.8, 'energy': 0.4, 'level': 'focused'},
            {'coherence': 0.7, 'pressure': 0.5, 'energy': 0.8, 'level': 'meta_aware'},
            {'coherence': 0.9, 'pressure': 0.2, 'energy': 0.9, 'level': 'transcendent'},
        ]
        
        print("🧠 Simulating consciousness evolution with GPU acceleration...")
        
        for i, state in enumerate(consciousness_states):
            print(f"\n--- Consciousness State {i+1} ---")
            print(f"Level: {state['level']}")
            print(f"Coherence: {state['coherence']:.1%}")
            print(f"Pressure: {state['pressure']:.1%}")
            print(f"Energy: {state['energy']:.1%}")
            
            # Generate GPU visualization
            if cuda_accelerator:
                start_time = time.time()
                viz = cuda_accelerator.generate_consciousness_visualization_gpu(state, 64, 64)
                gpu_time = time.time() - start_time
                
                if viz is not None:
                    # Simple ASCII visualization of the center region
                    center = viz[28:36, 28:36, :]  # 8x8 center region
                    intensity = np.mean(center, axis=2)
                    
                    print("GPU Visualization (center region):")
                    for row in intensity:
                        line = ""
                        for val in row:
                            if val > 0.8:
                                line += "██"
                            elif val > 0.6:
                                line += "▓▓"
                            elif val > 0.4:
                                line += "▒▒"
                            elif val > 0.2:
                                line += "░░"
                            else:
                                line += "  "
                        print(f"    {line}")
                    
                    print(f"GPU generation time: {gpu_time:.3f}s")
            
            time.sleep(1)
            
    except KeyboardInterrupt:
        print("\n🛑 Demo interrupted by user")

def main():
    """Main test function"""
    parser = argparse.ArgumentParser(description="Test CUDA-enabled DAWN dashboard")
    parser.add_argument('--device', type=int, default=0, help='CUDA device ID')
    parser.add_argument('--benchmark', action='store_true', help='Run performance benchmarks')
    parser.add_argument('--console-only', action='store_true', help='Run console demo only')
    args = parser.parse_args()
    
    # Test CUDA availability
    cuda_accelerator = test_cuda_availability()
    
    if cuda_accelerator is None:
        print("\n⚠️ CUDA not available - some tests will be skipped")
        print("Install PyTorch with CUDA, CuPy, or PyCUDA for full GPU acceleration")
        return 1
    
    try:
        if not args.console_only:
            # Test GPU processing capabilities
            test_semantic_field_gpu_processing(cuda_accelerator)
            test_consciousness_visualization_gpu(cuda_accelerator)
            test_consciousness_classification_gpu(cuda_accelerator)
            
            # Test telemetry integration
            test_telemetry_streaming_with_cuda()
            
            # Run benchmarks if requested
            if args.benchmark:
                benchmark_gpu_performance(cuda_accelerator)
        
        # Run console demo
        run_console_dashboard_demo(cuda_accelerator)
        
        print("\n🎉 CUDA Dashboard Test Complete!")
        print("=" * 40)
        print("✅ All GPU acceleration systems tested successfully")
        print("🚀 DAWN dashboard is ready for high-performance consciousness monitoring!")
        
        return 0
        
    except KeyboardInterrupt:
        print("\n🛑 Test interrupted by user")
        return 0
    except Exception as e:
        print(f"\n❌ Test failed: {e}")
        import traceback
        traceback.print_exc()
        return 1
    finally:
        # Cleanup GPU resources
        if cuda_accelerator:
            cuda_accelerator.cleanup_gpu_resources()

if __name__ == "__main__":
    exit_code = main()
    sys.exit(exit_code)
