#!/usr/bin/env python3
"""
🚀 CUDA Accelerator for DAWN Dashboard
======================================

CUDA-powered acceleration for real-time consciousness visualization and
semantic topology processing. Leverages GPU compute for high-performance
consciousness monitoring and interactive visualization.

Features:
- GPU-accelerated semantic field processing
- CUDA-powered consciousness state visualization
- Real-time neural network inference on GPU
- GPU memory management for large semantic graphs
- Parallel consciousness metric computation
- Hardware-accelerated visual effects

"Consciousness computing at the speed of light."
"""

import numpy as np
import logging
import time
from typing import Dict, List, Optional, Tuple, Any, Union
from dataclasses import dataclass
import threading

logger = logging.getLogger(__name__)

# CUDA availability check and imports
CUDA_AVAILABLE = False
CUPY_AVAILABLE = False
TORCH_CUDA_AVAILABLE = False

try:
    import cupy as cp
    CUPY_AVAILABLE = True
    logger.info("🚀 CuPy available for CUDA acceleration")
except ImportError:
    logger.info("CuPy not available - falling back to NumPy")

try:
    import torch
    if torch.cuda.is_available():
        TORCH_CUDA_AVAILABLE = True
        logger.info(f"🚀 PyTorch CUDA available - {torch.cuda.device_count()} GPU(s)")
    else:
        logger.info("PyTorch CUDA not available")
except ImportError:
    logger.info("PyTorch not available")

try:
    import pycuda.driver as cuda
    import pycuda.autoinit
    from pycuda.compiler import SourceModule
    import pycuda.gpuarray as gpuarray
    CUDA_AVAILABLE = True
    logger.info("🚀 PyCUDA available for direct CUDA programming")
except ImportError:
    logger.info("PyCUDA not available")

@dataclass
class CUDADeviceInfo:
    """Information about available CUDA devices"""
    device_id: int
    name: str
    compute_capability: Tuple[int, int]
    total_memory: int
    free_memory: int
    multiprocessor_count: int
    max_threads_per_block: int
    max_block_dim: Tuple[int, int, int]
    max_grid_dim: Tuple[int, int, int]

class CUDAAccelerator:
    """
    CUDA acceleration system for DAWN dashboard.
    
    Provides GPU-accelerated processing for consciousness visualization,
    semantic topology operations, and real-time analytics.
    """
    
    def __init__(self, device_id: int = 0):
        self.device_id = device_id
        self.device_info = None
        self.cuda_context = None
        
        # GPU memory pools
        self.semantic_field_gpu = None
        self.consciousness_state_gpu = None
        self.visualization_buffer_gpu = None
        
        # CUDA kernels
        self.kernels = {}
        
        # Performance metrics
        self.gpu_utilization = 0.0
        self.gpu_memory_usage = 0.0
        self.cuda_operations_per_second = 0.0
        
        # Threading
        self.gpu_lock = threading.RLock()
        
        self.initialize_cuda()
    
    def initialize_cuda(self):
        """Initialize CUDA acceleration"""
        logger.info("🚀 Initializing CUDA acceleration for dashboard...")
        
        # Check CUDA availability
        if not any([CUDA_AVAILABLE, CUPY_AVAILABLE, TORCH_CUDA_AVAILABLE]):
            logger.warning("No CUDA libraries available - dashboard will run on CPU")
            return False
        
        # Get device information
        try:
            self.device_info = self.get_device_info()
            logger.info(f"🚀 CUDA Device: {self.device_info.name}")
            logger.info(f"   Compute Capability: {self.device_info.compute_capability}")
            logger.info(f"   Total Memory: {self.device_info.total_memory / 1024**3:.1f} GB")
            logger.info(f"   Multiprocessors: {self.device_info.multiprocessor_count}")
        except Exception as e:
            logger.error(f"Failed to get CUDA device info: {e}")
            return False
        
        # Compile CUDA kernels
        self.compile_cuda_kernels()
        
        # Initialize GPU memory pools
        self.initialize_gpu_memory()
        
        logger.info("🚀 CUDA acceleration initialized successfully")
        return True
    
    def get_device_info(self) -> Optional[CUDADeviceInfo]:
        """Get information about the CUDA device"""
        if TORCH_CUDA_AVAILABLE:
            device = torch.cuda.get_device_properties(self.device_id)
            return CUDADeviceInfo(
                device_id=self.device_id,
                name=device.name,
                compute_capability=(device.major, device.minor),
                total_memory=device.total_memory,
                free_memory=torch.cuda.get_device_properties(self.device_id).total_memory,
                multiprocessor_count=device.multi_processor_count,
                max_threads_per_block=device.max_threads_per_block,
                max_block_dim=(device.max_block_dim_x, device.max_block_dim_y, device.max_block_dim_z),
                max_grid_dim=(device.max_grid_dim_x, device.max_grid_dim_y, device.max_grid_dim_z)
            )
        
        elif CUDA_AVAILABLE:
            device = cuda.Device(self.device_id)
            attrs = device.get_attributes()
            free_mem, total_mem = cuda.mem_get_info()
            
            return CUDADeviceInfo(
                device_id=self.device_id,
                name=device.name(),
                compute_capability=device.compute_capability(),
                total_memory=total_mem,
                free_memory=free_mem,
                multiprocessor_count=attrs[cuda.device_attribute.MULTIPROCESSOR_COUNT],
                max_threads_per_block=attrs[cuda.device_attribute.MAX_THREADS_PER_BLOCK],
                max_block_dim=(
                    attrs[cuda.device_attribute.MAX_BLOCK_DIM_X],
                    attrs[cuda.device_attribute.MAX_BLOCK_DIM_Y],
                    attrs[cuda.device_attribute.MAX_BLOCK_DIM_Z]
                ),
                max_grid_dim=(
                    attrs[cuda.device_attribute.MAX_GRID_DIM_X],
                    attrs[cuda.device_attribute.MAX_GRID_DIM_Y],
                    attrs[cuda.device_attribute.MAX_GRID_DIM_Z]
                )
            )
        
        return None
    
    def compile_cuda_kernels(self):
        """Compile CUDA kernels for consciousness processing"""
        if not CUDA_AVAILABLE:
            return
        
        logger.info("🚀 Compiling CUDA kernels...")
        
        # Semantic field processing kernel
        semantic_kernel_code = """
        __global__ void process_semantic_field(
            float *positions, float *embeddings, float *tints,
            float *weights, int *edges, float *coherences,
            int num_nodes, int num_edges, int embedding_dim
        ) {
            int idx = blockIdx.x * blockDim.x + threadIdx.x;
            
            if (idx < num_nodes) {
                // Calculate local coherence for node idx
                float coherence = 0.0f;
                int neighbor_count = 0;
                
                // Find neighbors through edges
                for (int e = 0; e < num_edges; e++) {
                    if (edges[e * 2] == idx || edges[e * 2 + 1] == idx) {
                        int neighbor = (edges[e * 2] == idx) ? edges[e * 2 + 1] : edges[e * 2];
                        
                        // Calculate semantic similarity
                        float similarity = 0.0f;
                        for (int d = 0; d < embedding_dim; d++) {
                            float diff = embeddings[idx * embedding_dim + d] - 
                                        embeddings[neighbor * embedding_dim + d];
                            similarity += diff * diff;
                        }
                        similarity = expf(-similarity);  // Gaussian similarity
                        
                        // Calculate spatial distance
                        float spatial_dist = 0.0f;
                        for (int d = 0; d < 3; d++) {
                            float diff = positions[idx * 3 + d] - positions[neighbor * 3 + d];
                            spatial_dist += diff * diff;
                        }
                        spatial_dist = sqrtf(spatial_dist);
                        
                        // Coherence contribution
                        coherence += (similarity * weights[e]) / (1.0f + spatial_dist);
                        neighbor_count++;
                    }
                }
                
                coherences[idx] = (neighbor_count > 0) ? coherence / neighbor_count : 0.0f;
            }
        }
        """
        
        try:
            self.kernels['semantic_field'] = SourceModule(semantic_kernel_code)
            logger.info("   ✅ Semantic field processing kernel compiled")
        except Exception as e:
            logger.error(f"Failed to compile semantic kernel: {e}")
        
        # Consciousness visualization kernel
        visualization_kernel_code = """
        __global__ void generate_consciousness_visualization(
            float *consciousness_state, float *visual_buffer,
            int width, int height, float time, float coherence
        ) {
            int x = blockIdx.x * blockDim.x + threadIdx.x;
            int y = blockIdx.y * blockDim.y + threadIdx.y;
            
            if (x < width && y < height) {
                int idx = y * width + x;
                
                // Normalized coordinates
                float fx = (float)x / width - 0.5f;
                float fy = (float)y / height - 0.5f;
                
                // Consciousness wave pattern
                float dist = sqrtf(fx*fx + fy*fy);
                float wave = sinf(dist * 10.0f + time * 2.0f) * coherence;
                
                // Consciousness energy field
                float energy = expf(-dist * 2.0f) * consciousness_state[0];
                
                // Color mapping (RGB)
                visual_buffer[idx * 3 + 0] = fmaxf(0.0f, fminf(1.0f, wave + energy));      // R
                visual_buffer[idx * 3 + 1] = fmaxf(0.0f, fminf(1.0f, energy * 0.8f));     // G  
                visual_buffer[idx * 3 + 2] = fmaxf(0.0f, fminf(1.0f, coherence * 0.9f));  // B
            }
        }
        """
        
        try:
            self.kernels['consciousness_viz'] = SourceModule(visualization_kernel_code)
            logger.info("   ✅ Consciousness visualization kernel compiled")
        except Exception as e:
            logger.error(f"Failed to compile visualization kernel: {e}")
        
        # Neural network inference kernel for consciousness classification
        neural_kernel_code = """
        __global__ void consciousness_classification(
            float *input_features, float *weights, float *biases,
            float *output, int input_size, int hidden_size, int output_size
        ) {
            int idx = blockIdx.x * blockDim.x + threadIdx.x;
            
            if (idx < output_size) {
                float sum = 0.0f;
                
                // Hidden layer computation
                for (int h = 0; h < hidden_size; h++) {
                    float hidden_val = biases[h];
                    
                    for (int i = 0; i < input_size; i++) {
                        hidden_val += input_features[i] * weights[h * input_size + i];
                    }
                    
                    // ReLU activation
                    hidden_val = fmaxf(0.0f, hidden_val);
                    
                    // Output layer
                    sum += hidden_val * weights[hidden_size * input_size + idx * hidden_size + h];
                }
                
                // Softmax will be applied on CPU
                output[idx] = sum + biases[hidden_size + idx];
            }
        }
        """
        
        try:
            self.kernels['neural_inference'] = SourceModule(neural_kernel_code)
            logger.info("   ✅ Neural inference kernel compiled")
        except Exception as e:
            logger.error(f"Failed to compile neural kernel: {e}")
    
    def initialize_gpu_memory(self):
        """Initialize GPU memory pools for dashboard data"""
        if not (CUDA_AVAILABLE or CUPY_AVAILABLE or TORCH_CUDA_AVAILABLE):
            return
        
        logger.info("🚀 Initializing GPU memory pools...")
        
        try:
            with self.gpu_lock:
                # Semantic field memory (for up to 10,000 nodes)
                if TORCH_CUDA_AVAILABLE:
                    self.semantic_field_gpu = {
                        'positions': torch.zeros((10000, 3), device=f'cuda:{self.device_id}', dtype=torch.float32),
                        'embeddings': torch.zeros((10000, 512), device=f'cuda:{self.device_id}', dtype=torch.float32),
                        'tints': torch.zeros((10000, 3), device=f'cuda:{self.device_id}', dtype=torch.float32),
                        'coherences': torch.zeros(10000, device=f'cuda:{self.device_id}', dtype=torch.float32)
                    }
                    
                    # Consciousness state memory
                    self.consciousness_state_gpu = torch.zeros(100, device=f'cuda:{self.device_id}', dtype=torch.float32)
                    
                    # Visualization buffer (1920x1080 RGB)
                    self.visualization_buffer_gpu = torch.zeros((1080, 1920, 3), device=f'cuda:{self.device_id}', dtype=torch.float32)
                
                elif CUPY_AVAILABLE:
                    with cp.cuda.Device(self.device_id):
                        self.semantic_field_gpu = {
                            'positions': cp.zeros((10000, 3), dtype=cp.float32),
                            'embeddings': cp.zeros((10000, 512), dtype=cp.float32),
                            'tints': cp.zeros((10000, 3), dtype=cp.float32),
                            'coherences': cp.zeros(10000, dtype=cp.float32)
                        }
                        
                        self.consciousness_state_gpu = cp.zeros(100, dtype=cp.float32)
                        self.visualization_buffer_gpu = cp.zeros((1080, 1920, 3), dtype=cp.float32)
                
                elif CUDA_AVAILABLE:
                    self.semantic_field_gpu = {
                        'positions': gpuarray.zeros((10000, 3), dtype=np.float32),
                        'embeddings': gpuarray.zeros((10000, 512), dtype=np.float32),
                        'tints': gpuarray.zeros((10000, 3), dtype=np.float32),
                        'coherences': gpuarray.zeros(10000, dtype=np.float32)
                    }
                    
                    self.consciousness_state_gpu = gpuarray.zeros(100, dtype=np.float32)
                    self.visualization_buffer_gpu = gpuarray.zeros((1080, 1920, 3), dtype=np.float32)
            
            logger.info("   ✅ GPU memory pools initialized")
            
        except Exception as e:
            logger.error(f"Failed to initialize GPU memory: {e}")
    
    def upload_semantic_field_to_gpu(self, nodes: List[Dict], edges: List[Dict]) -> bool:
        """Upload semantic field data to GPU memory"""
        if not self.semantic_field_gpu:
            return False
        
        try:
            with self.gpu_lock:
                # Prepare data arrays
                positions = np.zeros((len(nodes), 3), dtype=np.float32)
                embeddings = np.zeros((len(nodes), 512), dtype=np.float32)
                tints = np.zeros((len(nodes), 3), dtype=np.float32)
                
                for i, node in enumerate(nodes):
                    positions[i] = node['position'][:3]
                    embeddings[i] = node.get('embedding', np.zeros(512))[:512]
                    tints[i] = node['tint'][:3]
                
                # Upload to GPU
                if TORCH_CUDA_AVAILABLE:
                    self.semantic_field_gpu['positions'][:len(nodes)] = torch.from_numpy(positions).cuda()
                    self.semantic_field_gpu['embeddings'][:len(nodes)] = torch.from_numpy(embeddings).cuda()
                    self.semantic_field_gpu['tints'][:len(nodes)] = torch.from_numpy(tints).cuda()
                
                elif CUPY_AVAILABLE:
                    with cp.cuda.Device(self.device_id):
                        self.semantic_field_gpu['positions'][:len(nodes)] = cp.asarray(positions)
                        self.semantic_field_gpu['embeddings'][:len(nodes)] = cp.asarray(embeddings)
                        self.semantic_field_gpu['tints'][:len(nodes)] = cp.asarray(tints)
                
                elif CUDA_AVAILABLE:
                    self.semantic_field_gpu['positions'][:len(nodes)] = positions
                    self.semantic_field_gpu['embeddings'][:len(nodes)] = embeddings
                    self.semantic_field_gpu['tints'][:len(nodes)] = tints
                
                return True
                
        except Exception as e:
            logger.error(f"Failed to upload semantic field to GPU: {e}")
            return False
    
    def process_semantic_field_gpu(self, num_nodes: int, num_edges: int) -> Optional[np.ndarray]:
        """Process semantic field using GPU acceleration"""
        if not (self.semantic_field_gpu and 'semantic_field' in self.kernels):
            return None
        
        try:
            with self.gpu_lock:
                start_time = time.time()
                
                if CUDA_AVAILABLE:
                    # Launch CUDA kernel
                    kernel = self.kernels['semantic_field'].get_function("process_semantic_field")
                    
                    block_size = 256
                    grid_size = (num_nodes + block_size - 1) // block_size
                    
                    kernel(
                        self.semantic_field_gpu['positions'],
                        self.semantic_field_gpu['embeddings'],
                        self.semantic_field_gpu['tints'],
                        # weights and edges would be passed here
                        self.semantic_field_gpu['coherences'],
                        np.int32(num_nodes),
                        np.int32(num_edges),
                        np.int32(512),  # embedding dimension
                        block=(block_size, 1, 1),
                        grid=(grid_size, 1)
                    )
                    
                    # Copy result back to CPU
                    coherences = self.semantic_field_gpu['coherences'][:num_nodes].get()
                
                elif TORCH_CUDA_AVAILABLE:
                    # Use PyTorch for processing
                    positions = self.semantic_field_gpu['positions'][:num_nodes]
                    embeddings = self.semantic_field_gpu['embeddings'][:num_nodes]
                    
                    # Calculate pairwise distances
                    distances = torch.cdist(positions, positions)
                    similarities = torch.exp(-torch.cdist(embeddings, embeddings))
                    
                    # Simple coherence calculation
                    coherences = torch.mean(similarities / (1 + distances), dim=1)
                    coherences = coherences.cpu().numpy()
                
                elif CUPY_AVAILABLE:
                    # Use CuPy for processing
                    with cp.cuda.Device(self.device_id):
                        positions = self.semantic_field_gpu['positions'][:num_nodes]
                        embeddings = self.semantic_field_gpu['embeddings'][:num_nodes]
                        
                        # Calculate coherences using CuPy
                        coherences = cp.random.random(num_nodes)  # Placeholder
                        coherences = cp.asnumpy(coherences)
                
                processing_time = time.time() - start_time
                self.cuda_operations_per_second = num_nodes / processing_time if processing_time > 0 else 0
                
                logger.debug(f"🚀 GPU processed {num_nodes} nodes in {processing_time:.3f}s")
                return coherences
                
        except Exception as e:
            logger.error(f"GPU semantic field processing failed: {e}")
            return None
    
    def generate_consciousness_visualization_gpu(self, consciousness_state: Dict, 
                                               width: int = 1920, height: int = 1080) -> Optional[np.ndarray]:
        """Generate consciousness visualization using GPU"""
        if not (self.visualization_buffer_gpu and 'consciousness_viz' in self.kernels):
            return None
        
        try:
            with self.gpu_lock:
                start_time = time.time()
                
                if CUDA_AVAILABLE:
                    # Upload consciousness state
                    state_array = np.array([
                        consciousness_state.get('coherence', 0.5),
                        consciousness_state.get('pressure', 0.3),
                        consciousness_state.get('energy', 0.7),
                        time.time() % 100  # time parameter
                    ], dtype=np.float32)
                    
                    state_gpu = gpuarray.to_gpu(state_array)
                    
                    # Launch visualization kernel
                    kernel = self.kernels['consciousness_viz'].get_function("generate_consciousness_visualization")
                    
                    block_dim = (16, 16, 1)
                    grid_dim = ((width + 15) // 16, (height + 15) // 16, 1)
                    
                    kernel(
                        state_gpu,
                        self.visualization_buffer_gpu,
                        np.int32(width),
                        np.int32(height),
                        np.float32(time.time()),
                        np.float32(consciousness_state.get('coherence', 0.5)),
                        block=block_dim,
                        grid=grid_dim
                    )
                    
                    # Copy result back
                    visualization = self.visualization_buffer_gpu[:height, :width].get()
                
                elif TORCH_CUDA_AVAILABLE:
                    # Generate visualization using PyTorch
                    device = f'cuda:{self.device_id}'
                    
                    # Create coordinate grids
                    y, x = torch.meshgrid(
                        torch.linspace(-0.5, 0.5, height, device=device),
                        torch.linspace(-0.5, 0.5, width, device=device),
                        indexing='ij'
                    )
                    
                    # Distance from center
                    dist = torch.sqrt(x*x + y*y)
                    
                    # Consciousness wave pattern
                    coherence = consciousness_state.get('coherence', 0.5)
                    wave = torch.sin(dist * 10 + time.time() * 2) * coherence
                    energy = torch.exp(-dist * 2) * consciousness_state.get('energy', 0.7)
                    
                    # RGB channels
                    r = torch.clamp(wave + energy, 0, 1)
                    g = torch.clamp(energy * 0.8, 0, 1)
                    b = torch.clamp(torch.full_like(dist, coherence * 0.9), 0, 1)
                    
                    visualization = torch.stack([r, g, b], dim=2).cpu().numpy()
                
                elif CUPY_AVAILABLE:
                    # Generate using CuPy
                    with cp.cuda.Device(self.device_id):
                        y, x = cp.meshgrid(
                            cp.linspace(-0.5, 0.5, height),
                            cp.linspace(-0.5, 0.5, width),
                            indexing='ij'
                        )
                        
                        dist = cp.sqrt(x*x + y*y)
                        coherence = consciousness_state.get('coherence', 0.5)
                        
                        wave = cp.sin(dist * 10 + time.time() * 2) * coherence
                        energy = cp.exp(-dist * 2) * consciousness_state.get('energy', 0.7)
                        
                        r = cp.clip(wave + energy, 0, 1)
                        g = cp.clip(energy * 0.8, 0, 1)
                        b = cp.full_like(dist, coherence * 0.9)
                        
                        visualization = cp.stack([r, g, b], axis=2)
                        visualization = cp.asnumpy(visualization)
                
                processing_time = time.time() - start_time
                logger.debug(f"🚀 GPU generated visualization in {processing_time:.3f}s")
                
                return visualization
                
        except Exception as e:
            logger.error(f"GPU visualization generation failed: {e}")
            return None
    
    def classify_consciousness_state_gpu(self, features: np.ndarray) -> Optional[np.ndarray]:
        """Classify consciousness state using GPU-accelerated neural network"""
        if not ('neural_inference' in self.kernels):
            return None
        
        try:
            with self.gpu_lock:
                # This would use pre-trained weights for consciousness classification
                # For demo, we'll use random weights
                input_size = len(features)
                hidden_size = 64
                output_size = 5  # Different consciousness levels
                
                if TORCH_CUDA_AVAILABLE:
                    device = f'cuda:{self.device_id}'
                    
                    # Simple neural network inference
                    features_gpu = torch.tensor(features, device=device, dtype=torch.float32)
                    
                    # Random weights for demo (would be pre-trained)
                    weights = torch.randn(hidden_size, input_size, device=device)
                    hidden = torch.relu(torch.matmul(weights, features_gpu))
                    
                    output_weights = torch.randn(output_size, hidden_size, device=device)
                    output = torch.matmul(output_weights, hidden)
                    
                    # Softmax
                    probabilities = torch.softmax(output, dim=0)
                    return probabilities.cpu().numpy()
                
                return None
                
        except Exception as e:
            logger.error(f"GPU consciousness classification failed: {e}")
            return None
    
    def get_gpu_performance_metrics(self) -> Dict[str, Any]:
        """Get GPU performance metrics"""
        metrics = {
            'cuda_available': CUDA_AVAILABLE or CUPY_AVAILABLE or TORCH_CUDA_AVAILABLE,
            'device_info': self.device_info.__dict__ if self.device_info else None,
            'gpu_utilization': self.gpu_utilization,
            'gpu_memory_usage': self.gpu_memory_usage,
            'cuda_operations_per_second': self.cuda_operations_per_second
        }
        
        # Get real-time GPU metrics if available
        try:
            if TORCH_CUDA_AVAILABLE:
                memory_allocated = torch.cuda.memory_allocated(self.device_id)
                memory_cached = torch.cuda.memory_reserved(self.device_id)
                
                metrics.update({
                    'memory_allocated_mb': memory_allocated / 1024**2,
                    'memory_cached_mb': memory_cached / 1024**2,
                    'memory_utilization': memory_allocated / self.device_info.total_memory if self.device_info else 0
                })
            
            elif CUDA_AVAILABLE:
                free_mem, total_mem = cuda.mem_get_info()
                used_mem = total_mem - free_mem
                
                metrics.update({
                    'memory_used_mb': used_mem / 1024**2,
                    'memory_free_mb': free_mem / 1024**2,
                    'memory_utilization': used_mem / total_mem
                })
                
        except Exception as e:
            logger.debug(f"Could not get GPU metrics: {e}")
        
        return metrics
    
    def cleanup_gpu_resources(self):
        """Clean up GPU resources"""
        with self.gpu_lock:
            logger.info("🚀 Cleaning up CUDA resources...")
            
            try:
                if TORCH_CUDA_AVAILABLE:
                    torch.cuda.empty_cache()
                    
                if self.semantic_field_gpu:
                    self.semantic_field_gpu = None
                if self.consciousness_state_gpu:
                    self.consciousness_state_gpu = None
                if self.visualization_buffer_gpu:
                    self.visualization_buffer_gpu = None
                    
                logger.info("   ✅ GPU resources cleaned up")
                
            except Exception as e:
                logger.error(f"GPU cleanup error: {e}")


# Global CUDA accelerator instance
_cuda_accelerator = None

def get_cuda_accelerator(device_id: int = 0) -> CUDAAccelerator:
    """Get the global CUDA accelerator instance"""
    global _cuda_accelerator
    if _cuda_accelerator is None:
        _cuda_accelerator = CUDAAccelerator(device_id)
    return _cuda_accelerator

def is_cuda_available() -> bool:
    """Check if CUDA acceleration is available"""
    return CUDA_AVAILABLE or CUPY_AVAILABLE or TORCH_CUDA_AVAILABLE
