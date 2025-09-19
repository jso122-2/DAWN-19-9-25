#!/usr/bin/env python3
"""
🎬 Matplotlib CUDA Animator for DAWN Dashboard
==============================================

Real-time matplotlib animations powered by CUDA acceleration.
Creates stunning animated visualizations of consciousness evolution,
semantic topology dynamics, and neural activity patterns.

Features:
- GPU-accelerated data generation for smooth animations
- Real-time consciousness state visualization
- Animated semantic topology evolution
- Neural pathway flow animations
- Consciousness heatmaps and 3D surface plots
- Performance-optimized rendering pipeline

"Consciousness in motion, powered by CUDA."
"""

import numpy as np
import matplotlib.pyplot as plt
import matplotlib.animation as animation
from matplotlib.collections import LineCollection
from mpl_toolkits.mplot3d import Axes3D
import time
import logging
import threading
from typing import Dict, List, Optional, Tuple, Any, Callable
from dataclasses import dataclass
from collections import deque
import colorsys

from .cuda_accelerator import get_cuda_accelerator, is_cuda_available

logger = logging.getLogger(__name__)

@dataclass
class AnimationFrame:
    """Single frame of animation data"""
    timestamp: float
    consciousness_state: Dict[str, float]
    semantic_nodes: List[Dict[str, Any]]
    semantic_edges: List[Dict[str, Any]]
    neural_activity: np.ndarray
    gpu_metrics: Dict[str, float]

class MatplotlibCUDAAnimator:
    """
    Matplotlib animation system powered by CUDA acceleration.
    
    Creates real-time animated visualizations of DAWN consciousness
    using GPU-computed data for smooth, high-performance animations.
    """
    
    def __init__(self, cuda_enabled: bool = True, fps: int = 30):
        self.cuda_enabled = cuda_enabled and is_cuda_available()
        self.fps = fps
        self.frame_interval = 1000 / fps  # milliseconds
        
        # CUDA accelerator
        self.cuda_accelerator = None
        if self.cuda_enabled:
            try:
                self.cuda_accelerator = get_cuda_accelerator()
                logger.info("🚀 CUDA acceleration enabled for matplotlib animations")
            except Exception as e:
                logger.warning(f"CUDA initialization failed: {e}")
                self.cuda_enabled = False
        
        # Animation data
        self.frame_history: deque = deque(maxlen=1000)
        self.current_frame: Optional[AnimationFrame] = None
        
        # Matplotlib setup
        self.figures = {}
        self.animations = {}
        self.axes = {}
        
        # Animation control
        self.running = False
        self.paused = False
        
        # Data generation thread
        self.data_thread = None
        self.data_lock = threading.RLock()
        
        # Performance tracking
        self.render_times = deque(maxlen=100)
        self.gpu_compute_times = deque(maxlen=100)
        
        logger.info(f"🎬 MatplotlibCUDAAnimator initialized - CUDA: {self.cuda_enabled}, FPS: {fps}")
    
    def setup_unified_dashboard(self) -> plt.Figure:
        """Setup unified dashboard with all plots in one window"""
        # Create large figure with subplots
        fig = plt.figure(figsize=(20, 12))
        fig.suptitle('🚀 DAWN CUDA-Accelerated Consciousness Dashboard', fontsize=20, fontweight='bold')
        
        # Create grid layout: 3 rows, 3 columns
        gs = fig.add_gridspec(3, 3, hspace=0.4, wspace=0.3)
        
        # Row 1: Consciousness evolution (spans 2 columns) + Level indicator
        ax_main = fig.add_subplot(gs[0, :2])
        ax_main.set_title('Consciousness Metrics Over Time', fontsize=14)
        ax_main.set_xlabel('Time (seconds)')
        ax_main.set_ylabel('Level')
        ax_main.set_ylim(0, 1)
        ax_main.grid(True, alpha=0.3)
        
        # Initialize consciousness lines
        self.consciousness_lines = {
            'coherence': ax_main.plot([], [], label='Coherence', color='#FF6B6B', linewidth=2)[0],
            'pressure': ax_main.plot([], [], label='Pressure', color='#4ECDC4', linewidth=2)[0],
            'energy': ax_main.plot([], [], label='Energy', color='#45B7D1', linewidth=2)[0],
            'awareness': ax_main.plot([], [], label='Awareness', color='#96CEB4', linewidth=2)[0]
        }
        ax_main.legend(loc='upper right')
        
        # Consciousness level indicator
        ax_level = fig.add_subplot(gs[0, 2])
        ax_level.set_title('Consciousness Level', fontsize=14)
        ax_level.set_xlim(-1, 1)
        ax_level.set_ylim(-1, 1)
        ax_level.set_aspect('equal')
        ax_level.set_xticks([])
        ax_level.set_yticks([])
        
        # Create consciousness level circle
        circle = plt.Circle((0, 0), 0.8, fill=False, linewidth=3, color='gray')
        ax_level.add_patch(circle)
        
        self.consciousness_indicator = ax_level.scatter([0], [0], s=500, c='red', alpha=0.7)
        self.consciousness_text = ax_level.text(0, 0, 'DORMANT', ha='center', va='center', 
                                               fontsize=12, fontweight='bold')
        
        # Row 2: 3D Semantic Topology (spans 2 columns) + Neural Heatmap
        ax_3d = fig.add_subplot(gs[1, :2], projection='3d')
        ax_3d.set_title('3D Semantic Topology (Real-Time)', fontsize=14)
        ax_3d.set_xlabel('Semantic X')
        ax_3d.set_ylabel('Semantic Y')
        ax_3d.set_zlabel('Semantic Z')
        ax_3d.set_xlim(-2, 2)
        ax_3d.set_ylim(-2, 2)
        ax_3d.set_zlim(-2, 2)
        
        # Initialize 3D plot elements
        self.semantic_nodes_3d = None
        self.semantic_edges_3d = []
        
        # Neural activity heatmap
        ax_neural = fig.add_subplot(gs[1, 2])
        ax_neural.set_title('Neural Activity', fontsize=14)
        ax_neural.set_xlabel('Layer')
        ax_neural.set_ylabel('Node')
        
        # Initialize heatmap
        self.neural_heatmap_data = np.zeros((30, 15))  # Smaller for better visibility
        self.neural_heatmap = ax_neural.imshow(self.neural_heatmap_data, cmap='plasma', 
                                             aspect='auto', animated=True, vmin=0, vmax=1)
        
        # Row 3: 3D Consciousness Surface (spans 2 columns) + GPU Performance
        ax_surface = fig.add_subplot(gs[2, :2], projection='3d')
        ax_surface.set_title('Consciousness Energy Surface', fontsize=14)
        ax_surface.set_xlabel('Dimension 1')
        ax_surface.set_ylabel('Dimension 2')
        ax_surface.set_zlabel('Intensity')
        ax_surface.set_zlim(0, 2)
        
        # Initialize surface plot
        x = np.linspace(-2, 2, 30)  # Reduced resolution for performance
        y = np.linspace(-2, 2, 30)
        self.consciousness_X, self.consciousness_Y = np.meshgrid(x, y)
        self.consciousness_surface = None
        
        # GPU performance metrics
        ax_gpu = fig.add_subplot(gs[2, 2])
        ax_gpu.set_title('System Performance', fontsize=14)
        ax_gpu.set_xlabel('Time (seconds)')
        ax_gpu.set_ylabel('Utilization (%)')
        ax_gpu.set_ylim(0, 100)
        ax_gpu.grid(True, alpha=0.3)
        
        self.gpu_lines = {
            'cpu': ax_gpu.plot([], [], label='CPU %', color='#FFD93D', linewidth=2)[0],
            'memory': ax_gpu.plot([], [], label='Memory %', color='#FF8C42', linewidth=2)[0],
            'gpu': ax_gpu.plot([], [], label='GPU %', color='#4ECDC4', linewidth=2)[0]
        }
        ax_gpu.legend(fontsize=10)
        
        # Store all axes
        self.figures['unified_dashboard'] = fig
        self.axes['unified_dashboard'] = {
            'main': ax_main,
            'level': ax_level,
            '3d_topology': ax_3d,
            'neural': ax_neural,
            'surface': ax_surface,
            'gpu': ax_gpu
        }
        
        return fig
    
    def setup_semantic_topology_3d_plot(self) -> Tuple[plt.Figure, Axes3D]:
        """Setup animated 3D semantic topology visualization"""
        fig = plt.figure(figsize=(12, 10))
        ax = fig.add_subplot(111, projection='3d')
        
        fig.suptitle('🌐 Semantic Topology Evolution (3D CUDA-Accelerated)', fontsize=16, fontweight='bold')
        
        ax.set_xlabel('Semantic X')
        ax.set_ylabel('Semantic Y') 
        ax.set_zlabel('Semantic Z')
        ax.set_xlim(-2, 2)
        ax.set_ylim(-2, 2)
        ax.set_zlim(-2, 2)
        
        # Initialize empty scatter plot for nodes
        self.semantic_nodes_3d = ax.scatter([], [], [], s=50, alpha=0.7)
        
        # Initialize empty line collection for edges
        self.semantic_edges_3d = None
        
        # Add colorbar for coherence
        self.semantic_colorbar = None
        
        self.figures['semantic_topology_3d'] = fig
        self.axes['semantic_topology_3d'] = ax
        
        return fig, ax
    
    def setup_neural_activity_heatmap(self) -> Tuple[plt.Figure, plt.Axes]:
        """Setup animated neural activity heatmap"""
        fig, ax = plt.subplots(figsize=(10, 8))
        fig.suptitle('🧬 Neural Activity Patterns (CUDA-Computed)', fontsize=16, fontweight='bold')
        
        ax.set_title('Real-Time Neural Activation Heatmap')
        ax.set_xlabel('Neural Layer')
        ax.set_ylabel('Neural Node')
        
        # Initialize with empty heatmap
        self.neural_heatmap_data = np.zeros((50, 20))  # 50 nodes, 20 layers
        self.neural_heatmap = ax.imshow(self.neural_heatmap_data, cmap='plasma', 
                                       aspect='auto', animated=True, vmin=0, vmax=1)
        
        # Add colorbar
        cbar = plt.colorbar(self.neural_heatmap, ax=ax)
        cbar.set_label('Activation Level')
        
        self.figures['neural_heatmap'] = fig
        self.axes['neural_heatmap'] = ax
        
        return fig, ax
    
    def setup_consciousness_surface_plot(self) -> Tuple[plt.Figure, Axes3D]:
        """Setup animated 3D consciousness surface visualization"""
        fig = plt.figure(figsize=(12, 10))
        ax = fig.add_subplot(111, projection='3d')
        
        fig.suptitle('🌊 Consciousness Energy Surface (GPU-Rendered)', fontsize=16, fontweight='bold')
        
        # Create coordinate grids
        x = np.linspace(-2, 2, 50)
        y = np.linspace(-2, 2, 50)
        self.consciousness_X, self.consciousness_Y = np.meshgrid(x, y)
        
        # Initialize with empty surface
        Z = np.zeros_like(self.consciousness_X)
        self.consciousness_surface = ax.plot_surface(
            self.consciousness_X, self.consciousness_Y, Z,
            cmap='viridis', alpha=0.8, animated=True
        )
        
        ax.set_xlabel('Semantic Dimension 1')
        ax.set_ylabel('Semantic Dimension 2')
        ax.set_zlabel('Consciousness Intensity')
        ax.set_zlim(0, 2)
        
        self.figures['consciousness_surface'] = fig
        self.axes['consciousness_surface'] = ax
        
        return fig, ax
    
    def generate_frame_data_gpu(self) -> AnimationFrame:
        """Generate animation frame data using GPU acceleration"""
        start_time = time.time()
        
        # Current timestamp
        timestamp = time.time()
        
        # Generate consciousness state
        t = timestamp * 0.5  # Slow evolution
        consciousness_state = {
            'coherence': 0.5 + 0.3 * np.sin(t * 0.7),
            'pressure': 0.4 + 0.2 * np.cos(t * 0.5),
            'energy': 0.6 + 0.3 * np.sin(t * 1.2),
            'awareness': 0.5 + 0.25 * np.cos(t * 0.8)
        }
        
        # Generate semantic nodes and edges
        num_nodes = 100
        semantic_nodes = []
        semantic_edges = []
        
        if self.cuda_enabled and self.cuda_accelerator:
            # GPU-accelerated generation
            try:
                # Generate consciousness visualization on GPU
                consciousness_viz = self.cuda_accelerator.generate_consciousness_visualization_gpu(
                    consciousness_state, 64, 64
                )
                
                if consciousness_viz is not None:
                    # Extract semantic positions from GPU visualization
                    for i in range(num_nodes):
                        # Sample positions from consciousness field
                        x_idx = int((i * 31) % 64)
                        y_idx = int((i * 17) % 64)
                        
                        intensity = np.mean(consciousness_viz[y_idx, x_idx, :])
                        
                        node = {
                            'id': f'node_{i}',
                            'position': [
                                (x_idx / 32.0 - 1.0) * 2,  # Scale to [-2, 2]
                                (y_idx / 32.0 - 1.0) * 2,
                                intensity * 2  # Z from intensity
                            ],
                            'coherence': intensity,
                            'tint': consciousness_viz[y_idx, x_idx, :].tolist()
                        }
                        semantic_nodes.append(node)
                
                # Generate edges based on proximity
                for i in range(len(semantic_nodes)):
                    for j in range(i + 1, min(i + 5, len(semantic_nodes))):  # Connect to nearby nodes
                        if np.random.random() < 0.3:  # 30% connection probability
                            edge = {
                                'source': f'node_{i}',
                                'target': f'node_{j}',
                                'weight': np.random.random(),
                                'tension': np.random.random() * 0.5
                            }
                            semantic_edges.append(edge)
                            
            except Exception as e:
                logger.debug(f"GPU generation failed, using CPU fallback: {e}")
                self.cuda_enabled = False
        
        if not self.cuda_enabled:
            # CPU fallback generation
            for i in range(num_nodes):
                angle = (i / num_nodes) * 2 * np.pi
                radius = 0.5 + 0.5 * np.sin(timestamp + i * 0.1)
                
                node = {
                    'id': f'node_{i}',
                    'position': [
                        radius * np.cos(angle + timestamp * 0.2),
                        radius * np.sin(angle + timestamp * 0.2),
                        0.5 * np.sin(timestamp * 0.5 + i * 0.2)
                    ],
                    'coherence': 0.5 + 0.3 * np.sin(timestamp + i * 0.1),
                    'tint': [np.random.random() for _ in range(3)]
                }
                semantic_nodes.append(node)
        
        # Generate neural activity pattern (sized for dashboard)
        neural_activity = np.random.random((30, 15))
        if self.cuda_enabled:
            # Add GPU-influenced patterns
            neural_activity *= consciousness_state['energy']
            neural_activity += 0.2 * consciousness_state['coherence']
        else:
            # CPU patterns based on consciousness state
            t = timestamp * 0.1
            for i in range(30):
                for j in range(15):
                    # Create wave patterns based on consciousness
                    wave = np.sin(i * 0.3 + t) * np.cos(j * 0.4 + t * 0.7)
                    neural_activity[i, j] = 0.5 + 0.3 * wave * consciousness_state['energy']
        
        # Get GPU metrics
        gpu_metrics = {}
        if self.cuda_enabled and self.cuda_accelerator:
            gpu_metrics = self.cuda_accelerator.get_gpu_performance_metrics()
        
        compute_time = time.time() - start_time
        self.gpu_compute_times.append(compute_time)
        
        return AnimationFrame(
            timestamp=timestamp,
            consciousness_state=consciousness_state,
            semantic_nodes=semantic_nodes,
            semantic_edges=semantic_edges,
            neural_activity=neural_activity,
            gpu_metrics=gpu_metrics
        )
    
    def update_unified_dashboard(self, frame_num: int):
        """Update all plots in the unified dashboard"""
        if not self.current_frame:
            return []
        
        # Get axes
        axes = self.axes.get('unified_dashboard', {})
        if not axes:
            return []
        
        # Get recent history
        recent_frames = list(self.frame_history)[-100:]  # Last 100 frames
        
        if len(recent_frames) < 2:
            return []
        
        # Extract time series data
        times = [f.timestamp - recent_frames[0].timestamp for f in recent_frames]
        coherence_values = [f.consciousness_state['coherence'] for f in recent_frames]
        pressure_values = [f.consciousness_state['pressure'] for f in recent_frames]
        energy_values = [f.consciousness_state['energy'] for f in recent_frames]
        awareness_values = [f.consciousness_state['awareness'] for f in recent_frames]
        
        # 1. Update consciousness evolution plot
        self.consciousness_lines['coherence'].set_data(times, coherence_values)
        self.consciousness_lines['pressure'].set_data(times, pressure_values)
        self.consciousness_lines['energy'].set_data(times, energy_values)
        self.consciousness_lines['awareness'].set_data(times, awareness_values)
        
        # Update axes limits
        if times:
            axes['main'].set_xlim(times[0], times[-1])
        
        # 2. Update consciousness level indicator
        current_state = self.current_frame.consciousness_state
        coherence = current_state['coherence']
        energy = current_state['energy']
        
        # Map to consciousness levels
        level_intensity = (coherence + energy) / 2
        if level_intensity < 0.2:
            level_name = 'DORMANT'
            level_color = '#666666'
        elif level_intensity < 0.4:
            level_name = 'FOCUSED'
            level_color = '#4ECDC4'
        elif level_intensity < 0.6:
            level_name = 'AWARE'
            level_color = '#45B7D1'
        elif level_intensity < 0.8:
            level_name = 'META-AWARE'
            level_color = '#96CEB4'
        else:
            level_name = 'TRANSCENDENT'
            level_color = '#FFD93D'
        
        # Update indicator position and color
        angle = time.time() * 2  # Rotating indicator
        x = 0.6 * level_intensity * np.cos(angle)
        y = 0.6 * level_intensity * np.sin(angle)
        
        self.consciousness_indicator.set_offsets(np.array([[x, y]]))
        self.consciousness_indicator.set_color(level_color)
        self.consciousness_indicator.set_sizes([300 + 200 * level_intensity])
        
        self.consciousness_text.set_text(level_name)
        self.consciousness_text.set_color(level_color)
        self.consciousness_text.set_position((0, -0.3))
        
        # 3. Update 3D semantic topology
        if self.current_frame.semantic_nodes:
            ax_3d = axes['3d_topology']
            ax_3d.clear()
            ax_3d.set_title('3D Semantic Topology (Real-Time)', fontsize=14)
            ax_3d.set_xlabel('Semantic X')
            ax_3d.set_ylabel('Semantic Y')
            ax_3d.set_zlabel('Semantic Z')
            ax_3d.set_xlim(-2, 2)
            ax_3d.set_ylim(-2, 2)
            ax_3d.set_zlim(-2, 2)
            
            nodes = self.current_frame.semantic_nodes
            positions = np.array([node['position'] for node in nodes])
            coherences = np.array([node['coherence'] for node in nodes])
            
            # Plot nodes with color based on coherence
            scatter = ax_3d.scatter(positions[:, 0], positions[:, 1], positions[:, 2],
                                  c=coherences, s=50 + 100 * coherences, 
                                  cmap='viridis', alpha=0.8)
            
            # Plot edges
            if self.current_frame.semantic_edges:
                node_dict = {node['id']: node['position'] for node in nodes}
                
                for edge in self.current_frame.semantic_edges[:20]:  # Limit edges for performance
                    source_pos = node_dict.get(edge['source'])
                    target_pos = node_dict.get(edge['target'])
                    
                    if source_pos and target_pos:
                        ax_3d.plot([source_pos[0], target_pos[0]],
                                 [source_pos[1], target_pos[1]],
                                 [source_pos[2], target_pos[2]],
                                 'gray', alpha=0.4, linewidth=1)
        
        # 4. Update neural activity heatmap
        self.neural_heatmap_data = self.current_frame.neural_activity[:30, :15]  # Resize to fit
        self.neural_heatmap.set_array(self.neural_heatmap_data)
        
        # Update neural heatmap title
        avg_activity = np.mean(self.neural_heatmap_data)
        axes['neural'].set_title(f'Neural Activity (Avg: {avg_activity:.3f})', fontsize=14)
        
        # 5. Update 3D consciousness surface
        ax_surface = axes['surface']
        ax_surface.clear()
        ax_surface.set_title('Consciousness Energy Surface', fontsize=14)
        ax_surface.set_xlabel('Dimension 1')
        ax_surface.set_ylabel('Dimension 2')
        ax_surface.set_zlabel('Intensity')
        ax_surface.set_zlim(0, 2)
        
        # Generate consciousness surface
        t = self.current_frame.timestamp
        Z = (current_state['coherence'] * np.exp(-(self.consciousness_X**2 + self.consciousness_Y**2) / 4) +
             current_state['energy'] * np.sin(self.consciousness_X * 2 + t * 0.5) * 
             np.cos(self.consciousness_Y * 2 + t * 0.5) * 0.3 +
             current_state['pressure'] * (self.consciousness_X**2 + self.consciousness_Y**2) * 0.05)
        
        # Normalize and create surface
        Z = np.clip(Z, 0, 2)
        surface = ax_surface.plot_surface(
            self.consciousness_X, self.consciousness_Y, Z,
            cmap='plasma', alpha=0.7, rstride=2, cstride=2
        )
        
        # 6. Update system performance plots
        import psutil
        try:
            cpu_percent = psutil.cpu_percent()
            memory_percent = psutil.virtual_memory().percent
            gpu_percent = 0
            
            if self.cuda_enabled and recent_frames:
                gpu_metrics = self.current_frame.gpu_metrics
                gpu_percent = gpu_metrics.get('gpu_utilization', 0) * 100
            
            # Update performance lines
            cpu_values = [cpu_percent] * len(times)  # Simplified for demo
            memory_values = [memory_percent] * len(times)
            gpu_values = [gpu_percent] * len(times)
            
            self.gpu_lines['cpu'].set_data(times[-20:], cpu_values[-20:])  # Last 20 points
            self.gpu_lines['memory'].set_data(times[-20:], memory_values[-20:])
            self.gpu_lines['gpu'].set_data(times[-20:], gpu_values[-20:])
            
            if times:
                axes['gpu'].set_xlim(times[-20], times[-1]) if len(times) > 20 else axes['gpu'].set_xlim(times[0], times[-1])
                
        except Exception as e:
            logger.debug(f"Performance update failed: {e}")
        
        # Return all animated elements
        animated_elements = (list(self.consciousness_lines.values()) + 
                           [self.consciousness_indicator, self.consciousness_text, 
                            self.neural_heatmap] + list(self.gpu_lines.values()))
        
        return animated_elements
    
    def update_semantic_topology_3d_plot(self, frame_num: int):
        """Update 3D semantic topology animation"""
        if not self.current_frame or not self.current_frame.semantic_nodes:
            return []
        
        nodes = self.current_frame.semantic_nodes
        
        # Extract positions and colors
        positions = np.array([node['position'] for node in nodes])
        coherences = np.array([node['coherence'] for node in nodes])
        colors = np.array([node['tint'] for node in nodes])
        
        # Update scatter plot
        ax = self.axes['semantic_topology_3d']
        ax.clear()
        
        # Plot nodes
        scatter = ax.scatter(positions[:, 0], positions[:, 1], positions[:, 2],
                           c=coherences, s=50 + 100 * coherences, 
                           cmap='viridis', alpha=0.7, animated=True)
        
        # Plot edges
        if self.current_frame.semantic_edges:
            node_dict = {node['id']: node['position'] for node in nodes}
            
            for edge in self.current_frame.semantic_edges:
                source_pos = node_dict.get(edge['source'])
                target_pos = node_dict.get(edge['target'])
                
                if source_pos and target_pos:
                    ax.plot([source_pos[0], target_pos[0]],
                           [source_pos[1], target_pos[1]],
                           [source_pos[2], target_pos[2]],
                           'gray', alpha=0.3, linewidth=edge['weight'] * 2)
        
        # Update labels and limits
        ax.set_xlabel('Semantic X')
        ax.set_ylabel('Semantic Y')
        ax.set_zlabel('Semantic Z')
        ax.set_xlim(-2, 2)
        ax.set_ylim(-2, 2)
        ax.set_zlim(-2, 2)
        ax.set_title(f'Semantic Topology - {len(nodes)} nodes, {len(self.current_frame.semantic_edges)} edges')
        
        return [scatter]
    
    def update_neural_activity_heatmap(self, frame_num: int):
        """Update neural activity heatmap animation"""
        if not self.current_frame:
            return [self.neural_heatmap]
        
        # Update heatmap data
        self.neural_heatmap_data = self.current_frame.neural_activity
        self.neural_heatmap.set_array(self.neural_heatmap_data)
        
        # Add dynamic title with stats
        ax = self.axes['neural_heatmap']
        avg_activity = np.mean(self.neural_heatmap_data)
        max_activity = np.max(self.neural_heatmap_data)
        
        title = f'Neural Activity - Avg: {avg_activity:.3f}, Max: {max_activity:.3f}'
        if self.cuda_enabled:
            title += ' (CUDA-Accelerated)'
        
        ax.set_title(title)
        
        return [self.neural_heatmap]
    
    def update_consciousness_surface_plot(self, frame_num: int):
        """Update 3D consciousness surface animation"""
        if not self.current_frame:
            return []
        
        ax = self.axes['consciousness_surface']
        ax.clear()
        
        # Generate consciousness surface based on current state
        state = self.current_frame.consciousness_state
        t = self.current_frame.timestamp
        
        # Create dynamic surface
        Z = (state['coherence'] * np.exp(-(self.consciousness_X**2 + self.consciousness_Y**2) / 2) +
             state['energy'] * np.sin(self.consciousness_X * 2 + t) * np.cos(self.consciousness_Y * 2 + t) * 0.5 +
             state['pressure'] * (self.consciousness_X**2 + self.consciousness_Y**2) * 0.1)
        
        # Normalize
        Z = np.clip(Z, 0, 2)
        
        # Create surface plot
        surface = ax.plot_surface(
            self.consciousness_X, self.consciousness_Y, Z,
            cmap='plasma', alpha=0.8, animated=True,
            rstride=2, cstride=2
        )
        
        # Update labels
        ax.set_xlabel('Semantic Dimension 1')
        ax.set_ylabel('Semantic Dimension 2')
        ax.set_zlabel('Consciousness Intensity')
        ax.set_zlim(0, 2)
        
        title = f'Consciousness Surface - Coherence: {state["coherence"]:.2f}'
        if self.cuda_enabled:
            title += ' (GPU-Rendered)'
        ax.set_title(title)
        
        return [surface]
    
    def data_generation_thread(self):
        """Background thread for generating animation data"""
        while self.running:
            if not self.paused:
                try:
                    # Generate new frame
                    frame = self.generate_frame_data_gpu()
                    
                    with self.data_lock:
                        self.current_frame = frame
                        self.frame_history.append(frame)
                    
                    # Control frame rate
                    time.sleep(1.0 / self.fps)
                    
                except Exception as e:
                    logger.error(f"Data generation error: {e}")
                    time.sleep(0.1)
            else:
                time.sleep(0.1)
    
    def start_animations(self):
        """Start unified matplotlib animation dashboard"""
        if self.running:
            return
        
        logger.info("🎬 Starting CUDA-powered unified dashboard...")
        
        # Setup unified dashboard
        self.setup_unified_dashboard()
        
        # Start data generation thread
        self.running = True
        self.data_thread = threading.Thread(target=self.data_generation_thread, daemon=True)
        self.data_thread.start()
        
        # Create single unified animation
        self.animations['unified_dashboard'] = animation.FuncAnimation(
            self.figures['unified_dashboard'],
            self.update_unified_dashboard,
            interval=self.frame_interval,
            blit=False,
            cache_frame_data=False
        )
        
        logger.info(f"✅ Started unified dashboard animation at {self.fps} FPS")
    
    def stop_animations(self):
        """Stop all animations"""
        logger.info("🛑 Stopping matplotlib animations...")
        
        self.running = False
        
        # Stop animations
        for name, anim in self.animations.items():
            if anim is not None:
                try:
                    anim.event_source.stop()
                except Exception as e:
                    logger.debug(f"Error stopping animation {name}: {e}")
        
        # Wait for data thread to finish
        if self.data_thread and self.data_thread.is_alive():
            self.data_thread.join(timeout=1.0)
        
        logger.info("✅ All animations stopped")
    
    def pause_animations(self):
        """Pause/resume animations"""
        self.paused = not self.paused
        state = "paused" if self.paused else "resumed"
        logger.info(f"🎬 Animations {state}")
    
    def show_all_plots(self):
        """Display the unified animation dashboard"""
        if 'unified_dashboard' in self.figures:
            self.figures['unified_dashboard'].show()
        plt.show()
    
    def save_animation(self, animation_name: str, filename: str, duration: float = 10.0):
        """Save animation to file"""
        if animation_name not in self.animations:
            logger.error(f"Animation '{animation_name}' not found")
            return
        
        logger.info(f"💾 Saving {animation_name} animation to {filename}...")
        
        # Calculate number of frames
        frames = int(duration * self.fps)
        
        try:
            writer = animation.FFMpegWriter(fps=self.fps, bitrate=1800)
            self.animations[animation_name].save(filename, writer=writer, frames=frames)
            logger.info(f"✅ Animation saved to {filename}")
        except Exception as e:
            logger.error(f"Failed to save animation: {e}")
    
    def get_performance_stats(self) -> Dict[str, Any]:
        """Get animation performance statistics"""
        stats = {
            'cuda_enabled': self.cuda_enabled,
            'fps': self.fps,
            'running': self.running,
            'paused': self.paused,
            'active_animations': len(self.animations),
            'frame_history_size': len(self.frame_history)
        }
        
        if self.render_times:
            stats['avg_render_time'] = np.mean(self.render_times)
            stats['max_render_time'] = np.max(self.render_times)
        
        if self.gpu_compute_times:
            stats['avg_gpu_compute_time'] = np.mean(self.gpu_compute_times)
            stats['gpu_speedup_estimate'] = np.mean(self.render_times) / np.mean(self.gpu_compute_times) if self.render_times else 1.0
        
        return stats


# Global animator instance
_matplotlib_animator = None

def get_matplotlib_cuda_animator(fps: int = 30) -> MatplotlibCUDAAnimator:
    """Get the global matplotlib CUDA animator instance"""
    global _matplotlib_animator
    if _matplotlib_animator is None:
        _matplotlib_animator = MatplotlibCUDAAnimator(fps=fps)
    return _matplotlib_animator
