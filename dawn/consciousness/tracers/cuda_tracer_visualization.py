#!/usr/bin/env python3
"""
🚀 CUDA-Powered Tracer Visualization System
==========================================

Advanced GPU-accelerated visualization system for DAWN's tracer ecosystem.
Provides real-time 3D visualization of tracer movements, interactions, and
ecosystem dynamics with DAWN singleton integration.

Features:
- Real-time 3D tracer visualization
- GPU-accelerated particle systems
- Interactive ecosystem exploration
- CUDA-powered visual effects
- DAWN consciousness integration
- Live telemetry visualization

"Visualizing consciousness at the speed of light."
"""

import numpy as np
import logging
import time
import threading
from typing import Dict, List, Optional, Tuple, Any, Union
from dataclasses import dataclass, field
import uuid
import queue

# DAWN core imports
from dawn.core.singleton import get_dawn
from .base_tracer import BaseTracer, TracerType, TracerStatus
from .tracer_manager import TracerManager
from .cuda_tracer_engine import CUDATracerModelingEngine, get_cuda_tracer_engine

logger = logging.getLogger(__name__)

# Visualization imports
try:
    import matplotlib.pyplot as plt
    from matplotlib.animation import FuncAnimation
    from mpl_toolkits.mplot3d import Axes3D
    import matplotlib.patches as patches
    MATPLOTLIB_AVAILABLE = True
except ImportError:
    MATPLOTLIB_AVAILABLE = False
    logger.info("Matplotlib not available - 2D/3D plotting disabled")

try:
    import plotly.graph_objects as go
    import plotly.express as px
    from plotly.subplots import make_subplots
    import plotly.offline as pyo
    PLOTLY_AVAILABLE = True
except ImportError:
    PLOTLY_AVAILABLE = False
    logger.info("Plotly not available - interactive plotting disabled")

# CUDA visualization imports
CUDA_VIZ_AVAILABLE = False
try:
    import cupy as cp
    import cupyx.scipy.ndimage as ndimage
    CUDA_VIZ_AVAILABLE = True
    logger.info("🚀 CuPy available for CUDA visualization acceleration")
except ImportError:
    logger.info("CuPy not available - falling back to CPU visualization")

try:
    import torch
    if torch.cuda.is_available():
        TORCH_CUDA_VIZ = True
        logger.info("🚀 PyTorch CUDA available for visualization")
    else:
        TORCH_CUDA_VIZ = False
except ImportError:
    TORCH_CUDA_VIZ = False


@dataclass
class TracerVisualizationConfig:
    """Configuration for tracer visualization"""
    window_size: Tuple[int, int] = (1920, 1080)
    fps: float = 30.0
    max_trail_length: int = 100
    particle_size_scale: float = 1.0
    enable_3d: bool = True
    enable_trails: bool = True
    enable_interactions: bool = True
    enable_nutrient_field: bool = True
    color_scheme: str = "biological"  # biological, neon, pastel
    background_color: str = "black"
    auto_rotate: bool = True
    rotation_speed: float = 0.5


@dataclass
class TracerVisualState:
    """Visual state for a single tracer"""
    tracer_id: str
    tracer_type: TracerType
    position: Tuple[float, float, float]
    velocity: Tuple[float, float, float]
    trail_positions: List[Tuple[float, float, float]]
    color: Tuple[float, float, float, float]  # RGBA
    size: float
    activity_level: float
    age: int
    status: TracerStatus


class CUDATracerVisualizationEngine:
    """
    CUDA-accelerated tracer ecosystem visualization engine.
    
    Provides real-time 3D visualization of tracer movements, interactions,
    and ecosystem dynamics with GPU acceleration.
    """
    
    def __init__(self, config: Optional[TracerVisualizationConfig] = None):
        """Initialize the CUDA tracer visualization engine."""
        self.config = config or TracerVisualizationConfig()
        self.engine_id = str(uuid.uuid4())
        
        # DAWN singleton integration
        self.dawn = get_dawn()
        self.consciousness_bus = None
        self.telemetry_system = None
        
        # Tracer engine integration
        self.cuda_tracer_engine = None
        self.tracer_manager = None
        
        # Visualization state
        self.visual_tracers: Dict[str, TracerVisualState] = {}
        self.nutrient_field_visual = None
        self.interaction_lines = []
        
        # GPU visualization arrays
        self.gpu_viz_arrays = {}
        
        # Visualization thread and control
        self.visualization_running = False
        self.visualization_thread = None
        self.frame_queue = queue.Queue(maxsize=60)
        
        # Performance metrics
        self.viz_metrics = {
            'fps': 0.0,
            'frame_time_ms': 0.0,
            'gpu_memory_viz': 0.0,
            'particles_rendered': 0,
            'trails_rendered': 0
        }
        
        # Color schemes
        self.color_schemes = {
            'biological': {
                TracerType.CROW: (0.2, 0.2, 0.2, 1.0),      # Dark gray
                TracerType.WHALE: (0.0, 0.3, 0.8, 1.0),     # Deep blue
                TracerType.ANT: (0.6, 0.3, 0.0, 1.0),       # Brown
                TracerType.SPIDER: (0.5, 0.5, 0.5, 1.0),    # Gray
                TracerType.BEETLE: (0.2, 0.6, 0.2, 1.0),    # Green
                TracerType.BEE: (1.0, 0.8, 0.0, 1.0),       # Yellow
                TracerType.OWL: (0.6, 0.4, 0.2, 1.0),       # Brown
                TracerType.MEDIEVAL_BEE: (0.8, 0.6, 0.0, 1.0) # Gold
            },
            'neon': {
                TracerType.CROW: (0.8, 0.0, 0.8, 1.0),      # Magenta
                TracerType.WHALE: (0.0, 0.8, 1.0, 1.0),     # Cyan
                TracerType.ANT: (1.0, 0.4, 0.0, 1.0),       # Orange
                TracerType.SPIDER: (0.8, 0.8, 0.8, 1.0),    # White
                TracerType.BEETLE: (0.0, 1.0, 0.0, 1.0),    # Lime
                TracerType.BEE: (1.0, 1.0, 0.0, 1.0),       # Yellow
                TracerType.OWL: (0.6, 0.0, 1.0, 1.0),       # Purple
                TracerType.MEDIEVAL_BEE: (1.0, 0.6, 0.0, 1.0) # Gold
            }
        }
        
        logger.info(f"🚀 CUDA Tracer Visualization Engine initialized: {self.engine_id}")
        
        # Initialize components
        self._initialize_dawn_integration()
        self._initialize_visualization_arrays()
    
    def _initialize_dawn_integration(self):
        """Initialize integration with DAWN singleton"""
        try:
            if self.dawn.is_initialized:
                self.consciousness_bus = self.dawn.consciousness_bus
                self.telemetry_system = self.dawn.telemetry_system
                
                if self.consciousness_bus:
                    # Register visualization engine
                    self.consciousness_bus.register_module(
                        'cuda_tracer_visualization',
                        self,
                        capabilities=['tracer_visualization', 'gpu_rendering', 'real_time_display']
                    )
                    logger.info("✅ CUDA Visualization Engine registered with consciousness bus")
                
                if self.telemetry_system:
                    # Register telemetry metrics
                    self.telemetry_system.register_metric_source(
                        'cuda_tracer_visualization',
                        self._get_telemetry_metrics
                    )
                    logger.info("✅ CUDA Visualization Engine telemetry registered")
            
            # Get tracer engine
            self.cuda_tracer_engine = get_cuda_tracer_engine()
            
        except Exception as e:
            logger.error(f"Failed to initialize DAWN integration: {e}")
    
    def _initialize_visualization_arrays(self):
        """Initialize GPU arrays for visualization"""
        if not CUDA_VIZ_AVAILABLE:
            logger.info("CUDA visualization not available - using CPU arrays")
            return
        
        try:
            max_tracers = 1000
            max_trail_length = self.config.max_trail_length
            
            # Initialize GPU arrays for particle rendering
            self.gpu_viz_arrays = {
                'positions': cp.zeros((max_tracers, 3), dtype=cp.float32),
                'colors': cp.zeros((max_tracers, 4), dtype=cp.float32),
                'sizes': cp.ones(max_tracers, dtype=cp.float32),
                'trails': cp.zeros((max_tracers, max_trail_length, 3), dtype=cp.float32),
                'trail_colors': cp.zeros((max_tracers, max_trail_length, 4), dtype=cp.float32),
                'interaction_matrix': cp.zeros((max_tracers, max_tracers), dtype=cp.float32)
            }
            
            logger.info("✅ GPU visualization arrays initialized")
            
        except Exception as e:
            logger.error(f"Failed to initialize GPU visualization arrays: {e}")
    
    def connect_to_tracer_manager(self, tracer_manager: TracerManager):
        """Connect to a tracer manager for data source"""
        self.tracer_manager = tracer_manager
        logger.info("✅ Connected to TracerManager for visualization data")
    
    def update_visual_state(self):
        """Update visual state from tracer engine and manager"""
        if not self.cuda_tracer_engine or not self.tracer_manager:
            return
        
        try:
            # Get current tracer positions from CUDA engine
            gpu_positions = self.cuda_tracer_engine.get_tracer_positions()
            
            # Get active tracers from manager
            active_tracers = self.tracer_manager.active_tracers
            
            # Update visual tracers
            for tracer_id, tracer in active_tracers.items():
                if tracer_id not in self.visual_tracers:
                    # Create new visual state
                    color_scheme = self.color_schemes.get(self.config.color_scheme, self.color_schemes['biological'])
                    color = color_scheme.get(tracer.tracer_type, (1.0, 1.0, 1.0, 1.0))
                    
                    self.visual_tracers[tracer_id] = TracerVisualState(
                        tracer_id=tracer_id,
                        tracer_type=tracer.tracer_type,
                        position=(0.5, 0.5, 0.5),  # Default center
                        velocity=(0.0, 0.0, 0.0),
                        trail_positions=[],
                        color=color,
                        size=self._get_tracer_size(tracer.tracer_type),
                        activity_level=1.0,
                        age=tracer.get_age(0),
                        status=tracer.status
                    )
                
                # Update position from GPU if available
                if tracer_id in gpu_positions:
                    visual_tracer = self.visual_tracers[tracer_id]
                    old_position = visual_tracer.position
                    new_position = gpu_positions[tracer_id]
                    
                    # Calculate velocity
                    velocity = (
                        new_position[0] - old_position[0],
                        new_position[1] - old_position[1],
                        new_position[2] - old_position[2]
                    )
                    
                    # Update visual state
                    visual_tracer.position = new_position
                    visual_tracer.velocity = velocity
                    visual_tracer.status = tracer.status
                    visual_tracer.age = tracer.get_age(0)
                    
                    # Update trail
                    if self.config.enable_trails:
                        visual_tracer.trail_positions.append(new_position)
                        if len(visual_tracer.trail_positions) > self.config.max_trail_length:
                            visual_tracer.trail_positions.pop(0)
            
            # Remove inactive tracers
            inactive_tracers = [
                tid for tid in self.visual_tracers.keys() 
                if tid not in active_tracers
            ]
            for tid in inactive_tracers:
                del self.visual_tracers[tid]
            
            # Update GPU arrays if available
            if CUDA_VIZ_AVAILABLE and self.gpu_viz_arrays:
                self._update_gpu_visualization_arrays()
            
        except Exception as e:
            logger.error(f"Error updating visual state: {e}")
    
    def _update_gpu_visualization_arrays(self):
        """Update GPU arrays with current visual state"""
        try:
            n_tracers = len(self.visual_tracers)
            if n_tracers == 0:
                return
            
            # Prepare CPU arrays
            positions = np.zeros((n_tracers, 3), dtype=np.float32)
            colors = np.zeros((n_tracers, 4), dtype=np.float32)
            sizes = np.zeros(n_tracers, dtype=np.float32)
            
            for i, visual_tracer in enumerate(self.visual_tracers.values()):
                positions[i] = visual_tracer.position
                colors[i] = visual_tracer.color
                sizes[i] = visual_tracer.size * visual_tracer.activity_level
            
            # Transfer to GPU
            self.gpu_viz_arrays['positions'][:n_tracers] = cp.asarray(positions)
            self.gpu_viz_arrays['colors'][:n_tracers] = cp.asarray(colors)
            self.gpu_viz_arrays['sizes'][:n_tracers] = cp.asarray(sizes)
            
            # Update trails
            if self.config.enable_trails:
                self._update_gpu_trails(n_tracers)
            
        except Exception as e:
            logger.error(f"Error updating GPU visualization arrays: {e}")
    
    def _update_gpu_trails(self, n_tracers: int):
        """Update GPU trail arrays"""
        try:
            max_trail_length = self.config.max_trail_length
            
            for i, visual_tracer in enumerate(list(self.visual_tracers.values())[:n_tracers]):
                trail_positions = visual_tracer.trail_positions
                n_trail_points = len(trail_positions)
                
                if n_trail_points > 0:
                    # Fill trail array
                    trail_array = np.zeros((max_trail_length, 3), dtype=np.float32)
                    trail_array[:n_trail_points] = trail_positions
                    
                    # Create fading colors for trail
                    trail_colors = np.zeros((max_trail_length, 4), dtype=np.float32)
                    base_color = visual_tracer.color[:3]
                    
                    for j in range(n_trail_points):
                        alpha = (j + 1) / n_trail_points * visual_tracer.color[3]
                        trail_colors[j] = (*base_color, alpha)
                    
                    # Transfer to GPU
                    self.gpu_viz_arrays['trails'][i] = cp.asarray(trail_array)
                    self.gpu_viz_arrays['trail_colors'][i] = cp.asarray(trail_colors)
            
        except Exception as e:
            logger.error(f"Error updating GPU trails: {e}")
    
    def _get_tracer_size(self, tracer_type: TracerType) -> float:
        """Get visual size for tracer type"""
        size_map = {
            TracerType.CROW: 0.8,
            TracerType.WHALE: 2.0,
            TracerType.ANT: 0.5,
            TracerType.SPIDER: 1.2,
            TracerType.BEETLE: 1.0,
            TracerType.BEE: 0.7,
            TracerType.OWL: 1.5,
            TracerType.MEDIEVAL_BEE: 1.8
        }
        return size_map.get(tracer_type, 1.0) * self.config.particle_size_scale
    
    def render_frame_matplotlib(self, ax) -> Dict[str, Any]:
        """Render a frame using matplotlib (3D)"""
        if not MATPLOTLIB_AVAILABLE:
            return {'error': 'Matplotlib not available'}
        
        start_time = time.time()
        
        try:
            # Clear previous frame
            ax.clear()
            
            # Set up 3D space
            ax.set_xlim(0, 1)
            ax.set_ylim(0, 1)
            ax.set_zlim(0, 1)
            ax.set_xlabel('X')
            ax.set_ylabel('Y')
            ax.set_zlabel('Z')
            
            # Render nutrient field if enabled
            if self.config.enable_nutrient_field and self.cuda_tracer_engine:
                nutrient_field = self.cuda_tracer_engine.get_nutrient_field()
                if nutrient_field is not None:
                    self._render_nutrient_field_matplotlib(ax, nutrient_field)
            
            # Render tracers
            particles_rendered = 0
            trails_rendered = 0
            
            for visual_tracer in self.visual_tracers.values():
                # Render tracer particle
                ax.scatter(
                    visual_tracer.position[0],
                    visual_tracer.position[1], 
                    visual_tracer.position[2],
                    c=[visual_tracer.color[:3]],
                    s=visual_tracer.size * 100,
                    alpha=visual_tracer.color[3],
                    marker='o'
                )
                particles_rendered += 1
                
                # Render trail
                if self.config.enable_trails and len(visual_tracer.trail_positions) > 1:
                    trail_array = np.array(visual_tracer.trail_positions)
                    ax.plot(
                        trail_array[:, 0],
                        trail_array[:, 1],
                        trail_array[:, 2],
                        color=visual_tracer.color[:3],
                        alpha=0.3,
                        linewidth=1
                    )
                    trails_rendered += 1
            
            # Render interactions if enabled
            if self.config.enable_interactions:
                self._render_interactions_matplotlib(ax)
            
            # Update metrics
            frame_time = time.time() - start_time
            self.viz_metrics.update({
                'frame_time_ms': frame_time * 1000,
                'fps': 1.0 / max(frame_time, 0.001),
                'particles_rendered': particles_rendered,
                'trails_rendered': trails_rendered
            })
            
            return {
                'frame_time_ms': frame_time * 1000,
                'particles_rendered': particles_rendered,
                'trails_rendered': trails_rendered
            }
            
        except Exception as e:
            logger.error(f"Error rendering matplotlib frame: {e}")
            return {'error': str(e)}
    
    def _render_nutrient_field_matplotlib(self, ax, nutrient_field: np.ndarray):
        """Render nutrient field as volume visualization"""
        try:
            # Simple approach: show high-nutrient regions as transparent spheres
            grid_size = nutrient_field.shape[0]
            threshold = 0.7  # Only show high-nutrient areas
            
            x, y, z = np.meshgrid(
                np.linspace(0, 1, grid_size),
                np.linspace(0, 1, grid_size), 
                np.linspace(0, 1, grid_size)
            )
            
            # Find high-nutrient points
            high_nutrient_mask = nutrient_field > threshold
            if np.any(high_nutrient_mask):
                high_x = x[high_nutrient_mask]
                high_y = y[high_nutrient_mask]
                high_z = z[high_nutrient_mask]
                high_values = nutrient_field[high_nutrient_mask]
                
                ax.scatter(
                    high_x, high_y, high_z,
                    c=high_values,
                    cmap='Greens',
                    alpha=0.3,
                    s=20
                )
            
        except Exception as e:
            logger.error(f"Error rendering nutrient field: {e}")
    
    def _render_interactions_matplotlib(self, ax):
        """Render tracer interactions as lines"""
        try:
            if not self.cuda_tracer_engine or not self.cuda_tracer_engine.interaction_matrix_gpu:
                return
            
            # Get interaction matrix
            if CUDA_VIZ_AVAILABLE:
                interaction_matrix = cp.asnumpy(self.cuda_tracer_engine.interaction_matrix_gpu)
            else:
                interaction_matrix = self.cuda_tracer_engine.interaction_matrix_gpu
            
            tracer_list = list(self.visual_tracers.values())
            n_tracers = len(tracer_list)
            
            if n_tracers > len(interaction_matrix):
                return
            
            # Draw interaction lines for strong interactions
            for i in range(n_tracers):
                for j in range(i + 1, n_tracers):
                    interaction_strength = abs(interaction_matrix[i, j])
                    
                    if interaction_strength > 0.1:  # Threshold for visible interactions
                        tracer_i = tracer_list[i]
                        tracer_j = tracer_list[j]
                        
                        # Line color based on interaction type
                        color = 'red' if interaction_matrix[i, j] < 0 else 'blue'
                        alpha = min(interaction_strength * 2, 1.0)
                        
                        ax.plot(
                            [tracer_i.position[0], tracer_j.position[0]],
                            [tracer_i.position[1], tracer_j.position[1]],
                            [tracer_i.position[2], tracer_j.position[2]],
                            color=color,
                            alpha=alpha,
                            linewidth=interaction_strength * 2
                        )
            
        except Exception as e:
            logger.error(f"Error rendering interactions: {e}")
    
    def create_plotly_visualization(self) -> Optional[go.Figure]:
        """Create interactive Plotly visualization"""
        if not PLOTLY_AVAILABLE:
            logger.warning("Plotly not available for interactive visualization")
            return None
        
        try:
            fig = go.Figure()
            
            # Add tracer particles
            for visual_tracer in self.visual_tracers.values():
                fig.add_trace(go.Scatter3d(
                    x=[visual_tracer.position[0]],
                    y=[visual_tracer.position[1]],
                    z=[visual_tracer.position[2]],
                    mode='markers',
                    marker=dict(
                        size=visual_tracer.size * 10,
                        color=f'rgba({int(visual_tracer.color[0]*255)}, {int(visual_tracer.color[1]*255)}, {int(visual_tracer.color[2]*255)}, {visual_tracer.color[3]})',
                        opacity=visual_tracer.color[3]
                    ),
                    name=f'{visual_tracer.tracer_type.value}_{visual_tracer.tracer_id[:8]}',
                    text=f'Type: {visual_tracer.tracer_type.value}<br>Age: {visual_tracer.age}<br>Status: {visual_tracer.status.value}',
                    hoverinfo='text'
                ))
                
                # Add trail if enabled
                if self.config.enable_trails and len(visual_tracer.trail_positions) > 1:
                    trail_array = np.array(visual_tracer.trail_positions)
                    fig.add_trace(go.Scatter3d(
                        x=trail_array[:, 0],
                        y=trail_array[:, 1],
                        z=trail_array[:, 2],
                        mode='lines',
                        line=dict(
                            color=f'rgba({int(visual_tracer.color[0]*255)}, {int(visual_tracer.color[1]*255)}, {int(visual_tracer.color[2]*255)}, 0.3)',
                            width=2
                        ),
                        name=f'Trail_{visual_tracer.tracer_id[:8]}',
                        showlegend=False
                    ))
            
            # Update layout
            fig.update_layout(
                title='DAWN Tracer Ecosystem - Real-time 3D Visualization',
                scene=dict(
                    xaxis_title='X',
                    yaxis_title='Y',
                    zaxis_title='Z',
                    bgcolor=self.config.background_color,
                    camera=dict(
                        eye=dict(x=1.5, y=1.5, z=1.5)
                    )
                ),
                width=self.config.window_size[0],
                height=self.config.window_size[1]
            )
            
            return fig
            
        except Exception as e:
            logger.error(f"Error creating Plotly visualization: {e}")
            return None
    
    def start_real_time_visualization(self, backend: str = 'matplotlib'):
        """Start real-time visualization"""
        if self.visualization_running:
            logger.warning("Visualization already running")
            return
        
        self.visualization_running = True
        
        if backend == 'matplotlib' and MATPLOTLIB_AVAILABLE:
            self.visualization_thread = threading.Thread(
                target=self._matplotlib_visualization_loop,
                name="cuda_tracer_visualization",
                daemon=True
            )
        elif backend == 'plotly' and PLOTLY_AVAILABLE:
            self.visualization_thread = threading.Thread(
                target=self._plotly_visualization_loop,
                name="cuda_tracer_visualization",
                daemon=True
            )
        else:
            logger.error(f"Visualization backend '{backend}' not available")
            return
        
        self.visualization_thread.start()
        logger.info(f"🚀 Started real-time tracer visualization with {backend}")
    
    def stop_real_time_visualization(self):
        """Stop real-time visualization"""
        if not self.visualization_running:
            return
        
        self.visualization_running = False
        if self.visualization_thread and self.visualization_thread.is_alive():
            self.visualization_thread.join(timeout=5.0)
        
        logger.info("🛑 Stopped real-time tracer visualization")
    
    def _matplotlib_visualization_loop(self):
        """Matplotlib visualization loop"""
        if not MATPLOTLIB_AVAILABLE:
            return
        
        fig = plt.figure(figsize=(12, 9))
        ax = fig.add_subplot(111, projection='3d')
        
        def animate(frame):
            if not self.visualization_running:
                return
            
            # Update visual state
            self.update_visual_state()
            
            # Render frame
            self.render_frame_matplotlib(ax)
            
            # Update title with metrics
            ax.set_title(f'DAWN Tracer Ecosystem - FPS: {self.viz_metrics["fps"]:.1f} | '
                        f'Tracers: {self.viz_metrics["particles_rendered"]} | '
                        f'Frame: {self.viz_metrics["frame_time_ms"]:.1f}ms')
        
        try:
            ani = FuncAnimation(
                fig, animate, 
                interval=int(1000 / self.config.fps),
                blit=False
            )
            plt.show()
            
        except Exception as e:
            logger.error(f"Error in matplotlib visualization loop: {e}")
    
    def _plotly_visualization_loop(self):
        """Plotly visualization loop"""
        while self.visualization_running:
            try:
                # Update visual state
                self.update_visual_state()
                
                # Create figure
                fig = self.create_plotly_visualization()
                if fig:
                    # Save to file or display
                    pyo.plot(fig, filename='tracer_ecosystem.html', auto_open=False)
                
                # Wait for next frame
                time.sleep(1.0 / self.config.fps)
                
            except Exception as e:
                logger.error(f"Error in Plotly visualization loop: {e}")
                time.sleep(1.0)
    
    def _get_telemetry_metrics(self) -> Dict[str, Any]:
        """Get telemetry metrics for DAWN integration"""
        return {
            'cuda_tracer_visualization': {
                'engine_id': self.engine_id,
                'visualization_running': self.visualization_running,
                'visual_tracers_count': len(self.visual_tracers),
                'performance_metrics': self.viz_metrics.copy(),
                'config': {
                    'fps': self.config.fps,
                    'enable_3d': self.config.enable_3d,
                    'enable_trails': self.config.enable_trails,
                    'color_scheme': self.config.color_scheme
                },
                'capabilities': {
                    'matplotlib_available': MATPLOTLIB_AVAILABLE,
                    'plotly_available': PLOTLY_AVAILABLE,
                    'cuda_viz_available': CUDA_VIZ_AVAILABLE
                }
            }
        }
    
    def get_visualization_summary(self) -> Dict[str, Any]:
        """Get comprehensive visualization summary"""
        return {
            'engine_id': self.engine_id,
            'status': {
                'running': self.visualization_running,
                'visual_tracers': len(self.visual_tracers),
                'connected_to_tracer_engine': self.cuda_tracer_engine is not None,
                'connected_to_tracer_manager': self.tracer_manager is not None
            },
            'performance': self.viz_metrics.copy(),
            'configuration': {
                'fps': self.config.fps,
                'window_size': self.config.window_size,
                'max_trail_length': self.config.max_trail_length,
                'color_scheme': self.config.color_scheme,
                'features_enabled': {
                    'trails': self.config.enable_trails,
                    '3d': self.config.enable_3d,
                    'interactions': self.config.enable_interactions,
                    'nutrient_field': self.config.enable_nutrient_field
                }
            },
            'capabilities': {
                'matplotlib': MATPLOTLIB_AVAILABLE,
                'plotly': PLOTLY_AVAILABLE,
                'cuda_acceleration': CUDA_VIZ_AVAILABLE,
                'torch_cuda': TORCH_CUDA_VIZ
            },
            'dawn_integration': {
                'consciousness_bus_connected': self.consciousness_bus is not None,
                'telemetry_connected': self.telemetry_system is not None,
                'dawn_initialized': self.dawn.is_initialized if self.dawn else False
            }
        }


# Global visualization engine instance
_global_cuda_viz_engine: Optional[CUDATracerVisualizationEngine] = None
_cuda_viz_lock = threading.Lock()


def get_cuda_tracer_visualization_engine(config: Optional[TracerVisualizationConfig] = None) -> CUDATracerVisualizationEngine:
    """
    Get the global CUDA tracer visualization engine instance.
    
    Args:
        config: Optional configuration for the visualization engine
        
    Returns:
        CUDATracerVisualizationEngine instance
    """
    global _global_cuda_viz_engine
    
    with _cuda_viz_lock:
        if _global_cuda_viz_engine is None:
            _global_cuda_viz_engine = CUDATracerVisualizationEngine(config)
    
    return _global_cuda_viz_engine


def reset_cuda_tracer_visualization_engine():
    """Reset the global CUDA tracer visualization engine (use with caution)"""
    global _global_cuda_viz_engine
    
    with _cuda_viz_lock:
        if _global_cuda_viz_engine and _global_cuda_viz_engine.visualization_running:
            _global_cuda_viz_engine.stop_real_time_visualization()
        _global_cuda_viz_engine = None


if __name__ == "__main__":
    # Demo the CUDA tracer visualization engine
    logging.basicConfig(level=logging.INFO)
    
    print("🚀" * 30)
    print("🧠 DAWN CUDA TRACER VISUALIZATION ENGINE DEMO")
    print("🚀" * 30)
    
    # Create visualization engine
    viz_engine = CUDATracerVisualizationEngine()
    
    # Show configuration
    summary = viz_engine.get_visualization_summary()
    print(f"✅ Visualization Engine Summary: {summary}")
    
    # Test with dummy data if visualization libraries are available
    if MATPLOTLIB_AVAILABLE or PLOTLY_AVAILABLE:
        print("🚀 Visualization libraries available - creating test visualization")
        
        # Create some dummy visual tracers
        from .crow_tracer import CrowTracer
        from .whale_tracer import WhaleTracer
        
        dummy_tracers = {
            'crow_1': CrowTracer(),
            'whale_1': WhaleTracer()
        }
        
        # Create visual states
        for i, (tid, tracer) in enumerate(dummy_tracers.items()):
            viz_engine.visual_tracers[tid] = TracerVisualState(
                tracer_id=tid,
                tracer_type=tracer.tracer_type,
                position=(0.3 + i * 0.2, 0.5, 0.5),
                velocity=(0.01, 0.0, 0.0),
                trail_positions=[(0.3 + i * 0.2 - j * 0.05, 0.5, 0.5) for j in range(10)],
                color=viz_engine.color_schemes['biological'][tracer.tracer_type],
                size=viz_engine._get_tracer_size(tracer.tracer_type),
                activity_level=1.0,
                age=10,
                status=TracerStatus.ACTIVE
            )
        
        if PLOTLY_AVAILABLE:
            print("🚀 Creating Plotly visualization...")
            fig = viz_engine.create_plotly_visualization()
            if fig:
                print("✅ Plotly figure created successfully")
        
        print("✅ Visualization test complete")
    
    else:
        print("⚠️  No visualization libraries available - skipping visual test")
    
    print("🚀 CUDA Tracer Visualization Engine demo complete!")
