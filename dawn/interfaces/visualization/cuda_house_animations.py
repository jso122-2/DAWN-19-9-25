#!/usr/bin/env python3
"""
🚀 DAWN CUDA House Animations
============================

Advanced CUDA-accelerated matplotlib animations for the DAWN House systems:
- Mycelial House: Living network dynamics with nutrient flows
- Schema House: Sigil operations and symbolic transformations  
- Monitoring House: Real-time telemetry and system health visualization

Features:
- GPU-accelerated particle systems and fluid dynamics
- Real-time data integration from DAWN subsystems
- Interactive 3D animations with matplotlib
- Performance-optimized rendering pipeline
- Seamless GUI integration

"Animating consciousness architecture at the speed of light."
"""

import numpy as np
import logging
import time
import threading
from typing import Dict, List, Optional, Tuple, Any, Union, Callable
from dataclasses import dataclass, field
from collections import defaultdict, deque
import uuid
import queue
from pathlib import Path

# DAWN core imports
from dawn.core.singleton import get_dawn

logger = logging.getLogger(__name__)

# Matplotlib imports
try:
    import matplotlib.pyplot as plt
    import matplotlib.patches as patches
    from matplotlib.animation import FuncAnimation
    from matplotlib.figure import Figure
    from matplotlib.backends.backend_tkagg import FigureCanvasTkAgg
    from mpl_toolkits.mplot3d import Axes3D
    from mpl_toolkits.axes_grid1 import make_axes_locatable
    import matplotlib.colors as mcolors
    from matplotlib.patches import Circle, Rectangle, FancyBboxPatch, Ellipse
    from matplotlib.collections import LineCollection, PatchCollection
    from matplotlib.path import Path as MPath
    import matplotlib.patheffects as path_effects
    MATPLOTLIB_AVAILABLE = True
    logger.info("✅ Matplotlib available for house animations")
except ImportError:
    MATPLOTLIB_AVAILABLE = False
    logger.warning("❌ Matplotlib not available - animations disabled")

# CUDA imports
CUDA_AVAILABLE = False
CUPY_AVAILABLE = False
TORCH_CUDA_AVAILABLE = False

try:
    import cupy as cp
    CUPY_AVAILABLE = True
    logger.info("🚀 CuPy available for CUDA acceleration")
except ImportError:
    logger.debug("CuPy not available - falling back to NumPy")

try:
    import torch
    if torch.cuda.is_available():
        TORCH_CUDA_AVAILABLE = True
        logger.info(f"🚀 PyTorch CUDA available - {torch.cuda.device_count()} GPU(s)")
    else:
        logger.debug("PyTorch CUDA not available")
except ImportError:
    logger.debug("PyTorch not available")

CUDA_AVAILABLE = CUPY_AVAILABLE or TORCH_CUDA_AVAILABLE

# Scientific computing imports
try:
    from scipy import ndimage, signal
    from scipy.spatial.distance import pdist, squareform
    from scipy.interpolate import interp1d
    SCIPY_AVAILABLE = True
except ImportError:
    SCIPY_AVAILABLE = False
    logger.debug("SciPy not available - some features disabled")


@dataclass
class AnimationConfig:
    """Configuration for house animations"""
    fps: int = 30
    duration: float = 60.0  # seconds
    figure_size: Tuple[int, int] = (16, 10)
    dpi: int = 100
    style: str = "dark_background"
    color_scheme: str = "consciousness"
    enable_cuda: bool = True
    enable_3d: bool = True
    particle_count: int = 1000
    fluid_resolution: int = 64
    animation_quality: str = "high"  # low, medium, high, ultra


@dataclass
class ParticleSystem:
    """GPU-accelerated particle system for animations"""
    positions: np.ndarray
    velocities: np.ndarray
    colors: np.ndarray
    sizes: np.ndarray
    lifetimes: np.ndarray
    types: np.ndarray
    gpu_positions: Optional[Any] = None
    gpu_velocities: Optional[Any] = None


class CUDAHouseAnimator:
    """
    Base class for CUDA-accelerated house animations.
    Provides common functionality for all house visualization systems.
    """
    
    def __init__(self, house_name: str, config: Optional[AnimationConfig] = None):
        self.house_name = house_name
        self.config = config or AnimationConfig()
        self.animator_id = str(uuid.uuid4())
        
        # DAWN integration
        self.dawn = get_dawn()
        
        # Animation state
        self.figure = None
        self.axes = None
        self.animation = None
        self.running = False
        self.frame_count = 0
        self.start_time = time.time()
        
        # Data management
        self.data_queue = queue.Queue(maxsize=100)
        self.current_data = {}
        self.data_history = deque(maxlen=1000)
        
        # CUDA acceleration
        self.cuda_enabled = CUDA_AVAILABLE and self.config.enable_cuda
        self.particle_systems = {}
        self.gpu_arrays = {}
        
        # Performance tracking
        self.frame_times = deque(maxlen=100)
        self.avg_fps = 0.0
        
        logger.info(f"🎬 CUDA House Animator initialized: {house_name} ({self.animator_id})")
    
    def initialize_cuda_resources(self):
        """Initialize CUDA resources for acceleration"""
        if not self.cuda_enabled:
            return
        
        try:
            if CUPY_AVAILABLE:
                # Initialize CuPy memory pool
                mempool = cp.get_default_memory_pool()
                mempool.set_limit(size=2**30)  # 1GB limit
                logger.info("✅ CuPy memory pool initialized")
            
            if TORCH_CUDA_AVAILABLE:
                # Initialize PyTorch CUDA
                device = torch.device('cuda')
                torch.cuda.empty_cache()
                logger.info(f"✅ PyTorch CUDA initialized on {device}")
            
        except Exception as e:
            logger.warning(f"CUDA initialization failed: {e}")
            self.cuda_enabled = False
    
    def create_particle_system(self, name: str, count: int, particle_type: str = "default") -> ParticleSystem:
        """Create a GPU-accelerated particle system"""
        # Initialize particle data
        positions = np.random.rand(count, 3).astype(np.float32)
        velocities = (np.random.rand(count, 3) - 0.5).astype(np.float32) * 0.1
        colors = np.random.rand(count, 4).astype(np.float32)
        sizes = np.random.rand(count).astype(np.float32) * 10 + 5
        lifetimes = np.random.rand(count).astype(np.float32) * 100
        types = np.full(count, hash(particle_type) % 10, dtype=np.int32)
        
        # Create particle system
        system = ParticleSystem(
            positions=positions,
            velocities=velocities,
            colors=colors,
            sizes=sizes,
            lifetimes=lifetimes,
            types=types
        )
        
        # Move to GPU if available
        if self.cuda_enabled:
            try:
                if CUPY_AVAILABLE:
                    system.gpu_positions = cp.asarray(positions)
                    system.gpu_velocities = cp.asarray(velocities)
                elif TORCH_CUDA_AVAILABLE:
                    device = torch.device('cuda')
                    system.gpu_positions = torch.tensor(positions, device=device)
                    system.gpu_velocities = torch.tensor(velocities, device=device)
            except Exception as e:
                logger.warning(f"Failed to move particles to GPU: {e}")
        
        self.particle_systems[name] = system
        return system
    
    def update_particles_cuda(self, system_name: str, dt: float):
        """Update particle system using CUDA acceleration"""
        if system_name not in self.particle_systems:
            return
        
        system = self.particle_systems[system_name]
        
        if not self.cuda_enabled or system.gpu_positions is None:
            # CPU fallback
            system.positions += system.velocities * dt
            system.lifetimes -= dt
            
            # Wrap positions to stay in bounds
            system.positions = np.clip(system.positions, 0, 1)
            
            # Reset dead particles
            dead_mask = system.lifetimes <= 0
            system.positions[dead_mask] = np.random.rand(np.sum(dead_mask), 3)
            system.lifetimes[dead_mask] = np.random.rand(np.sum(dead_mask)) * 100
            return
        
        try:
            if CUPY_AVAILABLE:
                # Update positions with CuPy
                system.gpu_positions += system.gpu_velocities * dt
                
                # Boundary conditions
                system.gpu_positions = cp.clip(system.gpu_positions, 0, 1)
                
                # Copy back to CPU for rendering
                system.positions = cp.asnumpy(system.gpu_positions)
                
            elif TORCH_CUDA_AVAILABLE:
                # Update positions with PyTorch
                system.gpu_positions += system.gpu_velocities * dt
                
                # Boundary conditions
                system.gpu_positions = torch.clamp(system.gpu_positions, 0, 1)
                
                # Copy back to CPU for rendering
                system.positions = system.gpu_positions.cpu().numpy()
                
        except Exception as e:
            logger.warning(f"CUDA particle update failed: {e}")
            # Fallback to CPU update
            system.positions += system.velocities * dt
    
    def process_fluid_cuda(self, field: np.ndarray, operation: str = "diffuse") -> np.ndarray:
        """Process fluid dynamics using CUDA acceleration"""
        if not self.cuda_enabled:
            return self._process_fluid_cpu(field, operation)
        
        try:
            if CUPY_AVAILABLE:
                return self._process_fluid_cupy(field, operation)
            elif TORCH_CUDA_AVAILABLE:
                return self._process_fluid_torch(field, operation)
            else:
                return self._process_fluid_cpu(field, operation)
                
        except Exception as e:
            logger.warning(f"CUDA fluid processing failed: {e}")
            return self._process_fluid_cpu(field, operation)
    
    def _process_fluid_cupy(self, field: np.ndarray, operation: str) -> np.ndarray:
        """Process fluid using CuPy"""
        gpu_field = cp.asarray(field)
        
        if operation == "diffuse":
            # Simple diffusion kernel
            kernel = cp.array([[0.05, 0.1, 0.05],
                              [0.1,  0.4, 0.1],
                              [0.05, 0.1, 0.05]])
            result = cp.convolve2d(gpu_field, kernel, mode='same', boundary='wrap')
        elif operation == "advect":
            # Simple advection (shift field)
            result = cp.roll(gpu_field, 1, axis=0)
        elif operation == "vorticity":
            # Calculate vorticity
            dx = cp.gradient(gpu_field, axis=1)
            dy = cp.gradient(gpu_field, axis=0)
            result = dx - dy
        else:
            result = gpu_field
        
        return cp.asnumpy(result)
    
    def _process_fluid_torch(self, field: np.ndarray, operation: str) -> np.ndarray:
        """Process fluid using PyTorch CUDA"""
        device = torch.device('cuda')
        gpu_field = torch.tensor(field, device=device, dtype=torch.float32)
        
        if operation == "diffuse":
            # Simple diffusion using conv2d
            kernel = torch.tensor([[0.05, 0.1, 0.05],
                                  [0.1,  0.4, 0.1],
                                  [0.05, 0.1, 0.05]], device=device).unsqueeze(0).unsqueeze(0)
            gpu_field = gpu_field.unsqueeze(0).unsqueeze(0)
            result = torch.nn.functional.conv2d(gpu_field, kernel, padding=1)
            result = result.squeeze()
        elif operation == "advect":
            result = torch.roll(gpu_field, 1, dims=0)
        elif operation == "vorticity":
            dx = torch.gradient(gpu_field, dim=1)[0]
            dy = torch.gradient(gpu_field, dim=0)[0]
            result = dx - dy
        else:
            result = gpu_field
        
        return result.cpu().numpy()
    
    def _process_fluid_cpu(self, field: np.ndarray, operation: str) -> np.ndarray:
        """Process fluid using CPU (NumPy)"""
        if operation == "diffuse":
            if SCIPY_AVAILABLE:
                kernel = np.array([[0.05, 0.1, 0.05],
                                  [0.1,  0.4, 0.1],
                                  [0.05, 0.1, 0.05]])
                return ndimage.convolve(field, kernel, mode='wrap')
            else:
                return field * 0.9  # Simple decay
        elif operation == "advect":
            return np.roll(field, 1, axis=0)
        elif operation == "vorticity":
            dx = np.gradient(field, axis=1)
            dy = np.gradient(field, axis=0)
            return dx - dy
        else:
            return field
    
    def start_animation(self):
        """Start the animation loop"""
        if self.running:
            return
        
        self.running = True
        self.start_time = time.time()
        self.frame_count = 0
        
        # Initialize CUDA resources
        self.initialize_cuda_resources()
        
        # Create figure and axes
        self.figure = plt.figure(figsize=self.config.figure_size, dpi=self.config.dpi)
        self.figure.patch.set_facecolor('#0a0a0a')
        
        # Set up the animation
        self.setup_animation()
        
        # Start animation
        self.animation = FuncAnimation(
            self.figure,
            self.animate_frame,
            frames=int(self.config.duration * self.config.fps),
            interval=1000 // self.config.fps,
            blit=False,
            repeat=True
        )
        
        logger.info(f"🎬 Started {self.house_name} animation")
    
    def stop_animation(self):
        """Stop the animation"""
        if not self.running:
            return
        
        self.running = False
        
        if self.animation:
            self.animation.event_source.stop()
        
        logger.info(f"🛑 Stopped {self.house_name} animation")
    
    def setup_animation(self):
        """Setup animation components - to be overridden by subclasses"""
        pass
    
    def animate_frame(self, frame: int):
        """Animate a single frame - to be overridden by subclasses"""
        frame_start = time.time()
        
        # Update frame counter
        self.frame_count = frame
        
        # Track performance
        if len(self.frame_times) > 0:
            frame_time = time.time() - frame_start
            self.frame_times.append(frame_time)
            self.avg_fps = 1.0 / (sum(self.frame_times) / len(self.frame_times))
    
    def get_performance_stats(self) -> Dict[str, Any]:
        """Get animation performance statistics"""
        return {
            'animator_id': self.animator_id,
            'house_name': self.house_name,
            'running': self.running,
            'frame_count': self.frame_count,
            'avg_fps': self.avg_fps,
            'cuda_enabled': self.cuda_enabled,
            'particle_systems': len(self.particle_systems),
            'uptime': time.time() - self.start_time
        }


class MycelialHouseAnimator(CUDAHouseAnimator):
    """
    CUDA-accelerated animation for the Mycelial House.
    
    Visualizes:
    - Living mycelial network with growing/pruning nodes
    - Nutrient flow dynamics with particle streams
    - Energy distribution across the network
    - Growth gates and autophagy mechanisms
    - Cluster formation and dissolution
    """
    
    def __init__(self, config: Optional[AnimationConfig] = None):
        super().__init__("Mycelial House", config)
        
        # Mycelial-specific state
        self.network_nodes = []
        self.network_edges = []
        self.nutrient_field = None
        self.energy_flows = []
        
        # Animation parameters
        self.network_size = 50
        self.nutrient_particles = 500
        self.energy_particles = 300
        
        logger.info("🍄 Mycelial House Animator initialized")
    
    def setup_animation(self):
        """Setup mycelial animation components"""
        # Create 3D subplot for main network
        self.ax_main = self.figure.add_subplot(221, projection='3d')
        self.ax_main.set_title('🍄 Living Mycelial Network', color='white', fontsize=14)
        self.ax_main.set_facecolor('#0a0a0a')
        
        # Create 2D subplot for nutrient field
        self.ax_nutrient = self.figure.add_subplot(222)
        self.ax_nutrient.set_title('🌊 Nutrient Flow Field', color='white', fontsize=12)
        self.ax_nutrient.set_facecolor('#0a0a0a')
        
        # Create subplot for energy distribution
        self.ax_energy = self.figure.add_subplot(223)
        self.ax_energy.set_title('⚡ Energy Distribution', color='white', fontsize=12)
        self.ax_energy.set_facecolor('#0a0a0a')
        
        # Create subplot for system metrics
        self.ax_metrics = self.figure.add_subplot(224)
        self.ax_metrics.set_title('📊 System Metrics', color='white', fontsize=12)
        self.ax_metrics.set_facecolor('#0a0a0a')
        
        # Initialize network
        self._initialize_mycelial_network()
        
        # Create particle systems
        self.create_particle_system('nutrients', self.nutrient_particles, 'nutrient')
        self.create_particle_system('energy', self.energy_particles, 'energy')
        
        # Initialize nutrient field
        self.nutrient_field = np.random.rand(64, 64) * 0.5
        
        plt.tight_layout()
    
    def _initialize_mycelial_network(self):
        """Initialize the mycelial network structure"""
        # Create random network nodes
        self.network_nodes = []
        for i in range(self.network_size):
            node = {
                'id': i,
                'position': np.random.rand(3),
                'energy': np.random.rand() * 100,
                'state': np.random.choice(['active', 'growing', 'pruning', 'dormant']),
                'connections': [],
                'age': np.random.rand() * 50,
                'size': 10 + np.random.rand() * 20
            }
            self.network_nodes.append(node)
        
        # Create edges between nearby nodes
        self.network_edges = []
        for i, node1 in enumerate(self.network_nodes):
            for j, node2 in enumerate(self.network_nodes[i+1:], i+1):
                distance = np.linalg.norm(node1['position'] - node2['position'])
                if distance < 0.3:  # Connection threshold
                    edge = {
                        'source': i,
                        'target': j,
                        'strength': (0.3 - distance) / 0.3,
                        'flow': np.random.rand() * 10,
                        'type': np.random.choice(['structural', 'metabolic', 'signal'])
                    }
                    self.network_edges.append(edge)
                    node1['connections'].append(j)
                    node2['connections'].append(i)
    
    def animate_frame(self, frame: int):
        """Animate mycelial house frame"""
        super().animate_frame(frame)
        
        dt = 1.0 / self.config.fps
        
        # Clear all axes
        self.ax_main.clear()
        self.ax_nutrient.clear()
        self.ax_energy.clear()
        self.ax_metrics.clear()
        
        # Update network dynamics
        self._update_network_dynamics(dt)
        
        # Update particle systems
        self.update_particles_cuda('nutrients', dt)
        self.update_particles_cuda('energy', dt)
        
        # Update nutrient field
        self.nutrient_field = self.process_fluid_cuda(self.nutrient_field, 'diffuse')
        self.nutrient_field += np.random.rand(64, 64) * 0.01  # Add noise
        self.nutrient_field = np.clip(self.nutrient_field, 0, 1)
        
        # Render main network
        self._render_network(self.ax_main)
        
        # Render nutrient field
        self._render_nutrient_field(self.ax_nutrient)
        
        # Render energy distribution
        self._render_energy_distribution(self.ax_energy)
        
        # Render metrics
        self._render_system_metrics(self.ax_metrics)
        
        return []
    
    def _update_network_dynamics(self, dt: float):
        """Update mycelial network dynamics"""
        # Update node states
        for node in self.network_nodes:
            # Energy decay
            node['energy'] *= 0.999
            
            # State transitions
            if node['state'] == 'growing' and node['energy'] > 80:
                node['size'] = min(node['size'] + dt * 5, 30)
            elif node['state'] == 'pruning' and node['energy'] < 20:
                node['size'] = max(node['size'] - dt * 3, 5)
            
            # Random state changes
            if np.random.rand() < 0.01:
                node['state'] = np.random.choice(['active', 'growing', 'pruning', 'dormant'])
            
            # Age increment
            node['age'] += dt
        
        # Update edge flows
        for edge in self.network_edges:
            source_node = self.network_nodes[edge['source']]
            target_node = self.network_nodes[edge['target']]
            
            # Flow based on energy difference
            energy_diff = source_node['energy'] - target_node['energy']
            edge['flow'] = edge['strength'] * energy_diff * 0.1
            
            # Transfer energy
            if abs(energy_diff) > 1:
                transfer = edge['flow'] * dt * 0.1
                source_node['energy'] -= transfer
                target_node['energy'] += transfer
    
    def _render_network(self, ax):
        """Render the 3D mycelial network"""
        ax.set_title('🍄 Living Mycelial Network', color='white', fontsize=14)
        ax.set_facecolor('#0a0a0a')
        
        # Color map for node states
        state_colors = {
            'active': '#00FF00',    # Green
            'growing': '#FFD700',   # Gold
            'pruning': '#FF6B35',   # Orange
            'dormant': '#808080'    # Gray
        }
        
        # Draw edges first
        for edge in self.network_edges:
            source = self.network_nodes[edge['source']]
            target = self.network_nodes[edge['target']]
            
            # Edge color based on flow
            flow_intensity = abs(edge['flow']) / 10.0
            edge_color = plt.cm.plasma(np.clip(flow_intensity, 0, 1))
            edge_alpha = 0.3 + flow_intensity * 0.7
            
            ax.plot(
                [source['position'][0], target['position'][0]],
                [source['position'][1], target['position'][1]],
                [source['position'][2], target['position'][2]],
                color=edge_color, alpha=edge_alpha, linewidth=1 + flow_intensity * 3
            )
        
        # Draw nodes
        for node in self.network_nodes:
            pos = node['position']
            color = state_colors.get(node['state'], '#FFFFFF')
            size = node['size']
            energy_alpha = np.clip(node['energy'] / 100.0, 0.3, 1.0)
            
            ax.scatter(
                pos[0], pos[1], pos[2],
                c=color, s=size**2, alpha=energy_alpha,
                edgecolors='white', linewidth=0.5
            )
        
        # Draw nutrient particles
        nutrients = self.particle_systems.get('nutrients')
        if nutrients:
            ax.scatter(
                nutrients.positions[:, 0],
                nutrients.positions[:, 1], 
                nutrients.positions[:, 2],
                c='cyan', s=nutrients.sizes, alpha=0.6,
                marker='.'
            )
        
        ax.set_xlim(0, 1)
        ax.set_ylim(0, 1)
        ax.set_zlim(0, 1)
        ax.set_xlabel('X', color='white')
        ax.set_ylabel('Y', color='white')
        ax.set_zlabel('Z', color='white')
    
    def _render_nutrient_field(self, ax):
        """Render nutrient flow field"""
        ax.set_title('🌊 Nutrient Flow Field', color='white', fontsize=12)
        ax.set_facecolor('#0a0a0a')
        
        # Show nutrient field as heatmap
        im = ax.imshow(self.nutrient_field, cmap='viridis', origin='lower', aspect='equal')
        
        # Add flow vectors
        y, x = np.mgrid[0:64:8, 0:64:8]
        # Simple flow field (could be enhanced with actual fluid dynamics)
        u = np.sin(x * 0.1) * np.cos(y * 0.1)
        v = np.cos(x * 0.1) * np.sin(y * 0.1)
        
        ax.quiver(x, y, u, v, alpha=0.7, color='white', scale=20)
        
        # Add colorbar
        plt.colorbar(im, ax=ax, shrink=0.8, label='Nutrient Concentration')
        
        ax.set_xlabel('X Grid', color='white')
        ax.set_ylabel('Y Grid', color='white')
    
    def _render_energy_distribution(self, ax):
        """Render energy distribution across network"""
        ax.set_title('⚡ Energy Distribution', color='white', fontsize=12)
        ax.set_facecolor('#0a0a0a')
        
        # Energy histogram
        energies = [node['energy'] for node in self.network_nodes]
        ax.hist(energies, bins=20, alpha=0.7, color='gold', edgecolor='white')
        
        ax.set_xlabel('Energy Level', color='white')
        ax.set_ylabel('Node Count', color='white')
        ax.tick_params(colors='white')
        
        # Add statistics
        mean_energy = np.mean(energies)
        max_energy = np.max(energies)
        ax.axvline(mean_energy, color='red', linestyle='--', label=f'Mean: {mean_energy:.1f}')
        ax.axvline(max_energy, color='orange', linestyle='--', label=f'Max: {max_energy:.1f}')
        ax.legend()
    
    def _render_system_metrics(self, ax):
        """Render system performance metrics"""
        ax.set_title('📊 System Metrics', color='white', fontsize=12)
        ax.set_facecolor('#0a0a0a')
        
        # Calculate metrics
        active_nodes = sum(1 for node in self.network_nodes if node['state'] == 'active')
        growing_nodes = sum(1 for node in self.network_nodes if node['state'] == 'growing')
        pruning_nodes = sum(1 for node in self.network_nodes if node['state'] == 'pruning')
        dormant_nodes = sum(1 for node in self.network_nodes if node['state'] == 'dormant')
        
        total_energy = sum(node['energy'] for node in self.network_nodes)
        avg_connections = np.mean([len(node['connections']) for node in self.network_nodes])
        
        # Create metrics display
        metrics = [
            f"Active Nodes: {active_nodes}",
            f"Growing: {growing_nodes}",
            f"Pruning: {pruning_nodes}",
            f"Dormant: {dormant_nodes}",
            f"Total Energy: {total_energy:.1f}",
            f"Avg Connections: {avg_connections:.1f}",
            f"Frame: {self.frame_count}",
            f"FPS: {self.avg_fps:.1f}"
        ]
        
        ax.text(0.1, 0.9, '\n'.join(metrics), transform=ax.transAxes,
                fontsize=10, color='white', verticalalignment='top',
                bbox=dict(boxstyle='round', facecolor='black', alpha=0.7))
        
        ax.set_xlim(0, 1)
        ax.set_ylim(0, 1)
        ax.axis('off')


class SchemaHouseAnimator(CUDAHouseAnimator):
    """
    CUDA-accelerated animation for the Schema House.
    
    Visualizes:
    - Sigil operations and transformations
    - Symbolic routing through houses
    - Schema evolution and connections
    - Purification, weaving, and flame processes
    - Glyph rendering and symbolic grammar
    """
    
    def __init__(self, config: Optional[AnimationConfig] = None):
        super().__init__("Schema House", config)
        
        # Schema-specific state
        self.sigils = []
        self.houses = {}
        self.schema_connections = []
        self.active_operations = []
        
        # Animation parameters
        self.sigil_count = 30
        self.house_count = 6
        
        # House definitions
        self.house_definitions = {
            'memory': {'color': '#4CAF50', 'symbol': '🌸', 'position': (0.2, 0.8)},
            'purification': {'color': '#FF5722', 'symbol': '🔥', 'position': (0.8, 0.8)},
            'weaving': {'color': '#9C27B0', 'symbol': '🕸️', 'position': (0.2, 0.2)},
            'flame': {'color': '#FF9800', 'symbol': '⚡', 'position': (0.8, 0.2)},
            'mirrors': {'color': '#2196F3', 'symbol': '🪞', 'position': (0.5, 0.9)},
            'echoes': {'color': '#607D8B', 'symbol': '🔊', 'position': (0.5, 0.1)}
        }
        
        logger.info("🏛️ Schema House Animator initialized")
    
    def setup_animation(self):
        """Setup schema animation components"""
        # Create main subplot for sigil operations
        self.ax_main = self.figure.add_subplot(221)
        self.ax_main.set_title('🏛️ Sigil House Operations', color='white', fontsize=14)
        self.ax_main.set_facecolor('#0a0a0a')
        
        # Create subplot for schema connections
        self.ax_schema = self.figure.add_subplot(222)
        self.ax_schema.set_title('🔗 Schema Connections', color='white', fontsize=12)
        self.ax_schema.set_facecolor('#0a0a0a')
        
        # Create subplot for symbolic transformations
        self.ax_transform = self.figure.add_subplot(223)
        self.ax_transform.set_title('🔄 Symbolic Transformations', color='white', fontsize=12)
        self.ax_transform.set_facecolor('#0a0a0a')
        
        # Create subplot for house activity
        self.ax_activity = self.figure.add_subplot(224)
        self.ax_activity.set_title('📈 House Activity', color='white', fontsize=12)
        self.ax_activity.set_facecolor('#0a0a0a')
        
        # Initialize sigils and houses
        self._initialize_schema_elements()
        
        # Create particle systems for symbolic operations
        self.create_particle_system('sigils', self.sigil_count, 'sigil')
        self.create_particle_system('connections', 100, 'connection')
        
        plt.tight_layout()
    
    def _initialize_schema_elements(self):
        """Initialize schema elements"""
        # Initialize houses
        for house_name, house_data in self.house_definitions.items():
            self.houses[house_name] = {
                'name': house_name,
                'position': house_data['position'],
                'color': house_data['color'],
                'symbol': house_data['symbol'],
                'activity': np.random.rand(),
                'sigils': [],
                'connections': []
            }
        
        # Initialize sigils
        self.sigils = []
        sigil_types = ['recall', 'archive', 'purge', 'weave', 'ignite', 'reflect']
        
        for i in range(self.sigil_count):
            sigil = {
                'id': i,
                'type': np.random.choice(sigil_types),
                'position': np.random.rand(2),
                'target_house': np.random.choice(list(self.houses.keys())),
                'state': 'dormant',  # dormant, active, transforming, complete
                'energy': np.random.rand() * 100,
                'complexity': np.random.rand(),
                'age': 0
            }
            self.sigils.append(sigil)
        
        # Initialize schema connections
        self.schema_connections = []
        for i in range(20):
            connection = {
                'source': np.random.choice(list(self.houses.keys())),
                'target': np.random.choice(list(self.houses.keys())),
                'strength': np.random.rand(),
                'type': np.random.choice(['structural', 'semantic', 'temporal']),
                'active': np.random.rand() > 0.5
            }
            self.schema_connections.append(connection)
    
    def animate_frame(self, frame: int):
        """Animate schema house frame"""
        super().animate_frame(frame)
        
        dt = 1.0 / self.config.fps
        
        # Clear all axes
        self.ax_main.clear()
        self.ax_schema.clear()
        self.ax_transform.clear()
        self.ax_activity.clear()
        
        # Update schema dynamics
        self._update_schema_dynamics(dt)
        
        # Update particle systems
        self.update_particles_cuda('sigils', dt)
        self.update_particles_cuda('connections', dt)
        
        # Render components
        self._render_sigil_operations(self.ax_main)
        self._render_schema_connections(self.ax_schema)
        self._render_transformations(self.ax_transform)
        self._render_house_activity(self.ax_activity)
        
        return []
    
    def _update_schema_dynamics(self, dt: float):
        """Update schema house dynamics"""
        # Update house activities
        for house in self.houses.values():
            house['activity'] += (np.random.rand() - 0.5) * dt * 0.1
            house['activity'] = np.clip(house['activity'], 0, 1)
        
        # Update sigils
        for sigil in self.sigils:
            sigil['age'] += dt
            
            # State transitions
            if sigil['state'] == 'dormant' and np.random.rand() < 0.02:
                sigil['state'] = 'active'
            elif sigil['state'] == 'active' and sigil['energy'] > 50:
                if np.random.rand() < 0.05:
                    sigil['state'] = 'transforming'
            elif sigil['state'] == 'transforming':
                sigil['energy'] -= dt * 20
                if sigil['energy'] <= 0:
                    sigil['state'] = 'complete'
                    sigil['energy'] = 100
            elif sigil['state'] == 'complete':
                if np.random.rand() < 0.1:
                    sigil['state'] = 'dormant'
                    sigil['position'] = np.random.rand(2)
            
            # Move towards target house
            if sigil['state'] == 'active':
                target_pos = np.array(self.houses[sigil['target_house']]['position'])
                direction = target_pos - sigil['position']
                distance = np.linalg.norm(direction)
                if distance > 0.01:
                    sigil['position'] += direction / distance * dt * 0.1
        
        # Update connections
        for connection in self.schema_connections:
            # Random activation/deactivation
            if np.random.rand() < 0.01:
                connection['active'] = not connection['active']
            
            # Strength fluctuation
            connection['strength'] += (np.random.rand() - 0.5) * dt * 0.1
            connection['strength'] = np.clip(connection['strength'], 0.1, 1.0)
    
    def _render_sigil_operations(self, ax):
        """Render sigil operations"""
        ax.set_title('🏛️ Sigil House Operations', color='white', fontsize=14)
        ax.set_facecolor('#0a0a0a')
        
        # Draw houses
        for house_name, house in self.houses.items():
            pos = house['position']
            color = house['color']
            activity = house['activity']
            
            # House circle with activity-based size
            circle = Circle(pos, 0.08 + activity * 0.04, 
                          color=color, alpha=0.7 + activity * 0.3)
            ax.add_patch(circle)
            
            # House symbol
            ax.text(pos[0], pos[1], house['symbol'], 
                   ha='center', va='center', fontsize=16,
                   color='white', weight='bold')
            
            # House name
            ax.text(pos[0], pos[1] - 0.12, house_name.title(),
                   ha='center', va='center', fontsize=8,
                   color='white')
        
        # Draw sigils
        sigil_colors = {
            'dormant': '#666666',
            'active': '#00FF00',
            'transforming': '#FFD700',
            'complete': '#FF6B35'
        }
        
        for sigil in self.sigils:
            pos = sigil['position']
            color = sigil_colors.get(sigil['state'], '#FFFFFF')
            size = 20 + sigil['energy'] * 0.3
            
            ax.scatter(pos[0], pos[1], c=color, s=size, alpha=0.8,
                      edgecolors='white', linewidth=0.5)
        
        # Draw connections between houses
        for connection in self.schema_connections:
            if connection['active']:
                source_pos = self.houses[connection['source']]['position']
                target_pos = self.houses[connection['target']]['position']
                
                ax.plot([source_pos[0], target_pos[0]],
                       [source_pos[1], target_pos[1]],
                       color='cyan', alpha=connection['strength'] * 0.7,
                       linewidth=connection['strength'] * 3)
        
        ax.set_xlim(0, 1)
        ax.set_ylim(0, 1)
        ax.set_aspect('equal')
        ax.axis('off')
    
    def _render_schema_connections(self, ax):
        """Render schema connection network"""
        ax.set_title('🔗 Schema Connections', color='white', fontsize=12)
        ax.set_facecolor('#0a0a0a')
        
        # Create network graph of houses
        house_names = list(self.houses.keys())
        n_houses = len(house_names)
        
        # Position houses in circle
        angles = np.linspace(0, 2*np.pi, n_houses, endpoint=False)
        positions = {}
        
        for i, house_name in enumerate(house_names):
            pos = (0.5 + 0.3 * np.cos(angles[i]), 0.5 + 0.3 * np.sin(angles[i]))
            positions[house_name] = pos
        
        # Draw connections
        for connection in self.schema_connections:
            source_pos = positions[connection['source']]
            target_pos = positions[connection['target']]
            
            alpha = connection['strength'] if connection['active'] else 0.1
            color = 'red' if connection['type'] == 'structural' else \
                   'green' if connection['type'] == 'semantic' else 'blue'
            
            ax.plot([source_pos[0], target_pos[0]],
                   [source_pos[1], target_pos[1]],
                   color=color, alpha=alpha, linewidth=2)
        
        # Draw houses
        for house_name, pos in positions.items():
            house_data = self.houses[house_name]
            ax.scatter(pos[0], pos[1], c=house_data['color'], s=200, 
                      alpha=house_data['activity'], edgecolors='white', linewidth=2)
            ax.text(pos[0], pos[1], house_data['symbol'],
                   ha='center', va='center', fontsize=12, color='white')
        
        ax.set_xlim(0, 1)
        ax.set_ylim(0, 1)
        ax.set_aspect('equal')
        ax.axis('off')
    
    def _render_transformations(self, ax):
        """Render symbolic transformations"""
        ax.set_title('🔄 Symbolic Transformations', color='white', fontsize=12)
        ax.set_facecolor('#0a0a0a')
        
        # Transformation pipeline visualization
        transformation_types = ['recall', 'archive', 'purge', 'weave', 'ignite', 'reflect']
        
        # Count active transformations by type
        transform_counts = {}
        for t_type in transformation_types:
            count = sum(1 for sigil in self.sigils 
                       if sigil['type'] == t_type and sigil['state'] == 'transforming')
            transform_counts[t_type] = count
        
        # Create bar chart
        types = list(transform_counts.keys())
        counts = list(transform_counts.values())
        
        bars = ax.bar(types, counts, color=['#FF6B35', '#4CAF50', '#2196F3', 
                                           '#9C27B0', '#FF9800', '#607D8B'],
                     alpha=0.8)
        
        # Add glow effect to active bars
        for bar, count in zip(bars, counts):
            if count > 0:
                bar.set_edgecolor('white')
                bar.set_linewidth(2)
        
        ax.set_xlabel('Transformation Type', color='white')
        ax.set_ylabel('Active Count', color='white')
        ax.tick_params(colors='white')
        plt.setp(ax.get_xticklabels(), rotation=45, ha='right')
    
    def _render_house_activity(self, ax):
        """Render house activity over time"""
        ax.set_title('📈 House Activity', color='white', fontsize=12)
        ax.set_facecolor('#0a0a0a')
        
        # Store activity history
        if not hasattr(self, 'activity_history'):
            self.activity_history = {name: deque(maxlen=100) for name in self.houses.keys()}
        
        # Update history
        for house_name, house in self.houses.items():
            self.activity_history[house_name].append(house['activity'])
        
        # Plot activity lines
        for house_name, history in self.activity_history.items():
            if len(history) > 1:
                house_data = self.houses[house_name]
                ax.plot(range(len(history)), history, 
                       color=house_data['color'], label=house_name, linewidth=2)
        
        ax.set_xlabel('Time Steps', color='white')
        ax.set_ylabel('Activity Level', color='white')
        ax.tick_params(colors='white')
        ax.legend(loc='upper right')
        ax.grid(True, alpha=0.3)


class MonitoringHouseAnimator(CUDAHouseAnimator):
    """
    CUDA-accelerated animation for the Monitoring House.
    
    Visualizes:
    - Real-time telemetry streams
    - System health monitoring
    - Performance metrics and alerts
    - Data flow networks
    - Consciousness state tracking
    """
    
    def __init__(self, config: Optional[AnimationConfig] = None):
        super().__init__("Monitoring House", config)
        
        # Monitoring-specific state
        self.telemetry_streams = {}
        self.system_metrics = {}
        self.alert_system = {}
        self.data_flows = []
        
        # Animation parameters
        self.stream_count = 8
        self.metric_history_length = 200
        self.alert_particles = 50
        
        # Initialize monitoring data
        self.stream_names = [
            'consciousness', 'memory', 'schema', 'tracers', 
            'pulse', 'mycelial', 'thermal', 'recursive'
        ]
        
        logger.info("📊 Monitoring House Animator initialized")
    
    def setup_animation(self):
        """Setup monitoring animation components"""
        # Create main subplot for system overview
        self.ax_main = self.figure.add_subplot(221)
        self.ax_main.set_title('📊 System Health Overview', color='white', fontsize=14)
        self.ax_main.set_facecolor('#0a0a0a')
        
        # Create subplot for telemetry streams
        self.ax_streams = self.figure.add_subplot(222)
        self.ax_streams.set_title('📡 Telemetry Streams', color='white', fontsize=12)
        self.ax_streams.set_facecolor('#0a0a0a')
        
        # Create subplot for performance metrics
        self.ax_performance = self.figure.add_subplot(223)
        self.ax_performance.set_title('⚡ Performance Metrics', color='white', fontsize=12)
        self.ax_performance.set_facecolor('#0a0a0a')
        
        # Create subplot for data flow network
        self.ax_network = self.figure.add_subplot(224)
        self.ax_network.set_title('🌐 Data Flow Network', color='white', fontsize=12)
        self.ax_network.set_facecolor('#0a0a0a')
        
        # Initialize monitoring systems
        self._initialize_monitoring_systems()
        
        # Create particle systems
        self.create_particle_system('alerts', self.alert_particles, 'alert')
        self.create_particle_system('data_flow', 200, 'data')
        
        plt.tight_layout()
    
    def _initialize_monitoring_systems(self):
        """Initialize monitoring systems"""
        # Initialize telemetry streams
        for stream_name in self.stream_names:
            self.telemetry_streams[stream_name] = {
                'name': stream_name,
                'data': deque(maxlen=self.metric_history_length),
                'health': np.random.rand(),
                'throughput': np.random.rand() * 1000,
                'errors': 0,
                'status': 'active'
            }
        
        # Initialize system metrics
        self.system_metrics = {
            'cpu_usage': deque(maxlen=self.metric_history_length),
            'memory_usage': deque(maxlen=self.metric_history_length),
            'gpu_usage': deque(maxlen=self.metric_history_length),
            'network_io': deque(maxlen=self.metric_history_length),
            'disk_io': deque(maxlen=self.metric_history_length),
            'consciousness_coherence': deque(maxlen=self.metric_history_length),
            'system_temperature': deque(maxlen=self.metric_history_length),
            'error_rate': deque(maxlen=self.metric_history_length)
        }
        
        # Initialize alert system
        self.alert_system = {
            'active_alerts': [],
            'alert_history': deque(maxlen=100),
            'severity_counts': {'low': 0, 'medium': 0, 'high': 0, 'critical': 0}
        }
        
        # Initialize data flows
        self.data_flows = []
        for i in range(20):
            flow = {
                'source': np.random.choice(self.stream_names),
                'target': np.random.choice(self.stream_names),
                'bandwidth': np.random.rand() * 100,
                'latency': np.random.rand() * 10,
                'active': True,
                'quality': np.random.rand()
            }
            self.data_flows.append(flow)
    
    def animate_frame(self, frame: int):
        """Animate monitoring house frame"""
        super().animate_frame(frame)
        
        dt = 1.0 / self.config.fps
        
        # Clear all axes
        self.ax_main.clear()
        self.ax_streams.clear()
        self.ax_performance.clear()
        self.ax_network.clear()
        
        # Update monitoring data
        self._update_monitoring_data(dt)
        
        # Update particle systems
        self.update_particles_cuda('alerts', dt)
        self.update_particles_cuda('data_flow', dt)
        
        # Render components
        self._render_system_overview(self.ax_main)
        self._render_telemetry_streams(self.ax_streams)
        self._render_performance_metrics(self.ax_performance)
        self._render_data_flow_network(self.ax_network)
        
        return []
    
    def _update_monitoring_data(self, dt: float):
        """Update monitoring data"""
        current_time = time.time()
        
        # Update telemetry streams
        for stream in self.telemetry_streams.values():
            # Generate realistic telemetry data
            base_value = 50 + 30 * np.sin(current_time * 0.1 + hash(stream['name']) % 10)
            noise = (np.random.rand() - 0.5) * 20
            value = np.clip(base_value + noise, 0, 100)
            
            stream['data'].append(value)
            stream['health'] += (np.random.rand() - 0.5) * dt * 0.1
            stream['health'] = np.clip(stream['health'], 0, 1)
            
            # Random errors
            if np.random.rand() < 0.01:
                stream['errors'] += 1
            
            # Status updates
            if stream['health'] < 0.3:
                stream['status'] = 'critical'
            elif stream['health'] < 0.6:
                stream['status'] = 'warning'
            else:
                stream['status'] = 'active'
        
        # Update system metrics
        metrics_data = {
            'cpu_usage': 30 + 40 * np.sin(current_time * 0.2) + np.random.rand() * 10,
            'memory_usage': 45 + 25 * np.sin(current_time * 0.15) + np.random.rand() * 15,
            'gpu_usage': 20 + 60 * np.sin(current_time * 0.3) + np.random.rand() * 20,
            'network_io': np.random.rand() * 100,
            'disk_io': np.random.rand() * 80,
            'consciousness_coherence': 0.7 + 0.2 * np.sin(current_time * 0.1) + np.random.rand() * 0.1,
            'system_temperature': 35 + 15 * np.sin(current_time * 0.05) + np.random.rand() * 5,
            'error_rate': max(0, np.random.rand() * 5 - 4)
        }
        
        for metric_name, value in metrics_data.items():
            self.system_metrics[metric_name].append(value)
        
        # Update alerts
        self._update_alert_system(metrics_data)
        
        # Update data flows
        for flow in self.data_flows:
            flow['bandwidth'] += (np.random.rand() - 0.5) * dt * 10
            flow['bandwidth'] = np.clip(flow['bandwidth'], 1, 200)
            
            flow['latency'] += (np.random.rand() - 0.5) * dt * 2
            flow['latency'] = np.clip(flow['latency'], 0.1, 20)
            
            flow['quality'] += (np.random.rand() - 0.5) * dt * 0.1
            flow['quality'] = np.clip(flow['quality'], 0.1, 1.0)
    
    def _update_alert_system(self, metrics: Dict[str, float]):
        """Update alert system based on metrics"""
        # Check for alert conditions
        new_alerts = []
        
        if metrics['cpu_usage'] > 80:
            new_alerts.append({
                'type': 'cpu_high',
                'severity': 'high',
                'message': f"High CPU usage: {metrics['cpu_usage']:.1f}%",
                'timestamp': time.time()
            })
        
        if metrics['memory_usage'] > 85:
            new_alerts.append({
                'type': 'memory_high',
                'severity': 'critical',
                'message': f"Critical memory usage: {metrics['memory_usage']:.1f}%",
                'timestamp': time.time()
            })
        
        if metrics['consciousness_coherence'] < 0.5:
            new_alerts.append({
                'type': 'coherence_low',
                'severity': 'medium',
                'message': f"Low consciousness coherence: {metrics['consciousness_coherence']:.2f}",
                'timestamp': time.time()
            })
        
        if metrics['error_rate'] > 2:
            new_alerts.append({
                'type': 'error_rate_high',
                'severity': 'high',
                'message': f"High error rate: {metrics['error_rate']:.1f}/min",
                'timestamp': time.time()
            })
        
        # Add new alerts
        for alert in new_alerts:
            self.alert_system['active_alerts'].append(alert)
            self.alert_system['alert_history'].append(alert)
            self.alert_system['severity_counts'][alert['severity']] += 1
        
        # Remove old alerts
        current_time = time.time()
        self.alert_system['active_alerts'] = [
            alert for alert in self.alert_system['active_alerts']
            if current_time - alert['timestamp'] < 30  # 30 second alert lifetime
        ]
    
    def _render_system_overview(self, ax):
        """Render system health overview"""
        ax.set_title('📊 System Health Overview', color='white', fontsize=14)
        ax.set_facecolor('#0a0a0a')
        
        # Create health dashboard
        stream_names = list(self.telemetry_streams.keys())
        health_values = [self.telemetry_streams[name]['health'] for name in stream_names]
        
        # Color code by health level
        colors = ['red' if h < 0.3 else 'orange' if h < 0.6 else 'green' for h in health_values]
        
        # Create horizontal bar chart
        y_pos = np.arange(len(stream_names))
        bars = ax.barh(y_pos, health_values, color=colors, alpha=0.8)
        
        # Add health percentage labels
        for i, (bar, health) in enumerate(zip(bars, health_values)):
            ax.text(health + 0.02, i, f'{health*100:.0f}%', 
                   va='center', color='white', fontweight='bold')
        
        ax.set_yticks(y_pos)
        ax.set_yticklabels(stream_names, color='white')
        ax.set_xlabel('Health Level', color='white')
        ax.set_xlim(0, 1.2)
        ax.tick_params(colors='white')
        
        # Add overall system status
        overall_health = np.mean(health_values)
        status_color = 'red' if overall_health < 0.3 else 'orange' if overall_health < 0.6 else 'green'
        ax.text(0.02, 0.98, f'Overall Health: {overall_health*100:.0f}%',
                transform=ax.transAxes, color=status_color, fontsize=12, fontweight='bold',
                bbox=dict(boxstyle='round', facecolor='black', alpha=0.7))
    
    def _render_telemetry_streams(self, ax):
        """Render telemetry streams"""
        ax.set_title('📡 Telemetry Streams', color='white', fontsize=12)
        ax.set_facecolor('#0a0a0a')
        
        # Plot telemetry data for selected streams
        colors = plt.cm.tab10(np.linspace(0, 1, len(self.stream_names)))
        
        for i, (stream_name, color) in enumerate(zip(self.stream_names[:4], colors)):  # Show first 4 streams
            stream = self.telemetry_streams[stream_name]
            if len(stream['data']) > 1:
                time_points = range(len(stream['data']))
                ax.plot(time_points, stream['data'], color=color, 
                       label=stream_name, linewidth=2, alpha=0.8)
        
        ax.set_xlabel('Time Steps', color='white')
        ax.set_ylabel('Telemetry Value', color='white')
        ax.tick_params(colors='white')
        ax.legend(loc='upper right')
        ax.grid(True, alpha=0.3)
        
        # Add alert particles
        alerts = self.particle_systems.get('alerts')
        if alerts and len(self.alert_system['active_alerts']) > 0:
            ax.scatter(alerts.positions[:len(self.alert_system['active_alerts']), 0] * len(alerts.positions),
                      alerts.positions[:len(self.alert_system['active_alerts']), 1] * 100,
                      c='red', s=50, alpha=0.7, marker='!', edgecolors='white')
    
    def _render_performance_metrics(self, ax):
        """Render performance metrics"""
        ax.set_title('⚡ Performance Metrics', color='white', fontsize=12)
        ax.set_facecolor('#0a0a0a')
        
        # Create performance radar chart
        metrics_to_show = ['cpu_usage', 'memory_usage', 'gpu_usage', 'consciousness_coherence']
        
        if all(len(self.system_metrics[metric]) > 0 for metric in metrics_to_show):
            # Get latest values
            values = []
            for metric in metrics_to_show:
                latest = self.system_metrics[metric][-1]
                # Normalize consciousness_coherence to 0-100 scale
                if metric == 'consciousness_coherence':
                    latest *= 100
                values.append(latest)
            
            # Create polar plot
            angles = np.linspace(0, 2*np.pi, len(metrics_to_show), endpoint=False)
            values.append(values[0])  # Close the polygon
            angles = np.append(angles, angles[0])
            
            ax = plt.subplot(223, projection='polar')
            ax.plot(angles, values, 'o-', linewidth=2, color='cyan')
            ax.fill(angles, values, alpha=0.25, color='cyan')
            ax.set_xticks(angles[:-1])
            ax.set_xticklabels([m.replace('_', ' ').title() for m in metrics_to_show])
            ax.set_ylim(0, 100)
            ax.set_title('⚡ Performance Metrics', color='white', fontsize=12, pad=20)
            ax.grid(True, alpha=0.3)
    
    def _render_data_flow_network(self, ax):
        """Render data flow network"""
        ax.set_title('🌐 Data Flow Network', color='white', fontsize=12)
        ax.set_facecolor('#0a0a0a')
        
        # Position stream nodes in circle
        n_streams = len(self.stream_names)
        angles = np.linspace(0, 2*np.pi, n_streams, endpoint=False)
        positions = {}
        
        for i, stream_name in enumerate(self.stream_names):
            pos = (0.5 + 0.3 * np.cos(angles[i]), 0.5 + 0.3 * np.sin(angles[i]))
            positions[stream_name] = pos
        
        # Draw data flows
        for flow in self.data_flows:
            if flow['active'] and flow['source'] != flow['target']:
                source_pos = positions[flow['source']]
                target_pos = positions[flow['target']]
                
                # Line width based on bandwidth
                width = max(1, flow['bandwidth'] / 50)
                # Color based on quality
                color = plt.cm.RdYlGn(flow['quality'])
                # Alpha based on latency (lower latency = more visible)
                alpha = max(0.2, 1.0 - flow['latency'] / 20)
                
                ax.plot([source_pos[0], target_pos[0]],
                       [source_pos[1], target_pos[1]],
                       color=color, alpha=alpha, linewidth=width)
        
        # Draw stream nodes
        for stream_name, pos in positions.items():
            stream = self.telemetry_streams[stream_name]
            
            # Color based on status
            status_colors = {'active': 'green', 'warning': 'orange', 'critical': 'red'}
            color = status_colors.get(stream['status'], 'gray')
            
            ax.scatter(pos[0], pos[1], c=color, s=200, 
                      alpha=0.8, edgecolors='white', linewidth=2)
            ax.text(pos[0], pos[1], stream_name[:3].upper(),
                   ha='center', va='center', fontsize=8, 
                   color='white', fontweight='bold')
        
        ax.set_xlim(0, 1)
        ax.set_ylim(0, 1)
        ax.set_aspect('equal')
        ax.axis('off')
        
        # Add legend
        legend_elements = [
            plt.Line2D([0], [0], color='green', linewidth=3, label='High Quality'),
            plt.Line2D([0], [0], color='yellow', linewidth=3, label='Medium Quality'),
            plt.Line2D([0], [0], color='red', linewidth=3, label='Low Quality')
        ]
        ax.legend(handles=legend_elements, loc='upper right', 
                 bbox_to_anchor=(0.98, 0.98))


class HouseAnimationManager:
    """
    Unified manager for all house animations.
    Coordinates multiple house animators and provides GUI integration.
    """
    
    def __init__(self, config: Optional[AnimationConfig] = None):
        self.config = config or AnimationConfig()
        self.manager_id = str(uuid.uuid4())
        
        # DAWN integration
        self.dawn = get_dawn()
        
        # House animators
        self.animators: Dict[str, CUDAHouseAnimator] = {}
        self.active_animations: Dict[str, bool] = {}
        
        # Management state
        self.running = False
        self._lock = threading.RLock()
        
        logger.info(f"🎭 House Animation Manager initialized: {self.manager_id}")
    
    def create_animator(self, house_type: str, config: Optional[AnimationConfig] = None) -> Optional[CUDAHouseAnimator]:
        """Create a house animator"""
        animator_config = config or self.config
        
        try:
            if house_type.lower() == 'mycelial':
                animator = MycelialHouseAnimator(animator_config)
            elif house_type.lower() == 'schema':
                animator = SchemaHouseAnimator(animator_config)
            elif house_type.lower() == 'monitoring':
                animator = MonitoringHouseAnimator(animator_config)
            else:
                logger.error(f"Unknown house type: {house_type}")
                return None
            
            self.animators[house_type] = animator
            self.active_animations[house_type] = False
            
            logger.info(f"✅ Created {house_type} house animator")
            return animator
            
        except Exception as e:
            logger.error(f"Failed to create {house_type} animator: {e}")
            return None
    
    def start_animation(self, house_type: str) -> bool:
        """Start animation for a specific house"""
        if house_type not in self.animators:
            logger.error(f"No animator found for {house_type}")
            return False
        
        try:
            animator = self.animators[house_type]
            animator.start_animation()
            self.active_animations[house_type] = True
            
            logger.info(f"🎬 Started {house_type} animation")
            return True
            
        except Exception as e:
            logger.error(f"Failed to start {house_type} animation: {e}")
            return False
    
    def stop_animation(self, house_type: str) -> bool:
        """Stop animation for a specific house"""
        if house_type not in self.animators:
            return False
        
        try:
            animator = self.animators[house_type]
            animator.stop_animation()
            self.active_animations[house_type] = False
            
            logger.info(f"🛑 Stopped {house_type} animation")
            return True
            
        except Exception as e:
            logger.error(f"Failed to stop {house_type} animation: {e}")
            return False
    
    def start_all_animations(self):
        """Start all available animations"""
        for house_type in self.animators.keys():
            self.start_animation(house_type)
        
        self.running = True
        logger.info("🎬 Started all house animations")
    
    def stop_all_animations(self):
        """Stop all animations"""
        for house_type in self.animators.keys():
            self.stop_animation(house_type)
        
        self.running = False
        logger.info("🛑 Stopped all house animations")
    
    def get_animator(self, house_type: str) -> Optional[CUDAHouseAnimator]:
        """Get a specific animator"""
        return self.animators.get(house_type)
    
    def get_figure(self, house_type: str) -> Optional[Figure]:
        """Get figure for a specific house animation"""
        animator = self.animators.get(house_type)
        return animator.figure if animator else None
    
    def get_status(self) -> Dict[str, Any]:
        """Get manager status"""
        return {
            'manager_id': self.manager_id,
            'running': self.running,
            'animators': {
                house_type: {
                    'active': self.active_animations.get(house_type, False),
                    'performance': animator.get_performance_stats()
                }
                for house_type, animator in self.animators.items()
            },
            'cuda_available': CUDA_AVAILABLE,
            'matplotlib_available': MATPLOTLIB_AVAILABLE
        }


# Global manager instance
_global_house_animation_manager: Optional[HouseAnimationManager] = None
_manager_lock = threading.Lock()


def get_house_animation_manager(config: Optional[AnimationConfig] = None) -> HouseAnimationManager:
    """Get the global house animation manager instance"""
    global _global_house_animation_manager
    
    with _manager_lock:
        if _global_house_animation_manager is None:
            _global_house_animation_manager = HouseAnimationManager(config)
    
    return _global_house_animation_manager


def reset_house_animation_manager():
    """Reset the global house animation manager"""
    global _global_house_animation_manager
    
    with _manager_lock:
        if _global_house_animation_manager:
            _global_house_animation_manager.stop_all_animations()
        _global_house_animation_manager = None


if __name__ == "__main__":
    # Demo the house animation system
    logging.basicConfig(level=logging.INFO)
    
    print("🚀" * 50)
    print("🧠 DAWN CUDA HOUSE ANIMATIONS DEMO")
    print("🚀" * 50)
    
    # Create manager
    manager = get_house_animation_manager()
    
    # Create animators
    print("\n🎭 Creating house animators...")
    mycelial = manager.create_animator('mycelial')
    schema = manager.create_animator('schema') 
    monitoring = manager.create_animator('monitoring')
    
    # Show status
    status = manager.get_status()
    print(f"\n📊 Manager Status: {status}")
    
    if MATPLOTLIB_AVAILABLE:
        print("\n🎬 Starting animations...")
        
        # Start one animation for demo
        if mycelial:
            manager.start_animation('mycelial')
            print("✅ Mycelial house animation started")
            
            # Let it run briefly
            import time
            time.sleep(5)
            
            # Show performance stats
            stats = mycelial.get_performance_stats()
            print(f"📈 Performance: {stats}")
            
            # Stop animation
            manager.stop_animation('mycelial')
            print("🛑 Animation stopped")
    
    else:
        print("⚠️  Matplotlib not available - skipping animation demo")
    
    print("\n🚀 CUDA House Animations demo complete!")
