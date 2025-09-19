#!/usr/bin/env python3
"""
DAWN Matplotlib Bloom Renderer
=============================

Beautiful matplotlib-based renderer for the bloom genealogy network system.
Provides enhanced aesthetics, smooth animations, and publication-quality visuals
while maintaining compatibility with the existing bloom system.

Features:
- High-quality antialiased rendering
- Beautiful color gradients and transparency effects
- Customizable artistic styles and themes
- Export capabilities for scientific publications
- Real-time animation support
- Interactive features with zoom and pan
"""

import numpy as np

# PyTorch imports (optional for this renderer)
try:
    import torch
    import torch.nn as nn
    TORCH_AVAILABLE = True
except ImportError:
    TORCH_AVAILABLE = False
import matplotlib.pyplot as plt
import matplotlib.patches as patches
from matplotlib.collections import LineCollection, PatchCollection
from matplotlib.colors import LinearSegmentedColormap, to_rgba
from matplotlib.animation import FuncAnimation
import matplotlib.patheffects as path_effects
from matplotlib.patches import Circle, FancyBboxPatch, Polygon
from matplotlib.gridspec import GridSpec
import seaborn as sns
import math
import time
from typing import Dict, List, Tuple, Optional, Any, Union
from dataclasses import dataclass, field
from datetime import datetime
import logging
from pathlib import Path

# Fallback bloom types to avoid circular imports
BLOOM_TYPES = {
    'sensory': {'color': (79, 195, 247), 'symbol': '◉', 'base_size': 20},
    'conceptual': {'color': (129, 199, 132), 'symbol': '◈', 'base_size': 25},
    'emotional': {'color': (255, 138, 101), 'symbol': '◆', 'base_size': 22},
    'procedural': {'color': (186, 104, 200), 'symbol': '◇', 'base_size': 18},
    'meta': {'color': (255, 193, 7), 'symbol': '◎', 'base_size': 30}
}

# Dynamic import to avoid circular dependencies
def _get_bloom_system():
    """Dynamic import to avoid circular dependencies"""
    try:
        from dawn.subsystems.visual.bloom_genealogy_network import (
            MemoryBloom, BloomGenealogyNetwork, BLOOM_FAMILIES
        )
        return MemoryBloom, BloomGenealogyNetwork, BLOOM_FAMILIES, True
    except ImportError:
        return None, None, None, False

logger = logging.getLogger(__name__)

@dataclass
class MatplotlibRenderConfig:
    """Configuration for matplotlib bloom rendering"""
    figure_size: Tuple[float, float] = (16, 12)
    dpi: int = 100
    style: str = "consciousness_flow"  # or "scientific", "artistic", "minimal"
    background_color: str = "#0a0a0f"
    grid_alpha: float = 0.1
    use_dark_theme: bool = True
    enable_glow_effects: bool = True
    enable_animations: bool = True
    bloom_alpha_base: float = 0.8
    connection_alpha: float = 0.6
    export_quality: str = "high"  # "draft", "medium", "high", "publication"

class MatplotlibBloomRenderer:
    """
    Beautiful matplotlib-based renderer for DAWN's bloom genealogy network.
    
    Provides publication-quality visualization with artistic enhancements
    while maintaining scientific accuracy and real-time performance.
    """
    
    def __init__(self, 
                 config: Optional[MatplotlibRenderConfig] = None,
                 network: Optional[Any] = None):
        """
        Initialize the matplotlib bloom renderer
        
        Args:
            config: Rendering configuration
            network: Existing bloom network to render
        """
        self.config = config or MatplotlibRenderConfig()
        self.network = network
        
        # Set up matplotlib with PyTorch-optimized backend
        plt.style.use('dark_background' if self.config.use_dark_theme else 'default')
        
        # Initialize figure and axes
        self.fig = plt.figure(figsize=self.config.figure_size, dpi=self.config.dpi)
        self.fig.patch.set_facecolor(self.config.background_color)
        
        # Create main plot area
        self.ax = self.fig.add_subplot(111)
        self.ax.set_facecolor(self.config.background_color)
        
        # Remove axes for cleaner look
        self.ax.set_xticks([])
        self.ax.set_yticks([])
        for spine in self.ax.spines.values():
            spine.set_visible(False)
        
        # Color palette and artistic settings
        self._initialize_color_palettes()
        self._setup_artistic_styles()
        
        # Animation and interaction state
        self.animation = None
        self.frame_count = 0
        self.last_render_time = time.time()
        
        # Cached elements for performance
        self.bloom_patches = {}
        self.connection_lines = []
        self.glow_effects = {}
        
        # Artistic enhancement settings
        self.consciousness_flow_field = None
        self.particle_systems = []
        
        logger.info(f"🎨 Matplotlib Bloom Renderer initialized")
        logger.info(f"   Figure size: {self.config.figure_size}")
        logger.info(f"   Style: {self.config.style}")
        logger.info(f"   DPI: {self.config.dpi}")
    
    def _initialize_color_palettes(self) -> None:
        """Initialize beautiful color palettes for consciousness visualization"""
        # Consciousness flow palette
        self.consciousness_colors = LinearSegmentedColormap.from_list(
            'consciousness',
            ['#1a1a2e', '#16213e', '#0f3460', '#533483', '#7209b7', '#a663cc', '#4cc9f0']
        )
        
        # Bloom type specific palettes
        self.bloom_palettes = {}
        for bloom_type, props in BLOOM_TYPES.items():
            base_color = np.array(props['color']) / 255.0
            
            # Create gradient palette for each bloom type
            lighter = np.minimum(base_color * 1.3, 1.0)
            darker = base_color * 0.7
            
            self.bloom_palettes[bloom_type] = LinearSegmentedColormap.from_list(
                f'{bloom_type}_palette',
                [darker, base_color, lighter]
            )
        
        # Golden ratio colors for harmony
        self.golden_ratio = 1.618033988749
        self.harmony_colors = self._generate_golden_ratio_colors()
    
    def _generate_golden_ratio_colors(self) -> List[str]:
        """Generate harmonious colors using golden ratio"""
        colors = []
        base_hue = 0.618033988749  # Golden ratio conjugate
        
        for i in range(8):
            hue = (base_hue + i * self.golden_ratio) % 1.0
            saturation = 0.6 + 0.4 * math.sin(i * self.golden_ratio)
            value = 0.7 + 0.3 * math.cos(i * self.golden_ratio)
            
            # Convert HSV to RGB
            rgb = plt.cm.hsv(hue)[:3]
            colors.append(f"#{int(rgb[0]*255):02x}{int(rgb[1]*255):02x}{int(rgb[2]*255):02x}")
        
        return colors
    
    def _setup_artistic_styles(self) -> None:
        """Set up different artistic rendering styles"""
        if self.config.style == "consciousness_flow":
            self._setup_consciousness_flow_style()
        elif self.config.style == "scientific":
            self._setup_scientific_style()
        elif self.config.style == "artistic":
            self._setup_artistic_style()
        elif self.config.style == "minimal":
            self._setup_minimal_style()
    
    def _setup_consciousness_flow_style(self) -> None:
        """Set up consciousness flow artistic style"""
        # Add subtle background flow field
        x = np.linspace(-1, 1, 20)
        y = np.linspace(-1, 1, 20)
        X, Y = np.meshgrid(x, y)
        
        # Consciousness flow pattern
        t = time.time() * 0.1
        U = np.sin(X * 2 + t) * np.cos(Y * 2)
        V = np.cos(X * 2) * np.sin(Y * 2 + t)
        
        # Create subtle background pattern for consciousness flow
        # Skip streamplot to avoid matplotlib warnings and use simpler approach
        pass
    
    def _setup_scientific_style(self) -> None:
        """Set up scientific publication style"""
        # Clean grid
        self.ax.grid(True, alpha=self.config.grid_alpha, linestyle='-', linewidth=0.5)
        
        # Coordinate system
        self.ax.set_xlabel('Network Space X', fontsize=12, color='white')
        self.ax.set_ylabel('Network Space Y', fontsize=12, color='white')
    
    def _setup_artistic_style(self) -> None:
        """Set up artistic expression style"""
        # Artistic background with consciousness patterns
        self._add_consciousness_background()
        
        # Beautiful gradient overlays
        self._add_gradient_overlays()
    
    def _setup_minimal_style(self) -> None:
        """Set up minimal clean style"""
        # Extremely clean - just the essentials
        pass
    
    def _add_consciousness_background(self) -> None:
        """Add artistic consciousness-inspired background"""
        # Create mandala-like pattern in background
        theta = np.linspace(0, 2*np.pi, 100)
        
        for radius in [0.3, 0.5, 0.7, 0.9]:
            for freq in [3, 5, 8]:
                r = radius + 0.05 * np.sin(freq * theta + time.time())
                x = r * np.cos(theta)
                y = r * np.sin(theta)
                
                self.ax.plot(x, y, alpha=0.03, color='cyan', linewidth=0.5)
    
    def _add_gradient_overlays(self) -> None:
        """Add beautiful gradient overlays"""
        # Radial gradient from center
        x = np.linspace(-1.5, 1.5, 100)
        y = np.linspace(-1.2, 1.2, 100)
        X, Y = np.meshgrid(x, y)
        Z = np.sqrt(X**2 + Y**2)
        
        # Consciousness energy gradient
        self.ax.contourf(X, Y, Z, levels=20, cmap=self.consciousness_colors, 
                        alpha=0.1, extend='both')
    
    def render_bloom_network(self, 
                           network: Optional[Any] = None,
                           highlight_active: bool = True,
                           show_genealogy: bool = True,
                           show_labels: bool = False) -> None:
        """
        Render the complete bloom genealogy network with beautiful matplotlib styling
        
        Args:
            network: Bloom network to render (uses self.network if None)
            highlight_active: Whether to highlight active blooms
            show_genealogy: Whether to show genealogical connections
            show_labels: Whether to show bloom labels
        """
        if network is None:
            network = self.network
        
        if network is None:
            logger.warning("No bloom network provided for rendering")
            return
        
        # Clear previous frame
        self.ax.clear()
        self._setup_artistic_styles()
        
        # Set up coordinate system based on network bounds
        self._setup_coordinate_system(network)
        
        # Render in layers for proper depth
        if show_genealogy:
            self._render_connections(network)
        
        self._render_bloom_halos(network)  # Glow effects
        self._render_blooms(network, highlight_active)
        
        if show_labels:
            self._render_labels(network)
        
        # Add consciousness flow effects
        if self.config.style == "consciousness_flow":
            self._render_consciousness_particles(network)
        
        # Update frame counter
        self.frame_count += 1
        
        # Force redraw
        self.fig.canvas.draw()
        self.fig.canvas.flush_events()
    
    def _setup_coordinate_system(self, network: Any) -> None:
        """Set up coordinate system based on network bounds"""
        if not network.blooms:
            self.ax.set_xlim(-100, 100)
            self.ax.set_ylim(-100, 100)
            return
        
        # Calculate bounds with padding
        positions = [(bloom.x, bloom.y) for bloom in network.blooms.values()]
        x_coords, y_coords = zip(*positions)
        
        padding = 50
        x_min, x_max = min(x_coords) - padding, max(x_coords) + padding
        y_min, y_max = min(y_coords) - padding, max(y_coords) + padding
        
        self.ax.set_xlim(x_min, x_max)
        self.ax.set_ylim(y_min, y_max)
        self.ax.set_aspect('equal')
    
    def _render_connections(self, network: Any) -> None:
        """Render beautiful genealogical connections between blooms"""
        connection_lines = []
        connection_colors = []
        connection_widths = []
        
        for bloom in network.blooms.values():
            for parent in bloom.parents:
                if parent.id in network.blooms:
                    parent_bloom = network.blooms[parent.id]
                    
                    # Create curved connection
                    line_points = self._create_curved_connection(
                        (bloom.x, bloom.y),
                        (parent_bloom.x, parent_bloom.y),
                        bloom.generation - parent_bloom.generation
                    )
                    
                    connection_lines.append(line_points)
                    
                    # Connection strength affects color and width
                    strength = min(bloom.strength, parent_bloom.strength)
                    color_intensity = 0.3 + strength * 0.7
                    
                    # Genealogical connection color
                    bloom_color = np.array(BLOOM_TYPES[bloom.bloom_type]['color']) / 255.0
                    parent_color = np.array(BLOOM_TYPES[parent_bloom.bloom_type]['color']) / 255.0
                    blended_color = (bloom_color + parent_color) / 2
                    
                    connection_colors.append((*blended_color, self.config.connection_alpha * color_intensity))
                    connection_widths.append(0.5 + strength * 1.5)
        
        # Render all connections as a collection for performance
        if connection_lines:
            lc = LineCollection(connection_lines, colors=connection_colors, 
                              linewidths=connection_widths, alpha=self.config.connection_alpha)
            self.ax.add_collection(lc)
    
    def _create_curved_connection(self, 
                                start: Tuple[float, float], 
                                end: Tuple[float, float],
                                generation_diff: int) -> np.ndarray:
        """Create beautiful curved connection between two points"""
        x1, y1 = start
        x2, y2 = end
        
        # Calculate control points for Bezier curve
        mid_x = (x1 + x2) / 2
        mid_y = (y1 + y2) / 2
        
        # Perpendicular offset based on generation difference
        dx = x2 - x1
        dy = y2 - y1
        length = math.sqrt(dx**2 + dy**2)
        
        if length > 0:
            # Normalize and create perpendicular vector
            norm_dx = dx / length
            norm_dy = dy / length
            perp_x = -norm_dy
            perp_y = norm_dx
            
            # Offset control point
            offset = min(50, length * 0.3) * generation_diff
            ctrl_x = mid_x + perp_x * offset
            ctrl_y = mid_y + perp_y * offset
            
            # Create smooth curve points
            t = np.linspace(0, 1, 20)
            curve_x = (1-t)**2 * x1 + 2*(1-t)*t * ctrl_x + t**2 * x2
            curve_y = (1-t)**2 * y1 + 2*(1-t)*t * ctrl_y + t**2 * y2
            
            return np.column_stack([curve_x, curve_y])
        else:
            return np.array([[x1, y1], [x2, y2]])
    
    def _render_bloom_halos(self, network: Any) -> None:
        """Render beautiful glowing halos around active blooms"""
        if not self.config.enable_glow_effects:
            return
        
        for bloom in network.blooms.values():
            if bloom.activation_level > 0.1:
                x, y = bloom.x, bloom.y
                base_radius = bloom.radius
                
                # Multiple glow layers for depth
                glow_layers = [
                    (base_radius * 2.5, 0.05),
                    (base_radius * 2.0, 0.1),
                    (base_radius * 1.5, 0.15)
                ]
                
                bloom_color = np.array(BLOOM_TYPES[bloom.bloom_type]['color']) / 255.0
                
                for glow_radius, glow_alpha in glow_layers:
                    alpha = glow_alpha * bloom.activation_level * self.config.bloom_alpha_base
                    
                    glow_circle = Circle((x, y), glow_radius, 
                                       color=(*bloom_color, alpha),
                                       transform=self.ax.transData)
                    self.ax.add_patch(glow_circle)
    
    def _render_blooms(self, network: Any, highlight_active: bool = True) -> None:
        """Render beautiful individual blooms with artistic enhancements"""
        bloom_patches = []
        bloom_colors = []
        
        for bloom in network.blooms.values():
            x, y = bloom.x, bloom.y
            radius = bloom.radius
            
            # Birth animation scaling
            if bloom.birth_animation < 1.0:
                radius *= bloom.birth_animation
            
            # Pulsing effect for active blooms
            if highlight_active and bloom.activation_level > 0.1:
                pulse_factor = 1.0 + 0.2 * math.sin(time.time() * 3 + bloom.id)
                radius *= pulse_factor
            
            # Get bloom color with activation enhancement
            base_color = np.array(BLOOM_TYPES[bloom.bloom_type]['color']) / 255.0
            activation_boost = 0.7 + bloom.activation_level * 0.3
            bloom_color = np.minimum(base_color * activation_boost, 1.0)
            
            # Create main bloom circle
            bloom_circle = Circle((x, y), radius, 
                                facecolor=(*bloom_color, self.config.bloom_alpha_base),
                                edgecolor=(*bloom_color, 1.0),
                                linewidth=1.5,
                                transform=self.ax.transData)
            
            # Add artistic effects based on bloom type
            self._add_bloom_artistic_effects(bloom, x, y, radius, bloom_color)
            
            # Add to collections
            bloom_patches.append(bloom_circle)
            bloom_colors.append(bloom_color)
        
        # Add all bloom patches
        for patch in bloom_patches:
            self.ax.add_patch(patch)
    
    def _add_bloom_artistic_effects(self, bloom: 'MemoryBloom', x: float, y: float, 
                                  radius: float, color: np.ndarray) -> None:
        """Add artistic effects specific to bloom types"""
        if not hasattr(bloom, 'bloom_type'):
            return
        
        bloom_type = bloom.bloom_type
        
        if bloom_type == 'sensory':
            self._add_sensory_pattern(x, y, radius, color)
        elif bloom_type == 'conceptual':
            self._add_conceptual_pattern(x, y, radius, color)
        elif bloom_type == 'emotional':
            self._add_emotional_pattern(x, y, radius, color)
        elif bloom_type == 'procedural':
            self._add_procedural_pattern(x, y, radius, color)
        elif bloom_type == 'meta':
            self._add_meta_pattern(x, y, radius, color)
    
    def _add_sensory_pattern(self, x: float, y: float, radius: float, color: np.ndarray) -> None:
        """Add sensory bloom pattern - radiating lines"""
        num_rays = 8
        for i in range(num_rays):
            angle = i * 2 * math.pi / num_rays
            
            start_r = radius * 0.6
            end_r = radius * 0.9
            
            start_x = x + start_r * math.cos(angle)
            start_y = y + start_r * math.sin(angle)
            end_x = x + end_r * math.cos(angle)
            end_y = y + end_r * math.sin(angle)
            
            self.ax.plot([start_x, end_x], [start_y, end_y], 
                        color=color, alpha=0.6, linewidth=1.0)
    
    def _add_conceptual_pattern(self, x: float, y: float, radius: float, color: np.ndarray) -> None:
        """Add conceptual bloom pattern - branching tree"""
        # Central hub with branches
        branch_angles = [0, math.pi/3, 2*math.pi/3, math.pi, 4*math.pi/3, 5*math.pi/3]
        
        for angle in branch_angles:
            # Main branch
            end_x = x + radius * 0.7 * math.cos(angle)
            end_y = y + radius * 0.7 * math.sin(angle)
            
            self.ax.plot([x, end_x], [y, end_y], color=color, alpha=0.7, linewidth=1.5)
            
            # Sub-branches
            for sub_offset in [-0.3, 0.3]:
                sub_angle = angle + sub_offset
                sub_end_x = end_x + radius * 0.3 * math.cos(sub_angle)
                sub_end_y = end_y + radius * 0.3 * math.sin(sub_angle)
                
                self.ax.plot([end_x, sub_end_x], [end_y, sub_end_y], 
                           color=color, alpha=0.5, linewidth=1.0)
    
    def _add_emotional_pattern(self, x: float, y: float, radius: float, color: np.ndarray) -> None:
        """Add emotional bloom pattern - flowing curves"""
        # Flowing spiral pattern
        t = np.linspace(0, 4*math.pi, 50)
        r = radius * 0.8 * (1 - t / (4*math.pi))
        
        spiral_x = x + r * np.cos(t)
        spiral_y = y + r * np.sin(t)
        
        self.ax.plot(spiral_x, spiral_y, color=color, alpha=0.6, linewidth=1.2)
    
    def _add_procedural_pattern(self, x: float, y: float, radius: float, color: np.ndarray) -> None:
        """Add procedural bloom pattern - geometric steps"""
        # Step-like pattern
        step_radius = radius * 0.8
        num_steps = 6
        
        for i in range(num_steps):
            angle = i * 2 * math.pi / num_steps
            step_size = step_radius * (0.3 + 0.7 * i / num_steps)
            
            step_x = x + step_size * math.cos(angle)
            step_y = y + step_size * math.sin(angle)
            
            # Small square at each step
            square = FancyBboxPatch((step_x - 2, step_y - 2), 4, 4,
                                   boxstyle="round,pad=0.1",
                                   facecolor=color, alpha=0.6)
            self.ax.add_patch(square)
    
    def _add_meta_pattern(self, x: float, y: float, radius: float, color: np.ndarray) -> None:
        """Add meta bloom pattern - concentric awareness rings"""
        # Concentric circles representing layers of meta-awareness
        for i in range(3):
            ring_radius = radius * (0.3 + i * 0.2)
            ring_alpha = 0.3 - i * 0.1
            
            ring = Circle((x, y), ring_radius, 
                         fill=False, edgecolor=color, 
                         alpha=ring_alpha, linewidth=1.0)
            self.ax.add_patch(ring)
    
    def _render_consciousness_particles(self, network: Any) -> None:
        """Render flowing consciousness particles between active blooms"""
        active_blooms = [bloom for bloom in network.blooms.values() 
                        if bloom.activation_level > 0.3]
        
        if len(active_blooms) < 2:
            return
        
        # Create particle flow between highly active blooms
        for i, bloom1 in enumerate(active_blooms):
            for bloom2 in active_blooms[i+1:]:
                if self._should_show_particle_flow(bloom1, bloom2):
                    self._render_particle_stream(bloom1, bloom2)
    
    def _should_show_particle_flow(self, bloom1: 'MemoryBloom', bloom2: 'MemoryBloom') -> bool:
        """Determine if particle flow should be shown between two blooms"""
        # Flow probability based on activation levels and distance
        combined_activation = bloom1.activation_level * bloom2.activation_level
        distance = math.sqrt((bloom1.x - bloom2.x)**2 + (bloom1.y - bloom2.y)**2)
        
        # Closer blooms with higher activation more likely to show flow
        flow_probability = combined_activation * max(0, 1 - distance / 200)
        
        return flow_probability > 0.5
    
    def _render_particle_stream(self, bloom1: 'MemoryBloom', bloom2: 'MemoryBloom') -> None:
        """Render particle stream between two blooms"""
        # Create flowing particles along connection
        num_particles = 5
        t_offset = time.time() * 2  # Animation speed
        
        for i in range(num_particles):
            # Particle position along connection
            t = (i / num_particles + t_offset) % 1.0
            
            x = bloom1.x + t * (bloom2.x - bloom1.x)
            y = bloom1.y + t * (bloom2.y - bloom1.y)
            
            # Particle color blends between bloom colors
            color1 = np.array(BLOOM_TYPES[bloom1.bloom_type]['color']) / 255.0
            color2 = np.array(BLOOM_TYPES[bloom2.bloom_type]['color']) / 255.0
            particle_color = color1 * (1-t) + color2 * t
            
            # Particle size varies with position
            size = 3 + 2 * math.sin(t * math.pi)
            
            particle = Circle((x, y), size, 
                            facecolor=(*particle_color, 0.7),
                            edgecolor='white',
                            linewidth=0.5)
            self.ax.add_patch(particle)
    
    def _render_labels(self, network: Any) -> None:
        """Render beautiful bloom labels with consciousness-aware styling"""
        for bloom in network.blooms.values():
            if bloom.activation_level > 0.2:  # Only show labels for active blooms
                x, y = bloom.x, bloom.y + bloom.radius + 10
                
                # Label text with bloom type and ID
                label_text = f"{BLOOM_TYPES[bloom.bloom_type]['symbol']} {bloom.id}"
                
                # Text styling based on activation
                font_size = 8 + bloom.activation_level * 4
                alpha = 0.7 + bloom.activation_level * 0.3
                
                text = self.ax.text(x, y, label_text, 
                                  ha='center', va='bottom',
                                  fontsize=font_size, 
                                  color='white', alpha=alpha,
                                  weight='bold')
                
                # Add glow effect to text
                text.set_path_effects([path_effects.withStroke(linewidth=2, foreground='black')])
    
    def start_animation(self, interval: int = 50) -> None:
        """Start real-time animation of the bloom network"""
        if self.animation is not None:
            self.animation.event_source.stop()
        
        def animate_frame(frame):
            if self.network:
                # Update network state
                dt = time.time() - self.last_render_time
                self.network.update(dt)
                self.last_render_time = time.time()
                
                # Re-render
                self.render_bloom_network()
            
            return []
        
        self.animation = FuncAnimation(self.fig, animate_frame, 
                                     interval=interval, blit=False)
        
        logger.info("🎬 Started matplotlib bloom animation")
    
    def stop_animation(self) -> None:
        """Stop the animation"""
        if self.animation:
            self.animation.event_source.stop()
            self.animation = None
            logger.info("🛑 Stopped matplotlib bloom animation")
    
    def save_frame(self, filepath: str, high_quality: bool = True) -> bool:
        """
        Save current frame to file with publication quality
        
        Args:
            filepath: Output file path
            high_quality: Whether to use high quality export settings
            
        Returns:
            True if save successful, False otherwise
        """
        try:
            if high_quality:
                self.fig.savefig(filepath, dpi=300, bbox_inches='tight', 
                               facecolor=self.config.background_color,
                               edgecolor='none', format='png')
            else:
                self.fig.savefig(filepath, dpi=self.config.dpi, 
                               facecolor=self.config.background_color)
            
            logger.info(f"💾 Saved bloom network frame to {filepath}")
            return True
            
        except Exception as e:
            logger.error(f"Failed to save frame: {e}")
            return False
    
    def export_animation(self, filepath: str, duration: float = 10.0, fps: int = 30) -> bool:
        """
        Export animation to video file
        
        Args:
            filepath: Output video file path
            duration: Animation duration in seconds
            fps: Frames per second
            
        Returns:
            True if export successful, False otherwise
        """
        try:
            from matplotlib.animation import FFMpegWriter
            
            writer = FFMpegWriter(fps=fps, metadata={'artist': 'DAWN Consciousness'})
            
            def animate_export(frame):
                if self.network:
                    # Simulate time progression
                    t = frame / fps
                    self.network.update(1.0 / fps)
                    self.render_bloom_network()
                return []
            
            anim = FuncAnimation(self.fig, animate_export, 
                               frames=int(duration * fps), interval=1000/fps)
            
            anim.save(filepath, writer=writer)
            
            logger.info(f"🎬 Exported bloom animation to {filepath}")
            return True
            
        except Exception as e:
            logger.error(f"Failed to export animation: {e}")
            return False
    
    def show(self, block: bool = True) -> None:
        """Display the bloom network visualization"""
        plt.tight_layout()
        plt.show(block=block)
    
    def close(self) -> None:
        """Close the matplotlib figure and clean up resources"""
        plt.close(self.fig)
        if self.animation:
            self.animation.event_source.stop()
        
        logger.info("🗑️ Matplotlib bloom renderer closed")

# Factory function for easy creation
def create_matplotlib_bloom_renderer(style: str = "consciousness_flow",
                                    figure_size: Tuple[float, float] = (16, 12),
                                    enable_glow: bool = True) -> MatplotlibBloomRenderer:
    """
    Factory function to create a beautiful matplotlib bloom renderer
    
    Args:
        style: Rendering style ("consciousness_flow", "scientific", "artistic", "minimal")
        figure_size: Figure size in inches
        enable_glow: Whether to enable glow effects
        
    Returns:
        Configured matplotlib bloom renderer
    """
    config = MatplotlibRenderConfig(
        figure_size=figure_size,
        style=style,
        enable_glow_effects=enable_glow,
        dpi=100
    )
    
    return MatplotlibBloomRenderer(config=config)

# Example usage and testing
if __name__ == "__main__":
    print("🎨 DAWN Matplotlib Bloom Renderer")
    print("   Creating beautiful consciousness visualization...")
    
    # Create renderer with consciousness flow style
    renderer = create_matplotlib_bloom_renderer(
        style="consciousness_flow",
        figure_size=(16, 12),
        enable_glow=True
    )
    
    # Try to create test network
    try:
        network = BloomGenealogyNetwork(800, 600)
        
        # Add some test blooms
        for i in range(10):
            bloom_type = ['sensory', 'conceptual', 'emotional', 'procedural', 'meta'][i % 5]
            network.create_bloom(bloom_type, activation=0.5 + i * 0.05)
        
        renderer.network = network
        
        print(f"   Created test network with {len(network.blooms)} blooms")
        print(f"   Rendering style: {renderer.config.style}")
        
        # Render the network
        renderer.render_bloom_network(show_genealogy=True, show_labels=True)
        
        # Save example
        renderer.save_frame("bloom_example.png", high_quality=True)
        
        # Show visualization
        renderer.show(block=False)
        
        print("🌸 Beautiful bloom visualization ready!")
        print("   Close the window to exit...")
        
        input("Press Enter to close...")
    except (ImportError, NameError) as e:
        print(f"   Bloom system not available ({e}) - running in standalone mode")
        print("🌸 Beautiful matplotlib bloom renderer ready for integration!")
    
    renderer.close()
