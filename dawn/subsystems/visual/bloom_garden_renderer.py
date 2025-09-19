#!/usr/bin/env python3
"""
🌸 Bloom Garden Visualization System
═══════════════════════════════════

Visual rendering system for DAWN's fractal memory garden. Creates beautiful
ASCII and graphical representations of the bloom landscape, showing Juliet
flowers, shimmer intensities, and entropy variations.

"Consider all fractal encoding as a garden. Imagine Juliet sets rendered
geometrically and aesthetically - fractals with different entropy values
appear anomalous in color compared to other fractals."

Based on documentation: Fractal Memory/Bloom systems.rtf
"""

import numpy as np
import logging
import time
import threading
import math
from typing import Dict, List, Any, Optional, Tuple, Union
from dataclasses import dataclass, field
from enum import Enum
from collections import defaultdict
import json
import io
from PIL import Image, ImageDraw, ImageFont, ImageFilter
import matplotlib.pyplot as plt
import matplotlib.patches as patches
from matplotlib.colors import LinearSegmentedColormap
import seaborn as sns

logger = logging.getLogger(__name__)

class GardenViewMode(Enum):
    """Different visualization modes for the garden"""
    ASCII_SIMPLE = "ascii_simple"           # Simple ASCII representation
    ASCII_DETAILED = "ascii_detailed"       # Detailed ASCII with symbols
    FRACTAL_FIELD = "fractal_field"         # Mathematical fractal field
    SHIMMER_LANDSCAPE = "shimmer_landscape" # Shimmer intensity heatmap
    ENTROPY_TOPOLOGY = "entropy_topology"   # Entropy-based coloring
    JULIET_GARDEN = "juliet_garden"         # Focus on Juliet flowers
    INTERACTIVE_3D = "interactive_3d"       # 3D interactive view

class BloomSymbol(Enum):
    """ASCII symbols for different bloom types"""
    SEED = "·"              # Regular memory
    CHRYSALIS = "○"         # Building rebloom potential  
    JULIET_FLOWER = "✿"     # Full Juliet bloom
    FADING_BLOOM = "◦"      # Losing shimmer
    GHOST_TRACE = "░"       # Ghost trace remnant
    ASH_RESIDUE = "▓"       # Ash deposit
    SOOT_CLOUD = "▒"        # Soot accumulation
    MYCELIAL_NODE = "◈"     # Mycelial connection
    TRACER_PATH = "~"       # Tracer movement
    EMPTY_SPACE = " "       # No activity

@dataclass
class BloomVisualizationData:
    """Data structure for bloom visualization"""
    memory_id: str
    position: Tuple[float, float]
    bloom_type: str
    intensity: float
    entropy_value: float
    shimmer_level: float
    rebloom_depth: int = 0
    age_ticks: int = 0
    connections: List[str] = field(default_factory=list)
    visual_properties: Dict[str, Any] = field(default_factory=dict)

@dataclass
class GardenLayout:
    """Garden layout configuration"""
    width: int = 80
    height: int = 40
    scale: float = 1.0
    center_x: float = 0.0
    center_y: float = 0.0
    zoom_level: float = 1.0
    view_mode: GardenViewMode = GardenViewMode.ASCII_DETAILED

class BloomGardenRenderer:
    """
    Main renderer for the bloom garden visualization system.
    Creates beautiful representations of DAWN's memory landscape.
    """
    
    def __init__(self,
                 default_layout: Optional[GardenLayout] = None,
                 enable_caching: bool = True,
                 update_interval: float = 1.0):
        
        self.default_layout = default_layout or GardenLayout()
        self.enable_caching = enable_caching
        self.update_interval = update_interval
        
        # Visualization data
        self.bloom_data: Dict[str, BloomVisualizationData] = {}
        self.garden_history: List[Dict[str, Any]] = []
        self.cached_renders: Dict[str, Any] = {}
        
        # Color mappings for different entropy levels
        self.entropy_colors = {
            'very_low': '\033[94m',    # Blue
            'low': '\033[96m',         # Cyan  
            'normal': '\033[92m',      # Green
            'high': '\033[93m',        # Yellow
            'very_high': '\033[91m',   # Red
            'extreme': '\033[95m',     # Magenta
            'reset': '\033[0m'         # Reset
        }
        
        # Shimmer intensity mappings
        self.shimmer_symbols = {
            (0.9, 1.0): '✧',   # Brilliant
            (0.7, 0.9): '✦',   # Bright
            (0.5, 0.7): '✶',   # Moderate
            (0.3, 0.5): '✱',   # Dim
            (0.1, 0.3): '✢',   # Fading
            (0.0, 0.1): '·'    # Trace
        }
        
        # Statistics
        self.stats = {
            'renders_generated': 0,
            'cache_hits': 0,
            'cache_misses': 0,
            'garden_updates': 0,
            'last_render_time': 0.0
        }
        
        # Thread safety
        self._lock = threading.RLock()
        
        logger.info("🌸 BloomGardenRenderer initialized")
    
    def update_bloom_data(self, blooms: List[BloomVisualizationData]):
        """Update the bloom data for visualization"""
        with self._lock:
            # Clear old data
            self.bloom_data.clear()
            
            # Add new bloom data
            for bloom in blooms:
                self.bloom_data[bloom.memory_id] = bloom
            
            # Clear cache when data changes
            if self.enable_caching:
                self.cached_renders.clear()
            
            self.stats['garden_updates'] += 1
            
            logger.debug(f"🌸 Updated bloom data: {len(blooms)} blooms")
    
    def render_ascii_garden(self, 
                           layout: Optional[GardenLayout] = None,
                           show_colors: bool = True) -> str:
        """Render the garden as ASCII art"""
        layout = layout or self.default_layout
        
        with self._lock:
            # Check cache
            cache_key = f"ascii_{layout.width}x{layout.height}_{layout.view_mode.value}"
            if self.enable_caching and cache_key in self.cached_renders:
                self.stats['cache_hits'] += 1
                return self.cached_renders[cache_key]
            
            start_time = time.time()
            
            # Create garden grid
            garden_grid = [[BloomSymbol.EMPTY_SPACE.value for _ in range(layout.width)] 
                          for _ in range(layout.height)]
            color_grid = [['reset' for _ in range(layout.width)] 
                         for _ in range(layout.height)]
            
            # Map blooms to grid positions
            for bloom in self.bloom_data.values():
                grid_x, grid_y = self._world_to_grid(bloom.position, layout)
                
                if 0 <= grid_x < layout.width and 0 <= grid_y < layout.height:
                    # Determine symbol based on bloom type and intensity
                    symbol = self._get_bloom_symbol(bloom, layout.view_mode)
                    color = self._get_bloom_color(bloom)
                    
                    garden_grid[grid_y][grid_x] = symbol
                    color_grid[grid_y][grid_x] = color
            
            # Generate ASCII representation
            ascii_lines = []
            
            # Add header
            ascii_lines.append(f"{'=' * layout.width}")
            ascii_lines.append(f"🌸 DAWN Memory Garden - {layout.view_mode.value.replace('_', ' ').title()}")
            ascii_lines.append(f"Blooms: {len(self.bloom_data)} | Zoom: {layout.zoom_level:.1f}x")
            ascii_lines.append(f"{'=' * layout.width}")
            
            # Add garden rows
            for y in range(layout.height):
                line = ""
                for x in range(layout.width):
                    symbol = garden_grid[y][x]
                    color = color_grid[y][x]
                    
                    if show_colors and color != 'reset':
                        line += f"{self.entropy_colors.get(color, '')}{symbol}{self.entropy_colors['reset']}"
                    else:
                        line += symbol
                
                ascii_lines.append(line)
            
            # Add legend
            ascii_lines.append(f"{'=' * layout.width}")
            ascii_lines.extend(self._generate_legend(layout.view_mode))
            
            result = "\n".join(ascii_lines)
            
            # Cache result
            if self.enable_caching:
                self.cached_renders[cache_key] = result
                self.stats['cache_misses'] += 1
            
            self.stats['renders_generated'] += 1
            self.stats['last_render_time'] = time.time() - start_time
            
            return result
    
    def render_fractal_field(self,
                           layout: Optional[GardenLayout] = None,
                           resolution: Tuple[int, int] = (800, 600)) -> Image.Image:
        """Render the garden as a fractal field visualization"""
        layout = layout or self.default_layout
        
        with self._lock:
            # Create PIL image
            img = Image.new('RGB', resolution, (0, 0, 20))  # Dark blue background
            draw = ImageDraw.Draw(img)
            
            # Calculate scaling factors
            scale_x = resolution[0] / (layout.width * layout.scale)
            scale_y = resolution[1] / (layout.height * layout.scale)
            
            # Render each bloom as a fractal-inspired pattern
            for bloom in self.bloom_data.values():
                x, y = bloom.position
                
                # Convert to image coordinates
                img_x = int((x - layout.center_x) * scale_x + resolution[0] / 2)
                img_y = int((y - layout.center_y) * scale_y + resolution[1] / 2)
                
                if 0 <= img_x < resolution[0] and 0 <= img_y < resolution[1]:
                    self._draw_fractal_bloom(draw, img_x, img_y, bloom)
            
            # Add shimmer effects
            self._add_shimmer_effects(img, draw, layout, resolution)
            
            return img
    
    def render_shimmer_heatmap(self,
                              layout: Optional[GardenLayout] = None,
                              resolution: Tuple[int, int] = (800, 600)) -> plt.Figure:
        """Render shimmer intensity as a heatmap"""
        layout = layout or self.default_layout
        
        # Create data grid
        grid_size = 50
        shimmer_grid = np.zeros((grid_size, grid_size))
        
        with self._lock:
            for bloom in self.bloom_data.values():
                # Map bloom position to grid
                grid_x = int((bloom.position[0] + layout.width/2) / layout.width * grid_size)
                grid_y = int((bloom.position[1] + layout.height/2) / layout.height * grid_size)
                
                if 0 <= grid_x < grid_size and 0 <= grid_y < grid_size:
                    shimmer_grid[grid_y, grid_x] = max(shimmer_grid[grid_y, grid_x], bloom.shimmer_level)
        
        # Apply Gaussian blur for smooth heatmap
        from scipy.ndimage import gaussian_filter
        shimmer_grid = gaussian_filter(shimmer_grid, sigma=2)
        
        # Create matplotlib figure
        fig, ax = plt.subplots(figsize=(12, 8))
        
        # Custom colormap for shimmer
        colors = ['#000020', '#0040A0', '#00A0FF', '#40FF40', '#FFFF00', '#FF8000', '#FF0000']
        n_bins = 100
        shimmer_cmap = LinearSegmentedColormap.from_list('shimmer', colors, N=n_bins)
        
        # Plot heatmap
        im = ax.imshow(shimmer_grid, cmap=shimmer_cmap, origin='lower', 
                      extent=[-layout.width/2, layout.width/2, -layout.height/2, layout.height/2])
        
        # Add colorbar
        cbar = plt.colorbar(im, ax=ax)
        cbar.set_label('Shimmer Intensity', rotation=270, labelpad=20)
        
        # Overlay bloom positions
        for bloom in self.bloom_data.values():
            color = 'white' if bloom.shimmer_level > 0.5 else 'black'
            marker = self._get_matplotlib_marker(bloom)
            ax.scatter(bloom.position[0], bloom.position[1], 
                      c=color, marker=marker, s=30, alpha=0.8)
        
        ax.set_title('🌸 DAWN Memory Garden - Shimmer Landscape', fontsize=16)
        ax.set_xlabel('Garden X Position')
        ax.set_ylabel('Garden Y Position')
        
        return fig
    
    def render_entropy_topology(self,
                               layout: Optional[GardenLayout] = None) -> str:
        """Render entropy distribution as topological map"""
        layout = layout or self.default_layout
        
        with self._lock:
            # Create entropy height map
            entropy_lines = []
            entropy_lines.append("🌋 ENTROPY TOPOLOGY MAP")
            entropy_lines.append("=" * 60)
            
            # Group blooms by entropy levels
            entropy_groups = defaultdict(list)
            for bloom in self.bloom_data.values():
                entropy_level = self._classify_entropy(bloom.entropy_value)
                entropy_groups[entropy_level].append(bloom)
            
            # Render each entropy level
            for level in ['very_low', 'low', 'normal', 'high', 'very_high', 'extreme']:
                blooms = entropy_groups[level]
                if blooms:
                    color = self.entropy_colors[level]
                    reset = self.entropy_colors['reset']
                    
                    entropy_lines.append(f"{color}[{level.upper().replace('_', ' ')}]{reset}")
                    entropy_lines.append(f"  Count: {len(blooms)}")
                    
                    # Show sample positions
                    sample_blooms = blooms[:5]  # Show first 5
                    for bloom in sample_blooms:
                        symbol = self._get_bloom_symbol(bloom, GardenViewMode.ENTROPY_TOPOLOGY)
                        entropy_lines.append(f"  {color}{symbol}{reset} {bloom.memory_id[:12]}... "
                                           f"({bloom.position[0]:.1f}, {bloom.position[1]:.1f})")
                    
                    if len(blooms) > 5:
                        entropy_lines.append(f"  ... and {len(blooms) - 5} more")
                    entropy_lines.append("")
            
            return "\n".join(entropy_lines)
    
    def create_interactive_dashboard(self) -> Dict[str, Any]:
        """Create data for interactive dashboard"""
        with self._lock:
            dashboard_data = {
                'garden_overview': {
                    'total_blooms': len(self.bloom_data),
                    'bloom_types': defaultdict(int),
                    'entropy_distribution': defaultdict(int),
                    'shimmer_stats': {
                        'average': 0.0,
                        'max': 0.0,
                        'min': 1.0
                    }
                },
                'bloom_details': [],
                'spatial_clusters': [],
                'temporal_evolution': self.garden_history[-100:],  # Last 100 updates
                'performance_metrics': self.stats.copy()
            }
            
            # Calculate statistics
            shimmer_values = []
            for bloom in self.bloom_data.values():
                # Bloom type distribution
                dashboard_data['garden_overview']['bloom_types'][bloom.bloom_type] += 1
                
                # Entropy distribution  
                entropy_level = self._classify_entropy(bloom.entropy_value)
                dashboard_data['garden_overview']['entropy_distribution'][entropy_level] += 1
                
                # Shimmer statistics
                shimmer_values.append(bloom.shimmer_level)
                
                # Bloom details
                dashboard_data['bloom_details'].append({
                    'id': bloom.memory_id,
                    'position': bloom.position,
                    'type': bloom.bloom_type,
                    'intensity': bloom.intensity,
                    'entropy': bloom.entropy_value,
                    'shimmer': bloom.shimmer_level,
                    'age': bloom.age_ticks,
                    'connections': len(bloom.connections)
                })
            
            if shimmer_values:
                dashboard_data['garden_overview']['shimmer_stats'] = {
                    'average': np.mean(shimmer_values),
                    'max': np.max(shimmer_values),
                    'min': np.min(shimmer_values)
                }
            
            return dashboard_data
    
    def _world_to_grid(self, world_pos: Tuple[float, float], layout: GardenLayout) -> Tuple[int, int]:
        """Convert world coordinates to grid coordinates"""
        x, y = world_pos
        
        # Apply zoom and centering
        grid_x = int((x - layout.center_x) * layout.zoom_level + layout.width / 2)
        grid_y = int((y - layout.center_y) * layout.zoom_level + layout.height / 2)
        
        return grid_x, grid_y
    
    def _get_bloom_symbol(self, bloom: BloomVisualizationData, view_mode: GardenViewMode) -> str:
        """Get appropriate symbol for bloom based on view mode"""
        if view_mode == GardenViewMode.ASCII_SIMPLE:
            return "*" if bloom.intensity > 0.5 else "."
        
        elif view_mode == GardenViewMode.SHIMMER_LANDSCAPE:
            # Use shimmer-based symbols
            for (min_shimmer, max_shimmer), symbol in self.shimmer_symbols.items():
                if min_shimmer <= bloom.shimmer_level < max_shimmer:
                    return symbol
            return "·"
        
        elif view_mode == GardenViewMode.JULIET_GARDEN:
            if bloom.bloom_type == "juliet_flower":
                return BloomSymbol.JULIET_FLOWER.value
            elif bloom.bloom_type == "chrysalis":
                return BloomSymbol.CHRYSALIS.value
            else:
                return BloomSymbol.SEED.value
        
        else:  # ASCII_DETAILED or default
            symbol_map = {
                "seed": BloomSymbol.SEED.value,
                "chrysalis": BloomSymbol.CHRYSALIS.value,
                "juliet_flower": BloomSymbol.JULIET_FLOWER.value,
                "fading": BloomSymbol.FADING_BLOOM.value,
                "ghost_trace": BloomSymbol.GHOST_TRACE.value,
                "ash": BloomSymbol.ASH_RESIDUE.value,
                "soot": BloomSymbol.SOOT_CLOUD.value
            }
            return symbol_map.get(bloom.bloom_type, BloomSymbol.SEED.value)
    
    def _get_bloom_color(self, bloom: BloomVisualizationData) -> str:
        """Get color classification for bloom based on entropy"""
        return self._classify_entropy(bloom.entropy_value)
    
    def _classify_entropy(self, entropy_value: float) -> str:
        """Classify entropy value into color category"""
        if entropy_value < 0.1:
            return 'very_low'
        elif entropy_value < 0.3:
            return 'low'
        elif entropy_value < 0.6:
            return 'normal'
        elif entropy_value < 0.8:
            return 'high'
        elif entropy_value < 0.95:
            return 'very_high'
        else:
            return 'extreme'
    
    def _get_matplotlib_marker(self, bloom: BloomVisualizationData) -> str:
        """Get matplotlib marker for bloom type"""
        marker_map = {
            "seed": ".",
            "chrysalis": "o",
            "juliet_flower": "*",
            "fading": "x",
            "ghost_trace": "1",
            "ash": "s",
            "soot": "^"
        }
        return marker_map.get(bloom.bloom_type, ".")
    
    def _draw_fractal_bloom(self, draw: ImageDraw.Draw, x: int, y: int, bloom: BloomVisualizationData):
        """Draw a fractal-inspired bloom pattern"""
        # Base color from entropy
        entropy_colors = {
            'very_low': (100, 100, 255),    # Blue
            'low': (100, 255, 255),         # Cyan
            'normal': (100, 255, 100),      # Green  
            'high': (255, 255, 100),        # Yellow
            'very_high': (255, 100, 100),   # Red
            'extreme': (255, 100, 255)      # Magenta
        }
        
        color_key = self._classify_entropy(bloom.entropy_value)
        base_color = entropy_colors[color_key]
        
        # Adjust brightness based on shimmer
        shimmer_factor = bloom.shimmer_level
        color = tuple(int(c * shimmer_factor) for c in base_color)
        
        # Draw based on bloom type
        if bloom.bloom_type == "juliet_flower":
            # Draw flower pattern
            radius = int(10 * bloom.intensity)
            for i in range(6):  # 6 petals
                angle = i * math.pi / 3
                petal_x = x + int(radius * math.cos(angle))
                petal_y = y + int(radius * math.sin(angle))
                draw.ellipse([petal_x-3, petal_y-3, petal_x+3, petal_y+3], fill=color)
            
            # Center
            draw.ellipse([x-2, y-2, x+2, y+2], fill=(255, 255, 255))
            
        elif bloom.bloom_type == "chrysalis":
            # Draw cocoon shape
            radius = int(8 * bloom.intensity)
            draw.ellipse([x-radius, y-radius//2, x+radius, y+radius//2], fill=color)
            
        else:
            # Simple dot
            radius = max(1, int(5 * bloom.intensity))
            draw.ellipse([x-radius, y-radius, x+radius, y+radius], fill=color)
    
    def _add_shimmer_effects(self, img: Image.Image, draw: ImageDraw.Draw, 
                           layout: GardenLayout, resolution: Tuple[int, int]):
        """Add shimmer effects to the image"""
        # Create shimmer overlay
        shimmer_overlay = Image.new('RGBA', resolution, (0, 0, 0, 0))
        shimmer_draw = ImageDraw.Draw(shimmer_overlay)
        
        for bloom in self.bloom_data.values():
            if bloom.shimmer_level > 0.3:  # Only add shimmer to bright blooms
                x, y = bloom.position
                scale_x = resolution[0] / (layout.width * layout.scale)
                scale_y = resolution[1] / (layout.height * layout.scale)
                
                img_x = int((x - layout.center_x) * scale_x + resolution[0] / 2)
                img_y = int((y - layout.center_y) * scale_y + resolution[1] / 2)
                
                # Add shimmer glow
                glow_radius = int(15 * bloom.shimmer_level)
                alpha = int(50 * bloom.shimmer_level)
                
                shimmer_draw.ellipse([img_x-glow_radius, img_y-glow_radius,
                                    img_x+glow_radius, img_y+glow_radius],
                                   fill=(255, 255, 255, alpha))
        
        # Blur and composite shimmer
        shimmer_overlay = shimmer_overlay.filter(ImageFilter.GaussianBlur(radius=3))
        img.paste(shimmer_overlay, (0, 0), shimmer_overlay)
    
    def _generate_legend(self, view_mode: GardenViewMode) -> List[str]:
        """Generate legend for the visualization"""
        legend = ["LEGEND:"]
        
        if view_mode == GardenViewMode.ASCII_DETAILED:
            legend.extend([
                f"{BloomSymbol.SEED.value} Seed Memory",
                f"{BloomSymbol.CHRYSALIS.value} Chrysalis (Building Rebloom)",
                f"{BloomSymbol.JULIET_FLOWER.value} Juliet Flower (Full Bloom)",
                f"{BloomSymbol.FADING_BLOOM.value} Fading Bloom",
                f"{BloomSymbol.GHOST_TRACE.value} Ghost Trace",
                f"{BloomSymbol.ASH_RESIDUE.value} Ash Residue",
                f"{BloomSymbol.SOOT_CLOUD.value} Soot Accumulation"
            ])
        
        elif view_mode == GardenViewMode.SHIMMER_LANDSCAPE:
            legend.extend([
                "✧ Brilliant Shimmer (0.9-1.0)",
                "✦ Bright Shimmer (0.7-0.9)",
                "✶ Moderate Shimmer (0.5-0.7)",
                "✱ Dim Shimmer (0.3-0.5)",
                "✢ Fading Shimmer (0.1-0.3)",
                "· Trace Shimmer (0.0-0.1)"
            ])
        
        # Add color legend
        legend.append("")
        legend.append("ENTROPY COLORS:")
        legend.extend([
            "Blue: Very Low Entropy",
            "Cyan: Low Entropy", 
            "Green: Normal Entropy",
            "Yellow: High Entropy",
            "Red: Very High Entropy",
            "Magenta: Extreme Entropy"
        ])
        
        return legend
    
    def save_garden_snapshot(self, filename: str, view_mode: GardenViewMode = GardenViewMode.ASCII_DETAILED):
        """Save current garden state to file"""
        timestamp = time.strftime("%Y%m%d_%H%M%S")
        
        if view_mode in [GardenViewMode.ASCII_SIMPLE, GardenViewMode.ASCII_DETAILED]:
            # Save ASCII representation
            ascii_garden = self.render_ascii_garden(view_mode=view_mode)
            with open(f"{filename}_{timestamp}.txt", 'w', encoding='utf-8') as f:
                f.write(ascii_garden)
        
        elif view_mode == GardenViewMode.FRACTAL_FIELD:
            # Save fractal field image
            img = self.render_fractal_field()
            img.save(f"{filename}_{timestamp}.png")
        
        elif view_mode == GardenViewMode.SHIMMER_LANDSCAPE:
            # Save shimmer heatmap
            fig = self.render_shimmer_heatmap()
            fig.savefig(f"{filename}_{timestamp}.png", dpi=300, bbox_inches='tight')
            plt.close(fig)
        
        logger.info(f"🌸 Garden snapshot saved: {filename}_{timestamp}")


# Global bloom garden renderer instance
_bloom_renderer = None

def get_bloom_renderer(config: Optional[Dict[str, Any]] = None) -> BloomGardenRenderer:
    """Get the global bloom garden renderer instance"""
    global _bloom_renderer
    if _bloom_renderer is None:
        config = config or {}
        layout = GardenLayout(
            width=config.get('width', 80),
            height=config.get('height', 40),
            view_mode=GardenViewMode(config.get('view_mode', 'ascii_detailed'))
        )
        _bloom_renderer = BloomGardenRenderer(
            default_layout=layout,
            enable_caching=config.get('enable_caching', True),
            update_interval=config.get('update_interval', 1.0)
        )
    return _bloom_renderer


# Example usage and testing
if __name__ == "__main__":
    logging.basicConfig(level=logging.INFO)
    
    # Initialize renderer
    renderer = BloomGardenRenderer()
    
    # Create sample bloom data
    sample_blooms = [
        BloomVisualizationData(
            memory_id="memory_1",
            position=(10.0, 15.0),
            bloom_type="juliet_flower",
            intensity=0.9,
            entropy_value=0.3,
            shimmer_level=0.85,
            rebloom_depth=3
        ),
        BloomVisualizationData(
            memory_id="memory_2", 
            position=(-5.0, 8.0),
            bloom_type="chrysalis",
            intensity=0.6,
            entropy_value=0.7,
            shimmer_level=0.45,
            rebloom_depth=1
        ),
        BloomVisualizationData(
            memory_id="memory_3",
            position=(0.0, -10.0),
            bloom_type="seed",
            intensity=0.3,
            entropy_value=0.1,
            shimmer_level=0.2,
            rebloom_depth=0
        )
    ]
    
    # Update renderer with sample data
    renderer.update_bloom_data(sample_blooms)
    
    # Render ASCII garden
    ascii_garden = renderer.render_ascii_garden()
    print(ascii_garden)
    
    # Create dashboard data
    dashboard = renderer.create_interactive_dashboard()
    print(f"\nDashboard: {json.dumps(dashboard, indent=2)}")
    
    # Save snapshot
    renderer.save_garden_snapshot("test_garden")
