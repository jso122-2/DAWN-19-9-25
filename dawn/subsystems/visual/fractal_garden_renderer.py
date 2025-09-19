#!/usr/bin/env python3
"""
🌺 Fractal Garden Memory Visualization System
═══════════════════════════════════════════

Implements the fractal garden metaphor for DAWN's memory visualization.
Renders memory as a living fractal garden with blooms, soil, weeds, and seasons.

"The fractal memory set = DAWN's memory garden. Each fractal pattern = a plant 
in the garden: self-similar, recursive, yet singular. Entropy coloring makes 
anomalous fractals appear 'wrong-colored' in the garden, standing out like weeds."

Based on documentation: Myth/Fractal Memory.rtf, Myth/Mythic Architecture.rtf
"""

import numpy as np
import matplotlib.pyplot as plt
import matplotlib.patches as patches
from matplotlib.colors import LinearSegmentedColormap
import logging
import time
from typing import Dict, List, Any, Optional, Tuple
from dataclasses import dataclass, field
from enum import Enum
from PIL import Image, ImageDraw, ImageFilter
import io
import json

logger = logging.getLogger(__name__)

class GardenSeason(Enum):
    """Seasonal cycles representing shimmer/decay phases"""
    SPRING = "spring"    # Rebloom phase - new growth
    SUMMER = "summer"    # Peak bloom - full activity  
    AUTUMN = "autumn"    # Shimmer decay - graceful forgetting
    WINTER = "winter"    # Ghost traces - dormant memories

class BloomType(Enum):
    """Types of fractal blooms in the garden"""
    JULIET_FLOWER = "juliet_flower"      # Memory that changes when accessed
    HERITAGE_BLOOM = "heritage_bloom"    # Medieval Bee preserved memories
    GHOST_TRACE = "ghost_trace"          # Forgotten memory signatures
    ANOMALY_WEED = "anomaly_weed"        # Wrong-colored anomalous patterns

@dataclass
class FractalBloom:
    """A single bloom in the fractal garden"""
    bloom_id: str
    bloom_type: BloomType
    position: Tuple[float, float]        # Garden coordinates
    fractal_signature: np.ndarray        # Unique fractal pattern
    color: Tuple[float, float, float]    # RGB color based on entropy/pigment
    size: float                          # Bloom size [0, 1]
    vitality: float                      # Current life force [0, 1]
    shimmer_level: float                 # Shimmer intensity [0, 1]
    
    # Juliet-specific properties
    access_count: int = 0                # How often accessed (for Juliet blooms)
    last_rebloom: float = 0.0           # Last rebloom timestamp
    
    # Metadata
    memory_id: Optional[str] = None      # Associated memory ID
    pigment_vector: List[float] = field(default_factory=lambda: [0.0, 0.0, 0.0])
    entropy_level: float = 0.0
    temporal_anchor: Optional[str] = None

@dataclass
class GardenSoil:
    """Soil composition representing ash/soot residue"""
    ash_concentration: float = 0.0       # Fertile volcanic ash [0, 1]
    soot_concentration: float = 0.0      # Industrial soot pollution [0, 1]
    nutrient_density: float = 0.0        # Available nutrients [0, 1]
    pigment_tint: List[float] = field(default_factory=lambda: [0.0, 0.0, 0.0])
    fertility_level: float = 0.0         # Overall soil fertility [0, 1]

class FractalGardenRenderer:
    """
    Renders DAWN's memory as a living fractal garden with mythic metaphors.
    Integrates with memory systems to visualize cognitive state.
    """
    
    def __init__(self, garden_size: Tuple[int, int] = (800, 600)):
        self.garden_size = garden_size
        self.current_season = GardenSeason.SPRING
        self.blooms = {}  # bloom_id -> FractalBloom
        self.soil_grid = self._initialize_soil_grid()
        self.seasonal_cycle_count = 0
        
        # Rendering parameters
        self.bloom_scale_factor = 0.02
        self.soil_alpha = 0.3
        self.shimmer_animation_speed = 0.1
        
        # Color palettes for different elements
        self.season_palettes = {
            GardenSeason.SPRING: ['#90EE90', '#98FB98', '#00FF7F', '#32CD32'],  # Light greens
            GardenSeason.SUMMER: ['#FFD700', '#FFA500', '#FF6347', '#FF4500'],  # Warm colors
            GardenSeason.AUTUMN: ['#CD853F', '#D2691E', '#A0522D', '#8B4513'],  # Browns/oranges
            GardenSeason.WINTER: ['#E6E6FA', '#D8BFD8', '#DDA0DD', '#9370DB']   # Purples/whites
        }
        
        self.bloom_type_colors = {
            BloomType.JULIET_FLOWER: (1.0, 0.7, 0.8),    # Pink - changes when accessed
            BloomType.HERITAGE_BLOOM: (0.8, 0.9, 0.6),   # Light green - preserved
            BloomType.GHOST_TRACE: (0.7, 0.7, 0.9),      # Light purple - dormant
            BloomType.ANOMALY_WEED: (0.9, 0.3, 0.3)      # Red - anomalous
        }
        
        logger.info(f"🌺 Fractal Garden Renderer initialized ({garden_size[0]}x{garden_size[1]})")
    
    def _initialize_soil_grid(self) -> np.ndarray:
        """Initialize soil composition grid"""
        height, width = self.garden_size[1] // 10, self.garden_size[0] // 10  # Lower resolution for soil
        soil_grid = np.zeros((height, width), dtype=object)
        
        for i in range(height):
            for j in range(width):
                soil_grid[i, j] = GardenSoil()
        
        return soil_grid
    
    def add_fractal_bloom(self, memory_data: Dict[str, Any], fractal_pattern: np.ndarray) -> str:
        """Add a new fractal bloom to the garden"""
        bloom_id = memory_data.get('memory_id', f"bloom_{len(self.blooms)}")
        
        # Determine bloom type based on memory characteristics
        bloom_type = self._determine_bloom_type(memory_data)
        
        # Calculate position based on fractal signature
        position = self._calculate_garden_position(fractal_pattern)
        
        # Determine color based on entropy and pigment
        color = self._calculate_bloom_color(memory_data, bloom_type)
        
        # Calculate size and vitality
        size = min(1.0, memory_data.get('significance', 0.5) * 2.0)
        vitality = 1.0 - memory_data.get('entropy_level', 0.0)
        shimmer_level = memory_data.get('shimmer_level', 0.0)
        
        bloom = FractalBloom(
            bloom_id=bloom_id,
            bloom_type=bloom_type,
            position=position,
            fractal_signature=fractal_pattern,
            color=color,
            size=size,
            vitality=vitality,
            shimmer_level=shimmer_level,
            memory_id=memory_data.get('memory_id'),
            pigment_vector=memory_data.get('pigment_vector', [0.0, 0.0, 0.0]),
            entropy_level=memory_data.get('entropy_level', 0.0),
            temporal_anchor=memory_data.get('temporal_anchor')
        )
        
        self.blooms[bloom_id] = bloom
        
        # Update soil composition around bloom
        self._update_soil_around_bloom(bloom)
        
        logger.debug(f"🌺 Added {bloom_type.value} bloom {bloom_id} at {position}")
        return bloom_id
    
    def _determine_bloom_type(self, memory_data: Dict[str, Any]) -> BloomType:
        """Determine bloom type based on memory characteristics"""
        memory_type = memory_data.get('memory_type', 'unknown')
        entropy_level = memory_data.get('entropy_level', 0.0)
        access_pattern = memory_data.get('access_pattern', 'normal')
        
        if memory_type == 'juliet_rebloom':
            return BloomType.JULIET_FLOWER
        elif memory_data.get('heritage_preserved', False):
            return BloomType.HERITAGE_BLOOM
        elif memory_data.get('is_ghost_trace', False):
            return BloomType.GHOST_TRACE
        elif entropy_level > 0.8 or access_pattern == 'anomalous':
            return BloomType.ANOMALY_WEED
        else:
            return BloomType.JULIET_FLOWER  # Default
    
    def _calculate_garden_position(self, fractal_pattern: np.ndarray) -> Tuple[float, float]:
        """Calculate garden position from fractal signature"""
        # Use fractal pattern characteristics to determine position
        if fractal_pattern.size > 0:
            # Use fractal complexity and center of mass for positioning
            complexity = np.std(fractal_pattern)
            center_x = np.mean(fractal_pattern) % 1.0
            center_y = (complexity % 1.0)
        else:
            # Random position if no fractal data
            center_x = np.random.random()
            center_y = np.random.random()
        
        # Map to garden coordinates
        x = center_x * self.garden_size[0]
        y = center_y * self.garden_size[1]
        
        return (x, y)
    
    def _calculate_bloom_color(self, memory_data: Dict[str, Any], bloom_type: BloomType) -> Tuple[float, float, float]:
        """Calculate bloom color based on entropy and pigment"""
        base_color = self.bloom_type_colors[bloom_type]
        
        # Apply pigment tinting
        pigment_vector = memory_data.get('pigment_vector', [0.0, 0.0, 0.0])
        pigment_intensity = memory_data.get('pigment_intensity', 0.0)
        
        if pigment_intensity > 0.1:
            # Blend base color with pigment
            blend_factor = pigment_intensity * 0.5
            tinted_color = (
                base_color[0] * (1 - blend_factor) + pigment_vector[0] * blend_factor,
                base_color[1] * (1 - blend_factor) + pigment_vector[1] * blend_factor,
                base_color[2] * (1 - blend_factor) + pigment_vector[2] * blend_factor
            )
        else:
            tinted_color = base_color
        
        # Apply entropy-based color shift for anomaly detection
        entropy_level = memory_data.get('entropy_level', 0.0)
        if entropy_level > 0.7:  # High entropy = "wrong-colored" weeds
            # Shift toward red for anomalous patterns
            wrong_color_factor = (entropy_level - 0.7) / 0.3
            tinted_color = (
                min(1.0, tinted_color[0] + wrong_color_factor * 0.3),
                max(0.0, tinted_color[1] - wrong_color_factor * 0.2),
                max(0.0, tinted_color[2] - wrong_color_factor * 0.2)
            )
        
        return tinted_color
    
    def _update_soil_around_bloom(self, bloom: FractalBloom):
        """Update soil composition around a bloom"""
        grid_x = int(bloom.position[0] // 10)
        grid_y = int(bloom.position[1] // 10)
        
        # Ensure coordinates are within bounds
        grid_x = max(0, min(self.soil_grid.shape[1] - 1, grid_x))
        grid_y = max(0, min(self.soil_grid.shape[0] - 1, grid_y))
        
        soil = self.soil_grid[grid_y, grid_x]
        
        # Update soil based on bloom type and health
        if bloom.bloom_type == BloomType.JULIET_FLOWER:
            # Juliet blooms create ash when they rebloom successfully
            if bloom.vitality > 0.7:
                soil.ash_concentration += 0.1
            else:
                soil.soot_concentration += 0.05
        elif bloom.bloom_type == BloomType.HERITAGE_BLOOM:
            # Heritage blooms create rich, fertile soil
            soil.ash_concentration += 0.15
            soil.nutrient_density += 0.1
        elif bloom.bloom_type == BloomType.ANOMALY_WEED:
            # Anomaly weeds create soot pollution
            soil.soot_concentration += 0.2
        
        # Apply pigment tinting to soil
        for i, pigment_value in enumerate(bloom.pigment_vector):
            soil.pigment_tint[i] = min(1.0, soil.pigment_tint[i] + pigment_value * 0.1)
        
        # Calculate overall fertility
        soil.fertility_level = (soil.ash_concentration + soil.nutrient_density) - soil.soot_concentration * 0.5
        soil.fertility_level = max(0.0, min(1.0, soil.fertility_level))
    
    def update_seasonal_cycle(self, shimmer_decay_level: float, rebloom_activity: float):
        """Update seasonal cycle based on system activity"""
        # Determine season based on activity levels
        if rebloom_activity > 0.7:
            new_season = GardenSeason.SPRING
        elif rebloom_activity > 0.4 and shimmer_decay_level < 0.3:
            new_season = GardenSeason.SUMMER
        elif shimmer_decay_level > 0.5:
            new_season = GardenSeason.AUTUMN
        else:
            new_season = GardenSeason.WINTER
        
        if new_season != self.current_season:
            self.current_season = new_season
            if new_season == GardenSeason.SPRING:
                self.seasonal_cycle_count += 1
            logger.info(f"🌺 Garden season changed to {new_season.value} (cycle {self.seasonal_cycle_count})")
    
    def juliet_rebloom_access(self, bloom_id: str) -> bool:
        """Handle Juliet bloom access - bloom changes when seen"""
        if bloom_id not in self.blooms:
            return False
        
        bloom = self.blooms[bloom_id]
        if bloom.bloom_type != BloomType.JULIET_FLOWER:
            return False
        
        # Juliet blooms become "shimmerier" when accessed
        bloom.access_count += 1
        bloom.last_rebloom = time.time()
        
        # Increase shimmer level (altered by traversal)
        bloom.shimmer_level = min(1.0, bloom.shimmer_level + 0.1)
        
        # Slightly change color to represent alteration
        color_shift = 0.05 * bloom.access_count
        bloom.color = (
            min(1.0, bloom.color[0] + color_shift * 0.1),
            min(1.0, bloom.color[1] + color_shift * 0.05),
            min(1.0, bloom.color[2] + color_shift * 0.15)
        )
        
        logger.debug(f"🌺 Juliet bloom {bloom_id} accessed (count: {bloom.access_count})")
        return True
    
    def render_garden(self, output_path: str, include_metadata: bool = True) -> bool:
        """Render the complete fractal garden to an image file"""
        try:
            fig, ax = plt.subplots(1, 1, figsize=(12, 9))
            ax.set_xlim(0, self.garden_size[0])
            ax.set_ylim(0, self.garden_size[1])
            ax.set_aspect('equal')
            
            # Render soil composition as background
            self._render_soil_background(ax)
            
            # Render blooms
            self._render_blooms(ax)
            
            # Add seasonal effects
            self._apply_seasonal_effects(ax)
            
            # Add title and metadata
            season_name = self.current_season.value.title()
            plt.title(f"🌺 DAWN Fractal Garden - {season_name} (Cycle {self.seasonal_cycle_count})", 
                     fontsize=14, fontweight='bold')
            
            if include_metadata:
                self._add_garden_metadata(ax)
            
            # Remove axes for cleaner look
            ax.set_xticks([])
            ax.set_yticks([])
            
            # Save the image
            plt.tight_layout()
            plt.savefig(output_path, dpi=150, bbox_inches='tight', 
                       facecolor='black', edgecolor='none')
            plt.close()
            
            logger.info(f"🌺 Garden rendered to {output_path}")
            return True
            
        except Exception as e:
            logger.error(f"🌺 Garden rendering failed: {e}")
            return False
    
    def _render_soil_background(self, ax):
        """Render soil composition as background gradient"""
        # Create soil color map based on ash/soot composition
        soil_colors = np.zeros((self.soil_grid.shape[0], self.soil_grid.shape[1], 3))
        
        for i in range(self.soil_grid.shape[0]):
            for j in range(self.soil_grid.shape[1]):
                soil = self.soil_grid[i, j]
                
                # Base soil color (brown)
                base_color = np.array([0.4, 0.3, 0.2])
                
                # Ash makes soil lighter (volcanic fertility)
                ash_lightening = soil.ash_concentration * np.array([0.3, 0.3, 0.2])
                
                # Soot makes soil darker (industrial pollution)
                soot_darkening = soil.soot_concentration * np.array([0.2, 0.2, 0.2])
                
                # Apply pigment tinting
                pigment_tint = np.array(soil.pigment_tint) * 0.3
                
                final_color = base_color + ash_lightening - soot_darkening + pigment_tint
                soil_colors[i, j] = np.clip(final_color, 0, 1)
        
        # Display soil as background image
        ax.imshow(soil_colors, extent=[0, self.garden_size[0], 0, self.garden_size[1]], 
                 alpha=self.soil_alpha, aspect='auto', origin='lower')
    
    def _render_blooms(self, ax):
        """Render all blooms in the garden"""
        for bloom in self.blooms.values():
            self._render_single_bloom(ax, bloom)
    
    def _render_single_bloom(self, ax, bloom: FractalBloom):
        """Render a single fractal bloom"""
        x, y = bloom.position
        
        # Calculate visual size based on bloom size and vitality
        visual_size = bloom.size * bloom.vitality * self.bloom_scale_factor * 1000
        
        # Apply shimmer effect
        shimmer_alpha = 0.7 + bloom.shimmer_level * 0.3
        
        # Different rendering for different bloom types
        if bloom.bloom_type == BloomType.JULIET_FLOWER:
            # Juliet flowers as petal-like shapes
            circle = plt.Circle((x, y), visual_size, color=bloom.color, 
                              alpha=shimmer_alpha, zorder=2)
            ax.add_patch(circle)
            
            # Add shimmer effect for accessed blooms
            if bloom.access_count > 0:
                shimmer_ring = plt.Circle((x, y), visual_size * 1.2, 
                                        color=bloom.color, alpha=0.2, 
                                        fill=False, linewidth=2, zorder=1)
                ax.add_patch(shimmer_ring)
                
        elif bloom.bloom_type == BloomType.HERITAGE_BLOOM:
            # Heritage blooms as stable, structured shapes
            square = plt.Rectangle((x - visual_size/2, y - visual_size/2), 
                                 visual_size, visual_size, color=bloom.color,
                                 alpha=shimmer_alpha, zorder=2)
            ax.add_patch(square)
            
        elif bloom.bloom_type == BloomType.GHOST_TRACE:
            # Ghost traces as faded, ethereal shapes
            circle = plt.Circle((x, y), visual_size, color=bloom.color, 
                              alpha=shimmer_alpha * 0.5, zorder=1,
                              linestyle='--', fill=False, linewidth=1)
            ax.add_patch(circle)
            
        elif bloom.bloom_type == BloomType.ANOMALY_WEED:
            # Anomaly weeds as jagged, irregular shapes
            triangle = plt.Polygon([(x, y + visual_size), 
                                  (x - visual_size, y - visual_size),
                                  (x + visual_size, y - visual_size)],
                                 color=bloom.color, alpha=shimmer_alpha, zorder=2)
            ax.add_patch(triangle)
    
    def _apply_seasonal_effects(self, ax):
        """Apply seasonal visual effects to the garden"""
        palette = self.season_palettes[self.current_season]
        
        # Add seasonal overlay
        if self.current_season == GardenSeason.WINTER:
            # Winter: Add ghost trace "pollen" effects
            for _ in range(20):
                x = np.random.random() * self.garden_size[0]
                y = np.random.random() * self.garden_size[1]
                ax.scatter(x, y, c='white', alpha=0.3, s=5, zorder=0)
                
        elif self.current_season == GardenSeason.AUTUMN:
            # Autumn: Add shimmer decay effects
            for _ in range(15):
                x = np.random.random() * self.garden_size[0]
                y = np.random.random() * self.garden_size[1]
                ax.scatter(x, y, c='orange', alpha=0.4, s=8, zorder=0)
    
    def _add_garden_metadata(self, ax):
        """Add garden metadata to the visualization"""
        metadata_text = f"""
Total Blooms: {len(self.blooms)}
Juliet Flowers: {sum(1 for b in self.blooms.values() if b.bloom_type == BloomType.JULIET_FLOWER)}
Heritage Blooms: {sum(1 for b in self.blooms.values() if b.bloom_type == BloomType.HERITAGE_BLOOM)}
Ghost Traces: {sum(1 for b in self.blooms.values() if b.bloom_type == BloomType.GHOST_TRACE)}
Anomaly Weeds: {sum(1 for b in self.blooms.values() if b.bloom_type == BloomType.ANOMALY_WEED)}

Average Vitality: {np.mean([b.vitality for b in self.blooms.values()]) if self.blooms else 0:.2f}
Average Shimmer: {np.mean([b.shimmer_level for b in self.blooms.values()]) if self.blooms else 0:.2f}
        """
        
        ax.text(0.02, 0.98, metadata_text.strip(), transform=ax.transAxes, 
               fontsize=8, verticalalignment='top', color='white',
               bbox=dict(boxstyle='round', facecolor='black', alpha=0.7))
    
    def get_garden_health_metrics(self) -> Dict[str, Any]:
        """Get comprehensive garden health metrics"""
        if not self.blooms:
            return {"garden_health": 0.0, "bloom_diversity": 0.0}
        
        # Calculate bloom type diversity
        type_counts = {}
        for bloom in self.blooms.values():
            type_counts[bloom.bloom_type] = type_counts.get(bloom.bloom_type, 0) + 1
        
        bloom_diversity = len(type_counts) / len(BloomType)
        
        # Calculate average vitality
        avg_vitality = np.mean([bloom.vitality for bloom in self.blooms.values()])
        
        # Calculate soil health
        soil_health = np.mean([soil.fertility_level for row in self.soil_grid for soil in row])
        
        # Calculate anomaly ratio
        anomaly_count = sum(1 for b in self.blooms.values() if b.bloom_type == BloomType.ANOMALY_WEED)
        anomaly_ratio = anomaly_count / len(self.blooms)
        
        # Overall garden health
        garden_health = (avg_vitality * 0.4 + soil_health * 0.3 + 
                        bloom_diversity * 0.2 + (1.0 - anomaly_ratio) * 0.1)
        
        return {
            "garden_health": garden_health,
            "bloom_diversity": bloom_diversity,
            "average_vitality": avg_vitality,
            "soil_health": soil_health,
            "anomaly_ratio": anomaly_ratio,
            "total_blooms": len(self.blooms),
            "current_season": self.current_season.value,
            "seasonal_cycles": self.seasonal_cycle_count
        }


# Global renderer instance
_global_garden_renderer = None

def get_fractal_garden_renderer() -> FractalGardenRenderer:
    """Get global fractal garden renderer instance"""
    global _global_garden_renderer
    if _global_garden_renderer is None:
        _global_garden_renderer = FractalGardenRenderer()
    return _global_garden_renderer
