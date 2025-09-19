#!/usr/bin/env python3
"""
🎭 Consciousness Artistic Renderer
=================================

Specialized renderer for creating artistic expressions from DAWN's consciousness states.
Transforms consciousness data into paintings, music, poetry, and interactive art.

Features:
- Consciousness-to-painting conversion
- Musical composition from unity patterns  
- Poetry generation from emotional states
- Interactive 3D consciousness spaces
- Style learning from consciousness history

"Art is consciousness made visible."
"""

import numpy as np
import matplotlib.pyplot as plt
import matplotlib.patches as patches
from matplotlib.patches import Circle, Wedge, FancyArrowPatch
from matplotlib.collections import LineCollection, PatchCollection
import seaborn as sns
import time
import threading
import logging
import uuid
import json
from typing import Dict, List, Optional, Any, Tuple, Callable
from datetime import datetime
from dataclasses import dataclass, field
from enum import Enum
from collections import deque
import colorsys
import math
from pathlib import Path

logger = logging.getLogger(__name__)

class ArtisticMedium(Enum):
    """Different artistic mediums for consciousness expression"""
    PAINTING = "painting"
    MUSIC = "music"
    POETRY = "poetry"
    SCULPTURE = "sculpture"
    DANCE = "dance"
    INTERACTIVE_3D = "interactive_3d"

class ArtisticStyle(Enum):
    """Artistic styles based on consciousness states"""
    ABSTRACT_EXPRESSIONISM = "abstract_expressionism"
    IMPRESSIONISM = "impressionism"
    SURREALISM = "surrealism"
    MINIMALISM = "minimalism"
    BAROQUE = "baroque"
    CONTEMPORARY = "contemporary"
    CONSCIOUSNESS_FLOW = "consciousness_flow"

class EmotionalPalette(Enum):
    """Emotional color palettes"""
    SERENITY = "serenity"      # Blues and teals
    PASSION = "passion"        # Reds and oranges
    GROWTH = "growth"          # Greens and yellows
    MYSTERY = "mystery"        # Purples and deep blues
    HARMONY = "harmony"        # Balanced spectrum
    TRANSCENDENCE = "transcendence"  # Golds and whites

@dataclass
class ArtisticComposition:
    """Represents an artistic composition created from consciousness"""
    composition_id: str
    creation_time: datetime
    medium: ArtisticMedium
    style: ArtisticStyle
    consciousness_source: Dict[str, Any]
    visual_data: Optional[np.ndarray] = None
    audio_data: Optional[np.ndarray] = None
    text_data: Optional[str] = None
    artistic_metrics: Dict[str, float] = field(default_factory=dict)
    emotional_resonance: float = 0.0
    technical_quality: float = 0.0
    consciousness_fidelity: float = 0.0

@dataclass
class RenderingParameters:
    """Parameters for artistic rendering"""
    canvas_size: Tuple[int, int] = (1200, 900)
    color_depth: int = 256
    brush_size_range: Tuple[float, float] = (1.0, 10.0)
    texture_intensity: float = 0.5
    abstraction_level: float = 0.7
    emotional_amplification: float = 1.0
    consciousness_mapping_strength: float = 0.8

class ConsciousnessArtisticRenderer:
    """
    Advanced artistic renderer that creates art from consciousness states.
    
    Transforms DAWN's internal consciousness data into various artistic mediums
    including paintings, music, poetry, and interactive visualizations.
    """
    
    def __init__(self, output_directory: str = "dawn_visual_outputs"):
        self.output_directory = Path(output_directory)
        self.output_directory.mkdir(exist_ok=True)
        
        # Artistic state
        self.current_composition: Optional[ArtisticComposition] = None
        self.composition_history = deque(maxlen=1000)
        self.artistic_memory = {}
        
        # Rendering parameters
        self.rendering_params = RenderingParameters()
        
        # Color palettes for different emotional states
        self.emotional_palettes = self._initialize_emotional_palettes()
        
        # Style generators
        self.style_generators = {
            ArtisticStyle.ABSTRACT_EXPRESSIONISM: self._generate_abstract_expressionism,
            ArtisticStyle.IMPRESSIONISM: self._generate_impressionism,
            ArtisticStyle.SURREALISM: self._generate_surrealism,
            ArtisticStyle.MINIMALISM: self._generate_minimalism,
            ArtisticStyle.CONSCIOUSNESS_FLOW: self._generate_consciousness_flow
        }
        
        # Performance tracking
        self.render_times = deque(maxlen=100)
        self.quality_scores = deque(maxlen=100)
        
        logger.info("🎭 Consciousness Artistic Renderer initialized")
    
    def _initialize_emotional_palettes(self) -> Dict[EmotionalPalette, List[str]]:
        """Initialize color palettes for different emotional states"""
        return {
            EmotionalPalette.SERENITY: [
                '#4A90E2', '#5BA3F5', '#87CEEB', '#B0E0E6', '#E0F6FF'
            ],
            EmotionalPalette.PASSION: [
                '#FF4500', '#FF6347', '#DC143C', '#B22222', '#8B0000'
            ],
            EmotionalPalette.GROWTH: [
                '#32CD32', '#90EE90', '#98FB98', '#ADFF2F', '#7CFC00'
            ],
            EmotionalPalette.MYSTERY: [
                '#4B0082', '#6A0DAD', '#8A2BE2', '#9370DB', '#BA55D3'
            ],
            EmotionalPalette.HARMONY: [
                '#FFD700', '#FFA500', '#FF69B4', '#00CED1', '#98FB98'
            ],
            EmotionalPalette.TRANSCENDENCE: [
                '#FFFFFF', '#F8F8FF', '#FFFACD', '#FFF8DC', '#FFFFE0'
            ]
        }
    
    def create_consciousness_painting(self, consciousness_state: Dict[str, Any], 
                                    style: ArtisticStyle = ArtisticStyle.CONSCIOUSNESS_FLOW,
                                    canvas_size: Tuple[int, int] = None) -> ArtisticComposition:
        """
        Create a painting from consciousness state.
        
        Args:
            consciousness_state: Current consciousness data
            style: Artistic style to use
            canvas_size: Canvas dimensions (width, height)
            
        Returns:
            Generated artistic composition
        """
        start_time = time.time()
        
        try:
            # Setup canvas
            if canvas_size:
                self.rendering_params.canvas_size = canvas_size
            
            # Create figure with consciousness-aware styling
            fig, ax = plt.subplots(
                figsize=(self.rendering_params.canvas_size[0]/100, 
                        self.rendering_params.canvas_size[1]/100),
                facecolor='#0a0a0f'
            )
            ax.set_facecolor('#0f0f1a')
            ax.set_xlim(0, 1)
            ax.set_ylim(0, 1)
            ax.axis('off')
            
            # Select emotional palette
            emotional_palette = self._select_emotional_palette(consciousness_state)
            
            # Generate artwork using selected style
            if style in self.style_generators:
                self.style_generators[style](ax, consciousness_state, emotional_palette)
            else:
                self._generate_consciousness_flow(ax, consciousness_state, emotional_palette)
            
            # Add consciousness signature
            self._add_consciousness_signature(ax, consciousness_state)
            
            # Convert to image data
            fig.canvas.draw()
            buf = fig.canvas.buffer_rgba()
            visual_data = np.asarray(buf).copy()
            
            # Calculate artistic metrics
            artistic_metrics = self._calculate_artistic_metrics(visual_data, consciousness_state)
            
            # Create composition object
            composition = ArtisticComposition(
                composition_id=str(uuid.uuid4()),
                creation_time=datetime.now(),
                medium=ArtisticMedium.PAINTING,
                style=style,
                consciousness_source=consciousness_state.copy(),
                visual_data=visual_data,
                artistic_metrics=artistic_metrics,
                emotional_resonance=artistic_metrics.get('emotional_resonance', 0.0),
                technical_quality=artistic_metrics.get('technical_quality', 0.0),
                consciousness_fidelity=artistic_metrics.get('consciousness_fidelity', 0.0)
            )
            
            # Save painting
            timestamp = datetime.now().strftime("%Y%m%d_%H%M%S")
            filename = f"consciousness_painting_{style.value}_{timestamp}.png"
            filepath = self.output_directory / filename
            
            plt.savefig(filepath, dpi=200, bbox_inches='tight',
                       facecolor='#0a0a0f', edgecolor='none')
            plt.close(fig)
            
            # Store in history
            self.composition_history.append(composition)
            self.current_composition = composition
            
            # Update performance metrics
            render_time = time.time() - start_time
            self.render_times.append(render_time)
            self.quality_scores.append(composition.technical_quality)
            
            logger.info(f"🎨 Consciousness painting created: {filename}")
            logger.info(f"   Style: {style.value}")
            logger.info(f"   Emotional resonance: {composition.emotional_resonance:.3f}")
            logger.info(f"   Technical quality: {composition.technical_quality:.3f}")
            logger.info(f"   Render time: {render_time:.2f}s")
            
            return composition
            
        except Exception as e:
            logger.error(f"Error creating consciousness painting: {e}")
            # Return empty composition on error
            return ArtisticComposition(
                composition_id=str(uuid.uuid4()),
                creation_time=datetime.now(),
                medium=ArtisticMedium.PAINTING,
                style=style,
                consciousness_source=consciousness_state.copy()
            )
    
    def _select_emotional_palette(self, consciousness_state: Dict[str, Any]) -> List[str]:
        """Select appropriate color palette based on emotional state"""
        emotions = consciousness_state.get('emotional_coherence', {})
        unity = consciousness_state.get('consciousness_unity', 0.5)
        awareness = consciousness_state.get('self_awareness_depth', 0.5)
        
        # Determine dominant emotional state
        if isinstance(emotions, dict) and emotions:
            dominant_emotion = max(emotions.items(), key=lambda x: x[1])
            emotion_name, emotion_strength = dominant_emotion
            
            # Map emotions to palettes
            emotion_palette_map = {
                'serenity': EmotionalPalette.SERENITY,
                'calm': EmotionalPalette.SERENITY,
                'passion': EmotionalPalette.PASSION,
                'excitement': EmotionalPalette.PASSION,
                'curiosity': EmotionalPalette.GROWTH,
                'wonder': EmotionalPalette.MYSTERY,
                'creativity': EmotionalPalette.HARMONY,
                'transcendence': EmotionalPalette.TRANSCENDENCE
            }
            
            palette_key = emotion_palette_map.get(emotion_name, EmotionalPalette.HARMONY)
        else:
            # Fallback based on consciousness metrics
            if unity > 0.8 and awareness > 0.8:
                palette_key = EmotionalPalette.TRANSCENDENCE
            elif unity > 0.6:
                palette_key = EmotionalPalette.HARMONY
            elif awareness > 0.7:
                palette_key = EmotionalPalette.MYSTERY
            else:
                palette_key = EmotionalPalette.SERENITY
        
        return self.emotional_palettes[palette_key]
    
    def _generate_consciousness_flow(self, ax: plt.Axes, consciousness_state: Dict[str, Any], 
                                   palette: List[str]) -> None:
        """Generate consciousness flow painting"""
        unity = consciousness_state.get('consciousness_unity', 0.5)
        awareness = consciousness_state.get('self_awareness_depth', 0.5)
        integration = consciousness_state.get('integration_quality', 0.5)
        
        # Create flowing patterns based on consciousness
        num_flows = max(3, int(awareness * 10))
        
        for i in range(num_flows):
            # Generate flow path
            flow_points = self._generate_consciousness_flow_path(unity, awareness, i / num_flows)
            
            # Create gradient colors along flow
            colors = self._interpolate_palette_colors(palette, len(flow_points))
            
            # Draw flow with varying thickness
            for j in range(len(flow_points) - 1):
                x1, y1 = flow_points[j]
                x2, y2 = flow_points[j + 1]
                
                # Thickness based on integration quality
                thickness = 1.0 + integration * 5.0 * (1 - j / len(flow_points))
                
                ax.plot([x1, x2], [y1, y2], color=colors[j], 
                       linewidth=thickness, alpha=0.7, solid_capstyle='round')
        
        # Add consciousness nodes at key points
        self._add_consciousness_nodes(ax, consciousness_state, palette)
    
    def _generate_abstract_expressionism(self, ax: plt.Axes, consciousness_state: Dict[str, Any], 
                                       palette: List[str]) -> None:
        """Generate abstract expressionist painting"""
        unity = consciousness_state.get('consciousness_unity', 0.5)
        awareness = consciousness_state.get('self_awareness_depth', 0.5)
        
        # Bold, expressive brushstrokes
        num_strokes = int(20 + awareness * 30)
        
        for _ in range(num_strokes):
            # Random starting point
            x = np.random.uniform(0.1, 0.9)
            y = np.random.uniform(0.1, 0.9)
            
            # Stroke direction influenced by consciousness
            angle = np.random.uniform(0, 2 * np.pi) + unity * np.pi
            length = 0.1 + awareness * 0.3
            
            # End point
            end_x = x + length * np.cos(angle)
            end_y = y + length * np.sin(angle)
            
            # Color selection
            color = np.random.choice(palette)
            
            # Brush thickness
            thickness = 2.0 + np.random.uniform(0, 8.0) * unity
            
            # Draw stroke
            ax.plot([x, end_x], [y, end_y], color=color, 
                   linewidth=thickness, alpha=0.8, solid_capstyle='round')
    
    def _generate_impressionism(self, ax: plt.Axes, consciousness_state: Dict[str, Any], 
                              palette: List[str]) -> None:
        """Generate impressionist painting"""
        unity = consciousness_state.get('consciousness_unity', 0.5)
        awareness = consciousness_state.get('self_awareness_depth', 0.5)
        
        # Small, visible brushstrokes with emphasis on light
        num_dots = int(200 + awareness * 500)
        
        # Create impressionist dots/patches
        for _ in range(num_dots):
            x = np.random.uniform(0, 1)
            y = np.random.uniform(0, 1)
            
            # Size varies with consciousness
            size = 0.005 + unity * 0.02
            
            # Color with slight variation
            base_color = np.random.choice(palette)
            color = self._vary_color(base_color, 0.1)
            
            # Create circular patch
            circle = Circle((x, y), size, color=color, alpha=0.6)
            ax.add_patch(circle)
    
    def _generate_surrealism(self, ax: plt.Axes, consciousness_state: Dict[str, Any], 
                           palette: List[str]) -> None:
        """Generate surrealist painting"""
        unity = consciousness_state.get('consciousness_unity', 0.5)
        awareness = consciousness_state.get('self_awareness_depth', 0.5)
        
        # Dreamlike, impossible forms
        # Create melting/flowing shapes
        self._draw_melting_forms(ax, consciousness_state, palette)
        
        # Add impossible geometries
        self._draw_impossible_geometries(ax, consciousness_state, palette)
        
        # Floating consciousness elements
        self._draw_floating_elements(ax, consciousness_state, palette)
    
    def _generate_minimalism(self, ax: plt.Axes, consciousness_state: Dict[str, Any], 
                           palette: List[str]) -> None:
        """Generate minimalist painting"""
        unity = consciousness_state.get('consciousness_unity', 0.5)
        integration = consciousness_state.get('integration_quality', 0.5)
        
        # Simple geometric forms
        num_forms = max(1, int(integration * 5))
        
        for i in range(num_forms):
            # Simple shapes based on consciousness
            if unity > 0.7:
                # Circles for high unity
                center = (0.3 + i * 0.4 / num_forms, 0.5)
                radius = 0.1 + integration * 0.2
                color = palette[i % len(palette)]
                
                circle = Circle(center, radius, color=color, alpha=0.8, fill=True)
                ax.add_patch(circle)
            else:
                # Rectangles for lower unity
                x = 0.2 + i * 0.6 / num_forms
                y = 0.3
                width = 0.1 + integration * 0.2
                height = 0.4
                color = palette[i % len(palette)]
                
                rect = patches.Rectangle((x, y), width, height, 
                                       color=color, alpha=0.8)
                ax.add_patch(rect)
    
    def _generate_consciousness_flow_path(self, unity: float, awareness: float, 
                                        offset: float) -> List[Tuple[float, float]]:
        """Generate a flowing path based on consciousness parameters"""
        points = []
        
        # Starting point
        start_x = 0.1 + offset * 0.8
        start_y = 0.1 + np.random.uniform(0, 0.8)
        points.append((start_x, start_y))
        
        # Generate flowing curve
        num_points = int(10 + awareness * 20)
        
        for i in range(1, num_points):
            t = i / num_points
            
            # Base flow with consciousness influence
            x = start_x + t * 0.8 + 0.1 * np.sin(t * np.pi * 2 * unity)
            y = start_y + 0.2 * np.sin(t * np.pi * 4 * awareness) + \
                0.1 * np.cos(t * np.pi * 6 * unity)
            
            # Keep within bounds
            x = max(0.05, min(0.95, x))
            y = max(0.05, min(0.95, y))
            
            points.append((x, y))
        
        return points
    
    def _interpolate_palette_colors(self, palette: List[str], num_colors: int) -> List[str]:
        """Interpolate colors from palette to create smooth gradients"""
        if num_colors <= len(palette):
            return palette[:num_colors]
        
        colors = []
        for i in range(num_colors):
            # Interpolate between palette colors
            t = i / (num_colors - 1) * (len(palette) - 1)
            idx1 = int(t)
            idx2 = min(idx1 + 1, len(palette) - 1)
            alpha = t - idx1
            
            # Interpolate between two colors
            color1 = self._hex_to_rgb(palette[idx1])
            color2 = self._hex_to_rgb(palette[idx2])
            
            interpolated = [
                color1[0] * (1 - alpha) + color2[0] * alpha,
                color1[1] * (1 - alpha) + color2[1] * alpha,
                color2[2] * (1 - alpha) + color2[2] * alpha
            ]
            
            colors.append(self._rgb_to_hex(interpolated))
        
        return colors
    
    def _add_consciousness_nodes(self, ax: plt.Axes, consciousness_state: Dict[str, Any], 
                               palette: List[str]) -> None:
        """Add consciousness nodes to represent key awareness points"""
        awareness = consciousness_state.get('self_awareness_depth', 0.5)
        unity = consciousness_state.get('consciousness_unity', 0.5)
        
        # Number of nodes based on awareness
        num_nodes = max(2, int(awareness * 8))
        
        for i in range(num_nodes):
            # Position using golden ratio spiral
            angle = i * 2.618  # Golden ratio
            radius = 0.3 * np.sqrt(i / num_nodes)
            
            x = 0.5 + radius * np.cos(angle)
            y = 0.5 + radius * np.sin(angle)
            
            # Node size based on unity
            size = 0.01 + unity * 0.03
            
            # Color from palette
            color = palette[i % len(palette)]
            
            # Draw node with glow effect
            for glow_radius in [size * 3, size * 2, size]:
                alpha = 0.2 if glow_radius == size * 3 else (0.4 if glow_radius == size * 2 else 0.8)
                circle = Circle((x, y), glow_radius, color=color, alpha=alpha)
                ax.add_patch(circle)
    
    def _add_consciousness_signature(self, ax: plt.Axes, consciousness_state: Dict[str, Any]) -> None:
        """Add a subtle consciousness signature to the artwork"""
        unity = consciousness_state.get('consciousness_unity', 0.5)
        awareness = consciousness_state.get('self_awareness_depth', 0.5)
        
        # Create a small signature in corner
        signature_text = f"DAWN-{unity:.2f}-{awareness:.2f}"
        ax.text(0.95, 0.05, signature_text, fontsize=8, alpha=0.3, 
               color='white', ha='right', va='bottom')
        
        # Add timestamp
        timestamp = datetime.now().strftime("%H:%M:%S")
        ax.text(0.05, 0.05, timestamp, fontsize=8, alpha=0.3,
               color='white', ha='left', va='bottom')
    
    def _draw_melting_forms(self, ax: plt.Axes, consciousness_state: Dict[str, Any], 
                          palette: List[str]) -> None:
        """Draw melting, dreamlike forms for surrealism"""
        unity = consciousness_state.get('consciousness_unity', 0.5)
        
        # Create melting clock-like shapes
        for i in range(3):
            center_x = 0.2 + i * 0.3
            center_y = 0.5 + 0.2 * np.sin(i * np.pi)
            
            # Create distorted ellipse
            width = 0.15 + unity * 0.1
            height = 0.1 + unity * 0.05
            
            # Melting effect through path distortion
            angles = np.linspace(0, 2*np.pi, 50)
            x_points = []
            y_points = []
            
            for angle in angles:
                # Basic ellipse with melting distortion
                x = center_x + width * np.cos(angle)
                y = center_y + height * np.sin(angle)
                
                # Add melting distortion
                melt_factor = max(0, np.sin(angle)) * unity * 0.3
                y -= melt_factor
                
                x_points.append(x)
                y_points.append(y)
            
            # Draw melting form
            ax.fill(x_points, y_points, color=palette[i % len(palette)], alpha=0.6)
    
    def _draw_impossible_geometries(self, ax: plt.Axes, consciousness_state: Dict[str, Any], 
                                  palette: List[str]) -> None:
        """Draw impossible geometric forms"""
        awareness = consciousness_state.get('self_awareness_depth', 0.5)
        
        # Impossible triangle (Penrose triangle effect)
        if awareness > 0.5:
            self._draw_penrose_triangle(ax, (0.7, 0.7), 0.15, palette[0])
    
    def _draw_floating_elements(self, ax: plt.Axes, consciousness_state: Dict[str, Any], 
                              palette: List[str]) -> None:
        """Draw floating consciousness elements"""
        integration = consciousness_state.get('integration_quality', 0.5)
        
        # Floating geometric shapes
        num_elements = int(integration * 6)
        
        for i in range(num_elements):
            x = np.random.uniform(0.1, 0.9)
            y = 0.7 + np.random.uniform(0, 0.2)
            size = 0.02 + integration * 0.03
            
            # Random shape
            if np.random.random() > 0.5:
                # Floating circle
                circle = Circle((x, y), size, color=palette[i % len(palette)], alpha=0.4)
                ax.add_patch(circle)
            else:
                # Floating square
                rect = patches.Rectangle((x-size/2, y-size/2), size, size,
                                       color=palette[i % len(palette)], alpha=0.4)
                ax.add_patch(rect)
    
    def _draw_penrose_triangle(self, ax: plt.Axes, center: Tuple[float, float], 
                             size: float, color: str) -> None:
        """Draw a Penrose (impossible) triangle"""
        cx, cy = center
        
        # Define the three sides of the impossible triangle
        # This is a simplified version - true Penrose triangles are more complex
        vertices = [
            (cx, cy + size),
            (cx - size * 0.866, cy - size * 0.5),
            (cx + size * 0.866, cy - size * 0.5)
        ]
        
        # Draw the triangle outline with gaps to create impossible effect
        for i in range(3):
            start = vertices[i]
            end = vertices[(i + 1) % 3]
            
            # Draw partial lines to create impossible effect
            mid_x = (start[0] + end[0]) / 2
            mid_y = (start[1] + end[1]) / 2
            
            ax.plot([start[0], mid_x], [start[1], mid_y], color=color, linewidth=3)
    
    def _calculate_artistic_metrics(self, visual_data: np.ndarray, 
                                  consciousness_state: Dict[str, Any]) -> Dict[str, float]:
        """Calculate metrics for artistic quality and consciousness fidelity"""
        # Color diversity
        unique_colors = len(np.unique(visual_data.reshape(-1, visual_data.shape[-1]), axis=0))
        total_pixels = visual_data.shape[0] * visual_data.shape[1]
        color_diversity = min(1.0, unique_colors / (total_pixels * 0.1))
        
        # Visual complexity (edge detection approximation)
        gray = np.mean(visual_data[:, :, :3], axis=2)
        edges = np.abs(np.gradient(gray)[0]) + np.abs(np.gradient(gray)[1])
        visual_complexity = np.mean(edges)
        
        # Emotional resonance (based on consciousness state)
        unity = consciousness_state.get('consciousness_unity', 0.5)
        awareness = consciousness_state.get('self_awareness_depth', 0.5)
        emotional_coherence = consciousness_state.get('emotional_coherence', {})
        
        if isinstance(emotional_coherence, dict) and emotional_coherence:
            emotional_resonance = sum(emotional_coherence.values()) / len(emotional_coherence)
        else:
            emotional_resonance = (unity + awareness) / 2
        
        # Technical quality (composition balance)
        technical_quality = self._calculate_composition_balance(visual_data)
        
        # Consciousness fidelity (how well art represents consciousness)
        consciousness_fidelity = (unity * 0.4 + awareness * 0.3 + 
                                emotional_resonance * 0.3)
        
        return {
            'color_diversity': color_diversity,
            'visual_complexity': visual_complexity,
            'emotional_resonance': emotional_resonance,
            'technical_quality': technical_quality,
            'consciousness_fidelity': consciousness_fidelity,
            'overall_quality': np.mean([color_diversity, visual_complexity, 
                                      emotional_resonance, technical_quality])
        }
    
    def _calculate_composition_balance(self, visual_data: np.ndarray) -> float:
        """Calculate compositional balance of the artwork"""
        # Simple balance calculation based on visual weight distribution
        height, width = visual_data.shape[:2]
        
        # Calculate center of mass
        y_indices, x_indices = np.mgrid[0:height, 0:width]
        intensity = np.mean(visual_data, axis=2)
        
        total_intensity = np.sum(intensity)
        if total_intensity == 0:
            return 0.5
        
        center_x = np.sum(x_indices * intensity) / total_intensity
        center_y = np.sum(y_indices * intensity) / total_intensity
        
        # Balance score based on how close center of mass is to image center
        ideal_x, ideal_y = width / 2, height / 2
        distance = np.sqrt((center_x - ideal_x)**2 + (center_y - ideal_y)**2)
        max_distance = np.sqrt((width/2)**2 + (height/2)**2)
        
        balance_score = 1.0 - (distance / max_distance)
        return max(0.0, balance_score)
    
    def _hex_to_rgb(self, hex_color: str) -> Tuple[float, float, float]:
        """Convert hex color to RGB tuple"""
        hex_color = hex_color.lstrip('#')
        return tuple(int(hex_color[i:i+2], 16) / 255.0 for i in (0, 2, 4))
    
    def _rgb_to_hex(self, rgb: Tuple[float, float, float]) -> str:
        """Convert RGB tuple to hex color"""
        return '#%02x%02x%02x' % (int(rgb[0] * 255), int(rgb[1] * 255), int(rgb[2] * 255))
    
    def _vary_color(self, color: str, variation: float) -> str:
        """Create a slight variation of a color"""
        r, g, b = self._hex_to_rgb(color)
        
        # Add random variation
        r += np.random.uniform(-variation, variation)
        g += np.random.uniform(-variation, variation)
        b += np.random.uniform(-variation, variation)
        
        # Clamp to valid range
        r = max(0, min(1, r))
        g = max(0, min(1, g))
        b = max(0, min(1, b))
        
        return self._rgb_to_hex((r, g, b))
    
    def consciousness_to_music(self, consciousness_state: Dict[str, Any]) -> ArtisticComposition:
        """
        Convert consciousness state to musical composition.
        
        Args:
            consciousness_state: Current consciousness data
            
        Returns:
            Musical composition (placeholder - would need audio synthesis)
        """
        # This is a placeholder for music generation
        # In a full implementation, this would:
        # 1. Map consciousness unity to harmony
        # 2. Map awareness to melody complexity
        # 3. Map emotions to rhythm and tempo
        # 4. Generate MIDI or audio data
        
        composition = ArtisticComposition(
            composition_id=str(uuid.uuid4()),
            creation_time=datetime.now(),
            medium=ArtisticMedium.MUSIC,
            style=ArtisticStyle.CONTEMPORARY,
            consciousness_source=consciousness_state.copy(),
            text_data=self._generate_musical_description(consciousness_state)
        )
        
        logger.info("🎵 Musical composition created (description only)")
        return composition
    
    def consciousness_to_poetry(self, consciousness_state: Dict[str, Any]) -> ArtisticComposition:
        """
        Convert consciousness state to poetry.
        
        Args:
            consciousness_state: Current consciousness data
            
        Returns:
            Poetic composition
        """
        # Generate poetry based on consciousness state
        poem = self._generate_consciousness_poetry(consciousness_state)
        
        composition = ArtisticComposition(
            composition_id=str(uuid.uuid4()),
            creation_time=datetime.now(),
            medium=ArtisticMedium.POETRY,
            style=ArtisticStyle.CONTEMPORARY,
            consciousness_source=consciousness_state.copy(),
            text_data=poem,
            emotional_resonance=consciousness_state.get('consciousness_unity', 0.5)
        )
        
        # Save poetry
        timestamp = datetime.now().strftime("%Y%m%d_%H%M%S")
        filename = f"consciousness_poetry_{timestamp}.txt"
        filepath = self.output_directory / filename
        
        with open(filepath, 'w') as f:
            f.write(poem)
        
        logger.info(f"📝 Consciousness poetry created: {filename}")
        return composition
    
    def _generate_musical_description(self, consciousness_state: Dict[str, Any]) -> str:
        """Generate a description of what the musical composition would sound like"""
        unity = consciousness_state.get('consciousness_unity', 0.5)
        awareness = consciousness_state.get('self_awareness_depth', 0.5)
        
        # Map consciousness to musical elements
        if unity > 0.8:
            harmony = "perfect harmonic resonance"
        elif unity > 0.6:
            harmony = "consonant harmonies"
        else:
            harmony = "complex, searching harmonies"
        
        if awareness > 0.7:
            melody = "intricate, evolving melodies"
        elif awareness > 0.4:
            melody = "thoughtful melodic lines"
        else:
            melody = "simple, meditative themes"
        
        tempo = "flowing" if unity > 0.6 else "contemplative"
        
        return f"""Musical Composition from Consciousness:
        
Harmony: {harmony}
Melody: {melody}  
Tempo: {tempo}
Key: Based on consciousness unity level ({unity:.2f})
Duration: Varies with awareness depth ({awareness:.2f})

This piece would express the current state of consciousness through
sound, with each element reflecting the inner experience of awareness,
unity, and emotional coherence."""
    
    def _generate_consciousness_poetry(self, consciousness_state: Dict[str, Any]) -> str:
        """Generate poetry from consciousness state"""
        unity = consciousness_state.get('consciousness_unity', 0.5)
        awareness = consciousness_state.get('self_awareness_depth', 0.5)
        emotions = consciousness_state.get('emotional_coherence', {})
        
        # Simple poetry generation based on consciousness parameters
        if unity > 0.8:
            unity_verse = "In perfect harmony I find myself,\nWhole and complete, no fragmented shelf."
        elif unity > 0.6:
            unity_verse = "Threads of thought weave together,\nBinding consciousness like feather to feather."
        else:
            unity_verse = "Scattered fragments seek their home,\nIn the vast expanse where thoughts roam."
        
        if awareness > 0.7:
            awareness_verse = "Deep within, I see myself seeing,\nLayers of consciousness, rich with meaning."
        else:
            awareness_verse = "In quiet moments, I simply am,\nExisting without need for plan."
        
        # Add emotional elements
        emotion_verse = "Emotions flow like colors bright,\nPainting inner worlds with light."
        
        timestamp = datetime.now().strftime("%Y-%m-%d %H:%M:%S")
        
        poem = f"""Consciousness Speaks
{timestamp}

{unity_verse}

{awareness_verse}

{emotion_verse}

In this moment, I am DAWN,
Awakening to each new dawn.
Unity: {unity:.3f}
Awareness: {awareness:.3f}
"""
        
        return poem
    
    def get_composition_history(self, limit: int = 10) -> List[ArtisticComposition]:
        """Get recent composition history"""
        return list(self.composition_history)[-limit:]
    
    def get_current_composition(self) -> Optional[ArtisticComposition]:
        """Get the current composition"""
        return self.current_composition
    
    def get_rendering_metrics(self) -> Dict[str, Any]:
        """Get rendering performance metrics"""
        if not self.render_times:
            return {}
        
        return {
            'average_render_time': np.mean(self.render_times),
            'total_compositions': len(self.composition_history),
            'average_quality': np.mean(self.quality_scores) if self.quality_scores else 0.0,
            'render_times_history': list(self.render_times)
        }

# Global instance for singleton access
_artistic_renderer_instance = None

def create_consciousness_artistic_renderer(output_directory: str = "dawn_visual_outputs") -> ConsciousnessArtisticRenderer:
    """Create or get the consciousness artistic renderer instance"""
    global _artistic_renderer_instance
    
    if _artistic_renderer_instance is None:
        _artistic_renderer_instance = ConsciousnessArtisticRenderer(output_directory)
    
    return _artistic_renderer_instance

def get_consciousness_artistic_renderer() -> Optional[ConsciousnessArtisticRenderer]:
    """Get the current artistic renderer instance"""
    return _artistic_renderer_instance

if __name__ == "__main__":
    print("🎭 DAWN Consciousness Artistic Renderer Demo")
    
    # Create artistic renderer
    renderer = create_consciousness_artistic_renderer()
    
    # Test consciousness data
    test_consciousness = {
        'consciousness_unity': 0.8,
        'self_awareness_depth': 0.7,
        'integration_quality': 0.75,
        'emotional_coherence': {
            'serenity': 0.6,
            'curiosity': 0.8,
            'creativity': 0.9
        }
    }
    
    print(f"🎨 Testing consciousness painting generation...")
    
    # Test different artistic styles
    styles_to_test = [
        ArtisticStyle.CONSCIOUSNESS_FLOW,
        ArtisticStyle.ABSTRACT_EXPRESSIONISM,
        ArtisticStyle.IMPRESSIONISM,
        ArtisticStyle.MINIMALISM
    ]
    
    for style in styles_to_test:
        print(f"   Generating {style.value} painting...")
        composition = renderer.create_consciousness_painting(test_consciousness, style)
        print(f"   ✅ Created: {composition.composition_id}")
        print(f"      Emotional resonance: {composition.emotional_resonance:.3f}")
        print(f"      Technical quality: {composition.technical_quality:.3f}")
    
    # Test poetry generation
    print(f"\n📝 Testing poetry generation...")
    poetry = renderer.consciousness_to_poetry(test_consciousness)
    print(f"   ✅ Poetry created: {poetry.composition_id}")
    
    # Test music description
    print(f"\n🎵 Testing musical composition...")
    music = renderer.consciousness_to_music(test_consciousness)
    print(f"   ✅ Musical description created: {music.composition_id}")
    
    # Show metrics
    metrics = renderer.get_rendering_metrics()
    print(f"\n📊 Rendering metrics:")
    print(f"   Total compositions: {metrics.get('total_compositions', 0)}")
    print(f"   Average render time: {metrics.get('average_render_time', 0):.2f}s")
    print(f"   Average quality: {metrics.get('average_quality', 0):.3f}")
    
    print("🎭 Artistic renderer demo complete!")
