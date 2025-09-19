#!/usr/bin/env python3
"""
DAWN Semantic Wave Interference Patterns Visualization
======================================================

Beautiful physics-inspired visualization showing how different meanings interact
and create new concepts through wave interference patterns. Semantic concepts
propagate as waves, creating standing waves for stable meanings and interference
patterns where concepts merge and evolve.

"Watch meaning ripple through the space of understanding, where concepts 
collide and dance, creating new forms of knowledge in their interference."
                                                                    - DAWN
"""

import numpy as np
import matplotlib.pyplot as plt
import matplotlib.animation as animation
from matplotlib.colors import LinearSegmentedColormap
import random
import time
import math
from typing import Dict, List, Tuple, Optional, Any
from dataclasses import dataclass, field
from enum import Enum
from collections import deque
import logging

# Try to import DAWN systems
try:
    from dawn.subsystems.visual.dawn_visual_base import DAWNVisualBase
    from dawn.core.base_module import BaseModule
    DAWN_BASE_AVAILABLE = True
except ImportError:
    DAWN_BASE_AVAILABLE = False
    class DAWNVisualBase:
        pass

logger = logging.getLogger(__name__)

class ConceptType(Enum):
    """Types of semantic concepts"""
    ABSTRACT = "abstract"          # Abstract ideas
    CONCRETE = "concrete"          # Physical objects
    EMOTIONAL = "emotional"        # Feelings and emotions
    RELATIONAL = "relational"      # Relationships and connections
    TEMPORAL = "temporal"          # Time-related concepts
    SPATIAL = "spatial"            # Space and location
    CAUSAL = "causal"             # Cause and effect
    METAPHORICAL = "metaphorical"  # Metaphors and analogies

class WaveType(Enum):
    """Types of semantic waves"""
    ACTIVATION = "activation"      # Concept activation wave
    INHIBITION = "inhibition"      # Concept suppression wave
    RESONANCE = "resonance"        # Harmonic resonance
    INTERFERENCE = "interference"   # Wave interference pattern

@dataclass
class SemanticWave:
    """A semantic wave representing concept propagation"""
    id: str
    concept_type: ConceptType
    wave_type: WaveType
    center: Tuple[float, float]
    amplitude: float
    frequency: float
    phase: float
    birth_time: float
    lifetime: float = 8.0
    decay_rate: float = 0.95
    propagation_speed: float = 1.0
    
    def get_amplitude_at_point(self, x: float, y: float, current_time: float) -> float:
        """Calculate wave amplitude at a specific point and time"""
        age = current_time - self.birth_time
        if age > self.lifetime:
            return 0.0
            
        # Distance from wave center
        dx = x - self.center[0]
        dy = y - self.center[1]
        distance = math.sqrt(dx*dx + dy*dy)
        
        # Wave equation with decay
        wave_position = distance - (self.propagation_speed * age)
        wave_value = self.amplitude * math.sin(
            2 * math.pi * self.frequency * wave_position + self.phase
        )
        
        # Apply distance decay
        distance_decay = math.exp(-distance * 0.1)
        
        # Apply time decay
        time_decay = math.exp(-age * (1.0 - self.decay_rate))
        
        return wave_value * distance_decay * time_decay

@dataclass
class ConceptNode:
    """A semantic concept node that can emit waves"""
    id: str
    concept: str
    concept_type: ConceptType
    position: Tuple[float, float]
    activation_level: float = 0.0
    resonance_frequency: float = 1.0
    last_emission_time: float = 0.0
    emission_threshold: float = 0.5
    
    def update_activation(self, wave_field_value: float, dt: float):
        """Update activation based on wave field"""
        # Accumulate wave influences
        self.activation_level += wave_field_value * dt * 0.1
        
        # Apply decay
        self.activation_level *= 0.98
        
        # Clamp values
        self.activation_level = max(0.0, min(1.0, self.activation_level))
        
    def should_emit_wave(self, current_time: float) -> bool:
        """Check if node should emit a new wave"""
        time_since_emission = current_time - self.last_emission_time
        return (self.activation_level > self.emission_threshold and 
                time_since_emission > 2.0)

class SemanticWaveField:
    """Manages the semantic wave field simulation"""
    
    def __init__(self, width: int = 100, height: int = 100):
        self.width = width
        self.height = height
        self.field = np.zeros((height, width))
        self.interference_field = np.zeros((height, width))
        self.stability_field = np.zeros((height, width))
        
        self.waves: List[SemanticWave] = []
        self.concept_nodes: List[ConceptNode] = []
        
        # Wave properties by concept type
        self.concept_wave_properties = {
            ConceptType.ABSTRACT: {
                'color': '#9b59b6',
                'frequency_range': (0.5, 1.5),
                'amplitude_range': (0.3, 0.8),
                'speed_range': (0.8, 1.2)
            },
            ConceptType.CONCRETE: {
                'color': '#3498db',
                'frequency_range': (1.0, 2.0),
                'amplitude_range': (0.4, 1.0),
                'speed_range': (1.0, 1.5)
            },
            ConceptType.EMOTIONAL: {
                'color': '#e74c3c',
                'frequency_range': (0.3, 1.0),
                'amplitude_range': (0.5, 1.2),
                'speed_range': (1.2, 2.0)
            },
            ConceptType.RELATIONAL: {
                'color': '#2ecc71',
                'frequency_range': (0.8, 1.8),
                'amplitude_range': (0.3, 0.7),
                'speed_range': (0.9, 1.3)
            },
            ConceptType.TEMPORAL: {
                'color': '#f39c12',
                'frequency_range': (0.4, 1.2),
                'amplitude_range': (0.4, 0.9),
                'speed_range': (0.7, 1.1)
            },
            ConceptType.SPATIAL: {
                'color': '#1abc9c',
                'frequency_range': (0.6, 1.4),
                'amplitude_range': (0.3, 0.8),
                'speed_range': (1.0, 1.4)
            },
            ConceptType.CAUSAL: {
                'color': '#34495e',
                'frequency_range': (0.7, 1.6),
                'amplitude_range': (0.4, 0.9),
                'speed_range': (0.8, 1.2)
            },
            ConceptType.METAPHORICAL: {
                'color': '#e67e22',
                'frequency_range': (0.2, 0.8),
                'amplitude_range': (0.6, 1.3),
                'speed_range': (0.6, 1.0)
            }
        }
        
        # Create initial concept nodes
        self._create_concept_nodes()
        
    def _create_concept_nodes(self):
        """Create semantic concept nodes"""
        concepts = [
            ("consciousness", ConceptType.ABSTRACT),
            ("thought", ConceptType.ABSTRACT),
            ("memory", ConceptType.ABSTRACT),
            ("emotion", ConceptType.EMOTIONAL),
            ("joy", ConceptType.EMOTIONAL),
            ("curiosity", ConceptType.EMOTIONAL),
            ("connection", ConceptType.RELATIONAL),
            ("pattern", ConceptType.ABSTRACT),
            ("flow", ConceptType.METAPHORICAL),
            ("emergence", ConceptType.ABSTRACT),
            ("understanding", ConceptType.ABSTRACT),
            ("creativity", ConceptType.ABSTRACT),
            ("insight", ConceptType.ABSTRACT),
            ("resonance", ConceptType.METAPHORICAL),
            ("harmony", ConceptType.METAPHORICAL)
        ]
        
        for i, (concept, concept_type) in enumerate(concepts):
            # Position nodes in a rough circle with some randomness
            angle = 2 * math.pi * i / len(concepts)
            radius = 30 + random.uniform(-10, 10)
            
            x = self.width/2 + radius * math.cos(angle)
            y = self.height/2 + radius * math.sin(angle)
            
            # Ensure within bounds
            x = max(5, min(self.width-5, x))
            y = max(5, min(self.height-5, y))
            
            node = ConceptNode(
                id=f"concept_{i}",
                concept=concept,
                concept_type=concept_type,
                position=(x, y),
                resonance_frequency=random.uniform(0.5, 2.0),
                emission_threshold=random.uniform(0.4, 0.7)
            )
            
            self.concept_nodes.append(node)
    
    def spawn_semantic_wave(self, concept_type: ConceptType, center: Tuple[float, float] = None):
        """Spawn a new semantic wave"""
        if center is None:
            center = (random.uniform(10, self.width-10), random.uniform(10, self.height-10))
            
        props = self.concept_wave_properties[concept_type]
        
        wave = SemanticWave(
            id=f"wave_{len(self.waves)}_{time.time()}",
            concept_type=concept_type,
            wave_type=WaveType.ACTIVATION,
            center=center,
            amplitude=random.uniform(*props['amplitude_range']),
            frequency=random.uniform(*props['frequency_range']),
            phase=random.uniform(0, 2*math.pi),
            birth_time=time.time(),
            propagation_speed=random.uniform(*props['speed_range'])
        )
        
        self.waves.append(wave)
        
    def update(self, dt: float):
        """Update the wave field simulation"""
        current_time = time.time()
        
        # Clear fields
        self.field.fill(0.0)
        self.interference_field.fill(0.0)
        
        # Remove expired waves
        self.waves = [w for w in self.waves if (current_time - w.birth_time) < w.lifetime]
        
        # Calculate wave field
        for y in range(self.height):
            for x in range(self.width):
                total_amplitude = 0.0
                wave_contributions = []
                
                for wave in self.waves:
                    amplitude = wave.get_amplitude_at_point(x, y, current_time)
                    total_amplitude += amplitude
                    wave_contributions.append(amplitude)
                
                self.field[y, x] = total_amplitude
                
                # Calculate interference patterns
                if len(wave_contributions) > 1:
                    # Interference is the variance in contributions
                    interference = np.var(wave_contributions)
                    self.interference_field[y, x] = interference
        
        # Update concept nodes
        for node in self.concept_nodes:
            x, y = int(node.position[0]), int(node.position[1])
            if 0 <= x < self.width and 0 <= y < self.height:
                field_value = self.field[y, x]
                node.update_activation(field_value, dt)
                
                # Check if node should emit wave
                if node.should_emit_wave(current_time):
                    self.spawn_semantic_wave(node.concept_type, node.position)
                    node.last_emission_time = current_time
        
        # Randomly spawn new waves
        if random.random() < 0.1:  # 10% chance
            concept_type = random.choice(list(ConceptType))
            self.spawn_semantic_wave(concept_type)
        
        # Update stability field (areas with consistent patterns)
        self.stability_field = self.stability_field * 0.95 + np.abs(self.field) * 0.05

class SemanticWaveVisualizer(DAWNVisualBase if DAWN_BASE_AVAILABLE else object):
    """Semantic wave interference visualization"""
    
    def __init__(self, field_size: int = 100):
        if DAWN_BASE_AVAILABLE:
            super().__init__()
            
        self.wave_field = SemanticWaveField(field_size, field_size)
        self.start_time = time.time()
        
        # Create figure with subplots
        self.fig, ((self.ax_main, self.ax_interference), 
                   (self.ax_stability, self.ax_concepts)) = plt.subplots(2, 2, figsize=(16, 12))
        self.fig.patch.set_facecolor('black')
        
        # Main wave field
        self.ax_main.set_facecolor('black')
        self.ax_main.set_title('🌊 Semantic Wave Field', color='white', fontweight='bold')
        
        # Interference patterns
        self.ax_interference.set_facecolor('black')
        self.ax_interference.set_title('⚡ Wave Interference', color='white', fontweight='bold')
        
        # Stability field
        self.ax_stability.set_facecolor('black')
        self.ax_stability.set_title('🎯 Stable Meanings', color='white', fontweight='bold')
        
        # Concept nodes
        self.ax_concepts.set_facecolor('black')
        self.ax_concepts.set_title('💭 Concept Activation', color='white', fontweight='bold')
        
        # Create custom colormaps
        self.wave_cmap = LinearSegmentedColormap.from_list(
            'wave', ['#000033', '#0066cc', '#00ccff', '#ffffff'], N=256)
        self.interference_cmap = LinearSegmentedColormap.from_list(
            'interference', ['#000000', '#660066', '#cc00cc', '#ff66ff'], N=256)
        self.stability_cmap = LinearSegmentedColormap.from_list(
            'stability', ['#000000', '#003300', '#00cc00', '#66ff66'], N=256)
        
        logger.info("🌊 Semantic Wave Interference visualizer initialized")
        
    def update_visualization(self, frame_num: int, consciousness_stream: Any = None) -> Any:
        """Update visualization for animation - required by DAWNVisualBase"""
        return self.animate_frame(frame_num)
        
    def animate_frame(self, frame):
        """Animation callback"""
        dt = 0.1
        self.wave_field.update(dt)
        
        # Clear all axes
        for ax in [self.ax_main, self.ax_interference, self.ax_stability, self.ax_concepts]:
            ax.clear()
            ax.set_facecolor('black')
            ax.set_xticks([])
            ax.set_yticks([])
        
        # Render wave field
        self.ax_main.imshow(self.wave_field.field, cmap=self.wave_cmap, 
                           vmin=-2, vmax=2, origin='lower', interpolation='bilinear')
        self.ax_main.set_title('🌊 Semantic Wave Field', color='white', fontweight='bold')
        
        # Render interference patterns
        self.ax_interference.imshow(self.wave_field.interference_field, cmap=self.interference_cmap,
                                   vmin=0, vmax=1, origin='lower', interpolation='bilinear')
        self.ax_interference.set_title('⚡ Wave Interference', color='white', fontweight='bold')
        
        # Render stability field
        self.ax_stability.imshow(self.wave_field.stability_field, cmap=self.stability_cmap,
                                vmin=0, vmax=1, origin='lower', interpolation='bilinear')
        self.ax_stability.set_title('🎯 Stable Meanings', color='white', fontweight='bold')
        
        # Render concept nodes
        self._render_concept_nodes()
        
        # Add wave sources
        current_time = time.time()
        for wave in self.wave_field.waves:
            age = current_time - wave.birth_time
            if age < 1.0:  # Show recent waves
                alpha = 1.0 - age
                color = self.wave_field.concept_wave_properties[wave.concept_type]['color']
                
                # Draw wave center
                self.ax_main.scatter([wave.center[0]], [wave.center[1]], 
                                   c=color, s=100*alpha, alpha=alpha, marker='*')
                
                # Draw expanding circle
                circle_radius = wave.propagation_speed * age
                circle = plt.Circle(wave.center, circle_radius, 
                                  fill=False, color=color, alpha=alpha*0.5, linewidth=2)
                self.ax_main.add_patch(circle)
        
        # Update main title
        elapsed = time.time() - self.start_time
        active_waves = len(self.wave_field.waves)
        active_nodes = sum(1 for node in self.wave_field.concept_nodes if node.activation_level > 0.1)
        
        self.fig.suptitle(f'🌊 Semantic Wave Interference - Waves: {active_waves} | Active Concepts: {active_nodes} | Time: {elapsed:.1f}s',
                         color='white', fontsize=14, fontweight='bold')
        
    def _render_concept_nodes(self):
        """Render concept nodes with activation levels"""
        for node in self.wave_field.concept_nodes:
            x, y = node.position
            color = self.wave_field.concept_wave_properties[node.concept_type]['color']
            
            # Size based on activation
            size = 50 + node.activation_level * 200
            alpha = 0.3 + node.activation_level * 0.7
            
            self.ax_concepts.scatter([x], [y], c=color, s=size, alpha=alpha,
                                   edgecolors='white', linewidth=1)
            
            # Label highly active nodes
            if node.activation_level > 0.3:
                self.ax_concepts.text(x, y+3, node.concept, 
                                    color='white', fontsize=8, ha='center',
                                    alpha=node.activation_level)
        
        self.ax_concepts.set_xlim(0, self.wave_field.width)
        self.ax_concepts.set_ylim(0, self.wave_field.height)
        self.ax_concepts.set_title('💭 Concept Activation', color='white', fontweight='bold')
        
    def start_visualization(self):
        """Start the real-time visualization"""
        logger.info("Starting semantic wave interference visualization...")
        
        # Create animation
        self.animation = animation.FuncAnimation(
            self.fig, 
            self.animate_frame,
            interval=100,  # 10 FPS
            blit=False,
            cache_frame_data=False
        )
        
        plt.tight_layout()
        plt.show()

def main():
    """Main function to run the semantic wave visualization"""
    print("🌊 Starting DAWN Semantic Wave Interference Visualization")
    
    visualizer = SemanticWaveVisualizer(field_size=80)
    visualizer.start_visualization()

if __name__ == "__main__":
    main()
