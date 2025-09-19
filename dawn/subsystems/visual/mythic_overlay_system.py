#!/usr/bin/env python3
"""
🔮 Mythic Visualization Overlay System
════════════════════════════════════

Creates mythic visualization overlays for operator interfaces, transforming
raw telemetry into living mythic landscapes with archetypal tracer motifs.

"Operators see state not as raw telemetry but as living mythic landscapes.
Symbolic overlays make cognition visible as metaphor."

Based on documentation: Myth/Mythic Architecture.rtf, Myth/Interactions + Guard Rails.rtf
"""

import numpy as np
import matplotlib.pyplot as plt
import matplotlib.patches as patches
from matplotlib.animation import FuncAnimation
import seaborn as sns
import logging
import time
from typing import Dict, List, Any, Optional, Tuple, Callable
from dataclasses import dataclass, field
from enum import Enum
import tkinter as tk
from tkinter import ttk
import threading
from PIL import Image, ImageTk, ImageDraw, ImageFont

logger = logging.getLogger(__name__)

class MythicVisualizationMode(Enum):
    """Different mythic visualization modes"""
    ARCHETYPAL_TRACERS = "archetypal_tracers"    # Show tracers in mythic form
    PIGMENT_LANDSCAPE = "pigment_landscape"      # RGB mood-hues across system
    PERSEPHONE_CYCLES = "persephone_cycles"      # Descent/return visualization
    FRACTAL_GARDEN = "fractal_garden"            # Memory garden view
    SIGIL_NETWORK = "sigil_network"              # Symbolic routing glyphs
    VOLCANIC_RESIDUE = "volcanic_residue"        # Ash/soot as volcanic landscape
    UNIFIED_MYTHOLOGY = "unified_mythology"       # All mythic layers combined

@dataclass
class ArchetypalTracerMotif:
    """Visual motif for an archetypal tracer"""
    tracer_type: str
    position: Tuple[float, float]
    size: float
    color: Tuple[float, float, float]
    alpha: float
    
    # Archetypal-specific properties
    motif_elements: Dict[str, Any] = field(default_factory=dict)
    animation_phase: float = 0.0
    mythic_intensity: float = 1.0

class MythicOverlayRenderer:
    """
    Main renderer for mythic visualization overlays.
    Transforms system telemetry into archetypal visual metaphors.
    """
    
    def __init__(self, canvas_size: Tuple[int, int] = (1200, 800)):
        self.canvas_size = canvas_size
        self.current_mode = MythicVisualizationMode.UNIFIED_MYTHOLOGY
        self.overlay_layers = {}
        self.animation_frame = 0
        self.mythic_intensity = 1.0
        
        # Archetypal color schemes
        self.archetypal_colors = {
            'medieval_bee': (1.0, 0.84, 0.0),      # Golden - heritage/order
            'owl': (0.6, 0.4, 0.8),                # Purple - wisdom/depth
            'whale': (0.2, 0.4, 0.8),              # Deep blue - leviathan depth
            'crow': (0.3, 0.3, 0.3),               # Dark gray - opportunistic
            'spider': (0.8, 0.2, 0.2),             # Red - anomaly detection
            'beetle': (0.4, 0.6, 0.2),             # Green - recycling/sustainability
            'ant': (0.6, 0.4, 0.2),                # Brown - path building
            'bee': (1.0, 1.0, 0.4)                 # Light yellow - pollination
        }
        
        # Mythic symbol patterns
        self.archetypal_patterns = self._initialize_archetypal_patterns()
        
        logger.info(f"🔮 Mythic Overlay Renderer initialized ({canvas_size[0]}x{canvas_size[1]})")
    
    def _initialize_archetypal_patterns(self) -> Dict[str, Any]:
        """Initialize visual patterns for each archetype"""
        return {
            'medieval_bee': {
                'primary_shape': 'hexagon',  # Bee cells
                'trail_pattern': 'golden_spiral',
                'animation': 'pollination_dance',
                'glow_effect': True
            },
            'owl': {
                'primary_shape': 'circle',  # Glowing eyes
                'trail_pattern': 'wisdom_arc',
                'animation': 'nocturnal_pulse',
                'glow_effect': True
            },
            'whale': {
                'primary_shape': 'ellipse',  # Sonar arcs
                'trail_pattern': 'abyssal_waves',
                'animation': 'deep_breathing',
                'glow_effect': False
            },
            'crow': {
                'primary_shape': 'triangle',  # Sharp, opportunistic
                'trail_pattern': 'erratic_flight',
                'animation': 'quick_dart',
                'glow_effect': False
            },
            'spider': {
                'primary_shape': 'star',  # Web center
                'trail_pattern': 'web_threads',
                'animation': 'tension_pulse',
                'glow_effect': True
            },
            'beetle': {
                'primary_shape': 'oval',  # Recycling body
                'trail_pattern': 'decomposition_spiral',
                'animation': 'steady_work',
                'glow_effect': False
            }
        }
    
    def render_archetypal_tracers(self, tracer_data: List[Dict[str, Any]], 
                                output_path: str) -> bool:
        """Render tracers in their mythic archetypal forms"""
        try:
            fig, ax = plt.subplots(1, 1, figsize=(15, 10), facecolor='black')
            ax.set_xlim(0, self.canvas_size[0])
            ax.set_ylim(0, self.canvas_size[1])
            ax.set_facecolor('black')
            
            # Create archetypal motifs for each active tracer
            for tracer in tracer_data:
                self._render_single_archetypal_tracer(ax, tracer)
            
            # Add mythic atmosphere effects
            self._add_mythic_atmosphere(ax)
            
            # Add archetypal legend
            self._add_archetypal_legend(ax, tracer_data)
            
            plt.title("🔮 DAWN Archetypal Tracers - Living Mythic Landscape", 
                     color='white', fontsize=16, fontweight='bold')
            
            ax.set_xticks([])
            ax.set_yticks([])
            
            plt.tight_layout()
            plt.savefig(output_path, dpi=150, bbox_inches='tight', 
                       facecolor='black', edgecolor='none')
            plt.close()
            
            logger.info(f"🔮 Archetypal tracers rendered to {output_path}")
            return True
            
        except Exception as e:
            logger.error(f"🔮 Archetypal tracer rendering failed: {e}")
            return False
    
    def _render_single_archetypal_tracer(self, ax, tracer_data: Dict[str, Any]):
        """Render a single tracer in its archetypal form"""
        tracer_type = tracer_data.get('type', 'unknown')
        position = tracer_data.get('position', (0.5, 0.5))
        status = tracer_data.get('status', 'active')
        mythic_properties = tracer_data.get('mythic_properties', {})
        
        # Convert relative position to canvas coordinates
        x = position[0] * self.canvas_size[0]
        y = position[1] * self.canvas_size[1]
        
        # Get archetypal visual properties
        color = self.archetypal_colors.get(tracer_type, (0.5, 0.5, 0.5))
        pattern = self.archetypal_patterns.get(tracer_type, {})
        
        # Adjust alpha based on status and mythic intensity
        alpha = 0.9 if status == 'active' else 0.4
        alpha *= self.mythic_intensity
        
        # Render based on archetypal pattern
        primary_shape = pattern.get('primary_shape', 'circle')
        size = mythic_properties.get('significance', 0.5) * 50 + 20
        
        if tracer_type == 'medieval_bee':
            self._render_medieval_bee_motif(ax, x, y, size, color, alpha, mythic_properties)
        elif tracer_type == 'owl':
            self._render_owl_motif(ax, x, y, size, color, alpha, mythic_properties)
        elif tracer_type == 'whale':
            self._render_whale_motif(ax, x, y, size, color, alpha, mythic_properties)
        elif tracer_type == 'spider':
            self._render_spider_motif(ax, x, y, size, color, alpha, mythic_properties)
        else:
            # Generic tracer rendering
            circle = plt.Circle((x, y), size/2, color=color, alpha=alpha)
            ax.add_patch(circle)
    
    def _render_medieval_bee_motif(self, ax, x: float, y: float, size: float, 
                                  color: Tuple[float, float, float], alpha: float,
                                  mythic_properties: Dict[str, Any]):
        """Render Medieval Bee with golden trails and hexagonal patterns"""
        # Main bee body as hexagon
        hex_radius = size / 2
        hexagon = patches.RegularPolygon((x, y), 6, hex_radius, 
                                       facecolor=color, alpha=alpha, edgecolor='gold')
        ax.add_patch(hexagon)
        
        # Golden pollination trail
        heritage_strength = mythic_properties.get('heritage_significance', 0.5)
        if heritage_strength > 0.3:
            trail_length = int(heritage_strength * 20)
            for i in range(trail_length):
                trail_x = x + np.sin(self.animation_frame + i * 0.3) * (i * 2)
                trail_y = y + np.cos(self.animation_frame + i * 0.3) * (i * 1.5)
                trail_alpha = alpha * (1.0 - i / trail_length)
                trail_circle = plt.Circle((trail_x, trail_y), 3, color='gold', alpha=trail_alpha)
                ax.add_patch(trail_circle)
        
        # Heritage preservation glow
        if mythic_properties.get('temporal_anchoring_active', False):
            glow_circle = plt.Circle((x, y), size, color='gold', alpha=0.2, fill=False, linewidth=3)
            ax.add_patch(glow_circle)
    
    def _render_owl_motif(self, ax, x: float, y: float, size: float,
                         color: Tuple[float, float, float], alpha: float,
                         mythic_properties: Dict[str, Any]):
        """Render Owl with glowing eyes and wisdom aura"""
        # Main owl body
        owl_body = plt.Circle((x, y), size/2, color=color, alpha=alpha)
        ax.add_patch(owl_body)
        
        # Glowing eyes - symbol of Athena's wisdom
        eye_offset = size * 0.2
        left_eye = plt.Circle((x - eye_offset, y + eye_offset), size/8, 
                            color='white', alpha=alpha * 1.2)
        right_eye = plt.Circle((x + eye_offset, y + eye_offset), size/8, 
                             color='white', alpha=alpha * 1.2)
        ax.add_patch(left_eye)
        ax.add_patch(right_eye)
        
        # Wisdom aura - pulsing based on audit activity
        audit_activity = mythic_properties.get('audit_activity', 0.0)
        if audit_activity > 0.2:
            pulse_intensity = 0.3 + 0.2 * np.sin(self.animation_frame * 2)
            wisdom_aura = plt.Circle((x, y), size * 1.5, color=color, 
                                   alpha=pulse_intensity * alpha, fill=False, linewidth=2)
            ax.add_patch(wisdom_aura)
    
    def _render_whale_motif(self, ax, x: float, y: float, size: float,
                           color: Tuple[float, float, float], alpha: float,
                           mythic_properties: Dict[str, Any]):
        """Render Whale with abyssal sonar arcs"""
        # Main whale body as ellipse
        whale_body = patches.Ellipse((x, y), size, size * 0.6, 
                                   facecolor=color, alpha=alpha)
        ax.add_patch(whale_body)
        
        # Sonar arcs - Leviathan depth sensing
        depth_scanning = mythic_properties.get('depth_scanning_active', False)
        if depth_scanning:
            for i in range(3):
                arc_radius = size * (1.5 + i * 0.5)
                arc_alpha = alpha * (0.8 - i * 0.2)
                sonar_arc = plt.Circle((x, y), arc_radius, color=color, 
                                     alpha=arc_alpha, fill=False, linewidth=2)
                ax.add_patch(sonar_arc)
    
    def _render_spider_motif(self, ax, x: float, y: float, size: float,
                            color: Tuple[float, float, float], alpha: float,
                            mythic_properties: Dict[str, Any]):
        """Render Spider with web threads and tension indicators"""
        # Main spider body as star (web center)
        spider_star = patches.RegularPolygon((x, y), 8, size/2, 
                                           facecolor=color, alpha=alpha)
        ax.add_patch(spider_star)
        
        # Web threads extending outward
        web_connections = mythic_properties.get('web_connections', [])
        for connection in web_connections[:8]:  # Limit to 8 connections
            conn_x = connection.get('x', x + np.random.uniform(-100, 100))
            conn_y = connection.get('y', y + np.random.uniform(-100, 100))
            tension = connection.get('tension', 0.5)
            
            # Thread color based on tension
            thread_color = 'red' if tension > 0.7 else 'orange' if tension > 0.4 else 'white'
            ax.plot([x, conn_x], [y, conn_y], color=thread_color, 
                   alpha=alpha * 0.6, linewidth=1)
    
    def render_pigment_landscape(self, pigment_data: Dict[str, Any], 
                               output_path: str) -> bool:
        """Render system-wide pigment distribution as mood landscape"""
        try:
            fig, (ax1, ax2) = plt.subplots(1, 2, figsize=(16, 8), facecolor='black')
            
            # Left panel: RGB distribution heatmap
            self._render_pigment_heatmap(ax1, pigment_data)
            
            # Right panel: Pigment flow dynamics
            self._render_pigment_flow(ax2, pigment_data)
            
            fig.suptitle("🎨 DAWN Pigment Landscape - Belief Vector Distribution", 
                        color='white', fontsize=16, fontweight='bold')
            
            plt.tight_layout()
            plt.savefig(output_path, dpi=150, bbox_inches='tight', 
                       facecolor='black', edgecolor='none')
            plt.close()
            
            logger.info(f"🎨 Pigment landscape rendered to {output_path}")
            return True
            
        except Exception as e:
            logger.error(f"🎨 Pigment landscape rendering failed: {e}")
            return False
    
    def _render_pigment_heatmap(self, ax, pigment_data: Dict[str, Any]):
        """Render pigment distribution as heatmap"""
        ax.set_facecolor('black')
        
        # Extract pigment vectors
        pigment_vectors = pigment_data.get('system_pigment_vectors', [])
        if not pigment_vectors:
            ax.text(0.5, 0.5, 'No Pigment Data', ha='center', va='center', 
                   color='white', transform=ax.transAxes)
            return
        
        # Create RGB intensity map
        rgb_data = np.array(pigment_vectors)
        if rgb_data.shape[0] == 0:
            return
        
        # Reshape for heatmap visualization
        grid_size = int(np.sqrt(len(pigment_vectors))) + 1
        rgb_grid = np.zeros((grid_size, grid_size, 3))
        
        for i, vector in enumerate(pigment_vectors):
            row = i // grid_size
            col = i % grid_size
            if row < grid_size and col < grid_size:
                rgb_grid[row, col] = vector[:3]  # Take RGB channels
        
        # Display as RGB image
        ax.imshow(rgb_grid, aspect='auto')
        ax.set_title('Pigment Distribution Grid', color='white')
        ax.set_xticks([])
        ax.set_yticks([])
    
    def _render_pigment_flow(self, ax, pigment_data: Dict[str, Any]):
        """Render pigment flow dynamics"""
        ax.set_facecolor('black')
        
        # Pigment balance over time
        balance_history = pigment_data.get('balance_history', [])
        if balance_history:
            times = list(range(len(balance_history)))
            r_values = [b.get('red', 0) for b in balance_history]
            g_values = [b.get('green', 0) for b in balance_history]
            b_values = [b.get('blue', 0) for b in balance_history]
            
            ax.plot(times, r_values, color='red', alpha=0.8, label='Urgency (Red)')
            ax.plot(times, g_values, color='green', alpha=0.8, label='Balance (Green)')
            ax.plot(times, b_values, color='blue', alpha=0.8, label='Abstraction (Blue)')
            
            ax.set_title('Pigment Flow Dynamics', color='white')
            ax.legend()
            ax.tick_params(colors='white')
        else:
            ax.text(0.5, 0.5, 'No Flow Data', ha='center', va='center', 
                   color='white', transform=ax.transAxes)
    
    def _add_mythic_atmosphere(self, ax):
        """Add atmospheric effects to enhance mythic feeling"""
        # Add subtle background texture
        for _ in range(50):
            x = np.random.uniform(0, self.canvas_size[0])
            y = np.random.uniform(0, self.canvas_size[1])
            size = np.random.uniform(1, 3)
            alpha = np.random.uniform(0.1, 0.3)
            
            star = plt.Circle((x, y), size, color='white', alpha=alpha)
            ax.add_patch(star)
    
    def _add_archetypal_legend(self, ax, tracer_data: List[Dict[str, Any]]):
        """Add legend explaining archetypal symbols"""
        active_types = set(tracer['type'] for tracer in tracer_data)
        
        legend_elements = []
        for tracer_type in active_types:
            color = self.archetypal_colors.get(tracer_type, (0.5, 0.5, 0.5))
            legend_elements.append(plt.Line2D([0], [0], marker='o', color='w', 
                                            markerfacecolor=color, markersize=10,
                                            label=tracer_type.replace('_', ' ').title()))
        
        if legend_elements:
            ax.legend(handles=legend_elements, loc='upper right', 
                     facecolor='black', edgecolor='white', labelcolor='white')
    
    def create_unified_mythology_view(self, system_data: Dict[str, Any], 
                                    output_path: str) -> bool:
        """Create unified view combining all mythic layers"""
        try:
            fig = plt.figure(figsize=(20, 12), facecolor='black')
            
            # Create subplot grid for different mythic aspects
            gs = fig.add_gridspec(3, 3, hspace=0.3, wspace=0.3)
            
            # Archetypal tracers (top left)
            ax1 = fig.add_subplot(gs[0, 0])
            self._render_mini_archetypal_view(ax1, system_data.get('tracers', []))
            
            # Pigment landscape (top center)
            ax2 = fig.add_subplot(gs[0, 1])
            self._render_mini_pigment_view(ax2, system_data.get('pigments', {}))
            
            # Persephone cycles (top right)
            ax3 = fig.add_subplot(gs[0, 2])
            self._render_mini_persephone_view(ax3, system_data.get('persephone', {}))
            
            # Fractal garden (middle left)
            ax4 = fig.add_subplot(gs[1, 0])
            self._render_mini_garden_view(ax4, system_data.get('memory_garden', {}))
            
            # Volcanic residue (middle center)
            ax5 = fig.add_subplot(gs[1, 1])
            self._render_mini_volcanic_view(ax5, system_data.get('residue', {}))
            
            # Sigil network (middle right)
            ax6 = fig.add_subplot(gs[1, 2])
            self._render_mini_sigil_view(ax6, system_data.get('sigils', {}))
            
            # Mythic coherence status (bottom, spanning all)
            ax7 = fig.add_subplot(gs[2, :])
            self._render_coherence_status(ax7, system_data.get('mythic_coherence', {}))
            
            fig.suptitle("🔮 DAWN Unified Mythic Landscape - Living Symbolic Cognition", 
                        color='white', fontsize=18, fontweight='bold')
            
            plt.savefig(output_path, dpi=150, bbox_inches='tight', 
                       facecolor='black', edgecolor='none')
            plt.close()
            
            logger.info(f"🔮 Unified mythology view rendered to {output_path}")
            return True
            
        except Exception as e:
            logger.error(f"🔮 Unified mythology rendering failed: {e}")
            return False
    
    def _render_mini_archetypal_view(self, ax, tracer_data: List[Dict[str, Any]]):
        """Render mini archetypal tracer view"""
        ax.set_facecolor('black')
        ax.set_title('Archetypal Tracers', color='white', fontsize=10)
        
        for i, tracer in enumerate(tracer_data[:8]):  # Limit to 8 tracers
            angle = i * 2 * np.pi / len(tracer_data)
            x = 0.5 + 0.3 * np.cos(angle)
            y = 0.5 + 0.3 * np.sin(angle)
            
            tracer_type = tracer.get('type', 'unknown')
            color = self.archetypal_colors.get(tracer_type, (0.5, 0.5, 0.5))
            
            circle = plt.Circle((x, y), 0.05, color=color, alpha=0.8)
            ax.add_patch(circle)
        
        ax.set_xlim(0, 1)
        ax.set_ylim(0, 1)
        ax.set_xticks([])
        ax.set_yticks([])
    
    def _render_mini_pigment_view(self, ax, pigment_data: Dict[str, Any]):
        """Render mini pigment landscape view"""
        ax.set_facecolor('black')
        ax.set_title('Pigment Landscape', color='white', fontsize=10)
        
        # Simple RGB bar chart
        rgb_balance = pigment_data.get('current_balance', [0.33, 0.33, 0.33])
        colors = ['red', 'green', 'blue']
        labels = ['Urgency', 'Balance', 'Abstraction']
        
        bars = ax.bar(labels, rgb_balance, color=colors, alpha=0.7)
        ax.set_ylim(0, 1)
        ax.tick_params(colors='white')
    
    def _render_mini_persephone_view(self, ax, persephone_data: Dict[str, Any]):
        """Render mini Persephone cycle view"""
        ax.set_facecolor('black')
        ax.set_title('Persephone Cycles', color='white', fontsize=10)
        
        # Cycle visualization as spiral
        state = persephone_data.get('current_state', 'surface')
        cycle_progress = persephone_data.get('cycle_progress', 0.0)
        
        # Draw spiral representing descent/return cycle
        theta = np.linspace(0, 4*np.pi, 100)
        r = 0.1 + 0.3 * theta / (4*np.pi)
        x = 0.5 + r * np.cos(theta)
        y = 0.5 + r * np.sin(theta)
        
        ax.plot(x, y, color='purple', alpha=0.6)
        
        # Mark current position
        current_idx = int(cycle_progress * len(theta))
        if current_idx < len(x):
            ax.scatter(x[current_idx], y[current_idx], color='white', s=50)
        
        ax.set_xlim(0, 1)
        ax.set_ylim(0, 1)
        ax.set_xticks([])
        ax.set_yticks([])
    
    def _render_mini_garden_view(self, ax, garden_data: Dict[str, Any]):
        """Render mini fractal garden view"""
        ax.set_facecolor('black')
        ax.set_title('Fractal Garden', color='white', fontsize=10)
        
        # Simple bloom scatter
        bloom_count = garden_data.get('total_blooms', 0)
        if bloom_count > 0:
            x = np.random.random(min(bloom_count, 20))
            y = np.random.random(min(bloom_count, 20))
            colors = np.random.random((min(bloom_count, 20), 3))
            
            ax.scatter(x, y, c=colors, alpha=0.7, s=30)
        
        ax.set_xlim(0, 1)
        ax.set_ylim(0, 1)
        ax.set_xticks([])
        ax.set_yticks([])
    
    def _render_mini_volcanic_view(self, ax, residue_data: Dict[str, Any]):
        """Render mini volcanic residue view"""
        ax.set_facecolor('black')
        ax.set_title('Volcanic Residue', color='white', fontsize=10)
        
        # Ash vs Soot ratio as stacked bar
        ash_ratio = residue_data.get('ash_ratio', 0.5)
        soot_ratio = 1.0 - ash_ratio
        
        ax.bar(['Residue'], [ash_ratio], color='orange', alpha=0.8, label='Ash')
        ax.bar(['Residue'], [soot_ratio], bottom=[ash_ratio], color='gray', alpha=0.8, label='Soot')
        
        ax.set_ylim(0, 1)
        ax.legend(fontsize=8)
        ax.tick_params(colors='white')
    
    def _render_mini_sigil_view(self, ax, sigil_data: Dict[str, Any]):
        """Render mini sigil network view"""
        ax.set_facecolor('black')
        ax.set_title('Sigil Network', color='white', fontsize=10)
        
        # Simple network visualization
        house_count = sigil_data.get('active_houses', 6)
        if house_count > 0:
            angles = np.linspace(0, 2*np.pi, house_count, endpoint=False)
            x = 0.5 + 0.3 * np.cos(angles)
            y = 0.5 + 0.3 * np.sin(angles)
            
            # Draw houses as hexagons
            for i in range(house_count):
                hex_patch = patches.RegularPolygon((x[i], y[i]), 6, 0.05, 
                                                 facecolor='gold', alpha=0.7)
                ax.add_patch(hex_patch)
            
            # Draw connections
            for i in range(house_count):
                for j in range(i+1, house_count):
                    ax.plot([x[i], x[j]], [y[i], y[j]], color='white', alpha=0.3)
        
        ax.set_xlim(0, 1)
        ax.set_ylim(0, 1)
        ax.set_xticks([])
        ax.set_yticks([])
    
    def _render_coherence_status(self, ax, coherence_data: Dict[str, Any]):
        """Render mythic coherence status"""
        ax.set_facecolor('black')
        ax.set_title('Mythic Coherence Status', color='white', fontsize=12)
        
        # Coherence metrics
        overall_score = coherence_data.get('overall_score', 0.5)
        subsystem_scores = {
            'Pigment Balance': coherence_data.get('pigment_balance_score', 0.5),
            'Archetypal Alignment': coherence_data.get('archetypal_alignment_score', 0.5),
            'Persephone Cycles': coherence_data.get('persephone_cycle_score', 0.5),
            'Sigil Coherence': coherence_data.get('sigil_house_coherence_score', 0.5),
            'Temporal Continuity': coherence_data.get('temporal_continuity_score', 0.5)
        }
        
        # Horizontal bar chart
        y_pos = np.arange(len(subsystem_scores))
        scores = list(subsystem_scores.values())
        labels = list(subsystem_scores.keys())
        
        # Color bars based on score
        colors = ['red' if s < 0.5 else 'orange' if s < 0.7 else 'green' for s in scores]
        
        bars = ax.barh(y_pos, scores, color=colors, alpha=0.7)
        ax.set_yticks(y_pos)
        ax.set_yticklabels(labels)
        ax.set_xlim(0, 1)
        ax.tick_params(colors='white')
        
        # Add overall score text
        ax.text(0.02, 0.98, f'Overall Coherence: {overall_score:.2f}', 
               transform=ax.transAxes, color='white', fontsize=12, fontweight='bold',
               verticalalignment='top')
    
    def update_animation_frame(self):
        """Update animation frame for dynamic effects"""
        self.animation_frame += 0.1
        if self.animation_frame > 2 * np.pi:
            self.animation_frame = 0.0


# Global overlay renderer instance
_global_mythic_overlay = None

def get_mythic_overlay_renderer() -> MythicOverlayRenderer:
    """Get global mythic overlay renderer instance"""
    global _global_mythic_overlay
    if _global_mythic_overlay is None:
        _global_mythic_overlay = MythicOverlayRenderer()
    return _global_mythic_overlay
