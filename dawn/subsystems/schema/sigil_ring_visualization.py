#!/usr/bin/env python3
"""
DAWN Sigil Ring Visualization System
====================================

Implementation of visual representation for the DAWN sigil ring with
orbiting glyphs, house nodes, and real-time symbolic activity display.
Provides both programmatic and GUI-ready visualization data.

Based on DAWN's documented visual ring architecture.
"""

import time
import math
import logging
import json
from typing import Dict, List, Optional, Set, Tuple, Any, Union
from dataclasses import dataclass, field
from enum import Enum
import colorsys

from core.schema_anomaly_logger import log_anomaly, AnomalySeverity
from schema.registry import registry
from schema.sigil_glyph_codex import SigilHouse, sigil_glyph_codex, SigilGlyph
from schema.enhanced_sigil_ring import enhanced_sigil_ring
from schema.archetypal_house_operations import HOUSE_OPERATORS
from schema.tracer_house_alignment import tracer_house_alignment
from schema.symbolic_failure_detection import symbolic_failure_detector
from rhizome.propagation import emit_signal, SignalType
from utils.metrics_collector import metrics

logger = logging.getLogger(__name__)

class VisualizationTheme(Enum):
    """Visual themes for the ring display"""
    MYSTICAL = "mystical"       # Dark background, glowing elements
    TECHNICAL = "technical"     # Clean, data-focused
    ORGANIC = "organic"         # Natural, flowing elements
    NEON = "neon"              # Cyberpunk aesthetic
    MINIMAL = "minimal"        # Simple, high contrast

class AnimationType(Enum):
    """Types of animations for ring elements"""
    ORBIT = "orbit"             # Circular orbital motion
    PULSE = "pulse"             # Pulsing/breathing effect
    SPIRAL = "spiral"           # Spiral motion
    WAVE = "wave"              # Wave-like motion
    PARTICLE = "particle"      # Particle system effects

@dataclass
class VisualGlyph:
    """Visual representation of a sigil glyph"""
    symbol: str
    name: str
    position: Tuple[float, float]  # (x, y) coordinates
    size: float = 1.0
    color: str = "#FFFFFF"
    opacity: float = 1.0
    rotation: float = 0.0
    animation_type: Optional[AnimationType] = None
    animation_speed: float = 1.0
    trail_enabled: bool = False
    trail_length: int = 10
    metadata: Dict[str, Any] = field(default_factory=dict)

@dataclass
class VisualHouseNode:
    """Visual representation of a house node"""
    house: SigilHouse
    position: Tuple[float, float]
    emoji: str
    archetype_name: str
    activity_level: float = 0.0
    resonance: float = 0.0
    color: str = "#FFFFFF"
    size: float = 1.0
    pulse_intensity: float = 0.0
    connected_glyphs: List[str] = field(default_factory=list)
    connection_lines: List[Tuple[float, float]] = field(default_factory=list)

@dataclass
class VisualRing:
    """Complete visual representation of the sigil ring"""
    center: Tuple[float, float] = (400, 400)  # Ring center coordinates
    radius: float = 300.0
    house_nodes: List[VisualHouseNode] = field(default_factory=list)
    orbiting_glyphs: List[VisualGlyph] = field(default_factory=list)
    containment_boundary: Dict[str, Any] = field(default_factory=dict)
    background_effects: List[Dict[str, Any]] = field(default_factory=list)
    timestamp: float = field(default_factory=time.time)
    theme: VisualizationTheme = VisualizationTheme.MYSTICAL
    
    def get_total_elements(self) -> int:
        """Get total number of visual elements"""
        return len(self.house_nodes) + len(self.orbiting_glyphs)

class SigilRingVisualizationSystem:
    """
    Sigil Ring Visualization System
    
    Creates comprehensive visual representations of the DAWN sigil ring,
    including house nodes, orbiting glyphs, containment boundaries,
    and real-time activity visualization. Supports multiple themes
    and animation types.
    """
    
    def __init__(self):
        self.current_theme = VisualizationTheme.MYSTICAL
        self.animation_enabled = True
        self.animation_speed = 1.0
        
        # Visual parameters
        self.ring_center = (400, 400)
        self.ring_radius = 300.0
        self.house_node_size = 60.0
        self.glyph_size = 24.0
        
        # Color schemes for different themes
        self.color_schemes = self._initialize_color_schemes()
        
        # House positioning (evenly distributed around circle)
        self.house_positions = self._calculate_house_positions()
        
        # Animation state
        self.animation_time = 0.0
        self.last_update = time.time()
        
        # Visual cache
        self.cached_visual_ring: Optional[VisualRing] = None
        self.cache_timestamp = 0.0
        self.cache_duration = 0.1  # 100ms cache
        
        # Register with schema registry
        self._register()
        
        logger.info("🎨 Sigil Ring Visualization System initialized")
        logger.info(f"   Default theme: {self.current_theme.value}")
        logger.info(f"   Ring center: {self.ring_center}, radius: {self.ring_radius}")
    
    def _register(self):
        """Register with schema registry"""
        registry.register(
            component_id="schema.sigil_ring_visualization",
            name="Sigil Ring Visualization System",
            component_type="VISUALIZATION_SYSTEM",
            instance=self,
            capabilities=[
                "ring_visual_representation",
                "house_node_visualization",
                "glyph_orbit_animation",
                "containment_boundary_display",
                "real_time_activity_visualization",
                "multi_theme_support"
            ],
            version="1.0.0"
        )
    
    def _initialize_color_schemes(self) -> Dict[VisualizationTheme, Dict[str, Any]]:
        """Initialize color schemes for different themes"""
        return {
            VisualizationTheme.MYSTICAL: {
                "background": "#0A0A0F",
                "ring_border": "#4A4A8A",
                "house_colors": {
                    SigilHouse.MEMORY: "#FF69B4",      # Pink for blooms
                    SigilHouse.PURIFICATION: "#FF4500", # Orange-red for fire
                    SigilHouse.WEAVING: "#9370DB",     # Purple for web
                    SigilHouse.FLAME: "#FFD700",       # Gold for flame
                    SigilHouse.MIRRORS: "#87CEEB",     # Sky blue for reflection
                    SigilHouse.ECHOES: "#98FB98"       # Light green for sound
                },
                "glyph_colors": {
                    "primary": "#FFFFFF",
                    "composite": "#FFD700",
                    "core": "#FF69B4"
                },
                "effects": {
                    "glow": True,
                    "particles": True,
                    "trails": True
                }
            },
            VisualizationTheme.TECHNICAL: {
                "background": "#1E1E1E",
                "ring_border": "#00FF00",
                "house_colors": {
                    house: f"#{hex(hash(house.value) % 16777215)[2:].zfill(6)}" 
                    for house in SigilHouse
                },
                "glyph_colors": {
                    "primary": "#00FF00",
                    "composite": "#FFFF00",
                    "core": "#FF0000"
                },
                "effects": {
                    "glow": False,
                    "particles": False,
                    "trails": False
                }
            },
            VisualizationTheme.ORGANIC: {
                "background": "#2F4F2F",
                "ring_border": "#8FBC8F",
                "house_colors": {
                    SigilHouse.MEMORY: "#FFB6C1",
                    SigilHouse.PURIFICATION: "#F4A460",
                    SigilHouse.WEAVING: "#DDA0DD",
                    SigilHouse.FLAME: "#FFA500",
                    SigilHouse.MIRRORS: "#ADD8E6",
                    SigilHouse.ECHOES: "#90EE90"
                },
                "glyph_colors": {
                    "primary": "#F5F5DC",
                    "composite": "#D2B48C",
                    "core": "#CD853F"
                },
                "effects": {
                    "glow": True,
                    "particles": True,
                    "trails": True
                }
            },
            VisualizationTheme.NEON: {
                "background": "#000000",
                "ring_border": "#00FFFF",
                "house_colors": {
                    SigilHouse.MEMORY: "#FF00FF",
                    SigilHouse.PURIFICATION: "#FF0080",
                    SigilHouse.WEAVING: "#8000FF",
                    SigilHouse.FLAME: "#FFFF00",
                    SigilHouse.MIRRORS: "#00FFFF",
                    SigilHouse.ECHOES: "#00FF80"
                },
                "glyph_colors": {
                    "primary": "#FFFFFF",
                    "composite": "#00FFFF",
                    "core": "#FF00FF"
                },
                "effects": {
                    "glow": True,
                    "particles": True,
                    "trails": True
                }
            },
            VisualizationTheme.MINIMAL: {
                "background": "#FFFFFF",
                "ring_border": "#000000",
                "house_colors": {
                    house: "#808080" for house in SigilHouse
                },
                "glyph_colors": {
                    "primary": "#000000",
                    "composite": "#404040",
                    "core": "#800000"
                },
                "effects": {
                    "glow": False,
                    "particles": False,
                    "trails": False
                }
            }
        }
    
    def _calculate_house_positions(self) -> Dict[SigilHouse, Tuple[float, float]]:
        """Calculate positions for house nodes around the ring"""
        houses = list(SigilHouse)
        positions = {}
        
        for i, house in enumerate(houses):
            angle = (i / len(houses)) * 2 * math.pi - math.pi / 2  # Start at top
            x = self.ring_center[0] + self.ring_radius * math.cos(angle)
            y = self.ring_center[1] + self.ring_radius * math.sin(angle)
            positions[house] = (x, y)
        
        return positions
    
    def set_theme(self, theme: VisualizationTheme):
        """Set visualization theme"""
        self.current_theme = theme
        self._invalidate_cache()
        logger.info(f"🎨 Visualization theme set to: {theme.value}")
    
    def set_animation_speed(self, speed: float):
        """Set animation speed multiplier"""
        self.animation_speed = max(0.0, min(5.0, speed))  # Clamp to reasonable range
        logger.info(f"🎨 Animation speed set to: {self.animation_speed}x")
    
    def toggle_animation(self):
        """Toggle animation on/off"""
        self.animation_enabled = not self.animation_enabled
        logger.info(f"🎨 Animation {'enabled' if self.animation_enabled else 'disabled'}")
    
    def _invalidate_cache(self):
        """Invalidate visual cache"""
        self.cached_visual_ring = None
        self.cache_timestamp = 0.0
    
    def _update_animation_time(self):
        """Update animation time based on real time"""
        current_time = time.time()
        if self.animation_enabled:
            delta_time = current_time - self.last_update
            self.animation_time += delta_time * self.animation_speed
        self.last_update = current_time
    
    def generate_visual_ring(self, force_refresh: bool = False) -> VisualRing:
        """Generate complete visual representation of the sigil ring"""
        current_time = time.time()
        
        # Check cache
        if (not force_refresh and 
            self.cached_visual_ring and 
            current_time - self.cache_timestamp < self.cache_duration):
            return self.cached_visual_ring
        
        # Update animation time
        self._update_animation_time()
        
        # Create new visual ring
        visual_ring = VisualRing(
            center=self.ring_center,
            radius=self.ring_radius,
            theme=self.current_theme,
            timestamp=current_time
        )
        
        # Generate house nodes
        visual_ring.house_nodes = self._generate_house_nodes()
        
        # Generate orbiting glyphs
        visual_ring.orbiting_glyphs = self._generate_orbiting_glyphs()
        
        # Generate containment boundary
        visual_ring.containment_boundary = self._generate_containment_boundary()
        
        # Generate background effects
        visual_ring.background_effects = self._generate_background_effects()
        
        # Cache result
        self.cached_visual_ring = visual_ring
        self.cache_timestamp = current_time
        
        return visual_ring
    
    def _generate_house_nodes(self) -> List[VisualHouseNode]:
        """Generate visual house nodes"""
        house_nodes = []
        color_scheme = self.color_schemes[self.current_theme]
        
        for house in SigilHouse:
            # Get house operator for activity data
            operator = HOUSE_OPERATORS.get(house)
            
            # Calculate activity level and resonance
            activity_level = 0.0
            resonance = 0.0
            
            if operator:
                resonance = operator.get_average_resonance()
                # Simulate activity based on recent operations
                activity_level = min(1.0, operator.operations_performed / 10.0)
            
            # Get house emoji
            house_emojis = {
                SigilHouse.MEMORY: "🌸",
                SigilHouse.PURIFICATION: "🔥",
                SigilHouse.WEAVING: "🕸️",
                SigilHouse.FLAME: "⚡",
                SigilHouse.MIRRORS: "🪞",
                SigilHouse.ECHOES: "🔊"
            }
            
            # Calculate pulse intensity based on activity
            pulse_intensity = activity_level * (0.5 + 0.5 * math.sin(self.animation_time * 2))
            
            # Create house node
            house_node = VisualHouseNode(
                house=house,
                position=self.house_positions[house],
                emoji=house_emojis.get(house, "❓"),
                archetype_name=operator._get_archetype() if operator else "Unknown",
                activity_level=activity_level,
                resonance=resonance,
                color=color_scheme["house_colors"].get(house, "#FFFFFF"),
                size=self.house_node_size * (0.8 + 0.4 * activity_level),  # Size varies with activity
                pulse_intensity=pulse_intensity
            )
            
            house_nodes.append(house_node)
        
        return house_nodes
    
    def _generate_orbiting_glyphs(self) -> List[VisualGlyph]:
        """Generate orbiting glyphs around the ring"""
        orbiting_glyphs = []
        color_scheme = self.color_schemes[self.current_theme]
        
        # Get ring status for active glyphs
        ring_status = enhanced_sigil_ring.get_ring_status()
        
        # Get visual representation from ring
        ring_visual = enhanced_sigil_ring.get_visual_representation()
        
        if ring_visual.get("active", False):
            orbiting_glyph_data = ring_visual.get("orbiting_glyphs", [])
            
            for i, glyph_data in enumerate(orbiting_glyph_data):
                # Get glyph information
                glyph = sigil_glyph_codex.get_glyph(glyph_data["symbol"])
                
                # Calculate orbital position
                orbit_angle = (self.animation_time * 0.5 + i * (2 * math.pi / len(orbiting_glyph_data))) % (2 * math.pi)
                orbit_radius = self.ring_radius * 0.7  # Orbit inside the house nodes
                
                x = self.ring_center[0] + orbit_radius * math.cos(orbit_angle)
                y = self.ring_center[1] + orbit_radius * math.sin(orbit_angle)
                
                # Determine glyph color based on category
                glyph_color = color_scheme["glyph_colors"]["primary"]
                if glyph:
                    if glyph.category.value == "composite":
                        glyph_color = color_scheme["glyph_colors"]["composite"]
                    elif glyph.category.value == "core_minimal":
                        glyph_color = color_scheme["glyph_colors"]["core"]
                
                # Calculate size based on priority
                priority = glyph_data.get("priority", 5)
                size_multiplier = 0.5 + (priority / 10.0)
                
                # Create visual glyph
                visual_glyph = VisualGlyph(
                    symbol=glyph_data["symbol"],
                    name=glyph_data.get("name", "Unknown"),
                    position=(x, y),
                    size=self.glyph_size * size_multiplier,
                    color=glyph_color,
                    opacity=0.8 + 0.2 * math.sin(self.animation_time + i),
                    rotation=orbit_angle * 180 / math.pi,
                    animation_type=AnimationType.ORBIT,
                    animation_speed=0.5,
                    trail_enabled=color_scheme["effects"]["trails"],
                    trail_length=8,
                    metadata={
                        "priority": priority,
                        "house": glyph_data.get("house"),
                        "orbit_radius": orbit_radius,
                        "orbit_angle": orbit_angle
                    }
                )
                
                orbiting_glyphs.append(visual_glyph)
        
        return orbiting_glyphs
    
    def _generate_containment_boundary(self) -> Dict[str, Any]:
        """Generate containment boundary visualization"""
        ring_status = enhanced_sigil_ring.get_ring_status()
        color_scheme = self.color_schemes[self.current_theme]
        
        # Get containment level
        containment_info = ring_status.get("containment_boundary", {})
        containment_level = containment_info.get("level", "basic")
        
        # Calculate boundary properties
        boundary_thickness = {
            "open": 2,
            "basic": 4,
            "secured": 6,
            "sealed": 8,
            "quarantine": 12
        }.get(containment_level, 4)
        
        boundary_opacity = {
            "open": 0.2,
            "basic": 0.4,
            "secured": 0.6,
            "sealed": 0.8,
            "quarantine": 1.0
        }.get(containment_level, 0.4)
        
        # Pulse effect for higher containment levels
        pulse_factor = 1.0
        if containment_level in ["sealed", "quarantine"]:
            pulse_factor = 0.8 + 0.4 * math.sin(self.animation_time * 3)
        
        return {
            "center": self.ring_center,
            "radius": self.ring_radius + 20,  # Slightly outside house nodes
            "thickness": boundary_thickness * pulse_factor,
            "color": color_scheme["ring_border"],
            "opacity": boundary_opacity * pulse_factor,
            "containment_level": containment_level,
            "breach_count": containment_info.get("active_breaches", 0),
            "pulse_active": containment_level in ["sealed", "quarantine"]
        }
    
    def _generate_background_effects(self) -> List[Dict[str, Any]]:
        """Generate background effects"""
        effects = []
        color_scheme = self.color_schemes[self.current_theme]
        
        if not color_scheme["effects"]["particles"]:
            return effects
        
        # Generate floating particles
        particle_count = 20
        for i in range(particle_count):
            # Random position around ring
            angle = (i / particle_count) * 2 * math.pi + self.animation_time * 0.1
            distance = self.ring_radius * (0.3 + 0.7 * ((i * 7) % 100) / 100)
            
            x = self.ring_center[0] + distance * math.cos(angle)
            y = self.ring_center[1] + distance * math.sin(angle)
            
            particle = {
                "type": "particle",
                "position": (x, y),
                "size": 2 + 3 * math.sin(self.animation_time + i),
                "color": color_scheme["glyph_colors"]["primary"],
                "opacity": 0.1 + 0.2 * math.sin(self.animation_time * 0.5 + i),
                "animation_phase": i
            }
            
            effects.append(particle)
        
        return effects
    
    def get_ring_statistics(self) -> Dict[str, Any]:
        """Get visual ring statistics"""
        visual_ring = self.generate_visual_ring()
        
        # Calculate activity statistics
        total_activity = sum(node.activity_level for node in visual_ring.house_nodes)
        avg_activity = total_activity / len(visual_ring.house_nodes) if visual_ring.house_nodes else 0
        
        total_resonance = sum(node.resonance for node in visual_ring.house_nodes)
        avg_resonance = total_resonance / len(visual_ring.house_nodes) if visual_ring.house_nodes else 0
        
        # Most active house
        most_active_house = None
        if visual_ring.house_nodes:
            most_active_node = max(visual_ring.house_nodes, key=lambda n: n.activity_level)
            most_active_house = most_active_node.house.value
        
        return {
            "total_elements": visual_ring.get_total_elements(),
            "house_nodes": len(visual_ring.house_nodes),
            "orbiting_glyphs": len(visual_ring.orbiting_glyphs),
            "background_effects": len(visual_ring.background_effects),
            "average_activity": avg_activity,
            "average_resonance": avg_resonance,
            "most_active_house": most_active_house,
            "theme": visual_ring.theme.value,
            "animation_enabled": self.animation_enabled,
            "animation_speed": self.animation_speed,
            "containment_level": visual_ring.containment_boundary.get("containment_level", "unknown")
        }
    
    def export_visual_data(self, format: str = "json") -> Union[str, Dict[str, Any]]:
        """Export visual ring data in specified format"""
        visual_ring = self.generate_visual_ring()
        
        # Convert to serializable format
        data = {
            "ring": {
                "center": visual_ring.center,
                "radius": visual_ring.radius,
                "theme": visual_ring.theme.value,
                "timestamp": visual_ring.timestamp
            },
            "house_nodes": [
                {
                    "house": node.house.value,
                    "position": node.position,
                    "emoji": node.emoji,
                    "archetype": node.archetype_name,
                    "activity_level": node.activity_level,
                    "resonance": node.resonance,
                    "color": node.color,
                    "size": node.size,
                    "pulse_intensity": node.pulse_intensity
                }
                for node in visual_ring.house_nodes
            ],
            "orbiting_glyphs": [
                {
                    "symbol": glyph.symbol,
                    "name": glyph.name,
                    "position": glyph.position,
                    "size": glyph.size,
                    "color": glyph.color,
                    "opacity": glyph.opacity,
                    "rotation": glyph.rotation,
                    "animation_type": glyph.animation_type.value if glyph.animation_type else None,
                    "trail_enabled": glyph.trail_enabled,
                    "metadata": glyph.metadata
                }
                for glyph in visual_ring.orbiting_glyphs
            ],
            "containment_boundary": visual_ring.containment_boundary,
            "background_effects": visual_ring.background_effects,
            "statistics": self.get_ring_statistics()
        }
        
        if format.lower() == "json":
            return json.dumps(data, indent=2)
        else:
            return data
    
    def create_gui_frame_data(self) -> Dict[str, Any]:
        """Create frame data optimized for GUI rendering"""
        visual_ring = self.generate_visual_ring()
        
        # Simplified data structure for GUI
        return {
            "frame_id": int(self.animation_time * 60),  # 60 FPS frame counter
            "timestamp": visual_ring.timestamp,
            "theme": self.current_theme.value,
            "ring_center": self.ring_center,
            "ring_radius": self.ring_radius,
            
            # House nodes with simplified data
            "houses": [
                {
                    "id": node.house.value,
                    "x": node.position[0],
                    "y": node.position[1],
                    "emoji": node.emoji,
                    "color": node.color,
                    "size": node.size,
                    "pulse": node.pulse_intensity,
                    "activity": node.activity_level,
                    "resonance": node.resonance
                }
                for node in visual_ring.house_nodes
            ],
            
            # Glyphs with animation data
            "glyphs": [
                {
                    "symbol": glyph.symbol,
                    "x": glyph.position[0],
                    "y": glyph.position[1],
                    "size": glyph.size,
                    "color": glyph.color,
                    "opacity": glyph.opacity,
                    "rotation": glyph.rotation,
                    "trail": glyph.trail_enabled
                }
                for glyph in visual_ring.orbiting_glyphs
            ],
            
            # Containment boundary
            "containment": {
                "radius": visual_ring.containment_boundary.get("radius", self.ring_radius),
                "thickness": visual_ring.containment_boundary.get("thickness", 4),
                "color": visual_ring.containment_boundary.get("color", "#FFFFFF"),
                "opacity": visual_ring.containment_boundary.get("opacity", 0.4),
                "pulse": visual_ring.containment_boundary.get("pulse_active", False)
            },
            
            # Background effects
            "effects": visual_ring.background_effects,
            
            # Animation state
            "animation_time": self.animation_time,
            "animation_enabled": self.animation_enabled,
            "animation_speed": self.animation_speed
        }

# Global visualization system instance
sigil_ring_visualization = SigilRingVisualizationSystem()

# Export key functions for easy access
def generate_visual_ring() -> VisualRing:
    """Generate visual ring representation"""
    return sigil_ring_visualization.generate_visual_ring()

def set_visualization_theme(theme: VisualizationTheme):
    """Set visualization theme"""
    sigil_ring_visualization.set_theme(theme)

def get_gui_frame_data() -> Dict[str, Any]:
    """Get GUI frame data"""
    return sigil_ring_visualization.create_gui_frame_data()

def export_visual_data(format: str = "json") -> Union[str, Dict[str, Any]]:
    """Export visual data"""
    return sigil_ring_visualization.export_visual_data(format)
