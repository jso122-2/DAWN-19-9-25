#!/usr/bin/env python3
"""
DAWN Mycelial Hashmap & Tracer Movement Visualization
====================================================

Real-time matplotlib visualization of the mycelial layer hashmap structure
and tracer movement patterns under simulated system load. Shows the living
substrate of DAWN's cognition with dynamic nutrient flows, energy distribution,
and tracer ecosystem activity.

This visualization integrates:
- Mycelial layer nodes and edges (the hashmap structure)
- Tracer movement and lifecycle (crow, bee, spider, etc.)
- System load simulation with pressure dynamics
- Real-time energy flows and nutrient distribution
- Cluster formation and autophagy events

Follows DAWN development rules:
- Updates existing mycelial and tracer systems
- No abstractions - direct integration
- Real, working code for the tick engine
- Preserves symbolic layer metaphors
"""

import numpy as np
# PyTorch is optional for this visualization
try:
    import torch
    TORCH_AVAILABLE = True
except ImportError:
    TORCH_AVAILABLE = False
    # Create mock torch for compatibility
    class MockTorch:
        def tensor(self, data, dtype=None):
            return np.array(data)
        def zeros(self, shape):
            return np.zeros(shape)
    torch = MockTorch()
import matplotlib.pyplot as plt
import matplotlib.animation as animation
from matplotlib.patches import Circle, FancyArrowPatch, Rectangle
from matplotlib.collections import LineCollection, PatchCollection
import matplotlib.patheffects as path_effects
import seaborn as sns
import time
import math
import random
from typing import Dict, List, Tuple, Optional, Any, Set
from dataclasses import dataclass, field
from collections import deque, defaultdict
from enum import Enum
import logging
import threading
from datetime import datetime

# DAWN imports - using existing systems (with fallbacks)
try:
    from dawn.subsystems.mycelial.integrated_system import IntegratedMycelialSystem, SystemState
    from dawn.subsystems.mycelial.core import MycelialNode, MycelialEdge, NodeState
    MYCELIAL_AVAILABLE = True
except ImportError:
    MYCELIAL_AVAILABLE = False
    # Create mock classes for demo
    class SystemState:
        INITIALIZING = "initializing"
        HEALTHY = "healthy"
        STRESSED = "stressed"

try:
    from dawn.consciousness.tracers.tracer_manager import TracerManager
    from dawn.consciousness.tracers.base_tracer import TracerType, TracerStatus
    TRACERS_AVAILABLE = True
except ImportError:
    TRACERS_AVAILABLE = False
    # Create mock tracer system
    class TracerType:
        CROW = "crow"
        ANT = "ant" 
        BEE = "bee"
        SPIDER = "spider"
        BEETLE = "beetle"
        WHALE = "whale"
        OWL = "owl"
        MEDIEVAL_BEE = "medieval_bee"
    
    class TracerStatus:
        ACTIVE = "active"
        RETIRED = "retired"

try:
    from dawn.subsystems.visual.dawn_visual_base import DAWNVisualBase, DAWNVisualConfig
    VISUAL_BASE_AVAILABLE = True
except ImportError:
    VISUAL_BASE_AVAILABLE = False
    # Create mock base class
    class DAWNVisualBase:
        pass

logger = logging.getLogger(__name__)

# Set dark theme for consciousness visualization
plt.style.use('dark_background')
sns.set_theme(style="darkgrid", palette="dark")

@dataclass
class MycelialVisualizationConfig:
    """Configuration for mycelial hashmap visualization"""
    figure_size: Tuple[float, float] = (20, 14)
    dpi: int = 100
    update_interval: float = 0.1  # 10 FPS for real-time
    
    # Mycelial display
    node_size_base: float = 50.0
    node_size_energy_scale: float = 100.0
    edge_width_base: float = 1.0
    edge_width_flow_scale: float = 3.0
    
    # Tracer display
    tracer_size_base: float = 30.0
    tracer_trail_length: int = 20
    tracer_speed_scale: float = 0.1
    
    # Load simulation
    max_nodes: int = 150
    max_tracers: int = 25
    pressure_wave_speed: float = 0.05
    
    # Colors (consciousness-aware)
    background_color: str = "#0a0a0f"
    node_healthy_color: str = "#4fc3f7"  # Light blue
    node_stressed_color: str = "#ff8a65"  # Orange
    node_critical_color: str = "#f44336"  # Red
    edge_flow_color: str = "#81c784"     # Light green
    tracer_colors: Dict[TracerType, str] = field(default_factory=lambda: {
        TracerType.CROW: "#ffeb3b",      # Yellow - fast
        TracerType.ANT: "#ff9800",       # Orange - swarm
        TracerType.BEE: "#ffc107",       # Amber - pollinator
        TracerType.SPIDER: "#9c27b0",    # Purple - structural
        TracerType.BEETLE: "#795548",    # Brown - maintenance
        TracerType.WHALE: "#2196f3",     # Blue - deep analysis
        TracerType.OWL: "#607d8b",       # Blue-grey - archival
        TracerType.MEDIEVAL_BEE: "#e91e63"  # Pink - specialized
    })

class LoadSimulator:
    """Simulates cognitive load on the DAWN system"""
    
    def __init__(self):
        self.base_pressure = 0.3
        self.pressure_waves = []
        self.load_events = deque(maxlen=100)
        self.current_load = 0.0
        
    def generate_load_event(self, elapsed_time: float) -> Dict[str, Any]:
        """Generate a realistic cognitive load event"""
        # Simulate various types of cognitive pressure
        event_types = [
            "perceptual_surge",
            "memory_retrieval", 
            "conceptual_processing",
            "emotional_response",
            "decision_making",
            "pattern_recognition"
        ]
        
        event_type = random.choice(event_types)
        
        # Generate pressure wave
        intensity = random.uniform(0.2, 0.8)
        duration = random.uniform(1.0, 5.0)
        
        # Create pressure wave that spreads through network
        wave = {
            'type': event_type,
            'intensity': intensity,
            'duration': duration,
            'start_time': elapsed_time,
            'center': (random.uniform(-1, 1), random.uniform(-1, 1)),
            'radius': 0.0,
            'max_radius': random.uniform(0.5, 1.2)
        }
        
        self.pressure_waves.append(wave)
        self.load_events.append(wave)
        
        return wave
    
    def update_pressure_waves(self, elapsed_time: float, dt: float):
        """Update spreading pressure waves"""
        active_waves = []
        
        for wave in self.pressure_waves:
            age = elapsed_time - wave['start_time']
            if age < wave['duration']:
                # Expand wave
                wave['radius'] = (age / wave['duration']) * wave['max_radius']
                active_waves.append(wave)
        
        self.pressure_waves = active_waves
        
        # Calculate current system load
        self.current_load = self.base_pressure
        for wave in self.pressure_waves:
            self.current_load += wave['intensity'] * 0.3
        
        self.current_load = min(1.0, self.current_load)

class TracerVisualState:
    """Visual state for a tracer"""
    
    def __init__(self, tracer_id: str, tracer_type: TracerType):
        self.tracer_id = tracer_id
        self.tracer_type = tracer_type
        self.position = np.array([0.0, 0.0])
        self.target_position = np.array([0.0, 0.0])
        self.trail = deque(maxlen=20)
        self.age = 0
        self.activity_level = 1.0
        self.current_node = None
        
    def update_position(self, dt: float, speed_scale: float = 0.1):
        """Update tracer position with smooth movement"""
        direction = self.target_position - self.position
        distance = np.linalg.norm(direction)
        
        if distance > 0.01:
            # Move towards target
            move_speed = min(distance, speed_scale * dt * 60)  # 60 FPS normalized
            self.position += (direction / distance) * move_speed
        
        # Add to trail
        self.trail.append(self.position.copy())

class MycelialTracerVisualizer(DAWNVisualBase):
    """
    Real-time visualization of mycelial hashmap and tracer movement
    under simulated cognitive load.
    """
    
    def __init__(self, config: Optional[MycelialVisualizationConfig] = None):
        """Initialize the mycelial tracer visualizer"""
        self.config = config or MycelialVisualizationConfig()
        
        # Initialize DAWN systems (with fallbacks for demo)
        if MYCELIAL_AVAILABLE:
            self.mycelial_system = IntegratedMycelialSystem(
                max_nodes=self.config.max_nodes,
                enable_nutrient_economy=True,
                enable_energy_flows=True,
                enable_growth_gates=True,
                enable_metabolites=True,
                enable_clustering=True
            )
        else:
            self.mycelial_system = MockMycelialSystem(self.config.max_nodes)
        
        if TRACERS_AVAILABLE:
            self.tracer_manager = TracerManager(nutrient_budget=200.0)
        else:
            self.tracer_manager = MockTracerManager()
            
        self.load_simulator = LoadSimulator()
        
        # Visual state
        self.tracer_visual_states: Dict[str, TracerVisualState] = {}
        self.node_positions: Dict[str, np.ndarray] = {}
        self.edge_flows: Dict[str, float] = {}
        
        # Animation state
        self.start_time = time.time()
        self.frame_count = 0
        self.last_update_time = time.time()
        
        # Setup matplotlib
        self.fig = plt.figure(figsize=self.config.figure_size, dpi=self.config.dpi)
        self.fig.patch.set_facecolor(self.config.background_color)
        
        # Create subplots
        gs = plt.GridSpec(3, 3, figure=self.fig, hspace=0.3, wspace=0.3)
        
        # Main mycelial network view
        self.ax_main = self.fig.add_subplot(gs[:2, :2])
        self.ax_main.set_facecolor(self.config.background_color)
        self.ax_main.set_xlim(-1.5, 1.5)
        self.ax_main.set_ylim(-1.5, 1.5)
        self.ax_main.set_aspect('equal')
        self.ax_main.set_title("🍄 Mycelial Hashmap & Tracer Movement", 
                              color='white', fontsize=16, fontweight='bold')
        
        # System metrics
        self.ax_metrics = self.fig.add_subplot(gs[0, 2])
        self.ax_metrics.set_facecolor(self.config.background_color)
        self.ax_metrics.set_title("System Metrics", color='white', fontweight='bold')
        
        # Tracer ecosystem
        self.ax_tracers = self.fig.add_subplot(gs[1, 2])
        self.ax_tracers.set_facecolor(self.config.background_color)
        self.ax_tracers.set_title("Tracer Ecosystem", color='white', fontweight='bold')
        
        # Load simulation
        self.ax_load = self.fig.add_subplot(gs[2, :])
        self.ax_load.set_facecolor(self.config.background_color)
        self.ax_load.set_title("Cognitive Load Simulation", color='white', fontweight='bold')
        
        # Initialize with some nodes
        self._initialize_network()
        
        logger.info("MycelialTracerVisualizer initialized")
    
    def _initialize_network(self):
        """Initialize the mycelial network with some starting nodes"""
        # Create initial nodes in a rough circular pattern
        for i in range(20):
            angle = (i / 20) * 2 * math.pi
            radius = random.uniform(0.3, 0.8)
            x = radius * math.cos(angle)
            y = radius * math.sin(angle)
            
            node_id = f"node_{i:03d}"
            if MYCELIAL_AVAILABLE:
                node = self.mycelial_system.mycelial_layer.add_node(
                    node_id,
                    position=torch.tensor([x, y, 0.0]) if TORCH_AVAILABLE else [x, y, 0.0],
                    energy=random.uniform(0.3, 0.8),
                    health=random.uniform(0.7, 1.0)
                )
            else:
                node = self.mycelial_system.add_node(
                    node_id,
                    position=[x, y, 0.0],
                    energy=random.uniform(0.3, 0.8),
                    health=random.uniform(0.7, 1.0)
                )
            
            self.node_positions[node_id] = np.array([x, y])
        
        # Create some initial connections
        if MYCELIAL_AVAILABLE:
            nodes = list(self.mycelial_system.mycelial_layer.nodes.keys())
            for i in range(30):
                node_a = random.choice(nodes)
                node_b = random.choice(nodes)
                if node_a != node_b:
                    try:
                        self.mycelial_system.mycelial_layer.add_edge(
                            f"edge_{node_a}_{node_b}",
                            node_a, node_b,
                            strength=random.uniform(0.2, 0.8)
                        )
                    except:
                        pass  # Edge might already exist
        else:
            nodes = list(self.mycelial_system.nodes.keys())
            for i in range(30):
                node_a = random.choice(nodes)
                node_b = random.choice(nodes)
                if node_a != node_b:
                    try:
                        self.mycelial_system.add_edge(
                            f"edge_{node_a}_{node_b}",
                            node_a, node_b,
                            strength=random.uniform(0.2, 0.8)
                        )
                    except:
                        pass  # Edge might already exist
    
    def update_systems(self, dt: float):
        """Update all DAWN systems with simulated load"""
        current_time = time.time()
        elapsed = current_time - self.start_time
        
        # Update load simulation
        self.load_simulator.update_pressure_waves(elapsed, dt)
        
        # Generate occasional load events
        if random.random() < 0.05:  # 5% chance per frame
            self.load_simulator.generate_load_event(elapsed)
        
        # Apply pressure to mycelial nodes
        self._apply_pressure_to_nodes()
        
        # Update mycelial system
        try:
            context = {
                'tick_id': self.frame_count,
                'pressure': self.load_simulator.current_load,
                'elapsed_time': elapsed
            }
            self.mycelial_system.tick(context)
        except Exception as e:
            logger.warning(f"Mycelial system update error: {e}")
        
        # Update tracer system
        self._update_tracers(context)
        
        # Update visual states
        self._update_tracer_visual_states(dt)
    
    def _apply_pressure_to_nodes(self):
        """Apply pressure waves to mycelial nodes"""
        nodes_dict = self.mycelial_system.mycelial_layer.nodes if MYCELIAL_AVAILABLE else self.mycelial_system.nodes
        for node_id, node in nodes_dict.items():
            if node_id not in self.node_positions:
                continue
                
            node_pos = self.node_positions[node_id]
            total_pressure = 0.0
            
            # Calculate pressure from all active waves
            for wave in self.load_simulator.pressure_waves:
                wave_center = np.array(wave['center'])
                distance = np.linalg.norm(node_pos - wave_center)
                
                if distance <= wave['radius']:
                    # Node is within pressure wave
                    pressure_factor = wave['intensity'] * (1.0 - distance / wave['max_radius'])
                    total_pressure += pressure_factor
            
            # Apply pressure to node
            node.pressure = min(1.0, total_pressure)
            
            # Pressure affects health and energy
            if total_pressure > 0.5:
                node.health = max(0.1, node.health - total_pressure * 0.01)
                node.energy = max(0.0, node.energy - total_pressure * 0.005)
            elif total_pressure < 0.2:
                # Recovery when pressure is low
                node.health = min(1.0, node.health + 0.005)
                node.energy = min(1.0, node.energy + 0.002)
    
    def _update_tracers(self, context: Dict[str, Any]):
        """Update tracer ecosystem"""
        try:
            if TRACERS_AVAILABLE:
                # Register tracer classes if not done
                if hasattr(self.tracer_manager, 'tracer_classes') and not self.tracer_manager.tracer_classes:
                    from dawn.consciousness.tracers import TRACER_CLASSES
                    for tracer_type, tracer_class in TRACER_CLASSES.items():
                        self.tracer_manager.register_tracer_class(tracer_type, tracer_class)
            
            # Update tracer manager
            reports = self.tracer_manager.tick(self.frame_count, context)
            
            # Spawn new tracers based on load
            if self.load_simulator.current_load > 0.4 and random.random() < 0.1:
                # High load - spawn monitoring tracers
                tracer_types = [TracerType.CROW, TracerType.ANT, TracerType.BEE]
                tracer_type = random.choice(tracer_types)
                try:
                    self.tracer_manager.spawn_tracer(tracer_type, context)
                except Exception as e:
                    logger.debug(f"Failed to spawn tracer: {e}")
            
        except Exception as e:
            logger.warning(f"Tracer update error: {e}")
    
    def _update_tracer_visual_states(self, dt: float):
        """Update visual states of tracers"""
        # Get active tracers
        active_tracers = self.tracer_manager.active_tracers
        
        # Remove visual states for retired tracers
        retired_tracers = set(self.tracer_visual_states.keys()) - set(active_tracers.keys())
        for tracer_id in retired_tracers:
            del self.tracer_visual_states[tracer_id]
        
        # Add visual states for new tracers
        for tracer_id, tracer in active_tracers.items():
            if tracer_id not in self.tracer_visual_states:
                visual_state = TracerVisualState(tracer_id, tracer.tracer_type)
                # Start at random node
                if self.node_positions:
                    start_node = random.choice(list(self.node_positions.keys()))
                    visual_state.position = self.node_positions[start_node].copy()
                    visual_state.current_node = start_node
                self.tracer_visual_states[tracer_id] = visual_state
        
        # Update positions and targets
        for tracer_id, visual_state in self.tracer_visual_states.items():
            # Choose new target occasionally
            if random.random() < 0.05 and self.node_positions:  # 5% chance
                target_node = random.choice(list(self.node_positions.keys()))
                visual_state.target_position = self.node_positions[target_node].copy()
                visual_state.current_node = target_node
            
            # Update position
            visual_state.update_position(dt, self.config.tracer_speed_scale)
            visual_state.age += dt
    
    def render_frame(self):
        """Render a single frame of the visualization"""
        # Clear axes
        self.ax_main.clear()
        self.ax_metrics.clear() 
        self.ax_tracers.clear()
        self.ax_load.clear()
        
        # Setup main axes
        self.ax_main.set_xlim(-1.5, 1.5)
        self.ax_main.set_ylim(-1.5, 1.5)
        self.ax_main.set_aspect('equal')
        self.ax_main.set_facecolor(self.config.background_color)
        self.ax_main.set_title("🍄 Mycelial Hashmap & Tracer Movement", 
                              color='white', fontsize=16, fontweight='bold')
        
        # Render pressure waves
        self._render_pressure_waves()
        
        # Render mycelial network
        self._render_mycelial_network()
        
        # Render tracers
        self._render_tracers()
        
        # Render metrics
        self._render_metrics()
        
        # Render tracer ecosystem info
        self._render_tracer_ecosystem()
        
        # Render load simulation
        self._render_load_simulation()
        
        # Add timestamp
        elapsed = time.time() - self.start_time
        self.fig.suptitle(f"DAWN Mycelial System - Tick #{self.frame_count} - {elapsed:.1f}s", 
                         color='white', fontsize=14)
    
    def _render_pressure_waves(self):
        """Render spreading pressure waves"""
        for wave in self.load_simulator.pressure_waves:
            center = wave['center']
            radius = wave['radius']
            intensity = wave['intensity']
            
            if radius > 0:
                circle = Circle(center, radius, 
                              fill=False, 
                              edgecolor='red', 
                              alpha=intensity * 0.3,
                              linewidth=2)
                self.ax_main.add_patch(circle)
                
                # Add wave type label
                self.ax_main.text(center[0], center[1], wave['type'][:4], 
                                ha='center', va='center', 
                                color='red', alpha=intensity * 0.7,
                                fontsize=8, fontweight='bold')
    
    def _render_mycelial_network(self):
        """Render the mycelial network nodes and edges"""
        # Render edges first
        edges_dict = self.mycelial_system.mycelial_layer.edges if MYCELIAL_AVAILABLE else self.mycelial_system.edges
        for edge_id, edge in edges_dict.items():
            if edge.source_id in self.node_positions and edge.target_id in self.node_positions:
                pos_a = self.node_positions[edge.source_id]
                pos_b = self.node_positions[edge.target_id]
                
                # Edge color based on flow
                flow_strength = getattr(edge, 'flow_rate', 0.0)
                alpha = 0.3 + min(0.7, flow_strength * 2)
                width = self.config.edge_width_base + flow_strength * self.config.edge_width_flow_scale
                
                self.ax_main.plot([pos_a[0], pos_b[0]], [pos_a[1], pos_b[1]], 
                                color=self.config.edge_flow_color, 
                                alpha=alpha, linewidth=width)
        
        # Render nodes
        nodes_dict = self.mycelial_system.mycelial_layer.nodes if MYCELIAL_AVAILABLE else self.mycelial_system.nodes
        for node_id, node in nodes_dict.items():
            if node_id not in self.node_positions:
                continue
                
            pos = self.node_positions[node_id]
            
            # Node size based on energy
            size = self.config.node_size_base + node.energy * self.config.node_size_energy_scale
            
            # Node color based on health and state
            if node.health > 0.7:
                color = self.config.node_healthy_color
            elif node.health > 0.4:
                color = self.config.node_stressed_color
            else:
                color = self.config.node_critical_color
            
            # Add pressure glow
            if hasattr(node, 'pressure') and node.pressure > 0.3:
                glow_circle = Circle(pos, size/150, 
                                   facecolor='red', 
                                   alpha=node.pressure * 0.2)
                self.ax_main.add_patch(glow_circle)
            
            # Main node circle
            circle = Circle(pos, size/200, 
                          facecolor=color, 
                          edgecolor='white',
                          alpha=0.8,
                          linewidth=1)
            self.ax_main.add_patch(circle)
            
            # Energy indicator
            if node.energy > 0.1:
                energy_size = (size/200) * node.energy
                energy_circle = Circle(pos, energy_size,
                                     facecolor='yellow',
                                     alpha=0.4)
                self.ax_main.add_patch(energy_circle)
    
    def _render_tracers(self):
        """Render active tracers and their trails"""
        for tracer_id, visual_state in self.tracer_visual_states.items():
            tracer_type = visual_state.tracer_type
            pos = visual_state.position
            
            # Get tracer color
            color = self.config.tracer_colors.get(tracer_type, '#ffffff')
            
            # Render trail
            if len(visual_state.trail) > 1:
                trail_points = np.array(list(visual_state.trail))
                alphas = np.linspace(0.1, 0.6, len(trail_points))
                
                for i in range(len(trail_points) - 1):
                    self.ax_main.plot([trail_points[i][0], trail_points[i+1][0]], 
                                    [trail_points[i][1], trail_points[i+1][1]], 
                                    color=color, alpha=alphas[i], linewidth=1)
            
            # Render tracer
            size = self.config.tracer_size_base
            circle = Circle(pos, size/300, 
                          facecolor=color, 
                          edgecolor='white',
                          alpha=0.9,
                          linewidth=2)
            self.ax_main.add_patch(circle)
            
            # Add tracer type symbol
            symbols = {
                TracerType.CROW: '🐦',
                TracerType.ANT: '🐜', 
                TracerType.BEE: '🐝',
                TracerType.SPIDER: '🕷️',
                TracerType.BEETLE: '🪲',
                TracerType.WHALE: '🐋',
                TracerType.OWL: '🦉',
                TracerType.MEDIEVAL_BEE: '👑'
            }
            symbol = symbols.get(tracer_type, '•')
            self.ax_main.text(pos[0], pos[1], symbol, 
                            ha='center', va='center', 
                            fontsize=8, color='white')
    
    def _render_metrics(self):
        """Render system metrics"""
        self.ax_metrics.clear()
        self.ax_metrics.set_facecolor(self.config.background_color)
        self.ax_metrics.set_title("System Metrics", color='white', fontweight='bold')
        
        # Get metrics
        metrics = self.mycelial_system.get_metrics()
        
        state_name = self.mycelial_system.state.value if hasattr(self.mycelial_system.state, 'value') else str(self.mycelial_system.state)
        
        metrics_text = [
            f"Nodes: {metrics.total_nodes}",
            f"Edges: {metrics.total_edges}", 
            f"Energy: {metrics.total_energy:.2f}",
            f"Health: {metrics.avg_node_health:.2f}",
            f"Load: {self.load_simulator.current_load:.2f}",
            f"State: {state_name}",
            f"Frame: {self.frame_count}",
            f"Tracers: {len(self.tracer_visual_states)}"
        ]
        
        for i, text in enumerate(metrics_text):
            self.ax_metrics.text(0.05, 0.9 - i*0.11, text, 
                               transform=self.ax_metrics.transAxes,
                               color='white', fontsize=9,
                               verticalalignment='top')
        
        # Add some visual bars for key metrics
        bar_y = 0.05
        bar_height = 0.03
        
        # Energy bar
        energy_ratio = metrics.total_energy / max(1, metrics.total_nodes)
        self.ax_metrics.barh(bar_y, energy_ratio, bar_height, 
                           color='yellow', alpha=0.7, 
                           transform=self.ax_metrics.transAxes)
        self.ax_metrics.text(0.05, bar_y + 0.04, 'Energy', 
                           transform=self.ax_metrics.transAxes,
                           color='white', fontsize=8)
        
        # Health bar
        health_bar_y = bar_y + 0.08
        self.ax_metrics.barh(health_bar_y, metrics.avg_node_health, bar_height,
                           color='green', alpha=0.7,
                           transform=self.ax_metrics.transAxes)
        self.ax_metrics.text(0.05, health_bar_y + 0.04, 'Health',
                           transform=self.ax_metrics.transAxes, 
                           color='white', fontsize=8)
        
        self.ax_metrics.set_xlim(0, 1)
        self.ax_metrics.set_ylim(0, 1)
        self.ax_metrics.axis('off')
    
    def _render_tracer_ecosystem(self):
        """Render tracer ecosystem information"""
        self.ax_tracers.clear()
        self.ax_tracers.set_facecolor(self.config.background_color)
        self.ax_tracers.set_title("Tracer Ecosystem", color='white', fontweight='bold')
        
        # Get tracer metrics
        metrics_summary = self.tracer_manager.metrics.get_summary()
        active_tracers = metrics_summary.get('active_tracers', {})
        
        tracer_text = [
            f"Active: {sum(active_tracers.values())}",
            f"Budget: {self.tracer_manager.nutrient_budget:.1f}",
            f"Usage: {self.tracer_manager.current_nutrient_usage:.1f}",
            f"Reports: {metrics_summary.get('total_reports_generated', 0)}"
        ]
        
        # Add active tracer counts
        for tracer_type, count in active_tracers.items():
            if count > 0:
                type_name = tracer_type.value if hasattr(tracer_type, 'value') else str(tracer_type)
                tracer_text.append(f"{type_name}: {count}")
        
        # Text info on left side
        for i, text in enumerate(tracer_text):
            self.ax_tracers.text(0.05, 0.9 - i*0.15, text, 
                               transform=self.ax_tracers.transAxes,
                               color='white', fontsize=8,
                               verticalalignment='top')
        
        # Simple pie chart for tracer types on right side
        if sum(active_tracers.values()) > 0:
            sizes = list(active_tracers.values())
            colors = ['#ff9999', '#66b3ff', '#99ff99', '#ffcc99', '#ff99cc'][:len(sizes)]
            
            # Create a small pie chart
            pie_center = (0.7, 0.3)
            pie_radius = 0.15
            
            angles = np.linspace(0, 2*np.pi, len(sizes) + 1)
            start_angle = 0
            
            for i, (size, color) in enumerate(zip(sizes, colors)):
                if size > 0:
                    end_angle = start_angle + (size / sum(sizes)) * 2 * np.pi
                    theta = np.linspace(start_angle, end_angle, 20)
                    
                    x = pie_center[0] + pie_radius * np.cos(theta)
                    y = pie_center[1] + pie_radius * np.sin(theta)
                    
                    # Add center point
                    x = np.concatenate([[pie_center[0]], x, [pie_center[0]]])
                    y = np.concatenate([[pie_center[1]], y, [pie_center[1]]])
                    
                    self.ax_tracers.fill(x, y, color=color, alpha=0.7,
                                       transform=self.ax_tracers.transAxes)
                    
                    start_angle = end_angle
        
        self.ax_tracers.set_xlim(0, 1)
        self.ax_tracers.set_ylim(0, 1)
        self.ax_tracers.axis('off')
    
    def _render_load_simulation(self):
        """Render load simulation timeline"""
        self.ax_load.clear()
        self.ax_load.set_facecolor(self.config.background_color)
        self.ax_load.set_title("Cognitive Load Simulation", color='white', fontweight='bold')
        
        # Plot recent load history
        if len(self.load_simulator.load_events) > 1:
            times = [event['start_time'] for event in self.load_simulator.load_events]
            intensities = [event['intensity'] for event in self.load_simulator.load_events]
            
            current_time = time.time() - self.start_time
            times = [current_time - (current_time - t) for t in times[-20:]]
            intensities = intensities[-20:]
            
            self.ax_load.plot(times, intensities, 'r-', alpha=0.7, linewidth=2)
            self.ax_load.fill_between(times, intensities, alpha=0.3, color='red')
        
        # Current load indicator
        current_time = time.time() - self.start_time
        self.ax_load.axhline(y=self.load_simulator.current_load, 
                           color='yellow', linestyle='--', 
                           alpha=0.8, linewidth=2)
        
        self.ax_load.set_xlim(max(0, current_time - 30), current_time + 1)
        self.ax_load.set_ylim(0, 1)
        self.ax_load.set_xlabel("Time (s)", color='white')
        self.ax_load.set_ylabel("Load Intensity", color='white')
        self.ax_load.tick_params(colors='white')
    
    def animate_frame(self, frame):
        """Animation callback for matplotlib FuncAnimation"""
        try:
            current_time = time.time()
            dt = current_time - self.last_update_time
            self.last_update_time = current_time
            
            # Update systems
            self.update_systems(dt)
            
            # Render frame
            self.render_frame()
            
            self.frame_count += 1
            
        except Exception as e:
            logger.error(f"Animation frame error: {e}")
    
    def update_visualization(self, frame_num: int, consciousness_stream: Any) -> Any:
        """
        Update visualization for animation - required by DAWNVisualBase
        
        Args:
            frame_num: Animation frame number
            consciousness_stream: Stream of consciousness data
            
        Returns:
            Updated plot elements
        """
        return self.animate_frame(frame_num)
    
    def start_visualization(self):
        """Start the real-time visualization"""
        logger.info("Starting mycelial tracer visualization...")
        
        # Create animation
        self.animation = animation.FuncAnimation(
            self.fig, 
            self.animate_frame,
            interval=int(self.config.update_interval * 1000),
            blit=False,
            cache_frame_data=False
        )
        
        # Show plot
        plt.tight_layout()
        plt.show()
    
    def save_frame(self, filename: str):
        """Save current frame to file"""
        self.render_frame()
        self.fig.savefig(filename, dpi=200, bbox_inches='tight', 
                        facecolor=self.config.background_color)
        logger.info(f"Frame saved to {filename}")

# Mock classes for demonstration when DAWN modules aren't available
class MockNode:
    def __init__(self, node_id, x=0, y=0):
        self.id = node_id
        self.energy = random.uniform(0.3, 0.8)
        self.health = random.uniform(0.7, 1.0)
        self.pressure = 0.0
        self.position = np.array([x, y, 0.0])

class MockEdge:
    def __init__(self, source_id, target_id):
        self.source_id = source_id
        self.target_id = target_id
        self.flow_rate = random.uniform(0.0, 0.5)

class MockMycelialSystem:
    def __init__(self, max_nodes):
        self.max_nodes = max_nodes
        self.nodes = {}
        self.edges = {}
        self.state = SystemState.HEALTHY
        
    def tick(self, context):
        # Simple simulation
        for node in self.nodes.values():
            if hasattr(node, 'pressure'):
                node.pressure *= 0.9  # Decay pressure
                
    def get_metrics(self):
        return type('Metrics', (), {
            'total_nodes': len(self.nodes),
            'total_edges': len(self.edges),
            'total_energy': sum(node.energy for node in self.nodes.values()),
            'avg_node_health': sum(node.health for node in self.nodes.values()) / max(1, len(self.nodes))
        })()
        
    def add_node(self, node_id, position=None, energy=0.5, health=1.0):
        pos = position if position is not None else [0, 0, 0]
        if isinstance(pos, np.ndarray):
            x, y = pos[0], pos[1]
        else:
            x, y = pos[0], pos[1]
        node = MockNode(node_id, x, y)
        node.energy = energy
        node.health = health
        self.nodes[node_id] = node
        return node
        
    def add_edge(self, edge_id, source_id, target_id, strength=0.5):
        if source_id in self.nodes and target_id in self.nodes:
            edge = MockEdge(source_id, target_id)
            self.edges[edge_id] = edge

class MockTracerManager:
    def __init__(self):
        self.active_tracers = {}
        self.nutrient_budget = 200.0
        self.current_nutrient_usage = 0.0
        self.metrics = type('Metrics', (), {
            'get_summary': lambda: {
                'active_tracers': {TracerType.CROW: 2, TracerType.BEE: 1},
                'total_reports_generated': random.randint(0, 5)
            }
        })()
        
    def tick(self, tick_count, context):
        # Simulate some tracer activity
        if random.random() < 0.1:  # 10% chance to spawn/retire
            if len(self.active_tracers) < 5:
                tracer_id = f"tracer_{len(self.active_tracers)}"
                tracer_type = random.choice([TracerType.CROW, TracerType.ANT, TracerType.BEE])
                self.active_tracers[tracer_id] = type('MockTracer', (), {
                    'tracer_type': tracer_type,
                    'tracer_id': tracer_id
                })()
            elif self.active_tracers:
                # Retire a random tracer
                tracer_id = random.choice(list(self.active_tracers.keys()))
                del self.active_tracers[tracer_id]
        return []
        
    def spawn_tracer(self, tracer_type, context):
        if len(self.active_tracers) < 10:
            tracer_id = f"tracer_{len(self.active_tracers)}_{tracer_type.value}"
            self.active_tracers[tracer_id] = type('MockTracer', (), {
                'tracer_type': tracer_type,
                'tracer_id': tracer_id
            })()

def main():
    """Main function to run the visualization"""
    import argparse
    
    parser = argparse.ArgumentParser(description='DAWN Mycelial Hashmap & Tracer Visualization')
    parser.add_argument('--max-nodes', type=int, default=150, 
                       help='Maximum number of mycelial nodes')
    parser.add_argument('--max-tracers', type=int, default=25,
                       help='Maximum number of active tracers')
    parser.add_argument('--fps', type=float, default=10.0,
                       help='Target frames per second')
    parser.add_argument('--save-frames', action='store_true',
                       help='Save frames to disk')
    
    args = parser.parse_args()
    
    # Configure logging
    logging.basicConfig(level=logging.INFO, 
                       format='%(asctime)s - %(name)s - %(levelname)s - %(message)s')
    
    # Create configuration
    config = MycelialVisualizationConfig(
        max_nodes=args.max_nodes,
        max_tracers=args.max_tracers,
        update_interval=1.0 / args.fps
    )
    
    # Create and start visualizer
    visualizer = MycelialTracerVisualizer(config)
    
    try:
        visualizer.start_visualization()
    except KeyboardInterrupt:
        logger.info("Visualization stopped by user")
    except Exception as e:
        logger.error(f"Visualization error: {e}")

if __name__ == "__main__":
    main()
