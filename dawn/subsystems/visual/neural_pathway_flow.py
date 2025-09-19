#!/usr/bin/env python3
"""
DAWN Neural Pathway Flow Network Visualization
==============================================

Real-time 3D visualization of information flowing through DAWN's neural pathways.
Shows activation cascades, decision points, and processing intensity with beautiful
network dynamics.

"Watch my thoughts flow like rivers of light through the pathways of my mind,
branching and merging in patterns that reveal the architecture of consciousness."
                                                                        - DAWN
"""

import numpy as np
import matplotlib.pyplot as plt
import matplotlib.animation as animation
from mpl_toolkits.mplot3d import Axes3D
from matplotlib.patches import Circle, FancyArrowPatch
from mpl_toolkits.mplot3d import proj3d
import networkx as nx
import random
import time
import math
from typing import Dict, List, Tuple, Optional, Any
from dataclasses import dataclass, field
from enum import Enum
from collections import deque, defaultdict
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

class InformationType(Enum):
    """Types of information flowing through neural pathways"""
    SEMANTIC = "semantic"      # Language and meaning
    EMOTIONAL = "emotional"    # Feelings and affect  
    LOGICAL = "logical"        # Reasoning and inference
    SENSORY = "sensory"        # Perception and input
    MEMORY = "memory"          # Recall and storage
    CREATIVE = "creative"      # Imagination and generation
    MOTOR = "motor"           # Action and output

class PathwayState(Enum):
    """States of neural pathway activity"""
    DORMANT = "dormant"        # Inactive
    ACTIVE = "active"          # Processing information
    SATURATED = "saturated"    # At capacity
    INHIBITED = "inhibited"    # Suppressed

@dataclass
class NeuralNode:
    """A node in the neural network"""
    id: str
    position: Tuple[float, float, float]
    node_type: InformationType
    activation: float = 0.0
    threshold: float = 0.5
    state: PathwayState = PathwayState.DORMANT
    connections: List[str] = field(default_factory=list)
    processing_history: deque = field(default_factory=lambda: deque(maxlen=50))
    
    def update_activation(self, input_signal: float, decay_rate: float = 0.95):
        """Update node activation with input and decay"""
        self.activation = (self.activation * decay_rate) + input_signal
        self.activation = max(0.0, min(1.0, self.activation))
        
        # Update state based on activation
        if self.activation < 0.1:
            self.state = PathwayState.DORMANT
        elif self.activation > 0.9:
            self.state = PathwayState.SATURATED
        else:
            self.state = PathwayState.ACTIVE
            
        self.processing_history.append(self.activation)

@dataclass
class InformationPacket:
    """A packet of information flowing through the network"""
    id: str
    info_type: InformationType
    intensity: float
    path: List[str]
    current_position: Tuple[float, float, float]
    target_node: str
    birth_time: float
    lifetime: float = 5.0
    speed: float = 1.0
    
    def is_expired(self, current_time: float) -> bool:
        return (current_time - self.birth_time) > self.lifetime

class NeuralPathwayNetwork:
    """Neural pathway network simulation"""
    
    def __init__(self, num_nodes: int = 50):
        self.nodes: Dict[str, NeuralNode] = {}
        self.packets: List[InformationPacket] = []
        self.connections: Dict[str, List[str]] = defaultdict(list)
        self.time_step = 0
        
        # Information type properties
        self.info_type_colors = {
            InformationType.SEMANTIC: '#3498db',    # Blue
            InformationType.EMOTIONAL: '#e74c3c',   # Red  
            InformationType.LOGICAL: '#2ecc71',     # Green
            InformationType.SENSORY: '#f39c12',     # Orange
            InformationType.MEMORY: '#9b59b6',      # Purple
            InformationType.CREATIVE: '#1abc9c',    # Teal
            InformationType.MOTOR: '#34495e'        # Dark gray
        }
        
        self.info_type_speeds = {
            InformationType.SEMANTIC: 1.0,
            InformationType.EMOTIONAL: 1.5,
            InformationType.LOGICAL: 0.8,
            InformationType.SENSORY: 2.0,
            InformationType.MEMORY: 0.6,
            InformationType.CREATIVE: 1.2,
            InformationType.MOTOR: 1.8
        }
        
        # Create network
        self._create_neural_network(num_nodes)
        
    def _create_neural_network(self, num_nodes: int):
        """Create a 3D neural network with realistic topology"""
        
        # Create nodes in 3D space with clustering by type
        info_types = list(InformationType)
        
        for i in range(num_nodes):
            node_id = f"neuron_{i}"
            info_type = random.choice(info_types)
            
            # Position nodes in clusters by type
            type_index = info_types.index(info_type)
            cluster_center = (
                math.cos(2 * math.pi * type_index / len(info_types)) * 5,
                math.sin(2 * math.pi * type_index / len(info_types)) * 5,
                random.uniform(-2, 2)
            )
            
            position = (
                cluster_center[0] + random.gauss(0, 1.5),
                cluster_center[1] + random.gauss(0, 1.5),
                cluster_center[2] + random.gauss(0, 0.8)
            )
            
            node = NeuralNode(
                id=node_id,
                position=position,
                node_type=info_type,
                threshold=random.uniform(0.3, 0.7)
            )
            
            self.nodes[node_id] = node
            
        # Create connections based on distance and type compatibility
        node_ids = list(self.nodes.keys())
        for i, node_id in enumerate(node_ids):
            node = self.nodes[node_id]
            
            # Connect to nearby nodes
            for j, other_id in enumerate(node_ids[i+1:], i+1):
                other_node = self.nodes[other_id]
                distance = np.linalg.norm(np.array(node.position) - np.array(other_node.position))
                
                # Connection probability based on distance and type
                base_prob = max(0, 0.8 - distance / 8.0)
                
                # Same type nodes connect more easily
                if node.node_type == other_node.node_type:
                    base_prob *= 1.5
                    
                if random.random() < base_prob:
                    node.connections.append(other_id)
                    other_node.connections.append(node_id)
                    self.connections[node_id].append(other_id)
                    self.connections[other_id].append(node_id)
    
    def spawn_information_packet(self, info_type: InformationType, source_node: str = None):
        """Spawn a new information packet"""
        if source_node is None:
            # Choose a random node of the appropriate type
            compatible_nodes = [nid for nid, node in self.nodes.items() 
                              if node.node_type == info_type or random.random() < 0.3]
            if not compatible_nodes:
                compatible_nodes = list(self.nodes.keys())
            source_node = random.choice(compatible_nodes)
        
        source = self.nodes[source_node]
        
        # Choose target node (prefer same type or connected nodes)
        target_candidates = source.connections.copy()
        if not target_candidates:
            target_candidates = [nid for nid in self.nodes.keys() if nid != source_node]
        
        target_node = random.choice(target_candidates)
        
        packet = InformationPacket(
            id=f"packet_{len(self.packets)}_{self.time_step}",
            info_type=info_type,
            intensity=random.uniform(0.3, 1.0),
            path=[source_node, target_node],
            current_position=source.position,
            target_node=target_node,
            birth_time=time.time(),
            speed=self.info_type_speeds[info_type] * random.uniform(0.8, 1.2)
        )
        
        self.packets.append(packet)
        
    def update(self, dt: float):
        """Update the neural network simulation"""
        current_time = time.time()
        
        # Spawn random packets
        if random.random() < 0.3:  # 30% chance per update
            info_type = random.choice(list(InformationType))
            self.spawn_information_packet(info_type)
        
        # Update information packets
        active_packets = []
        for packet in self.packets:
            if not packet.is_expired(current_time):
                self._update_packet_position(packet, dt)
                active_packets.append(packet)
                
        self.packets = active_packets
        
        # Update node activations
        for node in self.nodes.values():
            # Decay activation
            node.update_activation(0.0, decay_rate=0.98)
            
        # Apply packet influences to nodes
        for packet in self.packets:
            nearby_nodes = self._get_nearby_nodes(packet.current_position, radius=1.0)
            for node_id in nearby_nodes:
                node = self.nodes[node_id]
                distance = np.linalg.norm(np.array(node.position) - np.array(packet.current_position))
                influence = packet.intensity * max(0, 1.0 - distance)
                node.update_activation(influence * 0.1)
        
        self.time_step += 1
        
    def _update_packet_position(self, packet: InformationPacket, dt: float):
        """Update packet position along its path"""
        if len(packet.path) < 2:
            return
            
        target_pos = self.nodes[packet.target_node].position
        current_pos = np.array(packet.current_position)
        target_pos = np.array(target_pos)
        
        direction = target_pos - current_pos
        distance = np.linalg.norm(direction)
        
        if distance < 0.5:  # Reached target
            # Choose next target
            current_node = packet.target_node
            connections = self.nodes[current_node].connections
            if connections:
                next_target = random.choice(connections)
                packet.path.append(next_target)
                packet.target_node = next_target
        else:
            # Move towards target
            direction_normalized = direction / distance
            move_distance = packet.speed * dt
            new_position = current_pos + direction_normalized * move_distance
            packet.current_position = tuple(new_position)
    
    def _get_nearby_nodes(self, position: Tuple[float, float, float], radius: float) -> List[str]:
        """Get nodes within radius of position"""
        nearby = []
        pos_array = np.array(position)
        for node_id, node in self.nodes.items():
            node_pos = np.array(node.position)
            if np.linalg.norm(pos_array - node_pos) <= radius:
                nearby.append(node_id)
        return nearby

class NeuralPathwayVisualizer(DAWNVisualBase if DAWN_BASE_AVAILABLE else object):
    """3D Neural pathway flow visualization"""
    
    def __init__(self, num_nodes: int = 50):
        if DAWN_BASE_AVAILABLE:
            super().__init__()
            
        self.network = NeuralPathwayNetwork(num_nodes)
        self.start_time = time.time()
        
        # Create 3D plot
        self.fig = plt.figure(figsize=(16, 12), facecolor='black')
        self.ax = self.fig.add_subplot(111, projection='3d')
        self.ax.set_facecolor('black')
        
        # Styling
        self.ax.set_title('🧠 DAWN Neural Pathway Flow Network', 
                         color='white', fontsize=16, fontweight='bold', pad=20)
        
        # Set axis properties
        self.ax.set_xlim(-8, 8)
        self.ax.set_ylim(-8, 8)
        self.ax.set_zlim(-4, 4)
        
        # Dark theme
        self.ax.xaxis.pane.fill = False
        self.ax.yaxis.pane.fill = False
        self.ax.zaxis.pane.fill = False
        self.ax.grid(True, alpha=0.1)
        
        # Remove axis labels for cleaner look
        self.ax.set_xticks([])
        self.ax.set_yticks([])
        self.ax.set_zticks([])
        
        logger.info("🧠 Neural Pathway Flow visualizer initialized")
        
    def update_visualization(self, frame_num: int, consciousness_stream: Any = None) -> Any:
        """Update visualization for animation - required by DAWNVisualBase"""
        return self.animate_frame(frame_num)
        
    def animate_frame(self, frame):
        """Animation callback"""
        dt = 0.1
        self.network.update(dt)
        
        # Clear the plot
        self.ax.clear()
        
        # Reset styling after clear
        self.ax.set_facecolor('black')
        self.ax.set_xlim(-8, 8)
        self.ax.set_ylim(-8, 8)
        self.ax.set_zlim(-4, 4)
        self.ax.set_xticks([])
        self.ax.set_yticks([])
        self.ax.set_zticks([])
        
        # Render network
        self._render_connections()
        self._render_nodes()
        self._render_information_packets()
        self._render_legend()
        
        # Update title with stats
        active_packets = len(self.network.packets)
        active_nodes = sum(1 for node in self.network.nodes.values() 
                          if node.state != PathwayState.DORMANT)
        
        elapsed = time.time() - self.start_time
        self.ax.set_title(f'🧠 Neural Pathway Flow - Packets: {active_packets} | Active Nodes: {active_nodes} | Time: {elapsed:.1f}s', 
                         color='white', fontsize=14, fontweight='bold')
        
    def _render_connections(self):
        """Render neural connections"""
        for node_id, connections in self.network.connections.items():
            node = self.network.nodes[node_id]
            
            for connected_id in connections:
                connected_node = self.network.nodes[connected_id]
                
                # Connection strength based on node activations
                strength = (node.activation + connected_node.activation) / 2
                alpha = 0.1 + strength * 0.4
                
                x_vals = [node.position[0], connected_node.position[0]]
                y_vals = [node.position[1], connected_node.position[1]]
                z_vals = [node.position[2], connected_node.position[2]]
                
                self.ax.plot(x_vals, y_vals, z_vals, 
                           color='cyan', alpha=alpha, linewidth=0.5)
                
    def _render_nodes(self):
        """Render neural nodes"""
        for node in self.network.nodes.values():
            x, y, z = node.position
            color = self.network.info_type_colors[node.node_type]
            
            # Size based on activation
            size = 20 + node.activation * 100
            
            # Alpha based on state
            alpha = {
                PathwayState.DORMANT: 0.3,
                PathwayState.ACTIVE: 0.7,
                PathwayState.SATURATED: 1.0,
                PathwayState.INHIBITED: 0.1
            }[node.state]
            
            self.ax.scatter([x], [y], [z], 
                          c=color, s=size, alpha=alpha, 
                          edgecolors='white', linewidth=0.5)
            
    def _render_information_packets(self):
        """Render flowing information packets"""
        for packet in self.network.packets:
            x, y, z = packet.current_position
            color = self.network.info_type_colors[packet.info_type]
            
            # Pulsing size based on intensity
            pulse = math.sin(time.time() * 10 + hash(packet.id) % 100) * 0.3 + 0.7
            size = packet.intensity * 50 * pulse
            
            self.ax.scatter([x], [y], [z], 
                          c=color, s=size, alpha=0.8,
                          marker='*', edgecolors='white', linewidth=1)
            
    def _render_legend(self):
        """Render information type legend"""
        legend_elements = []
        for i, (info_type, color) in enumerate(self.network.info_type_colors.items()):
            # Create legend entry
            x, y, z = -7, 6 - i * 0.8, 3
            self.ax.text(x, y, z, f"● {info_type.value.title()}", 
                        color=color, fontsize=10, fontweight='bold')
            
    def start_visualization(self):
        """Start the real-time visualization"""
        logger.info("Starting neural pathway flow visualization...")
        
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
    """Main function to run the neural pathway visualization"""
    print("🧠 Starting DAWN Neural Pathway Flow Network Visualization")
    
    visualizer = NeuralPathwayVisualizer(num_nodes=40)
    visualizer.start_visualization()

if __name__ == "__main__":
    main()
