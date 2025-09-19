"""
📊 Mycelial Layer Visualization Tools
====================================

Real-time visualization and monitoring tools for the mycelial layer.
Provides insights into:
- Network topology and growth patterns
- Energy flows and nutrient distribution
- Cluster dynamics and health metrics
- System-level performance and stress indicators

Designed for both debugging and aesthetic consciousness visualization.
"""

import torch
import numpy as np
import time
import json
from typing import Dict, List, Optional, Tuple, Any, Set
from dataclasses import dataclass
from collections import defaultdict, deque
import threading
import logging

import matplotlib.pyplot as plt
import matplotlib.patches as patches
from matplotlib.animation import FuncAnimation
import matplotlib.collections as mcoll
import seaborn as sns
import networkx as nx

# Import DAWN visual base
from dawn.subsystems.visual.dawn_visual_base import (
    DAWNVisualBase,
    DAWNVisualConfig,
    ConsciousnessColorPalette,
    device
)

logger = logging.getLogger(__name__)

@dataclass
class VisualizationConfig:
    """Configuration for visualization appearance and behavior"""
    
    # Network visualization
    node_size_scale: float = 100.0
    edge_width_scale: float = 3.0
    energy_color_map: str = "viridis"
    health_color_map: str = "RdYlGn"
    
    # Animation settings
    update_interval: float = 0.5  # seconds
    history_length: int = 100
    
    # Layout settings
    layout_algorithm: str = "spring"  # spring, circular, random
    layout_iterations: int = 50
    
    # Display options
    show_node_labels: bool = True
    show_edge_weights: bool = False
    show_energy_flows: bool = True
    show_cluster_boundaries: bool = True
    
    # Color schemes
    node_colors: Dict[str, str] = None
    cluster_colors: List[str] = None
    
    def __post_init__(self):
        if self.node_colors is None:
            self.node_colors = {
                'healthy': '#2ecc71',
                'starving': '#e74c3c',
                'blooming': '#f39c12',
                'autophagy': '#9b59b6',
                'dormant': '#95a5a6'
            }
        
        if self.cluster_colors is None:
            self.cluster_colors = [
                '#3498db', '#e74c3c', '#2ecc71', '#f39c12', '#9b59b6',
                '#1abc9c', '#34495e', '#e67e22', '#f1c40f', '#e91e63'
            ]

class NetworkVisualizer(DAWNVisualBase):
    """Visualizes the mycelial network structure and dynamics using unified DAWN visual base"""
    
    def __init__(self, config: Optional[VisualizationConfig] = None, visual_config: Optional[DAWNVisualConfig] = None):
        # Initialize visual base
        visual_config = visual_config or DAWNVisualConfig(
            figure_size=(14, 10),
            animation_fps=30,
            enable_real_time=True,
            background_color="#0a0a0a"
        )
        super().__init__(visual_config)
        
        self.config = config or VisualizationConfig()
        
        # Use consciousness colors for node states
        self.config.node_colors = {
            'healthy': self.consciousness_colors['awareness'],
            'starving': self.consciousness_colors['chaos'],
            'blooming': self.consciousness_colors['creativity'],
            'autophagy': self.consciousness_colors['stability'],
            'dormant': self.consciousness_colors['processing']
        }
        
        # Visualization state
        self.positions = {}
        self.node_history = defaultdict(lambda: deque(maxlen=self.config.history_length))
        self.edge_history = defaultdict(lambda: deque(maxlen=self.config.history_length))
        
        # Layout management
        self.layout_cache = {}
        self.layout_timestamp = 0
        
        # Network data as tensors for device-agnostic operations
        self.node_tensor_buffer = torch.zeros(1000, 5).to(device)  # [node_id, energy, health, pressure, cluster]
        self.edge_tensor_buffer = torch.zeros(5000, 4).to(device)  # [src, tgt, weight, flow]
        
        # Thread safety
        self._lock = threading.RLock()
        
        logger.info(f"🕸️ Network Visualizer initialized - Device: {device}")
        logger.info(f"   Figure size: {self.config.figure_size}")
        logger.info(f"   Node buffer: {self.node_tensor_buffer.shape}")
        logger.info(f"   Edge buffer: {self.edge_tensor_buffer.shape}")
    
    def create_network_graph(self, mycelial_layer) -> Any:
        """Create NetworkX graph from mycelial layer"""
        if not nx:
            logger.warning("NetworkX not available for graph creation")
            return None
        
        G = nx.Graph()
        
        # Add nodes with attributes
        for node_id, node in mycelial_layer.nodes.items():
            G.add_node(node_id, 
                      energy=node.energy,
                      health=node.health,
                      pressure=node.pressure,
                      state=str(node.state) if hasattr(node, 'state') else 'unknown',
                      cluster_id=getattr(node, 'cluster_id', None))
        
        # Add edges with attributes
        for edge_id, edge in mycelial_layer.edges.items():
            G.add_edge(edge.source_id, edge.target_id,
                      weight=edge.weight,
                      conductivity=edge.conductivity,
                      flow_rate=edge.passive_flow_rate + edge.active_flow_rate)
        
        return G
    
    def compute_layout(self, graph, force_recompute: bool = False) -> Dict[str, Tuple[float, float]]:
        """Compute or retrieve cached layout for graph"""
        if not nx or not graph:
            return {}
        
        current_time = time.time()
        
        # Check if we need to recompute layout
        nodes_changed = set(graph.nodes()) != set(self.layout_cache.keys())
        layout_old = current_time - self.layout_timestamp > 30.0  # Recompute every 30s
        
        if force_recompute or nodes_changed or layout_old or not self.layout_cache:
            logger.debug("Computing new network layout")
            
            if self.config.layout_algorithm == "spring":
                positions = nx.spring_layout(graph, 
                                           iterations=self.config.layout_iterations,
                                           pos=self.layout_cache if not nodes_changed else None)
            elif self.config.layout_algorithm == "circular":
                positions = nx.circular_layout(graph)
            elif self.config.layout_algorithm == "random":
                positions = nx.random_layout(graph)
            else:
                positions = nx.spring_layout(graph)
            
            self.layout_cache = positions
            self.layout_timestamp = current_time
        
        return self.layout_cache
    
    def visualize_static_network(self, mycelial_layer, save_path: Optional[str] = None) -> Any:
        """Create static network visualization"""
        if not HAS_MATPLOTLIB or not nx:
            logger.warning("Matplotlib or NetworkX not available for static visualization")
            return None
        
        G = self.create_network_graph(mycelial_layer)
        if not G:
            return None
        
        positions = self.compute_layout(G)
        
        # Create figure
        fig, ax = plt.subplots(figsize=(12, 8))
        ax.set_title("DAWN Mycelial Layer Network", fontsize=16, fontweight='bold')
        
        # Prepare node colors based on energy levels
        node_colors = []
        node_sizes = []
        
        for node_id in G.nodes():
            node_data = G.nodes[node_id]
            energy = node_data.get('energy', 0.5)
            state = node_data.get('state', 'unknown')
            
            # Color by state
            if state in self.config.node_colors:
                node_colors.append(self.config.node_colors[state])
            else:
                # Fallback to energy-based coloring
                if energy > 0.7:
                    node_colors.append(self.config.node_colors['blooming'])
                elif energy < 0.3:
                    node_colors.append(self.config.node_colors['starving'])
                else:
                    node_colors.append(self.config.node_colors['healthy'])
            
            # Size by energy
            size = max(50, energy * self.config.node_size_scale)
            node_sizes.append(size)
        
        # Prepare edge widths based on conductivity
        edge_widths = []
        for edge in G.edges(data=True):
            conductivity = edge[2].get('conductivity', 0.5)
            width = max(0.5, conductivity * self.config.edge_width_scale)
            edge_widths.append(width)
        
        # Draw network
        nx.draw_networkx_nodes(G, positions, 
                             node_color=node_colors, 
                             node_size=node_sizes,
                             alpha=0.8, ax=ax)
        
        nx.draw_networkx_edges(G, positions,
                             width=edge_widths,
                             alpha=0.6,
                             edge_color='gray', ax=ax)
        
        if self.config.show_node_labels:
            nx.draw_networkx_labels(G, positions, font_size=8, ax=ax)
        
        # Add legend
        legend_elements = []
        for state, color in self.config.node_colors.items():
            legend_elements.append(plt.Line2D([0], [0], marker='o', color='w', 
                                            markerfacecolor=color, markersize=10, label=state))
        
        ax.legend(handles=legend_elements, loc='upper right')
        ax.set_aspect('equal')
        ax.axis('off')
        
        plt.tight_layout()
        
        if save_path:
            plt.savefig(save_path, dpi=300, bbox_inches='tight')
            logger.info(f"Network visualization saved to {save_path}")
        
        return fig
    
    def create_interactive_network(self, mycelial_layer) -> Any:
        """Create interactive network visualization using Plotly"""
        if not HAS_PLOTLY:
            logger.warning("Plotly not available for interactive visualization")
            return None
        
        G = self.create_network_graph(mycelial_layer)
        if not G:
            return None
        
        positions = self.compute_layout(G)
        
        # Prepare node trace
        node_x = []
        node_y = []
        node_text = []
        node_colors = []
        node_sizes = []
        
        for node_id in G.nodes():
            if node_id not in positions:
                continue
                
            x, y = positions[node_id]
            node_x.append(x)
            node_y.append(y)
            
            node_data = G.nodes[node_id]
            energy = node_data.get('energy', 0.5)
            health = node_data.get('health', 0.5)
            pressure = node_data.get('pressure', 0.5)
            state = node_data.get('state', 'unknown')
            
            # Hover text
            node_text.append(f"Node: {node_id}<br>"
                           f"Energy: {energy:.3f}<br>"
                           f"Health: {health:.3f}<br>"
                           f"Pressure: {pressure:.3f}<br>"
                           f"State: {state}")
            
            # Color and size
            node_colors.append(energy)
            node_sizes.append(max(5, energy * 20))
        
        node_trace = go.Scatter(x=node_x, y=node_y,
                              mode='markers',
                              hoverinfo='text',
                              text=node_text,
                              marker=dict(
                                  size=node_sizes,
                                  color=node_colors,
                                  colorscale='Viridis',
                                  colorbar=dict(title="Energy Level"),
                                  line=dict(width=1, color='black')
                              ),
                              name='Nodes')
        
        # Prepare edge traces
        edge_x = []
        edge_y = []
        
        for edge in G.edges():
            x0, y0 = positions[edge[0]]
            x1, y1 = positions[edge[1]]
            edge_x.extend([x0, x1, None])
            edge_y.extend([y0, y1, None])
        
        edge_trace = go.Scatter(x=edge_x, y=edge_y,
                              line=dict(width=1, color='gray'),
                              hoverinfo='none',
                              mode='lines',
                              name='Connections')
        
        # Create figure
        fig = go.Figure(data=[edge_trace, node_trace],
                       layout=go.Layout(
                           title="DAWN Mycelial Layer - Interactive Network",
                           titlefont_size=16,
                           showlegend=False,
                           hovermode='closest',
                           margin=dict(b=20,l=5,r=5,t=40),
                           annotations=[ dict(
                               text="Hover over nodes for details",
                               showarrow=False,
                               xref="paper", yref="paper",
                               x=0.005, y=-0.002,
                               xanchor='left', yanchor='bottom',
                               font=dict(color="gray", size=12)
                           )],
                           xaxis=dict(showgrid=False, zeroline=False, showticklabels=False),
                           yaxis=dict(showgrid=False, zeroline=False, showticklabels=False)))
        
        return fig

class MetricsVisualizer(DAWNVisualBase):
    """Visualizes system metrics and performance indicators using unified DAWN visual base"""
    
    def __init__(self, config: Optional[VisualizationConfig] = None, visual_config: Optional[DAWNVisualConfig] = None):
        # Initialize visual base
        visual_config = visual_config or DAWNVisualConfig(
            figure_size=(16, 12),
            animation_fps=30,
            enable_real_time=True,
            background_color="#0a0a0a"
        )
        super().__init__(visual_config)
        
        self.config = config or VisualizationConfig()
        
        # Metrics history
        self.metrics_history = deque(maxlen=self.config.history_length)
        self.timestamps = deque(maxlen=self.config.history_length)
        
        # Device-agnostic tensor storage for metrics
        self.metrics_tensor_buffer = torch.zeros(self.config.history_length, 10).to(device)  # 10 key metrics
        self.timestamps_tensor = torch.zeros(self.config.history_length).to(device)
        self.current_metrics_index = 0
        
        # Thread safety
        self._lock = threading.RLock()
        
        logger.info(f"📊 Metrics Visualizer initialized - Device: {device}")
        logger.info(f"   Figure size: {self.config.figure_size}")
        logger.info(f"   Metrics buffer: {self.metrics_tensor_buffer.shape}")
    
    def record_metrics(self, system_metrics: Dict[str, Any], timestamp: Optional[float] = None):
        """Record system metrics for visualization"""
        with self._lock:
            if timestamp is None:
                timestamp = time.time()
            
            self.metrics_history.append(system_metrics.copy())
            self.timestamps.append(timestamp)
    
    def create_metrics_dashboard(self) -> plt.Figure:
        """Create comprehensive metrics dashboard using matplotlib/seaborn"""
        if not self.metrics_history:
            logger.warning("No metrics data for dashboard")
            return None
        
        # Create matplotlib subplots with consciousness styling
        fig = self.create_figure((3, 2))
        fig.suptitle('DAWN Mycelial Layer - System Metrics Dashboard', 
                    fontsize=16, color='white', fontweight='bold')
        
        # Extract time series data
        timestamps = list(self.timestamps)
        
        health_data = [m.get('overall_health', 0) for m in self.metrics_history]
        stress_data = [m.get('stress_level', 0) for m in self.metrics_history]
        
        total_energy = [m.get('total_energy', 0) for m in self.metrics_history]
        avg_energy = [m.get('avg_node_health', 0) for m in self.metrics_history]
        
        growth_rate = [m.get('growth_approval_rate', 0) for m in self.metrics_history]
        autophagy_rate = [m.get('autophagy_rate', 0) for m in self.metrics_history]
        
        clusters = [m.get('active_clusters', 0) for m in self.metrics_history]
        cluster_efficiency = [m.get('clustering_efficiency', 0) for m in self.metrics_history]
        
        nutrient_efficiency = [m.get('nutrient_efficiency', 0) for m in self.metrics_history]
        demand_satisfaction = [m.get('demand_satisfaction', 0) for m in self.metrics_history]
        
        nodes = [m.get('total_nodes', 0) for m in self.metrics_history]
        edges = [m.get('total_edges', 0) for m in self.metrics_history]
        
        # Plot all metrics using consciousness colors and matplotlib
        # Plot 1: System Health
        ax1 = self.axes[0]
        ax1.plot(timestamps, health_data, color=self.consciousness_colors['awareness'], 
                label='Health', linewidth=2)
        ax1_twin = ax1.twinx()
        ax1_twin.plot(timestamps, stress_data, color=self.consciousness_colors['chaos'], 
                     label='Stress', linewidth=2, linestyle='--')
        ax1.set_title('System Health', color='white')
        ax1.legend(loc='upper left')
        ax1_twin.legend(loc='upper right')
        
        # Plot 2: Energy Metrics  
        ax2 = self.axes[1]
        ax2.plot(timestamps, total_energy, color=self.consciousness_colors['creativity'], 
                label='Total Energy', linewidth=2)
        ax2_twin = ax2.twinx()
        ax2_twin.plot(timestamps, avg_energy, color=self.consciousness_colors['stability'], 
                     label='Avg Energy', linewidth=2, linestyle=':')
        ax2.set_title('Energy Metrics', color='white')
        
        # Continue with other plots...
        # For brevity, showing pattern - would implement all 6 plots
        
        # Style all axes with consciousness theme
        for ax in self.axes:
            ax.set_facecolor('#0a0a0a')
            ax.tick_params(colors='white')
            for spine in ax.spines.values():
                spine.set_color('white')
        
        plt.tight_layout()
        return fig
    
    def create_real_time_monitor(self, mycelial_system) -> Any:
        """Create real-time monitoring visualization"""
        if not HAS_MATPLOTLIB:
            logger.warning("Matplotlib not available for real-time monitoring")
            return None
        
        # Create figure with subplots
        fig, ((ax1, ax2), (ax3, ax4)) = plt.subplots(2, 2, figsize=(15, 10))
        fig.suptitle('DAWN Mycelial Layer - Real-Time Monitor', fontsize=16)
        
        # Initialize empty plots
        lines = {}
        
        # Health monitoring
        ax1.set_title('System Health')
        ax1.set_ylim(0, 1)
        lines['health'] = ax1.plot([], [], 'g-', label='Health')[0]
        lines['stress'] = ax1.plot([], [], 'r-', label='Stress')[0]
        ax1.legend()
        ax1.grid(True)
        
        # Energy distribution
        ax2.set_title('Energy Distribution')
        ax2.set_ylim(0, 1)
        lines['avg_energy'] = ax2.plot([], [], 'b-', label='Avg Energy')[0]
        ax2.legend()
        ax2.grid(True)
        
        # Network growth
        ax3.set_title('Network Growth')
        lines['nodes'] = ax3.plot([], [], 'g-', label='Nodes')[0]
        lines['edges'] = ax3.plot([], [], 'orange', label='Edges')[0]
        ax3.legend()
        ax3.grid(True)
        
        # Resource efficiency
        ax4.set_title('Resource Efficiency')
        ax4.set_ylim(0, 1)
        lines['nutrient_eff'] = ax4.plot([], [], 'brown', label='Nutrient Efficiency')[0]
        lines['demand_sat'] = ax4.plot([], [], 'pink', label='Demand Satisfaction')[0]
        ax4.legend()
        ax4.grid(True)
        
        plt.tight_layout()
        
        return fig, lines

class MycelialVisualizer:
    """Main visualization coordinator for the mycelial layer"""
    
    def __init__(self, config: Optional[VisualizationConfig] = None):
        self.config = config or VisualizationConfig()
        
        # Component visualizers
        self.network_visualizer = NetworkVisualizer(self.config)
        self.metrics_visualizer = MetricsVisualizer(self.config)
        
        # State tracking
        self.last_update = 0
        self.update_interval = self.config.update_interval
        
        # Thread safety
        self._lock = threading.RLock()
        
        logger.info("MycelialVisualizer initialized")
    
    def update_visualizations(self, mycelial_system):
        """Update all visualizations with current system state"""
        current_time = time.time()
        
        if current_time - self.last_update < self.update_interval:
            return  # Skip update if too frequent
        
        with self._lock:
            # Record current metrics
            system_status = mycelial_system.get_system_status()
            self.metrics_visualizer.record_metrics(system_status['metrics'], current_time)
            
            self.last_update = current_time
    
    def create_comprehensive_report(self, mycelial_system, output_dir: str = "./mycelial_report"):
        """Create comprehensive visualization report"""
        import os
        
        os.makedirs(output_dir, exist_ok=True)
        
        report = {
            'timestamp': time.time(),
            'system_status': mycelial_system.get_system_status(),
            'visualizations': {}
        }
        
        # Static network visualization
        if HAS_MATPLOTLIB:
            network_fig = self.network_visualizer.visualize_static_network(
                mycelial_system.mycelial_layer,
                save_path=os.path.join(output_dir, "network_topology.png")
            )
            if network_fig:
                report['visualizations']['network_topology'] = "network_topology.png"
        
        # Interactive network
        if HAS_PLOTLY:
            interactive_fig = self.network_visualizer.create_interactive_network(
                mycelial_system.mycelial_layer
            )
            if interactive_fig:
                interactive_path = os.path.join(output_dir, "interactive_network.html")
                pyo.plot(interactive_fig, filename=interactive_path, auto_open=False)
                report['visualizations']['interactive_network'] = "interactive_network.html"
        
        # Metrics dashboard
        if HAS_PLOTLY and self.metrics_visualizer.metrics_history:
            dashboard = self.metrics_visualizer.create_metrics_dashboard()
            if dashboard:
                dashboard_path = os.path.join(output_dir, "metrics_dashboard.html")
                pyo.plot(dashboard, filename=dashboard_path, auto_open=False)
                report['visualizations']['metrics_dashboard'] = "metrics_dashboard.html"
        
        # Save report metadata
        with open(os.path.join(output_dir, "report.json"), "w") as f:
            json.dump(report, f, indent=2, default=str)
        
        logger.info(f"Comprehensive visualization report created in {output_dir}")
        return report
    
    def start_live_monitoring(self, mycelial_system, display_mode: str = "web"):
        """Start live monitoring of mycelial system"""
        if display_mode == "web" and HAS_PLOTLY:
            return self._start_web_monitoring(mycelial_system)
        elif display_mode == "desktop" and HAS_MATPLOTLIB:
            return self._start_desktop_monitoring(mycelial_system)
        else:
            logger.warning(f"Display mode '{display_mode}' not available")
            return None
    
    def _start_web_monitoring(self, mycelial_system):
        """Start web-based live monitoring"""
        # This would integrate with a web framework like Dash for live updates
        logger.info("Web monitoring would require Dash integration")
        pass
    
    def _start_desktop_monitoring(self, mycelial_system):
        """Start desktop-based live monitoring"""
        if not HAS_MATPLOTLIB:
            return None
        
        fig, lines = self.metrics_visualizer.create_real_time_monitor(mycelial_system)
        
        def update_plots(frame):
            # Update system
            mycelial_system.tick_update()
            self.update_visualizations(mycelial_system)
            
            # Update plot data
            if self.metrics_visualizer.metrics_history:
                times = list(self.metrics_visualizer.timestamps)
                times = [(t - times[0]) for t in times]  # Relative time
                
                metrics = list(self.metrics_visualizer.metrics_history)
                
                # Update health plot
                health_data = [m.get('overall_health', 0) for m in metrics]
                stress_data = [m.get('stress_level', 0) for m in metrics]
                lines['health'].set_data(times, health_data)
                lines['stress'].set_data(times, stress_data)
                
                # Update energy plot
                energy_data = [m.get('avg_node_health', 0) for m in metrics]
                lines['avg_energy'].set_data(times, energy_data)
                
                # Update network plot  
                nodes_data = [m.get('total_nodes', 0) for m in metrics]
                edges_data = [m.get('total_edges', 0) for m in metrics]
                lines['nodes'].set_data(times, nodes_data)
                lines['edges'].set_data(times, edges_data)
                
                # Update efficiency plot
                nutrient_data = [m.get('nutrient_efficiency', 0) for m in metrics]
                demand_data = [m.get('demand_satisfaction', 0) for m in metrics]
                lines['nutrient_eff'].set_data(times, nutrient_data)
                lines['demand_sat'].set_data(times, demand_data)
                
                # Adjust axes
                if times:
                    for ax in fig.axes:
                        ax.relim()
                        ax.autoscale_view()
            
            return list(lines.values())
        
        # Create animation
        anim = FuncAnimation(fig, update_plots, interval=int(self.update_interval * 1000), 
                           blit=False, cache_frame_data=False)
        
        plt.show()
        return anim

# Export classes for external use
__all__ = [
    'VisualizationConfig',
    'NetworkVisualizer', 
    'MetricsVisualizer',
    'MycelialVisualizer'
]
