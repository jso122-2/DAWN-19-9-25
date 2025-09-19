#!/usr/bin/env python3
"""
DAWN Cognition Visualization #3: Heat Monitor (Matplotlib/Seaborn)
Foundation Tier - "Meeting DAWN"

Real-time radial gauge visualization of DAWN's cognitive intensity using the unified
matplotlib/seaborn visual base. Displays processing heat as a dynamic speedometer 
showing the engine's cognitive load, processing speed, and mental effort across time.

PyTorch Best Practices:
- Device-agnostic tensor operations (.to(device))
- Memory-efficient gradient handling
- Type hints for all functions
- Proper error handling with NaN checks
"""

import torch
import torch.nn as nn
import numpy as np
import matplotlib.pyplot as plt
import matplotlib.animation as animation
from matplotlib.patches import Wedge, Circle, Rectangle
import seaborn as sns
import time
import math
import json
from collections import deque
from typing import Dict, List, Tuple, Optional, Any, Union
from dataclasses import dataclass
import argparse
import logging

# Import DAWN visual base
from dawn.subsystems.visual.dawn_visual_base import (
    DAWNVisualBase,
    DAWNVisualConfig,
    ConsciousnessColorPalette,
    device
)

logger = logging.getLogger(__name__)

@dataclass
class HeatZone:
    """Heat zone definition for cognitive load visualization"""
    name: str
    min_heat: float
    max_heat: float
    color: str
    description: str

@dataclass
class HeatMonitorConfig:
    """Configuration for heat monitor visualization"""
    buffer_size: int = 50
    data_source: str = "stdin"
    save_frames: bool = False
    output_dir: str = './visual_output/heat_monitor'
    gauge_radius: float = 1.0
    gauge_start_angle: float = 225  # Bottom left
    gauge_end_angle: float = 315    # Bottom right (150 degree arc)
    heat_smoothing_factor: float = 0.1

class HeatMonitorVisualizer(DAWNVisualBase):
    """
    DAWN Heat Monitor Visualizer using matplotlib/seaborn
    
    Renders real-time cognitive intensity as a speedometer-style gauge with
    heat zones, historical tracking, and performance metrics using device-agnostic
    tensor operations and consciousness-aware styling.
    """
    
    def __init__(self,
                 heat_config: Optional[HeatMonitorConfig] = None,
                 visual_config: Optional[DAWNVisualConfig] = None):
        """
        Initialize heat monitor visualizer
        
        Args:
            heat_config: Heat monitor specific configuration
            visual_config: Base visual configuration
        """
        # Initialize visual base
        visual_config = visual_config or DAWNVisualConfig(
            figure_size=(14, 10),
            animation_fps=30,
            enable_real_time=True,
            background_color="#0a0a0a"
        )
        super().__init__(visual_config)
        
        # Heat monitor configuration
        self.heat_config = heat_config or HeatMonitorConfig()
        
        # Heat state tracking (as tensors for device-agnostic operations)
        self.heat_history: torch.Tensor = torch.zeros(self.heat_config.buffer_size).to(device)
        self.heat_current: float = 0.0
        self.heat_smoothed: float = 0.0
        self.heat_peak: float = 0.0
        self.heat_average: float = 0.0
        
        # Gauge parameters
        self.gauge_center = (0, 0)
        self.gauge_range = self.heat_config.gauge_end_angle - self.heat_config.gauge_start_angle
        
        # Heat zones and colors
        self.heat_zones = [
            HeatZone("Idle", 0.0, 0.2, self.consciousness_colors['stability'], "Minimal processing"),
            HeatZone("Active", 0.2, 0.5, self.consciousness_colors['awareness'], "Normal processing"),
            HeatZone("Intense", 0.5, 0.8, self.consciousness_colors['creativity'], "High cognitive load"),
            HeatZone("Critical", 0.8, 1.0, self.consciousness_colors['chaos'], "Maximum processing")
        ]
        
        # Performance metrics
        self.processing_metrics = {
            'cpu_usage': 0.0,
            'memory_usage': 0.0,
            'operations_per_second': 0.0,
            'error_rate': 0.0
        }
        
        # Neural network for heat processing
        self.heat_processor = self._create_heat_processor()
        
        logger.info(f"🔥 Heat Monitor Visualizer initialized - Device: {device}")
        logger.info(f"   Gauge range: {self.heat_config.gauge_start_angle}° to {self.heat_config.gauge_end_angle}°")
        logger.info(f"   Heat zones: {len(self.heat_zones)}")
    
    def _create_heat_processor(self) -> nn.Module:
        """Create neural processor for heat computation"""
        
        class HeatProcessor(nn.Module):
            def __init__(self):
                super(HeatProcessor, self).__init__()
                
                # Heat dynamics network
                self.heat_dynamics = nn.Sequential(
                    nn.Linear(4, 16),  # Input: [current_heat, load, complexity, time]
                    nn.ReLU(),
                    nn.Linear(16, 8),
                    nn.ReLU(),
                    nn.Linear(8, 1),   # Output: processed heat
                    nn.Sigmoid()       # Bound to [0,1]
                )
                
                # Initialize with consciousness-aware weights
                with torch.no_grad():
                    for layer in self.heat_dynamics:
                        if isinstance(layer, nn.Linear):
                            nn.init.xavier_uniform_(layer.weight)
                            
            def forward(self, heat_state: torch.Tensor) -> torch.Tensor:
                """
                Process heat state
                
                Args:
                    heat_state: [batch, 4] tensor with [current_heat, load, complexity, time]
                    
                Returns:
                    Processed heat value [batch, 1]
                """
                # Check for NaN values
                assert not torch.isnan(heat_state).any(), 'Heat state contains NaN values'
                
                return self.heat_dynamics(heat_state)
        
        return HeatProcessor().to(device)
    
    def render_frame(self, consciousness_data: torch.Tensor) -> plt.Figure:
        """
        Render heat monitor frame
        
        Args:
            consciousness_data: Consciousness tensor containing heat data
            
        Returns:
            Rendered matplotlib figure
        """
        start_time = time.time()
        
        try:
            # Process consciousness data
            heat_data = self.process_consciousness_tensor(consciousness_data)
            
            # Create figure with subplots
            fig = self.create_figure((2, 2))
            
            if not isinstance(self.axes, list) or len(self.axes) < 4:
                return fig
            
            ax_gauge, ax_history, ax_zones, ax_metrics = self.axes
            
            # Update heat data from consciousness tensor
            self._update_heat_data_from_tensor(heat_data)
            
            # Render main gauge
            self._render_heat_gauge(ax_gauge)
            
            # Render heat history
            self._render_heat_history(ax_history)
            
            # Render heat zones
            self._render_heat_zones(ax_zones)
            
            # Render performance metrics
            self._render_performance_metrics(ax_metrics)
            
            # Add heat indicators
            self._add_heat_indicators()
            
            # Update performance metrics
            self.frame_count += 1
            self.update_performance_metrics(time.time() - start_time)
            
            return fig
            
        except Exception as e:
            logger.error(f"Failed to render heat monitor frame: {e}")
            return self.create_figure()
    
    def update_visualization(self, frame_num: int, consciousness_stream: Any) -> Any:
        """
        Update heat monitor visualization for animation
        
        Args:
            frame_num: Animation frame number
            consciousness_stream: Stream of consciousness data
            
        Returns:
            Updated plot elements
        """
        try:
            # Get consciousness data
            if callable(consciousness_stream):
                consciousness_data = consciousness_stream()
            else:
                consciousness_data = self._generate_simulated_heat_data()
            
            # Ensure tensor format
            if not isinstance(consciousness_data, torch.Tensor):
                consciousness_data = torch.tensor(consciousness_data, dtype=torch.float32)
            
            # Clear previous plots
            for ax in self.axes:
                ax.clear()
            
            # Render new frame
            return self.render_frame(consciousness_data)
            
        except Exception as e:
            logger.error(f"Failed to update heat monitor visualization: {e}")
            return []
    
    def _update_heat_data_from_tensor(self, heat_tensor: torch.Tensor) -> None:
        """Update heat data from consciousness tensor"""
        if heat_tensor.numel() == 0:
            return
        
        # Extract heat information
        if heat_tensor.numel() >= 1:
            raw_heat = float(heat_tensor[0])
            
            # Process heat using neural network
            current_time = time.time() % 100  # Normalize time
            load_estimate = raw_heat * 1.2  # Simulate processing load
            
            # Calculate complexity safely, avoiding NaN
            valid_history = self.heat_history[self.heat_history != 0]
            if len(valid_history) > 1:
                complexity = float(torch.std(valid_history).cpu())
                # Check for NaN and replace with default
                if torch.isnan(torch.tensor(complexity)):
                    complexity = 0.1
            else:
                complexity = 0.1  # Default complexity
            
            heat_state = torch.tensor([
                raw_heat, load_estimate, complexity, current_time
            ], dtype=torch.float32).unsqueeze(0).to(device)
            
            # Check for NaN values in input before processing
            if torch.isnan(heat_state).any():
                logger.warning("NaN detected in heat state, using fallback values")
                heat_state = torch.tensor([0.5, 0.6, 0.1, current_time], dtype=torch.float32).unsqueeze(0).to(device)
            
            # Process through neural network
            with torch.no_grad():
                self.heat_processor.eval()
                processed_heat = self.heat_processor(heat_state)
                self.heat_current = float(processed_heat.squeeze())
            
            # Apply smoothing
            alpha = self.heat_config.heat_smoothing_factor
            self.heat_smoothed = alpha * self.heat_current + (1 - alpha) * self.heat_smoothed
            
            # Update statistics
            self.heat_peak = max(self.heat_peak, self.heat_current)
            
            # Update history
            self.heat_history = torch.roll(self.heat_history, -1)
            self.heat_history[-1] = self.heat_current
            
            # Calculate average
            valid_history = self.heat_history[self.heat_history != 0]
            if len(valid_history) > 0:
                self.heat_average = float(torch.mean(valid_history).cpu())
            
            # Update performance metrics (simulated)
            self._update_performance_metrics_simulation()
    
    def _update_performance_metrics_simulation(self) -> None:
        """Update simulated performance metrics based on heat"""
        # Simulate realistic performance metrics based on heat level
        base_cpu = 20 + self.heat_current * 60  # 20-80% CPU based on heat
        base_memory = 30 + self.heat_current * 40  # 30-70% memory
        
        # Add some noise
        noise_factor = 0.1
        self.processing_metrics.update({
            'cpu_usage': base_cpu + np.random.uniform(-5, 5),
            'memory_usage': base_memory + np.random.uniform(-3, 3),
            'operations_per_second': (1000 + self.heat_current * 2000) * (1 + np.random.uniform(-noise_factor, noise_factor)),
            'error_rate': max(0, (self.heat_current - 0.7) * 10 + np.random.uniform(-1, 1))
        })
    
    def _render_heat_gauge(self, ax: plt.Axes) -> None:
        """Render main heat gauge"""
        ax.set_xlim(-1.5, 1.5)
        ax.set_ylim(-1.5, 1.5)
        ax.set_aspect('equal')
        ax.set_facecolor(self.consciousness_colors['background'])
        
        # Draw gauge background
        self._draw_gauge_background(ax)
        
        # Draw heat zones on gauge
        self._draw_gauge_zones(ax)
        
        # Draw gauge needle
        self._draw_gauge_needle(ax)
        
        # Draw gauge center
        center_circle = Circle(self.gauge_center, 0.05, 
                             color=self.consciousness_colors['unity'], 
                             zorder=10)
        ax.add_patch(center_circle)
        
        # Add gauge labels
        self._add_gauge_labels(ax)
        
        # Current heat display
        heat_text = f"{self.heat_current:.1%}"
        ax.text(0, -0.3, heat_text, ha='center', va='center',
               fontsize=24, fontweight='bold',
               color=self._get_heat_color(self.heat_current))
        
        # Heat zone indicator
        current_zone = self._get_current_heat_zone()
        zone_text = f"Zone: {current_zone.name}"
        ax.text(0, -0.5, zone_text, ha='center', va='center',
               fontsize=12, color=current_zone.color)
        
        ax.set_title('Cognitive Heat Gauge\nProcessing Intensity Monitor', 
                    fontsize=14, color='white', weight='bold', pad=20)
        ax.axis('off')
    
    def _render_heat_history(self, ax: plt.Axes) -> None:
        """Render heat history graph"""
        ax.set_facecolor(self.consciousness_colors['background'])
        
        # Convert tensor to numpy
        history_data = self.heat_history.cpu().numpy()
        time_steps = np.arange(len(history_data))
        
        # Plot heat history
        valid_mask = history_data != 0
        if np.any(valid_mask):
            ax.plot(time_steps[valid_mask], history_data[valid_mask],
                   color=self.consciousness_colors['flow'],
                   linewidth=2, alpha=0.8, label='Heat Level')
            
            # Fill area
            ax.fill_between(time_steps[valid_mask], 0, history_data[valid_mask],
                          color=self.consciousness_colors['flow'], alpha=0.3)
        
        # Heat zones as background
        for zone in self.heat_zones:
            ax.axhspan(zone.min_heat, zone.max_heat, alpha=0.1, color=zone.color)
        
        # Average line
        if self.heat_average > 0:
            ax.axhline(y=self.heat_average, color=self.consciousness_colors['awareness'],
                      linestyle='--', alpha=0.7, label=f'Average: {self.heat_average:.2f}')
        
        # Peak line
        if self.heat_peak > 0:
            ax.axhline(y=self.heat_peak, color=self.consciousness_colors['chaos'],
                      linestyle=':', alpha=0.7, label=f'Peak: {self.heat_peak:.2f}')
        
        # Styling
        ax.set_title('Heat History\nTemporal Cognitive Load', 
                    fontsize=12, color='white', weight='bold')
        ax.set_xlabel('Time Steps', color='#cccccc', fontsize=10)
        ax.set_ylabel('Heat Level', color='#cccccc', fontsize=10)
        ax.set_ylim(0, 1.0)
        ax.legend(loc='upper right', fontsize=8, framealpha=0.8)
        ax.grid(True, alpha=0.3, color='#333333')
    
    def _render_heat_zones(self, ax: plt.Axes) -> None:
        """Render heat zones information"""
        ax.set_facecolor(self.consciousness_colors['background'])
        
        # Create horizontal bar chart for zones
        zone_names = [zone.name for zone in self.heat_zones]
        zone_ranges = [zone.max_heat - zone.min_heat for zone in self.heat_zones]
        zone_colors = [zone.color for zone in self.heat_zones]
        
        y_positions = np.arange(len(zone_names))
        
        # Draw zone bars
        bars = ax.barh(y_positions, zone_ranges, 
                      color=zone_colors, alpha=0.7,
                      edgecolor='white', linewidth=1)
        
        # Add zone descriptions
        for i, zone in enumerate(self.heat_zones):
            ax.text(zone_ranges[i] / 2, i, zone.description,
                   ha='center', va='center', fontsize=9,
                   color='white', weight='bold')
        
        # Current heat indicator
        current_zone_idx = self._get_current_heat_zone_index()
        if current_zone_idx is not None:
            # Highlight current zone
            bars[current_zone_idx].set_alpha(1.0)
            bars[current_zone_idx].set_edgecolor(self.consciousness_colors['unity'])
            bars[current_zone_idx].set_linewidth(3)
        
        # Styling
        ax.set_yticks(y_positions)
        ax.set_yticklabels(zone_names, color='white', fontsize=10)
        ax.set_xlabel('Zone Range', color='#cccccc', fontsize=10)
        ax.set_title('Heat Zones\nCognitive Load Categories', 
                    fontsize=12, color='white', weight='bold')
        ax.set_xlim(0, 1.0)
        ax.grid(axis='x', alpha=0.3, color='#333333')
    
    def _render_performance_metrics(self, ax: plt.Axes) -> None:
        """Render performance metrics panel"""
        ax.set_facecolor(self.consciousness_colors['background'])
        
        # Create metrics display
        metrics_text = "Performance Metrics\n\n"
        metrics_text += f"CPU Usage: {self.processing_metrics['cpu_usage']:.1f}%\n"
        metrics_text += f"Memory: {self.processing_metrics['memory_usage']:.1f}%\n"
        metrics_text += f"Ops/sec: {self.processing_metrics['operations_per_second']:.0f}\n"
        metrics_text += f"Error Rate: {self.processing_metrics['error_rate']:.2f}%\n\n"
        
        metrics_text += f"Heat Statistics\n"
        metrics_text += f"Current: {self.heat_current:.3f}\n"
        metrics_text += f"Smoothed: {self.heat_smoothed:.3f}\n"
        metrics_text += f"Average: {self.heat_average:.3f}\n"
        metrics_text += f"Peak: {self.heat_peak:.3f}"
        
        ax.text(0.05, 0.95, metrics_text, transform=ax.transAxes,
               color=self.consciousness_colors['unity'], fontsize=10,
               verticalalignment='top', fontfamily='monospace',
               bbox=dict(boxstyle='round,pad=0.5',
                        facecolor=self.consciousness_colors['background'],
                        alpha=0.8, edgecolor=self.consciousness_colors['awareness']))
        
        # Performance gauge (mini speedometer)
        self._draw_mini_performance_gauge(ax)
        
        ax.set_title('System Metrics\nPerformance & Statistics', 
                    fontsize=12, color='white', weight='bold')
        ax.axis('off')
    
    def _draw_gauge_background(self, ax: plt.Axes) -> None:
        """Draw gauge background arc"""
        # Main gauge arc
        gauge_arc = Wedge(self.gauge_center, self.heat_config.gauge_radius, 
                         self.heat_config.gauge_start_angle, self.heat_config.gauge_end_angle,
                         width=0.1, facecolor='#333333', edgecolor='white', alpha=0.3)
        ax.add_patch(gauge_arc)
    
    def _draw_gauge_zones(self, ax: plt.Axes) -> None:
        """Draw colored zones on gauge"""
        for zone in self.heat_zones:
            # Calculate angles for this zone
            zone_start_angle = (self.heat_config.gauge_start_angle + 
                              zone.min_heat * self.gauge_range)
            zone_end_angle = (self.heat_config.gauge_start_angle + 
                            zone.max_heat * self.gauge_range)
            
            # Draw zone arc
            zone_arc = Wedge(self.gauge_center, self.heat_config.gauge_radius,
                           zone_start_angle, zone_end_angle,
                           width=0.08, facecolor=zone.color, alpha=0.6)
            ax.add_patch(zone_arc)
    
    def _draw_gauge_needle(self, ax: plt.Axes) -> None:
        """Draw gauge needle pointing to current heat level"""
        # Calculate needle angle
        needle_angle = (self.heat_config.gauge_start_angle + 
                       self.heat_smoothed * self.gauge_range)
        needle_angle_rad = math.radians(needle_angle)
        
        # Needle end point
        needle_length = self.heat_config.gauge_radius * 0.8
        needle_x = needle_length * math.cos(needle_angle_rad)
        needle_y = needle_length * math.sin(needle_angle_rad)
        
        # Draw needle
        ax.plot([0, needle_x], [0, needle_y], 
               color=self._get_heat_color(self.heat_current),
               linewidth=4, alpha=0.9, zorder=5)
        
        # Needle tip
        tip_circle = Circle((needle_x, needle_y), 0.03,
                          color=self._get_heat_color(self.heat_current),
                          zorder=6)
        ax.add_patch(tip_circle)
    
    def _draw_mini_performance_gauge(self, ax: plt.Axes) -> None:
        """Draw mini performance gauge"""
        # Position in bottom right of metrics panel
        center_x, center_y = 0.75, 0.25
        radius = 0.15
        
        # CPU usage gauge
        cpu_pct = self.processing_metrics['cpu_usage'] / 100.0
        cpu_angle = 180 * cpu_pct  # 0-180 degrees
        
        # Background arc
        bg_arc = Wedge((center_x, center_y), radius, 0, 180,
                      width=0.03, facecolor='#333333', alpha=0.5,
                      transform=ax.transAxes)
        ax.add_patch(bg_arc)
        
        # CPU arc
        cpu_arc = Wedge((center_x, center_y), radius, 0, cpu_angle,
                       width=0.025, facecolor=self.consciousness_colors['creativity'],
                       alpha=0.8, transform=ax.transAxes)
        ax.add_patch(cpu_arc)
        
        # Label
        ax.text(center_x, center_y - 0.08, 'CPU', transform=ax.transAxes,
               ha='center', va='center', fontsize=8, color='white')
    
    def _add_gauge_labels(self, ax: plt.Axes) -> None:
        """Add labels around gauge"""
        # Heat level labels
        for i in range(0, 11, 2):  # 0, 20, 40, 60, 80, 100
            heat_level = i / 10.0
            angle = self.heat_config.gauge_start_angle + heat_level * self.gauge_range
            angle_rad = math.radians(angle)
            
            # Label position (outside gauge)
            label_radius = self.heat_config.gauge_radius * 1.15
            label_x = label_radius * math.cos(angle_rad)
            label_y = label_radius * math.sin(angle_rad)
            
            # Draw tick mark
            tick_start_radius = self.heat_config.gauge_radius * 0.95
            tick_start_x = tick_start_radius * math.cos(angle_rad)
            tick_start_y = tick_start_radius * math.sin(angle_rad)
            
            ax.plot([tick_start_x, label_x * 0.9], [tick_start_y, label_y * 0.9],
                   color='white', linewidth=1, alpha=0.7)
            
            # Label text
            ax.text(label_x, label_y, f'{int(heat_level * 100)}%',
                   ha='center', va='center', fontsize=8, color='white')
    
    def _get_heat_color(self, heat_level: float) -> str:
        """Get color for heat level"""
        current_zone = self._get_current_heat_zone()
        return current_zone.color
    
    def _get_current_heat_zone(self) -> HeatZone:
        """Get current heat zone"""
        for zone in self.heat_zones:
            if zone.min_heat <= self.heat_current <= zone.max_heat:
                return zone
        return self.heat_zones[-1]  # Default to highest zone
    
    def _get_current_heat_zone_index(self) -> Optional[int]:
        """Get current heat zone index"""
        for i, zone in enumerate(self.heat_zones):
            if zone.min_heat <= self.heat_current <= zone.max_heat:
                return i
        return None
    
    def _add_heat_indicators(self) -> None:
        """Add heat-specific consciousness indicators"""
        if self.figure is None:
            return
        
        # Heat state indicator
        current_zone = self._get_current_heat_zone()
        heat_text = f"Heat: {self.heat_current:.1%} ({current_zone.name}) | "
        heat_text += f"CPU: {self.processing_metrics['cpu_usage']:.1f}% | "
        heat_text += f"Frame: {self.frame_count}"
        
        self.figure.suptitle(f"DAWN Cognitive Heat Monitor", 
                           color=self.consciousness_colors['unity'], fontsize=16)
        
        self.figure.text(0.02, 0.95, heat_text, 
                        color=self.consciousness_colors['awareness'], fontsize=11)
        
        # Performance indicator
        stats = self.get_performance_stats()
        perf_text = f"Render FPS: {stats.get('fps', 0):.1f} | Device: {device}"
        self.figure.text(0.98, 0.02, perf_text, 
                        color=self.consciousness_colors['stability'], 
                        fontsize=8, ha='right')
    
    def _generate_simulated_heat_data(self) -> torch.Tensor:
        """Generate simulated heat data for testing"""
        t = time.time()
        
        # Simulate processing heat with various patterns
        base_heat = 0.3 + 0.4 * math.sin(t * 0.3)  # Slow oscillation
        spike_heat = 0.3 * max(0, math.sin(t * 2.0))  # Periodic spikes
        load_heat = 0.2 * (1 + math.sin(t * 0.1))  # Background load
        
        total_heat = base_heat + spike_heat + load_heat
        total_heat = max(0.0, min(1.0, total_heat))  # Clamp to [0,1]
        
        # Add processing complexity
        complexity = 0.5 + 0.3 * math.sin(t * 0.7)
        
        return torch.tensor([total_heat, complexity], dtype=torch.float32).to(device)
    
    def start_heat_monitoring(self, data_stream: Optional[Any] = None) -> None:
        """Start real-time heat monitoring"""
        if data_stream is None:
            data_stream = self._generate_simulated_heat_data
        
        self.start_real_time_visualization(data_stream, interval=100)  # 10 FPS
        logger.info("🔥 Heat monitoring started")
    
    def stop_heat_monitoring(self) -> None:
        """Stop heat monitoring"""
        self.stop_visualization()
        logger.info("🔥 Heat monitoring stopped")
    
    def _get_current_heat_zone(self) -> Optional[HeatZone]:
        """Get the current heat zone based on current heat level"""
        for zone in self.heat_zones:
            if zone.min_heat <= self.heat_current <= zone.max_heat:
                return zone
        return None  # No zone found
    
    def _get_current_heat_zone_index(self) -> Optional[int]:
        """Get the index of the current heat zone"""
        current_zone = self._get_current_heat_zone()
        if current_zone is None:
            return None
        return next(i for i, zone in enumerate(self.heat_zones) if zone.name == current_zone.name)
    
    def get_heat_metrics(self) -> Dict[str, float]:
        """Get current heat monitor metrics"""
        stats = self.get_performance_stats()
        
        heat_metrics = {
            'current_heat': self.heat_current,
            'smoothed_heat': self.heat_smoothed,
            'peak_heat': self.heat_peak,
            'average_heat': self.heat_average,
            'current_zone': self._get_current_heat_zone_index() or 0,
            'heat_variance': float(torch.var(self.heat_history[self.heat_history != 0]).cpu()) if torch.any(self.heat_history != 0) else 0.0
        }
        
        # Add performance metrics
        heat_metrics.update(self.processing_metrics)
        
        # Merge with base performance stats
        heat_metrics.update(stats)
        return heat_metrics


def create_heat_monitor_visualizer(heat_config: Optional[HeatMonitorConfig] = None,
                                  visual_config: Optional[DAWNVisualConfig] = None) -> HeatMonitorVisualizer:
    """
    Factory function to create heat monitor visualizer
    
    Args:
        heat_config: Heat monitor configuration
        visual_config: Visual configuration
        
    Returns:
        Configured heat monitor visualizer
    """
    return HeatMonitorVisualizer(heat_config, visual_config)


# Command line interface
def main():
    """Main function for command line usage"""
    parser = argparse.ArgumentParser(description='DAWN Heat Monitor Visualizer')
    parser.add_argument('--buffer-size', type=int, default=50,
                       help='Data buffer size')
    parser.add_argument('--fps', type=int, default=10,
                       help='Animation FPS')
    parser.add_argument('--save-frames', action='store_true',
                       help='Save animation frames')
    parser.add_argument('--output-dir', type=str, default='./visual_output/heat_monitor',
                       help='Output directory for saved frames')
    
    args = parser.parse_args()
    
    # Create configurations
    heat_config = HeatMonitorConfig(
        buffer_size=args.buffer_size,
        save_frames=args.save_frames,
        output_dir=args.output_dir
    )
    
    visual_config = DAWNVisualConfig(
        animation_fps=args.fps,
        enable_real_time=True,
        save_frames=args.save_frames,
        output_directory=args.output_dir
    )
    
    # Create visualizer
    visualizer = create_heat_monitor_visualizer(heat_config, visual_config)
    
    print(f"🔥 Starting DAWN Heat Monitor...")
    print(f"   Buffer size: {args.buffer_size}")
    print(f"   FPS: {args.fps}")
    print(f"   Device: {device}")
    
    try:
        # Start monitoring
        visualizer.start_heat_monitoring()
        
        # Show plot
        plt.show()
        
    except KeyboardInterrupt:
        print("\n⏹️ Stopping heat monitor...")
    finally:
        visualizer.stop_heat_monitoring()
        print("✅ Heat monitor stopped")


if __name__ == "__main__":
    main()
