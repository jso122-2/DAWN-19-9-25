#!/usr/bin/env python3
"""
DAWN Cognition Visualization #4: Entropy Flow (Matplotlib/Seaborn)
Foundation Tier - "Meeting DAWN"

Real-time vector field visualization of DAWN's information streams using the unified
matplotlib/seaborn visual base. Displays entropy flow as animated arrows showing 
the direction and intensity of information processing across cognitive space.

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
from matplotlib.patches import FancyArrowPatch
from matplotlib.collections import LineCollection
import seaborn as sns
import time
import math
import json
from collections import deque, defaultdict
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
class EntropyFlowConfig:
    """Configuration for entropy flow visualization"""
    buffer_size: int = 30
    grid_size: int = 12
    field_bounds: Tuple[float, float, float, float] = (-6, 6, -6, 6)  # x_min, x_max, y_min, y_max
    data_source: str = "stdin"
    save_frames: bool = False
    output_dir: str = './visual_output/entropy_flow'
    flow_speed_multiplier: float = 1.0
    arrow_scale: float = 20.0

@dataclass 
class StreamType:
    """Information stream type definition"""
    name: str
    color: str
    weight: float
    description: str

class EntropyFlowVisualizer(DAWNVisualBase):
    """
    DAWN Entropy Flow Visualizer using matplotlib/seaborn
    
    Renders real-time vector field visualization of information processing flows
    using device-agnostic tensor operations and consciousness-aware styling.
    """
    
    def __init__(self,
                 entropy_config: Optional[EntropyFlowConfig] = None,
                 visual_config: Optional[DAWNVisualConfig] = None):
        """
        Initialize entropy flow visualizer
        
        Args:
            entropy_config: Entropy flow specific configuration
            visual_config: Base visual configuration
        """
        # Initialize visual base
        visual_config = visual_config or DAWNVisualConfig(
            figure_size=(16, 8),
            animation_fps=30,
            enable_real_time=True,
            background_color="#0a0a0a"
        )
        super().__init__(visual_config)
        
        # Entropy flow configuration
        self.entropy_config = entropy_config or EntropyFlowConfig()
        
        # Create coordinate grids as tensors
        x = torch.linspace(self.entropy_config.field_bounds[0], 
                          self.entropy_config.field_bounds[1], 
                          self.entropy_config.grid_size).to(device)
        y = torch.linspace(self.entropy_config.field_bounds[2], 
                          self.entropy_config.field_bounds[3], 
                          self.entropy_config.grid_size).to(device)
        
        # Create meshgrid tensors
        self.X, self.Y = torch.meshgrid(x, y, indexing='xy')
        
        # Flow vectors (U, V components) as tensors
        self.U = torch.zeros_like(self.X)
        self.V = torch.zeros_like(self.Y)
        self.magnitude = torch.zeros_like(self.X)
        
        # Entropy state tracking
        self.entropy_history: torch.Tensor = torch.zeros(self.entropy_config.buffer_size).to(device)
        self.entropy_current: float = 0.5
        self.flow_phase: float = 0.0
        self.information_density = torch.zeros((self.entropy_config.grid_size, 
                                              self.entropy_config.grid_size)).to(device)
        
        # Information stream types
        self.stream_types = {
            'sensory': StreamType('Sensory', '#00aaff', 1.0, 'Input streams'),
            'memory': StreamType('Memory', '#aa00ff', 0.8, 'Memory recall'),
            'cognitive': StreamType('Cognitive', '#00ff88', 1.2, 'Active thinking'),
            'creative': StreamType('Creative', '#ffaa00', 0.9, 'Creative synthesis'),
            'output': StreamType('Output', '#ff4444', 1.1, 'Decision/action')
        }
        
        # Neural network for entropy field computation
        self.entropy_processor = self._create_entropy_processor()
        
        logger.info(f"🌊 Entropy Flow Visualizer initialized - Device: {device}")
        logger.info(f"   Grid size: {self.entropy_config.grid_size}x{self.entropy_config.grid_size}")
        logger.info(f"   Field bounds: {self.entropy_config.field_bounds}")
    
    def _create_entropy_processor(self) -> nn.Module:
        """Create neural processor for entropy field computation"""
        
        class EntropyFieldProcessor(nn.Module):
            def __init__(self, grid_size: int):
                super(EntropyFieldProcessor, self).__init__()
                self.grid_size = grid_size
                
                # Learnable parameters for entropy field dynamics
                self.flow_dynamics = nn.Linear(4, grid_size * grid_size * 2)  # Input: [entropy, phase, time, complexity]
                self.field_transformer = nn.TransformerEncoderLayer(
                    d_model=grid_size * 2,
                    nhead=4,
                    dim_feedforward=128,
                    dropout=0.1
                )
                
                # Initialize with consciousness-aware weights
                with torch.no_grad():
                    nn.init.xavier_uniform_(self.flow_dynamics.weight)
                    
            def forward(self, entropy_state: torch.Tensor) -> Tuple[torch.Tensor, torch.Tensor]:
                """
                Process entropy state to generate flow field
                
                Args:
                    entropy_state: [batch, 4] tensor with [entropy, phase, time, complexity]
                    
                Returns:
                    Tuple of (U, V) flow components [grid_size, grid_size]
                """
                batch_size = entropy_state.shape[0]
                
                # Generate flow field
                flow_field = self.flow_dynamics(entropy_state)  # [batch, grid_size*grid_size*2]
                flow_field = flow_field.view(batch_size, self.grid_size, self.grid_size, 2)
                
                # Apply transformer for temporal dynamics
                flow_reshaped = flow_field.view(batch_size, -1, 2)
                
                # Skip transformer for now to avoid dimension mismatch
                # TODO: Fix transformer dimension matching
                transformed = flow_reshaped.view(batch_size, self.grid_size, self.grid_size, 2)
                
                # Split into U and V components
                U = transformed[:, :, :, 0]
                V = transformed[:, :, :, 1]
                
                # Apply tanh activation for bounded flow
                U = torch.tanh(U)
                V = torch.tanh(V)
                
                return U, V
        
        return EntropyFieldProcessor(self.entropy_config.grid_size).to(device)
    
    def render_frame(self, consciousness_data: torch.Tensor) -> plt.Figure:
        """
        Render entropy flow frame
        
        Args:
            consciousness_data: Consciousness tensor containing entropy data
            
        Returns:
            Rendered matplotlib figure
        """
        start_time = time.time()
        
        try:
            # Process consciousness data
            entropy_data = self.process_consciousness_tensor(consciousness_data)
            
            # Create figure with two subplots
            fig = self.create_figure((1, 2))
            
            if not isinstance(self.axes, list) or len(self.axes) < 2:
                return fig
            
            ax_main, ax_metrics = self.axes[0], self.axes[1]
            
            # Update entropy field from consciousness data
            self._update_entropy_field_from_tensor(entropy_data)
            
            # Render main flow field
            self._render_flow_field(ax_main)
            
            # Render metrics panel
            self._render_metrics_panel(ax_metrics)
            
            # Add entropy indicators
            self._add_entropy_indicators()
            
            # Update performance metrics
            self.frame_count += 1
            self.update_performance_metrics(time.time() - start_time)
            
            return fig
            
        except Exception as e:
            logger.error(f"Failed to render entropy flow frame: {e}")
            return self.create_figure()
    
    def update_visualization(self, frame_num: int, consciousness_stream: Any) -> Any:
        """
        Update entropy flow visualization for animation
        
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
                consciousness_data = self._generate_simulated_entropy_data()
            
            # Ensure tensor format
            if not isinstance(consciousness_data, torch.Tensor):
                consciousness_data = torch.tensor(consciousness_data, dtype=torch.float32)
            
            # Clear previous plots
            for ax in self.axes:
                ax.clear()
            
            # Render new frame
            return self.render_frame(consciousness_data)
            
        except Exception as e:
            logger.error(f"Failed to update entropy flow visualization: {e}")
            return []
    
    def _update_entropy_field_from_tensor(self, entropy_tensor: torch.Tensor) -> None:
        """Update entropy field from consciousness tensor"""
        if entropy_tensor.numel() == 0:
            return
        
        # Extract entropy information
        if entropy_tensor.numel() >= 1:
            self.entropy_current = float(entropy_tensor[0])
        
        # Update flow phase
        self.flow_phase += 0.1
        
        # Update entropy history
        self.entropy_history = torch.roll(self.entropy_history, -1)
        self.entropy_history[-1] = self.entropy_current
        
        # Prepare state for neural processor
        current_time = time.time() % 100  # Normalize time
        complexity = torch.std(self.entropy_history).item()  # Entropy complexity
        
        entropy_state = torch.tensor([
            self.entropy_current,
            self.flow_phase, 
            current_time,
            complexity
        ], dtype=torch.float32).unsqueeze(0).to(device)  # Add batch dimension
        
        # Check for NaN values in input
        assert not torch.isnan(entropy_state).any(), 'Entropy state contains NaN values'
        
        # Generate flow field using neural processor
        with torch.no_grad():  # Inference mode
            self.entropy_processor.eval()
            U, V = self.entropy_processor(entropy_state)
            
            # Remove batch dimension and update flow components
            self.U = U.squeeze(0)
            self.V = V.squeeze(0)
            
            # Calculate magnitude
            self.magnitude = torch.sqrt(self.U**2 + self.V**2)
            
            # Update information density (based on magnitude)
            self.information_density = self.magnitude * self.entropy_current
    
    def _render_flow_field(self, ax: plt.Axes) -> None:
        """Render main vector field display"""
        # Convert tensors to numpy for matplotlib
        X_np = self.X.cpu().numpy()
        Y_np = self.Y.cpu().numpy()
        U_np = self.U.cpu().numpy()
        V_np = self.V.cpu().numpy()
        magnitude_np = self.magnitude.cpu().numpy()
        density_np = self.information_density.cpu().numpy()
        
        # Setup axes
        ax.set_xlim(self.entropy_config.field_bounds[0], self.entropy_config.field_bounds[1])
        ax.set_ylim(self.entropy_config.field_bounds[2], self.entropy_config.field_bounds[3])
        ax.set_aspect('equal')
        ax.set_facecolor(self.consciousness_colors['background'])
        
        # Create vector field quiver plot
        quiver = ax.quiver(X_np, Y_np, U_np, V_np, magnitude_np,
                          scale=self.entropy_config.arrow_scale,
                          scale_units='xy', angles='xy',
                          width=0.003, alpha=0.8, cmap='plasma')
        
        # Add information density contours
        if np.any(density_np > 0):
            contour = ax.contour(X_np, Y_np, density_np,
                               levels=8, colors='white', 
                               alpha=0.2, linewidths=0.5)
        
        # Add cognitive region labels
        self._add_cognitive_region_labels(ax)
        
        # Add stream type indicators
        self._add_stream_indicators(ax)
        
        # Styling
        ax.set_title('DAWN Information Flow Field\nEntropy Vector Visualization',
                    fontsize=14, color='white', weight='bold')
        ax.set_xlabel('Cognitive Space X', color='#cccccc', fontsize=10)
        ax.set_ylabel('Cognitive Space Y', color='#cccccc', fontsize=10)
        
        # Add colorbar for magnitude
        if hasattr(quiver, 'colorbar'):
            cbar = plt.colorbar(quiver, ax=ax, shrink=0.8, alpha=0.8)
            cbar.set_label('Flow Magnitude', color='white', fontsize=10)
            cbar.ax.yaxis.set_tick_params(color='white')
    
    def _render_metrics_panel(self, ax: plt.Axes) -> None:
        """Render entropy metrics panel"""
        ax.set_facecolor(self.consciousness_colors['background'])
        
        # Entropy history plot
        entropy_history_np = self.entropy_history.cpu().numpy()
        time_steps = np.arange(len(entropy_history_np))
        
        # Plot entropy timeline
        valid_mask = entropy_history_np != 0
        if np.any(valid_mask):
            ax.plot(time_steps[valid_mask], entropy_history_np[valid_mask],
                   color=self.consciousness_colors['flow'],
                   linewidth=2, alpha=0.8, label='Entropy Level')
            
            # Fill area
            ax.fill_between(time_steps[valid_mask], 0, entropy_history_np[valid_mask],
                          color=self.consciousness_colors['flow'], alpha=0.3)
        
        # Current entropy indicator
        if len(entropy_history_np) > 0:
            current_entropy = entropy_history_np[-1]
            ax.axhline(y=current_entropy, color=self.consciousness_colors['awareness'],
                      linestyle='--', alpha=0.7, label=f'Current: {current_entropy:.2f}')
        
        # Entropy zones
        ax.axhspan(0, 0.3, alpha=0.1, color=self.consciousness_colors['stability'], label='Low Entropy')
        ax.axhspan(0.3, 0.7, alpha=0.1, color=self.consciousness_colors['awareness'], label='Medium Entropy')
        ax.axhspan(0.7, 1.0, alpha=0.1, color=self.consciousness_colors['chaos'], label='High Entropy')
        
        # Flow statistics
        total_flow = float(torch.sum(self.magnitude).cpu())
        max_flow = float(torch.max(self.magnitude).cpu())
        
        stats_text = f"Total Flow: {total_flow:.1f}\n"
        stats_text += f"Max Flow: {max_flow:.2f}\n"
        stats_text += f"Flow Phase: {self.flow_phase:.1f}"
        
        ax.text(0.02, 0.98, stats_text, transform=ax.transAxes,
               color=self.consciousness_colors['unity'], fontsize=10,
               verticalalignment='top',
               bbox=dict(boxstyle='round,pad=0.3',
                        facecolor=self.consciousness_colors['background'],
                        alpha=0.7))
        
        # Styling
        ax.set_title('Entropy Metrics\nFlow Statistics & History',
                    fontsize=12, color='white', weight='bold')
        ax.set_xlabel('Time Steps', color='#cccccc', fontsize=10)
        ax.set_ylabel('Entropy Level', color='#cccccc', fontsize=10)
        ax.set_ylim(0, 1.0)
        ax.legend(loc='upper right', fontsize=8, framealpha=0.8)
        ax.grid(True, alpha=0.3, color='#333333')
    
    def _add_cognitive_region_labels(self, ax: plt.Axes) -> None:
        """Add cognitive region labels to flow field"""
        regions = [
            {'pos': (-4, 4), 'label': 'Sensory\nInput', 'color': self.stream_types['sensory'].color},
            {'pos': (4, 4), 'label': 'Memory\nRecall', 'color': self.stream_types['memory'].color},
            {'pos': (0, 0), 'label': 'Cognitive\nCore', 'color': self.stream_types['cognitive'].color},
            {'pos': (-4, -4), 'label': 'Creative\nSynthesis', 'color': self.stream_types['creative'].color},
            {'pos': (4, -4), 'label': 'Action\nOutput', 'color': self.stream_types['output'].color}
        ]
        
        for region in regions:
            ax.text(region['pos'][0], region['pos'][1], region['label'],
                   color=region['color'], fontsize=9, ha='center', va='center',
                   bbox=dict(boxstyle='round,pad=0.3', 
                           facecolor=self.consciousness_colors['background'],
                           alpha=0.7, edgecolor=region['color']))
    
    def _add_stream_indicators(self, ax: plt.Axes) -> None:
        """Add information stream type indicators"""
        legend_elements = []
        
        for stream_name, stream_type in self.stream_types.items():
            # Create legend entry
            from matplotlib.lines import Line2D
            legend_elements.append(
                Line2D([0], [0], color=stream_type.color, lw=3,
                      label=f"{stream_type.name} ({stream_type.weight:.1f})")
            )
        
        # Add legend for stream types
        ax.legend(handles=legend_elements, loc='upper left',
                 framealpha=0.8, fontsize=8, title="Stream Types",
                 title_fontsize=9)
    
    def _add_entropy_indicators(self) -> None:
        """Add entropy-specific consciousness indicators"""
        if self.figure is None:
            return
        
        # Entropy state indicator
        entropy_text = f"Entropy: {self.entropy_current:.3f} | "
        entropy_text += f"Flow Phase: {self.flow_phase:.1f} | "
        entropy_text += f"Frame: {self.frame_count}"
        
        self.figure.suptitle(f"DAWN Entropy Flow Monitor", 
                           color=self.consciousness_colors['unity'], fontsize=16)
        
        self.figure.text(0.02, 0.95, entropy_text, 
                        color=self.consciousness_colors['awareness'], fontsize=11)
        
        # Performance indicator
        stats = self.get_performance_stats()
        perf_text = f"Render FPS: {stats.get('fps', 0):.1f} | Device: {device}"
        self.figure.text(0.98, 0.02, perf_text, 
                        color=self.consciousness_colors['stability'], 
                        fontsize=8, ha='right')
    
    def _generate_simulated_entropy_data(self) -> torch.Tensor:
        """Generate simulated entropy data for testing"""
        t = time.time()
        
        # Simulate entropy with multiple influences
        base_entropy = 0.5 + 0.3 * math.sin(t * 0.2)
        chaos_component = 0.2 * math.sin(t * 1.5) * math.cos(t * 0.7)
        stability_component = 0.1 * math.sin(t * 0.05)
        
        entropy_value = base_entropy + chaos_component + stability_component
        entropy_value = max(0.0, min(1.0, entropy_value))  # Clamp to [0,1]
        
        # Add complexity and flow information
        flow_magnitude = 0.5 + 0.4 * math.sin(t * 0.8)
        complexity = 0.3 + 0.2 * math.sin(t * 0.3)
        
        return torch.tensor([entropy_value, flow_magnitude, complexity], 
                          dtype=torch.float32).to(device)
    
    def start_entropy_monitoring(self, data_stream: Optional[Any] = None) -> None:
        """Start real-time entropy flow monitoring"""
        if data_stream is None:
            data_stream = self._generate_simulated_entropy_data
        
        self.start_real_time_visualization(data_stream, interval=75)  # ~13 FPS
        logger.info("🌊 Entropy flow monitoring started")
    
    def stop_entropy_monitoring(self) -> None:
        """Stop entropy flow monitoring"""
        self.stop_visualization()
        logger.info("🌊 Entropy flow monitoring stopped")
    
    def get_entropy_metrics(self) -> Dict[str, float]:
        """Get current entropy flow metrics"""
        stats = self.get_performance_stats()
        
        entropy_metrics = {
            'current_entropy': self.entropy_current,
            'flow_phase': self.flow_phase,
            'total_flow_magnitude': float(torch.sum(self.magnitude).cpu()),
            'max_flow_magnitude': float(torch.max(self.magnitude).cpu()),
            'mean_information_density': float(torch.mean(self.information_density).cpu()),
            'entropy_variance': float(torch.var(self.entropy_history[self.entropy_history != 0]).cpu())
        }
        
        # Merge with base performance stats
        entropy_metrics.update(stats)
        return entropy_metrics


def create_entropy_flow_visualizer(entropy_config: Optional[EntropyFlowConfig] = None,
                                  visual_config: Optional[DAWNVisualConfig] = None) -> EntropyFlowVisualizer:
    """
    Factory function to create entropy flow visualizer
    
    Args:
        entropy_config: Entropy flow configuration
        visual_config: Visual configuration
        
    Returns:
        Configured entropy flow visualizer
    """
    return EntropyFlowVisualizer(entropy_config, visual_config)


# Command line interface
def main():
    """Main function for command line usage"""
    parser = argparse.ArgumentParser(description='DAWN Entropy Flow Visualizer')
    parser.add_argument('--grid-size', type=int, default=12,
                       help='Flow field grid size')
    parser.add_argument('--buffer-size', type=int, default=30,
                       help='Data buffer size')
    parser.add_argument('--fps', type=int, default=20,
                       help='Animation FPS')
    parser.add_argument('--save-frames', action='store_true',
                       help='Save animation frames')
    parser.add_argument('--output-dir', type=str, default='./visual_output/entropy_flow',
                       help='Output directory for saved frames')
    
    args = parser.parse_args()
    
    # Create configurations
    entropy_config = EntropyFlowConfig(
        grid_size=args.grid_size,
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
    visualizer = create_entropy_flow_visualizer(entropy_config, visual_config)
    
    print(f"🌊 Starting DAWN Entropy Flow Monitor...")
    print(f"   Grid size: {args.grid_size}x{args.grid_size}")
    print(f"   Buffer size: {args.buffer_size}")
    print(f"   FPS: {args.fps}")
    print(f"   Device: {device}")
    
    try:
        # Start monitoring
        visualizer.start_entropy_monitoring()
        
        # Show plot
        plt.show()
        
    except KeyboardInterrupt:
        print("\n⏹️ Stopping entropy flow monitor...")
    finally:
        visualizer.stop_entropy_monitoring()
        print("✅ Entropy flow monitor stopped")


if __name__ == "__main__":
    main()
