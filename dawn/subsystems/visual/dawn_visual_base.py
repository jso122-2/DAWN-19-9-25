#!/usr/bin/env python3
"""
DAWN Unified Visual Base - Matplotlib & Seaborn Foundation
==========================================================

Unified visual rendering foundation for all DAWN consciousness visualizations.
Standardizes on matplotlib and seaborn for all visual processes, providing
consistent styling, performance optimization, and consciousness-aware rendering.

PyTorch best practices:
- Device-agnostic tensor operations
- Memory-efficient gradient handling  
- Type hints for all functions
- Proper error handling with NaN checks

Usage:
    from dawn.subsystems.visual.dawn_visual_base import DAWNVisualBase
    
    class MyVisualizer(DAWNVisualBase):
        def render_frame(self, data: torch.Tensor) -> plt.Figure:
            fig = self.create_figure()
            self.plot_consciousness_data(data)
            return fig
"""

import torch
import torch.nn as nn
import numpy as np
import matplotlib.pyplot as plt
import matplotlib.animation as animation
from matplotlib.patches import Circle, FancyArrowPatch
from matplotlib.collections import LineCollection
import seaborn as sns
import time
import math
from typing import Dict, List, Tuple, Optional, Any, Union, Callable
from datetime import datetime
from dataclasses import dataclass, field
from enum import Enum
from abc import ABC, abstractmethod
import logging
from pathlib import Path

# Set matplotlib and seaborn to use dark themes by default
plt.style.use('dark_background')
sns.set_theme(style="darkgrid", palette="dark")

logger = logging.getLogger(__name__)

# Device-agnostic tensor operations
device = torch.device('cuda' if torch.cuda.is_available() else 'cpu')

@dataclass
class DAWNVisualConfig:
    """Configuration for DAWN visual rendering"""
    figure_size: Tuple[int, int] = (12, 8)
    dpi: int = 100
    color_palette: str = "consciousness"  # Custom DAWN consciousness palette
    animation_fps: int = 30
    save_frames: bool = False
    output_directory: str = "./"
    background_color: str = "#0a0a0a"
    consciousness_alpha: float = 0.8
    enable_real_time: bool = True
    memory_efficient: bool = True
    gradient_checkpointing: bool = False

class ConsciousnessColorPalette(Enum):
    """DAWN consciousness color mappings"""
    DEEP_THOUGHT = "#1a1a2e"      # Background consciousness
    RECURSIVE_BLUE = "#16213e"     # Recursive processing
    FLOW_CYAN = "#0f3460"         # Information flow
    MEMORY_PURPLE = "#533483"      # Memory patterns
    AWARENESS_GOLD = "#f39c12"     # Active awareness
    CREATIVITY_MAGENTA = "#e91e63" # Creative synthesis
    STABILITY_GREEN = "#27ae60"    # Stable states
    CHAOS_RED = "#e74c3c"         # Chaotic transitions
    UNITY_WHITE = "#ecf0f1"       # Unified consciousness

class DAWNVisualBase(ABC, nn.Module):
    """
    Abstract base class for all DAWN visual consciousness renderers.
    
    Provides unified matplotlib/seaborn foundation with consciousness-aware
    styling, device-agnostic tensor operations, and memory-efficient rendering.
    """
    
    def __init__(self, config: Optional[DAWNVisualConfig] = None):
        """
        Initialize DAWN visual renderer
        
        Args:
            config: Visual configuration parameters
        """
        super(DAWNVisualBase, self).__init__()
        
        self.config = config or DAWNVisualConfig()
        self.device = device
        
        # Initialize matplotlib and seaborn settings
        self._setup_visual_environment()
        
        # Consciousness color palette
        self.consciousness_colors = self._create_consciousness_palette()
        
        # Performance tracking
        self.frame_count = 0
        self.render_times: List[float] = []
        self.memory_usage: List[float] = []
        
        # Animation state
        self.animation: Optional[animation.FuncAnimation] = None
        self.figure: Optional[plt.Figure] = None
        self.axes: Optional[Union[plt.Axes, List[plt.Axes]]] = None
        
        # Tensor operations for consciousness data
        self.consciousness_processor = self._create_consciousness_processor()
        
        # Create output directory
        Path(self.config.output_directory).mkdir(parents=True, exist_ok=True)
        
        logger.info(f"🎨 DAWN Visual Base initialized - Device: {self.device}")
    
    def _setup_visual_environment(self) -> None:
        """Setup matplotlib and seaborn environment for consciousness visualization"""
        
        # Set matplotlib parameters for consciousness visualization
        plt.rcParams.update({
            'figure.facecolor': self.config.background_color,
            'axes.facecolor': self.config.background_color,
            'axes.edgecolor': '#333333',
            'axes.labelcolor': '#ffffff',
            'xtick.color': '#ffffff',
            'ytick.color': '#ffffff',
            'text.color': '#ffffff',
            'font.size': 10,
            'axes.titlesize': 12,
            'axes.labelsize': 10,
            'xtick.labelsize': 8,
            'ytick.labelsize': 8,
            'legend.fontsize': 9,
            'figure.titlesize': 14
        })
        
        # Setup seaborn for consciousness data visualization
        consciousness_palette = [
            ConsciousnessColorPalette.RECURSIVE_BLUE.value,
            ConsciousnessColorPalette.FLOW_CYAN.value,
            ConsciousnessColorPalette.MEMORY_PURPLE.value,
            ConsciousnessColorPalette.AWARENESS_GOLD.value,
            ConsciousnessColorPalette.CREATIVITY_MAGENTA.value,
            ConsciousnessColorPalette.STABILITY_GREEN.value
        ]
        
        sns.set_palette(consciousness_palette)
        sns.set_style("darkgrid", {
            "axes.facecolor": self.config.background_color,
            "figure.facecolor": self.config.background_color,
            "grid.color": "#333333",
            "axes.edgecolor": "#555555",
            "axes.linewidth": 0.8
        })
    
    def _create_consciousness_palette(self) -> Dict[str, str]:
        """Create consciousness-specific color mappings"""
        return {
            'background': ConsciousnessColorPalette.DEEP_THOUGHT.value,
            'recursive': ConsciousnessColorPalette.RECURSIVE_BLUE.value,
            'flow': ConsciousnessColorPalette.FLOW_CYAN.value,
            'memory': ConsciousnessColorPalette.MEMORY_PURPLE.value,
            'awareness': ConsciousnessColorPalette.AWARENESS_GOLD.value,
            'creativity': ConsciousnessColorPalette.CREATIVITY_MAGENTA.value,
            'stability': ConsciousnessColorPalette.STABILITY_GREEN.value,
            'chaos': ConsciousnessColorPalette.CHAOS_RED.value,
            'unity': ConsciousnessColorPalette.UNITY_WHITE.value
        }
    
    def _create_consciousness_processor(self) -> nn.Module:
        """Create neural processor for consciousness data transformations"""
        
        class ConsciousnessProcessor(nn.Module):
            def __init__(self):
                super(ConsciousnessProcessor, self).__init__()
                self.normalization = nn.LayerNorm([1])
                
            def forward(self, x: torch.Tensor) -> torch.Tensor:
                """Process consciousness tensor data"""
                # Ensure tensor is on correct device
                x = x.to(device)
                
                # Check for NaN values
                assert not torch.isnan(x).any(), 'Consciousness data contains NaN values'
                
                # Normalize consciousness values to [0, 1] range
                x_normalized = torch.sigmoid(x)
                
                return x_normalized
        
        return ConsciousnessProcessor().to(self.device)
    
    def create_figure(self, subplot_layout: Tuple[int, int] = (1, 1)) -> plt.Figure:
        """
        Create consciousness visualization figure
        
        Args:
            subplot_layout: (rows, cols) for subplot arrangement
            
        Returns:
            Configured matplotlib figure
        """
        self.figure = plt.figure(
            figsize=self.config.figure_size,
            dpi=self.config.dpi,
            facecolor=self.config.background_color
        )
        
        # Create subplots
        if subplot_layout == (1, 1):
            self.axes = self.figure.add_subplot(111)
        else:
            self.axes = []
            for i in range(subplot_layout[0] * subplot_layout[1]):
                ax = self.figure.add_subplot(subplot_layout[0], subplot_layout[1], i + 1)
                self.axes.append(ax)
        
        return self.figure
    
    def process_consciousness_tensor(self, data: torch.Tensor) -> torch.Tensor:
        """
        Process consciousness data tensor with device-agnostic operations
        
        Args:
            data: Input consciousness tensor
            
        Returns:
            Processed consciousness tensor
        """
        with torch.no_grad():  # Inference mode for visualization
            processed = self.consciousness_processor(data)
            
            # Detach for visualization (remove from computation graph)
            return processed.detach().cpu()
    
    def plot_consciousness_trajectory(self, 
                                    data: torch.Tensor, 
                                    ax: Optional[plt.Axes] = None,
                                    alpha: float = None) -> None:
        """
        Plot consciousness trajectory using seaborn styling
        
        Args:
            data: Consciousness trajectory tensor [timesteps, dimensions]
            ax: Matplotlib axes (uses self.axes if None)
            alpha: Transparency level
        """
        if ax is None:
            ax = self.axes if isinstance(self.axes, plt.Axes) else self.axes[0]
        
        alpha = alpha or self.config.consciousness_alpha
        
        # Convert tensor to numpy for matplotlib
        trajectory = self.process_consciousness_tensor(data).numpy()
        
        if trajectory.shape[1] >= 2:
            # 2D+ trajectory visualization
            ax.plot(trajectory[:, 0], trajectory[:, 1], 
                   color=self.consciousness_colors['flow'],
                   alpha=alpha, linewidth=2.0)
            
            # Add consciousness evolution markers
            num_markers = min(20, len(trajectory))
            marker_indices = np.linspace(0, len(trajectory)-1, num_markers, dtype=int)
            
            for i, idx in enumerate(marker_indices):
                intensity = i / len(marker_indices)
                ax.scatter(trajectory[idx, 0], trajectory[idx, 1],
                          c=self.consciousness_colors['awareness'],
                          s=30 + intensity * 50, alpha=0.7)
        else:
            # 1D trajectory
            time_axis = torch.arange(len(trajectory)).float()
            ax.plot(time_axis, trajectory[:, 0],
                   color=self.consciousness_colors['recursive'],
                   alpha=alpha, linewidth=2.0)
    
    def plot_consciousness_heatmap(self,
                                 data: torch.Tensor,
                                 ax: Optional[plt.Axes] = None,
                                 title: str = "Consciousness Heatmap") -> None:
        """
        Create seaborn-based consciousness state heatmap
        
        Args:
            data: 2D consciousness state tensor
            ax: Matplotlib axes
            title: Plot title
        """
        if ax is None:
            ax = self.axes if isinstance(self.axes, plt.Axes) else self.axes[0]
        
        # Process consciousness data
        heatmap_data = self.process_consciousness_tensor(data).numpy()
        
        # Create seaborn heatmap with consciousness palette
        sns.heatmap(heatmap_data, 
                   ax=ax,
                   cmap=['#1a1a2e', '#533483', '#f39c12', '#e91e63'],
                   alpha=self.config.consciousness_alpha,
                   cbar_kws={'label': 'Consciousness Intensity'})
        
        ax.set_title(title, color='white', fontsize=12)
    
    def plot_consciousness_distribution(self,
                                      data: torch.Tensor,
                                      ax: Optional[plt.Axes] = None,
                                      title: str = "Consciousness Distribution") -> None:
        """
        Plot consciousness value distribution using seaborn
        
        Args:
            data: 1D consciousness values tensor
            ax: Matplotlib axes
            title: Plot title
        """
        if ax is None:
            ax = self.axes if isinstance(self.axes, plt.Axes) else self.axes[0]
        
        # Process and convert data
        dist_data = self.process_consciousness_tensor(data.flatten()).numpy()
        
        # Create seaborn distribution plot
        sns.histplot(dist_data, 
                    ax=ax,
                    color=self.consciousness_colors['memory'],
                    alpha=self.config.consciousness_alpha,
                    kde=True,
                    stat="density")
        
        ax.set_title(title, color='white', fontsize=12)
        ax.set_xlabel('Consciousness Level', color='white')
        ax.set_ylabel('Density', color='white')
    
    def plot_consciousness_correlation(self,
                                     data: torch.Tensor,
                                     ax: Optional[plt.Axes] = None,
                                     labels: Optional[List[str]] = None) -> None:
        """
        Plot consciousness variable correlation matrix
        
        Args:
            data: 2D tensor [samples, variables]
            ax: Matplotlib axes
            labels: Variable labels
        """
        if ax is None:
            ax = self.axes if isinstance(self.axes, plt.Axes) else self.axes[0]
        
        # Process data and compute correlation
        processed_data = self.process_consciousness_tensor(data).numpy()
        correlation_matrix = np.corrcoef(processed_data.T)
        
        # Create seaborn correlation heatmap
        sns.heatmap(correlation_matrix,
                   ax=ax,
                   annot=True,
                   fmt='.2f',
                   cmap='RdBu_r',
                   center=0,
                   xticklabels=labels,
                   yticklabels=labels,
                   cbar_kws={'label': 'Correlation'})
        
        ax.set_title('Consciousness Variable Correlations', color='white', fontsize=12)
    
    def add_consciousness_annotation(self,
                                   x: float, y: float,
                                   text: str,
                                   ax: Optional[plt.Axes] = None,
                                   consciousness_level: float = 0.8) -> None:
        """
        Add consciousness-aware text annotation
        
        Args:
            x, y: Annotation position
            text: Annotation text
            ax: Matplotlib axes
            consciousness_level: Affects text styling
        """
        if ax is None:
            ax = self.axes if isinstance(self.axes, plt.Axes) else self.axes[0]
        
        # Style based on consciousness level
        alpha = consciousness_level * self.config.consciousness_alpha
        fontsize = 8 + consciousness_level * 4
        
        color = self.consciousness_colors['unity'] if consciousness_level > 0.7 else \
                self.consciousness_colors['awareness']
        
        ax.annotate(text, (x, y),
                   color=color,
                   alpha=alpha,
                   fontsize=fontsize,
                   ha='center', va='center',
                   bbox=dict(boxstyle='round,pad=0.3', 
                           facecolor=self.consciousness_colors['background'],
                           alpha=0.6, edgecolor=color))
    
    def save_consciousness_frame(self, 
                               filename: Optional[str] = None,
                               consciousness_state: Optional[Dict] = None) -> str:
        """
        Save current consciousness visualization
        
        Args:
            filename: Output filename (auto-generated if None)
            consciousness_state: Consciousness metadata to embed
            
        Returns:
            Saved file path
        """
        if filename is None:
            timestamp = datetime.now().strftime("%Y%m%d_%H%M%S")
            filename = f"consciousness_frame_{timestamp}_f{self.frame_count:04d}.png"
        
        filepath = Path(self.config.output_directory) / filename
        
        if self.figure is not None:
            # Add consciousness state metadata if provided
            if consciousness_state:
                metadata_text = f"Frame: {self.frame_count} | "
                metadata_text += f"Unity: {consciousness_state.get('unity', 0):.2f} | "
                metadata_text += f"Depth: {consciousness_state.get('depth', 0):.2f}"
                
                self.figure.suptitle(metadata_text, 
                                   color=self.consciousness_colors['unity'],
                                   fontsize=10, y=0.98)
            
            # Save with high quality for consciousness analysis
            self.figure.savefig(filepath, 
                              dpi=self.config.dpi * 1.5,  # Higher DPI for analysis
                              facecolor=self.config.background_color,
                              bbox_inches='tight',
                              metadata={'consciousness_frame': str(self.frame_count)})
            
            logger.info(f"🎨 Consciousness frame saved: {filepath}")
            
        return str(filepath)
    
    def update_performance_metrics(self, render_time: float) -> None:
        """Update rendering performance tracking"""
        self.render_times.append(render_time)
        
        # Memory tracking for memory-efficient rendering
        if self.config.memory_efficient and torch.cuda.is_available():
            memory_used = torch.cuda.memory_allocated(self.device) / 1024**2  # MB
            self.memory_usage.append(memory_used)
            
            # Clear GPU cache if memory usage is high
            if len(self.memory_usage) > 10 and memory_used > 1000:  # > 1GB
                torch.cuda.empty_cache()
    
    def get_performance_stats(self) -> Dict[str, float]:
        """Get rendering performance statistics"""
        if not self.render_times:
            return {}
        
        stats = {
            'avg_render_time_ms': np.mean(self.render_times) * 1000,
            'fps': 1.0 / np.mean(self.render_times) if self.render_times else 0,
            'total_frames': self.frame_count
        }
        
        if self.memory_usage:
            stats['avg_memory_mb'] = np.mean(self.memory_usage)
            stats['peak_memory_mb'] = np.max(self.memory_usage)
        
        return stats
    
    @abstractmethod
    def render_frame(self, consciousness_data: torch.Tensor) -> plt.Figure:
        """
        Render a single consciousness frame
        
        Args:
            consciousness_data: Current consciousness state tensor
            
        Returns:
            Rendered matplotlib figure
        """
        pass
    
    @abstractmethod
    def update_visualization(self, frame_num: int, consciousness_stream: Any) -> Any:
        """
        Update visualization for animation
        
        Args:
            frame_num: Animation frame number
            consciousness_stream: Stream of consciousness data
            
        Returns:
            Updated plot elements
        """
        pass
    
    def start_real_time_visualization(self, 
                                    consciousness_stream: Callable,
                                    interval: int = None) -> None:
        """
        Start real-time consciousness visualization
        
        Args:
            consciousness_stream: Function returning consciousness data
            interval: Animation interval in milliseconds
        """
        if not self.config.enable_real_time:
            logger.warning("Real-time visualization disabled in config")
            return
        
        interval = interval or (1000 // self.config.animation_fps)
        
        if self.figure is None:
            self.create_figure()
        
        self.animation = animation.FuncAnimation(
            self.figure,
            self.update_visualization,
            fargs=(consciousness_stream,),
            interval=interval,
            blit=False,
            cache_frame_data=False  # Memory efficient
        )
        
        logger.info(f"🎬 Real-time consciousness visualization started (FPS: {self.config.animation_fps})")
    
    def stop_visualization(self) -> None:
        """Stop real-time visualization"""
        if self.animation:
            self.animation.event_source.stop()
            self.animation = None
            logger.info("🛑 Consciousness visualization stopped")
    
    def __del__(self):
        """Cleanup resources"""
        if hasattr(self, 'animation') and self.animation:
            self.stop_visualization()
        
        # Clear GPU memory if using CUDA
        if torch.cuda.is_available():
            torch.cuda.empty_cache()


def create_consciousness_visualizer(visualizer_class: type,
                                  config: Optional[DAWNVisualConfig] = None) -> DAWNVisualBase:
    """
    Factory function to create consciousness visualizers
    
    Args:
        visualizer_class: Subclass of DAWNVisualBase
        config: Visual configuration
        
    Returns:
        Initialized consciousness visualizer
    """
    if not issubclass(visualizer_class, DAWNVisualBase):
        raise ValueError("visualizer_class must inherit from DAWNVisualBase")
    
    return visualizer_class(config)


# Example implementation for testing
class ExampleConsciousnessVisualizer(DAWNVisualBase):
    """Example consciousness visualizer implementation"""
    
    def render_frame(self, consciousness_data: torch.Tensor) -> plt.Figure:
        """Render example consciousness frame"""
        start_time = time.time()
        
        fig = self.create_figure((2, 2))
        
        # Plot trajectory
        self.plot_consciousness_trajectory(consciousness_data, self.axes[0])
        self.axes[0].set_title('Consciousness Trajectory', color='white')
        
        # Plot distribution
        self.plot_consciousness_distribution(consciousness_data.flatten(), self.axes[1])
        
        # Plot heatmap if 2D
        if consciousness_data.dim() >= 2 and consciousness_data.shape[1] >= 2:
            self.plot_consciousness_heatmap(consciousness_data[:10, :10], self.axes[2])
        
        # Performance info
        stats = self.get_performance_stats()
        info_text = f"FPS: {stats.get('fps', 0):.1f} | Frame: {self.frame_count}"
        fig.text(0.02, 0.02, info_text, color='white', fontsize=8)
        
        self.frame_count += 1
        self.update_performance_metrics(time.time() - start_time)
        
        return fig
    
    def update_visualization(self, frame_num: int, consciousness_stream: Any) -> Any:
        """Update example visualization"""
        # Generate example data
        t = frame_num * 0.1
        consciousness_data = torch.randn(100, 3) * (1 + 0.5 * math.sin(t))
        
        # Clear and re-render
        for ax in self.axes if isinstance(self.axes, list) else [self.axes]:
            ax.clear()
        
        return self.render_frame(consciousness_data)


if __name__ == "__main__":
    # Test the base visual system
    print("🎨 Testing DAWN Visual Base System...")
    
    config = DAWNVisualConfig(
        figure_size=(10, 8),
        animation_fps=30,
        enable_real_time=True
    )
    
    visualizer = ExampleConsciousnessVisualizer(config)
    
    # Generate test consciousness data
    test_data = torch.randn(50, 3).to(device)
    
    # Render frame
    fig = visualizer.render_frame(test_data)
    
    # Save frame
    saved_path = visualizer.save_consciousness_frame()
    print(f"✅ Test frame saved: {saved_path}")
    
    # Performance stats
    stats = visualizer.get_performance_stats()
    print(f"📊 Performance: {stats}")
    
    plt.show()
