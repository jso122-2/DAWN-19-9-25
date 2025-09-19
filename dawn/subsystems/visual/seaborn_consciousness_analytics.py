#!/usr/bin/env python3
"""
DAWN Seaborn Consciousness Analytics
===================================

Advanced statistical visualization components for DAWN consciousness analysis using
seaborn and matplotlib. Provides comprehensive mood tracking, state transition 
analysis, correlation studies, and temporal pattern recognition.

PyTorch Best Practices:
- Device-agnostic tensor operations (.to(device))
- Memory-efficient gradient handling
- Type hints for all functions
- Proper error handling with NaN checks
"""

import torch
import numpy as np
import pandas as pd
import matplotlib.pyplot as plt
import seaborn as sns
import time
from typing import Dict, List, Tuple, Optional, Any, Union
from dataclasses import dataclass
from collections import deque, defaultdict
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
class ConsciousnessAnalyticsConfig:
    """Configuration for consciousness analytics visualization"""
    
    # Statistical analysis settings
    window_size: int = 500
    correlation_threshold: float = 0.3
    transition_smoothing: float = 0.1
    
    # Visualization settings
    palette_style: str = "consciousness"  # Custom consciousness palette
    figure_style: str = "darkgrid"
    context: str = "notebook"
    
    # Analysis modes
    enable_mood_tracking: bool = True
    enable_state_transitions: bool = True
    enable_correlation_analysis: bool = True
    enable_temporal_patterns: bool = True
    
    # Output settings
    save_plots: bool = True
    output_format: str = "png"
    dpi: int = 150


class ConsciousnessAnalytics(DAWNVisualBase):
    """
    Advanced statistical visualization for consciousness analysis using seaborn
    
    Provides comprehensive analysis of:
    - Mood states and emotional patterns
    - State transitions and phase dynamics
    - Cross-dimensional correlations
    - Temporal consciousness patterns
    - Statistical distributions and outliers
    """
    
    def __init__(self, 
                 config: Optional[ConsciousnessAnalyticsConfig] = None,
                 visual_config: Optional[DAWNVisualConfig] = None):
        
        # Initialize visual base
        visual_config = visual_config or DAWNVisualConfig(
            figure_size=(16, 12),
            animation_fps=30,
            enable_real_time=False,  # Analytics is typically batch-based
            background_color="#0f0f23"
        )
        super().__init__(visual_config)
        
        self.analytics_config = config or ConsciousnessAnalyticsConfig()
        
        # Setup seaborn with consciousness theme
        self._setup_seaborn_style()
        
        # Data storage (device-agnostic tensors)
        buffer_size = self.analytics_config.window_size
        self.consciousness_data = torch.zeros(buffer_size, 4).to(device)  # SCUP
        self.mood_data = torch.zeros(buffer_size, 6).to(device)  # 6 mood dimensions
        self.temporal_data = torch.zeros(buffer_size).to(device)  # timestamps
        self.state_labels = []  # String labels for states
        
        # Analysis storage
        self.correlation_matrices = []
        self.transition_matrices = []
        self.mood_distributions = []
        
        # Data tracking
        self.current_index = 0
        self.buffer_full = False
        
        logger.info(f"📊 Consciousness Analytics initialized - Device: {device}")
        logger.info(f"   Buffer size: {buffer_size}")
        logger.info(f"   Analysis modes: {self._get_enabled_modes()}")
    
    def _setup_seaborn_style(self) -> None:
        """Setup seaborn with consciousness-aware styling"""
        # Set seaborn context and style
        sns.set_context(self.analytics_config.context)
        sns.set_style(self.analytics_config.figure_style)
        
        # Create custom consciousness color palette
        consciousness_palette = [
            self.consciousness_colors['stability'],    # Deep blue
            self.consciousness_colors['awareness'],    # Gold
            self.consciousness_colors['creativity'],   # Orange
            self.consciousness_colors['processing'],   # Green
            self.consciousness_colors['chaos'],        # Red
            self.consciousness_colors['transcendence'] # Purple
        ]
        
        # Register the custom palette
        sns.set_palette(consciousness_palette)
        
        # Set matplotlib parameters for dark theme
        plt.rcParams.update({
            'figure.facecolor': '#0f0f23',
            'axes.facecolor': '#1a1a2e',
            'axes.edgecolor': 'white',
            'axes.labelcolor': 'white',
            'text.color': 'white',
            'xtick.color': 'white',
            'ytick.color': 'white',
            'grid.color': '#333366',
            'grid.alpha': 0.3
        })
    
    def _get_enabled_modes(self) -> List[str]:
        """Get list of enabled analysis modes"""
        modes = []
        if self.analytics_config.enable_mood_tracking:
            modes.append("mood_tracking")
        if self.analytics_config.enable_state_transitions:
            modes.append("state_transitions")
        if self.analytics_config.enable_correlation_analysis:
            modes.append("correlation_analysis")
        if self.analytics_config.enable_temporal_patterns:
            modes.append("temporal_patterns")
        return modes
    
    def add_consciousness_data(self, 
                             scup_values: torch.Tensor,
                             mood_values: torch.Tensor,
                             state_label: str,
                             timestamp: Optional[float] = None) -> None:
        """
        Add consciousness data point for analysis
        
        Args:
            scup_values: Tensor [4] with Schema, Coherence, Utility, Pressure
            mood_values: Tensor [6] with mood dimensions (serenity, curiosity, etc.)
            state_label: String label for current consciousness state
            timestamp: Optional timestamp (uses current time if None)
        """
        # Ensure tensors are on correct device and properly shaped
        scup_values = scup_values.to(device).flatten()[:4]
        mood_values = mood_values.to(device).flatten()[:6]
        
        # Pad if necessary
        if scup_values.numel() < 4:
            padded_scup = torch.zeros(4).to(device)
            padded_scup[:scup_values.numel()] = scup_values
            scup_values = padded_scup
            
        if mood_values.numel() < 6:
            padded_mood = torch.zeros(6).to(device)
            padded_mood[:mood_values.numel()] = mood_values
            mood_values = padded_mood
        
        # Check for NaN values
        if torch.isnan(scup_values).any() or torch.isnan(mood_values).any():
            logger.warning("NaN values detected in consciousness data, skipping")
            return
        
        # Store data
        self.consciousness_data[self.current_index] = scup_values
        self.mood_data[self.current_index] = mood_values
        self.temporal_data[self.current_index] = timestamp or time.time()
        
        # Handle state labels (keep in sync with circular buffer)
        if len(self.state_labels) <= self.current_index:
            self.state_labels.extend(['unknown'] * (self.current_index - len(self.state_labels) + 1))
        self.state_labels[self.current_index] = state_label
        
        # Update index
        self.current_index = (self.current_index + 1) % self.analytics_config.window_size
        if self.current_index == 0:
            self.buffer_full = True
    
    def get_data_df(self, window_size: Optional[int] = None) -> pd.DataFrame:
        """
        Get current data as a pandas DataFrame for seaborn plotting
        
        Args:
            window_size: Number of recent samples to include (None for all available)
            
        Returns:
            DataFrame with consciousness and mood data
        """
        if not self.buffer_full and self.current_index == 0:
            return pd.DataFrame()
        
        # Determine data range
        if window_size is None:
            window_size = self.analytics_config.window_size
        
        if self.buffer_full:
            # Get last window_size entries
            start_idx = (self.current_index - window_size) % self.analytics_config.window_size
            if start_idx + window_size <= self.analytics_config.window_size:
                consciousness_slice = self.consciousness_data[start_idx:start_idx + window_size]
                mood_slice = self.mood_data[start_idx:start_idx + window_size]
                temporal_slice = self.temporal_data[start_idx:start_idx + window_size]
                labels_slice = self.state_labels[start_idx:start_idx + window_size]
            else:
                # Handle wrap-around
                part1_cons = self.consciousness_data[start_idx:]
                part2_cons = self.consciousness_data[:window_size - len(part1_cons)]
                consciousness_slice = torch.cat([part1_cons, part2_cons], dim=0)
                
                part1_mood = self.mood_data[start_idx:]
                part2_mood = self.mood_data[:window_size - len(part1_mood)]
                mood_slice = torch.cat([part1_mood, part2_mood], dim=0)
                
                part1_temporal = self.temporal_data[start_idx:]
                part2_temporal = self.temporal_data[:window_size - len(part1_temporal)]
                temporal_slice = torch.cat([part1_temporal, part2_temporal], dim=0)
                
                labels_slice = (self.state_labels[start_idx:] + 
                               self.state_labels[:window_size - len(self.state_labels[start_idx:])])
        else:
            # Buffer not full yet
            end_idx = min(self.current_index, window_size)
            consciousness_slice = self.consciousness_data[:end_idx]
            mood_slice = self.mood_data[:end_idx]
            temporal_slice = self.temporal_data[:end_idx]
            labels_slice = self.state_labels[:end_idx]
        
        # Convert to numpy for pandas
        consciousness_np = consciousness_slice.cpu().numpy()
        mood_np = mood_slice.cpu().numpy()
        temporal_np = temporal_slice.cpu().numpy()
        
        # Create DataFrame
        data = {
            'timestamp': temporal_np,
            'schema': consciousness_np[:, 0],
            'coherence': consciousness_np[:, 1],
            'utility': consciousness_np[:, 2],
            'pressure': consciousness_np[:, 3],
            'serenity': mood_np[:, 0],
            'curiosity': mood_np[:, 1],
            'creativity': mood_np[:, 2],
            'focus': mood_np[:, 3],
            'energy': mood_np[:, 4],
            'harmony': mood_np[:, 5],
            'state': labels_slice
        }
        
        return pd.DataFrame(data)
    
    def plot_mood_distributions(self, save_path: Optional[str] = None) -> plt.Figure:
        """
        Create seaborn distribution plots for mood dimensions
        
        Args:
            save_path: Optional path to save the plot
            
        Returns:
            matplotlib Figure object
        """
        df = self.get_data_df()
        if df.empty:
            logger.warning("No data available for mood distribution plot")
            return None
        
        # Create figure with subplots
        fig, axes = plt.subplots(2, 3, figsize=(18, 12))
        fig.suptitle('DAWN Consciousness - Mood Distributions', 
                    fontsize=16, color='white', fontweight='bold')
        
        mood_dimensions = ['serenity', 'curiosity', 'creativity', 'focus', 'energy', 'harmony']
        
        for i, mood in enumerate(mood_dimensions):
            ax = axes[i // 3, i % 3]
            
            # Create distribution plot with consciousness styling
            sns.histplot(data=df, x=mood, hue='state', multiple="stack", 
                        ax=ax, alpha=0.7, bins=20)
            
            # Overlay KDE plot
            sns.kdeplot(data=df, x=mood, ax=ax, color='white', alpha=0.8, linewidth=2)
            
            ax.set_title(f'{mood.capitalize()} Distribution', color='white', fontweight='bold')
            ax.set_xlabel(mood.capitalize(), color='white')
            ax.set_ylabel('Frequency', color='white')
            
            # Style the plot
            ax.set_facecolor('#1a1a2e')
            ax.tick_params(colors='white')
            for spine in ax.spines.values():
                spine.set_color('white')
            
            # Style legend
            legend = ax.get_legend()
            if legend:
                legend.set_frame_on(True)
                legend.get_frame().set_facecolor('#0f0f23')
                legend.get_frame().set_alpha(0.8)
                for text in legend.get_texts():
                    text.set_color('white')
        
        plt.tight_layout()
        
        if save_path and self.analytics_config.save_plots:
            plt.savefig(save_path, dpi=self.analytics_config.dpi, 
                       facecolor='#0f0f23', bbox_inches='tight')
            logger.info(f"Mood distributions saved to {save_path}")
        
        return fig
    
    def plot_state_transitions(self, save_path: Optional[str] = None) -> plt.Figure:
        """
        Create state transition heatmap using seaborn
        
        Args:
            save_path: Optional path to save the plot
            
        Returns:
            matplotlib Figure object
        """
        df = self.get_data_df()
        if df.empty or len(df) < 2:
            logger.warning("Insufficient data for state transition analysis")
            return None
        
        # Calculate transition matrix
        states = df['state'].unique()
        transition_matrix = pd.DataFrame(0, index=states, columns=states)
        
        for i in range(len(df) - 1):
            current_state = df.iloc[i]['state']
            next_state = df.iloc[i + 1]['state']
            transition_matrix.loc[current_state, next_state] += 1
        
        # Normalize to probabilities
        transition_matrix = transition_matrix.div(transition_matrix.sum(axis=1), axis=0).fillna(0)
        
        # Create heatmap
        fig, ax = plt.subplots(figsize=(12, 10))
        
        sns.heatmap(transition_matrix, annot=True, cmap='viridis', 
                   square=True, ax=ax, cbar_kws={'label': 'Transition Probability'})
        
        ax.set_title('DAWN Consciousness - State Transition Matrix', 
                    color='white', fontsize=16, fontweight='bold', pad=20)
        ax.set_xlabel('Next State', color='white', fontweight='bold')
        ax.set_ylabel('Current State', color='white', fontweight='bold')
        
        # Style the plot
        ax.tick_params(colors='white')
        for spine in ax.spines.values():
            spine.set_color('white')
        
        # Style colorbar
        cbar = ax.collections[0].colorbar
        cbar.ax.yaxis.label.set_color('white')
        cbar.ax.tick_params(colors='white')
        
        plt.tight_layout()
        
        if save_path and self.analytics_config.save_plots:
            plt.savefig(save_path, dpi=self.analytics_config.dpi, 
                       facecolor='#0f0f23', bbox_inches='tight')
            logger.info(f"State transitions saved to {save_path}")
        
        return fig
    
    def plot_correlation_matrix(self, save_path: Optional[str] = None) -> plt.Figure:
        """
        Create correlation matrix heatmap for all consciousness dimensions
        
        Args:
            save_path: Optional path to save the plot
            
        Returns:
            matplotlib Figure object
        """
        df = self.get_data_df()
        if df.empty:
            logger.warning("No data available for correlation analysis")
            return None
        
        # Select numeric columns for correlation
        numeric_cols = ['schema', 'coherence', 'utility', 'pressure', 
                       'serenity', 'curiosity', 'creativity', 'focus', 'energy', 'harmony']
        
        correlation_data = df[numeric_cols]
        correlation_matrix = correlation_data.corr()
        
        # Create correlation heatmap
        fig, ax = plt.subplots(figsize=(14, 12))
        
        # Create a mask for the upper triangle
        mask = np.triu(np.ones_like(correlation_matrix, dtype=bool))
        
        sns.heatmap(correlation_matrix, mask=mask, annot=True, cmap='RdBu_r', 
                   center=0, square=True, ax=ax, 
                   cbar_kws={'label': 'Correlation Coefficient'})
        
        ax.set_title('DAWN Consciousness - Correlation Matrix', 
                    color='white', fontsize=16, fontweight='bold', pad=20)
        
        # Style the plot
        ax.tick_params(colors='white')
        for spine in ax.spines.values():
            spine.set_color('white')
        
        # Style colorbar
        cbar = ax.collections[0].colorbar
        cbar.ax.yaxis.label.set_color('white')
        cbar.ax.tick_params(colors='white')
        
        plt.tight_layout()
        
        if save_path and self.analytics_config.save_plots:
            plt.savefig(save_path, dpi=self.analytics_config.dpi, 
                       facecolor='#0f0f23', bbox_inches='tight')
            logger.info(f"Correlation matrix saved to {save_path}")
        
        return fig
    
    def plot_temporal_patterns(self, save_path: Optional[str] = None) -> plt.Figure:
        """
        Create temporal pattern analysis using seaborn
        
        Args:
            save_path: Optional path to save the plot
            
        Returns:
            matplotlib Figure object
        """
        df = self.get_data_df()
        if df.empty:
            logger.warning("No data available for temporal pattern analysis")
            return None
        
        # Convert timestamp to relative time
        df['relative_time'] = (df['timestamp'] - df['timestamp'].min()) / 60  # minutes
        df['time_bin'] = pd.cut(df['relative_time'], bins=20, labels=False)
        
        # Create figure with multiple temporal views
        fig, axes = plt.subplots(2, 2, figsize=(16, 12))
        fig.suptitle('DAWN Consciousness - Temporal Patterns', 
                    fontsize=16, color='white', fontweight='bold')
        
        # 1. SCUP over time
        ax1 = axes[0, 0]
        scup_melted = df[['relative_time', 'schema', 'coherence', 'utility', 'pressure']].melt(
            id_vars=['relative_time'], var_name='dimension', value_name='value')
        sns.lineplot(data=scup_melted, x='relative_time', y='value', 
                    hue='dimension', ax=ax1, linewidth=2)
        ax1.set_title('SCUP Dimensions Over Time', color='white', fontweight='bold')
        ax1.set_xlabel('Time (minutes)', color='white')
        ax1.set_ylabel('SCUP Value', color='white')
        
        # 2. Mood over time
        ax2 = axes[0, 1]
        mood_melted = df[['relative_time', 'serenity', 'curiosity', 'creativity']].melt(
            id_vars=['relative_time'], var_name='mood', value_name='value')
        sns.lineplot(data=mood_melted, x='relative_time', y='value', 
                    hue='mood', ax=ax2, linewidth=2)
        ax2.set_title('Primary Moods Over Time', color='white', fontweight='bold')
        ax2.set_xlabel('Time (minutes)', color='white')
        ax2.set_ylabel('Mood Value', color='white')
        
        # 3. State distribution over time bins
        ax3 = axes[1, 0]
        if 'time_bin' in df.columns and not df['time_bin'].isna().all():
            state_time_crosstab = pd.crosstab(df['time_bin'], df['state'], normalize='index')
            sns.heatmap(state_time_crosstab.T, ax=ax3, cmap='viridis', 
                       cbar_kws={'label': 'Proportion'})
            ax3.set_title('State Distribution Over Time', color='white', fontweight='bold')
            ax3.set_xlabel('Time Bin', color='white')
            ax3.set_ylabel('State', color='white')
        
        # 4. Consciousness trajectory in 2D
        ax4 = axes[1, 1]
        sns.scatterplot(data=df, x='schema', y='coherence', hue='state', 
                       alpha=0.7, s=50, ax=ax4)
        ax4.set_title('Consciousness Trajectory (Schema vs Coherence)', 
                     color='white', fontweight='bold')
        ax4.set_xlabel('Schema', color='white')
        ax4.set_ylabel('Coherence', color='white')
        
        # Style all axes
        for ax in axes.flat:
            ax.set_facecolor('#1a1a2e')
            ax.tick_params(colors='white')
            for spine in ax.spines.values():
                spine.set_color('white')
            
            # Style legends
            legend = ax.get_legend()
            if legend:
                legend.set_frame_on(True)
                legend.get_frame().set_facecolor('#0f0f23')
                legend.get_frame().set_alpha(0.8)
                for text in legend.get_texts():
                    text.set_color('white')
        
        plt.tight_layout()
        
        if save_path and self.analytics_config.save_plots:
            plt.savefig(save_path, dpi=self.analytics_config.dpi, 
                       facecolor='#0f0f23', bbox_inches='tight')
            logger.info(f"Temporal patterns saved to {save_path}")
        
        return fig
    
    def create_comprehensive_report(self, output_dir: str = "./consciousness_analytics") -> Dict[str, str]:
        """
        Create comprehensive consciousness analytics report
        
        Args:
            output_dir: Directory to save all visualizations
            
        Returns:
            Dictionary mapping plot names to saved file paths
        """
        import os
        os.makedirs(output_dir, exist_ok=True)
        
        saved_files = {}
        
        # Generate all enabled visualizations
        if self.analytics_config.enable_mood_tracking:
            mood_path = os.path.join(output_dir, f"mood_distributions.{self.analytics_config.output_format}")
            fig = self.plot_mood_distributions(mood_path)
            if fig:
                saved_files['mood_distributions'] = mood_path
                plt.close(fig)
        
        if self.analytics_config.enable_state_transitions:
            transition_path = os.path.join(output_dir, f"state_transitions.{self.analytics_config.output_format}")
            fig = self.plot_state_transitions(transition_path)
            if fig:
                saved_files['state_transitions'] = transition_path
                plt.close(fig)
        
        if self.analytics_config.enable_correlation_analysis:
            correlation_path = os.path.join(output_dir, f"correlation_matrix.{self.analytics_config.output_format}")
            fig = self.plot_correlation_matrix(correlation_path)
            if fig:
                saved_files['correlation_matrix'] = correlation_path
                plt.close(fig)
        
        if self.analytics_config.enable_temporal_patterns:
            temporal_path = os.path.join(output_dir, f"temporal_patterns.{self.analytics_config.output_format}")
            fig = self.plot_temporal_patterns(temporal_path)
            if fig:
                saved_files['temporal_patterns'] = temporal_path
                plt.close(fig)
        
        logger.info(f"Consciousness analytics report generated: {len(saved_files)} visualizations saved to {output_dir}")
        
        return saved_files
    
    def get_analytics_summary(self) -> Dict[str, Any]:
        """Get summary statistics for current consciousness data"""
        df = self.get_data_df()
        if df.empty:
            return {}
        
        numeric_cols = ['schema', 'coherence', 'utility', 'pressure', 
                       'serenity', 'curiosity', 'creativity', 'focus', 'energy', 'harmony']
        
        summary = {
            'data_points': len(df),
            'time_span_minutes': (df['timestamp'].max() - df['timestamp'].min()) / 60 if len(df) > 1 else 0,
            'unique_states': df['state'].nunique(),
            'state_distribution': df['state'].value_counts().to_dict(),
            'mean_values': df[numeric_cols].mean().to_dict(),
            'std_values': df[numeric_cols].std().to_dict(),
            'correlation_summary': {
                'highest_positive': df[numeric_cols].corr().unstack().nlargest(20).to_dict(),
                'highest_negative': df[numeric_cols].corr().unstack().nsmallest(20).to_dict()
            }
        }
        
        return summary


def create_consciousness_analytics(config: Optional[ConsciousnessAnalyticsConfig] = None,
                                 visual_config: Optional[DAWNVisualConfig] = None) -> ConsciousnessAnalytics:
    """
    Factory function to create consciousness analytics visualizer
    
    Args:
        config: Analytics configuration
        visual_config: Visual configuration
        
    Returns:
        Configured ConsciousnessAnalytics instance
    """
    return ConsciousnessAnalytics(config, visual_config)


if __name__ == "__main__":
    """Example usage of consciousness analytics"""
    
    # Create analytics instance
    analytics = create_consciousness_analytics()
    
    # Simulate some consciousness data
    np.random.seed(42)
    torch.manual_seed(42)
    
    states = ['focused', 'creative', 'meditative', 'analytical', 'exploratory']
    
    for i in range(200):
        # Generate synthetic SCUP data
        scup = torch.rand(4)
        
        # Generate synthetic mood data
        mood = torch.rand(6)
        
        # Random state
        state = np.random.choice(states)
        
        # Add to analytics
        analytics.add_consciousness_data(scup, mood, state)
    
    # Generate comprehensive report
    print("🧠 Generating consciousness analytics report...")
    saved_files = analytics.create_comprehensive_report()
    
    print(f"📊 Analytics complete! Generated {len(saved_files)} visualizations:")
    for name, path in saved_files.items():
        print(f"   - {name}: {path}")
    
    # Get summary statistics
    summary = analytics.get_analytics_summary()
    print(f"📈 Data summary: {summary['data_points']} points over {summary['time_span_minutes']:.1f} minutes")
    print(f"🏷️ States analyzed: {list(summary['state_distribution'].keys())}")
