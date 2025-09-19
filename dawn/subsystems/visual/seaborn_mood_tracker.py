#!/usr/bin/env python3
"""
DAWN Seaborn Mood Tracker
=========================

Specialized mood tracking and emotional state analysis using seaborn statistical
visualizations. Focuses on emotional patterns, mood stability, and affective
consciousness dynamics.

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
from scipy import stats
from sklearn.decomposition import PCA
from sklearn.cluster import KMeans
import time
from typing import Dict, List, Tuple, Optional, Any, Union
from dataclasses import dataclass
from collections import deque
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
class MoodTrackerConfig:
    """Configuration for mood tracking visualization"""
    
    # Mood dimensions
    mood_dimensions: List[str] = None
    emotion_categories: List[str] = None
    
    # Analysis settings
    stability_window: int = 50
    cluster_count: int = 5
    pca_components: int = 3
    
    # Visualization settings
    violin_inner: str = "quart"  # "box", "quart", "point", "stick"
    swarm_size: float = 4.0
    pair_plot_kind: str = "scatter"
    
    # Statistical settings
    confidence_level: float = 0.95
    outlier_threshold: float = 2.5  # standard deviations
    
    def __post_init__(self):
        if self.mood_dimensions is None:
            self.mood_dimensions = [
                'serenity', 'curiosity', 'creativity', 'focus', 
                'energy', 'harmony', 'compassion', 'wonder'
            ]
        
        if self.emotion_categories is None:
            self.emotion_categories = [
                'contemplative', 'excited', 'peaceful', 'analytical',
                'inspired', 'focused', 'joyful', 'transcendent'
            ]


class MoodTracker(DAWNVisualBase):
    """
    Advanced mood tracking and emotional analysis using seaborn
    
    Provides specialized visualizations for:
    - Mood dimension distributions and patterns
    - Emotional state clustering and classification
    - Mood stability and volatility analysis
    - Cross-dimensional emotional correlations
    - Temporal mood evolution patterns
    """
    
    def __init__(self,
                 config: Optional[MoodTrackerConfig] = None,
                 visual_config: Optional[DAWNVisualConfig] = None):
        
        # Initialize visual base
        visual_config = visual_config or DAWNVisualConfig(
            figure_size=(16, 12),
            animation_fps=30,
            enable_real_time=False,
            background_color="#0f0f23"
        )
        super().__init__(visual_config)
        
        self.mood_config = config or MoodTrackerConfig()
        
        # Setup seaborn for mood visualization
        self._setup_mood_style()
        
        # Data storage (device-agnostic tensors)
        buffer_size = 1000
        n_moods = len(self.mood_config.mood_dimensions)
        
        self.mood_tensor = torch.zeros(buffer_size, n_moods).to(device)
        self.timestamp_tensor = torch.zeros(buffer_size).to(device)
        self.emotion_labels = []  # String labels for emotional states
        self.stability_scores = torch.zeros(buffer_size).to(device)
        
        # Analysis results
        self.mood_clusters = None
        self.pca_model = None
        self.mood_correlations = None
        
        # Data tracking
        self.current_index = 0
        self.buffer_full = False
        
        logger.info(f"😊 Mood Tracker initialized - Device: {device}")
        logger.info(f"   Mood dimensions: {len(self.mood_config.mood_dimensions)}")
        logger.info(f"   Buffer size: {buffer_size}")
    
    def _setup_mood_style(self) -> None:
        """Setup seaborn styling optimized for mood visualization"""
        # Create warm, emotional color palette
        mood_palette = [
            "#FF6B6B",  # Warm red - energy/passion
            "#4ECDC4",  # Teal - serenity/calm  
            "#45B7D1",  # Blue - focus/clarity
            "#96CEB4",  # Green - growth/harmony
            "#FFEAA7",  # Yellow - joy/wonder
            "#DDA0DD",  # Purple - creativity/transcendence
            "#FFB347",  # Orange - curiosity/enthusiasm
            "#F0E68C"   # Light yellow - compassion/warmth
        ]
        
        sns.set_palette(mood_palette)
        
        # Configure matplotlib for emotional visualization
        plt.rcParams.update({
            'figure.facecolor': '#0f0f23',
            'axes.facecolor': '#1a1a2e', 
            'grid.color': '#444466',
            'grid.alpha': 0.4,
            'text.color': 'white',
            'axes.labelcolor': 'white',
            'xtick.color': 'white',
            'ytick.color': 'white'
        })
    
    def add_mood_data(self,
                     mood_values: torch.Tensor,
                     emotion_label: str,
                     timestamp: Optional[float] = None) -> None:
        """
        Add mood data point for tracking
        
        Args:
            mood_values: Tensor with mood dimension values
            emotion_label: Emotional state label
            timestamp: Optional timestamp (uses current time if None)
        """
        # Ensure tensor is on correct device and properly sized
        mood_values = mood_values.to(device).flatten()
        n_expected = len(self.mood_config.mood_dimensions)
        
        if mood_values.numel() < n_expected:
            # Pad with neutral values (0.5)
            padded = torch.full((n_expected,), 0.5).to(device)
            padded[:mood_values.numel()] = mood_values
            mood_values = padded
        elif mood_values.numel() > n_expected:
            mood_values = mood_values[:n_expected]
        
        # Check for NaN values
        if torch.isnan(mood_values).any():
            logger.warning("NaN values in mood data, using neutral values")
            mood_values = torch.full((n_expected,), 0.5).to(device)
        
        # Calculate mood stability (variance from recent history)
        if self.current_index > 0:
            # Get recent mood data for stability calculation
            window_size = min(self.mood_config.stability_window, self.current_index)
            if self.buffer_full:
                start_idx = (self.current_index - window_size) % len(self.mood_tensor)
                if start_idx + window_size <= len(self.mood_tensor):
                    recent_moods = self.mood_tensor[start_idx:start_idx + window_size]
                else:
                    # Handle wraparound
                    part1 = self.mood_tensor[start_idx:]
                    part2 = self.mood_tensor[:window_size - len(part1)]
                    recent_moods = torch.cat([part1, part2], dim=0)
            else:
                recent_moods = self.mood_tensor[max(0, self.current_index - window_size):self.current_index]
            
            # Calculate stability as inverse of variance
            mood_variance = torch.var(recent_moods, dim=0).mean()
            stability = 1.0 / (1.0 + mood_variance)
        else:
            stability = 1.0  # Perfect stability for first data point
        
        # Store data
        self.mood_tensor[self.current_index] = mood_values
        self.timestamp_tensor[self.current_index] = timestamp or time.time()
        self.stability_scores[self.current_index] = stability
        
        # Handle emotion labels
        if len(self.emotion_labels) <= self.current_index:
            self.emotion_labels.extend(['neutral'] * (self.current_index - len(self.emotion_labels) + 1))
        self.emotion_labels[self.current_index] = emotion_label
        
        # Update index
        self.current_index = (self.current_index + 1) % len(self.mood_tensor)
        if self.current_index == 0:
            self.buffer_full = True
    
    def get_mood_df(self, window_size: Optional[int] = None) -> pd.DataFrame:
        """Get mood data as pandas DataFrame for seaborn plotting"""
        if not self.buffer_full and self.current_index == 0:
            return pd.DataFrame()
        
        # Determine data range
        if window_size is None:
            window_size = len(self.mood_tensor) if self.buffer_full else self.current_index
        
        # Extract data
        if self.buffer_full:
            start_idx = (self.current_index - window_size) % len(self.mood_tensor)
            if start_idx + window_size <= len(self.mood_tensor):
                mood_slice = self.mood_tensor[start_idx:start_idx + window_size]
                timestamp_slice = self.timestamp_tensor[start_idx:start_idx + window_size]
                stability_slice = self.stability_scores[start_idx:start_idx + window_size]
                labels_slice = self.emotion_labels[start_idx:start_idx + window_size]
            else:
                # Handle wraparound
                mood_part1 = self.mood_tensor[start_idx:]
                mood_part2 = self.mood_tensor[:window_size - len(mood_part1)]
                mood_slice = torch.cat([mood_part1, mood_part2], dim=0)
                
                time_part1 = self.timestamp_tensor[start_idx:]
                time_part2 = self.timestamp_tensor[:window_size - len(time_part1)]
                timestamp_slice = torch.cat([time_part1, time_part2], dim=0)
                
                stab_part1 = self.stability_scores[start_idx:]
                stab_part2 = self.stability_scores[:window_size - len(stab_part1)]
                stability_slice = torch.cat([stab_part1, stab_part2], dim=0)
                
                labels_slice = (self.emotion_labels[start_idx:] +
                               self.emotion_labels[:window_size - len(self.emotion_labels[start_idx:])])
        else:
            end_idx = min(self.current_index, window_size)
            mood_slice = self.mood_tensor[:end_idx]
            timestamp_slice = self.timestamp_tensor[:end_idx]
            stability_slice = self.stability_scores[:end_idx]
            labels_slice = self.emotion_labels[:end_idx]
        
        # Convert to numpy
        mood_np = mood_slice.cpu().numpy()
        timestamp_np = timestamp_slice.cpu().numpy()
        stability_np = stability_slice.cpu().numpy()
        
        # Create DataFrame
        data = {
            'timestamp': timestamp_np,
            'emotion': labels_slice,
            'stability': stability_np
        }
        
        # Add mood dimensions
        for i, mood_dim in enumerate(self.mood_config.mood_dimensions):
            data[mood_dim] = mood_np[:, i]
        
        return pd.DataFrame(data)
    
    def plot_mood_violin(self, save_path: Optional[str] = None) -> plt.Figure:
        """Create violin plots for mood dimensions by emotional state"""
        df = self.get_mood_df()
        if df.empty:
            logger.warning("No data for mood violin plot")
            return None
        
        # Melt DataFrame for seaborn
        mood_cols = self.mood_config.mood_dimensions
        melted = df[['emotion'] + mood_cols].melt(
            id_vars=['emotion'], var_name='mood_dimension', value_name='value')
        
        # Create violin plot
        fig, ax = plt.subplots(figsize=(16, 10))
        
        sns.violinplot(data=melted, x='mood_dimension', y='value', hue='emotion',
                      inner=self.mood_config.violin_inner, ax=ax)
        
        ax.set_title('DAWN Mood Distributions by Emotional State', 
                    color='white', fontsize=16, fontweight='bold')
        ax.set_xlabel('Mood Dimension', color='white', fontweight='bold')
        ax.set_ylabel('Mood Value', color='white', fontweight='bold')
        
        # Rotate x-axis labels for readability
        plt.xticks(rotation=45, ha='right')
        
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
        
        if save_path:
            plt.savefig(save_path, dpi=150, facecolor='#0f0f23', bbox_inches='tight')
            logger.info(f"Mood violin plot saved to {save_path}")
        
        return fig
    
    def plot_mood_stability_analysis(self, save_path: Optional[str] = None) -> plt.Figure:
        """Create mood stability analysis visualization"""
        df = self.get_mood_df()
        if df.empty:
            logger.warning("No data for stability analysis")
            return None
        
        # Calculate relative time
        df['relative_time'] = (df['timestamp'] - df['timestamp'].min()) / 60  # minutes
        
        # Create figure with subplots
        fig, axes = plt.subplots(2, 2, figsize=(16, 12))
        fig.suptitle('DAWN Mood Stability Analysis', 
                    fontsize=16, color='white', fontweight='bold')
        
        # 1. Stability over time
        ax1 = axes[0, 0]
        sns.lineplot(data=df, x='relative_time', y='stability', hue='emotion', 
                    ax=ax1, linewidth=2)
        ax1.set_title('Mood Stability Over Time', color='white', fontweight='bold')
        ax1.set_xlabel('Time (minutes)', color='white')
        ax1.set_ylabel('Stability Score', color='white')
        
        # 2. Stability distribution by emotion
        ax2 = axes[0, 1]
        sns.boxplot(data=df, x='emotion', y='stability', ax=ax2)
        ax2.set_title('Stability Distribution by Emotion', color='white', fontweight='bold')
        ax2.set_xlabel('Emotional State', color='white')
        ax2.set_ylabel('Stability Score', color='white')
        plt.setp(ax2.xaxis.get_majorticklabels(), rotation=45, ha='right')
        
        # 3. Mood variance heatmap
        ax3 = axes[1, 0]
        mood_vars = df[self.mood_config.mood_dimensions].var()
        mood_var_df = pd.DataFrame({'dimension': mood_vars.index, 'variance': mood_vars.values})
        sns.barplot(data=mood_var_df, x='dimension', y='variance', ax=ax3)
        ax3.set_title('Mood Dimension Variance', color='white', fontweight='bold')
        ax3.set_xlabel('Mood Dimension', color='white')
        ax3.set_ylabel('Variance', color='white')
        plt.setp(ax3.xaxis.get_majorticklabels(), rotation=45, ha='right')
        
        # 4. Stability vs average mood
        ax4 = axes[1, 1]
        df['avg_mood'] = df[self.mood_config.mood_dimensions].mean(axis=1)
        sns.scatterplot(data=df, x='avg_mood', y='stability', hue='emotion', 
                       alpha=0.7, s=50, ax=ax4)
        ax4.set_title('Stability vs Average Mood', color='white', fontweight='bold')
        ax4.set_xlabel('Average Mood', color='white')
        ax4.set_ylabel('Stability Score', color='white')
        
        # Style all axes
        for ax in axes.flat:
            ax.set_facecolor('#1a1a2e')
            ax.tick_params(colors='white')
            for spine in ax.spines.values():
                spine.set_color('white')
            
            legend = ax.get_legend()
            if legend:
                legend.set_frame_on(True)
                legend.get_frame().set_facecolor('#0f0f23')
                legend.get_frame().set_alpha(0.8)
                for text in legend.get_texts():
                    text.set_color('white')
        
        plt.tight_layout()
        
        if save_path:
            plt.savefig(save_path, dpi=150, facecolor='#0f0f23', bbox_inches='tight')
            logger.info(f"Stability analysis saved to {save_path}")
        
        return fig
    
    def plot_mood_clustering(self, save_path: Optional[str] = None) -> plt.Figure:
        """Create mood clustering analysis using PCA and K-means"""
        df = self.get_mood_df()
        if df.empty or len(df) < self.mood_config.cluster_count:
            logger.warning("Insufficient data for mood clustering")
            return None
        
        # Prepare data for clustering
        mood_data = df[self.mood_config.mood_dimensions].values
        
        # Apply PCA
        pca = PCA(n_components=self.mood_config.pca_components)
        mood_pca = pca.fit_transform(mood_data)
        
        # Apply K-means clustering
        kmeans = KMeans(n_clusters=self.mood_config.cluster_count, random_state=42)
        clusters = kmeans.fit_predict(mood_data)
        
        # Store models for later use
        self.pca_model = pca
        self.mood_clusters = kmeans
        
        # Create figure
        fig, axes = plt.subplots(2, 2, figsize=(16, 12))
        fig.suptitle('DAWN Mood Clustering Analysis', 
                    fontsize=16, color='white', fontweight='bold')
        
        # 1. PCA scatter plot (first 2 components)
        ax1 = axes[0, 0]
        scatter = ax1.scatter(mood_pca[:, 0], mood_pca[:, 1], c=clusters, 
                             cmap='viridis', alpha=0.7, s=50)
        ax1.set_title('PCA Mood Clusters (PC1 vs PC2)', color='white', fontweight='bold')
        ax1.set_xlabel(f'PC1 ({pca.explained_variance_ratio_[0]:.1%} variance)', color='white')
        ax1.set_ylabel(f'PC2 ({pca.explained_variance_ratio_[1]:.1%} variance)', color='white')
        plt.colorbar(scatter, ax=ax1)
        
        # 2. Cluster sizes
        ax2 = axes[0, 1]
        cluster_counts = pd.Series(clusters).value_counts().sort_index()
        sns.barplot(x=cluster_counts.index, y=cluster_counts.values, ax=ax2)
        ax2.set_title('Mood Cluster Sizes', color='white', fontweight='bold')
        ax2.set_xlabel('Cluster ID', color='white')
        ax2.set_ylabel('Number of Points', color='white')
        
        # 3. PCA component importance
        ax3 = axes[1, 0]
        component_importance = pd.DataFrame(
            pca.components_[:2].T,
            columns=['PC1', 'PC2'],
            index=self.mood_config.mood_dimensions
        )
        sns.heatmap(component_importance, annot=True, cmap='RdBu_r', center=0, ax=ax3)
        ax3.set_title('PCA Component Loadings', color='white', fontweight='bold')
        ax3.set_xlabel('Principal Component', color='white')
        ax3.set_ylabel('Mood Dimension', color='white')
        
        # 4. Cluster characteristics
        ax4 = axes[1, 1]
        cluster_means = pd.DataFrame(mood_data).groupby(clusters).mean()
        cluster_means.columns = self.mood_config.mood_dimensions
        sns.heatmap(cluster_means, annot=True, cmap='viridis', ax=ax4)
        ax4.set_title('Cluster Mood Profiles', color='white', fontweight='bold')
        ax4.set_xlabel('Mood Dimension', color='white')
        ax4.set_ylabel('Cluster ID', color='white')
        
        # Style all axes
        for ax in axes.flat:
            ax.set_facecolor('#1a1a2e')
            ax.tick_params(colors='white')
            for spine in ax.spines.values():
                spine.set_color('white')
        
        plt.tight_layout()
        
        if save_path:
            plt.savefig(save_path, dpi=150, facecolor='#0f0f23', bbox_inches='tight')
            logger.info(f"Mood clustering saved to {save_path}")
        
        return fig
    
    def plot_mood_correlations(self, save_path: Optional[str] = None) -> plt.Figure:
        """Create detailed mood correlation analysis"""
        df = self.get_mood_df()
        if df.empty:
            logger.warning("No data for correlation analysis")
            return None
        
        # Calculate correlations
        mood_corr = df[self.mood_config.mood_dimensions].corr()
        self.mood_correlations = mood_corr
        
        # Create figure with correlation analysis
        fig, axes = plt.subplots(2, 2, figsize=(16, 12))
        fig.suptitle('DAWN Mood Correlation Analysis', 
                    fontsize=16, color='white', fontweight='bold')
        
        # 1. Full correlation matrix
        ax1 = axes[0, 0]
        mask = np.triu(np.ones_like(mood_corr, dtype=bool))
        sns.heatmap(mood_corr, mask=mask, annot=True, cmap='RdBu_r', center=0,
                   square=True, ax=ax1, cbar_kws={'label': 'Correlation'})
        ax1.set_title('Mood Correlation Matrix', color='white', fontweight='bold')
        
        # 2. Strongest correlations
        ax2 = axes[0, 1]
        # Get upper triangle correlations
        corr_pairs = []
        for i in range(len(mood_corr.columns)):
            for j in range(i+1, len(mood_corr.columns)):
                corr_pairs.append({
                    'pair': f"{mood_corr.columns[i]}-{mood_corr.columns[j]}",
                    'correlation': mood_corr.iloc[i, j]
                })
        
        corr_df = pd.DataFrame(corr_pairs).sort_values('correlation', key=abs, ascending=False)
        top_corr = corr_df.head(10)
        
        sns.barplot(data=top_corr, y='pair', x='correlation', ax=ax2, orient='h')
        ax2.set_title('Strongest Mood Correlations', color='white', fontweight='bold')
        ax2.set_xlabel('Correlation Coefficient', color='white')
        ax2.set_ylabel('Mood Pair', color='white')
        
        # 3. Mood pairplot (subset for readability)
        if len(self.mood_config.mood_dimensions) > 4:
            subset_dims = self.mood_config.mood_dimensions[:4]
        else:
            subset_dims = self.mood_config.mood_dimensions
        
        # Create a small pairplot in the subplot
        mood_subset = df[subset_dims + ['emotion']]
        
        # Since we can't easily embed a full pairplot, show scatter of first two dimensions
        ax3 = axes[1, 0]
        sns.scatterplot(data=mood_subset, x=subset_dims[0], y=subset_dims[1], 
                       hue='emotion', alpha=0.7, s=50, ax=ax3)
        ax3.set_title(f'{subset_dims[0]} vs {subset_dims[1]}', color='white', fontweight='bold')
        ax3.set_xlabel(subset_dims[0], color='white')
        ax3.set_ylabel(subset_dims[1], color='white')
        
        # 4. Correlation strength distribution
        ax4 = axes[1, 1]
        corr_values = mood_corr.values[np.triu_indices_from(mood_corr.values, k=1)]
        sns.histplot(corr_values, bins=20, ax=ax4, alpha=0.7)
        ax4.axvline(x=0, color='white', linestyle='--', alpha=0.7)
        ax4.set_title('Distribution of Correlations', color='white', fontweight='bold')
        ax4.set_xlabel('Correlation Coefficient', color='white')
        ax4.set_ylabel('Frequency', color='white')
        
        # Style all axes
        for ax in axes.flat:
            ax.set_facecolor('#1a1a2e')
            ax.tick_params(colors='white')
            for spine in ax.spines.values():
                spine.set_color('white')
            
            legend = ax.get_legend()
            if legend:
                legend.set_frame_on(True)
                legend.get_frame().set_facecolor('#0f0f23')
                legend.get_frame().set_alpha(0.8)
                for text in legend.get_texts():
                    text.set_color('white')
        
        plt.tight_layout()
        
        if save_path:
            plt.savefig(save_path, dpi=150, facecolor='#0f0f23', bbox_inches='tight')
            logger.info(f"Mood correlations saved to {save_path}")
        
        return fig
    
    def generate_mood_report(self, output_dir: str = "./mood_analysis") -> Dict[str, str]:
        """Generate comprehensive mood analysis report"""
        import os
        os.makedirs(output_dir, exist_ok=True)
        
        saved_files = {}
        
        # Generate all visualizations
        plots = [
            ('mood_violin', self.plot_mood_violin),
            ('mood_stability', self.plot_mood_stability_analysis),
            ('mood_clustering', self.plot_mood_clustering),
            ('mood_correlations', self.plot_mood_correlations)
        ]
        
        for plot_name, plot_func in plots:
            try:
                save_path = os.path.join(output_dir, f"{plot_name}.png")
                fig = plot_func(save_path)
                if fig:
                    saved_files[plot_name] = save_path
                    plt.close(fig)
            except Exception as e:
                logger.error(f"Failed to generate {plot_name}: {e}")
        
        logger.info(f"Mood analysis report generated: {len(saved_files)} visualizations saved to {output_dir}")
        
        return saved_files


def create_mood_tracker(config: Optional[MoodTrackerConfig] = None,
                       visual_config: Optional[DAWNVisualConfig] = None) -> MoodTracker:
    """Factory function to create mood tracker"""
    return MoodTracker(config, visual_config)


if __name__ == "__main__":
    """Example usage of mood tracker"""
    
    # Create mood tracker
    tracker = create_mood_tracker()
    
    # Simulate mood data
    np.random.seed(42)
    torch.manual_seed(42)
    
    emotions = ['contemplative', 'excited', 'peaceful', 'analytical', 'inspired']
    
    for i in range(150):
        # Generate synthetic mood data
        base_mood = torch.rand(8) * 0.6 + 0.2  # Keep in reasonable range
        
        # Add some correlation structure
        if i > 0:
            # Make creativity correlate with curiosity
            base_mood[2] = 0.7 * base_mood[1] + 0.3 * torch.rand(1)
            # Make serenity inversely correlate with energy
            base_mood[0] = 1.0 - 0.5 * base_mood[4] + 0.2 * torch.rand(1)
        
        emotion = np.random.choice(emotions)
        
        tracker.add_mood_data(base_mood, emotion)
    
    # Generate report
    print("😊 Generating mood analysis report...")
    saved_files = tracker.generate_mood_report()
    
    print(f"📊 Mood analysis complete! Generated {len(saved_files)} visualizations:")
    for name, path in saved_files.items():
        print(f"   - {name}: {path}")
    
    # Get mood summary
    df = tracker.get_mood_df()
    if not df.empty:
        print(f"📈 Analyzed {len(df)} mood data points")
        print(f"😄 Emotional states: {df['emotion'].unique()}")
        print(f"📊 Average stability: {df['stability'].mean():.3f}")
