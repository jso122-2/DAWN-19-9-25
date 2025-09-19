#!/usr/bin/env python3
"""
DAWN Correlation Matrix Visualization (Unified)
==============================================

Enhanced correlation matrix visualization using the unified DAWN visual base
with consciousness-aware styling and device-agnostic tensor operations.
"""

import torch
import numpy as np
import matplotlib.pyplot as plt
import seaborn as sns
from pathlib import Path
import sys
import logging

# Import DAWN visual base
from dawn.subsystems.visual.dawn_visual_base import (
    DAWNVisualBase,
    DAWNVisualConfig,
    ConsciousnessColorPalette,
    device
)

logger = logging.getLogger(__name__)

class CorrelationMatrixVisualizer(DAWNVisualBase):
    """Enhanced correlation matrix visualizer using unified DAWN visual base"""
    
    def __init__(self):
        visual_config = DAWNVisualConfig(
            figure_size=(12, 10),
            background_color="#0a0a0a"
        )
        super().__init__(visual_config)
        
    def create_correlation_visualization(self, data: torch.Tensor, 
                                       labels: list = None,
                                       save_path: str = None) -> plt.Figure:
        """Create enhanced correlation matrix visualization with consciousness styling"""
        # Ensure data is on correct device and handle NaN values
        data = data.to(device)
        
        if torch.isnan(data).any():
            logger.warning("NaN values detected, removing affected samples")
            mask = ~torch.isnan(data).any(dim=1)
            data = data[mask]
        
        if data.shape[0] < 2:
            raise ValueError("Need at least 2 samples for correlation analysis")
        
        # Calculate correlation using PyTorch
        data_centered = data - data.mean(dim=0, keepdim=True)
        correlation_matrix = torch.corrcoef(data_centered.T)
        corr_np = correlation_matrix.cpu().numpy()
        
        # Create figure with consciousness styling
        fig = self.create_figure((1, 1))
        ax = self.axes[0]
        
        # Enhanced heatmap with seaborn
        mask = np.triu(np.ones_like(corr_np, dtype=bool))
        
        sns.heatmap(corr_np, 
                   mask=mask,
                   annot=True, 
                   cmap='RdBu_r', 
                   center=0,
                   square=True,
                   fmt='.2f',
                   cbar_kws={'label': 'Correlation Coefficient'},
                   ax=ax,
                   xticklabels=labels or [f'V{i}' for i in range(corr_np.shape[1])],
                   yticklabels=labels or [f'V{i}' for i in range(corr_np.shape[0])])
        
        ax.set_title('DAWN Consciousness - Correlation Matrix', 
                    color='white', fontsize=16, fontweight='bold')
        ax.set_facecolor('#0a0a0a')
        ax.tick_params(colors='white')
        
        # Style colorbar
        cbar = ax.collections[0].colorbar
        cbar.ax.yaxis.label.set_color('white')
        cbar.ax.tick_params(colors='white')
        
        plt.tight_layout()
        
        if save_path:
            self.save_consciousness_frame(save_path)
            logger.info(f"Enhanced correlation matrix saved to {save_path}")
        
        return fig


def main(*args, **kwargs):
    """Enhanced main function using unified DAWN visual system"""
    input_path = Path("data/correlation_data.npy")
    output_dir = Path("visual/outputs/correlation_matrix")
    output_dir.mkdir(parents=True, exist_ok=True)
    output_path = output_dir / "correlation_matrix_enhanced.png"

    if not input_path.exists():
        logger.error(f"Input data not found: {input_path}")
        # Generate sample data for demonstration
        logger.info("Generating sample correlation data for demonstration")
        sample_data = np.random.randn(100, 10)
        sample_data[:, 1] = 0.8 * sample_data[:, 0] + 0.2 * np.random.randn(100)  # Add correlation
        data_tensor = torch.from_numpy(sample_data).float().to(device)
    else:
        data_np = np.load(input_path)
        if data_np.size == 0:
            logger.error(f"Input data is empty: {input_path}")
            sys.exit(1)
        data_tensor = torch.from_numpy(data_np).float().to(device)
    
    logger.info(f"Loaded data: {data_tensor.shape} on {device}")
    
    # Create enhanced visualizer
    visualizer = CorrelationMatrixVisualizer()
    
    try:
        fig = visualizer.create_correlation_visualization(
            data_tensor, 
            save_path=str(output_path)
        )
        
        logger.info(f"SUCCESS: Enhanced correlation matrix saved to {output_path}")
        plt.close(fig)
        return str(output_path)
        
    except Exception as e:
        logger.error(f"Failed to create correlation visualization: {e}")
        sys.exit(1)

if __name__ == "__main__":
    main() 