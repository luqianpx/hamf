# models/attention_visualization.py
# author: px
# date: 2021-11-09

import torch
import numpy as np
import matplotlib.pyplot as plt
import seaborn as sns
from typing import Dict, List, Optional, Union
import logging
import pandas as pd
from pathlib import Path
import networkx as nx
from matplotlib.colors import LinearSegmentedColormap
import plotly.graph_objects as go
from plotly.subplots import make_subplots
import plotly.express as px
import json
from datetime import datetime

logging.basicConfig(level=logging.INFO)
logger = logging.getLogger(__name__)


class AttentionVisualizer:
    """
    Utility class for visualizing hierarchical attention weights
    """
    def __init__(self, save_dir: str = 'attention_plots'):
        """
        Args:
            save_dir: Directory to save visualization plots
        """
        self.save_dir = Path(save_dir)
        self.save_dir.mkdir(parents=True, exist_ok=True)
        
        # Set style
        plt.style.use('seaborn')
        self.modality_pairs = ['MRI-SNP', 'MRI-Clinical', 'SNP-Clinical']
        self.modalities = ['MRI', 'SNP', 'Clinical']
        self.colors = sns.color_palette("husl", 3)

    def plot_pairwise_attention(self, 
                              attention_weights: Dict[str, torch.Tensor],
                              batch_idx: int = 0,
                              save: bool = True) -> None:
        """
        Plot pairwise attention weights
        
        Args:
            attention_weights: Dictionary containing attention weights
            batch_idx: Index of batch sample to visualize
            save: Whether to save the plot
        """
        fig, axes = plt.subplots(1, 3, figsize=(15, 4))
        fig.suptitle('Pairwise Attention Weights', fontsize=16)
        
        for idx, key in enumerate(['mri_snp', 'mri_clinical', 'snp_clinical']):
            weights = attention_weights[key][batch_idx].cpu().detach().numpy()
            
            # Create heatmap
            sns.heatmap(
                weights.reshape(1, -1),
                ax=axes[idx],
                cmap='YlOrRd',
                cbar=True,
                xticklabels=self.modality_pairs[idx].split('-'),
                yticklabels=False,
                annot=True,
                fmt='.3f'
            )
            axes[idx].set_title(f'{self.modality_pairs[idx]} Attention')
        
        plt.tight_layout()
        if save:
            plt.savefig(self.save_dir / 'pairwise_attention.png', dpi=300, bbox_inches='tight')
            logger.info(f"Saved pairwise attention plot to {self.save_dir / 'pairwise_attention.png'}")
        plt.close()

    def plot_final_attention(self,
                           attention_weights: Dict[str, torch.Tensor],
                           batch_idx: int = 0,
                           save: bool = True) -> None:
        """
        Plot final fusion attention weights
        
        Args:
            attention_weights: Dictionary containing attention weights
            batch_idx: Index of batch sample to visualize
            save: Whether to save the plot
        """
        weights = attention_weights['final'][batch_idx].cpu().detach().numpy()
        
        plt.figure(figsize=(8, 6))
        plt.title('Final Fusion Attention Weights', fontsize=16)
        
        # Create bar plot
        bars = plt.bar(self.modality_pairs, weights, color=self.colors)
        
        # Add value labels on top of bars
        for bar in bars:
            height = bar.get_height()
            plt.text(bar.get_x() + bar.get_width()/2., height,
                    f'{height:.3f}',
                    ha='center', va='bottom')
        
        plt.ylabel('Attention Weight')
        plt.ylim(0, max(weights) * 1.2)  # Add some space for labels
        
        if save:
            plt.savefig(self.save_dir / 'final_attention.png', dpi=300, bbox_inches='tight')
            logger.info(f"Saved final attention plot to {self.save_dir / 'final_attention.png'}")
        plt.close()

    def plot_attention_flow(self,
                          attention_weights: Dict[str, torch.Tensor],
                          batch_idx: int = 0,
                          save: bool = True) -> None:
        """
        Plot hierarchical attention flow
        
        Args:
            attention_weights: Dictionary containing attention weights
            batch_idx: Index of batch sample to visualize
            save: Whether to save the plot
        """
        plt.figure(figsize=(12, 8))
        plt.title('Hierarchical Attention Flow', fontsize=16)
        
        # Plot modalities (first level)
        y_positions = [0, 1, 2]
        plt.scatter([0]*3, y_positions, c=self.colors, s=200, label='Modalities')
        
        for idx, mod in enumerate(self.modalities):
            plt.annotate(mod, (-0.1, y_positions[idx]), 
                        xytext=(-0.5, y_positions[idx]),
                        ha='right', va='center')
        
        # Plot pairwise fusion (second level)
        pair_positions = [0.5, 1, 1.5]
        plt.scatter([2]*3, pair_positions, c='lightgray', s=200, label='Pairwise Fusion')
        
        # Draw arrows for pairwise connections
        pairwise_weights = {
            'mri_snp': attention_weights['mri_snp'][batch_idx].cpu().detach().numpy(),
            'mri_clinical': attention_weights['mri_clinical'][batch_idx].cpu().detach().numpy(),
            'snp_clinical': attention_weights['snp_clinical'][batch_idx].cpu().detach().numpy()
        }
        
        self._draw_attention_arrows(pairwise_weights)
        
        # Plot final fusion (third level)
        final_weights = attention_weights['final'][batch_idx].cpu().detach().numpy()
        plt.scatter([4], [1], c='red', s=200, label='Final Fusion')
        
        # Draw arrows for final fusion
        for idx, weight in enumerate(final_weights):
            plt.arrow(2, pair_positions[idx], 1.8, 1-pair_positions[idx],
                     alpha=weight, width=0.02,
                     head_width=0.1, head_length=0.1,
                     fc='gray', ec='gray')
        
        plt.xlim(-1, 5)
        plt.ylim(-0.5, 2.5)
        plt.legend()
        plt.axis('off')
        
        if save:
            plt.savefig(self.save_dir / 'attention_flow.png', dpi=300, bbox_inches='tight')
            logger.info(f"Saved attention flow plot to {self.save_dir / 'attention_flow.png'}")
        plt.close()

    def _draw_attention_arrows(self, pairwise_weights: Dict[str, np.ndarray]) -> None:
        """Helper function to draw attention arrows for pairwise fusion"""
        connections = [
            ((0, 0), (2, 0.5)),  # MRI -> MRI-SNP
            ((0, 1), (2, 0.5)),  # SNP -> MRI-SNP
            ((0, 0), (2, 1.0)),  # MRI -> MRI-Clinical
            ((0, 2), (2, 1.0)),  # Clinical -> MRI-Clinical
            ((0, 1), (2, 1.5)),  # SNP -> SNP-Clinical
            ((0, 2), (2, 1.5))   # Clinical -> SNP-Clinical
        ]
        
        for idx, ((x1, y1), (x2, y2)) in enumerate(connections):
            weight_key = list(pairwise_weights.keys())[idx // 2]
            weight = pairwise_weights[weight_key][idx % 2]
            
            plt.arrow(x1, y1, x2-x1, y2-y1,
                     alpha=weight, width=0.02,
                     head_width=0.1, head_length=0.1,
                     fc='gray', ec='gray')

    def visualize_all(self,
                     attention_weights: Dict[str, torch.Tensor],
                     batch_idx: int = 0) -> None:
        """
        Generate all attention visualizations
        
        Args:
            attention_weights: Dictionary containing attention weights
            batch_idx: Index of batch sample to visualize
        """
        logger.info("Generating attention visualizations...")
        
        self.plot_pairwise_attention(attention_weights, batch_idx)
        self.plot_final_attention(attention_weights, batch_idx)
        self.plot_attention_flow(attention_weights, batch_idx)
        
        logger.info("Completed generating all visualizations")

# Example usage
def main():
    # Create dummy attention weights
    batch_size = 32
    attention_weights = {
        'mri_snp': torch.softmax(torch.randn(batch_size, 2), dim=1),
        'mri_clinical': torch.softmax(torch.randn(batch_size, 2), dim=1),
        'snp_clinical': torch.softmax(torch.randn(batch_size, 2), dim=1),
        'final': torch.softmax(torch.randn(batch_size, 3), dim=1)
    }
    
    # Create visualizer and generate plots
    visualizer = AttentionVisualizer(save_dir='attention_plots')
    visualizer.visualize_all(attention_weights, batch_idx=0)

if __name__ == "__main__":
    main()