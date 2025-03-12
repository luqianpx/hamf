# models/attention.py
# author: px
# date: 2021-11-09

# models/attention.py
import torch
import torch.nn as nn
import torch.nn.functional as F
import logging
from typing import List, Tuple, Dict
import math

logging.basicConfig(level=logging.INFO)
logger = logging.getLogger(__name__)

class ScaledDotProductAttention(nn.Module):
    """
    Scaled Dot-Product Attention mechanism
    """
    def __init__(self, temperature: float):
        super().__init__()
        self.temperature = temperature
        
    def forward(self, q: torch.Tensor, k: torch.Tensor, v: torch.Tensor) -> Tuple[torch.Tensor, torch.Tensor]:
        """
        Args:
            q: Query tensor [batch_size, query_len, dim]
            k: Key tensor [batch_size, key_len, dim]
            v: Value tensor [batch_size, value_len, dim]
            
        Returns:
            Tuple of (attended output, attention weights)
        """
        attn = torch.matmul(q / self.temperature, k.transpose(1, 2))
        attn = F.softmax(attn, dim=-1)
        output = torch.matmul(attn, v)
        
        return output, attn

class NonlinearGating(nn.Module):
    """
    Nonlinear Gating Module for mapping different modality features 
    into a common latent space with residual connections
    """
    def __init__(self, input_dims: List[int], output_dim: int = 64):
        super().__init__()
        
        self.gating_networks = nn.ModuleList([
            nn.Sequential(
                nn.Linear(dim, output_dim),
                nn.LayerNorm(output_dim),
                nn.ReLU(),
                nn.Dropout(0.2),
                nn.Linear(output_dim, output_dim),
                nn.LayerNorm(output_dim)
            ) for dim in input_dims
        ])
        
        # Residual projections if input dim != output dim
        self.residual_projections = nn.ModuleList([
            nn.Linear(dim, output_dim) if dim != output_dim else nn.Identity()
            for dim in input_dims
        ])
        
    def forward(self, features: List[torch.Tensor]) -> List[torch.Tensor]:
        """
        Args:
            features: List of feature tensors from different modalities
            
        Returns:
            List of gated features in common space
        """
        gated_features = []
        for feat, gate_net, res_proj in zip(
            features, self.gating_networks, self.residual_projections
        ):
            # Apply gating with residual connection
            gated = gate_net(feat) + res_proj(feat)
            gated_features.append(gated)
            
        return gated_features

class PairwiseAttention(nn.Module):
    """
    Enhanced Pairwise Attention mechanism with multi-head attention
    """
    def __init__(self, feature_dim: int = 64, num_heads: int = 4):
        super().__init__()
        
        assert feature_dim % num_heads == 0, "Feature dimension must be divisible by number of heads"
        
        self.num_heads = num_heads
        self.head_dim = feature_dim // num_heads
        self.scale = math.sqrt(self.head_dim)
        
        # Multi-head projection layers
        self.q_proj = nn.Linear(feature_dim, feature_dim)
        self.k_proj = nn.Linear(feature_dim, feature_dim)
        self.v_proj = nn.Linear(feature_dim, feature_dim)
        self.o_proj = nn.Linear(feature_dim, feature_dim)
        
        # Fusion layer
        self.fusion = nn.Sequential(
            nn.Linear(feature_dim * 3, feature_dim),  # Concatenated + element-wise
            nn.LayerNorm(feature_dim),
            nn.ReLU(),
            nn.Dropout(0.1)
        )
        
    def forward(self, x1: torch.Tensor, x2: torch.Tensor) -> Tuple[torch.Tensor, torch.Tensor]:
        batch_size = x1.size(0)
        
        # Multi-head attention
        q = self.q_proj(x1).view(batch_size, -1, self.num_heads, self.head_dim).transpose(1, 2)
        k = self.k_proj(x2).view(batch_size, -1, self.num_heads, self.head_dim).transpose(1, 2)
        v = self.v_proj(x2).view(batch_size, -1, self.num_heads, self.head_dim).transpose(1, 2)
        
        # Scaled dot-product attention
        attn_weights = torch.matmul(q, k.transpose(-2, -1)) / self.scale
        attn_weights = F.softmax(attn_weights, dim=-1)
        
        # Apply attention
        attended = torch.matmul(attn_weights, v)
        attended = attended.transpose(1, 2).contiguous().view(batch_size, -1)
        attended = self.o_proj(attended)
        
        # Fusion with different interactions
        fused = self.fusion(torch.cat([
            attended,
            x1 * x2,  # Element-wise interaction
            torch.abs(x1 - x2)  # Difference interaction
        ], dim=-1))
        
        return fused, attn_weights.mean(dim=1)  # Average attention across heads

class ModalityFusion(nn.Module):
    """
    Enhanced Final Fusion module with gating and residual connections
    """
    def __init__(self, feature_dim: int = 64):
        super().__init__()
        
        self.attention = nn.Sequential(
            nn.Linear(feature_dim * 3, feature_dim),
            nn.LayerNorm(feature_dim),
            nn.Tanh(),
            nn.Linear(feature_dim, 3)
        )
        
        # Gating mechanism
        self.gate = nn.Sequential(
            nn.Linear(feature_dim * 3, feature_dim),
            nn.LayerNorm(feature_dim),
            nn.Sigmoid()
        )
        
        # Fusion layers
        self.fusion = nn.Sequential(
            nn.Linear(feature_dim * 4, feature_dim),  # Including gated features
            nn.LayerNorm(feature_dim),
            nn.ReLU(),
            nn.Dropout(0.1),
            nn.Linear(feature_dim, feature_dim),
            nn.LayerNorm(feature_dim)
        )
        
    def forward(self, features: List[torch.Tensor]) -> Tuple[torch.Tensor, torch.Tensor]:
        # Concatenate all features for attention
        concat_features = torch.cat(features, dim=-1)
        
        # Calculate attention weights
        attention_weights = F.softmax(self.attention(concat_features), dim=-1)
        
        # Calculate gating weights
        gate_weights = self.gate(concat_features)
        
        # Apply attention and gating
        stacked = torch.stack(features, dim=1)
        attended = torch.sum(stacked * attention_weights.unsqueeze(-1), dim=1)
        gated = attended * gate_weights
        
        # Final fusion with residual connection
        fused = self.fusion(torch.cat([gated] + features, dim=-1))
        fused = fused + attended  # Residual connection
        
        return fused, attention_weights

class HierarchicalAttention(nn.Module):
    """
    Enhanced Hierarchical Attention mechanism with improved feature interaction
    """
    def __init__(self,
                 input_dims: List[int],
                 output_dim: int = 64,
                 num_heads: int = 4):
        super().__init__()
        
        # Nonlinear gating for initial feature mapping
        self.gating = NonlinearGating(input_dims, output_dim)
        
        # Pairwise attention modules
        self.mri_snp_attention = PairwiseAttention(output_dim, num_heads)
        self.mri_clinical_attention = PairwiseAttention(output_dim, num_heads)
        self.snp_clinical_attention = PairwiseAttention(output_dim, num_heads)
        
        # Final fusion module
        self.final_fusion = ModalityFusion(output_dim)
        
        # Layer normalization for outputs
        self.norm = nn.LayerNorm(output_dim)
        
        logger.info(f"Initialized HierarchicalAttention with dimensions: {input_dims} -> {output_dim}")
    
    def forward(self, features: List[torch.Tensor]) -> Tuple[torch.Tensor, Dict[str, torch.Tensor]]:
        """
        Forward pass through the hierarchical attention mechanism
        
        Args:
            features: List of [mri_features, snp_features, clinical_features]
            
        Returns:
            Tuple of (fused_features, attention_weights_dict)
        """
        # Map features to common space
        mri_feat, snp_feat, clinical_feat = self.gating(features)
        
        # First level: Pairwise fusion with attention
        mri_snp_fused, mri_snp_weights = self.mri_snp_attention(mri_feat, snp_feat)
        mri_clinical_fused, mri_clinical_weights = self.mri_clinical_attention(mri_feat, clinical_feat)
        snp_clinical_fused, snp_clinical_weights = self.snp_clinical_attention(snp_feat, clinical_feat)
        
        # Second level: Final fusion
        final_features, final_weights = self.final_fusion([
            mri_snp_fused,
            mri_clinical_fused,
            snp_clinical_fused
        ])
        
        # Apply final normalization
        final_features = self.norm(final_features)
        
        # Collect attention weights
        attention_weights = {
            'mri_snp': mri_snp_weights,
            'mri_clinical': mri_clinical_weights,
            'snp_clinical': snp_clinical_weights,
            'final': final_weights
        }
        
        return final_features, attention_weights