# models/snp_feature_extractor.py
import torch
import torch.nn as nn
import logging
import numpy as np
from typing import Tuple, Optional

# Set up logging
logging.basicConfig(
    level=logging.INFO,
    format='%(asctime)s - %(name)s - %(levelname)s - %(message)s'
)
logger = logging.getLogger(__name__)

class SNPEncoder(nn.Module):
    """
    Encoder part of the Stacked Denoising Autoencoder for SNP data
    """
    def __init__(self, 
                 input_dim: int = 450,
                 hidden_dims: list = [300, 200, 100],
                 dropout_rate: float = 0.3):
        """
        Args:
            input_dim: Number of input features (SNPs)
            hidden_dims: List of hidden layer dimensions
            dropout_rate: Dropout rate for noise
        """
        super(SNPEncoder, self).__init__()
        
        self.input_dim = input_dim
        self.hidden_dims = hidden_dims
        self.dropout_rate = dropout_rate
        
        # Build encoder layers
        layers = []
        prev_dim = input_dim
        
        for hidden_dim in hidden_dims:
            layers.extend([
                nn.Linear(prev_dim, hidden_dim),
                nn.BatchNorm1d(hidden_dim),
                nn.ReLU(),
                nn.Dropout(dropout_rate)
            ])
            prev_dim = hidden_dim
        
        self.encoder = nn.Sequential(*layers)
        logger.info(f"SNP Encoder architecture: {self.encoder}")
    
    def forward(self, x: torch.Tensor) -> torch.Tensor:
        return self.encoder(x)

class SNPDecoder(nn.Module):
    """
    Decoder part of the Stacked Denoising Autoencoder
    """
    def __init__(self, 
                 output_dim: int = 450,
                 hidden_dims: list = [100, 200, 300]):
        """
        Args:
            output_dim: Original input dimension
            hidden_dims: List of hidden layer dimensions (reverse of encoder)
        """
        super(SNPDecoder, self).__init__()
        
        # Build decoder layers
        layers = []
        prev_dim = hidden_dims[0]
        
        for hidden_dim in hidden_dims[1:]:
            layers.extend([
                nn.Linear(prev_dim, hidden_dim),
                nn.BatchNorm1d(hidden_dim),
                nn.ReLU()
            ])
            prev_dim = hidden_dim
        
        # Final reconstruction layer
        layers.extend([
            nn.Linear(prev_dim, output_dim),
            nn.Sigmoid()  # For binary SNP data
        ])
        
        self.decoder = nn.Sequential(*layers)
        logger.info(f"SNP Decoder architecture: {self.decoder}")
    
    def forward(self, x: torch.Tensor) -> torch.Tensor:
        return self.decoder(x)

class SNPSDAE(nn.Module):
    """
    Complete Stacked Denoising Autoencoder for SNP feature extraction
    """
    def __init__(self,
                 input_dim: int = 450,
                 hidden_dims: list = [300, 200, 100],
                 dropout_rate: float = 0.3):
        """
        Args:
            input_dim: Number of input features (SNPs)
            hidden_dims: List of hidden layer dimensions
            dropout_rate: Dropout rate for noise
        """
        super(SNPSDAE, self).__init__()
        
        self.encoder = SNPEncoder(input_dim, hidden_dims, dropout_rate)
        self.decoder = SNPDecoder(input_dim, hidden_dims[::-1])
        
        logger.info(f"Initialized SNPSDAE with input_dim={input_dim}, "
                   f"hidden_dims={hidden_dims}, dropout_rate={dropout_rate}")
    
    def add_noise(self, x: torch.Tensor, noise_factor: float = 0.3) -> torch.Tensor:
        """Add random noise to the input"""
        noise = torch.randn_like(x) * noise_factor
        corrupted = x + noise
        return torch.clamp(corrupted, 0., 1.)
    
    def forward(self, x: torch.Tensor, 
                add_noise: bool = True,
                return_decoder: bool = True) -> Tuple[torch.Tensor, Optional[torch.Tensor]]:
        """
        Forward pass through the SDAE
        
        Args:
            x: Input tensor
            add_noise: Whether to add noise during training
            return_decoder: Whether to return decoder output
            
        Returns:
            Tuple of (encoded features, decoded output if return_decoder=True)
        """
        if add_noise:
            x = self.add_noise(x)
        
        encoded = self.encoder(x)
        
        if return_decoder:
            decoded = self.decoder(encoded)
            return encoded, decoded
        return encoded, None

    def encode(self, x: torch.Tensor) -> torch.Tensor:
        """Encode input without noise for feature extraction"""
        return self.encoder(x)

# Training utilities
def train_epoch(model: SNPSDAE,
                dataloader: torch.utils.data.DataLoader,
                criterion: nn.Module,
                optimizer: torch.optim.Optimizer,
                device: torch.device) -> float:
    """
    Train the model for one epoch
    
    Returns:
        Average loss for the epoch
    """
    model.train()
    total_loss = 0
    
    for batch_idx, data in enumerate(dataloader):
        data = data.to(device)
        optimizer.zero_grad()
        
        # Forward pass
        _, decoded = model(data)
        loss = criterion(decoded, data)
        
        # Backward pass
        loss.backward()
        optimizer.step()
        
        total_loss += loss.item()
        
        if batch_idx % 100 == 0:
            logger.info(f'Batch {batch_idx}: Loss = {loss.item():.4f}')
    
    avg_loss = total_loss / len(dataloader)
    return avg_loss