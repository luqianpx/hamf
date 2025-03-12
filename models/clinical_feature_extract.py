# models/clinical_feature_extractor.py

import torch
import torch.nn as nn
import logging
from typing import List

# Set up logging
logging.basicConfig(
    level=logging.INFO,
    format='%(asctime)s - %(name)s - %(levelname)s - %(message)s'
)
logger = logging.getLogger(__name__)

class ClinicalDNN(nn.Module):
    """
    Two-layer DNN for clinical feature extraction
    """
    def __init__(self,
                 input_dim: int = 42,
                 hidden_dim: int = 64,
                 output_dim: int = 32,
                 dropout_rate: float = 0.2):
        """
        Args:
            input_dim: Number of input clinical features
            hidden_dim: Dimension of hidden layer
            output_dim: Dimension of output features
            dropout_rate: Dropout rate for regularization
        """
        super(ClinicalDNN, self).__init__()
        
        self.network = nn.Sequential(
            # First layer
            nn.Linear(input_dim, hidden_dim),
            nn.BatchNorm1d(hidden_dim),
            nn.ReLU(),
            nn.Dropout(dropout_rate),
            
            # Second layer
            nn.Linear(hidden_dim, output_dim),
            nn.BatchNorm1d(output_dim),
            nn.ReLU()
        )
        
        logger.info(f"Initialized ClinicalDNN with input_dim={input_dim}, "
                   f"hidden_dim={hidden_dim}, output_dim={output_dim}")
    
    def forward(self, x: torch.Tensor) -> torch.Tensor:
        """
        Forward pass through the network
        
        Args:
            x: Input clinical features tensor
            
        Returns:
            Extracted features tensor
        """
        return self.network(x)

# Training utilities
def train_epoch(model: ClinicalDNN,
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
    
    for batch_idx, (data, labels) in enumerate(dataloader):
        data = data.to(device)
        labels = labels.to(device)
        
        optimizer.zero_grad()
        
        # Forward pass
        outputs = model(data)
        loss = criterion(outputs, labels)
        
        # Backward pass
        loss.backward()
        optimizer.step()
        
        total_loss += loss.item()
        
        if batch_idx % 50 == 0:
            logger.info(f'Batch {batch_idx}: Loss = {loss.item():.4f}')
    
    avg_loss = total_loss / len(dataloader)
    return avg_loss

# Example usage
def main():
    # SNP SDAE example
    snp_model = SNPSDAE(
        input_dim=450,
        hidden_dims=[300, 200, 100],
        dropout_rate=0.3
    )
    
    # Clinical DNN example
    clinical_model = ClinicalDNN(
        input_dim=42,
        hidden_dim=64,
        output_dim=32,
        dropout_rate=0.2
    )
    
    logger.info("Models created successfully")

if __name__ == "__main__":
    main()