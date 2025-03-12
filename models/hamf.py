# models/hamf.py
# author: px
# date: 2021-11-09

import torch
import torch.nn as nn
import logging
from typing import List, Dict, Tuple, Optional, Union
from pathlib import Path

# Import our custom modules
from .snp_feature_extractor import SNPSDAE
from .clinical_feature_extractor import ClinicalDNN
from .attention import HierarchicalAttention
from .attention_visualization import AttentionVisualizer

class HAMF(nn.Module):
    def __init__(self,
                 snp_input_dim: int = 450,
                 clinical_input_dim: int = 42,
                 mri_feature_dim: int = 2048,
                 fusion_dim: int = 64,
                 num_classes: int = 2,
                 dropout_rate: float = 0.3,
                 use_batch_norm: bool = True):
        super(HAMF, self).__init__()
        
        self.fusion_dim = fusion_dim
        self.use_batch_norm = use_batch_norm
        
        # Initialize feature extractors
        self.snp_extractor = SNPSDAE(
            input_dim=snp_input_dim,
            hidden_dims=[300, 200, 100],
            dropout_rate=dropout_rate
        )
        
        self.clinical_extractor = ClinicalDNN(
            input_dim=clinical_input_dim,
            hidden_dim=fusion_dim,
            output_dim=fusion_dim // 2,
            dropout_rate=dropout_rate
        )
        
        # MRI feature projection (since they're pre-extracted)
        self.mri_projection = nn.Sequential(
            nn.Linear(mri_feature_dim, fusion_dim),
            nn.BatchNorm1d(fusion_dim) if use_batch_norm else nn.Identity(),
            nn.ReLU(),
            nn.Dropout(dropout_rate)
        )
        
        # Initialize hierarchical attention
        self.attention = HierarchicalAttention(
            input_dims=[fusion_dim, fusion_dim, fusion_dim // 2],  # Aligned dimensions
            output_dim=fusion_dim
        )
        
        # Classification head with residual connection
        self.classifier = nn.Sequential(
            nn.Linear(fusion_dim, fusion_dim // 2),
            nn.BatchNorm1d(fusion_dim // 2) if use_batch_norm else nn.Identity(),
            nn.ReLU(),
            nn.Dropout(dropout_rate),
            nn.Linear(fusion_dim // 2, num_classes)
        )
        
        # Initialize attention visualizer
        self.visualizer = AttentionVisualizer(save_dir='attention_plots')
        
        self._init_weights()
        
    def _init_weights(self):
        """Initialize model weights properly"""
        for m in self.modules():
            if isinstance(m, nn.Linear):
                nn.init.xavier_normal_(m.weight)
                if m.bias is not None:
                    nn.init.constant_(m.bias, 0)
            elif isinstance(m, nn.BatchNorm1d):
                nn.init.constant_(m.weight, 1)
                nn.init.constant_(m.bias, 0)
    
    def forward(self, 
                mri_features: torch.Tensor,
                snp_data: torch.Tensor,
                clinical_data: torch.Tensor,
                return_attention: bool = False) -> Union[torch.Tensor, Tuple[torch.Tensor, Dict]]:
        """
        Forward pass through HAMF model
        
        Args:
            mri_features: Pre-extracted MRI features [batch_size, mri_feature_dim]
            snp_data: SNP data [batch_size, snp_input_dim]
            clinical_data: Clinical data [batch_size, clinical_input_dim]
            return_attention: Whether to return attention weights
            
        Returns:
            predictions: Class predictions [batch_size, num_classes]
            attention_weights: Dictionary of attention weights (if return_attention=True)
        """
        # Extract features from each modality
        mri_features = self.mri_projection(mri_features)
        snp_features = self.snp_extractor(snp_data, return_decoder=False)[0]
        clinical_features = self.clinical_extractor(clinical_data)
        
        # Apply hierarchical attention fusion
        fused_features, attention_weights = self.attention([
            mri_features,
            snp_features,
            clinical_features
        ])
        
        # Classification
        predictions = self.classifier(fused_features)
        
        if return_attention:
            return predictions, attention_weights
        return predictions
    
    def visualize_attention(self, attention_weights: Dict[str, torch.Tensor], 
                          batch_idx: int = 0) -> None:
        """Visualize attention weights for a specific batch sample"""
        self.visualizer.visualize_all(attention_weights, batch_idx)

class HAMFTrainer:
    def __init__(self,
                 model: HAMF,
                 optimizer: torch.optim.Optimizer,
                 criterion: nn.Module,
                 device: torch.device,
                 scheduler_patience: int = 5,
                 scheduler_factor: float = 0.1,
                 early_stopping_patience: int = 10,
                 checkpoint_dir: str = 'checkpoints'):
        
        self.model = model.to(device)
        self.optimizer = optimizer
        self.criterion = criterion
        self.device = device
        
        # Learning rate scheduler
        self.scheduler = torch.optim.lr_scheduler.ReduceLROnPlateau(
            optimizer,
            mode='min',
            patience=scheduler_patience,
            factor=scheduler_factor,
            verbose=True
        )
        
        # Early stopping
        self.early_stopping_patience = early_stopping_patience
        self.best_val_loss = float('inf')
        self.patience_counter = 0
        
        # Checkpointing
        self.checkpoint_dir = Path(checkpoint_dir)
        self.checkpoint_dir.mkdir(parents=True, exist_ok=True)
        
        # Training history
        self.history = {
            'train_loss': [],
            'val_loss': [],
            'train_metrics': [],
            'val_metrics': [],
            'learning_rates': []
        }
        
        # Initialize metrics tracker
        self.metrics_tracker = MetricsTracker()
    # hamf.py - Add to HAMFTrainer class
def train(self, 
          train_loader: DataLoader,
          val_loader: DataLoader,
          num_epochs: int,
          save_freq: int = 5) -> None:
    """Training loop"""
    for epoch in range(num_epochs):
        # Training phase
        self.model.train()
        train_loss = 0
        train_metrics = MetricsTracker()
        
        for batch_idx, (data, labels) in enumerate(train_loader):
            # Move data to device
            data = {k: v.to(self.device) for k, v in data.items()}
            labels = labels.to(self.device)
            
            # Forward pass
            self.optimizer.zero_grad()
            outputs = self.model(
                mri_features=data['mri_features'],
                snp_data=data['snp_data'],
                clinical_data=data['clinical_data']
            )
            
            # Calculate loss
            loss = self.criterion(outputs, labels)
            
            # Backward pass
            loss.backward()
            self.optimizer.step()
            
            # Update metrics
            train_loss += loss.item()
            train_metrics.update(
                preds=outputs.argmax(dim=1),
                labels=labels,
                probs=F.softmax(outputs, dim=1)[:, 1]
            )
            
        # Validation phase
        val_loss, val_metrics = self.validate(val_loader)
        
        # Update learning rate scheduler
        self.scheduler.step(val_loss)
        
        # Save checkpoint
        if (epoch + 1) % save_freq == 0:
            self.save_checkpoint(epoch, val_loss, val_metrics)
            
        # Early stopping
        if val_loss < self.best_val_loss:
            self.best_val_loss = val_loss
            self.patience_counter = 0
        else:
            self.patience_counter += 1
            
        if self.patience_counter >= self.early_stopping_patience:
            logger.info("Early stopping triggered")
            break
            
        # Update history
        self.history['train_loss'].append(train_loss / len(train_loader))
        self.history['val_loss'].append(val_loss)
        self.history['train_metrics'].append(train_metrics.compute_metrics())
        self.history['val_metrics'].append(val_metrics)
        self.history['learning_rates'].append(
            self.optimizer.param_groups[0]['lr']
        )

def validate(self, val_loader: DataLoader) -> Tuple[float, Dict[str, float]]:
    """Validation loop"""
    self.model.eval()
    val_loss = 0
    val_metrics = MetricsTracker()
    
    with torch.no_grad():
        for data, labels in val_loader:
            # Move data to device
            data = {k: v.to(self.device) for k, v in data.items()}
            labels = labels.to(self.device)
            
            # Forward pass
            outputs = self.model(
                mri_features=data['mri_features'],
                snp_data=data['snp_data'],
                clinical_data=data['clinical_data']
            )
            
            # Calculate loss
            loss = self.criterion(outputs, labels)
            val_loss += loss.item()
            
            # Update metrics
            val_metrics.update(
                preds=outputs.argmax(dim=1),
                labels=labels,
                probs=F.softmax(outputs, dim=1)[:, 1]
            )
    
    return val_loss / len(val_loader), val_metrics.compute_metrics()
    
    def save_checkpoint(self, epoch: int, val_loss: float, val_metrics: Dict):
        """Save model checkpoint"""
        checkpoint = {
            'epoch': epoch,
            'model_state_dict': self.model.state_dict(),
            'optimizer_state_dict': self.optimizer.state_dict(),
            'scheduler_state_dict': self.scheduler.state_dict(),
            'val_loss': val_loss,
            'val_metrics': val_metrics,
            'history': self.history
        }
        
        # Save latest checkpoint
        torch.save(checkpoint, self.checkpoint_dir / 'latest_checkpoint.pt')
        
        # Save best model
        if val_loss < self.best_val_loss:
            self.best_val_loss = val_loss
            torch.save(checkpoint, self.checkpoint_dir / 'best_model.pt')
            
    def load_checkpoint(self, checkpoint_path: str):
        """Load model checkpoint"""
        checkpoint = torch.load(checkpoint_path, map_location=self.device)
        
        self.model.load_state_dict(checkpoint['model_state_dict'])
        self.optimizer.load_state_dict(checkpoint['optimizer_state_dict'])
        self.scheduler.load_state_dict(checkpoint['scheduler_state_dict'])
        self.history = checkpoint['history']
        
        return checkpoint['epoch']
