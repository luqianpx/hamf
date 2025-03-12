# models/trainer.py
import torch
import torch.nn as nn
import torch.nn.functional as F
from torch.utils.data import DataLoader
import logging
from pathlib import Path
from typing import Dict, Tuple
import json
from datetime import datetime

from .metrics import MetricsTracker
from .config import HAMFConfig

logger = logging.getLogger(__name__)

class HAMFTrainer:
    def __init__(self,
                 model: nn.Module,
                 config: HAMFConfig,
                 experiment_dir: Path,
                 device: torch.device):
        """
        Initialize trainer
        
        Args:
            model: HAMF model
            config: Training configuration
            experiment_dir: Directory for saving results
            device: Device to train on
        """
        self.model = model
        self.config = config
        self.device = device
        self.experiment_dir = experiment_dir
        
        # Initialize optimizer
        self.optimizer = torch.optim.Adam(
            model.parameters(),
            lr=config.learning_rate
        )
        
        # Initialize criterion
        self.criterion = nn.CrossEntropyLoss()
        
        # Initialize scheduler
        self.scheduler = torch.optim.lr_scheduler.ReduceLROnPlateau(
            self.optimizer,
            mode='min',
            patience=config.scheduler_patience,
            factor=config.scheduler_factor,
            verbose=True
        )
        
        # Initialize metrics tracker
        self.metrics_tracker = MetricsTracker()
        
        # Training state
        self.best_val_loss = float('inf')
        self.patience_counter = 0
        self.current_epoch = 0
        self.history = {
            'train_loss': [],
            'val_loss': [],
            'train_metrics': [],
            'val_metrics': [],
            'learning_rates': []
        }
        
    def train_epoch(self, train_loader: DataLoader) -> Tuple[float, Dict]:
        """Train for one epoch"""
        self.model.train()
        total_loss = 0
        self.metrics_tracker.reset()
        
        for batch_idx, (data, labels) in enumerate(train_loader):
            try:
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
                total_loss += loss.item()
                self.metrics_tracker.update(
                    preds=outputs.argmax(dim=1),
                    labels=labels,
                    probs=F.softmax(outputs, dim=1)[:, 1]
                )
                
                # Log progress
                if batch_idx % 10 == 0:
                    logger.info(f'Train Batch {batch_idx}/{len(train_loader)}: '
                              f'Loss: {loss.item():.4f}')
                    
            except Exception as e:
                logger.error(f"Error in training batch {batch_idx}: {str(e)}")
                continue
        
        avg_loss = total_loss / len(train_loader)
        metrics = self.metrics_tracker.compute_metrics()
        
        return avg_loss, metrics
    
    def validate(self, val_loader: DataLoader) -> Tuple[float, Dict]:
        """Validation loop"""
        self.model.eval()
        total_loss = 0
        self.metrics_tracker.reset()
        
        with torch.no_grad():
            for batch_idx, (data, labels) in enumerate(val_loader):
                try:
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
                    total_loss += loss.item()
                    
                    # Update metrics
                    self.metrics_tracker.update(
                        preds=outputs.argmax(dim=1),
                        labels=labels,
                        probs=F.softmax(outputs, dim=1)[:, 1]
                    )
                    
                except Exception as e:
                    logger.error(f"Error in validation batch {batch_idx}: {str(e)}")
                    continue
        
        avg_loss = total_loss / len(val_loader)
        metrics = self.metrics_tracker.compute_metrics()
        
        return avg_loss, metrics
    
    def train(self, train_loader: DataLoader, val_loader: DataLoader):
        """Full training loop"""
        logger.info("Starting training...")
        start_time = datetime.now()
        
        try:
            for epoch in range(self.config.num_epochs):
                self.current_epoch = epoch
                logger.info(f"\nEpoch {epoch+1}/{self.config.num_epochs}")
                
                # Training phase
                train_loss, train_metrics = self.train_epoch(train_loader)
                
                # Validation phase
                val_loss, val_metrics = self.validate(val_loader)
                
                # Update learning rate scheduler
                self.scheduler.step(val_loss)
                
                # Update history
                self.update_history(train_loss, val_loss, train_metrics, val_metrics)
                
                # Save checkpoint
                self.save_checkpoint(val_loss, val_metrics)
                
                # Early stopping check
                if self.check_early_stopping(val_loss):
                    logger.info("Early stopping triggered")
                    break
                
                # Log epoch results
                self.log_epoch_results(train_loss, val_loss, train_metrics, val_metrics)
            
            # Save final results
            self.save_final_results(start_time)
            
        except Exception as e:
            logger.error(f"Training failed: {str(e)}")
            raise
            
    def update_history(self, train_loss: float, val_loss: float,
                      train_metrics: Dict, val_metrics: Dict):
        """Update training history"""
        self.history['train_loss'].append(train_loss)
        self.history['val_loss'].append(val_loss)
        self.history['train_metrics'].append(train_metrics)
        self.history['val_metrics'].append(val_metrics)
        self.history['learning_rates'].append(
            self.optimizer.param_groups[0]['lr']
        )
    
    def save_checkpoint(self, val_loss: float, val_metrics: Dict):
        """Save model checkpoint"""
        checkpoint = {
            'epoch': self.current_epoch,
            'model_state_dict': self.model.state_dict(),
            'optimizer_state_dict': self.optimizer.state_dict(),
            'scheduler_state_dict': self.scheduler.state_dict(),
            'val_loss': val_loss,
            'val_metrics': val_metrics,
            'history': self.history
        }
        
        # Save latest checkpoint
        torch.save(checkpoint, 
                  self.experiment_dir / 'checkpoints' / 'latest_checkpoint.pt')
        
        # Save best model
        if val_loss < self.best_val_loss:
            self.best_val_loss = val_loss
            torch.save(checkpoint, 
                      self.experiment_dir / 'checkpoints' / 'best_model.pt')
    
    def check_early_stopping(self, val_loss: float) -> bool:
        """Check early stopping condition"""
        if val_loss < self.best_val_loss:
            self.patience_counter = 0
        else:
            self.patience_counter += 1
            
        return self.patience_counter >= self.config.early_stopping_patience
    
    def log_epoch_results(self, train_loss: float, val_loss: float,
                         train_metrics: Dict, val_metrics: Dict):
        """Log epoch results"""
        logger.info(
            f"Train Loss: {train_loss:.4f}, "
            f"Val Loss: {val_loss:.4f}\n"
            f"Train Metrics: {train_metrics}\n"
            f"Val Metrics: {val_metrics}"
        )
    
    def save_final_results(self, start_time: datetime):
        """Save final training results"""
        training_time = datetime.now() - start_time
        
        results = {
            'training_time': str(training_time),
            'best_val_loss': self.best_val_loss,
            'history': self.history,
            'config': self.config.__dict__
        }
        
        with open(self.experiment_dir / 'training_results.json', 'w') as f:
            json.dump(results, f, indent=4)