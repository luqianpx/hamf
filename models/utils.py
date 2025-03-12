# models/utils.py
import torch
from pathlib import Path
from typing import Dict, Optional
import logging

logger = logging.getLogger(__name__)

class ModelCheckpoint:
    def __init__(self, save_dir: str):
        self.save_dir = Path(save_dir)
        self.save_dir.mkdir(parents=True, exist_ok=True)
        
    def save_model(self, 
                  model: torch.nn.Module,
                  optimizer: torch.optim.Optimizer,
                  scheduler: torch.optim.lr_scheduler._LRScheduler,
                  epoch: int,
                  metrics: Dict,
                  filename: str):
        """Save model checkpoint"""
        checkpoint = {
            'epoch': epoch,
            'model_state_dict': model.state_dict(),
            'optimizer_state_dict': optimizer.state_dict(),
            'scheduler_state_dict': scheduler.state_dict(),
            'metrics': metrics
        }
        torch.save(checkpoint, self.save_dir / filename)
        
    def load_model(self,
                  model: torch.nn.Module,
                  optimizer: Optional[torch.optim.Optimizer] = None,
                  scheduler: Optional[torch.optim.lr_scheduler._LRScheduler] = None,
                  filename: str = 'best_model.pt') -> Dict:
        """Load model checkpoint"""
        checkpoint = torch.load(self.save_dir / filename)
        model.load_state_dict(checkpoint['model_state_dict'])
        
        if optimizer:
            optimizer.load_state_dict(checkpoint['optimizer_state_dict'])
        if scheduler:
            scheduler.load_state_dict(checkpoint['scheduler_state_dict'])
            
        return checkpoint