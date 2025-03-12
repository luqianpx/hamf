# models/experiment.py
# author: px
# date: 2021-11-09

import torch
import logging
from pathlib import Path
from typing import Dict, Optional
from datetime import datetime
import yaml
import json

from .hamf import HAMF
from .trainer import HAMFTrainer
from .data_load import MultimodalDataLoader
from .config import HAMFConfig
from .utils import ModelCheckpoint
from .attention_visualization import AttentionVisualizer

logger = logging.getLogger(__name__)

class HAMFExperiment:
    """
    Manages the entire HAMF experiment pipeline
    """
    def __init__(self, config_path: str):
        """
        Initialize experiment
        
        Args:
            config_path: Path to configuration file
        """
        self.config = self._load_config(config_path)
        self.experiment_dir = self._setup_experiment_dir()
        self.device = torch.device('cuda' if torch.cuda.is_available() else 'cpu')
        
        # Initialize components
        self.model = None
        self.trainer = None
        self.data_loader = None
        self.checkpoint_manager = ModelCheckpoint(self.experiment_dir / 'checkpoints')
        self.visualizer = AttentionVisualizer(self.experiment_dir / 'visualizations')
        
    def _load_config(self, config_path: str) -> HAMFConfig:
        """Load and validate configuration"""
        with open(config_path) as f:
            config_dict = yaml.safe_load(f)
        config = HAMFConfig(**config_dict)
        config.validate()  # Validate configuration
        return config
        
    def _setup_experiment_dir(self) -> Path:
        """Setup experiment directory structure"""
        timestamp = datetime.now().strftime('%Y%m%d_%H%M%S')
        exp_dir = Path(self.config.output_dir) / timestamp
        
        # Create directories
        for subdir in ['checkpoints', 'logs', 'visualizations', 'results']:
            (exp_dir / subdir).mkdir(parents=True, exist_ok=True)
            
        # Save configuration
        with open(exp_dir / 'config.yaml', 'w') as f:
            yaml.dump(self.config.__dict__, f)
            
        return exp_dir
        
    def setup(self):
        """Setup all components"""
        logger.info("Setting up experiment components...")
        
        # Initialize data loader
        self.data_loader = MultimodalDataLoader(
            mri_features_path=self.config.mri_features_path,
            snp_data_path=self.config.snp_data_path,
            clinical_data_path=self.config.clinical_data_path,
            labels_path=self.config.labels_path,
            batch_size=self.config.batch_size
        )
        
        # Initialize model
        self.model = HAMF(
            snp_input_dim=self.config.snp_input_dim,
            clinical_input_dim=self.config.clinical_input_dim,
            mri_feature_dim=self.config.mri_feature_dim,
            fusion_dim=self.config.fusion_dim,
            num_classes=self.config.num_classes,
            dropout_rate=self.config.dropout_rate,
            use_batch_norm=self.config.use_batch_norm
        ).to(self.device)
        
        # Initialize trainer
        self.trainer = HAMFTrainer(
            model=self.model,
            config=self.config,
            experiment_dir=self.experiment_dir,
            device=self.device
        )
        
        logger.info("Setup completed successfully")
        
    def train(self):
        """Run training"""
        logger.info("Starting training pipeline...")
        
        try:
            # Get data loaders
            train_loader = self.data_loader.get_train_loader()
            val_loader = self.data_loader.get_test_loader()
            
            # Train model
            self.trainer.train(train_loader, val_loader)
            
            # Save attention visualizations
            self._save_attention_visualizations(val_loader)
            
            # Save final results
            self._save_results()
            
            logger.info("Training completed successfully")
            
        except Exception as e:
            logger.error(f"Training failed: {str(e)}")
            raise
            
    def _save_attention_visualizations(self, val_loader):
        """Generate and save attention visualizations"""
        self.model.eval()
        with torch.no_grad():
            # Get one batch
            batch = next(iter(val_loader))
            data = {k: v.to(self.device) for k, v in batch[0].items()}
            
            # Get attention weights
            _, attention_weights = self.model(
                mri_features=data['mri_features'],
                snp_data=data['snp_data'],
                clinical_data=data['clinical_data']
            )
            
            # Generate visualizations
            self.visualizer.visualize_all(attention_weights)
            
    def _save_results(self):
        """Save final results and metrics"""
        results = {
            'best_val_loss': self.trainer.best_val_loss,
            'training_history': self.trainer.history,
            'config': self.config.__dict__
        }
        
        with open(self.experiment_dir / 'results' / 'final_results.json', 'w') as f:
            json.dump(results, f, indent=4)
            
    def evaluate(self, test_loader=None):
        """Evaluate model on test set"""
        if test_loader is None:
            test_loader = self.data_loader.get_test_loader()
            
        results, attention_weights = self.trainer.evaluate(test_loader)
        
        # Save evaluation results
        with open(self.experiment_dir / 'results' / 'evaluation_results.json', 'w') as f:
            json.dump(results, f, indent=4)
            
        # Generate visualizations for test set
        self.visualizer.visualize_all(attention_weights)
        
        return results