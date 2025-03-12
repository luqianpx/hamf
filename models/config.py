# models/config.py

from dataclasses import dataclass
from typing import List, Optional

@dataclass
class HAMFConfig:
    # Model architecture
    snp_input_dim: int = 450
    clinical_input_dim: int = 42
    mri_feature_dim: int = 2048
    fusion_dim: int = 64
    num_classes: int = 2
    
    # Training
    batch_size: int = 32
    learning_rate: float = 0.001
    num_epochs: int = 100
    dropout_rate: float = 0.3
    use_batch_norm: bool = True
    
    # Optimization
    scheduler_patience: int = 5
    scheduler_factor: float = 0.1
    early_stopping_patience: int = 10
    
    # Data paths
    data_root: str = "data"
    mri_features_path: str = "data/mri_features.npy"
    snp_data_path: str = "data/snp_data.csv"
    clinical_data_path: str = "data/clinical_data.csv"
    labels_path: str = "data/labels.csv"
    
    # Output paths
    checkpoint_dir: str = "checkpoints"
    log_dir: str = "logs"
    visualization_dir: str = "visualizations"
    
    def validate(self):
        """Validate configuration parameters"""
    # Validate dimensions
        assert self.snp_input_dim > 0, "SNP input dimension must be positive"
        assert self.clinical_input_dim > 0, "Clinical input dimension must be positive"
        assert self.mri_feature_dim > 0, "MRI feature dimension must be positive"
        assert self.fusion_dim > 0, "Fusion dimension must be positive"
        assert self.num_classes > 1, "Number of classes must be at least 2"
        
        # Validate training parameters
        assert 0 < self.dropout_rate < 1, "Dropout rate must be between 0 and 1"
        assert self.batch_size > 0, "Batch size must be positive"
        assert self.learning_rate > 0, "Learning rate must be positive"
        assert self.num_epochs > 0, "Number of epochs must be positive"
        
        # Validate file paths
        assert os.path.exists(self.data_root), f"Data root {self.data_root} does not exist"
        assert os.path.exists(self.mri_features_path), f"MRI features path {self.mri_features_path} does not exist"
        assert os.path.exists(self.snp_data_path), f"SNP data path {self.snp_data_path} does not exist"
        assert os.path.exists(self.clinical_data_path), f"Clinical data path {self.clinical_data_path} does not exist"