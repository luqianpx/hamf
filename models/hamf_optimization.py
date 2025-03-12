# models/hamf_optimization.py
# author: px
# date: 2021-11-09

import optuna
from optuna.trial import Trial
import torch
import torch.nn as nn
from torch.utils.data import DataLoader, SubsetRandomSampler
from sklearn.model_selection import KFold
import numpy as np
import logging
from pathlib import Path
import json
from typing import Dict, List, Tuple
from datetime import datetime

from .hamf import HAMF, HAMFTrainer

logging.basicConfig(level=logging.INFO)
logger = logging.getLogger(__name__)

class HAMFOptimizer:
    """
    Hyperparameter optimization for HAMF using Optuna and Cross-validation
    """
    def __init__(self,
                 dataset,
                 num_folds: int = 5,
                 n_trials: int = 100,
                 study_name: str = "hamf_optimization",
                 storage_name: str = "sqlite:///hamf_optimization.db"):
        """
        Args:
            dataset: Dataset object containing all modalities
            num_folds: Number of cross-validation folds
            n_trials: Number of optimization trials
            study_name: Name for the Optuna study
            storage_name: Path for Optuna storage database
        """
        self.dataset = dataset
        self.num_folds = num_folds
        self.n_trials = n_trials
        
        # Initialize Optuna study
        self.study = optuna.create_study(
            study_name=study_name,
            storage=storage_name,
            direction="minimize",
            load_if_exists=True
        )
        
        # Create results directory
        self.results_dir = Path(f"optimization_results_{datetime.now().strftime('%Y%m%d_%H%M%S')}")
        self.results_dir.mkdir(parents=True, exist_ok=True)
        
    def define_model_parameters(self, trial: Trial) -> Dict:
        """
        Define hyperparameter search space
        """
        params = {
            # Architecture parameters
            "fusion_dim": trial.suggest_int("fusion_dim", 32, 256),
            "snp_hidden_dims": [
                trial.suggest_int("snp_hidden_1", 200, 400),
                trial.suggest_int("snp_hidden_2", 100, 200),
                trial.suggest_int("snp_hidden_3", 50, 100)
            ],
            "clinical_hidden_dim": trial.suggest_int("clinical_hidden_dim", 32, 128),
            "num_attention_heads": trial.suggest_int("num_attention_heads", 2, 8),
            
            # Training parameters
            "batch_size": trial.suggest_int("batch_size", 16, 128),
            "learning_rate": trial.suggest_loguniform("learning_rate", 1e-5, 1e-2),
            "weight_decay": trial.suggest_loguniform("weight_decay", 1e-6, 1e-3),
            "dropout_rate": trial.suggest_float("dropout_rate", 0.1, 0.5),
            
            # Scheduler parameters
            "scheduler_patience": trial.suggest_int("scheduler_patience", 3, 10),
            "scheduler_factor": trial.suggest_float("scheduler_factor", 0.1, 0.5),
            
            # Early stopping parameters
            "early_stopping_patience": trial.suggest_int("early_stopping_patience", 5, 15)
        }
        
        return params
    
    def create_data_loaders(self, train_idx: np.ndarray, val_idx: np.ndarray,
                           batch_size: int) -> Tuple[DataLoader, DataLoader]:
        """
        Create train and validation data loaders for a fold
        """
        train_sampler = SubsetRandomSampler(train_idx)
        val_sampler = SubsetRandomSampler(val_idx)
        
        train_loader = DataLoader(
            self.dataset,
            batch_size=batch_size,
            sampler=train_sampler,
            num_workers=4,
            pin_memory=True
        )
        
        val_loader = DataLoader(
            self.dataset,
            batch_size=batch_size,
            sampler=val_sampler,
            num_workers=4,
            pin_memory=True
        )
        
        return train_loader, val_loader
    
    def objective(self, trial: Trial) -> float:
        """
        Objective function for Optuna optimization
        """
        # Get hyperparameters for this trial
        params = self.define_model_parameters(trial)
        
        # Initialize cross-validation
        kfold = KFold(n_splits=self.num_folds, shuffle=True, random_state=42)
        cv_scores = []
        
        # Cross-validation loop
        for fold, (train_idx, val_idx) in enumerate(kfold.split(self.dataset)):
            logger.info(f"Starting fold {fold + 1}/{self.num_folds}")
            
            # Create data loaders for this fold
            train_loader, val_loader = self.create_data_loaders(
                train_idx, val_idx, params["batch_size"]
            )
            
            # Initialize model with current hyperparameters
            model = HAMF(
                snp_input_dim=450,  # Fixed architecture parameters
                clinical_input_dim=42,
                mri_feature_dim=2048,
                fusion_dim=params["fusion_dim"],
                num_classes=2,
                dropout_rate=params["dropout_rate"]
            )
            
            # Initialize optimizer and criterion
            optimizer = torch.optim.Adam(
                model.parameters(),
                lr=params["learning_rate"],
                weight_decay=params["weight_decay"]
            )
            criterion = nn.CrossEntropyLoss()
            
            # Initialize trainer
            device = torch.device('cuda' if torch.cuda.is_available() else 'cpu')
            trainer = HAMFTrainer(
                model=model,
                optimizer=optimizer,
                criterion=criterion,
                device=device,
                scheduler_patience=params["scheduler_patience"],
                early_stopping_patience=params["early_stopping_patience"],
                checkpoint_dir=self.results_dir / f"trial_{trial.number}" / f"fold_{fold}"
            )
            
            # Train model
            try:
                trainer.train(
                    train_loader=train_loader,
                    val_loader=val_loader,
                    num_epochs=100  # Maximum epochs
                )
                
                # Get best validation loss for this fold
                best_val_loss = min(trainer.history['val_loss'])
                cv_scores.append(best_val_loss)
                
            except Exception as e:
                logger.error(f"Error in trial {trial.number}, fold {fold}: {str(e)}")
                raise optuna.TrialPruned()
            
            # Report intermediate value
            trial.report(best_val_loss, fold)
            
            # Handle pruning based on intermediate results
            if trial.should_prune():
                raise optuna.TrialPruned()
        
        # Calculate mean cross-validation score
        mean_score = np.mean(cv_scores)
        std_score = np.std(cv_scores)
        
        # Save trial results
        trial_results = {
            "trial_number": trial.number,
            "parameters": params,
            "cv_scores": cv_scores,
            "mean_score": float(mean_score),
            "std_score": float(std_score)
        }
        
        with open(self.results_dir / f"trial_{trial.number}_results.json", 'w') as f:
            json.dump(trial_results, f, indent=4)
        
        return mean_score
    
    def optimize(self) -> Dict:
        """
        Run hyperparameter optimization
        """
        logger.info("Starting hyperparameter optimization")
        
        # Run optimization
        self.study.optimize(self.objective, n_trials=self.n_trials)
        
        # Get best trial information
        best_trial = self.study.best_trial
        best_params = best_trial.params
        best_value = best_trial.value
        
        # Save optimization results
        results = {
            "best_parameters": best_params,
            "best_value": best_value,
            "n_trials": self.n_trials,
            "study_name": self.study.study_name,
            "datetime": datetime.now().strftime('%Y-%m-%d %H:%M:%S')
        }
        
        with open(self.results_dir / "optimization_results.json", 'w') as f:
            json.dump(results, f, indent=4)
        
        logger.info(f"Best trial value: {best_value}")
        logger.info("Best parameters:")
        for key, value in best_params.items():
            logger.info(f"    {key}: {value}")
        
        return results
    
    def train_final_model(self, best_params: Dict) -> HAMF:
        """
        Train final model using best parameters
        """
        logger.info("Training final model with best parameters")
        
        # Create data loaders for final training
        train_size = int(0.8 * len(self.dataset))
        val_size = len(self.dataset) - train_size
        train_dataset, val_dataset = torch.utils.data.random_split(
            self.dataset, [train_size, val_size]
        )
        
        train_loader = DataLoader(
            train_dataset,
            batch_size=best_params["batch_size"],
            shuffle=True,
            num_workers=4,
            pin_memory=True
        )
        
        val_loader = DataLoader(
            val_dataset,
            batch_size=best_params["batch_size"],
            shuffle=False,
            num_workers=4,
            pin_memory=True
        )
        
        # Initialize final model
        final_model = HAMF(
            snp_input_dim=450,
            clinical_input_dim=42,
            mri_feature_dim=2048,
            fusion_dim=best_params["fusion_dim"],
            num_classes=2,
            dropout_rate=best_params["dropout_rate"]
        )
        
        # Initialize optimizer and criterion
        optimizer = torch.optim.Adam(
            final_model.parameters(),
            lr=best_params["learning_rate"],
            weight_decay=best_params["weight_decay"]
        )
        criterion = nn.CrossEntropyLoss()
        
        # Train final model
        device = torch.device('cuda' if torch.cuda.is_available() else 'cpu')
        trainer = HAMFTrainer(
            model=final_model,
            optimizer=optimizer,
            criterion=criterion,
            device=device,
            scheduler_patience=best_params["scheduler_patience"],
            early_stopping_patience=best_params["early_stopping_patience"],
            checkpoint_dir=self.results_dir / "final_model"
        )
        
        trainer.train(
            train_loader=train_loader,
            val_loader=val_loader,
            num_epochs=100
        )
        
        return final_model

# Example usage
def main():
    # Initialize dataset
    dataset = YourDataset()  # Your custom dataset class
    
    # Initialize optimizer
    hamf_optimizer = HAMFOptimizer(
        dataset=dataset,
        num_folds=5,
        n_trials=100
    )
    
    # Run optimization
    best_results = hamf_optimizer.optimize()
    
    # Train final model with best parameters
    final_model = hamf_optimizer.train_final_model(best_results["best_parameters"])
    
    logger.info("Optimization and final training completed")

if __name__ == "__main__":
    main()