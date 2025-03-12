# models/hamf_evaluation.py
# author: px
# date: 2021-11-09

import torch
import torch.nn as nn
from torch.utils.data import DataLoader
import numpy as np
import pandas as pd
from sklearn.metrics import (
    accuracy_score, precision_recall_fscore_support,
    roc_auc_score, confusion_matrix, classification_report,
    roc_curve, precision_recall_curve
)
import matplotlib.pyplot as plt
import seaborn as sns
from pathlib import Path
import json
import logging
from typing import Dict, List, Tuple, Optional
import time

from .hamf import HAMF

logging.basicConfig(level=logging.INFO)
logger = logging.getLogger(__name__)

class HAMFEvaluator:
    """
    Comprehensive evaluation utilities for HAMF model
    """
    def __init__(self,
                 model: HAMF,
                 test_loader: DataLoader,
                 device: torch.device,
                 save_dir: str = 'evaluation_results'):
        """
        Args:
            model: Trained HAMF model
            test_loader: DataLoader for test data
            device: Device to run evaluation on
            save_dir: Directory to save evaluation results
        """
        self.model = model.to(device)
        self.test_loader = test_loader
        self.device = device
        self.save_dir = Path(save_dir)
        self.save_dir.mkdir(parents=True, exist_ok=True)
        
        self.results = {}
        self.attention_weights = {}
        
    def evaluate(self) -> Dict:
        """
        Perform comprehensive evaluation of the model
        
        Returns:
            Dictionary containing all evaluation metrics
        """
        logger.info("Starting model evaluation...")
        self.model.eval()
        
        # Initialize lists to store results
        all_predictions = []
        all_probabilities = []
        all_labels = []
        all_attention_weights = []
        all_features = []
        
        test_loss = 0
        criterion = nn.CrossEntropyLoss()
        
        start_time = time.time()
        
        with torch.no_grad():
            for batch_idx, (mri_feat, snp, clinical, labels) in enumerate(self.test_loader):
                # Move data to device
                mri_feat = mri_feat.to(self.device)
                snp = snp.to(self.device)
                clinical = clinical.to(self.device)
                labels = labels.to(self.device)
                
                # Forward pass
                predictions, attention_weights = self.model(
                    mri_feat, snp, clinical,
                    return_attention=True
                )
                
                # Calculate loss
                loss = criterion(predictions, labels)
                test_loss += loss.item()
                
                # Store results
                probabilities = torch.softmax(predictions, dim=1)
                pred_labels = torch.argmax(predictions, dim=1)
                
                all_predictions.extend(pred_labels.cpu().numpy())
                all_probabilities.extend(probabilities[:, 1].cpu().numpy())
                all_labels.extend(labels.cpu().numpy())
                all_attention_weights.append({
                    k: v.cpu().numpy() for k, v in attention_weights.items()
                })
                
        # Calculate metrics
        self.results = {
            'test_loss': test_loss / len(self.test_loader),
            'accuracy': accuracy_score(all_labels, all_predictions),
            'roc_auc': roc_auc_score(all_labels, all_probabilities),
            'confusion_matrix': confusion_matrix(all_labels, all_predictions).tolist(),
            'classification_report': classification_report(all_labels, all_predictions, output_dict=True)
        }
        
        # Store raw predictions and attention weights
        self.results['predictions'] = {
            'true_labels': all_labels,
            'predicted_labels': all_predictions,
            'probabilities': all_probabilities
        }
        self.attention_weights = all_attention_weights
        
        # Calculate evaluation time
        self.results['evaluation_time'] = time.time() - start_time
        
        # Save results
        self.save_results()
        
        logger.info("Evaluation completed successfully")
        return self.results
    
    def plot_confusion_matrix(self, save: bool = True) -> plt.Figure:
        """Plot confusion matrix"""
        cm = np.array(self.results['confusion_matrix'])
        
        plt.figure(figsize=(8, 6))
        sns.heatmap(
            cm,
            annot=True,
            fmt='d',
            cmap='Blues',
            xticklabels=['Class 0', 'Class 1'],
            yticklabels=['Class 0', 'Class 1']
        )
        plt.title('Confusion Matrix')
        plt.ylabel('True Label')
        plt.xlabel('Predicted Label')
        
        if save:
            plt.savefig(self.save_dir / 'confusion_matrix.png', dpi=300, bbox_inches='tight')
        
        return plt.gcf()
    
    def plot_roc_curve(self, save: bool = True) -> plt.Figure:
        """Plot ROC curve"""
        fpr, tpr, _ = roc_curve(
            self.results['predictions']['true_labels'],
            self.results['predictions']['probabilities']
        )
        
        plt.figure(figsize=(8, 6))
        plt.plot(fpr, tpr, label=f'ROC (AUC = {self.results["roc_auc"]:.3f})')
        plt.plot([0, 1], [0, 1], 'k--')
        plt.xlim([0.0, 1.0])
        plt.ylim([0.0, 1.05])
        plt.xlabel('False Positive Rate')
        plt.ylabel('True Positive Rate')
        plt.title('Receiver Operating Characteristic (ROC) Curve')
        plt.legend(loc="lower right")
        
        if save:
            plt.savefig(self.save_dir / 'roc_curve.png', dpi=300, bbox_inches='tight')
        
        return plt.gcf()
    
    def analyze_attention_patterns(self, save: bool = True) -> Dict:
        """Analyze attention patterns across test set"""
        attention_analysis = {}
        
        # Aggregate attention weights
        for key in self.attention_weights[0].keys():
            weights = np.concatenate([batch[key] for batch in self.attention_weights])
            attention_analysis[key] = {
                'mean': np.mean(weights, axis=0),
                'std': np.std(weights, axis=0),
                'max': np.max(weights, axis=0),
                'min': np.min(weights, axis=0)
            }
        
        if save:
            # Plot attention distributions
            for key, values in attention_analysis.items():
                plt.figure(figsize=(10, 6))
                sns.boxplot(data=pd.DataFrame([batch[key] for batch in self.attention_weights]))
                plt.title(f'Attention Weight Distribution: {key}')
                plt.xticks(rotation=45)
                plt.tight_layout()
                plt.savefig(self.save_dir / f'attention_{key}_distribution.png', dpi=300)
                plt.close()
        
        return attention_analysis
    
    def analyze_error_cases(self) -> pd.DataFrame:
        """Analyze cases where model made incorrect predictions"""
        predictions = self.results['predictions']
        error_indices = np.where(
            np.array(predictions['true_labels']) != 
            np.array(predictions['predicted_labels'])
        )[0]
        
        error_analysis = pd.DataFrame({
            'true_label': np.array(predictions['true_labels'])[error_indices],
            'predicted_label': np.array(predictions['predicted_labels'])[error_indices],
            'probability': np.array(predictions['probabilities'])[error_indices]
        })
        
        error_analysis.to_csv(self.save_dir / 'error_analysis.csv', index=False)
        return error_analysis
    
    def save_results(self):
        """Save all evaluation results"""
        # Save metrics
        with open(self.save_dir / 'evaluation_metrics.json', 'w') as f:
            json.dump(self.results, f, indent=4)
        
        # Generate and save plots
        self.plot_confusion_matrix()
        self.plot_roc_curve()
        
        # Analyze attention patterns
        attention_analysis = self.analyze_attention_patterns()
        with open(self.save_dir / 'attention_analysis.json', 'w') as f:
            json.dump(attention_analysis, f, indent=4)
        
        # Analyze error cases
        self.analyze_error_cases()
        
        # Create summary report
        self.create_evaluation_report()
    
    def create_evaluation_report(self):
        """Create comprehensive evaluation report"""
        report_path = self.save_dir / 'evaluation_report.html'
        
        with open(report_path, 'w') as f:
            f.write(f"""
            <html>
            <head>
                <title>HAMF Evaluation Report</title>
                <style>
                    body {{ font-family: Arial, sans-serif; margin: 20px; }}
                    .metric {{ margin: 10px 0; }}
                    .plot {{ margin: 20px 0; }}
                </style>
            </head>
            <body>
                <h1>HAMF Model Evaluation Report</h1>
                
                <h2>Performance Metrics</h2>
                <div class="metric">
                    <p>Test Loss: {self.results['test_loss']:.4f}</p>
                    <p>Accuracy: {self.results['accuracy']:.4f}</p>
                    <p>ROC AUC: {self.results['roc_auc']:.4f}</p>
                </div>
                
                <h2>Confusion Matrix</h2>
                <div class="plot">
                    <img src="confusion_matrix.png" alt="Confusion Matrix">
                </div>
                
                <h2>ROC Curve</h2>
                <div class="plot">
                    <img src="roc_curve.png" alt="ROC Curve">
                </div>
                
                <h2>Attention Analysis</h2>
                <div class="plot">
                    <img src="attention_mri_snp_distribution.png" alt="MRI-SNP Attention">
                    <img src="attention_mri_clinical_distribution.png" alt="MRI-Clinical Attention">
                    <img src="attention_snp_clinical_distribution.png" alt="SNP-Clinical Attention">
                </div>
                
                <h2>Classification Report</h2>
                <pre>{json.dumps(self.results['classification_report'], indent=4)}</pre>
                
                <p>Evaluation completed in {self.results['evaluation_time']:.2f} seconds</p>
            </body>
            </html>
            """)


def main():
    parser = argparse.ArgumentParser(description='Test HAMF model')
    parser.add_argument('--model_path', type=str, required=True,
                      help='Path to saved model checkpoint')
    parser.add_argument('--config', type=str, required=True,
                      help='Path to configuration file')
    parser.add_argument('--output_dir', type=str, default='test_results',
                      help='Directory to save test results')
    

                    
                      
    
    args = parser.parse_args()
    
    try:
        tester = HAMFTest(
            model_path=args.model_path,
            config_path=args.config,
            output_dir=args.output_dir
        )
        results = tester.run_test()
        
    except Exception as e:
        logger.error(f"Testing failed: {str(e)}")
        raise

if __name__ == "__main__":
    main()
if __name__ == "__main__":
    main()