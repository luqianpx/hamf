# test_hamf.py
import argparse
import logging
from pathlib import Path
import yaml
import torch
from typing import Dict

from models.hamf import HAMF
from models.data_load import MultimodalDataLoader
from models.utils import ModelCheckpoint
from models.config import HAMFConfig
from models.hamf_evaluation import HAMFEvaluator

logging.basicConfig(level=logging.INFO)
logger = logging.getLogger(__name__)

class ModelTest:
    """
    Wrapper class for model testing using HAMFEvaluator
    """
    def __init__(self, 
                 model_path: str, 
                 config_path: str,
                 output_dir: str):
        """
        Initialize test wrapper
        
        Args:
            model_path: Path to saved model checkpoint
            config_path: Path to configuration file
            output_dir: Directory to save test results
        """
        self.model_path = Path(model_path)
        self.output_dir = Path(output_dir)
        self.output_dir.mkdir(parents=True, exist_ok=True)
        
        # Load configuration
        self.config = self._load_config(config_path)
        
        # Setup device
        self.device = torch.device('cuda' if torch.cuda.is_available() else 'cpu')
        
        # Initialize components
        self.model = self._load_model()
        self.data_loader = self._setup_data_loader()
        
    def _load_config(self, config_path: str) -> HAMFConfig:
        """Load model configuration"""
        with open(config_path) as f:
            config_dict = yaml.safe_load(f)
        return HAMFConfig(**config_dict)
    
    def _load_model(self) -> HAMF:
        """Load trained model from checkpoint"""
        logger.info(f"Loading model from {self.model_path}")
        
        # Initialize model
        model = HAMF(
            snp_input_dim=self.config.snp_input_dim,
            clinical_input_dim=self.config.clinical_input_dim,
            mri_feature_dim=self.config.mri_feature_dim,
            fusion_dim=self.config.fusion_dim,
            num_classes=self.config.num_classes,
            dropout_rate=self.config.dropout_rate,
            use_batch_norm=self.config.use_batch_norm
        ).to(self.device)
        
        # Load weights
        checkpoint = ModelCheckpoint(self.model_path.parent)
        checkpoint.load_model(model, filename=self.model_path.name)
        
        return model
    
    def _setup_data_loader(self) -> MultimodalDataLoader:
        """Setup data loader for test data"""
        return MultimodalDataLoader(
            mri_features_path=self.config.test_mri_features_path,
            snp_data_path=self.config.test_snp_data_path,
            clinical_data_path=self.config.test_clinical_data_path,
            labels_path=self.config.test_labels_path,
            batch_size=self.config.batch_size
        )
    
    def run_test(self) -> Dict:
        """
        Run complete test evaluation using HAMFEvaluator
        
        Returns:
            Dictionary containing test results
        """
        logger.info("Starting model evaluation...")
        
        # Get test data loader
        test_loader = self.data_loader.get_test_loader()
        
        # Initialize evaluator
        evaluator = HAMFEvaluator(
            model=self.model,
            test_loader=test_loader,
            device=self.device,
            save_dir=self.output_dir
        )
        
        # Run evaluation
        try:
            results = evaluator.evaluate()
            logger.info("\nTest Results Summary:")
            logger.info(f"Test Accuracy: {results['accuracy']:.4f}")
            logger.info(f"ROC AUC: {results['roc_auc']:.4f}")
            logger.info(f"F1 Score: {results['f1_score']:.4f}")
            
            # Generate visualizations
            evaluator.plot_confusion_matrix()
            evaluator.plot_roc_curve()
            
            # Analyze attention patterns
            attention_analysis = evaluator.analyze_attention_patterns()
            
            # Analyze error cases
            error_analysis = evaluator.analyze_error_cases()
            
            logger.info(f"Results and visualizations saved to {self.output_dir}")
            
            return results
            
        except Exception as e:
            logger.error(f"Evaluation failed: {str(e)}")
            raise

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
        # Initialize test wrapper
        tester = ModelTest(
            model_path=args.model_path,
            config_path=args.config,
            output_dir=args.output_dir
        )
        
        # Run test evaluation
        results = tester.run_test()
        
    except Exception as e:
        logger.error(f"Testing failed: {str(e)}")
        raise

if __name__ == "__main__":
    main()