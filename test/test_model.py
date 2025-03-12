# tests/test_model.py
# author: px
# date: 2021-11-09
# only use for testing the model functions
import unittest
import torch
import numpy as np
from pathlib import Path
import tempfile
import shutil

from models.hamf import HAMF
from models.attention import ScaledDotProductAttention
from models.data_load import MultimodalDataLoader
from models.config import HAMFConfig

class TestHAMF(unittest.TestCase):
    """Test cases for HAMF model"""
    
    @classmethod
    def setUpClass(cls):
        """Set up test environment once for all tests"""
        cls.device = torch.device('cuda' if torch.cuda.is_available() else 'cpu')
        cls.batch_size = 4
        cls.snp_dim = 100
        cls.clinical_dim = 50
        cls.mri_dim = 1024
        cls.fusion_dim = 64
        cls.num_classes = 2
        
        # Create temporary directory for test data
        cls.test_dir = Path(tempfile.mkdtemp())
        
    @classmethod
    def tearDownClass(cls):
        """Clean up after all tests"""
        shutil.rmtree(cls.test_dir)
    
    def setUp(self):
        """Set up test fixtures before each test method"""
        self.model = HAMF(
            snp_input_dim=self.snp_dim,
            clinical_input_dim=self.clinical_dim,
            mri_feature_dim=self.mri_dim,
            fusion_dim=self.fusion_dim,
            num_classes=self.num_classes
        ).to(self.device)
        
        # Create dummy input data
        self.dummy_input = {
            'mri_features': torch.randn(self.batch_size, self.mri_dim).to(self.device),
            'snp_data': torch.randn(self.batch_size, self.snp_dim).to(self.device),
            'clinical_data': torch.randn(self.batch_size, self.clinical_dim).to(self.device)
        }
        
    def test_model_initialization(self):
        """Test model initialization"""
        self.assertIsInstance(self.model, HAMF)
        self.assertEqual(self.model.snp_input_dim, self.snp_dim)
        self.assertEqual(self.model.clinical_input_dim, self.clinical_dim)
        self.assertEqual(self.model.mri_feature_dim, self.mri_dim)
        self.assertEqual(self.model.fusion_dim, self.fusion_dim)
        self.assertEqual(self.model.num_classes, self.num_classes)
    
    def test_forward_pass(self):
        """Test forward pass of the model"""
        self.model.eval()
        with torch.no_grad():
            outputs, attention_weights = self.model(**self.dummy_input)
            
            # Check output shape
            self.assertEqual(outputs.shape, (self.batch_size, self.num_classes))
            
            # Check attention weights
            self.assertIn('mri_snp', attention_weights)
            self.assertIn('mri_clinical', attention_weights)
            self.assertIn('snp_clinical', attention_weights)
            self.assertIn('final', attention_weights)
    
    def test_attention_mechanism(self):
        """Test attention mechanism"""
        self.model.eval()
        with torch.no_grad():
            _, attention_weights = self.model(**self.dummy_input)
            
            # Check attention weight properties
            for attn_name, attn_weight in attention_weights.items():
                # Check if weights sum to approximately 1
                attn_sum = attn_weight.sum(dim=-1)
                self.assertTrue(torch.allclose(attn_sum, 
                                            torch.ones_like(attn_sum), 
                                            atol=1e-6))
                
                # Check if weights are between 0 and 1
                self.assertTrue((attn_weight >= 0).all())
                self.assertTrue((attn_weight <= 1).all())
    
    def test_model_training(self):
        """Test model training step"""
        self.model.train()
        
        # Create dummy labels
        labels = torch.randint(0, self.num_classes, (self.batch_size,)).to(self.device)
        
        # Define loss function and optimizer
        criterion = torch.nn.CrossEntropyLoss()
        optimizer = torch.optim.Adam(self.model.parameters())
        
        # Perform one training step
        optimizer.zero_grad()
        outputs, _ = self.model(**self.dummy_input)
        loss = criterion(outputs, labels)
        loss.backward()
        
        # Check if gradients are computed
        for param in self.model.parameters():
            if param.requires_grad:
                self.assertIsNotNone(param.grad)
    
    def test_model_saving_loading(self):
        """Test model saving and loading"""
        # Save model
        save_path = self.test_dir / 'test_model.pt'
        torch.save(self.model.state_dict(), save_path)
        
        # Load model
        new_model = HAMF(
            snp_input_dim=self.snp_dim,
            clinical_input_dim=self.clinical_dim,
            mri_feature_dim=self.mri_dim,
            fusion_dim=self.fusion_dim,
            num_classes=self.num_classes
        ).to(self.device)
        new_model.load_state_dict(torch.load(save_path))
        
        # Compare outputs
        self.model.eval()
        new_model.eval()
        with torch.no_grad():
            outputs1, attn1 = self.model(**self.dummy_input)
            outputs2, attn2 = new_model(**self.dummy_input)
            
            self.assertTrue(torch.allclose(outputs1, outputs2))
            for k in attn1.keys():
                self.assertTrue(torch.allclose(attn1[k], attn2[k]))
    
    def test_input_validation(self):
        """Test input validation"""
        # Test with wrong input dimensions
        wrong_input = {
            'mri_features': torch.randn(self.batch_size, self.mri_dim + 1).to(self.device),
            'snp_data': torch.randn(self.batch_size, self.snp_dim).to(self.device),
            'clinical_data': torch.randn(self.batch_size, self.clinical_dim).to(self.device)
        }
        
        with self.assertRaises(ValueError):
            self.model(**wrong_input)
    
    def test_batch_normalization(self):
        """Test batch normalization behavior"""
        # Test training mode
        self.model.train()
        out1, _ = self.model(**self.dummy_input)
        
        # Test eval mode
        self.model.eval()
        with torch.no_grad():
            out2, _ = self.model(**self.dummy_input)
        
        # Outputs should be different in train and eval modes
        self.assertFalse(torch.allclose(out1, out2))
    
    def test_dropout(self):
        """Test dropout behavior"""
        self.model.train()
        out1, _ = self.model(**self.dummy_input)
        out2, _ = self.model(**self.dummy_input)
        
        # Outputs should be different due to dropout
        self.assertFalse(torch.allclose(out1, out2))
    
    def test_gradient_flow(self):
        """Test gradient flow through the model"""
        self.model.train()
        outputs, _ = self.model(**self.dummy_input)
        loss = outputs.sum()
        loss.backward()
        
        # Check gradient flow through different components
        for name, param in self.model.named_parameters():
            if param.requires_grad:
                self.assertIsNotNone(param.grad)
                self.assertFalse(torch.all(param.grad == 0))

if __name__ == '__main__':
    unittest.main()