# data_load.py

# This file is used to load the data from the data folder and create a dataset and a data loader\
# this is only used for testing the model functions
# author: px
# date: 2021-11-09

import torch
from torch.utils.data import Dataset, DataLoader, SubsetRandomSampler
import numpy as np
import pandas as pd
from sklearn.model_selection import train_test_split
from sklearn.preprocessing import StandardScaler
import os
from pathlib import Path
import logging
from typing import Optional, Tuple, List
import sys

# Set up logging configuration
logging.basicConfig(
    level=logging.INFO,
    format='%(asctime)s - %(name)s - %(levelname)s - %(message)s',
    handlers=[
        logging.FileHandler('data_loading.log'),
        logging.StreamHandler(sys.stdout)
    ]
)
logger = logging.getLogger(__name__)

class DataLoadingError(Exception):
    """Custom exception for data loading errors"""
    pass

class MultimodalDataset(Dataset):
    """
    Custom Dataset class for handling multimodal data (MRI, SNP, and Clinical features)
    """
    def __init__(self, 
                 mri_features_path: str,
                 snp_data_path: str,
                 clinical_data_path: str,
                 labels_path: Optional[str] = None,
                 transform = None):
        """
        Initialize the dataset.
        
        Args:
            mri_features_path (str): Path to pre-extracted MRI features (.npy file)
            snp_data_path (str): Path to SNP data (.csv file)
            clinical_data_path (str): Path to clinical data (.csv file)
            labels_path (str, optional): Path to labels file
            transform (callable, optional): Optional transform to be applied on the data
        
        Raises:
            DataLoadingError: If there are issues loading or processing the data
            FileNotFoundError: If any of the required files are not found
            ValueError: If data dimensions or values are invalid
        """
        try:
            logger.info("Initializing MultimodalDataset")
            self._check_file_paths(mri_features_path, snp_data_path, clinical_data_path, labels_path)
            
            # Load MRI features
            logger.info(f"Loading MRI features from {mri_features_path}")
            try:
                self.mri_features = np.load(mri_features_path)
                logger.info(f"MRI features shape: {self.mri_features.shape}")
            except Exception as e:
                raise DataLoadingError(f"Error loading MRI features: {str(e)}")
            
            # Load SNP data
            logger.info(f"Loading SNP data from {snp_data_path}")
            try:
                self.snp_data = pd.read_csv(snp_data_path).values
                logger.info(f"SNP data shape: {self.snp_data.shape}")
            except Exception as e:
                raise DataLoadingError(f"Error loading SNP data: {str(e)}")
            
            # Load clinical data
            logger.info(f"Loading clinical data from {clinical_data_path}")
            try:
                self.clinical_data = pd.read_csv(clinical_data_path).values
                logger.info(f"Clinical data shape: {self.clinical_data.shape}")
            except Exception as e:
                raise DataLoadingError(f"Error loading clinical data: {str(e)}")
            
            # Load labels if provided
            self.labels = None
            if labels_path is not None:
                logger.info(f"Loading labels from {labels_path}")
                try:
                    self.labels = pd.read_csv(labels_path).values
                    logger.info(f"Labels shape: {self.labels.shape}")
                except Exception as e:
                    raise DataLoadingError(f"Error loading labels: {str(e)}")
            
            self.transform = transform
            
            # Verify data integrity
            self._verify_data()
            
            # Initialize and apply scalers
            self._initialize_scalers()
            
            logger.info("MultimodalDataset initialization completed successfully")
            
        except Exception as e:
            logger.error(f"Error during dataset initialization: {str(e)}")
            raise

    def _check_file_paths(self, *paths: Tuple[str]) -> None:
        """
        Verify that all provided file paths exist.
        
        Args:
            *paths: Variable number of file paths to check
            
        Raises:
            FileNotFoundError: If any required file is not found
        """
        for path in paths:
            if path is not None and not os.path.exists(path):
                logger.error(f"File not found: {path}")
                raise FileNotFoundError(f"File not found: {path}")

    def _verify_data(self) -> None:
        """
        Verify that all modalities have the same number of samples and valid values.
        
        Raises:
            ValueError: If data dimensions don't match or contains invalid values
        """
        try:
            n_samples = len(self.mri_features)
            
            # Check dimensions
            if len(self.snp_data) != n_samples:
                raise ValueError(f"SNP data samples ({len(self.snp_data)}) doesn't match MRI samples ({n_samples})")
            
            if len(self.clinical_data) != n_samples:
                raise ValueError(f"Clinical data samples ({len(self.clinical_data)}) doesn't match MRI samples ({n_samples})")
            
            if self.labels is not None and len(self.labels) != n_samples:
                raise ValueError(f"Labels samples ({len(self.labels)}) doesn't match MRI samples ({n_samples})")
            
            # Check for invalid values
            self._check_invalid_values(self.mri_features, "MRI features")
            self._check_invalid_values(self.snp_data, "SNP data")
            self._check_invalid_values(self.clinical_data, "Clinical data")
            
            if self.labels is not None:
                self._check_invalid_values(self.labels, "Labels")
            
            logger.info("Data verification completed successfully")
            
        except Exception as e:
            logger.error(f"Data verification failed: {str(e)}")
            raise

    def _check_invalid_values(self, data: np.ndarray, name: str) -> None:
        """
        Check for invalid values in the data.
        
        Args:
            data: numpy array to check
            name: name of the data for logging purposes
            
        Raises:
            ValueError: If invalid values are found
        """
        if np.isnan(data).any():
            logger.error(f"NaN values found in {name}")
            raise ValueError(f"NaN values found in {name}")
        if np.isinf(data).any():
            logger.error(f"Infinite values found in {name}")
            raise ValueError(f"Infinite values found in {name}")

    def _initialize_scalers(self) -> None:
        """
        Initialize and apply StandardScaler to each modality.
        """
        try:
            logger.info("Initializing data scalers")
            self.mri_scaler = StandardScaler()
            self.snp_scaler = StandardScaler()
            self.clinical_scaler = StandardScaler()
            
            self.mri_features = self.mri_scaler.fit_transform(self.mri_features)
            self.snp_data = self.snp_scaler.fit_transform(self.snp_data)
            self.clinical_data = self.clinical_scaler.fit_transform(self.clinical_data)
            
            logger.info("Data scaling completed successfully")
            
        except Exception as e:
            logger.error(f"Error during data scaling: {str(e)}")
            raise DataLoadingError(f"Error during data scaling: {str(e)}")

    def __len__(self) -> int:
        """Return the total number of samples"""
        return len(self.mri_features)

    def __getitem__(self, idx: int) -> Tuple[torch.Tensor, ...]:
        """
        Get a sample from the dataset.
        
        Args:
            idx: Index of the sample to get
            
        Returns:
            Tuple of tensors containing the sample data
        """
        try:
            # Get data for all modalities
            mri_feat = torch.FloatTensor(self.mri_features[idx])
            snp = torch.FloatTensor(self.snp_data[idx])
            clinical = torch.FloatTensor(self.clinical_data[idx])
            
            # Apply transforms if any
            if self.transform:
                mri_feat = self.transform(mri_feat)
                snp = self.transform(snp)
                clinical = self.transform(clinical)
            
            # Return with or without labels
            if self.labels is not None:
                label = torch.FloatTensor(self.labels[idx])
                return mri_feat, snp, clinical, label
            return mri_feat, snp, clinical
            
        except Exception as e:
            logger.error(f"Error retrieving sample at index {idx}: {str(e)}")
            raise

class MultimodalDataLoader:
    """
    Data loader class for handling the multimodal dataset with train/test splitting
    """
    def __init__(self,
                 mri_features_path: str,
                 snp_data_path: str,
                 clinical_data_path: str,
                 labels_path: Optional[str] = None,
                 batch_size: int = 32,
                 test_size: float = 0.2,
                 random_seed: int = 42):
        """
        Initialize the data loader.
        """
        try:
            logger.info("Initializing MultimodalDataLoader")
            
            if not 0 < test_size < 1:
                raise ValueError(f"Invalid test_size: {test_size}. Must be between 0 and 1")
            
            if batch_size <= 0:
                raise ValueError(f"Invalid batch_size: {batch_size}. Must be positive")
            
            self.batch_size = batch_size
            self.random_seed = random_seed
            
            # Create dataset
            self.dataset = MultimodalDataset(
                mri_features_path=mri_features_path,
                snp_data_path=snp_data_path,
                clinical_data_path=clinical_data_path,
                labels_path=labels_path
            )
            
            # Create train/test split indices
            self._create_split_indices(test_size)
            
            logger.info("MultimodalDataLoader initialization completed successfully")
            
        except Exception as e:
            logger.error(f"Error initializing MultimodalDataLoader: {str(e)}")
            raise

    def _create_split_indices(self, test_size: float) -> None:
        """
        Create train/test split indices.
        """
        try:
            dataset_size = len(self.dataset)
            indices = list(range(dataset_size))
            
            self.train_indices, self.test_indices = train_test_split(
                indices,
                test_size=test_size,
                random_state=self.random_seed
            )
            
            logger.info(f"Data split created: {len(self.train_indices)} train samples, "
                       f"{len(self.test_indices)} test samples")
            
        except Exception as e:
            logger.error(f"Error creating data split: {str(e)}")
            raise

    def get_train_loader(self) -> DataLoader:
        """Get train data loader"""
        try:
            train_sampler = SubsetRandomSampler(self.train_indices)
            train_loader = DataLoader(
                self.dataset,
                batch_size=self.batch_size,
                sampler=train_sampler,
                num_workers=4,
                pin_memory=True
            )
            logger.info("Train loader created successfully")
            return train_loader
        except Exception as e:
            logger.error(f"Error creating train loader: {str(e)}")
            raise

    def get_test_loader(self) -> DataLoader:
        """Get test data loader"""
        try:
            test_sampler = SubsetRandomSampler(self.test_indices)
            test_loader = DataLoader(
                self.dataset,
                batch_size=self.batch_size,
                sampler=test_sampler,
                num_workers=4,
                pin_memory=True
            )
            logger.info("Test loader created successfully")
            return test_loader
        except Exception as e:
            logger.error(f"Error creating test loader: {str(e)}")
            raise

def main():
   
    try:
        # Define paths
        data_root = "path/to/your/data"
        mri_features_path = os.path.join(data_root, "mri_features.npy")
        snp_data_path = os.path.join(data_root, "snp_data.csv")
        clinical_data_path = os.path.join(data_root, "clinical_data.csv")
        labels_path = os.path.join(data_root, "labels.csv")
        
        # Initialize data loader
        data_loader = MultimodalDataLoader(
            mri_features_path=mri_features_path,
            snp_data_path=snp_data_path,
            clinical_data_path=clinical_data_path,
            labels_path=labels_path,
            batch_size=32
        )
        
        # Get train and test loaders
        train_loader = data_loader.get_train_loader()
        test_loader = data_loader.get_test_loader()
        
        # Example of iterating through the data
        for batch_idx, (mri_feat, snp, clinical, labels) in enumerate(train_loader):
            logger.info(f"Batch {batch_idx}")
            logger.info(f"MRI features shape: {mri_feat.shape}")
            logger.info(f"SNP data shape: {snp.shape}")
            logger.info(f"Clinical data shape: {clinical.shape}")
            logger.info(f"Labels shape: {labels.shape}")
            break
            
    except Exception as e:
        logger.error(f"Error in main: {str(e)}")
        raise

if __name__ == "__main__":
    main()