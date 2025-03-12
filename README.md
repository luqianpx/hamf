# hamf
# HAMF: Hierarchical Attention-based Multi-modal Fusion for Medical Analysis

## Overview
HAMF is a deep learning framework that combines multiple medical data modalities (MRI, SNP, and Clinical data) using hierarchical attention mechanisms. The model automatically learns the importance of different modalities and their interactions for improved medical diagnosis and prediction.
### 
please keep in mind that this is only a test version of the model. If you find any bugs, please let me know. I will try to fix them as soon as possible.
And I did not include the data preprocessing and postprocessing in this version.
If you want to know how to preprocessing and postprocessing, please let me know I am willing to share the code. 

## Model Architecture

### Overview
HAMF uses a hierarchical attention mechanism to fuse multiple modalities:
1. **Feature Extraction**: Processes each modality independently
2. **Pairwise Attention**: Computes attention between modality pairs
3. **Hierarchical Fusion**: Combines modalities using learned attention weights

### Attention Mechanism

## Features
- 🧠 Multi-modal fusion of MRI, SNP, and Clinical data
- 🎯 Hierarchical attention mechanism for feature importance
- 📊 Comprehensive visualization tools
- 📈 Extensive evaluation metrics
- 🔄 Cross-validation support
- 🔍 Model interpretability

### Core Modules
- **`hamf.py`**: Defines the HAMF model and trainer.
- **`config.py`**: Defines the configuration settings for the experiment.
- **`train_hamf.py`**: Entry point for training and evaluating the HAMF model.
- **`evaluate_hamf.py`**: Script for evaluating a pre-trained HAMF model.
- **`experiment.py`**: Manages the full experiment pipeline.

### Feature Extraction Modules
- **`snp_feature_extract.py`**: Implements a Stacked Denoising Autoencoder (SDAE) for extracting SNP features.
- **`mri_feature_extract.py`**: Extracts features from MRI data.
- **`clinical_feature_extract.py`**: Implements a deep neural network for clinical feature extraction.

### Attention Mechanisms
- **`attention.py`**: Implements various attention mechanisms including:
- **`attention_visualization.py`**: Provides visualization utilities for attention weights.

### Model Training and Evaluation
- **`trainer.py`**: Contains the HAMFTrainer class for training and validation.
- **`metrics.py`**: Implements the MetricsTracker class to track model performance metrics.
- **`hamf_evaluation.py`**: Provides evaluation utilities for trained models.
- **`hamf_optimization.py`**: Implements hyperparameter optimization using Optuna.
- **`optimization_visualization.py`**: Provides visualization utilities for hyperparameter optimization results.

### Data Handling
- **`data_load.py`**: Defines data loading and preprocessing for multimodal datasets.
- **`utils.py`**: Implements the `ModelCheckpoint` class for saving/loading model checkpoints.

### Testing
- **`test_model.py`**: Unit tests for validating the HAMF model, including:
  - Model initialization
  - Forward pass validation
  - Attention mechanism verification
  - Model training and gradient flow tests
  - Model checkpointing (saving/loading)

## Project Structure
HAMF/
├── models/
│ ├── init.py
│ ├── hamf.py # Main model architecture
│ ├── attention.py # Attention mechanisms
│ ├── trainer.py # Training utilities
│ ├── data_load.py # Data loading utilities
│ ├── metrics.py # Evaluation metrics
│ ├── config.py # Configuration
│ └── visualization.py # Visualization tools
├── tests/
│ ├── init.py
│ ├── test_model.py # Model unit tests
│ └── test_data.py # Data pipeline tests
├── scripts/
│ ├── train.py # Training script
│ ├── evaluate.py # Evaluation script
│ └── visualize.py # Visualization script
├── configs/
│ └── default_config.yaml # Default configuration
├── notebooks/
│ └── examples.ipynb # Usage examples
├── requirements.txt
└── README.md
