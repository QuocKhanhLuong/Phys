"""
Configuration file for training and evaluation parameters.
Centralized configuration management for the entire project.
"""

import os
import torch


# =============================================================================
# --- Training Configuration ---
# =============================================================================

# Training hyperparameters
NUM_EPOCHS = 250
LEARNING_RATE = 1e-3
BATCH_SIZE = 24
EARLY_STOP_PATIENCE = 30

# Cross-validation
N_SPLITS = 5

# Model architecture
NUM_CLASSES = 4  # Background, RV, MYO, LV
NUM_SLICES = 5   # Number of 2.5D slices

# Image preprocessing
IMG_SIZE = 224


# =============================================================================
# --- Device Configuration ---
# =============================================================================

DEVICE = torch.device('cuda' if torch.cuda.is_available() else 'cpu')


# =============================================================================
# --- Data Paths ---
# =============================================================================

# Default paths - ACDC dataset location (relative to project root)
PROJECT_ROOT = os.path.dirname(os.path.abspath(__file__))
BASE_DATASET_ROOT = os.path.join(PROJECT_ROOT, 'database')
TRAIN_DATA_PATH = None  # Will be set in train.py if needed
TEST_DATA_PATH = None   # Will be set in train.py if needed


# =============================================================================
# --- Model Configuration ---
# =============================================================================

# ePURE settings
EPURE_BASE_CHANNELS = 32

# UNet++ settings
UNET_CHANNELS = [16, 32, 64, 128, 256]
DEEP_SUPERVISION = True

# Maxwell Solver settings
MAXWELL_HIDDEN_DIM = 32


# =============================================================================
# --- Loss Configuration ---
# =============================================================================

# Loss weights (must sum to 1.0)
INITIAL_LOSS_WEIGHTS = [0.4, 0.4, 0.2]  # [FocalLoss, FocalTverskyLoss, PhysicsLoss]

# Focal Loss parameters
FOCAL_GAMMA = 2.0

# Focal Tversky Loss parameters
TVERSKY_ALPHA = 0.2  # Weight for False Positives
TVERSKY_BETA = 0.8   # Weight for False Negatives
TVERSKY_GAMMA = 4.0 / 3.0

# Class indices for anatomical rules
CLASS_INDICES = {
    'BG': 0,
    'RV': 1,
    'MYO': 2,
    'LV': 3
}


# =============================================================================
# --- Data Augmentation ---
# =============================================================================

# Geometric transformations
ROTATE_LIMIT = 20
ROTATE_PROB = 0.7
HFLIP_PROB = 0.5
ELASTIC_ALPHA = 120
ELASTIC_SIGMA = 120 * 0.05
ELASTIC_PROB = 0.5

# Affine transformations
AFFINE_SCALE = (0.9, 1.1)
AFFINE_TRANSLATE = (-0.0625, 0.0625)
AFFINE_ROTATE = (-15, 15)
AFFINE_PROB = 0.7

# Intensity transformations
BRIGHTNESS_CONTRAST_PROB = 0.5

# Quantum noise injection
QUANTUM_T_MIN = 0.5
QUANTUM_T_MAX = 1.5
PAULI_PROB = {'X': 0.00096, 'Y': 0.00096, 'Z': 0.00096}


# =============================================================================
# --- B1 Map Configuration ---
# =============================================================================

B1_MAP_SAVE_PATH = "acdc_ultimate_b1_map.pth"
B1_N_COILS_RANGE = (4, 8)
B1_STRENGTH_RANGE = (0.5, 1.5)
B1_RADIUS_FACTOR_RANGE = (0.5, 1.5)


# =============================================================================
# --- Evaluation Configuration ---
# =============================================================================

# Test-time augmentation
USE_TTA = True
TTA_TRANSFORMS = ['original', 'hflip']  # Can add 'vflip', 'hvflip'

# Visualization
VIS_NUM_SAMPLES = 50
VIS_COLORMAP = {
    0: 'black',
    1: '#FF0000',  # Red for RV
    2: '#00FF00',  # Green for MYO
    3: '#0000FF'   # Blue for LV
}


# =============================================================================
# --- File Paths ---
# =============================================================================

MODEL_SAVE_PREFIX = "best_model_fold_"
FINAL_MODEL_PATH = "best_model.pth"


# =============================================================================
# --- Logging ---
# =============================================================================

VERBOSE = True
LOG_EVERY_N_EPOCHS = 1

