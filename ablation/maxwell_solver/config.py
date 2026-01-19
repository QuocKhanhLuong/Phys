"""
Configuration for Maxwell Solver Ablation Study
"""
import os
import sys
from pathlib import Path

# Add project root to path
PROJECT_ROOT = Path(__file__).parent.parent.parent
sys.path.insert(0, str(PROJECT_ROOT))


# =============================================================================
# MAXWELL SOLVER CONFIGURATIONS
# =============================================================================

MAXWELL_CONFIGS = {
    "Standard": {
        "name": "Standard (No Maxwell)",
        "use_maxwell": False,
        "description": "UNet++ decoder without Maxwell Solver"
    },
    "Physics": {
        "name": "Physics (Maxwell)",
        "use_maxwell": True,
        "description": "PIE-UNet decoder with Maxwell Solver (original)"
    }
}


# =============================================================================
# TRAINING CONFIGURATION (Same as train_acdc.py)
# =============================================================================

TRAINING_CONFIG = {
    "num_epochs": 250,
    "learning_rate": 1e-3,
    "weight_decay": 1e-4,
    "batch_size": 24,
    "early_stopping_patience": 30,
    "num_classes": 4,
    "num_slices": 5,
    "img_size": 224,
}


# =============================================================================
# DATA CONFIGURATION
# =============================================================================

DATA_CONFIG = {
    "preprocessed_dir": PROJECT_ROOT / "preprocessed_data" / "ACDC",
    "train_dir": PROJECT_ROOT / "preprocessed_data" / "ACDC" / "training",
    "test_dir": PROJECT_ROOT / "preprocessed_data" / "ACDC" / "testing",
}


# =============================================================================
# OUTPUT CONFIGURATION
# =============================================================================

OUTPUT_CONFIG = {
    "results_dir": Path(__file__).parent / "results",
    "weights_dir": Path(__file__).parent / "weights",
}

# Create directories
OUTPUT_CONFIG["results_dir"].mkdir(parents=True, exist_ok=True)
OUTPUT_CONFIG["weights_dir"].mkdir(parents=True, exist_ok=True)


# =============================================================================
# MEASUREMENT CONFIGURATION
# =============================================================================

MEASURE_CONFIG = {
    "input_size": 224,
    "num_warmup_runs": 10,
    "num_measure_runs": 100,
}


# =============================================================================
# PRETRAINED WEIGHTS (Physics uses original trained model)
# =============================================================================

PRETRAINED_WEIGHTS = {
    "Physics": PROJECT_ROOT / "weights" / "best_model_acdc_no_anatomical.pth"
}
