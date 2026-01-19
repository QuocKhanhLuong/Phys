import os
from pathlib import Path
from typing import Dict, Any

# Project root directory
PROJECT_ROOT = Path(__file__).parent.parent
SRC_ROOT = Path(__file__).parent


WEIGHTS_DIR = PROJECT_ROOT / "weights"
B1_MAPS_DIR = PROJECT_ROOT / "b1_maps"
RESULTS_DIR = PROJECT_ROOT / "results"

# Dataset Configuration
DATA_DIR = PROJECT_ROOT / "data"
ACDC_RAW_DIR = DATA_DIR / "ACDC"
ATLAS_RAW_DIR = DATA_DIR / "ATLAS"
MNM_RAW_DIR = DATA_DIR / "MnM"
SCD_RAW_DIR = DATA_DIR / "SCD"
EMIDEC_RAW_DIR = DATA_DIR / "EMIDEC"

# Preprocessed data directories
PREPROCESSED_DIR = PROJECT_ROOT / "preprocessed_data"
ACDC_PREPROCESSED_DIR = PREPROCESSED_DIR / "ACDC"
ATLAS_PREPROCESSED_DIR = PREPROCESSED_DIR / "atlas_npy"
MNM_PREPROCESSED_DIR = PREPROCESSED_DIR / "mnm"
SCD_PREPROCESSED_DIR = PREPROCESSED_DIR / "SCD"
EMIDEC_PREPROCESSED_DIR = PREPROCESSED_DIR / "EMIDEC"

# Cache directories
CACHE_DIR = PROJECT_ROOT / "cache"

# Model Configuration

# Model architecture
MODEL_CONFIG = {
    "input_channels": 5,  # 2.5D: 5 slices
    "num_classes": 4,     # Background + 3 foreground classes
    "base_filters": 64,
    "depth": 4,
    "dropout_rate": 0.1,
    "batch_norm": True,
    "activation": "relu"
}

# Training Configuration

TRAINING_CONFIG = {
    "num_epochs": 250,
    "learning_rate": 5e-5,
    "weight_decay": 1e-4,
    "batch_size": 24,
    "early_stopping_patience": 30,
    "save_every_n_epochs": 10,
    "validate_every_n_epochs": 5,
    "gradient_clip_norm": 1.0
}

# =============================================================================
# Data Loading Configuration
# =============================================================================

DATA_CONFIG = {
    "num_workers": 0,  # Must be 0 for ePURE augmentation
    "pin_memory": True,
    "image_size": 224,
    "num_slices": 5,
    "augmentation_prob": 0.5
}

# =============================================================================
# Loss Function Configuration
# =============================================================================

LOSS_CONFIG = {
    "dice_weight": 0.4,
    "focal_weight": 0.4,
    "physics_weight": 0.2,
    "focal_alpha": 0.25,
    "focal_gamma": 2.0,
    "smooth": 1e-6
}

# =============================================================================
# Dataset-Specific Configuration
# =============================================================================

ACDC_CONFIG = {
    "name": "ACDC",
    "class_map": {
        0: "Background",
        1: "Right Ventricle",
        2: "Myocardium",
        3: "Left Ventricle"
    },
    "num_classes": 4
}

MNM_CONFIG = {
    "name": "M&M",
    "class_map": {
        0: "Background",
        1: "Left Ventricle",
        2: "Myocardium",
        3: "Right Ventricle"
    },
    "num_classes": 4
}

SCD_CONFIG = {
    "name": "SCD",
    "class_map": {
        0: "Background",
        1: "Left Ventricle"
    },
    "num_classes": 2
}

EMIDEC_CONFIG = {
    "name": "EMIDEC",
    "class_map": {
        0: "Background",
        1: "Cavity",
        2: "Myocardium",
        3: "Infarction",
        4: "NoReflow"
    },
    "num_classes": 5
}

ATLAS_CONFIG = {
    "name": "ATLAS",
    "class_map": {
        0: "Background",
        1: "Lesion"
    },
    "num_classes": 2
}

# =============================================================================
# Experiment Configuration
# =============================================================================

EXPERIMENT_CONFIG = {
    "results_dir": RESULTS_DIR,
    "weights_dir": WEIGHTS_DIR,
    "b1_maps_dir": B1_MAPS_DIR,
    "visualizations_dir": RESULTS_DIR / "visualizations",
    "log_dir": RESULTS_DIR / "logs"
}

# =============================================================================
# Hardware Configuration
# =============================================================================

HARDWARE_CONFIG = {
    "device": "cuda" if os.environ.get("CUDA_VISIBLE_DEVICES") else "cpu",
    "mixed_precision": True,
    "compile_model": False,
    "num_gpus": 1
}

# =============================================================================
# Utility Functions
# =============================================================================

def get_dataset_config(dataset_name: str) -> Dict[str, Any]:
    """Get configuration for a specific dataset."""
    config_map = {
        "acdc": ACDC_CONFIG,
        "mnm": MNM_CONFIG,
        "scd": SCD_CONFIG,
        "emidec": EMIDEC_CONFIG,
        "atlas": ATLAS_CONFIG,
    }
    
    if dataset_name.lower() not in config_map:
        raise ValueError(f"Unknown dataset: {dataset_name}. Available: {list(config_map.keys())}")
    
    return config_map[dataset_name.lower()]

def get_experiment_path(experiment_name: str) -> Path:
    """Get the path for a specific experiment."""
    return EXPERIMENT_CONFIG["results_dir"] / experiment_name

def create_directories():
    """Create all necessary directories."""
    directories = [
        DATA_DIR,
        PREPROCESSED_DIR,
        CACHE_DIR,
        WEIGHTS_DIR,
        B1_MAPS_DIR,
        RESULTS_DIR,
        EXPERIMENT_CONFIG["visualizations_dir"],
        EXPERIMENT_CONFIG["log_dir"]
    ]
    
    for directory in directories:
        directory.mkdir(parents=True, exist_ok=True)

# =============================================================================
# Legacy Support (for backward compatibility)
# =============================================================================

NUM_EPOCHS = TRAINING_CONFIG["num_epochs"]
NUM_CLASSES = MODEL_CONFIG["num_classes"]
LEARNING_RATE = TRAINING_CONFIG["learning_rate"]
IMG_SIZE = DATA_CONFIG["image_size"]
BATCH_SIZE = TRAINING_CONFIG["batch_size"]
NUM_SLICES = DATA_CONFIG["num_slices"]
EARLY_STOP_PATIENCE = TRAINING_CONFIG["early_stopping_patience"]
