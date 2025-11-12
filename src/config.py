import os
from pathlib import Path
from typing import Dict, Any

# Project root directory
PROJECT_ROOT = Path(__file__).parent.parent
SRC_ROOT = Path(__file__).parent

# Dataset Configuration

# Raw data directories
DATA_DIR = PROJECT_ROOT / "data"
BRATS21_RAW_DIR = DATA_DIR / "BraTS21"
ACDC_RAW_DIR = DATA_DIR / "ACDC"
ATLAS_RAW_DIR = DATA_DIR / "ATLAS"

# Preprocessed data directories
PREPROCESSED_DIR = PROJECT_ROOT / "preprocessed_data"
BRATS21_PREPROCESSED_DIR = PREPROCESSED_DIR / "brats21" / "BraTS21_preprocessed_monai"
ACDC_PREPROCESSED_DIR = PREPROCESSED_DIR / "acdc"
ATLAS_PREPROCESSED_DIR = PREPROCESSED_DIR / "atlas"

# Cache directories
CACHE_DIR = PROJECT_ROOT / "cache"
MONAI_CACHE_DIR = CACHE_DIR / "monai"

# Model Configuration

# Model architecture
MODEL_CONFIG = {
    "input_channels": 4,  # BraTS21: T1, T1ce, T2, FLAIR
    "num_classes": 4,     # Background, NCR/NET, Edema, ET
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
    "use_monai_pipeline": True,
    "samples_per_patient": None,  # None for all slices
    "num_workers": 8,
    "prefetch_factor": 2,
    "pin_memory": True,
    "persistent_workers": True,
    "image_size": 224,
    "num_slices": 5,
    "augmentation_prob": 0.5
}

# =============================================================================
# Loss Function Configuration
# =============================================================================

LOSS_CONFIG = {
    "dice_weight": 1.0,
    "focal_weight": 0.5,
    "physics_weight": 0.3,
    "focal_alpha": 0.25,
    "focal_gamma": 2.0,
    "smooth": 1e-6
}

# =============================================================================
# Dataset-Specific Configuration
# =============================================================================

# BraTS21 Configuration
BRATS21_CONFIG = {
    "name": "BraTS21",
    "modalities": ["T1", "T1ce", "T2", "FLAIR"],
    "class_map": {
        0: "Background",
        1: "NCR/NET",  # Necrotic and non-enhancing tumor
        2: "Edema",    # Peritumoral edema
        3: "ET"        # Enhancing tumor
    },
    "region_names": {
        "ET": "Enhancing Tumor",
        "TC": "Tumor Core",  # NCR/NET + ET
        "WT": "Whole Tumor"  # NCR/NET + ET + Edema
    },
    "preprocessing": {
        "normalization": "z_score",
        "crop_to_nonzero": True,
        "resize_to": (224, 224),
        "slice_thickness": 1.0
    }
}

# ACDC Configuration (example)
ACDC_CONFIG = {
    "name": "ACDC",
    "modalities": ["ED", "ES"],  # End-diastolic, End-systolic
    "class_map": {
        0: "Background",
        1: "Right Ventricle",
        2: "Myocardium",
        3: "Left Ventricle"
    },
    "preprocessing": {
        "normalization": "min_max",
        "crop_to_nonzero": True,
        "resize_to": (224, 224),
        "slice_thickness": 1.0
    }
}

# =============================================================================
# Experiment Configuration
# =============================================================================

EXPERIMENT_CONFIG = {
    "results_dir": PROJECT_ROOT / "results",
    "experiments_dir": PROJECT_ROOT / "results" / "experiments",
    "visualizations_dir": PROJECT_ROOT / "results" / "visualizations",
    "checkpoint_dir": PROJECT_ROOT / "results" / "checkpoints",
    "log_dir": PROJECT_ROOT / "results" / "logs"
}

# =============================================================================
# Hardware Configuration
# =============================================================================

HARDWARE_CONFIG = {
    "device": "cuda" if os.environ.get("CUDA_VISIBLE_DEVICES") else "cpu",
    "mixed_precision": True,
    "compile_model": False,  # PyTorch 2.0 compilation
    "num_gpus": 1
}

# =============================================================================
# Utility Functions
# =============================================================================

def get_dataset_config(dataset_name: str) -> Dict[str, Any]:
    """Get configuration for a specific dataset."""
    config_map = {
        "brats21": BRATS21_CONFIG,
        "acdc": ACDC_CONFIG,
    }
    
    if dataset_name.lower() not in config_map:
        raise ValueError(f"Unknown dataset: {dataset_name}. Available: {list(config_map.keys())}")
    
    return config_map[dataset_name.lower()]

def get_experiment_path(experiment_name: str) -> Path:
    """Get the path for a specific experiment."""
    return EXPERIMENT_CONFIG["experiments_dir"] / experiment_name

def create_directories():
    """Create all necessary directories."""
    directories = [
        DATA_DIR,
        PREPROCESSED_DIR,
        CACHE_DIR,
        MONAI_CACHE_DIR,
        EXPERIMENT_CONFIG["results_dir"],
        EXPERIMENT_CONFIG["experiments_dir"],
        EXPERIMENT_CONFIG["visualizations_dir"],
        EXPERIMENT_CONFIG["checkpoint_dir"],
        EXPERIMENT_CONFIG["log_dir"]
    ]
    
    for directory in directories:
        directory.mkdir(parents=True, exist_ok=True)

# =============================================================================
# Legacy Support (for backward compatibility)
# =============================================================================

# Legacy constants for backward compatibility
PROJECT_ROOT_LEGACY = str(PROJECT_ROOT)
USE_MONAI_PIPELINE = DATA_CONFIG["use_monai_pipeline"]
MONAI_SAMPLES_PER_PATIENT = DATA_CONFIG["samples_per_patient"]
DATA_NUM_WORKERS = DATA_CONFIG["num_workers"]
DATA_PREFETCH_FACTOR = DATA_CONFIG["prefetch_factor"]

# Legacy paths
NPY_DIR = str(BRATS21_PREPROCESSED_DIR)
MONAI_NPY_DIR = str(BRATS21_PREPROCESSED_DIR)
MONAI_CACHE_DIR_LEGACY = str(MONAI_CACHE_DIR)
BRATS_RAW_DIR = str(BRATS21_RAW_DIR)

# Legacy training parameters
NUM_EPOCHS = TRAINING_CONFIG["num_epochs"]
NUM_CLASSES = MODEL_CONFIG["num_classes"]
LEARNING_RATE = TRAINING_CONFIG["learning_rate"]
IMG_SIZE = DATA_CONFIG["image_size"]
BATCH_SIZE = TRAINING_CONFIG["batch_size"]
NUM_SLICES = DATA_CONFIG["num_slices"]
EARLY_STOP_PATIENCE = TRAINING_CONFIG["early_stopping_patience"]

# Legacy class mappings
BRATS_CLASS_MAP = BRATS21_CONFIG["class_map"]
BRATS_REGION_NAMES = BRATS21_CONFIG["region_names"]

# Mở file src/config.py của bạn và thêm dòng này:

# Preprocessed data directories
PREPROCESSED_DIR = PROJECT_ROOT / "preprocessed_data"
BRATS21_PREPROCESSED_DIR = PREPROCESSED_DIR / "brats21" / "BraTS21_preprocessed_monai"
ACDC_PREPROCESSED_DIR = PREPROCESSED_DIR / "acdc"
ATLAS_RAW_DIR = DATA_DIR / "ATLAS"
# --- THÊM DÒNG NÀY ---
ATLAS_PREPROCESSED_DIR = PREPROCESSED_DIR / "atlas_npy"