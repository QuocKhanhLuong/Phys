"""
Configuration for PIE-UNet Profile Ablation Study
"""
import os
import sys
from pathlib import Path

# Add project root to path
PROJECT_ROOT = Path(__file__).parent.parent.parent
sys.path.insert(0, str(PROJECT_ROOT))


# | Profile     | C_in | depth |
# |-------------|------|-------|
# | PIE-UNet-T  | 3    | 4     |
# | PIE-UNet-S  | 5    | 4     |
# | PIE-UNet-M  | 5    | 5     |  ← Same as original model (baseline)
# | PIE-UNet-L  | 7    | 5     |
# | PIE-UNet-XL | 7    | 6     |

PROFILE_CONFIGS = {
    "T": {
        "name": "PIE-UNet-T",
        "n_channels": 3,  # C_in: number of input slices (2.5D)
        "depth": 4,       # Number of encoder levels
        "base_filters": 16,  # Base number of filters
        "description": "Tiny - smallest model"
    },
    "M": {
        "name": "PIE-UNet-M",
        "n_channels": 5,
        "depth": 5,
        "base_filters": 16,
        "description": "Medium - baseline configuration"
    },
    "XL": {
        "name": "PIE-UNet-XL",
        "n_channels": 7,
        "depth": 6,
        "base_filters": 16,
        "description": "Extra Large - largest model"
    }
}

TRAINING_CONFIG = {
    "num_epochs": 250,          # Same as train_acdc.py
    "learning_rate": 1e-3,
    "weight_decay": 1e-4,
    "batch_size": 16,           # Reverted to 16
    "early_stopping_patience": 30,  # Same as train_acdc.py
    "num_classes": 4,          # ACDC: BG, RV, MYO, LV
    "img_size": 224,
}

DATA_CONFIG = {
    "preprocessed_dir": PROJECT_ROOT / "preprocessed_data" / "ACDC",
    "train_dir": PROJECT_ROOT / "preprocessed_data" / "ACDC" / "training",
    "test_dir": PROJECT_ROOT / "preprocessed_data" / "ACDC" / "testing",
}

OUTPUT_CONFIG = {
    "results_dir": Path(__file__).parent / "results",
    "weights_dir": Path(__file__).parent / "weights",
}

# Create directories
OUTPUT_CONFIG["results_dir"].mkdir(parents=True, exist_ok=True)
OUTPUT_CONFIG["weights_dir"].mkdir(parents=True, exist_ok=True)


MEASURE_CONFIG = {
    "input_size": 224,
    "num_warmup_runs": 10,      # Warmup runs for latency measurement
    "num_measure_runs": 100,   # Number of runs to average for latency
}
