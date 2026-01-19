"""
Configuration for Encoder Ablation Study
"""
import os
import sys
from pathlib import Path

# Add project root to path
PROJECT_ROOT = Path(__file__).parent.parent.parent
sys.path.insert(0, str(PROJECT_ROOT))


# =============================================================================
# ENCODER CONFIGURATIONS
# =============================================================================

ENCODER_CONFIGS = {
    "Standard": {
        "name": "Standard Conv",
        "type": "standard",
        "description": "Basic double conv (Conv-BN-ReLU) × 2, like UNet++"
    },
    "SE": {
        "name": "Squeeze-Excite",
        "type": "se",
        "description": "Standard + SE channel attention"
    },
    "ResNet": {
        "name": "Residual",
        "type": "resnet",
        "description": "Standard + residual skip connection (BasicBlock)"
    },
    "CBAM": {
        "name": "CBAM Attention",
        "type": "cbam",
        "description": "Standard + CBAM (Channel + Spatial attention)"
    },
    "NAE": {
        "name": "NAE (Original)",
        "type": "nae",
        "description": "ePURE noise estimation + adaptive smoothing (original encoder)"
    },
    # --- New Encoders ---
    "ResNet50": {
        "name": "ResNet50 Bottleneck",
        "type": "resnet50",
        "description": "Bottleneck block (1x1→3x3→1x1) like ResNet-50"
    },
    "EfficientNet": {
        "name": "EfficientNet MBConv",
        "type": "efficientnet",
        "description": "MBConv blocks with SE attention (EfficientNet-B0 style)"
    },
    "Swin": {
        "name": "Swin Transformer",
        "type": "swin",
        "description": "Window-based self-attention (Swin Transformer)"
    },
    "ConvNeXt": {
        "name": "ConvNeXt",
        "type": "convnext",
        "description": "Modernized ConvNet (depthwise + GELU)"
    },
    "DenseNet": {
        "name": "DenseNet",
        "type": "densenet",
        "description": "Dense connections (concat features)"
    },
    "SAM": {
        "name": "SAM ViT",
        "type": "sam",
        "description": "SAM-inspired ViT encoder with patch embedding"
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
    "num_classes": 4,          # ACDC: BG, RV, MYO, LV
    "num_slices": 5,           # 2.5D input
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
