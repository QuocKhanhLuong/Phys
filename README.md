# PIE-UNet: Physics-Inspired Encoder for Medical Image Segmentation

A deep learning framework for cardiac MRI segmentation using physics-inspired encoders and Maxwell equation constraints.

## 📋 Table of Contents
- [Overview](#overview)
- [Installation](#installation)
- [Project Structure](#project-structure)
- [Quick Start](#quick-start)
- [Training the Main Model](#training-the-main-model)
- [Profile Ablation Study (T/M/XL)](#profile-ablation-study-tmxl)
- [Evaluation](#evaluation)
- [Datasets](#datasets)

---

## Overview

PIE-UNet integrates physics-inspired noise estimation (ePURE) with a UNet++ architecture enhanced by Maxwell equation constraints for robust cardiac segmentation. The model achieves state-of-the-art performance on the ACDC dataset.

**Key Features:**
- 2.5D input (multiple slices for context)
- Physics-informed loss with electromagnetic field constraints
- Deep supervision for stable training
- Multiple model profiles (T, M, XL) for different computational budgets

---

## Installation

### 1. Create Conda Environment

```bash
conda create -n physmed python=3.11 -y
conda activate physmed
```

### 2. Install Dependencies

```bash
# Install PyTorch (CUDA 12.4)
pip install torch==2.6.0 torchvision==0.21.0 --index-url https://download.pytorch.org/whl/cu124

# Install all other dependencies
pip install -r requirements.txt
```

---

## Project Structure

```
Phys/
├── src/                          # Core source code
│   ├── models/
│   │   ├── unet.py              # Main PIE-UNet model (RobustMedVFL_UNet)
│   │   ├── epure.py             # ePURE noise estimation module
│   │   └── maxwell_solver.py    # Maxwell equation solver
│   ├── modules/
│   │   └── losses.py            # Combined loss functions
│   └── data_utils/              # Dataset utilities
│
├── scripts/                      # Training & evaluation scripts
│   ├── train_acdc.py            # Main model training
│   ├── evaluate_acdc.py         # 3D volumetric evaluation
│   ├── preprocess_acdc.py       # Data preprocessing
│   └── ...
│
├── ablation/                     # Ablation studies
│   ├── profile/                 # Model size ablation (T/M/XL)
│   │   ├── config.py            # Profile configurations
│   │   ├── train_profile.py     # Training script
│   │   ├── evaluate_3d.py       # 3D evaluation
│   │   └── pie_unet.py          # Configurable PIE-UNet
│   └── encoder/                 # Encoder ablation study
│
├── weights/                      # Saved model weights
├── preprocessed_data/            # Preprocessed .npy data
└── data/                         # Raw datasets
```

---

## Quick Start

### 1. Preprocess Data

Before training, preprocess the raw ACDC data:

```bash
conda activate physmed
python scripts/preprocess_acdc.py
```

This creates `.npy` files in `preprocessed_data/ACDC/`.

### 2. Train Main Model

```bash
python scripts/train_acdc.py
```

### 3. Evaluate

```bash
python scripts/evaluate_acdc.py
```

---

## Training the Main Model

The main PIE-UNet model uses:
- **Input**: 5 slices (2.5D)
- **Output**: 4 classes (Background, RV, MYO, LV)
- **Architecture**: UNet++ with NAE encoder + Maxwell Solver

### Training Command

```bash
conda activate physmed
cd /path/to/Phys

# Full training (250 epochs with early stopping)
python scripts/train_acdc.py
```

### Configuration

Edit the configuration at the top of `scripts/train_acdc.py`:

```python
NUM_EPOCHS = 250
NUM_CLASSES = 4
LEARNING_RATE = 1e-3
BATCH_SIZE = 24
NUM_SLICES = 5        # 2.5D input
EARLY_STOP_PATIENCE = 30
```

### Output

Models are saved to `weights/`:
- `best_model_acdc_dice.pth` - Best Dice score
- `best_model_acdc_hd95.pth` - Best HD95 score
- `best_model_acdc_overall.pth` - Best combined score

---

## Profile Ablation Study (T/M/XL)

The profile ablation compares different model sizes:

| Profile | Input Slices | Depth | Description |
|---------|-------------|-------|-------------|
| **T** (Tiny) | 3 | 4 | Smallest, fastest |
| **M** (Medium) | 5 | 5 | Baseline (same as main model) |
| **XL** (Extra Large) | 7 | 6 | Largest, most accurate |

### Training Profiles

```bash
conda activate physmed
cd /path/to/Phys

# Train specific profile
python -m ablation.profile.train_profile --profile T    # Tiny
python -m ablation.profile.train_profile --profile M    # Medium
python -m ablation.profile.train_profile --profile XL   # Extra Large

# Train all profiles sequentially
python -m ablation.profile.run_full_ablation
```

### Evaluating Profiles (3D Metrics)

```bash
# Evaluate specific profile
python -m ablation.profile.evaluate_3d --profile T

# Evaluate all profiles with full metrics (Params, GFLOPs, Dice, HD95)
python -m ablation.profile.evaluate_3d
```

### Measuring Computational Metrics

```bash
# Measure CPU latency, params, GFLOPs for all profiles
python -m ablation.profile.measure_profile
```

### Profile Weights

Trained weights are saved to `ablation/profile/weights/`:
- `best_T_dice.pth`, `best_T_hd95.pth`, `best_T_overall.pth`
- `best_M_dice.pth`, `best_M_hd95.pth`, `best_M_overall.pth`
- `best_XL_dice.pth`, `best_XL_hd95.pth`, `best_XL_overall.pth`

---

## Evaluation

### 3D Volumetric Evaluation

For proper medical image evaluation, use 3D volumetric metrics:

```bash
# Evaluate main model on test set
python scripts/evaluate_acdc.py

# Evaluate profile models
python -m ablation.profile.evaluate_3d --profile M
```

### Metrics Reported

- **Dice Score**: Overlap between prediction and ground truth
- **HD95**: 95th percentile Hausdorff Distance (in pixels)
- **Per-class**: RV (Right Ventricle), MYO (Myocardium), LV (Left Ventricle)

---

## Datasets

### ACDC (Automated Cardiac Diagnosis Challenge)

Primary dataset for training and evaluation:
- 100 training patients, 50 testing patients
- 4 classes: Background, RV, MYO, LV
- Short-axis cardiac MRI

### Data Preprocessing

```bash
# ACDC
python scripts/preprocess_acdc.py

# M&M (Multi-Centre, Multi-Vendor)
python scripts/preprocess_mnm.py

# SCD (Sunnybrook Cardiac Data)
python scripts/preprocess_scd.py
```

---

## Example: Complete Training Pipeline

```bash
# 1. Activate environment
conda activate physmed
cd /path/to/Phys

# 2. Preprocess data (run once)
python scripts/preprocess_acdc.py

# 3. Train main model
python scripts/train_acdc.py

# 4. Evaluate on test set
python scripts/evaluate_acdc.py

# 5. (Optional) Run profile ablation
python -m ablation.profile.run_full_ablation

# 6. Compare profiles
python -m ablation.profile.evaluate_3d
```

---

## Hardware Requirements

- **GPU**: NVIDIA GPU with ≥8GB VRAM (tested on RTX 3080, 4090)
- **RAM**: ≥16GB
- **Storage**: ~10GB for preprocessed data

---

## Citation

If you use this code, please cite:

```bibtex
@article{pieunet2024,
  title={PIE-UNet: Physics-Inspired Encoder for Medical Image Segmentation},
  author={...},
  journal={...},
  year={2024}
}
```

---

## License

This project is for research purposes only.
