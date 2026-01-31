# PGE-UNet: Physics-Guided Encoder for Efficient Cine CMR Segmentation

---

## 📋 Table of Contents

- [Overview](#overview)
- [Installation](#installation)
- [Dataset Preparation](#dataset-preparation)
- [Usage](#usage)
  - [Preprocessing](#preprocessing)
  - [Training](#training)
  - [Evaluation](#evaluation)
- [Results](#results)
- [Ablation Study](#ablation-study)
- [Citation](#citation)
- [License](#license)

---

## Overview

**PGE-UNet** (Physics-Guided Encoder UNet) integrates physics-inspired noise estimation (ePURE) with a UNet++ architecture enhanced by Maxwell equation constraints for robust cardiac MRI segmentation. The model achieves state-of-the-art performance on multiple cardiac segmentation benchmarks.

### Key Features

| Feature | Description |
|---------|-------------|
| **2.5D Input** | 2.5D input for param-efficient |
| **Noise-Aware Encoder** | Physics-inspired noise estimation |
| **Physics-Regularized Decoder** | Electromagnetic field constraints via Maxwell equations |
| **Multiple Variants Support** | Support for various model configurations |

---

## Installation

### Requirements

- Python 3.11
- PyTorch 2.6
- CUDA 12.4

### Setup Environment

```bash
# 1. Clone the repository
git clone https://github.com/yourusername/PGE-UNet.git
cd PGE-UNet

# 2. Create conda environment
conda create -n pge-unet python=3.11 -y
conda activate pge-unet

# 3. Install PyTorch with CUDA 12.4
pip install torch==2.6.0 torchvision==0.21.0 --index-url https://download.pytorch.org/whl/cu124

# 4. Install dependencies
pip install -r requirements.txt
```

---

## Dataset Preparation

### Supported Datasets

| Dataset | Task | Classes | Download Link |
|---------|------|---------|---------------|
| **ACDC** | Cardiac Segmentation | 4 (BG, RV, MYO, LV) | [ACDC Challenge](https://www.creatis.insa-lyon.fr/Challenge/acdc/) |
| **M&M** | Multi-Centre Cardiac | 4 (BG, RV, MYO, LV) | [M&M Challenge](https://www.ub.edu/mnms/) |
| **SCD** | Sunnybrook Cardiac | 2 (BG, LV) | [Sunnybrook Dataset](http://www.cardiacatlas.org/studies/sunnybrook-cardiac-data/) |

### Directory Structure

Datasets must be downloaded, unzipped, and organized in `/data`. The standard structure:

```
Phys/
├── data/
│   ├── ACDC/
│   ├── MnM/
│   └── SCD/
│
├── preprocessed_data/
└── weights/
```

---

## Usage

### Preprocessing

Raw datasets must be preprocessed before training to convert to `.npy` format and generate pseudo B1 maps using simulator. Preprocessed data is saved to `/preprocessed_data`.

```bash
conda activate pge-unet
cd /path/to/Phys

# Preprocess ACDC dataset
python scripts/preprocess_acdc.py

# Preprocess M&M dataset
python scripts/preprocess_mnm.py

# Preprocess SCD dataset
python scripts/preprocess_scd.py
```

### Training

#### Main PGE-UNet Model

```bash
conda activate pge-unet
cd /path/to/Phys

# Train on ACDC (default: 250 epochs with early stopping)
python scripts/train_acdc.py

# Train on M&M
python scripts/train_mnm.py

# Train on SCD
python scripts/train_scd.py
```

**Output Weights** (saved to `weights/`):

Pretrained weights will be published after paper acceptance. Currently, weights are provided as requested from reviewers and editors via email.

### Evaluation

```bash
conda activate pge-unet
cd /path/to/Phys

# Evaluate main model on ACDC test set
python scripts/evaluate_acdc.py

# Evaluate on M&M test set
python scripts/evaluate_mnm.py

# Evaluate on SCD test set
python scripts/evaluate_scd.py
```

---

## Results

<p align="center">
  <img src="assets/patient_comparison_grid.png" alt="Segmentation Results" width="100%">
</p>

<p align="center">
  <em>Segmentation results on ACDC test patients. Rows: Input MRI, Ground Truth, PGE-UNet. Columns: Different patients (103-114).<br>
  Colors: 🔴 RV (Right Ventricle), 🟢 MYO (Myocardium), 🔵 LV (Left Ventricle)</em>
</p>

---

## Ablation Study

### Model Profile Comparison

| Profile | Input Slices | Depth | Params | GFLOPs | Mean Dice |
|---------|-------------|-------|--------|--------|-----------|
| **T** (Tiny) | 3 | 4 | 0.5M | 12.3 | 0.8921 |
| **M** (Medium) | 5 | 5 | 1.6M | 28.7 | 0.9152 |
| **XL** (Extra Large) | 7 | 6 | 4.2M | 56.4 | 0.9198 |

### Training Ablation Models

```bash
conda activate pge-unet
cd /path/to/Phys

# Train specific profile
python -m ablation.profile.train_profile --profile T    # Tiny
python -m ablation.profile.train_profile --profile M    # Medium
python -m ablation.profile.train_profile --profile XL   # Extra Large

# Train all profiles sequentially
python -m ablation.profile.run_full_ablation

# Evaluate specific profile
python -m ablation.profile.evaluate_3d --profile T

# Evaluate all profiles with full metrics
python -m ablation.profile.evaluate_3d

# Measure computational metrics (Params, GFLOPs, CPU Latency)
python -m ablation.profile.measure_profile
```

---

## Citation

If you find this work useful, please cite:

```bibtex
@article{pgeunet2024,
  title={PGE-UNet: Physics-Guided Encoder for Efficient Cardiac MRI Segmentation},
  author={Author Names},
  journal={Journal Name},
  year={2024}
}
```

---

## License

This project is for **research purposes only**. 

For commercial use, please contact the authors.

---

## Acknowledgements

- [ACDC Dataset](https://www.creatis.insa-lyon.fr/Challenge/acdc/)
- [M&M Challenge](https://www.ub.edu/mnms/)
- [nnUNet](https://github.com/MIC-DKFZ/nnUNet)
- [Swin-Unet](https://github.com/HuCaoFighting/Swin-Unet)
