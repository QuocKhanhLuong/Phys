# PhysicsMed - Physics-Informed Medical Image Segmentation

Deep learning framework for cardiac MRI segmentation using physics-informed learning with Maxwell equations and UNet++ architecture.

## 🎯 Features

- **UNet++**: Nested U-Net with dense skip connections
- **Physics-Informed Learning**: Maxwell equation integration via B1 field mapping
- **ePURE Module**: Enhanced noise profiling with SE attention
- **ASPP Bottleneck**: Multi-scale feature extraction
- **Quantum Noise Injection**: Advanced data augmentation

## 📁 Project Structure

```
PhysicsMed/
├── models.py              # Model architectures (UNet++, ePURE, Maxwell Solver)
├── losses.py              # Custom loss functions
├── utils.py               # Utility functions (B1 map, smoothing, noise injection)
├── data_utils.py          # Dataset and data loading
├── train.py               # Main training script
├── evaluate.py            # Evaluation and visualization
├── config.py              # Configuration
├── requirements.txt       # Dependencies
└── environment.yml        # Conda environment
```

## 🚀 Quick Start

### Installation

**Option 1: Conda (Recommended)**
```bash
conda create -n physicsmed python=3.11 -y
conda activate physicsmed
conda install pytorch=2.6.0 torchvision=0.21.0 pytorch-cuda=12.4 -c pytorch -c nvidia -y
pip install -r requirements.txt
```

**Option 2: From environment.yml**
```bash
conda env create -f environment.yml
conda activate physicsmed
```

### Dataset

Place ACDC dataset in `database/` folder:
```
PhysicsMed/
└── database/
    ├── training/
    └── testing/
```

### Training

```bash
# Test data loading
python test_data_loading.py

# Start training (5-fold cross-validation)
python train.py
```

## 📊 Expected Results

| Metric | Value |
|--------|-------|
| Dice Score (Foreground) | 0.85-0.92 |
| IoU (Foreground) | 0.75-0.85 |
| Accuracy | 0.95+ |

## 🛠️ Requirements

- Python 3.11
- PyTorch 2.6.0
- CUDA 12.4 (optional, for GPU)
- 16GB+ RAM
- See `requirements.txt` for full dependencies

## 📚 Documentation

- `CONDA_GUIDE.md` - Conda environment setup guide
- `config.py` - All hyperparameters and settings
- `test_data_loading.py` - Test script for data loading

## 🎓 Model Architecture

### UNet++ with Physics-Informed Learning
- **Encoder**: Custom blocks with ePURE noise profiling
- **Bottleneck**: ASPP for multi-scale features
- **Decoder**: Dense skip connections
- **Maxwell Solver**: Physics-based tissue property estimation

### Loss Function
- Focal Loss (γ=2.0)
- Focal Tversky Loss (α=0.2, β=0.8, γ=4/3)
- Physics Loss (Helmholtz equation residuals)
- Dynamic loss weighting

## 📈 Training Details

- **Dataset**: ACDC (100 training + 50 testing patients)
- **Input**: 2.5D (5 consecutive slices)
- **Cross-Validation**: 5-fold
- **Batch Size**: 24
- **Epochs**: 250 (with early stopping)
- **Optimizer**: AdamW (lr=1e-3)
- **Augmentation**: Rotation, flip, elastic transform, quantum noise injection

## 🔬 Key Components

### ePURE (Enhanced Parametric Uniform Residual Estimator)
- Noise profiling network
- SE attention mechanism
- Residual connections
- Adaptive smoothing

### Maxwell Solver
- Estimates permittivity (ε) and conductivity (σ)
- Helmholtz equation constraint
- Physics-informed regularization

### B1 Field Simulation
- Advanced multi-coil simulation
- Weighted averaging by image quality
- Spatial ROI weighting

## 📝 Citation

```bibtex
@article{physicsmed2025,
  title={Physics-Informed Deep Learning for Cardiac MRI Segmentation},
  journal={arXiv preprint},
  year={2025}
}
```

## 📄 License

MIT License

## 🙏 Acknowledgments

- ACDC Challenge for the dataset
- PyTorch team
- Medical imaging research community

---

**Note**: Dataset not included due to size. Please download ACDC dataset separately.

