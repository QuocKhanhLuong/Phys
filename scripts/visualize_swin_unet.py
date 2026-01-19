"""
Swin-Unet Visualization Script

Generates segmentation predictions from Swin-Unet and saves to individual images.
These can be combined with main comparison script output.

Usage:
    python scripts/visualize_swin_unet.py
"""

import os
import sys
import numpy as np
import torch
import matplotlib.pyplot as plt
import matplotlib.colors as mcolors
from pathlib import Path
from scipy.ndimage import zoom
from datetime import datetime

# Add project root
PROJECT_ROOT = Path(__file__).parent.parent
sys.path.insert(0, str(PROJECT_ROOT))

# ============================================================================
# CONFIGURATION
# ============================================================================

DEVICE = torch.device('cuda' if torch.cuda.is_available() else 'cpu')
NUM_CASES = 10
IMG_SIZE = 224
NUM_CLASSES = 4  # ACDC: BG, RV, MYO, LV

WEIGHTS_PATH = PROJECT_ROOT / 'comparison' / 'Swin-Unet' / 'acdc_out' / 'best_model.pth'
OUTPUT_DIR = PROJECT_ROOT / 'visualization_outputs' / 'swin_unet'


# ============================================================================
# MODEL LOADING
# ============================================================================

def load_swin_unet():
    """Load Swin-Unet with CORRECT architecture (depths=[2,2,2,2])."""
    swin_unet_path = PROJECT_ROOT / 'comparison' / 'Swin-Unet'
    sys.path.insert(0, str(swin_unet_path))
    
    from networks.swin_transformer_unet_skip_expand_decoder_sys import SwinTransformerSys
    
    # Correct config from swin_tiny_patch4_window7_224_lite.yaml
    model = SwinTransformerSys(
        img_size=IMG_SIZE,
        patch_size=4,
        in_chans=3,  # 3 channels
        num_classes=NUM_CLASSES,
        embed_dim=96,
        depths=[2, 2, 2, 2],  # CORRECT: lite version uses [2,2,2,2]
        depths_decoder=[2, 2, 2, 1],  # From config
        num_heads=[3, 6, 12, 24],
        window_size=7,
        mlp_ratio=4.,
        qkv_bias=True,
        drop_rate=0.,
        attn_drop_rate=0.,
        drop_path_rate=0.2,  # From config
    ).to(DEVICE)
    
    # Load state dict
    state_dict = torch.load(WEIGHTS_PATH, map_location=DEVICE)
    
    # Strip 'swin_unet.' prefix if present
    new_state_dict = {}
    for k, v in state_dict.items():
        if k.startswith('swin_unet.'):
            new_state_dict[k[10:]] = v
        else:
            new_state_dict[k] = v
    
    model.load_state_dict(new_state_dict)
    model.eval()
    print("Swin-Unet loaded successfully!")
    return model


# ============================================================================
# INFERENCE
# ============================================================================

def inference(model, image):
    """Run inference with Swin-Unet (3-channel input)."""
    if image.ndim == 3:  # Take middle slice if 2.5D
        image = image[image.shape[0] // 2]
    
    h, w = image.shape
    if h != IMG_SIZE or w != IMG_SIZE:
        image_resized = zoom(image, (IMG_SIZE / h, IMG_SIZE / w), order=3)
    else:
        image_resized = image
    
    # Repeat grayscale to 3 channels
    image_3ch = np.stack([image_resized, image_resized, image_resized], axis=0)
    x_tensor = torch.from_numpy(image_3ch).unsqueeze(0).float().to(DEVICE)
    
    with torch.no_grad():
        out = model(x_tensor)
        pred = torch.argmax(torch.softmax(out, dim=1), dim=1).squeeze(0)
    
    pred_np = pred.cpu().numpy()
    
    if h != IMG_SIZE or w != IMG_SIZE:
        pred_np = zoom(pred_np, (h / IMG_SIZE, w / IMG_SIZE), order=0)
    
    return pred_np


# ============================================================================
# VISUALIZATION
# ============================================================================

def create_figure(image, gt_mask, pred, case_name, output_path):
    """Create figure: Input | GT | Swin-Unet prediction."""
    
    fig, axes = plt.subplots(1, 3, figsize=(12, 4))
    
    if image.ndim == 3:
        display_img = image[image.shape[0] // 2]
    else:
        display_img = image
    
    colors_list = ['black', 'red', 'green', 'blue']
    cmap = mcolors.ListedColormap(colors_list)
    
    # Input
    axes[0].imshow(display_img, cmap='gray')
    axes[0].set_title('Ảnh MRI Gốc', fontsize=12, fontweight='bold')
    axes[0].axis('off')
    
    # Ground Truth
    axes[1].imshow(display_img, cmap='gray')
    gt_masked = np.ma.masked_where(gt_mask == 0, gt_mask)
    axes[1].imshow(gt_masked, cmap=cmap, alpha=0.6, vmin=0, vmax=NUM_CLASSES-1)
    axes[1].set_title('Ground Truth', fontsize=12, fontweight='bold')
    axes[1].axis('off')
    
    # Swin-Unet prediction
    axes[2].imshow(display_img, cmap='gray')
    pred_masked = np.ma.masked_where(pred == 0, pred)
    axes[2].imshow(pred_masked, cmap=cmap, alpha=0.6, vmin=0, vmax=NUM_CLASSES-1)
    axes[2].set_title('Swin-Unet', fontsize=12, fontweight='bold')
    axes[2].axis('off')
    
    # Legend
    legend_elements = [
        plt.Rectangle((0, 0), 1, 1, color='red', label='RV'),
        plt.Rectangle((0, 0), 1, 1, color='green', label='MYO'),
        plt.Rectangle((0, 0), 1, 1, color='blue', label='LV'),
    ]
    fig.legend(handles=legend_elements, loc='lower center', ncol=3, fontsize=10)
    
    fig.suptitle(f'{case_name}', fontsize=14, fontweight='bold')
    plt.tight_layout(rect=[0, 0.08, 1, 0.95])
    plt.savefig(output_path, dpi=150, bbox_inches='tight', facecolor='white')
    plt.close()
    print(f"  Saved: {output_path}")


# ============================================================================
# DATA LOADING
# ============================================================================

def load_test_data(num_cases=10):
    """Load test cases from ACDC preprocessed data."""
    from src.data_utils.acdc_dataset_optimized import ACDCDataset25DOptimized, get_acdc_volume_ids
    import albumentations as A
    from albumentations.pytorch import ToTensorV2
    
    test_npy_dir = str(PROJECT_ROOT / 'preprocessed_data' / 'ACDC' / 'testing')
    test_volume_ids = get_acdc_volume_ids(test_npy_dir)
    
    test_dataset = ACDCDataset25DOptimized(
        npy_dir=test_npy_dir,
        volume_ids=test_volume_ids,
        num_input_slices=5,
        transforms=A.Compose([ToTensorV2()]),
        max_cache_size=10,
        use_memmap=True
    )
    
    total_slices = len(test_dataset)
    step = max(1, total_slices // num_cases)
    selected_indices = [i * step for i in range(num_cases)]
    
    cases = []
    for idx in selected_indices[:num_cases]:
        image, mask = test_dataset[idx]
        image_np = image.numpy()
        mask_np = mask.numpy()
        
        vol_idx, slice_idx = test_dataset.index_map[idx]
        vol_id = os.path.basename(test_dataset.volume_paths[vol_idx]).replace('.npy', '')
        case_name = f"{vol_id}_slice{slice_idx}"
        
        cases.append({
            'image': image_np,
            'mask': mask_np,
            'case_name': case_name
        })
    
    return cases


# ============================================================================
# MAIN
# ============================================================================

def main():
    print("=" * 60)
    print("SWIN-UNET VISUALIZATION")
    print("=" * 60)
    print(f"Device: {DEVICE}")
    print(f"Weights: {WEIGHTS_PATH}")
    
    if not WEIGHTS_PATH.exists():
        print(f"ERROR: Weights not found!")
        return
    
    # Create output directory
    OUTPUT_DIR.mkdir(parents=True, exist_ok=True)
    timestamp = datetime.now().strftime("%Y%m%d_%H%M%S")
    run_dir = OUTPUT_DIR / timestamp
    run_dir.mkdir(parents=True, exist_ok=True)
    
    # Load model
    model = load_swin_unet()
    
    # Load test data
    print("\nLoading test data...")
    cases = load_test_data(NUM_CASES)
    print(f"Loaded {len(cases)} cases")
    
    # Generate visualizations
    print("\nGenerating visualizations...")
    
    for i, case_data in enumerate(cases):
        image = case_data['image']
        gt_mask = case_data['mask']
        case_name = case_data['case_name']
        
        print(f"Case {i+1}/{len(cases)}: {case_name}")
        
        pred = inference(model, image)
        
        output_path = run_dir / f"{i+1:02d}_{case_name}_swin_unet.png"
        create_figure(image, gt_mask, pred, case_name, output_path)
    
    print("\n" + "=" * 60)
    print(f"VISUALIZATION COMPLETE!")
    print(f"Output directory: {run_dir}")
    print("=" * 60)


if __name__ == "__main__":
    main()
