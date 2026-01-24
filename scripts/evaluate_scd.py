"""
SCD Evaluation Script
Adapted from evaluate_mnm.py
"""

import torch
import torch.nn.functional as F
import numpy as np
import os
import sys
import argparse
from torch.utils.data import DataLoader
import albumentations as A
from albumentations.pytorch import ToTensorV2
from tqdm import tqdm

# Add project root to path
sys.path.insert(0, os.path.abspath(os.path.join(os.path.dirname(__file__), '..')))

from src.models.unet import PIE_UNet
from src.data_utils.scd_dataset_optimized import (
    SCDDataset25DOptimized,
    get_scd_volume_ids,
)
from src import config
from src.modules.metrics import SegmentationMetrics, count_parameters

# Configuration
NUM_CLASSES = 2
DEVICE = torch.device('cuda' if torch.cuda.is_available() else 'cpu')
BATCH_SIZE = 24
NUM_SLICES = 5

SCD_CLASS_MAP = {
    0: 'Background',
    1: 'Left Ventricle'
}

def parse_args():
    parser = argparse.ArgumentParser(description="Evaluate SCD Model")
    parser.add_argument('--weights', type=str, default="weights/best_model_scd.pth", 
                        help="Path to model weights (default: weights/best_model_scd.pth)")
    return parser.parse_args()

def main():
    args = parse_args()

    print(f"Device: {DEVICE}")
    print(f"Evaluating SCD model on test set")

    test_transform = A.Compose([
        ToTensorV2(),
    ])

    print("\nLoading test data SCD")

    # Path to preprocessed testing data
    SCD_PREPROCESSED_DIR = os.path.join(config.PROJECT_ROOT, "preprocessed_data", "SCD", "testing")

    if not os.path.exists(SCD_PREPROCESSED_DIR):
        print(f"Error: Directory not found: {SCD_PREPROCESSED_DIR}")
        sys.exit(1)

    all_volume_ids = get_scd_volume_ids(SCD_PREPROCESSED_DIR, frame_type=None)
    print(f"Total volumes (All Frames): {len(all_volume_ids)}")

    test_volume_ids = all_volume_ids

    print(f"Using {len(test_volume_ids)} volumes for test set")

    test_dataset = SCDDataset25DOptimized(
        npy_dir=SCD_PREPROCESSED_DIR,
        volume_ids=test_volume_ids,
        num_input_slices=NUM_SLICES,
        transforms=test_transform,
        max_cache_size=10,
        use_memmap=True
    )

    test_loader = DataLoader(
        test_dataset, 
        batch_size=BATCH_SIZE, 
        shuffle=False, 
        num_workers=0, 
        pin_memory=True
    )

    print(f"Test set: {len(test_dataset)} slices")

    print(f"\n{'='*60}")
    print("Evaluating on test set")
    print(f"{'='*60}\n")

    MODEL_PATH = args.weights

    if not os.path.exists(MODEL_PATH):
        print(f"Error: Model not found: {MODEL_PATH}")
        sys.exit(1)

    print(f"Loading model from: {MODEL_PATH}")
    model = PIE_UNet(n_channels=NUM_SLICES, n_classes=NUM_CLASSES, deep_supervision=True).to(DEVICE)
    model.load_state_dict(torch.load(MODEL_PATH))
    model.eval()

    params_count = count_parameters(model)

    print(f"Model parameters: {params_count:,}")
    print("Model loaded")

    print("\nRunning evaluation on test set (3D Volumetric Evaluation)")

    # Dictionary to store slices for each volume
    # Structure: volume_predictions[vol_id] = {slice_idx: (pred_slice, gt_slice)}
    volume_data = {}
    
    with torch.no_grad():
        eval_pbar = tqdm(test_loader, desc="Evaluating", ncols=100)
        for i_batch, (imgs, tgts) in enumerate(eval_pbar):
            imgs, tgts = imgs.to(DEVICE), tgts.to(DEVICE)
            if imgs.size(0) == 0: 
                continue
            
            logits_list, _ = model(imgs)
            logits = logits_list[-1]
            preds = torch.argmax(F.softmax(logits, dim=1), dim=1)
            
            # Accumulate slices
            for b_idx in range(imgs.size(0)):
                 global_idx = i_batch * BATCH_SIZE + b_idx
                 if global_idx < len(test_dataset):
                     vol_idx, slice_idx = test_dataset.index_map[global_idx]
                     vol_id = os.path.basename(test_dataset.volume_paths[vol_idx]).replace('.npy', '')
                     
                     if vol_id not in volume_data:
                         volume_data[vol_id] = {}
                     
                     # Store as numpy arrays
                     volume_data[vol_id][slice_idx] = (preds[b_idx].cpu().numpy(), tgts[b_idx].cpu().numpy())

    # Compute 3D Metrics
    from medpy import metric
    
    vol_dice_scores = []
    vol_hd95_scores = []
    
    print("\nComputing 3D Volumetric Metrics...")
    
    sorted_vol_ids = sorted(volume_data.keys())
    for vol_id in tqdm(sorted_vol_ids, desc="Processing Volumes"):
        slices_dict = volume_data[vol_id]
        sorted_slice_indices = sorted(slices_dict.keys())
        
        # Reconstruct 3D volume
        # Assuming slices are contiguous and sorted_slice_indices are correct 0..D-1 relative to the loaded volume
        # We perform simple stacking based on sorted slice index
        
        pred_vol_list = []
        gt_vol_list = []
        
        for s_idx in sorted_slice_indices:
            p_slice, g_slice = slices_dict[s_idx]
            pred_vol_list.append(p_slice)
            gt_vol_list.append(g_slice)
            
        pred_vol = np.stack(pred_vol_list, axis=0) # (D, H, W)
        gt_vol = np.stack(gt_vol_list, axis=0)     # (D, H, W)
        
        # Compute Metrics for Foreground (Left Ventricle - Class 1)
        # Assuming Class 1 is the refined LV
        
        # Binary masks for LV
        pred_lv = (pred_vol == 1).astype(int)
        gt_lv = (gt_vol == 1).astype(int)
        
        # Dice
        if pred_lv.sum() > 0 and gt_lv.sum() > 0:
            d = metric.binary.dc(pred_lv, gt_lv)
            h = metric.binary.hd95(pred_lv, gt_lv)
        elif pred_lv.sum() == 0 and gt_lv.sum() == 0:
            d = 1.0
            h = np.nan
        else:
            d = 0.0
            h = np.nan
            
        vol_dice_scores.append(d)
        vol_hd95_scores.append(h)
        
        print(f"[Case: {vol_id}] 3D Dice: {d:.4f} - 3D HD95: {h:.4f}")

    # Aggregated Results
    avg_vol_dice = np.nanmean(vol_dice_scores)
    avg_vol_hd95 = np.nanmean(vol_hd95_scores)

    print(f"\n{'='*60}")
    print("TEST SET RESULTS (3D VOLUMETRIC)")
    print(f"{'='*60}")
    
    print(f"Model: {os.path.basename(MODEL_PATH)}")
    print(f"Params: {params_count:,}\n")
    
    print(f"Foreground (Left Ventricle) 3D Metrics:")
    print(f"=> Avg 3D Dice:      {avg_vol_dice:.4f}")
    print(f"=> Avg 3D HD95:      {avg_vol_hd95:.4f}")

    print(f"\n{'='*60}")
    
    # Save results
    RESULTS_DIR = "results"
    os.makedirs(RESULTS_DIR, exist_ok=True)
    results_file = os.path.join(RESULTS_DIR, "scd_test_results_3d.txt")
    
    with open(results_file, 'w') as f:
        f.write("="*60 + "\n")
        f.write("SCD TEST SET RESULTS (3D VOLUMETRIC)\n")
        f.write("="*60 + "\n\n")
        f.write(f"Model: {MODEL_PATH}\n")
        f.write(f"Parameters: {params_count:,}\n\n")
        
        f.write("Foreground (Left Ventricle) 3D Metrics:\n")
        f.write(f"  Avg 3D Dice: {avg_vol_dice:.4f}\n")
        f.write(f"  Avg 3D HD95: {avg_vol_hd95:.4f}\n")
        
        f.write("\nPer-Case Details:\n")
        for i, vid in enumerate(sorted_vol_ids):
             f.write(f"  {vid}: Dice={vol_dice_scores[i]:.4f}, HD95={vol_hd95_scores[i]:.4f}\n")

    print(f"Results saved to: {results_file}")

if __name__ == "__main__":
    main()
