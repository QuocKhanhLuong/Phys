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

from src.models.unet import RobustMedVFL_UNet 
from src.data_utils.mnm_dataset_optimized import (
    MnMDataset25DOptimized,
    get_mnm_volume_ids,
)
from src import config
from src.modules.metrics import SegmentationMetrics, count_parameters

# --- CONFIGURATION ---
NUM_CLASSES = 4
DEVICE = torch.device('cuda' if torch.cuda.is_available() else 'cpu')
BATCH_SIZE = 16
NUM_SLICES = 5

MNM_CLASS_MAP = {
    0: 'Background',
    1: 'Left Ventricle',
    2: 'Myocardium',
    3: 'Right Ventricle'
}

def parse_args():
    parser = argparse.ArgumentParser(description="Evaluate M&Ms Model")
    parser.add_argument('--weights', type=str, default="weights/best_model_mnm_norm.pth", 
                        help="Path to model weights (default: weights/best_model_mnm_norm.pth)")
    return parser.parse_args()

def main():
    args = parse_args()
    
    print(f"Device: {DEVICE}")
    print(f"Evaluating M&Ms model on test set")

    test_transform = A.Compose([
        ToTensorV2(),
    ])

    print("\nLoading test data M&Ms")

    MNM_PREPROCESSED_DIR = os.path.join(config.PROJECT_ROOT, "preprocessed_data", "mnm", "testing")

    if not os.path.exists(MNM_PREPROCESSED_DIR):
        print(f"Error: Directory not found: {MNM_PREPROCESSED_DIR}")
        sys.exit(1)

    all_volume_ids = get_mnm_volume_ids(MNM_PREPROCESSED_DIR)
    print(f"Total volumes: {len(all_volume_ids)}")

    test_volume_ids = all_volume_ids

    print(f"Using {len(test_volume_ids)} volumes for test set")

    test_dataset = MnMDataset25DOptimized(
        npy_dir=MNM_PREPROCESSED_DIR,
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
    model = RobustMedVFL_UNet(n_channels=NUM_SLICES, n_classes=NUM_CLASSES, deep_supervision=True).to(DEVICE)
    model.load_state_dict(torch.load(MODEL_PATH))
    model.eval()

    params_count = count_parameters(model)
    print(f"Model parameters: {params_count:,}")
    print("Model loaded")

    print("\nRunning evaluation on test set (3D Volumetric Evaluation)")
    
    # Structure: volume_data[vol_id][slice_idx] = (pred, gt)
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
    
    # Store results: per_class_dice = {1: [], 2: [], 3: []}
    per_class_dice = {c: [] for c in range(1, NUM_CLASSES)}
    per_class_hd95 = {c: [] for c in range(1, NUM_CLASSES)}
    
    print("\nComputing 3D Volumetric Metrics...")
    
    sorted_vol_ids = sorted(volume_data.keys())
    for vol_id in tqdm(sorted_vol_ids, desc="Processing Volumes"):
        slices_dict = volume_data[vol_id]
        sorted_slice_indices = sorted(slices_dict.keys())
        
        pred_vol_list = []
        gt_vol_list = []
        
        for s_idx in sorted_slice_indices:
            p_slice, g_slice = slices_dict[s_idx]
            pred_vol_list.append(p_slice)
            gt_vol_list.append(g_slice)
            
        pred_vol = np.stack(pred_vol_list, axis=0) # (D, H, W)
        gt_vol = np.stack(gt_vol_list, axis=0)     # (D, H, W)
        
        # Calculate metric for each foreground class
        for c in range(1, NUM_CLASSES):
            pred_c = (pred_vol == c).astype(int)
            gt_c = (gt_vol == c).astype(int)
            
            if pred_c.sum() > 0 and gt_c.sum() > 0:
                d = metric.binary.dc(pred_c, gt_c)
                h = metric.binary.hd95(pred_c, gt_c)
            elif pred_c.sum() == 0 and gt_c.sum() == 0:
                d = 1.0
                h = np.nan
            else:
                d = 0.0
                h = np.nan
            
            per_class_dice[c].append(d)
            per_class_hd95[c].append(h)

    # Aggregated Results
    print(f"\n{'='*60}")
    print("TEST SET RESULTS (3D VOLUMETRIC)")
    print(f"{'='*60}")
    
    print(f"Model: {os.path.basename(MODEL_PATH)}")
    print(f"Params: {params_count:,}\n")
    
    print("Per-Class 3D Metrics:")
    print(f"{'Class':<20} | {'Dice':<10} | {'HD95':<10}")
    print("-" * 50)
    
    overall_dice = []
    overall_hd95 = []
    
    for c in range(1, NUM_CLASSES):
        avg_d = np.nanmean(per_class_dice[c])
        avg_h = np.nanmean(per_class_hd95[c])
        
        overall_dice.append(avg_d)
        overall_hd95.append(avg_h)
        
        class_name = MNM_CLASS_MAP.get(c, f"Class {c}")
        print(f"{class_name:<20} | {avg_d:.4f}     | {avg_h:.4f}")
        
    print("-" * 50)
    
    mean_dice = np.nanmean(overall_dice)
    mean_hd95 = np.nanmean(overall_hd95)
    
    print(f"Foreground Mean      | {mean_dice:.4f}     | {mean_hd95:.4f}")
    
    print(f"\n{'='*60}")
    
    # Save results
    os.makedirs("results", exist_ok=True)
    results_file = "results/mnm_test_results_3d.txt"
    with open(results_file, 'w') as f:
        f.write("="*60 + "\n")
        f.write("M&Ms TEST SET RESULTS (3D VOLUMETRIC)\n")
        f.write("="*60 + "\n\n")
        f.write(f"Model: {MODEL_PATH}\n")
        f.write(f"Parameters: {params_count:,}\n\n")
        
        f.write("Per-Class 3D Metrics:\n")
        f.write(f"{'Class':<20} | {'Dice':<10} | {'HD95':<10}\n")
        f.write("-" * 50 + "\n")
        for c in range(1, NUM_CLASSES):
            avg_d = np.nanmean(per_class_dice[c])
            avg_h = np.nanmean(per_class_hd95[c])
            class_name = MNM_CLASS_MAP.get(c, f"Class {c}")
            f.write(f"{class_name:<20} | {avg_d:.4f}     | {avg_h:.4f}\n")
        f.write("-" * 50 + "\n")
        f.write(f"Foreground Mean      | {mean_dice:.4f}     | {mean_hd95:.4f}\n")
        
    print(f"Results saved to: {results_file}")

if __name__ == "__main__":
    main()

