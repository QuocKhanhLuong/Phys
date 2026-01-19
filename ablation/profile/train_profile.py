"""
Training Script for PIE-UNet Profile Ablation Study

Based on train_acdc.py but with configurable model profiles.
Trains and evaluates DICE/HD95 for each profile.
"""

import torch
import torch.nn as nn
import torch.optim as optim
import torch.nn.functional as F
import numpy as np
import time
import os
import sys
import argparse
from sklearn.model_selection import train_test_split
from torch.utils.data import DataLoader
import albumentations as A
from albumentations.pytorch import ToTensorV2
from itertools import chain
import cv2
from tqdm import tqdm
from monai.metrics import compute_hausdorff_distance
import warnings

warnings.filterwarnings("ignore")

# Add project root to path
PROJECT_ROOT = os.path.abspath(os.path.join(os.path.dirname(__file__), '..', '..'))
sys.path.insert(0, PROJECT_ROOT)

from ablation.profile.config import PROFILE_CONFIGS, TRAINING_CONFIG, DATA_CONFIG, OUTPUT_CONFIG
from ablation.profile.pie_unet import PIE_UNet
from src.models.epure import ePURE
from src.modules.losses import CombinedLoss
from src.data_utils.acdc_dataset_optimized import (
    ACDCDataset25DOptimized,
    get_acdc_volume_ids,
)
from src.utils.helpers import calculate_ultimate_common_b1_map


# =============================================================================
# CONFIGURATION
# =============================================================================

DEVICE = torch.device('cuda' if torch.cuda.is_available() else 'cpu')
ACDC_CLASS_MAP = {0: 'BG', 1: 'RV', 2: 'MYO', 3: 'LV'}


# =============================================================================
# AUGMENTATION PIPELINES
# =============================================================================

train_transform = A.Compose([
    A.Rotate(limit=20, p=0.7),
    A.HorizontalFlip(p=0.5),
    A.ElasticTransform(p=0.5, alpha=120, sigma=120 * 0.05),
    A.Affine(
        scale=(0.9, 1.1),
        translate_percent=(-0.0625, 0.0625),
        rotate=(-15, 15),
        p=0.7,
        border_mode=cv2.BORDER_CONSTANT
    ),
    A.RandomBrightnessContrast(p=0.5),
    ToTensorV2(),
])

val_test_transform = A.Compose([
    ToTensorV2(),
])


# =============================================================================
# TEST TIME AUGMENTATION (TTA)
# =============================================================================

def apply_tta(model, images):
    """Apply Test Time Augmentation and average predictions.
    
    Augmentations:
    - Original
    - Horizontal flip
    - Vertical flip  
    - 90° rotation
    - 180° rotation
    - 270° rotation
    """
    model.eval()
    all_logits = []
    
    with torch.no_grad():
        # Original
        logits_list, _ = model(images)
        all_logits.append(logits_list[-1])
        
        # Horizontal flip
        flipped_h = torch.flip(images, dims=[3])
        logits_h, _ = model(flipped_h)
        all_logits.append(torch.flip(logits_h[-1], dims=[3]))
        
        # Vertical flip
        flipped_v = torch.flip(images, dims=[2])
        logits_v, _ = model(flipped_v)
        all_logits.append(torch.flip(logits_v[-1], dims=[2]))
        
        # 90° rotation
        rot90 = torch.rot90(images, k=1, dims=[2, 3])
        logits_90, _ = model(rot90)
        all_logits.append(torch.rot90(logits_90[-1], k=-1, dims=[2, 3]))
        
        # 180° rotation
        rot180 = torch.rot90(images, k=2, dims=[2, 3])
        logits_180, _ = model(rot180)
        all_logits.append(torch.rot90(logits_180[-1], k=-2, dims=[2, 3]))
        
        # 270° rotation
        rot270 = torch.rot90(images, k=3, dims=[2, 3])
        logits_270, _ = model(rot270)
        all_logits.append(torch.rot90(logits_270[-1], k=-3, dims=[2, 3]))
    
    # Average all predictions
    avg_logits = torch.mean(torch.stack(all_logits), dim=0)
    return avg_logits


# =============================================================================
# EVALUATION METRICS
# =============================================================================

def evaluate_metrics(model, dataloader, device, num_classes=4, use_tta=False):
    """Evaluate segmentation metrics with optional TTA."""
    model.eval()
    dice_s = [0.0] * num_classes
    hd95_s = [0.0] * num_classes
    hd95_counts = [0] * num_classes
    batches = 0

    with torch.no_grad():
        for imgs, tgts in tqdm(dataloader, desc="Evaluating", leave=False):
            imgs, tgts = imgs.to(device), tgts.to(device)
            if imgs.size(0) == 0:
                continue
            
            if use_tta:
                logits = apply_tta(model, imgs)
            else:
                logits_list, _ = model(imgs)
                logits = logits_list[-1]
            preds = torch.argmax(F.softmax(logits, dim=1), dim=1)
            batches += 1

            for c in range(num_classes):
                pc_f = (preds == c).float().view(-1)
                tc_f = (tgts == c).float().view(-1)
                inter = (pc_f * tc_f).sum()
                dice_s[c] += ((2. * inter + 1e-6) / (pc_f.sum() + tc_f.sum() + 1e-6)).item()

            # HD95 on CPU
            preds_cpu = preds.detach().cpu()
            tgts_cpu = tgts.detach().cpu()
            preds_oh = F.one_hot(preds_cpu, num_classes=num_classes).permute(0, 3, 1, 2).float()
            tgts_oh = F.one_hot(tgts_cpu, num_classes=num_classes).permute(0, 3, 1, 2).float()
            
            try:
                hd95_batch = compute_hausdorff_distance(y_pred=preds_oh, y=tgts_oh, include_background=True, percentile=95.0)
                for c in range(num_classes):
                    valid_vals = hd95_batch[:, c]
                    mask = ~torch.isnan(valid_vals) & ~torch.isinf(valid_vals)
                    if mask.any():
                        hd95_s[c] += valid_vals[mask].sum().item()
                        hd95_counts[c] += mask.sum().item()
            except:
                pass

    metrics = {'dice_scores': [], 'hd95': []}
    if batches > 0:
        for c in range(num_classes):
            metrics['dice_scores'].append(dice_s[c] / batches)
            metrics['hd95'].append(hd95_s[c] / hd95_counts[c] if hd95_counts[c] > 0 else float('nan'))
    
    return metrics


# =============================================================================
# TRAINING FUNCTION
# =============================================================================

def train_profile(profile_name, num_epochs=None, batch_size=None, quick_test=False):
    """Train a specific profile and return metrics."""
    
    config = PROFILE_CONFIGS[profile_name]
    num_slices = config["n_channels"]
    num_classes = TRAINING_CONFIG["num_classes"]
    batch_size = batch_size if batch_size is not None else TRAINING_CONFIG["batch_size"]
    learning_rate = TRAINING_CONFIG["learning_rate"]
    
    if num_epochs is None:
        num_epochs = 5 if quick_test else TRAINING_CONFIG["num_epochs"]
    
    print(f"\n{'='*60}")
    print(f"Training Profile: {config['name']}")
    print(f"  n_channels: {num_slices}, depth: {config['depth']}")
    print(f"  epochs: {num_epochs}, batch_size: {batch_size}")
    print(f"{'='*60}")
    
    # Data paths
    train_npy_dir = str(DATA_CONFIG["train_dir"])
    test_npy_dir = str(DATA_CONFIG["test_dir"])
    
    # Get volume IDs and split
    all_train_volume_ids = get_acdc_volume_ids(train_npy_dir)
    indices = list(range(len(all_train_volume_ids)))
    train_indices, val_indices = train_test_split(indices, test_size=0.2, random_state=42)
    train_volume_ids = [all_train_volume_ids[i] for i in train_indices]
    val_volume_ids = [all_train_volume_ids[i] for i in val_indices]
    
    print(f"Data split: {len(train_volume_ids)} train, {len(val_volume_ids)} val volumes")
    
    # Initialize ePURE for noise injection
    ePURE_augmenter = ePURE(in_channels=num_slices).to(DEVICE)
    ePURE_augmenter.eval()
    
    # Create datasets
    train_dataset = ACDCDataset25DOptimized(
        npy_dir=train_npy_dir,
        volume_ids=train_volume_ids,
        num_input_slices=num_slices,
        transforms=train_transform,
        noise_injector_model=ePURE_augmenter,
        device=str(DEVICE),
        max_cache_size=15,
        use_memmap=True
    )
    
    val_dataset = ACDCDataset25DOptimized(
        npy_dir=train_npy_dir,
        volume_ids=val_volume_ids,
        num_input_slices=num_slices,
        transforms=val_test_transform,
        max_cache_size=10,
        use_memmap=True
    )
    
    train_dataloader = DataLoader(train_dataset, batch_size=batch_size, shuffle=True, num_workers=0, pin_memory=True)
    val_dataloader = DataLoader(val_dataset, batch_size=batch_size, shuffle=False, num_workers=0, pin_memory=True)
    
    print(f"Training slices: {len(train_dataset)}, Validation slices: {len(val_dataset)}")
    
    # Calculate B1 map
    print("Calculating B1 map...")
    def convert_npy_to_tensor_for_b1(npy_dir, volume_ids):
        volumes_dir = os.path.join(npy_dir, 'volumes')
        all_slices = []
        for vid in volume_ids:
            vol_path = os.path.join(volumes_dir, f'{vid}.npy')
            vol = np.load(vol_path)
            for i in range(vol.shape[2]):
                all_slices.append(torch.from_numpy(vol[:, :, i]).unsqueeze(0))
        return torch.stack(all_slices, dim=0).float()
    
    train_val_tensor = convert_npy_to_tensor_for_b1(train_npy_dir, train_volume_ids + val_volume_ids)
    common_b1_map = calculate_ultimate_common_b1_map(
        all_images=train_val_tensor,
        device=str(DEVICE),
        save_path=f"b1_maps/ablation_{profile_name}_b1_map.pth"
    )
    
    # Initialize model
    model = PIE_UNet(
        n_channels=num_slices,
        n_classes=num_classes,
        depth=config["depth"],
        base_filters=config["base_filters"],
        deep_supervision=True
    ).to(DEVICE)
    
    total_params = sum(p.numel() for p in model.parameters() if p.requires_grad)
    print(f"Model parameters: {total_params:,}")
    
    # Loss and optimizer
    criterion = CombinedLoss(num_classes=num_classes, initial_loss_weights=[0.4, 0.4, 0.2], class_indices_for_rules=None).to(DEVICE)
    optimizer = torch.optim.AdamW(chain(model.parameters(), criterion.parameters()), lr=learning_rate)
    scheduler = torch.optim.lr_scheduler.ReduceLROnPlateau(optimizer, mode='max', factor=0.5, patience=10)
    
    # Training loop
    best_dice = 0.0
    best_hd95 = float('inf')
    epochs_no_improve = 0
    weights_path = OUTPUT_CONFIG["weights_dir"] / f"best_model_{profile_name}.pth"
    
    for epoch in range(num_epochs):
        model.train()
        epoch_loss = 0.0
        
        # Reset peak memory tracking at start of epoch
        if torch.cuda.is_available():
            torch.cuda.reset_peak_memory_stats()
        
        train_pbar = tqdm(train_dataloader, desc=f"Epoch {epoch+1}/{num_epochs}", leave=False)
        for images, targets in train_pbar:
            images, targets = images.to(DEVICE), targets.to(DEVICE)
            optimizer.zero_grad()
            
            b1_map_batch = common_b1_map.expand(images.size(0), -1, -1, -1)
            logits_list, all_eps_sigma_tuples = model(images)
            
            total_loss = torch.tensor(0.0, device=DEVICE, requires_grad=True)
            for logits in logits_list:
                if logits.shape[2:] != targets.shape[1:]:
                    resized_targets = F.interpolate(targets.unsqueeze(1).float(), size=logits.shape[2:], mode='nearest').squeeze(1).long()
                else:
                    resized_targets = targets
                total_loss = total_loss + criterion(logits, resized_targets, b1_map_batch, all_eps_sigma_tuples)
            
            loss = total_loss / len(logits_list)
            loss.backward()
            optimizer.step()
            epoch_loss += loss.item()
            train_pbar.set_postfix({'loss': f'{loss.item():.4f}'})
        
        avg_loss = epoch_loss / len(train_dataloader)
        
        # Validation
        val_metrics = evaluate_metrics(model, val_dataloader, DEVICE, num_classes)
        torch.cuda.empty_cache()
        
        avg_fg_dice = np.mean(val_metrics['dice_scores'][1:])
        
        # Track peak GPU memory
        if torch.cuda.is_available():
            peak_gpu_memory_mb = torch.cuda.max_memory_allocated() / (1024 * 1024)
        else:
            peak_gpu_memory_mb = 0.0
        fg_hd95_vals = [h for h in val_metrics['hd95'][1:] if not np.isnan(h)]
        avg_fg_hd95 = np.mean(fg_hd95_vals) if fg_hd95_vals else float('inf')
        
        print(f"Epoch {epoch+1}: Loss={avg_loss:.4f}, Dice={avg_fg_dice:.4f}, HD95={avg_fg_hd95:.2f}")
        
        scheduler.step(avg_fg_dice)
        
        if avg_fg_dice > best_dice:
            best_dice = avg_fg_dice
            best_hd95 = avg_fg_hd95
            torch.save(model.state_dict(), weights_path)
            print(f"  ✓ New best model saved!")
            epochs_no_improve = 0
        else:
            epochs_no_improve += 1
            if epochs_no_improve >= TRAINING_CONFIG["early_stopping_patience"]:
                print(f"\nEarly stopping triggered after {TRAINING_CONFIG['early_stopping_patience']} epochs with no improvement.")
                break
    
    # =========================================================================
    # TEST SET EVALUATION (Same as train_acdc.py)
    # =========================================================================
    print(f"\n{'='*60}")
    print("EVALUATING ON TEST SET")
    print(f"{'='*60}")
    
    # Load best model
    model.load_state_dict(torch.load(weights_path))
    model.eval()
    
    # Create test dataset
    test_volume_ids = get_acdc_volume_ids(test_npy_dir)
    test_dataset = ACDCDataset25DOptimized(
        npy_dir=test_npy_dir,
        volume_ids=test_volume_ids,
        num_input_slices=num_slices,
        transforms=val_test_transform,
        max_cache_size=8,
        use_memmap=True
    )
    test_dataloader = DataLoader(test_dataset, batch_size=batch_size, shuffle=False, num_workers=0, pin_memory=True)
    
    print(f"Test slices: {len(test_dataset)}")
    
    # Evaluate on test set with TTA
    print("Evaluating with TTA (6 augmentations)...")
    test_metrics = evaluate_metrics(model, test_dataloader, DEVICE, num_classes, use_tta=True)
    torch.cuda.empty_cache()
    
    test_fg_dice = np.mean(test_metrics['dice_scores'][1:])
    test_fg_hd95_vals = [h for h in test_metrics['hd95'][1:] if not np.isnan(h)]
    test_fg_hd95 = np.mean(test_fg_hd95_vals) if test_fg_hd95_vals else float('inf')
    
    print(f"\n{'='*60}")
    print(f"TEST SET RESULTS for {config['name']}")
    print(f"{'='*60}")
    print(f"  Per-class Dice: {test_metrics['dice_scores']}")
    print(f"  Avg Foreground Dice: {test_fg_dice:.4f}")
    print(f"  Avg Foreground HD95: {test_fg_hd95:.4f}")
    
    print(f"\nTraining completed for {config['name']}")
    print(f"  Best Val Dice: {best_dice:.4f}")
    print(f"  Test Dice: {test_fg_dice:.4f}")
    print(f"  Test HD95: {test_fg_hd95:.4f}")
    
    return {
        "profile": profile_name,
        "name": config["name"],
        "n_channels": num_slices,
        "depth": config["depth"],
        "params": total_params,
        "best_val_dice": best_dice,
        "best_val_hd95": best_hd95,
        "test_dice": test_fg_dice,
        "test_hd95": test_fg_hd95,
        "peak_gpu_memory_mb": peak_gpu_memory_mb
    }


# =============================================================================
# MAIN
# =============================================================================

if __name__ == "__main__":
    parser = argparse.ArgumentParser(description="Train PIE-UNet profile")
    parser.add_argument("--profile", type=str, required=True, choices=list(PROFILE_CONFIGS.keys()), help="Profile to train")
    parser.add_argument("--epochs", type=int, default=None, help="Number of epochs")
    parser.add_argument("--batch-size", type=int, default=None, help="Batch size (default: 16)")
    parser.add_argument("--quick-test", action="store_true", help="Quick test mode (5 epochs)")
    args = parser.parse_args()
    
    print(f"Device: {DEVICE}")
    result = train_profile(args.profile, num_epochs=args.epochs, batch_size=args.batch_size, quick_test=args.quick_test)
    
    print("\n" + "="*60)
    print("FINAL RESULTS")
    print("="*60)
    print(f"Profile: {result['name']}")
    print(f"  n_channels: {result['n_channels']}, depth: {result['depth']}")
    print(f"  Params: {result['params']:,}")
    print(f"  Best Val Dice: {result['best_val_dice']:.4f}")
    print(f"  Best Val HD95: {result['best_val_hd95']:.4f}")
    print(f"  Test Dice: {result['test_dice']:.4f}")
    print(f"  Test HD95: {result['test_hd95']:.4f}")

