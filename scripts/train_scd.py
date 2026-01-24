"""
SCD Training Script
Adapted from ACDC Training Script.
"""

import torch
import torch.nn as nn
import torch.optim as optim
import torch.nn.functional as F
import numpy as np
import time
import os
import sys
from sklearn.model_selection import train_test_split
from torch.utils.data import DataLoader
import albumentations as A
from albumentations.pytorch import ToTensorV2
from itertools import chain
import cv2
from tqdm import tqdm
from monai.metrics import compute_hausdorff_distance  # Added for HD95
import warnings

warnings.filterwarnings("ignore")

# Add project root to path
sys.path.insert(0, os.path.abspath(os.path.join(os.path.dirname(__file__), '..')))

from src.models.unet import PIE_UNet
from src.models.epure import ePURE
from src.modules.losses import CombinedLoss
from src.data_utils.scd_dataset_optimized import (
    SCDDataset25DOptimized,
    get_scd_volume_ids
)
from src.utils.helpers import calculate_ultimate_common_b1_map


# =============================================================================
# CONFIGURATION
# =============================================================================

NUM_EPOCHS = 250
NUM_CLASSES = 2  # BG, LV (MYO is mapped to BG)
LEARNING_RATE = 1e-3
DEVICE = torch.device('cuda' if torch.cuda.is_available() else 'cpu')
IMG_SIZE = 224
BATCH_SIZE = 24
NUM_SLICES = 5  # 2.5D: 5 slices
EARLY_STOP_PATIENCE = 30

# SCD class mapping
SCD_CLASS_MAP = {0: 'BG', 1: 'LV'}
# Rules might need adjustment if class indices differ, but for now assuming 1=LV, 2=MYO
# If rules rely on specific anatomy, we should check. 
# For now, we'll keep the indices but be aware.
CLASS_INDICES_FOR_RULES = {'LV': 1} 

print(f"Device: {DEVICE}")
print(f"Configuration: {NUM_EPOCHS} epochs, batch size {BATCH_SIZE}, {NUM_SLICES} slices (2.5D)")


# =============================================================================
# AUGMENTATION PIPELINES
# =============================================================================

train_transform = A.Compose([
    A.Rotate(limit=20, p=0.7),
    A.HorizontalFlip(p=0.5),
    A.ElasticTransform(p=0.5, alpha=120, sigma=120 * 0.05),
    A.Affine(
        scale=(0.8, 1.2),
        translate_percent=(-0.1, 0.1),
        rotate=(-25, 25),
        p=0.8,
        border_mode=cv2.BORDER_CONSTANT
    ),
    A.RandomBrightnessContrast(p=0.5),
    A.GaussNoise(p=0.3),
    ToTensorV2(),
])

val_test_transform = A.Compose([
    ToTensorV2(),
])


# =============================================================================
# EVALUATION METRICS
# =============================================================================

def evaluate_metrics(model, dataloader, device, num_classes=3):
    """
    Hàm đánh giá các chỉ số cho mô hình phân đoạn.
    Updated to include HD95.
    """
    model.eval()
    tp = [0] * num_classes
    fp = [0] * num_classes
    fn = [0] * num_classes
    dice_s = [0.0] * num_classes
    iou_s = [0.0] * num_classes
    hd95_s = [0.0] * num_classes
    hd95_counts = [0] * num_classes  # Track valid HD95 batches per class
    
    batches = 0
    total_correct_pixels = 0
    total_pixels = 0

    with torch.no_grad():
        eval_pbar = tqdm(dataloader, desc="Evaluating", leave=False, ncols=80)
        for imgs, tgts in eval_pbar:
            imgs, tgts = imgs.to(device), tgts.to(device)
            if imgs.size(0) == 0: 
                continue
            
            # Model returns list of logits (deep supervision)
            logits_list, _ = model(imgs)
            
            # Use final output (most detailed)
            logits = logits_list[-1]
            
            preds = torch.argmax(F.softmax(logits, dim=1), dim=1)
            batches += 1
            total_correct_pixels += (preds == tgts).sum().item()
            total_pixels += tgts.numel()

            # --- Basic Metrics (Dice, IoU, Precision, Recall) ---
            for c in range(num_classes):
                pc_f = (preds == c).float().view(-1)
                tc_f = (tgts == c).float().view(-1)
                inter = (pc_f * tc_f).sum()

                dice_s[c] += ((2. * inter + 1e-6) / (pc_f.sum() + tc_f.sum() + 1e-6)).item()
                iou_s[c] += ((inter + 1e-6) / (pc_f.sum() + tc_f.sum() - inter + 1e-6)).item()
                tp[c] += inter.item()
                fp[c] += (pc_f.sum() - inter).item()
                fn[c] += (tc_f.sum() - inter).item()

            # --- HD95 Calculation (CPU to avoid VRAM leak) ---
            # Move to CPU first
            preds_cpu = preds.detach().cpu()
            tgts_cpu = tgts.detach().cpu()
            
            preds_oh = F.one_hot(preds_cpu, num_classes=num_classes).permute(0, 3, 1, 2).float()
            tgts_oh = F.one_hot(tgts_cpu, num_classes=num_classes).permute(0, 3, 1, 2).float()
            
            try:
                # compute_hausdorff_distance returns (Batch, Class)
                hd95_batch = compute_hausdorff_distance(
                    y_pred=preds_oh, 
                    y=tgts_oh, 
                    include_background=True, 
                    percentile=95.0
                )
                
                for c in range(num_classes):
                    # Check for valid HD95 values (not NaN, not Inf)
                    valid_vals = hd95_batch[:, c]
                    mask = ~torch.isnan(valid_vals) & ~torch.isinf(valid_vals)
                    
                    if mask.any():
                        hd95_s[c] += valid_vals[mask].sum().item()
                        hd95_counts[c] += mask.sum().item()
            except Exception:
                # Fallback if something fails in HD95 (e.g. empty batch shapes)
                pass

    metrics = {
        'accuracy': 0.0, 
        'dice_scores': [], 
        'iou': [], 
        'hd95': [], 
        'precision': [], 
        'recall': [], 
        'f1_score': []
    }

    if batches > 0:
        if total_pixels > 0:
            metrics['accuracy'] = total_correct_pixels / total_pixels
        
        for c in range(num_classes):
            metrics['dice_scores'].append(dice_s[c] / batches)
            metrics['iou'].append(iou_s[c] / batches)
            
            # HD95 average
            if hd95_counts[c] > 0:
                metrics['hd95'].append(hd95_s[c] / hd95_counts[c])
            else:
                metrics['hd95'].append(float('nan'))

            prec = tp[c] / (tp[c] + fp[c] + 1e-6)
            rec = tp[c] / (tp[c] + fn[c] + 1e-6)
            metrics['precision'].append(prec)
            metrics['recall'].append(rec)
            metrics['f1_score'].append(2 * prec * rec / (prec + rec + 1e-6) if (prec + rec > 0) else 0.0)
    else:
        for _ in range(num_classes):
            [metrics[key].append(0.0) for key in ['dice_scores', 'iou', 'hd95', 'precision', 'recall', 'f1_score']]
            
    return metrics


# =============================================================================
# DATA LOADING (OPTIMIZED with Memmap + LRU Cache)
# =============================================================================

print("LOADING SCD DATA (Optimized)")

# Preprocessed data paths
PREPROCESSED_ROOT = 'preprocessed_data/SCD'
train_npy_dir = os.path.join(PREPROCESSED_ROOT, 'training')
val_npy_dir = os.path.join(PREPROCESSED_ROOT, 'validate')
test_npy_dir = os.path.join(PREPROCESSED_ROOT, 'testing')

# Get volume IDs - Data is ALREADY split
print(f"Loading train volume IDs from: {train_npy_dir}")
train_volume_ids = get_scd_volume_ids(train_npy_dir, frame_type=None)

print(f"Loading val volume IDs from: {val_npy_dir}")
val_volume_ids = get_scd_volume_ids(val_npy_dir, frame_type=None)

print(f"Loading test volume IDs from: {test_npy_dir}")
test_volume_ids = get_scd_volume_ids(test_npy_dir, frame_type=None)

print(f"Split (Dynamic Frames, LV Only): {len(train_volume_ids)} train, {len(val_volume_ids)} val, {len(test_volume_ids)} test volumes")


# =============================================================================
# EPURE AUGMENTATION MODEL
# =============================================================================

print("INITIALIZING ePURE AUGMENTATION")

# Initialize ePURE model for noise injection
ePURE_augmenter = ePURE(in_channels=NUM_SLICES).to(DEVICE)
ePURE_augmenter.eval()

# Create optimized datasets
train_dataset = SCDDataset25DOptimized(
    npy_dir=train_npy_dir,
    volume_ids=train_volume_ids,
    num_input_slices=NUM_SLICES, 
    transforms=train_transform,
    noise_injector_model=ePURE_augmenter,
    device=str(DEVICE),
    max_cache_size=15,
    use_memmap=True
)

val_dataset = SCDDataset25DOptimized(
    npy_dir=val_npy_dir,
    volume_ids=val_volume_ids,
    num_input_slices=NUM_SLICES, 
    transforms=val_test_transform,
    max_cache_size=10,
    use_memmap=True
)

test_dataset = SCDDataset25DOptimized(
    npy_dir=test_npy_dir,
    volume_ids=test_volume_ids,
    num_input_slices=NUM_SLICES, 
    transforms=val_test_transform,
    max_cache_size=8,
    use_memmap=True
)

# Create dataloaders
train_dataloader = DataLoader(
    train_dataset, 
    batch_size=BATCH_SIZE, 
    shuffle=True, 
    num_workers=0,  # Must be 0 for ePURE
    pin_memory=True
)

val_dataloader = DataLoader(
    val_dataset, 
    batch_size=BATCH_SIZE, 
    shuffle=False, 
    num_workers=0,
    pin_memory=True
)

test_dataloader = DataLoader(
    test_dataset, 
    batch_size=BATCH_SIZE, 
    shuffle=False, 
    num_workers=0,
    pin_memory=True
)

print(f"Training slices: {len(train_dataset)}, Validation slices: {len(val_dataset)}, Test slices: {len(test_dataset)}")


# =============================================================================
# B1 MAP CALCULATION
# =============================================================================

print("CALCULATING B1 MAP")

def convert_npy_to_tensor_for_b1(npy_dir, volume_ids):
    """Load .npy volumes and convert to tensor for B1 map calculation"""
    volumes_dir = os.path.join(npy_dir, 'volumes')
    all_slices = []
    
    for vid in volume_ids:
        vol_path = os.path.join(volumes_dir, f'{vid}.npy')
        vol = np.load(vol_path)  # Shape: (H, W, Z)
        
        for i in range(vol.shape[2]):
            all_slices.append(torch.from_numpy(vol[:, :, i]).unsqueeze(0))
    
    return torch.stack(all_slices, dim=0).float()

# Combine all images from train/val/test splits
print("Converting volumes to tensor for B1 map...")
# Note: This might be memory intensive if dataset is huge. 
# If it crashes, we might need to compute B1 map on a subset or incrementally.
try:
    train_tensor = convert_npy_to_tensor_for_b1(train_npy_dir, train_volume_ids)
    val_tensor = convert_npy_to_tensor_for_b1(val_npy_dir, val_volume_ids)
    test_tensor = convert_npy_to_tensor_for_b1(test_npy_dir, test_volume_ids)
    all_images_tensor = torch.cat([train_tensor, val_tensor, test_tensor], dim=0)
    
    print(f"Total slices for B1 map: {all_images_tensor.shape[0]}")
    
    # Calculate or load B1 map
    dataset_name = "scd_cardiac"
    common_b1_map = calculate_ultimate_common_b1_map(
        all_images=all_images_tensor,
        device=str(DEVICE),
        save_path=f"b1_maps/{dataset_name}_ultimate_common_b1_map.pth"
    )
except Exception as e:
    print(f"Warning: Could not calculate B1 map from full dataset: {e}")
    print("Using a random initialization for B1 map as fallback.")
    common_b1_map = torch.randn(1, 1, IMG_SIZE, IMG_SIZE).to(DEVICE)


# =============================================================================
# MODEL, LOSS, OPTIMIZER
# =============================================================================

print("INITIALIZING MODEL & TRAINING COMPONENTS")

# Initialize model
model = PIE_UNet(n_channels=NUM_SLICES, n_classes=NUM_CLASSES, deep_supervision=True).to(DEVICE)

# Count parameters
total_params = sum(p.numel() for p in model.parameters() if p.requires_grad)
print(f"Model total parameters: {total_params:,}")

# Initialize loss
criterion = CombinedLoss(
    num_classes=NUM_CLASSES,
    initial_loss_weights=[0.4, 0.4, 0.2],  
    class_indices_for_rules=None  
).to(DEVICE)

# Optimizer
optimizer = torch.optim.AdamW(
    chain(model.parameters(), criterion.parameters()), 
    lr=LEARNING_RATE,
    weight_decay=1e-4
)

# Learning rate scheduler
scheduler = torch.optim.lr_scheduler.ReduceLROnPlateau(
    optimizer, mode='max', factor=0.5, patience=10
)

print("All components initialized successfully!")


# =============================================================================
# TRAINING LOOP
# =============================================================================

print("STARTING TRAINING")

best_val_metric = 0.0
# New trackers
best_val_hd95 = float('inf')
best_val_overall = float('-inf')

epochs_no_improve = 0
model_save_name_dice = "weights/best_model_scd_scratch.pth"
# New save names
model_save_name_hd95 = "weights/best_model_scd_hd95.pth"
model_save_name_overall = "weights/best_model_scd_overall.pth"

for epoch in range(NUM_EPOCHS):
    print(f"\n{'='*60}")
    print(f"EPOCH {epoch + 1}/{NUM_EPOCHS}")
    print(f"{'='*60}")
    
    # --- Training Phase ---
    model.train()
    epoch_train_loss = 0.0
    train_start_time = time.time()
    
    # Progress bar for training
    train_pbar = tqdm(train_dataloader, desc=f"Training Epoch {epoch+1}/{NUM_EPOCHS}", 
                      leave=False, ncols=100)
    
    for batch_idx, (images, targets) in enumerate(train_pbar):
        images, targets = images.to(DEVICE), targets.to(DEVICE)
        
        optimizer.zero_grad()
        
        # Expand B1 map for batch
        b1_map_for_loss = common_b1_map.expand(images.size(0), -1, -1, -1)
        
        # Forward pass
        logits_list, all_eps_sigma_tuples = model(images)

        # Calculate loss
        total_loss = torch.tensor(0.0, device=DEVICE, requires_grad=True)
        for logits in logits_list:
            if logits.shape[2:] != targets.shape[1:]:
                resized_targets = F.interpolate(
                    targets.unsqueeze(1).float(), 
                    size=logits.shape[2:], 
                    mode='nearest'
                ).squeeze(1).long()
            else:
                resized_targets = targets
            
            loss_component = criterion(logits, resized_targets, b1_map_for_loss, all_eps_sigma_tuples)
            total_loss = total_loss + loss_component
        
        loss = total_loss / len(logits_list)
        
        loss.backward()
        optimizer.step()
        
        epoch_train_loss += loss.item()
        train_pbar.set_postfix({'loss': f'{loss.item():.4f}'})
    
    avg_train_loss = epoch_train_loss / len(train_dataloader)
    train_time = time.time() - train_start_time
    
    print(f"Training Loss: {avg_train_loss:.4f} (Time: {train_time:.1f}s)")
    
    loss_weights = criterion.get_current_loss_weights()
    print(f"Loss Weights: FL={loss_weights['weight_FocalLoss']:.3f}, "
          f"FTL={loss_weights['weight_FocalTverskyLoss']:.3f}, "
          f"Physics={loss_weights['weight_Physics']:.3f}")

    # --- Validation Phase ---
    if val_dataloader and len(val_dataset) > 0:
        print("\nEvaluating on validation set...")
        val_start_time = time.time()
        val_metrics = evaluate_metrics(model, val_dataloader, DEVICE, NUM_CLASSES)
        val_time = time.time() - val_start_time
        
        # Clear cache after validation to release HD95 memory if any
        torch.cuda.empty_cache()
        
        val_accuracy = val_metrics['accuracy']
        all_dice = val_metrics['dice_scores']
        all_iou = val_metrics['iou']
        all_hd95 = val_metrics['hd95']  # Added
        all_precision = val_metrics['precision']
        all_recall = val_metrics['recall']
        all_f1 = val_metrics['f1_score']
        
        # Average foreground classes (1, 2)
        avg_fg_dice = np.mean(all_dice[1:])
        avg_fg_iou = np.mean(all_iou[1:])
        
        # Calculate Average Foreground HD95 (ignoring NaNs)
        fg_hd95_vals = [h for h in all_hd95[1:] if not np.isnan(h)]
        if len(fg_hd95_vals) > 0:
            avg_fg_hd95 = np.mean(fg_hd95_vals)
        else:
            avg_fg_hd95 = float('inf')

        avg_fg_precision = np.mean(all_precision[1:])
        avg_fg_recall = np.mean(all_recall[1:])
        avg_fg_f1 = np.mean(all_f1[1:])
        
        # Calculate Overall Score (Dice + 1/(HD95+1))
        safe_hd95 = avg_fg_hd95 if avg_fg_hd95 != float('inf') else 100.0
        overall_score = avg_fg_dice + (1.0 / (safe_hd95 + 1.0))
        
        current_lr = optimizer.param_groups[0]['lr']
        
        print(f"   --- Per-Class Metrics ---")
        for c_idx in range(NUM_CLASSES):
            class_name = SCD_CLASS_MAP.get(c_idx, f"Class {c_idx}")
            hd_str = f"{all_hd95[c_idx]:.4f}" if not np.isnan(all_hd95[c_idx]) else "NaN"
            print(f"=> {class_name:<15}: Dice: {all_dice[c_idx]:.4f}, IoU: {all_iou[c_idx]:.4f}, HD95: {hd_str}, "
                  f"Precision: {all_precision[c_idx]:.4f}, Recall: {all_recall[c_idx]:.4f}, F1: {all_f1[c_idx]:.4f}")
        
        print(f"   --- Summary Metrics ---")
        print(f"=> Avg Foreground: Dice: {avg_fg_dice:.4f}, IoU: {avg_fg_iou:.4f}, HD95: {avg_fg_hd95:.4f}, "
              f"Precision: {avg_fg_precision:.4f}, Recall: {avg_fg_recall:.4f}, F1: {avg_fg_f1:.4f}")
        print(f"=> Overall Score: {overall_score:.4f}")
        print(f"=> Overall Accuracy: {val_accuracy:.4f} | Current Learning Rate: {current_lr:.6f}")
        
        scheduler.step(avg_fg_dice)
        
        # --- Save Best Dice (Original logic) ---
        if avg_fg_dice > best_val_metric:
            best_val_metric = avg_fg_dice
            torch.save(model.state_dict(), model_save_name_dice)
            print(f"\n  ✓ New best model (DICE) saved! Avg Foreground Dice: {best_val_metric:.4f}")
            epochs_no_improve = 0
            # Also save as generic for compatibility locally if needed
            # torch.save(model.state_dict(), "weights/best_model_scd_scratch.pth") 
        else:
            epochs_no_improve += 1
            print(f"  No improvement for {epochs_no_improve} epoch(s)")

        # --- Save Best HD95 ---
        if avg_fg_hd95 < best_val_hd95:
            best_val_hd95 = avg_fg_hd95
            torch.save(model.state_dict(), model_save_name_hd95)
            print(f"  ✓ New best model (HD95) saved! Avg Foreground HD95: {best_val_hd95:.4f}")

        # --- Save Best Overall ---
        if overall_score > best_val_overall:
            best_val_overall = overall_score
            torch.save(model.state_dict(), model_save_name_overall)
            print(f"  ✓ New best model (OVERALL) saved! Score: {best_val_overall:.4f}")

    else:
        print("\nValidation dataset is empty. Skipping validation.")
    
    if epochs_no_improve >= EARLY_STOP_PATIENCE:
        print(f"\nEarly stopping triggered after {EARLY_STOP_PATIENCE} epochs with no improvement.")
        break

print("TRAINING COMPLETED")
print(f"Best validation Dice score: {best_val_metric:.4f}")
print(f"Best validation HD95 score: {best_val_hd95:.4f}")
print(f"Best validation Overall score: {best_val_overall:.4f}")
print(f"Model saved as: {model_save_name_dice}")


# =============================================================================
# TEST SET EVALUATION
# =============================================================================

print("EVALUATING ON TEST SET")

print("\nLoading best model (DICE)...")
model.load_state_dict(torch.load(model_save_name_dice))
model.eval()

print("Running evaluation on test set...")
test_start_time = time.time()
test_metrics = evaluate_metrics(model, test_dataloader, DEVICE, NUM_CLASSES)
test_time = time.time() - test_start_time

test_accuracy = test_metrics['accuracy']
test_dice = test_metrics['dice_scores']
test_iou = test_metrics['iou']
test_hd95 = test_metrics['hd95']
test_precision = test_metrics['precision']
test_recall = test_metrics['recall']
test_f1 = test_metrics['f1_score']

mean_dice_all = np.mean(test_dice)
mean_iou_all = np.mean(test_iou)
mean_precision_all = np.mean(test_precision)
mean_recall_all = np.mean(test_recall)
mean_f1_all = np.mean(test_f1)

avg_fg_dice_test = np.mean(test_dice[1:])
avg_fg_iou_test = np.mean(test_iou[1:])

# Calculate valid Test HD95
valid_fg_hd95_test = [h for h in test_hd95[1:] if not np.isnan(h)]
avg_fg_hd95_test = np.mean(valid_fg_hd95_test) if len(valid_fg_hd95_test) > 0 else float('nan')


print(f"\n{'='*60}")
print("TEST SET RESULTS")
print(f"{'='*60}")

print(f"\n  Test Results (Mean of ALL {NUM_CLASSES} Classes):")
print(f"    Accuracy: {test_accuracy:.4f}; Dice: {mean_dice_all:.4f}; IoU: {mean_iou_all:.4f}; "
      f"Precision: {mean_precision_all:.4f}; Recall: {mean_recall_all:.4f}; F1-score: {mean_f1_all:.4f}")

print(f"\n  Per-Class Metrics:")
for c_idx in range(NUM_CLASSES):
    class_name = SCD_CLASS_MAP.get(c_idx, f"Class {c_idx}")
    hd_str = f"{test_hd95[c_idx]:.4f}" if not np.isnan(test_hd95[c_idx]) else "NaN"
    print(f"    => {class_name:<20}: "
          f"Dice: {test_dice[c_idx]:.4f}, "
          f"IoU: {test_iou[c_idx]:.4f}, "
          f"HD95: {hd_str}, "
          f"Precision: {test_precision[c_idx]:.4f}, "
          f"Recall: {test_recall[c_idx]:.4f}, "
          f"F1: {test_f1[c_idx]:.4f}")

print(f"\n  Additional Info (Foreground Classes Only):")
print(f"    Avg Foreground Dice: {avg_fg_dice_test:.4f}")
print(f"    Avg Foreground IoU:  {avg_fg_iou_test:.4f}")
print(f"    Avg Foreground HD95: {avg_fg_hd95_test:.4f}")
print(f"    Evaluation Time:     {test_time:.1f}s")

print(f"\n{'='*60}")
print("EXPERIMENT COMPLETE")
print(f"{'='*60}")