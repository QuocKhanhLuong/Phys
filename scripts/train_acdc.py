"""
ACDC Training Script - From Notebook (93.92% Dice Score)
Converted from: final-application-maxwell-for-segmentation-task (3).ipynb
Exact reproduction of the notebook workflow.
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

# Add project root to path
sys.path.insert(0, os.path.abspath(os.path.join(os.path.dirname(__file__), '..')))

from src.models.unet import RobustMedVFL_UNet
from src.models.epure import ePURE
from src.modules.losses import CombinedLoss
from src.data_utils.acdc_dataset_direct import ACDCDataset25D, load_acdc_volumes
from src.utils.helpers import calculate_ultimate_common_b1_map


# =============================================================================
# CONFIGURATION (Exact from Notebook)
# =============================================================================

NUM_EPOCHS = 250
NUM_CLASSES = 4
LEARNING_RATE = 1e-3
DEVICE = torch.device('cuda' if torch.cuda.is_available() else 'cpu')
IMG_SIZE = 224
BATCH_SIZE = 24
NUM_SLICES = 5  # 2.5D: 5 slices
EARLY_STOP_PATIENCE = 30

# ACDC class mapping
ACDC_CLASS_MAP = {0: 'BG', 1: 'RV', 2: 'MYO', 3: 'LV'}  # Like notebook: BG instead of Background
CLASS_INDICES_FOR_RULES = {'RV': 1, 'MYO': 2, 'LV': 3}

print(f"Device: {DEVICE}")
print(f"Configuration: {NUM_EPOCHS} epochs, batch size {BATCH_SIZE}, {NUM_SLICES} slices (2.5D)")


# =============================================================================
# AUGMENTATION PIPELINES (Exact from Notebook)
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
# EVALUATION METRICS (Exact from Notebook)
# =============================================================================

def evaluate_metrics(model, dataloader, device, num_classes=4):
    """
    Hàm đánh giá các chỉ số cho mô hình phân đoạn.
    Đã được cập nhật để tương thích với output dạng list từ UNet++ (deep supervision).
    """
    model.eval()
    tp = [0] * num_classes
    fp = [0] * num_classes
    fn = [0] * num_classes
    dice_s = [0.0] * num_classes
    iou_s = [0.0] * num_classes
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

            for c in range(num_classes):
                pc_f = (preds == c).float().view(-1)
                tc_f = (tgts == c).float().view(-1)
                inter = (pc_f * tc_f).sum()

                dice_s[c] += ((2. * inter + 1e-6) / (pc_f.sum() + tc_f.sum() + 1e-6)).item()
                iou_s[c] += ((inter + 1e-6) / (pc_f.sum() + tc_f.sum() - inter + 1e-6)).item()
                tp[c] += inter.item()
                fp[c] += (pc_f.sum() - inter).item()
                fn[c] += (tc_f.sum() - inter).item()

    metrics = {
        'accuracy': 0.0,
        'dice_scores': [],
        'iou': [],
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
            prec = tp[c] / (tp[c] + fp[c] + 1e-6)
            rec = tp[c] / (tp[c] + fn[c] + 1e-6)
            metrics['precision'].append(prec)
            metrics['recall'].append(rec)
            metrics['f1_score'].append(2 * prec * rec / (prec + rec + 1e-6) if (prec + rec > 0) else 0.0)
    else:
        for _ in range(num_classes):
            [metrics[key].append(0.0) for key in ['dice_scores', 'iou', 'precision', 'recall', 'f1_score']]
            
    return metrics


# =============================================================================
# DATA LOADING (Exact from Notebook)
# =============================================================================

print("LOADING ACDC DATA")

# Data paths
ACDC_ROOT = '/home/linhdang/workspace/minhbao_workspace/Phys/data/ACDC'
train_data_path = os.path.join(ACDC_ROOT, 'training')
test_data_path = os.path.join(ACDC_ROOT, 'testing')

print(f"Loading training volumes from: {train_data_path}")
all_train_volumes, all_train_masks = load_acdc_volumes(
    train_data_path, target_size=(IMG_SIZE, IMG_SIZE)
)
print(f"Loaded {len(all_train_volumes)} training volumes.")

print(f"Loading testing volumes from: {test_data_path}")
all_test_volumes, all_test_masks = load_acdc_volumes(
    test_data_path, target_size=(IMG_SIZE, IMG_SIZE)
)
print(f"Loaded {len(all_test_volumes)} testing volumes.")

# Normalize pixel intensities (0-1 range)
print("Normalizing pixel intensities...")
for i in range(len(all_train_volumes)):
    max_val = np.max(all_train_volumes[i])
    if max_val > 0:
        all_train_volumes[i] /= max_val

for i in range(len(all_test_volumes)):
    max_val = np.max(all_test_volumes[i])
    if max_val > 0:
        all_test_volumes[i] /= max_val

# Split train/validation by volumes (patients)
indices = list(range(len(all_train_volumes)))
train_indices, val_indices = train_test_split(indices, test_size=0.2, random_state=42)

X_train_vols = [all_train_volumes[i] for i in train_indices]
y_train_vols = [all_train_masks[i] for i in train_indices]
X_val_vols = [all_train_volumes[i] for i in val_indices]
y_val_vols = [all_train_masks[i] for i in val_indices]

print(f"Train volumes: {len(X_train_vols)}, Validation volumes: {len(X_val_vols)}")


# =============================================================================
# EPURE AUGMENTATION MODEL (Exact from Notebook)
# =============================================================================

print("INITIALIZING ePURE AUGMENTATION")

# Initialize ePURE model for noise injection
# NOTE: This will be used INSIDE dataset.__getitem__, so num_workers MUST be 0!
ePURE_augmenter = ePURE(in_channels=NUM_SLICES).to(DEVICE)
ePURE_augmenter.eval()

# Create datasets
train_dataset = ACDCDataset25D(
    volumes_list=X_train_vols,
    masks_list=y_train_vols,
    num_input_slices=NUM_SLICES,
    transforms=train_transform,
    noise_injector_model=ePURE_augmenter,  # ePURE augmentation inside dataset
    device=str(DEVICE)
)

val_dataset = ACDCDataset25D(
    volumes_list=X_val_vols,
    masks_list=y_val_vols,
    num_input_slices=NUM_SLICES,
    transforms=val_test_transform
)

test_dataset = ACDCDataset25D(
    volumes_list=all_test_volumes,
    masks_list=all_test_masks,
    num_input_slices=NUM_SLICES,
    transforms=val_test_transform
)

# Create dataloaders - IMPORTANT: num_workers=0 because ePURE uses GPU in __getitem__
train_dataloader = DataLoader(
    train_dataset,
    batch_size=BATCH_SIZE,
    shuffle=True,
    num_workers=0,  # CRITICAL: Must be 0 to avoid CUDA multiprocessing error
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
# B1 MAP CALCULATION (Exact from Notebook)
# =============================================================================

print("CALCULATING B1 MAP")

def convert_volumes_to_tensor(volumes_list):
    """Convert volume list to tensor for B1 map calculation"""
    all_slices = []
    for vol in volumes_list:
        for i in range(vol.shape[2]):
            all_slices.append(torch.from_numpy(vol[:, :, i]).unsqueeze(0))
    return torch.stack(all_slices, dim=0).float()

# Combine all images for common B1 map
all_images_tensor = convert_volumes_to_tensor(X_train_vols + X_val_vols + all_test_volumes)

# Calculate or load B1 map
dataset_name = "acdc_cardiac"
common_b1_map = calculate_ultimate_common_b1_map(
    all_images=all_images_tensor,
    device=str(DEVICE),
    save_path=f"{dataset_name}_ultimate_common_b1_map.pth"
)


# =============================================================================
# MODEL, LOSS, OPTIMIZER (Exact from Notebook)
# =============================================================================

print("INITIALIZING MODEL & TRAINING COMPONENTS")

# Initialize model
model = RobustMedVFL_UNet(n_channels=NUM_SLICES, n_classes=NUM_CLASSES, deep_supervision=True).to(DEVICE)

# Count parameters
total_params = sum(p.numel() for p in model.parameters() if p.requires_grad)
print(f"Model total parameters: {total_params:,}")

# Initialize loss WITHOUT Anatomical Rule Loss (ABLATION STUDY)
criterion = CombinedLoss(
    num_classes=NUM_CLASSES,
    initial_loss_weights=[0.4, 0.4, 0.2],  # FL, FTL, Physics, Anatomical
    class_indices_for_rules=CLASS_INDICES_FOR_RULES
).to(DEVICE)

# Optimizer includes BOTH model and criterion parameters (for dynamic loss weighting)
optimizer = torch.optim.AdamW(
    chain(model.parameters(), criterion.parameters()),
    lr=LEARNING_RATE
)

# Learning rate scheduler (exact from notebook)
scheduler = torch.optim.lr_scheduler.ReduceLROnPlateau(
    optimizer, mode='max', factor=0.5, patience=10
)

print("All components initialized successfully!")


# =============================================================================
# TRAINING LOOP (Exact from Notebook)
# =============================================================================

print("STARTING TRAINING")

best_val_metric = 0.0
epochs_no_improve = 0

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
        
        # Forward pass - returns list of logits (deep supervision) + physics tuples
        logits_list, all_eps_sigma_tuples = model(images)

        # Calculate loss for each output (deep supervision)
        total_loss = torch.tensor(0.0, device=DEVICE, requires_grad=True)
        for logits in logits_list:
            # Handle size mismatch (resize targets if needed)
            if logits.shape[2:] != targets.shape[1:]:
                resized_targets = F.interpolate(
                    targets.unsqueeze(1).float(),
                    size=logits.shape[2:],
                    mode='nearest'
                ).squeeze(1).long()
            else:
                resized_targets = targets
            
            # Compute loss
            loss_component = criterion(logits, resized_targets, b1_map_for_loss, all_eps_sigma_tuples)
            total_loss = total_loss + loss_component
        
        # Average loss over all outputs
        loss = total_loss / len(logits_list)
        
        # Backward pass
        loss.backward()
        optimizer.step()
        
        epoch_train_loss += loss.item()
        
        # Update progress bar
        train_pbar.set_postfix({'loss': f'{loss.item():.4f}'})
    
    avg_train_loss = epoch_train_loss / len(train_dataloader)
    train_time = time.time() - train_start_time
    
    print(f"Training Loss: {avg_train_loss:.4f} (Time: {train_time:.1f}s)")
    
    # Print current loss weights
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
        
        # Extract ALL metrics (like notebook)
        val_accuracy = val_metrics['accuracy']
        all_dice = val_metrics['dice_scores']
        all_iou = val_metrics['iou']
        all_precision = val_metrics['precision']
        all_recall = val_metrics['recall']
        all_f1 = val_metrics['f1_score']
        
        # Average foreground classes (1, 2, 3)
        avg_fg_dice = np.mean(all_dice[1:])
        avg_fg_iou = np.mean(all_iou[1:])
        avg_fg_precision = np.mean(all_precision[1:])
        avg_fg_recall = np.mean(all_recall[1:])
        avg_fg_f1 = np.mean(all_f1[1:])
        
        # Get current learning rate
        current_lr = optimizer.param_groups[0]['lr']
        
        # Print per-class metrics (like notebook)
        print(f"   --- Per-Class Metrics ---")
        for c_idx in range(NUM_CLASSES):
            class_name = ACDC_CLASS_MAP.get(c_idx, f"Class {c_idx}")
            print(f"=> {class_name:<15}: Dice: {all_dice[c_idx]:.4f}, IoU: {all_iou[c_idx]:.4f}, "
                  f"Precision: {all_precision[c_idx]:.4f}, Recall: {all_recall[c_idx]:.4f}, F1: {all_f1[c_idx]:.4f}")
        
        print(f"   --- Summary Metrics ---")
        print(f"=> Avg Foreground: Dice: {avg_fg_dice:.4f}, IoU: {avg_fg_iou:.4f}, "
              f"Precision: {avg_fg_precision:.4f}, Recall: {avg_fg_recall:.4f}, F1: {avg_fg_f1:.4f}")
        print(f"=> Overall Accuracy: {val_accuracy:.4f} | Current Learning Rate: {current_lr:.6f}")
        
        # Update scheduler
        scheduler.step(avg_fg_dice)
        
        # Save best model
        if avg_fg_dice > best_val_metric:
            best_val_metric = avg_fg_dice
            torch.save(model.state_dict(), "best_model_acdc_no_anatomical.pth")
            print(f"\n  ✓ New best model saved! Avg Foreground Dice: {best_val_metric:.4f}")
            epochs_no_improve = 0
        else:
            epochs_no_improve += 1
            print(f"  No improvement for {epochs_no_improve} epoch(s)")
    else:
        print("\nValidation dataset is empty. Skipping validation.")
    
    # Early stopping check
    if epochs_no_improve >= EARLY_STOP_PATIENCE:
        print(f"\nEarly stopping triggered after {EARLY_STOP_PATIENCE} epochs with no improvement.")
        break

print("TRAINING COMPLETED")
print(f"Best validation Dice score: {best_val_metric:.4f}")
print(f"Model saved as: best_model_acdc_no_anatomical.pth")


# =============================================================================
# TEST SET EVALUATION (Using Best Model)
# =============================================================================

print("EVALUATING ON TEST SET")

# Load best model
print("\nLoading best model...")
model.load_state_dict(torch.load("best_model_acdc_no_anatomical.pth"))
model.eval()

# Evaluate on test set
print("Running evaluation on test set...")
test_start_time = time.time()
test_metrics = evaluate_metrics(model, test_dataloader, DEVICE, NUM_CLASSES)
test_time = time.time() - test_start_time

# Extract metrics
test_accuracy = test_metrics['accuracy']
test_dice = test_metrics['dice_scores']
test_iou = test_metrics['iou']
test_precision = test_metrics['precision']
test_recall = test_metrics['recall']
test_f1 = test_metrics['f1_score']

# Calculate averages - LIKE NOTEBOOK: ALL classes (including background)
mean_dice_all = np.mean(test_dice)
mean_iou_all = np.mean(test_iou)
mean_precision_all = np.mean(test_precision)
mean_recall_all = np.mean(test_recall)
mean_f1_all = np.mean(test_f1)

# Also calculate foreground only for comparison
avg_fg_dice_test = np.mean(test_dice[1:])
avg_fg_iou_test = np.mean(test_iou[1:])

print(f"\n{'='*60}")
print("TEST SET RESULTS")
print(f"{'='*60}")

# LIKE NOTEBOOK: Print mean of ALL classes first
print(f"\n  Test Results (Mean of ALL {NUM_CLASSES} Classes):")
print(f"    Accuracy: {test_accuracy:.4f}; Dice: {mean_dice_all:.4f}; IoU: {mean_iou_all:.4f}; "
      f"Precision: {mean_precision_all:.4f}; Recall: {mean_recall_all:.4f}; F1-score: {mean_f1_all:.4f}")

print(f"\n  Per-Class Metrics:")
for c_idx in range(NUM_CLASSES):
    class_name = ACDC_CLASS_MAP.get(c_idx, f"Class {c_idx}")
    print(f"    => {class_name:<20}: "
          f"Dice: {test_dice[c_idx]:.4f}, "
          f"IoU: {test_iou[c_idx]:.4f}, "
          f"Precision: {test_precision[c_idx]:.4f}, "
          f"Recall: {test_recall[c_idx]:.4f}, "
          f"F1: {test_f1[c_idx]:.4f}")

# Additional info: Foreground average
print(f"\n  Additional Info (Foreground Classes Only):")
print(f"    Avg Foreground Dice: {avg_fg_dice_test:.4f}")
print(f"    Avg Foreground IoU:  {avg_fg_iou_test:.4f}")
print(f"    Evaluation Time:     {test_time:.1f}s")

print(f"\n{'='*60}")
print("EXPERIMENT COMPLETE")
print(f"{'='*60}")
