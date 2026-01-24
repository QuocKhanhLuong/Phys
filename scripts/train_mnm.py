import torch
import torch.nn as nn
import torch.optim as optim
import torch.nn.functional as F
import numpy as np
import time
import os
import sys
from torch.utils.data import DataLoader
import albumentations as A
from albumentations.pytorch import ToTensorV2
from itertools import chain
import cv2
from tqdm import tqdm
from monai.metrics import compute_hausdorff_distance
import warnings

warnings.filterwarnings("ignore")

sys.path.insert(0, os.path.abspath(os.path.join(os.path.dirname(__file__), '..')))

from src.models.unet import PIE_UNet
from src.models.epure import ePURE
from src.modules.losses import CombinedLoss
from src.data_utils.mnm_dataset_optimized import (
    MnMDataset25DOptimized,
    get_mnm_volume_ids
)
from src.utils.helpers import calculate_ultimate_common_b1_map

NUM_EPOCHS = 250
NUM_CLASSES = 4
LEARNING_RATE = 1e-3
DEVICE = torch.device('cuda' if torch.cuda.is_available() else 'cpu')
IMG_SIZE = 224
BATCH_SIZE = 24
NUM_SLICES = 5
EARLY_STOP_PATIENCE = 30

RESULTS_DIR = "mnm_norm_results"
os.makedirs(RESULTS_DIR, exist_ok=True)

MNM_CLASS_MAP = {0: 'BG', 1: 'LV', 2: 'MYO', 3: 'RV'}

print(f"Device: {DEVICE}")
print(f"Configuration: {NUM_EPOCHS} epochs, batch size {BATCH_SIZE}, {NUM_SLICES} slices (2.5D)")
print(f"Experiment: MnM with Physics Loss (same as train_acdc)")
print(f"Results: {RESULTS_DIR}/")

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

def evaluate_metrics(model, dataloader, device, num_classes=4):
    model.eval()
    tp = [0] * num_classes
    fp = [0] * num_classes
    fn = [0] * num_classes
    dice_s = [0.0] * num_classes
    iou_s = [0.0] * num_classes
    hd95_s = [0.0] * num_classes
    hd95_counts = [0] * num_classes
    
    batches = 0
    total_correct_pixels = 0
    total_pixels = 0

    with torch.no_grad():
        eval_pbar = tqdm(dataloader, desc="Evaluating", leave=False, ncols=80)
        for imgs, tgts in eval_pbar:
            imgs, tgts = imgs.to(device), tgts.to(device)
            if imgs.size(0) == 0: continue
            
            logits_list, _ = model(imgs)
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

            # --- HD95 Calculation (CPU to avoid VRAM leak) ---
            # Move to CPU first
            preds_cpu = preds.detach().cpu()
            tgts_cpu = tgts.detach().cpu()

            preds_oh = F.one_hot(preds_cpu, num_classes=num_classes).permute(0, 3, 1, 2).float()
            tgts_oh = F.one_hot(tgts_cpu, num_classes=num_classes).permute(0, 3, 1, 2).float()
            
            try:
                hd95_batch = compute_hausdorff_distance(
                    y_pred=preds_oh, 
                    y=tgts_oh, 
                    include_background=True, 
                    percentile=95.0
                )
                
                for c in range(num_classes):
                    valid_vals = hd95_batch[:, c]
                    mask = ~torch.isnan(valid_vals) & ~torch.isinf(valid_vals)
                    if mask.any():
                        hd95_s[c] += valid_vals[mask].sum().item()
                        hd95_counts[c] += mask.sum().item()
            except Exception:
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

print("Loading M&M data")

PREPROCESSED_ROOT = '/home/linhdang/workspace/minhbao_workspace/Phys/preprocessed_data/mnm'
train_npy_dir = os.path.join(PREPROCESSED_ROOT, 'training')
val_npy_dir = os.path.join(PREPROCESSED_ROOT, 'validation')

if not os.path.exists(train_npy_dir):
    print(f"Error: Directory not found: {train_npy_dir}")
    print(f"Run preprocessing first:")
    print(f"  python scripts/preprocess_mnm.py --input data/MnM/Training/Labeled --output {train_npy_dir}")
    sys.exit(1)

if not os.path.exists(val_npy_dir):
    print(f"Error: Directory not found: {val_npy_dir}")
    print(f"Run preprocessing first:")
    print(f"  python scripts/preprocess_mnm.py --input data/MnM/M&M/Validation --output {val_npy_dir}")
    sys.exit(1)

print(f"Loading volume IDs from: {train_npy_dir}")
train_ids = get_mnm_volume_ids(train_npy_dir)

print(f"Loading volume IDs from: {val_npy_dir}")
val_ids = get_mnm_volume_ids(val_npy_dir)

print(f"Split: {len(train_ids)} train volumes, {len(val_ids)} val volumes")

print("Initializing ePURE")
ePURE_augmenter = ePURE(in_channels=NUM_SLICES).to(DEVICE)
ePURE_augmenter.eval()

print("Creating datasets")
train_dataset = MnMDataset25DOptimized(
    npy_dir=train_npy_dir,
    volume_ids=train_ids,
    num_input_slices=NUM_SLICES,
    transforms=train_transform,
    noise_injector_model=ePURE_augmenter,
    device=str(DEVICE),
    max_cache_size=15,
    use_memmap=True
)

val_dataset = MnMDataset25DOptimized(
    npy_dir=val_npy_dir,
    volume_ids=val_ids,
    num_input_slices=NUM_SLICES,
    transforms=val_test_transform,
    max_cache_size=10,
    use_memmap=True
)

train_loader = DataLoader(train_dataset, batch_size=BATCH_SIZE, shuffle=True, num_workers=0, pin_memory=True)
val_loader = DataLoader(val_dataset, batch_size=BATCH_SIZE, shuffle=False, num_workers=0, pin_memory=True)

print(f"Training slices: {len(train_dataset)}, Validation slices: {len(val_dataset)}")

print("Loading B1 map")
dataset_name = "mnm_norm_cardiac"
b1_map_path = f"{dataset_name}_ultimate_common_b1_map.pth"

def convert_npy_to_tensor_for_b1(npy_dir, volume_ids):
    volumes_dir = os.path.join(npy_dir, 'volumes')
    all_slices = []
    
    for vid in volume_ids:
        vol_path = os.path.join(volumes_dir, f'{vid}.npy')
        vol = np.load(vol_path)
        
        for i in range(vol.shape[2]):
            all_slices.append(torch.from_numpy(vol[:, :, i]).unsqueeze(0))
    
    return torch.stack(all_slices, dim=0).float()

print("Converting volumes to tensor for B1 map")
train_tensor = convert_npy_to_tensor_for_b1(train_npy_dir, train_ids)
val_tensor = convert_npy_to_tensor_for_b1(val_npy_dir, val_ids)
train_val_tensor = torch.cat([train_tensor, val_tensor], dim=0)

print(f"Total slices for B1 map: {train_val_tensor.shape[0]}")

common_b1_map = calculate_ultimate_common_b1_map(
    all_images=train_val_tensor,
    device=str(DEVICE),
    save_path=f"b1_maps/{dataset_name}_ultimate_common_b1_map.pth"
)

print("Initializing model")
model = PIE_UNet(n_channels=NUM_SLICES, n_classes=NUM_CLASSES, deep_supervision=True).to(DEVICE)

total_params = sum(p.numel() for p in model.parameters() if p.requires_grad)
print(f"Model parameters: {total_params:,}")

criterion = CombinedLoss(
    num_classes=NUM_CLASSES,
    initial_loss_weights=[0.4, 0.4, 0.2],  
    class_indices_for_rules=None  
).to(DEVICE)

optimizer = torch.optim.AdamW(
    chain(model.parameters(), criterion.parameters()),
    lr=LEARNING_RATE
)

scheduler = torch.optim.lr_scheduler.ReduceLROnPlateau(
    optimizer, mode='max', factor=0.5, patience=10
)

best_model_path_dice = "weights/best_model_mnm_dice.pth"
best_model_path_hd95 = "weights/best_model_mnm_hd95.pth"
best_model_path_overall = "weights/best_model_mnm_overall.pth"

print("Model initialized")

print("Starting training")

best_val_dice = 0.0
best_val_hd95 = float('inf')
best_val_overall = float('-inf')
epochs_no_improve = 0
history = {'train_loss': [], 'val_dice': [], 'val_acc': []}

for epoch in range(NUM_EPOCHS):
    print(f"\n{'='*60}")
    print(f"EPOCH {epoch + 1}/{NUM_EPOCHS}")
    print(f"{'='*60}")
    
    model.train()
    epoch_train_loss = 0.0
    train_start = time.time()
    
    train_pbar = tqdm(train_loader, desc=f"Training Epoch {epoch+1}/{NUM_EPOCHS}", leave=False, ncols=100)
    for images, targets in train_pbar:
        images, targets = images.to(DEVICE), targets.to(DEVICE)
        optimizer.zero_grad()
        
        logits_list, all_eps_sigma_tuples = model(images)
        
        total_loss = torch.tensor(0.0, device=DEVICE, requires_grad=True)
        for logits, eps_sigma_tuple in zip(logits_list, all_eps_sigma_tuples):
            if logits.shape[2:] != targets.shape[1:]:
                resized_targets = F.interpolate(
                    targets.unsqueeze(1).float(), size=logits.shape[2:],
                    mode='nearest'
                ).squeeze(1).long()
            else:
                resized_targets = targets
            
            if common_b1_map.shape[-2:] != logits.shape[-2:]:
                b1_resized = F.interpolate(
                    common_b1_map.unsqueeze(0).unsqueeze(0),
                    size=logits.shape[-2:],
                    mode='bilinear',
                    align_corners=False
                ).squeeze(0).squeeze(0)
            else:
                b1_resized = common_b1_map
            
            loss_component = criterion(
                logits=logits,
                targets=resized_targets,
                b1=b1_resized,
                all_es=[eps_sigma_tuple]
            )
            total_loss = total_loss + loss_component
        
        loss = total_loss / len(logits_list)
        loss.backward()
        torch.nn.utils.clip_grad_norm_(model.parameters(), max_norm=1.0)
        optimizer.step()
        epoch_train_loss += loss.item()
        train_pbar.set_postfix({'loss': f'{loss.item():.4f}'})
    
    avg_train_loss = epoch_train_loss / len(train_loader)
    train_time = time.time() - train_start
    
    current_weights = criterion.get_current_loss_weights()
    print(f"Training Loss: {avg_train_loss:.4f} (Time: {train_time:.1f}s)")
    print(f"Loss Weights: FL={current_weights['weight_FocalLoss']:.3f}, "
          f"FTL={current_weights['weight_FocalTverskyLoss']:.3f}, "
          f"Physics={current_weights['weight_Physics']:.3f}")
    
    print("Evaluating on validation set")
    val_metrics = evaluate_metrics(model, val_loader, DEVICE, NUM_CLASSES)
    
    # Explicitly empty cache after validation
    torch.cuda.empty_cache()
    
    val_accuracy = val_metrics['accuracy']
    all_dice = val_metrics['dice_scores']
    all_iou = val_metrics['iou']
    all_hd95 = val_metrics['hd95']
    all_precision = val_metrics['precision']
    all_recall = val_metrics['recall']
    all_f1 = val_metrics['f1_score']
    
    avg_fg_dice = np.mean(all_dice[1:])
    avg_fg_iou = np.mean(all_iou[1:])
    
    fg_hd95_vals = [h for h in all_hd95[1:] if not np.isnan(h)]
    if len(fg_hd95_vals) > 0:
        avg_fg_hd95 = np.mean(fg_hd95_vals)
    else:
        avg_fg_hd95 = float('inf')
        
    avg_fg_precision = np.mean(all_precision[1:])
    avg_fg_recall = np.mean(all_recall[1:])
    avg_fg_f1 = np.mean(all_f1[1:])
    
    # Calculate Overall Score
    safe_hd95 = avg_fg_hd95 if avg_fg_hd95 != float('inf') else 100.0
    overall_score = avg_fg_dice + (1.0 / (safe_hd95 + 1.0))

    current_lr = optimizer.param_groups[0]['lr']
    
    print(f"   Per-Class Metrics")
    for c_idx in range(NUM_CLASSES):
        class_name = MNM_CLASS_MAP.get(c_idx, f"Class {c_idx}")
        hd_str = f"{all_hd95[c_idx]:.4f}" if not np.isnan(all_hd95[c_idx]) else "NaN"
        print(f"=> {class_name:<15}: Dice: {all_dice[c_idx]:.4f}, IoU: {all_iou[c_idx]:.4f}, HD95: {hd_str}, "
              f"Precision: {all_precision[c_idx]:.4f}, Recall: {all_recall[c_idx]:.4f}, F1: {all_f1[c_idx]:.4f}")
    
    print(f"   Summary Metrics")
    print(f"=> Avg Foreground: Dice: {avg_fg_dice:.4f}, IoU: {avg_fg_iou:.4f}, HD95: {avg_fg_hd95:.4f}, "
          f"Precision: {avg_fg_precision:.4f}, Recall: {avg_fg_recall:.4f}, F1: {avg_fg_f1:.4f}")
    print(f"=> Overall Score: {overall_score:.4f}")
    print(f"=> Overall Accuracy: {val_accuracy:.4f} | Learning Rate: {current_lr:.6f}")
    
    history['train_loss'].append(avg_train_loss)
    history['val_dice'].append(avg_fg_dice)
    history['val_acc'].append(val_accuracy)
    
    scheduler.step(avg_fg_dice)
    
    # Save Best Dice
    if avg_fg_dice > best_val_dice:
        best_val_dice = avg_fg_dice
        torch.save(model.state_dict(), best_model_path_dice)
        print(f"\nNew best model (DICE) saved! Avg Foreground Dice: {best_val_dice:.4f}")
        epochs_no_improve = 0
    else:
        epochs_no_improve += 1
        print(f"No improvement for {epochs_no_improve} epoch(s)")

    # Save Best HD95
    if avg_fg_hd95 < best_val_hd95:
        best_val_hd95 = avg_fg_hd95
        torch.save(model.state_dict(), best_model_path_hd95)
        print(f"New best model (HD95) saved! Avg Foreground HD95: {best_val_hd95:.4f}")

    # Save Best Overall
    if overall_score > best_val_overall:
        best_val_overall = overall_score
        torch.save(model.state_dict(), best_model_path_overall)
        print(f"New best model (OVERALL) saved! Score: {best_val_overall:.4f}")
    
    if epochs_no_improve >= EARLY_STOP_PATIENCE:
        print(f"Early stopping triggered after {EARLY_STOP_PATIENCE} epochs with no improvement")
        break

print("Training completed")
print(f"Best validation Dice score: {best_val_dice:.4f}")
print(f"Best validation HD95 score: {best_val_hd95:.4f}")
print(f"Best validation Overall score: {best_val_overall:.4f}")
print(f"Model saved as: {best_model_path_dice}")

print(f"\n{'='*60}")
print("Experiment complete")
print(f"{'='*60}")

