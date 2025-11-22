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

sys.path.insert(0, os.path.abspath(os.path.join(os.path.dirname(__file__), '..')))

from src.models.unet import RobustMedVFL_UNet 
from src.models.epure import ePURE
from src.modules.losses import CombinedLoss
from src.data_utils.mnm2_dataset_optimized import (
    MnM2Dataset25DOptimized,
    get_mnm2_volume_ids
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

RESULTS_DIR = "mnm2_results"
os.makedirs(RESULTS_DIR, exist_ok=True)

MNM2_CLASS_MAP = {0: 'BG', 1: 'LV', 2: 'MYO', 3: 'RV'}

print(f"Device: {DEVICE}")
print(f"Configuration: {NUM_EPOCHS} epochs, batch size {BATCH_SIZE}, {NUM_SLICES} slices (2.5D)")
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

print("Loading M&Ms2 data")

PREPROCESSED_ROOT = '/home/linhdang/workspace/minhbao_workspace/Phys/preprocessed_data/mnm2'
train_npy_dir = PREPROCESSED_ROOT

if not os.path.exists(train_npy_dir):
    print(f"Error: Directory not found: {train_npy_dir}")
    print(f"Run preprocessing first: python scripts/preprocess_mnm2.py --input data/MnM2 --output {train_npy_dir}")
    sys.exit(1)

print(f"Loading volume IDs from: {train_npy_dir}")
all_volume_ids = get_mnm2_volume_ids(train_npy_dir)

indices = list(range(len(all_volume_ids)))
trainval_indices, test_indices = train_test_split(indices, test_size=0.2, random_state=42)
train_indices, val_indices = train_test_split(trainval_indices, test_size=0.125, random_state=42)

train_ids = [all_volume_ids[i] for i in train_indices]
val_ids = [all_volume_ids[i] for i in val_indices]
test_ids = [all_volume_ids[i] for i in test_indices]

print(f"Split: {len(train_ids)} train volumes, {len(val_ids)} val volumes, {len(test_ids)} test volumes")

print("Initializing ePURE")
ePURE_augmenter = ePURE(in_channels=NUM_SLICES).to(DEVICE)
ePURE_augmenter.eval()

print("Creating datasets")
train_dataset = MnM2Dataset25DOptimized(
    npy_dir=train_npy_dir,
    volume_ids=train_ids,
    num_input_slices=NUM_SLICES,
    transforms=train_transform,
    noise_injector_model=ePURE_augmenter,
    device=str(DEVICE),
    max_cache_size=15,
    use_memmap=True
)

val_dataset = MnM2Dataset25DOptimized(
    npy_dir=train_npy_dir,
    volume_ids=val_ids,
    num_input_slices=NUM_SLICES,
    transforms=val_test_transform,
    max_cache_size=10,
    use_memmap=True
)

test_dataset = MnM2Dataset25DOptimized(
    npy_dir=train_npy_dir,
    volume_ids=test_ids,
    num_input_slices=NUM_SLICES,
    transforms=val_test_transform,
    max_cache_size=8,
    use_memmap=True
)

train_loader = DataLoader(train_dataset, batch_size=BATCH_SIZE, shuffle=True, num_workers=0, pin_memory=True)
val_loader = DataLoader(val_dataset, batch_size=BATCH_SIZE, shuffle=False, num_workers=0, pin_memory=True)
test_loader = DataLoader(test_dataset, batch_size=BATCH_SIZE, shuffle=False, num_workers=0, pin_memory=True)

print(f"Training slices: {len(train_dataset)}, Validation slices: {len(val_dataset)}, Test slices: {len(test_dataset)}")

print("Loading B1 map")
dataset_name = "mnm2_cardiac"
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
train_val_tensor = convert_npy_to_tensor_for_b1(train_npy_dir, train_ids + val_ids)

print(f"Total slices for B1 map: {train_val_tensor.shape[0]}")

common_b1_map = calculate_ultimate_common_b1_map(
    all_images=train_val_tensor,
    device=str(DEVICE),
    save_path=f"{dataset_name}_ultimate_common_b1_map.pth"
)

print("Initializing model")
model = RobustMedVFL_UNet(n_channels=NUM_SLICES, n_classes=NUM_CLASSES, deep_supervision=True).to(DEVICE)

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

best_model_path = os.path.join(RESULTS_DIR, "best_model_mnm2.pth")

print("Model initialized")

print("Starting training")

best_val_dice = 0.0
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
    
    val_accuracy = val_metrics['accuracy']
    all_dice = val_metrics['dice_scores']
    all_iou = val_metrics['iou']
    all_precision = val_metrics['precision']
    all_recall = val_metrics['recall']
    all_f1 = val_metrics['f1_score']
    
    avg_fg_dice = np.mean(all_dice[1:])
    avg_fg_iou = np.mean(all_iou[1:])
    avg_fg_precision = np.mean(all_precision[1:])
    avg_fg_recall = np.mean(all_recall[1:])
    avg_fg_f1 = np.mean(all_f1[1:])
    
    current_lr = optimizer.param_groups[0]['lr']
    
    print(f"   Per-Class Metrics")
    for c_idx in range(NUM_CLASSES):
        class_name = MNM2_CLASS_MAP.get(c_idx, f"Class {c_idx}")
        print(f"=> {class_name:<15}: Dice: {all_dice[c_idx]:.4f}, IoU: {all_iou[c_idx]:.4f}, "
              f"Precision: {all_precision[c_idx]:.4f}, Recall: {all_recall[c_idx]:.4f}, F1: {all_f1[c_idx]:.4f}")
    
    print(f"   Summary Metrics")
    print(f"=> Avg Foreground: Dice: {avg_fg_dice:.4f}, IoU: {avg_fg_iou:.4f}, "
          f"Precision: {avg_fg_precision:.4f}, Recall: {avg_fg_recall:.4f}, F1: {avg_fg_f1:.4f}")
    print(f"=> Overall Accuracy: {val_accuracy:.4f} | Learning Rate: {current_lr:.6f}")
    
    history['train_loss'].append(avg_train_loss)
    history['val_dice'].append(avg_fg_dice)
    history['val_acc'].append(val_accuracy)
    
    scheduler.step(avg_fg_dice)
    
    if avg_fg_dice > best_val_dice:
        best_val_dice = avg_fg_dice
        torch.save(model.state_dict(), best_model_path)
        print(f"\nNew best model saved! Avg Foreground Dice: {best_val_dice:.4f}")
        epochs_no_improve = 0
    else:
        epochs_no_improve += 1
        print(f"No improvement for {epochs_no_improve} epoch(s)")
    
    if epochs_no_improve >= EARLY_STOP_PATIENCE:
        print(f"Early stopping triggered after {EARLY_STOP_PATIENCE} epochs with no improvement")
        break

print("Training completed")
print(f"Best validation Dice score: {best_val_dice:.4f}")
print(f"Model saved as: {best_model_path}")

print("Evaluating on test set")

print("Loading best model")
model.load_state_dict(torch.load(best_model_path))
model.eval()

print("Running evaluation on test set")
test_metrics = evaluate_metrics(model, test_loader, DEVICE, NUM_CLASSES)

test_accuracy = test_metrics['accuracy']
test_dice = test_metrics['dice_scores']
test_iou = test_metrics['iou']
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
avg_fg_precision_test = np.mean(test_precision[1:])
avg_fg_recall_test = np.mean(test_recall[1:])
avg_fg_f1_test = np.mean(test_f1[1:])

print(f"\n{'='*60}")
print("Test set results")
print(f"{'='*60}")

print(f"\n  Test Results (Mean of ALL {NUM_CLASSES} Classes):")
print(f"    Accuracy: {test_accuracy:.4f}; Dice: {mean_dice_all:.4f}; IoU: {mean_iou_all:.4f}; "
      f"Precision: {mean_precision_all:.4f}; Recall: {mean_recall_all:.4f}; F1-score: {mean_f1_all:.4f}")

print(f"\n  Per-Class Metrics:")
for c_idx in range(NUM_CLASSES):
    class_name = MNM2_CLASS_MAP.get(c_idx, f"Class {c_idx}")
    print(f"    => {class_name:<20}: "
          f"Dice: {test_dice[c_idx]:.4f}, "
          f"IoU: {test_iou[c_idx]:.4f}, "
          f"Precision: {test_precision[c_idx]:.4f}, "
          f"Recall: {test_recall[c_idx]:.4f}, "
          f"F1: {test_f1[c_idx]:.4f}")

print(f"\n  Foreground Classes Summary:")
print(f"    Avg Foreground Dice:      {avg_fg_dice_test:.4f}")
print(f"    Avg Foreground IoU:       {avg_fg_iou_test:.4f}")
print(f"    Avg Foreground Precision: {avg_fg_precision_test:.4f}")
print(f"    Avg Foreground Recall:    {avg_fg_recall_test:.4f}")
print(f"    Avg Foreground F1:        {avg_fg_f1_test:.4f}")

print(f"\n{'='*60}")
print("Experiment complete")
print(f"{'='*60}")

results_file = os.path.join(RESULTS_DIR, "test_results.txt")
with open(results_file, 'w') as f:
    f.write("="*60 + "\n")
    f.write("M&Ms2 TEST SET RESULTS\n")
    f.write("="*60 + "\n\n")
    f.write(f"Model: {best_model_path}\n")
    f.write(f"Test samples: {len(test_dataset)} slices\n\n")
    
    f.write(f"Mean of ALL {NUM_CLASSES} Classes:\n")
    f.write(f"  Accuracy: {test_accuracy:.4f}\n")
    f.write(f"  Dice: {mean_dice_all:.4f}\n")
    f.write(f"  IoU: {mean_iou_all:.4f}\n")
    f.write(f"  Precision: {mean_precision_all:.4f}\n")
    f.write(f"  Recall: {mean_recall_all:.4f}\n")
    f.write(f"  F1-score: {mean_f1_all:.4f}\n\n")
    
    f.write("Per-Class Metrics:\n")
    for c_idx in range(NUM_CLASSES):
        class_name = MNM2_CLASS_MAP.get(c_idx, f"Class {c_idx}")
        f.write(f"  {class_name}:\n")
        f.write(f"    Dice: {test_dice[c_idx]:.4f}\n")
        f.write(f"    IoU: {test_iou[c_idx]:.4f}\n")
        f.write(f"    Precision: {test_precision[c_idx]:.4f}\n")
        f.write(f"    Recall: {test_recall[c_idx]:.4f}\n")
        f.write(f"    F1: {test_f1[c_idx]:.4f}\n\n")
    
    f.write("Foreground Classes Summary:\n")
    f.write(f"  Avg Dice: {avg_fg_dice_test:.4f}\n")
    f.write(f"  Avg IoU: {avg_fg_iou_test:.4f}\n")
    f.write(f"  Avg Precision: {avg_fg_precision_test:.4f}\n")
    f.write(f"  Avg Recall: {avg_fg_recall_test:.4f}\n")
    f.write(f"  Avg F1: {avg_fg_f1_test:.4f}\n\n")
    
    f.write(f"Best Validation Dice: {best_val_dice:.4f}\n")

print(f"Test results saved to: {results_file}")
