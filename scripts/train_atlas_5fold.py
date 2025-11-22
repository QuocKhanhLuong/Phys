"""
Script huấn luyện ATLAS với 5-Fold Cross-Validation.
Chia 655 bệnh nhân thành Holdout 80-20, sau đó 5-fold CV trên 524 train set.
"""

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
from sklearn.model_selection import KFold, train_test_split
import json

# Thêm thư mục gốc vào path
sys.path.insert(0, os.path.abspath(os.path.join(os.path.dirname(__file__), '..')))

from src.models.unet import RobustMedVFL_UNet 
from src.models.epure import ePURE
from src.modules.losses import CombinedLoss, FocalLoss, FocalTverskyLoss
from src.data_utils.atlas import (
    ATLASDataset25DOptimized,
    get_atlas_volume_ids,
)
from src.utils.helpers import calculate_ultimate_common_b1_map
from src import config

# =============================================================================
# CẤU HÌNH
# =============================================================================

NUM_EPOCHS = config.NUM_EPOCHS
NUM_CLASSES = 2  # Background, Lesion
LEARNING_RATE = 1e-3
DEVICE = torch.device('cuda' if torch.cuda.is_available() else 'cpu')
IMG_SIZE = config.IMG_SIZE
BATCH_SIZE = config.BATCH_SIZE
NUM_SLICES = config.NUM_SLICES 
EARLY_STOP_PATIENCE = config.EARLY_STOP_PATIENCE

# 5-Fold CV config
N_FOLDS = 5
RESULTS_DIR = "atlas_5fold_results"
os.makedirs(RESULTS_DIR, exist_ok=True)

ATLAS_CLASS_MAP = {0: 'BG', 1: 'Lesion'}

print(f"Device: {DEVICE}")
print(f"Configuration: {NUM_EPOCHS} epochs, batch size {BATCH_SIZE}, {NUM_SLICES} slices (2.5D)")
print(f"5-Fold Cross-Validation - Results: {RESULTS_DIR}/")

# =============================================================================
# AUGMENTATION
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
# HÀM ĐÁNH GIÁ
# =============================================================================

def evaluate_metrics(model, dataloader, device, num_classes=2):
    """Hàm đánh giá cho ATLAS (2 lớp)."""
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

    metrics = {'accuracy': 0.0, 'dice_scores': [], 'iou': []}
    if batches > 0:
        if total_pixels > 0:
            metrics['accuracy'] = total_correct_pixels / total_pixels
        for c in range(num_classes):
            metrics['dice_scores'].append(dice_s[c] / batches)
            metrics['iou'].append(iou_s[c] / batches)
    else:
        metrics['dice_scores'] = [0.0] * num_classes
        metrics['iou'] = [0.0] * num_classes
            
    return metrics

# =============================================================================
# TẢI DỮ LIỆU VÀ CHIA HOLDOUT + 5-FOLD
# =============================================================================

print("\nĐANG TẢI DỮ LIỆU ATLAS")

train_npy_dir = os.path.join(config.ATLAS_PREPROCESSED_DIR, 'train')

# Tải tất cả volume IDs (655 bệnh nhân)
all_volume_ids = get_atlas_volume_ids(train_npy_dir)
print(f"Tổng số: {len(all_volume_ids)} bệnh nhân")

# BƯỚC 1: Chia Holdout 80-20
train_dev_ids, holdout_test_ids = train_test_split(
    all_volume_ids, 
    test_size=0.2, 
    random_state=42
)

print(f"\n📊 HOLDOUT SPLIT:")
print(f"  Train+Dev: {len(train_dev_ids)} BN (dùng cho 5-fold CV)")
print(f"  Holdout Test: {len(holdout_test_ids)} BN")

# BƯỚC 2: Setup 5-Fold CV
kfold = KFold(n_splits=N_FOLDS, shuffle=True, random_state=42)

# =============================================================================
# KHỞI TẠO ePURE & B1 MAP (Dùng chung cho tất cả folds)
# =============================================================================

print("\nKhởi tạo ePURE Augmentation")
ePURE_augmenter = ePURE(in_channels=NUM_SLICES).to(DEVICE)
ePURE_augmenter.eval()

# Tạo holdout test dataset
holdout_test_dataset = ATLASDataset25DOptimized(
    npy_dir=train_npy_dir,
    volume_ids=holdout_test_ids,
    num_input_slices=NUM_SLICES,
    transforms=val_test_transform,
    max_cache_size=10,
    use_memmap=True
)
holdout_test_dataloader = DataLoader(
    holdout_test_dataset, 
    batch_size=BATCH_SIZE, 
    shuffle=False, 
    num_workers=0, 
    pin_memory=True
)
print(f"✓ Holdout test set: {len(holdout_test_dataset)} slices")

# Tải hoặc tạo B1 MAP
print("\nTải B1 MAP...")
dataset_name = "atlas_t1w"
b1_map_path = f"{dataset_name}_ultimate_common_b1_map.pth"

if os.path.exists(b1_map_path):
    print(f"Đang tải B1 map từ: {b1_map_path}")
    saved_data = torch.load(b1_map_path, map_location=DEVICE)
    common_b1_map = saved_data['common_b1_map'].to(DEVICE)
else:
    print(f"Không tìm thấy B1 map, đang tạo mới...")
    def convert_volumes_to_tensor_for_b1(npy_dir, volume_ids):
        all_slices = []
        sample_ids = np.random.choice(volume_ids, min(len(volume_ids), 50), replace=False)
        for vid in tqdm(sample_ids, desc="Tải slices cho B1 map"):
            vol_path = os.path.join(npy_dir, 'volumes', f'{vid}.npy')
            vol_data = np.load(vol_path, mmap_mode='r')
            for i in range(vol_data.shape[2]):
                all_slices.append(torch.from_numpy(vol_data[:, :, i].copy()).unsqueeze(0))
        return torch.stack(all_slices, dim=0).float()
    
    all_tensors = convert_volumes_to_tensor_for_b1(train_npy_dir, all_volume_ids)
    print(f"Tổng cộng {all_tensors.shape[0]} slices")
    common_b1_map = calculate_ultimate_common_b1_map(
        all_images=all_tensors,
        device=str(DEVICE),
        save_path=b1_map_path
    )

# =============================================================================
# 5-FOLD CROSS-VALIDATION LOOP
# =============================================================================

fold_results = []

print(f"\n{'='*60}")
print(f"BẮT ĐẦU {N_FOLDS}-FOLD CROSS-VALIDATION")
print(f"{'='*60}\n")

train_dev_ids_array = np.array(train_dev_ids)

for fold_idx, (train_indices, val_indices) in enumerate(kfold.split(train_dev_ids_array)):
    print(f"\n{'#'*60}")
    print(f"### FOLD {fold_idx + 1}/{N_FOLDS}")
    print(f"{'#'*60}\n")
    
    # Lấy IDs cho fold này
    fold_train_ids = train_dev_ids_array[train_indices].tolist()
    fold_val_ids = train_dev_ids_array[val_indices].tolist()
    
    print(f"Train: {len(fold_train_ids)} BN, Val: {len(fold_val_ids)} BN")
    
    # Tạo datasets
    fold_train_dataset = ATLASDataset25DOptimized(
        npy_dir=train_npy_dir,
        volume_ids=fold_train_ids,
        num_input_slices=NUM_SLICES,
        transforms=train_transform,
        noise_injector_model=ePURE_augmenter,
        device=str(DEVICE),
        max_cache_size=15,
        use_memmap=True
    )
    
    fold_val_dataset = ATLASDataset25DOptimized(
        npy_dir=train_npy_dir,
        volume_ids=fold_val_ids,
        num_input_slices=NUM_SLICES,
        transforms=val_test_transform,
        max_cache_size=10,
        use_memmap=True
    )
    
    fold_train_loader = DataLoader(fold_train_dataset, batch_size=BATCH_SIZE, shuffle=True, num_workers=0, pin_memory=True)
    fold_val_loader = DataLoader(fold_val_dataset, batch_size=BATCH_SIZE, shuffle=False, num_workers=0, pin_memory=True)
    
    print(f"✓ Train: {len(fold_train_dataset)} slices, Val: {len(fold_val_dataset)} slices")
    
    # Khởi tạo model cho fold này
    print("\nKhởi tạo Model, Loss, Optimizer...")
    model = RobustMedVFL_UNet(n_channels=NUM_SLICES, n_classes=NUM_CLASSES, deep_supervision=True).to(DEVICE)
    
    initial_weights = [0.4, 0.4, 0.2]
    criterion = CombinedLoss(
        num_classes=NUM_CLASSES,
        initial_loss_weights=initial_weights,
        class_indices_for_rules=None
    ).to(DEVICE)

    # ====== IMPORTANT: do not optimize the dynamic loss weighter parameters jointly with the model
    # The dynamic weighter can collapse to extreme weights quickly when optimized together with model
    # For stability we freeze its params and only train model parameters here.
    try:
        criterion.loss_weighter.params.requires_grad = False
    except Exception:
        pass

    optimizer = torch.optim.AdamW(
        list(model.parameters()), 
        lr=LEARNING_RATE
    )
    scheduler = torch.optim.lr_scheduler.ReduceLROnPlateau(
        optimizer, mode='max', factor=0.5, patience=10
    )
    
    fold_model_path = os.path.join(RESULTS_DIR, f"fold_{fold_idx+1}_best_model.pth")
    
    print("✓ Model initialized with CombinedLoss (FL + FTL + Physics)")
    
    # Training loop cho fold này
    print(f"\nBẮT ĐẦU TRAINING FOLD {fold_idx + 1}")
    
    best_val_dice = 0.0
    epochs_no_improve = 0
    fold_history = {'train_loss': [], 'val_dice': [], 'val_acc': []}
    
    for epoch in range(NUM_EPOCHS):
        print(f"\n{'='*50}")
        print(f"FOLD {fold_idx+1} - EPOCH {epoch + 1}/{NUM_EPOCHS}")
        print(f"{'='*50}")
        
        # Training
        model.train()
        epoch_train_loss = 0.0
        train_start = time.time()
        
        train_pbar = tqdm(fold_train_loader, desc=f"Training", leave=False, ncols=100)
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
                
                # Resize B1 map
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
        
        avg_train_loss = epoch_train_loss / len(fold_train_loader)
        train_time = time.time() - train_start
        
        current_weights = criterion.get_current_loss_weights()
        print(f"Train Loss: {avg_train_loss:.4f} ({train_time:.1f}s)")
        print(f"  Weights: FL={current_weights['weight_FocalLoss']:.3f}, "
              f"FTL={current_weights['weight_FocalTverskyLoss']:.3f}, "
              f"Phy={current_weights['weight_Physics']:.3f}")
        
        # Validation
        print("\nValidation...")
        val_metrics = evaluate_metrics(model, fold_val_loader, DEVICE, NUM_CLASSES)
        val_dice_lesion = val_metrics['dice_scores'][1]
        val_acc = val_metrics['accuracy']
        
        print(f"=> BG: Dice={val_metrics['dice_scores'][0]:.4f}, IoU={val_metrics['iou'][0]:.4f}")
        print(f"=> Lesion: Dice={val_dice_lesion:.4f}, IoU={val_metrics['iou'][1]:.4f}")
        print(f"=> Accuracy: {val_acc:.4f}")
        
        fold_history['train_loss'].append(avg_train_loss)
        fold_history['val_dice'].append(val_dice_lesion)
        fold_history['val_acc'].append(val_acc)
        
        scheduler.step(val_dice_lesion)
        
        # Save best model
        if val_dice_lesion > best_val_dice:
            best_val_dice = val_dice_lesion
            torch.save(model.state_dict(), fold_model_path)
            print(f"\n  ✓ Best model saved! Dice: {best_val_dice:.4f}")
            epochs_no_improve = 0
        else:
            epochs_no_improve += 1
            print(f"  No improvement for {epochs_no_improve} epoch(s)")
        
        # Early stopping
        if epochs_no_improve >= EARLY_STOP_PATIENCE:
            print(f"\nEarly stopping at epoch {epoch + 1}")
            break
    
    # Lưu kết quả fold
    fold_results.append({
        'fold': fold_idx + 1,
        'best_val_dice': best_val_dice,
        'train_ids': fold_train_ids,
        'val_ids': fold_val_ids,
        'history': fold_history
    })
    
    print(f"\nFOLD {fold_idx + 1} HOÀN THÀNH - Best Val Dice: {best_val_dice:.4f}")

# =============================================================================
# TỔNG HỢP KẾT QUẢ 5 FOLDS
# =============================================================================

print(f"\n{'='*60}")
print("TỔNG HỢP KẾT QUẢ 5-FOLD CV")
print(f"{'='*60}\n")

all_fold_dices = [r['best_val_dice'] for r in fold_results]
mean_dice = np.mean(all_fold_dices)
std_dice = np.std(all_fold_dices)

for r in fold_results:
    print(f"Fold {r['fold']}: Dice = {r['best_val_dice']:.4f}")

print(f"\n📊 SUMMARY:")
print(f"  Mean Dice: {mean_dice:.4f} ± {std_dice:.4f}")
print(f"  Best Fold: Fold {np.argmax(all_fold_dices) + 1} ({max(all_fold_dices):.4f})")
print(f"  Worst Fold: Fold {np.argmin(all_fold_dices) + 1} ({min(all_fold_dices):.4f})")

# Lưu results
results_path = os.path.join(RESULTS_DIR, "cv_results.json")
with open(results_path, 'w') as f:
    json.dump({
        'n_folds': N_FOLDS,
        'mean_dice': float(mean_dice),
        'std_dice': float(std_dice),
        'fold_results': [
            {
                'fold': r['fold'],
                'best_val_dice': float(r['best_val_dice']),
                'num_train': len(r['train_ids']),
                'num_val': len(r['val_ids'])
            }
            for r in fold_results
        ]
    }, f, indent=2)
print(f"\n✓ Results saved to: {results_path}")

# =============================================================================
# ĐÁNH GIÁ TRÊN HOLDOUT TEST SET
# =============================================================================

print(f"\n{'='*60}")
print("ĐÁNH GIÁ TRÊN HOLDOUT TEST SET")
print(f"{'='*60}\n")

# Tải model của fold tốt nhất
best_fold_idx = np.argmax(all_fold_dices)
best_fold_model_path = os.path.join(RESULTS_DIR, f"fold_{best_fold_idx+1}_best_model.pth")

print(f"Sử dụng model từ Fold {best_fold_idx+1} (best fold)")
model = RobustMedVFL_UNet(n_channels=NUM_SLICES, n_classes=NUM_CLASSES, deep_supervision=True).to(DEVICE)
model.load_state_dict(torch.load(best_fold_model_path))
model.eval()

print("Đang đánh giá trên holdout test set...")
test_metrics = evaluate_metrics(model, holdout_test_dataloader, DEVICE, NUM_CLASSES)

test_dice_lesion = test_metrics['dice_scores'][1]
test_acc = test_metrics['accuracy']

print(f"\n📊 HOLDOUT TEST RESULTS:")
print(f"=> BG: Dice={test_metrics['dice_scores'][0]:.4f}, IoU={test_metrics['iou'][0]:.4f}")
print(f"=> Lesion: Dice={test_dice_lesion:.4f}, IoU={test_metrics['iou'][1]:.4f}")
print(f"=> Accuracy: {test_acc:.4f}")

print(f"\n{'='*60}")
print("HOÀN TẤT 5-FOLD CROSS-VALIDATION")
print(f"{'='*60}")
print(f"CV Mean Dice: {mean_dice:.4f} ± {std_dice:.4f}")
print(f"Holdout Test Dice: {test_dice_lesion:.4f}")
print(f"{'='*60}")
