"""
Script huấn luyện ATLAS - Dựa trên flow ACDC (Optimized).
Sử dụng memmap + LRU cache.
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

# Thêm thư mục gốc vào path
sys.path.insert(0, os.path.abspath(os.path.join(os.path.dirname(__file__), '..')))

# --- THAY ĐỔI: Imports ---
from src.models.unet import RobustMedVFL_UNet
from src.models.epure import ePURE
from src.modules.losses import CombinedLoss
from src.data_utils.atlas_dataset_optimized import (
    ATLASDataset25DOptimized,
    get_atlas_volume_ids,
    split_atlas_by_patient
)
from src.utils.helpers import calculate_ultimate_common_b1_map, adaptive_quantum_noise_injection
from src import config

# =============================================================================
# CẤU HÌNH (Tùy chỉnh cho ATLAS)
# =============================================================================

NUM_EPOCHS = config.NUM_EPOCHS
# --- THAY ĐỔI: 2 Lớp (0: Nền, 1: Lesion) ---
NUM_CLASSES = 2
LEARNING_RATE = 1e-3 # Giống ACDC
DEVICE = torch.device('cuda' if torch.cuda.is_available() else 'cpu')
IMG_SIZE = config.IMG_SIZE
BATCH_SIZE = config.BATCH_SIZE
NUM_SLICES = config.NUM_SLICES
EARLY_STOP_PATIENCE = config.EARLY_STOP_PATIENCE

# --- THAY ĐỔI: Class map và đường dẫn lưu model ---
ATLAS_CLASS_MAP = {0: 'BG', 1: 'Lesion'}
MODEL_SAVE_PATH = "best_model_atlas_optimized.pth"

print(f"Device: {DEVICE}")
print(f"Configuration: {NUM_EPOCHS} epochs, batch size {BATCH_SIZE}, {NUM_SLICES} slices (2.5D)")
print(f"Model save path: {MODEL_SAVE_PATH}")


# =============================================================================
# AUGMENTATION (Giữ nguyên)
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
# HÀM ĐÁNH GIÁ (Copy từ ACDC và sửa lại)
# =============================================================================

def evaluate_metrics(model, dataloader, device, num_classes=2):
    """
    Hàm đánh giá đã được sửa cho ATLAS (2 lớp).
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
# TẢI DỮ LIỆU (Tối ưu hóa)
# =============================================================================

print("ĐANG TẢI DỮ LIỆU ATLAS (TỐI ƯU HÓA - NPY)")

# --- THAY ĐỔI: Đường dẫn đến NPY ---
train_npy_dir = os.path.join(config.ATLAS_PREPROCESSED_DIR, 'train')
test_npy_dir = os.path.join(config.ATLAS_PREPROCESSED_DIR, 'test')

# Tải ID và tách tập train/val
print(f"Tải volume IDs từ: {train_npy_dir}")
all_train_volume_ids = get_atlas_volume_ids(train_npy_dir)
train_volume_ids, val_volume_ids = split_atlas_by_patient(
    all_train_volume_ids, 
    val_ratio=0.2, 
    random_state=42
)

print(f"Tải test volume IDs từ: {test_npy_dir}")
test_volume_ids = get_atlas_volume_ids(test_npy_dir)
print(f"Tìm thấy {len(test_volume_ids)} test volumes.")


# =============================================================================
# EPURE AUGMENTATION (Giữ nguyên)
# =============================================================================

print("Khởi tạo ePURE Augmentation")
ePURE_augmenter = ePURE(in_channels=NUM_SLICES).to(DEVICE)
ePURE_augmenter.eval()

# --- THAY ĐỔI: Dùng ATLASDataset25DOptimized ---
train_dataset = ATLASDataset25DOptimized(
    npy_dir=train_npy_dir,
    volume_ids=train_volume_ids,
    num_input_slices=NUM_SLICES,
    transforms=train_transform,
    noise_injector_model=ePURE_augmenter,
    device=str(DEVICE),
    max_cache_size=15,
    use_memmap=True
)

val_dataset = ATLASDataset25DOptimized(
    npy_dir=train_npy_dir,
    volume_ids=val_volume_ids,
    num_input_slices=NUM_SLICES,
    transforms=val_test_transform,
    max_cache_size=10,
    use_memmap=True
)

test_dataset = ATLASDataset25DOptimized(
    npy_dir=test_npy_dir,
    volume_ids=test_volume_ids,
    num_input_slices=NUM_SLICES,
    transforms=val_test_transform,
    max_cache_size=8,
    use_memmap=True
)

# --- THAY ĐỔI: num_workers BẮT BUỘC BẰNG 0 ---
train_dataloader = DataLoader(train_dataset, batch_size=BATCH_SIZE, shuffle=True, num_workers=0, pin_memory=True)
val_dataloader = DataLoader(val_dataset, batch_size=BATCH_SIZE, shuffle=False, num_workers=0, pin_memory=True)
test_dataloader = DataLoader(test_dataset, batch_size=BATCH_SIZE, shuffle=False, num_workers=0, pin_memory=True)

print(f"Training slices: {len(train_dataset)}, Validation slices: {len(val_dataset)}, Test slices: {len(test_dataset)}")
print("CẢNH BÁO: num_workers=0 là bắt buộc do ePURE augmentation (GPU) chạy trong Dataset.")

# =============================================================================
# B1 MAP (Tải file đã tính)
# =============================================================================

print("Tải B1 MAP (pre-computed)")

# --- THAY ĐỔI: Tên file B1 map của ATLAS ---
dataset_name = "atlas_t1w"
b1_map_path = f"{dataset_name}_ultimate_common_b1_map.pth"

if os.path.exists(b1_map_path):
    print(f"Đang tải B1 map từ: {b1_map_path}")
    saved_data = torch.load(b1_map_path, map_location=DEVICE)
    common_b1_map = saved_data['common_b1_map'].to(DEVICE)
else:
    # Nếu chưa có, tạo nó
    print(f"Không tìm thấy B1 map, đang tạo file mới...")
    def convert_volumes_to_tensor_for_b1(npy_dir, volume_ids):
        all_slices = []
        # Lấy mẫu một phần dữ liệu để tính B1 map cho nhanh
        sample_volume_ids = np.random.choice(volume_ids, min(len(volume_ids), 50), replace=False)
        print(f"  Sử dụng {len(sample_volume_ids)} volumes để tính B1 map...")
        for vid in tqdm(sample_volume_ids, desc="  Tải slices cho B1 map"):
            vol_path = os.path.join(npy_dir, 'volumes', f'{vid}.npy')
            vol_data = np.load(vol_path, mmap_mode='r')
            for i in range(vol_data.shape[2]):
                # Chỉ lấy 1 kênh (kênh 0) để tính B1
                all_slices.append(torch.from_numpy(vol_data[:, :, i]).unsqueeze(0))
        all_images_tensor = torch.stack(all_slices, dim=0).float()
        return all_images_tensor
        
    print("  Đang tải dữ liệu thô để tính B1 map...")
    train_tensors = convert_volumes_to_tensor_for_b1(train_npy_dir, train_volume_ids)
    val_tensors = convert_volumes_to_tensor_for_b1(train_npy_dir, val_volume_ids)
    test_tensors = convert_volumes_to_tensor_for_b1(test_npy_dir, test_volume_ids)
    all_images_tensor = torch.cat([train_tensors, val_tensors, test_tensors], dim=0)
    
    print(f"  Tổng cộng {all_images_tensor.shape[0]} slices được dùng để tính B1 map.")
    common_b1_map = calculate_ultimate_common_b1_map(
        all_images=all_images_tensor,
        device=str(DEVICE),
        save_path=b1_map_path
    )


# =============================================================================
# MODEL, LOSS, OPTIMIZER (Thay đổi)
# =============================================================================

print("Khởi tạo Model, Loss, Optimizer...")

# --- THAY ĐỔI: n_channels=5 (T1w 2.5D), num_classes=2 ---
model = RobustMedVFL_UNet(n_channels=NUM_SLICES, n_classes=NUM_CLASSES, deep_supervision=True).to(DEVICE)

total_params = sum(p.numel() for p in model.parameters() if p.requires_grad)
print(f"Model total parameters: {total_params:,}")

# --- THAY ĐỔI: Cấu hình loss cho ATLAS (TẮT Physics, TẮT Anatomical) ---
# Chúng ta sẽ tự định nghĩa loss trong vòng lặp train để tránh sửa file src/modules/losses.py
criterion_fl = FocalLoss(gamma=2.0).to(DEVICE)
criterion_ftl = FocalTverskyLoss(num_classes=NUM_CLASSES, alpha=0.2, beta=0.8, gamma=4.0/3.0).to(DEVICE)

print("ĐÃ TẮT PHYSICS LOSS VÀ ANATOMICAL LOSS.")
print("Sử dụng 50% FocalLoss + 50% FocalTverskyLoss (Fixed).")


optimizer = torch.optim.AdamW(
    model.parameters(), # Chỉ train tham số model
    lr=LEARNING_RATE
)
scheduler = torch.optim.lr_scheduler.ReduceLROnPlateau(
    optimizer, mode='max', factor=0.5, patience=10
)

print("Tất cả thành phần đã được khởi tạo!")


# =============================================================================
# VÒNG LẶP HUẤN LUYỆN (Thay đổi)
# =============================================================================

print("BẮT ĐẦU HUẤN LUYỆN")

best_val_metric = 0.0
epochs_no_improve = 0

for epoch in range(NUM_EPOCHS):
    print(f"\n{'='*60}")
    print(f"EPOCH {epoch + 1}/{NUM_EPOCHS}")
    print(f"{'='*60}")
    
    model.train()
    epoch_train_loss = 0.0
    train_start_time = time.time()
    train_pbar = tqdm(train_dataloader, desc=f"Training Epoch {epoch+1}", leave=False, ncols=100)
    
    for batch_idx, (images, targets) in enumerate(train_pbar):
        images, targets = images.to(DEVICE), targets.to(DEVICE)
        optimizer.zero_grad()
        
        logits_list, all_eps_sigma_tuples = model(images)

        total_loss = torch.tensor(0.0, device=DEVICE, requires_grad=True)
        for logits in logits_list:
            if logits.shape[2:] != targets.shape[1:]:
                resized_targets = F.interpolate(
                    targets.unsqueeze(1).float(), size=logits.shape[2:],
                    mode='nearest'
                ).squeeze(1).long()
            else:
                resized_targets = targets
            
            # --- THAY ĐỔI: Tính loss thủ công (50/50) ---
            l_fl = criterion_fl(logits, resized_targets)
            l_ftl = criterion_ftl(logits, resized_targets)
            loss_component = (0.5 * l_fl) + (0.5 * l_ftl)
            
            total_loss = total_loss + loss_component
        
        loss = total_loss / len(logits_list)
        loss.backward()
        optimizer.step()
        epoch_train_loss += loss.item()
        train_pbar.set_postfix({'loss': f'{loss.item():.4f}'})
    
    avg_train_loss = epoch_train_loss / len(train_dataloader)
    train_time = time.time() - train_start_time
    print(f"Training Loss: {avg_train_loss:.4f} (Time: {train_time:.1f}s)")

    # --- Validation Phase (Thay đổi) ---
    if val_dataloader and len(val_dataset) > 0:
        print("\nĐang đánh giá trên tập validation...")
        val_metrics = evaluate_metrics(model, val_dataloader, DEVICE, NUM_CLASSES)
        
        val_accuracy = val_metrics['accuracy']
        all_dice = val_metrics['dice_scores']
        all_iou = val_metrics['iou']
        
        # --- THAY ĐỔI: Metric chính là Dice Lớp 1 (Lesion) ---
        avg_fg_dice = all_dice[1] # Dice cho lớp 1 (Lesion)
        
        current_lr = optimizer.param_groups[0]['lr']
        
        print(f"   --- Per-Class Metrics ---")
        print(f"=> {ATLAS_CLASS_MAP[0]:<15}: Dice: {all_dice[0]:.4f}, IoU: {all_iou[0]:.4f}")
        print(f"=> {ATLAS_CLASS_MAP[1]:<15}: Dice: {all_dice[1]:.4f}, IoU: {all_iou[1]:.4f}")
        
        print(f"   --- Summary Metrics ---")
        print(f"=> Metric chính (Lesion Dice): {avg_fg_dice:.4f}")
        print(f"=> Overall Accuracy: {val_accuracy:.4f} | Current LR: {current_lr:.6f}")
        
        scheduler.step(avg_fg_dice)
        
        if avg_fg_dice > best_val_metric:
            best_val_metric = avg_fg_dice
            torch.save(model.state_dict(), MODEL_SAVE_PATH)
            print(f"\n  ✓ Model mới đã được lưu! Avg Lesion Dice: {best_val_metric:.4f}")
            epochs_no_improve = 0
        else:
            epochs_no_improve += 1
            print(f"  Không cải thiện trong {epochs_no_improve} epoch(s)")
    
    if epochs_no_improve >= EARLY_STOP_PATIENCE:
        print(f"\nDừng sớm sau {EARLY_STOP_PATIENCE} epochs không cải thiện.")
        break

print("HUẤN LUYỆN HOÀN TẤT")
print(f"Best validation Dice score (Lesion): {best_val_metric:.4f}")
print(f"Model đã lưu tại: {MODEL_SAVE_PATH}")


# =============================================================================
# ĐÁNH GIÁ TRÊN TẬP TEST
# =============================================================================

print("\nĐANG ĐÁNH GIÁ TRÊN TẬP TEST")
print("Tải model tốt nhất...")
try:
    model.load_state_dict(torch.load(MODEL_SAVE_PATH))
    model.eval()
except FileNotFoundError:
    print(f"LỖI: Không tìm thấy file model '{MODEL_SAVE_PATH}'. Bỏ qua bước test.")
    sys.exit(1)
except Exception as e:
    print(f"LỖI khi tải model: {e}. Bỏ qua bước test.")
    sys.exit(1)


print("Bắt đầu đánh giá tập test...")
test_metrics = evaluate_metrics(model, test_dataloader, DEVICE, NUM_CLASSES)

print(f"\n{'='*60}")
print("KẾT QUẢ TRÊN TẬP TEST")
print(f"{'='*60}")
test_accuracy = test_metrics['accuracy']
test_all_dice = test_metrics['dice_scores']
test_all_iou = test_metrics['iou']
test_fg_dice = test_all_dice[1] # Dice cho lớp 1 (Lesion)

print(f"   --- Per-Class Metrics (Test) ---")
print(f"=> {ATLAS_CLASS_MAP[0]:<15}: Dice: {test_all_dice[0]:.4f}, IoU: {test_all_iou[0]:.4f}")
print(f"=> {ATLAS_CLASS_MAP[1]:<15}: Dice: {test_all_dice[1]:.4f}, IoU: {test_all_iou[1]:.4f}")
print(f"   --- Summary Metrics (Test) ---")
print(f"=> Metric chính (Lesion Dice): {test_fg_dice:.4f}")
print(f"=> Overall Accuracy: {test_accuracy:.4f}")
print(f"{'='*60}")
print("THÍ NGHIỆM HOÀN TẤT")
print(f"{'='*60}")