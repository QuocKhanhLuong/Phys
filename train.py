"""
Main training script for cardiac MRI segmentation with physics-informed learning.
Supports K-fold cross-validation, deep supervision, and early stopping.
"""

import os
import numpy as np
import torch
import torch.nn.functional as F
import torch.multiprocessing as mp
from torch.utils.data import DataLoader
from sklearn.model_selection import KFold
from itertools import chain
import albumentations as A
from albumentations.pytorch import ToTensorV2
import cv2

from models import RobustMedVFL_UNet, ePURE
from losses import CombinedLoss
from data_utils import ACDCDataset25D, load_acdc_volumes
from utils import calculate_ultimate_common_b1_map
from evaluate import evaluate_metrics, evaluate_metrics_with_tta


# =============================================================================
# --- Configuration ---
# =============================================================================

NUM_EPOCHS = 250
NUM_CLASSES = 4
LEARNING_RATE = 1e-3
DEVICE = torch.device('cuda' if torch.cuda.is_available() else 'cpu')
IMG_SIZE = 224
BATCH_SIZE = 24
NUM_SLICES = 5  # Số lát cắt cho 2.5D input
EARLY_STOP_PATIENCE = 30
N_SPLITS = 5  # Số fold cho cross-validation


# =============================================================================
# --- Helper Functions ---
# =============================================================================

def convert_volumes_to_tensor(volumes_list):
    """Chuyển đổi danh sách các volume 3D thành tensor để tính B1 map."""
    return torch.stack([
        torch.from_numpy(vol[:, :, i]).unsqueeze(0) 
        for vol in volumes_list 
        for i in range(vol.shape[2])
    ], dim=0).float()


def print_epoch_metrics(epoch, num_epochs, fold, train_loss, val_metrics, current_lr):
    """In các metrics theo từng epoch."""
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

    print(f"\n--- Fold {fold}, Epoch {epoch}/{num_epochs} ---")
    print(f"   Training Loss: {train_loss:.4f}")
    print("   --- Per-Class Metrics ---")
    
    class_map = {0: 'BG', 1: 'RV', 2: 'MYO', 3: 'LV'}
    for c_idx in range(NUM_CLASSES):
        class_name = class_map.get(c_idx, f"Class {c_idx}")
        print(f"   => {class_name:<15}: Dice: {all_dice[c_idx]:.4f}, IoU: {all_iou[c_idx]:.4f}, "
              f"Precision: {all_precision[c_idx]:.4f}, Recall: {all_recall[c_idx]:.4f}, F1: {all_f1[c_idx]:.4f}")

    print("   --- Summary Metrics ---")
    print(f"   => Avg Foreground: Dice: {avg_fg_dice:.4f}, IoU: {avg_fg_iou:.4f}, "
          f"Precision: {avg_fg_precision:.4f}, Recall: {avg_fg_recall:.4f}, F1: {avg_fg_f1:.4f}")
    print(f"   => Overall Accuracy: {val_accuracy:.4f} | Current Learning Rate: {current_lr:.6f}")
    
    return avg_fg_dice


# =============================================================================
# --- Training Function ---
# =============================================================================

def train_one_fold(fold, train_indices, val_indices, all_train_volumes, all_train_masks, 
                   common_b1_map, train_transform, val_transform):
    """
    Huấn luyện model cho một fold trong cross-validation.
    
    Returns:
        best_val_metric (float): Dice score tốt nhất trên validation set.
    """
    print(f"\n{'='*30} FOLD {fold + 1}/{N_SPLITS} {'='*30}")

    # Chuẩn bị dữ liệu
    X_train_vols = [all_train_volumes[i] for i in train_indices]
    y_train_vols = [all_train_masks[i] for i in train_indices]
    X_val_vols = [all_train_volumes[i] for i in val_indices]
    y_val_vols = [all_train_masks[i] for i in val_indices]
    
    # Tạo noise injector cho augmentation
    ePURE_augmenter = ePURE(in_channels=NUM_SLICES).to(DEVICE).eval()
    
    # Tạo datasets
    train_dataset = ACDCDataset25D(X_train_vols, y_train_vols, NUM_SLICES, train_transform, 
                                   ePURE_augmenter, DEVICE)
    val_dataset = ACDCDataset25D(X_val_vols, y_val_vols, NUM_SLICES, val_transform)
    
    # Tạo dataloaders
    train_dataloader = DataLoader(train_dataset, batch_size=BATCH_SIZE, shuffle=True, 
                                 num_workers=0, pin_memory=True)
    val_dataloader = DataLoader(val_dataset, batch_size=BATCH_SIZE, shuffle=False, 
                               num_workers=0, pin_memory=True)
    
    print(f"Fold {fold+1} - Số mẫu training: {len(train_dataset)}, Validation: {len(val_dataset)}")

    # Khởi tạo model và loss
    model = RobustMedVFL_UNet(n_channels=NUM_SLICES, n_classes=NUM_CLASSES).to(DEVICE)
    my_class_indices = {'RV': 1, 'MYO': 2, 'LV': 3}
    criterion = CombinedLoss(NUM_CLASSES, [0.4, 0.4, 0.2], my_class_indices).to(DEVICE)
    
    # Optimizer và scheduler
    optimizer = torch.optim.AdamW(chain(model.parameters(), criterion.parameters()), 
                                 lr=LEARNING_RATE)
    scheduler = torch.optim.lr_scheduler.ReduceLROnPlateau(optimizer, mode='max', 
                                                           factor=0.5, patience=10)
    
    # Training loop
    best_val_metric = 0.0
    epochs_no_improve = 0
    model_save_path = f"best_model_fold_{fold}.pth"

    for epoch in range(1, NUM_EPOCHS + 1):
        # Training phase
        model.train()
        epoch_train_loss = 0.0
        
        for images, targets in train_dataloader:
            images, targets = images.to(DEVICE), targets.to(DEVICE)
            optimizer.zero_grad()
            
            b1_map_for_loss = common_b1_map.expand(images.size(0), -1, -1, -1)
            logits_list, all_es_tuples = model(images)
            
            # Deep supervision: tính loss cho tất cả các output
            total_loss = sum(
                criterion(
                    F.interpolate(logits, size=targets.shape[1:], mode='bilinear', align_corners=False),
                    targets, b1_map_for_loss, all_es_tuples
                ) 
                for logits in logits_list
            )
            loss = total_loss / len(logits_list)
            
            loss.backward()
            optimizer.step()
            epoch_train_loss += loss.item()
        
        avg_train_loss = epoch_train_loss / len(train_dataloader)
        
        # Validation phase
        if val_dataloader.dataset and len(val_dataloader.dataset) > 0:
            val_metrics = evaluate_metrics(model, val_dataloader, DEVICE, NUM_CLASSES)
            avg_fg_dice = print_epoch_metrics(epoch, NUM_EPOCHS, fold+1, avg_train_loss, 
                                             val_metrics, optimizer.param_groups[0]['lr'])
            
            scheduler.step(avg_fg_dice)
            
            # Save best model
            if avg_fg_dice > best_val_metric:
                best_val_metric = avg_fg_dice
                torch.save(model.state_dict(), model_save_path)
                print(f"   >>> New best model for fold {fold+1} saved with Dice: {best_val_metric:.4f} <<<")
                epochs_no_improve = 0
            else:
                epochs_no_improve += 1
        
        # Early stopping
        if epochs_no_improve >= EARLY_STOP_PATIENCE:
            print(f"Early stopping for fold {fold+1} at epoch {epoch}.")
            break
    
    print(f"--->>> Best Validation Dice for Fold {fold+1}: {best_val_metric:.4f} <<<---")
    return best_val_metric


# =============================================================================
# --- Main Training Loop ---
# =============================================================================

def main():
    """Main training function with K-fold cross-validation."""
    mp.set_start_method('spawn', force=True)

    print(f"Thiết bị đang sử dụng: {DEVICE}")
    
    # --- Augmentation pipelines ---
    train_transform = A.Compose([
        A.Rotate(limit=20, p=0.7),
        A.HorizontalFlip(p=0.5),
        A.ElasticTransform(p=0.5, alpha=120, sigma=120 * 0.05),
        A.Affine(scale=(0.9, 1.1), translate_percent=(-0.0625, 0.0625), 
                rotate=(-15, 15), p=0.7, border_mode=cv2.BORDER_CONSTANT),
        A.RandomBrightnessContrast(p=0.5),
        ToTensorV2()
    ])
    val_test_transform = A.Compose([ToTensorV2()])

    # --- Nạp và Chuẩn bị Dữ liệu ---
    # Get project root directory (where train.py is located)
    project_root = os.path.dirname(os.path.abspath(__file__))
    base_dataset_root = os.path.join(project_root, 'database')
    train_data_path = os.path.join(base_dataset_root, 'training')
    test_data_path = os.path.join(base_dataset_root, 'testing')
    
    print(f"Nạp các volume training từ: {train_data_path}...")
    all_train_volumes, all_train_masks = load_acdc_volumes(train_data_path, 
                                                           target_size=(IMG_SIZE, IMG_SIZE))
    print(f"Đã nạp {len(all_train_volumes)} training volumes.")

    print(f"Nạp các volume testing từ: {test_data_path}...")
    all_test_volumes, all_test_masks = load_acdc_volumes(test_data_path, 
                                                         target_size=(IMG_SIZE, IMG_SIZE))
    print(f"Đã nạp {len(all_test_volumes)} testing volumes.")

    # Normalize volumes
    for vol_list in [all_train_volumes, all_test_volumes]:
        for i in range(len(vol_list)):
            max_val = np.max(vol_list[i])
            if max_val > 0:
                vol_list[i] /= max_val

    # Create test dataloader
    test_dataset = ACDCDataset25D(volumes_list=all_test_volumes, 
                                 masks_list=all_test_masks, 
                                 num_input_slices=NUM_SLICES, 
                                 transforms=val_test_transform)
    test_dataloader = DataLoader(test_dataset, batch_size=BATCH_SIZE, shuffle=False, 
                                num_workers=0, pin_memory=True)

    # Calculate B1 map
    all_images_tensor_for_b1 = convert_volumes_to_tensor(all_train_volumes + all_test_volumes)
    common_b1_map = calculate_ultimate_common_b1_map(
        all_images=all_images_tensor_for_b1, 
        device=str(DEVICE), 
        save_path="acdc_ultimate_b1_map.pth"
    )

    # K-Fold Cross-Validation
    kf = KFold(n_splits=N_SPLITS, shuffle=True, random_state=42)
    all_train_indices = np.array(range(len(all_train_volumes)))
    fold_validation_results = []

    print("\n" + "="*80)
    print(f"BẮT ĐẦU QUÁ TRÌNH HUẤN LUYỆN {N_SPLITS}-FOLD CROSS-VALIDATION")
    print("="*80)

    for fold, (train_indices, val_indices) in enumerate(kf.split(all_train_indices)):
        best_val_metric = train_one_fold(
            fold, train_indices, val_indices, 
            all_train_volumes, all_train_masks, 
            common_b1_map, train_transform, val_test_transform
        )
        fold_validation_results.append(best_val_metric)

    # --- Tổng hợp kết quả ---
    print("\n" + "="*80)
    print("TỔNG HỢP KẾT QUẢ CROSS-VALIDATION")
    print("="*80)
    mean_dice, std_dice = np.mean(fold_validation_results), np.std(fold_validation_results)
    print(f"Điểm Dice trên tập validation của các folds: {[f'{d:.4f}' for d in fold_validation_results]}")
    print(f"===> Dice trung bình trên {N_SPLITS} Folds: {mean_dice:.4f} ± {std_dice:.4f}")
    
    # --- Đánh giá cuối cùng trên Test Set ---
    print("\n" + "="*80)
    print("ĐÁNH GIÁ CUỐI CÙNG TRÊN TẬP TEST")
    print("="*80)
    
    all_folds_test_metrics = []
    class_map = {0: 'BG', 1: 'RV', 2: 'MYO', 3: 'LV'}

    for fold in range(N_SPLITS):
        print(f"\n--- Đánh giá model từ Fold {fold+1} trên Test Set ---")
        model = RobustMedVFL_UNet(n_channels=NUM_SLICES, n_classes=NUM_CLASSES).to(DEVICE)
        model_path = f"best_model_fold_{fold}.pth"
        
        if os.path.exists(model_path):
            model.load_state_dict(torch.load(model_path, map_location=DEVICE, weights_only=True))
            test_metrics = evaluate_metrics_with_tta(model, test_dataloader, DEVICE, NUM_CLASSES)
            all_folds_test_metrics.append(test_metrics)
            
            # In kết quả chi tiết cho từng fold
            print(f"   --- Metrics chi tiết cho Fold {fold+1} ---")
            for key, value in test_metrics.items():
                if isinstance(value, list):
                    print(f"     {key}:")
                    for c_idx, v in enumerate(value):
                        class_name = class_map.get(c_idx, f"Class {c_idx}")
                        print(f"       => {class_name:<15}: {v:.4f}")
                else:
                    print(f"     {key}: {value:.4f}")
        else:
            print(f"   Lỗi: Không tìm thấy file model '{model_path}'. Bỏ qua fold này.")

    # Tính trung bình các metrics từ tất cả các folds
    if all_folds_test_metrics:
        avg_test_metrics = {}
        metric_keys = all_folds_test_metrics[0].keys()
    
        for key in metric_keys:
            all_values = [m[key] for m in all_folds_test_metrics]
            avg_test_metrics[key] = np.mean(all_values, axis=0)
            
        print("\n--- KẾT QUẢ TEST TRUNG BÌNH TRÊN TẤT CẢ CÁC FOLDS ---")
        print(f"===> Accuracy           : {avg_test_metrics['accuracy']:.4f}")
        print(f"===> Dice (Foreground)    : {np.mean(avg_test_metrics['dice_scores'][1:]):.4f}")
        print(f"===> IoU (Foreground)     : {np.mean(avg_test_metrics['iou'][1:]):.4f}")
        print(f"===> Precision (Foreground): {np.mean(avg_test_metrics['precision'][1:]):.4f}")
        print(f"===> Recall (Foreground)   : {np.mean(avg_test_metrics['recall'][1:]):.4f}")
        print(f"===> F1-Score (Foreground): {np.mean(avg_test_metrics['f1_score'][1:]):.4f}")
        print("="*80)


if __name__ == "__main__":
    main()

