"""
Evaluation and visualization functions for medical image segmentation.
Includes metrics calculation, test-time augmentation, and result visualization.
"""

import os
import random
import numpy as np
import torch
import torch.nn.functional as F
import matplotlib.pyplot as plt
import matplotlib.colors as mcolors
import albumentations as A
from albumentations.pytorch import ToTensorV2


# =============================================================================
# --- Evaluation Metrics ---
# =============================================================================

def evaluate_metrics(model, dataloader, device, num_classes=4):
    """
    Hàm đánh giá các chỉ số cho mô hình phân đoạn.
    Tương thích với output dạng list từ UNet++ (deep supervision).
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
        for imgs, tgts in dataloader:
            imgs, tgts = imgs.to(device), tgts.to(device)
            if imgs.size(0) == 0:
                continue
            
            logits_list, _ = model(imgs)
            logits = logits_list[-1]  # Lấy output cuối cùng
            
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
            for key in ['dice_scores', 'iou', 'precision', 'recall', 'f1_score']:
                metrics[key].append(0.0)
            
    return metrics


def evaluate_metrics_with_tta(model, dataloader, device, num_classes=4):
    """
    Hàm đánh giá CÓ TÍCH HỢP TTA (Test-Time Augmentation).
    Sử dụng 2 phép biến đổi: gốc và lật ngang.
    """
    model.eval()
    
    # Khởi tạo các biến để lưu tổng các chỉ số
    total_dice = np.zeros(num_classes)
    total_iou = np.zeros(num_classes)
    total_precision = np.zeros(num_classes)
    total_recall = np.zeros(num_classes)
    total_f1 = np.zeros(num_classes)
    total_correct_pixels = 0
    total_pixels = 0
    num_batches = 0

    with torch.no_grad():
        for imgs, tgts in dataloader:
            imgs, tgts = imgs.to(device), tgts.to(device)
            if imgs.size(0) == 0:
                continue
            
            num_batches += 1
            
            # --- TTA Logic ---
            img_orig = imgs
            img_hflip = torch.flip(imgs, dims=[-1])  # Lật ngang

            # Gộp lại thành một batch lớn
            tta_batch = torch.cat([img_orig, img_hflip], dim=0)
            
            # Dự đoán trên cả batch
            logits_list, _ = model(tta_batch)
            probs_batch = torch.softmax(logits_list[-1], dim=1)
            
            # Tách kết quả cho từng phiên bản
            prob_orig, prob_hflip = torch.chunk(probs_batch, 2, dim=0)

            # Hoàn tác phép biến đổi trên kết quả
            prob_hflip_restored = torch.flip(prob_hflip, dims=[-1])
            
            # Lấy trung bình 2 bản đồ xác suất
            avg_probs = (prob_orig + prob_hflip_restored) / 2.0
            
            # Lấy dự đoán cuối cùng
            preds = torch.argmax(avg_probs, dim=1)
            # --- End TTA Logic ---
            
            # Tính toán metrics cho batch hiện tại
            total_correct_pixels += (preds == tgts).sum().item()
            total_pixels += tgts.numel()

            for c in range(num_classes):
                pred_mask = (preds == c)
                true_mask = (tgts == c)
                
                tp = (pred_mask & true_mask).sum().item()
                fp = (pred_mask & ~true_mask).sum().item()
                fn = (~pred_mask & true_mask).sum().item()
                
                total_dice[c] += (2. * tp) / (2 * tp + fp + fn + 1e-8)
                total_iou[c] += tp / (tp + fp + fn + 1e-8)
                
                precision = tp / (tp + fp + 1e-8)
                recall = tp / (tp + fn + 1e-8)
                
                total_precision[c] += precision
                total_recall[c] += recall
                total_f1[c] += (2 * precision * recall) / (precision + recall + 1e-8)

    # Tính trung bình các metrics
    metrics = {
        'accuracy': total_correct_pixels / total_pixels if total_pixels > 0 else 0,
        'dice_scores': (total_dice / num_batches).tolist() if num_batches > 0 else [0]*num_classes,
        'iou': (total_iou / num_batches).tolist() if num_batches > 0 else [0]*num_classes,
        'precision': (total_precision / num_batches).tolist() if num_batches > 0 else [0]*num_classes,
        'recall': (total_recall / num_batches).tolist() if num_batches > 0 else [0]*num_classes,
        'f1_score': (total_f1 / num_batches).tolist() if num_batches > 0 else [0]*num_classes,
    }
    return metrics


# =============================================================================
# --- Visualization ---
# =============================================================================

# Định nghĩa cho việc trực quan hóa
ACDC_CLASS_MAP = {
    0: "Background",
    1: "Right Ventricle (RV)",
    2: "Myocardium (MYO)",
    3: "Left Ventricle (LV)"
}

ACDC_COLOR_MAP = {
    0: 'black',
    1: '#FF0000',
    2: '#00FF00',
    3: '#0000FF'
}


def run_and_print_test_evaluation(model, test_dataloader, device, num_classes):
    """
    Đánh giá model trên tập test với TTA và in ra các chỉ số metrics.
    """
    print("\n--- Evaluating on Test Set with TTA ---")
    
    if test_dataloader and test_dataloader.dataset and len(test_dataloader.dataset) > 0:
        test_metrics = evaluate_metrics_with_tta(model, test_dataloader, device, num_classes)
        
        test_accuracy = test_metrics['accuracy']
        
        # Tính trung bình trên tất cả các class
        mean_dice = np.mean(test_metrics['dice_scores'])
        mean_iou = np.mean(test_metrics['iou'])
        mean_precision = np.mean(test_metrics['precision'])
        mean_recall = np.mean(test_metrics['recall'])
        mean_f1 = np.mean(test_metrics['f1_score'])

        print(f"\n  Test Results (Mean of ALL {num_classes} Classes):")
        print(f"    Accuracy: {test_accuracy:.4f}; Dice: {mean_dice:.4f}; IoU: {mean_iou:.4f}; "
              f"Precision: {mean_precision:.4f}; Recall: {mean_recall:.4f}; F1-score: {mean_f1:.4f}")
        
        print("\n  Per-Class Metrics:")
        for c_idx in range(num_classes):
            class_name = ACDC_CLASS_MAP.get(c_idx, f"Class {c_idx}")
            print(f"    => {class_name:<20}: "
                  f"Dice: {test_metrics['dice_scores'][c_idx]:.4f}, "
                  f"IoU: {test_metrics['iou'][c_idx]:.4f}, "
                  f"Precision: {test_metrics['precision'][c_idx]:.4f}, "
                  f"Recall: {test_metrics['recall'][c_idx]:.4f}, "
                  f"F1: {test_metrics['f1_score'][c_idx]:.4f}")
        
        return test_metrics
    else:
        print("\nTest dataset not available or empty. Skipping test evaluation.")
        return None


def visualize_final_results_2_5D(model, volumes_np, masks_np, num_classes, num_samples, num_slices, device):
    """
    Trực quan hóa kết quả phân đoạn.
    
    Args:
        model: Mô hình đã được huấn luyện.
        volumes_np: Danh sách các volume ảnh 3D.
        masks_np: Danh sách các volume mask 3D tương ứng.
        num_classes: Số lượng class.
        num_samples: Số lượng mẫu để hiển thị.
        num_slices: Số lượng slices đầu vào (2.5D).
        device: Thiết bị (CPU/GPU).
    """
    if not volumes_np:
        print("Không có dữ liệu test để trực quan hóa.")
        return
        
    print("\n--- Visualizing Final Results ---")
    
    model.eval()
    
    vis_transform = A.Compose([ToTensorV2()])

    # Tạo index map để chọn ngẫu nhiên các lát cắt
    index_map = []
    for vol_idx, vol in enumerate(volumes_np):
        for slice_idx in range(vol.shape[2]):
            index_map.append((vol_idx, slice_idx))
            
    if not index_map:
        print("Không có lát cắt nào để hiển thị.")
        return
    
    sample_indices = random.sample(range(len(index_map)), min(num_samples, len(index_map)))

    # Tạo colormap tùy chỉnh
    colors = [ACDC_COLOR_MAP.get(i, 'black') for i in range(num_classes)]
    cmap = mcolors.ListedColormap(colors)

    for i, idx in enumerate(sample_indices):
        vol_idx, center_slice_idx = index_map[idx]
        
        original_image_slice = volumes_np[vol_idx][:, :, center_slice_idx]
        ground_truth_mask_slice = masks_np[vol_idx][:, :, center_slice_idx]
        
        # Chuẩn bị input 2.5D với đúng số lát cắt
        current_volume = volumes_np[vol_idx]
        num_slices_in_vol = current_volume.shape[2]
        
        radius = (num_slices - 1) // 2
        slice_indices_for_stack = []
        for offset in range(-radius, radius + 1):
            slice_idx = np.clip(center_slice_idx + offset, 0, num_slices_in_vol - 1)
            slice_indices_for_stack.append(slice_idx)
            
        image_stack_np = np.stack(
            [current_volume[:, :, s] for s in slice_indices_for_stack], axis=-1
        ).astype(np.float32)
        
        # Áp dụng transform
        transformed = vis_transform(image=image_stack_np)
        model_input = transformed['image'].unsqueeze(0).to(device)

        # Lấy dự đoán từ model
        with torch.no_grad():
            logits_list, _ = model(model_input)
            logits = logits_list[-1] 
            probabilities = torch.softmax(logits, dim=1)
            prediction = torch.argmax(probabilities, dim=1).squeeze(0)
            
        predicted_mask_slice = prediction.cpu().numpy()

        # Vẽ kết quả
        fig, axes = plt.subplots(1, 3, figsize=(18, 6))
        fig.suptitle(f'Sample {i+1} (Volume: {vol_idx}, Slice: {center_slice_idx})', fontsize=16)
        
        axes[0].imshow(original_image_slice, cmap='gray')
        axes[0].set_title('Ảnh MRI Gốc')
        axes[0].axis('off')

        axes[1].imshow(original_image_slice, cmap='gray')
        pred_masked_display = np.ma.masked_where(predicted_mask_slice == 0, predicted_mask_slice)
        axes[1].imshow(pred_masked_display, cmap=cmap, alpha=0.6, vmin=0, vmax=num_classes-1)
        axes[1].set_title('Dự đoán (Model Tốt Nhất)')
        axes[1].axis('off')
        
        axes[2].imshow(original_image_slice, cmap='gray')
        gt_masked_display = np.ma.masked_where(ground_truth_mask_slice == 0, ground_truth_mask_slice)
        axes[2].imshow(gt_masked_display, cmap=cmap, alpha=0.6, vmin=0, vmax=num_classes-1)
        axes[2].set_title('Mặt nạ Ground Truth')
        axes[2].axis('off')

        legend_elements = [
            plt.Rectangle((0, 0), 1, 1, color=ACDC_COLOR_MAP[i], label=ACDC_CLASS_MAP[i])
            for i in range(1, num_classes)
        ]
        fig.legend(handles=legend_elements, loc='lower center', ncol=3, bbox_to_anchor=(0.5, -0.02))
        
        plt.tight_layout(rect=[0, 0.05, 1, 0.95])
        plt.show()

