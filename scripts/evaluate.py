import torch
import torch.nn.functional as F
import numpy as np
import matplotlib.pyplot as plt
import random
import albumentations as A
from albumentations.pytorch import ToTensorV2
import matplotlib.colors as mcolors
import os


BRATS_CLASS_MAP = {
    0: "Background",
    1: "NCR/NET",
    2: "Edema",
    3: "ET"
}

BRATS_COLOR_MAP = {
    0: 'black',
    1: '#FF0000',
    2: '#00FF00',
    3: '#0000FF'
}


def calculate_brats_regions_dice(preds, targets, num_classes=4):
    """
    Calculate Dice scores for BraTS regions: ET, TC, WT
    
    BraTS labels: 0=Background, 1=NCR/NET, 2=Edema, 3=ET
    Regions:
        ET (Enhancing Tumor) = label 3
        TC (Tumor Core) = label 1 + label 3
        WT (Whole Tumor) = label 1 + label 2 + label 3
    """
    et_pred = (preds == 3).float()
    et_target = (targets == 3).float()
    
    tc_pred = ((preds == 1) | (preds == 3)).float()
    tc_target = ((targets == 1) | (targets == 3)).float()
    
    wt_pred = ((preds == 1) | (preds == 2) | (preds == 3)).float()
    wt_target = ((targets == 1) | (targets == 2) | (targets == 3)).float()
    
    def dice_score(pred, target):
        intersection = (pred * target).sum()
        return (2. * intersection + 1e-6) / (pred.sum() + target.sum() + 1e-6)
    
    et_dice = dice_score(et_pred, et_target).item()
    tc_dice = dice_score(tc_pred, tc_target).item()
    wt_dice = dice_score(wt_pred, wt_target).item()
    
    return {
        'ET': et_dice,
        'TC': tc_dice,
        'WT': wt_dice,
        'mean': (et_dice + tc_dice + wt_dice) / 3.0
    }


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
    
    et_dice_sum = 0.0
    tc_dice_sum = 0.0
    wt_dice_sum = 0.0

    with torch.no_grad():
        for imgs, tgts in dataloader:
            imgs, tgts = imgs.to(device), tgts.to(device)
            if imgs.size(0) == 0: continue
            
            logits_list, _ = model(imgs)
            logits = logits_list[-1]
            preds = torch.argmax(F.softmax(logits, dim=1), dim=1)
            batches += 1
            total_correct_pixels += (preds == tgts).sum().item()
            total_pixels += tgts.numel()
            
            brats_regions = calculate_brats_regions_dice(preds, tgts, num_classes)
            et_dice_sum += brats_regions['ET']
            tc_dice_sum += brats_regions['TC']
            wt_dice_sum += brats_regions['WT']

            for c in range(num_classes):
                pc_f = (preds == c).float().view(-1)
                tc_f = (tgts == c).float().view(-1)
                inter = (pc_f * tc_f).sum()

                dice_s[c] += ((2. * inter + 1e-6) / (pc_f.sum() + tc_f.sum() + 1e-6)).item()
                iou_s[c] += ((inter + 1e-6) / (pc_f.sum() + tc_f.sum() - inter + 1e-6)).item()
                tp[c] += inter.item()
                fp[c] += (pc_f.sum() - inter).item()
                fn[c] += (tc_f.sum() - inter).item()

    metrics = {'accuracy': 0.0, 'dice_scores': [], 'iou': [], 'precision': [], 'recall': [], 'f1_score': [],
               'ET': 0.0, 'TC': 0.0, 'WT': 0.0, 'avg_regions': 0.0}

    if batches > 0:
        if total_pixels > 0:
            metrics['accuracy'] = total_correct_pixels / total_pixels
        
        metrics['ET'] = et_dice_sum / batches
        metrics['TC'] = tc_dice_sum / batches
        metrics['WT'] = wt_dice_sum / batches
        metrics['avg_regions'] = (metrics['ET'] + metrics['TC'] + metrics['WT']) / 3.0
        
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


def evaluate_metrics_with_tta(model, dataloader, device, num_classes=4):
    model.eval()
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
            if imgs.size(0) == 0: continue
            
            num_batches += 1
            
            img_orig = imgs
            img_hflip = torch.flip(imgs, dims=[-1])
            tta_batch = torch.cat([img_orig, img_hflip], dim=0)
            
            logits_list, _ = model(tta_batch)
            probs_batch = torch.softmax(logits_list[-1], dim=1)
            prob_orig, prob_hflip = torch.chunk(probs_batch, 2, dim=0)

            prob_hflip_restored = torch.flip(prob_hflip, dims=[-1])
            avg_probs = (prob_orig + prob_hflip_restored) / 2.0
            preds = torch.argmax(avg_probs, dim=1)
            
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

    metrics = {
        'accuracy': total_correct_pixels / total_pixels if total_pixels > 0 else 0,
        'dice_scores': (total_dice / num_batches).tolist() if num_batches > 0 else [0]*num_classes,
        'iou': (total_iou / num_batches).tolist() if num_batches > 0 else [0]*num_classes,
        'precision': (total_precision / num_batches).tolist() if num_batches > 0 else [0]*num_classes,
        'recall': (total_recall / num_batches).tolist() if num_batches > 0 else [0]*num_classes,
        'f1_score': (total_f1 / num_batches).tolist() if num_batches > 0 else [0]*num_classes,
    }
    return metrics


def run_and_print_test_evaluation(test_dataloader, device, num_classes, num_slices):
    print("\n--- Evaluating on Test Set with TTA ---")
    from src.models.unet_archetype import RobustMedVFL_UNet
    
    model = RobustMedVFL_UNet(n_channels=num_slices, n_classes=num_classes)
    model_path = "best_model.pth"
    if os.path.exists(model_path):
        print(f"Loading model from '{model_path}'...")
        model.load_state_dict(torch.load(model_path, map_location=device))
        model.to(device)
    else:
        print(f"Error: Model file '{model_path}' not found.")
        return
        
    if test_dataloader and test_dataloader.dataset and len(test_dataloader.dataset) > 0:
        test_metrics = evaluate_metrics_with_tta(model, test_dataloader, device, num_classes)
        test_accuracy = test_metrics['accuracy']
        
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
            class_name = BRATS_CLASS_MAP.get(c_idx, f"Class {c_idx}")
            print(f"    => {class_name:<20}: "
                  f"Dice: {test_metrics['dice_scores'][c_idx]:.4f}, "
                  f"IoU: {test_metrics['iou'][c_idx]:.4f}, "
                  f"Precision: {test_metrics['precision'][c_idx]:.4f}, "
                  f"Recall: {test_metrics['recall'][c_idx]:.4f}, "
                  f"F1: {test_metrics['f1_score'][c_idx]:.4f}")
    else:
        print("\nTest dataset not available or empty.")


def visualize_final_results_2_5D(volumes_np, masks_np, num_classes, num_samples, device, num_slices, model_path="best_model.pth"):
    if not volumes_np:
        print("No test data for visualization.")
        return
        
    print("\n--- Visualizing Final Results ---")
    from src.models.unet_archetype import RobustMedVFL_UNet
    
    model = RobustMedVFL_UNet(n_channels=num_slices, n_classes=num_classes)
    
    if os.path.exists(model_path):
        print(f"Loading model from '{model_path}'...")
        model.load_state_dict(torch.load(model_path, map_location=device))
        model.to(device)
    else:
        print(f"Error: Model file '{model_path}' not found.")
        return

    model.eval()
    vis_transform = A.Compose([ToTensorV2()])

    index_map = []
    for vol_idx, vol in enumerate(volumes_np):
        for slice_idx in range(vol.shape[3]):
            index_map.append((vol_idx, slice_idx))
            
    if not index_map:
        print("No slices to visualize.")
        return
        
    sample_indices = random.sample(range(len(index_map)), min(num_samples, len(index_map)))

    colors = [BRATS_COLOR_MAP.get(i, 'black') for i in range(num_classes)]
    cmap = mcolors.ListedColormap(colors)

    for i, idx in enumerate(sample_indices):
        vol_idx, center_slice_idx = index_map[idx]
        
        original_image_slice = volumes_np[vol_idx][0, :, :, center_slice_idx]
        ground_truth_mask_slice = masks_np[vol_idx][:, :, center_slice_idx]
        
        current_volume = volumes_np[vol_idx]
        num_slices_in_vol = current_volume.shape[3]
        
        slice_indices_for_stack = []
        for offset in [-2, -1, 0, 1, 2]:
            slice_idx = np.clip(center_slice_idx + offset, 0, num_slices_in_vol - 1)
            slice_indices_for_stack.append(slice_idx)
            
        image_stack_np = np.stack(
            [current_volume[:, :, :, s].transpose(1, 2, 0) for s in slice_indices_for_stack],
            axis=-1
        ).astype(np.float32)
        
        image_stack_np = image_stack_np.reshape(image_stack_np.shape[0], image_stack_np.shape[1], -1)
        
        transformed = vis_transform(image=image_stack_np)
        model_input = transformed['image'].unsqueeze(0).to(device)

        with torch.no_grad():
            logits_list, _ = model(model_input)
            logits = logits_list[-1] 
            probabilities = torch.softmax(logits, dim=1)
            prediction = torch.argmax(probabilities, dim=1).squeeze(0)
            
        predicted_mask_slice = prediction.cpu().numpy()

        fig, axes = plt.subplots(1, 3, figsize=(18, 6))
        fig.suptitle(f'Sample {i+1} (Volume: {vol_idx}, Slice: {center_slice_idx})', fontsize=16)
        
        axes[0].imshow(original_image_slice, cmap='gray')
        axes[0].set_title('Original MRI')
        axes[0].axis('off')

        axes[1].imshow(original_image_slice, cmap='gray')
        pred_masked_display = np.ma.masked_where(predicted_mask_slice == 0, predicted_mask_slice)
        axes[1].imshow(pred_masked_display, cmap=cmap, alpha=0.6, vmin=0, vmax=num_classes-1)
        axes[1].set_title('Prediction')
        axes[1].axis('off')
        
        axes[2].imshow(original_image_slice, cmap='gray')
        gt_masked_display = np.ma.masked_where(ground_truth_mask_slice == 0, ground_truth_mask_slice)
        axes[2].imshow(gt_masked_display, cmap=cmap, alpha=0.6, vmin=0, vmax=num_classes-1)
        axes[2].set_title('Ground Truth')
        axes[2].axis('off')

        import matplotlib.patches as mpatches

        legend_elements = [
            mpatches.Patch(color=BRATS_COLOR_MAP[i], label=BRATS_CLASS_MAP[i])
            for i in range(1, num_classes)
        ]
        fig.legend(handles=legend_elements, loc='lower center', ncol=3, bbox_to_anchor=(0.5, -0.02))
        plt.tight_layout(rect=[0, 0.05, 1, 0.95])
        
        os.makedirs('visualizations', exist_ok=True)
        save_path = f'visualizations/sample_{i+1}_vol{vol_idx}_slice{center_slice_idx}.png'
        plt.savefig(save_path, dpi=150, bbox_inches='tight')
        print(f"   Saved: {save_path}")
        
        plt.show()
        plt.close()

