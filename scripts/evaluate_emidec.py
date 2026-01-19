import torch
import torch.nn.functional as F
import numpy as np
import os
import sys
from torch.utils.data import DataLoader
import albumentations as A
from albumentations.pytorch import ToTensorV2
from tqdm import tqdm

sys.path.insert(0, os.path.abspath(os.path.join(os.path.dirname(__file__), '..')))

from src.models.unet import RobustMedVFL_UNet 
from src.data_utils.emidec_dataset_optimized import (
    EmidecDataset25DOptimized,
    get_emidec_volume_ids,
)
from src import config

NUM_CLASSES = 5
DEVICE = torch.device('cuda' if torch.cuda.is_available() else 'cpu')
BATCH_SIZE = 16
NUM_SLICES = 5

EMIDEC_CLASS_MAP = {0: 'BG', 1: 'Cavity', 2: 'MYO', 3: 'Infarction', 4: 'NoReflow'}

print(f"Device: {DEVICE}")
print(f"Evaluating EMIDEC model on test set")

test_transform = A.Compose([
    ToTensorV2(),
])

def evaluate_metrics(model, dataloader, device, num_classes=5):
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
        eval_pbar = tqdm(dataloader, desc="Evaluating", ncols=100)
        for imgs, tgts in eval_pbar:
            imgs, tgts = imgs.to(device), tgts.to(device)
            if imgs.size(0) == 0: 
                continue
            
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

print("\nLoading test data EMIDEC")

PREPROCESSED_ROOT = '/home/linhdang/workspace/minhbao_workspace/Phys/preprocessed_data/EMIDEC'
TEST_NPY_DIR = os.path.join(PREPROCESSED_ROOT, "test")

if not os.path.exists(TEST_NPY_DIR):
    print(f"Error: Directory not found: {TEST_NPY_DIR}")
    sys.exit(1)

all_volume_ids = get_emidec_volume_ids(TEST_NPY_DIR)
print(f"Total volumes: {len(all_volume_ids)}")

test_volume_ids = all_volume_ids

print(f"Using {len(test_volume_ids)} volumes for test set")

test_dataset = EmidecDataset25DOptimized(
    npy_dir=TEST_NPY_DIR,
    volume_ids=test_volume_ids,
    num_input_slices=NUM_SLICES,
    transforms=test_transform,
    max_cache_size=10,
    use_memmap=True
)

test_loader = DataLoader(
    test_dataset, 
    batch_size=BATCH_SIZE, 
    shuffle=False, 
    num_workers=0, 
    pin_memory=True
)

print(f"Test set: {len(test_dataset)} slices")

print(f"\n{'='*60}")
print("Evaluating on test set")
print(f"{'='*60}\n")

MODEL_PATH = "weights/best_model_emidec.pth"

if not os.path.exists(MODEL_PATH):
    print(f"Error: Model not found: {MODEL_PATH}")
    sys.exit(1)

print(f"Loading model from: {MODEL_PATH}")
model = RobustMedVFL_UNet(n_channels=NUM_SLICES, n_classes=NUM_CLASSES, deep_supervision=True).to(DEVICE)
model.load_state_dict(torch.load(MODEL_PATH))
model.eval()

print("Model loaded")

print("\nRunning evaluation on test set")
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

avg_fg_dice = np.mean(test_dice[1:])
avg_fg_iou = np.mean(test_iou[1:])
avg_fg_precision = np.mean(test_precision[1:])
avg_fg_recall = np.mean(test_recall[1:])
avg_fg_f1 = np.mean(test_f1[1:])

print(f"\n{'='*60}")
print("Test set results")
print(f"{'='*60}")

print(f"\n  Test Results (Mean of ALL {NUM_CLASSES} Classes):")
print(f"    Accuracy: {test_accuracy:.4f}; Dice: {mean_dice_all:.4f}; IoU: {mean_iou_all:.4f}; "
      f"Precision: {mean_precision_all:.4f}; Recall: {mean_recall_all:.4f}; F1-score: {mean_f1_all:.4f}")

print(f"\n  Per-Class Metrics:")
for c_idx in range(NUM_CLASSES):
    class_name = EMIDEC_CLASS_MAP.get(c_idx, f"Class {c_idx}")
    print(f"    => {class_name:<20}: "
          f"Dice: {test_dice[c_idx]:.4f}, "
          f"IoU: {test_iou[c_idx]:.4f}, "
          f"Precision: {test_precision[c_idx]:.4f}, "
          f"Recall: {test_recall[c_idx]:.4f}, "
          f"F1: {test_f1[c_idx]:.4f}")

print(f"\n  Foreground Classes Summary:")
print(f"    Avg Foreground Dice:      {avg_fg_dice:.4f}")
print(f"    Avg Foreground IoU:       {avg_fg_iou:.4f}")
print(f"    Avg Foreground Precision: {avg_fg_precision:.4f}")
print(f"    Avg Foreground Recall:    {avg_fg_recall:.4f}")
print(f"    Avg Foreground F1:        {avg_fg_f1:.4f}")

print(f"\n{'='*60}")
print("Evaluation complete")
print(f"{'='*60}")

os.makedirs("results", exist_ok=True)
results_file = "results/emidec_test_results.txt"
with open(results_file, 'w') as f:
    f.write("="*60 + "\n")
    f.write("EMIDEC TEST SET RESULTS\n")
    f.write("="*60 + "\n\n")
    f.write(f"Model: {MODEL_PATH}\n")
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
        class_name = EMIDEC_CLASS_MAP.get(c_idx, f"Class {c_idx}")
        f.write(f"  {class_name}:\n")
        f.write(f"    Dice: {test_dice[c_idx]:.4f}\n")
        f.write(f"    IoU: {test_iou[c_idx]:.4f}\n")
        f.write(f"    Precision: {test_precision[c_idx]:.4f}\n")
        f.write(f"    Recall: {test_recall[c_idx]:.4f}\n")
        f.write(f"    F1: {test_f1[c_idx]:.4f}\n\n")
    
    f.write("Foreground Classes Summary:\n")
    f.write(f"  Avg Dice: {avg_fg_dice:.4f}\n")
    f.write(f"  Avg IoU: {avg_fg_iou:.4f}\n")
    f.write(f"  Avg Precision: {avg_fg_precision:.4f}\n")
    f.write(f"  Avg Recall: {avg_fg_recall:.4f}\n")
    f.write(f"  Avg F1: {avg_fg_f1:.4f}\n")

print(f"\nResults saved to: {results_file}")
