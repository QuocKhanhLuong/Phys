"""
Encoder Ablation Training Script

Trains PIE-UNet with different encoder types for ablation study.
Duplicated from scripts/train_acdc.py with encoder selection.

Usage:
    python ablation/encoder/train_encoder.py --encoder standard
    python ablation/encoder/train_encoder.py --encoder se
    python ablation/encoder/train_encoder.py --encoder resnet
    python ablation/encoder/train_encoder.py --encoder cbam
    python ablation/encoder/train_encoder.py --encoder nae
"""

import torch
import torch.nn as nn
import torch.optim as optim
import torch.nn.functional as F
import numpy as np
import time
import os
import sys
import argparse
from pathlib import Path
from sklearn.model_selection import train_test_split
from torch.utils.data import DataLoader
import albumentations as A
from albumentations.pytorch import ToTensorV2
from itertools import chain
import cv2
from tqdm import tqdm
from monai.metrics import compute_hausdorff_distance
import warnings

warnings.filterwarnings("ignore")

# Add project root to path
PROJECT_ROOT = Path(__file__).parent.parent.parent
sys.path.insert(0, str(PROJECT_ROOT))

from ablation.encoder.config import ENCODER_CONFIGS, TRAINING_CONFIG, DATA_CONFIG, OUTPUT_CONFIG
from ablation.encoder.pie_unet_encoder import PIE_UNet_Encoder
from src.models.epure import ePURE
from src.modules.losses import CombinedLoss
from src.data_utils.acdc_dataset_optimized import (
    ACDCDataset25DOptimized,
    get_acdc_volume_ids,
)
from src.utils.helpers import calculate_ultimate_common_b1_map


# =============================================================================
# AUGMENTATION PIPELINES (Same as train_acdc.py)
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

val_test_transform = A.Compose([ToTensorV2()])


# =============================================================================
# EVALUATION METRICS (Same as train_acdc.py)
# =============================================================================

def evaluate_metrics(model, dataloader, device, num_classes=4, compute_hd95=True):
    """Evaluate segmentation metrics (Dice, and optional HD95)."""
    model.eval()
    dice_s = [0.0] * num_classes
    hd95_s = [0.0] * num_classes  # Only used if compute_hd95=True
    hd95_counts = [0] * num_classes
    batches = 0

    with torch.no_grad():
        for imgs, tgts in tqdm(dataloader, desc="Evaluating", leave=False):
            imgs, tgts = imgs.to(device), tgts.to(device)
            if imgs.size(0) == 0:
                continue
            
            logits_list, _ = model(imgs)
            logits = logits_list[-1]
            preds = torch.argmax(F.softmax(logits, dim=1), dim=1)
            batches += 1

            for c in range(num_classes):
                pc_f = (preds == c).float().view(-1)
                tc_f = (tgts == c).float().view(-1)
                inter = (pc_f * tc_f).sum()
                dice_s[c] += ((2. * inter + 1e-6) / (pc_f.sum() + tc_f.sum() + 1e-6)).item()

            # HD95 (Optimization: Skip during training validation to fix "stuck" issue)
            if compute_hd95:
                # Keep on GPU if possible, but MONAI might need CPU for some versions
                # Try keeping on GPU first for speed? Safe bet is CPU for now but conditional.
                preds_cpu = preds.detach().cpu()
                tgts_cpu = tgts.detach().cpu()
                preds_oh = F.one_hot(preds_cpu, num_classes=num_classes).permute(0, 3, 1, 2).float()
                tgts_oh = F.one_hot(tgts_cpu, num_classes=num_classes).permute(0, 3, 1, 2).float()
                
                try:
                    hd95_batch = compute_hausdorff_distance(y_pred=preds_oh, y=tgts_oh, include_background=True, percentile=95.0)
                    for c in range(num_classes):
                        valid_vals = hd95_batch[:, c]
                        mask = ~torch.isnan(valid_vals) & ~torch.isinf(valid_vals)
                        if mask.any():
                            hd95_s[c] += valid_vals[mask].sum().item()
                            hd95_counts[c] += mask.sum().item()
                except:
                    pass

    metrics = {'dice_scores': [], 'hd95': []}
    if batches > 0:
        for c in range(num_classes):
            metrics['dice_scores'].append(dice_s[c] / batches)
            if compute_hd95:
                metrics['hd95'].append(hd95_s[c] / hd95_counts[c] if hd95_counts[c] > 0 else float('nan'))
            else:
                metrics['hd95'].append(float('nan'))
    
    return metrics


# =============================================================================
# TRAINING FUNCTION
# =============================================================================

def train_encoder(encoder_name, num_epochs=None):
    """Train a model with the specified encoder type."""
    
    if encoder_name not in ENCODER_CONFIGS:
        raise ValueError(f"Unknown encoder: {encoder_name}. Available: {list(ENCODER_CONFIGS.keys())}")
    
    enc_config = ENCODER_CONFIGS[encoder_name]
    encoder_type = enc_config["type"]
    
    # Configuration
    DEVICE = torch.device('cuda' if torch.cuda.is_available() else 'cpu')
    NUM_CLASSES = TRAINING_CONFIG["num_classes"]
    NUM_SLICES = TRAINING_CONFIG["num_slices"]
    BATCH_SIZE = TRAINING_CONFIG["batch_size"]
    LEARNING_RATE = TRAINING_CONFIG["learning_rate"]
    EARLY_STOP_PATIENCE = TRAINING_CONFIG["early_stopping_patience"]
    
    if num_epochs is None:
        num_epochs = TRAINING_CONFIG["num_epochs"]
    
    print("=" * 70)
    print(f"ENCODER ABLATION TRAINING: {enc_config['name']}")
    print(f"  Type: {encoder_type}")
    print(f"  Description: {enc_config['description']}")
    print("=" * 70)
    print(f"Device: {DEVICE}")
    print(f"Epochs: {num_epochs}, Batch size: {BATCH_SIZE}")
    
    # ==========================================================================
    # DATA LOADING
    # ==========================================================================
    
    train_npy_dir = str(DATA_CONFIG["train_dir"])
    test_npy_dir = str(DATA_CONFIG["test_dir"])
    
    all_train_volume_ids = get_acdc_volume_ids(train_npy_dir)
    indices = list(range(len(all_train_volume_ids)))
    train_indices, val_indices = train_test_split(indices, test_size=0.2, random_state=42)
    train_volume_ids = [all_train_volume_ids[i] for i in train_indices]
    val_volume_ids = [all_train_volume_ids[i] for i in val_indices]
    test_volume_ids = get_acdc_volume_ids(test_npy_dir)
    
    print(f"Data: {len(train_volume_ids)} train, {len(val_volume_ids)} val, {len(test_volume_ids)} test")
    
    # ePURE NOT NEEDED for encoder ablation (encoders don't use noise estimation)
    # ePURE_augmenter = ePURE(in_channels=NUM_SLICES).to(DEVICE)
    # ePURE_augmenter.eval()
    
    # Datasets (NO ePURE for encoder ablation - not needed!)
    train_dataset = ACDCDataset25DOptimized(
        npy_dir=train_npy_dir,
        volume_ids=train_volume_ids,
        num_input_slices=NUM_SLICES,
        transforms=train_transform,
        noise_injector_model=None,  # NO ePURE for encoder ablation!
        device=str(DEVICE),
        max_cache_size=15,
        use_memmap=True
    )
    
    val_dataset = ACDCDataset25DOptimized(
        npy_dir=train_npy_dir,
        volume_ids=val_volume_ids,
        num_input_slices=NUM_SLICES,
        transforms=val_test_transform,
        max_cache_size=10,
        use_memmap=True
    )
    
    test_dataset = ACDCDataset25DOptimized(
        npy_dir=test_npy_dir,
        volume_ids=test_volume_ids,
        num_input_slices=NUM_SLICES,
        transforms=val_test_transform,
        max_cache_size=8,
        use_memmap=True
    )
    
    # DataLoaders 
    NUM_WORKERS = 12  
    train_dataloader = DataLoader(train_dataset, batch_size=BATCH_SIZE, shuffle=True, num_workers=NUM_WORKERS, pin_memory=True, persistent_workers=True)
    val_dataloader = DataLoader(val_dataset, batch_size=BATCH_SIZE, shuffle=False, num_workers=NUM_WORKERS, pin_memory=True, persistent_workers=True)
    test_dataloader = DataLoader(test_dataset, batch_size=BATCH_SIZE, shuffle=False, num_workers=NUM_WORKERS, pin_memory=True, persistent_workers=True)
    
    print(f"Slices: {len(train_dataset)} train, {len(val_dataset)} val, {len(test_dataset)} test")
    
    # ==========================================================================
    # B1 MAP CALCULATION
    # ==========================================================================
    
    def convert_npy_to_tensor_for_b1(npy_dir, volume_ids):
        volumes_dir = os.path.join(npy_dir, 'volumes')
        all_slices = []
        for vid in volume_ids:
            vol = np.load(os.path.join(volumes_dir, f'{vid}.npy'))
            for i in range(vol.shape[2]):
                all_slices.append(torch.from_numpy(vol[:, :, i]).unsqueeze(0))
        return torch.stack(all_slices, dim=0).float()
    
    train_val_tensor = convert_npy_to_tensor_for_b1(train_npy_dir, train_volume_ids + val_volume_ids)
    test_tensor = convert_npy_to_tensor_for_b1(test_npy_dir, test_volume_ids)
    all_images_tensor = torch.cat([train_val_tensor, test_tensor], dim=0)
    
    common_b1_map = calculate_ultimate_common_b1_map(
        all_images=all_images_tensor,
        device=str(DEVICE),
        save_path="b1_maps/acdc_cardiac_ultimate_common_b1_map.pth"
    )
    
    # ==========================================================================
    # MODEL, LOSS, OPTIMIZER
    # ==========================================================================
    
    model = PIE_UNet_Encoder(
        n_channels=NUM_SLICES,
        n_classes=NUM_CLASSES,
        encoder_type=encoder_type,
        deep_supervision=True
    ).to(DEVICE)
    
    total_params = sum(p.numel() for p in model.parameters() if p.requires_grad)
    print(f"Model params: {total_params:,}")
    
    criterion = CombinedLoss(
        num_classes=NUM_CLASSES,
        initial_loss_weights=[0.4, 0.4, 0.2],
        class_indices_for_rules=None
    ).to(DEVICE)
    
    optimizer = torch.optim.AdamW(chain(model.parameters(), criterion.parameters()), lr=LEARNING_RATE)
    scheduler = torch.optim.lr_scheduler.ReduceLROnPlateau(optimizer, mode='max', factor=0.5, patience=10)
    
    # ==========================================================================
    # TRAINING LOOP
    # ==========================================================================
    
    best_dice = 0.0
    best_hd95 = float('inf')
    epochs_no_improve = 0
    weights_path = OUTPUT_CONFIG["weights_dir"] / f"best_{encoder_name}.pth"
    
    # Reset GPU peak memory
    if torch.cuda.is_available():
        torch.cuda.reset_peak_memory_stats()
    
    for epoch in range(num_epochs):
        model.train()
        epoch_loss = 0.0
        
        train_pbar = tqdm(train_dataloader, desc=f"Epoch {epoch+1}/{num_epochs}", leave=False)
        for images, targets in train_pbar:
            images, targets = images.to(DEVICE), targets.to(DEVICE)
            optimizer.zero_grad()
            
            b1_map_batch = common_b1_map.expand(images.size(0), -1, -1, -1)
            logits_list, all_eps_sigma_tuples = model(images)
            
            total_loss = torch.tensor(0.0, device=DEVICE, requires_grad=True)
            for logits in logits_list:
                if logits.shape[2:] != targets.shape[1:]:
                    resized_targets = F.interpolate(targets.unsqueeze(1).float(), size=logits.shape[2:], mode='nearest').squeeze(1).long()
                else:
                    resized_targets = targets
                total_loss = total_loss + criterion(logits, resized_targets, b1_map_batch, all_eps_sigma_tuples)
            
            loss = total_loss / len(logits_list)
            loss.backward()
            optimizer.step()
            epoch_loss += loss.item()
            train_pbar.set_postfix({'loss': f'{loss.item():.4f}'})
        
        avg_loss = epoch_loss / len(train_dataloader)
        
        # Validation - DISABLE HD95 for speed!
        val_metrics = evaluate_metrics(model, val_dataloader, DEVICE, NUM_CLASSES, compute_hd95=False)
        torch.cuda.empty_cache()
        
        avg_fg_dice = np.mean(val_metrics['dice_scores'][1:])
        # No HD95 during val
        avg_fg_hd95 = 0.0
        
        scheduler.step(avg_fg_dice)
        
        print(f"Epoch {epoch+1}/{num_epochs} | Loss: {avg_loss:.4f} | Val Dice: {avg_fg_dice:.4f} | Val HD95: {avg_fg_hd95:.4f}")
        
        # Save best and early stopping
        if avg_fg_dice > best_dice:
            best_dice = avg_fg_dice
            best_hd95 = avg_fg_hd95
            torch.save(model.state_dict(), weights_path)
            epochs_no_improve = 0
            print(f"  *** New best! Saved to {weights_path}")
        else:
            epochs_no_improve += 1
            if epochs_no_improve >= EARLY_STOP_PATIENCE:
                print(f"Early stopping after {epoch+1} epochs")
                break
    
    # ==========================================================================
    # TEST SET EVALUATION
    # ==========================================================================
    
    print("\nEvaluating on test set...")
    model.load_state_dict(torch.load(weights_path))
    # Test set - Enable HD95 here for final result
    test_metrics = evaluate_metrics(model, test_dataloader, DEVICE, NUM_CLASSES, compute_hd95=True)
    
    test_fg_dice = np.mean(test_metrics['dice_scores'][1:])
    test_fg_hd95_vals = [h for h in test_metrics['hd95'][1:] if not np.isnan(h)]
    test_fg_hd95 = np.mean(test_fg_hd95_vals) if test_fg_hd95_vals else float('inf')
    
    # Peak GPU memory
    if torch.cuda.is_available():
        peak_gpu_mb = torch.cuda.max_memory_allocated() / (1024 * 1024)
    else:
        peak_gpu_mb = 0.0
    
    print(f"\n{'='*60}")
    print(f"TRAINING COMPLETE: {enc_config['name']}")
    print(f"{'='*60}")
    print(f"  Params: {total_params:,}")
    print(f"  Best Val Dice: {best_dice:.4f}")
    print(f"  Test Dice: {test_fg_dice:.4f}")
    print(f"  Test HD95: {test_fg_hd95:.4f}")
    print(f"  Peak GPU: {peak_gpu_mb:.0f}MB")
    
    return {
        "encoder": encoder_name,
        "name": enc_config["name"],
        "type": encoder_type,
        "params": total_params,
        "best_val_dice": best_dice,
        "best_val_hd95": best_hd95,
        "test_dice": test_fg_dice,
        "test_hd95": test_fg_hd95,
        "peak_gpu_mb": peak_gpu_mb
    }


# =============================================================================
# MAIN
# =============================================================================

if __name__ == "__main__":
    parser = argparse.ArgumentParser(description="Train encoder ablation")
    parser.add_argument("--encoder", type=str, required=True, 
                        choices=list(ENCODER_CONFIGS.keys()),
                        help="Encoder type to train")
    parser.add_argument("--epochs", type=int, default=None, help="Number of epochs")
    args = parser.parse_args()
    
    train_encoder(args.encoder, num_epochs=args.epochs)
