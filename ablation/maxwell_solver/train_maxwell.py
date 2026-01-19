"""
Maxwell Solver Ablation Training Script

Trains PIE-UNet with/without Maxwell Solver.

Usage:
    python ablation/maxwell_solver/train_maxwell.py --variant Standard
    python ablation/maxwell_solver/train_maxwell.py --variant Physics
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
import warnings

warnings.filterwarnings("ignore")

PROJECT_ROOT = Path(__file__).parent.parent.parent
sys.path.insert(0, str(PROJECT_ROOT))

from ablation.maxwell_solver.config import MAXWELL_CONFIGS, TRAINING_CONFIG, DATA_CONFIG, OUTPUT_CONFIG
from ablation.maxwell_solver.pie_unet_maxwell import PIE_UNet_Maxwell
from src.models.epure import ePURE
from src.modules.losses import CombinedLoss
from src.data_utils.acdc_dataset_optimized import ACDCDataset25DOptimized, get_acdc_volume_ids
from src.utils.helpers import calculate_ultimate_common_b1_map


# Augmentation
train_transform = A.Compose([
    A.Rotate(limit=20, p=0.7),
    A.HorizontalFlip(p=0.5),
    A.ElasticTransform(p=0.5, alpha=120, sigma=120 * 0.05),
    A.Affine(scale=(0.9, 1.1), translate_percent=(-0.0625, 0.0625), rotate=(-15, 15), p=0.7, border_mode=cv2.BORDER_CONSTANT),
    A.RandomBrightnessContrast(p=0.5),
    ToTensorV2(),
])
val_test_transform = A.Compose([ToTensorV2()])


def train_maxwell_variant(variant_name, num_epochs=None):
    """Train a Maxwell Solver ablation variant."""
    
    if variant_name not in MAXWELL_CONFIGS:
        raise ValueError(f"Unknown variant: {variant_name}")
    
    config = MAXWELL_CONFIGS[variant_name]
    
    DEVICE = torch.device('cuda' if torch.cuda.is_available() else 'cpu')
    NUM_CLASSES = TRAINING_CONFIG["num_classes"]
    NUM_SLICES = TRAINING_CONFIG["num_slices"]
    BATCH_SIZE = TRAINING_CONFIG["batch_size"]
    LEARNING_RATE = TRAINING_CONFIG["learning_rate"]
    EARLY_STOP_PATIENCE = TRAINING_CONFIG["early_stopping_patience"]
    
    if num_epochs is None:
        num_epochs = TRAINING_CONFIG["num_epochs"]
    
    print("=" * 70)
    print(f"MAXWELL SOLVER ABLATION: {config['name']}")
    print(f"  use_maxwell: {config['use_maxwell']}")
    print("=" * 70)
    
    # Data loading
    train_npy_dir = str(DATA_CONFIG["train_dir"])
    test_npy_dir = str(DATA_CONFIG["test_dir"])
    
    all_train_ids = get_acdc_volume_ids(train_npy_dir)
    indices = list(range(len(all_train_ids)))
    train_idx, val_idx = train_test_split(indices, test_size=0.2, random_state=42)
    train_ids = [all_train_ids[i] for i in train_idx]
    val_ids = [all_train_ids[i] for i in val_idx]
    test_ids = get_acdc_volume_ids(test_npy_dir)
    
    ePURE_augmenter = ePURE(in_channels=NUM_SLICES).to(DEVICE)
    ePURE_augmenter.eval()
    
    train_dataset = ACDCDataset25DOptimized(train_npy_dir, train_ids, NUM_SLICES, train_transform, 
                                             noise_injector_model=ePURE_augmenter, device=str(DEVICE), 
                                             max_cache_size=15, use_memmap=True)
    val_dataset = ACDCDataset25DOptimized(train_npy_dir, val_ids, NUM_SLICES, val_test_transform, 
                                           max_cache_size=10, use_memmap=True)
    
    train_loader = DataLoader(train_dataset, BATCH_SIZE, shuffle=True, num_workers=0, pin_memory=True)
    val_loader = DataLoader(val_dataset, BATCH_SIZE, shuffle=False, num_workers=0, pin_memory=True)
    
    # B1 map
    def convert_npy_to_tensor(npy_dir, vol_ids):
        volumes_dir = os.path.join(npy_dir, 'volumes')
        slices = []
        for vid in vol_ids:
            vol = np.load(os.path.join(volumes_dir, f'{vid}.npy'))
            for i in range(vol.shape[2]):
                slices.append(torch.from_numpy(vol[:, :, i]).unsqueeze(0))
        return torch.stack(slices, dim=0).float()
    
    all_tensor = torch.cat([convert_npy_to_tensor(train_npy_dir, train_ids + val_ids),
                           convert_npy_to_tensor(test_npy_dir, test_ids)], dim=0)
    common_b1_map = calculate_ultimate_common_b1_map(all_tensor, str(DEVICE), 
                                                      "b1_maps/acdc_cardiac_ultimate_common_b1_map.pth")
    
    # Model
    model = PIE_UNet_Maxwell(n_channels=NUM_SLICES, n_classes=NUM_CLASSES, 
                             use_maxwell=config['use_maxwell'], deep_supervision=True).to(DEVICE)
    params = sum(p.numel() for p in model.parameters())
    print(f"Params: {params:,}")
    
    criterion = CombinedLoss(num_classes=NUM_CLASSES, initial_loss_weights=[0.4, 0.4, 0.2], 
                             class_indices_for_rules=None).to(DEVICE)
    optimizer = torch.optim.AdamW(chain(model.parameters(), criterion.parameters()), lr=LEARNING_RATE)
    scheduler = torch.optim.lr_scheduler.ReduceLROnPlateau(optimizer, mode='max', factor=0.5, patience=10)
    
    # Training
    best_dice = 0.0
    epochs_no_improve = 0
    weights_path = OUTPUT_CONFIG["weights_dir"] / f"best_{variant_name}.pth"
    
    for epoch in range(num_epochs):
        model.train()
        epoch_loss = 0.0
        
        for images, targets in tqdm(train_loader, desc=f"Epoch {epoch+1}", leave=False):
            images, targets = images.to(DEVICE), targets.to(DEVICE)
            optimizer.zero_grad()
            
            b1_batch = common_b1_map.expand(images.size(0), -1, -1, -1)
            logits_list, eps_sigma = model(images)
            
            total_loss = torch.tensor(0.0, device=DEVICE, requires_grad=True)
            for logits in logits_list:
                if logits.shape[2:] != targets.shape[1:]:
                    resized = F.interpolate(targets.unsqueeze(1).float(), logits.shape[2:], mode='nearest').squeeze(1).long()
                else:
                    resized = targets
                total_loss = total_loss + criterion(logits, resized, b1_batch, eps_sigma)
            
            loss = total_loss / len(logits_list)
            loss.backward()
            optimizer.step()
            epoch_loss += loss.item()
        
        # Validation
        model.eval()
        dice_sum = 0.0
        batches = 0
        with torch.no_grad():
            for imgs, tgts in val_loader:
                imgs, tgts = imgs.to(DEVICE), tgts.to(DEVICE)
                logits_list, _ = model(imgs)
                preds = torch.argmax(F.softmax(logits_list[-1], dim=1), dim=1)
                batches += 1
                for c in range(1, NUM_CLASSES):
                    pc = (preds == c).float().view(-1)
                    tc = (tgts == c).float().view(-1)
                    inter = (pc * tc).sum()
                    dice_sum += ((2. * inter + 1e-6) / (pc.sum() + tc.sum() + 1e-6)).item()
        
        avg_dice = dice_sum / (batches * (NUM_CLASSES - 1)) if batches > 0 else 0.0
        scheduler.step(avg_dice)
        
        print(f"Epoch {epoch+1}/{num_epochs} | Loss: {epoch_loss/len(train_loader):.4f} | Val Dice: {avg_dice:.4f}")
        
        if avg_dice > best_dice:
            best_dice = avg_dice
            torch.save(model.state_dict(), weights_path)
            epochs_no_improve = 0
        else:
            epochs_no_improve += 1
            if epochs_no_improve >= EARLY_STOP_PATIENCE:
                print(f"Early stopping at epoch {epoch+1}")
                break
    
    print(f"\nTraining complete! Best Dice: {best_dice:.4f}")
    return {"variant": variant_name, "params": params, "best_dice": best_dice}


if __name__ == "__main__":
    parser = argparse.ArgumentParser()
    parser.add_argument("--variant", type=str, required=True, choices=list(MAXWELL_CONFIGS.keys()))
    parser.add_argument("--epochs", type=int, default=None)
    args = parser.parse_args()
    
    train_maxwell_variant(args.variant, args.epochs)
