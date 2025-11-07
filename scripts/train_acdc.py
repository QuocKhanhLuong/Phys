import torch
import torch.nn as nn
import torch.nn.functional as F
import numpy as np
import os
import sys
import time
import gc
from pathlib import Path
from sklearn.model_selection import train_test_split
from torch.utils.data import DataLoader
from itertools import chain
from tqdm import tqdm

project_root = os.path.abspath(os.path.join(os.path.dirname(__file__), '..'))
sys.path.append(project_root)

from src.models.unet_archetype import RobustMedVFL_UNet, print_model_parameters
from src.modules.losses import CombinedLoss
from src.data_utils.data_utils import get_patient_ids_from_npy
from evaluate import evaluate_metrics, run_and_print_test_evaluation
from src import config

import kornia.augmentation as K
import argparse

NUM_EPOCHS = config.NUM_EPOCHS
NUM_CLASSES = 4  # ACDC: Background, Right Ventricle, Myocardium, Left Ventricle
LEARNING_RATE = config.LEARNING_RATE
DEVICE = torch.device('cuda' if torch.cuda.is_available() else 'cpu')
IMG_SIZE = config.IMG_SIZE
BATCH_SIZE = config.BATCH_SIZE
EARLY_STOP_PATIENCE = config.EARLY_STOP_PATIENCE


def parse_args():
    parser = argparse.ArgumentParser(description="Train ACDC script with optional resume from checkpoint")
    parser.add_argument('--checkpoint', type=str, default=None,
                        help='Path to a checkpoint file to load. Can be a full checkpoint (dict) or model state_dict.')
    parser.add_argument('--start-epoch', type=int, default=0,
                        help='Starting epoch if loading model-only weights (no epoch info in checkpoint).')
    return parser.parse_args()

def calculate_gflops(model, input_size=(1, 2, 224, 224), device='cuda'):
    """Calculate GFLOPs (Giga Floating Point Operations) for the model."""
    model.eval()
    input_tensor = torch.randn(input_size).to(device)
    
    total_ops = 0
    hooks = []
    
    def count_conv2d(m, x, y):
        nonlocal total_ops
        batch_size = y.shape[0]
        output_height, output_width = y.shape[2:]
        kernel_height, kernel_width = m.kernel_size
        in_channels = m.in_channels
        out_channels = m.out_channels
        groups = m.groups
        
        ops_per_position = kernel_height * kernel_width * (in_channels // groups)
        total_positions = batch_size * output_height * output_width * out_channels
        total_ops += ops_per_position * total_positions
    
    def count_linear(m, x, y):
        nonlocal total_ops
        total_ops += m.in_features * m.out_features * y.shape[0]
    
    def count_batchnorm(m, x, y):
        nonlocal total_ops
        total_ops += y.numel() * 2
    
    for m in model.modules():
        if isinstance(m, torch.nn.Conv2d):
            hooks.append(m.register_forward_hook(count_conv2d))
        elif isinstance(m, torch.nn.Linear):
            hooks.append(m.register_forward_hook(count_linear))
        elif isinstance(m, (torch.nn.BatchNorm2d, torch.nn.GroupNorm)):
            hooks.append(m.register_forward_hook(count_batchnorm))
    
    with torch.no_grad():
        model(input_tensor)
    
    for hook in hooks:
        hook.remove()
    
    gflops = total_ops / 1e9
    return gflops

class GpuAugmentation2D(nn.Module):
    """2D Augmentation cho ACDC dataset"""
    def __init__(self):
        super().__init__()

        # 2D Geometric augmentations cho images và masks
        self.geometric_aug_list = nn.Sequential(
            K.RandomHorizontalFlip(p=0.5),
            K.RandomVerticalFlip(p=0.5),
            K.RandomAffine(degrees=20, translate=(0.1, 0.1), p=0.7),
        )
        
        # Intensity augmentations chỉ cho images
        self.intensity_aug_list = nn.Sequential(
            K.RandomBrightness(brightness=(0.8, 1.2), p=0.5),
            K.RandomContrast(contrast=(0.8, 1.2), p=0.5),
        )

    @torch.no_grad()  
    def forward(self, x_2d, y_2d):
        # x_2d: (B, 2, H, W) - 2 channels (ED, ES)
        # y_2d: (B, H, W) - mask
        
        # Expand mask để match shape với geometric aug
        y_2d_expanded = y_2d.unsqueeze(1).float()  # (B, 1, H, W)
        
        # Apply geometric augmentations
        x_aug = self.geometric_aug_list(x_2d)
        y_aug = self.geometric_aug_list(y_2d_expanded)
        
        # Apply intensity augmentations chỉ cho images
        x_final = self.intensity_aug_list(x_aug)
        
        # Convert mask back to long
        y_final = y_aug.squeeze(1).long()
        
        return x_final, y_final

if __name__ == "__main__":
    print(f"Device: {DEVICE}")

    args = parse_args()

    npy_dir = str(config.ACDC_PREPROCESSED_DIR)
    print(f"Data directory: {npy_dir}")
    if not os.path.exists(npy_dir):
        raise FileNotFoundError(f"ACDC data not found at: {npy_dir}")
    
    patient_ids = get_patient_ids_from_npy(npy_dir)
    train_val_ids, test_ids = train_test_split(patient_ids, test_size=0.15, random_state=42)
    train_ids, val_ids = train_test_split(train_val_ids, test_size=0.176, random_state=42)
    print(f"Split: {len(train_ids)} train, {len(val_ids)} val, {len(test_ids)} test")
    print("-" * 60)

    from src.data_utils.acdc_dataset import build_acdc_dataset, CacheLocalitySampler, _worker_init
    
    samples_per_patient = config.MONAI_SAMPLES_PER_PATIENT
    train_dataset = build_acdc_dataset(npy_dir, train_ids, samples_per_patient)
    val_dataset = build_acdc_dataset(npy_dir, val_ids, samples_per_patient)
    test_dataset = build_acdc_dataset(npy_dir, test_ids, samples_per_patient)
    
    train_sampler = CacheLocalitySampler(train_dataset, BATCH_SIZE, shuffle=True)
    val_sampler = CacheLocalitySampler(val_dataset, BATCH_SIZE, shuffle=False)
    test_sampler = CacheLocalitySampler(test_dataset, BATCH_SIZE, shuffle=False)
    
    train_dataloader = DataLoader(train_dataset, batch_size=BATCH_SIZE, sampler=train_sampler, 
                                  num_workers=config.DATA_NUM_WORKERS, pin_memory=True, 
                                  persistent_workers=True, prefetch_factor=config.DATA_PREFETCH_FACTOR, 
                                  worker_init_fn=_worker_init)
    val_dataloader = DataLoader(val_dataset, batch_size=BATCH_SIZE, sampler=val_sampler, 
                               num_workers=config.DATA_NUM_WORKERS, pin_memory=True, 
                               persistent_workers=True, prefetch_factor=config.DATA_PREFETCH_FACTOR)
    test_dataloader = DataLoader(test_dataset, batch_size=BATCH_SIZE, sampler=test_sampler, 
                                num_workers=config.DATA_NUM_WORKERS, pin_memory=True, 
                                persistent_workers=True, prefetch_factor=config.DATA_PREFETCH_FACTOR)
    print(f"Training samples: {len(train_dataset)}, Validation: {len(val_dataset)}, Test: {len(test_dataset)}")
    print("-" * 60)

    gpu_augmenter = GpuAugmentation2D().to(DEVICE)

    print("Initializing model...")
    # ACDC: 2 input channels (ED, ES)
    model = RobustMedVFL_UNet(n_channels=2, n_classes=NUM_CLASSES).to(DEVICE)
    criterion = CombinedLoss(
        num_classes=NUM_CLASSES,
        use_physics=True,
        fixed_weights=False,
        initial_loss_weights=[0.2, 0.75, 0.05]
    ).to(DEVICE)
    optimizer = torch.optim.AdamW(model.parameters(), lr=LEARNING_RATE)
    scheduler = torch.optim.lr_scheduler.ReduceLROnPlateau(optimizer, mode='max', factor=0.5, patience=10)
    
    print_model_parameters(model)
    gflops = calculate_gflops(model, input_size=(1, 2, IMG_SIZE, IMG_SIZE), device=str(DEVICE))
    print(f"Model GFLOPs: {gflops:.2f} G")
    print("All components initialized. GPU 2D augmentation is ENABLED.")
    print("-" * 60)
    
    best_val_metric = 0.0
    epochs_no_improve = 0
    total_train_start = time.time()

    # Resume / checkpoint loading support
    start_epoch = 0
    if args.checkpoint is not None and os.path.exists(args.checkpoint):
        print(f"Found checkpoint at {args.checkpoint}, attempting to load...")
        ckpt = torch.load(args.checkpoint, map_location=DEVICE)
        if isinstance(ckpt, dict) and ('model_state' in ckpt or 'model' in ckpt or 'epoch' in ckpt):
            model_state = ckpt.get('model_state', ckpt.get('model', None))
            if model_state is not None:
                model.load_state_dict(model_state)
            else:
                try:
                    model.load_state_dict(ckpt)
                except Exception:
                    print('Warning: checkpoint dict did not contain recognizable model state; skipping model load')

            if 'optimizer_state' in ckpt and ckpt['optimizer_state'] is not None:
                try:
                    optimizer.load_state_dict(ckpt['optimizer_state'])
                except Exception:
                    print('Warning: failed to load optimizer state (incompatible)')

            if 'scheduler_state' in ckpt and ckpt['scheduler_state'] is not None:
                try:
                    scheduler.load_state_dict(ckpt['scheduler_state'])
                except Exception:
                    print('Warning: failed to load scheduler state (incompatible)')

            start_epoch = ckpt.get('epoch', 0) + 1
            best_val_metric = ckpt.get('best_val_metric', best_val_metric)
            print(f"Resuming from epoch {start_epoch} (best_val_metric={best_val_metric:.4f})")
        else:
            try:
                model.load_state_dict(ckpt)
                start_epoch = args.start_epoch
                print(f"Loaded model weights only. Starting from epoch {start_epoch} (use --start-epoch to set)")
            except Exception:
                print('Warning: failed to load checkpoint as model state_dict. Skipping checkpoint load.')

    for epoch in range(start_epoch, NUM_EPOCHS):
        epoch_start_time = time.time()
        print(f"\n--- Epoch {epoch + 1}/{NUM_EPOCHS} ---")
        
        model.train()
        epoch_train_loss = 0.0
        
        data_load_times, gpu_transfer_times, forward_times, backward_times, total_iter_times = [], [], [], [], []
        
        train_pbar = tqdm(train_dataloader, desc="Training", ncols=120)
        
        iter_start = time.time()
        for batch_idx, (images, targets) in enumerate(train_pbar):
            data_load_time = time.time() - iter_start
            data_load_times.append(data_load_time)
            
            transfer_start = time.time()
            images, targets = images.to(DEVICE), targets.to(DEVICE)
            images, targets = gpu_augmenter(images, targets)
            gpu_transfer_time = time.time() - transfer_start
            gpu_transfer_times.append(gpu_transfer_time)
            
            optimizer.zero_grad()
            
            forward_start = time.time()
            logits_list, _ = model(images)
            loss = torch.stack([criterion(logits, targets) for logits in logits_list]).mean()
            forward_time = time.time() - forward_start
            forward_times.append(forward_time)
            
            backward_start = time.time()
            loss.backward()
            torch.nn.utils.clip_grad_norm_(model.parameters(), max_norm=1.0)
            optimizer.step()
            backward_time = time.time() - backward_start
            backward_times.append(backward_time)
            
            epoch_train_loss += loss.item()
            total_iter_time = time.time() - iter_start
            total_iter_times.append(total_iter_time)
            
            if batch_idx % 50 == 0 and batch_idx > 0:
                train_pbar.set_postfix({
                    'loss': f'{loss.item():.4f}',
                    'data_load': f'{np.mean(data_load_times[-50:]):.3f}s',
                    'gpu_aug': f'{np.mean(gpu_transfer_times[-50:]):.3f}s',
                    'forward': f'{np.mean(forward_times[-50:]):.3f}s',
                    'backward': f'{np.mean(backward_times[-50:]):.3f}s'
                })
            else:
                train_pbar.set_postfix({'loss': f'{loss.item():.4f}'})

            iter_start = time.time()

        if data_load_times:
            print(f"\n   --- Profiling Summary (Epoch Avg) ---")
            print(f"   - Data Loading:    {np.mean(data_load_times):.3f}s")
            print(f"   - GPU Transfer+Aug:  {np.mean(gpu_transfer_times):.3f}s")
            print(f"   - Forward Pass:    {np.mean(forward_times):.3f}s")
            print(f"   - Backward Pass:   {np.mean(backward_times):.3f}s")
            print(f"   - Total/Iteration: {np.mean(total_iter_times):.3f}s")
        
        avg_train_loss = epoch_train_loss / len(train_dataloader)
        epoch_time = time.time() - epoch_start_time
        print(f"\n   === Epoch {epoch+1} Summary ===")
        print(f"   Training Loss: {avg_train_loss:.4f} | Time: {epoch_time/60:.2f} min")

        # Validation
        if len(val_dataloader) > 0:
            val_metrics = evaluate_metrics(model, val_dataloader, DEVICE, NUM_CLASSES)
            
            avg_fg_dice = np.mean(val_metrics['dice_scores'][1:])
            print("   --- Validation Per-Class Dice ---")
            for c_idx in range(1, NUM_CLASSES):
                class_name = config.ACDC_CONFIG["class_map"][c_idx]
                print(f"   - {class_name}: {val_metrics['dice_scores'][c_idx]:.4f}")

            print(f"   - Avg Foreground Dice: {avg_fg_dice:.4f}")
            
            scheduler.step(avg_fg_dice)
            if avg_fg_dice > best_val_metric:
                best_val_metric = avg_fg_dice
                ckpt_to_save = {
                    'epoch': epoch,
                    'model_state': model.state_dict(),
                    'optimizer_state': optimizer.state_dict(),
                    'scheduler_state': getattr(scheduler, 'state_dict', lambda: None)(),
                    'best_val_metric': best_val_metric
                }
                torch.save(ckpt_to_save, "best_checkpoint_acdc.pth")
                torch.save(model.state_dict(), "best_model_acdc.pth")
                print(f"   >>> New best model saved with Dice: {best_val_metric:.4f} <<<")
                epochs_no_improve = 0
            else:
                epochs_no_improve += 1

        if epochs_no_improve >= EARLY_STOP_PATIENCE:
            print(f"\nEarly stopping after {EARLY_STOP_PATIENCE} epochs with no improvement.")
            break
        
        gc.collect()
        torch.cuda.empty_cache()

    total_train_time = time.time() - total_train_start
    print("\n" + "=" * 60)
    print(f"--- Training Finished ---")
    print(f"Total Training Time: {total_train_time/3600:.2f} hours")
    print(f"Best Validation Dice Score: {best_val_metric:.4f}")
    print("=" * 60 + "\n")
    
    run_and_print_test_evaluation(
        test_dataloader=test_dataloader,
        device=DEVICE,
        num_classes=NUM_CLASSES,
        num_slices=2  # ACDC: 2 channels
    )

