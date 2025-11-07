import torch
import torch.nn as nn
import torch.nn.functional as F
import numpy as np
import os
import time
import gc
from sklearn.model_selection import train_test_split
from torch.utils.data import DataLoader
from itertools import chain
from tqdm import tqdm

# Import các thành phần cần thiết từ project của bạn
from models import RobustMedVFL_UNet, print_model_parameters
from losses import CombinedLoss
from data_utils import get_patient_ids_from_npy, load_brats21_volumes
from evaluate import evaluate_metrics, run_and_print_test_evaluation, visualize_final_results_2_5D
import config

import kornia.augmentation as K
import kornia.geometry.transform as K_tf

NUM_EPOCHS = config.NUM_EPOCHS
NUM_CLASSES = config.NUM_CLASSES
LEARNING_RATE = config.LEARNING_RATE
DEVICE = torch.device('cuda' if torch.cuda.is_available() else 'cpu')
IMG_SIZE = config.IMG_SIZE
BATCH_SIZE = config.BATCH_SIZE
NUM_SLICES = config.NUM_SLICES
EARLY_STOP_PATIENCE = config.EARLY_STOP_PATIENCE

def calculate_gflops(model, input_size=(1, 20, 224, 224), device='cuda'):
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

class GpuAugmentation(nn.Module):
    def __init__(self, num_slices_2_5d: int):
        super().__init__()
        self.num_slices = num_slices_2_5d

        # Separate augmentations for images and masks
        self.image_aug_list = nn.Sequential(
            K.RandomHorizontalFlip3D(p=0.5),
            K.RandomVerticalFlip3D(p=0.5),
            K.RandomDepthicalFlip3D(p=0.3),
            K.RandomAffine3D(degrees=(20, 20, 20), translate=(0.1, 0.1, 0.1), p=0.7),
        )
        
        self.mask_aug_list = nn.Sequential(
            K.RandomHorizontalFlip3D(p=0.5),
            K.RandomVerticalFlip3D(p=0.5),
            K.RandomDepthicalFlip3D(p=0.3),
            K.RandomAffine3D(degrees=(20, 20, 20), translate=(0.1, 0.1, 0.1), p=0.7),
        )
        
        self.intensity_aug_list = nn.Sequential(
            K.RandomBrightness(brightness=(0.8, 1.2), p=0.5),
            K.RandomContrast(contrast=(0.8, 1.2), p=0.5),
        )

    @torch.no_grad()  
    def forward(self, x_2_5d, y_2d):
        # 1. Reshape 3D
        b, _, h, w = x_2_5d.shape
        x_3d = x_2_5d.view(b, 4, self.num_slices, h, w)
        y_3d = y_2d.view(b, 1, 1, h, w).expand(-1, -1, self.num_slices, -1, -1).float()

        # 2. Apply 3D augmentations separately
        x_aug_3d = self.image_aug_list(x_3d)
        y_aug_3d = self.mask_aug_list(y_3d)

        # 3. Reshape 2.5D
        x_aug_2_5d = x_aug_3d.view(b, -1, h, w)
        
        # 4. 2D Intensity augment
        x_final_aug_2_5d = self.intensity_aug_list(x_aug_2_5d)
        
        # 5. center mask
        center_slice_idx = self.num_slices // 2
        y_aug_2d = y_aug_3d[:, 0, center_slice_idx, :, :].long()
        
        return x_final_aug_2_5d, y_aug_2d

if __name__ == "__main__":
    print(f"Device: {DEVICE}")

    npy_dir = config.MONAI_NPY_DIR
    print(f"Data directory: {npy_dir}")
    if not os.path.exists(npy_dir):
        raise FileNotFoundError(f"MONAI data not found at: {npy_dir}")
    
    patient_ids = get_patient_ids_from_npy(npy_dir)
    train_val_ids, test_ids = train_test_split(patient_ids, test_size=0.15, random_state=42)
    train_ids, val_ids = train_test_split(train_val_ids, test_size=0.176, random_state=42)
    print(f"Split: {len(train_ids)} train, {len(val_ids)} val, {len(test_ids)} test")
    print("-" * 60)

    from monai_dataset import build_monai_persistent_dataset, CacheLocalitySampler, _worker_init
    
    samples_per_patient = config.MONAI_SAMPLES_PER_PATIENT
    train_dataset = build_monai_persistent_dataset(npy_dir, train_ids, NUM_SLICES, samples_per_patient)
    val_dataset = build_monai_persistent_dataset(npy_dir, val_ids, NUM_SLICES, samples_per_patient)
    test_dataset = build_monai_persistent_dataset(npy_dir, test_ids, NUM_SLICES, samples_per_patient)
    
    train_sampler = CacheLocalitySampler(train_dataset, BATCH_SIZE, shuffle=True)
    val_sampler = CacheLocalitySampler(val_dataset, BATCH_SIZE, shuffle=False)
    test_sampler = CacheLocalitySampler(test_dataset, BATCH_SIZE, shuffle=False)
    
    train_dataloader = DataLoader(train_dataset, batch_size=BATCH_SIZE, sampler=train_sampler, num_workers=config.DATA_NUM_WORKERS, pin_memory=True, persistent_workers=True, prefetch_factor=config.DATA_PREFETCH_FACTOR, worker_init_fn=_worker_init)
    val_dataloader = DataLoader(val_dataset, batch_size=BATCH_SIZE, sampler=val_sampler, num_workers=config.DATA_NUM_WORKERS, pin_memory=True, persistent_workers=True, prefetch_factor=config.DATA_PREFETCH_FACTOR)
    test_dataloader = DataLoader(test_dataset, batch_size=BATCH_SIZE, sampler=test_sampler, num_workers=config.DATA_NUM_WORKERS, pin_memory=True, persistent_workers=True, prefetch_factor=config.DATA_PREFETCH_FACTOR)
    print(f"Training samples: {len(train_dataset)}, Validation: {len(val_dataset)}, Test: {len(test_dataset)}")
    print("-" * 60)

    gpu_augmenter = GpuAugmentation(NUM_SLICES).to(DEVICE)

    print("Initializing model...")
    model = RobustMedVFL_UNet(n_channels=NUM_SLICES * 4, n_classes=NUM_CLASSES).to(DEVICE)
    criterion = CombinedLoss(num_classes=NUM_CLASSES, use_physics=False, fixed_weights=True).to(DEVICE) # Tạm thời tắt physics loss
    optimizer = torch.optim.AdamW(model.parameters(), lr=LEARNING_RATE)
    scheduler = torch.optim.lr_scheduler.ReduceLROnPlateau(optimizer, mode='max', factor=0.5, patience=10)
    
    print_model_parameters(model)
    gflops = calculate_gflops(model, input_size=(1, NUM_SLICES * 4, IMG_SIZE, IMG_SIZE), device=str(DEVICE))
    print(f"Model GFLOPs: {gflops:.2f} G")
    print("All components initialized. GPU 3D augmentation is ENABLED.")
    print("-" * 60)
    
    best_val_metric = 0.0
    epochs_no_improve = 0
    total_train_start = time.time()

    for epoch in range(NUM_EPOCHS):
        epoch_start_time = time.time()
        print(f"\n--- Epoch {epoch + 1}/{NUM_EPOCHS} ---")
        
        model.train()
        epoch_train_loss = 0.0
        
        data_load_times, gpu_transfer_times, forward_times, backward_times, total_iter_times = [], [], [], [], []
        
        train_pbar = tqdm(train_dataloader, desc="Training", ncols=120) # Mở rộng thanh progress
        
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

            iter_start = time.time() # Reset timer cho vòng lặp tiếp theo

        # In profiling summary cuối mỗi epoch
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

        # --- Đánh giá trên tập validation ---
        if len(val_dataloader) > 0:
            val_metrics = evaluate_metrics(model, val_dataloader, DEVICE, NUM_CLASSES)
            
            if 'brats_regions' in val_metrics:
                 avg_fg_dice = np.mean(list(val_metrics['brats_regions'].values()))
                 print("   --- Validation BraTS Regions ---")
                 print(f"   - ET: {val_metrics['brats_regions']['ET']:.4f}, TC: {val_metrics['brats_regions']['TC']:.4f}, WT: {val_metrics['brats_regions']['WT']:.4f}")
            else:
                 avg_fg_dice = np.mean(val_metrics['dice_scores'][1:])
                 print("   --- Validation Per-Class Dice ---")
                 for c_idx in range(1, NUM_CLASSES):
                     print(f"   - Class {c_idx}: {val_metrics['dice_scores'][c_idx]:.4f}")

            print(f"   - Avg Foreground Dice: {avg_fg_dice:.4f}")
            
            scheduler.step(avg_fg_dice)
            if avg_fg_dice > best_val_metric:
                best_val_metric = avg_fg_dice
                torch.save(model.state_dict(), "best_model.pth")
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
        num_slices=NUM_SLICES * 4
    )
    
    print("\nLoading test volumes for visualization...")
    test_vols_viz, test_masks_viz = load_brats21_volumes(config.BRATS_RAW_DIR, target_size=(IMG_SIZE, IMG_SIZE), max_patients=10)
    for i in range(len(test_vols_viz)):
        for mod_idx in range(4):
            max_val = np.max(test_vols_viz[i][mod_idx])
            if max_val > 0:
                test_vols_viz[i][mod_idx] /= max_val

    visualize_final_results_2_5D(
        volumes_np=test_vols_viz,
        masks_np=test_masks_viz,
        num_classes=NUM_CLASSES,
        num_samples=min(10, len(test_vols_viz)),
        device=DEVICE,
        num_slices=NUM_SLICES * 4
    )