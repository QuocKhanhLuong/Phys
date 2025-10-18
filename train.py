import torch
import torch.nn.functional as F
import numpy as np
import os
import time
import gc
from sklearn.model_selection import train_test_split
from torch.utils.data import DataLoader
import albumentations as A
from albumentations.pytorch import ToTensorV2
from itertools import chain
import cv2
import torch.multiprocessing as mp
from tqdm import tqdm

from models import ePURE, RobustMedVFL_UNet, print_model_parameters
from losses import CombinedLoss
from data_utils import BraTS21Dataset25D, load_brats21_volumes, get_brats21_patient_paths, get_patient_ids_from_npy
from typing import Optional
from evaluate import evaluate_metrics, run_and_print_test_evaluation, visualize_final_results_2_5D
from utils import calculate_ultimate_common_b1_map
import config


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


if __name__ == "__main__":
    print(f"Device: {DEVICE}")
    
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

    use_monai = getattr(config, 'USE_MONAI_PIPELINE', False)
    
    if use_monai:
        npy_dir = getattr(config, 'MONAI_NPY_DIR', config.NPY_DIR)
        print(f"MONAI Pipeline Selected")
        print(f"Data directory: {npy_dir}")
        if not os.path.exists(npy_dir):
            print(f"ERROR: MONAI preprocessed data not found: {npy_dir}")
            print(f"Run: python monai_preprocess.py --input_dir {config.BRATS_RAW_DIR} --output_dir {npy_dir}")
            raise FileNotFoundError(f"MONAI data not found: {npy_dir}")
    else:
        npy_dir = config.NPY_DIR
        print(f"Original Pipeline Selected")
        print(f"Data directory: {npy_dir}")
        if not os.path.exists(npy_dir):
            print(f"ERROR: NPY directory not found: {npy_dir}")
            print("Run: python preprocess.py")
            raise FileNotFoundError(f"NPY directory not found: {npy_dir}")
    
    patient_ids = get_patient_ids_from_npy(npy_dir)
    print(f"Found {len(patient_ids)} preprocessed patients")
    
    from sklearn.model_selection import train_test_split as split
    train_val_ids, test_ids = split(patient_ids, test_size=0.15, random_state=42)
    train_ids, val_ids = split(train_val_ids, test_size=0.176, random_state=42)
    
    print(f"Split: {len(train_ids)} train, {len(val_ids)} val, {len(test_ids)} test")
    
    ePURE_augmenter = ePURE(in_channels=NUM_SLICES * 4).to(DEVICE)
    ePURE_augmenter.eval()
    
    if getattr(config, 'USE_MONAI_PIPELINE', False):
        from monai_dataset import build_monai_persistent_dataset, CacheLocalitySampler, _worker_init

        print("=" * 60)
        print("USING MONAI PIPELINE (Z-score normalized + 2.5D + ULTRA-OPTIMIZED)")
        print("=" * 60)
        
        monai_npy_dir = getattr(config, 'MONAI_NPY_DIR', npy_dir)
        if not os.path.exists(monai_npy_dir):
            raise FileNotFoundError(
                f"MONAI preprocessed data not found at: {monai_npy_dir}\n"
                f"Please run: python monai_preprocess.py --input_dir {config.BRATS_RAW_DIR} "
                f"--output_dir {monai_npy_dir}"
            )
        
        print(f"Loading from: {monai_npy_dir}")
        
        samples_per_patient = getattr(config, 'MONAI_SAMPLES_PER_PATIENT', 10)
        if samples_per_patient is None or samples_per_patient <= 0:
            print(f"Sampling mode: FULL DATASET (all valid slices per patient)")
        else:
            print(f"Sampling mode: Random {samples_per_patient} slices per patient")
        
        train_dataset = build_monai_persistent_dataset(
            npy_dir=monai_npy_dir,
            patient_ids=train_ids,
            num_slices_25d=NUM_SLICES,
            samples_per_patient=samples_per_patient,
            transforms=train_transform  # Apply augmentation for training
        )
        val_dataset = build_monai_persistent_dataset(
            npy_dir=monai_npy_dir,
            patient_ids=val_ids,
            num_slices_25d=NUM_SLICES,
            samples_per_patient=samples_per_patient,
            transforms=val_test_transform  # No augmentation, only ToTensorV2
        )
        test_dataset = build_monai_persistent_dataset(
            npy_dir=monai_npy_dir,
            patient_ids=test_ids,
            num_slices_25d=NUM_SLICES,
            samples_per_patient=samples_per_patient,
            transforms=val_test_transform  # No augmentation, only ToTensorV2
        )

        # Create cache-aware samplers for better locality
        train_sampler = CacheLocalitySampler(train_dataset, BATCH_SIZE, shuffle=True)
        val_sampler = CacheLocalitySampler(val_dataset, BATCH_SIZE, shuffle=False)
        test_sampler = CacheLocalitySampler(test_dataset, BATCH_SIZE, shuffle=False)

        NUM_WORKERS = getattr(config, 'DATA_NUM_WORKERS', min(4, os.cpu_count() or 2))
        PREFETCH_FACTOR = getattr(config, 'DATA_PREFETCH_FACTOR', 2)
        print(f"Using {NUM_WORKERS} workers, prefetch_factor={PREFETCH_FACTOR} (optimized for mmap)")
        
        # Use sampler instead of shuffle
        train_dataloader = DataLoader(train_dataset, batch_size=BATCH_SIZE, sampler=train_sampler, num_workers=NUM_WORKERS, pin_memory=True, persistent_workers=True if NUM_WORKERS > 0 else False, prefetch_factor=PREFETCH_FACTOR if NUM_WORKERS > 0 else None, worker_init_fn=_worker_init)
        val_dataloader = DataLoader(val_dataset, batch_size=BATCH_SIZE, sampler=val_sampler, num_workers=NUM_WORKERS, pin_memory=True, persistent_workers=True if NUM_WORKERS > 0 else False, prefetch_factor=PREFETCH_FACTOR if NUM_WORKERS > 0 else None)
        test_dataloader = DataLoader(test_dataset, batch_size=BATCH_SIZE, sampler=test_sampler, num_workers=NUM_WORKERS, pin_memory=True, persistent_workers=True if NUM_WORKERS > 0 else False, prefetch_factor=PREFETCH_FACTOR if NUM_WORKERS > 0 else None)
        
        print(f"MONAI datasets created with optimizations:")
        print(f"  - Train: {len(train_ids)} patients, {len(train_dataset)} slices")
        print(f"  - Val:   {len(val_ids)} patients, {len(val_dataset)} slices")
        print(f"  - Test:  {len(test_ids)} patients, {len(test_dataset)} slices")
        print(f"  - Workers: {NUM_WORKERS}")
        print(f"  - Smart cache-locality sampler: ENABLED")
        print(f"  - Memory-mapped I/O: ENABLED")
        print("=" * 60)
    else:
        print("=" * 60)
        print("USING OPTIMIZED PIPELINE (BraTS21Dataset25D + LRU Cache + Memmap)")
        print("=" * 60)
        
        # Optimized dataset with LRU cache and memmap
        train_dataset = BraTS21Dataset25D(
            npy_dir=npy_dir,
            patient_ids=train_ids,
            num_input_slices=NUM_SLICES,
            transforms=train_transform,
            max_cache_size=15,  # Cache up to 15 volumes
            use_memmap=True     # Use memory-mapped files
        )
        val_dataset = BraTS21Dataset25D(
            npy_dir=npy_dir,
            patient_ids=val_ids,
            num_input_slices=NUM_SLICES,
            transforms=val_test_transform,
            max_cache_size=10,
            use_memmap=True
        )
        test_dataset = BraTS21Dataset25D(
            npy_dir=npy_dir,
            patient_ids=test_ids,
            num_input_slices=NUM_SLICES,
            transforms=val_test_transform,
            max_cache_size=10,
            use_memmap=True
        )

        NUM_WORKERS = min(4, os.cpu_count() or 2)
        print(f"✓ Datasets created with optimizations:")
        print(f"  - LRU Cache: 15 volumes (train), 10 (val/test)")
        print(f"  - Memmap: Enabled")
        print(f"  - Workers: {NUM_WORKERS}")
        print(f"  - Prefetch: 2 batches")
        print("=" * 60)
        
        train_dataloader = DataLoader(
            train_dataset, 
            batch_size=BATCH_SIZE, 
            shuffle=True, 
            num_workers=NUM_WORKERS, 
            pin_memory=True,
            persistent_workers=True if NUM_WORKERS > 0 else False,
            prefetch_factor=2 if NUM_WORKERS > 0 else None
        )
        val_dataloader = DataLoader(
            val_dataset, 
            batch_size=BATCH_SIZE, 
            shuffle=False, 
            num_workers=NUM_WORKERS, 
            pin_memory=True,
            persistent_workers=True if NUM_WORKERS > 0 else False,
            prefetch_factor=2 if NUM_WORKERS > 0 else None
        )
        test_dataloader = DataLoader(
            test_dataset, 
            batch_size=BATCH_SIZE, 
            shuffle=False, 
            num_workers=NUM_WORKERS, 
            pin_memory=True,
            persistent_workers=True if NUM_WORKERS > 0 else False,
            prefetch_factor=2 if NUM_WORKERS > 0 else None
        )
    
    print(f"\nTraining samples: {len(train_dataset)}, Validation: {len(val_dataset)}, Test: {len(test_dataset)}")
    print("-" * 60)

    dataset_name = "brats21"
    b1_map_path = f"{dataset_name}_ultimate_common_b1_map.pth"
    
    if os.path.exists(b1_map_path):
        print(f"\nB1 map found at {b1_map_path}, loading...")
        b1_map_data = torch.load(b1_map_path, map_location=DEVICE)
        if isinstance(b1_map_data, dict):
            common_b1_map = b1_map_data.get('b1_map', b1_map_data.get('common_b1_map', None))
            if common_b1_map is None:
                raise ValueError("B1 map dict doesn't contain expected keys")
        else:
            common_b1_map = b1_map_data
        print(f"B1 map loaded successfully, shape: {common_b1_map.shape}")
    else:
        print(f"\nB1 map not found, calculating from scratch...")
        print("Loading subset of volumes for B1 map calculation...")
        brats_raw_dir = os.path.join(config.PROJECT_ROOT, 'BraTS21')
        subset_volumes, _ = load_brats21_volumes(brats_raw_dir, target_size=(IMG_SIZE, IMG_SIZE), max_patients=200)
        
        for i in range(len(subset_volumes)):
            for mod_idx in range(4):
                max_val = np.max(subset_volumes[i][mod_idx])
                if max_val > 0:
                    subset_volumes[i][mod_idx] /= max_val
        
        def convert_volumes_to_tensor(volumes_list):
            all_slices = []
            for vol in volumes_list:
                for i in range(vol.shape[3]):
                    all_slices.append(torch.from_numpy(vol[:, :, :, i]).mean(dim=0, keepdim=True))
            return torch.stack(all_slices, dim=0).float()
        
        all_images_tensor = convert_volumes_to_tensor(subset_volumes)
        common_b1_map = calculate_ultimate_common_b1_map(
            all_images=all_images_tensor,
            device=str(DEVICE),
            save_path=b1_map_path
        )
        
        del subset_volumes
        gc.collect()
        print(f"B1 map calculated and saved to {b1_map_path}")
    
    print("Initializing model...")
    model = RobustMedVFL_UNet(n_channels=NUM_SLICES * 4, n_classes=NUM_CLASSES).to(DEVICE)
    print_model_parameters(model)
    
    gflops = calculate_gflops(model, input_size=(1, NUM_SLICES * 4, IMG_SIZE, IMG_SIZE), device=str(DEVICE))
    print(f"Model GFLOPs: {gflops:.2f} G")
    
    criterion = CombinedLoss(
        num_classes=NUM_CLASSES,
        initial_loss_weights=[0.4, 0.4, 0.2]
    ).to(DEVICE)
    
    optimizer = torch.optim.AdamW(chain(model.parameters(), criterion.parameters()), lr=LEARNING_RATE)
    scheduler = torch.optim.lr_scheduler.ReduceLROnPlateau(optimizer, mode='max', factor=0.5, patience=10)
    print("All components initialized.")
    print("-" * 60)
    
    best_val_metric = 0.0
    epochs_no_improve = 0
    total_train_start = time.time()

    data_load_times = []
    gpu_transfer_times = []
    forward_times = []
    backward_times = []
    total_iter_times = []
    
    for epoch in range(NUM_EPOCHS):
        epoch_start_time = time.time()
        print(f"\n--- Epoch {epoch + 1}/{NUM_EPOCHS} ---")
        
        model.train()
        epoch_train_loss = 0.0
        
        train_pbar = tqdm(train_dataloader, desc=f"Training", ncols=100)
        iter_start = time.time()
        
        try:
            for batch_idx, (images, targets) in enumerate(train_pbar):
                data_load_time = time.time() - iter_start
                data_load_times.append(data_load_time)
                
                transfer_start = time.time()
                images, targets = images.to(DEVICE), targets.to(DEVICE)
                gpu_transfer_time = time.time() - transfer_start
                gpu_transfer_times.append(gpu_transfer_time)
                
                # ePURE noise injection
                if ePURE_augmenter is not None:
                    from utils import adaptive_quantum_noise_injection
                    with torch.no_grad():
                        noise_map = ePURE_augmenter(images)
                        images = adaptive_quantum_noise_injection(images, noise_map)
                
                optimizer.zero_grad()
                
                forward_start = time.time()
                b1_map_for_loss = common_b1_map.expand(images.size(0), -1, -1, -1)
                logits_list, all_eps_sigma_tuples = model(images)

                total_loss = 0
                for logits in logits_list:
                    if logits.shape[2:] != targets.shape[1:]:
                        resized_targets = F.interpolate(
                            targets.unsqueeze(1).float(), 
                            size=logits.shape[2:], 
                            mode='nearest'
                        ).squeeze(1).long()
                    else:
                        resized_targets = targets
                    
                    loss_component = criterion(logits, resized_targets, b1_map_for_loss, all_eps_sigma_tuples)
                    total_loss += loss_component
                
                loss = total_loss / len(logits_list)
                forward_time = time.time() - forward_start
                forward_times.append(forward_time)
                
                backward_start = time.time()
                loss.backward()
                optimizer.step()
                backward_time = time.time() - backward_start
                backward_times.append(backward_time)
                
                epoch_train_loss += loss.item()
                
                total_iter_time = time.time() - iter_start
                total_iter_times.append(total_iter_time)
                
                # Print loss weights every 100 iterations for monitoring
                if batch_idx % 100 == 0 and batch_idx > 0:
                    weights = criterion.get_current_loss_weights()
                    print(f"\n   [Iter {batch_idx}] Loss: {loss.item():.6f}")
                    print(f"   Loss weights: FL={weights['weight_FocalLoss']:.4f}, "
                          f"FTL={weights['weight_FocalTverskyLoss']:.4f}, "
                          f"Phy={weights['weight_Physics']:.4f}")
                
                if batch_idx % 50 == 0 and batch_idx > 0:
                    avg_data = np.mean(data_load_times[-50:])
                    avg_transfer = np.mean(gpu_transfer_times[-50:])
                    avg_forward = np.mean(forward_times[-50:])
                    avg_backward = np.mean(backward_times[-50:])
                    avg_total = np.mean(total_iter_times[-50:])
                    
                    train_pbar.set_postfix({
                        'loss': f'{loss.item():.6f}',
                        'data': f'{avg_data:.2f}s',
                        'gpu': f'{avg_transfer:.3f}s',
                        'fwd': f'{avg_forward:.2f}s',
                        'bwd': f'{avg_backward:.2f}s',
                        'total': f'{avg_total:.2f}s'
                    })
                else:
                    train_pbar.set_postfix({'loss': f'{loss.item():.6f}'})
                
                # Explicit memory cleanup (avoid empty_cache per iteration - too slow!)
                del images, targets, logits_list, all_eps_sigma_tuples, loss, total_loss
                # Note: torch.cuda.empty_cache() moved to end of epoch only
                
                iter_start = time.time()
        finally:
            train_pbar.close()
            
            if len(data_load_times) > 0:
                print(f"\n   Profiling Summary (last 100 iters):")
                print(f"   - Data Loading:    {np.mean(data_load_times[-100:]):.3f}s ({np.mean(data_load_times[-100:])/np.mean(total_iter_times[-100:])*100:.1f}%)")
                print(f"   - GPU Transfer:    {np.mean(gpu_transfer_times[-100:]):.3f}s ({np.mean(gpu_transfer_times[-100:])/np.mean(total_iter_times[-100:])*100:.1f}%)")
                print(f"   - Forward Pass:    {np.mean(forward_times[-100:]):.3f}s ({np.mean(forward_times[-100:])/np.mean(total_iter_times[-100:])*100:.1f}%)")
                print(f"   - Backward Pass:   {np.mean(backward_times[-100:]):.3f}s ({np.mean(backward_times[-100:])/np.mean(total_iter_times[-100:])*100:.1f}%)")
                print(f"   - Total/Iteration: {np.mean(total_iter_times[-100:]):.3f}s")
                print(f"   - Estimated epoch time: {np.mean(total_iter_times[-100:]) * len(train_dataloader) / 3600:.2f} hours")
            
        avg_train_loss = epoch_train_loss / len(train_dataloader)
        epoch_time = time.time() - epoch_start_time
        print(f"   Training Loss: {avg_train_loss:.4f} | Time: {epoch_time:.2f}s")

        if val_dataloader.dataset and len(val_dataloader.dataset) > 0:
            print("   Evaluating on validation set...")
            val_metrics = evaluate_metrics(model, val_dataloader, DEVICE, NUM_CLASSES)
            
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
            
            current_lr = optimizer.param_groups[0]['lr']

            print("   --- Per-Class Metrics ---")
            for c_idx in range(NUM_CLASSES):
                print(f"=> Class {c_idx:<3}: Dice: {all_dice[c_idx]:.4f}, IoU: {all_iou[c_idx]:.4f}, "
                      f"Precision: {all_precision[c_idx]:.4f}, Recall: {all_recall[c_idx]:.4f}, F1: {all_f1[c_idx]:.4f}")

            print("   --- BraTS Regions (Benchmark Metrics) ---")
            print(f"=> ET (Enhancing Tumor): {val_metrics['ET']:.4f}")
            print(f"=> TC (Tumor Core):      {val_metrics['TC']:.4f}")
            print(f"=> WT (Whole Tumor):     {val_metrics['WT']:.4f}")
            print(f"=> Average (ET+TC+WT):   {val_metrics['avg_regions']:.4f}")
            
            print("   --- Summary Metrics ---")
            print(f"=> Avg Foreground: Dice: {avg_fg_dice:.4f}, IoU: {avg_fg_iou:.4f}, "
                  f"Precision: {avg_fg_precision:.4f}, Recall: {avg_fg_recall:.4f}, F1: {avg_fg_f1:.4f}")
            print(f"=> Overall Accuracy: {val_accuracy:.4f} | Current LR: {current_lr:.6f}")
            print(f"=> Epoch Time: {epoch_time:.2f}s")

            scheduler.step(avg_fg_dice)
            if avg_fg_dice > best_val_metric:
                best_val_metric = avg_fg_dice
                torch.save(model.state_dict(), "best_model.pth")
                print(f"   >>> New best model saved with Dice: {best_val_metric:.4f} <<<")
                epochs_no_improve = 0
            else:
                epochs_no_improve += 1
        else:
            print("   Validation dataset empty. Skipping validation.")

        if epochs_no_improve >= EARLY_STOP_PATIENCE:
            print(f"\nEarly stopping triggered after {EARLY_STOP_PATIENCE} epochs with no improvement.")
            break
        
        # Garbage collection at end of epoch
        gc.collect()
        if torch.cuda.is_available():
            torch.cuda.empty_cache()
        
        # Clear profiling lists to prevent memory buildup
        data_load_times.clear()
        gpu_transfer_times.clear()
        forward_times.clear()
        backward_times.clear()
        total_iter_times.clear()

    total_train_time = time.time() - total_train_start
    hours = int(total_train_time // 3600)
    minutes = int((total_train_time % 3600) // 60)
    seconds = int(total_train_time % 60)
    
    print("\n" + "=" * 60)
    print(f"--- Training Finished ---")
    print(f"Total Training Time: {hours}h {minutes}m {seconds}s ({total_train_time:.2f}s)")
    print(f"Best Validation Dice Score: {best_val_metric:.4f}")
    print("=" * 60 + "\n")
    
    run_and_print_test_evaluation(
        test_dataloader=test_dataloader,
        device=DEVICE,
        num_classes=NUM_CLASSES,
        num_slices=NUM_SLICES * 4
    )
    
    print("\nLoading test volumes for visualization...")
    test_vols_viz, test_masks_viz = load_brats21_volumes(brats_raw_dir, target_size=(IMG_SIZE, IMG_SIZE), max_patients=10)
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

