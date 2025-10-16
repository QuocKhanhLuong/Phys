import torch
import torch.nn.functional as F
import numpy as np
import os
import time
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
from data_utils import BraTS21Dataset25D, load_brats21_volumes
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
    mp.set_start_method('spawn', force=True)
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

    train_data_path = os.path.join(config.PROJECT_ROOT, 'data', 'training')
    test_data_path = os.path.join(config.PROJECT_ROOT, 'data', 'testing')
    
    print(f"Loading training volumes from: {train_data_path}...")
    all_train_volumes, all_train_masks = load_brats21_volumes(
        train_data_path, target_size=(IMG_SIZE, IMG_SIZE)
    )
    print(f"Loaded {len(all_train_volumes)} training volumes.")

    print(f"Loading testing volumes from: {test_data_path}...")
    all_test_volumes, all_test_masks = load_brats21_volumes(
        test_data_path, target_size=(IMG_SIZE, IMG_SIZE)
    )
    print(f"Loaded {len(all_test_volumes)} testing volumes.")

    for i in range(len(all_train_volumes)):
        for mod_idx in range(4):
            max_val = np.max(all_train_volumes[i][mod_idx])
            if max_val > 0:
                all_train_volumes[i][mod_idx] /= max_val
    
    for i in range(len(all_test_volumes)):
        for mod_idx in range(4):
            max_val = np.max(all_test_volumes[i][mod_idx])
            if max_val > 0:
                all_test_volumes[i][mod_idx] /= max_val

    indices = list(range(len(all_train_volumes)))
    train_indices, val_indices = train_test_split(indices, test_size=0.2, random_state=42)

    X_train_vols = [all_train_volumes[i] for i in train_indices]
    y_train_vols = [all_train_masks[i] for i in train_indices]
    X_val_vols = [all_train_volumes[i] for i in val_indices]
    y_val_vols = [all_train_masks[i] for i in val_indices]
    
    ePURE_augmenter = ePURE(in_channels=NUM_SLICES * 4).to(DEVICE)
    ePURE_augmenter.eval()

    train_dataset = BraTS21Dataset25D(
        volumes_list=X_train_vols, 
        masks_list=y_train_vols, 
        num_input_slices=NUM_SLICES, 
        transforms=train_transform,
        noise_injector_model=ePURE_augmenter,
        device=str(DEVICE)
    )
    val_dataset = BraTS21Dataset25D(
        volumes_list=X_val_vols, 
        masks_list=y_val_vols, 
        num_input_slices=NUM_SLICES, 
        transforms=val_test_transform
    )
    test_dataset = BraTS21Dataset25D(
        volumes_list=all_test_volumes, 
        masks_list=all_test_masks, 
        num_input_slices=NUM_SLICES, 
        transforms=val_test_transform
    )

    train_dataloader = DataLoader(train_dataset, batch_size=BATCH_SIZE, shuffle=True, num_workers=0, pin_memory=True)
    val_dataloader = DataLoader(val_dataset, batch_size=BATCH_SIZE, shuffle=False, num_workers=0, pin_memory=True)
    test_dataloader = DataLoader(test_dataset, batch_size=BATCH_SIZE, shuffle=False, num_workers=0, pin_memory=True)
    
    print(f"\nTraining samples: {len(train_dataset)}, Validation: {len(val_dataset)}, Test: {len(test_dataset)}")
    print("-" * 60)

    def convert_volumes_to_tensor(volumes_list):
        all_slices = []
        for vol in volumes_list:
            for i in range(vol.shape[3]):
                all_slices.append(torch.from_numpy(vol[:, :, :, i]).mean(dim=0, keepdim=True))
        return torch.stack(all_slices, dim=0).float()
    
    all_images_tensor = convert_volumes_to_tensor(X_train_vols + X_val_vols + all_test_volumes)
    dataset_name = "brats21"
    common_b1_map = calculate_ultimate_common_b1_map(
        all_images=all_images_tensor,
        device=str(DEVICE),
        save_path=f"{dataset_name}_ultimate_common_b1_map.pth"
    )
    
    print("Initializing model...")
    model = RobustMedVFL_UNet(n_channels=NUM_SLICES * 4, n_classes=NUM_CLASSES).to(DEVICE)
    print_model_parameters(model)
    
    gflops = calculate_gflops(model, input_size=(1, NUM_SLICES * 4, IMG_SIZE, IMG_SIZE), device=str(DEVICE))
    print(f"Model GFLOPs: {gflops:.2f} G")
    
    criterion = CombinedLoss(
        num_classes=NUM_CLASSES,
        initial_loss_weights=[0.5, 0.4, 0.1]
    ).to(DEVICE)
    
    optimizer = torch.optim.AdamW(chain(model.parameters(), criterion.parameters()), lr=LEARNING_RATE)
    scheduler = torch.optim.lr_scheduler.ReduceLROnPlateau(optimizer, mode='max', factor=0.5, patience=10)
    print("All components initialized.")
    print("-" * 60)
    
    best_val_metric = 0.0
    epochs_no_improve = 0
    total_train_start = time.time()

    for epoch in range(NUM_EPOCHS):
        epoch_start_time = time.time()
        print(f"\n--- Epoch {epoch + 1}/{NUM_EPOCHS} ---")
        
        model.train()
        epoch_train_loss = 0.0
        
        train_pbar = tqdm(train_dataloader, desc=f"Training", ncols=100)
        for images, targets in train_pbar:
            images, targets = images.to(DEVICE), targets.to(DEVICE)
            optimizer.zero_grad()
            
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
            loss.backward()
            optimizer.step()
            epoch_train_loss += loss.item()
            
            train_pbar.set_postfix({'loss': f'{loss.item():.4f}'})
            
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

    visualize_final_results_2_5D(
        volumes_np=all_test_volumes,
        masks_np=all_test_masks,
        num_classes=NUM_CLASSES,
        num_samples=10,
        device=DEVICE,
        num_slices=NUM_SLICES * 4
    )

