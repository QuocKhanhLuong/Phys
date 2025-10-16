import torch
import torch.nn.functional as F
import numpy as np
import os
from sklearn.model_selection import train_test_split
from torch.utils.data import DataLoader
import albumentations as A
from albumentations.pytorch import ToTensorV2
from itertools import chain
import cv2
import torch.multiprocessing as mp

from models import ePURE, RobustMedVFL_UNet, print_model_parameters
from losses import CombinedLoss
from data_utils import BraTS21Dataset25D, load_brats21_volumes
from evaluate import evaluate_metrics, run_and_print_test_evaluation, visualize_final_results_2_5D
from utils import calculate_ultimate_common_b1_map


NUM_EPOCHS = 250
NUM_CLASSES = 4
LEARNING_RATE = 1e-3
DEVICE = torch.device('cuda' if torch.cuda.is_available() else 'cpu')
IMG_SIZE = 224
BATCH_SIZE = 24
NUM_SLICES = 5
EARLY_STOP_PATIENCE = 30


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

    base_dataset_root = '/Users/alvinluong/PhysicsMed/BraTS21'
    train_data_path = os.path.join(base_dataset_root, 'training')
    test_data_path = os.path.join(base_dataset_root, 'testing')
    
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
                all_slices.append(torch.from_numpy(vol[:, :, :, i]).unsqueeze(0).mean(dim=1, keepdim=True))
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

    for epoch in range(NUM_EPOCHS):
        print(f"\n--- Epoch {epoch + 1}/{NUM_EPOCHS} ---")
        
        model.train()
        epoch_train_loss = 0.0
        for images, targets in train_dataloader:
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
            
        avg_train_loss = epoch_train_loss / len(train_dataloader)
        print(f"   Epoch {epoch+1} - Training Loss: {avg_train_loss:.4f}")

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

    print("\n--- Training Finished ---")
    
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

