import os
import sys
import argparse
import numpy as np
import nibabel as nib
import matplotlib.pyplot as plt
from pathlib import Path
from typing import List, Tuple, Optional
import json
from tqdm import tqdm
import time

# MONAI imports
try:
    from monai.transforms.compose import Compose
    from monai.transforms.io.dictionary import LoadImaged
    from monai.transforms.utility.dictionary import EnsureChannelFirstd, EnsureTyped
    from monai.transforms.spatial.dictionary import Orientationd, Spacingd, Resized
    from monai.transforms.intensity.dictionary import NormalizeIntensityd, ScaleIntensityRanged
    from monai.transforms.croppad.dictionary import CropForegroundd
    from monai.utils.misc import set_determinism
except ImportError as e:
    print(f"MONAI import error: {e}")
    print("Please install MONAI: pip install monai")
    sys.exit(1)


def get_core_preprocessing_transforms(
    target_spacing: Tuple[float, float, float] = (1.0, 1.0, 1.0),
    target_size: Tuple[int, int, int] = (224, 224, 155)
):
    """
    Defines the core 3D preprocessing pipeline (offline).
    Applies standard steps: Load, Reorient, Respace, Normalize, Crop, Resize.
    """
    return Compose([
        # 1. Load 4 modalities and ensure channel-first format
        LoadImaged(keys=["image", "label"]),
        EnsureChannelFirstd(keys=["image", "label"]),
        EnsureTyped(keys=["image", "label"], data_type="tensor"), # Ensure tensor for MONAI transforms

        # 2. Reorient to a standard orientation (e.g., RAS)
        Orientationd(keys=["image", "label"], axcodes="RAS"),

        # 3. Resample to isotropic spacing
        Spacingd(
            keys=["image", "label"],
            pixdim=target_spacing,
            mode=("bilinear", "nearest") # Bilinear for image, Nearest for label
        ),

        # 4. Normalize intensity using Z-score based on non-zero voxels
        NormalizeIntensityd(keys="image", nonzero=True, channel_wise=True),

        # 5. Crop unnecessary background
        CropForegroundd(keys=["image", "label"], source_key="image"),

        # 6. Resize to target spatial dimensions
        Resized(keys=["image", "label"], spatial_size=target_size, mode=("trilinear", "nearest")),

        # 7. Ensure final type for saving
        EnsureTyped(keys=["image", "label"], data_type="numpy"),
    ])


def _canonicalize_shape(volume: np.ndarray, is_label: bool) -> np.ndarray:
    """Ensure volume is (C, H, W, D) and label is (H, W, D)."""
    # Remove batch dim if present
    if volume.ndim == 5 and volume.shape[0] == 1:
        volume = np.squeeze(volume, axis=0)

    if is_label:
        # Expected shape (1, H, W, D) -> (H, W, D)
        if volume.ndim == 4 and volume.shape[0] == 1:
            volume = np.squeeze(volume, axis=0)
    else:
        # Expected shape (C, H, W, D)
        if volume.ndim == 3: # Add channel dim if missing
            volume = np.expand_dims(volume, axis=0)
            
    # Ensure channel dim is first for volume
    if not is_label and volume.ndim == 4 and volume.shape[-1] == 4: # Assuming 4 channels if last dim is 4
         volume = np.moveaxis(volume, -1, 0)

    return volume


def preprocess_brats21_with_monai_core(
    input_dir: str,
    output_dir: str,
    target_spacing: Tuple[float, float, float] = (1.0, 1.0, 1.0),
    target_size: Tuple[int, int, int] = (224, 224, 155),
    min_slices_required: int = 50,
    visualize_samples: bool = True,
    num_visualization_samples: int = 5
):
    """
    Runs the core preprocessing pipeline (offline) and saves results as .npy files.
    Augmentations are NOT applied here, they should be done online during training.
    """
    print("=" * 60)
    print("MONAI CORE 3D PREPROCESSING FOR BRATS21")
    print("=" * 60)

    volumes_dir = os.path.join(output_dir, 'volumes')
    masks_dir = os.path.join(output_dir, 'masks')
    viz_dir = os.path.join(output_dir, 'visualizations_preprocessed')

    os.makedirs(volumes_dir, exist_ok=True)
    os.makedirs(masks_dir, exist_ok=True)
    if visualize_samples:
        os.makedirs(viz_dir, exist_ok=True)

    patient_dirs = sorted([d for d in os.listdir(input_dir) if os.path.isdir(os.path.join(input_dir, d)) and d.startswith('BraTS')])
    print(f"Found {len(patient_dirs)} patients in {input_dir}")

    # Get the core preprocessing transforms
    transforms = get_core_preprocessing_transforms(target_spacing, target_size)
    print("Using core preprocessing pipeline (Load, Orient, Space, Normalize, Crop, Resize).")

    processed_count = 0
    visualization_count = 0
    patient_info = {}

    for patient_id in tqdm(patient_dirs, desc="Preprocessing patients"):
        patient_path = os.path.join(input_dir, patient_id)

        # Define file paths for MONAI LoadImaged
        # MONAI LoadImaged expects a list for multi-channel images
        data = {
            "image": [
                os.path.join(patient_path, f"{patient_id}_t1.nii.gz"),
                os.path.join(patient_path, f"{patient_id}_t1ce.nii.gz"),
                os.path.join(patient_path, f"{patient_id}_t2.nii.gz"),
                os.path.join(patient_path, f"{patient_id}_flair.nii.gz"),
            ],
            "label": os.path.join(patient_path, f"{patient_id}_seg.nii.gz")
        }

        # Check if all files exist before processing
        all_files_exist = all(os.path.exists(p) for p in data["image"]) and os.path.exists(data["label"])
        if not all_files_exist:
            print(f"Skipping {patient_id}: Missing one or more raw NIfTI files.")
            continue

        try:
            # Apply the core preprocessing transforms
            processed_data = transforms(data)

            processed_volume = processed_data["image"] # Output is already numpy due to EnsureTyped
            processed_seg = processed_data["label"]     # Output is already numpy

            # Ensure shapes are correct after transforms
            processed_volume = _canonicalize_shape(processed_volume, is_label=False)
            processed_seg = _canonicalize_shape(processed_seg, is_label=True)

            # Check minimum slices requirement *after* preprocessing (cropping might reduce slices)
            if processed_volume.shape[-1] < min_slices_required:
                print(f"Skipping {patient_id}: Only {processed_volume.shape[-1]} slices after preprocessing (min: {min_slices_required})")
                continue

            # Remap BraTS labels {1, 2, 4} -> {1, 2, 3} (Important!)
            remapped_seg = np.zeros_like(processed_seg, dtype=np.uint8)
            remapped_seg[processed_seg == 1] = 1
            remapped_seg[processed_seg == 2] = 2
            remapped_seg[processed_seg == 4] = 3 # Or use original labels if model expects them

            # Save preprocessed data as .npy
            np.save(os.path.join(volumes_dir, f"{patient_id}.npy"), processed_volume.astype(np.float32))
            np.save(os.path.join(masks_dir, f"{patient_id}.npy"), remapped_seg.astype(np.uint8)) # Save remapped label

            patient_info[patient_id] = {'num_slices': processed_volume.shape[-1]}
            processed_count += 1

            if visualize_samples and visualization_count < num_visualization_samples:
                # Visualize the *preprocessed* data, not augmented data
                create_visualization_samples(processed_volume, remapped_seg, patient_id, viz_dir)
                visualization_count += 1

        except Exception as e:
            print(f"Error processing {patient_id}: {e}")
            import traceback
            traceback.print_exc()

    metadata = {
        "processed_patients": processed_count,
        "target_spacing": target_spacing,
        "target_size": target_size,
        "min_slices_required": min_slices_required,
        "preprocessing_transforms": [
            "LoadImaged", "EnsureChannelFirstd", "Orientationd", "Spacingd",
            "NormalizeIntensityd", "CropForegroundd", "Resized"
        ],
        "label_remapping": "{1,2,4} -> {1,2,3}",
        "patient_info": patient_info
    }

    with open(os.path.join(output_dir, 'metadata.json'), 'w') as f:
        json.dump(metadata, f, indent=2)

    print(f"\nPreprocessing completed!")
    print(f"Processed: {processed_count}/{len(patient_dirs)} patients")
    print(f"Output directory: {output_dir}")
    if visualize_samples:
        print(f"Preprocessed visualization samples saved to: {viz_dir}")

# --- Hàm load_patient_data không cần thiết nữa vì LoadImaged xử lý ---

def create_visualization_samples(volume: np.ndarray, segmentation: np.ndarray,
                               patient_id: str, viz_dir: str):
    """Creates a visualization of the preprocessed volume."""
    if segmentation.max() > 0:
        tumor_areas = np.sum(segmentation > 0, axis=(0, 1))
        slice_to_show = np.argmax(tumor_areas)
    else:
        slice_to_show = volume.shape[-1] // 2

    fig, axes = plt.subplots(2, 4, figsize=(20, 10))
    fig.suptitle(f'Patient {patient_id} - Preprocessed Data (Slice {slice_to_show})', fontsize=16)

    modalities = ['T1', 'T1ce', 'T2', 'FLAIR']
    for i, mod_name in enumerate(modalities):
        img_slice = volume[i, :, :, slice_to_show]
        seg_slice = segmentation[:, :, slice_to_show]

        # Plot modality
        axes[0, i].imshow(np.rot90(img_slice), cmap='gray')
        axes[0, i].set_title(f'Modality: {mod_name}')
        axes[0, i].axis('off')

        # Plot with segmentation overlay
        axes[1, i].imshow(np.rot90(img_slice), cmap='gray')
        if seg_slice.max() > 0:
            masked_seg = np.ma.masked_where(seg_slice == 0, seg_slice)
            cmap = plt.get_cmap('jet', 4) # Colormap with 4 distinct colors
            axes[1, i].imshow(np.rot90(masked_seg), cmap=cmap, alpha=0.5, vmin=0, vmax=3)
        axes[1, i].set_title(f'{mod_name} + Seg Mask')
        axes[1, i].axis('off')

    plt.tight_layout(rect=[0, 0, 1, 0.96])
    plt.savefig(os.path.join(viz_dir, f'{patient_id}_preprocessed.png'), dpi=150)
    plt.close(fig)


def main():
    parser = argparse.ArgumentParser(description='MONAI Core 3D Preprocessing for BraTS21')
    parser.add_argument('--input_dir', type=str, required=True, help='Path to BraTS21 raw data directory')
    parser.add_argument('--output_dir', type=str, required=True, help='Path to save preprocessed .npy data')
    parser.add_argument('--target_size', type=int, nargs=3, default=[224, 224, 155], help='Target volume size (H, W, D)')
    parser.add_argument('--target_spacing', type=float, nargs=3, default=[1.0, 1.0, 1.0], help='Target voxel spacing (X, Y, Z)')
    parser.add_argument('--num_viz_samples', type=int, default=5, help='Number of visualization samples to create')
    parser.add_argument('--no_visualization', action='store_true', help='Disable visualization samples')

    args = parser.parse_args()

    # Chỉ gọi set_determinism một lần duy nhất
    set_determinism(seed=42)

    preprocess_brats21_with_monai_core(
        input_dir=args.input_dir,
        output_dir=args.output_dir,
        target_spacing=tuple(args.target_spacing),
        target_size=tuple(args.target_size),
        visualize_samples=not args.no_visualization,
        num_visualization_samples=args.num_viz_samples
    )

if __name__ == "__main__":
    main()