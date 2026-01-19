"""
Preprocess SCD dataset: Convert NIfTI files -> .npy files for fast memmap loading.
Run this ONCE before training.
Follows M&M style: Scans for 4D files and extracts ANY frames with annotations.

Usage:
    python scripts/preprocess_scd.py --input data/SCD --output preprocessed_data/SCD
"""

import os
import sys
import argparse
import numpy as np
import nibabel as nib
from pathlib import Path
from tqdm import tqdm
import json
from skimage.transform import resize
import glob

# Add project root to path
sys.path.append(str(Path(__file__).parent.parent))


def preprocess_single_patient_scd(patient_path, target_size=(224, 224)):
    """
    Process a single patient directory containing 4D NIfTI files.
    Extracts frames that have ground truth annotations.
    """
    patient_id = os.path.basename(patient_path)
    
    # Logic to find the 4D files. 
    # M&M uses `patient_id_sa.nii.gz`, SCD uses `SCDXXXXXX.nii.gz`?
    # Based on `ls` output: `SCD0001001.nii.gz` and `SCD0001001_gt.nii.gz`
    
    gt_glob = glob.glob(os.path.join(patient_path, "*_gt.nii.gz"))
    if not gt_glob:
        return []
        
    mask_path = gt_glob[0]
    img_path = mask_path.replace("_gt.nii.gz", ".nii.gz")
    
    if not os.path.exists(img_path):
        # Try finding the image file if naming is different, though usually it matches
        img_glob = glob.glob(os.path.join(patient_path, "*.nii.gz"))
        img_candidates = [f for f in img_glob if "_gt.nii.gz" not in f]
        if img_candidates:
            img_path = img_candidates[0]
        else:
            return []

    try:
        img_nii = nib.load(img_path)
        mask_nii = nib.load(mask_path)
        
        # Load data as float32/uint8
        img_data = img_nii.get_fdata().astype(np.float32)
        mask_data = mask_nii.get_fdata().astype(np.uint8)
        
        results = []
        
        # 4D Case: (H, W, Z, T)
        if img_data.ndim == 4 and mask_data.ndim == 4:
            num_frames = mask_data.shape[3]
            for t in range(num_frames):
                mask_t = mask_data[:, :, :, t]
                # Check if this frame has any annotations (sum > 0)
                if np.sum(mask_t) > 0:
                    img_t = img_data[:, :, :, t]
                    # Create a frame ID like SCD0001001_t00
                    frame_id = f"{patient_id}_t{t:02d}"
                    results.append((img_t, mask_t, frame_id))
                    
        # 3D Case: (H, W, Z) - Single frame
        elif img_data.ndim == 3 and mask_data.ndim == 3:
            if np.sum(mask_data) > 0:
                results.append((img_data, mask_data, patient_id))
        else:
            return []
            
        processed_results = []
        for img_vol, mask_vol, vol_id in results:
            # Squeeze unneeded dimensions
            img_vol = np.squeeze(img_vol)
            mask_vol = np.squeeze(mask_vol)
            
            # Ensure 3D (H, W, Z)
            if img_vol.ndim == 2: img_vol = img_vol[..., np.newaxis]
            if mask_vol.ndim == 2: mask_vol = mask_vol[..., np.newaxis]
            
            # Match slices if mismatched
            if img_vol.shape[2] != mask_vol.shape[2]:
                min_slices = min(img_vol.shape[2], mask_vol.shape[2])
                img_vol = img_vol[:, :, :min_slices]
                mask_vol = mask_vol[:, :, :min_slices]
                
            num_slices = img_vol.shape[2]
            
            # Resize
            resized_img = np.zeros((target_size[0], target_size[1], num_slices), dtype=np.float32)
            resized_mask = np.zeros((target_size[0], target_size[1], num_slices), dtype=np.uint8)

            for i in range(num_slices):
                resized_img[:, :, i] = resize(
                    img_vol[:, :, i], 
                    target_size, 
                    order=1, 
                    preserve_range=True, 
                    anti_aliasing=True, 
                    mode='reflect'
                )
                resized_mask[:, :, i] = np.round(resize(
                    mask_vol[:, :, i].astype(np.float32), 
                    target_size, 
                    order=0, 
                    preserve_range=True, 
                    anti_aliasing=False, 
                    mode='reflect'
                )).astype(np.uint8)

            # Normalize Intensity (Min-Max)
            # Safe normalization
            min_val = resized_img.min()
            max_val = resized_img.max()
            if max_val > min_val:
                resized_img = (resized_img - min_val) / (max_val - min_val)
            else:
                resized_img = resized_img * 0
                
            processed_results.append((resized_img, resized_mask, vol_id))
            
        return processed_results

    except Exception as e:
        print(f"Error processing {patient_id}: {e}")
        return []

def preprocess_scd_subset(input_subset_dir, output_subset_dir, target_size=(224, 224), skip_existing=True):
    volumes_dir = os.path.join(output_subset_dir, 'volumes')
    masks_dir = os.path.join(output_subset_dir, 'masks')
    os.makedirs(volumes_dir, exist_ok=True)
    os.makedirs(masks_dir, exist_ok=True)
    
    # Recursively find all directories that look like they contain data
    # (i.e., contain *_gt.nii.gz)
    print(f"Scanning {input_subset_dir}...")
    gt_files = glob.glob(os.path.join(input_subset_dir, "**", "*_gt.nii.gz"), recursive=True)
    
    # Get unique patient directories
    patient_dirs = sorted(list(set([os.path.dirname(f) for f in gt_files])))
    print(f"Found {len(patient_dirs)} patients.")
    
    volume_info = {}
    
    count = 0
    for p_dir in tqdm(patient_dirs, desc=f"Processing"):
        results = preprocess_single_patient_scd(p_dir, target_size)
        
        for vol, mask, vol_id in results:
            out_vol_path = os.path.join(volumes_dir, f"{vol_id}.npy")
            out_mask_path = os.path.join(masks_dir, f"{vol_id}.npy")
            
            # Store info for metadata
            # We need to know num_slices. Load it if we skip, or from vol if we process.
            
            if skip_existing and os.path.exists(out_vol_path) and os.path.exists(out_mask_path):
                 try:
                     # Check existing to get metadata
                     # To avoid full load, maybe just load header or assume valid?
                     # Let's verify by loading shape
                     existing_mask = np.load(out_mask_path, mmap_mode='r')
                     volume_info[vol_id] = {
                         'num_slices': int(existing_mask.shape[2]),
                         'target_size': list(target_size)
                     }
                     continue
                 except:
                     pass # If corrupt, reprocess
                
            np.save(out_vol_path, vol)
            np.save(out_mask_path, mask)
            
            volume_info[vol_id] = {
                 'num_slices': int(mask.shape[2]),
                 'target_size': list(target_size)
            }
            count += 1
            
    # Save metadata
    metadata = {
        'dataset': 'SCD',
        'subset': os.path.basename(input_subset_dir),
        'target_size': list(target_size),
        'total_volumes': len(volume_info),
        'processed_volumes': count,
        'volume_info': volume_info,
        'num_classes': 2, # LV + BG (Actually 3 in file, but 2 used)
        'class_names': ['Background', 'LV']
    }
    
    metadata_path = os.path.join(output_subset_dir, 'metadata.json')
    with open(metadata_path, 'w') as f:
        json.dump(metadata, f, indent=2)

    print(f"Saved {count} volumes to {output_subset_dir} (volumes/masks). Metadata saved.")


def main():
    parser = argparse.ArgumentParser(description='Preprocess SCD dataset to .npy files')
    parser.add_argument('--input', type=str, required=True,
                       help='Input root directory (containing training, validate, test)')
    parser.add_argument('--output', type=str, required=True,
                       help='Output root directory')
    parser.add_argument('--size', type=int, default=224,
                       help='Target image size (default: 224)')
    parser.add_argument('--no-skip', action='store_true',
                       help='Reprocess even if .npy files exist')
    
    args = parser.parse_args()
    
    target_size = (args.size, args.size)
    skip_existing = not args.no_skip
    
    # Map output subset name to possible input directory names
    subsets_map = {
        'training': ['Training', 'training'],
        'validate': ['Validate', 'validate'],
        'testing': ['Testing', 'testing', 'Test', 'test']
    }
    
    print(f"Preprocessing SCD Dataset (M&M Style - Dynamic Frame Extraction)")
    print(f"Input: {args.input}")
    print(f"Output: {args.output}")
    print(f"Target Size: {target_size}")
    print(f"{'='*60}")
    
    for out_name, possible_inputs in subsets_map.items():
        found = False
        for inp_name in possible_inputs:
            input_subset = os.path.join(args.input, inp_name)
            if os.path.exists(input_subset):
                found = True
                print(f"\nProcessing subset: {out_name} (found source: {inp_name})")
                output_subset = os.path.join(args.output, out_name)
                
                preprocess_scd_subset(
                    input_subset_dir=input_subset,
                    output_subset_dir=output_subset,
                    target_size=target_size,
                    skip_existing=skip_existing
                )
                break
        
        if not found:
             print(f"Warning: Source for subset '{out_name}' not found. Checked: {possible_inputs}")

    print(f"\n{'='*60}")
    print("SCD Preprocessing Complete!")
    print(f"{'='*60}")


if __name__ == '__main__':
    main()
