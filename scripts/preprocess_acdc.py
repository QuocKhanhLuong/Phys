"""
Preprocess ACDC dataset: Convert NIfTI files → .npy files for fast memmap loading.
Run this ONCE before training.

Usage:
    python scripts/preprocess_acdc.py --input data/ACDC --output preprocessed_data/ACDC
"""

import os
import sys
import argparse
import configparser
import numpy as np
import nibabel as nib
from pathlib import Path
from tqdm import tqdm
import json
from skimage.transform import resize
import cv2

# Add project root to path
sys.path.append(str(Path(__file__).parent.parent))


def preprocess_single_patient_acdc(patient_path, target_size=(224, 224)):
    """
    Process one ACDC patient: load ED and ES frames, resize, return volumes.
    
    Returns:
        List of (volume, mask, volume_id) tuples
    """
    patient_folder = os.path.basename(patient_path)
    info_cfg_path = os.path.join(patient_path, 'Info.cfg')
    
    # Read ED/ES frames
    if not os.path.exists(info_cfg_path):
        return []
    
    try:
        parser = configparser.ConfigParser()
        with open(info_cfg_path, 'r') as f:
            config_string = '[DEFAULT]\n' + f.read()
        parser.read_string(config_string)
        ed_frame = int(parser['DEFAULT']['ED'])
        es_frame = int(parser['DEFAULT']['ES'])
    except Exception as e:
        print(f"  Error reading Info.cfg for {patient_folder}: {e}")
        return []
    
    results = []
    
    for frame_num, frame_name in [(ed_frame, 'ED'), (es_frame, 'ES')]:
        img_filename = f'{patient_folder}_frame{frame_num:02d}.nii.gz'
        mask_filename = f'{patient_folder}_frame{frame_num:02d}_gt.nii.gz'
        
        # Try both .nii.gz and .nii
        img_path = None
        mask_path = None
        
        for suffix in ['.gz', '']:
            test_img = os.path.join(patient_path, img_filename.replace('.gz', '') if suffix == '' else img_filename)
            test_mask = os.path.join(patient_path, mask_filename.replace('.gz', '') if suffix == '' else mask_filename)
            
            if os.path.exists(test_img):
                img_path = test_img
                mask_path = test_mask
                break
        
        if img_path is None or not os.path.exists(img_path):
            continue
        
        try:
            # Load NIfTI
            img_data = nib.load(img_path).get_fdata()
            
            mask_data = None
            if os.path.exists(mask_path):
                mask_data = nib.load(mask_path).get_fdata()
            else:
                print(f"  Warning: No mask for {patient_folder} frame {frame_num}")
                continue
            
            num_slices = img_data.shape[2]
            
            # Resize all slices using cv2 (faster than skimage)
            resized_img = np.zeros((target_size[0], target_size[1], num_slices), dtype=np.float32)
            resized_mask = np.zeros((target_size[0], target_size[1], num_slices), dtype=np.uint8)
            
            for i in range(num_slices):
                resized_img[:, :, i] = cv2.resize(
                    img_data[:, :, i].astype(np.float32),
                    target_size,
                    interpolation=cv2.INTER_LINEAR
                )
                resized_mask[:, :, i] = cv2.resize(
                    mask_data[:, :, i].astype(np.uint8),
                    target_size,
                    interpolation=cv2.INTER_NEAREST
                )
            
            # Normalize volume
            max_val = resized_img.max()
            if max_val > 0:
                resized_img /= max_val
            
            volume_id = f"{patient_folder}_{frame_name}"
            results.append((resized_img, resized_mask, volume_id))
            
        except Exception as e:
            print(f"  Error processing {patient_folder} frame {frame_num}: {e}")
            continue
    
    return results


def preprocess_acdc_dataset(input_dir, output_dir, target_size=(224, 224), skip_existing=True):
    """
    Preprocess entire ACDC dataset (training or testing).
    
    Args:
        input_dir: Path to ACDC/training or ACDC/testing
        output_dir: Path to save preprocessed .npy files
        target_size: Resize target
        skip_existing: Skip if .npy already exists
    """
    os.makedirs(output_dir, exist_ok=True)
    
    volumes_dir = os.path.join(output_dir, 'volumes')
    masks_dir = os.path.join(output_dir, 'masks')
    os.makedirs(volumes_dir, exist_ok=True)
    os.makedirs(masks_dir, exist_ok=True)
    
    # Get all patient folders
    patient_folders = sorted([
        os.path.join(input_dir, d) 
        for d in os.listdir(input_dir) 
        if os.path.isdir(os.path.join(input_dir, d)) and d.startswith('patient')
    ])
    
    print(f"Found {len(patient_folders)} patients in {input_dir}")
    print(f"Output: {output_dir}")
    print(f"Target size: {target_size}")
    
    volume_info = {}
    processed_count = 0
    skipped_count = 0
    failed_count = 0
    
    for patient_path in tqdm(patient_folders, desc="Preprocessing ACDC"):
        patient_results = preprocess_single_patient_acdc(patient_path, target_size)
        
        if not patient_results:
            failed_count += 1
            continue
        
        for volume, mask, volume_id in patient_results:
            volume_save_path = os.path.join(volumes_dir, f'{volume_id}.npy')
            mask_save_path = os.path.join(masks_dir, f'{volume_id}.npy')
            
            # Skip if exists
            if skip_existing and os.path.exists(volume_save_path) and os.path.exists(mask_save_path):
                try:
                    test_mask = np.load(mask_save_path)
                    volume_info[volume_id] = {
                        'num_slices': int(test_mask.shape[2]),
                        'target_size': list(target_size)
                    }
                    skipped_count += 1
                    continue
                except:
                    pass
            
            # Save as .npy
            try:
                np.save(volume_save_path, volume)
                np.save(mask_save_path, mask)
                
                volume_info[volume_id] = {
                    'num_slices': int(mask.shape[2]),
                    'target_size': list(target_size)
                }
                processed_count += 1
                
            except Exception as e:
                print(f"  Error saving {volume_id}: {e}")
                failed_count += 1
    
    # Save metadata
    metadata = {
        'dataset': 'ACDC',
        'target_size': list(target_size),
        'total_volumes': len(volume_info),
        'newly_processed': processed_count,
        'skipped_existing': skipped_count,
        'failed': failed_count,
        'volume_info': volume_info,
        'num_classes': 4,
        'class_names': ['Background', 'RV', 'MYO', 'LV']
    }
    
    metadata_path = os.path.join(output_dir, 'metadata.json')
    with open(metadata_path, 'w') as f:
        json.dump(metadata, f, indent=2)
    
    print(f"\n{'='*60}")
    print(f"ACDC Preprocessing Complete!")
    print(f"{'='*60}")
    print(f"  Processed: {processed_count} volumes")
    print(f"  Skipped:   {skipped_count} volumes (already exists)")
    print(f"  Failed:    {failed_count} patients")
    print(f"  Total:     {len(volume_info)} volumes")
    print(f"  Metadata:  {metadata_path}")
    print(f"{'='*60}")
    
    return processed_count, len(volume_info)


def main():
    parser = argparse.ArgumentParser(description='Preprocess ACDC dataset to .npy files')
    parser.add_argument('--input', type=str, required=True,
                       help='Input directory (e.g., data/ACDC/training)')
    parser.add_argument('--output', type=str, required=True,
                       help='Output directory (e.g., preprocessed_data/ACDC/training)')
    parser.add_argument('--size', type=int, default=224,
                       help='Target image size (default: 224)')
    parser.add_argument('--no-skip', action='store_true',
                       help='Reprocess even if .npy files exist')
    
    args = parser.parse_args()
    
    target_size = (args.size, args.size)
    skip_existing = not args.no_skip
    
    preprocess_acdc_dataset(
        input_dir=args.input,
        output_dir=args.output,
        target_size=target_size,
        skip_existing=skip_existing
    )


if __name__ == '__main__':
    main()

