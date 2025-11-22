import os
import sys
import argparse
import numpy as np
import nibabel as nib
from pathlib import Path
from tqdm import tqdm
import json
from skimage.transform import resize

sys.path.append(str(Path(__file__).parent.parent))

def preprocess_single_patient_mnm(patient_path, target_size=(224, 224)):
    patient_folder = os.path.basename(patient_path)
    
    img_filename = f'{patient_folder}_sa.nii.gz'
    mask_filename = f'{patient_folder}_sa_gt.nii.gz'
    
    img_path = os.path.join(patient_path, img_filename)
    mask_path = os.path.join(patient_path, mask_filename)
    
    if not os.path.exists(img_path):
        return []
    if not os.path.exists(mask_path):
        return []
    
    try:
        img_data = nib.nifti1.load(img_path).get_fdata()
        mask_data = nib.nifti1.load(mask_path).get_fdata()
        
        results = []
        
        if len(img_data.shape) == 4 and len(mask_data.shape) == 4:
            for t in range(mask_data.shape[3]):
                mask_t = mask_data[:, :, :, t]
                if len(np.unique(mask_t)) > 1:
                    img_t = img_data[:, :, :, t]
                    results.append((img_t, mask_t, f'{patient_folder}_t{t:02d}'))
        elif len(img_data.shape) == 3 and len(mask_data.shape) == 3:
            results.append((img_data, mask_data, patient_folder))
        else:
            return []
        
        processed_results = []
        for img_vol, mask_vol, vol_id in results:
            img_vol = np.squeeze(img_vol)
            mask_vol = np.squeeze(mask_vol)
            mask_vol = np.round(mask_vol).astype(np.uint8)
            
            if len(img_vol.shape) == 2:
                img_vol = img_vol[:, :, np.newaxis]
            if len(mask_vol.shape) == 2:
                mask_vol = mask_vol[:, :, np.newaxis]
            
            if len(img_vol.shape) != 3 or len(mask_vol.shape) != 3:
                continue
            
            if img_vol.shape[2] != mask_vol.shape[2]:
                min_slices = min(img_vol.shape[2], mask_vol.shape[2])
                img_vol = img_vol[:, :, :min_slices]
                mask_vol = mask_vol[:, :, :min_slices]
            
            num_slices = img_vol.shape[2]
            
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
            
            max_val = resized_img.max()
            if max_val > 0:
                resized_img /= max_val
            
            processed_results.append((resized_img, resized_mask, vol_id))
        
        return processed_results
        
    except Exception as e:
        return []

def preprocess_mnm_dataset(input_dir, output_dir, target_size=(224, 224), skip_existing=True):
    os.makedirs(output_dir, exist_ok=True)
    
    volumes_dir = os.path.join(output_dir, 'volumes')
    masks_dir = os.path.join(output_dir, 'masks')
    os.makedirs(volumes_dir, exist_ok=True)
    os.makedirs(masks_dir, exist_ok=True)
    
    patient_folders = sorted([
        os.path.join(input_dir, d) 
        for d in os.listdir(input_dir) 
        if os.path.isdir(os.path.join(input_dir, d))
    ])
    
    if not patient_folders:
        print(f"No patient directories found in {input_dir}")
        return
    
    print(f"Found {len(patient_folders)} patients in {input_dir}")
    print(f"Output: {output_dir}")
    print(f"Target size: {target_size}")
    
    volume_info = {}
    processed_count = 0
    skipped_count = 0
    failed_count = 0
    
    for patient_path in tqdm(patient_folders, desc="Preprocessing M&M"):
        results = preprocess_single_patient_mnm(patient_path, target_size)
        
        if not results:
            failed_count += 1
            continue
        
        for volume, mask, volume_id in results:
            volume_save_path = os.path.join(volumes_dir, f'{volume_id}.npy')
            mask_save_path = os.path.join(masks_dir, f'{volume_id}.npy')
            
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
            
            try:
                np.save(volume_save_path, volume)
                np.save(mask_save_path, mask)
                
                volume_info[volume_id] = {
                    'num_slices': int(mask.shape[2]),
                    'target_size': list(target_size)
                }
                processed_count += 1
                
            except Exception as e:
                failed_count += 1
    
    metadata = {
        'dataset': 'M&M',
        'target_size': list(target_size),
        'total_volumes': len(volume_info),
        'newly_processed': processed_count,
        'skipped_existing': skipped_count,
        'failed': failed_count,
        'volume_info': volume_info,
        'num_classes': 4,
        'class_names': ['Background', 'LV', 'MYO', 'RV']
    }
    
    metadata_path = os.path.join(output_dir, 'metadata.json')
    with open(metadata_path, 'w') as f:
        json.dump(metadata, f, indent=2)
    
    print(f"\n{'='*60}")
    print(f"M&M Preprocessing Complete!")
    print(f"{'='*60}")
    print(f"  Processed: {processed_count} volumes")
    print(f"  Skipped:   {skipped_count} volumes")
    print(f"  Failed:    {failed_count} patients")
    print(f"  Total:     {len(volume_info)} volumes")
    print(f"  Metadata:  {metadata_path}")
    print(f"{'='*60}")
    
    return processed_count, len(volume_info)

if __name__ == '__main__':
    parser = argparse.ArgumentParser(description='Preprocess M&M dataset to .npy files')
    parser.add_argument('--input', type=str, required=True, help='Input directory')
    parser.add_argument('--output', type=str, required=True, help='Output directory')
    parser.add_argument('--size', type=int, default=224, help='Target image size')
    parser.add_argument('--no-skip', action='store_true', help='Reprocess even if .npy files exist')
    
    args = parser.parse_args()
    
    target_size = (args.size, args.size)
    skip_existing = not args.no_skip
    
    preprocess_mnm_dataset(
        input_dir=args.input,
        output_dir=args.output,
        target_size=target_size,
        skip_existing=skip_existing
    )
