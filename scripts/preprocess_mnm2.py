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

def preprocess_single_patient_mnm2(patient_path, target_size=(224, 224)):
    patient_folder = os.path.basename(patient_path)
    files = os.listdir(patient_path)
    
    image_file = None
    seg_file = None
    
    for f in files:
        if f.endswith('.nii.gz') or f.endswith('.nii'):
            if 'gt' in f.lower() or 'seg' in f.lower() or 'label' in f.lower():
                seg_file = f
            else:
                image_file = f
    
    if not image_file:
        print(f"  {patient_folder}: Missing image file")
        return None
    if not seg_file:
        print(f"  {patient_folder}: Missing segmentation file")
        return None
    
    try:
        img_path = os.path.join(patient_path, image_file)
        seg_path = os.path.join(patient_path, seg_file)
        
        img_data = nib.load(img_path).get_fdata()
        mask_data = nib.load(seg_path).get_fdata()
        
        if len(img_data.shape) == 4:
            mid_t = img_data.shape[3] // 2
            img_data = img_data[:, :, :, mid_t]
        
        if len(mask_data.shape) == 4:
            mid_t = mask_data.shape[3] // 2
            mask_data = mask_data[:, :, :, mid_t]
        
        img_data = np.squeeze(img_data)
        mask_data = np.squeeze(mask_data)
        
        if len(img_data.shape) == 2:
            img_data = img_data[:, :, np.newaxis]
        if len(mask_data.shape) == 2:
            mask_data = mask_data[:, :, np.newaxis]
        
        if len(img_data.shape) != 3 or len(mask_data.shape) != 3:
            print(f"  {patient_folder}: Invalid dimensions - img {img_data.shape}, mask {mask_data.shape}")
            return None
        
        num_slices = max(img_data.shape[2], mask_data.shape[2])
        
        if img_data.shape[2] != mask_data.shape[2]:
            min_slices = min(img_data.shape[2], mask_data.shape[2])
            img_data = img_data[:, :, :min_slices]
            mask_data = mask_data[:, :, :min_slices]
            num_slices = min_slices
        
        resized_img = np.zeros((target_size[0], target_size[1], num_slices), dtype=np.float32)
        resized_mask = np.zeros((target_size[0], target_size[1], num_slices), dtype=np.uint8)
        
        for i in range(num_slices):
            resized_img[:, :, i] = resize(
                img_data[:, :, i], 
                target_size, 
                order=1, 
                preserve_range=True,
                anti_aliasing=True, 
                mode='reflect'
            )
            resized_mask[:, :, i] = resize(
                mask_data[:, :, i], 
                target_size, 
                order=0, 
                preserve_range=True,
                anti_aliasing=False, 
                mode='reflect'
            )
        
        max_val = resized_img.max()
        if max_val > 0:
            resized_img /= max_val
        
        volume_id = patient_folder
        return (resized_img, resized_mask, volume_id)
        
    except Exception as e:
        print(f"  Error processing {patient_folder}: {str(e)}")
        return None

def preprocess_mnm2_dataset(input_dir, output_dir, target_size=(224, 224), skip_existing=True):
    os.makedirs(output_dir, exist_ok=True)
    
    volumes_dir = os.path.join(output_dir, 'volumes')
    masks_dir = os.path.join(output_dir, 'masks')
    os.makedirs(volumes_dir, exist_ok=True)
    os.makedirs(masks_dir, exist_ok=True)
    
    patient_dirs = []
    for root, dirs, files in os.walk(input_dir):
        for d in dirs:
            patient_path = os.path.join(root, d)
            nii_files = [f for f in os.listdir(patient_path) if f.endswith('.nii.gz') or f.endswith('.nii')]
            if nii_files:
                patient_dirs.append(patient_path)
    
    if not patient_dirs:
        print(f"No patient directories found in {input_dir}")
        return
    
    print(f"Found {len(patient_dirs)} patients in {input_dir}")
    print(f"Output: {output_dir}")
    print(f"Target size: {target_size}")
    
    volume_info = {}
    processed_count = 0
    skipped_count = 0
    failed_count = 0
    
    for patient_path in tqdm(patient_dirs, desc="Preprocessing M&Ms2"):
        result = preprocess_single_patient_mnm2(patient_path, target_size)
        
        if result is None:
            failed_count += 1
            continue
        
        volume, mask, volume_id = result
        
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
            print(f"  Error saving {volume_id}: {e}")
            failed_count += 1
    
    metadata = {
        'dataset': 'M&Ms2',
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
    print(f"M&Ms2 Preprocessing Complete!")
    print(f"{'='*60}")
    print(f"  Processed: {processed_count} volumes")
    print(f"  Skipped:   {skipped_count} volumes (already exists)")
    print(f"  Failed:    {failed_count} patients")
    print(f"  Total:     {len(volume_info)} volumes")
    print(f"  Metadata:  {metadata_path}")
    print(f"{'='*60}")
    
    return processed_count, len(volume_info)

if __name__ == '__main__':
    parser = argparse.ArgumentParser(description='Preprocess M&Ms2 dataset to .npy files')
    parser.add_argument('--input', type=str, required=True, help='Input directory')
    parser.add_argument('--output', type=str, required=True, help='Output directory')
    parser.add_argument('--size', type=int, default=224, help='Target image size')
    parser.add_argument('--no-skip', action='store_true', help='Reprocess even if .npy files exist')
    
    args = parser.parse_args()
    
    target_size = (args.size, args.size)
    skip_existing = not args.no_skip
    
    preprocess_mnm2_dataset(
        input_dir=args.input,
        output_dir=args.output,
        target_size=target_size,
        skip_existing=skip_existing
    )
