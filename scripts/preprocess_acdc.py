import sys
from pathlib import Path
import os

# Add project root to path for imports
project_root = os.path.abspath(os.path.join(os.path.dirname(__file__), '..'))
sys.path.append(project_root)

from src import config
from typing import Optional
import json
import glob
import re

def normalize_volume_simple(volume):
    """Normalize volume to [0, 1] range"""
    v_min, v_max = volume.min(), volume.max()
    if v_max > v_min:
        volume = (volume - v_min) / (v_max - v_min)
    return volume


def update_metadata_with_patient_info(npy_dir):
    """Update metadata.json with patient info (num_slices, etc.)"""
    metadata_path = os.path.join(npy_dir, 'metadata.json')
    masks_dir = os.path.join(npy_dir, 'masks')
    
    # Create metadata if it doesn't exist
    if not os.path.exists(metadata_path):
        metadata = {
            'target_size': [224, 224],
            'patient_info': {}
        }
        with open(metadata_path, 'w') as f:
            json.dump(metadata, f, indent=2)
    
    with open(metadata_path, 'r') as f:
        metadata = json.load(f)
    
    if 'patient_info' in metadata and metadata['patient_info']:
        print(f"Metadata already has patient_info ({len(metadata['patient_info'])} entries)")
        return True
    
    print("Building patient_info from mask files...")
    patient_info = {}
    mask_files = sorted(glob.glob(os.path.join(masks_dir, '*.npy')))
    
    import numpy as np
    from tqdm import tqdm
    
    for mask_file in tqdm(mask_files, desc="Scanning"):
        patient_id = os.path.basename(mask_file).replace('.npy', '')
        try:
            mask = np.load(mask_file)
            patient_info[patient_id] = {
                'num_slices': int(mask.shape[2]),
                'target_size': list(metadata.get('target_size', [224, 224]))
            }
        except Exception as e:
            print(f"Failed {patient_id}: {e}")
    
    metadata['patient_info'] = patient_info
    
    with open(metadata_path, 'w') as f:
        json.dump(metadata, f, indent=2)
    
    print(f"Updated metadata ({len(patient_info)} patients)")
    return True


def preprocess_acdc_dataset(input_dir: str, output_dir: str, target_size: tuple = (224, 224),
                           max_patients: Optional[int] = None, num_workers: int = 8,
                           skip_existing: bool = True):
    """
    Preprocess ACDC dataset tương tự như BraTS nhưng cho 2 modalities (ED, ES)
    
    ACDC structure:
    - Mỗi patient có 2 phases: ED (end-diastolic) và ES (end-systolic)
    - Mỗi phase có 1 image và 1 mask
    - Output: volume shape (2, H, W, Z) với 2 channels là ED và ES
    """
    import numpy as np
    import nibabel as nib
    import cv2
    from tqdm import tqdm
    import glob
    
    os.makedirs(output_dir, exist_ok=True)
    volumes_dir = os.path.join(output_dir, 'volumes')
    masks_dir = os.path.join(output_dir, 'masks')
    os.makedirs(volumes_dir, exist_ok=True)
    os.makedirs(masks_dir, exist_ok=True)
    
    # Create initial metadata.json if it doesn't exist
    metadata_path = os.path.join(output_dir, 'metadata.json')
    if not os.path.exists(metadata_path):
        initial_metadata = {
            'target_size': list(target_size),
            'patient_info': {}
        }
        with open(metadata_path, 'w') as f:
            json.dump(initial_metadata, f, indent=2)
    
    # Tìm tất cả các patient folders
    # ACDC có cấu trúc: data/ACDC/training/patient001/, data/ACDC/testing/patient001/
    patient_dirs = []
    
    # Kiểm tra nếu có thư mục training và testing
    training_dir = os.path.join(input_dir, 'training')
    testing_dir = os.path.join(input_dir, 'testing')
    
    if os.path.exists(training_dir):
        training_patients = [os.path.join(training_dir, d) for d in os.listdir(training_dir) 
                           if os.path.isdir(os.path.join(training_dir, d)) and d.startswith('patient')]
        patient_dirs.extend(training_patients)
    
    if os.path.exists(testing_dir):
        testing_patients = [os.path.join(testing_dir, d) for d in os.listdir(testing_dir) 
                          if os.path.isdir(os.path.join(testing_dir, d)) and d.startswith('patient')]
        patient_dirs.extend(testing_patients)
    
    # Nếu không có training/testing, tìm trực tiếp trong input_dir
    if not patient_dirs:
        patient_dirs = [os.path.join(input_dir, d) for d in os.listdir(input_dir) 
                       if os.path.isdir(os.path.join(input_dir, d)) and d.startswith('patient')]
    
    patient_dirs.sort()
    
    if max_patients:
        patient_dirs = patient_dirs[:max_patients]
    
    print(f"Found {len(patient_dirs)} patients")
    
    processed = 0
    failed = 0
    
    for patient_dir in tqdm(patient_dirs, desc="Preprocessing"):
        try:
            # patient_dir đã là full path
            patient_id = os.path.basename(patient_dir)
            
            # Tìm ED và ES images và masks
            # ACDC format: patient001_frame01.nii (ED), patient001_frame12.nii (ES)
            #             patient001_frame01_gt.nii (ED mask), patient001_frame12_gt.nii (ES mask)
            
            ed_img_path = None
            es_img_path = None
            ed_mask_path = None
            es_mask_path = None
            
            # Tìm files trong patient directory
            files = os.listdir(patient_dir)
            
            # Tìm tất cả các frame files
            frame_files = {}
            for file in files:
                if not (file.endswith('.nii') or file.endswith('.nii.gz')):
                    continue
                
                file_lower = file.lower()
                file_path = os.path.join(patient_dir, file)
                
                # Skip 4d files và Info.cfg
                if '4d' in file_lower or 'info' in file_lower:
                    continue
                
                # Extract frame number từ tên file (ví dụ: patient001_frame01.nii -> frame01)
                frame_match = re.search(r'frame(\d+)', file_lower)
                if frame_match:
                    frame_num = frame_match.group(1)
                    if frame_num not in frame_files:
                        frame_files[frame_num] = {'img': None, 'mask': None}
                    
                    if 'gt' in file_lower or 'seg' in file_lower:
                        frame_files[frame_num]['mask'] = file_path
                    else:
                        frame_files[frame_num]['img'] = file_path
            
            # ED thường là frame01, ES là frame khác (thường là frame12 hoặc frame08)
            if '01' in frame_files:
                ed_img_path = frame_files['01']['img']
                ed_mask_path = frame_files['01']['mask']
            
            # Tìm ES frame (frame khác frame01, ưu tiên frame12, sau đó frame08, sau đó frame đầu tiên khác)
            es_frame_candidates = ['12', '08', '10', '11', '13', '14', '15']
            for candidate in es_frame_candidates:
                if candidate in frame_files and candidate != '01':
                    es_img_path = frame_files[candidate]['img']
                    es_mask_path = frame_files[candidate]['mask']
                    break
            
            # Nếu không tìm thấy trong candidates, tìm frame đầu tiên khác frame01
            if not es_img_path:
                for frame_num in sorted(frame_files.keys()):
                    if frame_num != '01' and frame_files[frame_num]['img']:
                        es_img_path = frame_files[frame_num]['img']
                        es_mask_path = frame_files[frame_num]['mask']
                        break
            
            if not all([ed_img_path, es_img_path, ed_mask_path, es_mask_path]):
                print(f"Warning: Missing files for {patient_id}")
                print(f"  ED img: {ed_img_path}, ES img: {es_img_path}")
                print(f"  ED mask: {ed_mask_path}, ES mask: {es_mask_path}")
                failed += 1
                continue
            
            # Load images và masks
            ed_image = nib.load(ed_img_path).get_fdata()
            es_image = nib.load(es_img_path).get_fdata()
            ed_mask = nib.load(ed_mask_path).get_fdata().astype(np.uint8)
            es_mask = nib.load(es_mask_path).get_fdata().astype(np.uint8)
            
            # Normalize images
            ed_image = normalize_volume_simple(ed_image)
            es_image = normalize_volume_simple(es_image)
            
            # ACDC mask values: 0=Background, 1=Right Ventricle, 2=Myocardium, 3=Left Ventricle
            # Đảm bảo mask values trong range [0, 3]
            ed_mask = np.clip(ed_mask, 0, 3).astype(np.uint8)
            es_mask = np.clip(es_mask, 0, 3).astype(np.uint8)
            
            # Resize volumes
            num_slices = ed_image.shape[2] if len(ed_image.shape) == 3 else ed_image.shape[3]
            
            # Handle 3D or 4D volumes
            if len(ed_image.shape) == 3:
                # 3D volume: (H, W, Z)
                resized_ed = np.zeros((target_size[0], target_size[1], num_slices), dtype=np.float32)
                resized_es = np.zeros((target_size[0], target_size[1], num_slices), dtype=np.float32)
                resized_mask = np.zeros((target_size[0], target_size[1], num_slices), dtype=np.uint8)
                
                for i in range(num_slices):
                    resized_ed[:, :, i] = cv2.resize(
                        ed_image[:, :, i], target_size, interpolation=cv2.INTER_LINEAR
                    )
                    resized_es[:, :, i] = cv2.resize(
                        es_image[:, :, i], target_size, interpolation=cv2.INTER_LINEAR
                    )
                    resized_mask[:, :, i] = cv2.resize(
                        ed_mask[:, :, i], target_size, interpolation=cv2.INTER_NEAREST
                    )
            else:
                # 4D volume: (H, W, Z, T) - take first time point
                resized_ed = np.zeros((target_size[0], target_size[1], num_slices), dtype=np.float32)
                resized_es = np.zeros((target_size[0], target_size[1], num_slices), dtype=np.float32)
                resized_mask = np.zeros((target_size[0], target_size[1], num_slices), dtype=np.uint8)
                
                for i in range(num_slices):
                    resized_ed[:, :, i] = cv2.resize(
                        ed_image[:, :, i, 0], target_size, interpolation=cv2.INTER_LINEAR
                    )
                    resized_es[:, :, i] = cv2.resize(
                        es_image[:, :, i, 0], target_size, interpolation=cv2.INTER_LINEAR
                    )
                    resized_mask[:, :, i] = cv2.resize(
                        ed_mask[:, :, i, 0], target_size, interpolation=cv2.INTER_NEAREST
                    )
            
            # Combine ED và ES thành volume (2, H, W, Z)
            volume = np.stack([resized_ed, resized_es], axis=0)  # (2, H, W, Z)
            
            # Save
            volume_path = os.path.join(volumes_dir, f'{patient_id}.npy')
            mask_path = os.path.join(masks_dir, f'{patient_id}.npy')
            
            np.save(volume_path, volume)
            np.save(mask_path, resized_mask)
            
            processed += 1
            
        except Exception as e:
            print(f"Error processing {patient_id}: {e}")
            import traceback
            traceback.print_exc()
            failed += 1
            continue
    
    # Update metadata
    update_metadata_with_patient_info(output_dir)
    
    return processed, failed


if __name__ == '__main__':
    INPUT_DIR = str(config.ACDC_RAW_DIR) if hasattr(config, 'ACDC_RAW_DIR') else './data/ACDC'
    OUTPUT_DIR = str(config.ACDC_PREPROCESSED_DIR) if hasattr(config, 'ACDC_PREPROCESSED_DIR') else './preprocessed_data/acdc'
    TARGET_SIZE = (224, 224)
    NUM_WORKERS = 8
    
    if len(sys.argv) > 1 and sys.argv[1] == '--update-metadata':
        print(f"Updating metadata: {OUTPUT_DIR}")
        update_metadata_with_patient_info(OUTPUT_DIR)
        sys.exit(0)
    
    print("="*60)
    print("ACDC Preprocessing (2D)")
    print("="*60)
    print(f"Input: {INPUT_DIR}")
    print(f"Output: {OUTPUT_DIR}")
    print(f"Size: {TARGET_SIZE[0]}x{TARGET_SIZE[1]}, Workers: {NUM_WORKERS}")
    print("="*60)
    
    processed, failed = preprocess_acdc_dataset(
        input_dir=INPUT_DIR,
        output_dir=OUTPUT_DIR,
        target_size=TARGET_SIZE,
        max_patients=None,
        num_workers=NUM_WORKERS,
        skip_existing=True
    )
    
    print(f"\nDone. Processed: {processed}, Failed: {failed}")

