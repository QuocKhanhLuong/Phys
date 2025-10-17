import torch
from torch.utils.data import Dataset
import numpy as np
import os
import nibabel as nib  # type: ignore
from skimage.transform import resize
import glob
from tqdm import tqdm
import json
import cv2
from multiprocessing import Pool
from functools import partial

class BraTS21Dataset25D(Dataset):
    def __init__(self, volumes_list, masks_list, num_input_slices=5, transforms=None, 
                 noise_injector_model=None, device: str = 'cpu', lazy_load=False, patient_paths=None,
                 use_npy=False, npy_dir=None, patient_ids=None, cache_volumes=True):
        if num_input_slices % 2 == 0:
            raise ValueError("num_input_slices must be odd.")
        
        self.lazy_load = lazy_load
        self.use_npy = use_npy
        self.num_input_slices = num_input_slices
        self.transforms = transforms
        self.noise_injector_model = noise_injector_model
        self.device = device
        self.cache_volumes = cache_volumes
        self._volume_cache = {} if cache_volumes else None
        
        if use_npy and npy_dir is not None:
            volumes_dir = os.path.join(npy_dir, 'volumes')
            masks_dir = os.path.join(npy_dir, 'masks')
            
            if patient_ids is not None:
                self.volume_paths = [os.path.join(volumes_dir, f'{p}.npy') for p in patient_ids]
                self.mask_paths = [os.path.join(masks_dir, f'{p}.npy') for p in patient_ids]
            else:
                self.volume_paths = sorted(glob.glob(os.path.join(volumes_dir, '*.npy')))
                self.mask_paths = sorted(glob.glob(os.path.join(masks_dir, '*.npy')))
            
            self.volumes = None
            self.masks = None
            self.patient_paths = None
        elif lazy_load and patient_paths is not None:
            self.patient_paths = patient_paths
            self.volumes = None
            self.masks = None
            self.volume_paths = None
            self.mask_paths = None
        else:
            self.volumes = volumes_list
            self.masks = masks_list
            self.patient_paths = None
            self.volume_paths = None
            self.mask_paths = None
        
        self.index_map = []
        radius = (self.num_input_slices - 1) // 2
        
        if use_npy and self.volume_paths is not None and self.mask_paths is not None and npy_dir is not None:
            metadata_path = os.path.join(npy_dir, 'metadata.json')
            patient_info = {}
            
            if os.path.exists(metadata_path):
                try:
                    with open(metadata_path, 'r') as f:
                        metadata = json.load(f)
                        patient_info = metadata.get('patient_info', {})
                    print(f"Loaded metadata: {len(patient_info)} patients")
                except Exception as e:
                    print(f"Metadata load failed: {e}, using fallback")
            
            for vol_idx in range(len(self.volume_paths)):
                patient_id = os.path.basename(self.volume_paths[vol_idx]).replace('.npy', '')
                
                if patient_id in patient_info:
                    num_slices = patient_info[patient_id]['num_slices']
                else:
                    mask = np.load(self.mask_paths[vol_idx])
                    num_slices = mask.shape[2]
                
                for slice_idx in range(radius, num_slices - radius):
                    self.index_map.append((vol_idx, slice_idx))
        elif not lazy_load and self.volumes is not None:
            for vol_idx, vol in enumerate(self.volumes):
                num_slices = vol.shape[3]
                for slice_idx in range(radius, num_slices - radius):
                    self.index_map.append((vol_idx, slice_idx))
        elif lazy_load and self.patient_paths is not None:
            for vol_idx in range(len(self.patient_paths)):
                sample_path = self.patient_paths[vol_idx]
                seg_file = os.path.join(sample_path, f"{os.path.basename(sample_path)}_seg.nii.gz")
                if os.path.exists(seg_file):
                    seg_data = nib.load(seg_file)
                    num_slices = seg_data.shape[2]
                    for slice_idx in range(radius, num_slices - radius):
                        self.index_map.append((vol_idx, slice_idx))
    
    def __len__(self):
        return len(self.index_map)

    def __getitem__(self, idx):
        vol_idx, center_slice_idx = self.index_map[idx]
        
        if self._volume_cache is not None and vol_idx in self._volume_cache:
            current_volume, current_mask_volume = self._volume_cache[vol_idx]
        else:
            if self.use_npy and self.volume_paths is not None and self.mask_paths is not None:
                current_volume = np.load(self.volume_paths[vol_idx])
                current_mask_volume = np.load(self.mask_paths[vol_idx])
            elif self.lazy_load and self.patient_paths is not None:
                current_volume, current_mask_volume = self._load_volume_on_demand(vol_idx)
            elif self.volumes is not None and self.masks is not None:
                current_volume = self.volumes[vol_idx]
                current_mask_volume = self.masks[vol_idx]
            else:
                raise ValueError("Dataset not properly initialized")
            
            if self._volume_cache is not None:
                self._volume_cache[vol_idx] = (current_volume, current_mask_volume)
        
        num_slices_in_vol = current_volume.shape[3]
    
        radius = (self.num_input_slices - 1) // 2
        offsets = range(-radius, radius + 1)
        
        slice_indices = [np.clip(center_slice_idx + offset, 0, num_slices_in_vol - 1) for offset in offsets]
        
        image_stack = np.stack(
            [current_volume[:, :, :, i].transpose(1, 2, 0) for i in slice_indices],
            axis=-1
        ).astype(np.float32)
        
        image_stack = image_stack.reshape(image_stack.shape[0], image_stack.shape[1], -1)
        
        mask = current_mask_volume[:, :, center_slice_idx].astype(np.int64)
        
        if self.transforms:
            augmented = self.transforms(image=image_stack, mask=mask)
            image_tensor = augmented['image']
            mask_tensor = augmented['mask']
        else:
            image_tensor = torch.from_numpy(image_stack.transpose(2, 0, 1))
            mask_tensor = torch.from_numpy(mask)
    
        if self.noise_injector_model is not None:
            from utils import adaptive_quantum_noise_injection
            with torch.no_grad():
                img_on_gpu_with_batch = image_tensor.to(self.device).unsqueeze(0)
                noise_map = self.noise_injector_model(img_on_gpu_with_batch)
                image_tensor_with_noise_gpu = adaptive_quantum_noise_injection(
                    img_on_gpu_with_batch,
                    noise_map
                )
                image_tensor = image_tensor_with_noise_gpu.squeeze(0).cpu()
                
        return image_tensor, mask_tensor.long()
    
    def _load_volume_on_demand(self, vol_idx):
        """Load and resize volume on-the-fly for lazy loading"""
        if self.patient_paths is None:
            raise ValueError("Patient paths not initialized for lazy loading")
        
        patient_path = self.patient_paths[vol_idx]
        patient_id = os.path.basename(patient_path)
        
        t1_path = os.path.join(patient_path, f'{patient_id}_t1.nii.gz')
        t1ce_path = os.path.join(patient_path, f'{patient_id}_t1ce.nii.gz')
        t2_path = os.path.join(patient_path, f'{patient_id}_t2.nii.gz')
        flair_path = os.path.join(patient_path, f'{patient_id}_flair.nii.gz')
        seg_path = os.path.join(patient_path, f'{patient_id}_seg.nii.gz')
        
        t1 = nib.load(t1_path).get_fdata()
        t1ce = nib.load(t1ce_path).get_fdata()
        t2 = nib.load(t2_path).get_fdata()
        flair = nib.load(flair_path).get_fdata()
        volume = np.stack([t1, t1ce, t2, flair], axis=0)
        
        mask_data = nib.load(seg_path).get_fdata()
        mask = np.zeros_like(mask_data, dtype=np.uint8)
        mask[mask_data == 1] = 1
        mask[mask_data == 2] = 2
        mask[mask_data == 4] = 3
        
        target_size = (224, 224)  # Use config.IMG_SIZE in production
        num_slices = volume.shape[3]
        resized_volume = np.zeros((4, target_size[0], target_size[1], num_slices), dtype=np.float32)
        resized_mask = np.zeros((target_size[0], target_size[1], num_slices), dtype=np.uint8)
        
        for i in range(num_slices):
            for mod_idx in range(4):
                resized_volume[mod_idx, :, :, i] = resize(
                    volume[mod_idx, :, :, i], target_size, order=1, preserve_range=True,
                    anti_aliasing=True, mode='reflect'
                )
            resized_mask[:, :, i] = resize(
                mask[:, :, i], target_size, order=0, preserve_range=True,
                anti_aliasing=False, mode='reflect'
            )
        
        for mod_idx in range(4):
            max_val = np.max(resized_volume[mod_idx])
            if max_val > 0:
                resized_volume[mod_idx] /= max_val
        
        return resized_volume, resized_mask


def load_brats21_volumes(directory, target_size=(224, 224), max_patients=None):
    volumes_list = []
    masks_list = []
    
    if not os.path.exists(directory):
        print(f"Error: Dataset directory not found at {directory}.")
        return [], []

    patient_folders = sorted(glob.glob(os.path.join(directory, 'BraTS2021_*')))
    if not patient_folders:
        patient_folders = sorted(glob.glob(os.path.join(directory, 'patient*')))
    
    volume_count = 0

    for patient_path in patient_folders:
        if max_patients and volume_count >= max_patients:
            break
        
        if not os.path.isdir(patient_path):
            continue

        patient_id = os.path.basename(patient_path)
        
        t1_path = os.path.join(patient_path, f'{patient_id}_t1.nii.gz')
        t1ce_path = os.path.join(patient_path, f'{patient_id}_t1ce.nii.gz')
        t2_path = os.path.join(patient_path, f'{patient_id}_t2.nii.gz')
        flair_path = os.path.join(patient_path, f'{patient_id}_flair.nii.gz')
        seg_path = os.path.join(patient_path, f'{patient_id}_seg.nii.gz')
        
        file_4d = os.path.join(patient_path, f'{patient_id}_4d.nii')

        try:
            if all(os.path.exists(p) for p in [t1_path, t1ce_path, t2_path, flair_path]):
                t1 = nib.load(t1_path).get_fdata()
                t1ce = nib.load(t1ce_path).get_fdata()
                t2 = nib.load(t2_path).get_fdata()
                flair = nib.load(flair_path).get_fdata()
                volume = np.stack([t1, t1ce, t2, flair], axis=0)
                
                if os.path.exists(seg_path):
                    mask_data = nib.load(seg_path).get_fdata()
                    mask = np.zeros_like(mask_data, dtype=np.uint8)
                    mask[mask_data == 1] = 1
                    mask[mask_data == 2] = 2
                    mask[mask_data == 4] = 3
                else:
                    print(f"Warning: No segmentation for {patient_id}. Skipping.")
                    continue
                
                num_slices = volume.shape[3]
                resized_volume = np.zeros((4, target_size[0], target_size[1], num_slices), dtype=np.float32)
                resized_mask = np.zeros((target_size[0], target_size[1], num_slices), dtype=np.uint8)
                
                for i in range(num_slices):
                    for mod_idx in range(4):
                        resized_volume[mod_idx, :, :, i] = resize(
                            volume[mod_idx, :, :, i], target_size, order=1, preserve_range=True,
                            anti_aliasing=True, mode='reflect'
                        )
                    resized_mask[:, :, i] = resize(
                        mask[:, :, i], target_size, order=0, preserve_range=True,
                        anti_aliasing=False, mode='reflect'
                    )
                
                volumes_list.append(resized_volume)
                masks_list.append(resized_mask)
                volume_count += 1
                print(f"Loaded {patient_id}")
                    
            elif os.path.exists(file_4d):
                data_4d = nib.load(file_4d).get_fdata()
                gt_files = sorted(glob.glob(os.path.join(patient_path, f'{patient_id}_frame*_gt.nii')))
                
                if not gt_files:
                    print(f"Warning: No ground truth files for {patient_id}. Skipping.")
                    continue
                
                for gt_file in gt_files:
                    gt_basename = os.path.basename(gt_file)
                    frame_str = gt_basename.split('_')[1]
                    frame_num = int(frame_str.replace('frame', ''))
                    frame_idx = frame_num - 1
                    
                    if frame_idx >= data_4d.shape[3]:
                        print(f"Warning: Frame {frame_num} out of range for {patient_id}. Skipping.")
                        continue
                    
                    frame_volume = data_4d[:, :, :, frame_idx]
                    volume = np.stack([frame_volume] * 4, axis=0)
                    
                    gt_data = nib.load(gt_file).get_fdata()
                    mask = gt_data.astype(np.uint8)
                    
                    num_slices = volume.shape[3]
                    resized_volume = np.zeros((4, target_size[0], target_size[1], num_slices), dtype=np.float32)
                    resized_mask = np.zeros((target_size[0], target_size[1], num_slices), dtype=np.uint8)
                    
                    for i in range(num_slices):
                        for mod_idx in range(4):
                            resized_volume[mod_idx, :, :, i] = resize(
                                volume[mod_idx, :, :, i], target_size, order=1, preserve_range=True,
                                anti_aliasing=True, mode='reflect'
                            )
                        resized_mask[:, :, i] = resize(
                            mask[:, :, i], target_size, order=0, preserve_range=True,
                            anti_aliasing=False, mode='reflect'
                        )
                    
                    volumes_list.append(resized_volume)
                    masks_list.append(resized_mask)
                    volume_count += 1
                    print(f"Loaded {patient_id} - {frame_str}")
            else:
                print(f"Warning: No data files found for {patient_id}. Skipping.")
                continue
            
        except Exception as e:
            print(f"Error processing {patient_id}: {e}")
            import traceback
            traceback.print_exc()
            continue
    
    print(f"\nTotal: {len(volumes_list)} volumes from {len(patient_folders)} patients")
    return volumes_list, masks_list


def get_brats21_patient_paths(directory):
    """Get list of patient paths for lazy loading"""
    if not os.path.exists(directory):
        print(f"Error: Dataset directory not found at {directory}.")
        return []
    
    patient_folders = sorted(glob.glob(os.path.join(directory, 'BraTS2021_*')))
    if not patient_folders:
        patient_folders = sorted(glob.glob(os.path.join(directory, 'patient*')))
    
    valid_paths = []
    for patient_path in patient_folders:
        if not os.path.isdir(patient_path):
            continue
        
        patient_id = os.path.basename(patient_path)
        t1_path = os.path.join(patient_path, f'{patient_id}_t1.nii.gz')
        seg_path = os.path.join(patient_path, f'{patient_id}_seg.nii.gz')
        
        if os.path.exists(t1_path) and os.path.exists(seg_path):
            valid_paths.append(patient_path)
    
    print(f"Found {len(valid_paths)} valid patient folders")
    return valid_paths


def normalize_volume_simple(volume):
    """Simple and fast per-volume normalization"""
    max_val = volume.max()
    if max_val > 0:
        return volume / max_val
    return volume


def preprocess_single_patient(patient_path, target_size=(224, 224)):
    patient_id = os.path.basename(patient_path)
    
    t1_path = os.path.join(patient_path, f'{patient_id}_t1.nii.gz')
    t1ce_path = os.path.join(patient_path, f'{patient_id}_t1ce.nii.gz')
    t2_path = os.path.join(patient_path, f'{patient_id}_t2.nii.gz')
    flair_path = os.path.join(patient_path, f'{patient_id}_flair.nii.gz')
    seg_path = os.path.join(patient_path, f'{patient_id}_seg.nii.gz')
    
    if not all(os.path.exists(p) for p in [t1_path, t1ce_path, t2_path, flair_path, seg_path]):
        return None, None, patient_id
    
    try:
        t1 = nib.load(t1_path).get_fdata()
        t1ce = nib.load(t1ce_path).get_fdata()
        t2 = nib.load(t2_path).get_fdata()
        flair = nib.load(flair_path).get_fdata()
        mask = nib.load(seg_path).get_fdata().astype(np.uint8)
        
        t1 = normalize_volume_simple(t1)
        t1ce = normalize_volume_simple(t1ce)
        t2 = normalize_volume_simple(t2)
        flair = normalize_volume_simple(flair)
        
        mask[mask == 4] = 3
        
        volume = np.stack([t1, t1ce, t2, flair], axis=0)
        
        num_slices = volume.shape[3]
        resized_volume = np.zeros((4, target_size[0], target_size[1], num_slices), dtype=np.float32)
        resized_mask = np.zeros((target_size[0], target_size[1], num_slices), dtype=np.uint8)
        
        for i in range(num_slices):
            for mod_idx in range(4):
                resized_volume[mod_idx, :, :, i] = cv2.resize(
                    volume[mod_idx, :, :, i],
                    target_size,
                    interpolation=cv2.INTER_LINEAR
                )
            resized_mask[:, :, i] = cv2.resize(
                mask[:, :, i],
                target_size,
                interpolation=cv2.INTER_NEAREST
            )
        
        return resized_volume, resized_mask, patient_id
    
    except Exception as e:
        print(f"Error processing {patient_id}: {e}")
        return None, None, patient_id


def preprocess_and_save_patient(patient_path, output_dir, target_size, skip_existing):
    patient_id = os.path.basename(patient_path)
    
    volumes_dir = os.path.join(output_dir, 'volumes')
    masks_dir = os.path.join(output_dir, 'masks')
    
    volume_save_path = os.path.join(volumes_dir, f'{patient_id}.npy')
    mask_save_path = os.path.join(masks_dir, f'{patient_id}.npy')
    
    if skip_existing and os.path.exists(volume_save_path) and os.path.exists(mask_save_path):
        try:
            mask = np.load(mask_save_path)
            num_slices = mask.shape[2]
            return True, patient_id, "exists", num_slices
        except:
            return True, patient_id, "exists", None
    
    volume, mask, patient_id = preprocess_single_patient(patient_path, target_size)
    
    if volume is None or mask is None:
        return False, patient_id, "failed", None
    
    try:
        np.save(volume_save_path, volume)
        np.save(mask_save_path, mask)
        num_slices = mask.shape[2]
        return True, patient_id, "success", num_slices
    except Exception as e:
        return False, patient_id, f"error: {str(e)}", None


def preprocess_brats21_dataset(input_dir, output_dir, target_size=(224, 224), 
                               max_patients=None, num_workers=8, skip_existing=True):
    os.makedirs(output_dir, exist_ok=True)
    
    volumes_dir = os.path.join(output_dir, 'volumes')
    masks_dir = os.path.join(output_dir, 'masks')
    os.makedirs(volumes_dir, exist_ok=True)
    os.makedirs(masks_dir, exist_ok=True)
    
    patient_folders = sorted(glob.glob(os.path.join(input_dir, 'BraTS2021_*')))
    
    if max_patients:
        patient_folders = patient_folders[:max_patients]
    
    print(f"Processing {len(patient_folders)} patients ({num_workers} workers)")
    
    process_fn = partial(
        preprocess_and_save_patient,
        output_dir=output_dir,
        target_size=target_size,
        skip_existing=skip_existing
    )
    
    with Pool(num_workers) as pool:
        results = list(tqdm(
            pool.imap(process_fn, patient_folders),
            total=len(patient_folders),
            desc="Preprocessing"
        ))
    
    success_results = [r for r in results if r[0]]
    failed_results = [r for r in results if not r[0]]
    existing_count = len([r for r in success_results if r[2] == "exists"])
    processed_count = len([r for r in success_results if r[2] == "success"])
    
    patient_info = {}
    for success, patient_id, status, num_slices in success_results:
        if num_slices is not None:
            patient_info[patient_id] = {
                'num_slices': int(num_slices),
                'target_size': target_size
            }
    
    print(f"\nResults: {processed_count} processed, {existing_count} skipped, {len(failed_results)} failed")
    
    if failed_results:
        print("Failed patients:")
        for result in failed_results[:5]:
            success, patient_id, error = result[0], result[1], result[2]
            print(f"  {patient_id}: {error}")
        if len(failed_results) > 5:
            print(f"  ... and {len(failed_results) - 5} more")
    
    metadata = {
        'target_size': target_size,
        'total_processed': len(success_results),
        'newly_processed': processed_count,
        'skipped_existing': existing_count,
        'failed': len(failed_results),
        'patient_info': patient_info,
        'modalities': ['t1', 't1ce', 't2', 'flair'],
        'num_classes': 4,
        'num_workers': num_workers
    }
    
    with open(os.path.join(output_dir, 'metadata.json'), 'w') as f:
        json.dump(metadata, f, indent=2)
    
    print(f"Saved metadata ({len(patient_info)} patients)")
    
    return processed_count, len(failed_results)


def get_patient_ids_from_npy(npy_dir):
    volumes_dir = os.path.join(npy_dir, 'volumes')
    volume_files = sorted(glob.glob(os.path.join(volumes_dir, '*.npy')))
    patient_ids = [os.path.basename(f).replace('.npy', '') for f in volume_files]
    return patient_ids


def update_metadata_with_patient_info(npy_dir):
    metadata_path = os.path.join(npy_dir, 'metadata.json')
    masks_dir = os.path.join(npy_dir, 'masks')
    
    if not os.path.exists(metadata_path):
        print(f"Error: No metadata.json found at {metadata_path}")
        return False
    
    with open(metadata_path, 'r') as f:
        metadata = json.load(f)
    
    if 'patient_info' in metadata and metadata['patient_info']:
        print(f"Metadata already has patient_info ({len(metadata['patient_info'])} entries)")
        return True
    
    print("Building patient_info from mask files...")
    patient_info = {}
    mask_files = sorted(glob.glob(os.path.join(masks_dir, '*.npy')))
    
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

