import os
import numpy as np
import torch
from typing import List, Optional
from torch.utils.data import Dataset, Sampler
import json
import random
import config

class MONAI25DDataset(Dataset):
    def __init__(self, npy_dir: str, patient_ids: List[str], num_slices_25d: int, 
                 samples_per_patient: Optional[int] = None, random_seed: int = 42):
        self.npy_dir = npy_dir
        self.num_slices_25d = num_slices_25d
        self.radius_25d = num_slices_25d // 2
        self.samples_per_patient = samples_per_patient
        self._mmap_cache = {}  
        
        random.seed(random_seed)
        np.random.seed(random_seed)
        
        metadata_path = os.path.join(npy_dir, 'metadata.json')
        with open(metadata_path, 'r') as f:
            metadata = json.load(f)
        
        patient_info = metadata.get('patient_info', {})
        
        self.samples = []
        patient_sample_groups = []
        
        for pid in patient_ids:
            vol_path = os.path.join(npy_dir, 'volumes', f'{pid}.npy')
            mask_path = os.path.join(npy_dir, 'masks', f'{pid}.npy')
            
            if pid in patient_info:
                num_z = patient_info[pid].get('num_slices', 0)
                valid_slices = list(range(self.radius_25d, num_z - self.radius_25d))
                
                if not valid_slices:
                    continue

                if samples_per_patient is None or samples_per_patient <= 0:
                    sampled_slices = valid_slices
                else:
                    n_samples = min(samples_per_patient, len(valid_slices))
                    sampled_slices = random.sample(valid_slices, n_samples)
                
                patient_group = [(vol_path, mask_path, z) for z in sampled_slices]
                patient_sample_groups.append(patient_group)
        
        for group in patient_sample_groups:
            self.samples.extend(group)
        
        avg_slices_per_patient = len(self.samples) / len(patient_ids) if len(patient_ids) > 0 else 0
        
        print(f"Dataset Initialized: {len(self.samples)} slices from {len(patient_ids)} patients.")
        print(f"  - Sampling: {'ALL slices' if samples_per_patient is None else f'{samples_per_patient} slices/patient'}")
        print(f"  - Augmentation: DISABLED (Handled by GPU)")

    def __len__(self):
        return len(self.samples)
    
    def _get_mmap(self, path):
        if path not in self._mmap_cache:
            self._mmap_cache[path] = np.load(path, mmap_mode='r')
        return self._mmap_cache[path]
    
    def __getitem__(self, idx):
        vol_path, mask_path, center_z = self.samples[idx]
        
        volume = self._get_mmap(vol_path).copy()
        mask = self._get_mmap(mask_path).copy()

        start_z = center_z - self.radius_25d
        end_z = center_z + self.radius_25d + 1
        image_patch_3d = volume[:, :, :, start_z:end_z] # Shape: (4, H, W, num_slices)
        
        # (4, H, W, 5) -> (20, H, W)
        image_chw = image_patch_3d.transpose(0, 3, 1, 2).reshape(-1, config.IMG_SIZE, config.IMG_SIZE)
        mask_hw = mask[:, :, center_z]
        image_tensor = torch.from_numpy(image_chw).float()
        label_tensor = torch.from_numpy(mask_hw).long()
        
        return image_tensor, label_tensor


class CacheLocalitySampler(Sampler):
    def __init__(self, dataset, batch_size, shuffle=True):
        self.dataset = dataset
        self.batch_size = batch_size
        self.shuffle = shuffle
        
        self.patient_groups = {}
        for idx, (vol_path, _, _) in enumerate(dataset.samples):
            if vol_path not in self.patient_groups:
                self.patient_groups[vol_path] = []
            self.patient_groups[vol_path].append(idx)
        
        self.num_samples = len(dataset)
    
    def __iter__(self):
        all_patient_keys = list(self.patient_groups.keys())
        if self.shuffle:
            random.shuffle(all_patient_keys)
        
        indices = []
        for patient_key in all_patient_keys:
            patient_indices = self.patient_groups[patient_key].copy()
            if self.shuffle:
                random.shuffle(patient_indices)
            indices.extend(patient_indices)
        
        return iter(indices)
    
    def __len__(self):
        return self.num_samples


def _worker_init(_):
    """Khởi tạo mỗi worker với cài đặt đơn luồng để tránh xung đột tài nguyên."""
    os.environ["OMP_NUM_THREADS"] = "1"
    os.environ["MKL_NUM_THREADS"] = "1"
    os.environ["OPENBLAS_NUM_THREADS"] = "1"
    os.environ["NUMEXPR_NUM_THREADS"] = "1"
    try:
        torch.set_num_threads(1)
    except Exception:
        pass


def build_monai_persistent_dataset(npy_dir: str, patient_ids: List[str], num_slices_25d: int,
                                   samples_per_patient: Optional[int] = None):
    """Hàm helper để khởi tạo dataset đã được tối ưu."""
    return MONAI25DDataset(npy_dir, patient_ids, num_slices_25d, samples_per_patient)