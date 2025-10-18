import os
import numpy as np
import torch
from typing import List, Optional
from torch.utils.data import Dataset, Sampler
import json
import random


class MONAI25DDataset(Dataset):
    """Ultra-optimized MONAI 2.5D dataset with smart batching and zero-copy operations"""
    
    def __init__(self, npy_dir: str, patient_ids: List[str], num_slices_25d: int, 
                 samples_per_patient: Optional[int] = None, random_seed: int = 42, 
                 smart_batching: bool = True, transforms=None):
        self.npy_dir = npy_dir
        self.num_slices = num_slices_25d
        self.radius = num_slices_25d // 2
        self.samples_per_patient = samples_per_patient
        self._mmap_cache = {}  
        self.smart_batching = smart_batching
        self.transforms = transforms
        
        random.seed(random_seed)
        np.random.seed(random_seed)
        
        metadata_path = os.path.join(npy_dir, 'metadata.json')
        with open(metadata_path, 'r') as f:
            metadata = json.load(f)
        
        patient_slices = metadata.get('patient_slices', {})
        
        # Build samples list (grouped by patient for cache locality)
        self.samples = []
        patient_sample_groups = []
        
        for pid in patient_ids:
            vol_path = os.path.join(npy_dir, 'volumes', f'{pid}.npy')
            mask_path = os.path.join(npy_dir, 'masks', f'{pid}.npy')
            
            if pid in patient_slices:
                num_z = patient_slices[pid]
                valid_slices = list(range(self.radius, num_z - self.radius))
                
                if len(valid_slices) > 0:
                    # If samples_per_patient is None or <= 0, use ALL valid slices
                    if samples_per_patient is None or samples_per_patient <= 0:
                        sampled_slices = valid_slices
                    else:
                        # Random sample up to samples_per_patient
                        n_samples = min(samples_per_patient, len(valid_slices))
                        sampled_slices = random.sample(valid_slices, n_samples)
                    
                    patient_group = []
                    for z in sampled_slices:
                        patient_group.append((vol_path, mask_path, z))
                    patient_sample_groups.append(patient_group)
        
        # Smart batching: group consecutive samples from same patient
        if smart_batching:
            # Flatten but maintain patient locality
            for group in patient_sample_groups:
                self.samples.extend(group)
        else:
            # Random shuffle (old behavior)
            for group in patient_sample_groups:
                self.samples.extend(group)
            random.shuffle(self.samples)
        
        avg_slices_per_patient = len(self.samples) / len(patient_ids) if len(patient_ids) > 0 else 0
        
        print(f"MONAI 2.5D Dataset (Ultra-optimized): {len(self.samples)} slices from {len(patient_ids)} patients")
        if samples_per_patient is None or samples_per_patient <= 0:
            print(f"  - Sampling mode: FULL DATASET (all valid slices, avg {avg_slices_per_patient:.1f}/patient)")
        else:
            print(f"  - Sampling mode: Random {samples_per_patient} slices/patient (avg {avg_slices_per_patient:.1f}/patient)")
        print(f"  - Smart batching: {'ENABLED (cache-friendly)' if smart_batching else 'DISABLED'}")
        print(f"  - Memory mapping: ENABLED")
    
    def __len__(self):
        return len(self.samples)
    
    def _get_mmap(self, path):
        """Get or create memory-mapped array (zero-copy)"""
        if path not in self._mmap_cache:
            self._mmap_cache[path] = np.load(path, mmap_mode='r')
        return self._mmap_cache[path]
    
    def __getitem__(self, idx):
        vol_path, mask_path, center_z = self.samples[idx]
        
        # Memory-mapped arrays (OS handles caching)
        volume = self._get_mmap(vol_path)
        mask = self._get_mmap(mask_path)
        
        # Pre-compute indices (avoid redundant calculations)
        z_start = max(0, center_z - self.radius)
        z_end = min(volume.shape[-1], center_z + self.radius + 1)
        
        # Extract slices: (C, H, W, num_slices)
        image_slices = volume[:, :, :, z_start:z_end]
        
        # Efficient reshape: minimize memory operations
        # (C, H, W, nz) -> (C, nz, H, W) -> (C*nz, H, W)
        C, H, W, nz = image_slices.shape
        
        # Convert to float32
        if image_slices.dtype != np.float32:
            image_slices = image_slices.astype(np.float32)
        
        # Transpose and reshape: (C, H, W, nz) -> (C, nz, H, W) -> (C*nz, H, W)
        image_2d = image_slices.transpose(0, 3, 1, 2).reshape(C * nz, H, W)
        
        # Extract label slice
        label_2d = mask[:, :, center_z]
        
        # Apply transforms if provided
        if self.transforms:
            # Convert to albumentations format: (C, H, W) -> (H, W, C)
            image_hwc = image_2d.transpose(1, 2, 0)  # (H, W, C*nz)
            
            # Apply augmentation
            augmented = self.transforms(image=image_hwc, mask=label_2d)
            image_tensor = augmented['image']  # Already a tensor from ToTensorV2
            label_tensor = augmented['mask']   # Already a tensor
        else:
            # No transforms - direct conversion
            image_tensor = torch.from_numpy(image_2d.copy())
            label_tensor = torch.from_numpy(label_2d.astype(np.int64).copy())
        
        return image_tensor, label_tensor.long()


class CacheLocalitySampler(Sampler):
    """Smart sampler that groups indices by patient for better cache hits"""
    
    def __init__(self, dataset, batch_size, shuffle=True):
        self.dataset = dataset
        self.batch_size = batch_size
        self.shuffle = shuffle
        
        # Group indices by patient (same vol_path)
        self.patient_groups = {}
        for idx, (vol_path, _, _) in enumerate(dataset.samples):
            if vol_path not in self.patient_groups:
                self.patient_groups[vol_path] = []
            self.patient_groups[vol_path].append(idx)
        
        self.num_samples = len(dataset)
    
    def __iter__(self):
        # Create batches that maximize cache locality
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
    """Initialize each worker with single-threaded settings to avoid thread contention"""
    # Set single thread for each worker to prevent thread competition
    os.environ["OMP_NUM_THREADS"] = "1"
    os.environ["MKL_NUM_THREADS"] = "1"
    os.environ["OPENBLAS_NUM_THREADS"] = "1"
    os.environ["NUMEXPR_NUM_THREADS"] = "1"
    try:
        torch.set_num_threads(1)
    except Exception:
        pass


def build_monai_persistent_dataset(npy_dir: str, patient_ids: List[str], num_slices_25d: int,
                                   cache_dir: str = "./monai_cache",
                                   crop_hw: int = 224,
                                   samples_per_patient: Optional[int] = None,
                                   transforms=None):
    return MONAI25DDataset(npy_dir, patient_ids, num_slices_25d, samples_per_patient, transforms=transforms)
