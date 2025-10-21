import os
import numpy as np
import torch
from typing import List, Optional
from torch.utils.data import Dataset, Sampler
import json
import random
import config

# MONAI imports for 3D augmentation
from monai.transforms.compose import Compose
from monai.transforms.spatial.dictionary import RandFlipd, RandAffined
from monai.transforms.intensity.dictionary import (
    RandScaleIntensityd, RandShiftIntensityd, RandGaussianNoised,
    RandGibbsNoised, RandBiasFieldd, NormalizeIntensityd
)
from monai.transforms.croppad.dictionary import RandCropByPosNegLabeld
from monai.transforms.utility.dictionary import Lambdad


class MONAI25DDataset(Dataset):
    def __init__(self, npy_dir: str, patient_ids: List[str], num_slices_25d: int, 
                 samples_per_patient: Optional[int] = None, random_seed: int = 42, 
                 smart_batching: bool = True, transforms=None, use_3d_augmentation: bool = True):
        self.npy_dir = npy_dir
        self.num_slices_25d = num_slices_25d
        self.radius_25d = num_slices_25d // 2
        self.samples_per_patient = samples_per_patient
        self._mmap_cache = {}  
        self.smart_batching = smart_batching
        self.transforms = transforms
        self.use_3d_augmentation = use_3d_augmentation
        
        random.seed(random_seed)
        np.random.seed(random_seed)
        
        # Define online transforms (3D augmentation + 2.5D cropping)
        self.online_transforms = self._create_online_transforms()
        
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
                valid_slices = list(range(self.radius_25d, num_z - self.radius_25d))
                
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
        print(f"  - 3D Augmentation: {'ENABLED' if use_3d_augmentation else 'DISABLED'}")
    
    def _create_online_transforms(self):
        """Create online transforms for 3D augmentation + 2.5D cropping"""
        transforms_list = []
        
        if self.use_3d_augmentation:
            # 3D Augmentation transforms
            transforms_list.extend([
                RandFlipd(keys=["image", "label"], prob=0.5, spatial_axis=0),
                RandFlipd(keys=["image", "label"], prob=0.5, spatial_axis=1),
                RandFlipd(keys=["image", "label"], prob=0.3, spatial_axis=2),
                RandAffined(
                    keys=["image", "label"], 
                    prob=0.5, 
                    rotate_range=(np.pi/18, np.pi/18, np.pi/18),  # ±10 degrees
                    scale_range=(0.1, 0.1, 0.1),  # ±10% scaling
                    translate_range=(10, 10, 5),   # Translation
                    mode=("bilinear", "nearest"),
                    padding_mode="zeros"
                ),
                RandScaleIntensityd(keys="image", factors=0.1, prob=0.5),
                RandShiftIntensityd(keys="image", offsets=0.1, prob=0.5),
                RandGaussianNoised(keys="image", prob=0.3, std=0.01),
                RandGibbsNoised(keys="image", prob=0.3, alpha=(0.4, 0.8)),
                RandBiasFieldd(keys="image", prob=0.4, degree=3),
            ])
        
        # 2.5D Cropping
        transforms_list.append(
            RandCropByPosNegLabeld(
                keys=["image", "label"], 
                label_key="label",
                spatial_size=(config.IMG_SIZE, config.IMG_SIZE, self.num_slices_25d),
                num_samples=1  # Only need 1 sample for each __getitem__ call
            )
        )
        
        # Reshape 2.5D -> 2D
        transforms_list.extend([
            Lambdad(keys="image", func=lambda x: x.reshape(-1, x.shape[1], x.shape[2])),
            Lambdad(keys="label", func=lambda x: x[:, :, :, self.radius_25d]),  # Take center slice
        ])
        
        return Compose(transforms_list)
    
    def __len__(self):
        return len(self.samples)
    
    def _get_mmap(self, path):
        """Get or create memory-mapped array (zero-copy)"""
        if path not in self._mmap_cache:
            self._mmap_cache[path] = np.load(path, mmap_mode='r')
        return self._mmap_cache[path]
    
    def __getitem__(self, idx):
        vol_path, mask_path, center_z = self.samples[idx]
        
        # 1. Load 3D volume and mask (memory-mapped)
        volume = self._get_mmap(vol_path)
        mask = self._get_mmap(mask_path)
        
        # 2. Prepare data for MONAI transforms
        data = {"image": volume, "label": mask}
        
        # 3. Apply online transforms (3D augmentation + 2.5D cropping + reshape)
        try:
            transformed_data = self.online_transforms(data)
            image_tensor = transformed_data["image"]
            label_tensor = transformed_data["label"].long()  
        except Exception as e:
            print(f"Error applying online transform for index {idx}: {e}")
            # Return empty data or skip problematic sample
            return torch.zeros((config.NUM_SLICES * 4, config.IMG_SIZE, config.IMG_SIZE)), torch.zeros((config.IMG_SIZE, config.IMG_SIZE)).long()
        
        if self.transforms and not self.use_3d_augmentation:
            image_np_hwc = image_tensor.numpy().transpose(1, 2, 0)
            label_np_hw = label_tensor.numpy().squeeze()  # Remove channel dim if exists
            
            # Apply 2D augmentation
            augmented = self.transforms(image=image_np_hwc, mask=label_np_hw)
            image_tensor = augmented['image']  # From ToTensorV2()
            label_tensor = augmented['mask'].long()  # From ToTensorV2()
        
        return image_tensor, label_tensor


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
                                   transforms=None, use_3d_augmentation: bool = True):
    return MONAI25DDataset(npy_dir, patient_ids, num_slices_25d, samples_per_patient, 
                          transforms=transforms, use_3d_augmentation=use_3d_augmentation)


def create_optimized_datasets(npy_dir: str, train_patient_ids: List[str], val_patient_ids: List[str], 
                             num_slices_25d: int = 5, samples_per_patient: Optional[int] = None):
    # Training dataset: 3D augmentation only (recommended)
    train_dataset = MONAI25DDataset(
        npy_dir=npy_dir,
        patient_ids=train_patient_ids,
        num_slices_25d=num_slices_25d,
        samples_per_patient=samples_per_patient,
        use_3d_augmentation=True,  # 3D augmentation enabled
        transforms=None,  # No 2D augmentation to avoid over-augmentation
        smart_batching=True
    )
    
    # Validation dataset: No augmentation
    val_dataset = MONAI25DDataset(
        npy_dir=npy_dir,
        patient_ids=val_patient_ids,
        num_slices_25d=num_slices_25d,
        samples_per_patient=samples_per_patient,
        use_3d_augmentation=False,  # No 3D augmentation
        transforms=None,  # No 2D augmentation
        smart_batching=True
    )
    
    print("✅ Optimized datasets created:")
    print(f"  - Training: {len(train_dataset)} samples with 3D augmentation")
    print(f"  - Validation: {len(val_dataset)} samples without augmentation")
    print("  - Augmentation strategy: 3D-only (recommended)")
    
    return train_dataset, val_dataset
