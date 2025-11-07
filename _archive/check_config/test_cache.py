#!/usr/bin/env python3
"""
Quick test script to check cache performance for MONAI dataset
"""
import os
import sys
import time
import numpy as np
from torch.utils.data import DataLoader
import albumentations as A
from albumentations.pytorch import ToTensorV2

# Add parent directory to path for imports
sys.path.append(os.path.dirname(os.path.dirname(os.path.abspath(__file__))))

# Import from project
from monai_dataset import build_monai_persistent_dataset
from data_utils import get_patient_ids_from_npy
import config

def test_monai_mmap_cache():
    """Test and display MONAI mmap cache performance"""
    print("=" * 60)
    print("MONAI MMAP CACHE PERFORMANCE TEST")
    print("=" * 60)
    
    # Use MONAI preprocessed directory
    npy_dir = './BraTS21_preprocessed_monai'
    
    if not os.path.exists(npy_dir):
        print(f"ERROR: MONAI data directory not found: {npy_dir}")
        print(f"Run: python monai_preprocess.py")
        sys.exit(1)
    
    patient_ids = get_patient_ids_from_npy(npy_dir)
    print(f"Found {len(patient_ids)} patients in {npy_dir}")
    
    # Use a subset for testing
    test_patients = patient_ids[:30]  # Test with first 30 patients
    
    # Test different sampling modes
    first_test = True
    for samples_per_patient in [5, 10, None]:
        mode_name = f"{samples_per_patient} samples/patient" if samples_per_patient else "ALL slices"
        print(f"\n{'='*60}")
        print(f"Testing with: {mode_name}")
        print(f"{'='*60}")
        
        transform = A.Compose([ToTensorV2()])
        
        dataset = build_monai_persistent_dataset(
            npy_dir=npy_dir,
            patient_ids=test_patients,
            num_slices_25d=config.NUM_SLICES,
            samples_per_patient=samples_per_patient,
            transforms=transform
        )
        
        dataloader = DataLoader(
            dataset, 
            batch_size=config.BATCH_SIZE,
            shuffle=True,
            num_workers=2,
            pin_memory=True,
            persistent_workers=True
        )
        
        print(f"\nDataset: {len(dataset)} slices from {len(test_patients)} patients")
        print(f"Batches: {len(dataloader)}")
        
        # Warm-up for first test (persistent workers initialization)
        if first_test:
            print("Warming up workers...")
            for batch_idx, (images, masks) in enumerate(dataloader):
                if batch_idx >= 2:  # Just 2 batches for warmup
                    break
            print("Warm-up complete. Starting timed test...")
            first_test = False
        
        # Run through one epoch (timed)
        start_time = time.time()
        batches_processed = 0
        
        for batch_idx, (images, masks) in enumerate(dataloader):
            batches_processed += 1
            if batch_idx % 50 == 0:
                print(f"  Batch {batch_idx:3d}: Processing... (shape: {images.shape})")
        
        elapsed = time.time() - start_time
        
        # Stats
        print(f"\n  RESULTS:")
        print(f"  ├─ Time:             {elapsed:.2f}s ({len(dataset)/elapsed:.1f} samples/s)")
        print(f"  ├─ Throughput:       {batches_processed*config.BATCH_SIZE/elapsed:.1f} images/s")
        print(f"  ├─ Batches:          {batches_processed}")
        print(f"  └─ Avg time/batch:   {elapsed/batches_processed:.3f}s")
        
        # Note about mmap with workers
        print(f"\n  ℹ️  Note: With num_workers>0, each worker has its own mmap cache.")
        print(f"     OS page cache handles memory efficiently across workers.")

def test_dataloader_speed():
    """Test dataloader speed with different worker configurations"""
    print("\n" + "=" * 60)
    print("DATALOADER WORKER PERFORMANCE TEST")
    print("=" * 60)
    
    npy_dir = './BraTS21_preprocessed_monai'
    patient_ids = get_patient_ids_from_npy(npy_dir)[:20]
    transform = A.Compose([ToTensorV2()])
    
    dataset = build_monai_persistent_dataset(
        npy_dir=npy_dir,
        patient_ids=patient_ids,
        num_slices_25d=config.NUM_SLICES,
        samples_per_patient=10,
        transforms=transform
    )
    
    for num_workers in [0, 2, 4]:
        print(f"\n--- num_workers={num_workers} ---")
        
        dataloader = DataLoader(
            dataset,
            batch_size=config.BATCH_SIZE,
            shuffle=False,
            num_workers=num_workers,
            pin_memory=True,
            persistent_workers=(num_workers > 0)
        )
        
        start = time.time()
        for batch_idx, (images, masks) in enumerate(dataloader):
            if batch_idx >= 50:  # Test first 50 batches
                break
        elapsed = time.time() - start
        
        batches_processed = min(50, len(dataloader))
        print(f"  Time for {batches_processed} batches: {elapsed:.3f}s ({elapsed/batches_processed:.3f}s per batch)")
        print(f"  Throughput: {batches_processed*config.BATCH_SIZE/elapsed:.1f} images/s")

if __name__ == "__main__":
    print("\n🔍 Testing MONAI cache system...\n")
    
    try:
        test_monai_mmap_cache()
        test_dataloader_speed()
        
        print("\n" + "=" * 60)
        print("✓ MONAI cache test completed!")
        print("=" * 60 + "\n")
        
    except KeyboardInterrupt:
        print("\n\nTest interrupted by user.")
    except Exception as e:
        print(f"\n❌ Error during test: {e}")
        import traceback
        traceback.print_exc()

