#!/usr/bin/env python3
"""
Find optimal number of workers for your system
"""
import os
import sys
import time
import torch
import gc
import numpy as np
from torch.utils.data import DataLoader

# Add parent directory to path for imports
sys.path.append(os.path.dirname(os.path.dirname(os.path.abspath(__file__))))

from monai_dataset import build_monai_persistent_dataset, CacheLocalitySampler, _worker_init
from data_utils import get_patient_ids_from_npy
import config

def get_system_info():
    """Get system resource information"""
    cpu_count = os.cpu_count() or 4
    
    print("=" * 60)
    print("SYSTEM INFORMATION")
    print("=" * 60)
    print(f"CPU Cores: {cpu_count}")
    print(f"GPU: {torch.cuda.get_device_name(0) if torch.cuda.is_available() else 'None'}")
    print("=" * 60 + "\n")

def benchmark_workers(num_workers, num_epochs=2):
    """Benchmark training with specific number of workers"""
    print(f"\n{'='*60}")
    print(f"TESTING: {num_workers} workers")
    print(f"{'='*60}")
    
    npy_dir = config.MONAI_NPY_DIR
    patient_ids = get_patient_ids_from_npy(npy_dir)[:50]  # Test with 50 patients
    
    # Remove the transforms parameter - it's not supported by build_monai_persistent_dataset
    dataset = build_monai_persistent_dataset(
        npy_dir=npy_dir,
        patient_ids=patient_ids,
        num_slices_25d=config.NUM_SLICES,
        samples_per_patient=10
    )
    
    sampler = CacheLocalitySampler(dataset, config.BATCH_SIZE, shuffle=True)
    
    dataloader = DataLoader(
        dataset,
        batch_size=config.BATCH_SIZE,
        sampler=sampler,
        num_workers=num_workers,
        pin_memory=True,
        persistent_workers=(num_workers > 0),
        prefetch_factor=2 if num_workers > 0 else None,
        worker_init_fn=_worker_init if num_workers > 0 else None
    )
    
    # Warm-up
    print("Warming up...")
    for i, (images, masks) in enumerate(dataloader):
        if i >= 3:
            break
    
    # Benchmark multiple epochs
    epoch_times = []
    
    for epoch in range(num_epochs):
        print(f"\nEpoch {epoch + 1}/{num_epochs}")
        
        start_time = time.time()
        batch_times = []
        
        for batch_idx, (images, masks) in enumerate(dataloader):
            batch_start = time.time()
            
            # Simulate some GPU work
            if torch.cuda.is_available():
                images = images.cuda()
                masks = masks.cuda()
                _ = images.mean()  # Simple operation
            
            batch_time = time.time() - batch_start
            batch_times.append(batch_time)
            
            if batch_idx >= 50:  # Test 50 batches per epoch
                break
        
        epoch_time = time.time() - start_time
        epoch_times.append(epoch_time)
        
        avg_batch_time = np.mean(batch_times)
        throughput = config.BATCH_SIZE / avg_batch_time
        
        print(f"  Time: {epoch_time:.2f}s")
        print(f"  Avg batch time: {avg_batch_time:.3f}s")
        print(f"  Throughput: {throughput:.1f} images/s")
    
    # Results
    avg_epoch_time = np.mean(epoch_times)
    std_epoch_time = np.std(epoch_times)
    
    print(f"\n{'─'*60}")
    print(f"RESULTS for {num_workers} workers:")
    print(f"  ├─ Avg epoch time:   {avg_epoch_time:.2f}s ± {std_epoch_time:.2f}s")
    print(f"  └─ Stability:        {'✅ GOOD' if std_epoch_time < 0.5 else '⚠️  POOR (slowdown detected!)'}")
    
    # Clean up
    del dataloader, dataset, sampler
    gc.collect()
    if torch.cuda.is_available():
        torch.cuda.empty_cache()
    
    return {
        'num_workers': num_workers,
        'avg_time': avg_epoch_time,
        'std_time': std_epoch_time
    }

if __name__ == "__main__":
    get_system_info()
    
    # Test different worker counts
    worker_counts = [2, 4, 6, 8, 10, 12, 16]
    results = []
    
    print("🔍 Finding optimal worker count...")
    print("Testing with augmentation to simulate real training\n")
    
    for num_workers in worker_counts:
        try:
            result = benchmark_workers(num_workers, num_epochs=2)
            results.append(result)
        except Exception as e:
            print(f"\n❌ Error with {num_workers} workers: {e}")
            continue
    
    # Summary
    print("\n" + "="*60)
    print("SUMMARY")
    print("="*60)
    print(f"{'Workers':<10} {'Avg Time':<15} {'Std Dev':<15} {'Stability'}")
    print("-"*60)
    
    for r in results:
        stability = "✅ GOOD" if r['std_time'] < 0.5 else "⚠️  POOR"
        print(f"{r['num_workers']:<10} {r['avg_time']:>6.2f}s        {r['std_time']:>6.2f}s        {stability}")
    
    # Recommendation
    if results:  # Check if we have any results
        # Prioritize: fast AND stable
        stable_results = [r for r in results if r['std_time'] < 0.5]
        if stable_results:
            best = min(stable_results, key=lambda x: x['avg_time'])
            print("\n" + "="*60)
            print(f"✅ RECOMMENDED: {best['num_workers']} workers")
            print(f"   (Fastest stable time: {best['avg_time']:.2f}s ± {best['std_time']:.2f}s)")
        else:
            best = min(results, key=lambda x: x['avg_time'])
            print("\n" + "="*60)
            print(f"⚠️  RECOMMENDED: {best['num_workers']} workers")
            print(f"   (Fastest but unstable: {best['avg_time']:.2f}s ± {best['std_time']:.2f}s)")
            print(f"   Warning: All configurations show instability!")
        
        print("="*60)
        
        print(f"\n💡 Update config.py:")
        print(f"   DATA_NUM_WORKERS = {best['num_workers']}")
    else:
        print("\n❌ No successful tests completed. Check your data and configuration.")
    
    print("="*60)