#!/usr/bin/env python3
"""
Quick benchmark script để test data loading performance
Chạy: python benchmark_dataloader.py
"""

import os
import sys
import torch
import time
import numpy as np
from torch.utils.data import DataLoader

# Add parent directory to path for imports
sys.path.append(os.path.dirname(os.path.dirname(os.path.abspath(__file__))))

import config
from monai_dataset import build_monai_persistent_dataset, CacheLocalitySampler
from data_utils import get_patient_ids_from_npy
from sklearn.model_selection import train_test_split

print("=" * 70)
print("DATA LOADING BENCHMARK")
print("=" * 70)

# Setup
DEVICE = torch.device('cuda' if torch.cuda.is_available() else 'cpu')
BATCH_SIZE = config.BATCH_SIZE
NUM_SLICES = config.NUM_SLICES
npy_dir = config.MONAI_NPY_DIR

print(f"Device: {DEVICE}")
print(f"Batch size: {BATCH_SIZE}")
print(f"Data directory: {npy_dir}")

# Load patient IDs
patient_ids = get_patient_ids_from_npy(npy_dir)
train_val_ids, test_ids = train_test_split(patient_ids, test_size=0.15, random_state=42)
train_ids, val_ids = train_test_split(train_val_ids, test_size=0.176, random_state=42)

print(f"Using {len(train_ids)} training patients")

# Build dataset
samples_per_patient = getattr(config, 'MONAI_SAMPLES_PER_PATIENT', 10)
train_dataset = build_monai_persistent_dataset(
    npy_dir=npy_dir,
    patient_ids=train_ids,
    num_slices_25d=NUM_SLICES,
    samples_per_patient=samples_per_patient
)

# Create sampler and dataloader
train_sampler = CacheLocalitySampler(train_dataset, BATCH_SIZE, shuffle=True)
NUM_WORKERS = getattr(config, 'DATA_NUM_WORKERS', 4)
PREFETCH_FACTOR = getattr(config, 'DATA_PREFETCH_FACTOR', 2)

print(f"\nDataLoader config:")
print(f"  - Workers: {NUM_WORKERS}")
print(f"  - Prefetch factor: {PREFETCH_FACTOR}")
print(f"  - Total samples: {len(train_dataset)}")
print(f"  - Batches per epoch: {len(train_dataset) // BATCH_SIZE}")

train_dataloader = DataLoader(
    train_dataset,
    batch_size=BATCH_SIZE,
    sampler=train_sampler,
    num_workers=NUM_WORKERS,
    pin_memory=True,
    persistent_workers=True if NUM_WORKERS > 0 else False,
    prefetch_factor=PREFETCH_FACTOR if NUM_WORKERS > 0 else None
)

print("\n" + "=" * 70)
print("BENCHMARK: First 50 batches (includes warmup)")
print("=" * 70)

times = {
    'data_load': [],
    'gpu_transfer': [],
    'total': []
}

iter_start = time.time()
for i, (images, targets) in enumerate(train_dataloader):
    if i >= 50:
        break
    
    data_time = time.time() - iter_start
    times['data_load'].append(data_time)
    
    # Transfer to GPU
    transfer_start = time.time()
    images = images.to(DEVICE)
    targets = targets.to(DEVICE)
    transfer_time = time.time() - transfer_start
    times['gpu_transfer'].append(transfer_time)
    
    total_time = time.time() - iter_start
    times['total'].append(total_time)
    
    if i % 10 == 0:
        print(f"Batch {i:3d}: Data={data_time:.4f}s, Transfer={transfer_time:.4f}s, Total={total_time:.4f}s")
    
    iter_start = time.time()

print("\n" + "=" * 70)
print("RESULTS (excluding first 10 warmup batches)")
print("=" * 70)

# Exclude warmup batches
warmup = 10
data_times = times['data_load'][warmup:]
transfer_times = times['gpu_transfer'][warmup:]
total_times = times['total'][warmup:]

avg_data = np.mean(data_times)
avg_transfer = np.mean(transfer_times)
avg_total = np.mean(total_times)

print(f"\nAverage per iteration:")
print(f"  Data loading:  {avg_data:.4f}s ({avg_data/avg_total*100:.1f}%)")
print(f"  GPU transfer:  {avg_transfer:.4f}s ({avg_transfer/avg_total*100:.1f}%)")
print(f"  Total:         {avg_total:.4f}s")

print(f"\nEstimates:")
total_batches = len(train_dataset) // BATCH_SIZE
estimated_data_time = avg_total * total_batches
print(f"  Time per epoch (data only): {estimated_data_time:.1f}s ({estimated_data_time/60:.1f} min)")
print(f"  Samples per second:         {BATCH_SIZE / avg_total:.1f}")

# Assuming model forward+backward takes ~0.1s (typical)
model_time = 0.1
estimated_total = (avg_total + model_time) * total_batches
print(f"\n  Estimated epoch time (with model): {estimated_total:.1f}s ({estimated_total/60:.1f} min)")
print(f"    (assuming forward+backward ~{model_time}s)")

print("\n" + "=" * 70)
print("INTERPRETATION")
print("=" * 70)

if avg_data / avg_total > 0.5:
    print("⚠️  Data loading is BOTTLENECK (>50% of time)")
    print("   Suggestions:")
    print("   - Try reducing DATA_NUM_WORKERS to 2 or 3")
    print("   - Check disk I/O: run 'iostat -x 2' in another terminal")
    print("   - Verify mmap is working (check code)")
elif avg_data / avg_total > 0.3:
    print("⚡ Data loading is ACCEPTABLE (30-50% of time)")
    print("   Minor improvements possible:")
    print("   - Fine-tune DATA_NUM_WORKERS")
    print("   - Consider reducing DATA_PREFETCH_FACTOR to 1")
else:
    print("✅ Data loading is OPTIMAL (<30% of time)")
    print("   GPU is the bottleneck - this is ideal!")
    print("   Consider optimizing model or increasing batch size")

print("\n" + "=" * 70)

