#!/usr/bin/env python3
"""
Find optimal batch size for your system
Tests different batch sizes to find the best balance between speed and memory usage
"""
import os
import sys
import time
import torch
import gc
import numpy as np
from torch.utils.data import DataLoader
import psutil

# Add parent directory to path for imports
sys.path.append(os.path.dirname(os.path.dirname(os.path.abspath(__file__))))

from monai_dataset import build_monai_persistent_dataset, CacheLocalitySampler, _worker_init
from data_utils import get_patient_ids_from_npy
import config

def get_system_info():
    """Get system resource information"""
    cpu_count = os.cpu_count() or 4
    gpu_name = torch.cuda.get_device_name(0) if torch.cuda.is_available() else 'None'
    gpu_memory = torch.cuda.get_device_properties(0).total_memory / 1024**3 if torch.cuda.is_available() else 0
    ram_gb = psutil.virtual_memory().total / 1024**3
    
    print("=" * 70)
    print("SYSTEM INFORMATION")
    print("=" * 70)
    print(f"CPU Cores: {cpu_count}")
    print(f"GPU: {gpu_name}")
    print(f"GPU Memory: {gpu_memory:.1f} GB")
    print(f"RAM: {ram_gb:.1f} GB")
    print("=" * 70 + "\n")

def get_memory_usage():
    """Get current memory usage"""
    if torch.cuda.is_available():
        gpu_memory = torch.cuda.memory_allocated() / 1024**3
        gpu_cached = torch.cuda.memory_reserved() / 1024**3
        return gpu_memory, gpu_cached
    return 0, 0

def benchmark_batch_size(batch_size, num_epochs=2, max_batches=30):
    """Benchmark training with specific batch size"""
    print(f"\n{'='*70}")
    print(f"TESTING: Batch Size {batch_size}")
    print(f"{'='*70}")
    
    npy_dir = config.MONAI_NPY_DIR
    patient_ids = get_patient_ids_from_npy(npy_dir)[:30]  # Test with 30 patients
    
    try:
        dataset = build_monai_persistent_dataset(
            npy_dir=npy_dir,
            patient_ids=patient_ids,
            num_slices_25d=config.NUM_SLICES,
            samples_per_patient=10
        )
        
        sampler = CacheLocalitySampler(dataset, batch_size, shuffle=True)
        
        dataloader = DataLoader(
            dataset,
            batch_size=batch_size,
            sampler=sampler,
            num_workers=config.DATA_NUM_WORKERS,
            pin_memory=True,
            persistent_workers=True,
            prefetch_factor=config.DATA_PREFETCH_FACTOR,
            worker_init_fn=_worker_init
        )
        
        # Warm-up
        print("Warming up...")
        for i, (images, masks) in enumerate(dataloader):
            if i >= 3:
                break
        
        # Clear memory before testing
        if torch.cuda.is_available():
            torch.cuda.empty_cache()
        
        # Benchmark multiple epochs
        epoch_times = []
        memory_usage = []
        throughputs = []
        
        for epoch in range(num_epochs):
            print(f"\nEpoch {epoch + 1}/{num_epochs}")
            
            start_time = time.time()
            batch_times = []
            
            for batch_idx, (images, masks) in enumerate(dataloader):
                batch_start = time.time()
                
                # Move to GPU and simulate training
                if torch.cuda.is_available():
                    images = images.cuda()
                    masks = masks.cuda()
                    
                    # Simulate model forward pass (simple operations)
                    _ = images.mean()
                    _ = masks.sum()
                    
                    # Check memory usage
                    gpu_mem, gpu_cached = get_memory_usage()
                    memory_usage.append(gpu_mem)
                
                batch_time = time.time() - batch_start
                batch_times.append(batch_time)
                
                if batch_idx >= max_batches:  # Limit batches for faster testing
                    break
            
            epoch_time = time.time() - start_time
            epoch_times.append(epoch_time)
            
            avg_batch_time = np.mean(batch_times)
            throughput = batch_size / avg_batch_time
            throughputs.append(throughput)
            
            print(f"  Time: {epoch_time:.2f}s")
            print(f"  Avg batch time: {avg_batch_time:.3f}s")
            print(f"  Throughput: {throughput:.1f} images/s")
            if torch.cuda.is_available():
                print(f"  GPU Memory: {gpu_mem:.2f} GB")
        
        # Results
        avg_epoch_time = np.mean(epoch_times)
        std_epoch_time = np.std(epoch_times)
        avg_throughput = np.mean(throughputs)
        max_memory = max(memory_usage) if memory_usage else 0
        
        print(f"\n{'─'*70}")
        print(f"RESULTS for Batch Size {batch_size}:")
        print(f"  ├─ Avg epoch time:   {avg_epoch_time:.2f}s ± {std_epoch_time:.2f}s")
        print(f"  ├─ Avg throughput:   {avg_throughput:.1f} images/s")
        print(f"  ├─ Max GPU memory:   {max_memory:.2f} GB")
        print(f"  └─ Stability:        {'✅ GOOD' if std_epoch_time < 1.0 else '⚠️  POOR'}")
        
        # Check if we're running out of memory
        gpu_total = torch.cuda.get_device_properties(0).total_memory / 1024**3 if torch.cuda.is_available() else 0
        memory_usage_ratio = max_memory / gpu_total if gpu_total > 0 else 0
        
        if memory_usage_ratio > 0.9:
            print(f"  ⚠️  WARNING: High memory usage ({memory_usage_ratio:.1%})")
            return None  # Skip this batch size if too much memory
        
        # Clean up
        del dataloader, dataset, sampler
        gc.collect()
        if torch.cuda.is_available():
            torch.cuda.empty_cache()
        
        return {
            'batch_size': batch_size,
            'avg_time': avg_epoch_time,
            'std_time': std_epoch_time,
            'throughput': avg_throughput,
            'max_memory': max_memory,
            'memory_ratio': memory_usage_ratio,
            'stability': std_epoch_time < 1.0
        }
        
    except RuntimeError as e:
        if "out of memory" in str(e).lower():
            print(f"❌ Out of memory with batch size {batch_size}")
            return None
        else:
            print(f"❌ Error with batch size {batch_size}: {e}")
            return None
    except Exception as e:
        print(f"❌ Error with batch size {batch_size}: {e}")
        return None

def find_optimal_batch_size():
    """Find the optimal batch size"""
    print("🔍 Finding optimal batch size...")
    print("Testing different batch sizes to find the best balance\n")
    
    # Test different batch sizes
    # Start with powers of 2, then add some intermediate values
    batch_sizes = [1, 2, 4, 8, 16, 24, 32, 48, 64, 96, 128]
    
    # If GPU memory is limited, start with smaller sizes
    if torch.cuda.is_available():
        gpu_memory_gb = torch.cuda.get_device_properties(0).total_memory / 1024**3
        if gpu_memory_gb < 8:  # Less than 8GB GPU memory
            batch_sizes = [1, 2, 4, 8, 12, 16, 24, 32, 48, 64]
        elif gpu_memory_gb < 12:  # Less than 12GB GPU memory
            batch_sizes = [1, 2, 4, 8, 16, 24, 32, 48, 64, 96]
    
    results = []
    
    for batch_size in batch_sizes:
        result = benchmark_batch_size(batch_size, num_epochs=2, max_batches=20)
        if result is not None:
            results.append(result)
        
        # Small delay between tests
        time.sleep(1)
    
    return results

def print_summary(results):
    """Print summary of results"""
    if not results:
        print("\n❌ No successful tests completed. Check your data and configuration.")
        return
    
    print("\n" + "="*70)
    print("SUMMARY")
    print("="*70)
    print(f"{'Batch Size':<12} {'Avg Time':<12} {'Throughput':<12} {'GPU Memory':<12} {'Stability'}")
    print("-"*70)
    
    for r in results:
        stability = "✅ GOOD" if r['stability'] else "⚠️  POOR"
        print(f"{r['batch_size']:<12} {r['avg_time']:>8.2f}s    {r['throughput']:>8.1f} img/s  {r['max_memory']:>8.2f} GB    {stability}")
    
    # Find optimal batch size based on different criteria
    if results:
        # Best throughput
        best_throughput = max(results, key=lambda x: x['throughput'])
        
        # Best time (fastest)
        best_time = min(results, key=lambda x: x['avg_time'])
        
        # Best memory efficiency (good throughput with reasonable memory)
        memory_efficient = [r for r in results if r['memory_ratio'] < 0.8 and r['stability']]
        if memory_efficient:
            best_memory = max(memory_efficient, key=lambda x: x['throughput'])
        else:
            best_memory = min(results, key=lambda x: x['memory_ratio'])
        
        # Overall best (balanced)
        # Score based on throughput, stability, and memory efficiency
        for r in results:
            throughput_score = r['throughput'] / max(r['throughput'] for r in results)
            memory_score = 1 - r['memory_ratio']  # Lower memory usage is better
            stability_score = 1 if r['stability'] else 0.5
            r['overall_score'] = throughput_score * 0.4 + memory_score * 0.3 + stability_score * 0.3
        
        best_overall = max(results, key=lambda x: x['overall_score'])
        
        print("\n" + "="*70)
        print("RECOMMENDATIONS")
        print("="*70)
        print(f"🚀 Best Throughput:     {best_throughput['batch_size']} ( {best_throughput['throughput']:.1f} img/s )")
        print(f"⚡ Fastest Training:    {best_time['batch_size']} ( {best_time['avg_time']:.2f}s/epoch )")
        print(f"💾 Memory Efficient:    {best_memory['batch_size']} ( {best_memory['max_memory']:.2f} GB )")
        print(f"⭐ Overall Best:        {best_overall['batch_size']} ( balanced )")
        
        print("\n" + "="*70)
        print("CONFIGURATION UPDATE")
        print("="*70)
        print(f"Update config.py:")
        print(f"   BATCH_SIZE = {best_overall['batch_size']}")
        
        # Additional recommendations
        print(f"\n💡 Additional Tips:")
        if best_overall['memory_ratio'] > 0.7:
            print(f"   - Consider reducing batch size if you encounter OOM errors")
        if not best_overall['stability']:
            print(f"   - Consider using gradient accumulation for more stable training")
        if best_overall['batch_size'] < 16:
            print(f"   - Small batch size detected - consider using batch normalization")
        
        print("="*70)

if __name__ == "__main__":
    get_system_info()
    
    results = find_optimal_batch_size()
    print_summary(results)
