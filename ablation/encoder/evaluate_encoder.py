"""
Encoder Ablation Evaluation Script

Evaluates trained encoder variants with:
- Computational metrics: #Params, GFLOPs, CPU Latency
- Performance metrics: 3D Dice, 3D HD95 (same as evaluate_acdc.py)

Usage:
    python ablation/encoder/evaluate_encoder.py              # Evaluate all encoders
    python ablation/encoder/evaluate_encoder.py --encoder SE # Evaluate single encoder
"""

import torch
import torch.nn.functional as F
import numpy as np
import os
import sys
import argparse
import csv
import time
from pathlib import Path
from datetime import datetime
from torch.utils.data import DataLoader
import albumentations as A
from albumentations.pytorch import ToTensorV2
from tqdm import tqdm

# Add project root to path
PROJECT_ROOT = Path(__file__).parent.parent.parent
sys.path.insert(0, str(PROJECT_ROOT))

from ablation.encoder.config import ENCODER_CONFIGS, DATA_CONFIG, OUTPUT_CONFIG, MEASURE_CONFIG, TRAINING_CONFIG
from ablation.encoder.pie_unet_encoder import PIE_UNet_Encoder
from src.data_utils.acdc_dataset_optimized import ACDCDataset25DOptimized, get_acdc_volume_ids

# 3D metrics
from medpy import metric

# FLOPs measurement
try:
    from thop import profile, clever_format
    HAS_THOP = True
except ImportError:
    HAS_THOP = False
    print("Warning: thop not installed. GFLOPs will not be measured.")

DEVICE = torch.device('cuda' if torch.cuda.is_available() else 'cpu')
NUM_CLASSES = TRAINING_CONFIG["num_classes"]
ACDC_CLASS_MAP = {0: 'Background', 1: 'Right Ventricle', 2: 'Myocardium', 3: 'Left Ventricle'}


# =============================================================================
# COMPUTATIONAL METRICS
# =============================================================================

def measure_flops(model, input_tensor):
    """Measure MACs, GMACs, GFLOPs using thop."""
    if not HAS_THOP:
        return 0, 0.0, 0.0
    
    model.eval()
    with torch.no_grad():
        macs, params = profile(model, inputs=(input_tensor,), verbose=False)
    
    g_macs = macs / 1e9
    gflops = g_macs * 2  # GFLOPs = MACs * 2
    
    return macs, g_macs, gflops


def get_cpu_info():
    """Get CPU information."""
    import platform
    import multiprocessing
    
    info = {"processor": platform.processor() or "Unknown", "num_cores": multiprocessing.cpu_count()}
    try:
        with open("/proc/cpuinfo", "r") as f:
            for line in f:
                if "model name" in line:
                    info["model_name"] = line.split(":")[1].strip()
                    break
    except:
        pass
    return info


def measure_cpu_latency(model, input_tensor, num_warmup=10, num_runs=100, verbose=False):
    """Measure CPU inference latency with CUDA disabled."""
    import os
    
    original_cuda_visible = os.environ.get('CUDA_VISIBLE_DEVICES', None)
    os.environ['CUDA_VISIBLE_DEVICES'] = ''
    
    num_threads = 4
    torch.set_num_threads(num_threads)
    
    model.eval()
    model_cpu = model.cpu()
    input_cpu = input_tensor.cpu()
    
    if verbose:
        cpu_info = get_cpu_info()
        print(f"  CPU: {cpu_info.get('model_name', cpu_info['processor'])}")
        print(f"  Threads: {num_threads}, Cores: {cpu_info['num_cores']}")
    
    # Warmup
    with torch.no_grad():
        for _ in range(num_warmup):
            _ = model_cpu(input_cpu)
    
    # Measure
    latencies = []
    with torch.no_grad():
        for _ in range(num_runs):
            start = time.perf_counter()
            _ = model_cpu(input_cpu)
            end = time.perf_counter()
            latencies.append((end - start) * 1000)
    
    avg_latency = sum(latencies) / len(latencies)
    std_latency = (sum((l - avg_latency) ** 2 for l in latencies) / len(latencies)) ** 0.5
    
    # Restore CUDA
    if original_cuda_visible is not None:
        os.environ['CUDA_VISIBLE_DEVICES'] = original_cuda_visible
    elif 'CUDA_VISIBLE_DEVICES' in os.environ:
        del os.environ['CUDA_VISIBLE_DEVICES']
    
    return avg_latency, std_latency


# =============================================================================
# 3D VOLUMETRIC EVALUATION
# =============================================================================

def evaluate_3d_volumetric(model, test_dataset, test_loader, device, num_classes=4):
    """Evaluate using TRUE 3D volumetric metrics (same as evaluate_acdc.py)."""
    model.eval()
    volume_data = {}
    
    with torch.no_grad():
        for i_batch, (imgs, tgts) in enumerate(tqdm(test_loader, desc="Inference", leave=False)):
            imgs, tgts = imgs.to(device), tgts.to(device)
            if imgs.size(0) == 0:
                continue
            
            logits_list, _ = model(imgs)
            logits = logits_list[-1]
            preds = torch.argmax(F.softmax(logits, dim=1), dim=1)
            
            batch_size = imgs.size(0)
            for b_idx in range(batch_size):
                global_idx = i_batch * test_loader.batch_size + b_idx
                if global_idx < len(test_dataset):
                    vol_idx, slice_idx = test_dataset.index_map[global_idx]
                    vol_id = os.path.basename(test_dataset.volume_paths[vol_idx]).replace('.npy', '')
                    
                    if vol_id not in volume_data:
                        volume_data[vol_id] = {}
                    
                    volume_data[vol_id][slice_idx] = (preds[b_idx].cpu().numpy(), tgts[b_idx].cpu().numpy())
    
    # Compute 3D metrics
    per_class_dice = {c: [] for c in range(1, num_classes)}
    per_class_hd95 = {c: [] for c in range(1, num_classes)}
    
    for vol_id in tqdm(sorted(volume_data.keys()), desc="3D Metrics", leave=False):
        slices_dict = volume_data[vol_id]
        sorted_indices = sorted(slices_dict.keys())
        
        pred_vol = np.stack([slices_dict[i][0] for i in sorted_indices], axis=0)
        gt_vol = np.stack([slices_dict[i][1] for i in sorted_indices], axis=0)
        
        for c in range(1, num_classes):
            pred_c = (pred_vol == c).astype(int)
            gt_c = (gt_vol == c).astype(int)
            
            if pred_c.sum() > 0 and gt_c.sum() > 0:
                d = metric.binary.dc(pred_c, gt_c)
                h = metric.binary.hd95(pred_c, gt_c)
            elif pred_c.sum() == 0 and gt_c.sum() == 0:
                d, h = 1.0, np.nan
            else:
                d, h = 0.0, np.nan
            
            per_class_dice[c].append(d)
            per_class_hd95[c].append(h)
    
    # Aggregate with std
    all_dice = []
    all_hd95 = []
    for c in range(1, num_classes):
        all_dice.extend([d for d in per_class_dice[c] if not np.isnan(d)])
        all_hd95.extend([h for h in per_class_hd95[c] if not np.isnan(h)])
    
    return {
        'mean_dice': np.nanmean(all_dice) if all_dice else 0.0,
        'std_dice': np.nanstd(all_dice) if all_dice else 0.0,
        'mean_hd95': np.nanmean(all_hd95) if all_hd95 else 0.0,
        'std_hd95': np.nanstd(all_hd95) if all_hd95 else 0.0
    }


# =============================================================================
# EVALUATE SINGLE ENCODER
# =============================================================================

def evaluate_encoder(encoder_name, verbose=True):
    """Evaluate a single encoder with full metrics."""
    
    if encoder_name not in ENCODER_CONFIGS:
        raise ValueError(f"Unknown encoder: {encoder_name}")
    
    config = ENCODER_CONFIGS[encoder_name]
    
    # NAE uses pretrained weights from main training
    if encoder_name == "NAE":
        weights_path = PROJECT_ROOT / "weights" / "best_model_acdc_no_anatomical.pth"
    else:
        weights_path = OUTPUT_CONFIG["weights_dir"] / f"best_{encoder_name}.pth"
    
    if not weights_path.exists():
        print(f"Weights not found: {weights_path}")
        return None
    
    if verbose:
        print(f"\n{'='*70}")
        print(f"EVALUATING: {config['name']}")
        print(f"  Weights: {weights_path}")
        print(f"{'='*70}")
    
    # Load model
    NUM_SLICES = TRAINING_CONFIG["num_slices"]
    model = PIE_UNet_Encoder(
        n_channels=NUM_SLICES,
        n_classes=NUM_CLASSES,
        encoder_type=config["type"],
        deep_supervision=True
    ).to(DEVICE)
    model.load_state_dict(torch.load(weights_path))
    model.eval()
    
    params = sum(p.numel() for p in model.parameters())
    
    # Computational metrics
    dummy_input = torch.randn(1, NUM_SLICES, MEASURE_CONFIG["input_size"], MEASURE_CONFIG["input_size"])
    macs, g_macs, gflops = measure_flops(model, dummy_input.to(DEVICE))
    cpu_lat, cpu_std = measure_cpu_latency(model, dummy_input, verbose=verbose)
    
    if verbose:
        print(f"  Params: {params:,}")
        print(f"  GMACs: {g_macs:.4f}, GFLOPs: {gflops:.4f}")
        print(f"  CPU Latency: {cpu_lat:.2f}±{cpu_std:.2f}ms")
    
    # Reset GPU peak memory
    if torch.cuda.is_available():
        torch.cuda.reset_peak_memory_stats()
    
    # Move back to GPU for evaluation
    model = model.to(DEVICE)
    
    # Load test data
    test_npy_dir = str(DATA_CONFIG["test_dir"])
    test_volume_ids = get_acdc_volume_ids(test_npy_dir)
    
    test_dataset = ACDCDataset25DOptimized(
        npy_dir=test_npy_dir,
        volume_ids=test_volume_ids,
        num_input_slices=NUM_SLICES,
        transforms=A.Compose([ToTensorV2()]),
        max_cache_size=8,
        use_memmap=True
    )
    
    test_loader = DataLoader(test_dataset, batch_size=16, shuffle=False, num_workers=0, pin_memory=True)
    
    if verbose:
        print(f"  Test volumes: {len(test_volume_ids)}, slices: {len(test_dataset)}")
    
    # 3D evaluation
    results_3d = evaluate_3d_volumetric(model, test_dataset, test_loader, DEVICE, NUM_CLASSES)
    
    # Peak GPU
    if torch.cuda.is_available():
        torch.cuda.synchronize()
        peak_gpu_mb = torch.cuda.max_memory_allocated() / (1024 * 1024)
    else:
        peak_gpu_mb = 0.0
    
    if verbose:
        print(f"  Peak GPU: {peak_gpu_mb:.0f}MB")
        print(f"  Dice (3D): {results_3d['mean_dice']:.4f}")
        print(f"  HD95 (3D): {results_3d['mean_hd95']:.4f}")
    
    return {
        "encoder": encoder_name,
        "name": config["name"],
        "type": config["type"],
        "params": params,
        "g_macs": g_macs,
        "gflops": gflops,
        "cpu_latency_ms": cpu_lat,
        "cpu_latency_std": cpu_std,
        "peak_gpu_mb": peak_gpu_mb,
        "test_dice_3d": results_3d['mean_dice'],
        "test_dice_std": results_3d['std_dice'],
        "test_hd95_3d": results_3d['mean_hd95'],
        "test_hd95_std": results_3d['std_hd95']
    }


# =============================================================================
# EVALUATE ALL ENCODERS
# =============================================================================

def evaluate_all_encoders():
    """Evaluate all trained encoder variants."""
    
    print("=" * 100)
    print("ENCODER ABLATION - FULL EVALUATION")
    print("=" * 100)
    
    all_results = []
    
    for encoder_name in ENCODER_CONFIGS.keys():
        result = evaluate_encoder(encoder_name, verbose=True)
        if result:
            all_results.append(result)
    
    if not all_results:
        print("No trained encoders found! Run train_encoder.py first.")
        return []
    
    # Save results
    timestamp = datetime.now().strftime("%Y%m%d_%H%M%S")
    output_dir = OUTPUT_CONFIG["results_dir"]
    
    # CSV
    csv_path = output_dir / f"encoder_ablation_{timestamp}.csv"
    with open(csv_path, 'w', newline='') as f:
        fieldnames = ['name', 'params', 'g_macs', 'gflops', 'cpu_latency_ms', 'cpu_latency_std', 
                      'peak_gpu_mb', 'test_dice_3d', 'test_dice_std', 'test_hd95_3d', 'test_hd95_std']
        writer = csv.DictWriter(f, fieldnames=fieldnames, extrasaction='ignore')
        writer.writeheader()
        writer.writerows(all_results)
    
    # Markdown
    md_path = output_dir / "encoder_ablation_results.md"
    with open(md_path, 'w') as f:
        f.write("# Encoder Ablation Study Results\n\n")
        f.write(f"Generated: {datetime.now().strftime('%Y-%m-%d %H:%M:%S')}\n\n")
        f.write("## Results (3D Volumetric Metrics)\n\n")
        f.write("| Encoder | #Params | GMACs | GFLOPs | Dice (3D) | HD95 (3D) |\n")
        f.write("|---------|---------|-------|--------|-----------|----------|\n")
        for r in all_results:
            f.write(f"| {r['name']} | {r['params']:,} | {r['g_macs']:.4f} | {r['gflops']:.4f} | "
                    f"{r['test_dice_3d']:.4f} | "
                    f"{r['test_hd95_3d']:.4f} |\n")
    
    # Print summary
    print("\n" + "=" * 120)
    print("ENCODER ABLATION COMPLETE")
    print("=" * 120)
    print(f"\n{'Encoder':<20} {'Params':<12} {'GMACs':<10} {'GFLOPs':<10} {'Dice 3D':<10} {'HD95 3D':<10}")
    print("-" * 80)
    for r in all_results:
        print(f"{r['name']:<20} {r['params']:<12,} {r['g_macs']:<10.4f} {r['gflops']:<10.4f} "
              f"{r['test_dice_3d']:<10.4f} {r['test_hd95_3d']:<10.4f}")
    print("=" * 80)
    print(f"\nResults saved to:")
    print(f"  - {csv_path}")
    print(f"  - {md_path}")
    
    return all_results


# =============================================================================
# MAIN
# =============================================================================

if __name__ == "__main__":
    parser = argparse.ArgumentParser(description="Evaluate encoder ablation")
    parser.add_argument("--encoder", type=str, choices=list(ENCODER_CONFIGS.keys()),
                        help="Specific encoder to evaluate (default: all)")
    args = parser.parse_args()
    
    print(f"Device: {DEVICE}")
    
    if args.encoder:
        evaluate_encoder(args.encoder, verbose=True)
    else:
        evaluate_all_encoders()
