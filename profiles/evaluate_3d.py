"""
Re-evaluate Ablation Models with 3D Volumetric Metrics

This script re-evaluates saved ablation weights using TRUE 3D volumetric
metrics (same as evaluate_acdc.py) instead of 2D slice-by-slice.

Usage:
    python profile/evaluate_3d.py              # Evaluate all profiles
    python profile/evaluate_3d.py --profile M  # Evaluate single profile
"""

import torch
import torch.nn.functional as F
import numpy as np
import os
import sys
import argparse
from pathlib import Path
from torch.utils.data import DataLoader
import albumentations as A
from albumentations.pytorch import ToTensorV2
from tqdm import tqdm
from datetime import datetime

# Add project root to path
PROJECT_ROOT = Path(__file__).parent.parent.parent
sys.path.insert(0, str(PROJECT_ROOT))

from profiles.config import PROFILE_CONFIGS, DATA_CONFIG, OUTPUT_CONFIG, MEASURE_CONFIG
from profiles.pie_unet import PIE_UNet
from profiles.measure_profile import measure_profile
from src.data_utils.acdc_dataset_optimized import (
    ACDCDataset25DOptimized,
    get_acdc_volume_ids,
)

# 3D metrics
from medpy import metric

DEVICE = torch.device('cuda' if torch.cuda.is_available() else 'cpu')
NUM_CLASSES = 4
ACDC_CLASS_MAP = {0: 'Background', 1: 'Right Ventricle', 2: 'Myocardium', 3: 'Left Ventricle'}


def find_profile_weights(profile_name, weights_dir):
    """
    Auto-detect all weight files for a given profile in the weights directory.
    
    Returns:
        List of tuples: [(weight_path, weight_name), ...]
    """
    weights_dir = Path(weights_dir)
    if not weights_dir.exists():
        return []
    
    # Find all .pth files matching the profile
    pattern = f"best_model_{profile_name}*.pth"
    weight_files = list(weights_dir.glob(pattern))
    
    results = []
    for w_path in sorted(weight_files):
        # Extract weight type from filename
        # E.g., "best_model_XL_dice.pth" -> "dice"
        #       "best_model_XL.pth" -> "default"
        filename = w_path.stem  # Remove .pth
        base_name = f"best_model_{profile_name}"
        
        if filename == base_name:
            weight_name = "default"
        else:
            # Remove base_name and leading underscore
            weight_name = filename[len(base_name)+1:]
        
        results.append((w_path, weight_name))
    
    return results


def evaluate_3d_volumetric(model, test_dataset, test_loader, device, num_classes=4):
    """
    Evaluate using TRUE 3D volumetric metrics (same as evaluate_acdc.py).
    Reconstructs full volumes then computes Dice and HD95.
    """
    model.eval()
    
    # Structure: volume_data[vol_id][slice_idx] = (pred, gt)
    volume_data = {}
    
    with torch.no_grad():
        for i_batch, (imgs, tgts) in enumerate(tqdm(test_loader, desc="Inference", leave=False)):
            imgs, tgts = imgs.to(device), tgts.to(device)
            if imgs.size(0) == 0:
                continue
            
            logits_list, _ = model(imgs)
            logits = logits_list[-1]
            preds = torch.argmax(F.softmax(logits, dim=1), dim=1)
            
            # Accumulate slices by volume
            batch_size = imgs.size(0)
            for b_idx in range(batch_size):
                global_idx = i_batch * test_loader.batch_size + b_idx
                if global_idx < len(test_dataset):
                    vol_idx, slice_idx = test_dataset.index_map[global_idx]
                    vol_id = os.path.basename(test_dataset.volume_paths[vol_idx]).replace('.npy', '')
                    
                    if vol_id not in volume_data:
                        volume_data[vol_id] = {}
                    
                    volume_data[vol_id][slice_idx] = (
                        preds[b_idx].cpu().numpy(),
                        tgts[b_idx].cpu().numpy()
                    )
    
    # Compute 3D metrics for each volume
    per_class_dice = {c: [] for c in range(1, num_classes)}
    per_class_hd95 = {c: [] for c in range(1, num_classes)}
    
    print("Computing 3D Volumetric Metrics...")
    for vol_id in tqdm(sorted(volume_data.keys()), desc="Processing Volumes", leave=False):
        slices_dict = volume_data[vol_id]
        sorted_slice_indices = sorted(slices_dict.keys())
        
        pred_vol_list = []
        gt_vol_list = []
        
        for s_idx in sorted_slice_indices:
            p_slice, g_slice = slices_dict[s_idx]
            pred_vol_list.append(p_slice)
            gt_vol_list.append(g_slice)
        
        pred_vol = np.stack(pred_vol_list, axis=0)  # (D, H, W)
        gt_vol = np.stack(gt_vol_list, axis=0)      # (D, H, W)
        
        # Calculate metric for each foreground class
        for c in range(1, num_classes):
            pred_c = (pred_vol == c).astype(int)
            gt_c = (gt_vol == c).astype(int)
            
            if pred_c.sum() > 0 and gt_c.sum() > 0:
                d = metric.binary.dc(pred_c, gt_c)
                h = metric.binary.hd95(pred_c, gt_c)
            elif pred_c.sum() == 0 and gt_c.sum() == 0:
                d = 1.0
                h = np.nan
            else:
                d = 0.0
                h = np.nan
            
            per_class_dice[c].append(d)
            per_class_hd95[c].append(h)
    
    # Aggregate results with mean and std
    results = {
        'per_class_dice': {},
        'per_class_hd95': {},
        'mean_fg_dice': 0.0,
        'std_fg_dice': 0.0,
        'mean_fg_hd95': 0.0,
        'std_fg_hd95': 0.0
    }
    
    # Collect all foreground dice/hd95 values across all volumes
    all_fg_dice = []
    all_fg_hd95 = []
    
    for c in range(1, num_classes):
        avg_d = np.nanmean(per_class_dice[c])
        avg_h = np.nanmean(per_class_hd95[c])
        
        results['per_class_dice'][c] = avg_d
        results['per_class_hd95'][c] = avg_h
        
        # Collect per-volume values for std calculation
        all_fg_dice.extend([d for d in per_class_dice[c] if not np.isnan(d)])
        all_fg_hd95.extend([h for h in per_class_hd95[c] if not np.isnan(h)])
    
    # Calculate mean and std across all foreground classes and volumes
    results['mean_fg_dice'] = np.nanmean(all_fg_dice) if all_fg_dice else 0.0
    results['std_fg_dice'] = np.nanstd(all_fg_dice) if all_fg_dice else 0.0
    results['mean_fg_hd95'] = np.nanmean(all_fg_hd95) if all_fg_hd95 else 0.0
    results['std_fg_hd95'] = np.nanstd(all_fg_hd95) if all_fg_hd95 else 0.0
    
    return results


def evaluate_profile_3d(profile_name, weights_path=None, weight_name=None, verbose=True):
    """
    Evaluate a single profile with FULL metrics:
    - Computational: Params, MACs, GFLOPs, CPU Latency
    - Peak GPU VRAM during inference
    - 3D Volumetric: Dice, HD95 (same as evaluate_acdc.py)
    
    Args:
        profile_name: Profile to evaluate (T, M, XL, etc.)
        weights_path: Path to weight file (if None, will auto-detect)
        weight_name: Name/identifier for this weight variant
        verbose: Print detailed output
    """
    
    config = PROFILE_CONFIGS[profile_name]
    
    # If no weights_path provided, try to find default
    if weights_path is None:
        detected_weights = find_profile_weights(profile_name, OUTPUT_CONFIG["weights_dir"])
        if not detected_weights:
            print(f"Error: No weights found for profile {profile_name}")
            return None
        weights_path, weight_name = detected_weights[0]  # Use first one
        if verbose:
            print(f"Auto-detected weight: {weights_path}")
    else:
        weights_path = Path(weights_path)
    
    if not weights_path.exists():
        print(f"Error: Weights not found: {weights_path}")
        return None
    
    if verbose:
        print(f"\n{'='*70}")
        print(f"FULL EVALUATION: {config['name']}")
        print(f"  Weights: {weights_path}")
        print(f"{'='*70}")
    
    # =========================================================================
    # STEP 1: Measure computational metrics (Params, MACs, GFLOPs, CPU Latency)
    # =========================================================================
    if verbose:
        print("\n[Step 1] Measuring computational metrics...")
    comp_metrics = measure_profile(profile_name, n_classes=NUM_CLASSES, verbose=verbose)
    
    # =========================================================================
    # STEP 2: Load model and evaluate with Peak GPU VRAM tracking
    # =========================================================================
    if verbose:
        print("\n[Step 2] Running 3D volumetric evaluation...")
    
    # Reset GPU peak memory
    if torch.cuda.is_available():
        torch.cuda.reset_peak_memory_stats()
        torch.cuda.synchronize()
    
    # Load model
    model = PIE_UNet(
        n_channels=config["n_channels"],
        n_classes=NUM_CLASSES,
        depth=config["depth"],
        base_filters=config["base_filters"],
        deep_supervision=True
    ).to(DEVICE)
    
    model.load_state_dict(torch.load(weights_path))
    model.eval()
    
    # Load test dataset
    test_npy_dir = str(DATA_CONFIG["test_dir"])
    test_volume_ids = get_acdc_volume_ids(test_npy_dir)
    
    test_transform = A.Compose([ToTensorV2()])
    
    test_dataset = ACDCDataset25DOptimized(
        npy_dir=test_npy_dir,
        volume_ids=test_volume_ids,
        num_input_slices=config["n_channels"],
        transforms=test_transform,
        max_cache_size=8,
        use_memmap=True
    )
    
    test_loader = DataLoader(
        test_dataset,
        batch_size=16,
        shuffle=False,
        num_workers=0,
        pin_memory=True
    )
    
    if verbose:
        print(f"  Test volumes: {len(test_volume_ids)}")
        print(f"  Test slices: {len(test_dataset)}")
    
    # Run 3D evaluation
    results = evaluate_3d_volumetric(model, test_dataset, test_loader, DEVICE, NUM_CLASSES)
    
    # Get Peak GPU VRAM
    if torch.cuda.is_available():
        torch.cuda.synchronize()
        peak_gpu_mb = torch.cuda.max_memory_allocated() / (1024 * 1024)
    else:
        peak_gpu_mb = 0.0
    
    if verbose:
        print(f"\n  Peak GPU VRAM: {peak_gpu_mb:.0f} MB")
        print(f"\n  Per-Class 3D Metrics:")
        print(f"  {'Class':<25} | {'Dice':<10} | {'HD95':<10}")
        print(f"  {'-'*55}")
        for c in range(1, NUM_CLASSES):
            class_name = f"{ACDC_CLASS_MAP[c]}"
            d = results['per_class_dice'][c]
            h = results['per_class_hd95'][c]
            hd_str = f"{h:.4f}" if not np.isnan(h) else "NaN"
            print(f"  {class_name:<25} | {d:.4f}     | {hd_str:<10}")
        print(f"  {'-'*55}")
        print(f"  {'Foreground Mean':<25} | {results['mean_fg_dice']:.4f}     | {results['mean_fg_hd95']:.4f}")
    
    return {
        "profile": profile_name,
        "name": config["name"],
        "weight_file": str(weights_path.name),
        "weight_name": weight_name if weight_name else "unknown",
        "n_channels": config["n_channels"],
        "depth": config["depth"],
        # Computational metrics
        "params": comp_metrics["params"],  # Full number
        "params_m": comp_metrics["params_m"],
        "g_macs": comp_metrics["g_macs"],
        "gflops": comp_metrics["gflops"],
        "cpu_latency_ms": comp_metrics["cpu_latency_ms"],
        "cpu_latency_std": comp_metrics["cpu_latency_std"],
        # Peak GPU VRAM (during evaluation)
        "peak_gpu_mb": peak_gpu_mb,
        # 3D Volumetric metrics with std
        "test_dice_3d": results['mean_fg_dice'],
        "test_dice_std": results['std_fg_dice'],
        "test_hd95_3d": results['mean_fg_hd95'],
        "test_hd95_std": results['std_fg_hd95'],
        "per_class_dice": results['per_class_dice'],
        "per_class_hd95": results['per_class_hd95']
    }


def evaluate_all_profiles_3d():
    """Re-evaluate all profiles with FULL metrics for ALL weight variants found in directory."""
    
    print("=" * 100)
    print("FULL ABLATION EVALUATION (Computational + 3D Volumetric) - AUTO-DETECTING WEIGHTS")
    print("=" * 100)
    
    all_results = []
    
    for profile_name in PROFILE_CONFIGS.keys():
        # Auto-detect all weights for this profile
        detected_weights = find_profile_weights(profile_name, OUTPUT_CONFIG["weights_dir"])
        
        if not detected_weights:
            print(f"\n⚠️  No weights found for profile {profile_name}, skipping...")
            continue
        
        print(f"\n{'='*100}")
        print(f"Profile {profile_name}: Found {len(detected_weights)} weight file(s)")
        print(f"{'='*100}")
        
        for weights_path, weight_name in detected_weights:
            print(f"\n  → Evaluating: {weights_path.name} ({weight_name})")
            result = evaluate_profile_3d(profile_name, weights_path=weights_path, weight_name=weight_name, verbose=True)
            if result:
                all_results.append(result)
    
    # Save results
    timestamp = datetime.now().strftime("%Y%m%d_%H%M%S")
    output_dir = OUTPUT_CONFIG["results_dir"]
    
    # CSV with all fields including weight info
    import csv
    csv_path = output_dir / f"ablation_full_3d_{timestamp}.csv"
    with open(csv_path, 'w', newline='') as f:
        fieldnames = ['name', 'weight_name', 'weight_file', 'n_channels', 'depth', 'params', 'g_macs', 'gflops', 
                      'cpu_latency_ms', 'cpu_latency_std', 'peak_gpu_mb', 
                      'test_dice_3d', 'test_dice_std', 'test_hd95_3d', 'test_hd95_std']
        writer = csv.DictWriter(f, fieldnames=fieldnames, extrasaction='ignore')
        writer.writeheader()
        writer.writerows(all_results)
    
    # Markdown with full format including weight info
    md_path = output_dir / "ablation_full_3d_results.md"
    with open(md_path, 'w') as f:
        f.write("# PIE-UNet Ablation - FULL Evaluation Results (All Detected Weights)\n\n")
        f.write(f"Generated: {datetime.now().strftime('%Y-%m-%d %H:%M:%S')}\n\n")
        f.write("## Computational Metrics + 3D Volumetric Accuracy\n\n")
        f.write("| Profile | Weight | Weight File | C_in | Depth | Params | G-MACs | GFLOPs | CPU Latency | Peak GPU | Dice (3D) | HD95 (3D) |\n")
        f.write("|---------|--------|-------------|------|-------|--------|--------|--------|-------------|----------|-----------|----------|\n")
        for r in all_results:
            f.write(f"| {r['name']} | {r['weight_name']} | {r['weight_file']} | {r['n_channels']} | {r['depth']} | "
                    f"{r['params']:,} | {r['g_macs']:.4f} | {r['gflops']:.4f} | "
                    f"{r['cpu_latency_ms']:.2f}±{r['cpu_latency_std']:.2f}ms | {r['peak_gpu_mb']:.0f}MB | "
                    f"{r['test_dice_3d']:.4f}±{r['test_dice_std']:.4f} | {r['test_hd95_3d']:.4f}±{r['test_hd95_std']:.4f} |\n")
    
    # Print summary with weight info
    print("\n" + "=" * 170)
    print("ABLATION STUDY COMPLETE - FULL RESULTS (ALL DETECTED WEIGHTS)")
    print("=" * 170)
    print(f"\n{'Profile':<14} {'Weight':<12} {'Weight File':<30} {'C_in':<5} {'Depth':<6} {'Params':<12} {'G-MACs':<10} {'GFLOPs':<10} "
          f"{'CPU Latency':<16} {'Peak GPU':<10} {'Dice 3D':<16} {'HD95 3D':<16}")
    print("-" * 170)
    for r in all_results:
        print(f"{r['name']:<14} {r['weight_name']:<12} {r['weight_file']:<30} {r['n_channels']:<5} {r['depth']:<6} "
              f"{r['params']:<12,} {r['g_macs']:<10.4f} {r['gflops']:<10.4f} "
              f"{r['cpu_latency_ms']:.2f}±{r['cpu_latency_std']:.2f}ms{'':<4} {r['peak_gpu_mb']:.0f}MB{'':<6} "
              f"{r['test_dice_3d']:.4f}±{r['test_dice_std']:.4f}{'':<4} {r['test_hd95_3d']:.4f}±{r['test_hd95_std']:.4f}")
    print("=" * 170)
    print(f"\nResults saved to:")
    print(f"  - {csv_path}")
    print(f"  - {md_path}")
    
    return all_results


if __name__ == "__main__":
    parser = argparse.ArgumentParser(description="Re-evaluate ablation models with 3D metrics")
    parser.add_argument("--profile", type=str, choices=list(PROFILE_CONFIGS.keys()),
                        help="Specific profile to evaluate (default: all profiles)")
    parser.add_argument("--weights-path", type=str,
                        help="Specific weight file path to evaluate")
    parser.add_argument("--weights-dir", type=str,
                        help="Directory containing weight files (default: profile/weights)")
    args = parser.parse_args()
    
    print(f"Device: {DEVICE}")
    
    # Override weights directory if specified
    if args.weights_dir:
        OUTPUT_CONFIG["weights_dir"] = Path(args.weights_dir)
        print(f"Using weights directory: {OUTPUT_CONFIG['weights_dir']}")
    
    if args.profile:
        # Evaluate specific profile
        if args.weights_path:
            # Specific weight file provided
            weight_name = Path(args.weights_path).stem
            evaluate_profile_3d(args.profile, weights_path=args.weights_path, weight_name=weight_name, verbose=True)
        else:
            # Auto-detect all weights for this profile
            detected_weights = find_profile_weights(args.profile, OUTPUT_CONFIG["weights_dir"])
            if not detected_weights:
                print(f"Error: No weights found for profile {args.profile}")
            else:
                print(f"\nFound {len(detected_weights)} weight file(s) for profile {args.profile}:")
                for i, (w_path, w_name) in enumerate(detected_weights, 1):
                    print(f"  {i}. {w_path.name} ({w_name})")
                print("\nEvaluating all detected weights...\n")
                for w_path, w_name in detected_weights:
                    evaluate_profile_3d(args.profile, weights_path=w_path, weight_name=w_name, verbose=True)
    else:
        # Evaluate all profiles with all detected weights
        evaluate_all_profiles_3d()
