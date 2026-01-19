"""
Maxwell Solver Ablation Evaluation Script

Evaluates with delta metrics: ΔParams, ΔGFLOPs, Dice, HD95.
Standard variant is the baseline (delta = 0).

Usage:
    python ablation/maxwell_solver/evaluate_maxwell.py
"""

import torch
import torch.nn.functional as F
import numpy as np
import os
import sys
import csv
import time
from pathlib import Path
from datetime import datetime
from torch.utils.data import DataLoader
import albumentations as A
from albumentations.pytorch import ToTensorV2
from tqdm import tqdm

PROJECT_ROOT = Path(__file__).parent.parent.parent
sys.path.insert(0, str(PROJECT_ROOT))

from ablation.maxwell_solver.config import MAXWELL_CONFIGS, DATA_CONFIG, OUTPUT_CONFIG, MEASURE_CONFIG, TRAINING_CONFIG, PRETRAINED_WEIGHTS
from ablation.maxwell_solver.pie_unet_maxwell import PIE_UNet_Maxwell
from src.data_utils.acdc_dataset_optimized import ACDCDataset25DOptimized, get_acdc_volume_ids

from medpy import metric

try:
    from thop import profile
    HAS_THOP = True
except ImportError:
    HAS_THOP = False

DEVICE = torch.device('cuda' if torch.cuda.is_available() else 'cpu')
NUM_CLASSES = TRAINING_CONFIG["num_classes"]


def measure_flops(model, input_tensor):
    """Measure MACs, GMACs, GFLOPs."""
    if not HAS_THOP:
        return 0, 0.0, 0.0
    model.eval()
    with torch.no_grad():
        macs, _ = profile(model, inputs=(input_tensor,), verbose=False)
    g_macs = macs / 1e9
    gflops = g_macs * 2
    return macs, g_macs, gflops


def evaluate_3d_volumetric(model, test_dataset, test_loader, device, num_classes=4):
    """3D volumetric evaluation (same as evaluate_acdc.py)."""
    model.eval()
    volume_data = {}
    
    with torch.no_grad():
        for i_batch, (imgs, tgts) in enumerate(tqdm(test_loader, desc="Inference", leave=False)):
            imgs, tgts = imgs.to(device), tgts.to(device)
            if imgs.size(0) == 0:
                continue
            
            logits_list, _ = model(imgs)
            preds = torch.argmax(F.softmax(logits_list[-1], dim=1), dim=1)
            
            for b_idx in range(imgs.size(0)):
                global_idx = i_batch * test_loader.batch_size + b_idx
                if global_idx < len(test_dataset):
                    vol_idx, slice_idx = test_dataset.index_map[global_idx]
                    vol_id = os.path.basename(test_dataset.volume_paths[vol_idx]).replace('.npy', '')
                    if vol_id not in volume_data:
                        volume_data[vol_id] = {}
                    volume_data[vol_id][slice_idx] = (preds[b_idx].cpu().numpy(), tgts[b_idx].cpu().numpy())
    
    per_class_dice = {c: [] for c in range(1, num_classes)}
    per_class_hd95 = {c: [] for c in range(1, num_classes)}
    
    for vol_id in tqdm(sorted(volume_data.keys()), desc="3D Metrics", leave=False):
        slices_dict = volume_data[vol_id]
        sorted_idx = sorted(slices_dict.keys())
        pred_vol = np.stack([slices_dict[i][0] for i in sorted_idx], axis=0)
        gt_vol = np.stack([slices_dict[i][1] for i in sorted_idx], axis=0)
        
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
    
    all_dice = [d for c in range(1, num_classes) for d in per_class_dice[c] if not np.isnan(d)]
    all_hd95 = [h for c in range(1, num_classes) for h in per_class_hd95[c] if not np.isnan(h)]
    
    return {
        'mean_dice': np.nanmean(all_dice) if all_dice else 0.0,
        'std_dice': np.nanstd(all_dice) if all_dice else 0.0,
        'mean_hd95': np.nanmean(all_hd95) if all_hd95 else 0.0,
        'std_hd95': np.nanstd(all_hd95) if all_hd95 else 0.0
    }


def evaluate_variant(variant_name, verbose=True):
    """Evaluate a single Maxwell variant."""
    
    config = MAXWELL_CONFIGS[variant_name]
    
    # Weights path
    if variant_name == "Physics" and variant_name in PRETRAINED_WEIGHTS:
        weights_path = PRETRAINED_WEIGHTS[variant_name]
    else:
        weights_path = OUTPUT_CONFIG["weights_dir"] / f"best_{variant_name}.pth"
    
    if not weights_path.exists():
        print(f"Weights not found: {weights_path}")
        return None
    
    if verbose:
        print(f"\n{'='*70}")
        print(f"EVALUATING: {config['name']}")
        print(f"  use_maxwell: {config['use_maxwell']}")
        print(f"  Weights: {weights_path}")
        print(f"{'='*70}")
    
    NUM_SLICES = TRAINING_CONFIG["num_slices"]
    model = PIE_UNet_Maxwell(n_channels=NUM_SLICES, n_classes=NUM_CLASSES, 
                             use_maxwell=config['use_maxwell'], deep_supervision=True).to(DEVICE)
    model.load_state_dict(torch.load(weights_path))
    model.eval()
    
    params = sum(p.numel() for p in model.parameters())
    
    # FLOPs
    dummy = torch.randn(1, NUM_SLICES, MEASURE_CONFIG["input_size"], MEASURE_CONFIG["input_size"])
    macs, g_macs, gflops = measure_flops(model, dummy.to(DEVICE))
    
    if verbose:
        print(f"  Params: {params:,}")
        print(f"  GMACs: {g_macs:.4f}, GFLOPs: {gflops:.4f}")
    
    # 3D Evaluation
    test_npy_dir = str(DATA_CONFIG["test_dir"])
    test_ids = get_acdc_volume_ids(test_npy_dir)
    
    test_dataset = ACDCDataset25DOptimized(test_npy_dir, test_ids, NUM_SLICES, 
                                           A.Compose([ToTensorV2()]), max_cache_size=8, use_memmap=True)
    test_loader = DataLoader(test_dataset, batch_size=16, shuffle=False, num_workers=0, pin_memory=True)
    
    results = evaluate_3d_volumetric(model, test_dataset, test_loader, DEVICE, NUM_CLASSES)
    
    if verbose:
        print(f"  Dice (3D): {results['mean_dice']:.4f}±{results['std_dice']:.4f}")
        print(f"  HD95 (3D): {results['mean_hd95']:.4f}±{results['std_hd95']:.4f}")
    
    return {
        "variant": variant_name,
        "name": config["name"],
        "use_maxwell": config["use_maxwell"],
        "params": params,
        "g_macs": g_macs,
        "gflops": gflops,
        "dice": results['mean_dice'],
        "dice_std": results['std_dice'],
        "hd95": results['mean_hd95'],
        "hd95_std": results['std_hd95']
    }


def evaluate_all():
    """Evaluate all variants with delta metrics."""
    
    print("=" * 80)
    print("MAXWELL SOLVER ABLATION EVALUATION")
    print("=" * 80)
    
    results = []
    for name in MAXWELL_CONFIGS.keys():
        r = evaluate_variant(name, verbose=True)
        if r:
            results.append(r)
    
    if not results:
        print("No results!")
        return []
    
    # Find baseline (Standard)
    baseline = next((r for r in results if r['variant'] == 'Standard'), results[0])
    
    # Calculate deltas (ΔGFLOPs = GMACs_Physics - GMACs_Standard)
    for r in results:
        r['delta_params'] = r['params'] - baseline['params']
        r['delta_gmacs'] = r['g_macs'] - baseline['g_macs']
        r['delta_gflops'] = r['delta_gmacs']  # Just the GMACs difference
    
    # Save results
    output_dir = OUTPUT_CONFIG["results_dir"]
    timestamp = datetime.now().strftime("%Y%m%d_%H%M%S")
    
    # CSV
    csv_path = output_dir / f"maxwell_ablation_{timestamp}.csv"
    with open(csv_path, 'w', newline='') as f:
        fieldnames = ['name', 'params', 'delta_params', 'gflops', 'delta_gflops', 'dice', 'dice_std', 'hd95', 'hd95_std']
        writer = csv.DictWriter(f, fieldnames=fieldnames, extrasaction='ignore')
        writer.writeheader()
        writer.writerows(results)
    
    # Markdown
    md_path = output_dir / "maxwell_ablation_results.md"
    with open(md_path, 'w') as f:
        f.write("# Maxwell Solver Ablation Results\n\n")
        f.write(f"Generated: {datetime.now().strftime('%Y-%m-%d %H:%M:%S')}\n\n")
        f.write("## Results\n\n")
        f.write("| Maxwell Solver | ΔParams | ΔGFLOPs | DICE | HD95 |\n")
        f.write("|----------------|---------|---------|------|------|\n")
        for r in results:
            delta_p = f"+{r['delta_params']:,}" if r['delta_params'] > 0 else str(r['delta_params'])
            delta_g = f"+{r['delta_gflops']:.4f}" if r['delta_gflops'] > 0 else f"{r['delta_gflops']:.4f}"
            f.write(f"| {r['name']} | {delta_p} | {delta_g} | "
                    f"{r['dice']:.4f}±{r['dice_std']:.4f} | {r['hd95']:.4f}±{r['hd95_std']:.4f} |\n")
    
    # Print summary
    print("\n" + "=" * 100)
    print("MAXWELL SOLVER ABLATION COMPLETE")
    print("=" * 100)
    print(f"\n{'Maxwell Solver':<25} {'ΔParams':<15} {'ΔGFLOPs':<12} {'DICE':<18} {'HD95':<18}")
    print("-" * 100)
    for r in results:
        delta_p = f"+{r['delta_params']:,}" if r['delta_params'] > 0 else str(r['delta_params'])
        delta_g = f"+{r['delta_gflops']:.4f}" if r['delta_gflops'] > 0 else f"{r['delta_gflops']:.4f}"
        print(f"{r['name']:<25} {delta_p:<15} {delta_g:<12} "
              f"{r['dice']:.4f}±{r['dice_std']:.4f}{'':<4} {r['hd95']:.4f}±{r['hd95_std']:.4f}")
    print("=" * 100)
    print(f"\nResults saved to: {md_path}")
    
    return results


if __name__ == "__main__":
    print(f"Device: {DEVICE}")
    evaluate_all()
