"""
Run Full Ablation Study: Training + Metrics for All Profiles

This script trains all 5 PIE-UNet profiles and collects:
- Computational metrics: Params, GFLOPs, CPU Latency, Peak RAM
- Accuracy metrics: DICE, HD95 (from training)
"""

import sys
import os
import argparse
import csv
from pathlib import Path
from datetime import datetime

# Add project root to path
PROJECT_ROOT = Path(__file__).parent.parent.parent
sys.path.insert(0, str(PROJECT_ROOT))

from ablation.profile.config import PROFILE_CONFIGS, OUTPUT_CONFIG, TRAINING_CONFIG
from ablation.profile.measure_profile import measure_profile
from ablation.profile.train_profile import train_profile


def run_full_ablation(epochs=None, quick_test=False):
    """Run complete ablation study for all profiles."""
    
    if epochs is None:
        epochs = 5 if quick_test else TRAINING_CONFIG["num_epochs"]
    
    print("=" * 70)
    print("FULL PIE-UNet ABLATION STUDY")
    print("=" * 70)
    print(f"Profiles: {list(PROFILE_CONFIGS.keys())}")
    print(f"Training epochs: {epochs}")
    print(f"Quick test: {quick_test}")
    print("=" * 70)
    
    all_results = []
    
    for profile_name in PROFILE_CONFIGS.keys():
        print(f"\n{'#'*70}")
        print(f"# PROFILE: {profile_name}")
        print(f"{'#'*70}")
        
        # Step 1: Measure computational metrics
        print("\n[Step 1] Measuring computational metrics...")
        comp_metrics = measure_profile(profile_name, n_classes=4, verbose=True)
        
        # Step 2: Train and get accuracy metrics
        print("\n[Step 2] Training model...")
        train_result = train_profile(profile_name, num_epochs=epochs, quick_test=False)
        
        # Combine results (use TEST metrics, not validation)
        result = {
            "profile": profile_name,
            "name": PROFILE_CONFIGS[profile_name]["name"],
            "n_channels": PROFILE_CONFIGS[profile_name]["n_channels"],
            "depth": PROFILE_CONFIGS[profile_name]["depth"],
            "params": comp_metrics["params"],
            "params_m": comp_metrics["params_m"],
            "g_macs": comp_metrics["g_macs"],
            "gflops": comp_metrics["gflops"],
            "cpu_latency_ms": comp_metrics["cpu_latency_ms"],
            "peak_gpu_mb": train_result["peak_gpu_memory_mb"],  # From training, not static measurement
            "val_dice": train_result["best_val_dice"],
            "test_dice": train_result["test_dice"],
            "test_hd95": train_result["test_hd95"]
        }
        
        all_results.append(result)
        
        # Print summary for this profile
        print(f"\n{'='*50}")
        print(f"PROFILE {profile_name} COMPLETE")
        print(f"  Params: {result['params_m']:.2f}M")
        print(f"  G-MACs: {result['g_macs']:.3f}, GFLOPs: {result['gflops']:.3f}")
        print(f"  Test DICE: {result['test_dice']:.4f}")
        print(f"  Test HD95: {result['test_hd95']:.2f}")
        print(f"{'='*50}")
    
    # Save final results
    output_dir = OUTPUT_CONFIG["results_dir"]
    timestamp = datetime.now().strftime("%Y%m%d_%H%M%S")
    
    # CSV
    csv_path = output_dir / f"full_ablation_{timestamp}.csv"
    with open(csv_path, 'w', newline='') as f:
        writer = csv.DictWriter(f, fieldnames=all_results[0].keys())
        writer.writeheader()
        writer.writerows(all_results)
    
    # Latest CSV
    latest_path = output_dir / "full_ablation_latest.csv"
    with open(latest_path, 'w', newline='') as f:
        writer = csv.DictWriter(f, fieldnames=all_results[0].keys())
        writer.writeheader()
        writer.writerows(all_results)
    
    # Markdown report
    md_path = output_dir / "full_ablation_results.md"
    with open(md_path, 'w') as f:
        f.write("# PIE-UNet Full Ablation Study Results\n\n")
        f.write(f"Generated: {datetime.now().strftime('%Y-%m-%d %H:%M:%S')}\n\n")
        f.write(f"Training epochs: {epochs} (with early stopping)\n\n")
        f.write("## Results (Evaluated on TEST SET)\n\n")
        f.write("| Profile | C_in | Depth | #Params | G-MACs | GFLOPs | Peak GPU | Test DICE | Test HD95 |\n")
        f.write("|---------|------|-------|---------|--------|--------|----------|-----------|----------|\n")
        for r in all_results:
            f.write(f"| {r['name']} | {r['n_channels']} | {r['depth']} | "
                    f"{r['params_m']:.2f}M | {r['g_macs']:.3f} | {r['gflops']:.3f} | "
                    f"{r['peak_gpu_mb']:.0f}MB | {r['test_dice']:.4f} | {r['test_hd95']:.2f} |\n")
    
    # Print final summary
    print("\n" + "=" * 110)
    print("ABLATION STUDY COMPLETE!")
    print("=" * 110)
    print(f"\n{'Profile':<12} {'C_in':<6} {'Depth':<6} {'Params':<10} {'G-MACs':<8} {'GFLOPs':<8} {'Peak GPU':<10} {'Test DICE':<10} {'Test HD95':<10}")
    print("-" * 110)
    for r in all_results:
        print(f"{r['name']:<12} {r['n_channels']:<6} {r['depth']:<6} "
              f"{r['params_m']:.2f}M{'':<4} {r['g_macs']:<8.3f} {r['gflops']:<8.3f} "
              f"{r['peak_gpu_mb']:.0f}MB{'':<4} {r['test_dice']:<10.4f} {r['test_hd95']:<10.2f}")
    print("=" * 110)
    print(f"\nResults saved to:")
    print(f"  - {csv_path}")
    print(f"  - {md_path}")
    
    return all_results


if __name__ == "__main__":
    parser = argparse.ArgumentParser(description="Run full PIE-UNet ablation study")
    parser.add_argument("--epochs", type=int, default=None, help="Number of training epochs (default: from config)")
    parser.add_argument("--quick-test", action="store_true", help="Quick test mode (5 epochs)")
    args = parser.parse_args()
    
    run_full_ablation(epochs=args.epochs, quick_test=args.quick_test)
