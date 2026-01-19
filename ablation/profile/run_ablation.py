"""
Run All PIE-UNet Profile Ablation Studies

This script runs the complete ablation study:
1. Measures computational metrics (Params, GFLOPs, Latency, RAM)
2. Outputs a comprehensive comparison table
"""

import sys
import argparse
from pathlib import Path
from datetime import datetime

# Add project root to path
PROJECT_ROOT = Path(__file__).parent.parent.parent
sys.path.insert(0, str(PROJECT_ROOT))

from ablation.profile.config import PROFILE_CONFIGS, OUTPUT_CONFIG
from ablation.profile.measure_profile import measure_all_profiles, print_results_table, save_results_csv


def main():
    parser = argparse.ArgumentParser(description="Run PIE-UNet Profile Ablation Study")
    parser.add_argument("--output-dir", type=str, default=None,
                        help="Output directory for results")
    args = parser.parse_args()
    
    output_dir = Path(args.output_dir) if args.output_dir else OUTPUT_CONFIG["results_dir"]
    output_dir.mkdir(parents=True, exist_ok=True)
    
    print("=" * 70)
    print("PIE-UNet Profile Ablation Study")
    print("=" * 70)
    print(f"\nProfiles to evaluate: {list(PROFILE_CONFIGS.keys())}")
    print(f"Output directory: {output_dir}")
    
    # Step 1: Measure computational metrics
    print("\n" + "-" * 70)
    print("Step 1: Measuring Computational Metrics")
    print("-" * 70)
    
    results = measure_all_profiles(n_classes=4)
    
    # Print results table
    print_results_table(results)
    
    # Save results
    timestamp = datetime.now().strftime("%Y%m%d_%H%M%S")
    csv_path = output_dir / f"profile_ablation_{timestamp}.csv"
    save_results_csv(results, csv_path)
    
    # Also save as latest
    latest_path = output_dir / "profile_ablation_latest.csv"
    save_results_csv(results, latest_path)
    
    print("\n" + "=" * 70)
    print("Ablation Study Completed!")
    print("=" * 70)
    print(f"\nResults saved to:")
    print(f"  - {csv_path}")
    print(f"  - {latest_path}")
    
    # Generate markdown table
    md_path = output_dir / "profile_ablation_results.md"
    with open(md_path, 'w') as f:
        f.write("# PIE-UNet Profile Ablation Study Results\n\n")
        f.write(f"Generated: {datetime.now().strftime('%Y-%m-%d %H:%M:%S')}\n\n")
        f.write("## Computational Metrics\n\n")
        f.write("| Profile | C_in | Depth | #Params | GFLOPs | CPU Latency | Peak RAM |\n")
        f.write("|---------|------|-------|---------|--------|-------------|----------|\n")
        for r in results:
            f.write(f"| {r['name']} | {r['n_channels']} | {r['depth']} | "
                    f"{r['params_m']:.2f}M | {r['gflops']:.3f} | "
                    f"{r['cpu_latency_ms']:.2f}ms | {r['peak_ram_mb']:.1f}MB |\n")
        
        f.write("\n## Notes\n\n")
        f.write("- **C_in**: Number of input slices (2.5D context)\n")
        f.write("- **Depth**: Number of encoder levels in UNet++\n")
        f.write("- **CPU Latency**: Average inference time on CPU (100 runs)\n")
        f.write("- **Peak RAM**: Peak memory usage during inference\n")
    
    print(f"  - {md_path}")


if __name__ == "__main__":
    main()
