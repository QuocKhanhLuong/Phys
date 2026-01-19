"""
Run Full Encoder Ablation Study

Trains and evaluates all 5 encoder variants.

Usage:
    python ablation/encoder/run_encoder_ablation.py
    python ablation/encoder/run_encoder_ablation.py --epochs 50  # Quick test
"""

import sys
import argparse
from pathlib import Path
from datetime import datetime

PROJECT_ROOT = Path(__file__).parent.parent.parent
sys.path.insert(0, str(PROJECT_ROOT))

from ablation.encoder.config import ENCODER_CONFIGS, OUTPUT_CONFIG
from ablation.encoder.train_encoder import train_encoder
from ablation.encoder.evaluate_encoder import evaluate_encoder, evaluate_all_encoders


def run_full_encoder_ablation(epochs=None, skip_training=False):
    """Run complete encoder ablation study."""
    
    print("=" * 80)
    print("FULL ENCODER ABLATION STUDY")
    print("=" * 80)
    print(f"Encoders: {list(ENCODER_CONFIGS.keys())}")
    print(f"Skip training: {skip_training}")
    print("=" * 80)
    
    all_results = []
    
    for encoder_name in ENCODER_CONFIGS.keys():
        print(f"\n{'#'*80}")
        print(f"# ENCODER: {encoder_name}")
        print(f"{'#'*80}")
        
        if not skip_training:
            print("\n[Step 1] Training...")
            train_result = train_encoder(encoder_name, num_epochs=epochs)
        else:
            print("\n[Step 1] Skipping training (using existing weights)")
        
        print("\n[Step 2] Evaluating...")
        eval_result = evaluate_encoder(encoder_name, verbose=True)
        
        if eval_result:
            all_results.append(eval_result)
    
    # Save combined results
    if all_results:
        import csv
        timestamp = datetime.now().strftime("%Y%m%d_%H%M%S")
        output_dir = OUTPUT_CONFIG["results_dir"]
        
        md_path = output_dir / "encoder_ablation_full.md"
        with open(md_path, 'w') as f:
            f.write("# Encoder Ablation Study - Full Results\n\n")
            f.write(f"Generated: {datetime.now().strftime('%Y-%m-%d %H:%M:%S')}\n\n")
            f.write("## Computational + Accuracy Metrics\n\n")
            f.write("| Encoder | #Params | GFLOPs | Dice (3D) | HD95 (3D) |\n")
            f.write("|---------|---------|--------|-----------|----------|\n")
            for r in all_results:
                f.write(f"| {r['name']} | {r['params']:,} | {r['gflops']:.4f} | "
                        f"{r['test_dice_3d']:.4f}±{r['test_dice_std']:.4f} | "
                        f"{r['test_hd95_3d']:.4f}±{r['test_hd95_std']:.4f} |\n")
        
        print(f"\nResults saved to: {md_path}")
    
    return all_results


if __name__ == "__main__":
    parser = argparse.ArgumentParser(description="Run full encoder ablation")
    parser.add_argument("--epochs", type=int, default=None, help="Number of epochs per encoder")
    parser.add_argument("--skip-training", action="store_true", help="Skip training, only evaluate")
    args = parser.parse_args()
    
    run_full_encoder_ablation(epochs=args.epochs, skip_training=args.skip_training)
