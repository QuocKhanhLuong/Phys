#!/bin/bash
"""
Ablation study runner script for comprehensive experiments.
This script runs all ablation studies automatically.
"""

set -e  # Exit on any error

echo "=========================================="
echo "ABLATION STUDY RUNNER"
echo "=========================================="

# Configuration
DATASET="brats21"
BASE_EXPERIMENT="brats21_ablation"
RESULTS_DIR="results/experiments"

# Create results directory
mkdir -p $RESULTS_DIR

echo "Starting ablation studies for dataset: $DATASET"
echo "Results will be saved to: $RESULTS_DIR"
echo ""

# Function to run experiment
run_experiment() {
    local exp_name=$1
    local extra_args=$2
    local description=$3
    
    echo "Running experiment: $exp_name"
    echo "Description: $description"
    echo "Command: python scripts/train.py --dataset $DATASET --experiment_name $exp_name $extra_args"
    echo "----------------------------------------"
    
    python scripts/train.py \
        --dataset $DATASET \
        --experiment_name $exp_name \
        $extra_args
    
    if [ $? -eq 0 ]; then
        echo "✓ Experiment $exp_name completed successfully"
    else
        echo "❌ Experiment $exp_name failed"
        exit 1
    fi
    echo ""
}

# 1. Baseline experiment (no physics loss)
run_experiment "${BASE_EXPERIMENT}_baseline" \
    "--no_physics_loss" \
    "Baseline model without physics-informed loss"

# 2. Physics loss with different weights
run_experiment "${BASE_EXPERIMENT}_physics_0.1" \
    "--physics_weight 0.1" \
    "Physics loss with weight 0.1"

run_experiment "${BASE_EXPERIMENT}_physics_0.3" \
    "--physics_weight 0.3" \
    "Physics loss with weight 0.3 (default)"

run_experiment "${BASE_EXPERIMENT}_physics_0.5" \
    "--physics_weight 0.5" \
    "Physics loss with weight 0.5"

# 3. Different loss combinations
run_experiment "${BASE_EXPERIMENT}_dice_only" \
    "--dice_weight 1.0 --focal_weight 0.0 --physics_weight 0.0" \
    "Dice loss only"

run_experiment "${BASE_EXPERIMENT}_focal_only" \
    "--dice_weight 0.0 --focal_weight 1.0 --physics_weight 0.0" \
    "Focal loss only"

run_experiment "${BASE_EXPERIMENT}_dice_focal" \
    "--dice_weight 0.5 --focal_weight 0.5 --physics_weight 0.0" \
    "Dice + Focal loss combination"

# 4. Different model architectures
run_experiment "${BASE_EXPERIMENT}_unet2d" \
    "--model unet2d" \
    "2D UNet architecture"

run_experiment "${BASE_EXPERIMENT}_unet3d" \
    "--model unet3d" \
    "3D UNet architecture"

# 5. Different batch sizes
run_experiment "${BASE_EXPERIMENT}_batch_8" \
    "--batch_size 8" \
    "Batch size 8"

run_experiment "${BASE_EXPERIMENT}_batch_16" \
    "--batch_size 16" \
    "Batch size 16"

run_experiment "${BASE_EXPERIMENT}_batch_32" \
    "--batch_size 32" \
    "Batch size 32"

# 6. Different learning rates
run_experiment "${BASE_EXPERIMENT}_lr_1e4" \
    "--learning_rate 1e-4" \
    "Learning rate 1e-4"

run_experiment "${BASE_EXPERIMENT}_lr_5e5" \
    "--learning_rate 5e-5" \
    "Learning rate 5e-5 (default)"

run_experiment "${BASE_EXPERIMENT}_lr_1e5" \
    "--learning_rate 1e-5" \
    "Learning rate 1e-5"

echo "=========================================="
echo "ALL ABLATION EXPERIMENTS COMPLETED"
echo "=========================================="
echo ""
echo "Results summary:"
echo "- All experiments saved to: $RESULTS_DIR"
echo "- Each experiment contains:"
echo "  - best_model.pth (trained model)"
echo "  - training.log (training metrics)"
echo "  - evaluation.json (final metrics)"
echo "  - visualizations/ (prediction images)"
echo ""
echo "To analyze results, run:"
echo "python scripts/analyze_results.py --results_dir $RESULTS_DIR"
