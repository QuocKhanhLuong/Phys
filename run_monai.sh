#!/bin/bash
# Script to run MONAI pipeline: preprocessing + training

set -e

echo "============================================================"
echo "MONAI PIPELINE - Preprocessing + Training"
echo "============================================================"

BRATS_RAW="/home/linhdang/workspace/minhbao_workspace/Phys/BraTS21"
MONAI_OUTPUT="/home/linhdang/workspace/minhbao_workspace/Phys/BraTS21_preprocessed_monai"

if [ -d "$MONAI_OUTPUT" ] && [ -f "$MONAI_OUTPUT/metadata.json" ]; then
    echo "✓ MONAI preprocessed data already exists at:"
    echo "  $MONAI_OUTPUT"
    echo ""
    read -p "Skip preprocessing and start training? (Y/n) " -n 1 -r
    echo
    if [[ $REPLY =~ ^[Nn]$ ]]; then
        echo "Re-running preprocessing..."
        python monai_preprocess.py \
            --input_dir "$BRATS_RAW" \
            --output_dir "$MONAI_OUTPUT" \
            --spacing 1.0 1.0 1.0
    else
        echo "Skipping preprocessing, starting training..."
    fi
else
    echo "Running MONAI preprocessing..."
    echo "Input:  $BRATS_RAW"
    echo "Output: $MONAI_OUTPUT"
    echo ""
    
    python monai_preprocess.py \
        --input_dir "$BRATS_RAW" \
        --output_dir "$MONAI_OUTPUT" \
        --spacing 1.0 1.0 1.0
fi

echo ""
echo "============================================================"
echo "Training with MONAI Pipeline"
echo "============================================================"
echo ""

# Step 2: Start training
python train.py

echo ""
echo "Done!"

