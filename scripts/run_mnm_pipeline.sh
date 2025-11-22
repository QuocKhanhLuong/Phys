#!/bin/bash

set -e

echo "============================================================"
echo "M&M Dataset: Preprocessing + Training Pipeline"
echo "============================================================"
echo ""

PROJECT_ROOT="/home/linhdang/workspace/minhbao_workspace/Phys"
cd $PROJECT_ROOT

echo "Step 1: Preprocessing Training Set"
echo "============================================================"
python scripts/preprocess_mnm.py \
  --input data/MnM/Training/Labeled \
  --output preprocessed_data/mnm/training

echo ""
echo "Step 2: Preprocessing Validation Set"
echo "============================================================"
python scripts/preprocess_mnm.py \
  --input "data/MnM/M&M/Validation" \
  --output preprocessed_data/mnm/validation

echo ""
echo "Step 3: Training Model"
echo "============================================================"
python scripts/train_mnm.py

echo ""
echo "============================================================"
echo "Pipeline Complete!"
echo "============================================================"
echo "Results saved in: mnm_results/"
echo "Best model: mnm_results/best_model_mnm.pth"

