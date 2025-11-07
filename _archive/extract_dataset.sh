#!/bin/bash

cd BraTS21

echo "Extracting BraTS2021 Training Data..."
tar -xf BraTS2021_Training_Data.tar

echo "Extracting BraTS2021_00621..."
tar -xf BraTS2021_00621.tar

echo "Extracting BraTS2021_00495..."
tar -xf BraTS2021_00495.tar

echo "Dataset extraction completed!"
echo "Training data should be in: BraTS21/training/"
echo "Testing data should be in: BraTS21/testing/"

