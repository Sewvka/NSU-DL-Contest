#!/bin/bash
set -e

echo "--- Starting Stage 1: Preprocessing (CPU) ---"
python preprocess_v5.py

echo "--- Starting Stage 2: Training (GPU) ---"
python train_model_gpu.py

echo "--- Pipeline Finished! ---"
