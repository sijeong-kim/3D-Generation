#!/bin/bash

# Script to run all 5 methods: wo, svgd+rbf, svgd+cosine, rlsd+rbf, rlsd+cosine
# Usage: ./run_all_methods.sh [prompt]

# Set default prompt if not provided
PROMPT=${1:-"a photo of a hamburger"}

echo "Running all 5 methods with prompt: '$PROMPT'"
echo "Methods: wo, svgd+rbf, svgd+cosine, rlsd+rbf, rlsd+cosine"
echo "=================================================="

# Method 1: wo (without repulsion)
echo "Running Method 1: wo (without repulsion)"
python hp_ours.py \
    --config configs/text_ours.yaml \
    --sweep_config configs/text_ours_combined_methods.yaml \
    --sweep_name exp_wo \
    --prompts "$PROMPT" \
    --outdir=logs

echo "Method 1 completed!"
echo "=================================================="

# Method 2: svgd+rbf
echo "Running Method 2: svgd+rbf"
python hp_ours.py \
    --config configs/text_ours.yaml \
    --sweep_config configs/text_ours_combined_methods.yaml \
    --sweep_name exp_svgd_rbf \
    --prompts "$PROMPT" \
    --outdir=logs

echo "Method 2 completed!"
echo "=================================================="

# Method 3: svgd+cosine
echo "Running Method 3: svgd+cosine"
python hp_ours.py \
    --config configs/text_ours.yaml \
    --sweep_config configs/text_ours_combined_methods.yaml \
    --sweep_name exp_svgd_cosine \
    --prompts "$PROMPT" \
    --outdir=logs

echo "Method 3 completed!"
echo "=================================================="

# Method 4: rlsd+rbf
echo "Running Method 4: rlsd+rbf"
python hp_ours.py \
    --config configs/text_ours.yaml \
    --sweep_config configs/text_ours_combined_methods.yaml \
    --sweep_name exp_rlsd_rbf \
    --prompts "$PROMPT" \
    --outdir=logs

echo "Method 4 completed!"
echo "=================================================="

# Method 5: rlsd+cosine
echo "Running Method 5: rlsd+cosine"
python hp_ours.py \
    --config configs/text_ours.yaml \
    --sweep_config configs/text_ours_combined_methods.yaml \
    --sweep_name exp_rlsd_cosine \
    --prompts "$PROMPT" \
    --outdir=logs

echo "Method 5 completed!"
echo "=================================================="

echo "All 5 methods completed!"
echo "Results saved in logs/ directory:"
echo "  - logs/wo_method/"
echo "  - logs/svgd_rbf_method/"
echo "  - logs/svgd_cosine_method/"
echo "  - logs/rlsd_rbf_method/"
echo "  - logs/rlsd_cosine_method/"
