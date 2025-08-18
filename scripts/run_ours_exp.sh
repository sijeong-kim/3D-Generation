#!/bin/bash

# Simple script to run hyperparameter tuning experiments
# Usage: ./scripts/run_ours_exp.sh <experiment_name> [--dry-run]
#
# Examples:
#   ./scripts/run_ours_exp.sh exp1_repulsion_lambda_sweep
#   ./scripts/run_ours_exp.sh exp1_wo_method --dry-run

# Parse arguments
EXPERIMENT_NAME="$1"
DRY_RUN=""

if [ "$2" = "--dry-run" ]; then
    DRY_RUN="--dry_run"
fi

# Validate experiment name
if [ -z "$EXPERIMENT_NAME" ]; then
    echo "[ERROR] Experiment name is required"
    echo "Usage: $0 <experiment_name> [--dry-run]"
    echo ""
    echo "Available experiments:"
    echo "  - exp1_repulsion_lambda_sweep"
    echo "  - exp1_wo_method"
    echo ""
    echo "Examples:"
    echo "  $0 exp1_repulsion_lambda_sweep"
    echo "  $0 exp1_wo_method --dry-run"
    exit 1
fi

# Get the directory where this script is located
SCRIPT_DIR="$(cd "$(dirname "${BASH_SOURCE[0]}")" && pwd)"
PROJECT_DIR="$(dirname "$SCRIPT_DIR")"

echo "========== EXPERIMENT RUNNER =========="
echo "Project Directory: $PROJECT_DIR"
echo "Experiment: $EXPERIMENT_NAME"
echo "Dry Run: $([ -n "$DRY_RUN" ] && echo "Yes" || echo "No")"
echo "======================================="

# Change to project directory
cd "$PROJECT_DIR"

# Run the hyperparameter tuning script
python hp_ours.py \
    --config configs/text_ours.yaml \
    --sweep_config configs/text_ours_exp.yaml \
    --sweep_name "$EXPERIMENT_NAME" \
    --outdir logs \
    $DRY_RUN

echo ""
echo "======================================="
echo "Experiment completed!"
echo "Check logs/$EXPERIMENT_NAME/ for results"
echo "======================================="
