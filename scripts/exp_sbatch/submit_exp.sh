#!/bin/bash

# Wrapper script to submit experiment jobs via sbatch
# Usage: ./submit_exp.sh [sweep_name] [options...]

set -euo pipefail

# Default values
SWEEP_NAME=""
OUTDIR="exp"
BASE_CONFIG="configs/text_ours.yaml"
SWEEP_CONFIG="configs/text_ours_exp.yaml"
GPUS="0"
CPU_CORES="0-7"
THREADS="8"
MEM_SOFT_MB="32768"
TIMEOUT="14400"
SLEEP_BETWEEN="60"
DRY_RUN="false"

# Parse arguments
while [[ $# -gt 0 ]]; do
  case "$1" in
    --outdir) OUTDIR="$2"; shift 2 ;;
    --base_config) BASE_CONFIG="$2"; shift 2 ;;
    --sweep_config) SWEEP_CONFIG="$2"; shift 2 ;;
    --gpus) GPUS="$2"; shift 2 ;;
    --cpu_cores) CPU_CORES="$2"; shift 2 ;;
    --threads) THREADS="$2"; shift 2 ;;
    --mem_soft_mb) MEM_SOFT_MB="$2"; shift 2 ;;
    --timeout) TIMEOUT="$2"; shift 2 ;;
    --sleep_between) SLEEP_BETWEEN="$2"; shift 2 ;;
    --dry_run) DRY_RUN="true"; shift ;;
    --help)
      echo "Usage: $0 [sweep_name] [OPTIONS]"
      echo ""
      echo "Arguments:"
      echo "  sweep_name            Name of the experiment sweep (required)"
      echo ""
      echo "Options:"
      echo "  --outdir DIR          Output directory (default: exp)"
      echo "  --base_config PATH    Base config file (default: configs/text_ours.yaml)"
      echo "  --sweep_config PATH   Sweep config file (default: configs/text_ours_exp.yaml)"
      echo "  --gpus CSV            GPU list (default: 0)"
      echo "  --cpu_cores STR       CPU cores (default: 0-7)"
      echo "  --threads N           Number of threads (default: 8)"
      echo "  --mem_soft_mb MB      Memory limit in MB (default: 32768)"
      echo "  --timeout SEC         Timeout in seconds (default: 14400)"
      echo "  --sleep_between SEC   Sleep between experiments (default: 60)"
      echo "  --dry_run             Show the sbatch command without submitting"
      echo "  --help                Show this help"
      echo ""
      echo "Examples:"
      echo "  $0 my_experiment"
      echo "  $0 my_experiment --outdir results --gpus 0,1 --timeout 28800"
      echo "  $0 my_experiment --dry_run"
      exit 0
      ;;
    --*) echo "Unknown option: $1"; exit 1 ;;
    *)
      if [[ -z "$SWEEP_NAME" ]]; then
        SWEEP_NAME="$1"
      else
        echo "Multiple sweep names provided or unknown argument: $1"
        exit 1
      fi
      shift
      ;;
  esac
done

# Check if sweep name is provided
if [[ -z "$SWEEP_NAME" ]]; then
  echo "Error: Sweep name is required"
  echo "Use --help for usage information"
  exit 1
fi

# Build the sbatch command
SBATCH_CMD="sbatch scripts/exp_sbatch/run_exp_sbatch.sh \
  --sweep_name \"${SWEEP_NAME}\" \
  --outdir \"${OUTDIR}\" \
  --base_config \"${BASE_CONFIG}\" \
  --sweep_config \"${SWEEP_CONFIG}\" \
  --gpus \"${GPUS}\" \
  --cpu_cores \"${CPU_CORES}\" \
  --threads \"${THREADS}\" \
  --mem_soft_mb \"${MEM_SOFT_MB}\" \
  --timeout \"${TIMEOUT}\" \
  --sleep_between \"${SLEEP_BETWEEN}\""

echo "Sweep Name: ${SWEEP_NAME}"
echo "Output Dir: ${OUTDIR}"
echo "Base Config: ${BASE_CONFIG}"
echo "Sweep Config: ${SWEEP_CONFIG}"
echo "GPUs: ${GPUS}"
echo "CPU Cores: ${CPU_CORES}"
echo "Threads: ${THREADS}"
echo "Memory Limit: ${MEM_SOFT_MB} MB"
echo "Timeout: ${TIMEOUT} seconds"
echo "Sleep Between: ${SLEEP_BETWEEN} seconds"
echo ""

if [[ "$DRY_RUN" == "true" ]]; then
  echo "DRY RUN - Would execute:"
  echo "$SBATCH_CMD"
else
  echo "Submitting job..."
  eval "$SBATCH_CMD"
  echo "Job submitted successfully!"
fi
