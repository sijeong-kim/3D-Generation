#!/bin/bash

# Unified runner for 3D Generation experiments
# - Generates per-run configs via exp_config.py
# - Runs main_ours.py directly for each experiment
# - Per-run CPU/GPU initialization + teardown
# - Logs/summary/resume/retry support

set -euo pipefail

# =======================
# Pretty printing helpers
# =======================
RED='\033[0;31m'; GREEN='\033[0;32m'; YELLOW='\033[1;33m'; BLUE='\033[0;34m'; PURPLE='\033[0;35m'; NC='\033[0m'
print_info()    { echo -e "${BLUE}[INFO]${NC} $*"; }
print_success() { echo -e "${GREEN}[SUCCESS]${NC} $*"; }
print_warning() { echo -e "${YELLOW}[WARNING]${NC} $*"; }
print_error()   { echo -e "${RED}[ERROR]${NC} $*"; }
print_header()  { echo -e "${PURPLE}$*${NC}"; }

usage() {
  cat <<'EOF'
Usage: run_sweep.sh [OPTIONS] <sweep_name>

OPTIONS:
  --outdir DIR          Base output directory (default: exp)
  --dry_run             Print commands without running
  --no_resume           Start fresh (ignore previous summary/.done)
  --retry_failed_only   Only retry previously failed runs
  --gpus CSV            GPU list for round-robin, e.g. "0,1" (default: none)
  --cpu_cores STR       CPU pinning, e.g. "0-15" or "0,2,4,6" (default: all)
  --threads N           BLAS/torch CPU threads (default: env/system default)
  --mem_soft_mb MB      Soft virtual memory cap for child (ulimit -v) (default: none)
  --timeout SEC         Per-run timeout in seconds (default: 7200)
  --base_config PATH    Base yaml (default: configs/text_ours.yaml)
  --sweep_config PATH   Sweep yaml (default: configs/text_ours_exp.yaml)
  --help                Show this help

Example:
  ./run_sweep.sh exp1_lambda_coarse_svgd \
    --outdir logs_direct \
    --gpus 0,1 \
    --cpu_cores 0-15 \
    --threads 8 \
    --mem_soft_mb 65536 \
    --timeout 14400
EOF
}

# =======================
# Global defaults
# =======================
OUTDIR="exp"
DRY_RUN="false"
NO_RESUME="false"
RETRY_FAILED_ONLY="false"
GPUS=""
CPU_CORES=""
THREADS=""
MEM_SOFT_MB=""
TIMEOUT="7200"
BASE_CONFIG="configs/text_ours.yaml"
SWEEP_CONFIG="configs/text_ours_exp.yaml"
SWEEP_NAME=""
SLEEP_BETWEEN="60"

# =======================
# Parse args
# =======================
while [[ $# -gt 0 ]]; do
  case "$1" in
    --outdir) OUTDIR="$2"; shift 2 ;;
    --dry_run) DRY_RUN="true"; shift ;;
    --no_resume) NO_RESUME="true"; shift ;;
    --retry_failed_only) RETRY_FAILED_ONLY="true"; shift ;;
    --gpus) GPUS="$2"; shift 2 ;;
    --cpu_cores) CPU_CORES="$2"; shift 2 ;;
    --threads) THREADS="$2"; shift 2 ;;
    --mem_soft_mb) MEM_SOFT_MB="$2"; shift 2 ;;
    --timeout) TIMEOUT="$2"; shift 2 ;;
    --base_config) BASE_CONFIG="$2"; shift 2 ;;
    --sweep_config) SWEEP_CONFIG="$2"; shift 2 ;;
    --help) usage; exit 0 ;;
    --sleep_between) SLEEP_BETWEEN="$2"; shift 2 ;;
    --*) print_error "Unknown option: $1"; usage; exit 1 ;;
    *) if [[ -z "$SWEEP_NAME" ]]; then SWEEP_NAME="$1"; shift; else print_error "Multiple sweep names provided"; exit 1; fi ;;
  esac
done

[[ -z "$SWEEP_NAME" ]] && { print_error "Missing <sweep_name>"; usage; exit 1; }
[[ -f "$BASE_CONFIG" ]] || { print_error "Base config not found: $BASE_CONFIG"; exit 1; }
[[ -f "$SWEEP_CONFIG" ]] || { print_error "Sweep config not found: $SWEEP_CONFIG"; exit 1; }

export DRY_RUN GPUS CPU_CORES THREADS MEM_SOFT_MB TIMEOUT SLEEP_BETWEEN

# =======================
# Helpers
# =======================
pick_gpu_rr() {
  local idx=$1; local csv="$2"
  [[ -z "$csv" ]] && { echo ""; return; }
  IFS=',' read -r -a arr <<< "$csv"
  local n=${#arr[@]}
  echo "${arr[$((idx % n))]}"
}

check_cuda_availability() {
  if python3 -c "import torch; print(torch.cuda.is_available())" 2>/dev/null; then
    local cnt=$(python3 -c "import torch; print(torch.cuda.device_count())" 2>/dev/null || echo 0)
    print_info "CUDA available with $cnt GPU(s)"
    return 0
  else
    print_warning "CUDA not available (or PyTorch missing)"
    return 1
  fi
}

collect_run_metadata() {
  local out="$1"
  {
    echo "timestamp: $(date -Iseconds)"
    echo "hostname: $(hostname)"
    echo "user: ${USER:-unknown}"
    if command -v git >/dev/null 2>&1 && [[ -d .git ]]; then
      echo "git_commit: $(git rev-parse HEAD 2>/dev/null || echo unknown)"
      echo "git_branch: $(git rev-parse --abbrev-ref HEAD 2>/dev/null || echo unknown)"
    fi
    if command -v python3 >/dev/null 2>&1; then
      echo "python_version: $(python3 --version 2>&1)"
      python3 - <<'PY'
try:
  import torch
  print("torch_version:", torch.__version__)
  print("cuda_version:", getattr(torch.version,'cuda','N/A'))
except Exception:
  print("torch_version:", "N/A")
  print("cuda_version:", "N/A")
PY
    fi
  } > "$out"
}

# =======================
# Per-run executor
# =======================
run_single_experiment() {
  local config_file="$1"
  local output_dir="$2"
  local run_idx="$3"

  mkdir -p "$output_dir" "$output_dir/figures"

  print_info "Running experiment: $output_dir"
  printf '%.0s-' {1..80}; echo

  # ---- Resource init ----
  local gpu_id="$(pick_gpu_rr $((run_idx-1)) "$GPUS")"
  if [[ -n "$gpu_id" ]]; then
    export CUDA_VISIBLE_DEVICES="$gpu_id"
    print_info "Assigned GPU (CUDA_VISIBLE_DEVICES): $gpu_id"
  else
    # Default to GPU 0 if no GPUs specified
    export CUDA_VISIBLE_DEVICES="0"
    gpu_id="0"
    print_info "No GPU specified, using default GPU 0"
  fi

  # Threads
  if [[ -n "$THREADS" ]]; then
    export OMP_NUM_THREADS="$THREADS"
    export MKL_NUM_THREADS="$THREADS"
    export OPENBLAS_NUM_THREADS="$THREADS"
    export NUMEXPR_NUM_THREADS="$THREADS"
    export TORCH_NUM_THREADS="$THREADS"
  fi

  # PyTorch allocator (helps multi-run fragmentation)
  export PYTORCH_CUDA_ALLOC_CONF="expandable_segments:True,max_split_size_mb:128"
  

  # Optional soft memory cap for child
  local ULIMIT_PREFIX=""
  if [[ -n "${MEM_SOFT_MB}" ]]; then
    local KB=$((MEM_SOFT_MB * 1024))
    ULIMIT_PREFIX="ulimit -v $KB; "
  fi

  # CPU pinning
  local EXEC_PREFIX=""
  if command -v numactl >/dev/null 2>&1 && [[ -n "$CPU_CORES" ]]; then
    EXEC_PREFIX="numactl --physcpubind=$CPU_CORES --localalloc"
  elif command -v taskset >/dev/null 2>&1 && [[ -n "$CPU_CORES" ]]; then
    EXEC_PREFIX="taskset -c $CPU_CORES"
  fi

  # Snapshots (before)
  if command -v nvidia-smi >/dev/null 2>&1; then
    nvidia-smi -i "${gpu_id:-0}" \
      --query-gpu=index,name,memory.total,memory.used,memory.free \
      --format=csv,noheader > "$output_dir/nvidia_smi_before.txt" 2>/dev/null || true
  fi
  free -h > "$output_dir/memory_before.txt" 2>/dev/null || true
  command -v lscpu >/dev/null 2>&1 && lscpu > "$output_dir/cpu_info.txt" 2>/dev/null || true

  # Outer CUDA cache clear
  python3 - <<'PY' >/dev/null 2>&1 || true
import gc
try:
  import torch
  if torch.cuda.is_available():
    torch.cuda.empty_cache()
    torch.cuda.synchronize()
except Exception:
  pass
gc.collect()
PY

  # Metadata (start)
  collect_run_metadata "$output_dir/run_metadata.yaml"
  local start_time=$(date -Iseconds)
  {
    echo "start_time: $start_time"
    echo "gpu_id: ${gpu_id:-none}"
    echo "threads: ${THREADS:-default}"
    echo "cpu_cores: ${CPU_CORES:-all}"
    echo "PYTORCH_CUDA_ALLOC_CONF: ${PYTORCH_CUDA_ALLOC_CONF}"
  } >> "$output_dir/run_metadata.yaml"

  # ---- Launch ----
  local result=0
  local error_msg=""
  export TQDM_DISABLE=1

  if [[ "$DRY_RUN" == "true" ]]; then
    print_info "DRY RUN - Would run: main_ours.py --config $config_file"
  else
    print_info "Starting main_ours.py..."
    local CMD="python3 main_ours.py --config \"$config_file\""
    [[ -n "$ULIMIT_PREFIX" ]] && CMD="${ULIMIT_PREFIX}${CMD}"
    [[ -n "$EXEC_PREFIX"   ]] && CMD="${EXEC_PREFIX} ${CMD}"

    # run with timeout
    if timeout "${TIMEOUT}" bash -lc "$CMD" > "$output_dir/out" 2> "$output_dir/err"; then
      result=0
    else
      result=$?
      error_msg="Training failed with exit code $result"
    fi
  fi

  # ---- Teardown ----
  local end_time=$(date -Iseconds)
  local duration_sec=$(( $(date -d "$end_time" +%s) - $(date -d "$start_time" +%s) ))
  {
    echo "end_time: $end_time"
    echo "duration_sec: $duration_sec"
    echo "return_code: $result"
    [[ $result -ne 0 ]] && echo "error: $error_msg"
  } >> "$output_dir/run_metadata.yaml"

  if command -v nvidia-smi >/dev/null 2>&1; then
    nvidia-smi -i "${gpu_id:-0}" \
      --query-gpu=index,name,memory.total,memory.used,memory.free \
      --format=csv,noheader > "$output_dir/nvidia_smi_after.txt" 2>/dev/null || true
  fi
  free -h > "$output_dir/memory_after.txt" 2>/dev/null || true

  python3 - <<'PY' >/dev/null 2>&1 || true
import gc
try:
  import torch
  if torch.cuda.is_available():
    torch.cuda.empty_cache()
    torch.cuda.ipc_collect()
    torch.cuda.synchronize()
except Exception:
  pass
gc.collect()
PY

  if [[ $result -eq 0 ]]; then
    echo "$end_time" > "$output_dir/.done"
    print_success "Experiment OK: $output_dir (duration ${duration_sec}s)"
  else
    print_error "Experiment FAILED: $output_dir (rc=$result)  (see $output_dir/err)"
  fi

  # sleep for sync
  sleep 60

  return $result
}

# =======================
# Main flow
# =======================
print_info "Loading experiment configs via exp_config.py ..."
python3 scripts/experiments/exp_config.py "$BASE_CONFIG" "$SWEEP_CONFIG" "$SWEEP_NAME" --save-files \
  > /tmp/_exp_config.out 2>&1 || { print_error "exp_config.py failed"; cat /tmp/_exp_config.out; exit 1; }

# exp_config.py is expected to create "exp/<sweep_name>/experiment_summary.txt"
EXP_DIR="exp/$SWEEP_NAME"
SUMMARY_TXT="$EXP_DIR/experiment_summary.txt"
[[ -f "$SUMMARY_TXT" ]] || { print_error "Summary file not found: $SUMMARY_TXT"; exit 1; }

CONFIG_COUNT=$(grep -E "Total configurations:" "$SUMMARY_TXT" | awk '{print $3}')
print_info "Generated $CONFIG_COUNT configuration(s) under $EXP_DIR"

# Extract output directories listed like: "1: prompt=..._kernel=... -> <path>"
mapfile -t OUTPUT_DIR_NAMES < <(awk -F': ' '/^[0-9]+: /{print $2}' "$SUMMARY_TXT" | sed 's/ ->.*$//')

print_header "$(printf '%.0s=' {1..80})"
print_info "Experiment: $SWEEP_NAME"
print_info "Total experiments: $CONFIG_COUNT"
print_header "$(printf '%.0s=' {1..80})"

check_cuda_availability || true
print_header "$(printf '%.0s=' {1..80})"

BASE_OUT="$OUTDIR/$SWEEP_NAME"
mkdir -p "$BASE_OUT"

SUMMARY_YAML="$BASE_OUT/experiment_summary.yaml"
START_TIME=$(date -Iseconds)
cat > "$SUMMARY_YAML" <<EOF
experiment_name: $SWEEP_NAME
parameter_combinations: $CONFIG_COUNT
total_experiments: $CONFIG_COUNT
start_time: $START_TIME
combinations: []
EOF

successful_runs=0
failed_runs=0

print_info "Starting execution of $CONFIG_COUNT experiments..."
for i in "${!OUTPUT_DIR_NAMES[@]}"; do
  idx=$((i+1))
  out_name="${OUTPUT_DIR_NAMES[$i]}"
  run_dir="$BASE_OUT/$out_name"
  cfg="$run_dir/config.yaml"

  print_info "[$idx/$CONFIG_COUNT] $run_dir"
  [[ -f "$cfg" ]] || { print_error "Missing config: $cfg"; failed_runs=$((failed_runs+1));
    {
      echo "- experiment_id: $idx"
      echo "  output_dir: $run_dir"
      echo "  status: failed"
    } >> "$SUMMARY_YAML"
    continue; }

  # resume / retry logic
  if [[ "$NO_RESUME" == "false" && -f "$run_dir/.done" && "$RETRY_FAILED_ONLY" == "false" ]]; then
    print_info "Skip: already done"
    {
      echo "- experiment_id: $idx"
      echo "  output_dir: $run_dir"
      echo "  status: skipped"
    } >> "$SUMMARY_YAML"
    continue
  fi

  {
    echo "- experiment_id: $idx"
    echo "  output_dir: $run_dir"
    echo "  status: pending"
  } >> "$SUMMARY_YAML"

  if [[ "$DRY_RUN" == "true" ]]; then
    print_info "DRY RUN - Would execute"
    continue
  fi

  if run_single_experiment "$cfg" "$run_dir" "$idx"; then
    successful_runs=$((successful_runs+1))
    sed -i "0,/status: pending/{s/status: pending/status: success/}" "$SUMMARY_YAML"
  else
    failed_runs=$((failed_runs+1))
    sed -i "0,/status: pending/{s/status: pending/status: failed/}" "$SUMMARY_YAML"
  fi

  END_TIME=$(date -Iseconds)
  # Refresh summary tail fields (idempotent)
  sed -i "/^end_time:/d" "$SUMMARY_YAML";          echo "end_time: $END_TIME" >> "$SUMMARY_YAML"
  sed -i "/^successful_runs:/d" "$SUMMARY_YAML";   echo "successful_runs: $successful_runs" >> "$SUMMARY_YAML"
  sed -i "/^failed_runs:/d" "$SUMMARY_YAML";       echo "failed_runs: $failed_runs" >> "$SUMMARY_YAML"

  # After summary refresh, sleep between experiments if configured
  if [[ "${SLEEP_BETWEEN}" != "0" ]]; then
    print_info "Sleeping ${SLEEP_BETWEEN}s between experiments..."
    # 파일/버퍼 동기화 후 대기 (선택)
    sync || true
    sleep "${SLEEP_BETWEEN}"
  fi

done

print_header "$(printf '%.0s=' {1..80})"
print_header "EXPERIMENT SUMMARY"
print_header "$(printf '%.0s=' {1..80})"
print_info "Experiment: $SWEEP_NAME"
print_info "Processed:  $CONFIG_COUNT"
print_info "Successes:  $successful_runs"
print_info "Failures:   $failed_runs"
if [[ "$CONFIG_COUNT" =~ ^[0-9]+$ && "$CONFIG_COUNT" -gt 0 ]]; then
  rate=$(python3 - <<'PY' 2>/dev/null || echo "0.0"
s=$successful_runs; n=$CONFIG_COUNT
print(f"{(s*100.0/n):.1f}")
PY
)
  print_info "Success rate: ${rate}%"
fi
print_info "Results dir: $BASE_OUT"
print_info "Summary:     $SUMMARY_YAML"

if [[ $failed_runs -gt 0 ]]; then
  print_warning "$failed_runs run(s) failed. Re-run with --retry_failed_only to target failures."
else
  print_success "All experiments completed successfully!"
fi
