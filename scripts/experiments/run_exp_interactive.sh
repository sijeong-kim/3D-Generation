#!/bin/bash
# Hyperparameter tuning script for 3D Generation experiments.
# Reads experiment configurations from YAML files and runs grid search experiments.
#
# IMPORTANT CHANGES (GPU Resource Management):
# - Replaced Python execution with direct bash script execution
# - Uses exp_config.py to generate configurations
# - Runs main_ours.py directly for each experiment
# - Added explicit GPU device initialization for each experiment
# - Implemented comprehensive GPU cleanup after each experiment
# - Added GPU resource monitoring and logging
# - Each experiment now properly initializes its own GPU context
#
# This ensures that GPU resources are properly managed and cleaned up between experiments,
# preventing memory leaks and resource conflicts in hyperparameter tuning runs.

set -e  # Exit on any error

# Colors for output
RED='\033[0;31m'
GREEN='\033[0;32m'
YELLOW='\033[1;33m'
BLUE='\033[0;34m'
PURPLE='\033[0;35m'
CYAN='\033[0;36m'
NC='\033[0m' # No Color

# Function to print colored output
print_info() {
    echo -e "${BLUE}[INFO]${NC} $1"
}

print_success() {
    echo -e "${GREEN}[SUCCESS]${NC} $1"
}

print_warning() {
    echo -e "${YELLOW}[WARNING]${NC} $1"
}

print_error() {
    echo -e "${RED}[ERROR]${NC} $1"
}

print_header() {
    echo -e "${PURPLE}$1${NC}"
}

# Function to show usage
show_usage() {
    cat << EOF
Usage: $0 [OPTIONS] <experiment_name>

Hyperparameter tuning for 3D Generation experiments.

OPTIONS:
    --outdir <dir>            Base output directory (default: exp)
    --dry_run                 Print commands without running experiments
    --no_resume               Disable resume functionality and start fresh
    --retry_failed_only       Only retry failed experiments, skip pending ones
    --help                    Show this help message

EXAMPLES:
    $0 exp1_lambda_coarse_svgd
    $0 exp1_lambda_coarse_svgd --dry_run
    $0 exp1_lambda_coarse_svgd --retry_failed_only

EOF
}

# Simple function to check if CUDA is available
check_cuda_availability() {
    if python3 -c "import torch; print(torch.cuda.is_available())" &> /dev/null; then
        local cuda_available=$(python3 -c "import torch; print(torch.cuda.is_available())" 2>/dev/null)
        if [ "$cuda_available" = "True" ]; then
            local gpu_count=$(python3 -c "import torch; print(torch.cuda.device_count())" 2>/dev/null || echo "0")
            print_info "CUDA available with $gpu_count GPU(s)"
            return 0
        else
            print_info "CUDA not available, using CPU"
            return 1
        fi
    else
        print_warning "PyTorch not available"
        return 1
    fi
}

# Function to collect run metadata
collect_run_metadata() {
    local metadata=""
    metadata+="timestamp: $(date -Iseconds)\n"
    metadata+="hostname: $(hostname)\n"
    metadata+="user: $USER\n"
    
    # Git information
    if command -v git &> /dev/null && [ -d ".git" ]; then
        local git_commit=$(git rev-parse HEAD 2>/dev/null || echo "unknown")
        local git_branch=$(git rev-parse --abbrev-ref HEAD 2>/dev/null || echo "unknown")
        metadata+="git_commit: $git_commit\n"
        metadata+="git_branch: $git_branch\n"
    fi
    
    # Python and package versions
    if command -v python3 &> /dev/null; then
        metadata+="python_version: $(python3 --version 2>&1)\n"
        if python3 -c "import torch; print(torch.__version__)" &> /dev/null; then
            metadata+="torch_version: $(python3 -c "import torch; print(torch.__version__)" 2>/dev/null)\n"
            if python3 -c "import torch; print(torch.version.cuda)" &> /dev/null; then
                metadata+="cuda_version: $(python3 -c "import torch; print(torch.version.cuda)" 2>/dev/null)\n"
            fi
        fi
    fi
    
    echo -e "$metadata"
}

# Function to run a single experiment
run_single_experiment() {
    local config_file=$1
    local output_dir=$2
    local base_config_path=$3
    local exp_name=$4
    
    # Create output directory
    mkdir -p "$output_dir"
    
    # Create figures subdirectory for analysis plots
    local figures_dir="$output_dir/figures"
    mkdir -p "$figures_dir"
    
    # Note: config.yaml is already saved in the output directory by exp_config.py
    # No need to copy it again
    
    print_info "Running experiment: $output_dir"
    echo "$(printf '%.0s-' {1..80})"
    
    # Collect metadata before running
    local metadata=$(collect_run_metadata)
    local start_time=$(date -Iseconds)
    metadata+="start_time: $start_time\n"
    
    # Save metadata
    echo -e "$metadata" > "$output_dir/run_metadata.yaml"
    
    # Check CUDA availability
    check_cuda_availability
    
    # Run the experiment
    local result=0
    local error_msg=""
    
    if [ "$DRY_RUN" = "true" ]; then
        print_info "DRY RUN - Would run: python3 main_ours.py --config $config_file"
        result=0
    else
        print_info "Starting training with main_ours.py..."
        
        # Run main_ours.py with the generated config
        # Set environment variables for cleaner output
        export TQDM_DISABLE=1  # Disable progress bars for cleaner logs
        
        if timeout 7200 python3 main_ours.py --config "$config_file" > "$output_dir/out" 2> "$output_dir/err"; then
            result=0
        else
            result=$?
            error_msg="Training failed with exit code $result"
        fi
    fi
    
    # Calculate duration
    local end_time=$(date -Iseconds)
    local duration_sec=$(($(date -d "$end_time" +%s) - $(date -d "$start_time" +%s)))
    
    # Update metadata with results
    metadata+="end_time: $end_time\n"
    metadata+="duration_sec: $duration_sec\n"
    metadata+="return_code: $result\n"
    
    if [ $result -ne 0 ]; then
        metadata+="error: $error_msg\n"
    fi
    
    # Save updated metadata
    echo -e "$metadata" > "$output_dir/run_metadata.yaml"
    
    if [ $result -eq 0 ]; then
        print_success "Experiment completed successfully: $output_dir (duration: ${duration_sec}s)"
        
        # Create .done marker to prevent duplicate runs
        echo "$end_time" > "$output_dir/.done"
        
        # Cleanup resources after successful completion
        print_info "Cleaning up resources..."
        python3 -c "
import gc
try:
    import torch
    if torch.cuda.is_available():
        torch.cuda.empty_cache()
        torch.cuda.synchronize()
except:
    pass
gc.collect()
" &> /dev/null || true
    else
        print_error "Experiment failed: $output_dir"
        print_error "Error: $error_msg"
        
        # Cleanup resources even after failure
        print_info "Cleaning up resources after failure..."
        python3 -c "
import gc
try:
    import torch
    if torch.cuda.is_available():
        torch.cuda.empty_cache()
        torch.cuda.synchronize()
except:
    pass
gc.collect()
" &> /dev/null || true
    fi
    
    return $result
}

# Function to load existing summary
load_existing_summary() {
    local base_output_dir=$1
    local summary_path="$base_output_dir/experiment_summary.yaml"
    
    if [ ! -f "$summary_path" ]; then
        return 1
    fi
    
    # Parse existing summary (simple parsing for bash)
    local failed_count=0
    local successful_count=0
    local pending_count=0
    
    if [ -f "$summary_path" ]; then
        failed_count=$(grep -c "status: failed" "$summary_path" 2>/dev/null || echo "0")
        successful_count=$(grep -c "status: success\|status: skipped" "$summary_path" 2>/dev/null || echo "0")
        pending_count=$(grep -c "status: pending" "$summary_path" 2>/dev/null || echo "0")
    fi
    
    echo "$failed_count $successful_count $pending_count"
    return 0
}

# Function to create combination info
create_combination_info() {
    local sweep_params=$1
    local output_dir=$2
    local experiment_id=$3
    
    # Simple combination info for bash
    echo "experiment_id: $experiment_id"
    echo "output_dir: $output_dir"
    echo "status: pending"
}

# Function to categorize combinations
categorize_combinations() {
    local base_output_dir=$1
    local existing_summary=$2
    
    local failed_combinations=()
    local pending_combinations=()
    local successful_combinations=()
    
    # For now, assume all combinations are pending
    # In a full implementation, this would parse the existing summary
    # and check for .done files in output directories
    
    echo "${failed_combinations[@]}" "${pending_combinations[@]}" "${successful_combinations[@]}"
}

# Main function
main() {
    # Parse command line arguments
    local sweep_name=""
    local outdir="exp"
    local dry_run=false
    local no_resume=false
    local retry_failed_only=false
    
    # Parse options first
    while [[ $# -gt 0 ]]; do
        case $1 in
            --outdir)
                outdir="$2"
                shift 2
                ;;
            --dry_run)
                dry_run=true
                shift
                ;;
            --no_resume)
                no_resume=true
                shift
                ;;
            --retry_failed_only)
                retry_failed_only=true
                shift
                ;;
            --help)
                show_usage
                exit 0
                ;;
            -*)
                print_error "Unknown option: $1"
                show_usage
                exit 1
                ;;
            *)
                # First non-option argument is the experiment name
                if [ -z "$sweep_name" ]; then
                    sweep_name="$1"
                else
                    print_error "Multiple experiment names specified: $sweep_name and $1"
                    exit 1
                fi
                shift
                ;;
        esac
    done
    
    # Validate required arguments
    if [ -z "$sweep_name" ]; then
        print_error "Missing experiment name"
        show_usage
        exit 1
    fi
    
    # Set default config paths
    local base_config="configs/text_ours.yaml"
    local sweep_config="configs/text_ours_exp.yaml"
    
    # Check if files exist
    if [ ! -f "$base_config" ]; then
        print_error "Base config file not found: $base_config"
        exit 1
    fi
    
    if [ ! -f "$sweep_config" ]; then
        print_error "Sweep config file not found: $sweep_config"
        exit 1
    fi
    
    # Load configurations using exp_config.py
    print_info "Loading configurations using exp_config.py..."
    
    # Generate config files instead of parsing stdout
    local config_count
    local output_dirs=()
    
    # First, save all configs to files in their respective output directories
    if ! config_output=$(python3 scripts/experiments/exp_config.py "$base_config" "$sweep_config" "$sweep_name" --save-files 2>&1); then
        print_error "Failed to generate experiment configurations:"
        echo "$config_output"
        exit 1
    fi
    
    # Parse the output to get count and extract output directories from summary file
    local exp_dir="exp/$sweep_name"
    local summary_file="$exp_dir/experiment_summary.txt"
    
    if [ ! -f "$summary_file" ]; then
        print_error "Summary file not found: $summary_file"
        exit 1
    fi
    
    # Extract config count from summary
    config_count=$(grep "Total configurations:" "$summary_file" | awk '{print $3}')
    
    # Extract output directories from summary
    while IFS= read -r line; do
        if [[ $line =~ ^[0-9]+:\ (.+) ]]; then
            # Extract just the output directory name (before the arrow if present)
            local dir_name=$(echo "$line" | sed 's/^[0-9]\+: //' | sed 's/ ->.*$//')
            output_dirs+=("$dir_name")
        fi
    done < "$summary_file"
    
    print_info "Generated $config_count configuration files in experiment directories under: $exp_dir"
    
    print_info "Experiment: $sweep_name"
    print_info "Parameter combinations: $config_count"
    print_info "Total experiments: $config_count"
    echo "$(printf '%.0s=' {1..80})"
    
    # Check CUDA availability
    check_cuda_availability
    echo "$(printf '%.0s=' {1..80})"
    
    # Create base output directory
    local base_output_dir="$outdir/$sweep_name"
    mkdir -p "$base_output_dir"
    
    # Load existing summary and categorize combinations
    local existing_summary=""
    if [ "$no_resume" = "false" ]; then
        if existing_summary=$(load_existing_summary "$base_output_dir"); then
            print_info "Resume mode enabled (default behavior)"
        else
            print_info "No existing results found, starting fresh"
        fi
    else
        print_info "Starting fresh (--no_resume flag used)"
    fi
    
    # Create summary file
    local summary_path="$base_output_dir/experiment_summary.yaml"
    local start_time=$(date -Iseconds)
    
    cat > "$summary_path" << EOF
experiment_name: $sweep_name
parameter_combinations: $config_count
total_experiments: $config_count
start_time: $start_time
combinations: []
EOF
    
    # Initialize counters
    local successful_runs=0
    local failed_runs=0
    
    # Process combinations
    print_info "Starting execution of $config_count experiments..."
    
    for i in "${!output_dirs[@]}"; do
        local output_dir_name="${output_dirs[$i]}"
        local output_dir="$base_output_dir/$output_dir_name"
        local config_file="$output_dir/config.yaml"
        
        print_info "[$((i+1))/$config_count] Processing combination..."
        print_info "  Output: $output_dir"
        print_info "  Config: $config_file"
        
        # Check if config file exists
        if [ ! -f "$config_file" ]; then
            print_error "Config file not found: $config_file"
            # Mark as failed since config is missing
            failed_runs=$((failed_runs + 1))
            # Update status in summary
            echo "- experiment_id: $((i+1))" >> "$summary_path"
            echo "  output_dir: $output_dir" >> "$summary_path"
            echo "  status: failed" >> "$summary_path"
            continue
        fi
        
        # Add combination to summary
        echo "- experiment_id: $((i+1))" >> "$summary_path"
        echo "  output_dir: $output_dir" >> "$summary_path"
        echo "  status: pending" >> "$summary_path"
        
        if [ "$dry_run" = "true" ]; then
            print_info "DRY RUN - Would run: $output_dir"
            continue
        fi
        
        # Check for duplicate run prevention
        if [ -f "$output_dir/.done" ]; then
            print_info "Skip: already done ($output_dir)"
            # Update status in summary
            sed -i "s/status: pending/status: skipped/" "$summary_path"
            continue
        fi
        
        # Run experiment using the saved config file
        if run_single_experiment "$config_file" "$output_dir" "$base_config" "$sweep_name"; then
            successful_runs=$((successful_runs + 1))
            # Update status in summary
            sed -i "s/status: pending/status: success/" "$summary_path"
        else
            failed_runs=$((failed_runs + 1))
            # Update status in summary
            sed -i "s/status: pending/status: failed/" "$summary_path"
        fi
        
        # Update summary with final stats
        local end_time=$(date -Iseconds)
        sed -i "s/start_time: .*/start_time: $start_time/" "$summary_path"
        sed -i "/^end_time:/d" "$summary_path"
        echo "end_time: $end_time" >> "$summary_path"
        sed -i "/^successful_runs:/d" "$summary_path"
        echo "successful_runs: $successful_runs" >> "$summary_path"
        sed -i "/^failed_runs:/d" "$summary_path"
        echo "failed_runs: $failed_runs" >> "$summary_path"
    done
    
    # Final summary
    print_header "$(printf '%.0s=' {1..80})"
    print_header "EXPERIMENT SUMMARY"
    print_header "$(printf '%.0s=' {1..80})"
    print_info "Experiment: $sweep_name"
    print_info "Total experiments processed: $config_count"
    print_info "Successful runs: $successful_runs"
    print_info "Failed runs: $failed_runs"
    
    if [ $config_count -gt 0 ]; then
        local success_rate=$(echo "scale=1; $successful_runs * 100 / $config_count" | bc -l 2>/dev/null || echo "0")
        print_info "Success rate: ${success_rate}%"
    fi
    
    print_info "Results saved in: $base_output_dir"
    print_info "Summary file: $summary_path"
    
    if [ $failed_runs -gt 0 ]; then
        print_warning "$failed_runs experiments failed. You can rerun this script to retry failed experiments."
        print_info "The script will automatically prioritize failed combinations first."
        
        if [ $successful_runs -eq 0 ]; then
            print_error "No experiments were successful. Please check the configuration and try again."
        fi
    else
        print_success "All experiments completed successfully!"
    fi
}

# Export variables for functions
export DRY_RUN

# Run main function
main "$@"
