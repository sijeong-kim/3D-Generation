#!/bin/bash

# Example usage of the sbatch experiment scripts
# This file shows different ways to run experiments via SLURM

echo "=== Example Usage of SBATCH Experiment Scripts ==="
echo ""

echo "1. Basic usage - run with default parameters:"
echo "   ./submit_exp.sh my_experiment_name"
echo ""

echo "2. Run with custom output directory:"
echo "   ./submit_exp.sh my_experiment --outdir results/my_exp"
echo ""

echo "3. Run with multiple GPUs and custom timeout:"
echo "   ./submit_exp.sh my_experiment --gpus 0,1 --timeout 28800"
echo ""

echo "4. Run with custom config files:"
echo "   ./submit_exp.sh my_experiment --base_config configs/my_base.yaml --sweep_config configs/my_sweep.yaml"
echo ""

echo "5. Dry run to see what would be executed:"
echo "   ./submit_exp.sh my_experiment --dry_run"
echo ""

echo "6. Run with custom CPU cores and memory limits:"
echo "   ./submit_exp.sh my_experiment --cpu_cores 0-15 --mem_soft_mb 65536 --threads 16"
echo ""

echo "7. Direct sbatch submission (if you prefer):"
echo "   sbatch scripts/exp_sbatch/run_exp_sbatch.sh --sweep_name my_experiment --outdir results"
echo ""

echo "=== Available Parameters ==="
echo "  --outdir DIR          Output directory (default: exp)"
echo "  --base_config PATH    Base config file (default: configs/text_ours.yaml)"
echo "  --sweep_config PATH   Sweep config file (default: configs/text_ours_exp_v1.yaml)"
echo "  --gpus CSV            GPU list (default: 0)"
echo "  --cpu_cores STR       CPU cores (default: 0-7)"
echo "  --threads N           Number of threads (default: 8)"
echo "  --mem_soft_mb MB      Memory limit in MB (default: 32768)"
echo "  --timeout SEC         Timeout in seconds (default: 14400)"
echo "  --sleep_between SEC   Sleep between experiments (default: 60)"
echo ""

echo "=== SLURM Resource Allocation ==="
echo "The sbatch script is configured with:"
echo "  - Job name: exp_interactive"
echo "  - Partition: gpgpu"
echo "  - GPUs: 1"
echo "  - CPUs: 8"
echo "  - Memory: 32GB"
echo "  - Time limit: 24 hours"
echo "  - Email notifications: END,FAIL"
echo ""

echo "=== Output Files ==="
echo "SLURM will create output files in:"
echo "  - outputs/<job_id>/exp_output.out  (stdout)"
echo "  - outputs/<job_id>/exp_error.err   (stderr)"
echo ""

echo "=== Monitoring Jobs ==="
echo "To monitor your jobs:"
echo "  squeue -u \$USER                    # Show your jobs"
echo "  scontrol show job <job_id>         # Show job details"
echo "  scancel <job_id>                   # Cancel a job"
echo ""

echo "=== Example Workflow ==="
echo "1. First, do a dry run to check parameters:"
echo "   ./submit_exp.sh my_test --dry_run"
echo ""
echo "2. Submit the actual job:"
echo "   ./submit_exp.sh my_test"
echo ""
echo "3. Monitor the job:"
echo "   squeue -u \$USER"
echo ""
echo "4. Check the output:"
echo "   tail -f outputs/<job_id>/exp_output.out"
echo ""
