#!/bin/bash
#
# Grid Search Master Script for Scene Boundary Classification
# Launches 8 experiments in parallel with different hyperparameters
#
# Usage: bash launch_grid_search.sh
#

echo "========================================="
echo "Launching Hyperparameter Grid Search"
echo "Time: $(date)"
echo "========================================="
echo ""

# Create logs directory if it doesn't exist
mkdir -p logs/grid_search

# Define the grid search experiments
# Format: exp_name learning_rate epochs lora_r

experiments=(
    "exp1_baseline      2e-4  5  16"
    "exp2_low_lr        1e-4  5  16"
    "exp3_high_lr       4e-4  5  16"
    "exp4_more_epochs   2e-4  8  16"
    "exp5_fewer_epochs  2e-4  3  16"
    "exp6_high_rank     2e-4  5  32"
    "exp7_low_rank      2e-4  5  8"
    "exp8_aggressive    4e-4  8  32"
)

# Submit each experiment
job_ids=()
for exp in "${experiments[@]}"; do
    # Parse experiment parameters
    read -r exp_name lr epochs lora_r <<< "$exp"

    echo "Submitting experiment: $exp_name"
    echo "  Learning rate: $lr"
    echo "  Epochs: $epochs"
    echo "  LoRA rank: $lora_r"

    # Submit job and capture job ID
    job_output=$(sbatch \
        --job-name="scene-$exp_name" \
        --output="logs/grid_search/${exp_name}_%j.log" \
        --error="logs/grid_search/${exp_name}_%j.err" \
        --nodes=1 \
        --gres=gpu:h100:1 \
        --mem=64G \
        --time=5:00:00 \
        --qos=cs \
        --export=ALL,EXP_NAME=$exp_name,LR=$lr,EPOCHS=$epochs,LORA_R=$lora_r \
        grid_search_worker.sh)

    # Extract job ID from "Submitted batch job 12345"
    job_id=$(echo $job_output | grep -oP '\d+')
    job_ids+=($job_id)

    echo "  Job ID: $job_id"
    echo ""
done

# Save job IDs for reference
echo "========================================="
echo "All jobs submitted!"
echo "Job IDs: ${job_ids[@]}"
echo "========================================="
echo ""
echo "${job_ids[@]}" > logs/grid_search/job_ids.txt

# Show queue status
echo "Current queue status:"
squeue -u $USER

echo ""
echo "To monitor progress:"
echo "  squeue -u $USER"
echo "  tail -f logs/grid_search/exp*_*.log"
echo ""
echo "After completion, run:"
echo "  python collect_results.py"
