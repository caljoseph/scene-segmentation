#!/bin/bash
#
# Grid Search Worker Script
# Runs a single experiment with specified hyperparameters
#
# Environment variables expected:
#   EXP_NAME, LR, EPOCHS, LORA_R
#

# Print job info
echo "========================================="
echo "Grid Search Experiment: $EXP_NAME"
echo "Job ID: $SLURM_JOB_ID"
echo "Node: $SLURM_NODELIST"
echo "Start time: $(date)"
echo "========================================="
echo ""
echo "Hyperparameters:"
echo "  Learning rate: $LR"
echo "  Epochs: $EPOCHS"
echo "  LoRA rank: $LORA_R"
echo ""

# Set offline mode for HuggingFace (no network on compute nodes)
export HF_HUB_OFFLINE=1
export TRANSFORMERS_OFFLINE=1
export HF_DATASETS_OFFLINE=1

# Change to project directory
cd /home/cbradsh4/scene-segmentation

# Show GPU info
echo "GPU Information:"
nvidia-smi --query-gpu=name,memory.total --format=csv,noheader
echo ""

# Activate virtual environment
source .venv/bin/activate

# Run training with specified hyperparameters
echo "Starting training..."
echo ""
python generate_data_and_train_parametric.py \
    --exp_name "$EXP_NAME" \
    --lr "$LR" \
    --epochs "$EPOCHS" \
    --lora_r "$LORA_R"

# Print completion info
echo ""
echo "========================================="
echo "Experiment $EXP_NAME completed at: $(date)"
echo "========================================="
