#!/bin/bash
#
# run_table1_experiments.sh
#
# Automated script for running Table 1 experiments (Main Results).
# This script trains CRaFT on all 4 LIBERO task suites and evaluates each checkpoint.
#
# Usage:
#   bash craft_experiments/01_main_results/run_table1_experiments.sh
#

set -e  # Exit on error

# ============================================================================
# Configuration
# ============================================================================

# Paths
PROJECT_ROOT="E:/VLA-Adapter"
FINETUNE_SCRIPT="${PROJECT_ROOT}/vla-scripts/finetune.py"
EVAL_SCRIPT="${PROJECT_ROOT}/experiments/robot/libero/run_libero_eval.py"
RESULTS_LOG="${PROJECT_ROOT}/craft_experiments/01_main_results/table1_results.log"
EVAL_LOGS_DIR="${PROJECT_ROOT}/craft_experiments/01_main_results/eval_logs"

# Model configuration
PRETRAINED_CHECKPOINT="openvla/openvla-7b"
DATA_ROOT_DIR="${PROJECT_ROOT}/datasets/rlds"
RUN_ROOT_DIR="${PROJECT_ROOT}/runs"

# Training hyperparameters
BATCH_SIZE=8
LEARNING_RATE=5e-4
MAX_STEPS=20000
SAVE_FREQ=5000
GRAD_ACCUMULATION_STEPS=1

# CRaFT hyperparameters
USE_CRAFT=True
CRAFT_RETENTION_BUDGET=0.1
CRAFT_DUAL_LR=0.01
CRAFT_ENABLE_PROJECTION=True

# Evaluation configuration
NUM_TRIALS_PER_TASK=50
NUM_IMAGES_IN_INPUT=2

# Task suites for Table 1
TASK_SUITES=("libero_spatial" "libero_object" "libero_goal" "libero_10")

# ============================================================================
# Setup
# ============================================================================

echo "=========================================="
echo "CRaFT Table 1 Experiments"
echo "=========================================="
echo ""
echo "Project root: ${PROJECT_ROOT}"
echo "Results will be saved to: ${RESULTS_LOG}"
echo ""

# Create directories
mkdir -p "${EVAL_LOGS_DIR}"

# Clear previous results
> "${RESULTS_LOG}"

echo "Starting experiments at $(date)" | tee -a "${RESULTS_LOG}"
echo "" | tee -a "${RESULTS_LOG}"

# ============================================================================
# Main Experiment Loop
# ============================================================================

for TASK_SUITE in "${TASK_SUITES[@]}"; do
    echo "=========================================="
    echo "Task Suite: ${TASK_SUITE}"
    echo "=========================================="
    
    # Define run ID
    RUN_ID="craft-${TASK_SUITE}-table1"
    RUN_DIR="${RUN_ROOT_DIR}/${RUN_ID}"
    
    echo "Run ID: ${RUN_ID}"
    echo "Run directory: ${RUN_DIR}"
    echo ""
    
    # ------------------------------------------------------------------------
    # Step 1: Training
    # ------------------------------------------------------------------------
    
    echo "[$(date)] Starting training for ${TASK_SUITE}..."
    
    python "${FINETUNE_SCRIPT}" \
        --config_file_path "${PRETRAINED_CHECKPOINT}" \
        --data_root_dir "${DATA_ROOT_DIR}" \
        --dataset_name "${TASK_SUITE}" \
        --run_root_dir "${RUN_ROOT_DIR}" \
        --run_id_override "${RUN_ID}" \
        --batch_size ${BATCH_SIZE} \
        --learning_rate ${LEARNING_RATE} \
        --max_steps ${MAX_STEPS} \
        --save_freq ${SAVE_FREQ} \
        --grad_accumulation_steps ${GRAD_ACCUMULATION_STEPS} \
        --use_l1_regression True \
        --use_lora True \
        --lora_rank 32 \
        --use_proprio True \
        --num_images_in_input ${NUM_IMAGES_IN_INPUT} \
        --image_aug True \
        --use_craft ${USE_CRAFT} \
        --craft_retention_budget ${CRAFT_RETENTION_BUDGET} \
        --craft_dual_lr ${CRAFT_DUAL_LR} \
        --craft_enable_projection ${CRAFT_ENABLE_PROJECTION} \
        --wandb_project "craft-table1" \
        --wandb_entity "your-entity"
    
    TRAIN_EXIT_CODE=$?
    
    if [ ${TRAIN_EXIT_CODE} -ne 0 ]; then
        echo "[ERROR] Training failed for ${TASK_SUITE} with exit code ${TRAIN_EXIT_CODE}"
        echo "${TASK_SUITE}: TRAINING_FAILED" | tee -a "${RESULTS_LOG}"
        continue
    fi
    
    echo "[$(date)] Training completed for ${TASK_SUITE}"
    echo ""
    
    # ------------------------------------------------------------------------
    # Step 2: Find latest checkpoint
    # ------------------------------------------------------------------------
    
    echo "[$(date)] Finding latest checkpoint..."
    
    # Find the latest checkpoint directory
    LATEST_CHECKPOINT=$(ls -td "${RUN_ROOT_DIR}/${RUN_ID}"--*_chkpt 2>/dev/null | head -1)
    
    if [ -z "${LATEST_CHECKPOINT}" ]; then
        echo "[ERROR] No checkpoint found for ${TASK_SUITE}"
        echo "${TASK_SUITE}: NO_CHECKPOINT" | tee -a "${RESULTS_LOG}"
        continue
    fi
    
    echo "Latest checkpoint: ${LATEST_CHECKPOINT}"
    echo ""
    
    # ------------------------------------------------------------------------
    # Step 3: Evaluation
    # ------------------------------------------------------------------------
    
    echo "[$(date)] Starting evaluation for ${TASK_SUITE}..."
    
    EVAL_LOG_FILE="${EVAL_LOGS_DIR}/eval_${TASK_SUITE}_$(date +%Y%m%d_%H%M%S).txt"
    
    python "${EVAL_SCRIPT}" \
        --pretrained_checkpoint "${LATEST_CHECKPOINT}" \
        --task_suite_name "${TASK_SUITE}" \
        --num_trials_per_task ${NUM_TRIALS_PER_TASK} \
        --use_l1_regression True \
        --use_proprio True \
        --num_images_in_input ${NUM_IMAGES_IN_INPUT} \
        --center_crop True \
        --run_id_note "craft-table1" \
        --local_log_dir "${EVAL_LOGS_DIR}" \
        --seed 7 \
        2>&1 | tee "${EVAL_LOG_FILE}"
    
    EVAL_EXIT_CODE=$?
    
    if [ ${EVAL_EXIT_CODE} -ne 0 ]; then
        echo "[ERROR] Evaluation failed for ${TASK_SUITE} with exit code ${EVAL_EXIT_CODE}"
        echo "${TASK_SUITE}: EVAL_FAILED" | tee -a "${RESULTS_LOG}"
        continue
    fi
    
    echo "[$(date)] Evaluation completed for ${TASK_SUITE}"
    echo ""
    
    # ------------------------------------------------------------------------
    # Step 4: Extract success rate
    # ------------------------------------------------------------------------
    
    echo "[$(date)] Extracting success rate..."
    
    # Extract success rate from evaluation log
    SUCCESS_RATE=$(python "${PROJECT_ROOT}/craft_experiments/common_utils/log_parser.py" "${EVAL_LOG_FILE}" | grep "Success rate:" | awk '{print $3}')
    
    if [ -z "${SUCCESS_RATE}" ]; then
        echo "[ERROR] Could not extract success rate for ${TASK_SUITE}"
        echo "${TASK_SUITE}: PARSE_FAILED" | tee -a "${RESULTS_LOG}"
        continue
    fi
    
    echo "Success rate: ${SUCCESS_RATE}"
    echo ""
    
    # Record result
    echo "${TASK_SUITE}: ${SUCCESS_RATE}" | tee -a "${RESULTS_LOG}"
    echo "" | tee -a "${RESULTS_LOG}"
    
    echo "=========================================="
    echo ""
    
done

# ============================================================================
# Summary
# ============================================================================

echo "=========================================="
echo "All experiments completed at $(date)"
echo "=========================================="
echo ""
echo "Results summary:"
cat "${RESULTS_LOG}"
echo ""
echo "Full results saved to: ${RESULTS_LOG}"
echo "Evaluation logs saved to: ${EVAL_LOGS_DIR}"
echo ""

# Generate markdown table
echo "Generating results table..."
python - <<EOF
import sys
sys.path.append("${PROJECT_ROOT}/craft_experiments/common_utils")
from log_parser import parse_all_results, format_results_table

results = parse_all_results("${RESULTS_LOG}")
table = format_results_table(results)

print("\n=== Table 1: Main Results ===\n")
print(table)

# Save table to file
with open("${PROJECT_ROOT}/craft_experiments/01_main_results/table1_formatted.md", "w") as f:
    f.write("# Table 1: Main Results - CRaFT on LIBERO\n\n")
    f.write(table)
    f.write("\n")

print("\nFormatted table saved to: craft_experiments/01_main_results/table1_formatted.md")
EOF

echo ""
echo "Done!"

