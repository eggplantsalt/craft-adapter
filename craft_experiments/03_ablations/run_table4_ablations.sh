#!/bin/bash
#
# run_table4_ablations.sh
#
# Automated script for running Table 4 experiments (Ablation Studies).
# This script evaluates the contribution of each CRaFT component.
#
# Usage:
#   bash craft_experiments/03_ablations/run_table4_ablations.sh
#

set -e  # Exit on error

# ============================================================================
# Configuration
# ============================================================================

# Paths
PROJECT_ROOT="E:/VLA-Adapter"
FINETUNE_SCRIPT="${PROJECT_ROOT}/vla-scripts/finetune.py"
EVAL_SCRIPT="${PROJECT_ROOT}/experiments/robot/libero/run_libero_eval.py"
RESULTS_LOG="${PROJECT_ROOT}/craft_experiments/03_ablations/table4_ablations_results.log"
EVAL_LOGS_DIR="${PROJECT_ROOT}/craft_experiments/03_ablations/eval_logs"

# Model configuration
PRETRAINED_CHECKPOINT="openvla/openvla-7b"
DATA_ROOT_DIR="${PROJECT_ROOT}/datasets/rlds"
RUN_ROOT_DIR="${PROJECT_ROOT}/runs"

# Target dataset for ablation studies (Long-horizon complex tasks)
TASK_SUITE="libero_10"

# Training hyperparameters
BATCH_SIZE=8
LEARNING_RATE=5e-4
MAX_STEPS=20000
SAVE_FREQ=5000
GRAD_ACCUMULATION_STEPS=1

# CRaFT hyperparameters (for full model)
CRAFT_RETENTION_BUDGET=0.1
CRAFT_DUAL_LR=0.01

# Evaluation configuration
NUM_TRIALS_PER_TASK=50
NUM_IMAGES_IN_INPUT=2

# ============================================================================
# Setup
# ============================================================================

echo "=========================================="
echo "CRaFT Table 4: Ablation Studies"
echo "=========================================="
echo ""
echo "Project root: ${PROJECT_ROOT}"
echo "Target dataset: ${TASK_SUITE}"
echo "Results will be saved to: ${RESULTS_LOG}"
echo ""

# Create directories
mkdir -p "${EVAL_LOGS_DIR}"

# Clear previous results
> "${RESULTS_LOG}"

echo "Starting ablation experiments at $(date)" | tee -a "${RESULTS_LOG}"
echo "" | tee -a "${RESULTS_LOG}"
echo "Experimental Setup:" | tee -a "${RESULTS_LOG}"
echo "  - Dataset: ${TASK_SUITE}" | tee -a "${RESULTS_LOG}"
echo "  - Training steps: ${MAX_STEPS}" | tee -a "${RESULTS_LOG}"
echo "  - Ablations: Full, w/o Projection, w/o Dual, AQ Only, Raw Only" | tee -a "${RESULTS_LOG}"
echo "" | tee -a "${RESULTS_LOG}"

# ============================================================================
# Experiment 1: Full CRaFT (Ours)
# ============================================================================

echo "=========================================="
echo "Experiment 1: Full CRaFT (Ours)"
echo "=========================================="
echo ""

RUN_ID="craft-full-${TASK_SUITE}-ablation"
RUN_DIR="${RUN_ROOT_DIR}/${RUN_ID}"

echo "[$(date)] Training Full CRaFT..."

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
    --use_craft True \
    --craft_enable_projection True \
    --craft_enable_dual True \
    --craft_anchor_type concat \
    --craft_retention_budget ${CRAFT_RETENTION_BUDGET} \
    --craft_dual_lr ${CRAFT_DUAL_LR} \
    --wandb_project "craft-table4-ablations" \
    --wandb_entity "your-entity"

TRAIN_EXIT_CODE=$?

if [ ${TRAIN_EXIT_CODE} -ne 0 ]; then
    echo "[ERROR] Full CRaFT training failed with exit code ${TRAIN_EXIT_CODE}"
    echo "full_craft: TRAINING_FAILED" | tee -a "${RESULTS_LOG}"
else
    echo "[$(date)] Full CRaFT training completed"
    
    # Find latest checkpoint
    LATEST_CHECKPOINT=$(ls -td "${RUN_ROOT_DIR}/${RUN_ID}"--*_chkpt 2>/dev/null | head -1)
    
    if [ -z "${LATEST_CHECKPOINT}" ]; then
        echo "[ERROR] No checkpoint found for Full CRaFT"
        echo "full_craft: NO_CHECKPOINT" | tee -a "${RESULTS_LOG}"
    else
        echo "Latest checkpoint: ${LATEST_CHECKPOINT}"
        
        # Evaluate
        echo "[$(date)] Evaluating Full CRaFT..."
        EVAL_LOG_FILE="${EVAL_LOGS_DIR}/eval_full_craft_$(date +%Y%m%d_%H%M%S).txt"
        
        python "${EVAL_SCRIPT}" \
            --pretrained_checkpoint "${LATEST_CHECKPOINT}" \
            --task_suite_name "${TASK_SUITE}" \
            --num_trials_per_task ${NUM_TRIALS_PER_TASK} \
            --use_l1_regression True \
            --use_proprio True \
            --num_images_in_input ${NUM_IMAGES_IN_INPUT} \
            --center_crop True \
            --run_id_note "full-craft" \
            --local_log_dir "${EVAL_LOGS_DIR}" \
            --seed 7 \
            2>&1 | tee "${EVAL_LOG_FILE}"
        
        EVAL_EXIT_CODE=$?
        
        if [ ${EVAL_EXIT_CODE} -ne 0 ]; then
            echo "[ERROR] Full CRaFT evaluation failed"
            echo "full_craft: EVAL_FAILED" | tee -a "${RESULTS_LOG}"
        else
            # Extract success rate
            SUCCESS_RATE=$(python "${PROJECT_ROOT}/craft_experiments/common_utils/log_parser.py" "${EVAL_LOG_FILE}" | grep "Success rate:" | awk '{print $3}')
            
            if [ -z "${SUCCESS_RATE}" ]; then
                echo "[ERROR] Could not extract success rate for Full CRaFT"
                echo "full_craft: PARSE_FAILED" | tee -a "${RESULTS_LOG}"
            else
                echo "Full CRaFT success rate: ${SUCCESS_RATE}"
                echo "full_craft: ${SUCCESS_RATE}" | tee -a "${RESULTS_LOG}"
            fi
        fi
    fi
fi

echo ""

# ============================================================================
# Experiment 2: w/o Projection
# ============================================================================

echo "=========================================="
echo "Experiment 2: w/o Projection"
echo "=========================================="
echo ""

RUN_ID="craft-wo-projection-${TASK_SUITE}-ablation"
RUN_DIR="${RUN_ROOT_DIR}/${RUN_ID}"

echo "[$(date)] Training CRaFT w/o Projection..."

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
    --use_craft True \
    --craft_enable_projection False \
    --craft_enable_dual True \
    --craft_anchor_type concat \
    --craft_retention_budget ${CRAFT_RETENTION_BUDGET} \
    --craft_dual_lr ${CRAFT_DUAL_LR} \
    --wandb_project "craft-table4-ablations" \
    --wandb_entity "your-entity"

TRAIN_EXIT_CODE=$?

if [ ${TRAIN_EXIT_CODE} -ne 0 ]; then
    echo "[ERROR] w/o Projection training failed"
    echo "wo_projection: TRAINING_FAILED" | tee -a "${RESULTS_LOG}"
else
    echo "[$(date)] w/o Projection training completed"
    
    LATEST_CHECKPOINT=$(ls -td "${RUN_ROOT_DIR}/${RUN_ID}"--*_chkpt 2>/dev/null | head -1)
    
    if [ -z "${LATEST_CHECKPOINT}" ]; then
        echo "[ERROR] No checkpoint found"
        echo "wo_projection: NO_CHECKPOINT" | tee -a "${RESULTS_LOG}"
    else
        echo "Latest checkpoint: ${LATEST_CHECKPOINT}"
        
        echo "[$(date)] Evaluating w/o Projection..."
        EVAL_LOG_FILE="${EVAL_LOGS_DIR}/eval_wo_projection_$(date +%Y%m%d_%H%M%S).txt"
        
        python "${EVAL_SCRIPT}" \
            --pretrained_checkpoint "${LATEST_CHECKPOINT}" \
            --task_suite_name "${TASK_SUITE}" \
            --num_trials_per_task ${NUM_TRIALS_PER_TASK} \
            --use_l1_regression True \
            --use_proprio True \
            --num_images_in_input ${NUM_IMAGES_IN_INPUT} \
            --center_crop True \
            --run_id_note "wo-projection" \
            --local_log_dir "${EVAL_LOGS_DIR}" \
            --seed 7 \
            2>&1 | tee "${EVAL_LOG_FILE}"
        
        EVAL_EXIT_CODE=$?
        
        if [ ${EVAL_EXIT_CODE} -ne 0 ]; then
            echo "[ERROR] Evaluation failed"
            echo "wo_projection: EVAL_FAILED" | tee -a "${RESULTS_LOG}"
        else
            SUCCESS_RATE=$(python "${PROJECT_ROOT}/craft_experiments/common_utils/log_parser.py" "${EVAL_LOG_FILE}" | grep "Success rate:" | awk '{print $3}')
            
            if [ -z "${SUCCESS_RATE}" ]; then
                echo "[ERROR] Could not extract success rate"
                echo "wo_projection: PARSE_FAILED" | tee -a "${RESULTS_LOG}"
            else
                echo "w/o Projection success rate: ${SUCCESS_RATE}"
                echo "wo_projection: ${SUCCESS_RATE}" | tee -a "${RESULTS_LOG}"
            fi
        fi
    fi
fi

echo ""

# ============================================================================
# Experiment 3: w/o Dual
# ============================================================================

echo "=========================================="
echo "Experiment 3: w/o Dual (Fixed λ=0.1)"
echo "=========================================="
echo ""

RUN_ID="craft-wo-dual-${TASK_SUITE}-ablation"
RUN_DIR="${RUN_ROOT_DIR}/${RUN_ID}"

echo "[$(date)] Training CRaFT w/o Dual..."

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
    --use_craft True \
    --craft_enable_projection True \
    --craft_enable_dual False \
    --craft_fixed_lambda 0.1 \
    --craft_anchor_type concat \
    --craft_retention_budget ${CRAFT_RETENTION_BUDGET} \
    --wandb_project "craft-table4-ablations" \
    --wandb_entity "your-entity"

TRAIN_EXIT_CODE=$?

if [ ${TRAIN_EXIT_CODE} -ne 0 ]; then
    echo "[ERROR] w/o Dual training failed"
    echo "wo_dual: TRAINING_FAILED" | tee -a "${RESULTS_LOG}"
else
    echo "[$(date)] w/o Dual training completed"
    
    LATEST_CHECKPOINT=$(ls -td "${RUN_ROOT_DIR}/${RUN_ID}"--*_chkpt 2>/dev/null | head -1)
    
    if [ -z "${LATEST_CHECKPOINT}" ]; then
        echo "[ERROR] No checkpoint found"
        echo "wo_dual: NO_CHECKPOINT" | tee -a "${RESULTS_LOG}"
    else
        echo "Latest checkpoint: ${LATEST_CHECKPOINT}"
        
        echo "[$(date)] Evaluating w/o Dual..."
        EVAL_LOG_FILE="${EVAL_LOGS_DIR}/eval_wo_dual_$(date +%Y%m%d_%H%M%S).txt"
        
        python "${EVAL_SCRIPT}" \
            --pretrained_checkpoint "${LATEST_CHECKPOINT}" \
            --task_suite_name "${TASK_SUITE}" \
            --num_trials_per_task ${NUM_TRIALS_PER_TASK} \
            --use_l1_regression True \
            --use_proprio True \
            --num_images_in_input ${NUM_IMAGES_IN_INPUT} \
            --center_crop True \
            --run_id_note "wo-dual" \
            --local_log_dir "${EVAL_LOGS_DIR}" \
            --seed 7 \
            2>&1 | tee "${EVAL_LOG_FILE}"
        
        EVAL_EXIT_CODE=$?
        
        if [ ${EVAL_EXIT_CODE} -ne 0 ]; then
            echo "[ERROR] Evaluation failed"
            echo "wo_dual: EVAL_FAILED" | tee -a "${RESULTS_LOG}"
        else
            SUCCESS_RATE=$(python "${PROJECT_ROOT}/craft_experiments/common_utils/log_parser.py" "${EVAL_LOG_FILE}" | grep "Success rate:" | awk '{print $3}')
            
            if [ -z "${SUCCESS_RATE}" ]; then
                echo "[ERROR] Could not extract success rate"
                echo "wo_dual: PARSE_FAILED" | tee -a "${RESULTS_LOG}"
            else
                echo "w/o Dual success rate: ${SUCCESS_RATE}"
                echo "wo_dual: ${SUCCESS_RATE}" | tee -a "${RESULTS_LOG}"
            fi
        fi
    fi
fi

echo ""

# ============================================================================
# Experiment 4: Anchor - AQ Only
# ============================================================================

echo "=========================================="
echo "Experiment 4: Anchor - AQ Only"
echo "=========================================="
echo ""

RUN_ID="craft-aq-only-${TASK_SUITE}-ablation"
RUN_DIR="${RUN_ROOT_DIR}/${RUN_ID}"

echo "[$(date)] Training CRaFT with AQ Only..."

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
    --use_craft True \
    --craft_enable_projection True \
    --craft_enable_dual True \
    --craft_anchor_type aq_only \
    --craft_retention_budget ${CRAFT_RETENTION_BUDGET} \
    --craft_dual_lr ${CRAFT_DUAL_LR} \
    --wandb_project "craft-table4-ablations" \
    --wandb_entity "your-entity"

TRAIN_EXIT_CODE=$?

if [ ${TRAIN_EXIT_CODE} -ne 0 ]; then
    echo "[ERROR] AQ Only training failed"
    echo "aq_only: TRAINING_FAILED" | tee -a "${RESULTS_LOG}"
else
    echo "[$(date)] AQ Only training completed"
    
    LATEST_CHECKPOINT=$(ls -td "${RUN_ROOT_DIR}/${RUN_ID}"--*_chkpt 2>/dev/null | head -1)
    
    if [ -z "${LATEST_CHECKPOINT}" ]; then
        echo "[ERROR] No checkpoint found"
        echo "aq_only: NO_CHECKPOINT" | tee -a "${RESULTS_LOG}"
    else
        echo "Latest checkpoint: ${LATEST_CHECKPOINT}"
        
        echo "[$(date)] Evaluating AQ Only..."
        EVAL_LOG_FILE="${EVAL_LOGS_DIR}/eval_aq_only_$(date +%Y%m%d_%H%M%S).txt"
        
        python "${EVAL_SCRIPT}" \
            --pretrained_checkpoint "${LATEST_CHECKPOINT}" \
            --task_suite_name "${TASK_SUITE}" \
            --num_trials_per_task ${NUM_TRIALS_PER_TASK} \
            --use_l1_regression True \
            --use_proprio True \
            --num_images_in_input ${NUM_IMAGES_IN_INPUT} \
            --center_crop True \
            --run_id_note "aq-only" \
            --local_log_dir "${EVAL_LOGS_DIR}" \
            --seed 7 \
            2>&1 | tee "${EVAL_LOG_FILE}"
        
        EVAL_EXIT_CODE=$?
        
        if [ ${EVAL_EXIT_CODE} -ne 0 ]; then
            echo "[ERROR] Evaluation failed"
            echo "aq_only: EVAL_FAILED" | tee -a "${RESULTS_LOG}"
        else
            SUCCESS_RATE=$(python "${PROJECT_ROOT}/craft_experiments/common_utils/log_parser.py" "${EVAL_LOG_FILE}" | grep "Success rate:" | awk '{print $3}')
            
            if [ -z "${SUCCESS_RATE}" ]; then
                echo "[ERROR] Could not extract success rate"
                echo "aq_only: PARSE_FAILED" | tee -a "${RESULTS_LOG}"
            else
                echo "AQ Only success rate: ${SUCCESS_RATE}"
                echo "aq_only: ${SUCCESS_RATE}" | tee -a "${RESULTS_LOG}"
            fi
        fi
    fi
fi

echo ""

# ============================================================================
# Experiment 5: Anchor - Raw Only
# ============================================================================

echo "=========================================="
echo "Experiment 5: Anchor - Raw Only"
echo "=========================================="
echo ""

RUN_ID="craft-raw-only-${TASK_SUITE}-ablation"
RUN_DIR="${RUN_ROOT_DIR}/${RUN_ID}"

echo "[$(date)] Training CRaFT with Raw Only..."

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
    --use_craft True \
    --craft_enable_projection True \
    --craft_enable_dual True \
    --craft_anchor_type raw_only \
    --craft_retention_budget ${CRAFT_RETENTION_BUDGET} \
    --craft_dual_lr ${CRAFT_DUAL_LR} \
    --wandb_project "craft-table4-ablations" \
    --wandb_entity "your-entity"

TRAIN_EXIT_CODE=$?

if [ ${TRAIN_EXIT_CODE} -ne 0 ]; then
    echo "[ERROR] Raw Only training failed"
    echo "raw_only: TRAINING_FAILED" | tee -a "${RESULTS_LOG}"
else
    echo "[$(date)] Raw Only training completed"
    
    LATEST_CHECKPOINT=$(ls -td "${RUN_ROOT_DIR}/${RUN_ID}"--*_chkpt 2>/dev/null | head -1)
    
    if [ -z "${LATEST_CHECKPOINT}" ]; then
        echo "[ERROR] No checkpoint found"
        echo "raw_only: NO_CHECKPOINT" | tee -a "${RESULTS_LOG}"
    else
        echo "Latest checkpoint: ${LATEST_CHECKPOINT}"
        
        echo "[$(date)] Evaluating Raw Only..."
        EVAL_LOG_FILE="${EVAL_LOGS_DIR}/eval_raw_only_$(date +%Y%m%d_%H%M%S).txt"
        
        python "${EVAL_SCRIPT}" \
            --pretrained_checkpoint "${LATEST_CHECKPOINT}" \
            --task_suite_name "${TASK_SUITE}" \
            --num_trials_per_task ${NUM_TRIALS_PER_TASK} \
            --use_l1_regression True \
            --use_proprio True \
            --num_images_in_input ${NUM_IMAGES_IN_INPUT} \
            --center_crop True \
            --run_id_note "raw-only" \
            --local_log_dir "${EVAL_LOGS_DIR}" \
            --seed 7 \
            2>&1 | tee "${EVAL_LOG_FILE}"
        
        EVAL_EXIT_CODE=$?
        
        if [ ${EVAL_EXIT_CODE} -ne 0 ]; then
            echo "[ERROR] Evaluation failed"
            echo "raw_only: EVAL_FAILED" | tee -a "${RESULTS_LOG}"
        else
            SUCCESS_RATE=$(python "${PROJECT_ROOT}/craft_experiments/common_utils/log_parser.py" "${EVAL_LOG_FILE}" | grep "Success rate:" | awk '{print $3}')
            
            if [ -z "${SUCCESS_RATE}" ]; then
                echo "[ERROR] Could not extract success rate"
                echo "raw_only: PARSE_FAILED" | tee -a "${RESULTS_LOG}"
            else
                echo "Raw Only success rate: ${SUCCESS_RATE}"
                echo "raw_only: ${SUCCESS_RATE}" | tee -a "${RESULTS_LOG}"
            fi
        fi
    fi
fi

echo ""

# ============================================================================
# Summary
# ============================================================================

echo "=========================================="
echo "All ablation experiments completed at $(date)"
echo "=========================================="
echo ""
echo "Results summary:"
cat "${RESULTS_LOG}"
echo ""
echo "Full results saved to: ${RESULTS_LOG}"
echo "Evaluation logs saved to: ${EVAL_LOGS_DIR}"
echo ""

# Generate comparison table
echo "Generating comparison table..."
python - <<EOF
import sys
import re
from pathlib import Path

def parse_ablation_results(results_log_path):
    results = {}
    try:
        with open(results_log_path, 'r', encoding='utf-8') as f:
            for line in f:
                # Pattern: "full_craft: 0.7600" or "wo_projection: 0.7200"
                match = re.match(r'(full_craft|wo_projection|wo_dual|aq_only|raw_only):\s+([\d.]+)', line.strip())
                if match:
                    config = match.group(1)
                    success_rate = float(match.group(2))
                    results[config] = success_rate
    except Exception as e:
        print(f'Error parsing results log: {e}')
    return results

def format_comparison_table(results):
    # Define order and labels
    config_order = [
        ('full_craft', 'Ours (Full CRaFT)'),
        ('wo_projection', 'w/o Projection'),
        ('wo_dual', 'w/o Dual'),
        ('aq_only', 'Anchor: AQ Only'),
        ('raw_only', 'Anchor: Raw Only'),
    ]
    
    table = '| Configuration | Success Rate | Δ from Full |\n'
    table += '|---------------|--------------|-------------|\n'
    
    full_craft_rate = results.get('full_craft', 0.0)
    
    for config_key, config_label in config_order:
        rate = results.get(config_key, 0.0)
        delta = rate - full_craft_rate
        delta_str = f'{delta:+.4f} ({delta/full_craft_rate*100:+.1f}%)' if full_craft_rate > 0 and config_key != 'full_craft' else '-'
        
        table += f'| {config_label} | {rate:.4f} ({rate*100:.1f}%) | {delta_str} |\n'
    
    return table

results = parse_ablation_results('${RESULTS_LOG}')
table = format_comparison_table(results)

print('\n=== Table 4: Ablation Study Results ===\n')
print(table)

# Save table to file
output_path = '${PROJECT_ROOT}/craft_experiments/03_ablations/table4_ablations_formatted.md'
with open(output_path, 'w', encoding='utf-8') as f:
    f.write('# Table 4: Ablation Study - Component Analysis\n\n')
    f.write(f'**Dataset**: ${TASK_SUITE} (Long-horizon complex tasks)\n\n')
    f.write(table)
    f.write('\n')
    f.write('## Key Findings\n\n')
    f.write('- **Gradient Projection**: Resolves conflicts between action and retention objectives\n')
    f.write('- **Dual Optimization**: Adaptively balances retention constraint vs. fixed weight\n')
    f.write('- **Feature Selection**: Concatenating both C_R and C_AQ provides richer representation\n')
    f.write('  - C_R (Raw Latent): Captures multi-modal fusion from intermediate layers\n')
    f.write('  - C_AQ (ActionQuery): Captures task-specific action semantics from final layer\n')

print(f'\nFormatted table saved to: {output_path}')
EOF

echo ""
echo "Done!"

