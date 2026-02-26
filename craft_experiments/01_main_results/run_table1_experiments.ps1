# run_table1_experiments.ps1
#
# Automated script for running Table 1 experiments (Main Results).
# This script trains CRaFT on all 4 LIBERO task suites and evaluates each checkpoint.
#
# Usage:
#   powershell -ExecutionPolicy Bypass -File craft_experiments/01_main_results/run_table1_experiments.ps1
#

# ============================================================================
# Configuration
# ============================================================================

# Paths
$PROJECT_ROOT = "E:/VLA-Adapter"
$FINETUNE_SCRIPT = "$PROJECT_ROOT/vla-scripts/finetune.py"
$EVAL_SCRIPT = "$PROJECT_ROOT/experiments/robot/libero/run_libero_eval.py"
$RESULTS_LOG = "$PROJECT_ROOT/craft_experiments/01_main_results/table1_results.log"
$EVAL_LOGS_DIR = "$PROJECT_ROOT/craft_experiments/01_main_results/eval_logs"

# Model configuration
$PRETRAINED_CHECKPOINT = "openvla/openvla-7b"
$DATA_ROOT_DIR = "$PROJECT_ROOT/datasets/rlds"
$RUN_ROOT_DIR = "$PROJECT_ROOT/runs"

# Training hyperparameters
$BATCH_SIZE = 8
$LEARNING_RATE = "5e-4"
$MAX_STEPS = 20000
$SAVE_FREQ = 5000
$GRAD_ACCUMULATION_STEPS = 1

# CRaFT hyperparameters
$USE_CRAFT = "True"
$CRAFT_RETENTION_BUDGET = 0.1
$CRAFT_DUAL_LR = 0.01
$CRAFT_ENABLE_PROJECTION = "True"

# Evaluation configuration
$NUM_TRIALS_PER_TASK = 50
$NUM_IMAGES_IN_INPUT = 2

# Task suites for Table 1
$TASK_SUITES = @("libero_spatial", "libero_object", "libero_goal", "libero_10")

# ============================================================================
# Setup
# ============================================================================

Write-Host "==========================================" -ForegroundColor Cyan
Write-Host "CRaFT Table 1 Experiments" -ForegroundColor Cyan
Write-Host "==========================================" -ForegroundColor Cyan
Write-Host ""
Write-Host "Project root: $PROJECT_ROOT"
Write-Host "Results will be saved to: $RESULTS_LOG"
Write-Host ""

# Create directories
New-Item -ItemType Directory -Path $EVAL_LOGS_DIR -Force | Out-Null

# Clear previous results
"" | Out-File -FilePath $RESULTS_LOG -Encoding UTF8

$startTime = Get-Date
"Starting experiments at $startTime" | Tee-Object -FilePath $RESULTS_LOG -Append
"" | Tee-Object -FilePath $RESULTS_LOG -Append

# ============================================================================
# Main Experiment Loop
# ============================================================================

foreach ($TASK_SUITE in $TASK_SUITES) {
    Write-Host "==========================================" -ForegroundColor Yellow
    Write-Host "Task Suite: $TASK_SUITE" -ForegroundColor Yellow
    Write-Host "==========================================" -ForegroundColor Yellow
    
    # Define run ID
    $RUN_ID = "craft-$TASK_SUITE-table1"
    $RUN_DIR = "$RUN_ROOT_DIR/$RUN_ID"
    
    Write-Host "Run ID: $RUN_ID"
    Write-Host "Run directory: $RUN_DIR"
    Write-Host ""
    
    # ------------------------------------------------------------------------
    # Step 1: Training
    # ------------------------------------------------------------------------
    
    Write-Host "[$(Get-Date)] Starting training for $TASK_SUITE..." -ForegroundColor Green
    
    python $FINETUNE_SCRIPT `
        --config_file_path $PRETRAINED_CHECKPOINT `
        --data_root_dir $DATA_ROOT_DIR `
        --dataset_name $TASK_SUITE `
        --run_root_dir $RUN_ROOT_DIR `
        --run_id_override $RUN_ID `
        --batch_size $BATCH_SIZE `
        --learning_rate $LEARNING_RATE `
        --max_steps $MAX_STEPS `
        --save_freq $SAVE_FREQ `
        --grad_accumulation_steps $GRAD_ACCUMULATION_STEPS `
        --use_l1_regression True `
        --use_lora True `
        --lora_rank 32 `
        --use_proprio True `
        --num_images_in_input $NUM_IMAGES_IN_INPUT `
        --image_aug True `
        --use_craft $USE_CRAFT `
        --craft_retention_budget $CRAFT_RETENTION_BUDGET `
        --craft_dual_lr $CRAFT_DUAL_LR `
        --craft_enable_projection $CRAFT_ENABLE_PROJECTION `
        --wandb_project "craft-table1" `
        --wandb_entity "your-entity"
    
    if ($LASTEXITCODE -ne 0) {
        Write-Host "[ERROR] Training failed for $TASK_SUITE with exit code $LASTEXITCODE" -ForegroundColor Red
        "$TASK_SUITE`: TRAINING_FAILED" | Tee-Object -FilePath $RESULTS_LOG -Append
        continue
    }
    
    Write-Host "[$(Get-Date)] Training completed for $TASK_SUITE" -ForegroundColor Green
    Write-Host ""
    
    # ------------------------------------------------------------------------
    # Step 2: Find latest checkpoint
    # ------------------------------------------------------------------------
    
    Write-Host "[$(Get-Date)] Finding latest checkpoint..." -ForegroundColor Green
    
    # Find the latest checkpoint directory
    $checkpointDirs = Get-ChildItem -Path $RUN_ROOT_DIR -Directory -Filter "${RUN_ID}--*_chkpt" -ErrorAction SilentlyContinue
    
    if ($checkpointDirs.Count -eq 0) {
        Write-Host "[ERROR] No checkpoint found for $TASK_SUITE" -ForegroundColor Red
        "$TASK_SUITE`: NO_CHECKPOINT" | Tee-Object -FilePath $RESULTS_LOG -Append
        continue
    }
    
    # Get the latest checkpoint (sort by step number)
    $LATEST_CHECKPOINT = ($checkpointDirs | Sort-Object {
        if ($_.Name -match '--(\d+)_chkpt') { [int]$matches[1] } else { 0 }
    } -Descending | Select-Object -First 1).FullName
    
    Write-Host "Latest checkpoint: $LATEST_CHECKPOINT"
    Write-Host ""
    
    # ------------------------------------------------------------------------
    # Step 3: Evaluation
    # ------------------------------------------------------------------------
    
    Write-Host "[$(Get-Date)] Starting evaluation for $TASK_SUITE..." -ForegroundColor Green
    
    $timestamp = Get-Date -Format "yyyyMMdd_HHmmss"
    $EVAL_LOG_FILE = "$EVAL_LOGS_DIR/eval_${TASK_SUITE}_${timestamp}.txt"
    
    python $EVAL_SCRIPT `
        --pretrained_checkpoint $LATEST_CHECKPOINT `
        --task_suite_name $TASK_SUITE `
        --num_trials_per_task $NUM_TRIALS_PER_TASK `
        --use_l1_regression True `
        --use_proprio True `
        --num_images_in_input $NUM_IMAGES_IN_INPUT `
        --center_crop True `
        --run_id_note "craft-table1" `
        --local_log_dir $EVAL_LOGS_DIR `
        --seed 7 `
        2>&1 | Tee-Object -FilePath $EVAL_LOG_FILE
    
    if ($LASTEXITCODE -ne 0) {
        Write-Host "[ERROR] Evaluation failed for $TASK_SUITE with exit code $LASTEXITCODE" -ForegroundColor Red
        "$TASK_SUITE`: EVAL_FAILED" | Tee-Object -FilePath $RESULTS_LOG -Append
        continue
    }
    
    Write-Host "[$(Get-Date)] Evaluation completed for $TASK_SUITE" -ForegroundColor Green
    Write-Host ""
    
    # ------------------------------------------------------------------------
    # Step 4: Extract success rate
    # ------------------------------------------------------------------------
    
    Write-Host "[$(Get-Date)] Extracting success rate..." -ForegroundColor Green
    
    # Extract success rate from evaluation log
    $logContent = Get-Content $EVAL_LOG_FILE -Raw
    if ($logContent -match "Overall success rate:\s+([\d.]+)\s+\(([\d.]+)%\)") {
        $SUCCESS_RATE = $matches[1]
        Write-Host "Success rate: $SUCCESS_RATE" -ForegroundColor Cyan
        Write-Host ""
        
        # Record result
        "$TASK_SUITE`: $SUCCESS_RATE" | Tee-Object -FilePath $RESULTS_LOG -Append
        "" | Tee-Object -FilePath $RESULTS_LOG -Append
    } else {
        Write-Host "[ERROR] Could not extract success rate for $TASK_SUITE" -ForegroundColor Red
        "$TASK_SUITE`: PARSE_FAILED" | Tee-Object -FilePath $RESULTS_LOG -Append
        continue
    }
    
    Write-Host "==========================================" -ForegroundColor Yellow
    Write-Host ""
}

# ============================================================================
# Summary
# ============================================================================

$endTime = Get-Date
$duration = $endTime - $startTime

Write-Host "==========================================" -ForegroundColor Cyan
Write-Host "All experiments completed at $endTime" -ForegroundColor Cyan
Write-Host "Total duration: $duration" -ForegroundColor Cyan
Write-Host "==========================================" -ForegroundColor Cyan
Write-Host ""
Write-Host "Results summary:" -ForegroundColor Green
Get-Content $RESULTS_LOG
Write-Host ""
Write-Host "Full results saved to: $RESULTS_LOG" -ForegroundColor Green
Write-Host "Evaluation logs saved to: $EVAL_LOGS_DIR" -ForegroundColor Green
Write-Host ""

# Generate markdown table
Write-Host "Generating results table..." -ForegroundColor Green

$pythonScript = @"
import sys
import re
from pathlib import Path

def parse_all_results(results_log_path):
    results = {}
    try:
        with open(results_log_path, 'r', encoding='utf-8') as f:
            for line in f:
                match = re.match(r'(\w+):\s+([\d.]+)', line.strip())
                if match:
                    task_suite = match.group(1)
                    success_rate = float(match.group(2))
                    results[task_suite] = success_rate
    except Exception as e:
        print(f'Error parsing results log: {e}')
    return results

def format_results_table(results):
    table = '| Task Suite | Success Rate |\n'
    table += '|------------|-------------|\n'
    
    for task_suite, success_rate in sorted(results.items()):
        table += f'| {task_suite} | {success_rate:.4f} ({success_rate*100:.1f}%) |\n'
    
    if results:
        avg_success_rate = sum(results.values()) / len(results)
        table += '|------------|-------------|\n'
        table += f'| **Average** | **{avg_success_rate:.4f} ({avg_success_rate*100:.1f}%)** |\n'
    
    return table

results = parse_all_results('$RESULTS_LOG')
table = format_results_table(results)

print('\n=== Table 1: Main Results ===\n')
print(table)

# Save table to file
output_path = '$PROJECT_ROOT/craft_experiments/01_main_results/table1_formatted.md'
with open(output_path, 'w', encoding='utf-8') as f:
    f.write('# Table 1: Main Results - CRaFT on LIBERO\n\n')
    f.write(table)
    f.write('\n')

print(f'\nFormatted table saved to: {output_path}')
"@

$pythonScript | python

Write-Host ""
Write-Host "Done!" -ForegroundColor Green

