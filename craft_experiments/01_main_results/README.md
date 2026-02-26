# CRaFT Experiments - Table 1: Main Results

This directory contains automated scripts for running the main experiments (Table 1) of the CRaFT paper.

## Overview

Table 1 evaluates CRaFT's performance on 4 LIBERO task suites:
- `libero_spatial`: Spatial reasoning tasks
- `libero_object`: Object manipulation tasks
- `libero_goal`: Goal-conditioned tasks
- `libero_10`: Long-horizon tasks (10 tasks)

## Prerequisites

1. **Environment Setup**: Ensure you have installed all dependencies:
   ```bash
   pip install -r requirements.txt
   ```

2. **Data Preparation**: Download LIBERO datasets to `datasets/rlds/`:
   ```bash
   # Follow VLA-Adapter's data preparation instructions
   ```

3. **Pretrained Model**: Ensure you have access to the pretrained checkpoint:
   - Default: `openvla/openvla-7b` (from HuggingFace Hub)
   - Or specify a local path in the script

## Usage

### On Linux/Mac (Bash)

```bash
cd /path/to/VLA-Adapter
bash craft_experiments/01_main_results/run_table1_experiments.sh
```

### On Windows (PowerShell)

```powershell
cd E:\VLA-Adapter
powershell -ExecutionPolicy Bypass -File craft_experiments/01_main_results/run_table1_experiments.ps1
```

## What the Script Does

For each task suite, the script:

1. **Training Phase**:
   - Trains a CRaFT model for 20,000 steps
   - Uses batch size 8, learning rate 5e-4
   - Enables CRaFT with ε=0.1, η_λ=0.01
   - Saves checkpoints every 5,000 steps

2. **Evaluation Phase**:
   - Automatically finds the latest checkpoint
   - Runs 50 evaluation trials per task
   - Logs results to `eval_logs/`

3. **Results Aggregation**:
   - Extracts success rates from evaluation logs
   - Saves results to `table1_results.log`
   - Generates a formatted markdown table

## Configuration

You can modify the following parameters in the script:

### Training Hyperparameters
```bash
BATCH_SIZE=8
LEARNING_RATE=5e-4
MAX_STEPS=20000
SAVE_FREQ=5000
```

### CRaFT Hyperparameters
```bash
CRAFT_RETENTION_BUDGET=0.1    # ε: representation drift budget
CRAFT_DUAL_LR=0.01            # η_λ: dual variable learning rate
CRAFT_ENABLE_PROJECTION=True  # Enable gradient projection
```

### Evaluation Configuration
```bash
NUM_TRIALS_PER_TASK=50        # Number of rollouts per task
NUM_IMAGES_IN_INPUT=2         # Use wrist camera (1=no wrist, 2=with wrist)
```

## Output Files

After running the script, you will find:

```
craft_experiments/01_main_results/
├── table1_results.log          # Raw results (task_suite: success_rate)
├── table1_formatted.md         # Formatted markdown table
└── eval_logs/                  # Detailed evaluation logs
    ├── eval_libero_spatial_*.txt
    ├── eval_libero_object_*.txt
    ├── eval_libero_goal_*.txt
    └── eval_libero_10_*.txt
```

### Example Output

**table1_results.log**:
```
Starting experiments at 2026-02-27 10:00:00

libero_spatial: 0.8500

libero_object: 0.9200

libero_goal: 0.8800

libero_10: 0.7600
```

**table1_formatted.md**:
```markdown
| Task Suite | Success Rate |
|------------|-------------|
| libero_10 | 0.7600 (76.0%) |
| libero_goal | 0.8800 (88.0%) |
| libero_object | 0.9200 (92.0%) |
| libero_spatial | 0.8500 (85.0%) |
|------------|-------------|
| **Average** | **0.8525 (85.2%)** |
```

## Troubleshooting

### Training Fails
- Check GPU memory (requires ~19GB for CRaFT)
- Reduce `BATCH_SIZE` if OOM occurs
- Check data paths are correct

### Evaluation Fails
- Ensure LIBERO environment is properly installed
- Check checkpoint path exists
- Verify `num_images_in_input` matches training configuration

### Success Rate Not Extracted
- Check evaluation log file exists
- Verify log format matches expected pattern
- Run log parser manually: `python common_utils/log_parser.py <log_file>`

## Estimated Runtime

On a single RTX 4090 (24GB):
- Training: ~4-6 hours per task suite (20k steps)
- Evaluation: ~2-3 hours per task suite (50 trials × 10 tasks)
- **Total for all 4 suites: ~24-36 hours**

## Notes

- The script runs experiments **sequentially** (one task suite at a time)
- Training checkpoints are saved to `runs/craft-{task_suite}-table1/`
- Evaluation videos are saved to `experiments/logs/rollout_videos/`
- WandB logging is enabled by default (configure `wandb_entity` in script)

## Next Steps

After completing Table 1 experiments:
1. Review results in `table1_formatted.md`
2. Compare with baseline results (without CRaFT)
3. Proceed to Table 2 (multi-task) and Table 3 (few-shot) experiments
4. Run ablation studies (Table 4)

## Citation

If you use these scripts, please cite:

```bibtex
@article{craft2024,
  title={CRaFT: Constrained Representation and Fine-Tuning for Vision-Language-Action Models},
  author={Your Name},
  journal={arXiv preprint},
  year={2024}
}
```

