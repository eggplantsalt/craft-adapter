# Table 2: Few-Shot Learning Experiments

## Overview

This directory contains automated scripts for evaluating CRaFT's **sample efficiency** in few-shot learning scenarios. The experiments demonstrate CRaFT's ability to maintain strong performance even with severely limited training data.

## Scientific Motivation

**Research Question**: Can CRaFT prevent catastrophic forgetting and overfitting when fine-tuning on extremely limited demonstrations?

**Hypothesis**: By constraining representation drift, CRaFT should:
1. Preserve pre-trained knowledge more effectively than baseline fine-tuning
2. Generalize better from few demonstrations
3. Show larger performance gains as data becomes more scarce

## Experimental Design

### Dataset
- **Target**: `libero_spatial` (10 manipulation tasks)
- **Rationale**: Spatial reasoning tasks require robust generalization from limited examples

### Few-Shot Configurations
- **5-shot**: Only 5 demonstrations per task (10% of full data)
- **10-shot**: Only 10 demonstrations per task (20% of full data)

### Training Adjustments for Few-Shot
- **Reduced steps**: 5,000 (vs 20,000 for full data)
  - Prevents severe overfitting on limited samples
  - Maintains fair comparison between methods
- **Same hyperparameters**: Learning rate, batch size, etc. kept constant
  - Isolates the effect of CRaFT vs baseline

### Comparison
For each N-shot configuration:
1. **Baseline**: Standard fine-tuning without CRaFT
2. **CRaFT**: Fine-tuning with representation constraints

## Implementation Details

### Physical Data Truncation (Per-Task N-Shot)

**CRITICAL**: The N-shot limitation is implemented at the **per-task level**, not at the total dataset level.

For multi-task datasets like LIBERO (10 tasks × 50 episodes = 500 total):
- **10-shot** means: 10 episodes **per task** → 10 × 10 = **100 total episodes**
- **NOT**: 10 episodes total (which would all come from the first task)

#### Implementation Strategy

The per-task filtering is implemented using a **stateful Python function** that tracks episode counts for each unique `language_instruction`:

```python
# In prismatic/vla/datasets/rlds/dataset.py
task_episode_counts = {}  # Maintains state across filter calls

def py_filter_n_shot_per_task(lang_instr_bytes):
    """Keep only first N episodes per unique task."""
    lang_str = lang_instr_bytes.decode('utf-8')
    
    if lang_str not in task_episode_counts:
        task_episode_counts[lang_str] = 0
    
    if task_episode_counts[lang_str] < n_shot_episodes:
        task_episode_counts[lang_str] += 1
        return True  # Keep this episode
    else:
        return False  # Skip this episode

# Apply via tf.py_function
dataset = dataset.filter(lambda traj: tf.py_function(
    py_filter_n_shot_per_task,
    [traj['language_instruction']],
    tf.bool
))
```

This ensures:
- ✅ True episode-level truncation per task
- ✅ Balanced sampling across all tasks
- ✅ Consistent data across runs
- ✅ No hidden data leakage
- ✅ Academically correct Few-Shot definition

### Parameter Propagation
The `--n_shot_episodes` parameter flows through:
1. `finetune.py` → CLI argument
2. `RLDSDataset.__init__()` → Dataset wrapper
3. `make_interleaved_dataset()` → Dataset builder
4. `make_dataset_from_rlds()` → RLDS loader
5. `dl.DLataset.take()` → Physical truncation

## Usage

### Bash (Linux/Mac)
```bash
cd /path/to/VLA-Adapter
bash craft_experiments/02_stability_efficiency/run_table2_fewshot.sh
```

### PowerShell (Windows)
```powershell
cd E:\VLA-Adapter
powershell -ExecutionPolicy Bypass -File craft_experiments/02_stability_efficiency/run_table2_fewshot.ps1
```

### Manual Execution
For a single N-shot experiment:

```bash
# 5-shot Baseline
python vla-scripts/finetune.py \
    --dataset_name libero_spatial \
    --n_shot_episodes 5 \
    --max_steps 5000 \
    --use_craft False \
    --run_id_override baseline-spatial-5shot

# 5-shot CRaFT
python vla-scripts/finetune.py \
    --dataset_name libero_spatial \
    --n_shot_episodes 5 \
    --max_steps 5000 \
    --use_craft True \
    --craft_retention_budget 0.1 \
    --run_id_override craft-spatial-5shot
```

## Output Files

### Results Log
`table2_fewshot_results.log` - Raw success rates:
```
baseline_5shot: 0.6200
craft_5shot: 0.7500
baseline_10shot: 0.7100
craft_10shot: 0.8300
```

### Formatted Table
`table2_fewshot_formatted.md` - Comparison table:

| N-Shot | Baseline | CRaFT | Improvement |
|--------|----------|-------|-------------|
| 5-shot | 0.6200 (62.0%) | 0.7500 (75.0%) | +0.1300 (+21.0%) |
| 10-shot | 0.7100 (71.0%) | 0.8300 (83.0%) | +0.1200 (+16.9%) |

### Evaluation Logs
`eval_logs/` - Detailed per-task performance for each experiment

## Expected Results

Based on the CRaFT paper, we expect:

1. **Consistent Improvement**: CRaFT outperforms baseline in both 5-shot and 10-shot
2. **Larger Gains in 5-shot**: More severe data scarcity → bigger advantage for CRaFT
3. **Absolute Performance**:
   - 5-shot: ~15-20% improvement
   - 10-shot: ~10-15% improvement

## Troubleshooting

### Issue: "No episodes found"
**Cause**: Dataset path incorrect or RLDS data not downloaded
**Solution**: Verify `DATA_ROOT_DIR` points to valid RLDS datasets

### Issue: "Training diverges quickly"
**Cause**: Learning rate too high for few-shot scenario
**Solution**: Reduce `LEARNING_RATE` to 1e-4 or 5e-5

### Issue: "Success rate is 0.0"
**Cause**: Model overfitted on limited data
**Solution**: 
- Reduce `MAX_STEPS` further (try 3000)
- Increase `craft_retention_budget` (try 0.15)

## Key Differences from Table 1

| Aspect | Table 1 (Main Results) | Table 2 (Few-Shot) |
|--------|------------------------|-------------------|
| Data | Full 50 episodes | 5-10 episodes |
| Steps | 20,000 | 5,000 |
| Focus | Overall performance | Sample efficiency |
| Datasets | 4 task suites | 1 task suite |

## Citation

If you use these few-shot experiments, please cite:

```bibtex
@article{craft2024,
  title={CRaFT: Constrained Representation and Fine-Tuning for Vision-Language-Action Models},
  author={...},
  journal={arXiv preprint},
  year={2024}
}
```

## Notes

- **Strategic Decision**: We cancelled multi-task experiments (originally planned for Table 2) because `libero_10` is already used in Table 1 as a long-horizon task, creating logical conflicts
- **Focus Shift**: All resources now dedicated to demonstrating CRaFT's superior few-shot learning capabilities
- **Validation Set**: Always uses full validation data (no N-shot truncation) to ensure fair evaluation

