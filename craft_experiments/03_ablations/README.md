# Table 4: Ablation Studies

## Overview

This directory contains automated scripts for evaluating the contribution of each CRaFT component through systematic ablation studies.

## Scientific Motivation

**Research Question**: Which components of CRaFT are essential for its superior performance?

**Ablation Strategy**: Remove or modify one component at a time while keeping everything else constant.

## Experimental Design

### Target Dataset
- **libero_10**: Long-horizon complex manipulation tasks (10 tasks)
- **Rationale**: Complex tasks best reveal the importance of each component

### Ablation Configurations

| Configuration | Description | Purpose |
|---------------|-------------|---------|
| **Ours (Full CRaFT)** | All components enabled | Baseline for comparison |
| **w/o Projection** | `craft_enable_projection=False` | Test gradient conflict resolution |
| **w/o Dual** | `craft_enable_dual=False`, `λ=0.1` (fixed) | Test adaptive vs. fixed weighting |
| **Anchor: AQ Only** | `craft_anchor_type=aq_only` | Test action-specific features alone |
| **Anchor: Raw Only** | `craft_anchor_type=raw_only` | Test multi-modal features alone |

### Component Explanations

#### 1. Gradient Projection
- **What**: Conflict-aware projection when action and retention gradients oppose
- **Formula**: $\tilde{g}_{act} = g_{act} - \frac{\langle g_{act}, g_{ret} \rangle}{\|g_{ret}\|^2 + \delta} g_{ret}$ (when $\langle g_{act}, g_{ret} \rangle < 0$)
- **Hypothesis**: Without projection, conflicting gradients harm both objectives

#### 2. Dual Optimization
- **What**: Adaptive λ update based on constraint violation
- **Formula**: $\lambda \leftarrow \max(0, \lambda + \eta_\lambda (\mathcal{L}_{ret} - \varepsilon))$
- **Hypothesis**: Adaptive weighting outperforms fixed λ

#### 3. Feature Selection
- **concat**: $f_\theta = [\text{pool}(C_R); \text{pool}(C_{AQ})]$ (dimension: 2D)
- **aq_only**: $f_\theta = \text{pool}(C_{AQ})$ (dimension: D)
- **raw_only**: $f_\theta = \text{pool}(C_R)$ (dimension: D)
- **Hypothesis**: Both features are complementary and necessary

## Implementation Details

### Feature Dimension Handling

The feature extractor automatically handles dimension changes:

```python
# In CRaFTFeatureExtractor.forward()
if self.anchor_type == "concat":
    combined_features = torch.cat([pooled_raw, pooled_action], dim=-1)  # (B, 2*D)
elif self.anchor_type == "aq_only":
    combined_features = pooled_action  # (B, D)
elif self.anchor_type == "raw_only":
    combined_features = pooled_raw  # (B, D)
```

The retention loss (MSE) automatically adapts to the feature dimension.

### Dual Optimization Control

```python
# In CRaFTDualOptimizer.step()
if not self.enable_dual:
    return  # Keep lambda fixed

# Otherwise, update adaptively
violation = retention_loss - self.budget
self.lambda_val = max(0.0, self.lambda_val + self.dual_lr * violation)
```

## Usage

### Bash (Linux/Mac)
```bash
cd /path/to/VLA-Adapter
bash craft_experiments/03_ablations/run_table4_ablations.sh
```

### PowerShell (Windows)
```powershell
cd E:\VLA-Adapter
powershell -ExecutionPolicy Bypass -File craft_experiments/03_ablations/run_table4_ablations.ps1
```

### Manual Execution

```bash
# Full CRaFT
python vla-scripts/finetune.py \
    --dataset_name libero_10 \
    --use_craft True \
    --craft_enable_projection True \
    --craft_enable_dual True \
    --craft_anchor_type concat

# w/o Projection
python vla-scripts/finetune.py \
    --dataset_name libero_10 \
    --use_craft True \
    --craft_enable_projection False \
    --craft_enable_dual True \
    --craft_anchor_type concat

# w/o Dual (Fixed λ=0.1)
python vla-scripts/finetune.py \
    --dataset_name libero_10 \
    --use_craft True \
    --craft_enable_projection True \
    --craft_enable_dual False \
    --craft_fixed_lambda 0.1 \
    --craft_anchor_type concat

# AQ Only
python vla-scripts/finetune.py \
    --dataset_name libero_10 \
    --use_craft True \
    --craft_enable_projection True \
    --craft_enable_dual True \
    --craft_anchor_type aq_only

# Raw Only
python vla-scripts/finetune.py \
    --dataset_name libero_10 \
    --use_craft True \
    --craft_enable_projection True \
    --craft_enable_dual True \
    --craft_anchor_type raw_only
```

## Output Files

### Results Log
`table4_ablations_results.log` - Raw success rates:
```
full_craft: 0.7600
wo_projection: 0.7200
wo_dual: 0.7300
aq_only: 0.7100
raw_only: 0.7000
```

### Formatted Table
`table4_ablations_formatted.md` - Comparison table with deltas

### Evaluation Logs
`eval_logs/` - Detailed per-task performance for each configuration

## Expected Results

Based on the CRaFT paper, we expect:

1. **Full CRaFT**: Highest performance (baseline)
2. **w/o Projection**: ~3-5% drop (gradient conflicts hurt performance)
3. **w/o Dual**: ~2-4% drop (fixed λ is suboptimal)
4. **AQ Only**: ~5-7% drop (missing multi-modal context)
5. **Raw Only**: ~6-8% drop (missing action-specific semantics)

## Key Insights

- **All components contribute**: Each ablation should show performance degradation
- **Feature complementarity**: Both C_R and C_AQ are necessary
- **Adaptive > Fixed**: Dual optimization outperforms fixed weighting
- **Conflict resolution matters**: Gradient projection prevents objective interference

## Citation

If you use these ablation experiments, please cite:

```bibtex
@article{craft2024,
  title={CRaFT: Constrained Representation and Fine-Tuning for Vision-Language-Action Models},
  author={...},
  journal={arXiv preprint},
  year={2024}
}
```

