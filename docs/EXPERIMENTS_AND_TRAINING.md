# 训练与评估完全指南 (Training and Evaluation Guide)

本文档提供 VLA-Adapter + CRaFT 训练的完整指南，包括训练配置、指标解读、实验脚本使用和常见问题排查。

---

## 📋 目录

1. [训练配置详解](#training-config)
2. [Baseline vs CRaFT 对比](#baseline-vs-craft)
3. [核心指标深度解读](#metrics-explained)
4. [自动化实验脚本使用](#experiment-scripts)
5. [训练监控与调试](#monitoring-debugging)
6. [常见报错与排查](#troubleshooting)

---

## <a name="training-config"></a>1. 训练配置详解

### 1.1 基础训练参数

```bash
CUDA_VISIBLE_DEVICES=0 torchrun --standalone --nnodes 1 --nproc-per-node 1 vla-scripts/finetune.py \
  --vlm_path pretrained_models/prism-qwen25-extra-dinosiglip-224px-0_5b \
  --config_file_path pretrained_models/configs \
  --data_root_dir data/libero \
  --dataset_name "libero_spatial_no_noops" \
  --batch_size 4 \
  --grad_accumulation_steps 4 \
  --learning_rate 2e-4 \
  --num_steps_before_decay 150000 \
  --max_steps 20000 \
  --save_freq 5000 \
  --use_wandb True \
  --wandb_project "vla-experiments"
```

也可以直接使用仓库内置参数化脚本：

```bash
bash vla-scripts/run_finetune_libero.sh
```

**参数说明**：

- **`config_file_path`**：预训练 VLA 模型的路径
  - 可以是 HuggingFace Hub 上的模型 ID（如 `openvla/openvla-7b`）
  - 也可以是本地路径（如 `runs/previous-experiment--10000_chkpt`）

- **`batch_size`**：单 GPU 的批次大小
  - 总批次大小 = `batch_size × num_gpus × grad_accumulation_steps`
  - 推荐值：8-16（取决于 GPU 显存）

- **`learning_rate`**：优化器学习率
  - Baseline 推荐：5e-4
  - CRaFT 推荐：5e-4（与 Baseline 保持一致）

- **`max_steps`**：最大训练步数
  - LIBERO 数据集推荐：20000-50000 步
  - Few-Shot 场景可适当减少

- **`num_steps_before_decay`**：学习率衰减里程碑
  - 当前脚本使用 MultiStepLR，在该步后将学习率乘以 0.1

- **`use_wandb`**：是否启用 WandB 记录
  - `True`：初始化 WandB 并记录训练指标
  - `False`：不初始化 WandB，仅保留终端/本地日志

- **`console_log_freq`**：终端历史日志输出频率（按 step）
  - 同步写入运行目录下 `train_progress.log`

### 1.2 动作表示配置

```bash
--use_l1_regression True \     # 使用连续动作空间 + L1 回归
--use_diffusion False          # 不使用扩散模型（CRaFT 暂不支持）
```

**说明**：
- CRaFT 目前仅支持 **L1 回归** 的连续动作表示
- 离散动作表示（Next-Token Prediction）和扩散模型暂不支持

### 1.3 数据增强配置

```bash
--image_aug True \             # 启用图像增强（强烈推荐）
--shuffle_buffer_size 100000   # 数据打乱缓冲区大小
```

**图像增强的作用**：
- 提升模型对视觉变化的鲁棒性
- 防止过拟合
- **强烈推荐保持启用**

---

## <a name="baseline-vs-craft"></a>2. Baseline vs CRaFT 对比

### 2.1 Baseline 训练（标准 VLA 微调）

```bash
CUDA_VISIBLE_DEVICES=0 torchrun --standalone --nnodes 1 --nproc-per-node 1 vla-scripts/finetune.py \
  --vlm_path pretrained_models/prism-qwen25-extra-dinosiglip-224px-0_5b \
  --config_file_path pretrained_models/configs \
  --data_root_dir data/libero \
  --dataset_name "libero_spatial_no_noops" \
  --batch_size 4 \
  --grad_accumulation_steps 4 \
  --learning_rate 2e-4 \
  --num_steps_before_decay 150000 \
  --max_steps 20000 \
  --use_craft False \
  --use_wandb False \
  --console_log_freq 10 \
  --wandb_project "vla-baseline"
```

**Baseline 特点**：
- 仅优化动作预测损失 L_act
- 无表征保留约束
- 容易出现表征坍塌（尤其在 Few-Shot 场景）

### 2.2 CRaFT 训练（约束优化微调）

```bash
CUDA_VISIBLE_DEVICES=0 torchrun --standalone --nnodes 1 --nproc-per-node 1 vla-scripts/finetune.py \
  --vlm_path pretrained_models/prism-qwen25-extra-dinosiglip-224px-0_5b \
  --config_file_path pretrained_models/configs \
  --data_root_dir data/libero \
  --dataset_name "libero_spatial_no_noops" \
  --batch_size 4 \
  --grad_accumulation_steps 4 \
  --learning_rate 2e-4 \
  --num_steps_before_decay 150000 \
  --max_steps 20000 \
  --use_craft True \
  --craft_retention_budget 0.1 \
  --craft_dual_lr 0.01 \
  --craft_enable_projection True \
  --craft_enable_dual True \
  --craft_anchor_type "concat" \
  --use_wandb True \
  --console_log_freq 10 \
  --wandb_project "vla-craft"
```

**CRaFT 核心参数详解**：

| 参数 | 默认值 | 推荐范围 | 说明 |
|------|--------|----------|------|
| `craft_retention_budget` | 0.1 | 0.05-0.2 | 允许的最大表征漂移量 ε（越小越保守） |
| `craft_dual_lr` | 0.01 | 0.001-0.1 | 对偶变量学习率 η_λ（控制 λ 更新速度） |
| `craft_enable_projection` | True | True/False | 是否启用冲突感知梯度投影 |
| `craft_enable_dual` | True | True/False | 是否启用自适应 λ（False 则使用固定 λ） |
| `craft_fixed_lambda` | 0.1 | 0.01-1.0 | 当 `enable_dual=False` 时的固定 λ 值 |
| `craft_anchor_type` | "concat" | concat/aq_only/raw_only | 锚点特征类型（消融实验用） |
| `craft_anchor_layer_idx` | None | 整数 | C_R 的 hidden_states 层索引（None=中间层，负数=从末尾倒数） |
| `craft_cr_token_mode` | vision_only | vision_only/vision_plus_prompt | C_R 使用的 token 范围 |

---

## <a name="metrics-explained"></a>3. 核心指标深度解读

### 3.1 标准训练指标

#### `VLA Train/Loss`（动作预测损失）
- **定义**：L1 回归损失，衡量预测动作与真实动作的差距
- **计算公式**：`L_act = ||π_θ(o, l) - A_t||_1`
- **期望趋势**：持续下降，最终收敛到 0.05-0.15
- **异常情况**：
  - 损失不下降：学习率过小或数据问题
  - 损失震荡：学习率过大或批次大小过小

#### `VLA Train/Curr Action L1 Loss`（当前动作损失）
- **定义**：仅计算当前时间步动作的 L1 损失
- **意义**：反映模型对即时动作的预测能力
- **期望值**：< 0.1（越小越好）

#### `VLA Train/Next Actions L1 Loss`（未来动作损失）
- **定义**：计算未来 K 步动作的平均 L1 损失
- **意义**：反映模型的长期规划能力
- **期望值**：通常比当前动作损失略高（0.1-0.2）

### 3.2 CRaFT 核心指标（⭐ 论文关键证据）

#### `CRaFT/Retention Loss`（表征保留损失）
- **定义**：当前特征与锚点特征的均方误差
- **计算公式**：`L_ret = ||f_θ(o, l) - f̃(o, l)||²`
- **物理意义**：衡量模型表征相对于预训练状态的漂移程度
- **期望趋势**：
  - 训练初期：快速上升（模型开始适应下游任务）
  - 训练中期：稳定在 ε 附近（对偶优化生效）
  - 训练后期：保持稳定或略微下降
- **异常情况**：
  - 持续上升超过 ε：对偶学习率 `craft_dual_lr` 过小
  - 始终接近 0：`craft_retention_budget` 过小，模型被过度约束

#### `CRaFT/Retention NonFinite`
- **定义**：表征保留损失是否出现非有限数（`NaN/Inf`）的标记（0 或 1）
- **意义**：用于快速定位异常 batch 的数值稳定性问题
- **期望值**：长期保持 0

#### `CRaFT/Lambda`（拉格朗日乘子）
- **定义**：对偶变量 λ，控制表征保留损失的权重
- **更新规则**：`λ ← max(0, λ + η_λ × (L_ret - ε))`
- **物理意义**：
  - λ 增大：模型表征漂移超出预算，需要加强约束
  - λ 减小（趋向 0）：表征漂移在预算内，可以放松约束
- **期望趋势**：
  - 训练初期：从 0 快速上升
  - 训练中期：在某个稳定值附近震荡
  - 训练后期：保持稳定
- **典型值范围**：0.1-1.0（取决于 `craft_dual_lr` 和 `craft_retention_budget`）

#### `CRaFT/Lambda Before` / `CRaFT/Lambda After`
- **定义**：同一个 step 内，对偶变量更新前后的 λ
- **意义**：直接观测该 step 是否触发 λ 更新
- **经验解读**：若两者都接近 0，通常代表当前 `L_ret <= ε`，不一定是实现问题

#### `CRaFT/Conflict Ratio`（⭐ 梯度冲突率，论文核心指标）
- **定义**：在当前 step 中，参与 CRaFT 梯度合并的参数张量里，出现"动作梯度"与"表征梯度"几何冲突的张量比例
- **计算公式**：`Conflict Ratio = (冲突参数张量数) / (总参数张量数)`
- **冲突判定**：当 `<g_act, g_ret> < 0` 时，认为发生冲突
- **物理意义**：
  - **高冲突率 (>30%)**：模型正在经历严重的表征坍塌，动作优化与表征保留存在大量冲突
  - **低冲突率 (<10%)**：两个优化目标基本一致，表征稳定
  - **中等冲突率 (10%-30%)**：正常的优化过程，CRaFT 的梯度投影机制正在化解冲突
- **期望趋势**：
  - **Baseline（无 CRaFT）**：冲突率持续高企（30%-50%），说明表征坍塌严重
  - **CRaFT（有梯度投影）**：冲突率逐渐下降并稳定在较低水平（5%-15%）
- **论文价值**：
  - 这是证明"表征坍塌"现象存在的直接证据
  - CRaFT 的梯度投影能有效化解冲突，是论文的核心卖点

### 3.3 工程健壮性指标（Phase 7.5 新增）

#### `VLA Train/Gradient Norm`（梯度范数）
- **定义**：所有可训练参数梯度的 L2 范数
- **计算公式**：`||∇θ||_2 = sqrt(Σ ||∇θ_i||²)`
- **物理意义**：衡量梯度的整体大小，反映训练稳定性
- **期望趋势**：
  - 训练初期：较大（1.0-10.0）
  - 训练中期：逐渐下降
  - 训练后期：稳定在较小值（0.1-1.0）
- **异常情况**：
  - 梯度爆炸：梯度范数突然飙升到 >100，需要降低学习率或启用梯度裁剪
  - 梯度消失：梯度范数 <0.001，模型停止学习

#### `VLA Train/Learning Rate`（学习率）
- **定义**：当前优化器的学习率
- **调度策略**：
  - Warmup 阶段：从 10% 线性增长到 100%
  - 稳定阶段：保持恒定
  - Decay 阶段：在 `num_steps_before_decay` 后衰减 10 倍
- **期望趋势**：
  - 0-1000 步：从 5e-5 增长到 5e-4（Warmup）
  - 1000-20000 步：保持 5e-4
  - 20000+ 步：衰减到 5e-5

---

## <a name="experiment-scripts"></a>4. 自动化实验脚本使用

我们提供了完整的自动化脚本，可一键复现论文中的所有实验。

### 4.1 主实验：Table 1（LIBERO 四个 Suite）

**脚本位置**：`craft_experiments/01_main_results/run_table1_experiments.sh`

**运行方式**：
```bash
cd craft_experiments/01_main_results
bash run_table1_experiments.sh
```

**实验内容**：
- 在 LIBERO 四个 Suite（Spatial, Object, Goal, Long）上分别运行：
  - Baseline 训练（无 CRaFT）
  - CRaFT 训练（完整配置）
- 每个实验使用 3 个随机种子（42, 123, 456）
- 自动保存 WandB 日志和模型 Checkpoint

**预期运行时间**：
- 单个实验：约 4-8 小时（取决于 GPU 性能）
- 总计：约 3-5 天（8 个配置 × 3 个种子 × 4-8 小时）

**输出结果**：
- WandB 日志：`wandb_project/table1-{suite}-{method}-seed{seed}`
- Checkpoint：`runs/table1-{suite}-{method}-seed{seed}/`

### 4.2 极少样本实验（Few-Shot）

**脚本位置**：`craft_experiments/02_stability_efficiency/run_table2_fewshot.sh`

**运行方式**：
```bash
cd craft_experiments/02_stability_efficiency
bash run_table2_fewshot.sh
```

**实验内容**：
- 在 LIBERO-Spatial 上测试：
  - 5-Shot（每个任务 5 条轨迹）
  - 10-Shot（每个任务 10 条轨迹）
- 对比 Baseline vs CRaFT
- 每个实验使用 3 个随机种子

**核心参数**：
```bash
--n_shot_episodes 5   # 或 10
```

### 4.3 消融实验（Ablation Study）

**脚本位置**：`craft_experiments/03_ablations/run_table4_ablations.sh`

**运行方式**：
```bash
cd craft_experiments/03_ablations
bash run_table4_ablations.sh
```

**实验内容**：

| 实验名称 | 配置 | 目的 |
|---------|------|------|
| CRaFT (Full) | `projection=True, dual=True` | 完整 CRaFT |
| w/o Projection | `projection=False, dual=True` | 验证梯度投影的作用 |
| w/o Dual | `projection=True, dual=False, fixed_lambda=0.1` | 验证自适应 λ 的作用 |
| AQ Only | `anchor_type=aq_only` | 仅使用动作查询特征 |
| Raw Only | `anchor_type=raw_only` | 仅使用原始潜在特征 |

---

## <a name="monitoring-debugging"></a>5. 训练监控与调试

### 5.1 实时监控 WandB 日志

训练启动后，访问 WandB 项目页面查看实时指标：

```
https://wandb.ai/{your-entity}/{your-project}
```

**关键图表**：
1. **Loss 曲线**：`VLA Train/Loss` 应持续下降
2. **CRaFT 指标**：
   - `CRaFT/Retention Loss` 应稳定在 ε 附近
   - `CRaFT/Lambda` 应在训练中期稳定
   - `CRaFT/Conflict Ratio` 应逐渐下降
3. **梯度范数**：`VLA Train/Gradient Norm` 应保持稳定

### 5.2 终端日志解读

训练过程中，终端会显示实时进度条：

```
Step 1234/20000 | Loss: 0.1234 | Ret: 0.0821/0.1000 | λ: 0.000->0.000 | Conflict: 12.34% | GradNorm: 1.23 | LR: 5.00e-04
```

**字段说明**：
- `Step`：当前 step / 最大 step
- `Loss`：当前批次的动作损失
- `Ret`：表征保留损失 / 预算 ε
- `λ`：当前 step 的 λ 更新（前->后）
- `Conflict`：当前批次的梯度冲突率
- `GradNorm`：当前批次的梯度范数
- `LR`：当前学习率

此外，历史日志会按 `console_log_freq` 频率写入运行目录下 `train_progress.log`。

### 5.3 关于 epoch 与 step

- 当前 RLDS 训练链路是 step-driven：底层数据在 interleave 阶段使用 `repeat()` 持续供数。
- 训练停止条件由 `max_steps` 控制，而不是传统 epoch 结束。
- 调参与监控应重点关注 step 相关参数（如 `max_steps`、`save_freq`、`wandb_log_freq`、`console_log_freq`）。

### 5.3 Checkpoint 管理

**保存策略**：
- 默认每 5000 步保存一次 Checkpoint
- 可通过 `--save_freq` 调整保存频率
- 可通过 `--save_latest_checkpoint_only True` 仅保存最新 Checkpoint（节省磁盘空间）

**Checkpoint 内容**：
```
runs/experiment-name--10000_chkpt/
├── config.json                    # 模型配置
├── processor_config.json          # 处理器配置
├── lora_adapter/                  # LoRA 权重（如果使用 LoRA）
├── action_head--10000_checkpoint.pt        # 动作头权重
├── dataset_statistics.json        # 数据集统计信息
└── training_state.pt              # 训练状态（Optimizer + Scheduler）
```

### 5.4 断点续训

```bash
python vla-scripts/finetune.py \
  --config_file_path "outputs/experiment-name--10000_chkpt" \
    --resume True \
    --resume_step 10000 \
  --resum_vla_path "outputs/experiment-name--10000_chkpt" \
    --max_steps 20000 \
    --use_craft True
```

或使用仓库脚本（推荐）：

```bash
RESUME=True \
RESUME_STEP=10000 \
RESUME_VLA_PATH=outputs/experiment-name--10000_chkpt \
bash vla-scripts/run_finetune_libero.sh
```

**说明**：
- 脚本会校验 `RESUME_STEP` 与 `RESUME_VLA_PATH` 是否匹配，并在可推断时自动修正 step
- 当前主流程恢复的是模块 checkpoint（如 `action_head`、`proprio_projector` 等）并从 `resume_step` 继续计步
- `training_state.pt` 已保存 Optimizer/Scheduler 状态，用于后续扩展完整状态恢复

---

## <a name="troubleshooting"></a>6. 常见报错与排查

### 问题 1：CUDA Out of Memory (OOM)

**错误信息**：
```
RuntimeError: CUDA out of memory. Tried to allocate X.XX GiB
```

**原因**：GPU 显存不足。

**解决方案**：
```bash
# 方案 1：减小 Batch Size
--batch_size 4  # 从 8 降低到 4

# 方案 2：启用梯度累积（保持总批次大小不变）
--batch_size 4 \
--grad_accumulation_steps 2  # 总批次 = 4 × 2 = 8

# 方案 3：减小 Shuffle Buffer Size
--shuffle_buffer_size 10000  # 从 100000 降低到 10000
```

### 问题 2：Loss 不下降或震荡

**可能原因**：
1. 学习率过大或过小
2. 批次大小过小
3. 数据问题

**排查步骤**：
```bash
# 1. 降低学习率
--learning_rate 1e-4  # 从 5e-4 降低到 1e-4

# 2. 增大批次大小
--batch_size 16 \
--grad_accumulation_steps 2  # 总批次 = 32

# 3. 检查数据集是否正确加载
python -c "
from prismatic.vla.datasets import RLDSDataset
dataset = RLDSDataset(...)
print(f'Dataset size: {len(dataset)}')
"
```

### 问题 3：CRaFT Retention Loss 持续上升

**错误现象**：`CRaFT/Retention Loss` 远超 `craft_retention_budget`，且 `Lambda` 持续增大。

**原因**：对偶学习率 `craft_dual_lr` 过小，λ 更新速度不够快。

**解决方案**：
```bash
# 增大对偶学习率
--craft_dual_lr 0.05  # 从 0.01 增大到 0.05
```

### 问题 4：梯度爆炸

**错误信息**：
```
Loss: nan | GradNorm: inf
```

**原因**：梯度范数过大，导致数值不稳定。

**解决方案**：
```bash
# 方案 1：降低学习率
--learning_rate 1e-4

# 方案 2：启用梯度裁剪（需修改代码）
# 在 finetune.py 中添加：
torch.nn.utils.clip_grad_norm_(trainable_params, max_norm=1.0)
```

### 问题 5：DDP 多卡训练报错

**错误信息**：
```
RuntimeError: Expected to have finished reduction in the prior iteration
```

**原因**：DDP 梯度同步问题，通常由 `find_unused_parameters` 引起。

**解决方案**：
```python
# 在 finetune.py 中检查 DDP 配置
vla = DDP(vla, device_ids=[device_id], find_unused_parameters=True)
```

### 问题 6：WandB 离线模式

**现象**：WandB 日志未上传到云端。

**原因**：训练脚本默认使用 `mode="offline"`，或设置了 `--use_wandb False`。

**解决方案**：
```python
# 修改 finetune.py 中的 WandB 初始化
wandb.init(project=cfg.wandb_project, name=f"ft+{run_id}", mode="online")  # 改为 online
```

或直接通过参数关闭 WandB：
```bash
--use_wandb False
```

---

## 📚 相关文档

- **[数据集准备指南](DATASETS.md)**：LIBERO 数据集下载和配置
- **[项目结构详解](craft/PROJECT_STRUCTURE.md)**：代码库架构深度解析
- **[主 README](../README.md)**：项目概览和快速开始

---

## 🤝 需要帮助？

如果遇到训练相关问题：
1. 查看本文档的"常见报错与排查"章节
2. 检查 WandB 日志中的异常指标
3. 提交 GitHub Issue 并附上：
   - 完整的训练命令
   - 错误日志
   - WandB 日志链接（如果有）

---

**最后更新**：2024-02-27 | **维护者**：VLA-Adapter Team

