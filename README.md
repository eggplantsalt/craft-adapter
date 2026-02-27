# VLA-Adapter with CRaFT

<div align="center">

**CRaFT: 克服视觉-语言-动作模型微调中的表征坍塌**

*Constrained Representation and Fine-Tuning for Vision-Language-Action Models*

[English](#english) | [中文文档](#chinese)

---

</div>

## <a name="chinese"></a>🇨🇳 中文文档

### 📖 项目简介

本项目在开源的 **VLA-Adapter** (基于 Prismatic VLM) 基础上，实现了我们提出的 **CRaFT (Constrained Representation and Fine-Tuning)** 训练框架。

**核心问题**：在标准的视觉-语言-动作 (VLA) 模型微调中，仅使用低维的动作监督信号会导致严重的"**表征坍塌 (Representation Collapse)**"——模型为了走捷径拟合下游动作任务，破坏了从预训练 VLM 中继承的通用多模态感知能力。

**CRaFT 解决方案**：将下游微调显式表述为一个**带有表征漂移预算的约束优化问题**，通过**冲突感知梯度投影**化解优化冲突，从而实现稳定、高泛化性的微调。

---

### 🎯 核心贡献

1. **在线权重切换 (Online Weight Swapping)**：避免离线缓存带来的数据对齐风险，零显存开销提取锚点特征
2. **冲突感知梯度投影**：仅在检测到梯度冲突时触发投影，保持训练效率
3. **自适应对偶优化**：动态调整拉格朗日乘子 λ，平衡动作性能和表征保留
4. **梯度冲突率统计**：实时监控训练过程中的梯度冲突频率，为论文提供核心实验证据

---

### 🚀 快速开始 (Quickstart)

#### 1. 环境安装

```bash
# 克隆仓库
git clone https://github.com/your-repo/VLA-Adapter.git
cd VLA-Adapter

# 创建 Conda 环境 (推荐 Python 3.10)
conda create -n vla python=3.10 -y
conda activate vla

# 安装核心依赖
pip install -e .
pip install -r requirements.txt

# 安装 LIBERO 数据集支持 (如果使用 LIBERO)
cd LIBERO
pip install -e .
cd ..
```

#### 2. 验证 Baseline 能否运行

在开始 CRaFT 训练前，建议先验证原版 VLA-Adapter 的 Baseline 训练能否正常运行：

```bash
# 测试 Baseline 训练 (不使用 CRaFT)
python vla-scripts/finetune.py \
    --config_file_path "openvla/openvla-7b" \
    --dataset_name "libero_spatial_no_noops" \
    --batch_size 8 \
    --learning_rate 5e-4 \
    --max_steps 5000 \
    --use_l1_regression True \
    --use_craft False \
    --wandb_project "vla-baseline-test"
```

如果上述命令能正常运行并开始训练，说明环境配置正确。

#### 3. 启动 CRaFT 训练

```bash
# CRaFT 训练 (启用所有核心功能)
python vla-scripts/finetune.py \
    --config_file_path "openvla/openvla-7b" \
    --dataset_name "libero_spatial_no_noops" \
    --batch_size 8 \
    --learning_rate 5e-4 \
    --max_steps 20000 \
    --use_l1_regression True \
    --use_craft True \
    --craft_retention_budget 0.1 \
    --craft_dual_lr 0.01 \
    --craft_enable_projection True \
    --craft_enable_dual True \
    --wandb_project "vla-craft-training"
```

**关键参数说明**：
- `--use_craft True`：启用 CRaFT 训练框架
- `--craft_retention_budget 0.1`：表征漂移预算 ε (论文核心超参数)
- `--craft_dual_lr 0.01`：对偶变量学习率 η_λ
- `--craft_enable_projection True`：启用冲突感知梯度投影
- `--craft_enable_dual True`：启用自适应 λ 更新

---

### 📂 项目结构指南

```
VLA-Adapter/
├── prismatic/                          # 核心模型代码
│   ├── models/                         # VLA 模型架构
│   │   └── backbones/                  # 视觉和语言骨干网络
│   └── training/
│       ├── craft_utils.py              # ⭐ CRaFT 核心实现 (在线权重切换、梯度投影、对偶优化)
│       └── train_utils.py              # 原版训练工具
│
├── vla-scripts/
│   └── finetune.py                     # ⭐ 主训练脚本 (集成 CRaFT 训练循环)
    │
├── craft_experiments/                  # ⭐ CRaFT 实验脚本 (一键复现论文结果)
│   ├── 01_main_results/                # 主实验：Table 1 (LIBERO 四个 Suite)
│   │   └── run_table1_experiments.sh   # 自动化脚本：Baseline vs CRaFT
│   ├── 02_stability_efficiency/        # 极少样本实验 (5-shot, 10-shot)
│   │   └── run_fewshot_experiments.sh
│   └── 03_ablations/                   # 消融实验 (梯度投影、对偶优化、锚点类型)
│       └── run_ablation_experiments.sh
│
├── craft_experiments/common_utils/     # 实验辅助工具
│   └── log_parser.py                   # ⭐ 日志解析器 (提取 WandB 指标、生成 LaTeX 表格)
│
└── docs/                               # 📚 完整中文文档
    ├── DATASETS.md                     # 数据集准备指南 (LIBERO RLDS 格式)
    ├── EXPERIMENTS_AND_TRAINING.md     # 训练与评估完全指南
    ├── CHANGELOG.md                    # 文档变更记录
    └── craft/                          # CRaFT 项目文档
        ├── PROJECT_STRUCTURE.md        # 详细项目结构解析
        └── craft_progress.md           # 开发进度追踪
```

**核心文件说明**：

- **`prismatic/training/craft_utils.py`**：CRaFT 算法的完整实现，包含：
  - `CRaFTWeightManager`：在线权重切换管理器
  - `CRaFTFeatureExtractor`：桥接特征提取器
  - `CRaFTGradientProjector`：冲突感知梯度投影器（含冲突率统计）
  - `CRaFTDualOptimizer`：自适应对偶变量优化器

- **`vla-scripts/finetune.py`**：主训练脚本，集成了：
  - CRaFT 两阶段反向传播（动作损失 + 表征损失）
  - 梯度冲突率实时统计
  - 梯度范数监控（训练稳定性）
  - 学习率追踪
  - 优化器/调度器状态保存（断点续训支持）

- **`craft_experiments/`**：一键复现论文实验的自动化脚本

---

### 📊 监控训练指标

CRaFT 训练过程中，WandB 会记录以下关键指标：

#### 标准训练指标
- `VLA Train/Loss`：动作预测损失 (L1 Loss)
- `VLA Train/Curr Action L1 Loss`：当前动作的 L1 损失
- `VLA Train/Next Actions L1 Loss`：未来动作的 L1 损失

#### CRaFT 核心指标 (⭐ 论文关键证据)
- `CRaFT/Retention Loss`：表征保留损失 L_ret (衡量表征漂移程度)
- `CRaFT/Lambda`：拉格朗日乘子 λ 的动态变化
- **`CRaFT/Conflict Ratio`**：**梯度冲突率** (证明表征坍塌与梯度冲突的核心指标)

#### 工程健壮性指标 (Phase 7.5 新增)
- `VLA Train/Gradient Norm`：梯度范数 (监控训练稳定性)
- `VLA Train/Learning Rate`：学习率变化曲线

**梯度冲突率的物理意义**：
- 该指标统计了在所有参数中，有多少比例的参数出现了"动作梯度"与"表征梯度"的几何冲突 (内积 < 0)
- **高冲突率 (>30%)** 说明模型正在经历严重的表征坍塌
- CRaFT 的梯度投影机制能有效化解这些冲突，保持表征稳定性

---

### 🔬 一键复现论文实验

我们提供了完整的自动化实验脚本，可一键复现论文中的所有实验结果：

#### 实验 1：主实验 (Table 1)

 ```bash
cd craft_experiments/01_main_results
bash run_table1_experiments.sh
```

该脚本会自动运行：
- LIBERO 四个 Suite (Spatial, Object, Goal, Long) 的 Baseline 和 CRaFT 训练
- 每个实验使用 3 个随机种子
- 自动保存 WandB 日志和模型 Checkpoint

#### 实验 2：极少样本实验 (Few-Shot)

 ```bash
cd craft_experiments/02_stability_efficiency
bash run_fewshot_experiments.sh
```

测试 CRaFT 在 5-shot 和 10-shot 场景下的表现。

#### 实验 3：消融实验 (Ablation Study)

 ```bash
cd craft_experiments/03_ablations
bash run_ablation_experiments.sh
```

包含：
- 梯度投影的作用
- 对偶优化的作用
- 不同锚点特征类型的影响

---

### 📈 结果分析与可视化

训练完成后，使用我们提供的日志解析工具提取指标：

```bash
python craft_experiments/common_utils/log_parser.py \
    --wandb_project "vla-craft-training" \
    --output_dir "results/" \
    --generate_latex_table
```

该工具会：
1. 从 WandB 下载所有实验日志
2. 提取关键指标 (成功率、冲突率、表征漂移等)
3. 计算均值和标准差
4. 生成 LaTeX 格式的论文表格

---

### 🛠️ 断点续训

CRaFT 支持完整的断点续训功能（Phase 7.5 新增）：

```bash
# 从 Step 10000 的 Checkpoint 继续训练
python vla-scripts/finetune.py \
    --config_file_path "runs/your-experiment-name--10000_chkpt" \
    --resume True \
    --resume_step 10000 \
    --max_steps 20000 \
    --use_craft True
```

保存的 `training_state.pt` 包含：
- Optimizer 状态 (Adam 的动量和二阶矩估计)
- LR Scheduler 状态
- 当前训练步数

---

### 📚 完整文档导航

- **[数据集准备指南](docs/DATASETS.md)**：LIBERO 数据集下载、RLDS 格式说明、Few-Shot 数据截断机制
- **[训练与评估指南](docs/EXPERIMENTS_AND_TRAINING.md)**：详细的训练配置、指标解读、常见报错排查
- **[项目结构详解](docs/craft/PROJECT_STRUCTURE.md)**：代码库架构深度解析
- **[开发进度追踪](docs/craft/craft_progress.md)**：CRaFT 项目开发历史 (Phase 1-8)

---

### 🤝 贡献指南

我们欢迎社区贡献！如果您发现 Bug 或有改进建议，请：

1. 提交 Issue 描述问题
2. Fork 本仓库并创建新分支
3. 提交 Pull Request

---

### 📄 引用

如果本项目对您的研究有帮助，请引用我们的论文：

```bibtex
@article{craft2024,
  title={CRaFT: Overcoming Representation Collapse in Behavior Cloning},
  author={Your Name et al.},
  journal={arXiv preprint arXiv:XXXX.XXXXX},
  year={2024}
}
```

---

### 📧 联系方式

如有任何问题，欢迎通过以下方式联系我们：
- 提交 GitHub Issue
- 发送邮件至：your-email@example.com

---

## <a name="english"></a>🇬🇧 English Documentation

### 📖 Project Overview

This project implements **CRaFT (Constrained Representation and Fine-Tuning)** on top of the open-source **VLA-Adapter** (based on Prismatic VLM).

**Core Problem**: Standard VLA fine-tuning with low-dimensional action supervision leads to severe **Representation Collapse** — the model takes shortcuts to fit downstream tasks, destroying the general multimodal perception inherited from pre-trained VLMs.

**CRaFT Solution**: Formulates fine-tuning as a **constrained optimization problem with representation drift budget**, using **conflict-aware gradient projection** to resolve optimization conflicts for stable, high-generalization fine-tuning.

---

### 🚀 Quick Start

```bash
# Install dependencies
pip install -e .
pip install -r requirements.txt

# Run CRaFT training
python vla-scripts/finetune.py \
    --config_file_path "openvla/openvla-7b" \
    --dataset_name "libero_spatial_no_noops" \
    --use_craft True \
    --craft_retention_budget 0.1 \
    --craft_dual_lr 0.01
```

For detailed documentation, see:
- [Dataset Preparation](docs/DATASETS.md)
- [Training Guide](docs/EXPERIMENTS_AND_TRAINING.md)
- [Project Structure](docs/craft/PROJECT_STRUCTURE.md)

---

### 📊 Key Metrics

CRaFT introduces novel training metrics:
- **Conflict Ratio**: Percentage of parameters experiencing gradient conflicts (core evidence for representation collapse)
- **Retention Loss**: Measures representation drift from pre-trained features
- **Adaptive Lambda**: Lagrangian multiplier balancing action performance and representation preservation

---

### 🔬 Reproduce Paper Results

```bash
# Main experiments (Table 1)
cd craft_experiments/01_main_results
bash run_table1_experiments.sh

# Few-shot experiments
cd craft_experiments/02_stability_efficiency
bash run_fewshot_experiments.sh

# Ablation studies
cd craft_experiments/03_ablations
bash run_ablation_experiments.sh
```

---

### 📄 Citation

```bibtex
@article{craft2024,
  title={CRaFT: Overcoming Representation Collapse in Behavior Cloning},
  author={Your Name et al.},
  journal={arXiv preprint arXiv:XXXX.XXXXX},
  year={2024}
}
```

---

**License**: MIT | **Maintained by**: VLA-Adapter Team
