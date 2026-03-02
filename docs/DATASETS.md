# 数据集准备指南 (Dataset Preparation Guide)

本文档详细说明如何准备和配置 VLA-Adapter + CRaFT 训练所需的数据集。

---

## 📋 目录

1. [LIBERO 数据集概述](#libero-overview)
2. [数据集下载与安装](#download-install)
3. [RLDS 格式说明](#rlds-format)
4. [数据集存放路径规则](#path-rules)
5. [Few-Shot 数据截断机制](#few-shot)
6. [数据集统计信息](#dataset-stats)
7. [常见问题排查](#troubleshooting)

---

## <a name="libero-overview"></a>1. LIBERO 数据集概述

**LIBERO (Lifelong Benchmark for Robot Learning)** 是一个专为机器人操作任务设计的基准数据集，包含多个难度递增的任务套件 (Suite)。

### 1.1 LIBERO 四大 Suite

| Suite 名称 | 任务数量 | 难度等级 | 描述 |
|-----------|---------|---------|------|
| **LIBERO-Spatial** | 10 | ⭐ 简单 | 空间推理任务 (如"把物体放到左边") |
| **LIBERO-Object** | 10 | ⭐⭐ 中等 | 物体识别任务 (如"拿起红色方块") |
| **LIBERO-Goal** | 10 | ⭐⭐⭐ 困难 | 目标导向任务 (如"打开抽屉并放入物体") |
| **LIBERO-Long** | 10 | ⭐⭐⭐⭐ 极难 | 长时序任务 (需要多步推理) |

### 1.2 数据集规模

- **每个任务**：50 条专家演示轨迹 (Expert Demonstrations)
- **每条轨迹**：平均 100-300 个时间步 (Timesteps)
- **观测空间**：
  - 第三人称相机图像 (Third-Person RGB): 128×128 或 224×224
  - 可选：腕部相机图像 (Wrist RGB)
  - 可选：机器人本体感知状态 (Proprioception): 7-DoF 关节角度 + 夹爪状态
- **动作空间**：7-DoF 连续动作 (6-DoF 末端执行器位姿 + 1-DoF 夹爪开合)

---

## <a name="download-install"></a>2. 数据集下载与安装

### 2.1 安装 LIBERO 库

```bash
# 克隆 LIBERO 仓库 (已作为 Submodule 集成到本项目)
cd VLA-Adapter/LIBERO
pip install -e .
```

### 2.2 下载预处理的 RLDS 格式数据集

VLA-Adapter 使用 **RLDS (Reinforcement Learning Datasets)** 格式存储数据，这是 Google 开源的标准化机器人数据集格式。

```bash
# 创建数据集根目录
mkdir -p datasets/rlds

# 下载 LIBERO-Spatial (示例)
cd datasets/rlds
wget https://example.com/libero_spatial_no_noops.tar.gz
tar -xzvf libero_spatial_no_noops.tar.gz

# 下载其他 Suite (根据需要)
wget https://example.com/libero_object_no_noops.tar.gz
wget https://example.com/libero_goal_no_noops.tar.gz
wget https://example.com/libero_long_no_noops.tar.gz
```

**注意**：实际下载链接请参考 LIBERO 官方文档或联系数据集维护者。

### 2.3 从原始 LIBERO 数据转换为 RLDS 格式

如果您需要从原始 LIBERO HDF5 文件转换为 RLDS 格式：

```bash
# 当前仓库未内置 LIBERO -> RLDS 的一键转换脚本
# 请先在外部完成转换，再将产物放到 --data_root_dir 对应目录下
```

---

## <a name="rlds-format"></a>3. RLDS 格式说明

### 3.1 RLDS 数据结构

RLDS 数据集以 **TFRecord** 格式存储，每个文件包含多条轨迹 (Episodes)。每条轨迹的结构如下：

```python
{
    'episode_metadata': {
        'file_path': str,           # 原始数据文件路径
        'episode_id': int,          # 轨迹 ID
    },
    'steps': [                      # 时间步序列
        {
            'observation': {
                'image': np.ndarray,        # RGB 图像 (H, W, 3)
                'wrist_image': np.ndarray,  # 腕部图像 (可选)
                'state': np.ndarray,        # 本体感知状态 (可选)
            },
            'action': np.ndarray,           # 动作 (7-DoF)
            'reward': float,                # 奖励 (通常为 0/1)
            'is_terminal': bool,            # 是否为终止状态
            'is_first': bool,               # 是否为初始状态
            'language_instruction': str,    # 任务描述 (如 "put the red block in the drawer")
        },
        ...
    ]
}
```

### 3.2 关键字段说明

- **`language_instruction`**：自然语言任务描述，是 VLA 模型的核心输入之一
  - 示例：`"put the red block in the drawer"`
  - CRaFT 使用该字段进行 **Few-Shot 数据截断** (见第 5 节)

- **`action`**：7 维连续动作向量
  - 前 6 维：末端执行器的 6-DoF 位姿变化 (Δx, Δy, Δz, Δroll, Δpitch, Δyaw)
  - 第 7 维：夹爪开合指令 (0=关闭, 1=打开)

- **`observation/image`**：第三人称相机图像
  - 原始分辨率：128×128 或 224×224
  - 训练时会进行数据增强 (如果启用 `--image_aug`)

---

## <a name="path-rules"></a>4. 数据集存放路径规则

### 4.1 标准目录结构

```
VLA-Adapter/
└── datasets/
    └── rlds/                                   # RLDS 数据集根目录
        ├── libero_spatial_no_noops/            # LIBERO-Spatial Suite
        │   ├── 0.1.0/                          # 数据集版本号
        │   │   ├── dataset_info.json           # 数据集元信息
        │   │   └── libero_spatial_no_noops-train.tfrecord-*  # TFRecord 文件
        │   └── features.json                   # 特征定义
        ├── libero_object_no_noops/             # LIBERO-Object Suite
        ├── libero_goal_no_noops/               # LIBERO-Goal Suite
        └── libero_long_no_noops/               # LIBERO-Long Suite
```

### 4.2 训练脚本中的路径配置

在 `vla-scripts/finetune.py` 中，通过以下参数指定数据集：

```bash
python vla-scripts/finetune.py \
    --data_root_dir "datasets/rlds" \           # RLDS 根目录
    --dataset_name "libero_spatial_no_noops"    # 具体数据集名称
```

代码会自动拼接完整路径：`datasets/rlds/libero_spatial_no_noops/`

说明：仓库内置脚本 `vla-scripts/run_finetune_libero.sh` 的默认值是 `--data_root_dir data/libero`，
只要目录下包含对应数据集子目录（如 `data/libero/libero_spatial_no_noops`）即可正常读取。

### 4.3 验证数据集路径

运行以下命令检查数据集是否正确加载：

```bash
python -c "
from prismatic.vla.datasets import RLDSDataset

dataset = RLDSDataset(
    data_root_dir='datasets/rlds',
    dataset_name='libero_spatial_no_noops',
    batch_transform=None,
    resize_resolution=(224, 224),
    shuffle_buffer_size=10000,
)
print(f'Dataset loaded successfully! Total episodes: {len(dataset)}')
"
```

### 4.4 结合梯度累积估算有效训练步数

训练时，参数更新次数由 `len(dataloader)` 和 `grad_accumulation_steps` 共同决定：

```text
可执行优化步 ≈ floor(len(dataloader) / grad_accumulation_steps)
实际训练步 = min(max_steps, 可执行优化步)
```

示例：若 `len(dataloader)=500` 且 `grad_accumulation_steps=8`，最多约 `62` 次参数更新。

---

## <a name="few-shot"></a>5. Few-Shot 数据截断机制

CRaFT 支持 **Few-Shot 学习实验**，通过 `--n_shot_episodes` 参数限制每个任务使用的轨迹数量。

### 5.1 工作原理

代码底层实现了基于 `language_instruction` 的**物理截断机制**：

1. **任务分组**：根据 `language_instruction` 将数据集分组
   - 例如：LIBERO-Spatial 有 10 个不同的任务描述
   - 每个任务对应 50 条轨迹

2. **截断逻辑**：对每个任务，只保留前 N 条轨迹
   - `--n_shot_episodes 10`：每个任务使用 10/50 = 20% 的数据
   - `--n_shot_episodes 5`：每个任务使用 5/50 = 10% 的数据

3. **实现位置**：`prismatic/vla/datasets/rlds/dataset.py` 与 `prismatic/vla/datasets/datasets.py` 中的数据加载逻辑

### 5.2 使用示例

```bash
# 10-Shot 训练 (每个任务 10 条轨迹)
python vla-scripts/finetune.py \
    --dataset_name "libero_spatial_no_noops" \
    --n_shot_episodes 10 \
    --use_craft True

# 5-Shot 训练 (每个任务 5 条轨迹)
python vla-scripts/finetune.py \
    --dataset_name "libero_spatial_no_noops" \
    --n_shot_episodes 5 \
    --use_craft True

# 完整数据训练 (默认，每个任务 50 条轨迹)
python vla-scripts/finetune.py \
    --dataset_name "libero_spatial_no_noops" \
    --use_craft True
```

### 5.3 Few-Shot 实验的意义

- **测试数据效率**：CRaFT 在极少样本下能否保持性能？
- **验证表征保留**：少量数据更容易导致过拟合和表征坍塌，CRaFT 的约束机制是否有效？
- **论文实验**：Few-Shot 结果是 CRaFT 论文的重要实验证据

---

## <a name="dataset-stats"></a>6. 数据集统计信息

### 6.1 自动保存统计信息

训练开始时，代码会自动计算并保存数据集的统计信息（用于动作归一化）：

```python
# 保存位置：runs/{experiment_name}/dataset_statistics.json
{
    "action": {
        "mean": [0.01, -0.02, 0.03, ...],  # 7 维动作均值
        "std": [0.15, 0.12, 0.18, ...],    # 7 维动作标准差
        "min": [-0.5, -0.4, -0.6, ...],    # 最小值
        "max": [0.5, 0.4, 0.6, ...]        # 最大值
    },
    "proprio": {                            # 如果使用本体感知
        "mean": [...],
        "std": [...]
    }
}
```

### 6.2 统计信息的作用

- **训练时**：将动作归一化到标准正态分布 N(0, 1)，加速收敛
- **推理时**：将模型预测的归一化动作反归一化为真实动作值

### 6.3 手动查看统计信息

```bash
# 查看已保存的统计信息
cat runs/your-experiment-name/dataset_statistics.json | python -m json.tool
```

---

## <a name="troubleshooting"></a>7. 常见问题排查

### 问题 1：`FileNotFoundError: Dataset not found`

**原因**：数据集路径配置错误或数据集未下载。

**解决方案**：
```bash
# 检查路径是否存在
ls datasets/rlds/libero_spatial_no_noops/

# 确认 TFRecord 文件存在
ls datasets/rlds/libero_spatial_no_noops/0.1.0/*.tfrecord*
```

---

### 问题 2：`OutOfMemoryError` (OOM) 在数据加载阶段

**原因**：`shuffle_buffer_size` 设置过大，TensorFlow 缓冲区占用过多内存。

**解决方案**：
```bash
# 减小 shuffle_buffer_size (默认 100000)
python vla-scripts/finetune.py \
    --shuffle_buffer_size 10000  # 从 100000 降低到 10000
```

**说明**：
- `shuffle_buffer_size` 控制 TensorFlow 数据加载器的随机打乱缓冲区大小
- 较小的值会减少内存占用，但可能降低数据随机性
- 推荐值：10000-50000（取决于可用内存）

---

### 问题 3：Few-Shot 模式下数据量不符合预期

**原因**：`n_shot_episodes` 参数未正确传递或数据集任务数量不匹配。

**解决方案**：
```bash
# 检查数据集实际加载的轨迹数量
python -c "
from prismatic.vla.datasets import RLDSDataset

dataset = RLDSDataset(
    data_root_dir='datasets/rlds',
    dataset_name='libero_spatial_no_noops',
    batch_transform=None,
    resize_resolution=(224, 224),
    shuffle_buffer_size=10000,
    n_shot_episodes=10,  # 设置 Few-Shot
)
print(f'Total episodes loaded: {len(dataset)}')
print(f'Expected: 10 tasks × 10 episodes = 100 episodes')
"
```

---

### 问题 4：图像分辨率不匹配

**原因**：数据集图像分辨率与模型配置不一致。

**解决方案**：
```bash
# 检查模型配置的图像尺寸
python -c "
from transformers import AutoConfig
config = AutoConfig.from_pretrained('openvla/openvla-7b', trust_remote_code=True)
print(f'Model expects image size: {config.image_sizes}')
"

# 确保数据集加载时使用正确的 resize_resolution
# 在 finetune.py 中，resize_resolution 会自动从模型配置读取
```

---

### 问题 5：数据增强导致训练不稳定

**原因**：过强的图像增强可能破坏视觉特征。

**解决方案**：
```bash
# 关闭图像增强（不推荐，会降低泛化性能）
python vla-scripts/finetune.py \
    --image_aug False

# 或者调整增强强度（需修改代码中的增强参数）
```

**说明**：
- 默认的图像增强包括：随机裁剪、颜色抖动、随机翻转
- 这些增强对提升泛化性能至关重要，建议保持启用

---

## 📚 相关文档

- **[训练与评估指南](EXPERIMENTS_AND_TRAINING.md)**：详细的训练配置和指标解读
- **[项目结构详解](craft/PROJECT_STRUCTURE.md)**：代码库架构深度解析
- **[主 README](../README.md)**：项目概览和快速开始

---

## 🤝 需要帮助？

如果遇到数据集相关问题：
1. 查看本文档的"常见问题排查"章节
2. 提交 GitHub Issue 并附上完整的错误日志
3. 联系数据集维护者获取最新的下载链接

---

**最后更新**：2024-02-27 | **维护者**：VLA-Adapter Team

