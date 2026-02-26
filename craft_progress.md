# CRaFT Implementation Progress Tracker

## Project Overview
实现 CRaFT (Constrained Representation and Fine-Tuning) 算法到 VLA-Adapter 代码库中。

## Current Phase: Phase 2 - 特征提取与缓存机制实现

**Status**: ✅ COMPLETED

**Start Date**: 2026-02-26

---

## 下一步行动计划

### Phase 3: 梯度投影与对偶优化 (待执行)
1. 在 `finetune.py` 中添加 CRaFT 配置参数
2. 修改 `run_forward_pass()` 以支持双损失计算
3. 实现梯度投影逻辑（在 backward 后、optimizer.step 前）
4. 集成对偶变量 λ 的更新
5. 添加 CRaFT 相关指标的 WandB 日志

### Phase 4: 集成测试与调试 (待执行)
1. 端到端训练测试
2. 验证 DDP 兼容性
3. 性能优化与内存分析
4. 编写使用文档和训练脚本

---

## 文件清单

### 新增文件
- ✅ `prismatic/training/craft_utils.py` - CRaFT 核心工具模块
- ✅ `vla-scripts/build_craft_cache.py` - 离线特征缓存脚本
- ✅ `craft_progress.md` - 项目进度跟踪文档

### 修改文件
- ✅ `prismatic/extern/hf/modeling_prismatic.py` - 添加特征提取逻辑

### 待修改文件 (Phase 3)
- ⏳ `vla-scripts/finetune.py` - 集成 CRaFT 训练逻辑
- ⏳ `prismatic/training/craft_utils.py` - 完善梯度投影的批量处理

---

## Phase 1: 代码库深度调研与特征提取架构设计

### 调研目标
1. ✅ 理解 CRaFT 算法的核心逻辑和数学公式
2. ✅ 追踪 VLA 模型的 Forward 流程，定位桥接特征的计算位置
3. ✅ 分析训练循环结构和分布式训练配置
4. ✅ 提出特征提取的最优实现方案

### 关键发现

#### 1. 模型架构分析

**核心类层次结构**:
```
OpenVLAForActionPrediction (prismatic/extern/hf/modeling_prismatic.py)
  └─ PrismaticForConditionalGeneration
      ├─ vision_backbone: PrismaticVisionBackbone
      ├─ projector: PrismaticProjector  
      ├─ language_model: AutoModelForCausalLM (Qwen2.5-0.5B)
      └─ action_queries: nn.Embedding(NUM_TOKENS, llm_dim)
```

**Forward 流程** (`PrismaticForConditionalGeneration.forward()`):
1. Vision Backbone 提取视觉特征 → `patch_features` (B, num_patches, vision_dim)
2. Projector 投影到 LLM 空间 → `projected_patch_embeddings` (B, num_patches, llm_dim)
3. 构建多模态输入：`[BOS, vision_patches, text_tokens, action_queries, STOP]`
4. LLM Forward → `language_model_output.hidden_states` (所有层的隐藏状态)

#### 2. 桥接特征 (Bridge Conditions) 定位

根据代码分析，CRaFT 需要的两个桥接特征在以下位置：

**特征 1: Raw Latent $C_R^{(m)}$ - 中间层视觉-语言融合特征**
- **位置**: `language_model_output.hidden_states[m]` 的 **vision patch 部分**
- **形状**: `(B, num_patches, llm_dim)`
- **语义**: 中间层（如第 12 层）承载的多模态原始特征，包含视觉和任务语言的融合信息

**特征 2: ActionQuery Latent $C_{AQ}^{(M)}$ - 深层动作查询特征**
- **位置**: `language_model_output.hidden_states[-1]` 的 **action_queries 部分**
- **形状**: `(B, NUM_TOKENS, llm_dim)` 其中 `NUM_TOKENS = ACTION_DIM * NUM_ACTIONS_CHUNK`
- **语义**: 最后一层的动作查询 token 特征，直接用于动作预测

**当前代码中的特征提取逻辑** (在 `finetune.py` 的 `run_forward_pass()` 中):
```python
multi_layer_hidden_states = []
for item in output.hidden_states[0:]:
    text_hidden_states = item[:, num_patches:-1]
    actions_hidden_states = text_hidden_states[current_action_mask | next_actions_mask]
    task_latten_states = item[:, :num_patches]
    all_hidden_states = torch.cat((task_latten_states, actions_hidden_states), 2)
    multi_layer_hidden_states.append(all_hidden_states)
```

#### 3. 训练循环分析

**训练脚本**: `vla-scripts/finetune.py`

**分布式训练配置**:
- 使用 **DDP (DistributedDataParallel)** 而非 FSDP
- 通过 `accelerate.PartialState` 管理分布式状态
- 模型通过 `wrap_ddp()` 包装

**训练循环结构**:
```python
for batch_idx, batch in enumerate(dataloader):
    # 1. Forward Pass
    loss, metrics = run_forward_pass(vla, action_head, ...)
    
    # 2. Backward Pass
    normalized_loss = loss / grad_accumulation_steps
    normalized_loss.backward()
    
    # 3. Gradient Accumulation
    if (batch_idx + 1) % grad_accumulation_steps == 0:
        optimizer.step()
        scheduler.step()
        optimizer.zero_grad()
```

**关键观察**:
- Loss 计算在 `run_forward_pass()` 中完成
- 使用梯度累积 (gradient accumulation)
- 支持混合精度训练 (`torch.autocast`)

#### 4. Action Head 架构

**类**: `L1RegressionActionHead` (prismatic/models/action_heads.py)

**核心组件**:
- `MLPResNet`: 24 层 ResNet 块，带有 cross-attention 机制
- 输入: `multi_layer_hidden_states` (B, num_layers, num_patches + NUM_TOKENS, llm_dim)
- 输出: 连续动作 (B, NUM_ACTIONS_CHUNK, ACTION_DIM)

**特征使用**:
- 使用 **所有层** 的隐藏状态 (不仅仅是最后一层)
- 每个 ResNet 块接收对应层的 task 和 action 特征作为条件

---

## 特征提取方案设计

### 方案对比分析

#### 方案 A: PyTorch Forward Hook
**优点**:
- 非侵入式，不修改原始 forward 逻辑
- 易于开关 (通过 register/remove hook)

**缺点**:
- ❌ **在 DDP 环境下可能有同步问题**
- ❌ Hook 在 autocast 上下文外执行，可能导致精度不一致
- ❌ 需要额外的全局变量或闭包来存储特征
- ❌ 调试困难，错误信息不清晰

#### 方案 B: 修改 Forward 返回字典 (推荐)
**优点**:
- ✅ **与 DDP/混合精度训练完全兼容**
- ✅ 特征提取在同一计算图内，梯度流清晰
- ✅ 易于调试和维护
- ✅ 符合 HuggingFace 的设计模式 (返回 dataclass)

**缺点**:
- 需要修改 `PrismaticForConditionalGeneration.forward()` 的返回值
- 需要修改 `run_forward_pass()` 来接收额外的特征

**实现策略**:
1. 在 `PrismaticCausalLMOutputWithPast` 中添加字段:
   - `raw_latent_features: Optional[torch.FloatTensor]`
   - `action_query_features: Optional[torch.FloatTensor]`
2. 在 `forward()` 中提取并返回这些特征
3. 通过配置参数 `output_craft_features: bool` 控制是否提取

---

## 提议的实现路径

### 核心修改点

#### 1. 扩展输出数据结构
**文件**: `prismatic/extern/hf/modeling_prismatic.py`

在 `PrismaticCausalLMOutputWithPast` 中添加:
```python
@dataclass
class PrismaticCausalLMOutputWithPast(ModelOutput):
    # ... 现有字段 ...
    
    # CRaFT 特征
    raw_latent_features: Optional[torch.FloatTensor] = None      # C_R: 中间层特征
    action_query_features: Optional[torch.FloatTensor] = None    # C_AQ: 动作查询特征
```

#### 2. 修改 Forward 方法
**文件**: `prismatic/extern/hf/modeling_prismatic.py`

在 `PrismaticForConditionalGeneration.forward()` 中:
- 添加参数 `output_craft_features: bool = False`
- 当 `output_craft_features=True` 时，从 `language_model_output.hidden_states` 提取特征
- 返回扩展后的输出

#### 3. 创建 CRaFT 工具模块
**新文件**: `prismatic/training/craft_utils.py`

包含:
- `CRaFTFeatureExtractor`: 特征提取和池化
- `CRaFTGradientProjector`: 梯度投影逻辑
- `CRaFTDualOptimizer`: 对偶变量 λ 的管理和更新
- `CRaFTConfig`: CRaFT 超参数配置

#### 4. 修改训练循环
**文件**: `vla-scripts/finetune.py`

- 在 `FinetuneConfig` 中添加 CRaFT 相关参数
- 在 `run_forward_pass()` 中接收桥接特征
- 在主训练循环中实现梯度投影逻辑

---

## Phase 2: 特征提取与缓存机制实现

**Status**: ✅ COMPLETED

**Completion Date**: 2026-02-26

### 实施目标
1. ✅ 修改 `PrismaticCausalLMOutputWithPast` 数据结构，添加 CRaFT 特征字段
2. ✅ 在 `PrismaticForConditionalGeneration.forward()` 中实现特征提取逻辑
3. ✅ 创建 `craft_utils.py` 核心工具模块
4. ✅ 编写离线特征缓存脚本 `build_craft_cache.py`

### 代码修改清单

#### 1. 扩展输出数据结构
**文件**: `prismatic/extern/hf/modeling_prismatic.py`

**修改内容**:
- 在 `PrismaticCausalLMOutputWithPast` 中添加两个新字段：
  ```python
  raw_latent_features: Optional[torch.FloatTensor] = None      # C_R: 中间层特征
  action_query_features: Optional[torch.FloatTensor] = None    # C_AQ: 动作查询特征
  ```

#### 2. 实现特征提取逻辑
**文件**: `prismatic/extern/hf/modeling_prismatic.py`

**修改内容**:
- 在 `forward()` 方法签名中添加参数 `output_craft_features: Optional[bool] = None`
- 在方法开始处初始化特征占位符：
  ```python
  raw_latent_features = None
  action_query_features = None
  ```
- 在返回语句之前添加特征提取逻辑：
  - 从 `language_model_output.hidden_states[middle_layer]` 提取 vision patch 部分作为 $C_R$
  - 从 `language_model_output.hidden_states[-1]` 提取 action query 部分作为 $C_{AQ}$
  - 自动计算中间层索引 (`num_layers // 2`)
  - 自动计算 action query 的位置索引
- 在返回的 `PrismaticCausalLMOutputWithPast` 中包含提取的特征

**特征提取位置计算**:
```python
# 序列结构: [BOS, vision_patches, prompt_tokens, action_queries, STOP]
num_patches = projected_patch_embeddings.shape[1]
prompt_length = input_ids.shape[1] - 1
action_start_idx = 1 + num_patches + prompt_length
action_end_idx = action_start_idx + num_action_tokens
```

#### 3. 创建 CRaFT 工具模块
**新文件**: `prismatic/training/craft_utils.py`

**实现的类和函数**:

1. **`CRaFTConfig`** (dataclass)
   - 配置参数：anchor_layer_idx, use_mean_pooling, retention_weight, retention_budget, dual_lr, projection_eps 等
   - 用于统一管理 CRaFT 的所有超参数

2. **`CRaFTFeatureExtractor`** (nn.Module)
   - `pool_features()`: 对特征进行 Mean/Max Pooling
   - `forward()`: 接收 $C_R$ 和 $C_{AQ}$，池化后拼接为 $f_\theta$
   - 输入: (B, seq_len, D) → 输出: (B, 2*D)

3. **`CRaFTGradientProjector`**
   - `project_gradients()`: 实现单个梯度的投影逻辑
   - 公式: $\tilde{g}_{act} = g_{act} - \frac{\langle g_{act}, g_{ret} \rangle}{\|g_{ret}\|^2 + \delta} g_{ret}$
   - 仅在梯度冲突时 (dot product < 0) 执行投影

4. **`CRaFTDualOptimizer`**
   - `step()`: 更新对偶变量 λ
   - 公式: $\lambda \leftarrow \max(0, \lambda + \eta_\lambda (\mathcal{L}_{ret} - \varepsilon))$
   - `get_lambda()`: 获取当前 λ 值
   - `reset()`: 重置 λ 到初始值

5. **辅助函数**:
   - `compute_retention_loss()`: 计算 MSE 损失
   - `load_cached_features()`: 加载缓存特征 (占位符，Phase 3 实现)

#### 4. 离线特征缓存脚本
**新文件**: `vla-scripts/build_craft_cache.py`

**功能**:
- 加载冻结的预训练 VLA 模型
- 遍历整个下游数据集 (如 Libero)
- 提取桥接特征并通过 `CRaFTFeatureExtractor` 处理
- 分片保存到磁盘 (避免内存 OOM)

**配置参数** (`CacheBuildConfig`):
- `pretrained_checkpoint`: 预训练模型路径
- `data_root_dir`: RLDS 数据集根目录
- `dataset_name`: 数据集名称 (如 "libero_spatial")
- `batch_size`: 批次大小
- `output_dir`: 缓存输出目录
- `shard_size`: 每个分片的样本数 (默认 1000)

**输出格式**:
- 每个分片: `features_shard_XXXX.pt`，包含 `[{'sample_idx': int, 'features': Tensor}, ...]`
- 元数据: `metadata.pt`，包含数据集信息、特征维度、层索引等

**使用方法**:
```bash
python vla-scripts/build_craft_cache.py \
    --pretrained_checkpoint openvla/openvla-7b \
    --data_root_dir datasets/rlds \
    --dataset_name libero_spatial \
    --output_dir cache/craft_features \
    --batch_size 8 \
    --shard_size 1000
```

### 技术细节

#### 特征提取的精确性
- **中间层选择**: 自动选择 `len(hidden_states) // 2` 作为锚点层
- **Vision Patch 提取**: `hidden_states[m][:, 1:1+num_patches, :]` (跳过 BOS token)
- **Action Query 提取**: 通过 `_process_action_masks()` 精确定位 action token 位置

#### 防御性编程
- 使用 `torch.no_grad()` 确保缓存时不构建计算图
- 特征提取后立即移到 CPU，避免 GPU 内存累积
- 分片保存机制，避免单个文件过大导致 OOM
- 仅在 main process 执行文件 I/O，避免分布式冲突

#### 数据集兼容性
- 完全复用 `finetune.py` 的数据加载逻辑
- 支持 RLDS 格式数据集
- 支持 `RLDSBatchTransform` 和 `PaddedCollatorForActionPrediction`
- 禁用图像增强 (`image_aug=False`) 确保缓存一致性

### 验证与测试

#### 如何运行缓存脚本

**前提条件**:
1. 已下载预训练模型 (如 `openvla/openvla-7b`)
2. 已准备 RLDS 格式的下游数据集 (如 Libero)
3. 确保有足够的磁盘空间存储缓存

**运行命令**:
```bash
# 基本用法
python vla-scripts/build_craft_cache.py \
    --pretrained_checkpoint <path_to_checkpoint> \
    --data_root_dir <path_to_rlds_data> \
    --dataset_name <dataset_name> \
    --output_dir cache/craft_features

# 示例：为 Libero Spatial 数据集构建缓存
python vla-scripts/build_craft_cache.py \
    --pretrained_checkpoint openvla/openvla-7b \
    --data_root_dir datasets/rlds \
    --dataset_name libero_spatial \
    --output_dir cache/craft_features \
    --batch_size 8 \
    --shard_size 1000 \
    --log_freq 10
```

**预期输出**:
```
Building CRaFT feature cache for dataset: libero_spatial
Loading pretrained VLA model...
Model loaded successfully on device 0
Loading dataset...
Dataset loaded: 1250 batches
Extracting features...
Caching features: 100%|████████████| 1250/1250 [15:30<00:00, 1.35it/s]
Saved shard 0 with 1000 samples to cache/craft_features/libero_spatial/features_shard_0000.pt
Saved shard 1 with 1000 samples to cache/craft_features/libero_spatial/features_shard_0001.pt
...
Saved final shard 9 with 500 samples to cache/craft_features/libero_spatial/features_shard_0009.pt
Saved metadata to cache/craft_features/libero_spatial/metadata.pt

Cache building complete!
Total samples cached: 9500
Total shards: 10
Feature dimension: 1792
```

**验证缓存**:
```python
import torch

# 加载元数据
metadata = torch.load('cache/craft_features/libero_spatial/metadata.pt')
print(f"Dataset: {metadata['dataset_name']}")
print(f"Samples: {metadata['num_samples']}")
print(f"Feature dim: {metadata['feature_dim']}")

# 加载第一个分片
shard = torch.load('cache/craft_features/libero_spatial/features_shard_0000.pt')
print(f"Shard 0 contains {len(shard)} samples")
print(f"Sample 0 feature shape: {shard[0]['features'].shape}")
```

### 已知限制与注意事项

1. **内存管理**: 
   - 大数据集建议调小 `batch_size` 和 `shard_size`
   - 特征会先在 GPU 上计算，然后移到 CPU 存储

2. **分布式支持**:
   - 当前仅支持单 GPU 缓存构建
   - 多 GPU 环境下仅 main process 执行 I/O

3. **数据一致性**:
   - 缓存时禁用了图像增强，确保特征可复现
   - 使用相同的 `shuffle_buffer_size` 确保样本顺序一致

4. **特征维度**:
   - 对于 Qwen2.5-0.5B (llm_dim=896)，最终特征维度为 2*896=1792
   - 不同模型的特征维度会不同

---

## 下一步行动计划

### Phase 2: 特征提取与缓存机制实现
1. 修改 `PrismaticCausalLMOutputWithPast` 数据结构
2. 实现特征提取逻辑
3. 创建 `craft_utils.py` 基础框架
4. 实现离线特征缓存脚本

### Phase 3: 梯度投影与对偶优化
1. 实现冲突感知梯度投影
2. 实现对偶变量 λ 的更新逻辑
3. 集成到训练循环

### Phase 4: 集成测试与调试
1. 端到端测试
2. 验证 DDP 兼容性
3. 性能优化

---

## 已知风险与注意事项

1. **DDP 梯度同步**: 在操作 `.grad` 之前必须确保梯度已同步
2. **混合精度**: 所有 CRaFT 相关计算需要在正确的 dtype 下进行
3. **内存开销**: 需要存储额外的特征和梯度，可能需要梯度检查点
4. **超参数敏感性**: λ 的初始化和学习率需要仔细调优

---

## 参考资料

- VLA-Adapter 原始代码: `vla-scripts/finetune.py`
- 模型定义: `prismatic/extern/hf/modeling_prismatic.py`
- Action Head: `prismatic/models/action_heads.py`
- CRaFT 算法: `docs/IDEA.md`

