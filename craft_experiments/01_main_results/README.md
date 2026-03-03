# 01 Main Results（重构版）

本目录现在采用“轻量、参数化、可组合”的方式做主实验评估：

- 一个脚本完成多 suite 评估
- 可选自动 merge LoRA（用于你们训练产生的中间 checkpoint）
- 不再把训练 + 评估 + 结果格式化强耦合在一条超长流水线里

---

## 脚本

- `run_table1_experiments.sh`：主入口（评估用）
  - 输入一个模型目录
  - 通过 `SUITES` 选择一个或多个 LIBERO suites
  - 通过 `AUTO_MERGE=True/False` 控制是否先 merge LoRA

---

## 什么时候需要 merge LoRA

你们本地训练的 checkpoint（例如 `outputs/xxx--900_chkpt`）通常包含：

- `lora_adapter/`
- `action_head--xxx_checkpoint.pt`
- `proprio_projector--xxx_checkpoint.pt`

这类目录默认**不是完整 HF 权重目录**。若要直接用于 `run_libero_eval.py`，建议先 merge。  
主脚本已支持自动 merge：

- `AUTO_MERGE=True` 且目录中有 `lora_adapter/` 时，自动调用 `vla-scripts/merge_lora_weights_and_save.py`
- merge 后会在同目录生成 `model.safetensors`（或等价完整权重）
- 若设置 `MERGE_OUTPUT_DIR`，脚本会先复制到该目录再 merge，避免污染原始 `MODEL_DIR`

---

## 用法

### 1) 评估官方完整模型（通常无需 merge）

```bash
MODEL_DIR=outputs/LIBERO-Spatial-Pro \
SUITES=libero_spatial \
AUTO_MERGE=False \
bash craft_experiments/01_main_results/run_table1_experiments.sh
```

### 2) 评估你们自己的 LoRA checkpoint（自动 merge）

```bash
MODEL_DIR=/workspace/craft-adapter/outputs/configs+libero_spatial_no_noops+b32+lr-0.0002+lora-r64+dropout-0.0--image_aug--VLA-Adapter--craft--libero_spatial_no_noops--20260302_160719--6000_chkpt \
SUITES=libero_spatial \
AUTO_MERGE=True \
USE_MINIVLA=True \
VLM_PATH=pretrained_models/prism-qwen25-extra-dinosiglip-224px-0_5b \
MERGE_OUTPUT_DIR=/opt/data/private/models \
bash craft_experiments/01_main_results/run_table1_experiments.sh
```

---

## 关键参数

- `MODEL_DIR`（必填）: 待评估模型目录
- `SUITES`: 逗号分隔列表，默认 `libero_spatial`
- `AUTO_MERGE`: `True/False`，默认 `False`
- `MERGE_OUTPUT_DIR`: 可选。设置后 merge 在该目录进行，评估也使用该 merge 产物目录
- `USE_MINIVLA`: merge 时是否走 minivla 路径，默认 `True`
- `VLM_PATH`: `USE_MINIVLA=True` 时的基座路径
- `BASE_CHECKPOINT`: `USE_MINIVLA=False` 时必须显式设置（默认不再给 `openvla/openvla-7b`）
- `NUM_TRIALS_PER_TASK`: 每任务评估回合数，默认 `50`
- `NUM_IMAGES_IN_INPUT`: 默认 `2`
- `USE_PROPRIO`: 默认 `True`
- `USE_PRO_VERSION`: 默认 `True`

---

## 输出

- 汇总结果：`craft_experiments/01_main_results/table1_results.log`
- 详细日志：`craft_experiments/01_main_results/eval_logs/`

`table1_results.log` 示例：

```text
libero_spatial: 0.972
libero_object: 0.918
libero_goal: 0.884
libero_10: 0.801
```

---

## `AUTO_MERGE=False` 的含义

- `AUTO_MERGE=False` **不是**“自动按 LoRA 方式评估”
- 它表示脚本不执行 merge，直接把 `MODEL_DIR` 传给评估脚本
- 如果 `MODEL_DIR` 已是完整模型目录（如含 `model.safetensors`），可直接评估
- 如果 `MODEL_DIR` 只有 `lora_adapter/` 等中间产物，通常需要先 merge 再评估
