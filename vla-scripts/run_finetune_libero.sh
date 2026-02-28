#!/usr/bin/env bash
set -euo pipefail

# =========================
# Editable config section
# =========================
MODE=${MODE:-craft}                              # craft | baseline
DATA_NAME=${DATA_NAME:-libero_spatial_no_noops}

# GPU / torchrun
CUDA_VISIBLE_DEVICES=${CUDA_VISIBLE_DEVICES:-0}
NPROC_PER_NODE=${NPROC_PER_NODE:-1}
NNODES=${NNODES:-1}

# Paths
VLM_PATH=${VLM_PATH:-pretrained_models/prism-qwen25-extra-dinosiglip-224px-0_5b}
CONFIG_FILE_PATH=${CONFIG_FILE_PATH:-pretrained_models/configs}
DATA_ROOT_DIR=${DATA_ROOT_DIR:-data/libero}
RUN_ROOT_DIR=${RUN_ROOT_DIR:-outputs}

# Resume / checkpoint
RESUME=${RESUME:-True}                          # True | False
RESUME_STEP=${RESUME_STEP:-500}                   # e.g. 2500
RESUME_VLA_PATH=${RESUME_VLA_PATH:-/workspace/craft-adapter/outputs/configs+libero_spatial_no_noops+b4+lr-0.0002+lora-r64+dropout-0.0--image_aug--VLA-Adapter--craft--libero_spatial_no_noops--20260228_193036--500_chkpt}            # e.g. outputs/<run_id>--2500_chkpt

# Training hyperparameters
BATCH_SIZE=${BATCH_SIZE:-8}
GRAD_ACCUM_STEPS=${GRAD_ACCUM_STEPS:-8}
LEARNING_RATE=${LEARNING_RATE:-2e-4}
LORA_RANK=${LORA_RANK:-64}
MAX_STEPS=${MAX_STEPS:-5000}
NUM_STEPS_BEFORE_DECAY=${NUM_STEPS_BEFORE_DECAY:-1800}
SAVE_FREQ=${SAVE_FREQ:-1000}
SHUFFLE_BUFFER_SIZE=${SHUFFLE_BUFFER_SIZE:-2000}

# Logging
USE_WANDB=${USE_WANDB:-True}                    # True | False
WANDB_ENTITY=${WANDB_ENTITY:-qzy84511}
WANDB_PROJECT=${WANDB_PROJECT:-$DATA_NAME}
WANDB_LOG_FREQ=${WANDB_LOG_FREQ:-10}
CONSOLE_LOG_FREQ=${CONSOLE_LOG_FREQ:-10}

# CRaFT hyperparameters (only used when MODE=craft)
CRAFT_RETENTION_BUDGET=${CRAFT_RETENTION_BUDGET:-0.005}
CRAFT_DUAL_LR=${CRAFT_DUAL_LR:-0.01}
CRAFT_ENABLE_PROJECTION=${CRAFT_ENABLE_PROJECTION:-True}
CRAFT_ENABLE_DUAL=${CRAFT_ENABLE_DUAL:-True}
CRAFT_ANCHOR_TYPE=${CRAFT_ANCHOR_TYPE:-concat}
CRAFT_ANCHOR_LAYER_IDX=${CRAFT_ANCHOR_LAYER_IDX:-}  # empty=middle; int=指定 hidden_states 索引; 负数=从末尾倒数
CRAFT_CR_TOKEN_MODE=${CRAFT_CR_TOKEN_MODE:-vision_only}  # vision_only | vision_plus_prompt

# Run tag
CURRENT_TIME=${CURRENT_TIME:-$(date +%Y%m%d_%H%M%S)}
RUN_NOTE_PREFIX=${RUN_NOTE_PREFIX:-VLA-Adapter}

# Optional extra args passthrough
EXTRA_ARGS=${EXTRA_ARGS:-}

# =========================
# Script body
# =========================
SCRIPT_DIR="$(cd "$(dirname "${BASH_SOURCE[0]}")" && pwd)"
REPO_ROOT="$(cd "${SCRIPT_DIR}/.." && pwd)"
cd "$REPO_ROOT"

mkdir -p logs

if [[ "$MODE" != "craft" && "$MODE" != "baseline" ]]; then
  echo "[ERROR] MODE must be 'craft' or 'baseline', got: $MODE"
  exit 1
fi

RUN_NOTE="${RUN_NOTE_PREFIX}--${MODE}--${DATA_NAME}--${CURRENT_TIME}"
LOG_FILE="logs/${RUN_NOTE}.log"

COMMON_ARGS=(
  --vlm_path "$VLM_PATH"
  --config_file_path "$CONFIG_FILE_PATH"
  --data_root_dir "$DATA_ROOT_DIR"
  --dataset_name "$DATA_NAME"
  --run_root_dir "$RUN_ROOT_DIR"
  --use_film False
  --num_images_in_input 2
  --use_proprio True
  --use_lora True
  --use_fz False
  --use_minivlm True
  --image_aug True
  --num_steps_before_decay "$NUM_STEPS_BEFORE_DECAY"
  --max_steps "$MAX_STEPS"
  --save_freq "$SAVE_FREQ"
  --save_latest_checkpoint_only False
  --merge_lora_during_training False
  --batch_size "$BATCH_SIZE"
  --grad_accumulation_steps "$GRAD_ACCUM_STEPS"
  --learning_rate "$LEARNING_RATE"
  --lora_rank "$LORA_RANK"
  --shuffle_buffer_size "$SHUFFLE_BUFFER_SIZE"
  --use_pro_version True
  --use_wandb "$USE_WANDB"
  --wandb_entity "$WANDB_ENTITY"
  --wandb_project "$WANDB_PROJECT"
  --wandb_log_freq "$WANDB_LOG_FREQ"
  --console_log_freq "$CONSOLE_LOG_FREQ"
  --run_id_note "$RUN_NOTE"
)

if [[ "$RESUME" == "True" ]]; then
  if [[ -z "$RESUME_VLA_PATH" ]]; then
    echo "[ERROR] RESUME=True 时必须设置 RESUME_VLA_PATH (如 outputs/<run_id>--2500_chkpt)"
    exit 1
  fi

  if [[ ! -d "$RESUME_VLA_PATH" ]]; then
    echo "[ERROR] RESUME_VLA_PATH 不存在: $RESUME_VLA_PATH"
    exit 1
  fi

  EXPECTED_ACTION_CKPT="$RESUME_VLA_PATH/action_head--${RESUME_STEP}_checkpoint.pt"
  EXPECTED_PROPRIO_CKPT="$RESUME_VLA_PATH/proprio_projector--${RESUME_STEP}_checkpoint.pt"

  # 若 step 与目录不一致，优先从目录名自动解析（如: xxx--250_chkpt）
  if [[ ! -f "$EXPECTED_ACTION_CKPT" || ! -f "$EXPECTED_PROPRIO_CKPT" ]]; then
    DIR_BASENAME="$(basename "$RESUME_VLA_PATH")"
    if [[ "$DIR_BASENAME" =~ --([0-9]+)_chkpt$ ]]; then
      AUTO_STEP="${BASH_REMATCH[1]}"
      AUTO_ACTION_CKPT="$RESUME_VLA_PATH/action_head--${AUTO_STEP}_checkpoint.pt"
      AUTO_PROPRIO_CKPT="$RESUME_VLA_PATH/proprio_projector--${AUTO_STEP}_checkpoint.pt"
      if [[ -f "$AUTO_ACTION_CKPT" && -f "$AUTO_PROPRIO_CKPT" ]]; then
        echo "[WARN] RESUME_STEP=$RESUME_STEP 与目录不匹配，自动更正为 $AUTO_STEP"
        RESUME_STEP="$AUTO_STEP"
      fi
    fi
  fi

  # 最终校验
  if [[ ! -f "$RESUME_VLA_PATH/action_head--${RESUME_STEP}_checkpoint.pt" ]]; then
    echo "[ERROR] 缺少 checkpoint: $RESUME_VLA_PATH/action_head--${RESUME_STEP}_checkpoint.pt"
    echo "[INFO] 可用 action_head checkpoint:"
    ls "$RESUME_VLA_PATH"/action_head--*_checkpoint.pt 2>/dev/null || true
    exit 1
  fi

  if [[ ! -f "$RESUME_VLA_PATH/proprio_projector--${RESUME_STEP}_checkpoint.pt" ]]; then
    echo "[ERROR] 缺少 checkpoint: $RESUME_VLA_PATH/proprio_projector--${RESUME_STEP}_checkpoint.pt"
    echo "[INFO] 可用 proprio_projector checkpoint:"
    ls "$RESUME_VLA_PATH"/proprio_projector--*_checkpoint.pt 2>/dev/null || true
    exit 1
  fi

  COMMON_ARGS+=(
    --resume True
    --resume_step "$RESUME_STEP"
    --resum_vla_path "$RESUME_VLA_PATH"
    --config_file_path "$RESUME_VLA_PATH"
  )
fi

if [[ "$MODE" == "craft" ]]; then
  MODE_ARGS=(
    --use_craft True
    --craft_retention_budget "$CRAFT_RETENTION_BUDGET"
    --craft_dual_lr "$CRAFT_DUAL_LR"
    --craft_enable_projection "$CRAFT_ENABLE_PROJECTION"
    --craft_enable_dual "$CRAFT_ENABLE_DUAL"
    --craft_anchor_type "$CRAFT_ANCHOR_TYPE"
    --craft_cr_token_mode "$CRAFT_CR_TOKEN_MODE"
  )
  if [[ -n "$CRAFT_ANCHOR_LAYER_IDX" ]]; then
    MODE_ARGS+=(--craft_anchor_layer_idx "$CRAFT_ANCHOR_LAYER_IDX")
  fi
else
  MODE_ARGS=(
    --use_craft False
  )
fi

echo "[INFO] Repo: $REPO_ROOT"
echo "[INFO] Mode: $MODE"
echo "[INFO] Dataset: $DATA_NAME"
echo "[INFO] GPUs: $CUDA_VISIBLE_DEVICES | nproc_per_node=$NPROC_PER_NODE"
echo "[INFO] Log: $LOG_FILE"
echo "[INFO] Resume: $RESUME"
if [[ "$RESUME" == "True" ]]; then
  echo "[INFO] Resume step: $RESUME_STEP"
  echo "[INFO] Resume path: $RESUME_VLA_PATH"
fi

# shellcheck disable=SC2086
CUDA_VISIBLE_DEVICES="$CUDA_VISIBLE_DEVICES" torchrun \
  --standalone \
  --nnodes "$NNODES" \
  --nproc-per-node "$NPROC_PER_NODE" \
  vla-scripts/finetune.py \
  "${COMMON_ARGS[@]}" \
  "${MODE_ARGS[@]}" \
  $EXTRA_ARGS \
  2>&1 | tee "$LOG_FILE"
