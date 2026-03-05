#!/usr/bin/env bash
set -euo pipefail

# =============================================================================
# run_table1_experiments.sh (Refactored)
#
# 主实验评估入口（单脚本参数化）：
#   1) 可选：将本地 LoRA checkpoint 自动 merge 成可评估目录
#   2) 在一个或多个 LIBERO suites 上执行 evaluation
#   3) 输出每个 suite 的 success rate 汇总
#
# 典型用法：
#   MODEL_DIR=outputs/xxx--900_chkpt \
#   SUITES=libero_spatial,libero_object,libero_goal,libero_10 \
#   AUTO_MERGE=True \
#   bash craft_experiments/01_main_results/run_table1_experiments.sh
# =============================================================================

SCRIPT_DIR="$(cd "$(dirname "${BASH_SOURCE[0]}")" && pwd)"
PROJECT_ROOT="$(cd "${SCRIPT_DIR}/../.." && pwd)"

EVAL_SCRIPT="${PROJECT_ROOT}/experiments/robot/libero/run_libero_eval.py"
MERGE_SCRIPT="${PROJECT_ROOT}/vla-scripts/merge_lora_weights_and_save.py"

# -------------------- 必填参数 --------------------
MODEL_DIR=${MODEL_DIR:-}

# -------------------- 评估参数 --------------------
SUITES=${SUITES:-libero_spatial}
NUM_TRIALS_PER_TASK=${NUM_TRIALS_PER_TASK:-50}
NUM_IMAGES_IN_INPUT=${NUM_IMAGES_IN_INPUT:-2}
USE_PROPRIO=${USE_PROPRIO:-True}
USE_FILM=${USE_FILM:-False}
USE_PRO_VERSION=${USE_PRO_VERSION:-True}
SAVE_ROLLOUT_VIDEO=${SAVE_ROLLOUT_VIDEO:-False}
CENTER_CROP=${CENTER_CROP:-True}
SEED=${SEED:-7}

# -------------------- merge 参数 --------------------
AUTO_MERGE=${AUTO_MERGE:-False}
USE_MINIVLA=${USE_MINIVLA:-True}
VLM_PATH=${VLM_PATH:-pretrained_models/prism-qwen25-extra-dinosiglip-224px-0_5b}
BASE_CHECKPOINT=${BASE_CHECKPOINT:-}
MERGE_OUTPUT_DIR=${MERGE_OUTPUT_DIR:-}

# -------------------- 日志参数 --------------------
LOCAL_LOG_DIR=${LOCAL_LOG_DIR:-${PROJECT_ROOT}/craft_experiments/01_main_results/eval_logs}
RESULTS_LOG=${RESULTS_LOG:-${PROJECT_ROOT}/craft_experiments/01_main_results/table1_results.log}
RUN_ID_NOTE=${RUN_ID_NOTE:-craft-table1}

# -------------------- 环境参数 --------------------
CUDA_VISIBLE_DEVICES=${CUDA_VISIBLE_DEVICES:-0}

mkdir -p "${LOCAL_LOG_DIR}"
: > "${RESULTS_LOG}"

if [[ -z "${MODEL_DIR}" ]]; then
  echo "[ERROR] MODEL_DIR 不能为空。"
  echo "示例：MODEL_DIR=outputs/xxx--900_chkpt bash craft_experiments/01_main_results/run_table1_experiments.sh"
  exit 1
fi

if [[ ! -d "${MODEL_DIR}" ]]; then
  echo "[ERROR] MODEL_DIR 不存在: ${MODEL_DIR}"
  exit 1
fi

if [[ ! -f "${EVAL_SCRIPT}" ]]; then
  echo "[ERROR] 评估脚本不存在: ${EVAL_SCRIPT}"
  exit 1
fi

EVAL_MODEL_DIR="${MODEL_DIR}"

has_merged_weights() {
  local model_dir="$1"
  [[ -f "${model_dir}/model.safetensors" || -f "${model_dir}/pytorch_model.bin" ]]
}

echo "=========================================="
echo "🚀 CRaFT Table1 评估入口（重构版）"
echo "=========================================="
echo "[INFO] PROJECT_ROOT   : ${PROJECT_ROOT}"
echo "[INFO] MODEL_DIR      : ${MODEL_DIR}"
echo "[INFO] SUITES         : ${SUITES}"
echo "[INFO] AUTO_MERGE     : ${AUTO_MERGE}"
echo "[INFO] SAVE_ROLLOUT_VIDEO: ${SAVE_ROLLOUT_VIDEO}"
echo "[INFO] MERGE_OUTPUT_DIR: ${MERGE_OUTPUT_DIR:-<same-as-model-dir>}"
echo "[INFO] RESULTS_LOG    : ${RESULTS_LOG}"
echo "[INFO] LOCAL_LOG_DIR  : ${LOCAL_LOG_DIR}"
echo ""

# Optional merge
if [[ "${AUTO_MERGE}" == "True" || "${AUTO_MERGE}" == "true" ]]; then
  MERGE_TARGET_DIR="${MODEL_DIR}"
  if [[ -n "${MERGE_OUTPUT_DIR}" ]]; then
    mkdir -p "${MERGE_OUTPUT_DIR}"
    MERGE_TARGET_DIR="${MERGE_OUTPUT_DIR}/$(basename "${MODEL_DIR}")--merged"
    if [[ ! -d "${MERGE_TARGET_DIR}" ]]; then
      echo "[INFO] 复制源 checkpoint 到 merge 目录: ${MERGE_TARGET_DIR}"
      cp -a "${MODEL_DIR}" "${MERGE_TARGET_DIR}"
    else
      echo "[INFO] merge 目录已存在，将复用: ${MERGE_TARGET_DIR}"
    fi
  fi

  if has_merged_weights "${MERGE_TARGET_DIR}"; then
    echo "[INFO] 检测到已存在 merged 权重（model.safetensors / pytorch_model.bin），跳过 merge。"
  elif [[ -d "${MERGE_TARGET_DIR}/lora_adapter" ]]; then
    echo "[INFO] 检测到 LoRA adapter，开始 merge 以用于 evaluation..."

    if [[ ! -f "${MERGE_SCRIPT}" ]]; then
      echo "[ERROR] merge 脚本不存在: ${MERGE_SCRIPT}"
      exit 1
    fi

    if [[ "${USE_MINIVLA}" == "True" || "${USE_MINIVLA}" == "true" ]]; then
      CUDA_VISIBLE_DEVICES="${CUDA_VISIBLE_DEVICES}" python "${MERGE_SCRIPT}" \
        --use_minivla True \
        --vlm_path "${VLM_PATH}" \
        --lora_finetuned_checkpoint_dir "${MERGE_TARGET_DIR}"
    else
      if [[ -z "${BASE_CHECKPOINT}" ]]; then
        echo "[ERROR] USE_MINIVLA=False 时必须显式设置 BASE_CHECKPOINT（完整基础模型路径）。"
        exit 1
      fi
      CUDA_VISIBLE_DEVICES="${CUDA_VISIBLE_DEVICES}" python "${MERGE_SCRIPT}" \
        --base_checkpoint "${BASE_CHECKPOINT}" \
        --lora_finetuned_checkpoint_dir "${MERGE_TARGET_DIR}"
    fi

    if ! has_merged_weights "${MERGE_TARGET_DIR}"; then
      echo "[ERROR] merge 执行后仍未找到 merged 权重，请检查：${MERGE_TARGET_DIR}"
      exit 1
    fi

    echo "[INFO] merge 完成。"
  else
    echo "[WARN] 目标目录中既没有 merged 权重，也没有 lora_adapter：${MERGE_TARGET_DIR}"
    echo "[WARN] 将继续评估，但大概率会失败。"
  fi

  EVAL_MODEL_DIR="${MERGE_TARGET_DIR}"
fi

IFS=',' read -r -a SUITE_ARRAY <<< "${SUITES}"

echo "⏰ 开始时间: $(date)" | tee -a "${RESULTS_LOG}"
echo "model_dir: ${MODEL_DIR}" | tee -a "${RESULTS_LOG}"
echo "eval_model_dir: ${EVAL_MODEL_DIR}" | tee -a "${RESULTS_LOG}"
echo "" | tee -a "${RESULTS_LOG}"

for suite in "${SUITE_ARRAY[@]}"; do
  suite="$(echo "${suite}" | xargs)"
  if [[ -z "${suite}" ]]; then
    continue
  fi

  echo "=========================================="
  echo "📦 Suite: ${suite}"
  echo "=========================================="

  timestamp="$(date +%Y%m%d_%H%M%S)"
  eval_log_file="${LOCAL_LOG_DIR}/eval_${suite}_${timestamp}.txt"

  CUDA_VISIBLE_DEVICES="${CUDA_VISIBLE_DEVICES}" python "${EVAL_SCRIPT}" \
    --pretrained_checkpoint "${EVAL_MODEL_DIR}" \
    --task_suite_name "${suite}" \
    --num_trials_per_task "${NUM_TRIALS_PER_TASK}" \
    --use_l1_regression True \
    --use_proprio "${USE_PROPRIO}" \
    --num_images_in_input "${NUM_IMAGES_IN_INPUT}" \
    --use_film "${USE_FILM}" \
    --use_pro_version "${USE_PRO_VERSION}" \
    --save_rollout_video "${SAVE_ROLLOUT_VIDEO}" \
    --center_crop "${CENTER_CROP}" \
    --run_id_note "${RUN_ID_NOTE}" \
    --local_log_dir "${LOCAL_LOG_DIR}" \
    --seed "${SEED}" \
    2>&1 | tee "${eval_log_file}"

  eval_exit_code=$?
  if [[ ${eval_exit_code} -ne 0 ]]; then
    echo "${suite}: EVAL_FAILED" | tee -a "${RESULTS_LOG}"
    echo "" | tee -a "${RESULTS_LOG}"
    continue
  fi

  success_line="$(grep -E "Overall success rate:" "${eval_log_file}" | tail -1 || true)"
  success_rate="$(echo "${success_line}" | awk '{print $4}')"

  if [[ -z "${success_rate}" ]]; then
    echo "${suite}: PARSE_FAILED" | tee -a "${RESULTS_LOG}"
  else
    echo "${suite}: ${success_rate}" | tee -a "${RESULTS_LOG}"
  fi
  echo "" | tee -a "${RESULTS_LOG}"
done

echo "=========================================="
echo "✅ 评估完成"
echo "⏰ 结束时间: $(date)"
echo "📊 结果文件: ${RESULTS_LOG}"
echo "📝 评估日志: ${LOCAL_LOG_DIR}"
echo "=========================================="
