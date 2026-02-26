#!/bin/bash
#
# run_table1_experiments.sh
#
# 【CRaFT 论文 Table 1 主实验自动化脚本】
#
# 功能说明：
#   本脚本自动化执行 CRaFT 论文中 Table 1 的所有实验，包括：
#   1. 在 LIBERO 的 4 个任务套件上训练 CRaFT 模型
#   2. 自动评估每个训练好的 checkpoint
#   3. 提取成功率并生成结果表格
#
# 实验覆盖的任务套件：
#   - libero_spatial (空间推理任务)
#   - libero_object (物体操作任务)
#   - libero_goal (目标导向任务)
#   - libero_10 (长序列任务)
#
# 使用方法：
#   bash craft_experiments/01_main_results/run_table1_experiments.sh
#
# 注意事项：
#   - 确保已安装所有依赖（见 README.md）
#   - 确保 LIBERO 数据集已下载到 datasets/rlds/ 目录
#   - 训练过程可能需要数小时到数天（取决于 GPU 配置）
#   - 建议使用 tmux 或 screen 在后台运行
#
# 输出文件：
#   - craft_experiments/01_main_results/table1_results.log - 汇总结果
#   - craft_experiments/01_main_results/eval_logs/ - 详细评估日志
#   - craft_experiments/01_main_results/table1_formatted.md - 格式化表格
#

set -e  # 遇到错误立即退出

# ============================================================================
# 配置区域 (Configuration)
# ============================================================================

# -------------------- 路径配置 --------------------
PROJECT_ROOT="E:/VLA-Adapter"                                                    # 项目根目录
FINETUNE_SCRIPT="${PROJECT_ROOT}/vla-scripts/finetune.py"                       # 训练脚本路径
EVAL_SCRIPT="${PROJECT_ROOT}/experiments/robot/libero/run_libero_eval.py"      # 评估脚本路径
RESULTS_LOG="${PROJECT_ROOT}/craft_experiments/01_main_results/table1_results.log"  # 结果汇总日志
EVAL_LOGS_DIR="${PROJECT_ROOT}/craft_experiments/01_main_results/eval_logs"    # 评估日志目录

# -------------------- 模型配置 --------------------
PRETRAINED_CHECKPOINT="openvla/openvla-7b"                                      # 预训练模型路径（或 HuggingFace ID）
DATA_ROOT_DIR="${PROJECT_ROOT}/datasets/rlds"                                   # RLDS 数据集根目录
RUN_ROOT_DIR="${PROJECT_ROOT}/runs"                                             # 训练输出目录

# -------------------- 训练超参数 --------------------
BATCH_SIZE=8                                                                     # 每个 GPU 的 batch size
LEARNING_RATE=5e-4                                                               # 学习率
MAX_STEPS=20000                                                                  # 最大训练步数
SAVE_FREQ=5000                                                                   # Checkpoint 保存频率（每 N 步）
GRAD_ACCUMULATION_STEPS=1                                                        # 梯度累积步数

# -------------------- CRaFT 超参数 --------------------
USE_CRAFT=True                                                                   # 是否启用 CRaFT
CRAFT_RETENTION_BUDGET=0.1                                                       # 表征漂移预算 ε（论文中的关键参数）
CRAFT_DUAL_LR=0.01                                                               # 对偶变量学习率 η_λ
CRAFT_ENABLE_PROJECTION=True                                                     # 是否启用梯度投影

# -------------------- 评估配置 --------------------
NUM_TRIALS_PER_TASK=50                                                           # 每个任务的评估次数（LIBERO 标准：50 episodes）
NUM_IMAGES_IN_INPUT=2                                                            # 输入图像数量（1=第三人称视角，2=第三人称+腕部视角）

# -------------------- 任务套件列表 --------------------
# Table 1 实验涵盖 LIBERO 的 4 个任务套件
TASK_SUITES=("libero_spatial" "libero_object" "libero_goal" "libero_10")

# ============================================================================
# 初始化设置 (Setup)
# ============================================================================

echo "=========================================="
echo "🚀 CRaFT Table 1 主实验"
echo "=========================================="
echo ""
echo "📁 项目根目录: ${PROJECT_ROOT}"
echo "📊 结果将保存到: ${RESULTS_LOG}"
echo "📝 评估日志目录: ${EVAL_LOGS_DIR}"
echo ""

# 步骤 1: 创建必要的目录
mkdir -p "${EVAL_LOGS_DIR}"

# 步骤 2: 清空之前的结果（避免混淆）
> "${RESULTS_LOG}"

# 步骤 3: 记录实验开始时间
echo "⏰ 实验开始时间: $(date)" | tee -a "${RESULTS_LOG}"
echo "" | tee -a "${RESULTS_LOG}"

# ============================================================================
# 主实验循环 (Main Experiment Loop)
# ============================================================================

for TASK_SUITE in "${TASK_SUITES[@]}"; do
    echo "=========================================="
    echo "📦 任务套件: ${TASK_SUITE}"
    echo "=========================================="
    
    # 定义本次运行的唯一标识符
    RUN_ID="craft-${TASK_SUITE}-table1"
    RUN_DIR="${RUN_ROOT_DIR}/${RUN_ID}"
    
    echo "🏷️  Run ID: ${RUN_ID}"
    echo "📂 运行目录: ${RUN_DIR}"
    echo ""
    
    # ------------------------------------------------------------------------
    # 步骤 1: 训练 CRaFT 模型
    # ------------------------------------------------------------------------
    
    echo "⏰ [$(date)] 开始训练 ${TASK_SUITE}..."
    echo "📊 训练配置:"
    echo "   - Batch Size: ${BATCH_SIZE}"
    echo "   - Learning Rate: ${LEARNING_RATE}"
    echo "   - Max Steps: ${MAX_STEPS}"
    echo "   - CRaFT Retention Budget (ε): ${CRAFT_RETENTION_BUDGET}"
    echo "   - CRaFT Dual LR (η_λ): ${CRAFT_DUAL_LR}"
    echo ""
    
    # 执行训练命令
    # 注意：这里使用 Python 直接调用训练脚本，所有参数通过命令行传递
    python "${FINETUNE_SCRIPT}" \
        --config_file_path "${PRETRAINED_CHECKPOINT}" \
        --data_root_dir "${DATA_ROOT_DIR}" \
        --dataset_name "${TASK_SUITE}" \
        --run_root_dir "${RUN_ROOT_DIR}" \
        --run_id_override "${RUN_ID}" \
        --batch_size ${BATCH_SIZE} \
        --learning_rate ${LEARNING_RATE} \
        --max_steps ${MAX_STEPS} \
        --save_freq ${SAVE_FREQ} \
        --grad_accumulation_steps ${GRAD_ACCUMULATION_STEPS} \
        --use_l1_regression True \
        --use_lora True \
        --lora_rank 32 \
        --use_proprio True \
        --num_images_in_input ${NUM_IMAGES_IN_INPUT} \
        --image_aug True \
        --use_craft ${USE_CRAFT} \
        --craft_retention_budget ${CRAFT_RETENTION_BUDGET} \
        --craft_dual_lr ${CRAFT_DUAL_LR} \
        --craft_enable_projection ${CRAFT_ENABLE_PROJECTION} \
        --wandb_project "craft-table1" \
        --wandb_entity "your-entity"
    
    # 捕获训练脚本的退出码
    TRAIN_EXIT_CODE=$?
    
    # 检查训练是否成功
    if [ ${TRAIN_EXIT_CODE} -ne 0 ]; then
        echo "❌ [错误] ${TASK_SUITE} 训练失败，退出码: ${TRAIN_EXIT_CODE}"
        echo "${TASK_SUITE}: TRAINING_FAILED" | tee -a "${RESULTS_LOG}"
        continue  # 跳过当前任务，继续下一个
    fi
    
    echo "✅ [$(date)] ${TASK_SUITE} 训练完成"
    echo ""
    
    # ------------------------------------------------------------------------
    # 步骤 2: 查找最新的 checkpoint
    # ------------------------------------------------------------------------
    
    echo "🔍 [$(date)] 正在查找最新的 checkpoint..."
    
    # 使用 ls 命令查找所有 checkpoint 目录，按时间排序，取最新的
    LATEST_CHECKPOINT=$(ls -td "${RUN_ROOT_DIR}/${RUN_ID}"--*_chkpt 2>/dev/null | head -1)
    
    # 检查是否找到 checkpoint
    if [ -z "${LATEST_CHECKPOINT}" ]; then
        echo "❌ [错误] 未找到 ${TASK_SUITE} 的 checkpoint"
        echo "${TASK_SUITE}: NO_CHECKPOINT" | tee -a "${RESULTS_LOG}"
        continue  # 跳过当前任务，继续下一个
    fi
    
    echo "📦 最新 checkpoint: ${LATEST_CHECKPOINT}"
    echo ""
    
    # ------------------------------------------------------------------------
    # 步骤 3: 评估模型
    # ------------------------------------------------------------------------
    
    echo "🎯 [$(date)] 开始评估 ${TASK_SUITE}..."
    echo "📊 评估配置:"
    echo "   - 每个任务的试验次数: ${NUM_TRIALS_PER_TASK}"
    echo "   - 输入图像数量: ${NUM_IMAGES_IN_INPUT}"
    echo ""
    
    # 生成评估日志文件名（包含时间戳）
    EVAL_LOG_FILE="${EVAL_LOGS_DIR}/eval_${TASK_SUITE}_$(date +%Y%m%d_%H%M%S).txt"
    
    # 执行评估命令（输出同时保存到日志文件和终端）
    python "${EVAL_SCRIPT}" \
        --pretrained_checkpoint "${LATEST_CHECKPOINT}" \
        --task_suite_name "${TASK_SUITE}" \
        --num_trials_per_task ${NUM_TRIALS_PER_TASK} \
        --use_l1_regression True \
        --use_proprio True \
        --num_images_in_input ${NUM_IMAGES_IN_INPUT} \
        --center_crop True \
        --run_id_note "craft-table1" \
        --local_log_dir "${EVAL_LOGS_DIR}" \
        --seed 7 \
        2>&1 | tee "${EVAL_LOG_FILE}"
    
    # 捕获评估脚本的退出码
    EVAL_EXIT_CODE=$?
    
    # 检查评估是否成功
    if [ ${EVAL_EXIT_CODE} -ne 0 ]; then
        echo "❌ [错误] ${TASK_SUITE} 评估失败，退出码: ${EVAL_EXIT_CODE}"
        echo "${TASK_SUITE}: EVAL_FAILED" | tee -a "${RESULTS_LOG}"
        continue  # 跳过当前任务，继续下一个
    fi
    
    echo "✅ [$(date)] ${TASK_SUITE} 评估完成"
    echo ""
    
    # ------------------------------------------------------------------------
    # 步骤 4: 提取成功率
    # ------------------------------------------------------------------------
    
    echo "📈 [$(date)] 正在提取成功率..."
    
    # 使用日志解析器提取成功率
    SUCCESS_RATE=$(python "${PROJECT_ROOT}/craft_experiments/common_utils/log_parser.py" "${EVAL_LOG_FILE}" | grep "Success rate:" | awk '{print $3}')
    
    # 检查是否成功提取
    if [ -z "${SUCCESS_RATE}" ]; then
        echo "❌ [错误] 无法提取 ${TASK_SUITE} 的成功率"
        echo "${TASK_SUITE}: PARSE_FAILED" | tee -a "${RESULTS_LOG}"
        continue  # 跳过当前任务，继续下一个
    fi
    
    echo "🎉 成功率: ${SUCCESS_RATE}"
    echo ""
    
    # 记录结果到汇总日志
    echo "${TASK_SUITE}: ${SUCCESS_RATE}" | tee -a "${RESULTS_LOG}"
    echo "" | tee -a "${RESULTS_LOG}"
    
    echo "=========================================="
    echo ""
    
done

# ============================================================================
# 结果汇总 (Summary)
# ============================================================================

echo "=========================================="
echo "🎊 所有实验已完成！"
echo "⏰ 完成时间: $(date)"
echo "=========================================="
echo ""
echo "📊 结果汇总:"
cat "${RESULTS_LOG}"
echo ""
echo "💾 完整结果已保存到: ${RESULTS_LOG}"
echo "📝 评估日志已保存到: ${EVAL_LOGS_DIR}"
echo ""

# 生成格式化的 Markdown 表格
echo "📋 正在生成结果表格..."
python - <<EOF
import sys
sys.path.append("${PROJECT_ROOT}/craft_experiments/common_utils")
from log_parser import parse_all_results, format_results_table

# 解析所有结果
results = parse_all_results("${RESULTS_LOG}")
table = format_results_table(results)

print("\n=== Table 1: 主实验结果 (Main Results) ===\n")
print(table)

# 保存表格到文件
with open("${PROJECT_ROOT}/craft_experiments/01_main_results/table1_formatted.md", "w", encoding="utf-8") as f:
    f.write("# Table 1: 主实验结果 - CRaFT on LIBERO\n\n")
    f.write("本表格展示了 CRaFT 在 LIBERO 四个任务套件上的性能表现。\n\n")
    f.write(table)
    f.write("\n")
    f.write("## 实验配置\n\n")
    f.write(f"- Batch Size: ${BATCH_SIZE}\n")
    f.write(f"- Learning Rate: ${LEARNING_RATE}\n")
    f.write(f"- Max Steps: ${MAX_STEPS}\n")
    f.write(f"- CRaFT Retention Budget (ε): ${CRAFT_RETENTION_BUDGET}\n")
    f.write(f"- CRaFT Dual LR (η_λ): ${CRAFT_DUAL_LR}\n")
    f.write(f"- Gradient Projection: ${CRAFT_ENABLE_PROJECTION}\n")
    f.write("\n")

print("\n✅ 格式化表格已保存到: craft_experiments/01_main_results/table1_formatted.md")
EOF

echo ""
echo "🎉 实验流程全部完成！"
echo ""
echo "📌 下一步操作建议:"
echo "   1. 查看详细结果: cat ${RESULTS_LOG}"
echo "   2. 查看格式化表格: cat craft_experiments/01_main_results/table1_formatted.md"
echo "   3. 查看评估日志: ls ${EVAL_LOGS_DIR}"
echo "   4. 分析 WandB 日志: 访问 https://wandb.ai/your-entity/craft-table1"
echo ""

