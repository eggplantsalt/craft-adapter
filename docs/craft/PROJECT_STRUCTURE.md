# VLA-Adapter + CRaFT 项目目录结构

**更新时间**: 2026-03-02  
**版本**: 与当前训练脚本行为对齐

---

## 📁 完整目录树

```
VLA-Adapter/
│
├── 📄 README.md                                    # 项目主文档（英文）
├── 📄 LICENSE                                      # 开源协议
├── 📄 pyproject.toml                               # Python 项目配置
├── 📄 our_envs.txt                                 # 环境依赖列表
│
├── 📁 docs/                                        # 📚 文档目录（当前有效）
│   ├── 📄 CHANGELOG.md                             # 文档变更日志
│   ├── 📄 DATASETS.md                              # 数据集准备与路径规则
│   ├── 📄 EXPERIMENTS_AND_TRAINING.md              # 训练与评估指南
│   ├── 🔒 CONTEXT.md                               # 项目上下文（受保护）
│   ├── 🔒 IDEA.md                                  # CRaFT 核心思想（受保护）
│   └── 📁 craft/                                   # CRaFT 开发与归档文档
│       ├── 📄 craft_progress.md                    # 开发进度记录
│       ├── 📄 FINAL_COMPLETION_SUMMARY.md          # 阶段总结
│       ├── 📄 PHASE_7.5_SUMMARY.md                 # Phase 7.5 完成报告
│       ├── 📄 PROJECT_COMPLETION_REPORT.md         # 项目完成报告
│       ├── 📄 PROJECT_STRUCTURE.md                 # 本文档
│       ├── 📄 STEP1_2_COMPLETION_REPORT.md         # Step1/2 完成报告
│       └── 📄 STEP1_2_FINAL_REPORT.md              # Step1/2 最终报告
│
├── 📁 prismatic/                                   # 🧠 核心模型代码
│   ├── 📁 training/                                # 训练相关模块
│   │   ├── 📄 craft_utils.py                       # ⭐ CRaFT 核心工具（517行中文注释）
│   │   ├── 📄 train_utils.py                       # 训练工具函数
│   │   └── 📄 metrics.py                           # 指标计算
│   ├── 📁 models/                                  # 模型架构
│   │   ├── 📁 backbones/                           # 骨干网络
│   │   │   ├── 📁 llm/                             # 语言模型
│   │   │   └── 📁 vision/                          # 视觉模型
│   │   ├── 📄 action_heads.py                      # 动作预测头
│   │   ├── 📄 projectors.py                        # 投影器
│   │   └── 📄 load.py                              # 模型加载
│   ├── 📁 vla/                                     # VLA 专用模块
│   │   ├── 📁 datasets/                            # 数据集加载
│   │   ├── 📄 action_tokenizer.py                  # 动作分词器
│   │   └── 📄 constants.py                         # 常量定义
│   └── 📁 extern/hf/                               # HuggingFace 集成
│       ├── 📄 modeling_prismatic.py                # Prismatic 模型定义
│       └── 📄 processing_prismatic.py              # 数据处理
│
├── 📁 vla-scripts/                                 # 🚀 VLA 训练/评估脚本
│   ├── 📄 finetune.py                              # ⭐ 主训练脚本（已集成 CRaFT）
│   ├── 📄 evaluate_calvin.py                       # CALVIN 评估
│   └── 📁 extern/                                  # 外部工具
│       └── 📄 convert_openvla_weights_to_hf.py     # 权重转换
│
├── 📁 craft_experiments/                           # 🧪 CRaFT 实验脚本
│   ├── 📁 01_main_results/                         # Table 1: 主实验
│   │   ├── 📄 run_table1_experiments.sh            # ✅ 自动化脚本（已添加中文注释）
│   │   └── 📄 README.md                            # 实验说明
│   ├── 📁 02_stability_efficiency/                 # Table 2: Few-shot 实验
│   │   ├── 📄 run_table2_fewshot.sh                # 自动化脚本
│   │   └── 📄 README.md                            # 实验说明
│   ├── 📁 03_ablations/                            # Table 4: 消融实验
│   │   ├── 📄 run_table4_ablations.sh              # 自动化脚本
│   │   └── 📄 README.md                            # 实验说明
│   └── 📁 common_utils/                            # 通用工具
│       └── 📄 log_parser.py                        # ✅ 日志解析器（已添加中文注释）
│
├── 📁 experiments/robot/                           # 🤖 机器人评估代码
│   ├── 📁 libero/                                  # LIBERO 评估
│   │   ├── 📄 run_libero_eval.py                   # LIBERO 评估脚本
│   │   └── 📄 libero_utils.py                      # LIBERO 工具函数
│   ├── 📄 openvla_utils.py                         # OpenVLA 工具
│   └── 📄 robot_utils.py                           # 机器人通用工具
│
├── 📁 pretrained_models/                           # 💾 预训练模型
│   ├── 📁 configs/                                 # 模型配置文件
│   │   ├── 📄 config.json
│   │   ├── 📄 modeling_prismatic.py
│   │   └── 📄 processing_prismatic.py
│   └── 📁 prism-qwen25-extra-dinosiglip-224px-0_5b/  # Prismatic VLM
│       └── 📄 README.md
│
├── 📁 data/                                        # 💿 数据集目录（用户自行下载）
│   └── 📁 libero/                                  # LIBERO 数据集
│       ├── 📁 libero_spatial_no_noops/
│       ├── 📁 libero_object_no_noops/
│       ├── 📁 libero_goal_no_noops/
│       └── 📁 libero_10_no_noops/
│
├── 📁 runs/                                        # 📊 训练输出目录
│   └── (训练生成的 checkpoint 和日志)
│
├── 📁 eval_logs/                                   # 📝 评估日志
│   └── (评估生成的日志文件)
│
└── 📁 figure/                                      # 🖼️ 图片资源
    ├── 📄 Framework.png                            # 框架图
    └── 📄 Teaser.png                               # 效果图
```

---

## 🎯 关键文件说明

### ⭐ CRaFT 核心代码（必读）

| 文件路径 | 说明 | 中文注释 |
|---------|------|---------|
| `prismatic/training/craft_utils.py` | CRaFT 核心算法实现 | ✅ 517行详细注释 |
| `vla-scripts/finetune.py` | 训练脚本（已集成 CRaFT） | ⚠️ 部分待补充 |

### 🧪 实验脚本（一键运行）

| 文件路径 | 说明 | 中文注释 |
|---------|------|---------|
| `craft_experiments/01_main_results/run_table1_experiments.sh` | Table 1 主实验 | ✅ 完整注释 |
| `craft_experiments/02_stability_efficiency/run_table2_fewshot.sh` | Table 2 Few-shot | 📝 待补充 |
| `craft_experiments/03_ablations/run_table4_ablations.sh` | Table 4 消融实验 | 📝 待补充 |

### 🛠️ 工具脚本

| 文件路径 | 说明 | 中文注释 |
|---------|------|---------|
| `craft_experiments/common_utils/log_parser.py` | 日志解析与结果提取 | ✅ 完整注释 |

### 📚 文档（中文）

| 文件路径 | 说明 | 状态 |
|---------|------|------|
| `docs/CHANGELOG.md` | 文档变更日志 | ✅ 已维护 |
| `docs/DATASETS.md` | 数据集与路径说明 | ✅ 已维护 |
| `docs/EXPERIMENTS_AND_TRAINING.md` | 训练与调试指南 | ✅ 已维护 |
| `docs/craft/craft_progress.md` | 开发进度记录 | ✅ 已归档 |

---

## 🔍 目录结构特点

### ✅ 优点

1. **物理隔离清晰**
   - CRaFT 核心代码在 `prismatic/training/craft_utils.py`
   - 实验脚本统一在 `craft_experiments/`
   - 原版 VLA-Adapter 代码保持不变

2. **文档组织合理**
   - 开发文档归档到 `docs/craft/`
   - 用户文档规划在 `docs/zh-CN/`
   - 根目录保持整洁

3. **实验脚本模块化**
   - 按论文表格分目录（Table 1/2/4）
   - 每个实验有独立的 README
   - 通用工具统一在 `common_utils/`

4. **易于维护**
   - 变更日志追踪所有修改
   - 中文注释降低理解门槛
   - 目录结构一目了然

---

## 📝 与原版 VLA-Adapter 的区别

### 新增目录/文件

```diff
+ docs/CHANGELOG.md                              # 文档变更日志
+ docs/craft/                                    # CRaFT 开发文档目录
+ docs/zh-CN/                                    # 中文文档目录
+ prismatic/training/craft_utils.py             # CRaFT 核心工具
+ craft_experiments/                             # CRaFT 实验脚本
+ vla-scripts/finetune.py (已修改)              # 集成 CRaFT 训练逻辑
```

### 保持不变的核心文件

```
✅ prismatic/models/                             # 模型架构（未修改）
✅ prismatic/vla/                                # VLA 模块（未修改）
✅ experiments/robot/                            # 评估代码（未修改）
✅ pretrained_models/                            # 预训练模型（未修改）
```

---

## 🚀 快速导航

### 我想...

- **了解 CRaFT 核心原理** → 阅读 `docs/IDEA.md`（受保护文档）
- **查看开发进度** → 阅读 `docs/craft/craft_progress.md`
- **运行 Table 1 实验** → 执行 `craft_experiments/01_main_results/run_table1_experiments.sh`
- **理解 CRaFT 代码** → 阅读 `prismatic/training/craft_utils.py`（517行中文注释）
- **解析评估日志** → 使用 `craft_experiments/common_utils/log_parser.py`
- **查看文档变更** → 阅读 `docs/CHANGELOG.md`

### 我是...

- **新手开发者** → 先读 `README.md` + `docs/EXPERIMENTS_AND_TRAINING.md`
- **论文复现者** → 查看 `craft_experiments/*/README.md` 和实验脚本
- **代码贡献者** → 阅读 `docs/CONTEXT.md` 和 `craft_utils.py`
- **文档维护者** → 遵循 `docs/CHANGELOG.md` 的记录规范

---

## 📌 重要提醒

1. **受保护文件**: `docs/CONTEXT.md` 和 `docs/IDEA.md` 绝对不可修改
2. **中文优先**: 所有新增文档和注释必须使用中文
3. **变更追踪**: 所有文档变更必须记录到 `docs/CHANGELOG.md`
4. **代码优先**: 文档必须基于实际代码撰写，不能臆测

---

**目录结构整理完成！准备进入 Step 3：撰写全套中文文档。**

