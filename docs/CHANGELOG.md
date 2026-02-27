# 文档变更日志 (Documentation Changelog)

本文件记录 VLA-Adapter + CRaFT 项目中所有文档的创建、修改、删除操作。

---

## 2025-02-27: Phase 7.5 完成 + 文档大扫除启动

### 文件移动
- ✅ `PHASE_7.5_SUMMARY.md` → `docs/craft/PHASE_7.5_SUMMARY.md`
  - **原因**: 将 CRaFT 开发相关文档统一归档到 `docs/craft/` 目录
- ✅ `craft_progress.md` → `docs/craft/craft_progress.md`
  - **原因**: 将 CRaFT 开发进度记录归档到专门目录

### 新建文件
- ✅ `docs/CHANGELOG.md` (本文件)
  - **用途**: 追踪所有文档变更操作
- ✅ `docs/craft/STEP1_2_COMPLETION_REPORT.md`
  - **用途**: Step 1 & 2 完成报告
- ✅ `docs/craft/PROJECT_STRUCTURE.md`
  - **用途**: 项目目录结构说明文档

### 代码注释增强
- ✅ `craft_experiments/common_utils/log_parser.py`
  - **改动**: 添加完整的中文文档字符串和注释
  - **内容**: 每个函数都有详细的参数说明、使用示例和工作原理
- ✅ `craft_experiments/01_main_results/run_table1_experiments.sh`
  - **改动**: 添加详细的中文注释
  - **内容**: 脚本头部说明、配置区域注释、每个步骤的中文说明

### 待创建文档 (Step 3)
以下文档将在 Step 3 中创建：
1. `docs/zh-CN/01_项目结构指南.md` - 项目目录结构说明
2. `docs/zh-CN/02_快速开始.md` - 环境配置与 Baseline 验证
3. `docs/zh-CN/03_数据集准备.md` - RLDS 格式与路径配置
4. `docs/zh-CN/04_训练指南.md` - Baseline 与 CRaFT 训练
5. `docs/zh-CN/05_评估指南.md` - 模型评测流程
6. `docs/zh-CN/06_架构深度解析.md` - CRaFT 核心原理
7. `docs/zh-CN/07_故障排查.md` - 常见问题与解决方案
8. `docs/zh-CN/08_API参考.md` - 代码接口文档

### 保护文件（不可修改）
- 🔒 `docs/CONTEXT.md` - 项目上下文（用户指定保护）
- 🔒 `docs/IDEA.md` - CRaFT 核心思想（用户指定保护）

---

## 变更记录格式说明

每次文档变更请按以下格式记录：

```markdown
## YYYY-MM-DD: 变更主题

### 新建文件
- ✅ `路径/文件名.md`
  - **用途**: 简要说明

### 修改文件
- 📝 `路径/文件名.md`
  - **改动**: 简要说明修改内容

### 删除文件
- ❌ `路径/文件名.md`
  - **原因**: 简要说明删除原因

### 移动文件
- ✅ `旧路径` → `新路径`
  - **原因**: 简要说明移动原因
```

---

## 文档维护原则

1. **中文优先**: 所有新增文档必须使用中文撰写
2. **傻瓜式**: 操作指南必须达到"复制粘贴即可跑通"的水平
3. **模块化**: 不同主题的文档分文件存放，避免单文件过长
4. **版本追踪**: 所有变更必须在本文件中记录
5. **保护机制**: `CONTEXT.md` 和 `IDEA.md` 绝对不可修改

