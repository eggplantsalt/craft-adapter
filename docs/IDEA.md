我们要做的idea是：

CRaFT: 克服行为克隆中的表征坍塌 - 全局项目上下文

1. 项目背景与目标

我们正在现有的开源项目 VLA-Adapter (基于 Prismatic VLM) 的基础上，实现我们提出的 CRaFT (Constrained Representation and Fine-Tuning) 训练框架。
在标准的 VLA 模型微调中，仅使用低维的动作监督信号会导致严重的“表征坍塌（Representation Collapse）”——模型为了走捷径拟合下游动作，破坏了从预训练 VLM 中继承的通用多模态感知能力。
CRaFT 的目标是将下游微调显式表述为一个带有表征漂移预算的约束优化问题，通过梯度投影化解优化冲突，从而实现稳定、高泛化性的微调。

2. 核心数学与算法逻辑

CRaFT 的核心由以下几个数学模块组成，我们需要在代码中严格实现它们：

2.1 动作损失函数 (Action Loss)

沿用 VLA-Adapter 的原始设定，动作预测使用连续动作空间的 $L_1$ 回归损失：


$$\mathcal{L}_{act}(\theta) = \mathbb{E} \| \pi_\theta(o, \ell) - A_t \|_1$$

2.2 锚点特征提取 (Bridge Conditions)

我们需要在模型 Forward 过程中，提取 VLA-Adapter 传递给 Action Head 的显式桥接特征（Bridge Conditions）作为锚点特征（Anchor Features）：

Raw Latent 中间层 ($C_R^{(m)}$)：中间层承载的多模态原始特征。

ActionQuery 深层 ($C_{AQ}^{(M)}$)：深层动作查询标记的特征。
我们需要将它们提取、Pool（池化）并拼接，作为最终的特征表示 $f_\theta$：


$$f_\theta(o,\ell) = [\text{Pool}(C_R^{(m)}); \text{Pool}(C_{AQ}^{(M)})]$$

2.3 表征保留损失 (Representation Retention Loss)

计算当前网络特征 $f_\theta$ 与冻结的初始预训练网络快照（Snapshot）特征 $\tilde f$ 之间的均方误差 (MSE)：


$$\mathcal{L}_{ret}(\theta) = \mathbb{E} \| f_\theta(o,\ell) - \tilde f(o,\ell) \|_2^2$$

2.4 冲突感知梯度投影 (Conflict-Aware Gradient Projection)

在反向传播（Backward）计算出梯度后、优化器更新参数（Step）前，我们需要对动作梯度 $g_{act} = \nabla_\theta \mathcal{L}_{act}$ 和表征梯度 $g_{ret} = \nabla_\theta \mathcal{L}_{ret}$ 进行干预。
当 $\langle g_{act}, g_{ret} \rangle < 0$ 时（发生几何冲突），对动作梯度进行正交投影：


$$\tilde g_{act} = g_{act} - \frac{\langle g_{act}, g_{ret} \rangle}{\|g_{ret}\|_2^2 + \delta} g_{ret}$$


否则，$\tilde g_{act} = g_{act}$。

2.5 原对偶自适应更新 (Primal-Dual Adaptation)

引入对偶变量 $\lambda$（Lagrange Multiplier，初始值为 0），参数更新法则为：


$$\theta \leftarrow \theta - \eta (\tilde g_{act} + \lambda g_{ret})$$


在每次训练 Step 后，根据预算 $\varepsilon$ 动态更新标量 $\lambda$：


$$\lambda \leftarrow \max(0, \lambda + \eta_\lambda (\mathcal{L}_{ret}(\theta) - \varepsilon))$$

3. VLA-Adapter 代码库修改指南 (Implementation Strategy)

为了绝对不破坏原版 VLA-Adapter 的架构和预训练兼容性，必须采取**最小侵入式（Minimal Intrusion）**的修改策略：

特征提取 (Feature Hooking):

深入阅读 prismatic/models/vlas/ 或 prismatic/models/backbones/llm/ 相关的 Forward 逻辑。

建议利用 PyTorch 的 register_forward_hook 机制，或者在不破坏原有 loss 返回字典的前提下，优雅地把 $C_R$ 和 $C_{AQ}$ 暴露出来。

离线缓存机制 (Offline Caching):

编写一个独立的脚本（如 vla-scripts/build_craft_cache.py），加载预训练权重（requires_grad=False），运行下游数据集（无动作监督），将 $\tilde f$ 缓存为本地 Tensor (如 .pt 或 .safetensors 文件)。这能彻底避免训练时双模型前向传播的 OOM 问题。

梯度手术 (Gradient Surgery):

在主训练循环 vla-scripts/finetune.py 中，实现核心的投影逻辑。

注意 DDP/FSDP 兼容性：VLA-Adapter 通常使用 DDP 或 FSDP 进行分布式训练。在操作 .grad 之前，必须确保正确处理了跨 GPU 的梯度同步和分片问题（如果使用了 FSDP，获取全量梯度需要特定的 Context Manager）。

对偶状态管理 (Dual State Management):

维护 $\lambda$ 标量，并确保其随训练步数正确迭代，并记录到 Wandb 中。

4. 开发原则

模块化: CRaFT 的核心算法（如梯度投影函数、$\lambda$ 调度器、Hook 注册器）必须封装在独立的文件中，例如新建 prismatic/training/craft_utils.py。

可逆性: 主训练脚本 vla-scripts/finetune.py 中的所有 CRaFT 逻辑必须被 if args.use_craft: 包裹，确保传入 --use_craft False 时，模型能完全回退到原生 VLA-Adapter 的普通 SFT 行为。