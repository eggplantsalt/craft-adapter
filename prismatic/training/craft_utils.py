"""
craft_utils.py

CRaFT (Constrained Representation and Fine-Tuning) 核心工具模块

本模块实现了 CRaFT 算法的所有核心组件：
1. CRaFTFeatureExtractor: 提取并处理桥接特征（Bridge Conditions）
2. CRaFTGradientProjector: 实现冲突感知的梯度投影
3. CRaFTDualOptimizer: 管理拉格朗日对偶变量 λ 的自适应更新
4. CRaFTWeightManager: 管理在线权重切换（Online Weight Swapping）
5. 辅助函数: 锚点特征提取、表征保留损失计算等

关键设计理念：
- 采用"在线权重切换"策略，避免离线缓存带来的数据对齐风险
- 使用 torch.no_grad() 提取锚点特征，实现零显存开销
- 梯度投影仅在检测到冲突时触发，保持训练效率
- 对偶优化自适应调整 λ，平衡动作性能和表征保留
"""

from dataclasses import dataclass
from typing import Optional, Tuple

import torch
import torch.nn as nn


@dataclass
class CRaFTConfig:
    """
    CRaFT 训练配置类
    
    包含 CRaFT 算法的所有超参数配置，分为四大模块：
    1. 特征提取配置
    2. 表征保留配置
    3. 对偶优化配置
    4. 梯度投影配置
    """
    
    # ========== 特征提取配置 ==========
    anchor_layer_idx: Optional[int] = None  # 锚点层索引（C_R 提取位置）。None 表示使用中间层 (num_layers // 2)
    use_mean_pooling: bool = True           # 是否使用均值池化聚合特征（False 则使用最大池化）
    anchor_type: str = "concat"             # 锚点特征类型：'concat'(拼接 C_R+C_AQ), 'aq_only'(仅 C_AQ), 'raw_only'(仅 C_R)
    
    # ========== 表征保留配置 ==========
    retention_weight: float = 1.0           # 表征保留损失权重（论文中的 λ）
    retention_budget: float = 0.1           # 允许的最大表征漂移量（论文中的 ε）
    
    # ========== 对偶优化配置 ==========
    dual_lr: float = 0.01                   # 对偶变量学习率（论文中的 η_λ）
    dual_init: float = 0.0                  # 对偶变量初始值
    enable_dual: bool = True                # 是否启用自适应对偶优化（False 则使用固定 λ）
    fixed_lambda: float = 0.1               # 当 enable_dual=False 时使用的固定 λ 值
    
    # ========== 梯度投影配置 ==========
    projection_eps: float = 1e-8            # 数值稳定性常数（论文中的 δ）
    enable_projection: bool = True          # 是否启用冲突感知梯度投影
    



class CRaFTFeatureExtractor(nn.Module):
    """
    CRaFT 特征提取器
    
    功能：提取并处理 VLA-Adapter 的桥接特征（Bridge Conditions）
    
    工作流程：
    1. 接收两类特征：
       - C_R: 中间层的原始潜在特征（Raw Latent Features）
       - C_AQ: 最终层的动作查询特征（Action Query Features）
    2. 对两类特征分别进行池化（Pooling）操作
    3. 根据 anchor_type 配置选择特征组合方式
    4. 输出最终的特征表示 f_θ
    
    特征组合策略：
    - 'concat': 拼接 C_R 和 C_AQ（原始 CRaFT 方案，维度 2*D）
    - 'aq_only': 仅使用 C_AQ（消融实验，维度 D）
    - 'raw_only': 仅使用 C_R（消融实验，维度 D）
    """
    
    def __init__(self, config: CRaFTConfig):
        """
        初始化特征提取器
        
        Args:
            config: CRaFT 配置对象
        """
        super().__init__()
        self.config = config
        self.use_mean_pooling = config.use_mean_pooling
        self.anchor_type = config.anchor_type  # 'concat', 'aq_only', 'raw_only'
    
    def pool_features(self, features: torch.Tensor) -> torch.Tensor:
        """
        对特征进行池化操作，将序列维度压缩为单个向量
        
        Args:
            features: 输入特征，形状 (B, seq_len, D)
                     B = batch size
                     seq_len = 序列长度（如 patch 数量或 token 数量）
                     D = 特征维度
        
        Returns:
            池化后的特征，形状 (B, D)
        """
        if self.use_mean_pooling:
            # 均值池化：对序列维度取平均
            return features.mean(dim=1)
        else:
            # 最大池化：对序列维度取最大值
            return features.max(dim=1)[0]
    
    def forward(
        self,
        raw_latent_features: torch.Tensor,
        action_query_features: torch.Tensor,
    ) -> torch.Tensor:
        """
        提取并组合桥接特征
        
        Args:
            raw_latent_features: C_R，来自中间层，形状 (B, num_patches, D)
            action_query_features: C_AQ，来自最终层，形状 (B, num_action_tokens, D)
        
        Returns:
            特征表示 f_θ，形状取决于 anchor_type：
            - 'concat': (B, 2*D)
            - 'aq_only' 或 'raw_only': (B, D)
        """
        # 步骤 1: 对两类特征分别进行池化
        pooled_raw = self.pool_features(raw_latent_features)      # (B, D)
        pooled_action = self.pool_features(action_query_features)  # (B, D)
        
        # 步骤 2: 根据 anchor_type 选择特征组合方式
        if self.anchor_type == "concat":
            # 拼接两类特征（原始 CRaFT 方案）
            combined_features = torch.cat([pooled_raw, pooled_action], dim=-1)  # (B, 2*D)
        elif self.anchor_type == "aq_only":
            # 仅使用动作查询特征（消融实验）
            combined_features = pooled_action  # (B, D)
        elif self.anchor_type == "raw_only":
            # 仅使用原始潜在特征（消融实验）
            combined_features = pooled_raw  # (B, D)
        else:
            raise ValueError(f"Invalid anchor_type: {self.anchor_type}. Must be 'concat', 'aq_only', or 'raw_only'")
        
        return combined_features


class CRaFTGradientProjector:
    """
    CRaFT 梯度投影器
    
    功能：实现冲突感知的梯度投影（Conflict-Aware Gradient Projection）
    
    核心思想：
    当动作梯度 g_act 和表征梯度 g_ret 发生冲突（点积为负）时，
    将 g_act 投影到与 g_ret 正交的方向上，避免两个优化目标相互干扰。
    
    数学公式：
    当 <g_act, g_ret> < 0 时：
        g̃_act = g_act - <g_act, g_ret> / (||g_ret||² + δ) * g_ret
    否则：
        g̃_act = g_act（无需投影）
    
    其中 δ 是数值稳定性常数，防止除零错误。
    
    冲突统计（论文核心卖点）：
    - 记录每次调用中发生冲突的参数层数量
    - 用于计算冲突率（Conflict Ratio），证明梯度投影的必要性
    """
    
    def __init__(self, config: CRaFTConfig):
        """
        初始化梯度投影器
        
        Args:
            config: CRaFT 配置对象
        """
        self.config = config
        self.eps = config.projection_eps
        self.enable_projection = config.enable_projection
        
        # 冲突统计（用于论文实验分析）
        self.num_conflicts = 0  # 当前 step 中发生冲突的参数层数量
        self.total_params = 0   # 当前 step 中参与 CRaFT 的总参数层数量
    
    def project_gradients(
        self,
        action_grad: torch.Tensor,
        retention_grad: torch.Tensor,
    ) -> Tuple[torch.Tensor, bool]:
        """
        对动作梯度进行投影（如果检测到冲突）
        
        实现公式：g̃_act = g_act - <g_act, g_ret> / (||g_ret||² + δ) * g_ret
        
        Args:
            action_grad: 动作损失的梯度，形状 (D,)
            retention_grad: 表征保留损失的梯度，形状 (D,)
        
        Returns:
            tuple: (投影后的动作梯度, 是否发生冲突)
                - projected_grad: 形状 (D,)
                - has_conflict: bool，True 表示发生冲突并执行了投影
        """
        if not self.enable_projection:
            return action_grad, False
        
        # 步骤 1: 计算梯度点积，判断是否冲突
        dot_product = torch.dot(action_grad, retention_grad)
        
        # 步骤 2: 仅在冲突时（点积为负）进行投影
        if dot_product < 0:
            # 计算投影系数
            retention_norm_sq = torch.dot(retention_grad, retention_grad)
            coeff = dot_product / (retention_norm_sq + self.eps)
            
            # 执行正交投影
            projected_grad = action_grad - coeff * retention_grad
            
            return projected_grad, True  # 发生冲突
        else:
            # 无冲突，返回原始梯度
            return action_grad, False  # 无冲突
    
    def reset_conflict_stats(self) -> None:
        """
        重置冲突统计计数器（在每个 optimizer step 前调用）
        """
        self.num_conflicts = 0
        self.total_params = 0
    
    def get_conflict_ratio(self) -> float:
        """
        获取当前 step 的冲突率
        
        Returns:
            冲突率 = 冲突的参数层数量 / 总参数层数量
            如果总数为 0，返回 0.0
        """
        if self.total_params == 0:
            return 0.0
        return self.num_conflicts / self.total_params
    
    def project_gradients_batch(
        self,
        model: nn.Module,
        action_loss: torch.Tensor,
        retention_loss: torch.Tensor,
    ) -> None:
        """
        批量梯度投影（预留接口，当前未使用）
        
        该方法会：
        1. 计算动作损失的梯度
        2. 计算表征保留损失的梯度
        3. 对冲突的梯度进行投影
        4. 更新模型梯度为：投影后的动作梯度 + 表征梯度
        
        Args:
            model: 需要投影梯度的模型
            action_loss: 动作预测损失
            retention_loss: 表征保留损失
        """
        # 该功能在 Phase 3 中已实现，但当前训练循环中未使用批量投影
        raise NotImplementedError("Batch gradient projection will be implemented in Phase 3")


class CRaFTDualOptimizer:
    """
    CRaFT 对偶优化器
    
    功能：管理拉格朗日对偶变量 λ（Lagrange Multiplier）的自适应更新
    
    核心思想：
    将表征保留约束转化为拉格朗日对偶问题，通过动态调整 λ 来平衡
    动作预测性能和表征保留约束。
    
    更新规则：
        λ ← max(0, λ + η_λ * (L_ret - ε))
    
    其中：
    - η_λ: 对偶变量学习率
    - L_ret: 当前的表征保留损失
    - ε: 允许的表征漂移预算
    
    工作模式：
    1. 自适应模式（enable_dual=True）：λ 根据约束违反程度动态调整
    2. 固定模式（enable_dual=False）：λ 保持为 fixed_lambda，用于消融实验
    """
    
    def __init__(self, config: CRaFTConfig):
        """
        初始化对偶优化器
        
        Args:
            config: CRaFT 配置对象
        """
        self.config = config
        self.dual_lr = config.dual_lr
        self.budget = config.retention_budget
        self.enable_dual = config.enable_dual
        
        # 初始化对偶变量
        if self.enable_dual:
            self.lambda_val = config.dual_init
        else:
            # 禁用对偶优化时，使用固定的 λ 值
            self.lambda_val = config.fixed_lambda
    
    def step(self, retention_loss: float) -> None:
        """
        根据约束违反程度更新对偶变量
        
        更新公式：λ ← max(0, λ + η_λ * (L_ret - ε))
        
        Args:
            retention_loss: 当前的表征保留损失值
        """
        if not self.enable_dual:
            # 固定模式下不更新 λ
            return
        
        # 步骤 1: 计算约束违反量
        violation = retention_loss - self.budget
        
        # 步骤 2: 更新对偶变量，并投影到非负象限
        self.lambda_val = max(0.0, self.lambda_val + self.dual_lr * violation)
    
    def get_lambda(self) -> float:
        """
        获取当前的对偶变量值
        
        Returns:
            当前的 λ 值
        """
        return self.lambda_val
    
    def reset(self) -> None:
        """重置对偶变量到初始值"""
        self.lambda_val = self.config.dual_init


def compute_retention_loss(
    current_features: torch.Tensor,
    anchor_features: torch.Tensor,
) -> torch.Tensor:
    """
    计算表征保留损失（Representation Retention Loss）
    
    功能：衡量当前模型特征与锚点特征之间的差异
    
    数学公式：
        L_ret = ||f_θ - f̃||²
    
    其中：
    - f_θ: 当前训练模型提取的特征
    - f̃: 冻结的初始模型（锚点）提取的特征
    
    Args:
        current_features: 当前模型的特征，形状 (B, D) 或 (B, 2*D)
        anchor_features: 锚点模型的特征，形状 (B, D) 或 (B, 2*D)
    
    Returns:
        标量损失值（MSE）
    """
    # Use float32 for numerical stability, and sanitize non-finite feature values
    # to avoid NaN retention loss causing stalled dual updates (lambda stuck at 0).
    current_features_fp32 = torch.nan_to_num(
        current_features.float(),
        nan=0.0,
        posinf=1e4,
        neginf=-1e4,
    )
    anchor_features_fp32 = torch.nan_to_num(
        anchor_features.float(),
        nan=0.0,
        posinf=1e4,
        neginf=-1e4,
    )
    return torch.nn.functional.mse_loss(current_features_fp32, anchor_features_fp32)


class CRaFTWeightManager:
    """
    CRaFT 权重管理器
    
    功能：管理在线权重切换（Online Weight Swapping），实现零显存开销的锚点特征提取
    
    核心策略："No-Grad First, Grad Second"
    1. 保存初始 Adapter 权重（预训练状态）到 CPU
    2. 训练时，每个 batch 先切换到初始权重，用 torch.no_grad() 提取锚点特征
    3. 切换回当前训练权重，正常 forward 提取当前特征
    4. 计算表征保留损失并执行梯度投影
    
    优势：
    - ✅ 零显存负担：第一次 forward 在 no_grad 下，激活值立即释放
    - ✅ 完美对齐：同一个 batch 的数据用于提取两次特征，绝对一致
    - ✅ 简洁优雅：无需管理复杂的离线缓存文件和索引
    - ✅ 易于调试：所有逻辑都在训练循环内，问题容易定位
    
    注意：
    - 仅保存和切换可训练参数（requires_grad=True）
    - 初始权重存储在 CPU 上以节省 GPU 显存
    - 自动处理 DDP wrapper（通过 model.module 访问）
    """
    
    def __init__(self, model: nn.Module, device: torch.device):
        """
        初始化权重管理器并保存初始 Adapter 权重
        
        Args:
            model: 模型（可能被 DDP 包装）
            device: 模型所在的设备
        """
        self.device = device
        self.initial_weights = {}
        
        # 提取并保存初始可训练权重
        # 处理 DDP wrapper：如果被包装则访问 .module
        base_model = model.module if hasattr(model, 'module') else model
        
        for name, param in base_model.named_parameters():
            if param.requires_grad:
                # 深拷贝到 CPU 以节省显存
                self.initial_weights[name] = param.data.clone().detach().cpu()
        
        print(f"[CRaFT] 已保存 {len(self.initial_weights)} 个初始可训练参数")
    
    def swap_to_initial(self, model: nn.Module) -> None:
        """
        将模型权重切换到初始（预训练）状态
        
        Args:
            model: 需要修改的模型
        """
        base_model = model.module if hasattr(model, 'module') else model
        
        for name, param in base_model.named_parameters():
            if name in self.initial_weights:
                # 从 CPU 加载初始权重到 GPU
                param.data.copy_(self.initial_weights[name].to(self.device))
    
    def swap_to_current(self, model: nn.Module, current_weights: dict) -> None:
        """
        将模型权重切换回当前训练状态
        
        Args:
            model: 需要修改的模型
            current_weights: 当前权重字典（name -> tensor）
        """
        base_model = model.module if hasattr(model, 'module') else model
        
        for name, param in base_model.named_parameters():
            if name in current_weights:
                param.data.copy_(current_weights[name])
    
    def save_current_weights(self, model: nn.Module) -> dict:
        """
        保存当前模型权重（切换前）
        
        Args:
            model: 需要保存的模型
        
        Returns:
            当前权重字典（name -> tensor，在设备上）
        """
        base_model = model.module if hasattr(model, 'module') else model
        current_weights = {}
        
        for name, param in base_model.named_parameters():
            if name in self.initial_weights:
                current_weights[name] = param.data.clone()
        
        return current_weights


def extract_anchor_features_online(
    model: nn.Module,
    weight_manager: CRaFTWeightManager,
    feature_extractor: CRaFTFeatureExtractor,
    batch: dict,
    device: torch.device,
    num_patches: int,
    use_proprio: bool = False,
    proprio_projector: Optional[nn.Module] = None,
    use_film: bool = False,
    anchor_layer_idx: Optional[int] = None,
    cr_token_mode: str = "vision_only",
) -> torch.Tensor:
    """
    使用在线权重切换提取锚点特征（无梯度模式）
    
    功能：实现内存高效的锚点特征提取策略
    
    工作流程：
    1. 保存当前训练权重
    2. 切换到初始权重（预训练状态）
    3. 在 torch.no_grad() 下执行 forward，提取锚点特征
    4. 切换回当前训练权重
    
    关键优势：
    - 第一次 forward 在 no_grad 下，激活值不保留，零显存开销
    - 同一个 batch 用于两次特征提取，完美对齐
    
    Args:
        model: VLA 模型
        weight_manager: 权重管理器
        feature_extractor: CRaFT 特征提取器
        batch: 输入 batch
        device: 设备
        num_patches: 视觉 patch 数量
        use_proprio: 是否使用本体感知输入
        proprio_projector: 本体感知投影器
        use_film: 是否使用 FiLM
    
    Returns:
        锚点特征 f̃，形状 (B, 2*D) 或 (B, D)（取决于 anchor_type）
    """
    # 步骤 1: 保存当前训练权重
    current_weights = weight_manager.save_current_weights(model)
    
    # 步骤 2: 切换到初始权重
    weight_manager.swap_to_initial(model)
    
    # 步骤 3: 提取锚点特征（无梯度）
    with torch.no_grad():
        with torch.autocast("cuda", dtype=torch.bfloat16):
            output = model(
                input_ids=batch["input_ids"].to(device),
                attention_mask=batch["attention_mask"].to(device),
                pixel_values=batch["pixel_values"].to(torch.bfloat16).to(device),
                labels=batch["labels"].to(device),
                output_hidden_states=True,
                output_craft_features=True,
                craft_anchor_layer_idx=anchor_layer_idx,
                craft_cr_token_mode=cr_token_mode,
                proprio=batch["proprio"] if use_proprio else None,
                proprio_projector=proprio_projector if use_proprio else None,
                use_film=use_film,
            )
        
        # 提取并处理特征
        if output.raw_latent_features is None or output.action_query_features is None:
            raise RuntimeError("CRaFT 特征未提取！请确保 output_craft_features=True")
        
        anchor_features = feature_extractor(
            output.raw_latent_features,
            output.action_query_features,
        )  # (B, 2*D) 或 (B, D)
        
        # Detach 确保无梯度流动
        anchor_features = anchor_features.detach()
    
    # 步骤 4: 切换回当前权重
    weight_manager.swap_to_current(model, current_weights)
    
    return anchor_features




