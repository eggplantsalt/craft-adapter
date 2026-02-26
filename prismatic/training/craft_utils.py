"""
craft_utils.py

Core utilities for CRaFT (Constrained Representation and Fine-Tuning) implementation.
Provides feature extraction, gradient projection, and dual optimization components.
"""

from dataclasses import dataclass
from typing import Optional, Tuple

import torch
import torch.nn as nn


@dataclass
class CRaFTConfig:
    """Configuration for CRaFT training."""
    
    # Feature extraction
    anchor_layer_idx: Optional[int] = None  # If None, uses middle layer (num_layers // 2)
    use_mean_pooling: bool = True           # Whether to use mean pooling for feature aggregation
    anchor_type: str = "concat"             # Feature type: 'concat', 'aq_only', 'raw_only'
    
    # Representation retention
    retention_weight: float = 1.0           # Weight for retention loss (λ in the paper)
    retention_budget: float = 0.1           # Maximum allowed representation drift (ε)
    
    # Dual optimization
    dual_lr: float = 0.01                   # Learning rate for dual variable (η_λ)
    dual_init: float = 0.0                  # Initial value for dual variable
    enable_dual: bool = True                # Whether to enable adaptive dual optimization
    fixed_lambda: float = 0.1               # Fixed lambda when enable_dual=False
    
    # Gradient projection
    projection_eps: float = 1e-8            # Small constant for numerical stability (δ)
    enable_projection: bool = True          # Whether to enable conflict-aware gradient projection
    



class CRaFTFeatureExtractor(nn.Module):
    """
    Extracts and processes bridging features for CRaFT.
    
    Takes raw latent features (C_R) and action query features (C_AQ) from the model,
    applies pooling, and concatenates them to form the final representation f_θ.
    """
    
    def __init__(self, config: CRaFTConfig):
        """
        Initialize the feature extractor.
        
        Args:
            config: CRaFT configuration object
        """
        super().__init__()
        self.config = config
        self.use_mean_pooling = config.use_mean_pooling
        self.anchor_type = config.anchor_type  # 'concat', 'aq_only', 'raw_only'
    
    def pool_features(self, features: torch.Tensor) -> torch.Tensor:
        """
        Apply pooling operation to features.
        
        Args:
            features: Input features of shape (B, seq_len, D)
        
        Returns:
            Pooled features of shape (B, D)
        """
        if self.use_mean_pooling:
            # Mean pooling across sequence dimension
            return features.mean(dim=1)
        else:
            # Max pooling across sequence dimension
            return features.max(dim=1)[0]
    
    def forward(
        self,
        raw_latent_features: torch.Tensor,
        action_query_features: torch.Tensor,
    ) -> torch.Tensor:
        """
        Extract and concatenate bridging features.
        
        Args:
            raw_latent_features: C_R from intermediate layer, shape (B, num_patches, D)
            action_query_features: C_AQ from final layer, shape (B, num_action_tokens, D)
        
        Returns:
            Feature representation f_θ of shape (B, D) or (B, 2*D) depending on anchor_type
        """
        # Pool both feature types
        pooled_raw = self.pool_features(raw_latent_features)      # (B, D)
        pooled_action = self.pool_features(action_query_features)  # (B, D)
        
        # Select features based on anchor_type
        if self.anchor_type == "concat":
            # Concatenate both features (original CRaFT)
            combined_features = torch.cat([pooled_raw, pooled_action], dim=-1)  # (B, 2*D)
        elif self.anchor_type == "aq_only":
            # Use only ActionQuery features
            combined_features = pooled_action  # (B, D)
        elif self.anchor_type == "raw_only":
            # Use only Raw Latent features
            combined_features = pooled_raw  # (B, D)
        else:
            raise ValueError(f"Invalid anchor_type: {self.anchor_type}. Must be 'concat', 'aq_only', or 'raw_only'")
        
        return combined_features


class CRaFTGradientProjector:
    """
    Implements conflict-aware gradient projection for CRaFT.
    
    When the action gradient and retention gradient conflict (negative dot product),
    projects the action gradient to be orthogonal to the retention gradient.
    """
    
    def __init__(self, config: CRaFTConfig):
        """
        Initialize the gradient projector.
        
        Args:
            config: CRaFT configuration object
        """
        self.config = config
        self.eps = config.projection_eps
        self.enable_projection = config.enable_projection
    
    def project_gradients(
        self,
        action_grad: torch.Tensor,
        retention_grad: torch.Tensor,
    ) -> torch.Tensor:
        """
        Project action gradient if it conflicts with retention gradient.
        
        Implements: g̃_act = g_act - <g_act, g_ret> / (||g_ret||² + δ) * g_ret
        
        Args:
            action_grad: Gradient from action loss, shape (D,)
            retention_grad: Gradient from retention loss, shape (D,)
        
        Returns:
            Projected action gradient, shape (D,)
        """
        if not self.enable_projection:
            return action_grad
        
        # Compute dot product
        dot_product = torch.dot(action_grad, retention_grad)
        
        # Only project if gradients conflict (negative dot product)
        if dot_product < 0:
            # Compute projection coefficient
            retention_norm_sq = torch.dot(retention_grad, retention_grad)
            coeff = dot_product / (retention_norm_sq + self.eps)
            
            # Project action gradient
            projected_grad = action_grad - coeff * retention_grad
            
            return projected_grad
        else:
            # No conflict, return original gradient
            return action_grad
    
    def project_gradients_batch(
        self,
        model: nn.Module,
        action_loss: torch.Tensor,
        retention_loss: torch.Tensor,
    ) -> None:
        """
        Compute gradients for both losses and apply projection in-place.
        
        This method:
        1. Computes gradients for action loss
        2. Computes gradients for retention loss
        3. Projects action gradients where conflicts occur
        4. Updates model gradients with projected action gradients + retention gradients
        
        Args:
            model: The model whose gradients to project
            action_loss: Action prediction loss
            retention_loss: Representation retention loss
        """
        # This will be implemented in Phase 3
        raise NotImplementedError("Batch gradient projection will be implemented in Phase 3")


class CRaFTDualOptimizer:
    """
    Manages the dual variable (Lagrange multiplier) λ for CRaFT.
    
    Updates λ based on the constraint violation: λ ← max(0, λ + η_λ * (L_ret - ε))
    """
    
    def __init__(self, config: CRaFTConfig):
        """
        Initialize the dual optimizer.
        
        Args:
            config: CRaFT configuration object
        """
        self.config = config
        self.dual_lr = config.dual_lr
        self.budget = config.retention_budget
        self.enable_dual = config.enable_dual
        
        # Initialize dual variable
        if self.enable_dual:
            self.lambda_val = config.dual_init
        else:
            # Use fixed lambda when dual optimization is disabled
            self.lambda_val = config.fixed_lambda
    
    def step(self, retention_loss: float) -> None:
        """
        Update the dual variable based on constraint violation.
        
        Args:
            retention_loss: Current value of retention loss
        """
        if not self.enable_dual:
            # Keep lambda fixed when dual optimization is disabled
            return
        
        # Compute constraint violation
        violation = retention_loss - self.budget
        
        # Update dual variable with projection to non-negative orthant
        self.lambda_val = max(0.0, self.lambda_val + self.dual_lr * violation)
    
    def get_lambda(self) -> float:
        """
        Get current value of dual variable.
        
        Returns:
            Current λ value
        """
        return self.lambda_val
    
    def reset(self) -> None:
        """Reset dual variable to initial value."""
        self.lambda_val = self.config.dual_init


def compute_retention_loss(
    current_features: torch.Tensor,
    anchor_features: torch.Tensor,
) -> torch.Tensor:
    """
    Compute representation retention loss (MSE between current and anchor features).
    
    Implements: L_ret = ||f_θ - f̃||²
    
    Args:
        current_features: Features from current model, shape (B, D)
        anchor_features: Features from frozen snapshot model, shape (B, D)
    
    Returns:
        Scalar retention loss
    """
    return torch.nn.functional.mse_loss(current_features, anchor_features)


class CRaFTWeightManager:
    """
    Manages weight swapping for online anchor feature extraction.
    
    This class implements the "No-Grad First, Grad Second" strategy to minimize
    peak GPU memory usage. It stores initial adapter weights and provides methods
    to swap between initial and current weights during training.
    """
    
    def __init__(self, model: nn.Module, device: torch.device):
        """
        Initialize weight manager and save initial adapter weights.
        
        Args:
            model: The model (potentially wrapped with DDP)
            device: Device where the model resides
        """
        self.device = device
        self.initial_weights = {}
        
        # Extract and save initial trainable weights
        # Handle DDP wrapper: access .module if wrapped
        base_model = model.module if hasattr(model, 'module') else model
        
        for name, param in base_model.named_parameters():
            if param.requires_grad:
                # Deep copy to CPU to save memory
                self.initial_weights[name] = param.data.clone().detach().cpu()
        
        print(f"[CRaFT] Saved {len(self.initial_weights)} initial trainable parameters")
    
    def swap_to_initial(self, model: nn.Module) -> None:
        """
        Swap model weights to initial (pretrained) state.
        
        Args:
            model: The model to modify
        """
        base_model = model.module if hasattr(model, 'module') else model
        
        for name, param in base_model.named_parameters():
            if name in self.initial_weights:
                param.data.copy_(self.initial_weights[name].to(self.device))
    
    def swap_to_current(self, model: nn.Module, current_weights: dict) -> None:
        """
        Swap model weights back to current training state.
        
        Args:
            model: The model to modify
            current_weights: Dictionary of current weights (name -> tensor)
        """
        base_model = model.module if hasattr(model, 'module') else model
        
        for name, param in base_model.named_parameters():
            if name in current_weights:
                param.data.copy_(current_weights[name])
    
    def save_current_weights(self, model: nn.Module) -> dict:
        """
        Save current model weights (before swapping).
        
        Args:
            model: The model to save from
        
        Returns:
            Dictionary of current weights (name -> tensor on device)
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
) -> torch.Tensor:
    """
    Extract anchor features using online weight swapping (no-grad mode).
    
    This function implements the memory-efficient strategy:
    1. Save current weights
    2. Swap to initial weights
    3. Forward pass with torch.no_grad() to get anchor features
    4. Swap back to current weights
    
    Args:
        model: The VLA model
        weight_manager: Weight manager for swapping
        feature_extractor: CRaFT feature extractor
        batch: Input batch
        device: Device
        num_patches: Number of vision patches
        use_proprio: Whether to use proprioceptive input
        proprio_projector: Proprioceptive projector
        use_film: Whether to use FiLM
    
    Returns:
        Anchor features f̃, shape (B, 2*D)
    """
    # Step 1: Save current training weights
    current_weights = weight_manager.save_current_weights(model)
    
    # Step 2: Swap to initial weights
    weight_manager.swap_to_initial(model)
    
    # Step 3: Extract anchor features (no gradient)
    with torch.no_grad():
        with torch.autocast("cuda", dtype=torch.bfloat16):
            output = model(
                input_ids=batch["input_ids"].to(device),
                attention_mask=batch["attention_mask"].to(device),
                pixel_values=batch["pixel_values"].to(torch.bfloat16).to(device),
                labels=batch["labels"].to(device),
                output_hidden_states=True,
                output_craft_features=True,
                proprio=batch["proprio"] if use_proprio else None,
                proprio_projector=proprio_projector if use_proprio else None,
                use_film=use_film,
            )
        
        # Extract and process features
        if output.raw_latent_features is None or output.action_query_features is None:
            raise RuntimeError("CRaFT features not extracted! Ensure output_craft_features=True")
        
        anchor_features = feature_extractor(
            output.raw_latent_features,
            output.action_query_features,
        )  # (B, 2*D)
        
        # Detach to ensure no gradient flow
        anchor_features = anchor_features.detach()
    
    # Step 4: Swap back to current weights
    weight_manager.swap_to_current(model, current_weights)
    
    return anchor_features




