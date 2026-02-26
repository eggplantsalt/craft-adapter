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
    
    # Representation retention
    retention_weight: float = 1.0           # Weight for retention loss (λ in the paper)
    retention_budget: float = 0.1           # Maximum allowed representation drift (ε)
    
    # Dual optimization
    dual_lr: float = 0.01                   # Learning rate for dual variable (η_λ)
    dual_init: float = 0.0                  # Initial value for dual variable
    
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
            Concatenated feature representation f_θ of shape (B, 2*D)
        """
        # Pool both feature types
        pooled_raw = self.pool_features(raw_latent_features)      # (B, D)
        pooled_action = self.pool_features(action_query_features)  # (B, D)
        
        # Concatenate along feature dimension
        combined_features = torch.cat([pooled_raw, pooled_action], dim=-1)  # (B, 2*D)
        
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
        action_grads: dict,
        retention_grads: dict,
    ) -> dict:
        """
        Apply conflict-aware gradient projection for all trainable parameters.
        
        This method projects action gradients when they conflict with retention gradients,
        then combines them with the dual-weighted retention gradients.
        
        Args:
            model: The model whose gradients to project
            action_grads: Dictionary of action gradients {param_name: grad_tensor}
            retention_grads: Dictionary of retention gradients {param_name: grad_tensor}
        
        Returns:
            Dictionary of projected gradients ready for optimizer.step()
        """
        if not self.enable_projection:
            # No projection, just return action gradients
            return action_grads
        
        projected_grads = {}
        
        for name, param in model.named_parameters():
            if not param.requires_grad or name not in action_grads:
                continue
            
            g_act = action_grads[name]
            g_ret = retention_grads[name]
            
            # Flatten gradients for dot product computation
            g_act_flat = g_act.flatten()
            g_ret_flat = g_ret.flatten()
            
            # Compute dot product
            dot_product = torch.dot(g_act_flat, g_ret_flat)
            
            # Only project if gradients conflict (negative dot product)
            if dot_product < 0:
                # Compute projection coefficient
                retention_norm_sq = torch.dot(g_ret_flat, g_ret_flat)
                coeff = dot_product / (retention_norm_sq + self.eps)
                
                # Project action gradient (reshape back to original shape)
                g_act_projected_flat = g_act_flat - coeff * g_ret_flat
                projected_grads[name] = g_act_projected_flat.reshape_as(g_act)
            else:
                # No conflict, keep original action gradient
                projected_grads[name] = g_act
        
        return projected_grads


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
        
        # Initialize dual variable
        self.lambda_val = config.dual_init
    
    def step(self, retention_loss: float) -> None:
        """
        Update the dual variable based on constraint violation.
        
        Args:
            retention_loss: Current value of retention loss
        """
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
    
    This class handles saving initial weights and swapping between current and initial
    weights to enable zero-overhead online reference feature extraction.
    """
    
    def __init__(self, model: nn.Module, device: torch.device):
        """
        Initialize weight manager and save initial trainable weights.
        
        Args:
            model: The model (can be DDP-wrapped)
            device: Device where model resides
        """
        self.device = device
        self.initial_weights = {}
        
        # Extract and save initial weights (only trainable parameters)
        # Handle both DDP-wrapped and non-wrapped models
        if hasattr(model, 'module'):
            # DDP-wrapped model
            base_model = model.module
        else:
            base_model = model
        
        for name, param in base_model.named_parameters():
            if param.requires_grad:
                # Deep copy to CPU to save memory
                self.initial_weights[name] = param.data.clone().detach().cpu()
        
        print(f"[CRaFT] Saved {len(self.initial_weights)} initial weight tensors")
    
    def swap_to_initial(self, model: nn.Module) -> dict:
        """
        Swap model weights to initial weights and return current weights.
        
        Args:
            model: The model to swap weights for
        
        Returns:
            Dictionary of current weights (before swap)
        """
        current_weights = {}
        
        # Handle DDP wrapper
        if hasattr(model, 'module'):
            base_model = model.module
        else:
            base_model = model
        
        for name, param in base_model.named_parameters():
            if name in self.initial_weights:
                # Save current weight
                current_weights[name] = param.data.clone()
                
                # Load initial weight
                param.data.copy_(self.initial_weights[name].to(self.device))
        
        return current_weights
    
    def swap_to_current(self, model: nn.Module, current_weights: dict) -> None:
        """
        Restore model weights to current training weights.
        
        Args:
            model: The model to restore weights for
            current_weights: Dictionary of current weights to restore
        """
        # Handle DDP wrapper
        if hasattr(model, 'module'):
            base_model = model.module
        else:
            base_model = model
        
        for name, param in base_model.named_parameters():
            if name in current_weights:
                param.data.copy_(current_weights[name])

