"""
finetune.py

Fine-tunes Qwen2.5-0.5B via LoRA.
"""

import os
import time
from collections import deque
from dataclasses import dataclass
from pathlib import Path
from typing import Dict, Optional, Tuple, Type
import torch.nn.functional as F
import draccus
import torch
import torch.distributed as dist
import torch.nn as nn
import tqdm
from accelerate import PartialState
from huggingface_hub import HfApi, snapshot_download
from peft import LoraConfig, PeftModel, get_peft_model
from torch.nn.parallel import DistributedDataParallel as DDP
from torch.optim import AdamW
from torch.optim.lr_scheduler import MultiStepLR, CosineAnnealingLR
from torch.utils.data import DataLoader
from transformers import AutoConfig, AutoImageProcessor, AutoModelForVision2Seq, AutoProcessor
from transformers.modeling_outputs import CausalLMOutputWithPast
import wandb

from experiments.robot.openvla_utils import (
    check_model_logic_mismatch,
    model_is_on_hf_hub,
    update_auto_map
)
from prismatic.extern.hf.configuration_prismatic import OpenVLAConfig
from prismatic.extern.hf.modeling_prismatic import OpenVLAForActionPrediction
from prismatic.extern.hf.processing_prismatic import PrismaticImageProcessor, PrismaticProcessor
from prismatic.models.action_heads import L1RegressionActionHead
from prismatic.models.backbones.llm.prompting import PurePromptBuilder
from prismatic.models.film_vit_wrapper import FiLMedPrismaticVisionBackbone
from prismatic.models.projectors import ProprioProjector
from prismatic.training.train_utils import (
    compute_actions_l1_loss,
    compute_token_accuracy,
    get_current_action_mask,
    get_next_actions_mask
)
from prismatic.training.craft_utils import (
    CRaFTConfig,
    CRaFTFeatureExtractor,
    CRaFTGradientProjector,
    CRaFTDualOptimizer,
    CRaFTWeightManager,
    extract_anchor_features_online,
    compute_retention_loss,
)
from prismatic.util.data_utils import PaddedCollatorForActionPrediction
from prismatic.vla.action_tokenizer import ActionTokenizer
from prismatic.vla.constants import (
    ACTION_DIM,
    ACTION_PROPRIO_NORMALIZATION_TYPE,
    NUM_ACTIONS_CHUNK,
    PROPRIO_DIM,
    NUM_TOKENS
)
from prismatic.vla.datasets import RLDSDataset, RLDSBatchTransform
from prismatic.vla.datasets.rlds.utils.data_utils import save_dataset_statistics
from prismatic.models import load, load_vla



# Sane Defaults
os.environ["TOKENIZERS_PARALLELISM"] = "false"

@dataclass
class FinetuneConfig:
    # fmt: off
    config_file_path: str = "openvla/openvla-7b"     # Path to necessary config files of LA-Adapter
    vlm_path: str = "openvla/openvla-7b"             # Path to OpenVLA model (on HuggingFace Hub or stored locally)
    use_minivlm: bool = False                        # 
    resum_vla_path: str = "openvla/openvla-7b"       # Path to OpenVLA model (on HuggingFace Hub or stored locally)

    # Dataset
    data_root_dir: Path = Path("datasets/rlds")      # Directory containing RLDS datasets
    dataset_name: str = "aloha_scoop_x_into_bowl"    # Name of fine-tuning dataset (e.g., `aloha_scoop_x_into_bowl`)
    run_root_dir: Path = Path("runs")                # Path to directory to store logs & checkpoints
    shuffle_buffer_size: int = 100_000               # Dataloader shuffle buffer size (can reduce if OOM errors occur)

    # Algorithm and architecture
    use_l1_regression: bool = True                   # If True, trains continuous action head with L1 regression objective
    use_diffusion: bool = False                      # If True, trains continuous action head with diffusion modeling objective (DDIM)
    num_diffusion_steps: int = 50                    # (When `diffusion==True`) Number of diffusion steps for training 
    use_film: bool = False                           # If True, uses FiLM to infuse language inputs into visual features
    num_images_in_input: int = 1                     # Number of images in the VLA input (default: 1)
    use_proprio: bool = False                        # If True, includes robot proprioceptive state in input
    phase1_path: str = "None"

    # Training configuration
    batch_size: int = 8                              # Batch size per device (total batch size = batch_size * num GPUs)
    learning_rate: float = 5e-4                      # Learning rate
    lr_warmup_steps: int = 0.1                       # Number of steps to warm up learning rate (from 10% to 100%)
    num_steps_before_decay: int = 100000             # Number of steps before LR decays by 10x
    grad_accumulation_steps: int = 1                 # Number of gradient accumulation steps
    max_steps: int = 200000                          # Max number of training steps
    use_val_set: bool = False                        # If True, uses validation set and log validation metrics
    val_freq: int = 10_000                           # (When `use_val_set==True`) Validation set logging frequency in steps
    val_time_limit: int = 180                        # (When `use_val_set==True`) Time limit for computing validation metrics
    save_freq: int = 10_000                          # Checkpoint saving frequency in steps
    save_latest_checkpoint_only: bool = False        # If True, saves only 1 checkpoint, overwriting latest checkpoint
                                                     #   (If False, saves all checkpoints)
    resume: bool = False                             # If True, resumes from checkpoint
    resume_step: Optional[int] = None                # (When `resume==True`) Step number that we are resuming from
    image_aug: bool = True                           # If True, trains with image augmentations (HIGHLY RECOMMENDED)
    diffusion_sample_freq: int = 50                  # (When `use_diffusion==True`) Frequency for sampling in steps

    # LoRA
    use_lora: bool = False                           # If True, uses LoRA fine-tuning
    lora_rank: int = 32                              # Rank of LoRA weight matrix
    lora_dropout: float = 0.0                        # Dropout applied to LoRA weights
    merge_lora_during_training: bool = False         # If True, merges LoRA weights and saves result during training
                                                     #   Note: Merging can be very slow on some machines. If so, set to
                                                     #         False and merge final checkpoint offline!

    # Full Finetune
    use_fz: bool = False                             # If True, uses LoRA fine-tuning

    # Logging
    wandb_entity: str = "your-wandb-entity"          # Name of WandB entity
    wandb_project: str = "your-wandb-project"        # Name of WandB project
    use_wandb: bool = True                            # If False, disable all WandB init/logging
    run_id_note: Optional[str] = None                # Extra note to add to end of run ID for logging
    run_id_override: Optional[str] = None            # Optional string to override the run ID with
    wandb_log_freq: int = 10                         # WandB logging frequency in steps
    console_log_freq: int = 10                       # Console/history logging frequency in steps

    # revision version
    use_pro_version: bool = True                             # the version number
    phase: str = "Training"
    
    # CRaFT configuration
    use_craft: bool = False                                  # Enable CRaFT training
    craft_retention_weight: float = 1.0                      # Weight for retention loss (λ)
    craft_retention_budget: float = 0.1                      # Maximum allowed representation drift (ε)
    craft_dual_lr: float = 0.01                              # Learning rate for dual variable (η_λ)
    craft_projection_eps: float = 1e-8                       # Numerical stability constant (δ)
    craft_enable_projection: bool = True                     # Enable conflict-aware gradient projection
    craft_enable_dual: bool = True                           # Enable adaptive dual variable optimization
    craft_fixed_lambda: float = 0.1                          # Fixed lambda when craft_enable_dual=False
    craft_anchor_type: str = "concat"                        # Anchor feature type: 'concat', 'aq_only', 'raw_only'
    craft_anchor_layer_idx: Optional[int] = None             # Layer index for C_R (None = middle layer)
    craft_log_freq: int = 10                                 # CRaFT metrics logging frequency
    
    # Few-shot configuration
    n_shot_episodes: Optional[int] = None                    # If provided, limits training to first N episodes per task (for few-shot experiments)
    # fmt: on



def remove_ddp_in_checkpoint(state_dict) -> dict:
    """
    Removes the 'module.' prefix from parameter names in a PyTorch model state dictionary that was saved using
    DistributedDataParallel (DDP).

    When a model is trained using PyTorch's DistributedDataParallel, the saved state dictionary contains parameters
    prefixed with 'module.'. This function removes these prefixes to make the state dictionary compatible when
    loading into models that are not yet wrapped in DDP.

    Args:
        state_dict (dict): PyTorch model state dictionary.

    Returns:
        dict: A new state dictionary with the same contents but with 'module.' prefixes removed from parameter names.
              Parameters without the 'module.' prefix remain unchanged.
    """
    new_state_dict = {}
    for k, v in state_dict.items():
        if k[:7] == "module.":
            new_state_dict[k[7:]] = v
        else:
            new_state_dict[k] = v
    return new_state_dict



def get_run_id(cfg) -> str:
    """
    Generates or retrieves an identifier string for an experiment run.

    Args:
        cfg (FinetuneConfig): Training configuration.

    Returns:
        str: Experiment run ID.
    """
    if cfg.run_id_override is not None:
        # Override the run ID with the user-provided ID
        run_id = cfg.run_id_override
    elif cfg.resume:
        # Override run ID with the previous resumed run's ID
        run_id = cfg.config_file_path.split("/")[-1]
        # Remove the "--XXX_chkpt" suffix from the run ID if it exists
        if "chkpt" in run_id.split("--")[-1]:
            run_id = "--".join(run_id.split("--")[:-1])
    else:
        run_id = (
            f"{cfg.config_file_path.split('/')[-1]}+{cfg.dataset_name}"
            f"+b{cfg.batch_size * cfg.grad_accumulation_steps}"
            f"+lr-{cfg.learning_rate}"
        )
        if cfg.use_fz:
            run_id += f"+frozen+dropout-{cfg.lora_dropout}"
        if cfg.use_lora:
            run_id += f"+lora-r{cfg.lora_rank}+dropout-{cfg.lora_dropout}"
        if cfg.image_aug:
            run_id += "--image_aug"
        if cfg.run_id_note is not None:
            run_id += f"--{cfg.run_id_note}"
    return run_id



def load_checkpoint(module_name: str, path: str, step: int, device: str = "cpu") -> dict:
    """
    Loads a checkpoint for a given module.

    Args:
        module_name (str): Name of model component to load checkpoint for.
        path (str): Path to checkpoint directory.
        step (int): Gradient step number of saved checkpoint.
        device (str): String specifying how to remap storage locations (default = "cpu").

    Returns:
        dict: PyTorch model state dictionary.
    """
    checkpoint_path = os.path.join(path, f"{module_name}--{step}_checkpoint.pt")
    print(f"Loading checkpoint: {checkpoint_path}")
    state_dict = torch.load(checkpoint_path, weights_only=True, map_location=device)
    return remove_ddp_in_checkpoint(state_dict)



def wrap_ddp(module: nn.Module, device_id: int, find_unused: bool = False) -> DDP:
    """
    Wrap a module with DistributedDataParallel.

    Args:
        module (nn.Module): PyTorch module.
        device_id (str): Device ID.
        find_unused (bool): Whether to detect parameters without gradients in distributed training.

    Returns:
        DistributedDataParallel: PyTorch module wrapped with DDP.
    """
    return DDP(module, device_ids=[device_id], find_unused_parameters=find_unused, gradient_as_bucket_view=True)



def count_parameters(module: nn.Module, name: str) -> None:
    """
    Counts and prints the number of trainable parameters in a module.

    Args:
        module (nn.Module): PyTorch module.
        module_name (str): Name of model component.

    Returns:
        None.
    """
    num_params = sum(p.numel() for p in module.parameters() if p.requires_grad)
    
    print(f"# trainable params in {name}: {num_params}")



def init_module(
    module_class: Type[nn.Module],
    module_name: str,
    cfg: FinetuneConfig,
    device_id: int,
    module_args: dict,
    to_bf16: bool = False,
    find_unused_params: bool = False,
) -> DDP:
    """
    Initializes a module, optionally loads checkpoint, moves to device, and wraps with DDP.

    Args:
        module_class (Type[nn.Module]): Class of PyTorch module to initialize.
        module_name (str): Name of model component to load checkpoint for.
        cfg (FinetuneConfig): Training configuration.
        device_id (str): Device ID.
        module_args (dict): Args for initializing the module.
        to_bf16 (bool): Whether to convert to torch.bfloat16 data type.
        find_unused_params (bool): Whether to detect parameters without gradients in distributed training.

    Returns:
        DistributedDataParallel: PyTorch module wrapped with DDP.
    """
    module = module_class(**module_args)
    count_parameters(module, module_name)

    if cfg.resume:
        state_dict = load_checkpoint(module_name, cfg.resum_vla_path, cfg.resume_step)
        module.load_state_dict(state_dict)
        print('loaded!!!!!!!!!')

    if to_bf16:
        module = module.to(torch.bfloat16)
    module = module.to(device_id)

    return wrap_ddp(module, device_id, find_unused_params)



def run_forward_pass(
    vla,
    action_head,
    proprio_projector,
    batch,
    action_tokenizer,
    device_id,
    use_l1_regression,
    use_proprio,
    use_film,
    num_patches,
    compute_diffusion_l1=False,
    use_pro_version=True,
    cfg=None
) -> Tuple[torch.Tensor, Dict[str, float]]:
    """
    Compute model forward pass and metrics for both training and validation.

    Args:
        vla (OpenVLAForActionPrediction): Vision-language-action policy.
        action_head (nn.Module): Action head module.
        noisy_action_projector (nn.Module): Noisy action projector module (only used for diffusion).
        proprio_projector (nn.Module): Proprioceptive state projector module.
        batch (dict): Input batch.
        action_tokenizer (ActionTokenizer): Action tokenizer.
        device_id (str): Device ID.
        use_l1_regression (bool): Whether to use L1 regression.
        use_diffusion (bool): Whether to use diffusion.
        use_proprio (bool): Whether to use proprioceptive state as input.
        use_film (bool): Whether to use FiLM for better language following.
        num_patches (int): Number of vision patches.
        compute_diffusion_l1 (bool): Whether to sample actions and compute L1 loss for diffusion (do this once every
                                    diffusion_sample_freq steps during training; do it every batch for validation)
        num_diffusion_steps (int): Number of diffusion steps (only used for diffusion).

    Returns:
        tuple: (loss, metrics_dict)
            loss: The loss tensor with gradient for backpropagation.
            metrics_dict: Dictionary of computed metrics (detached values for logging).
    """
    metrics = {}

    # Get ground-truth action labels
    ground_truth_actions = batch["actions"].to(device_id).to(torch.bfloat16)
    noise, noisy_actions, diffusion_timestep_embeddings = None, None, None

    # VLA forward pass
    with torch.autocast("cuda", dtype=torch.bfloat16):
        output: CausalLMOutputWithPast = vla(
            input_ids=batch["input_ids"].to(device_id),
            attention_mask=batch["attention_mask"].to(device_id),
            pixel_values=batch["pixel_values"].to(torch.bfloat16).to(device_id),
            labels=batch["labels"],
            output_hidden_states=True,
            proprio=batch["proprio"] if use_proprio else None,
            proprio_projector=proprio_projector if use_proprio else None,
            noisy_actions=None,
            noisy_action_projector=None,
            diffusion_timestep_embeddings=None,
            use_film=use_film,
            )

    # Get action masks needed for logging
    ground_truth_token_ids = batch["labels"][:,1:].to(device_id)
    current_action_mask = get_current_action_mask(ground_truth_token_ids)
    next_actions_mask = get_next_actions_mask(ground_truth_token_ids)

    # Compute metrics for discrete action representation (next-token prediction)
    if not (use_l1_regression):
        loss = output.loss
        predicted_token_ids = output.logits[:, num_patches:-1].argmax(dim=2)

        curr_action_accuracy = compute_token_accuracy(
            predicted_token_ids, 
            ground_truth_token_ids, 
            mask=current_action_mask
            )
        curr_action_l1_loss = compute_actions_l1_loss(
            action_tokenizer, 
            predicted_token_ids, 
            ground_truth_token_ids, 
            mask=current_action_mask
            )
        next_actions_accuracy = compute_token_accuracy(
            predicted_token_ids, 
            ground_truth_token_ids, 
            mask=next_actions_mask
            )
        next_actions_l1_loss = compute_actions_l1_loss(
            action_tokenizer, 
            predicted_token_ids, 
            ground_truth_token_ids, 
            mask=next_actions_mask
            )
        
        metrics.update(
            {
                "loss_value": loss.item(),  # Detached value for logging
                "curr_action_accuracy": curr_action_accuracy.item(),
                "curr_action_l1_loss": curr_action_l1_loss.item(),
                "next_actions_accuracy": next_actions_accuracy.item(),
                "next_actions_l1_loss": next_actions_l1_loss.item(),
                }
            )
        
    # Compute metrics for continuous action representations (L1 regression)
    else:
        # Get last layer hidden states
        multi_layer_hidden_states = []
        
        for item in output.hidden_states[0:]:
            # last_hidden_states = output.hidden_states[-1]  # (B, seq_len, D)
            # Get hidden states for text portion of prompt+response (after the vision patches)
            text_hidden_states = item[:, num_patches:-1]
            # Get hidden states for action portion of response
            batch_size = batch["input_ids"].shape[0]
            # actions_hidden_states = text_hidden_states[:, -1, :].reshape(batch_size, 1, -1).to(torch.bfloat16)
            actions_hidden_states = text_hidden_states[current_action_mask | next_actions_mask].reshape(batch_size, 1,NUM_TOKENS, -1).to(torch.bfloat16)
            task_latten_states = item[:, :num_patches].reshape(batch_size, 1, num_patches , -1)
            all_hidden_states = torch.cat((task_latten_states, actions_hidden_states),2)
            multi_layer_hidden_states.append(all_hidden_states)
        multi_layer_hidden_states = torch.cat(multi_layer_hidden_states, dim = 1)

        predicted_actions = action_head.module.predict_action(
            multi_layer_hidden_states,
            proprio=batch["proprio"] if use_proprio else None,
            proprio_projector=proprio_projector if use_proprio else None,
            phase=cfg.phase,
            )

        loss = torch.nn.L1Loss()(predicted_actions, ground_truth_actions)

        metrics.update(
            {
                "loss_value": loss.item(),  # Detached value for logging
            }
        )

        # Get detailed L1 losses for logging
        should_log_l1_loss = use_l1_regression
        if should_log_l1_loss:
            ground_truth_curr_action = ground_truth_actions[:, 0]
            predicted_curr_action = predicted_actions[:, 0]
            ground_truth_next_actions = ground_truth_actions[:, 1:]
            predicted_next_actions = predicted_actions[:, 1:]
            curr_action_l1_loss = torch.nn.L1Loss()(ground_truth_curr_action, predicted_curr_action)
            next_actions_l1_loss = torch.nn.L1Loss()(ground_truth_next_actions, predicted_next_actions)
            if compute_diffusion_l1:
                print('curr: ',curr_action_l1_loss.item())
                # print('next: ',next_actions_l1_loss.item())

            metrics.update(
                {
                    "curr_action_l1_loss": curr_action_l1_loss.item(),
                    "next_actions_l1_loss": next_actions_l1_loss.item(),
                }
            )

    # Return both the loss tensor (with gradients) and the metrics dictionary (with detached values)
    return loss, metrics



def run_forward_pass_craft(
    vla,
    action_head,
    proprio_projector,
    batch,
    action_tokenizer,
    device_id,
    use_l1_regression,
    use_proprio,
    use_film,
    num_patches,
    compute_diffusion_l1=False,
    use_pro_version=True,
    cfg=None,
    feature_extractor=None,
) -> Tuple[torch.Tensor, Dict[str, float], torch.Tensor]:
    """
    CRaFT version of run_forward_pass that also extracts and returns current features.
    
    Returns:
        tuple: (loss, metrics_dict, current_features)
            loss: The loss tensor with gradient for backpropagation.
            metrics_dict: Dictionary of computed metrics (detached values for logging).
            current_features: Current bridging features f_θ, shape (B, 2*D)
    """
    metrics = {}

    # Get ground-truth action labels
    ground_truth_actions = batch["actions"].to(device_id).to(torch.bfloat16)

    # VLA forward pass with CRaFT feature extraction
    with torch.autocast("cuda", dtype=torch.bfloat16):
        output: CausalLMOutputWithPast = vla(
            input_ids=batch["input_ids"].to(device_id),
            attention_mask=batch["attention_mask"].to(device_id),
            pixel_values=batch["pixel_values"].to(torch.bfloat16).to(device_id),
            labels=batch["labels"],
            output_hidden_states=True,
            output_craft_features=True,  # Enable CRaFT feature extraction
            proprio=batch["proprio"] if use_proprio else None,
            proprio_projector=proprio_projector if use_proprio else None,
            noisy_actions=None,
            noisy_action_projector=None,
            diffusion_timestep_embeddings=None,
            use_film=use_film,
        )

    # Extract current features for CRaFT
    if output.raw_latent_features is None or output.action_query_features is None:
        raise RuntimeError("CRaFT features not extracted! Ensure output_craft_features=True")
    
    current_features = feature_extractor(
        output.raw_latent_features,
        output.action_query_features,
    )  # (B, 2*D)

    # Get action masks needed for logging
    ground_truth_token_ids = batch["labels"][:,1:].to(device_id)
    current_action_mask = get_current_action_mask(ground_truth_token_ids)
    next_actions_mask = get_next_actions_mask(ground_truth_token_ids)

    # Compute metrics for continuous action representations (L1 regression)
    if use_l1_regression:
        # Get last layer hidden states
        multi_layer_hidden_states = []
        
        for item in output.hidden_states[0:]:
            text_hidden_states = item[:, num_patches:-1]
            batch_size = batch["input_ids"].shape[0]
            actions_hidden_states = text_hidden_states[current_action_mask | next_actions_mask].reshape(batch_size, 1, NUM_TOKENS, -1).to(torch.bfloat16)
            task_latten_states = item[:, :num_patches].reshape(batch_size, 1, num_patches, -1)
            all_hidden_states = torch.cat((task_latten_states, actions_hidden_states), 2)
            multi_layer_hidden_states.append(all_hidden_states)
        multi_layer_hidden_states = torch.cat(multi_layer_hidden_states, dim=1)

        predicted_actions = action_head.module.predict_action(
            multi_layer_hidden_states,
            proprio=batch["proprio"] if use_proprio else None,
            proprio_projector=proprio_projector if use_proprio else None,
            phase=cfg.phase,
        )

        loss = torch.nn.L1Loss()(predicted_actions, ground_truth_actions)

        metrics.update({
            "loss_value": loss.item(),
        })

        # Get detailed L1 losses for logging
        ground_truth_curr_action = ground_truth_actions[:, 0]
        predicted_curr_action = predicted_actions[:, 0]
        ground_truth_next_actions = ground_truth_actions[:, 1:]
        predicted_next_actions = predicted_actions[:, 1:]
        curr_action_l1_loss = torch.nn.L1Loss()(ground_truth_curr_action, predicted_curr_action)
        next_actions_l1_loss = torch.nn.L1Loss()(ground_truth_next_actions, predicted_next_actions)

        metrics.update({
            "curr_action_l1_loss": curr_action_l1_loss.item(),
            "next_actions_l1_loss": next_actions_l1_loss.item(),
        })
    else:
        # Discrete action prediction (not typically used with CRaFT)
        loss = output.loss
        predicted_token_ids = output.logits[:, num_patches:-1].argmax(dim=2)

        curr_action_accuracy = compute_token_accuracy(predicted_token_ids, ground_truth_token_ids, mask=current_action_mask)
        curr_action_l1_loss = compute_actions_l1_loss(action_tokenizer, predicted_token_ids, ground_truth_token_ids, mask=current_action_mask)
        next_actions_accuracy = compute_token_accuracy(predicted_token_ids, ground_truth_token_ids, mask=next_actions_mask)
        next_actions_l1_loss = compute_actions_l1_loss(action_tokenizer, predicted_token_ids, ground_truth_token_ids, mask=next_actions_mask)
        
        metrics.update({
            "loss_value": loss.item(),
            "curr_action_accuracy": curr_action_accuracy.item(),
            "curr_action_l1_loss": curr_action_l1_loss.item(),
            "next_actions_accuracy": next_actions_accuracy.item(),
            "next_actions_l1_loss": next_actions_l1_loss.item(),
        })

    # Return loss, metrics, and current features
    return loss, metrics, current_features



def compute_smoothened_metrics(metrics_deques) -> dict:
    """
    Compute smoothened metrics from recent deques.

    Args:
        metrics_deques (dict): Dictionary of deques containing recent metrics.

    Returns:
        dict: Dictionary of smoothened metrics.
    """
    smoothened_metrics = {}
    for name, deque in metrics_deques.items():
        if deque and len(deque) > 0:
            smoothened_metrics[name] = sum(deque) / len(deque)
    return smoothened_metrics



def log_metrics_to_wandb(metrics, prefix, step, wandb_entity) -> None:
    """
    Log metrics to Weights & Biases.

    Args:
        metrics (dict): Dictionary of metrics to log
        prefix (str): Prefix for metric names
        step (int): Training step
        wandb_entity (str): W&B entity instance

    Returns:
        None.
    """
    if wandb_entity is None:
        return

    log_dict = {}
    for name, value in metrics.items():
        # Map loss_value to Loss for better readability in W&B
        if name == "loss_value":
            log_dict[f"{prefix}/Loss"] = value
        # Keep other metrics as is
        else:
            log_dict[f"{prefix}/{name.replace('_', ' ').title()}"] = value
    wandb_entity.log(log_dict, step=step)



def save_training_checkpoint(
    cfg,
    run_dir,
    log_step,
    vla,
    processor,
    proprio_projector,
    noisy_action_projector,
    action_head,
    train_dataset,
    distributed_state,
    new_state_dict,
    optimizer=None,
    scheduler=None,
) -> None:
    """
    Save all training checkpoints including model components, LoRA adapter, dataset statistics, and training state.

    Args:
        cfg (FinetuneConfig): Training configuration.
        run_dir (Path): Experiment run directory path.
        log_step (int): Current logging step.
        vla (OpenVLAForActionPrediction): Vision-language-action policy.
        processor (PrismaticProcessor): OpenVLA inputs processor.
        proprio_projector (nn.Module): Proprioceptive state projector module.
        noisy_action_projector (nn.Module): Noisy action projector module (only used for diffusion).
        action_head (nn.Module): Action head module.
        train_dataset (RLDSDataset): Training dataset.
        distributed_state (PartialState): Distributed training state.
        optimizer (torch.optim.Optimizer): Optimizer instance (for resume training).
        scheduler (torch.optim.lr_scheduler._LRScheduler): LR scheduler instance (for resume training).

    Returns:
        None.
    """
    # Determine checkpoint paths and naming
    if cfg.save_latest_checkpoint_only:
        checkpoint_dir = run_dir
        checkpoint_name_suffix = "latest_checkpoint.pt"
    else:
        checkpoint_dir = Path(str(run_dir) + f"--{log_step}_chkpt")
        checkpoint_name_suffix = f"{log_step}_checkpoint.pt"

    adapter_dir = checkpoint_dir / "lora_adapter"

    # Create directories and save dataset statistics (main process only)
    if distributed_state.is_main_process:
        os.makedirs(checkpoint_dir, exist_ok=True)
        os.makedirs(adapter_dir, exist_ok=True)
        save_dataset_statistics(train_dataset.dataset_statistics, checkpoint_dir)
        print(f"Saving Model Checkpoint for Step {log_step}")

    # Wait for directories to be created
    dist.barrier()

    # Save model components (main process only)
    if distributed_state.is_main_process:
        # Save processor and LoRA adapter
        processor.save_pretrained(checkpoint_dir)

        if cfg.use_fz:
            vla.module.save_pretrained(checkpoint_dir) # directly save checkpoint without lora
        else:
            vla.module.save_pretrained(adapter_dir)

        # Save other components
        if cfg.use_proprio and proprio_projector is not None:
            torch.save(proprio_projector.state_dict(), checkpoint_dir / f"proprio_projector--{checkpoint_name_suffix}")

        if cfg.use_diffusion and noisy_action_projector is not None:
            torch.save(
                noisy_action_projector.state_dict(), checkpoint_dir / f"noisy_action_projector--{checkpoint_name_suffix}"
            )

        if cfg.use_l1_regression and action_head is not None:
            torch.save(action_head.state_dict(), checkpoint_dir / f"action_head--{checkpoint_name_suffix}")

        if cfg.use_film:
            # To be safe, just save the entire vision backbone (not just FiLM components)
            torch.save(
                vla.module.vision_backbone.state_dict(), checkpoint_dir / f"vision_backbone--{checkpoint_name_suffix}"
            )
        
        # === Phase 7.5: 保存 Optimizer 和 Scheduler 状态（断点续训支持）===
        if optimizer is not None:
            training_state = {
                'optimizer_state_dict': optimizer.state_dict(),
                'step': log_step,
            }
            if scheduler is not None:
                training_state['scheduler_state_dict'] = scheduler.state_dict()
            
            torch.save(training_state, checkpoint_dir / "training_state.pt")
            print(f"[Phase 7.5] Saved optimizer and scheduler state for step {log_step}")

    # Wait for model components to be saved
    dist.barrier()

    # Merge LoRA weights into base model and save resulting model checkpoint
    # Note: Can be very slow on some devices; if so, we recommend merging offline
    if cfg.use_lora and cfg.merge_lora_during_training:
        if cfg.use_minivlm:
            config = AutoConfig.from_pretrained("pretrained_models/configs/config.json")
            base_vla = AutoModelForVision2Seq.from_config(config, torch_dtype=torch.bfloat16)  # Create a new model with configuration, the parameters are randomly initialized
            # print(new_state_dict['action_queries.weight'])
            new_state_dict['action_queries.weight'] = vla.state_dict()['module.base_model.model.action_queries.weight'].cpu()
            missing_keys, unexpected_keys = base_vla.load_state_dict(new_state_dict, strict=False)
            
        else:
            base_vla = AutoModelForVision2Seq.from_pretrained(
            cfg.config_file_path, torch_dtype=torch.bfloat16, low_cpu_mem_usage=False, trust_remote_code=False
        )


        merged_vla = PeftModel.from_pretrained(base_vla, adapter_dir)
        merged_vla = merged_vla.merge_and_unload()

        if distributed_state.is_main_process:
            merged_vla.save_pretrained(checkpoint_dir)
            print(f"Saved merged model for Step {log_step} at: {checkpoint_dir}")
        
        # Wait for merged model to be saved
        dist.barrier()



def run_validation(
    vla,
    action_head,
    noisy_action_projector,
    proprio_projector,
    val_dataloader,
    action_tokenizer,
    device_id,
    cfg,
    num_patches,
    log_step,
    distributed_state,
    val_time_limit,
) -> None:
    """
    Compute validation set metrics for logging.

    Args:
        vla (OpenVLAForActionPrediction): Vision-language-action policy.
        action_head (nn.Module): Action head module.
        noisy_action_projector (nn.Module): Noisy action projector module (only used for diffusion).
        proprio_projector (nn.Module): Proprioceptive state projector module.
        val_dataloader (DataLoader): Validation data loader.
        action_tokenizer (ActionTokenizer): Action tokenizer.
        device_id (str): Device ID.
        cfg (FinetuneConfig): Training configuration.
        num_patches (int): Number of vision patches.
        log_step (int): Current logging step.
        distributed_state (PartialState): Distributed training state.
        val_time_limit (int): Time limit for computing validation metrics.

    Returns:
        None.
    """
    val_start_time = time.time()
    vla.eval()
    val_batches_count = 0

    # List to store validation metrics
    all_val_metrics = []

    with torch.no_grad():
        for batch in val_dataloader:
            # Always compute L1 loss for validation, even for diffusion
            _, metrics = run_forward_pass(
                vla=vla,
                action_head=action_head,
                proprio_projector=proprio_projector,
                batch=batch,
                action_tokenizer=action_tokenizer,
                device_id=device_id,
                use_l1_regression=cfg.use_l1_regression,
                use_proprio=cfg.use_proprio,
                use_film=cfg.use_film,
                num_patches=num_patches,
                compute_diffusion_l1=True,
                use_pro_version=cfg.use_pro_version
            )

            # Add the loss value to the metrics
            metrics["loss"] = metrics["loss_value"]
            all_val_metrics.append(metrics)
            val_batches_count += 1

            # Cut testing on validation set short if it exceeds time limit
            if time.time() - val_start_time > val_time_limit:
                break

    # Compute average validation metrics
    avg_val_metrics = {}
    for metric_name in all_val_metrics[0].keys():
        values = [metrics[metric_name] for metrics in all_val_metrics if metric_name in metrics]
        if values:
            avg_val_metrics[metric_name] = sum(values) / len(values)

    # Add batch count to metrics
    avg_val_metrics["val_batches_count"] = val_batches_count

    # Log validation metrics to W&B
    if distributed_state.is_main_process:
        log_metrics_to_wandb(avg_val_metrics, "VLA Val", log_step, wandb)



@draccus.wrap()
def finetune(cfg: FinetuneConfig) -> None:
    """
    Fine-tunes base VLA on demonstration dataset via LoRA.

    Allows toggling different action representations (discrete vs. continuous), different learning objectives
    (next-token prediction vs. L1 regression vs. diffusion), FiLM. Also allows for additional model inputs,
    such as additional camera images and robot proprioceptive state. Assumes parallel action generation with
    action chunking.

    Args:
        cfg (FinetuneConfig): Training configuration.

    Returns:
        None.
    """ 

    global RAW_STATE_DICT

    assert not (cfg.use_l1_regression and cfg.use_diffusion), (
        "Cannot do both L1 regression and diffusion. Please pick one of them!"
    )

    # Trim trailing forward slash ('/') in VLA path if it exists
    cfg.config_file_path = cfg.config_file_path.rstrip("/")
    print(f"Fine-tuning OpenVLA Model `{cfg.config_file_path}` on `{cfg.dataset_name}`")

    # Get experiment run ID
    run_id = get_run_id(cfg)

    # Create experiment run directory
    run_dir = cfg.run_root_dir / run_id
    os.makedirs(run_dir, exist_ok=True)

    # GPU setup
    distributed_state = PartialState()
    device_id = distributed_state.local_process_index
    torch.cuda.set_device(device_id)
    torch.cuda.empty_cache()

    # Initialize wandb logging
    if distributed_state.is_main_process and cfg.use_wandb:
        wandb.init(project=cfg.wandb_project, name=f"ft+{run_id}", mode="offline")

    # Print detected constants
    print(
        "Detected constants:\n"
        f"\tNUM_ACTIONS_CHUNK: {NUM_ACTIONS_CHUNK}\n"
        f"\tACTION_DIM: {ACTION_DIM}\n"
        f"\tPROPRIO_DIM: {PROPRIO_DIM}\n"
        f"\tACTION_PROPRIO_NORMALIZATION_TYPE: {ACTION_PROPRIO_NORMALIZATION_TYPE}"
    )
    
    # Print few-shot configuration if enabled
    if cfg.n_shot_episodes is not None:
        print(f"\n[Few-Shot Mode] Training with only {cfg.n_shot_episodes} episodes per task")
        print(f"[Few-Shot Mode] This is {cfg.n_shot_episodes}/50 = {cfg.n_shot_episodes/50*100:.1f}% of full data\n")

    # Two options:
    # (1) Base model is on Hugging Face Hub
    #   - Then download it and record the path to the download directory
    # (2) Base model is stored locally
    #   - Then register model config in HF Auto Classes
    # In both cases, we want to check whether any changes have been made to
    # the `modeling_prismatic.py` file in this codebase; if so, we will copy
    # the file to the downloaded or locally stored checkpoint directory so
    # that the user's changes to the VLA class logic go into effect

    if model_is_on_hf_hub(cfg.config_file_path):
        # Download model directly from Hugging Face Hub
        vla_download_path = snapshot_download(repo_id=cfg.config_file_path)
        # Overwrite VLA path
        cfg.config_file_path = vla_download_path
    else:
        # Register OpenVLA model to HF Auto Classes (not needed if the model is on HF Hub)
        AutoConfig.register("openvla", OpenVLAConfig)
        AutoImageProcessor.register(OpenVLAConfig, PrismaticImageProcessor)
        AutoProcessor.register(OpenVLAConfig, PrismaticProcessor)
        AutoModelForVision2Seq.register(OpenVLAConfig, OpenVLAForActionPrediction)


    # Update config.json and sync model files
    if distributed_state.is_main_process:
        update_auto_map(cfg.config_file_path)
        check_model_logic_mismatch(cfg.config_file_path)

    # Wait for model files to be synced
    dist.barrier()

    # Load processor and VLA
    AutoProcessor.register(OpenVLAConfig, PrismaticProcessor)
    processor = AutoProcessor.from_pretrained(cfg.config_file_path, trust_remote_code=True)

    if cfg.use_minivlm:
        hf_token = ''
        if 'prism-qwen25-extra-dinosiglip-224px-0_5b' in cfg.vlm_path:
            
            vlm = load(cfg.vlm_path, hf_token=hf_token, load_for_training=True)
        else:
            vlm = load_vla(
                cfg.vlm_path,
                hf_token=hf_token,
                load_for_training=True,
                )
        config = AutoConfig.from_pretrained("pretrained_models/configs/config.json")
        vla = AutoModelForVision2Seq.from_config(config, torch_dtype=torch.bfloat16).to(device_id)  # Create a new model with configuration, the parameters are randomly initialized
        # for name, param in model.named_parameters():
        #     print(f"{name}: {param.shape}")
        replace_map = [
            ("vision_backbone.dino_featurizer", "vision_backbone.featurizer"),
            ("vision_backbone.siglip_featurizer", "vision_backbone.fused_featurizer"),
            ("llm_backbone.llm", "language_model"),
            ("projector.projector.0", "projector.fc1"),
            ("projector.projector.2", "projector.fc2"),
            ("projector.projector.4", "projector.fc3"),
            ("gamma", "scale_factor"),
            ]

        def rename_state_dict_keys(state_dict, replace_map):
            new_state_dict = {}
            for k, v in state_dict.items():
                new_k = k
                for old, new in replace_map:
                    if old in new_k:
                        new_k = new_k.replace(old, new)
                new_state_dict[new_k] = v
            return new_state_dict
        
        old_state_dict = vlm.state_dict()
        RAW_STATE_DICT = rename_state_dict_keys(old_state_dict, replace_map)
    
        missing_keys, unexpected_keys = vla.load_state_dict(RAW_STATE_DICT, strict=False)
        del old_state_dict

    else:
        RAW_STATE_DICT ={}
        vla = AutoModelForVision2Seq.from_pretrained(
            cfg.config_file_path,
            torch_dtype=torch.bfloat16,
            low_cpu_mem_usage=False,
            trust_remote_code=False,
            ).to(device_id)

    # Set number of images in VLA input
    vla.vision_backbone.set_num_images_in_input(cfg.num_images_in_input)

    # vla.set_version(cfg.version)

    if cfg.use_lora:
        lora_config = LoraConfig(
            r=cfg.lora_rank,
            lora_alpha= 2 * cfg.lora_rank,
            lora_dropout=cfg.lora_dropout,
            target_modules="all-linear",
            init_lora_weights="gaussian",
        )
        vla = get_peft_model(vla, lora_config)
        for name, param in vla.named_parameters():
            if "action_queries" in name:
                param.requires_grad = True
        vla.print_trainable_parameters()

    else:
        for name, param in vla.named_parameters():
            if "action_queries" in name:
                param.requires_grad = True

    # FiLM setup
    if cfg.use_film:
        count_parameters(vla.vision_backbone, "vla.vision_backbone (original)")
        # Wrap vision backbone with FiLM wrapper
        # Important: For this, must specify `vla.model.vision_backbone` instead of just `vla.vision_backbone`, since the
        # latter would cause the new wrapped backbone to be saved as a new attribute of `vla` instead of overwriting the
        # original one (due to the LoRA wrapper)
        vla.model.vision_backbone = FiLMedPrismaticVisionBackbone(
            vision_backbone=vla.model.vision_backbone,
            llm_dim=vla.llm_dim,
        )
        count_parameters(vla.vision_backbone, "vla.vision_backbone (post-wrap)")
        if cfg.resume:
            state_dict = load_checkpoint("vision_backbone", cfg.config_file_path, cfg.resume_step)
            vla.model.vision_backbone.load_state_dict(state_dict)
        vla.model.vision_backbone = vla.model.vision_backbone.to(device_id)

    # Wrap VLA with DDP
    vla = wrap_ddp(vla, device_id, find_unused=True)

    # === Initialize CRaFT Components (if enabled) ===
    craft_weight_manager = None
    craft_feature_extractor = None
    craft_gradient_projector = None
    craft_dual_optimizer = None
    
    if cfg.use_craft:
        print("\n" + "="*60)
        print("Initializing CRaFT (Constrained Representation and Fine-Tuning)")
        print("="*60)
        
        # Create CRaFT configuration
        craft_config = CRaFTConfig(
            anchor_layer_idx=cfg.craft_anchor_layer_idx,
            use_mean_pooling=True,
            anchor_type=cfg.craft_anchor_type,
            retention_weight=cfg.craft_retention_weight,
            retention_budget=cfg.craft_retention_budget,
            dual_lr=cfg.craft_dual_lr,
            dual_init=0.0,
            enable_dual=cfg.craft_enable_dual,
            fixed_lambda=cfg.craft_fixed_lambda,
            projection_eps=cfg.craft_projection_eps,
            enable_projection=cfg.craft_enable_projection,
        )
        
        # Initialize weight manager (saves initial adapter weights)
        craft_weight_manager = CRaFTWeightManager(vla, device_id)
        
        # Initialize feature extractor
        craft_feature_extractor = CRaFTFeatureExtractor(craft_config).to(device_id)
        craft_feature_extractor.eval()
        
        # Initialize gradient projector
        craft_gradient_projector = CRaFTGradientProjector(craft_config)
        
        # Initialize dual optimizer
        craft_dual_optimizer = CRaFTDualOptimizer(craft_config)
        
        print(f"[CRaFT] Retention budget (ε): {cfg.craft_retention_budget}")
        print(f"[CRaFT] Dual learning rate (η_λ): {cfg.craft_dual_lr}")
        print(f"[CRaFT] Gradient projection: {'Enabled' if cfg.craft_enable_projection else 'Disabled'}")
        print(f"[CRaFT] Dual optimization: {'Enabled' if cfg.craft_enable_dual else f'Disabled (λ={cfg.craft_fixed_lambda})'}")
        print(f"[CRaFT] Anchor type: {cfg.craft_anchor_type}")
        print("="*60 + "\n")

    # If applicable, instantiate proprio projector
    if cfg.use_proprio:
        proprio_projector = init_module(
            ProprioProjector,
            "proprio_projector",
            cfg,
            device_id,
            {"llm_dim": vla.module.llm_dim, "proprio_dim": PROPRIO_DIM},
            to_bf16=True,
        )

    # If applicable, instantiate continuous action head for L1 regression
    if cfg.use_l1_regression:
        action_head = init_module(
        L1RegressionActionHead,
        "action_head",
        cfg,
        device_id,
        {
            "input_dim": vla.module.llm_dim, 
            "hidden_dim": vla.module.llm_dim, 
            "action_dim": ACTION_DIM,
            "use_pro_version": cfg.use_pro_version,
            },
        to_bf16=True,
        )

    # Get number of vision patches
    NUM_PATCHES = vla.module.vision_backbone.get_num_patches() * vla.module.vision_backbone.get_num_images_in_input()
    # If we have proprio inputs, a single proprio embedding is appended to the end of the vision patch embeddings

    # Instantiate optimizer
    trainable_params = [param for param in vla.parameters() if param.requires_grad]
    if cfg.use_l1_regression:
        trainable_params += [param for param in action_head.parameters() if param.requires_grad]

    if cfg.use_proprio:
        trainable_params += [param for param in proprio_projector.parameters() if param.requires_grad]
    print(f"# total trainable params: {sum(p.numel() for p in trainable_params)}")
    optimizer = AdamW(trainable_params, lr=cfg.learning_rate)

    # Record original learning rate
    original_lr = optimizer.param_groups[0]["lr"]

    # Create learning rate scheduler
    # 1. MultiStepLR
    scheduler = MultiStepLR(
        optimizer,
        milestones=[cfg.num_steps_before_decay],  # Number of steps after which LR will change
        gamma=0.1,  # Multiplicative factor of learning rate decay
    )
    # 2. CosineAnnealingLR
    # scheduler = CosineAnnealingLR(
    #         optimizer,
    #         T_max=cfg.num_steps_before_decay, 
    #         eta_min=0.0001,          
    #         )

    # Create Action Tokenizer
    action_tokenizer = ActionTokenizer(processor.tokenizer)

    # Load Fine-tuning Dataset =>> note that we use an RLDS-formatted dataset following Open X-Embodiment by default.
    #   =>> If you want to use a non-RLDS dataset (e.g., a standard PyTorch Dataset) see the following commented block.
    #   =>> Note that our training code does not loop over epochs because the RLDS loader does this implicitly; if using
    #       your own Dataset, make sure to add the appropriate logic to the training loop!
    #
    # ---
    # from prismatic.vla.datasets import DummyDataset
    #
    # train_dataset = DummyDataset(
    #     action_tokenizer,
    #     processor.tokenizer,
    #     image_transform=processor.image_processor.apply_transform,
    #     prompt_builder_fn=PurePromptBuilder,
    # )
    # ---

    # We assume that the model takes as input one third-person camera image and 1 or 2 optional wrist camera image(s)
    use_wrist_image = cfg.num_images_in_input > 1

    # Create training and optional validation datasets
    batch_transform = RLDSBatchTransform(
        action_tokenizer,
        processor.tokenizer,
        image_transform=processor.image_processor.apply_transform,
        prompt_builder_fn=PurePromptBuilder,
        use_wrist_image=use_wrist_image,
        use_proprio=cfg.use_proprio,
        use_minivlm=cfg.use_minivlm
        )
    train_dataset = RLDSDataset(
        cfg.data_root_dir,
        cfg.dataset_name,
        batch_transform,
        resize_resolution=tuple(vla.module.config.image_sizes),
        shuffle_buffer_size=cfg.shuffle_buffer_size,
        image_aug=cfg.image_aug,
        n_shot_episodes=cfg.n_shot_episodes,
    )
    if cfg.use_val_set:
        val_dataset = RLDSDataset(
            cfg.data_root_dir,
            cfg.dataset_name,
            batch_transform,
            resize_resolution=tuple(vla.module.config.image_sizes),
            shuffle_buffer_size=cfg.shuffle_buffer_size // 10,
            image_aug=cfg.image_aug,
            train=False,
            n_shot_episodes=None,  # Always use full validation set
        )

    # [Important] Save dataset statistics so that we can unnormalize actions during inference
    if distributed_state.is_main_process:
        save_dataset_statistics(train_dataset.dataset_statistics, run_dir)

    # Create collator and dataloader
    collator = PaddedCollatorForActionPrediction(
        processor.tokenizer.model_max_length, processor.tokenizer.pad_token_id, padding_side="right"
    )
    dataloader = DataLoader(
        train_dataset,
        batch_size=cfg.batch_size,
        sampler=None,
        collate_fn=collator,
        num_workers=0,  # Important: Set to 0 if using RLDS, which uses its own parallelism
    )
    print('Len of dataloader: ', len(dataloader))
    if cfg.use_val_set:
        val_batch_size = cfg.batch_size
        val_dataloader = DataLoader(
            val_dataset,
            batch_size=val_batch_size,
            sampler=None,
            collate_fn=collator,
            num_workers=0,  # Important: Set to 0 if using RLDS, which uses its own parallelism
        )

    # Deque to store recent train metrics (used for computing smoothened metrics for gradient accumulation)
    recent_metrics = {
        "loss_value": deque(maxlen=cfg.grad_accumulation_steps),
        "curr_action_accuracy": deque(maxlen=cfg.grad_accumulation_steps),
        "curr_action_l1_loss": deque(maxlen=cfg.grad_accumulation_steps),
        "next_actions_accuracy": deque(maxlen=cfg.grad_accumulation_steps),
        "next_actions_l1_loss": deque(maxlen=cfg.grad_accumulation_steps),
    }

    console_log_path = run_dir / "train_progress.log"
    console_log_freq = max(1, cfg.console_log_freq)

    # Start training
    with tqdm.tqdm(total=cfg.max_steps, leave=True) as progress:
        vla.train()
        optimizer.zero_grad()
        for batch_idx, batch in enumerate(dataloader):
            # === CRaFT: Extract anchor features (if enabled) ===
            anchor_features = None
            if cfg.use_craft:
                anchor_features = extract_anchor_features_online(
                    model=vla,
                    weight_manager=craft_weight_manager,
                    feature_extractor=craft_feature_extractor,
                    batch=batch,
                    device=device_id,
                    num_patches=NUM_PATCHES,
                    use_proprio=cfg.use_proprio,
                    proprio_projector=proprio_projector if cfg.use_proprio else None,
                    use_film=cfg.use_film,
                )  # (B, 2*D), detached
            
            # === Standard Forward Pass (with gradient) ===
            compute_diffusion_l1 = (cfg.use_l1_regression and batch_idx % cfg.diffusion_sample_freq == 0) or (cfg.use_diffusion and batch_idx % cfg.diffusion_sample_freq == 0)
            
            # Modified run_forward_pass to also return current features if CRaFT is enabled
            if cfg.use_craft:
                loss, metrics, current_features = run_forward_pass_craft(
                    vla=vla,
                    action_head=action_head,
                    proprio_projector=proprio_projector if cfg.use_proprio else None,
                    batch=batch,
                    action_tokenizer=action_tokenizer,
                    device_id=device_id,
                    use_l1_regression=cfg.use_l1_regression,
                    use_proprio=cfg.use_proprio,
                    use_film=cfg.use_film,
                    num_patches=NUM_PATCHES,
                    compute_diffusion_l1=compute_diffusion_l1,
                    use_pro_version=cfg.use_pro_version,
                    cfg=cfg,
                    feature_extractor=craft_feature_extractor,
                )
            else:
                loss, metrics = run_forward_pass(
                    vla=vla,
                    action_head=action_head,
                    proprio_projector=proprio_projector if cfg.use_proprio else None,
                    batch=batch,
                    action_tokenizer=action_tokenizer,
                    device_id=device_id,
                    use_l1_regression=cfg.use_l1_regression,
                    use_proprio=cfg.use_proprio,
                    use_film=cfg.use_film,
                    num_patches=NUM_PATCHES,
                    compute_diffusion_l1=compute_diffusion_l1,
                    use_pro_version=cfg.use_pro_version,
                    cfg=cfg,
                )

            # Normalize loss to account for gradient accumulation
            normalized_loss = loss / cfg.grad_accumulation_steps

            # === CRaFT: Two-Stage Backward with Gradient Projection ===
            if cfg.use_craft:
                # Stage 1: Backward for action loss, save gradients
                normalized_loss.backward(retain_graph=True)
                
                # Save action gradients
                action_grads = {}
                base_model = vla.module if hasattr(vla, 'module') else vla
                for name, param in base_model.named_parameters():
                    if param.requires_grad and param.grad is not None:
                        action_grads[name] = param.grad.clone()
                
                # Clear gradients
                optimizer.zero_grad()
                
                # Stage 2: Compute retention loss and backward
                retention_loss = compute_retention_loss(current_features, anchor_features)
                retention_loss_scaled = retention_loss / cfg.grad_accumulation_steps
                retention_loss_scaled.backward()
                
                # Save retention gradients
                retention_grads = {}
                for name, param in base_model.named_parameters():
                    if param.requires_grad and param.grad is not None:
                        retention_grads[name] = param.grad.clone()
                
                # Clear gradients again
                optimizer.zero_grad()
                
                # Stage 3: Gradient projection and combination (with conflict tracking)
                lambda_val = craft_dual_optimizer.get_lambda()
                
                # 重置冲突统计计数器
                craft_gradient_projector.reset_conflict_stats()
                
                for name, param in base_model.named_parameters():
                    if param.requires_grad and name in action_grads and name in retention_grads:
                        g_act = action_grads[name].flatten()
                        g_ret = retention_grads[name].flatten()
                        
                        # Project action gradient if conflict exists (返回投影后的梯度和冲突标志)
                        g_act_projected, has_conflict = craft_gradient_projector.project_gradients(g_act, g_ret)
                        
                        # 更新冲突统计
                        craft_gradient_projector.total_params += 1
                        if has_conflict:
                            craft_gradient_projector.num_conflicts += 1
                        
                        # Combine: g_final = g_act_projected + λ * g_ret
                        g_final = g_act_projected + lambda_val * g_ret
                        
                        # Reshape and assign back
                        param.grad = g_final.reshape(param.shape)
                
                # 计算冲突率（论文核心指标）
                conflict_ratio = craft_gradient_projector.get_conflict_ratio()
                
                # Update dual variable
                craft_dual_optimizer.step(retention_loss.item())
                lambda_after = craft_dual_optimizer.get_lambda()
                
                # Add CRaFT metrics
                metrics['retention_loss'] = retention_loss.item()
                metrics['retention_budget'] = cfg.craft_retention_budget
                metrics['lambda_before'] = lambda_val
                metrics['lambda_after'] = lambda_after
                metrics['lambda'] = lambda_after
                metrics['conflict_ratio'] = conflict_ratio  # 新增：冲突率统计
            else:
                # Standard backward pass
                normalized_loss.backward()
            
            # === 计算梯度范数（用于监控训练稳定性）===
            # 在 backward 之后、optimizer.step() 之前计算
            grad_norm = torch.nn.utils.clip_grad_norm_(
                trainable_params, 
                max_norm=float('inf')  # 不裁剪，只计算范数
            ).item()
            metrics['grad_norm'] = grad_norm
            
            # === 获取当前学习率 ===
            current_lr = optimizer.param_groups[0]['lr']
            metrics['learning_rate'] = current_lr

            # Store recent train metrics
            for metric_name, value in metrics.items():
                if metric_name in recent_metrics:
                    recent_metrics[metric_name].append(value)

            # Compute gradient step index
            gradient_step_idx = batch_idx // cfg.grad_accumulation_steps

            # Compute smoothened train metrics
            smoothened_metrics = compute_smoothened_metrics(recent_metrics)

            # Push Metrics to W&B (every wandb_log_freq gradient steps)
            log_step = gradient_step_idx if not cfg.resume else cfg.resume_step + gradient_step_idx
            if distributed_state.is_main_process and log_step % cfg.wandb_log_freq == 0:
                if cfg.use_wandb:
                    log_metrics_to_wandb(smoothened_metrics, "VLA Train", log_step, wandb)
                
                # Log CRaFT-specific metrics (包含冲突率)
                if cfg.use_wandb and cfg.use_craft and log_step % cfg.craft_log_freq == 0:
                    wandb.log({
                        "CRaFT/Retention Loss": metrics.get('retention_loss', 0.0),
                        "CRaFT/Retention Budget": metrics.get('retention_budget', cfg.craft_retention_budget),
                        "CRaFT/Lambda": metrics.get('lambda', 0.0),
                        "CRaFT/Lambda Before": metrics.get('lambda_before', 0.0),
                        "CRaFT/Lambda After": metrics.get('lambda_after', 0.0),
                        "CRaFT/Conflict Ratio": metrics.get('conflict_ratio', 0.0),  # 新增：冲突率
                    }, step=log_step)
                
                # Log gradient norm and learning rate (顶会级日志)
                if cfg.use_wandb:
                    wandb.log({
                        "VLA Train/Gradient Norm": metrics.get('grad_norm', 0.0),
                        "VLA Train/Learning Rate": metrics.get('learning_rate', optimizer.param_groups[0]['lr']),
                    }, step=log_step)

            # [If applicable] Linearly warm up learning rate from 10% to 100% of original
            if cfg.lr_warmup_steps > 0:
                lr_progress = min((gradient_step_idx + 1) / cfg.lr_warmup_steps, 1.0)  # Cap at 1.0
                current_lr = original_lr * (0.1 + 0.9 * lr_progress)
                for param_group in optimizer.param_groups:
                    param_group["lr"] = current_lr

            if distributed_state.is_main_process and gradient_step_idx % cfg.wandb_log_freq == 0:
                # 已经在上面的日志块中记录了学习率，这里移除重复
                pass

            # Optimizer and LR scheduler step
            if (batch_idx + 1) % cfg.grad_accumulation_steps == 0:
                optimizer.step()
                scheduler.step()
                optimizer.zero_grad()

                completed_step = (batch_idx + 1) // cfg.grad_accumulation_steps
                display_step = completed_step if not cfg.resume else cfg.resume_step + completed_step
                
                # 更新 tqdm 进度条，显示关键指标
                progress_desc = f"Step {display_step}/{cfg.max_steps} | Loss: {metrics.get('loss_value', 0.0):.4f}"
                if cfg.use_craft:
                    progress_desc += (
                        f" | Ret: {metrics.get('retention_loss', 0.0):.4f}/{metrics.get('retention_budget', cfg.craft_retention_budget):.4f}"
                        f" | λ: {metrics.get('lambda_before', 0.0):.3f}->{metrics.get('lambda_after', 0.0):.3f}"
                        f" | Conflict: {metrics.get('conflict_ratio', 0.0):.2%}"
                    )
                progress_desc += f" | GradNorm: {metrics.get('grad_norm', 0.0):.2f} | LR: {metrics.get('learning_rate', 0.0):.2e}"
                progress.set_description(progress_desc)
                progress.update()

                if distributed_state.is_main_process and display_step % console_log_freq == 0:
                    history_line = progress_desc
                    tqdm.tqdm.write(history_line)
                    with open(console_log_path, "a", encoding="utf-8") as history_log:
                        history_log.write(history_line + "\n")

            # Save model checkpoint: either keep latest checkpoint only or all checkpoints
            if gradient_step_idx > 0 and log_step % cfg.save_freq == 0:
                save_training_checkpoint(
                    cfg=cfg,
                    run_dir=run_dir,
                    log_step=log_step,
                    vla=vla,
                    processor=processor,
                    proprio_projector=proprio_projector if cfg.use_proprio else None,
                    noisy_action_projector=None,
                    action_head=action_head,
                    train_dataset=train_dataset,
                    distributed_state=distributed_state,
                    new_state_dict=RAW_STATE_DICT,
                    optimizer=optimizer,  # Phase 7.5: 传入 optimizer
                    scheduler=scheduler,  # Phase 7.5: 传入 scheduler
                )

            # Test model on validation set
            if cfg.use_val_set and log_step > 0 and log_step % cfg.val_freq == 0:
                run_validation(
                    vla=vla,
                    action_head=action_head,
                    noisy_action_projector=None,
                    proprio_projector=proprio_projector if cfg.use_proprio else None,
                    val_dataloader=val_dataloader,
                    action_tokenizer=action_tokenizer,
                    device_id=device_id,
                    cfg=cfg,
                    num_patches=NUM_PATCHES,
                    log_step=log_step,
                    distributed_state=distributed_state,
                    val_time_limit=cfg.val_time_limit,
                )
                # Set model back to training mode after validation
                vla.train()

            # Stop training when max_steps is reached
            if log_step == cfg.max_steps:
                print(f"Max step {cfg.max_steps} reached! Stopping training...")
                break


if __name__ == "__main__":
    finetune()
