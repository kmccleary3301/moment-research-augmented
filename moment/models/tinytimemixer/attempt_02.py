import torch
import torch.nn as nn
from torch.nn import functional as F
from dataclasses import dataclass
from typing import Optional, Tuple, Union, List, Dict
from transformers.modeling_outputs import ModelOutput
from argparse import Namespace # Using Namespace for config simplicity like MOMENT
import math
import warnings
import random
import numpy as np
from moment.utils.utils import NamespaceWithDefaults
from moment.data.base import TimeseriesOutputs


from typing import Optional, Tuple, Union, List, Dict, Any # Add Any


# --- Configuration ---
# Let's define a config similar to MOMENT's Namespace approach for flexibility
class AugmentedTTMConfig(Namespace):
    def __init__(self, **kwargs):
        # TTM Base Params
        self.context_length: int = 512
        self.patch_length: int = 16
        self.num_input_channels: int = 1 # Default, adjust as needed
        self.d_model: int = 128
        self.num_layers: int = 6
        self.expansion_factor: int = 2
        self.dropout: float = 0.1
        self.mode: str = "common_channel" # TTM's default
        self.gated_attn: bool = True # TTM's default
        self.norm_mlp: str = "LayerNorm" # TTM's default
        self.scaling: Union[str, bool] = "std" # 'std', 'mean', False
        self.use_positional_encoding: bool = True
        self.target_dropout: float = 0.0 # Specific dropout in head? MOMENT uses head_dropout

        # MOMENT/Augmentation Inspired Params
        self.task_name: str = "pretrain" # 'pretrain', 'forecast', 'classification', 'imputation', 'anomaly_detection'
        self.mask_ratio: float = 0.4 # Masking ratio for pretraining
        self.revin_affine: bool = False # Use RevIN affine parameters?
        self.use_revin: bool = True # Use RevIN instead of TTM's scaler?

        # Task-Specific Params
        self.forecast_horizon: int = 96
        self.n_classes: int = 2 # For classification
        self.head_dropout: float = 0.1 # General dropout for heads

        # TTM Advanced Params (Optional, add if needed based on full TTM implementation)
        # self.resolution_prefix_tuning: bool = False
        # self.frequency_token_vocab_size: int = 10 # Example
        # self.adaptive_patching: bool = False # If implementing adaptive TTM blocks

        # Internal/Derived Params (Set by __post_init__)
        self.patch_stride_len: Optional[int] = None
        self.num_patches: Optional[int] = None
        self.seq_len: Optional[int] = None # Alias for context_length

        # Update with provided kwargs
        for key, value in kwargs.items():
            setattr(self, key, value)

        self.__post_init__() # Call post-init manually

    def __post_init__(self):
        self.patch_stride_len = self.patch_length # Assuming non-overlapping patches like MOMENT/TTM
        self.num_patches = (self.context_length - self.patch_length) // self.patch_stride_len + 1
        self.seq_len = self.context_length # Alias

    def getattr(self, key, default=None):
         return getattr(self, key, default)

# # --- Output Dataclasses ---
# @dataclass
# class AugmentedTTMOutput(ModelOutput):
#     loss: Optional[torch.FloatTensor] = None
#     reconstruction: Optional[torch.FloatTensor] = None # Shape: B x C x SeqLen
#     forecast: Optional[torch.FloatTensor] = None # Shape: B x C x Horizon
#     classification_logits: Optional[torch.FloatTensor] = None # Shape: B x N_Classes
#     anomaly_scores: Optional[torch.FloatTensor] = None # Shape: B x C x SeqLen or B x 1 or ...
#     backbone_output: Optional[torch.FloatTensor] = None # Shape: B x C x N_Patches x D_Model
#     pretrain_mask: Optional[torch.BoolTensor] = None # Shape: B x SeqLen (original time steps masked)
#     input_mask: Optional[torch.BoolTensor] = None # Shape: B x SeqLen (padding mask)
#     norm_metadata: Optional[Dict] = None # For RevIN loc/scale


# --- New Output Dataclass (Mimicking TimeseriesOutputs Structure with Tensors) ---
@dataclass
class AugmentedTTMOutput(ModelOutput): # Still inherit from ModelOutput for HF compatibility if needed
    # Fields matching TimeseriesOutputs (using Tensors where appropriate)
    forecast: Optional[torch.FloatTensor] = None           # Equivalent to TimeseriesOutputs.forecast
    anomaly_scores: Optional[torch.FloatTensor] = None     # Equivalent to TimeseriesOutputs.anomaly_scores
    # labels: int = None # Classification labels usually passed separately, not returned by model forward pass
    input_mask: Optional[torch.BoolTensor] = None          # Equivalent to TimeseriesOutputs.input_mask (using Tensor)
    pretrain_mask: Optional[torch.BoolTensor] = None       # Equivalent to TimeseriesOutputs.pretrain_mask (using Tensor, True=MASKED)
    reconstruction: Optional[torch.FloatTensor] = None     # Equivalent to TimeseriesOutputs.reconstruction
    embeddings: Optional[torch.FloatTensor] = None         # Equivalent to TimeseriesOutputs.embeddings (interpret as backbone output)
    metadata: Optional[Dict[str, Any]] = None              # Equivalent to TimeseriesOutputs.metadata

    # Additional fields needed/useful
    loss: Optional[torch.FloatTensor] = None               # Keep the calculated loss
    classification_logits: Optional[torch.FloatTensor] = None # Specific output for classification task

    # illegal_output: bool = False # Can add if needed for debugging weights

    # Make it behave a bit like a dictionary for easier access
    def __getitem__(self, item):
        return getattr(self, item)

    def __setitem__(self, item, value):
        setattr(self, item, value)


# --- RevIN Implementation (borrowed structure from MOMENT's likely usage) ---
class RevIN(nn.Module):
    def __init__(self, num_features: int, eps=1e-5, affine=True):
        super().__init__()
        self.num_features = num_features
        self.eps = eps
        self.affine = affine
        if self.affine:
            self.affine_weight = nn.Parameter(torch.ones(1, num_features, 1))
            self.affine_bias = nn.Parameter(torch.zeros(1, num_features, 1))

    def forward(self, x: torch.Tensor, mask: Optional[torch.Tensor] = None, mode: str = 'norm'):
        """
        x: [batch_size x n_channels x seq_len]
        mask: [batch_size x seq_len], 1 for observed, 0 for masked/padded.
              Needs broadcasting to [batch_size x n_channels x seq_len].
        mode: 'norm' or 'denorm'
        """
        if mode == 'norm':
            self._get_statistics(x, mask)
            x = self._normalize(x)
        elif mode == 'denorm':
            x = self._denormalize(x)
        else: raise NotImplementedError
        return x

    def _get_statistics(self, x, mask=None):
        dim2reduce = (-1,) # Reduce along seq_len
        if mask is None:
            self.mean = torch.mean(x, dim=dim2reduce, keepdim=True).detach()
            self.stdev = torch.sqrt(torch.var(x, dim=dim2reduce, keepdim=True, unbiased=False) + self.eps).detach()
        else:
            # Expand mask if needed: B x SeqLen -> B x C x SeqLen
            if mask.ndim == 2:
                 mask = mask.unsqueeze(1).expand_as(x)
            mask = mask.float()
            # Masked mean/std calculation
            masked_x = x * mask
            num_observed = mask.sum(dim=dim2reduce, keepdim=True)
            # Avoid division by zero for completely masked series/channels
            safe_num_observed = torch.clamp(num_observed, min=1.0)

            self.mean = (masked_x.sum(dim=dim2reduce, keepdim=True) / safe_num_observed).detach()
            # Calculate variance with masked values excluded
            var = (((masked_x - self.mean * mask)**2).sum(dim=dim2reduce, keepdim=True) / safe_num_observed).detach()
            self.stdev = torch.sqrt(var + self.eps).detach()


    def _normalize(self, x):
        x = x - self.mean
        x = x / self.stdev
        if self.affine:
            x = x * self.affine_weight
            x = x * self.affine_bias # Should be addition? Check MOMENT paper/code. Usually it's y = gamma * (x-mean)/std + beta
            # Assuming standard y = gamma * norm(x) + beta
            x = x * self.affine_weight + self.affine_bias
        return x

    def _denormalize(self, x):
        if self.affine:
            x = (x - self.affine_bias) / (self.affine_weight + self.eps*self.eps) # Add eps for safety
        x = x * self.stdev
        x = x + self.mean
        return x

# --- Simple Scalers from TTM (if not using RevIN) ---
class TinyTimeMixerStdScaler(nn.Module):
    def __init__(self, config):
        super().__init__()
        self.eps = 1e-5

    def forward(self, x, mask):
        # mask: [batch_size x seq_len x n_channels] -> transpose before use
        # x: [batch_size x n_channels x seq_len]
        mask = mask.transpose(1, 2) # B x C x SeqLen
        mask = mask.float()
        num_observed = torch.clamp(mask.sum(dim=-1, keepdim=True), min=1.0) # B x C x 1

        masked_x = x * mask
        mean = (masked_x.sum(dim=-1, keepdim=True) / num_observed).detach() # B x C x 1
        var = (((masked_x - mean * mask)**2).sum(dim=-1, keepdim=True) / num_observed).detach() # B x C x 1
        stdev = torch.sqrt(var + self.eps).detach() # B x C x 1

        loc = mean
        scale = stdev
        return (x - loc) / scale, loc, scale

class TinyTimeMixerMeanScaler(nn.Module):
     def __init__(self, config):
        super().__init__()

     def forward(self, x, mask):
        # mask: [batch_size x seq_len x n_channels] -> transpose before use
        # x: [batch_size x n_channels x seq_len]
        mask = mask.transpose(1, 2) # B x C x SeqLen
        mask = mask.float()
        num_observed = torch.clamp(mask.sum(dim=-1, keepdim=True), min=1.0) # B x C x 1

        masked_x = x * mask
        mean = (masked_x.sum(dim=-1, keepdim=True) / num_observed).detach() # B x C x 1

        loc = mean
        scale = torch.ones_like(loc) # Scale is 1 for mean scaling
        return (x - loc), loc, scale

class TinyTimeMixerNOPScaler(nn.Module):
    def __init__(self, config):
        super().__init__()

    def forward(self, x, mask):
        loc = torch.zeros((x.shape[0], x.shape[1], 1), device=x.device).detach()
        scale = torch.ones((x.shape[0], x.shape[1], 1), device=x.device).detach()
        return x, loc, scale


# --- Patching ---
class Patching(nn.Module):
    def __init__(self, patch_len, stride):
        super().__init__()
        self.patch_len = patch_len
        self.stride = stride

    def forward(self, x):
        """
        x: [batch_size x n_channels x seq_len]
        output: [batch_size x n_channels x n_patches x patch_len]
        """
        batch_size, n_channels, seq_len = x.shape
        n_patches = (seq_len - self.patch_len) // self.stride + 1
        # Use unfold to create patches
        # The size will be (B, C, num_patches, patch_len)
        # The stride is applied to the sequence length dimension (dim=2)
        patches = x.unfold(dimension=2, size=self.patch_len, step=self.stride)
        # Check if unfold output matches expected (B, C, num_patches, patch_len)
        # print(f"Patches shape after unfold: {patches.shape}") # Debug shape
        return patches


# --- Masking ---
class Masking(nn.Module):
    def __init__(self, mask_ratio):
        super().__init__()
        self.mask_ratio = mask_ratio

    def generate_mask(self, x: torch.Tensor, input_mask: Optional[torch.Tensor]=None):
        """
        Generates a mask for pre-training.
        x: Tensor of shape [batch_size x n_channels x seq_len] or [batch_size x seq_len]
        input_mask: Optional tensor of shape [batch_size x seq_len], 1 indicates valid data, 0 indicates padding.
        Returns a boolean mask of shape [batch_size x seq_len]. True indicates KEEP, False indicates MASK.
        """
        batch_size, seq_len = x.shape[0], x.shape[-1]
        device = x.device

        len_keep = int(seq_len * (1 - self.mask_ratio))

        noise = torch.rand(batch_size, seq_len, device=device)  # noise in [0, 1]

        # Adjust noise based on input_mask: We don't want to select padded positions to be kept
        if input_mask is not None:
            # Set noise for padded positions to infinity so they are never chosen among the top-k to keep
            noise[input_mask == 0] = float('inf')
            # Calculate how many valid positions exist
            num_valid = input_mask.sum(dim=1) # B
            # Ensure len_keep doesn't exceed the number of valid positions for each batch item
            actual_len_keep = torch.clamp(torch.full_like(num_valid, len_keep), max=num_valid).long() # B
        else:
            actual_len_keep = torch.full((batch_size,), len_keep, device=device).long()

        # Sort noise and keep indices
        ids_shuffle = torch.argsort(noise, dim=1)  # ascend: small is keep, large is remove
        ids_restore = torch.argsort(ids_shuffle, dim=1)

        # Generate the boolean mask: True for keep, False for mask
        # We compare the rank of each position with actual_len_keep for that batch item
        mask = torch.zeros(batch_size, seq_len, device=device, dtype=torch.bool)
        ranks = ids_restore # Rank of each position after sorting noise (0 is smallest noise)
        # For each batch item i, mark as True if rank < actual_len_keep[i]
        mask = ranks < actual_len_keep.unsqueeze(1)

        # Ensure padded positions are never masked (always "kept" in terms of attention, but handled by attention_mask)
        # This mask is about which tokens are *replaced* by MASK token for reconstruction objective.
        # We DO want to potentially mask valid tokens. Padded tokens should just be ignored by attention.
        # So, the generated mask is correct. `mask`=True means keep original, `mask`=False means replace with [MASK]

        return mask # True = KEEP, False = MASK

    @staticmethod
    def convert_seq_to_patch_view(seq_mask: torch.Tensor, patch_len: int) -> torch.Tensor:
        """
        Converts a sequence mask [B, SeqLen] to a patch view mask [B, NumPatches].
        A patch is considered valid (1) if *all* its corresponding timesteps in seq_mask are valid (1).
        """
        if seq_mask is None: return None
        batch_size, seq_len = seq_mask.shape
        num_patches = (seq_len - patch_len) // patch_len + 1 # Assuming stride == patch_len
        # Reshape seq_mask to [B, NumPatches, PatchLen]
        seq_mask_patches = seq_mask[:, :num_patches * patch_len].view(batch_size, num_patches, patch_len)
        # A patch is valid if all time steps within it are valid
        patch_mask = torch.all(seq_mask_patches == 1, dim=-1).long() # [B, NumPatches]
        return patch_mask


# --- Patch Embedding (modified from MOMENT to handle mask token replacement) ---
class PatchEmbedding(nn.Module):
    def __init__(self, d_model, patch_len, stride, dropout, add_pos_emb, value_emb_bias, orth_gain):
        super().__init__()
        self.patch_len = patch_len
        self.stride = stride
        self.add_positional_embedding = add_pos_emb
        self.value_embedding = nn.Linear(patch_len, d_model, bias=value_emb_bias)
        if self.add_positional_embedding:
            self.positional_embedding = nn.Parameter(torch.zeros(1, 1, 1000, d_model)) # Max 1000 patches
            nn.init.trunc_normal_(self.positional_embedding, std=0.02) # Initialize pos emb

        self.dropout = nn.Dropout(dropout)

        if orth_gain is not None and orth_gain > 0:
             torch.nn.init.orthogonal_(self.value_embedding.weight, gain=orth_gain)
             if value_emb_bias: self.value_embedding.bias.data.zero_()

        # Learnable MASK token
        self.mask_token = nn.Parameter(torch.zeros(1, 1, 1, d_model))
        nn.init.trunc_normal_(self.mask_token, std=0.02) # Initialize mask token

    def forward(self, x: torch.Tensor, time_step_mask: Optional[torch.Tensor] = None):
        """
        x: Patched input [batch_size x n_channels x n_patches x patch_len]
        time_step_mask: Boolean mask [batch_size x seq_len]. True means KEEP, False means MASK.
                       Used to replace patches corresponding to masked time steps with the mask token.
        Returns:
        enc_in: Embeddings [batch_size x n_channels x n_patches x d_model]
        """
        batch_size, n_channels, n_patches, _ = x.shape
        device = x.device

        x = self.value_embedding(x)  # [B x C x N_Patches x D_Model]

        if time_step_mask is not None:
             # Convert time_step_mask [B, SeqLen] -> patch mask [B, N_Patches]
             # A patch is masked if ANY of its time steps are masked (i.e., time_step_mask is False)
             seq_mask_patches = time_step_mask[:, :n_patches * self.patch_len].view(batch_size, n_patches, self.patch_len)
             patch_keep_mask = torch.all(seq_mask_patches, dim=-1) # [B, N_Patches], True if patch should be kept

             # Expand patch_keep_mask for broadcasting: [B, 1, N_Patches, 1]
             patch_keep_mask = patch_keep_mask.unsqueeze(1).unsqueeze(-1).expand(-1, n_channels, -1, x.shape[-1])

             # Replace embeddings where patch_keep_mask is False with self.mask_token
             mask_token_expanded = self.mask_token.expand(batch_size, n_channels, n_patches, -1)
             x = torch.where(patch_keep_mask, x, mask_token_expanded)


        # Add positional embedding
        if self.add_positional_embedding:
             # Check if n_patches exceeds positional embedding size
            if n_patches > self.positional_embedding.shape[2]:
                raise ValueError(f"Number of patches ({n_patches}) exceeds positional embedding size ({self.positional_embedding.shape[2]})")
            x = x + self.positional_embedding[:, :, :n_patches, :]

        return self.dropout(x)


# --- TTM Backbone Components (Simplified based on provided code) ---

class TinyTimeMixerLayerNorm(nn.Module):
    def __init__(self, config: AugmentedTTMConfig):
        super().__init__()
        if config.norm_mlp == "LayerNorm":
            self.norm = nn.LayerNorm(config.d_model)
        elif config.norm_mlp == "BatchNorm":
            # Needs careful handling of dimensions for BatchNorm on patches/features
            # Using LayerNorm as default and simpler option here
            warnings.warn("BatchNorm not fully implemented for MLP norm, using LayerNorm.")
            self.norm = nn.LayerNorm(config.d_model)
        else:
            self.norm = nn.Identity()

    def forward(self, x: torch.Tensor) -> torch.Tensor:
        return self.norm(x)

class TinyTimeMixerMLP(nn.Module):
    def __init__(self, config: AugmentedTTMConfig, for_patch: bool = True):
        super().__init__()
        self.for_patch = for_patch
        in_features = config.num_patches if for_patch else config.d_model
        out_features = in_features
        expanded_features = int(in_features * config.expansion_factor)

        self.norm = TinyTimeMixerLayerNorm(config)
        self.fc1 = nn.Linear(in_features, expanded_features)
        self.activation = nn.GELU()
        self.fc2 = nn.Linear(expanded_features, out_features)
        self.dropout = nn.Dropout(config.dropout)

    def forward(self, x: torch.Tensor) -> torch.Tensor:
        """Input shape: B x C x N_Patches x D_Model"""
        residual = x
        x = self.norm(x)

        if self.for_patch:
            x = x.transpose(-1, -2)  # B x C x D_Model x N_Patches
            x = self.fc1(x)
            x = self.activation(x)
            x = self.fc2(x)
            x = x.transpose(-1, -2)  # B x C x N_Patches x D_Model
        else: # Feature mixing
            x = self.fc1(x)
            x = self.activation(x)
            x = self.fc2(x)

        x = self.dropout(x)
        return x + residual

class TinyTimeMixerGatedAttention(nn.Module):
    def __init__(self, config: AugmentedTTMConfig):
        super().__init__()
        self.norm = TinyTimeMixerLayerNorm(config)
        self.in_proj = nn.Linear(config.d_model, config.d_model * 2)
        self.out_proj = nn.Linear(config.d_model, config.d_model)
        self.dropout = nn.Dropout(config.dropout)

    def forward(self, x: torch.Tensor) -> torch.Tensor:
        """Input shape: B x C x N_Patches x D_Model"""
        residual = x
        x = self.norm(x)
        # Apply projection and split into value and gate
        v_g = self.in_proj(x)
        v, g = torch.chunk(v_g, chunks=2, dim=-1)
        # Apply activation to gate and multiply with value
        x = self.activation(g) * v # Using GELU from MLP activation
        x = self.out_proj(x)
        x = self.dropout(x)
        return x + residual

    def activation(self, x):
         # Using GELU consistent with MLP layers
        return F.gelu(x)

class TinyTimeMixerBlock(nn.Module):
    def __init__(self, config: AugmentedTTMConfig):
        super().__init__()
        self.patch_mixer = TinyTimeMixerMLP(config, for_patch=True)
        self.feature_mixer = TinyTimeMixerMLP(config, for_patch=False)

        if config.mode == "mix_channel":
            self.channel_mixer = TinyTimeMixerMLP(config, for_patch=True) # Applying MLP across channels similar to patch mixing
            warnings.warn("mix_channel mode in TTM block might need different MLP structure")
        elif config.mode == "common_channel":
            self.channel_mixer = None
        else:
            raise ValueError(f"Unknown mode: {config.mode}")

        self.gated_attn = TinyTimeMixerGatedAttention(config) if config.gated_attn else None

    def forward(self, x: torch.Tensor, output_hidden_states: bool = False):
        """Input shape: B x C x N_Patches x D_Model"""
        # Apply patch mixing
        x = self.patch_mixer(x)

        # Apply channel mixing if enabled
        if self.channel_mixer is not None:
            x = x.transpose(1, 2) # B x N_Patches x C x D_Model
            x = self.channel_mixer(x) # Mix across channel dimension (C)
            x = x.transpose(1, 2) # B x C x N_Patches x D_Model

        # Apply feature mixing
        x = self.feature_mixer(x)

        # Apply gated attention if enabled
        if self.gated_attn is not None:
            x = self.gated_attn(x)

        # TTM paper mentions TSMixer block includes partition/merge for adaptive patching.
        # This simplified block does not include that. For full TTM replication, that logic would be needed here or around this block.

        hidden_state = x if output_hidden_states else None
        return x, hidden_state # Only return last state and optionally intermediate


class TinyTimeMixerEncoder(nn.Module):
    def __init__(self, config: AugmentedTTMConfig):
        super().__init__()
        self.layers = nn.ModuleList([TinyTimeMixerBlock(config) for _ in range(config.num_layers)])
        self.norm = TinyTimeMixerLayerNorm(config) # Final norm

    def forward(self, x: torch.Tensor, output_hidden_states: bool = False):
        """Input shape: B x C x N_Patches x D_Model"""
        all_hidden_states = []
        for layer in self.layers:
            x, hidden = layer(x, output_hidden_states=output_hidden_states)
            if output_hidden_states:
                all_hidden_states.append(hidden)

        x = self.norm(x) # Apply final layer norm

        if output_hidden_states:
            all_hidden_states.append(x) # Add final output too

        return x, all_hidden_states if output_hidden_states else None


# --- Heads (Inspired by MOMENT) ---
class ReconstructionHead(nn.Module):
    def __init__(self, config: AugmentedTTMConfig):
        super().__init__()
        self.dropout = nn.Dropout(config.head_dropout)
        self.linear = nn.Linear(config.d_model, config.patch_length)
        # Consider orthogonal init like MOMENT?
        # torch.nn.init.orthogonal_(self.linear.weight, gain=1.41)
        # self.linear.bias.data.zero_()

    def forward(self, x):
        """
        x: [batch_size x n_channels x n_patches x d_model]
        output: [batch_size x n_channels x seq_len], where seq_len = n_patches * patch_len
        """
        x = self.linear(self.dropout(x))  # [B x C x N_Patches x PatchLen]
        # Flatten patches back into sequence length
        # Assuming non-overlapping patches, simple reshape/view works
        batch_size, n_channels, n_patches, patch_len = x.shape
        seq_len = n_patches * patch_len
        x = x.view(batch_size, n_channels, seq_len)
        return x

class ClassificationHead(nn.Module):
    def __init__(self, config: AugmentedTTMConfig):
        super().__init__()
        self.config = config
        # Pool features across patches. Using mean pooling.
        self.pooling = nn.AdaptiveAvgPool1d(1) # Pool the patch dimension
        self.flatten = nn.Flatten(start_dim=1) # Flatten C * D_Model
        self.dropout = nn.Dropout(config.head_dropout)
        # Input size depends on pooling strategy. Mean pooling -> C * d_model
        self.linear = nn.Linear(config.num_input_channels * config.d_model, config.n_classes)

    def forward(self, x):
        """
        x: [batch_size x n_channels x n_patches x d_model]
        output: [batch_size x n_classes]
        """
        # Pool across the patch dimension
        # Input to pool needs to be (B*C, D_Model, N_Patches) or similar
        batch_size, n_channels, n_patches, d_model = x.shape
        x = x.permute(0, 1, 3, 2) # B x C x D_Model x N_Patches
        x = x.reshape(batch_size * n_channels, d_model, n_patches)
        x = self.pooling(x) # B*C x D_Model x 1
        x = x.squeeze(-1) # B*C x D_Model
        x = x.view(batch_size, n_channels * d_model) # B x (C * D_Model)

        # Flatten, Dropout, Linear
        # x = self.flatten(x) # Already flattened from pooling reshape
        x = self.dropout(x)
        y = self.linear(x)  # y: batch_size x n_classes
        return y

class ForecastingHead(nn.Module):
    def __init__(self, config: AugmentedTTMConfig):
        super().__init__()
        self.config = config
        # Flatten features across patches
        self.head_nf = config.d_model * config.num_patches
        self.flatten = nn.Flatten(start_dim=-2) # Flatten patch and feature dims
        self.dropout = nn.Dropout(config.head_dropout)
        self.linear = nn.Linear(self.head_nf, config.forecast_horizon)

    def forward(self, x):
        """
        x: [batch_size x n_channels x n_patches x d_model]
        output: [batch_size x n_channels x forecast_horizon]
        """
        x = self.flatten(x)  # B x C x (N_Patches * D_Model)
        x = self.dropout(x)
        x = self.linear(x)  # B x C x Horizon
        return x

# --- Main Augmented TTM Model ---
class AugmentedTTM(nn.Module):
    def __init__(self, config: AugmentedTTMConfig | NamespaceWithDefaults):
        super().__init__()
        
        print(f"Initializing AugmentedTTM with config: {str(type(config))}")
        
        if isinstance(config, NamespaceWithDefaults):
            self.config : AugmentedTTMConfig = AugmentedTTMConfig(**config.to_json())
        else:
            self.config : AugmentedTTMConfig = config

        print(f"AugmentedTTMConfig: {self.config}")

        # Normalization
        if config.use_revin:
            self.normalizer = RevIN(num_features=self.config.num_input_channels, affine=self.config.revin_affine)
        else:
            scaler_map = {"std": TinyTimeMixerStdScaler, "mean": TinyTimeMixerMeanScaler}
            scaler_class = scaler_map.get(self.config.scaling, TinyTimeMixerNOPScaler)
            self.scaler = scaler_class(self.config)

        # Patching
        self.patching = Patching(patch_len=self.config.patch_length, stride=self.config.patch_stride_len)

        # Masking Generator (for pre-training)
        self.mask_generator = Masking(mask_ratio=self.config.mask_ratio)

        # Embedding (includes mask token logic)
        self.patch_embedding = PatchEmbedding(
            d_model=self.config.d_model,
            patch_len=self.config.patch_length,
            stride=self.config.patch_stride_len,
            dropout=self.config.dropout,
            add_pos_emb=self.config.use_positional_encoding,
            value_emb_bias=False, # Example, align with TTM/MOMENT if needed
            orth_gain=-1 # Example, disable orthogonal init for now
        )

        # TTM Backbone
        self.encoder = TinyTimeMixerEncoder(self.config)

        # Task-specific Heads
        self.reconstruction_head = ReconstructionHead(self.config)
        self.forecasting_head = ForecastingHead(self.config) # For fine-tuned forecasting
        self.classification_head = ClassificationHead(self.config)
        
        self.head = self.reconstruction_head
        
        if config.task_name == 'pre-training' and config.mask_ratio == 0.0:
             warnings.warn("Task is 'pre-training' but mask_ratio is 0.0. No masking will occur.")

    def forward(
        self,
        x_enc: torch.Tensor, # Input: B x C x SeqLen
        input_mask: Optional[torch.Tensor] = None, # Padding mask: B x SeqLen, 1=valid, 0=pad
        mask: Optional[torch.Tensor] = None, # Allow external mask override if needed by Pretraining class?
        **kwargs
    ) -> AugmentedTTMOutput: # Return the revised output class

        # --- Initializations ---
        batch_size, n_channels, seq_len = x_enc.shape
        device = x_enc.device
        task = self.config.task_name

        internal_loss = None # Loss calculated inside the model (set to None for pretrain)
        reconstruction_final = None # De-normalized reconstruction
        forecast_final = None
        classification_logits_final = None
        anomaly_scores_final = None
        pretrain_mask_final = None # True = MASKED
        backbone_output_final = None # Store backbone output for embeddings field

        # --- Input Mask Handling ---
        if input_mask is None:
            input_mask = torch.ones(batch_size, seq_len, device=device, dtype=torch.long)
        input_mask_bool = input_mask.bool()

        # --- Normalization ---
        norm_loc, norm_scale = None, None
        # ... (Normalization logic remains the same: calculate x_enc_normalized, norm_loc, norm_scale) ...
        if self.config.use_revin:
            norm_mask = input_mask_bool # Default norm mask
            if task == "pretrain": # Generate mask only if needed
                if mask is None: # Allow external mask override
                     pretrain_mask_keep_true = self.mask_generator.generate_mask(x_enc, input_mask=input_mask_bool) # Keep=True, Mask=False
                else:
                     # Assuming external mask follows Keep=True convention
                     pretrain_mask_keep_true = mask.bool()

                if pretrain_mask_keep_true is not None:
                    norm_mask = pretrain_mask_keep_true & input_mask_bool # Use only kept & valid points for norm stats
                    pretrain_mask_final = ~pretrain_mask_keep_true # Store final mask (True=MASKED)

            x_enc_normalized = self.normalizer(x_enc, mask=norm_mask, mode='norm')
            # Store stats IF they were computed (might not be if just loaded for inference)
            if hasattr(self.normalizer, 'mean') and hasattr(self.normalizer, 'stdev'):
                norm_loc = self.normalizer.mean
                norm_scale = self.normalizer.stdev
            else: # Need to handle inference case where stats aren't present yet
                _ = self.normalizer(x_enc, mask=input_mask_bool, mode='norm') # Recompute just to get stats
                norm_loc = self.normalizer.mean
                norm_scale = self.normalizer.stdev

        else: # TTM Scaler
            input_mask_ttm = input_mask_bool.unsqueeze(-1).expand(-1, -1, n_channels)
            x_enc_normalized, norm_loc, norm_scale = self.scaler(x_enc, input_mask_ttm)
             # Generate pretrain mask (Keep=True) if needed for embedding step
            if task == "pretrain":
                 if mask is None:
                     pretrain_mask_keep_true = self.mask_generator.generate_mask(x_enc, input_mask=input_mask_bool) # Keep=True, Mask=False
                 else:
                     pretrain_mask_keep_true = mask.bool()
                 if pretrain_mask_keep_true is not None:
                     pretrain_mask_final = ~pretrain_mask_keep_true # Store final mask (True=MASKED)
            else:
                 pretrain_mask_keep_true = None # No masking needed for embedding

        x_enc_normalized = torch.nan_to_num(x_enc_normalized)

        # --- Patching ---
        patches = self.patching(x_enc_normalized) # B x C x N_Patches x PatchLen

        # --- Embedding & Apply Pretrain Mask (if applicable) ---
        # Pass the Keep=True mask to embedding layer
        embeddings = self.patch_embedding(patches, time_step_mask=pretrain_mask_keep_true if task == "pretrain" else None) # B x C x N_Patches x D_Model

        # --- Backbone ---
        embeddings_reshaped = embeddings.view(batch_size * n_channels, self.config.num_patches, self.config.d_model)
        attention_patch_mask = Masking.convert_seq_to_patch_view(input_mask_bool, self.config.patch_length)
        attention_patch_mask = attention_patch_mask.repeat_interleave(n_channels, dim=0)
        backbone_output_reshaped, _ = self.encoder(embeddings_reshaped)
        backbone_output_final = backbone_output_reshaped.view(batch_size, n_channels, self.config.num_patches, self.config.d_model)

        # --- Task-Specific Heads ---

        # ** ALWAYS calculate reconstruction for pre-training task **
        if task == "pretrain":
            # Ensure head gets the correct backbone output
            if backbone_output_final is None:
                 raise RuntimeError("Backbone output is None during pre-training head calculation.")

            reconstruction_normalized = self.reconstruction_head(backbone_output_final)

            # De-normalize using stored/recomputed stats
            if norm_loc is None or norm_scale is None:
                 raise RuntimeError("Normalization stats (loc/scale) are None before de-normalization.")

            if self.config.use_revin:
                 # RevIN denorm needs the stats attached to the instance
                 self.normalizer.mean = norm_loc
                 self.normalizer.stdev = norm_scale
                 reconstruction_final = self.normalizer(reconstruction_normalized, mode='denorm')
            else:
                 reconstruction_final = reconstruction_normalized * norm_scale + norm_loc

            # Ensure pretrain_mask_final is set (True=MASKED)
            if pretrain_mask_final is None:
                 # This should have been set during normalization if mask was generated
                 # If mask ratio was 0.0, create a mask indicating nothing is masked
                 warnings.warn("Pretrain mask was None after normalization step, creating all False mask.")
                 pretrain_mask_final = torch.zeros_like(input_mask_bool)
            pretrain_mask_final = pretrain_mask_final.long()


        elif task == "forecast":
             # Handle zero-shot vs fine-tuned forecast
             if self.config.getattr("finetuning_mode", "finetune") == "zero-shot":
                 # Use reconstruction head (logic might need careful review for zero-shot shift/mask)
                 # For simplicity, let's assume pretrain mask handled shift/mask generation earlier
                 reconstruction_normalized = self.reconstruction_head(backbone_output_final)
                 if self.config.use_revin:
                     self.normalizer.mean = norm_loc
                     self.normalizer.stdev = norm_scale
                     reconstruction_denorm = self.normalizer(reconstruction_normalized, mode='denorm')
                 else:
                     reconstruction_denorm = reconstruction_normalized * norm_scale + norm_loc

                 # Extract forecast part
                 horizon = self.config.forecast_horizon
                 num_masked_patches = math.ceil(horizon / self.config.patch_length)
                 num_masked_timesteps = num_masked_patches * self.config.patch_length
                 end_idx = -num_masked_timesteps + horizon
                 end_idx = None if end_idx == 0 else end_idx
                 forecast_final = reconstruction_denorm[:, :, -num_masked_timesteps:end_idx]
                 reconstruction_final = reconstruction_denorm # Also provide full reconstruction if needed

             else: # Fine-tuned forecast head
                 forecast_normalized = self.forecasting_head(backbone_output_final)
                 # De-normalize
                 if self.config.use_revin:
                     self.normalizer.mean = norm_loc
                     self.normalizer.stdev = norm_scale
                     forecast_final = self.normalizer(forecast_normalized, mode='denorm')
                 else:
                     forecast_final = forecast_normalized * norm_scale + norm_loc
                 # No reconstruction in this path

        elif task == "classification":
            classification_logits_final = self.classification_head(backbone_output_final)
            # No reconstruction in this path

        elif task == "imputation" or task == "anomaly_detection":
             # Use reconstruction head, assume pretrain_mask defines missing/corrupt areas
             reconstruction_normalized = self.reconstruction_head(backbone_output_final)
             if self.config.use_revin:
                 self.normalizer.mean = norm_loc
                 self.normalizer.stdev = norm_scale
                 reconstruction_final = self.normalizer(reconstruction_normalized, mode='denorm')
             else:
                 reconstruction_final = reconstruction_normalized * norm_scale + norm_loc

             if task == "anomaly_detection":
                 anomaly_scores_final = F.mse_loss(reconstruction_final, x_enc, reduction='none').mean(dim=[1, 2])

        # --- Metadata ---
        metadata_final = {'norm': {'loc': norm_loc, 'scale': norm_scale}}
        # print(f"Pretrain mask final: {pretrain_mask_final.shape}")
        
        # --- Return using TimeseriesOutputs ---
        # Ensure all required fields for pretraining are populated
        if task == "pretrain" and reconstruction_final is None:
             raise RuntimeError("Reconstruction is None before returning outputs in pre-training task.")
        if task == "pretrain" and pretrain_mask_final is None:
             raise RuntimeError("Pretrain mask is None before returning outputs in pre-training task.")

        return TimeseriesOutputs(
            reconstruction=reconstruction_final,
            forecast=forecast_final,
            # labels=None, # classification_logits_final handled externally
            anomaly_scores=anomaly_scores_final,
            embeddings=backbone_output_final,
            input_mask=input_mask, # Return the original LongTensor input mask
            pretrain_mask=pretrain_mask_final, # Return LongTensor (1=MASKED)
            metadata=metadata_final
            # illegal_output=False # Add if needed
        )
    
    # def forward(
    #     self,
    #     x_enc: torch.Tensor, # Input: B x C x SeqLen
    #     input_mask: Optional[torch.Tensor] = None, # Padding mask: B x SeqLen, 1=valid, 0=pad
    #     **kwargs
    # ) -> AugmentedTTMOutput:

    #     batch_size, n_channels, seq_len = x_enc.shape
    #     device = x_enc.device
    #     task = self.config.task_name

    #     # --- Input Mask Handling ---
    #     if input_mask is None:
    #         input_mask = torch.ones(batch_size, seq_len, device=device, dtype=torch.long)
    #     # Ensure input_mask is boolean or long for subsequent operations
    #     input_mask_bool = input_mask.bool() # Use boolean for masking logic

        
    #     # --- Preparations before final return ---
    #     task = self.config.task_name
    #     loss = None
    #     reconstruction_normalized = None # Store normalized reconstruction for loss calc
    #     reconstruction_denormalized = None # Store denormalized for output field
    #     forecast_final = None
    #     classification_logits_final = None
    #     anomaly_scores_final = None
    #     pretrain_mask_final = None # True = MASKED
        
    #     # --- Normalization ---
    #     norm_loc, norm_scale = None, None
    #     if self.config.use_revin:
    #         # RevIN needs the mask of VALID (non-padded AND non-pretrain-masked) points for norm stats
    #         # But applies normalization to all points before backbone.
    #         # Let's decide: For pretraining, normalize based on unmasked+unpadded?
    #         # For downstream, normalize based on unpadded?
    #         # MOMENT uses mask * input_mask for pretrain norm. Let's follow that.
    #         norm_mask = input_mask_bool # Default: use only padding mask for norm stats

    #         # Pre-training Mask Generation (only if needed)
    #         pretrain_mask = None # Keep=True, Mask=False
    #         if task in ["pretrain", "imputation", "anomaly_detection"] or \
    #            (task == "forecast" and self.config.getattr("finetuning_mode", "finetune") == "zero-shot"): # Using pretrain head for zero-shot forecast
    #              if task == "pretrain":
    #                  pretrain_mask = self.mask_generator.generate_mask(x_enc, input_mask=input_mask_bool) # Keep=True, Mask=False
    #              elif task == "imputation":
    #                  # Assume missing values are NaN in x_enc, create mask from them
    #                  pretrain_mask = ~torch.isnan(x_enc).any(dim=1) # Keep=True where NOT NaN
    #                  x_enc = torch.nan_to_num(x_enc) # Replace NaNs with 0 for processing
    #              elif task == "anomaly_detection":
    #                   pretrain_mask = torch.ones_like(input_mask_bool) # Keep all for reconstruction
    #              elif task == "forecast": # Zero-shot forecast via reconstruction head
    #                  # Mask the future part
    #                  horizon = self.config.forecast_horizon
    #                  num_masked_patches = math.ceil(horizon / self.config.patch_length)
    #                  num_masked_timesteps = num_masked_patches * self.config.patch_length

    #                  # Shift input, mask end
    #                  x_enc = torch.roll(x_enc, shifts=-num_masked_timesteps, dims=2)
    #                  input_mask_bool = torch.roll(input_mask_bool, shifts=-num_masked_timesteps, dims=1)
    #                  # Create pretrain_mask: Keep beginning, mask end
    #                  pretrain_mask = torch.ones_like(input_mask_bool)
    #                  pretrain_mask[:, -num_masked_timesteps:] = False # Mask=False

    #                  # Also update the input_mask for attention?
    #                  # MOMENT does: input_mask[:, -num_masked_timesteps:] = 0 (don't attend)
    #                  # OR input_mask[:, -num_masked_timesteps:] = 1 (attend to mask tokens)
    #                  # Let's try attending to mask tokens for forecast
    #                  input_mask_bool[:, -num_masked_timesteps:] = True


    #         if pretrain_mask is not None:
    #              norm_mask = pretrain_mask & input_mask_bool # Use only kept & valid points for norm stats

    #         x_enc_normalized = self.normalizer(x_enc, mask=norm_mask, mode='norm')
    #         norm_loc = self.normalizer.mean # Store for denorm
    #         norm_scale = self.normalizer.stdev # Store for denorm

    #     else: # TTM Scaler
    #         # Needs mask in B x SeqLen x C format
    #         input_mask_ttm = input_mask_bool.unsqueeze(-1).expand(-1, -1, n_channels)
    #         x_enc_normalized, norm_loc, norm_scale = self.scaler(x_enc, input_mask_ttm)
    #         # TTM Scaler loc/scale are B x C x 1, RevIN are B x C x 1 too. Consistent.

    #     x_enc_normalized = torch.nan_to_num(x_enc_normalized) # Handle potential NaNs after norm

    #     # --- Patching ---
    #     patches = self.patching(x_enc_normalized) # B x C x N_Patches x PatchLen

    #     # --- Embedding & Apply Pretrain Mask (if applicable) ---
    #     # The pretrain_mask (Keep=True, Mask=False) is passed to embedding layer
    #     # Embedding layer replaces patches where pretrain_mask=False with mask token
    #     embeddings = self.patch_embedding(patches, time_step_mask=pretrain_mask) # B x C x N_Patches x D_Model

    #     # --- Backbone ---
    #     # Reshape for backbone: B*C x N_Patches x D_Model
    #     embeddings_reshaped = embeddings.view(batch_size * n_channels, self.config.num_patches, self.config.d_model)

    #     # Create attention mask from input_mask (padding mask)
    #     # Convert seq padding mask B x SeqLen -> patch padding mask B x N_Patches
    #     attention_patch_mask = Masking.convert_seq_to_patch_view(input_mask_bool, self.config.patch_length)
    #     # Repeat for channels B x N_Patches -> B*C x N_Patches
    #     attention_patch_mask = attention_patch_mask.repeat_interleave(n_channels, dim=0)

    #     # TTM Encoder doesn't explicitly take attention mask in provided code, assuming it handles padding internally or doesn't need it
    #     # If needed, modify TinyTimeMixerEncoder/Block to accept and use `attention_mask`
    #     backbone_output_reshaped, _ = self.encoder(embeddings_reshaped) # B*C x N_Patches x D_Model

    #     # Reshape back: B x C x N_Patches x D_Model
    #     backbone_output = backbone_output_reshaped.view(batch_size, n_channels, self.config.num_patches, self.config.d_model)

    #     # --- Task-Specific Heads ---
    #     if task in ["pretrain", "imputation", "anomaly_detection"] or \
    #        (task == "forecast" and self.config.getattr("finetuning_mode", "finetune") == "zero-shot"):

    #         reconstruction_normalized = self.reconstruction_head(backbone_output) # B x C x SeqLen (Normalized)

    #         # De-normalize for the output field
    #         if self.config.use_revin:
    #              reconstruction_denormalized = self.normalizer(reconstruction_normalized, mode='denorm')
    #         else:
    #              reconstruction_denormalized = reconstruction_normalized * norm_scale + norm_loc

    #         if task == "anomaly_detection":
    #              anomaly_scores_final = F.mse_loss(reconstruction_denormalized, x_enc, reduction='none').mean(dim=[1, 2]) # B

    #         if task == "forecast": # Zero-shot case using reconstruction head
    #              horizon = self.config.forecast_horizon
    #              num_masked_patches = math.ceil(horizon / self.config.patch_length)
    #              num_masked_timesteps = num_masked_patches * self.config.patch_length
    #              end_idx = -num_masked_timesteps + horizon
    #              end_idx = None if end_idx == 0 else end_idx
    #              # Extract from DENORMALIZED reconstruction
    #              forecast_final = reconstruction_denormalized[:, :, -num_masked_timesteps:end_idx] # B x C x Horizon

    #         # Calculate PRETRAIN loss using NORMALIZED values
    #         if task == "pretrain" and pretrain_mask is not None and reconstruction_normalized is not None:
    #             loss_mask_bool = ~pretrain_mask & input_mask_bool # Masked (False in pretrain_mask) AND Valid (True in input_mask)
    #             if loss_mask_bool.any(): # Only compute loss if there are masked tokens
    #                 loss_mask_expanded = loss_mask_bool.unsqueeze(1).expand_as(x_enc_normalized)
    #                 loss = F.mse_loss(reconstruction_normalized[loss_mask_expanded], x_enc_normalized[loss_mask_expanded], reduction='mean')
    #             else:
    #                 loss = torch.tensor(0.0, device=x_enc.device) # Or handle as needed if no tokens were masked

    #         if pretrain_mask is not None:
    #              pretrain_mask_final = ~pretrain_mask # Output mask: True means MASKED


    #     elif task == "forecast": # Fine-tuned forecast
    #         forecast_normalized = self.forecasting_head(backbone_output) # B x C x Horizon (Normalized)
    #         # De-normalize
    #         if self.config.use_revin:
    #             if not hasattr(self.normalizer,'mean') or not hasattr(self.normalizer, 'stdev'):
    #                  norm_mask = input_mask_bool
    #                  _ = self.normalizer(x_enc, mask=norm_mask, mode='norm')
    #             forecast_final = self.normalizer(forecast_normalized, mode='denorm')
    #         else:
    #             if norm_loc is None or norm_scale is None:
    #                  input_mask_ttm = input_mask_bool.unsqueeze(-1).expand(-1, n_channels)
    #                  _, norm_loc, norm_scale = self.scaler(x_enc, input_mask_ttm)
    #             forecast_final = forecast_normalized * norm_scale + norm_loc

    #         # Loss for fine-tuning forecast would typically be calculated externally
    #         # using forecast_final and the ground truth future values.

    #     elif task == "classification":
    #         classification_logits_final = self.classification_head(backbone_output) # B x N_Classes
    #         # Loss for classification calculated externally using classification_logits_final and labels.

    #     # --- Metadata ---
    #     norm_metadata = None
    #     if self.config.use_revin and hasattr(self.normalizer, 'mean') and hasattr(self.normalizer, 'stdev'):
    #          norm_metadata = {'loc': self.normalizer.mean, 'scale': self.normalizer.stdev}
    #     elif not self.config.use_revin and norm_loc is not None and norm_scale is not None:
    #          norm_metadata = {'loc': norm_loc, 'scale': norm_scale}

    #     metadata_final = {'norm': norm_metadata} # Example structure

    #     # --- Return using the new output class ---
    #     return AugmentedTTMOutput(
    #         loss=loss,                            # Calculated loss (mainly for pretrain)
    #         reconstruction=reconstruction_denormalized, # De-normalized reconstruction
    #         forecast=forecast_final,              # De-normalized forecast
    #         classification_logits=classification_logits_final, # Logits for classification
    #         anomaly_scores=anomaly_scores_final,  # Anomaly scores
    #         embeddings=backbone_output,           # Raw backbone output B x C x N_Patches x D_Model
    #         input_mask=input_mask_bool,           # Padding mask (True=VALID)
    #         pretrain_mask=pretrain_mask_final,    # Mask used for pretraining (True=MASKED)
    #         metadata=metadata_final               # Dictionary for extra info like norm stats
    #     )

# --- Example Usage ---
if __name__ == '__main__':
    # --- Config ---
    config = AugmentedTTMConfig(
        context_length=96,
        patch_length=16,
        num_input_channels=7,
        d_model=128,
        num_layers=4,
        task_name="pretrain", # Try 'forecast', 'classification' etc.
        mask_ratio=0.4,
        use_revin=True,
        forecast_horizon=24,
        n_classes=3,
        head_dropout=0.1,
        dropout=0.1
    )

    # --- Model ---
    model = AugmentedTTM(config)
    # print(model)
    print(f"Model parameters: {sum(p.numel() for p in model.parameters() if p.requires_grad):,}")


    # --- Dummy Input ---
    batch_size = 4
    seq_len = config.context_length
    n_channels = config.num_input_channels
    dummy_input = torch.randn(batch_size, n_channels, seq_len)
    dummy_padding_mask = torch.ones(batch_size, seq_len) # Assume no padding
    # Add some padding for testing
    dummy_padding_mask[:, -10:] = 0
    dummy_input = dummy_input * dummy_padding_mask.unsqueeze(1) # Zero out padded values


    # --- Forward Pass ---
    print(f"\n--- Running task: {config.task_name} ---")
    model.train() # Set to train mode for dropout etc.
    output = model(dummy_input, input_mask=dummy_padding_mask)

    # --- Print Shapes ---
    print("Output Shapes:")
    if output.loss is not None: print(f"  Loss: {output.loss.item()}")
    if output.reconstruction is not None: print(f"  Reconstruction: {output.reconstruction.shape}")
    if output.forecast is not None: print(f"  Forecast: {output.forecast.shape}")
    if output.classification_logits is not None: print(f"  Classification Logits: {output.classification_logits.shape}")
    if output.anomaly_scores is not None: print(f"  Anomaly Scores: {output.anomaly_scores.shape}")
    if output.backbone_output is not None: print(f"  Backbone Output: {output.backbone_output.shape}")
    if output.pretrain_mask is not None: print(f"  Pretrain Mask (True=Masked): {output.pretrain_mask.shape}, Num Masked: {output.pretrain_mask.float().sum().item()}/{batch_size*seq_len}")
    if output.input_mask is not None: print(f"  Input Mask (True=Valid): {output.input_mask.shape}")
    if output.norm_metadata and output.norm_metadata.get('loc') is not None : print(f"  Norm Loc: {output.norm_metadata['loc'].shape}")
    if output.norm_metadata and output.norm_metadata.get('scale') is not None : print(f"  Norm Scale: {output.norm_metadata['scale'].shape}")


    # --- Example: Switch task to Forecast (fine-tune) ---
    config.task_name = "forecast"
    # config.finetuning_mode = "zero-shot" # Test zero-shot forecast
    config.finetuning_mode = "finetune" # Test fine-tune forecast head
    model.config = config # Update model's config reference
    print(f"\n--- Running task: {config.task_name} ({config.getattr('finetuning_mode', '')}) ---")
    model.eval() # Set to eval mode
    with torch.no_grad():
        output_forecast = model(dummy_input, input_mask=dummy_padding_mask)
    print("Output Shapes (Forecast):")
    if output_forecast.forecast is not None: print(f"  Forecast: {output_forecast.forecast.shape}")
    if output_forecast.reconstruction is not None: print(f"  Reconstruction (from zero-shot): {output_forecast.reconstruction.shape}")

     # --- Example: Switch task to Classification ---
    config.task_name = "classification"
    model.config = config # Update model's config reference
    print(f"\n--- Running task: {config.task_name} ---")
    model.eval() # Set to eval mode
    with torch.no_grad():
        output_class = model(dummy_input, input_mask=dummy_padding_mask)
    print("Output Shapes (Classification):")
    if output_class.classification_logits is not None: print(f"  Classification Logits: {output_class.classification_logits.shape}")