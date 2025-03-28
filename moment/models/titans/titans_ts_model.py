import logging
import math
import numpy as np
from dataclasses import dataclass
from typing import Optional, Tuple, Union, List

import torch
import torch.nn as nn
import torch.nn.functional as F
from torch.nn import LayerNorm

from transformers.utils import ModelOutput
from transformers.modeling_utils import PreTrainedModel

# Import TITANs modules
from titans_pytorch import NeuralMemory
from titans_pytorch.mac_transformer import flex_attention, SegmentedAttention, MemoryAsContextTransformer

# Import TransformerEncoder and TransformerConfig
from transformers.models.bert.modeling_bert import BertConfig as TransformerConfig
from transformers.models.bert.modeling_bert import BertEncoder as TransformerEncoder
from transformers.modeling_outputs import BaseModelOutput


# Custom RMSNorm implementation for compatibility with TITANs
class RMSNorm(nn.Module):
    def __init__(self, dim, eps=1e-6):
        super().__init__()
        self.eps = eps
        self.scale = nn.Parameter(torch.ones(dim))

    def forward(self, x):
        # Calculate RMS
        rms = torch.sqrt(torch.mean(x ** 2, dim=-1, keepdim=True) + self.eps)
        # Normalize and scale
        x_normed = x / rms
        return self.scale * x_normed


# Add RMSNorm to torch.nn namespace for TITANs compatibility
torch.nn.RMSNorm = RMSNorm


class TimeSeriesRevIN(nn.Module):
    """
    Reversible Instance Normalization for time series.
    Adapted from RevIN: https://arxiv.org/abs/2105.11203
    
    This normalizes data across the time dimension, making it easier for 
    models to learn temporal patterns regardless of scale.
    """
    def __init__(self, num_features: int, eps=1e-5, affine=True, subtract_last=False):
        """
        Initialize RevIN.
        
        Args:
            num_features: Number of features/channels in the time series
            eps: Small constant for numerical stability
            affine: Whether to include learnable affine parameters
            subtract_last: Whether to subtract the last value instead of the mean
        """
        super().__init__()
        self.num_features = num_features
        self.eps = eps
        self.affine = affine
        self.subtract_last = subtract_last
        
        if affine:
            self.affine_weight = nn.Parameter(torch.ones(num_features))
            self.affine_bias = nn.Parameter(torch.zeros(num_features))
            
    def forward(self, x, mask=None, mode='norm', normalize=None):
        """
        Apply normalization or denormalization.
        
        Args:
            x: Input tensor of shape [batch_size, num_features, seq_len]
            mask: Optional mask tensor of shape [batch_size, seq_len]
            mode: Operation mode - 'norm' for normalization, 'denorm' for denormalization
            normalize: For backward compatibility - if True, normalize; if False, denormalize
            
        Returns:
            Normalized or denormalized tensor
        """
        # For backward compatibility
        if normalize is not None:
            mode = 'norm' if normalize else 'denorm'
            
        if mode not in ['norm', 'denorm']:
            raise ValueError(f"Mode must be 'norm' or 'denorm', got {mode}")
        
        # Check input shape
        if x.dim() != 3:
            raise ValueError(f"Expected 3D tensor, got {x.dim()}D tensor of shape {x.shape}")
            
        if mode == 'norm':
            return self._normalize(x, mask)
        else:  # mode == 'denorm'
            return self._denormalize(x, mask)
    
    def _normalize(self, x, mask=None):
        """
        Normalize the input time series.
        
        Args:
            x: Input tensor of shape [batch_size, num_features, seq_len]
            mask: Optional mask tensor of shape [batch_size, seq_len]
            
        Returns:
            Normalized tensor
        """
        batch_size, num_features, seq_len = x.shape
        
        # Handle mask
        if mask is not None:
            # Make mask compatible with x - expand to [batch_size, 1, seq_len]
            mask_expanded = mask.unsqueeze(1)
            
            # Count valid (non-padding) positions
            valid_count = mask.sum(dim=1, keepdim=True)
            
            # Calculate mean and standard deviation only over valid positions
            mean = (x * mask_expanded).sum(dim=2, keepdim=True) / (valid_count.unsqueeze(1) + self.eps)
            
            # Calculate variance over valid positions using unbiased=False to match PyTorch's default
            var = ((x - mean) * mask_expanded).pow(2).sum(dim=2, keepdim=True) / (valid_count.unsqueeze(1) + self.eps)
            
            # Add epsilon for numerical stability
            std = torch.sqrt(var + self.eps)
            
        else:
            # Normal mean and standard deviation over the time dimension
            mean = torch.mean(x, dim=2, keepdim=True)
            var = torch.var(x, dim=2, keepdim=True, unbiased=False)
            std = torch.sqrt(var + self.eps)
        
        # Handle channels without variation
        std = torch.max(std, torch.full_like(std, self.eps))
        
        # Normalize
        x_normed = (x - mean) / std
        
        # Apply affine transform if requested
        if self.affine:
            x_normed = x_normed * self.affine_weight.view(1, -1, 1) + self.affine_bias.view(1, -1, 1)
            
        return x_normed
        
    def _denormalize(self, x, mask=None):
        """
        Denormalize the input time series.
        
        Args:
            x: Input tensor of shape [batch_size, num_features, seq_len]
            mask: Optional mask tensor of shape [batch_size, seq_len]
            
        Returns:
            Denormalized tensor
        """
        # If no statistics are saved, can't denormalize
        if not hasattr(self, '_mean') or not hasattr(self, '_std'):
            # Just undo the affine transform if present
            if self.affine:
                x = (x - self.affine_bias.view(1, -1, 1)) / self.affine_weight.view(1, -1, 1)
            return x
            
        # Apply affine transform if requested
        if self.affine:
            x = (x - self.affine_bias.view(1, -1, 1)) / self.affine_weight.view(1, -1, 1)
            
        # Denormalize
        return x * self._std + self._mean


class TSAdaptivePatcher(nn.Module):
    """
    Time Series Adaptive Patcher for converting time series into patches.
    
    This module divides a time series into patches of equal size,
    where each patch is a segment of consecutive time steps.
    """
    
    def __init__(self, patch_size, stride=None):
        """
        Initialize the time series patcher.
        
        Args:
            patch_size (int): Size of each patch
            stride (int, optional): Stride between consecutive patches. Defaults to patch_size.
        """
        super().__init__()
        self.patch_size = patch_size
        self.stride = stride if stride is not None else patch_size
        
    def forward(self, x, mask=None):
        """
        Convert a time series into patches.
        
        Args:
            x (torch.Tensor): Input time series of shape [batch_size, channels, seq_len]
            mask (torch.Tensor, optional): Mask tensor of shape [batch_size, seq_len]
                indicating valid positions (1) vs padding (0)
        
        Returns:
            Tuple[torch.Tensor, torch.Tensor]: Patches and corresponding padding mask
        """
        batch_size, channels, seq_len = x.shape
        
        print(f"[Patcher] Input shape: {x.shape}, mask shape: {None if mask is None else mask.shape}")
        
        # If sequence is too short, pad to patch_size
        if seq_len < self.patch_size:
            print(f"[Patcher] Sequence too short ({seq_len} < {self.patch_size}), padding to {self.patch_size}")
            padding = torch.zeros(batch_size, channels, self.patch_size - seq_len, device=x.device)
            x = torch.cat([x, padding], dim=2)
            seq_len = self.patch_size
            
            # Update mask if provided
            if mask is not None:
                mask_padding = torch.zeros(batch_size, self.patch_size - mask.shape[1], device=mask.device)
                mask = torch.cat([mask, mask_padding], dim=1)
        
        # Calculate number of patches
        n_patches = 1 + (seq_len - self.patch_size) // self.stride
        print(f"[Patcher] Number of patches: {n_patches}")
        
        # Create patches tensor
        patches = []
        for i in range(n_patches):
            start = i * self.stride
            end = min(start + self.patch_size, seq_len)
            
            # If last patch is smaller, pad it
            if end - start < self.patch_size:
                patch = x[:, :, start:end]
                padding = torch.zeros(batch_size, channels, self.patch_size - (end - start), device=x.device)
                patch = torch.cat([patch, padding], dim=2)
            else:
                patch = x[:, :, start:end]
                
            patches.append(patch)
            
        # Stack patches along new dimension
        patches = torch.stack(patches, dim=1)  # [batch_size, n_patches, channels, patch_size]
        
        # Reshape to flatten channels and patch_size
        patches = patches.view(batch_size, n_patches, -1)
        
        print(f"[Patcher] Output patches shape: {patches.shape}")
        
        # Create padding mask based on the original mask
        padding_mask = None
        if mask is not None:
            print(f"[Patcher] Creating padding mask from input mask")
            padding_mask = torch.zeros(batch_size, n_patches, device=x.device)
            
            for i in range(n_patches):
                start = i * self.stride
                end = min(start + self.patch_size, seq_len)
                
                # A patch is valid if at least 50% of its positions are valid according to the mask
                for b in range(batch_size):
                    valid_ratio = mask[b, start:end].float().mean()
                    padding_mask[b, i] = (valid_ratio >= 0.5)
            
            print(f"[Patcher] Padding mask shape: {padding_mask.shape}")
        
        return patches, padding_mask


class PatchEmbedding(nn.Module):
    """
    Embeds patches into a higher-dimensional space and adds positional embeddings.
    """
    def __init__(self, patch_size=16, in_channels=1, embed_dim=512, dropout=0.1, max_len=1024):
        super().__init__()
        self.value_embedding = nn.Linear(patch_size * in_channels, embed_dim, bias=False)
        self.position_embedding = PositionalEmbedding(max_len=max_len, embed_dim=embed_dim)
        self.dropout = nn.Dropout(p=dropout)
        
    def forward(self, x):
        # Input: [batch_size, num_patches, channels, patch_size]
        batch_size, num_patches, channels, patch_size = x.shape
        
        # Flatten the patches
        x = x.reshape(batch_size, num_patches, channels * patch_size)
        
        # Project to embedding dimension
        x = self.value_embedding(x)  # [batch_size, num_patches, embed_dim]
        
        # Add positional embedding
        x = self.position_embedding(x)
        
        # Apply dropout
        x = self.dropout(x)
        
        return x


class PositionalEmbedding(nn.Module):
    """
    Adds positional information to input embeddings.
    Uses sinusoidal positional embeddings.
    """
    def __init__(self, max_len=1024, embed_dim=512):
        super().__init__()
        # Create positional embeddings
        pe = torch.zeros(max_len, embed_dim)
        position = torch.arange(0, max_len).unsqueeze(1)
        div_term = torch.exp(torch.arange(0, embed_dim, 2) * -(math.log(10000.0) / embed_dim))
        
        pe[:, 0::2] = torch.sin(position * div_term)
        pe[:, 1::2] = torch.cos(position * div_term)
        self.register_buffer('pe', pe.unsqueeze(0))
        self.max_len = max_len
        
    def forward(self, x):
        # X shape: [batch_size, seq_len, embed_dim]
        seq_len = x.size(1)
        if seq_len > self.max_len:
            raise ValueError(f"Sequence length ({seq_len}) exceeds maximum length ({self.max_len})")
            
        x = x + self.pe[:, :seq_len]
        return x


class SlidingWindowAttention(nn.Module):
    """
    Attention mechanism with a sliding window to restrict context.
    This allows for more efficient computation when dealing with long sequences.
    """
    def __init__(self, dim=512, window_size=64, num_heads=8, dropout=0.1):
        super().__init__()
        self.dim = dim
        self.window_size = window_size
        self.num_heads = num_heads
        self.head_dim = dim // num_heads
        self.scale = self.head_dim ** -0.5
        
        self.qkv = nn.Linear(dim, dim * 3)
        self.proj = nn.Linear(dim, dim)
        self.dropout = nn.Dropout(dropout)
        
    def forward(self, x, mask=None):
        # x shape: [batch_size, seq_len, dim]
        batch_size, seq_len, _ = x.shape
        
        # Calculate QKV
        qkv = self.qkv(x).reshape(batch_size, seq_len, 3, self.num_heads, self.head_dim)
        qkv = qkv.permute(2, 0, 3, 1, 4)  # [3, batch_size, num_heads, seq_len, head_dim]
        q, k, v = qkv[0], qkv[1], qkv[2]
        
        # Scaled dot-product attention with sliding window
        attn_output = self._sliding_window_attention(q, k, v, mask)
        
        # Reshape and project
        attn_output = attn_output.transpose(1, 2).contiguous().view(batch_size, seq_len, self.dim)
        attn_output = self.proj(attn_output)
        attn_output = self.dropout(attn_output)
        
        return attn_output
    
    def _sliding_window_attention(self, q, k, v, mask=None):
        batch_size, num_heads, seq_len, head_dim = q.shape
        
        # If sequence length is shorter than window size, do regular attention
        if seq_len <= self.window_size:
            attn = (q @ k.transpose(-2, -1)) * self.scale
            if mask is not None:
                # Ensure mask has the right dimensions
                if mask.dim() == 2 and mask.shape[1] == seq_len:
                    # Add dimensions for broadcasting with attention matrix
                    mask = mask.unsqueeze(1).unsqueeze(1)
                else:
                    logging.warning(f"SlidingWindowAttention: Mask shape {mask.shape} incompatible with sequence length {seq_len}")
                    # Create a default mask that allows attention to all positions
                    mask = None
                
                if mask is not None:
                    attn = attn.masked_fill(~mask, float('-inf'))
            
            attn = F.softmax(attn, dim=-1)
            output = attn @ v
            return output
        
        # For longer sequences, use sliding window attention
        outputs = []
        
        for i in range(seq_len):
            # Define window boundaries
            window_start = max(0, i - self.window_size // 2)
            window_end = min(seq_len, i + self.window_size // 2 + 1)
            
            # Extract query, key, value for this position/window
            q_i = q[:, :, i:i+1]  # query for current position
            k_w = k[:, :, window_start:window_end]  # keys in window
            v_w = v[:, :, window_start:window_end]  # values in window
            
            # Compute attention for this position
            attn = (q_i @ k_w.transpose(-2, -1)) * self.scale
            
            if mask is not None:
                # Handle mask for the current window
                if mask.dim() == 2 and mask.shape[1] == seq_len:
                    # Extract window mask and add dimensions for broadcasting
                    window_mask = mask[:, window_start:window_end].unsqueeze(1).unsqueeze(1)
                    attn = attn.masked_fill(~window_mask, float('-inf'))
                else:
                    # If mask dimensions don't match, ignore it for this window
                    pass
                
            attn = F.softmax(attn, dim=-1)
            output_i = attn @ v_w
            outputs.append(output_i)
        
        # Concatenate all position outputs
        output = torch.cat(outputs, dim=2)
        return output


class MLPBlock(nn.Module):
    """
    MLP block with residual connection for transformer-like architectures.
    """
    def __init__(self, in_features=512, hidden_features=1024, dropout=0.1):
        super().__init__()
        self.fc1 = nn.Linear(in_features, hidden_features)
        self.act = nn.GELU()
        self.fc2 = nn.Linear(hidden_features, in_features)
        self.dropout = nn.Dropout(dropout)
        
    def forward(self, x):
        shortcut = x
        x = self.fc1(x)
        x = self.act(x)
        x = self.dropout(x)
        x = self.fc2(x)
        x = self.dropout(x)
        x = x + shortcut
        return x


class TitanShortTermBlock(nn.Module):
    """
    Transformer-like block for processing local dependencies in the time series.
    """
    def __init__(self, dim=512, window_size=64, num_heads=8, mlp_ratio=4, dropout=0.1):
        super().__init__()
        # Pre-normalization for attention
        self.norm1 = LayerNorm(dim)
        
        # Sliding window self-attention
        self.self_attn = SlidingWindowAttention(
            dim=dim,
            window_size=window_size,
            num_heads=num_heads,
            dropout=dropout
        )
        
        # Pre-normalization for MLP
        self.norm2 = LayerNorm(dim)
        
        # MLP block
        self.ff = MLPBlock(
            in_features=dim,
            hidden_features=int(dim * mlp_ratio),
            dropout=dropout
        )
        
    def forward(self, x):
        # Apply layernorm before attention (pre-norm transformer design)
        x = x + self.self_attn(self.norm1(x))
        
        # Apply layernorm before MLP
        x = x + self.ff(self.norm2(x))
        
        return x


class NeuralMemoryBlock(nn.Module):
    """
    Block that wraps the TITANs NeuralMemory module for compatibility with our architecture.
    Uses the official TITANs implementation.
    """
    def __init__(
        self, 
        dim=512, 
        chunk_size=64, 
        heads=4, 
        momentum=True, 
        weight_decay=True,  # This parameter is used internally, but not passed to NeuralMemory directly
        num_kv_per_token=2,
        dropout=0.1
    ):
        super().__init__()
        
        # Initialize official TITANs NeuralMemory module
        self.memory_module = NeuralMemory(
            dim=dim,
            chunk_size=chunk_size,
            heads=heads,
            momentum=momentum,
            # Note: weight_decay is handled internally in this class, not passed to NeuralMemory
            num_kv_per_token=num_kv_per_token,
        )
        
        # Store the weight_decay parameter for our own use
        self.use_weight_decay = weight_decay
        
        # Additional projection to ensure proper dimensionality
        self.projection = nn.Linear(dim, dim)
        self.dropout = nn.Dropout(dropout)
        self.layer_norm = LayerNorm(dim)
        
    def forward(self, x, store_mask=None):
        # Apply layer normalization
        normed_x = self.layer_norm(x)
        
        # Process through TITANs neural memory
        # Handle the mask if provided
        if store_mask is not None:
            # Get the expected sequence length from the chunk size and input dimensions
            batch_size, seq_len, _ = normed_x.shape
            chunk_size = getattr(self.memory_module, 'chunk_size', 64)
            
            # Check if dimensions are compatible
            if store_mask.shape[-1] != seq_len:
                logging.info(f"NeuralMemoryBlock: Mask dimension {store_mask.shape[-1]} doesn't match sequence length {seq_len}")
                
                # If mask is provided but dimensions don't match, create a compatibility mask
                new_mask = torch.ones((batch_size, seq_len), dtype=torch.bool, device=x.device)
                
                # If the mask is smaller than seq_len, copy the available values
                if store_mask.shape[-1] < seq_len:
                    new_mask[:, :store_mask.shape[-1]] = store_mask.bool()
                    logging.info(f"Using first {store_mask.shape[-1]} values from smaller mask")
                else:
                    # If mask is larger, use only what we need
                    new_mask = store_mask[:, :seq_len].bool()
                    logging.info(f"Using first {seq_len} values from larger mask")
                
                store_mask = new_mask
            
            # Ensure the mask is boolean
            store_mask = store_mask.bool()
        
        # Process through neural memory
        memory_output, _ = self.memory_module(normed_x, store_mask=store_mask)
        
        # Apply residual connection
        output = x + self.dropout(self.projection(memory_output))
        
        return output


class PersistentMemoryModule(nn.Module):
    """
    Module for persistent memory tokens that are learned during training
    and remain fixed at test time.
    """
    def __init__(self, dim=512, num_tokens=16):
        super().__init__()
        self.tokens = nn.Parameter(torch.randn(num_tokens, dim))
        self.proj = nn.Linear(dim, dim)
        
    def forward(self, x):
        # x shape: [batch_size, seq_len, dim]
        batch_size, seq_len, _ = x.shape
        
        # Expand persistent tokens to match batch size and sequence length
        expanded_tokens = self.tokens.unsqueeze(0).unsqueeze(1).expand(batch_size, seq_len, -1, -1)
        
        # Average across token dimension to get a fixed-size representation
        pooled_tokens = expanded_tokens.mean(dim=2)
        
        # Project to ensure proper dimensionality
        token_features = self.proj(pooled_tokens)
        
        return token_features


class FusionLayer(nn.Module):
    """
    Layer for fusing outputs from different branches (short-term, long-term memory, persistent memory).
    """
    def __init__(self, dim=512, method='concat-linear'):
        super().__init__()
        self.method = method
        
        if method == 'concat-linear':
            self.projection = nn.Linear(dim * 3, dim)
        elif method == 'gated':
            self.gate_short = nn.Linear(dim, dim)
            self.gate_long = nn.Linear(dim, dim)
            self.gate_persistent = nn.Linear(dim, dim)
            self.norm = nn.LayerNorm(dim)
        elif method == 'weighted':
            self.weights = nn.Parameter(torch.ones(3) / 3)
        
    def forward(self, short_term, long_term, persistent):
        # All inputs should have shape [batch_size, seq_len, dim]
        
        if self.method == 'concat-linear':
            # Concatenate along feature dimension and project back
            fused = torch.cat([short_term, long_term, persistent], dim=-1)
            fused = self.projection(fused)
        
        elif self.method == 'gated':
            # Apply gating mechanism to each branch
            gate_short = torch.sigmoid(self.gate_short(short_term))
            gate_long = torch.sigmoid(self.gate_long(long_term))
            gate_persistent = torch.sigmoid(self.gate_persistent(persistent))
            
            # Normalize gates to sum to 1
            gate_sum = gate_short + gate_long + gate_persistent
            gate_short = gate_short / gate_sum
            gate_long = gate_long / gate_sum
            gate_persistent = gate_persistent / gate_sum
            
            # Apply gates
            fused = gate_short * short_term + gate_long * long_term + gate_persistent * persistent
            fused = self.norm(fused)
            
        elif self.method == 'weighted':
            # Apply learned weights
            normalized_weights = F.softmax(self.weights, dim=0)
            fused = (normalized_weights[0] * short_term + 
                    normalized_weights[1] * long_term + 
                    normalized_weights[2] * persistent)
        
        else:
            # Default to simple addition
            fused = short_term + long_term + persistent
            
        return fused


class TITANsEncoder(nn.Module):
    """
    Encoder module that combines short-term processing, long-term memory,
    and persistent memory.
    """
    def __init__(
        self,
        dim=512,
        num_short_term_layers=6,
        window_size=64,
        num_heads=8,
        mlp_ratio=4,
        dropout=0.1,
        chunk_size=64,
        memory_heads=4,
        momentum=True,
        weight_decay=True,
        num_persistent_tokens=16,
        fusion_method='concat-linear'
    ):
        super().__init__()
        
        # Short-term processing branch with multiple layers
        self.short_term = nn.ModuleList([
            TitanShortTermBlock(
                dim=dim,
                window_size=window_size,
                num_heads=num_heads,
                mlp_ratio=mlp_ratio,
                dropout=dropout
            )
            for _ in range(num_short_term_layers)
        ])
        
        # Long-term memory branch using official TITANs implementation
        self.long_term_memory = NeuralMemoryBlock(
            dim=dim,
            chunk_size=chunk_size,
            heads=memory_heads,
            momentum=momentum,
            weight_decay=weight_decay,
            num_kv_per_token=2,
            dropout=dropout
        )
        
        # Persistent memory tokens
        self.persistent_memory = PersistentMemoryModule(
            dim=dim,
            num_tokens=num_persistent_tokens
        )
        
        # Fusion layer to combine branches
        self.fusion = FusionLayer(
            dim=dim,
            method=fusion_method
        )
        
    def forward(self, x, mask=None):
        # Process through short-term branch
        short_term_output = x
        for layer in self.short_term:
            short_term_output = layer(short_term_output)
        
        # Check mask dimensions for compatibility with the input
        batch_size, seq_len, _ = x.shape
        if mask is not None:
            if mask.shape[-1] != seq_len:
                logging.info(f"TITANsEncoder: Mask dimension {mask.shape[-1]} doesn't match sequence length {seq_len}")
                
                # If mask is provided but dimensions don't match, create a compatibility mask
                new_mask = torch.ones((batch_size, seq_len), dtype=torch.bool, device=x.device)
                
                # If the mask is smaller than seq_len, copy the available values
                if mask.shape[-1] < seq_len:
                    new_mask[:, :mask.shape[-1]] = mask.bool()
                    logging.info(f"Using first {mask.shape[-1]} values from smaller mask")
                else:
                    # If mask is larger, use only what we need
                    new_mask = mask[:, :seq_len].bool()
                    logging.info(f"Using first {seq_len} values from larger mask")
                
                mask = new_mask
            else:
                mask = mask.bool()
                
            if mask.dim() < 2:
                logging.warning(f"Mask has unexpected dimensions: {mask.shape}. Creating compatible mask.")
                mask = torch.ones((batch_size, seq_len), dtype=torch.bool, device=x.device)
        
        # Process through long-term memory branch
        long_term_output = self.long_term_memory(x, store_mask=mask)
        
        # Add persistent memory tokens and process
        persistent_output = self.persistent_memory(x)
        
        # Fuse outputs from all branches
        fused_output = self.fusion(short_term_output, long_term_output, persistent_output)
        
        return fused_output


class PredictionHead(nn.Module):
    """
    Prediction head for TITANsTS model supporting multiple tasks.
    """
    
    def __init__(
        self,
        dim,
        output_dim=None,
        dropout=0.1,
        task_type="reconstruction",
        n_channels=1,
    ):
        """
        Initialize prediction head.
        
        Args:
            dim (int): Input dimension (hidden state dimension)
            output_dim (int, optional): Output dimension. For classification tasks.
            dropout (float, optional): Dropout rate
            task_type (str, optional): Task type - "classification", "forecasting", "reconstruction", "anomaly"
            n_channels (int, optional): Number of channels in the input time series
        """
        super().__init__()
        self.dim = dim
        self.output_dim = output_dim
        self.dropout = dropout
        self.task_type = task_type
        self.n_channels = n_channels
        
        # Different head based on task type
        if task_type == "classification":
            self.output_dim = output_dim if output_dim is not None else 2
            self.head = nn.Sequential(
                nn.Linear(dim, dim // 2),
                nn.GELU(),
                nn.Dropout(dropout),
                nn.Linear(dim // 2, self.output_dim)
            )
        
        elif task_type == "forecasting":
            # For forecasting, output dimension depends on forecast horizon
            self.output_dim = output_dim if output_dim is not None else 1
            self.head = nn.Sequential(
                nn.Linear(dim, dim // 2),
                nn.GELU(),
                nn.Dropout(dropout),
                nn.Linear(dim // 2, self.output_dim)
            )
        
        elif task_type == "reconstruction":
            # For reconstruction, output should match input dimensions
            self.head = nn.Sequential(
                nn.Linear(dim, dim // 2),
                nn.GELU(),
                nn.Dropout(dropout),
                nn.Linear(dim // 2, n_channels)
            )
            
        elif task_type == "anomaly":
            # For anomaly detection, output is a binary value per patch
            self.head = nn.Sequential(
                nn.Linear(dim, dim // 2),
                nn.GELU(),
                nn.Dropout(dropout),
                nn.Linear(dim // 2, 1)
            )
        
        else:
            raise ValueError(f"Unknown task type: {task_type}")
    
    def forward(self, x, task_type=None):
        """
        Forward pass through prediction head.
        
        Args:
            x (torch.Tensor): Encoded hidden states [batch_size, seq_len, hidden_dim]
            task_type (str, optional): Task type to use (overrides the default)
            
        Returns:
            torch.Tensor: Predictions
        """
        # Use provided task_type if given, otherwise use default
        task = task_type if task_type is not None else self.task_type
        print(f"[PredictionHead] Input shape: {x.shape}, task: {task}")
        
        if task == "classification":
            # For classification, we use the CLS token (first token)
            # or average pooling over sequence length for class prediction
            if x.size(1) > 1:
                # Apply pooling over sequence length
                x = x.mean(dim=1)  # [batch_size, hidden_dim]
            
            # Apply head
            output = self.head(x)  # [batch_size, num_classes]
            print(f"[PredictionHead] Classification output: {output.shape}")
            
        elif task == "forecasting":
            # For forecasting, we use the last tokens to predict future values
            # Apply head to all tokens
            output = self.head(x)  # [batch_size, seq_len, forecast_length]
            print(f"[PredictionHead] Forecasting output: {output.shape}")
            
        elif task == "reconstruction":
            # For reconstruction, we output values for all patches
            # Apply head to all tokens
            output = self.head(x)  # [batch_size, seq_len, n_channels]
            print(f"[PredictionHead] Reconstruction output: {output.shape}")
            
        elif task == "anomaly":
            # For anomaly detection, we output an anomaly score for each position
            # Apply head to all tokens
            output = self.head(x).squeeze(-1)  # [batch_size, seq_len]
            print(f"[PredictionHead] Anomaly output: {output.shape}")
            
        else:
            raise ValueError(f"Unknown task type: {task}")
        
        return output


@dataclass
class TITANsTSModelOutput(ModelOutput):
    """
    Output class for TITANsTS model.
    
    Args:
        loss (torch.FloatTensor, optional): Loss value if training
        logits (torch.FloatTensor): Main output tensor
        hidden_states (tuple of torch.FloatTensor, optional): Hidden states from intermediate layers
        attention_weights (tuple of torch.FloatTensor, optional): Attention weights from attention layers
    """
    loss: Optional[torch.FloatTensor] = None
    logits: torch.FloatTensor = None
    hidden_states: Optional[Tuple[torch.FloatTensor]] = None
    attention_weights: Optional[Tuple[torch.FloatTensor]] = None


class TITANsTSModel(nn.Module):
    """
    TITANsTS Model: Transformer with Input Normalization for Time Series
    
    This model is designed for various time series tasks including:
    - Forecasting
    - Classification
    - Reconstruction
    - Anomaly detection
    
    It combines patching, normalization, and transformer-based encoding
    for effective time series representation.
    """
    
    def __init__(
        self,
        d_model=512,
        patch_size=16,
        stride=8,
        in_channels=1,
        num_layers=4,
        num_heads=8,
        dim_feedforward=2048,
        dropout=0.1,
        attention_dropout=0.1,
        output_dim=None,
        task_type="reconstruction",
        activation="gelu",
        norm_layer="layer_norm",
        head_dropout=0.1,
        **kwargs
    ):
        """
        Initialize TITANsTS model.
        
        Args:
            d_model: Hidden dimension of the model
            patch_size: Size of each patch
            stride: Stride between consecutive patches
            in_channels: Number of input channels
            num_layers: Number of transformer layers
            num_heads: Number of attention heads
            dim_feedforward: Dimension of feedforward network
            dropout: Dropout rate
            attention_dropout: Attention dropout rate
            output_dim: Output dimension for prediction head
            task_type: Task type - "classification", "forecasting", "reconstruction", "anomaly"
            activation: Activation function
            norm_layer: Normalization layer type
            head_dropout: Dropout rate for prediction head
        """
        super().__init__()
        self.d_model = d_model
        self.patch_size = patch_size
        self.stride = stride
        self.in_channels = in_channels
        self.num_layers = num_layers
        self.task_type = task_type
        
        print(f"[TITANsTS Model] Initializing with patch_size={patch_size}, stride={stride}, in_channels={in_channels}")
        
        # Input normalization
        self.normalizer = TimeSeriesRevIN(num_features=in_channels)
        
        # Patching module
        self.patching = TSAdaptivePatcher(
            patch_size=patch_size,
            stride=stride
        )
        
        # Patch embedding
        self.patch_embedding = nn.Linear(in_channels * patch_size, d_model)
        
        # Encoder configuration
        encoder_config = TransformerConfig(
            hidden_size=d_model,
            num_hidden_layers=num_layers,
            num_attention_heads=num_heads,
            intermediate_size=dim_feedforward,
            hidden_dropout_prob=dropout,
            attention_probs_dropout_prob=attention_dropout,
            hidden_act=activation,
            layer_norm_eps=1e-12,
        )
        
        # Encoder
        self.encoder = TransformerEncoder(encoder_config)
        
        # Prediction head
        self.head = PredictionHead(
            dim=d_model,
            output_dim=output_dim,
            dropout=head_dropout,
            task_type=task_type,
            n_channels=in_channels
        )
        
        # Initialize weights
        self.apply(self._init_weights)
        
    def _init_weights(self, module):
        """Initialize the weights."""
        if isinstance(module, nn.Linear):
            # Initialize linear layer weights with Xavier uniform
            nn.init.xavier_uniform_(module.weight)
            if module.bias is not None:
                nn.init.zeros_(module.bias)
                
    def forward(
        self,
        x: torch.Tensor = None,
        mask: torch.Tensor = None,
        task_type: str = None,
        labels: torch.Tensor = None,
        output_hidden_states: bool = False,
        return_dict: bool = True
    ) -> Union[Tuple, TITANsTSModelOutput]:
        """
        Forward pass of TITANsTS model for time series forecasting, reconstruction, classification, and anomaly detection.
        
        Args:
            x: Input time series of shape [batch_size, n_channels, seq_len]
            mask: Mask tensor for identifying valid (non-padding) positions
            task_type: Task type, one of "classification", "forecasting", "reconstruction", "anomaly"
            labels: Task labels for classification
            output_hidden_states: Whether to output hidden states
            return_dict: Whether to return outputs as dict
            
        Returns:
            TITANsTSModelOutput: Output with losses and logits
        """
        if task_type is None:
            task_type = self.task_type
            
        print(f"[TITANsTS] forward called with x shape: {x.shape}, mask shape: {None if mask is None else mask.shape}")
        print(f"[TITANsTS] task_type: {task_type}")
        
        # If y_mask is not provided, assume all positions are valid
        if mask is None:
            print("[TITANsTS] No mask provided, assuming all positions are valid")
            mask = torch.ones(x.shape[0], x.shape[2], device=x.device)
        
        # Normalize input
        x_in = self.normalizer(x)
        
        # Patch and embed the input
        patches, padding_mask = self.patching(x_in, mask)
        print(f"[TITANsTS] After patching: {patches.shape}, padding_mask: {None if padding_mask is None else padding_mask.shape}")
        
        # Embed patches
        hidden_states = self.patch_embedding(patches)
        print(f"[TITANsTS] After embedding: {hidden_states.shape}")
        
        # Encode with transformer
        encoder_outputs = self.encoder(
            hidden_states=hidden_states,
            attention_mask=padding_mask,
            output_hidden_states=output_hidden_states,
            return_dict=return_dict,
        )
        
        if return_dict:
            hidden_states = encoder_outputs.last_hidden_state
            output_hidden_states = encoder_outputs.hidden_states if output_hidden_states else None
        else:
            hidden_states = encoder_outputs[0]
            output_hidden_states = encoder_outputs[1] if output_hidden_states else None
        
        print(f"[TITANsTS] After encoder: {hidden_states.shape}")
        
        # Calculate loss and compute logits based on task type
        loss = None
        if task_type == "classification" and labels is not None:
            # Classification task
            logits = self.head(hidden_states, task_type)
            print(f"[TITANsTS] Classification logits: {logits.shape}")
            
            # Calculate loss if labels are provided
            loss_fct = torch.nn.CrossEntropyLoss()
            loss = loss_fct(logits.view(-1, self.head.output_dim), labels.view(-1))
            
        elif task_type == "forecasting":
            # Forecasting task
            logits = self.head(hidden_states, task_type)
            print(f"[TITANsTS] Forecasting logits: {logits.shape}")
            
            # Calculate loss if labels are provided
            if labels is not None:
                loss_fct = torch.nn.MSELoss()
                loss = loss_fct(logits, labels)
                
        elif task_type == "reconstruction":
            # Reconstruction task
            logits = self.head(hidden_states, task_type)
            print(f"[TITANsTS] Reconstruction logits before processing: {logits.shape}")
            
            # Ensure the output shape matches the input shape [batch_size, n_channels, seq_len]
            batch_size, n_patches, hidden_dim = hidden_states.shape
            
            # Check if the logits output is already [batch_size, n_channels, seq_len]
            if len(logits.shape) == 3 and logits.shape[1] == self.in_channels:
                # No need to do anything, already in the right format
                pass
            else:
                # Need to transpose from [batch_size, n_patches, n_channels] to [batch_size, n_channels, n_patches]
                logits = logits.permute(0, 2, 1)  # [batch_size, n_channels, n_patches]
            
            # Get the original sequence length from the input
            original_seq_len = x.shape[2]
            
            # Check if we need to resize to match original sequence length
            current_seq_len = logits.shape[2]
            print(f"[TITANsTS] Current length: {current_seq_len}, Original length: {original_seq_len}")
            
            if current_seq_len != original_seq_len:
                print(f"[TITANsTS] Resizing output from {logits.shape}")
                # Resize to match original sequence length
                logits = torch.nn.functional.interpolate(
                    logits, 
                    size=original_seq_len,
                    mode='linear', 
                    align_corners=False
                )
                print(f"[TITANsTS] Resized to {logits.shape}")
            
            # Calculate loss if input is provided
            if labels is not None:
                loss_fct = torch.nn.MSELoss()
                loss = loss_fct(logits, labels)
            
        elif task_type == "anomaly":
            # Anomaly detection task
            logits = self.head(hidden_states, task_type)
            print(f"[TITANsTS] Anomaly logits: {logits.shape}")
            
            # Calculate loss if labels are provided
            if labels is not None:
                loss_fct = torch.nn.BCEWithLogitsLoss()
                loss = loss_fct(logits.view(-1), labels.view(-1))
        
        else:
            raise ValueError(f"Unknown task type: {task_type}")
        
        print(f"[TITANsTS] Final logits shape: {logits.shape}")
        
        if not return_dict:
            output = (loss, logits)
            if output_hidden_states:
                output = output + (output_hidden_states,)
            return output
            
        return TITANsTSModelOutput(
            loss=loss,
            logits=logits,
            hidden_states=output_hidden_states if output_hidden_states else None,
            attentions=None,  # We don't store attention weights currently
        ) 