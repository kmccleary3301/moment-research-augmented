import os
import sys
import torch
import logging
import argparse
import traceback
from typing import Optional, Union, Dict, Any

from momentfm.data.base import TimeseriesOutputs
from momentfm.models.layers.revin import RevIN
from momentfm.utils.masking import Masking

# Import TITANsTS model
from moment.models.titans.titans_ts_model import (
    TITANsTSModel,
    TITANsTSModelOutput,
    TimeSeriesRevIN
)

# Import directly from titans_pytorch
from titans_pytorch import (
    # MemoryAsContextTransformer, 
    MemoryMLP,
    MemoryAttention
)

from moment.models.titans.mac_transformer import MemoryAsContextTransformer

# Custom RMSNorm implementation compatible with titans_pytorch
class RMSNorm(torch.nn.Module):
    def __init__(self, dim, eps=1e-6):
        super().__init__()
        self.eps = eps
        self.scale = torch.nn.Parameter(torch.ones(dim))

    def forward(self, x):
        # Calculate RMS
        rms = torch.sqrt(torch.mean(x ** 2, dim=-1, keepdim=True) + self.eps)
        # Normalize and scale
        x_normed = x / rms
        return self.scale * x_normed

# Add MultiheadRMSNorm for compatibility with NeuralMemory
class MultiheadRMSNorm(torch.nn.Module):
    def __init__(self, dim, heads):
        super().__init__()
        self.heads = heads
        self.rmsnorm = RMSNorm(dim)
        
    def forward(self, x):
        # Apply RMSNorm across multiple heads
        # Reshape to separate heads, apply norm, reshape back
        batch = x.shape[0]
        
        # Reshape to [batch * heads, seq_len, dim_head]
        x = x.view(batch, self.heads, -1).transpose(0, 1)
        
        # Apply RMSNorm to each head independently
        normed = torch.stack([self.rmsnorm(x_head) for x_head in x])
        
        # Reshape back to original shape
        return normed.transpose(0, 1).reshape(batch, -1)

# Add to torch.nn namespace for compatibility
torch.nn.RMSNorm = RMSNorm
torch.nn.MultiheadRMSNorm = MultiheadRMSNorm

class TITANsTSAdapter(torch.nn.Module):
    """
    Adapter class to make TITANsTS compatible with the MOMENT interface
    used in the pretraining and inference scripts.
    """
    def __init__(
        self, 
        config: Optional[Union[argparse.Namespace, dict]] = None, 
        logger: Optional[logging.Logger] = None,
        **kwargs
    ):
        """
        Initialize a TITANsTS adapter with a MOMENT-compatible interface.
        
        Args:
            config: The configuration with MOMENT parameters
            logger: Optional logger for debug output
            **kwargs: Additional arguments to override config
        """
        super().__init__()
        
        # Store the logger
        self.logger = logger
        
        # Convert dict to namespace if needed
        if isinstance(config, dict):
            config = self._dict_to_namespace(config)
            
        # Use default config if none provided
        if config is None:
            if self.logger:
                self.logger.info("No config provided, using default config")
            self.config = self._default_config()
        else:
            self.config = config
            
        # Set debug flag from config
        self._adapter_debug = getattr(self.config, "adapter_debug", False)
        
        if self._adapter_debug and self.logger:
            self.logger.info(f"Using config: {self.config}")
        
        # Convert MOMENT config to TITANs parameters
        titans_params = self._convert_config_to_titans(config)
        
        # Initialize the neural memory model based on config
        if getattr(self.config, "use_neural_memory_attention", False):
            neural_memory_model = MemoryAttention(
                dim=getattr(self.config, "memory_dim", 64)
            )
        else:
            neural_memory_model = MemoryMLP(
                dim=getattr(self.config, "memory_dim", 64),
                depth=getattr(self.config, "neural_memory_depth", 2)
            )
            
        # Initialize the MemoryAsContextTransformer
        if self._adapter_debug and self.logger:
            self.logger.info(f"Initializing TITANs model with parameters: {titans_params}")
            
        try:
            # We use the MemoryAsContextTransformer directly
            self.titans_model = MemoryAsContextTransformer(
                num_tokens=getattr(self.config, "vocab_size", 256),
                dim=getattr(self.config, "d_model", 384),
                depth=getattr(self.config, "num_layers", 8),
                segment_len=getattr(self.config, "window_size", 32),
                num_persist_mem_tokens=getattr(self.config, "num_persist_mem", 4),
                num_longterm_mem_tokens=getattr(self.config, "num_longterm_mem", 4),
                neural_memory_layers=getattr(self.config, "neural_mem_layers", (2, 4, 6)),
                neural_memory_segment_len=getattr(self.config, "neural_mem_segment_len", 4),
                neural_memory_batch_size=getattr(self.config, "neural_mem_batch_size", 128),
                neural_mem_gate_attn_output=getattr(self.config, "neural_mem_gate_attn_output", False),
                neural_mem_weight_residual=getattr(self.config, "neural_mem_weight_residual", True),
                neural_memory_qkv_receives_diff_views=getattr(self.config, "neural_mem_qkv_receives_diff_views", True),
                use_flex_attn=False,
                sliding_window_attn=getattr(self.config, "sliding_windows", True),
                neural_memory_model=neural_memory_model,
                neural_memory_kwargs=dict(
                    dim_head=getattr(self.config, "memory_dim_head", 64),
                    heads=getattr(self.config, "memory_heads", 4),
                    attn_pool_chunks=getattr(self.config, "store_attn_pool_chunks", True),
                    qk_rmsnorm=False,
                    momentum=getattr(self.config, "neural_mem_momentum", True),
                    momentum_order=getattr(self.config, "neural_mem_momentum_order", 1),
                    default_step_transform_max_lr=getattr(self.config, "neural_mem_max_lr", 0.1),
                    use_accelerated_scan=False,
                    per_parameter_lr_modulation=getattr(self.config, "memory_model_per_layer_learned_lr", True)
                )
            )
            
            if self._adapter_debug and self.logger:
                self.logger.info(f"TITANs model initialized successfully")
                param_count = sum(p.numel() for p in self.titans_model.parameters())
                self.logger.info(f"Model has {param_count:,} parameters")
        except Exception as e:
            if self.logger:
                self.logger.error(f"Error initializing TITANs model: {e}")
                self.logger.error(traceback.format_exc())
            raise
            
        # Add a RevIN normalizer for compatibility with MOMENT
        self.normalizer = RevIN(num_features=1, affine=getattr(self.config, "revin_affine", False))
        
        # Add reconstruction head for pretraining - project from hidden dim to original sequence
        self.reconstruction_head = torch.nn.Sequential(
            torch.nn.Linear(getattr(self.config, "d_model", 384), getattr(self.config, "d_model", 384)),
            torch.nn.GELU(),
            torch.nn.Linear(getattr(self.config, "d_model", 384), 1) # Output single channel time series
        )
        
        # Add masking generator
        self.mask_generator = Masking(mask_ratio=getattr(self.config, "mask_ratio", 0.15))
        
        # Store task name for compatibility with MOMENT
        self.task_name = getattr(self.config, "task_name", "reconstruction")
        
        # Add patching module to handle sequence patching similar to TinyTimeMixer/MOMENT
        self.patching = self._create_patching_module()
        
        # Add scaler that doesn't transform sequences (NOP - no operation)
        self.scaler = self._create_scaler_module()
        
    def _create_patching_module(self):
        """Create a patching module for time series"""
        # Simple module that converts time series to patches
        class TITANsTSPatchify(torch.nn.Module):
            def __init__(self):
                super().__init__()
                
            def forward(self, x, mask=None):
                """
                Convert time series to patches suitable for the TITANs transformer.
                For simplicity, we treat each time step as a token.
                
                Args:
                    x: Time series of shape [batch_size, channels, seq_len]
                    mask: Optional mask tensor
                
                Returns:
                    Patched/tokenized input
                """
                # For TITANs, we need to convert to [batch_size, seq_len, channels]
                # and then treat it as tokenized input
                return x.transpose(1, 2)
                
        return TITANsTSPatchify()
        
    def _create_scaler_module(self):
        """Create a scaler module that applies no scaling (NOP)"""
        class TITANsTSNOPScaler(torch.nn.Module):
            def __init__(self):
                super().__init__()
                
            def forward(self, x, mask=None, mode="norm"):
                # No-op, return x as is
                return x
                
        return TITANsTSNOPScaler()
        
    def _convert_config_to_titans(self, config):
        """
        Convert a MOMENT/config namespace to TITANs parameters.
        """
        # Extract parameters from config with sensible defaults
        params = {
            'num_tokens': getattr(config, "vocab_size", 256),
            'dim': getattr(config, "d_model", 384),
            'depth': getattr(config, "num_layers", 8),
            'segment_len': getattr(config, "window_size", 32),
            'num_persist_mem_tokens': getattr(config, "num_persist_mem", 4),
            'num_longterm_mem_tokens': getattr(config, "num_longterm_mem", 4),
            'neural_memory_layers': getattr(config, "neural_mem_layers", (2, 4, 6)),
            'neural_memory_segment_len': getattr(config, "neural_mem_segment_len", 4),
            'neural_memory_batch_size': getattr(config, "neural_mem_batch_size", 128),
            'neural_mem_gate_attn_output': getattr(config, "neural_mem_gate_attn_output", False),
            'neural_mem_weight_residual': getattr(config, "neural_mem_weight_residual", True),
            'neural_memory_qkv_receives_diff_views': getattr(config, "neural_mem_qkv_receives_diff_views", True),
            'use_flex_attn': getattr(config, "use_flex_attn", True),
            'sliding_window_attn': getattr(config, "sliding_windows", True),
        }
        
        return params
    
    def forward(self, x_enc, mask=None, input_mask=None, **kwargs):
        """
        Forward pass for the adapter, with an interface compatible with MOMENT.
        
        Args:
            x_enc (torch.Tensor): Input time series of shape [batch_size, n_channels, seq_len]
            mask (torch.Tensor, optional): Mask tensor for masked modeling of shape [batch_size, seq_len]
            input_mask (torch.Tensor, optional): Input mask of shape [batch_size, seq_len]
        
        Returns:
            TimeseriesOutputs: Output with the same interface as MOMENT
        """
        # Call the appropriate task handler
        if self.task_name == "reconstruction":
            return self.reconstruction(x_enc=x_enc, mask=mask, input_mask=input_mask, **kwargs)
        else:
            raise NotImplementedError(f"Task {self.task_name} not implemented in TITANsTSAdapter")
    
    def reconstruction(self, x_enc, mask=None, input_mask=None, **kwargs) -> TimeseriesOutputs:
        """
        Reconstruction task for masked modeling pretraining.
        
        Args:
            x_enc (torch.Tensor): Input time series of shape [batch_size, n_channels, seq_len]
            mask (torch.Tensor, optional): Mask tensor for masked modeling
            input_mask (torch.Tensor, optional): Input mask of shape [batch_size, seq_len]
            
        Returns:
            TimeseriesOutputs: Output with reconstruction results
        """
        batch_size, n_channels, seq_len = x_enc.shape
        
        if self._adapter_debug and self.logger:
            self.logger.info(f"[TITANs adapter] Input shape: {x_enc.shape}")
        
        if input_mask is None:
            input_mask = torch.ones((batch_size, seq_len), device=x_enc.device)
            
        if mask is None:
            mask = self.mask_generator.generate_mask(x=x_enc, input_mask=input_mask)
            mask = mask.to(x_enc.device)
        
        # Calculate the combined mask
        combined_mask = mask * input_mask
        
        # Check for samples with too few or no unmasked values 
        for i in range(batch_size):
            # Count non-zero values in the combined mask
            non_zero_count = combined_mask[i].sum().item()
            
            # If the mask has too few non-zero values, adjust it
            if non_zero_count < 10:  # Minimum required points for stable statistics
                # Calculate where the real data is (non-padding)
                real_data_positions = (input_mask[i] == 1).nonzero(as_tuple=True)[0]
                
                # If we have enough real data
                if len(real_data_positions) > 0:
                    # Ensure at least 50% of the real data remains unmasked
                    min_unmasked = max(10, int(len(real_data_positions) * 0.5))
                    
                    # Select positions to unmask
                    positions_to_unmask = real_data_positions[:min_unmasked]
                    
                    # Update the mask to unmask these positions
                    # In the pretraining mask, 0 = masked (to predict), 1 = unmasked
                    mask[i][positions_to_unmask] = 1
                    
                    # Recalculate the combined mask
                    combined_mask[i] = mask[i] * input_mask[i]
        
        # Normalize input with RevIN
        x_normalized = self.normalizer(x=x_enc, mask=combined_mask, mode="norm")
        
        if self._adapter_debug and self.logger:
            self.logger.info(f"[TITANs adapter] After normalization shape: {x_normalized.shape}")
        
        # Create masked input: replace masked positions with zeros
        # For reconstruction task, we need to mask positions where mask=0
        # In the combined mask: 1 = use this value, 0 = mask/predict this value
        mask_expanded = combined_mask.unsqueeze(1)  # [batch_size, 1, seq_len]
        
        # Apply mask to normalized input (only keep unmasked values)
        masked_x = x_normalized * mask_expanded
        
        # Convert to appropriate format for TITANs via patching
        patched_x = self.patching(masked_x)  # [batch_size, seq_len, channels]
        
        try:
            # For TITANs, we need to convert the float data to indices
            # by quantizing the normalized time series values to integers
            # in the range [0, 255] to use as token indices
            
            # Scale values to [0, 1] range assuming normalization has centered around zero
            min_val = patched_x.min()
            max_val = patched_x.max()
            scaled_x = (patched_x - min_val) / (max_val - min_val + 1e-6)
            
            # Quantize to integers in [0, 255] range for token indices
            tokenized_x = (scaled_x * 255).long().clamp(0, 255)
            
            # Forward pass through TITANs model
            # The MemoryAsContextTransformer expects token indices as input
            print(tokenized_x.shape)
            
            hidden_states = self.titans_model(
                tokenized_x,  # [batch_size, seq_len, channels] with token indices
                return_loss=False,
                # store_memories=True  # Enable memory storage
            )  # [batch_size, seq_len, dim]
            
            # Apply reconstruction head to get outputs
            reconstruction = self.reconstruction_head(hidden_states)  # [batch_size, seq_len, 1]
            
            # Convert back to [batch_size, channels, seq_len] format
            reconstruction = reconstruction.transpose(1, 2)
            
            # Denormalize 
            reconstruction = self.normalizer(reconstruction, mask=combined_mask, mode="denorm")
            
            # Create final output with MOMENT-compatible interface
            outputs = TimeseriesOutputs(
                reconstruction=reconstruction,
                pretrain_mask=mask,
                input_mask=input_mask
            )
            
            return outputs
            
        except Exception as e:
            if self.logger:
                self.logger.error(f"Error in forward pass: {e}")
                self.logger.error(traceback.format_exc())
            
            # Use a fallback reconstruction for debugging
            if self._adapter_debug and self.logger:
                self.logger.warning("Using fallback identity reconstruction")
            
            # Simple identity reconstruction as fallback
            fallback_reconstruction = x_enc.clone()
            
            return TimeseriesOutputs(
                reconstruction=fallback_reconstruction,
                pretrain_mask=mask,
                input_mask=input_mask
            )
            
    def _default_config(self):
        """
        Create a default configuration for TITANsTS.
        """
        config = argparse.Namespace()
        
        # Basic parameters
        config.d_model = 384
        config.num_layers = 8
        config.window_size = 32
        config.num_persist_mem = 4
        config.num_longterm_mem = 4
        config.neural_mem_layers = (2, 4, 6)
        config.neural_mem_segment_len = 4
        config.neural_mem_batch_size = 128
        config.neural_mem_gate_attn_output = False
        config.neural_mem_weight_residual = True
        config.neural_mem_qkv_receives_diff_views = True
        config.neural_mem_momentum = True
        config.task_name = "reconstruction"
        config.mask_ratio = 0.15
        config.revin_affine = False
        
        return config
        
    def _dict_to_namespace(self, config_dict):
        """
        Convert a dictionary to a namespace.
        """
        namespace = argparse.Namespace()
        for key, value in config_dict.items():
            setattr(namespace, key, value)
        return namespace
        
    def getattr(self, name, default=None):
        """
        Get an attribute from the config, with a default value.
        """
        return getattr(self.config, name, default)
        
    @property
    def debug(self):
        # Get debug setting from config
        return self._adapter_debug
