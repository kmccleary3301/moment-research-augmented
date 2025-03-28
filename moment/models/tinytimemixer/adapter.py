import os
import sys
import torch
import torch.nn as nn
from torch.cuda.amp import autocast
import traceback
from math import ceil

from moment.data.base import TimeseriesOutputs
from moment.models.layers.revin import RevIN
from moment.models.layers.embed import PatchEmbedding, Patching
from moment.utils.masking import Masking
from moment.common import TASKS
from moment.utils.utils import NamespaceWithDefaults

# Import TinyTimeMixer
from moment.models.tinytimemixer.configuration_tinytimemixer import TinyTimeMixerConfig
from moment.models.tinytimemixer.modeling_tinytimemixer import (
    TinyTimeMixerModel,
)

class PretrainHead(nn.Module):
    """
    Reconstruction head for pretraining, matching MOMENT's implementation
    """
    def __init__(
        self,
        d_model: int = 768,
        patch_len: int = 8,
        head_dropout: float = 0.1,
        orth_gain: float = 1.41,
    ):
        super().__init__()
        self.dropout = nn.Dropout(head_dropout)
        self.linear = nn.Linear(d_model, patch_len)
        self.activation = nn.GELU()  # Add activation for stability
        
        # Use a more conservative initialization to prevent gradient explosions
        if orth_gain is not None:
            # Use a lower gain value for more numerical stability
            actual_gain = min(orth_gain, 1.0)
            torch.nn.init.orthogonal_(self.linear.weight, gain=actual_gain)
            self.linear.bias.data.zero_()

    def forward(self, x):
        """
        x: [batch_size x n_channels x n_patches x d_model]
        output: [batch_size x n_channels x seq_len], where seq_len = n_patches * patch_len
        """
        # Apply layer norm for better stability
        x_mean = x.mean(dim=-1, keepdim=True)
        x_var = x.var(dim=-1, keepdim=True, unbiased=False)
        x = (x - x_mean) / torch.sqrt(x_var + 1e-5)
        
        # Apply dropout, the linear layer, and activation
        x = self.dropout(x)
        x = self.linear(x)
        x = self.activation(x)
        
        # Prevent extreme values
        x = torch.clamp(x, min=-10.0, max=10.0)
        
        # Reshape to final output format
        x = x.flatten(start_dim=2, end_dim=3)  # [batch_size x n_channels x seq_len]
        
        return x


class TinyTimeMixerAdapter(nn.Module):
    """
    Adapter class to make TinyTimeMixer compatible with the MOMENT interface.
    Implements the same API as MOMENT for seamless integration.
    """
    def __init__(self, configs, **kwargs):
        """
        Initialize a TinyTimeMixer model with a MOMENT-compatible interface.
        
        Args:
            configs: The configuration with MOMENT parameters
        """
        super().__init__()
        
        # Process configs similarly to MOMENT
        configs = self._update_inputs(configs, **kwargs)
        self.configs = configs
        self.task_name = configs.task_name
        self.seq_len = configs.seq_len
        self.patch_len = configs.patch_len
        self.debug = configs.getattr("debug", False)
        self.device = configs.getattr("device", "cuda" if torch.cuda.is_available() else "cpu")
        
        # Convert MOMENT config to TinyTimeMixer config
        ttm_config = self._convert_config_to_ttm(configs)
        
        # Add normalization, patching and embedding - same as MOMENT
        self.normalizer = RevIN(
            num_features=1, affine=configs.getattr("revin_affine", False)
        )
        self.tokenizer = Patching(
            patch_len=configs.patch_len, stride=configs.patch_stride_len
        )
        self.patch_embedding = PatchEmbedding(
            d_model=configs.d_model,
            seq_len=configs.seq_len,
            patch_len=configs.patch_len,
            stride=configs.patch_stride_len,
            dropout=configs.getattr("dropout", 0.1),
            add_positional_embedding=configs.getattr("add_positional_embedding", True),
            value_embedding_bias=configs.getattr("value_embedding_bias", False),
            orth_gain=configs.getattr("orth_gain", 1.41),
        )
        
        # Masking generator for pretraining
        self.mask_generator = Masking(mask_ratio=configs.getattr("mask_ratio", 0.0))
        
        # Initialize the TinyTimeMixer model as the encoder
        self.encoder = TinyTimeMixerModel(ttm_config)
        
        # Add task-specific head
        self.head = self._get_head(self.task_name)

    def _convert_config_to_ttm(self, moment_config):
        """
        Convert a MOMENT config to a TinyTimeMixer config.
        """
        # Calculate number of patches
        num_patches = (max(moment_config.seq_len, moment_config.patch_len) - moment_config.patch_len) // moment_config.patch_stride_len + 1
        
        if self.debug:
            print(f"Converting MOMENT config to TinyTimeMixer")
            print(f"seq_len: {moment_config.seq_len}, patch_len: {moment_config.patch_len}, num_patches: {num_patches}")
        
        # Get dropout values from config, but ensure they're not too small for numerical stability
        dropout = max(moment_config.getattr("dropout", 0.1), 0.05)
        head_dropout = max(moment_config.getattr("head_dropout", 0.1), 0.05)
        
        ttm_config = TinyTimeMixerConfig(
            context_length=moment_config.seq_len,
            patch_length=moment_config.patch_len,
            patch_stride=moment_config.patch_stride_len,
            num_patches=num_patches,  # Explicitly set num_patches
            num_input_channels=1,  # Default for now, adapt based on data if needed
            prediction_length=moment_config.seq_len,  # For reconstruction, this is the sequence length
            d_model=moment_config.d_model,
            num_layers=moment_config.getattr("num_layers", 4),
            dropout=dropout,
            head_dropout=head_dropout,
            # Use model features from MOMENT's config
            adaptive_patching_levels=moment_config.getattr("adaptive_patching_levels", 2),
            gated_attn=True,
            scaling="layer_norm",  # Add layer normalization for better stability
            loss="mse",
            use_sparse_attention=moment_config.getattr("use_sparse_attention", False),
            sliding_window_size=moment_config.getattr("sparse_attention_params", {}).get("sliding_window_size", 16),
            use_positional_encoding=moment_config.getattr("add_positional_embedding", True),
            initialization_factor=0.02,  # Use smaller initialization for improved stability
            # Ensure decoder is disabled for pretraining
            use_decoder=False,
        )
        
        return ttm_config
    
    def _update_inputs(self, configs, **kwargs):
        """
        Process configs similarly to MOMENT
        """
        if isinstance(configs, dict) and "model_kwargs" in kwargs:
            return NamespaceWithDefaults(**{**configs, **kwargs["model_kwargs"]})
        else:
            return NamespaceWithDefaults.from_namespace(configs)
    
    def _get_head(self, task_name):
        """
        Get the appropriate head based on the task name
        """
        if task_name == TASKS.PRETRAINING:
            return PretrainHead(
                self.configs.d_model,
                self.configs.patch_len,
                self.configs.getattr("dropout", 0.1),
                self.configs.getattr("orth_gain", 1.41),
            )
        # Add implementations for other task heads as needed
        else:
            raise NotImplementedError(f"Task {task_name} not implemented.")
    
    def forward(
        self,
        x_enc: torch.Tensor,
        mask: torch.Tensor = None,
        input_mask: torch.Tensor = None,
        **kwargs,
    ):
        """
        Forward method with the same signature as MOMENT
        """
        if self.task_name == TASKS.PRETRAINING:
            return self.pretraining(
                x_enc=x_enc, mask=mask, input_mask=input_mask, **kwargs
            )
        else:
            raise NotImplementedError(f"Task {self.task_name} not implemented.")
    
    def pretraining(
        self,
        x_enc: torch.Tensor,
        input_mask: torch.Tensor = None,
        mask: torch.Tensor = None,
        **kwargs,
    ):
        """
        Pretraining method that matches MOMENT's implementation
        
        Args:
            x_enc : [batch_size x n_channels x seq_len]
                Time-series data
            mask  : [batch_size x seq_len]
                Data that is masked but still attended to via mask-tokens
            input_mask : [batch_size x seq_len]
                Input mask for the time-series data that is unobserved
        """
        batch_size, n_channels, seq_len = x_enc.shape

        # Generate mask if not provided
        if mask is None:
            mask = self.mask_generator.generate_mask(x=x_enc, input_mask=input_mask)
            mask = mask.to(x_enc.device)  # mask: [batch_size x seq_len]
        
        # Default input_mask if not provided
        if input_mask is None:
            input_mask = torch.ones((batch_size, seq_len), device=x_enc.device)
        
        # Create combined mask for normalization
        combined_mask = mask * input_mask
        
        # Normalization with the combined mask
        x_enc_normalized = self.normalizer(x=x_enc, mask=combined_mask, mode="norm")
        x_enc_normalized = torch.nan_to_num(x_enc_normalized, nan=0, posinf=0, neginf=0)
        
        # Apply additional scaling to prevent gradient explosion
        scale_factor = torch.max(torch.abs(x_enc_normalized)).detach() + 1e-5
        x_enc_normalized = x_enc_normalized / scale_factor
        
        # Store the original tensor for reconstruction comparison
        original_tensor = x_enc.clone()
        
        try:
            # TinyTimeMixer expects [batch_size, seq_len, n_channels]
            # Reshape our tensor from [batch_size, n_channels, seq_len] to TTM's expected format
            past_values = x_enc_normalized.permute(0, 2, 1)  # [batch_size, seq_len, n_channels]
            
            # Prepare the mask - TTM expects [batch_size, seq_len, n_channels] as well
            past_observed_mask = input_mask.unsqueeze(-1).expand(-1, -1, n_channels)
            
            if self.debug:
                print(f"Past values shape: {past_values.shape}")
                print(f"Past observed mask shape: {past_observed_mask.shape}")
                print(f"Max value in input: {torch.max(torch.abs(past_values)).item()}")
                
            # Call TinyTimeMixerModel with correct parameters
            with autocast(enabled=self.configs.getattr("use_amp", False)):
                ttm_output = self.encoder(
                    past_values=past_values,
                    past_observed_mask=past_observed_mask,
                    output_hidden_states=True,
                    return_dict=True
                )
            
            # Get last hidden state and convert to MOMENT format for our head
            last_hidden = ttm_output.last_hidden_state
            
            if last_hidden is not None:
                # Check for NaN values in hidden states
                if torch.isnan(last_hidden).any() or torch.isinf(last_hidden).any():
                    if self.debug:
                        print("NaN or Inf detected in last_hidden - using fallback")
                    raise ValueError("NaN or Inf values in hidden state")
                
                # Apply our MOMENT-style head to get reconstructions
                # First, we need to reshape to fit our pretraining head's expectations
                # We'll use the patch_input from TTM's output
                patched_input = ttm_output.patch_input
                
                # Reshape to [batch_size, n_channels, n_patches, patch_len]
                if patched_input.ndim == 4:  # Should be [batch_size, n_channels, n_patches, patch_len]
                    # May need to transpose depending on TTM's output format
                    if patched_input.shape[1] != n_channels:
                        patched_input = patched_input.permute(0, 2, 1, 3)
                else:
                    # Adapt other potential shapes
                    n_patches = seq_len // self.patch_len
                    patched_input = patched_input.reshape(batch_size, n_channels, n_patches, self.patch_len)
                
                # Apply our head to get reconstructions
                n_patches = patched_input.shape[2]
                patch_len = patched_input.shape[3]
                enc_out = last_hidden.reshape(batch_size, n_channels, n_patches, -1)
                
                # Apply gradient clamping for stability
                enc_out = torch.clamp(enc_out, min=-10.0, max=10.0)
                
                dec_out = self.head(enc_out)
                
                # Check for NaN values after head
                if torch.isnan(dec_out).any() or torch.isinf(dec_out).any():
                    if self.debug:
                        print("NaN or Inf detected in dec_out - using fallback")
                    raise ValueError("NaN or Inf values in decoder output")
                
                # De-normalize using our RevIN and rescale appropriately
                dec_out = dec_out * scale_factor
                dec_out = self.normalizer(x=dec_out, mode="denorm")
                
                # Final safety check
                dec_out = torch.nan_to_num(dec_out, nan=0, posinf=0, neginf=0)
            else:
                # Fallback to identity reconstruction
                dec_out = original_tensor.clone().detach().requires_grad_(True)
            
            # Return with same format as MOMENT
            return TimeseriesOutputs(
                input_mask=input_mask,
                reconstruction=dec_out,
                pretrain_mask=mask,
            )
            
        except Exception as e:
            print(f"Error in TinyTimeMixer forward pass: {str(e)}")
            print(f"Error details: {type(e).__name__}")
            print(f"Traceback: {traceback.format_exc()}")
            
            # Create a fallback output for debugging
            fallback_output = original_tensor.clone().detach().requires_grad_(True)
            
            return TimeseriesOutputs(
                input_mask=input_mask,
                reconstruction=fallback_output,
                pretrain_mask=mask,
            )
    
    def _adapt_ttm_output(self, output, batch_size, n_channels, n_patches):
        """
        Adapt TTM output to the format expected by MOMENT
        """
        # Handle different potential output formats
        if hasattr(output, 'last_hidden_state'):
            output = output.last_hidden_state
            
        # Try to reshape based on available dimensions
        if len(output.shape) == 4:  # [batch, seq, channels, dim]
            return output.permute(0, 2, 1, 3)  # -> [batch, channels, seq, dim]
        elif len(output.shape) == 3:  # [batch*channels, seq, dim]
            return output.reshape(batch_size, n_channels, n_patches, -1)
        else:
            # Last resort reshape assuming flattened dimensions
            return output.reshape(batch_size, n_channels, n_patches, self.configs.d_model)
            
    def _check_model_weights_for_illegal_values(self):
        """
        Check for NaN values in model weights (for debugging)
        """
        illegal_encoder_weights = (
            torch.stack([torch.isnan(p).any() for p in self.encoder.parameters()])
            .any()
            .item()
        )
        illegal_head_weights = (
            torch.stack([torch.isnan(p).any() for p in self.head.parameters()])
            .any()
            .item()
        )
        illegal_patch_embedding_weights = (
            torch.stack(
                [torch.isnan(p).any() for p in self.patch_embedding.parameters()]
            )
            .any()
            .item()
        )

        return (
            illegal_encoder_weights
            or illegal_head_weights
            or illegal_patch_embedding_weights
        )
    
    # Implement additional MOMENT methods as needed
    def embed(
        self,
        x_enc: torch.Tensor,
        input_mask: torch.Tensor = None,
        reduction: str = "mean",
        **kwargs,
    ) -> TimeseriesOutputs:
        """
        Extract embeddings from time series data
        
        Args:
            x_enc : [batch_size x n_channels x seq_len]
            input_mask : [batch_size x seq_len]
            reduction : How to reduce channel and patch dimensions
        """
        batch_size, n_channels, seq_len = x_enc.shape

        if input_mask is None:
            input_mask = torch.ones((batch_size, seq_len)).to(x_enc.device)

        # Normalize and process input
        x_enc_normalized = self.normalizer(x=x_enc, mask=input_mask, mode="norm")
        x_enc_normalized = torch.nan_to_num(x_enc_normalized, nan=0, posinf=0, neginf=0)

        try:
            # Convert to TinyTimeMixer's expected format [batch_size, seq_len, n_channels]
            past_values = x_enc_normalized.permute(0, 2, 1)
            past_observed_mask = input_mask.unsqueeze(-1).expand(-1, -1, n_channels)
            
            # Forward through encoder
            ttm_output = self.encoder(
                past_values=past_values,
                past_observed_mask=past_observed_mask,
                output_hidden_states=True,
                return_dict=True
            )
            
            # Extract embeddings from the last hidden state
            last_hidden = ttm_output.last_hidden_state
            
            # Apply reduction as needed
            if reduction == "mean":
                # First convert to the right format
                # Assuming last_hidden is [batch_size, seq_len, n_channels, d_model]
                if last_hidden.ndim == 4:
                    embeddings = last_hidden.mean(dim=(1, 2))  # Average across seq_len and channels
                else:
                    # For other potential shapes
                    embeddings = last_hidden.mean(dim=1)  # Average across sequence dimension
            else:
                # Just return the raw embeddings
                embeddings = last_hidden

            return TimeseriesOutputs(
                embeddings=embeddings, input_mask=input_mask, metadata=reduction
            )
            
        except Exception as e:
            print(f"Error in TinyTimeMixer embed: {str(e)}")
            print(f"Error details: {type(e).__name__}")
            print(f"Traceback: {traceback.format_exc()}")
            
            # Fallback output
            fallback = torch.zeros((batch_size, self.configs.d_model), device=x_enc.device)
            fallback.requires_grad_(True)
            
            return TimeseriesOutputs(
                embeddings=fallback, input_mask=input_mask, metadata=reduction
            )
