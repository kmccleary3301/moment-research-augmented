import os
import sys
import torch
import logging
import argparse
import traceback
from typing import Optional, Union, Dict, Any

from moment.data.base import TimeseriesOutputs
from moment.models.layers.revin import RevIN
from moment.utils.masking import Masking

class SimpleSequentialAdapter(torch.nn.Module):
    """
    This gets good results on pretraining for some reason.
    """
    def __init__(
        self, 
        config: Optional[Union[argparse.Namespace, dict]] = None, 
        logger: Optional[logging.Logger] = None,
        **kwargs
    ):
        """
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
        self.debug = getattr(self.config, "debug", False)
        
        if self.debug and self.logger:
            self.logger.info(f"Using config: {self.config}")
        
        # Get model dimension
        self.d_model = getattr(self.config, "d_model", 256)
        
        # Get patch parameters
        self.patch_size = getattr(self.config, "patch_len", 16)
        self.stride = getattr(self.config, "patch_stride_len", 8)
        
        # Create a simple encoder model as substitute for full TITANsTS
        self.encoder = torch.nn.Sequential(
            torch.nn.Linear(self.patch_size, self.d_model),
            torch.nn.GELU(),
            torch.nn.Dropout(getattr(self.config, "dropout", 0.1)),
            torch.nn.Linear(self.d_model, self.d_model),
            torch.nn.GELU(),
            torch.nn.Dropout(getattr(self.config, "dropout", 0.1)),
        )
        
        # Create a decoder for reconstruction
        self.decoder = torch.nn.Sequential(
            torch.nn.Linear(self.d_model, self.d_model),
            torch.nn.GELU(),
            torch.nn.Dropout(getattr(self.config, "dropout", 0.1)),
            torch.nn.Linear(self.d_model, self.patch_size),
        )
        
        # Add a RevIN normalizer for compatibility with MOMENT
        self.normalizer = RevIN(num_features=1, affine=getattr(self.config, "revin_affine", False))
        
        # Add masking generator
        self.mask_generator = Masking(mask_ratio=getattr(self.config, "mask_ratio", 0.15))
        
        # Store task name for compatibility with MOMENT
        self.task_name = getattr(self.config, "task_name", "reconstruction")
        
        if self.debug and self.logger:
            self.logger.info(f"Created simplified TITANsTSAdapter with dimensions: {self.d_model}")
            self.logger.info(f"Patch size: {self.patch_size}, stride: {self.stride}")
        
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
            input_mask (torch.Tensor, optional): Input mask
            
        Returns:
            TimeseriesOutputs: Output with reconstruction results
        """
        batch_size, n_channels, seq_len = x_enc.shape
        
        if self.debug and self.logger:
            self.logger.info(f"Input shape: {x_enc.shape}")
        
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
        
        # Prevent too short time-series from causing NaNs
        x_normalized = torch.nan_to_num(x_normalized, nan=0, posinf=0, neginf=0)
        
        try:
            # Ensure input is properly shaped
            reconstructions = []
            
            # Process each channel separately
            for c in range(n_channels):
                # Get the current channel
                channel_data = x_normalized[:, c]  # [batch_size, seq_len]
                
                # Create patches
                patches = []
                for i in range(0, seq_len - self.patch_size + 1, self.stride):
                    patch = channel_data[:, i:i+self.patch_size]  # [batch_size, patch_size]
                    patches.append(patch)
                
                if not patches:  # Handle case where sequence is shorter than patch_size
                    # Use the entire sequence as one patch with padding if needed
                    patch = channel_data
                    if patch.shape[1] < self.patch_size:
                        padding = torch.zeros((batch_size, self.patch_size - patch.shape[1]), device=patch.device)
                        patch = torch.cat([patch, padding], dim=1)
                    patches.append(patch)
                
                # Stack patches
                stacked_patches = torch.stack(patches, dim=1)  # [batch_size, num_patches, patch_size]
                
                # Encode each patch
                encoded_patches = []
                for p in range(stacked_patches.shape[1]):
                    encoded = self.encoder(stacked_patches[:, p])  # [batch_size, d_model]
                    encoded_patches.append(encoded)
                
                encoded_sequence = torch.stack(encoded_patches, dim=1)  # [batch_size, num_patches, d_model]
                
                # Decode each patch
                decoded_patches = []
                for p in range(encoded_sequence.shape[1]):
                    decoded = self.decoder(encoded_sequence[:, p])  # [batch_size, patch_size]
                    decoded_patches.append(decoded)
                
                # Combine the patches back into a sequence
                # We'll use a simple overlap-add approach
                reconstructed = torch.zeros((batch_size, seq_len), device=channel_data.device)
                counts = torch.zeros((batch_size, seq_len), device=channel_data.device)
                
                for i, p in enumerate(range(0, seq_len - self.patch_size + 1, self.stride)):
                    if i < len(decoded_patches):
                        reconstructed[:, p:p+self.patch_size] += decoded_patches[i]
                        counts[:, p:p+self.patch_size] += 1
                
                # Average the overlapping regions
                counts[counts == 0] = 1  # Avoid division by zero
                reconstructed = reconstructed / counts
                
                # Add channel dimension back
                reconstructed = reconstructed.unsqueeze(1)  # [batch_size, 1, seq_len]
                reconstructions.append(reconstructed)
            
            # Combine all channels
            reconstruction = torch.cat(reconstructions, dim=1)  # [batch_size, n_channels, seq_len]
            
            # Denormalize
            reconstruction = self.normalizer(reconstruction, mask=combined_mask, mode="denorm")
            
            # Final sanity check
            if reconstruction.shape != x_enc.shape:
                if self.debug and self.logger:
                    self.logger.warning(f"Reconstruction shape {reconstruction.shape} doesn't match input shape {x_enc.shape}")
                # Resize to match if needed
                reconstruction = torch.nn.functional.interpolate(
                    reconstruction, 
                    size=seq_len,
                    mode='linear', 
                    align_corners=False
                )
            
            # Create the final output
            return TimeseriesOutputs(
                reconstruction=reconstruction,
                input_mask=input_mask,
                pretrain_mask=mask,
                metadata={}
            )
            
        except Exception as e:
            if self.logger:
                self.logger.error(f"Error in TITANsTSAdapter forward pass: {e}")
                self.logger.error(f"Input tensor shape: {x_enc.shape}")
                self.logger.error(f"Normalized tensor shape: {x_normalized.shape}")
                self.logger.error(traceback.format_exc())
            raise e

    def _default_config(self):
        """
        Create a default configuration for the model.
        """
        config = argparse.Namespace()
        config.d_model = 256
        config.num_layers = 6
        config.num_heads = 8
        config.mlp_ratio = 4
        config.dropout = 0.1
        config.head_dropout = 0.1
        config.seq_len = 512
        config.patch_len = 16
        config.patch_stride_len = 8
        config.task_name = 'reconstruction'
        config.memory_heads = 4
        config.mask_ratio = 0.15
        config.revin_affine = False
        config.debug = False
        return config

    def _dict_to_namespace(self, config_dict):
        """
        Convert a dictionary to a namespace.
        """
        config = argparse.Namespace()
        for key, value in config_dict.items():
            setattr(config, key, value)
        return config
        
    def getattr(self, name, default=None):
        """
        Get attribute from config with a default value.
        """
        return getattr(self.config, name, default)
