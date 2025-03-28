#!/usr/bin/env python
# -*- coding: utf-8 -*-

"""Test the MOMENT model with a simple inference task."""

import os
import sys
import math
import logging
import warnings
import argparse
from argparse import Namespace
from copy import deepcopy
from tqdm import tqdm
import random
import time
import datetime
import json
import traceback
from typing import Callable, Dict, Any, List, Optional, Union, Tuple

import torch
import numpy as np
from torch.utils.data import DataLoader, Subset, random_split
import torch.nn.functional as F

# Filter warnings about deprecated pytree functions
warnings.filterwarnings("ignore", message=".*_register_pytree_node.*", category=FutureWarning)
# Suppress gradient checkpointing warnings from transformers
warnings.filterwarnings("ignore", 
                      message=".*Passing `gradient_checkpointing` to a config initialization is deprecated.*",
                      category=UserWarning)

import os, sys
# Change to directory of the script, whether it's a notebook or a .py file
try:
    file_dir = globals()['_dh'][0]
except:
	file_dir = os.path.dirname(__file__)

try:
    # Try to import from local installed package
    from momentfm.models.moment import MOMENT, MOMENTPipeline
except ImportError:
    # Try to import from model repo directory
    sys.path.append(os.path.dirname(file_dir))
    from momentfm.models.moment import MOMENT, MOMENTPipeline

from momentfm.utils.utils import NamespaceWithDefaults
from momentfm.data.base import TimeseriesOutputs
from momentfm.models.layers.revin import RevIN
from momentfm.utils.masking import Masking
from momentfm.models.tinytimemixer.adapter import TinyTimeMixerAdapter
from momentfm.models.titans.adapter import TITANsTSAdapter
from momentfm.data.modified_dataset_loader_new import TimeSeriesPILEDataset, set_seed
from momentfm.data.timeseries_data_validator import is_valid_time_series, print_validation_stats

# Add the TinyTimeMixer repo to the path
sys.path.append('/home/kmccleary/network_project/network_paper/cloned_model_repos/granite-tsfm')

# Import TinyTimeMixer
from moment.models.tinytimemixer.configuration_tinytimemixer import TinyTimeMixerConfig
from momentfm.models.tinytimemixer.modeling_tinytimemixer import (
    TinyTimeMixerModel, 
    TinyTimeMixerForPrediction,
    TinyTimeMixerPreTrainedModel
)

# Import the modified dataset loader
from momentfm.data.modified_dataset_loader_new import TimeSeriesPILEDataset, set_seed
from momentfm.utils.safe_serialize import safe_serialize

# Import the MOMENT model
try:
    # Try to import from local installed package
    from momentfm.models.moment import MOMENT, MOMENTPipeline
    from momentfm.utils.utils import NamespaceWithDefaults
except ImportError:
    # Try to import from model repo directory
    sys.path.append(os.path.join(os.path.dirname(__file__), 'cloned_model_repos/moment'))
    from momentfm.models.moment import MOMENT, MOMENTPipeline
    from momentfm.utils.utils import NamespaceWithDefaults

# Import TITANsTS model
from momentfm.models.titans.titans_ts_model import (
    TITANsTSModel,
    TITANsTSModelOutput,
    TimeSeriesRevIN,
    TSAdaptivePatcher,
    PatchEmbedding
)

# Import the TITANs modules
from titans_pytorch import NeuralMemory
from titans_pytorch.mac_transformer import flex_attention, SegmentedAttention, MemoryAsContextTransformer
from argparse import Namespace
from momentfm.models.tinytimemixer.adapter import TinyTimeMixerAdapter
from momentfm.models.titans.adapter import TITANsTSAdapter

# Set up logging
logging.basicConfig(
    level=logging.INFO,
    format='%(asctime)s - %(name)s - %(levelname)s - %(message)s'
)
logger = logging.getLogger(__name__)



def custom_collate(batch, dataset=None, print_warnings=True):
    """Custom collation function for batching time series data.
    
    This function handles problematic samples that might have been missed during dataset filtering.
    """
    # Extract all items
    time_series = []
    masks = []
    idxs = []
    
    for item in batch:
        time_series.append(item['time_series'])
        masks.append(item['mask'])
        idxs.append(item['idx'])
    
    # Check for empty batch
    if len(time_series) == 0:
        return {
            'time_series': torch.tensor([]),
            'mask': torch.tensor([]),
            'idx': torch.tensor([])
        }
    
    # Determine sequence length from first item
    seq_len = time_series[0].shape[0]
    
    # Process and filter samples
    valid_time_series = []
    valid_masks = []
    valid_indices = []
    
    for i in range(len(time_series)):
        ts = time_series[i]
        
        # Skip if tensor is None
        if ts is None:
            continue
            
        # Convert to tensor if not already
        if not isinstance(ts, torch.Tensor):
            if isinstance(ts, np.ndarray):
                ts = torch.from_numpy(ts)
            else:
                ts = torch.tensor(ts, dtype=torch.float32)
        
        # Last-chance validation before training (for samples that might have been missed by filtering)
        # Only do this if we have access to dataset and sample index
        if dataset is not None and i < len(idxs):
            try:
                source_file = dataset.get_sample_source(idxs[i])
            except:
                source_file = f"Unknown source for sample {idxs[i]}"
                
            # Use the validator to check if this sample is valid
            if not is_valid_time_series(ts.cpu().numpy(), source_file, idxs[i]):
                if print_warnings:
                    logger.warning(f"Skipping problematic sample {idxs[i]} detected in collate function")
                continue
        
        # Skip invalid data
        if torch.isnan(ts).any() or torch.isinf(ts).any():
            if print_warnings:
                logger.warning(f"Skipping sample {idxs[i]} with NaN/Inf values")
            continue
            
        # Skip extreme values
        if torch.max(torch.abs(ts)) > 1e6:
            if print_warnings:
                logger.warning(f"Skipping sample {idxs[i]} with extreme values")
            continue
            
        # Resize tensor if needed
        if ts.shape[0] != seq_len:
            if ts.shape[0] > seq_len:
                ts = ts[:seq_len]
            else:
                padding = torch.zeros(seq_len - ts.shape[0], dtype=torch.float32)
                ts = torch.cat([ts, padding])
        
        # Process mask
        mask = masks[i]
        if not isinstance(mask, torch.Tensor):
            if isinstance(mask, np.ndarray):
                mask = torch.from_numpy(mask)
            else:
                mask = torch.tensor(mask, dtype=torch.float32)
            
        if mask.shape[0] != seq_len:
            if mask.shape[0] > seq_len:
                mask = mask[:seq_len]
            else:
                padding = torch.zeros(seq_len - mask.shape[0], dtype=torch.float32)
                mask = torch.cat([mask, padding])
        
        # Append valid data
        valid_time_series.append(ts)
        valid_masks.append(mask)
        valid_indices.append(idxs[i])
    
    # If no valid samples were found
    if len(valid_time_series) == 0:
        # Create a dummy batch with small random noise
        if print_warnings:
            logger.warning("No valid samples after processing - creating synthetic batch")
        valid_time_series = [torch.randn(seq_len) for _ in range(1)]
        valid_masks = [torch.ones(seq_len) for _ in range(1)]
        valid_indices = [0]
    
    # Stack tensors
    stacked_time_series = torch.stack(valid_time_series)
    stacked_masks = torch.stack(valid_masks)
    
    return {
        'time_series': stacked_time_series,
        'mask': stacked_masks,
        'idx': torch.tensor(valid_indices)
    }
    
def create_random_mask(x, mask_ratio=0.15):
    """Create a random mask for masked modeling pretraining."""
    batch_size, seq_len = x.shape
    
    # Determine number of tokens to mask
    num_masks = int(seq_len * mask_ratio)
    
    # Create masks for each sample in the batch
    mask = torch.zeros((batch_size, seq_len), device=x.device)
    
    for i in range(batch_size):
        # Randomly select indices to mask
        mask_indices = torch.randperm(seq_len)[:num_masks]
        mask[i, mask_indices] = 1.0
    
    return mask


def load_dataset(device):
    full_dataset = TimeSeriesPILEDataset(
        seq_len=512,
        seed=42,
        max_files=12,
        overlap_pct=0,
        filter_problematic=True
    )
    
    total_samples = len(full_dataset)
    train_size = int(0.8 * total_samples)
    val_size = int(0.1 * total_samples)
    test_size = total_samples - train_size - val_size

    _, val_dataset, _ = random_split(
        full_dataset, 
        [train_size, val_size, test_size],
        generator=torch.Generator().manual_seed(42)
    )
    
    data_loader = DataLoader(
        val_dataset,
        batch_size=20,
        shuffle=False,
        num_workers=4,
        collate_fn=lambda batch: custom_collate(
            batch, 
            val_dataset,
            True
        ),
        pin_memory=True if device == 'cuda' else False,
        drop_last=False
    )
    
    return data_loader


def load_models():
    # Load the MOMENT model
    config_namespace_moment = Namespace(**{
        'task_name': 'reconstruction',
        'transformer_type': 'encoder_only', 
        'transformer_backbone': 'default', 
        'randomly_initialize_backbone': True, 
        'freeze_embedder': False, 
        'freeze_encoder': False, 
        'freeze_head': False, 
        'add_positional_embedding': True, 
        'revin_affine': False, 
        'mask_ratio': 0.15, 
        'seq_len': 512, 
        'batch_size': 256, 
        'use_sparse_attention': False, 
        'd_model': 512, 
        'n_layers': 8, 
        'n_heads': 8, 
        'dim_feedforward': 2048, 
        'patch_len': 16, 
        'patch_stride_len': 16, 
        'dropout': 0.1, 
        'attention_dropout': 0.1, 
        'patch_dropout': 0.1, 
        'head_dropout': 0.1, 
        't5_config': {
            'model_type': 't5', 
            'architectures': ['T5ForConditionalGeneration'], 
            'd_model': 512, 
            'd_ff': 2048, 
            'num_layers': 8, 
            'num_heads': 8, 
            'relative_attention_num_buckets': 32, 
            'relative_attention_max_distance': 128, 
            'feed_forward_proj': 'relu', 
            'is_encoder_decoder': False, 
            'use_cache': False, 
            'decoder_start_token_id': 0, 
            'layer_norm_epsilon': 1e-06, 
            'initializer_factor': 1.0, 
            'tie_word_embeddings': False, 
            'pad_token_id': 0, 
            'eos_token_id': 1, 
            'attn_implementation': None
            }
        })
    
    model_config_moment = NamespaceWithDefaults.from_namespace(config_namespace_moment)
    
    # Initialize the model
    moment_model = MOMENT(model_config_moment)
    
    # Enable gradient checkpointing if supported
    try:
        if hasattr(moment_model, 'transformer_backbone') and hasattr(moment_model.transformer_backbone, 'gradient_checkpointing_enable'):
            moment_model.transformer_backbone.gradient_checkpointing_enable()
            logging.info("Enabled gradient checkpointing for MOMENT model.")
    except Exception as e:
        logging.warning(f"Failed to enable gradient checkpointing for MOMENT model: {e}")

    config_namespace_ttm = Namespace(**{
        "task_name": "reconstruction",
        "transformer_type": "encoder_only",
        "transformer_backbone": "default",
        "randomly_initialize_backbone": True,
        "freeze_embedder": False,
        "freeze_encoder": False,
        "freeze_head": False,
        "add_positional_embedding": True,
        "revin_affine": False,
        "mask_ratio": 0.15,
        "seq_len": 512,
        "batch_size": 256,
        "adaptive_patching_levels": 2,
        "gated_attn": True,
        "self_attn": False,
        "expansion_factor": 2,
        "mode": "common_channel",
        "use_sparse_attention": False,
        "d_model": 512,
        "num_layers": 8,
        "n_heads": 8,
        "dim_feedforward": 2048,
        "patch_len": 16,
        "patch_stride_len": 16,
        "dropout": 0.1,
        "attention_dropout": 0.1,
        "patch_dropout": 0.1,
        "head_dropout": 0.1,
        "t5_config": {
            "model_type": "t5",
            "vocab_size": 32128,
            "d_model": 512,
            "d_kv": 64,
            "d_ff": 2048,
            "num_layers": 8,
            "num_decoder_layers": 0,
            "num_heads": 8,
            "relative_attention_num_buckets": 32,
            "dropout_rate": 0.1,
            "layer_norm_epsilon": 1e-06,
            "initializer_factor": 1.0,
            "feed_forward_proj": "relu",
            "use_cache": True,
            "tie_word_embeddings": False,
            "pad_token_id": 0,
            "eos_token_id": 1
        }
    })
    
    model_config_ttm = NamespaceWithDefaults.from_namespace(config_namespace_ttm)
    
    # Initialize the TinyTimeMixer adapter model
    ttm_model = TinyTimeMixerAdapter(model_config_ttm)
    
    # Enable gradient checkpointing for TTM model if supported
    try:
        if hasattr(ttm_model, 'ttm_model') and hasattr(ttm_model.ttm_model, 'gradient_checkpointing_enable'):
            ttm_model.ttm_model.gradient_checkpointing_enable()
            logging.info("Enabled gradient checkpointing for TTM model.")
    except Exception as e:
        logging.warning(f"Failed to enable gradient checkpointing for TTM model: {e}")

    config_namespace_titans = Namespace(**{
        "adaptive_patching_levels": 2,
        "batch_size": 8,
        "checkpoint": None,
        "checkpoint_dir": "checkpoints",
        "d_model": 256,
        "debug": False,
        "debug_verbose": False,
        "dropout": 0.1,
        "early_stopping": 10,
        "epochs": 1,
        "expansion_factor": 2,
        "file_types": ".ts,.tsf,.csv,.json,.npy",
        "fusion_method": "concat-linear",
        "gated_attn": False,
        "grad_clip": 1.0,
        "half_precision": False,
        "head_dropout": 0.1,
        "keep_top_n": 3,
        "lr": 0.0001,
        "mask_ratio": 0.15,
        "mask_value": None,
        "max_files": 1,
        "memory_heads": 4,
        "mlp_ratio": 4,
        "mode": "common_channel",
        "num_heads": 8,
        "num_layers": 6,
        "num_persistent_tokens": 16,
        "num_short_term_layers": 4,
        "num_workers": 4,
        "output_dir": "output/titans",
        "patch_len": 16,
        "patch_stride_len": 8,
        "patience": 15,
        "run_id": "20250321_130419",
        "save_every": 10,
        "seed": 42,
        "self_attn": False,
        "seq_len": 512,
        "subsample_factor": 1.0,
        "task_name": "reconstruction",
        "use_momentum": False,
        "use_random_mask": False,
        "use_weight_decay": False,
        "val_batch_cap": 100,
        "val_fraction": 0.025,
        "val_freq": 1,
        "weight_decay": 0.01,
        "window_size": 64
    })

    model_config_titans = NamespaceWithDefaults.from_namespace(config_namespace_titans)
    
    # Initialize the TITANsTS adapter model
    titans_model = TITANsTSAdapter(model_config_titans)

    return moment_model, ttm_model, titans_model


def evaluate_model(model, dataset_loader, device):
    model.eval()
    with torch.no_grad():
        for batch in tqdm(dataset_loader, desc="Evaluating model"):
            # Move data to device
            val_time_series = batch['time_series'].to(device)
            val_mask = batch['mask'].to(device)
            
            # Add channel dimension if needed for MOMENT
            if len(val_time_series.shape) == 2:
                val_time_series = val_time_series.unsqueeze(1)
            
            # Create masking for validation
            val_mask_indices = create_random_mask(val_time_series.squeeze(1), 0.14)
            
            # Forward pass for validation
            val_outputs = model(x_enc=val_time_series, mask=val_mask_indices, input_mask=val_mask)
    print("Output shape: ", val_outputs.reconstruction.shape)


def main():
    moment_model, ttm_model, titans_model = load_models()
    
    device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
    moment_model.to(device)
    ttm_model.to(device)
    titans_model.to(device)
    
    print("TITANs model:", titans_model)
    
    dataset_loader = load_dataset(device)
    # Test the MOMENT model
    evaluate_model(moment_model, dataset_loader, device)
    print("Moment model done")
    evaluate_model(ttm_model, dataset_loader, device)
    print("TTM model done")
    
    
    
    evaluate_model(titans_model, dataset_loader, device)
    print("TITANs model done")
    

if __name__ == "__main__":
    main()


