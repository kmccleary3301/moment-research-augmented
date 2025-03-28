#!/usr/bin/env python
# Pretraining script for TinyTimeMixer model on the TimeSeriesPILE dataset

import os
import sys
import argparse
import logging
import numpy as np
import torch
import random
import time
from typing import Callable, Dict, Any, List, Optional, Union, Tuple
from torch.utils.data import DataLoader, Subset, random_split
from torch.optim import AdamW
from torch.optim.lr_scheduler import CosineAnnealingLR, ReduceLROnPlateau
from torch.cuda.amp import autocast, GradScaler
import torch.nn.functional as F
import datetime
import json
from tqdm import tqdm
import matplotlib.pyplot as plt
from itertools import cycle
from momentfm.data.timeseries_data_validator import is_valid_time_series, print_validation_stats
import traceback

# Import the MOMENT model for its TimeseriesOutputs class and other utilities
try:
    # Try to import from local installed package
    from momentfm.models.moment import MOMENT, MOMENTPipeline
    from momentfm.utils.utils import NamespaceWithDefaults
    from momentfm.data.base import TimeseriesOutputs
    from momentfm.models.layers.revin import RevIN
    from momentfm.utils.masking import Masking
except ImportError:
    # Try to import from model repo directory
    sys.path.append(os.path.join(os.path.dirname(__file__), 'cloned_model_repos/moment'))
    from momentfm.models.moment import MOMENT, MOMENTPipeline
    from momentfm.utils.utils import NamespaceWithDefaults
    from momentfm.data.base import TimeseriesOutputs
    from momentfm.models.layers.revin import RevIN
    from momentfm.utils.masking import Masking

# Add the TinyTimeMixer repo to the path
sys.path.append('/home/kmccleary/network_project/network_paper/cloned_model_repos/granite-tsfm')

# Import TinyTimeMixer
from momentfm.models.tinytimemixer.configuration_tinytimemixer import TinyTimeMixerConfig
from momentfm.models.tinytimemixer.modeling_tinytimemixer import (
    TinyTimeMixerModel, 
    TinyTimeMixerForPrediction,
    TinyTimeMixerPreTrainedModel
)

# Import the modified dataset loader
from momentfm.data.modified_dataset_loader_new import (
    TimeSeriesPILEDataset, 
    set_seed, 
    sha256_of_tensor,
    sha256_of_string
)
from momentfm.utils.safe_serialize import safe_serialize
from momentfm.models.tinytimemixer.adapter import TinyTimeMixerAdapter

# Change to directory of the script, whether it's a notebook or a .py file
try:
    os.chdir(globals()['_dh'][0])
except:
	os.chdir(os.path.dirname(__file__))

# Set up logging
logging.basicConfig(
    level=logging.INFO,
    format='%(asctime)s - %(name)s - %(levelname)s - %(message)s'
)
logger = logging.getLogger(__name__)

# Define the dataset cache directory
DATASET_CACHE_DIR = "/mnt/drive_2/timeseries_pile_data"

# Make sure the cache directory exists
os.makedirs(DATASET_CACHE_DIR, exist_ok=True)

# Helper functions for the training process
def setup_logging(args):
    """Set up logging configuration based on args."""
    log_level = logging.DEBUG if args.debug_verbose else logging.INFO
    logger.setLevel(log_level)
    
    # Create output directory if it doesn't exist
    os.makedirs(args.output_dir, exist_ok=True)
    
    # Create logs directory
    logs_dir = os.path.join(args.output_dir, 'logs')
    os.makedirs(logs_dir, exist_ok=True)
    
    # Add file handler if needed
    log_file = os.path.join(logs_dir, f"train_{time.strftime('%Y%m%d_%H%M%S')}.log")
    file_handler = logging.FileHandler(log_file)
    file_handler.setFormatter(logging.Formatter('%(asctime)s - %(name)s - %(levelname)s - %(message)s'))
    logger.addHandler(file_handler)
    
    logger.info(f"Logging to {log_file}")
    return log_file

def setup_device(args):
    """Set up and return the device for training."""
    if args.device == 'cuda' and not torch.cuda.is_available():
        logger.warning("CUDA is not available, falling back to CPU")
        device = torch.device('cpu')
    else:
        device = torch.device(args.device)
    
    logger.info(f"Using device: {device}")
    
    if device.type == 'cuda':
        logger.info(f"CUDA Device: {torch.cuda.get_device_name(0)}")
        logger.info(f"CUDA Memory: {torch.cuda.get_device_properties(0).total_memory / 1e9:.2f} GB")
    
    return device

def seed_everything(seed):
    """Set random seeds for reproducibility."""
    random.seed(seed)
    np.random.seed(seed)
    torch.manual_seed(seed)
    if torch.cuda.is_available():
        torch.cuda.manual_seed_all(seed)
    logger.info(f"Random seed set to {seed}")

class TokenCounter:
    """Class to track token counts during training."""
    def __init__(self):
        self.tokens = 0
        self.masked_tokens = 0
    
    def update(self, batch_size, seq_len, mask=None):
        """Update token counts."""
        tokens_this_batch = batch_size * seq_len
        masked_tokens_this_batch = tokens_this_batch * 0.15  # Default mask ratio
        if mask is not None:
            masked_tokens_this_batch = mask.sum().item()
            
        self.tokens += tokens_this_batch
        self.masked_tokens += masked_tokens_this_batch
    
    def get_tokens(self):
        """Get total token count."""
        return self.tokens
    
    def get_masked_tokens(self):
        """Get masked token count."""
        return self.masked_tokens

def create_model(args, config):
    """Create and initialize the TinyTimeMixer model with MOMENT compatibility layer."""
    try:
        # Create a Namespace from the config dictionary
        from argparse import Namespace
        config_namespace = Namespace(**config)
        
        print("ttm model config:", json.dumps(config, indent=4))
        
        # Convert to NamespaceWithDefaults
        model_config = NamespaceWithDefaults.from_namespace(config_namespace)
        
        # Add debug flag if present in args
        if hasattr(args, 'debug'):
            model_config.debug = args.debug
            if args.debug:
                logger.info("Debug mode enabled for model")
        
        # Initialize the TinyTimeMixer adapter model
        model = TinyTimeMixerAdapter(model_config)
        logger.info(f"Created TinyTimeMixer model with {sum(p.numel() for p in model.parameters()):,} parameters")
        return model
    except Exception as e:
        logger.error(f"Error creating model: {e}")
        raise

def create_optimizer(model, args):
    """Create and initialize the optimizer."""
    return AdamW(
        model.parameters(),
        lr=args.learning_rate,
        weight_decay=args.weight_decay
    )

def create_scheduler(optimizer, args, steps_per_epoch):
    """Create and initialize the learning rate scheduler."""
    total_steps = args.epochs * steps_per_epoch
    return torch.optim.lr_scheduler.OneCycleLR(
        optimizer,
        max_lr=args.learning_rate,
        total_steps=total_steps,
        pct_start=0.3,  # Percentage of steps for the LR to increase
        div_factor=25,  # Initial LR is max_lr/div_factor
        final_div_factor=10000,  # Final LR is max_lr/(div_factor*final_div_factor)
        anneal_strategy='cos'  # Use cosine annealing
    )
    

def save_checkpoint(model, optimizer, scheduler, epoch, train_loss, val_loss, 
                   path, keep_top_n=3, scaler=None, token_counter=None):
    """Save a model checkpoint."""
    # Ensure directory exists
    os.makedirs(os.path.dirname(path), exist_ok=True)
    
    # Create checkpoint dictionary
    checkpoint = {
        'epoch': epoch,
        'model': model.state_dict(),
        'optimizer': optimizer.state_dict(),
        'train_loss': train_loss,
        'val_loss': val_loss,
        'config': model.config if hasattr(model, 'config') else None,
    }
    
    # Add optional components if available
    if scheduler is not None:
        checkpoint['scheduler'] = scheduler.state_dict()
    
    if scaler is not None:
        checkpoint['scaler'] = scaler.state_dict()
    
    if token_counter is not None:
        checkpoint['tokens'] = token_counter.get_tokens()
        checkpoint['masked_tokens'] = token_counter.get_masked_tokens()
    
    # Save the checkpoint
    torch.save(checkpoint, path)
    logger.info(f"Saved checkpoint to {path}")
    
    # Manage checkpoint history if keeping top N
    if keep_top_n > 0:
        dir_path = os.path.dirname(path)
        base_filename = os.path.basename(path).split('_')[0]
        checkpoints = []
        
        # Find all relevant checkpoints
        for filename in os.listdir(dir_path):
            if filename.startswith(base_filename) and filename.endswith('.pt') and 'epoch' in filename:
                try:
                    checkpoint_epoch = int(filename.split('_')[-1].split('.')[0])
                    checkpoint_path = os.path.join(dir_path, filename)
                    checkpoint_data = torch.load(checkpoint_path, map_location='cpu')
                    checkpoints.append((checkpoint_path, checkpoint_data['val_loss'], checkpoint_epoch))
                except (ValueError, KeyError, FileNotFoundError):
                    continue
        
        # Sort by validation loss (ascending)
        checkpoints.sort(key=lambda x: x[1])
        
        # Keep only the top N checkpoints by validation loss
        to_keep = checkpoints[:keep_top_n]
        to_keep_paths = {x[0] for x in to_keep}
        
        # Delete others
        for checkpoint_path, _, _ in checkpoints:
            if checkpoint_path not in to_keep_paths and checkpoint_path != path:
                try:
                    os.remove(checkpoint_path)
                    logger.info(f"Removed checkpoint {checkpoint_path} to keep top {keep_top_n}")
                except OSError as e:
                    logger.warning(f"Error removing checkpoint {checkpoint_path}: {e}")

def parse_args():
    """Parse command-line arguments."""
    parser = argparse.ArgumentParser(description="Pretrain TinyTimeMixer on the TimeSeriesPILE dataset")
    
    # Model size options
    parser.add_argument('--model_size', type=str, default='base', choices=['small', 'base', 'large'],
                        help='Model size to use (default: base)')
    
    # TinyTimeMixer specific parameters
    parser.add_argument('--adaptive_patching_levels', type=int, default=2, 
                        help='Number of adaptive patching levels for TinyTimeMixer')
    parser.add_argument('--gated_attn', action='store_true', default=True,
                        help='Enable gated attention in TinyTimeMixer')
    parser.add_argument('--self_attn', action='store_true', default=False, 
                        help='Enable self attention across patches in TinyTimeMixer')
    parser.add_argument('--expansion_factor', type=int, default=2,
                        help='Expansion factor to use inside MLP in TinyTimeMixer')
    parser.add_argument('--mode', type=str, default='common_channel', choices=['common_channel', 'mix_channel'],
                        help='Mixer mode in TinyTimeMixer')
    
    # Debug parameters
    parser.add_argument('--debug', action='store_true', help='Run in debug mode with limited data')
    parser.add_argument('--debug_verbose', action='store_true', help='Print verbose debug information')
    parser.add_argument('--debug_max_files', type=int, default=None, help='Maximum number of files to use in debug mode')
    
    # Dataset parameters
    parser.add_argument('--dataset_dir', type=str, default=DATASET_CACHE_DIR,
                        help='Path to the directory containing the dataset')
    parser.add_argument('--max_files', type=int, default=None, help='Maximum number of files to use')
    parser.add_argument('--file_types', type=str, default=".ts,.tsf,.csv,.json,.npy",
                        help='Comma-separated list of file extensions to load')
    parser.add_argument('--seq_len', type=int, default=512, help='Sequence length to use')
    parser.add_argument('--use_unfiltered_data', action='store_true', 
                        help='Use the original unfiltered dataset even if a filtered version exists')
    
    # Masking parameters
    parser.add_argument('--mask_ratio', type=float, default=0.15, 
                        help='Ratio of tokens to mask during pretraining (default: 0.15)')
    
    # Training parameters
    parser.add_argument('--batch_size', type=int, default=256, help='Batch size to use')
    parser.add_argument('--epochs', type=int, default=10, help='Number of epochs to train for')
    parser.add_argument('--learning_rate', type=float, default=5e-4, help='Learning rate to use')
    parser.add_argument('--min_learning_rate', type=float, default=1e-5, help='Minimum learning rate for scheduler')
    parser.add_argument('--weight_decay', type=float, default=0.01, help='Weight decay to use')
    parser.add_argument('--warmup_steps', type=int, default=1000, help='Number of warmup steps')
    parser.add_argument('--gradient_clip', type=float, default=1.0, help='Gradient clipping value')
    parser.add_argument('--early_stopping', type=float, default=-1, 
                        help='Early stopping threshold for validation loss (negative value to disable)')
    
    # Logging parameters
    parser.add_argument('--check_nan_frequency', type=int, default=100, 
                        help='Check for NaN weights every N batches')
    parser.add_argument('--log_token_counts', action='store_true', 
                        help='Log token counts during training')
    parser.add_argument('--log_interval', type=int, default=20, 
                        help='Log metrics every N batches')
    parser.add_argument('--save_interval', type=int, default=500, 
                        help='Save checkpoints every N batches')
    
    
    
    # Output parameters
    parser.add_argument('--load_comparison', type=str, default=None, 
                        help='Directory of model to load for comparison')
    parser.add_argument('--output_dir', type=str, default='output/ttm', help='Output directory for checkpoints and logs')
    parser.add_argument('--save_every', type=int, default=1, help='Save model every N epochs (0 to disable)')
    parser.add_argument('--keep_top_n', type=int, default=3, 
                        help='Number of top checkpoints to keep (0 to keep all)')
    parser.add_argument('--resume_checkpoint', type=str, default=None, 
                        help='Path to checkpoint to resume training from')
    
    # Hardware parameters
    parser.add_argument('--device', type=str, default='cuda' if torch.cuda.is_available() else 'cpu',
                        help='Device to use (cuda or cpu)')
    parser.add_argument('--seed', type=int, default=42, help='Random seed for reproducibility')
    parser.add_argument('--num_workers', type=int, default=4, help='Number of dataloader workers')
    parser.add_argument('--mixed_precision', action='store_true', help='Use mixed precision training')
    parser.add_argument('--persistent_workers', action='store_true', 
                        help='Keep workers alive between epochs')
    parser.add_argument('--model_label', type=str, default=None, 
                        help='Label for the model for usage in graphs')
    
    # Sparse Attention parameters
    parser.add_argument('--use_sparse_attention', action='store_true', 
                        help='Use sparse attention mechanism for transformer')
    parser.add_argument('--sliding_window_size', type=int, default=16, 
                        help='Size of sliding window for sparse attention (if enabled)')
    
    args = parser.parse_args()
    
    if args.model_label is None:
        args.model_label = "TTM"
    
    # Apply debug settings if requested
    if args.debug:
        args.batch_size = min(args.batch_size, 16)
        args.debug_max_files = args.debug_max_files or 5
        args.epochs = min(args.epochs, 3)
    
    # Map mixed_precision to use_amp for compatibility with existing code
    args.use_amp = args.mixed_precision
    
    return args

def get_model_config(args):
    """Get model configuration based on model size."""
    # Base configuration values that the model expects
    config = {
        'task_name': 'reconstruction',
        'transformer_type': 'encoder_only',
        'transformer_backbone': 'default',
        'randomly_initialize_backbone': True,
        'freeze_embedder': False,
        'freeze_encoder': False,
        'freeze_head': False,
        'add_positional_embedding': True,
        'revin_affine': False,
        'mask_ratio': args.mask_ratio,
        'seq_len': args.seq_len,
        'batch_size': args.batch_size,
        
        # TinyTimeMixer specific parameters
        'adaptive_patching_levels': args.adaptive_patching_levels,
        'gated_attn': args.gated_attn,
        'self_attn': args.self_attn,
        'expansion_factor': args.expansion_factor,
        'mode': args.mode,
        # Add sparse attention parameter
        'use_sparse_attention': args.use_sparse_attention,
    }
    
    # If sparse attention is enabled, add parameters
    if args.use_sparse_attention:
        sparse_attention_params = {
            'sliding_window_size': args.sliding_window_size,
        }
        config['sparse_attention_params'] = sparse_attention_params
    
    # Add size-specific configuration
    if args.model_size == 'small':
        d_model = 128
        n_layers = 4
        n_heads = 4
        dim_feedforward = 256
        config.update({
            'd_model': d_model,
            'num_layers': n_layers,  # Changed from n_layers to num_layers for TTM
            'n_heads': n_heads,
            'dim_feedforward': dim_feedforward,
            'patch_len': 16,
            'patch_stride_len': 16,  # Set equal to patch_len to avoid warnings
            'dropout': 0.1,
            'attention_dropout': 0.1,
            'patch_dropout': 0.1,
            'head_dropout': 0.1,
        })
    elif args.model_size == 'base':
        d_model = 512
        n_layers = 8
        n_heads = 8
        dim_feedforward = 2048
        config.update({
            'd_model': d_model,
            'num_layers': n_layers,  # Changed from n_layers to num_layers for TTM
            'n_heads': n_heads,
            'dim_feedforward': dim_feedforward,
            'patch_len': 16,
            'patch_stride_len': 16,  # Set equal to patch_len to avoid warnings
            'dropout': 0.1,
            'attention_dropout': 0.1,
            'patch_dropout': 0.1,
            'head_dropout': 0.1,
        })
    elif args.model_size == 'large':
        d_model = 1024
        n_layers = 12
        n_heads = 16
        dim_feedforward = 4096
        config.update({
            'd_model': d_model,
            'num_layers': n_layers,  # Changed from n_layers to num_layers for TTM
            'n_heads': n_heads,
            'dim_feedforward': dim_feedforward,
            'patch_len': 16,
            'patch_stride_len': 16,  # Set equal to patch_len to avoid warnings
            'dropout': 0.1,
            'attention_dropout': 0.1,
            'patch_dropout': 0.1,
            'head_dropout': 0.1,
        })
    
    # Add T5 configuration
    t5_config = {
        'model_type': 't5',
        'vocab_size': 32128,
        'd_model': d_model,
        'd_kv': d_model // n_heads,
        'd_ff': dim_feedforward,
        'num_layers': n_layers,
        'num_decoder_layers': 0,  # Encoder-only
        'num_heads': n_heads,
        'relative_attention_num_buckets': 32,
        'dropout_rate': 0.1,
        'layer_norm_epsilon': 1e-6,
        'initializer_factor': 1.0,
        'feed_forward_proj': 'relu',
        'use_cache': True,
        'tie_word_embeddings': False,
        'pad_token_id': 0,
        'eos_token_id': 1,
        'gradient_checkpointing': True,
    }
    
    config['t5_config'] = t5_config
    
    return config

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

def train_epoch(model, data_loader, optimizer, scheduler, device, epoch, args, metrics_file, scaler=None, token_counter=None, continuous_train_losses=None, continuous_train_tokens=None, 
             val_loader=None, continuous_val_losses=None, continuous_val_tokens=None, val_samples_seen=None):
    """Train the model for one epoch."""
    model.train()
    total_loss = 0
    start_time = time.time()
    
    # Token tracking
    tokens_this_epoch = 0
    masked_tokens_this_epoch = 0
    
    # Create a cyclic iterator over the validation set if needed
    val_iter = None
    if val_loader is not None and continuous_val_losses is not None:
        val_iter = cycle(val_loader)
        
    # Create progress bar
    pbar = tqdm(enumerate(data_loader), total=len(data_loader))
    
    # Track NaN occurrences
    nan_batches = 0
    valid_batches = 0
    
    # Track data statistics for debugging
    all_means = []
    all_stds = []
    all_mins = []
    all_maxs = []
    lowest_valid_length = 99999999
    
    # Define criterion at the beginning of main() or training
    criterion = torch.nn.MSELoss(reduction='sum')
    
    for batch_idx, batch in pbar:
        # Get inputs from batch
        time_series = batch['time_series'].to(device)  # [batch_size, seq_len]
        mask = batch['mask'].to(device)  # [batch_size, seq_len]
        
        # Count tokens in this batch
        batch_size = time_series.size(0)
        seq_len = time_series.size(1)
        tokens_in_batch = batch_size * seq_len
        tokens_this_epoch += tokens_in_batch
        
        if token_counter is not None:
            token_counter.update(batch_size, seq_len)
        
        # Data validation and statistics
        current_means = torch.mean(time_series, dim=1)
        current_stds = torch.std(time_series, dim=1)
        current_mins = torch.min(time_series, dim=1).values
        current_maxs = torch.max(time_series, dim=1).values
        
        all_means.append(torch.mean(current_means).item())
        all_stds.append(torch.mean(current_stds).item())
        all_mins.append(torch.mean(current_mins).item())
        all_maxs.append(torch.mean(current_maxs).item())
        
        # Check that time series values are reasonable
        if torch.isnan(time_series).any() or torch.isinf(time_series).any():
            logger.warning(f"NaN/Inf in input time series at batch {batch_idx}. Skipping.")
            nan_batches += 1
            continue
            
        if torch.max(torch.abs(time_series)) > 1e6:
            logger.warning(f"Extreme values in input time series at batch {batch_idx}. Skipping.")
            nan_batches += 1
            continue
        
        # Add channel dimension if needed for MOMENT
        if len(time_series.shape) == 2:
            time_series = time_series.unsqueeze(1)  # [batch_size, 1, seq_len]
            
        # Assert that inputs have the expected shape
        assert time_series.dim() == 3, f"Expected 3D tensor (batch, channel, seq_len), got shape {time_series.shape}"
        assert time_series.shape[2] == args.seq_len, f"Expected sequence length {args.seq_len}, got {time_series.shape[2]}"
        assert mask.dim() == 2, f"Expected 2D tensor (batch, seq_len), got shape {mask.shape}"
        
        # Create masking for pretraining (which indices to predict)
        mask_indices = create_random_mask(time_series.squeeze(1), args.mask_ratio)
        
        # Count masked tokens
        masked_tokens_in_batch = mask_indices.sum().item()
        masked_tokens_this_epoch += masked_tokens_in_batch
        
        if token_counter is not None:
            token_counter.update(batch_size, seq_len, mask_indices)
        
        # Check if mask has a reasonable number of masked indices
        num_masked = mask_indices.sum().item()
        if num_masked < 1:
            logger.warning(f"Batch {batch_idx} has no masked positions, skipping")
            continue
            
        # Assert mask has the right shape and contains only binary values
        assert mask_indices.shape == mask.shape, f"Mask shape {mask_indices.shape} doesn't match expected {mask.shape}"
        assert torch.all((mask_indices == 0) | (mask_indices == 1)), "Mask contains non-binary values"
        
        # Zero the parameter gradients
        optimizer.zero_grad()
        
        # Forward pass with automatic mixed precision if enabled
        if args.use_amp:
            with autocast():
                outputs = model(x_enc=time_series, mask=mask_indices, input_mask=mask)
                # Calculate MSE loss on reconstructed values (focusing on masked positions)
                reconstruction = outputs.reconstruction
                
                # Calculate MSE loss only on masked positions
                if reconstruction.shape != time_series.shape:
                    reconstruction = reconstruction.view(time_series.shape)
                
                # Create binary mask tensor (1 for masked, 0 for observed)
                masked_positions = mask_indices.bool()
                
                # Get original and reconstructed values at masked positions
                original_masked = time_series * masked_positions.unsqueeze(1)
                reconstructed_masked = reconstruction * masked_positions.unsqueeze(1)
                
                # Calculate MSE loss only on masked positions with protection against NaN
                masked_values_count = masked_positions.sum().float()
                
                # Ensure we don't divide by zero or a very small number
                if masked_values_count > 10:  # Only compute loss if we have enough masked values
                    mse_loss = criterion(reconstructed_masked, original_masked)
                    loss = mse_loss / masked_values_count
                else:
                    # If too few masked positions, use a small but valid loss
                    logger.warning(f"Batch {batch_idx} has too few masked positions ({masked_values_count}), using dummy loss")
                    loss = torch.tensor(0.01, device=device, requires_grad=True)
        else:
            outputs = model(x_enc=time_series, mask=mask_indices, input_mask=mask)
            # Calculate MSE loss on reconstructed values (focusing on masked positions)
            reconstruction = outputs.reconstruction
            
            # Calculate MSE loss only on masked positions
            if reconstruction.shape != time_series.shape:
                reconstruction = reconstruction.view(time_series.shape)
            
            # Create binary mask tensor (1 for masked, 0 for observed)
            masked_positions = mask_indices.bool()
            
            # Get original and reconstructed values at masked positions
            original_masked = time_series * masked_positions.unsqueeze(1)
            reconstructed_masked = reconstruction * masked_positions.unsqueeze(1)
            
            # Calculate MSE loss only on masked positions with protection against NaN
            masked_values_count = masked_positions.sum().float()
            
            # Ensure we don't divide by zero or a very small number
            if masked_values_count > 10:  # Only compute loss if we have enough masked values
                mse_loss = criterion(reconstructed_masked, original_masked)
                loss = mse_loss / masked_values_count
            else:
                # If too few masked positions, use a small but valid loss
                logger.warning(f"Batch {batch_idx} has too few masked positions ({masked_values_count}), using dummy loss")
                loss = torch.tensor(0.01, device=device, requires_grad=True)
        
        # After loss computation and before backprop, check for NaN loss
        if torch.isnan(loss).item() or torch.isinf(loss).item():
            time_series_get = np.array(batch['time_series'].tolist())
            mask_get = np.array(batch['mask'].tolist())
            idx_get = np.array(batch['idx'].tolist())
            
            time_series_no_padding = time_series_get[mask_get == 1]
            
            time_series_is_2d = time_series_no_padding.ndim == 2
            
            ts_lengths = [len(ts) for ts in time_series_no_padding] if time_series_is_2d else [len(time_series_no_padding)]
            
            comp_string = f"Failed with minimum sequence length {min(ts_lengths)}, vs minimum successful length {lowest_valid_length}"
            
            logger.warning(f"NaN or Inf loss detected at batch {batch_idx}. Performing detailed analysis. {comp_string}")
            
            # Create directory for NaN diagnostic files if it doesn't exist
            nan_log_dir = os.path.join(args.output_dir, "nan_diagnostics")
            os.makedirs(nan_log_dir, exist_ok=True)
            
            # Detailed report filename with timestamp for uniqueness
            report_filename = f"NaN_autopsy_{time.strftime('%Y%m%d_%H%M%S')}_{batch_idx}.json"
            report_path = os.path.join(nan_log_dir, report_filename)
            
            # Create a detailed diagnostic report
            diagnostic_report = {
                "batch_info": {
                    "batch_idx": batch_idx,
                    "epoch": epoch,
                    "time_series_shape": time_series.shape,
                    "mask_shape": mask.shape,
                    "loss_value": str(loss.item()),
                    "is_nan": torch.isnan(loss).item(),
                    "is_inf": torch.isinf(loss).item()
                },
                "data_statistics": {
                    "time_series": {
                        "mean": float(torch.mean(time_series).cpu().item()) if not torch.isnan(torch.mean(time_series)).item() else "NaN",
                        "std": float(torch.std(time_series).cpu().item()) if not torch.isnan(torch.std(time_series)).item() else "NaN",
                        "min": float(torch.min(time_series).cpu().item()) if not torch.isnan(torch.min(time_series)).item() else "NaN",
                        "max": float(torch.max(time_series).cpu().item()) if not torch.isnan(torch.max(time_series)).item() else "NaN",
                        "has_nan": torch.isnan(time_series).any().item(),
                        "has_inf": torch.isinf(time_series).any().item()
                    }
                },
                "model_outputs": {
                    "reconstruction_shape": list(reconstruction.shape),
                    "reconstruction_statistics": {
                        "mean": float(torch.mean(reconstruction).cpu().item()) if not torch.isnan(torch.mean(reconstruction)).item() else "NaN",
                        "std": float(torch.std(reconstruction).cpu().item()) if not torch.isnan(torch.std(reconstruction)).item() else "NaN",
                        "min": float(torch.min(reconstruction).cpu().item()) if not torch.isnan(torch.min(reconstruction)).item() else "NaN",
                        "max": float(torch.max(reconstruction).cpu().item()) if not torch.isnan(torch.max(reconstruction)).item() else "NaN",
                        "has_nan": torch.isnan(reconstruction).any().item(),
                        "has_inf": torch.isinf(reconstruction).any().item()
                    }
                },
                "masking_info": {
                    "masked_positions_count": int(masked_positions.sum().item()),
                    "total_positions": int(masked_positions.numel()),
                    "mask_ratio": float(masked_positions.sum().item() / masked_positions.numel())
                },
                "loss_components": {
                    "original_masked_statistics": {
                        "mean": float(torch.mean(original_masked).cpu().item()) if not torch.isnan(torch.mean(original_masked)).item() else "NaN",
                        "std": float(torch.std(original_masked).cpu().item()) if not torch.isnan(torch.std(original_masked)).item() else "NaN",
                        "min": float(torch.min(original_masked).cpu().item()) if not torch.isnan(torch.min(original_masked)).item() else "NaN",
                        "max": float(torch.max(original_masked).cpu().item()) if not torch.isnan(torch.max(original_masked)).item() else "NaN",
                        "has_nan": torch.isnan(original_masked).any().item(),
                        "has_inf": torch.isinf(original_masked).any().item()
                    },
                    "reconstructed_masked_statistics": {
                        "mean": float(torch.mean(reconstructed_masked).cpu().item()) if not torch.isnan(torch.mean(reconstructed_masked)).item() else "NaN",
                        "std": float(torch.std(reconstructed_masked).cpu().item()) if not torch.isnan(torch.std(reconstructed_masked)).item() else "NaN",
                        "min": float(torch.min(reconstructed_masked).cpu().item()) if not torch.isnan(torch.min(reconstructed_masked)).item() else "NaN",
                        "max": float(torch.max(reconstructed_masked).cpu().item()) if not torch.isnan(torch.max(reconstructed_masked)).item() else "NaN",
                        "has_nan": torch.isnan(reconstructed_masked).any().item(),
                        "has_inf": torch.isinf(reconstructed_masked).any().item()
                    },
                    "masked_values_count": float(masked_values_count.item())
                },
                "sample_data": {
                    "time_series_sample": time_series_no_padding.tolist() if len(time_series_no_padding) <= 5 else "Too large to include",
                    "mean": np.mean(time_series_no_padding).tolist() if not np.isnan(np.mean(time_series_no_padding)) else "NaN",
                    "std": np.std(time_series_no_padding).tolist() if not np.isnan(np.std(time_series_no_padding)) else "NaN",
                    "total_samples": len(time_series_no_padding) if time_series_is_2d else 1,
                    "time_series_lengths": ts_lengths
                }
            }
            
            # EXTENDED REVIN ANALYSIS: Check if we can find and analyze the RevIN normalizer
            logger.info("Performing detailed RevIN analysis...")
            revin_data = {}
            
            # Find RevIN instances in the model
            for name, module in model.named_modules():
                if "normalizer" in name.lower() or isinstance(module, torch.nn.Module) and hasattr(module, "__class__") and "revin" in module.__class__.__name__.lower():
                    logger.info(f"Found RevIN component: {name}")
                    
                    # Check if this module has the diagnostics we added
                    if hasattr(module, "stats_log") and module.stats_log:
                        revin_data[name] = {
                            "module_type": str(type(module)),
                            "stats_log": module.stats_log
                        }
                        
                        # Add specific checks for key problem indicators
                        if hasattr(module, "mean") and hasattr(module, "stdev"):
                            try:
                                # Analyze mean
                                mean_data = module.mean.detach().cpu()
                                revin_data[name]["mean_stats"] = {
                                    "shape": list(mean_data.shape),
                                    "min": float(mean_data.min().item()) if not torch.isnan(mean_data.min()).item() else "NaN",
                                    "max": float(mean_data.max().item()) if not torch.isnan(mean_data.max()).item() else "NaN",
                                    "has_nan": torch.isnan(mean_data).any().item(),
                                    "has_inf": torch.isinf(mean_data).any().item()
                                }
                                
                                # Analyze stdev - this is the critical part
                                stdev_data = module.stdev.detach().cpu()
                                
                                # Calculate stats and find problematic values
                                min_stdev = float(stdev_data.min().item()) if not torch.isnan(stdev_data.min()).item() else "NaN"
                                
                                # Check for very small standard deviations
                                if isinstance(min_stdev, float) and min_stdev < 1e-4:
                                    # Find channels with very small stdev
                                    small_stdevs = (stdev_data < 1e-4).nonzero(as_tuple=True)
                                    small_stdev_indices = [(int(i.item()), int(j.item())) for i, j in zip(*small_stdevs)]
                                    
                                    # Get the actual stdev values for these indices
                                    small_stdev_values = [float(stdev_data[i, j].item()) for i, j in small_stdev_indices]
                                    
                                    revin_data[name]["problematic_stdevs"] = {
                                        "count": len(small_stdev_indices),
                                        "indices": small_stdev_indices[:10] if len(small_stdev_indices) > 10 else small_stdev_indices,
                                        "values": small_stdev_values[:10] if len(small_stdev_values) > 10 else small_stdev_values,
                                        "eps_value": getattr(module, "eps", 1e-5)
                                    }
                                    
                                    logger.warning(f"Found {len(small_stdev_indices)} channels with stdev < 1e-4 in {name}")
                                
                                revin_data[name]["stdev_stats"] = {
                                    "shape": list(stdev_data.shape),
                                    "min": min_stdev,
                                    "max": float(stdev_data.max().item()) if not torch.isnan(stdev_data.max()).item() else "NaN",
                                    "median": float(torch.median(stdev_data).item()) if not torch.isnan(torch.median(stdev_data)).item() else "NaN",
                                    "has_nan": torch.isnan(stdev_data).any().item(),
                                    "has_inf": torch.isinf(stdev_data).any().item()
                                }
                                
                            except Exception as e:
                                logger.error(f"Error analyzing RevIN component {name}: {str(e)}")
                                revin_data[name]["error"] = str(e)
            
            # Add RevIN analysis to the diagnostics
            diagnostic_report["revin_detailed_analysis"] = revin_data
            
            # Deep model tracing to identify the source of NaNs
            # Rerun the model with hooks to trace intermediate activations
            logger.info("Tracing model to identify the source of NaNs...")
            
            # Storage for activation traces
            activation_traces = {}
            nan_detected_at = None
            
            # Hook function to capture intermediate activations
            def hook_fn(name):
                def fn(module, input, output):
                    # Check if output contains NaN or Inf
                    has_nan = torch.isnan(output).any().item() if isinstance(output, torch.Tensor) else False
                    has_inf = torch.isinf(output).any().item() if isinstance(output, torch.Tensor) else False
                    
                    # For tuple/list outputs (like in some attention mechanisms)
                    if isinstance(output, tuple) or isinstance(output, list):
                        for i, out in enumerate(output):
                            if isinstance(out, torch.Tensor):
                                has_nan = has_nan or torch.isnan(out).any().item()
                                has_inf = has_inf or torch.isinf(out).any().item()
                    
                    # For dict outputs
                    if isinstance(output, dict):
                        for k, v in output.items():
                            if isinstance(v, torch.Tensor):
                                has_nan = has_nan or torch.isnan(v).any().item()
                                has_inf = has_inf or torch.isinf(v).any().item()
                    
                    # Store statistics about the activation
                    if isinstance(output, torch.Tensor):
                        # For single tensor outputs
                        try:
                            mean_val = float(torch.mean(output).cpu().item()) if not torch.isnan(torch.mean(output)).item() else "NaN"
                            std_val = float(torch.std(output).cpu().item()) if not torch.isnan(torch.std(output)).item() else "NaN"
                            min_val = float(torch.min(output).cpu().item()) if not torch.isnan(torch.min(output)).item() else "NaN"
                            max_val = float(torch.max(output).cpu().item()) if not torch.isnan(torch.max(output)).item() else "NaN"
                            
                            # Add more detailed statistics for NaN tracking
                            nan_count = torch.isnan(output).sum().item() if has_nan else 0
                            inf_count = torch.isinf(output).sum().item() if has_inf else 0
                            
                            activation_traces[name] = {
                                "shape": list(output.shape),
                                "has_nan": has_nan,
                                "has_inf": has_inf,
                                "nan_count": nan_count,
                                "inf_count": inf_count,
                                "nan_percentage": (nan_count / output.numel() * 100) if output.numel() > 0 else 0,
                                "mean": mean_val,
                                "std": std_val,
                                "min": min_val,
                                "max": max_val
                            }
                            
                            # Only for problematic outputs, get more detailed info
                            if has_nan or has_inf:
                                # If it's not too large, try to detect patterns in NaN positions
                                if output.numel() < 10000:
                                    nan_positions = torch.isnan(output).nonzero(as_tuple=True)
                                    if nan_positions[0].numel() > 0:
                                        # Sample up to 10 positions
                                        sample_size = min(10, nan_positions[0].numel())
                                        indices = torch.randperm(nan_positions[0].numel())[:sample_size]
                                        
                                        # Get positions
                                        sampled_positions = []
                                        for i in range(sample_size):
                                            pos = tuple(p[indices[i]].item() for p in nan_positions)
                                            sampled_positions.append(pos)
                                        
                                        activation_traces[name]["sample_nan_positions"] = sampled_positions
                        except:
                            activation_traces[name] = {
                                "shape": list(output.shape) if hasattr(output, "shape") else "Unknown",
                                "has_nan": has_nan,
                                "has_inf": has_inf,
                                "error": "Could not compute statistics"
                            }
                    else:
                        # For non-tensor outputs, just log the type
                        activation_traces[name] = {
                            "type": str(type(output)),
                            "has_nan": has_nan,
                            "has_inf": has_inf
                        }
                    
                    # If this is the first layer with NaN, mark it
                    nonlocal nan_detected_at
                    if (has_nan or has_inf) and nan_detected_at is None:
                        nan_detected_at = name
                        logger.warning(f"First NaN/Inf detected in layer: {name}")
                        
                        # For even more detailed inspection, if it's a tensor, save some example NaN values
                        if isinstance(output, torch.Tensor) and has_nan:
                            nan_mask = torch.isnan(output)
                            if nan_mask.any():
                                # Get indices of NaNs
                                nan_indices = nan_mask.nonzero(as_tuple=True)
                                
                                # Sample a few locations for detailed logging
                                indices_sample = [(dim[i].item() for dim in nan_indices) 
                                                for i in range(min(5, len(nan_indices[0])))]
                                
                                logger.warning(f"Sample NaN locations in {name}: {indices_sample}")
                return fn
            
            # Register hooks for all layers
            hooks = []
            for name, module in model.named_modules():
                if name != "":  # Skip the root module
                    hooks.append(module.register_forward_hook(hook_fn(name)))
            
            # Try to run the forward pass again to trace activations
            try:
                with torch.no_grad():
                    model.eval()  # Set to eval mode for tracing
                    traced_outputs = model(x_enc=time_series, mask=mask_indices, input_mask=mask)
                    model.train()  # Set back to train mode
            except Exception as e:
                # If the forward pass fails completely, note the error
                activation_traces["forward_pass_error"] = str(e)
            
            # Remove all hooks
            for hook in hooks:
                hook.remove()
            
            # Add activation traces to the diagnostic report
            diagnostic_report["layer_traces"] = {
                "first_nan_layer": nan_detected_at,
                "activations": activation_traces
            }
            
            # Add an input gradient analysis if we have access to the input gradients
            if time_series.grad is not None:
                try:
                    input_grad = time_series.grad.detach().cpu()
                    diagnostic_report["input_gradients"] = {
                        "has_nan": torch.isnan(input_grad).any().item(),
                        "has_inf": torch.isinf(input_grad).any().item(),
                        "mean": float(torch.mean(input_grad).item()) if not torch.isnan(torch.mean(input_grad)).item() else "NaN",
                        "std": float(torch.std(input_grad).item()) if not torch.isnan(torch.std(input_grad)).item() else "NaN",
                        "min": float(torch.min(input_grad).item()) if not torch.isnan(torch.min(input_grad)).item() else "NaN",
                        "max": float(torch.max(input_grad).item()) if not torch.isnan(torch.max(input_grad)).item() else "NaN"
                    }
                except:
                    diagnostic_report["input_gradients"] = {
                        "error": "Could not analyze input gradients"
                    }
            
            # Analyze model internals
            # Look at key components that might cause NaN/Inf
            try:
                # Check if model has specific components like RevIN, normalization layers, or attention
                has_revin = False
                has_layer_norm = False
                has_attention = False
                
                for name, module in model.named_modules():
                    if "revin" in name.lower():
                        has_revin = True
                    if "norm" in name.lower() or "layernorm" in name.lower():
                        has_layer_norm = True
                    if "attention" in name.lower():
                        has_attention = True
                
                diagnostic_report["model_components"] = {
                    "has_revin": has_revin,
                    "has_layer_norm": has_layer_norm, 
                    "has_attention": has_attention
                }
                
                # If model has RevIN normalization, check its parameters
                if has_revin:
                    revin_components = []
                    for name, module in model.named_modules():
                        if "revin" in name.lower():
                            try:
                                # Check if module has affine parameters
                                affine_params = {}
                                if hasattr(module, "affine") and module.affine:
                                    if hasattr(module, "weight"):
                                        weight = module.weight.detach().cpu()
                                        affine_params["weight"] = {
                                            "mean": float(torch.mean(weight).item()),
                                            "min": float(torch.min(weight).item()),
                                            "max": float(torch.max(weight).item()),
                                            "has_nan": torch.isnan(weight).any().item(),
                                            "has_inf": torch.isinf(weight).any().item()
                                        }
                                    if hasattr(module, "bias"):
                                        bias = module.bias.detach().cpu()
                                        affine_params["bias"] = {
                                            "mean": float(torch.mean(bias).item()),
                                            "min": float(torch.min(bias).item()),
                                            "max": float(torch.max(bias).item()),
                                            "has_nan": torch.isnan(bias).any().item(),
                                            "has_inf": torch.isinf(bias).any().item()
                                        }
                                
                                revin_components.append({
                                    "name": name,
                                    "affine_params": affine_params
                                })
                            except Exception as e:
                                revin_components.append({
                                    "name": name,
                                    "error": str(e)
                                })
                
                    diagnostic_report["revin_analysis"] = revin_components
            except Exception as e:
                diagnostic_report["model_components_error"] = str(e)
            
            # Add model weight statistics
            weight_stats = {}
            potential_issues = []
            
            for name, param in model.named_parameters():
                try:
                    if param.requires_grad:
                        # Compute parameter statistics
                        param_data = param.data.detach().cpu()
                        
                        # Check for NaN/Inf in weights
                        has_nan = torch.isnan(param_data).any().item()
                        has_inf = torch.isinf(param_data).any().item()
                        
                        # Only calculate statistics if the parameter doesn't have NaN/Inf
                        if not has_nan and not has_inf:
                            mean_val = float(torch.mean(param_data).item())
                            std_val = float(torch.std(param_data).item())
                            min_val = float(torch.min(param_data).item())
                            max_val = float(torch.max(param_data).item())
                            norm_val = float(torch.norm(param_data).item())
                        else:
                            mean_val = "NaN"
                            std_val = "NaN"
                            min_val = "NaN"
                            max_val = "NaN"
                            norm_val = "NaN"
                        
                        # Check the gradient if it exists
                        if param.grad is not None:
                            grad_data = param.grad.detach().cpu()
                            grad_has_nan = torch.isnan(grad_data).any().item()
                            grad_has_inf = torch.isinf(grad_data).any().item()
                            
                            if not grad_has_nan and not grad_has_inf:
                                grad_mean = float(torch.mean(grad_data).item())
                                grad_std = float(torch.std(grad_data).item())
                                grad_min = float(torch.min(grad_data).item())
                                grad_max = float(torch.max(grad_data).item())
                                grad_norm = float(torch.norm(grad_data).item())
                            else:
                                grad_mean = "NaN"
                                grad_std = "NaN"
                                grad_min = "NaN"
                                grad_max = "NaN"
                                grad_norm = "NaN"
                        else:
                            grad_has_nan = False
                            grad_has_inf = False
                            grad_mean = None
                            grad_std = None
                            grad_min = None
                            grad_max = None
                            grad_norm = None
                        
                        # Record potential issues
                        if has_nan or has_inf:
                            potential_issues.append(f"Parameter {name} has NaN/Inf values")
                        if grad_has_nan or grad_has_inf:
                            potential_issues.append(f"Gradient for {name} has NaN/Inf values")
                        if grad_norm is not None and (grad_norm > 10.0 or np.isclose(grad_norm, 0.0)):
                            potential_issues.append(f"Gradient for {name} has unusual norm: {grad_norm}")
                        
                        # Store statistics
                        weight_stats[name] = {
                            "shape": list(param.shape),
                            "weights": {
                                "mean": mean_val,
                                "std": std_val,
                                "min": min_val,
                                "max": max_val,
                                "norm": norm_val,
                                "has_nan": has_nan,
                                "has_inf": has_inf
                            },
                            "gradients": {
                                "mean": grad_mean,
                                "std": grad_std,
                                "min": grad_min,
                                "max": grad_max,
                                "norm": grad_norm,
                                "has_nan": grad_has_nan,
                                "has_inf": grad_has_inf
                            }
                        }
                except Exception as e:
                    weight_stats[name] = {"error": str(e)}
            
            diagnostic_report["model_weights"] = weight_stats
            diagnostic_report["potential_issues"] = potential_issues
            
            # Analysis of which components might be causing the problem
            if torch.isnan(mse_loss).item():
                diagnostic_report["nan_source_analysis"] = "NaN originates in MSE loss calculation"
            elif torch.isnan(reconstruction).any().item():
                diagnostic_report["nan_source_analysis"] = "NaN found in model's reconstruction output"
            elif torch.isnan(time_series).any().item():
                diagnostic_report["nan_source_analysis"] = "NaN found in input time series"
            else:
                diagnostic_report["nan_source_analysis"] = "Could not determine exact source of NaN"
            
            # Save reconstructed outputs for detailed inspection
            reconstruction_sample = reconstruction.detach().cpu().numpy()
            diagnostic_report["reconstruction_sample"] = reconstruction_sample[:2].tolist() if len(reconstruction_sample) > 0 else []
            
            # Add root cause analysis based on layer traces
            if nan_detected_at is not None:
                diagnostic_report["root_cause_analysis"] = {
                    "first_nan_layer": nan_detected_at,
                    "layer_type": str(type(model.get_submodule(nan_detected_at) if nan_detected_at else None)),
                    "possible_causes": [
                        "Division by zero or very small number",
                        "Log of zero or negative number",
                        "Square root of negative number",
                        "Overflow in exp() function",
                        "Extremely large inputs to softmax",
                        "Unhandled edge cases in custom functions"
                    ],
                    "recommendations": [
                        "Add epsilon to denominators",
                        "Apply clipping to inputs of exponential functions",
                        "Check for proper normalization of inputs",
                        "Consider gradient clipping",
                        "Reduce learning rate",
                        "Check for extreme weight initialization"
                    ]
                }
            
            # Save detailed diagnostic information
            with open(report_path, "w") as f:
                json.dump(diagnostic_report, f, indent=2)
            
            logger.error(f"NaN/Inf analysis complete. Full report saved to {report_path}")
            if nan_detected_at:
                logger.error(f"First NaN detected at layer: {nan_detected_at}")
            logger.error(f"Top potential issues: {', '.join(potential_issues[:5])}" if potential_issues else "No obvious issues found in weights")
            
            # Stop the training run as requested
            raise ValueError(f"Training stopped due to NaN/Inf loss. See detailed report at {report_path}")
        
        valid_batches += 1
        
        # Backward + optimize
        if args.use_amp:
            scaler.scale(loss).backward()
            
            # Gradient clipping
            if args.gradient_clip > 0:
                scaler.unscale_(optimizer)
                torch.nn.utils.clip_grad_norm_(model.parameters(), args.gradient_clip)
            
            scaler.step(optimizer)
            scaler.update()
        else:
            loss.backward()
            
            # Gradient clipping
            if args.gradient_clip > 0:
                torch.nn.utils.clip_grad_norm_(model.parameters(), args.gradient_clip)
            
            optimizer.step()
        
        # Update learning rate
        scheduler.step()
        
        # Update statistics
        total_loss += loss.item()
        
        # Record continuous training loss and token count for this batch
        if continuous_train_losses is not None and continuous_train_tokens is not None:
            # ALWAYS record the loss for EVERY SINGLE SAMPLE as requested
            # This is critical for detailed loss curve tracking
            loss_value = loss.item()
            token_count = token_counter.get_tokens()
            continuous_train_losses.append(loss_value)
            continuous_train_tokens.append(token_count)
            logger.debug(f"Recording continuous loss: {loss_value:.6f} at token count: {token_count}")
        
        # Process a validation batch alongside training
        if val_iter is not None and continuous_val_losses is not None and continuous_val_tokens is not None:
            # Get a batch from the validation set
            try:
                val_batch = next(val_iter)
                
                # Process validation batch without affecting gradients
                with torch.no_grad():
                    model.eval()  # Set model to evaluation mode
                    
                    # Move data to device
                    val_time_series = val_batch['time_series'].to(device)
                    val_mask = val_batch['mask'].to(device)
                    
                    # Add channel dimension if needed for MOMENT
                    if len(val_time_series.shape) == 2:
                        val_time_series = val_time_series.unsqueeze(1)
                    
                    # Create masking for validation
                    val_mask_indices = create_random_mask(val_time_series.squeeze(1), args.mask_ratio)
                    
                    # Forward pass for validation
                    val_outputs = model(x_enc=val_time_series, mask=val_mask_indices, input_mask=val_mask)
                    
                    # Calculate validation loss
                    val_reconstruction = val_outputs.reconstruction
                    if val_reconstruction.shape != val_time_series.shape:
                        val_reconstruction = val_reconstruction.view(val_time_series.shape)
                    
                    # Create binary mask tensor and calculate loss on masked positions
                    val_masked_positions = val_mask_indices.bool()
                    val_original_masked = val_time_series * val_masked_positions.unsqueeze(1)
                    val_reconstructed_masked = val_reconstruction * val_masked_positions.unsqueeze(1)
                    
                    val_masked_values_count = val_masked_positions.sum().float()
                    if val_masked_values_count > 0:
                        val_mse_loss = criterion(val_reconstructed_masked, val_original_masked)
                        val_loss = val_mse_loss / val_masked_values_count
                    else:
                        val_loss = torch.tensor(0.0, device=device)
                    
                    # Record validation loss and token count
                    val_batch_size = val_time_series.size(0)
                    val_seq_len = val_time_series.size(1)
                    val_tokens_in_batch = val_batch_size * val_seq_len
                    
                    if val_samples_seen is not None:
                        val_samples_seen.append(val_tokens_in_batch)
                    
                    continuous_val_losses.append(val_loss.item())
                    continuous_val_tokens.append(token_counter.get_tokens())  # Use same token counter as training
                    
                    # Set model back to training mode
                    model.train()
            except StopIteration:
                # This shouldn't happen with cycle, but just in case
                val_iter = cycle(val_loader)
            except Exception as e:
                logger.warning(f"Error processing validation batch: {str(e)}")
                # Continue with training even if validation fails
                model.train()
        
        # Periodically check model for NaN weights
        if batch_idx % args.check_nan_frequency == 0:
            has_nan = False
            for name, param in model.named_parameters():
                if torch.isnan(param).any() or torch.isinf(param).any():
                    logger.warning(f"NaN or Inf detected in {name} parameter")
                    has_nan = True
            
            if has_nan:
                logger.error("NaN parameters detected in model. Consider restarting with a lower learning rate.")
        
        # Update progress bar with token information
        if args.log_token_counts:
            pbar.set_description(f"Epoch {epoch+1} | Loss: {loss.item():.4f} | NaN: {nan_batches}/{batch_idx+1} | Tokens: {tokens_this_epoch:,}")
        else:
            pbar.set_description(f"Epoch {epoch+1} | Loss: {loss.item():.4f} | NaN: {nan_batches}/{batch_idx+1}")
        
        # Log metrics to file
        if batch_idx % args.log_interval == 0:
            global_step = epoch * len(data_loader) + batch_idx
            current_lr = scheduler.get_last_lr()[0]
            
            # Log to file
            metrics = {
                'epoch': epoch + 1,
                'batch': batch_idx,
                'global_step': global_step,
                'train_loss': loss.item(),
                'learning_rate': current_lr,
                'nan_batches': nan_batches,
                'valid_batches': valid_batches,
                'tokens_processed': tokens_this_epoch,
                'masked_tokens': masked_tokens_this_epoch,
                'timestamp': time.strftime('%Y-%m-%d %H:%M:%S')
            }
            
            metrics_file.write(json.dumps(metrics) + '\n')
            metrics_file.flush()
        
        # # Save checkpoint periodically
        # if (batch_idx + 1) % args.save_interval == 0:
        #     checkpoint_path = os.path.join(args.output_dir, f"moment_{args.model_size}_epoch{epoch+1}_batch{batch_idx+1}.pt")
        #     torch.save({
        #         'epoch': epoch,
        #         'batch_idx': batch_idx,
        #         'model_state_dict': model.state_dict(),
        #         'optimizer_state_dict': optimizer.state_dict(),
        #         'scheduler_state_dict': scheduler.state_dict(),
        #         'loss': loss.item(),
        #     }, checkpoint_path)
        #     logger.info(f"Saved checkpoint to {checkpoint_path}")
    
    # Log data statistics
    if len(all_means) > 0:
        logger.info(f"Data statistics - Avg Mean: {np.mean(all_means):.4f}, Avg Std: {np.mean(all_stds):.4f}, "
                   f"Avg Min: {np.mean(all_mins):.4f}, Avg Max: {np.mean(all_maxs):.4f}")
    
    # Skip computing average loss if no valid batches were processed
    if valid_batches == 0:
        logger.error("No valid batches in epoch. Stopping training.")
        raise ValueError("No valid batches in epoch. Training cannot continue.")
    
    # Compute average loss for the epoch (only from valid batches)
    avg_loss = total_loss / valid_batches
    elapsed = time.time() - start_time
    
    logger.info(f"Epoch {epoch+1} | Avg Loss: {avg_loss:.4f} | Time: {elapsed:.2f}s | NaN batches: {nan_batches}")
    
    # Log token statistics
    logger.info(f"Tokens processed this epoch: {tokens_this_epoch:,} (masked: {masked_tokens_this_epoch:,}, {masked_tokens_this_epoch/tokens_this_epoch*100:.2f}%)")
    if token_counter is not None:
        logger.info(f"Total tokens processed so far: {token_counter.get_tokens():,} (masked: {token_counter.get_masked_tokens():,}, {token_counter.get_masked_tokens()/token_counter.get_tokens()*100:.2f}%)")
    
    return avg_loss

def validate(model, data_loader, device, epoch, args, metrics_file):
    """Validate the model on the validation dataset."""
    model.eval()
    total_loss = 0
    start_time = time.time()
    
    # Create progress bar
    pbar = tqdm(enumerate(data_loader), total=len(data_loader), desc=f"Epoch {epoch+1} Validation")
    # Define criterion at the beginning of main() or training
    criterion = torch.nn.MSELoss(reduction='sum')
    
    with torch.no_grad():
        for batch_idx, batch in pbar:
            # Move data to device
            time_series = batch['time_series'].to(device)  # [batch_size, seq_len]
            mask = batch['mask'].to(device)  # [batch_size, seq_len]
            
            # Add channel dimension if needed for MOMENT
            if len(time_series.shape) == 2:
                time_series = time_series.unsqueeze(1)  # [batch_size, 1, seq_len]
            
            # Create masking for validation (which indices to predict)
            mask_indices = create_random_mask(time_series.squeeze(1), args.mask_ratio)
            
            # Forward pass
            outputs = model(x_enc=time_series, mask=mask_indices, input_mask=mask)
            
            # Calculate MSE loss on reconstructed values (focusing on masked positions)
            reconstruction = outputs.reconstruction
            
            # Calculate MSE loss only on masked positions
            if reconstruction.shape != time_series.shape:
                reconstruction = reconstruction.view(time_series.shape)
            
            # Create binary mask tensor (1 for masked, 0 for observed)
            masked_positions = mask_indices.bool()
            
            # Get original and reconstructed values at masked positions
            original_masked = time_series * masked_positions.unsqueeze(1)
            reconstructed_masked = reconstruction * masked_positions.unsqueeze(1)
            
            # Calculate MSE loss only on masked positions
            masked_values_count = masked_positions.sum().float()
            if masked_values_count > 0:
                loss = criterion(reconstructed_masked, original_masked) / masked_values_count
            else:
                loss = torch.tensor(0.0, device=device)
            
            # Update total loss
            total_loss += loss.item()
            
            # Update progress bar
            pbar.set_postfix(loss=loss.item())
    
    # Calculate average loss
    avg_loss = total_loss / len(data_loader)
    
    # Calculate time taken
    time_taken = time.time() - start_time
    
    logger.info(f"Validation complete - Avg loss: {avg_loss:.6f}, Time: {time_taken:.2f}s")
    
    return avg_loss

def check_dataset_statistics(dataset, sample_size=1000):
    """Check statistics of the dataset to verify data quality."""
    logger.info(f"Checking dataset statistics with {min(sample_size, len(dataset))} samples...")
    
    # Sample indices to check
    sample_size = min(sample_size, len(dataset))
    indices = random.sample(range(len(dataset)), sample_size)
    
    # Track statistics
    shapes = []
    means = []
    stds = []
    mins = []
    maxs = []
    nan_count = 0
    inf_count = 0
    small_std_count = 0
    
    for idx in tqdm(indices, desc="Checking data samples"):
        try:
            logger.info(f"Processing sample {idx}")
            sample = dataset[idx]
            
            logger.info(f"Sample keys: {sample.keys()}")
            ts = sample['time_series']
            logger.info(f"Time series type: {type(ts)}, shape/len: {ts.shape if hasattr(ts, 'shape') else len(ts)}")
            
            # Convert to tensor if not already
            if not isinstance(ts, torch.Tensor):
                logger.info(f"Converting to tensor, original type: {type(ts)}")
                ts = torch.tensor(ts, dtype=torch.float32)
            
            logger.info(f"Tensor shape: {ts.shape}, device: {ts.device}, dtype: {ts.dtype}")
            
            # Track shape
            shapes.append(ts.shape)
            
            # Check for NaN/Inf using torch functions
            if torch.isnan(ts).any():
                logger.info(f"NaN values found in sample {idx}")
                nan_count += 1
                continue
                
            if torch.isinf(ts).any():
                logger.info(f"Inf values found in sample {idx}")
                inf_count += 1
                continue
            
            # Compute statistics with torch
            mean_val = torch.mean(ts.float()).item()
            std_val = torch.std(ts.float()).item()
            min_val = torch.min(ts.float()).item()
            max_val = torch.max(ts.float()).item()
            
            logger.info(f"Sample {idx} stats - mean: {mean_val:.2f}, std: {std_val:.2f}, min: {min_val:.2f}, max: {max_val:.2f}")
            
            means.append(mean_val)
            stds.append(std_val)
            mins.append(min_val)
            maxs.append(max_val)
            
            if std_val < 1e-6:
                logger.info(f"Low variance detected in sample {idx}")
                small_std_count += 1
                
        except Exception as e:
            logger.error(f"Error processing sample {idx}: {str(e)}", exc_info=True)
            continue
    
    # Log results
    logger.info(f"Dataset check results:")
    logger.info(f"- Samples processed: {len(shapes)}/{sample_size}")
    
    if shapes:
        unique_shapes = set(str(s) for s in shapes)
        logger.info(f"- Unique shapes ({len(unique_shapes)}): {unique_shapes}")
    
    logger.info(f"- NaN samples: {nan_count}/{sample_size} ({nan_count/sample_size*100:.2f}%)")
    logger.info(f"- Inf samples: {inf_count}/{sample_size} ({inf_count/sample_size*100:.2f}%)")
    logger.info(f"- Low variance samples: {small_std_count}/{sample_size} ({small_std_count/sample_size*100:.2f}%)")
    
    if means:
        logger.info(f"- Mean statistics: min={min(means):.4f}, max={max(means):.4f}, avg={np.mean(means):.4f}")
        logger.info(f"- Std statistics: min={min(stds):.4f}, max={max(stds):.4f}, avg={np.mean(stds):.4f}")
        logger.info(f"- Value range: min={min(mins):.4f}, max={max(maxs):.4f}")
    
    if nan_count > 0 or inf_count > 0:
        logger.warning("Dataset contains NaN or Inf values that will be filtered!")
    
    if small_std_count > sample_size * 0.5:
        logger.warning("More than 50% of samples have near-zero standard deviation!")
    
    return {
        'nan_ratio': nan_count / sample_size,
        'inf_ratio': inf_count / sample_size,
        'low_variance_ratio': small_std_count / sample_size,
        'mean_stats': [min(means), max(means), np.mean(means)] if means else None,
        'std_stats': [min(stds), max(stds), np.mean(stds)] if stds else None,
        'value_range': [min(mins), max(maxs)] if mins else None,
        'unique_shapes': list(unique_shapes) if shapes else None
    }

def main():
    """Main function to run the training."""
    # Parse args
    args = parse_args()
    
    # Setup logging and devices
    setup_logging(args)
    device = setup_device(args)
    
    # Set seed for reproducibility
    seed_everything(args.seed)
    
    logger.info(f"Using dataset from: {args.dataset_dir}")
    
    # Create model config
    config = get_model_config(args)
    
    # Check if we're using filtered data
    dataset_dir = args.dataset_dir
    filtered_path = os.path.join(args.dataset_dir, "filtered")
    if os.path.exists(filtered_path) and not args.use_unfiltered_data:
        logger.info(f"Using pre-filtered dataset from: {filtered_path}")
        dataset_dir = filtered_path
    else:
        logger.info("Using original dataset (not pre-filtered)")
    
    # Load dataset
    full_dataset = TimeSeriesPILEDataset(
        download_dir=dataset_dir,
        seq_len=args.seq_len,
        seed=args.seed,
        max_files=args.max_files,
        overlap_pct=0,
        filter_problematic=True  # Enable in-memory filtering
    )
    
    # Split dataset into train, validation, and test (80/10/10 split)
    total_samples = len(full_dataset)
    train_size = int(0.8 * total_samples)
    val_size = int(0.1 * total_samples)
    test_size = total_samples - train_size - val_size

    train_dataset, val_dataset, test_dataset = random_split(
        full_dataset, 
        [train_size, val_size, test_size],
        generator=torch.Generator().manual_seed(args.seed)
    )
    
    logger.info(f"Dataset loaded with {len(full_dataset)} samples")
    logger.info(f"Training on {len(train_dataset)} samples, validating on {len(val_dataset)} samples, testing on {len(test_dataset)} samples")
    
    # Create data loaders
    train_loader = DataLoader(
        train_dataset,
        batch_size=args.batch_size,
        shuffle=True,
        num_workers=args.num_workers,
        collate_fn=lambda batch: custom_collate(
            batch, 
            train_dataset,
            True
        ),
        pin_memory=True if args.device == 'cuda' else False,
        drop_last=True,
        persistent_workers=args.persistent_workers
    )
    
    val_loader = DataLoader(
        val_dataset,
        batch_size=args.batch_size,
        shuffle=False,
        num_workers=args.num_workers,
        collate_fn=lambda batch: custom_collate(
            batch, 
            val_dataset,
            True
        ),
        pin_memory=True if args.device == 'cuda' else False,
        drop_last=False
    )
    
    test_loader = DataLoader(
        test_dataset,
        batch_size=args.batch_size,
        shuffle=False,
        num_workers=args.num_workers,
        collate_fn=lambda batch: custom_collate(batch, test_dataset, True),
        pin_memory=True if args.device == 'cuda' else False,
        drop_last=False
    )
    
    
    # We'll hash the data loaders to validate complete reproducibility
    # Across model tests, ensuring that all batches are processed in the same order
    # and that everything is exactly the same
    # Essentially, we are guaranteeing that circumstances are completely identical
    # across model tests
    data_loaders_sample_sets = {k: set() for k in ["train", "val", "test"]}
    hash_data_loaders_thorough = {k: "a" for k in ["train", "val", "test", "full"]}
    for (loader, key) in zip(
        [train_loader, val_loader, test_loader], 
        ["train", "val", "test"]
    ):
        for batch in tqdm(loader, desc=f"Hashing {key} data"):
            sample_hash = sha256_of_tensor(batch["time_series"])
            data_loaders_sample_sets[key].add(sample_hash)
            
            hash_data_loaders_thorough[key] += sample_hash
            hash_data_loaders_thorough[key] += sha256_of_tensor(batch["mask"])
            hash_data_loaders_thorough[key] += sha256_of_tensor(batch["idx"])
            
            hash_data_loaders_thorough[key] = sha256_of_string(hash_data_loaders_thorough[key])
        
        hash_data_loaders_thorough["full"] += hash_data_loaders_thorough[key]
    hash_data_loaders_thorough["full"] = sha256_of_string(hash_data_loaders_thorough["full"])
    
    
    # Check that intersections are empty
    for key in data_loaders_sample_sets:
        for other_key in data_loaders_sample_sets:
            if key == other_key:
                continue
            
            intersection = data_loaders_sample_sets[key].intersection(data_loaders_sample_sets[other_key])
            assert len(intersection) == 0, \
                f"Intersections are not empty for {key} and {other_key} ({len(intersection)} samples in common)"
    
    
    
    # Initialize lists to capture loss curves data
    train_loss_series = []
    val_loss_series = []
    tokens_series = []
    # Lists for continuous (per-batch) loss tracking
    continuous_train_losses = []
    continuous_train_tokens = []
    # Lists for continuous validation loss tracking
    continuous_val_losses = []
    continuous_val_tokens = []
    val_samples_seen = []

    # Create the model
    model = create_model(args, config)
    model.to(device)
    
    # Create optimizer and scheduler
    optimizer = create_optimizer(model, args)
    scheduler = create_scheduler(optimizer, args, len(train_loader))
    
    # Mixed precision if enabled
    scaler = None
    if args.mixed_precision and args.device == 'cuda':
        scaler = torch.cuda.amp.GradScaler()
    
    # Resume from checkpoint if provided
    start_epoch = 0
    if args.resume_checkpoint:
        if os.path.isfile(args.resume_checkpoint):
            logger.info(f"Loading checkpoint from {args.resume_checkpoint}")
            checkpoint = torch.load(args.resume_checkpoint, map_location=device)
            start_epoch = checkpoint['epoch'] + 1
            model.load_state_dict(checkpoint['model'])
            optimizer.load_state_dict(checkpoint['optimizer'])
            if scheduler is not None and 'scheduler' in checkpoint:
                scheduler.load_state_dict(checkpoint['scheduler'])
            if scaler is not None and 'scaler' in checkpoint:
                scaler.load_state_dict(checkpoint['scaler'])
        else:
            logger.warning(f"Checkpoint file not found: {args.resume_checkpoint}")
    
    # Create metrics file
    metrics_file_path = os.path.join(args.output_dir, f"metrics_{time.strftime('%Y%m%d_%H%M%S')}.jsonl")
    metrics_file = open(metrics_file_path, 'w')
    
    # Also create a CSV summary file for high-level metrics
    summary_file_path = os.path.join(args.output_dir, f"summary_{time.strftime('%Y%m%d_%H%M%S')}.csv")
    with open(summary_file_path, 'w') as f:
        f.write("epoch,train_loss,val_loss,learning_rate,time_elapsed\n")
    
    # Create token counter
    token_counter = TokenCounter()
    
    # Train loop
    logger.info("Starting training loop")
    
    try:
        for epoch in range(start_epoch, args.epochs):
            epoch_start_time = time.time()
            
            # Train
            train_loss = train_epoch(
                model, train_loader, optimizer, scheduler, device, epoch, args, metrics_file, scaler, token_counter, continuous_train_losses, continuous_train_tokens,
                val_loader, continuous_val_losses, continuous_val_tokens, val_samples_seen
            )
            
            # Validate using test set as the final validation dataset
            val_loss = validate(model, test_loader, device, epoch, args, metrics_file)
            
            # Record loss and token counts for plotting
            train_loss_series.append(train_loss)
            val_loss_series.append(val_loss)
            tokens_series.append(token_counter.get_tokens())
            
            # # Save model checkpoint TODO: Uncomment this, for saving checkpoints
            # if args.save_every > 0 and (epoch + 1) % args.save_every == 0:
            #     save_checkpoint(
            #         model, optimizer, scheduler, epoch, train_loss, val_loss, 
            #         os.path.join(args.output_dir, f"checkpoint_{epoch+1}.pt"),
            #         args.keep_top_n, scaler, token_counter
            #     )
            
            # Log the epoch
            epoch_time = time.time() - epoch_start_time
            logger.info(
                f"Epoch {epoch+1}/{args.epochs}: train_loss={train_loss:.6f}, val_loss={val_loss:.6f}, "
                f"time={epoch_time:.2f}s, tokens={token_counter.get_tokens():,}"
            )
            
            # Early stopping
            if args.early_stopping > 0 and val_loss < args.early_stopping:
                logger.info(f"Early stopping triggered at epoch {epoch+1} with val_loss={val_loss:.6f}")
                break
    finally:
        # Make sure to close the metrics file
        metrics_file.close()
    
    # Print validation statistics
    print_validation_stats()
    
    
    
    # Load comparison model
    if args.load_comparison:
        logger.info(f"Loading comparison model from {args.load_comparison}")
        comparison_model = MOMENTPipeline.from_pretrained(args.load_comparison)
        comparison_model.to(device)
        
        logger.info(f"Type of model: {type(model)}")
        logger.info(f"Type of comparison model: {type(comparison_model)}")
        
        val_loss_comparison = validate(comparison_model, test_loader, device, epoch, args, metrics_file)
        
        logger.info(f"Final val loss (new  model): {val_loss:.6f}")
        logger.info(f"Final val loss (comp model): {val_loss_comparison:.6f}")
    else:
        logger.info(f"Final val loss: {val_loss:.6f}")
    
    
    logger.info("Training complete!")
    
    # Final checkpoint
    final_checkpoint_path = os.path.join(args.output_dir, "checkpoint_final.pt")
    save_checkpoint(
        model, optimizer, scheduler, epoch, train_loss, val_loss, 
        final_checkpoint_path, 0, scaler, token_counter
    )
    
    logger.info(f"Final model saved to {final_checkpoint_path}")
    logger.info(f"Total tokens processed: {token_counter.get_tokens():,}")

    # Save the loss curves as CSV for further analysis
    continuous_loss_csv_path = os.path.join(args.output_dir, 'continuous_loss_data.csv')
    with open(continuous_loss_csv_path, 'w') as f:
        f.write("tokens,train_loss,val_loss\n")
        for i in range(len(continuous_train_losses)):
            val_loss_str = continuous_val_losses[i] if i < len(continuous_val_losses) else ""
            f.write(f"{continuous_train_tokens[i]},{continuous_train_losses[i]},{val_loss_str}\n")
    logger.info(f"Continuous loss data saved to {continuous_loss_csv_path}")
    with open(os.path.join(args.output_dir, 'model_print.txt'), 'w') as f:
        f.write(str(model))

    # Function to apply moving average smoothing
    def moving_average(data, window_size):
        weights = np.ones(window_size) / window_size
        return np.convolve(data, weights, mode='valid')
    
    # Plot loss curves for training and validation
    # Use a window size of 20 for the moving average as requested
    window_size = 20
    
    # Apply smoothing if we have enough data points
    if len(continuous_train_losses) >= window_size:
        smoothed_train_losses = moving_average(continuous_train_losses, window_size)
        # Adjust token counts to match the length of smoothed losses
        smoothed_train_tokens = continuous_train_tokens[window_size-1:len(continuous_train_losses)]
    else:
        # If not enough data points, just use the raw data
        smoothed_train_losses = continuous_train_losses
        smoothed_train_tokens = continuous_train_tokens
        logger.warning(f"Not enough data points for a window size of {window_size}. Using raw data.")
    
    # Apply smoothing to validation losses if we have enough data points
    if len(continuous_val_losses) >= window_size:
        smoothed_val_losses = moving_average(continuous_val_losses, window_size)
        # Adjust token counts to match the length of smoothed losses
        smoothed_val_tokens = continuous_val_tokens[window_size-1:len(continuous_val_losses)]
    else:
        # If not enough data points, just use the raw data
        smoothed_val_losses = continuous_val_losses
        smoothed_val_tokens = continuous_val_tokens
        logger.warning(f"Not enough validation data points for a window size of {window_size}. Using raw data.")
    
    plt.figure()
    # Plot the per-batch (continuous) training losses with moving average
    if len(smoothed_train_losses) > 0:
        plt.plot(smoothed_train_tokens, smoothed_train_losses, 'b-', alpha=0.7, color='blue',
                 label=f'Training Loss (Moving Avg {window_size})')
    
    # Plot the per-batch (continuous) validation losses with moving average
    if len(smoothed_val_losses) > 0:
        plt.plot(smoothed_val_tokens, smoothed_val_losses, 'g-', alpha=0.7, color='orange',
                 label=f'Validation Loss (Moving Avg {window_size})')
    
    # Also plot the epoch level validation losses for comparison
    plt.plot(tokens_series, val_loss_series, label='Epoch Validation Loss')
    plt.xlabel('Tokens Processed')
    plt.ylabel('Loss')
    plt.title('Loss Curves Over Training')
    plt.legend()
    plot_path = os.path.join(args.output_dir, 'loss_curves.pdf')
    plt.savefig(plot_path)
    plt.close()
    logger.info(f"Loss curve graph saved to {plot_path}")

    # Create a second, more detailed plot showing both raw and smoothed training loss
    if len(continuous_train_losses) > 0:
        plt.figure(figsize=(12, 6))
        
        # Plot raw per-batch losses as light scatter points
        plt.scatter(continuous_train_tokens, continuous_train_losses, 
                   s=1, alpha=0.3, c='blue', label='Raw Training Loss (Per Batch)')
        
        # Plot raw validation losses as light scatter points
        if len(continuous_val_losses) > 0:
            plt.scatter(continuous_val_tokens, continuous_val_losses,
                       s=1, alpha=0.3, c='green', label='Raw Validation Loss (Per Batch)')
        
        # Plot smoothed version on top
        if len(smoothed_train_losses) > 0:
            plt.plot(smoothed_train_tokens, smoothed_train_losses, 'r-', linewidth=2,
                     label=f'Smoothed Training Loss (Window Size {window_size})')
        
        # Plot smoothed validation loss on top
        if len(smoothed_val_losses) > 0:
            plt.plot(smoothed_val_tokens, smoothed_val_losses, 'g-', linewidth=2,
                     label=f'Smoothed Validation Loss (Window Size {window_size})')
        
        plt.xlabel('Tokens Processed')
        plt.ylabel('Loss')
        plt.title('Detailed Loss Curves Over Time')
        plt.legend()
        plt.grid(True, alpha=0.3)
        
        # Add some annotations to help with interpretation
        if len(continuous_train_losses) > 0:
            min_train_loss = min(continuous_train_losses)
            max_train_loss = max(continuous_train_losses)
            min_val_loss = min(continuous_val_losses) if continuous_val_losses else float('inf')
            max_val_loss = max(continuous_val_losses) if continuous_val_losses else float('-inf')
            
            plt.annotate(f'Min Train Loss: {min_train_loss:.4f}', 
                        xy=(0.02, 0.05), xycoords='axes fraction')
            plt.annotate(f'Max Train Loss: {max_train_loss:.4f}', 
                        xy=(0.02, 0.10), xycoords='axes fraction')
            plt.annotate(f'Min Val Loss: {min_val_loss:.4f}', 
                        xy=(0.02, 0.15), xycoords='axes fraction')
            plt.annotate(f'Max Val Loss: {max_val_loss:.4f}', 
                        xy=(0.02, 0.20), xycoords='axes fraction')
            plt.annotate(f'Train Samples: {len(continuous_train_losses)}', 
                        xy=(0.02, 0.25), xycoords='axes fraction')
            plt.annotate(f'Val Samples: {len(continuous_val_losses)}', 
                        xy=(0.02, 0.30), xycoords='axes fraction')
        
        detailed_plot_path = os.path.join(args.output_dir, 'detailed_loss_curve.png')
        
        all_detailed_loss_data = {
            "train_losses": continuous_train_losses,
            "val_losses": continuous_val_losses,
            "train_tokens": continuous_train_tokens,
            "val_tokens": continuous_val_tokens,
            "smoothed_train_losses": smoothed_train_losses,
            "smoothed_val_losses": smoothed_val_losses,
            "smoothed_train_tokens": smoothed_train_tokens,
            "smoothed_val_tokens": smoothed_val_tokens,
        }
        
        total_params, prefix = sum(p.numel() for p in model.parameters()), ""
        
        for value in ["K", "M", "B"]:
            if total_params < 1000:
                break
            total_params /= 1000
            prefix = value
        
        with open(os.path.join(args.output_dir, 'detailed_loss_data.json'), 'w') as f:
            f.write(safe_serialize({
                "name": f"{args.model_label} {total_params:.1f}{prefix}",
                "dataset_hash": full_dataset.hash,
                "data_loaders_thorough_hash": hash_data_loaders_thorough,
                "all_data": all_detailed_loss_data
            }, indent=4))
        
        plt.savefig(detailed_plot_path)
        plt.close()
        logger.info(f"Detailed loss curve saved to {detailed_plot_path}")

    return model, optimizer, scheduler, final_checkpoint_path

if __name__ == "__main__":
    main() 