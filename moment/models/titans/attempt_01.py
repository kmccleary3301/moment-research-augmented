import logging
import warnings
from argparse import Namespace
from math import ceil
from dataclasses import dataclass
from typing import Callable, Optional

import torch
from torch import nn, stack, cat
import torch.nn.functional as F
from torch.nn import Module, ModuleList, Linear
from functools import partial

from einops import rearrange, repeat

# --- Re-use components from MAC Transformer ---
# (Assuming MAC Transformer code is available in the environment)
# Need: SegmentedAttention, FeedForward, NeuralMemory (optional), RotaryEmbedding, Attend, etc.
# Import necessary components - adjust paths if needed


from titans_pytorch.mac_transformer import ( # Assuming the provided code is saved as titans_pytorch.py
    FeedForward, NeuralMemory,
    LinearNoBias, GEGLU, AttnIntermediates,
    exists, default, divisible_by, round_up_multiple,
    pad_at_dim, pad_and_segment_with_inverse, pack_with_inverse,
    create_mac_block_mask, flex_attention # Make sure flex_attention is imported if used
)
from .segmented_attn_modified import SegmentedAttention
from rotary_embedding_torch import RotaryEmbedding
from axial_positional_embedding import ContinuousAxialPositionalEmbedding # Keep for potential future use? Unlikely needed.
from hyper_connections import get_init_and_expand_reduce_stream_functions
from titans_pytorch import MemoryAttention, MemoryMLP
# --- End MAC Component Imports ---

# --- Re-use components from MOMENT ---
# (Assuming MOMENT code is available in the environment)
from moment.common import TASKS
# from moment.data.base import TimeseriesOutputs # Provided in the prompt
from moment.utils.masking import Masking
from moment.utils.utils import NamespaceWithDefaults
from moment.models.moment import ( # Adjust import path if needed
    PretrainHead, ClassificationHead, ForecastingHead,
    PatchEmbedding, Patching, RevIN
)
from moment.data.base import TimeseriesOutputs
# --- End MOMENT Component Imports ---

class MAC_Moment(nn.Module):
    def __init__(self, configs: Namespace | dict, **kwargs: dict):
        super().__init__()
        configs = self._update_inputs(configs, **kwargs)
        
        self.use_flex_attn = configs.getattr("mac_use_flex_attn", False) and exists(flex_attention)
        
        configs = self._validate_inputs(configs) # Add custom validations if needed
        self.configs = configs
        self.task_name = configs.task_name
        self.seq_len = configs.seq_len
        self.patch_len = configs.patch_len
        self.d_model = configs.d_model
        self.num_input_channels = configs.num_input_channels # Assuming num_input_channels is in configs

        # --- MOMENT Input Processing ---
        self.normalizer = RevIN(
            num_features=1, # MOMENT processes channel-independently before encoder
            affine=configs.getattr("revin_affine", False)
        )
        self.tokenizer = Patching(
            patch_len=configs.patch_len, stride=configs.patch_stride_len
        )
        self.patch_embedding = PatchEmbedding(
            d_model=configs.d_model,
            seq_len=configs.seq_len, # Needed for PE calculation if used
            patch_len=configs.patch_len,
            stride=configs.patch_stride_len,
            dropout=configs.getattr("dropout", 0.1),
            add_positional_embedding=configs.getattr("add_positional_embedding", True),
            value_embedding_bias=configs.getattr("value_embedding_bias", False),
            orth_gain=configs.getattr("orth_gain", 1.41),
        ).to(configs.device)
        self.mask_generator = Masking(mask_ratio=configs.getattr("mask_ratio", 0.0))

        # --- MAC Transformer Core ---
        self.depth = configs.getattr("mac_depth", 8)
        self.heads = configs.getattr("mac_heads", 8)
        self.dim_head = configs.getattr("mac_dim_head", self.d_model // self.heads)
        self.ff_mult = configs.getattr("mac_ff_mult", 4)
        self.num_persist_mem_tokens = configs.getattr("mac_num_persist_mem_tokens", 0)
        # Omit longterm_mems for simplicity
        # self.num_longterm_mem_tokens = configs.getattr("mac_num_longterm_mem_tokens", 0)
        self.num_longterm_mem_tokens = 0 # Hardcoded omission

        # Segmentation must be in terms of patches
        self.mac_segment_len_patches = configs.getattr("mac_segment_len_patches", 32) # e.g., 32 patches per segment

        self.sliding_window_attn = configs.getattr("mac_sliding_window_attn", False)
        

        # Neural Memory Config (optional)
        self.use_neural_memory = configs.getattr("mac_use_neural_memory", False)
        self.neural_memory_layers = configs.getattr("mac_neural_memory_layers", tuple(range(1, self.depth + 1))) # Default: all layers
        self.neural_mem_segment_len_patches = configs.getattr("mac_neural_mem_segment_len_patches", self.mac_segment_len_patches) # Default: same as attn segment
        self.neural_mem_batch_size = configs.getattr("mac_neural_mem_batch_size", None)
        self.neural_mem_gate_attn_output = configs.getattr("mac_neural_mem_gate_attn_output", False)
        self.neural_mem_weight_residual = configs.getattr("mac_neural_mem_weight_residual", False)
        self.neural_mem_kwargs = configs.getattr("mac_neural_memory_kwargs", {})
        self.neural_memory_model_config = configs.getattr("mac_neural_memory_model_config", None) # Config to build the mem model


        # Hyper-connections (optional, default off for simplicity)
        self.num_residual_streams = configs.getattr("mac_num_residual_streams", 1)
        init_hyper_conn, self.expand_streams, self.reduce_streams = get_init_and_expand_reduce_stream_functions(
            self.num_residual_streams, dim=self.d_model, add_stream_embed=True, disable=self.num_residual_streams == 1
        )

        self.layers = ModuleList([])
        is_first_neural_mem = True

        for layer_idx in range(self.depth):
            layer_num = layer_idx + 1
            is_first_layer = layer_num == 1

            attn = SegmentedAttention(
                dim=self.d_model,
                dim_head=self.dim_head,
                heads=self.heads,
                segment_len=self.mac_segment_len_patches, # Use patch-based segment length
                use_flex_attn=self.use_flex_attn,
                accept_value_residual=not is_first_layer, # Allow value residual after first layer
                num_longterm_mem_tokens=self.num_longterm_mem_tokens, # = 0
                num_persist_mem_tokens=self.num_persist_mem_tokens,
                sliding=self.sliding_window_attn
            )

            mem = None
            mem_hyper_conn = None
            mem_qkv_layer_selector = None # Simpler: don't use qkv selector initially

            if self.use_neural_memory and layer_num in self.neural_memory_layers:
                mem_hyper_conn = init_hyper_conn(add_branch_out_to_residual=not self.neural_mem_gate_attn_output)

                # Build neural memory model if config provided
                neural_memory_model = None
                if self.neural_memory_model_config is not None:
                    # Example: Assuming config specifies model type and params
                    # This part needs customization based on how you want to configure the memory model
                    if self.neural_memory_model_config.type == 'mlp':
                         
                        neural_memory_model = MemoryMLP(dim=self.d_model, depth=self.neural_memory_model_config.depth)
                    elif self.neural_memory_model_config.type == 'attn':
                         
                        neural_memory_model = MemoryAttention(dim=self.d_model, dim_head=64, heads=4)
                    # Add more types as needed
                    logging.info(f"Built Neural Memory model: {self.neural_memory_model_config.type}")


                mem = NeuralMemory(
                    dim=self.d_model,
                    chunk_size=self.neural_mem_segment_len_patches, # Use patch-based chunk size
                    batch_size=self.neural_mem_batch_size,
                    model=neural_memory_model, # Pass the built model
                    qkv_receives_diff_views=False, # Simpler: disable qkv selection
                    accept_weight_residual=self.neural_mem_weight_residual and not is_first_neural_mem,
                    **self.neural_mem_kwargs
                )
                is_first_neural_mem = False

            ff = FeedForward(dim=self.d_model, mult=self.ff_mult)

            self.layers.append(ModuleList([
                mem_hyper_conn,
                init_hyper_conn(), # attn hyper conn
                init_hyper_conn(), # ff hyper conn
                mem_qkv_layer_selector, # = None
                mem,
                attn,
                ff,
            ]))

        self.final_norm = nn.RMSNorm(self.d_model) # Use RMSNorm consistent with MAC

        # --- MOMENT Output Head ---
        self.head = self._get_head(self.task_name) # Reuse MOMENT's head selection

    # --- Helper methods copied/adapted from MOMENT ---
    def _update_inputs(
        self, configs: Namespace | dict, **kwargs
    ) -> NamespaceWithDefaults:
        # Combine base configs and model_kwargs if provided
        combined_configs = {}
        if isinstance(configs, Namespace):
            combined_configs.update(vars(configs))
        elif isinstance(configs, dict):
            combined_configs.update(configs)

        if "model_kwargs" in kwargs:
             combined_configs.update(kwargs["model_kwargs"])

        # Ensure essential MOMENT configs are present
        defaults = {
            'task_name': TASKS.PRETRAINING,
            'seq_len': 512,
            'patch_len': 8,
            'patch_stride_len': 8,
            'd_model': 512,
            'dropout': 0.1,
            'revin_affine': False,
            'add_positional_embedding': True,
            'value_embedding_bias': False,
            'orth_gain': 1.41,
            'mask_ratio': 0.4,
            'num_input_channels': 512, # Default, should be overridden by data
            'device': 'cuda' if torch.cuda.is_available() else 'cpu',
            'debug': False,
            # Add MAC specific defaults if needed
            'mac_depth': 8,
            'mac_heads': 8,
            'mac_num_persist_mem_tokens': 4,
            'mac_segment_len_patches': 16,
             'mac_use_neural_memory': False, # Default off
        }
        return NamespaceWithDefaults(_defaults=defaults, **combined_configs)


    def _validate_inputs(self, configs: NamespaceWithDefaults) -> NamespaceWithDefaults:
        # Add specific validation for MAC_Moment if necessary
        if configs.patch_stride_len != configs.patch_len:
            warnings.warn("Patch stride length is not equal to patch length.")
        if not divisible_by(configs.d_model, configs.mac_heads):
             warnings.warn(f"d_model ({configs.d_model}) is not divisible by mac_heads ({configs.mac_heads}).")
             configs.mac_dim_head = configs.d_model // configs.mac_heads # Recalculate dim_head
        else:
             configs.mac_dim_head = default(configs.getattr("mac_dim_head"), configs.d_model // configs.mac_heads)

        if self.use_flex_attn and not torch.cuda.is_available():
            warnings.warn("FlexAttention requested but CUDA is not available. Disabling FlexAttention.")
            self.use_flex_attn = False
            configs.mac_use_flex_attn = False # Update config state

        return configs

    def _get_head(self, task_name: str) -> nn.Module:
        # Directly reuse MOMENT's head logic
        head_dropout = self.configs.getattr("head_dropout", self.configs.getattr("dropout", 0.1))

        if task_name in {
            TASKS.PRETRAINING,
            TASKS.ANOMALY_DETECTION,
            TASKS.IMPUTATION,
        } or (
            task_name == TASKS.SHORT_HORIZON_FORECASTING
            and self.configs.getattr("finetuning_mode", "linear-probing") == "zero-shot" # Check finetuning_mode
        ):
            return PretrainHead(
                self.configs.d_model,
                self.configs.patch_len,
                head_dropout,
                self.configs.getattr("orth_gain", 1.41),
            )
        elif task_name == TASKS.CLASSIFICATION:
            # Ensure num_class is defined in configs for classification
            if not hasattr(self.configs, 'num_class'):
                 raise ValueError("Configuration must include 'num_class' for classification task.")
            return ClassificationHead(
                self.configs.n_channels,
                self.configs.d_model,
                self.configs.num_class,
                head_dropout,
            )
        elif (task_name == TASKS.LONG_HORIZON_FORECASTING) or (
            task_name == TASKS.SHORT_HORIZON_FORECASTING
            and self.configs.getattr("finetuning_mode", "linear-probing") != "zero-shot"
        ):
             # Ensure forecast_horizon is defined
            if not hasattr(self.configs, 'forecast_horizon'):
                 raise ValueError("Configuration must include 'forecast_horizon' for forecasting task.")

            # Calculate num_patches correctly based on MOMENT's PatchEmbedding logic
            num_patches = self.patch_embedding.num_patches
            # Original MOMENT calculation (might be slightly different if seq_len < patch_len)
            # num_patches = (max(self.configs.seq_len, self.configs.patch_len) - self.configs.patch_len) // self.configs.patch_stride_len + 1

            head_nf = self.configs.d_model * num_patches
            return ForecastingHead(
                head_nf,
                self.configs.forecast_horizon,
                head_dropout,
            )
        else:
            raise NotImplementedError(f"Task {task_name} head not implemented.")

    # --- Main Forward Function ---
    def forward(
        self,
        x_enc: torch.Tensor,                # Input: [batch_size x n_channels x seq_len]
        input_mask: Optional[torch.Tensor] = None, # Padding mask: [batch_size x seq_len]
        mask: Optional[torch.Tensor] = None,       # Pretraining mask: [batch_size x seq_len]
        **kwargs, # Allow for extra arguments like in MOMENT
    ):
        # Default masks if not provided (like in MOMENT)
        if input_mask is None:
             input_mask = torch.ones((x_enc.shape[0], self.seq_len), device=x_enc.device, dtype=torch.long)
        if mask is None and self.task_name == TASKS.PRETRAINING:
             mask = self.mask_generator.generate_mask(x=x_enc, input_mask=input_mask)
             mask = mask.to(x_enc.device)
        elif mask is None: # For tasks other than pretraining, assume no masking unless provided
             mask = torch.ones((x_enc.shape[0], self.seq_len), device=x_enc.device, dtype=torch.long)


        # --- Input Processing (MOMENT Style) ---
        batch_size, n_channels, seq_len = x_enc.shape
        # assert n_channels == self.num_input_channels, f"Input channel mismatch: expected {self.num_input_channels}, got {n_channels}"
        assert seq_len == self.seq_len, f"Input sequence length mismatch: expected {self.seq_len}, got {seq_len}"

        # 1. Normalization (use combined mask for normalization stats)
        norm_mask = mask * input_mask if self.task_name == TASKS.PRETRAINING else input_mask
        x_enc = self.normalizer(x=x_enc, mask=norm_mask, mode="norm")
        x_enc = torch.nan_to_num(x_enc, nan=0, posinf=0, neginf=0) # Handle potential NaNs

        # 2. Tokenize (Patching)
        x_patched = self.tokenizer(x=x_enc) # [B, C, NumPatches, PatchLen]

        # 3. Embed Patches (handles pretraining mask internally)
        # The mask passed here determines which patches become mask tokens
        enc_in = self.patch_embedding(x_patched, mask=mask) # [B, C, NumPatches, d_model]
        num_patches = enc_in.shape[2]

        # 4. Reshape for Transformer: [B*C, NumPatches, d_model]
        enc_in = rearrange(enc_in, 'b c n d -> (b c) n d')

        # 5. Create Attention Mask (from input_mask, for padding)
        # Convert seq mask to patch mask, repeat for channels
        attention_mask = Masking.convert_seq_to_patch_view(
            input_mask, self.patch_len
        ).repeat_interleave(n_channels, dim=0) # [B*C, NumPatches]

        # --- MAC Transformer Layers ---
        value_residual = None # For SegmentedAttention
        mem_weight_residual = None # For NeuralMemory
        # mem_input_layers = [] # Not using QKV selector for now

        # Prep Flex Attention (if used)
        flex_attn_fn = None
        if self.use_flex_attn and enc_in.is_cuda:
            # Need to ensure create_mac_block_mask works with NumPatches
            # Assuming NumPatches is the sequence length for the MAC layers
            block_mask = create_mac_block_mask(
                 num_patches, # Q_LEN = NumPatches
                 self.mac_segment_len_patches, # window_size = segment length in patches
                 self.num_persist_mem_tokens,
                 self.sliding_window_attn
            )
            # flex_attention function is imported and potentially compiled
            flex_attn_fn = partial(flex_attention, block_mask=block_mask)


        # Apply hyper-connection stream expansion if needed
        enc_in = self.expand_streams(enc_in) # Identity if num_streams=1

        # Iterate through MAC layers
        for layer_idx, (mem_hyper_conn, attn_hyper_conn, ff_hyper_conn, _, mem, attn, ff) in enumerate(self.layers):
            layer_num = layer_idx + 1
            retrieved = None
            attn_out_gates = None
            # next_neural_mem_cache = None # Not handling caching here

            # Neural Memory (if enabled and configured for this layer)
            if exists(mem) and exists(mem_hyper_conn):
                mem_input, add_residual = mem_hyper_conn(enc_in)

                # Simplified: Use mem_input for Q, K, V directly
                qkv_mem_input = stack((mem_input, mem_input, mem_input))

                # Note: Caching for NeuralMemory state is not implemented here for simplicity
                # It would require passing/returning state through the forward call
                retrieved, mem_state_update = mem.forward(
                    qkv_mem_input,
                    # state = None, # No cache handling
                    prev_weights = mem_weight_residual
                )

                if self.neural_mem_weight_residual:
                    mem_weight_residual = mem_state_update.updates # Assuming .updates holds the weights/residuals

                if self.neural_mem_gate_attn_output:
                    attn_out_gates = retrieved.sigmoid()
                else:
                    enc_in = add_residual(retrieved) # Add memory output back to residual stream

            # Attention
            attn_in, add_residual = attn_hyper_conn(enc_in)

            # Pass attention_mask to SegmentedAttention
            # SegmentedAttention needs to handle the mask format correctly
            # MAC's default Attend class expects mask [B, H, Nq, Nk] or similar
            # We have [B*C, Np]. Need to adapt or ensure Attend handles it.
            # Let's assume SegmentedAttention's internal Attend handles broadcast/conversion.
            # If using FlexAttention, the block_mask in flex_attn_fn handles masking.
            attn_kwargs = {}
            # --- MODIFICATION START ---
            # Remove the attn_kwargs logic for the mask
            # attn_kwargs = {} # No longer needed for mask
            # if not self.use_flex_attn:
            #     attn_kwargs['mask'] = attention_mask # DELETE

            attn_out, (values, _) = attn( # Ignore cache output from attn
                attn_in,
                value_residual=value_residual,
                flex_attn_fn=flex_attn_fn,
                output_gating=attn_out_gates,
                # Pass mask directly if not using flex, otherwise None (flex handles mask via block_mask)
                mask=attention_mask if not self.use_flex_attn else None
                # **attn_kwargs # DELETE THIS UNPACKING
            )
            # --- MODIFICATION END ---

            # Store value residual *after* the first layer
            if layer_num > 0:
                 value_residual = default(value_residual, values) # Keep first value residual
                 # Or update: value_residual = values # Always use latest

            enc_in = add_residual(attn_out)

            # Feedforward
            ff_in, add_ff_residual = ff_hyper_conn(enc_in)
            ff_out = ff(ff_in)
            enc_in = add_ff_residual(ff_out)

        # Apply final normalization
        enc_out = self.final_norm(enc_in) # [B*C, NumPatches, d_model]

        # Reduce streams if hyper-connections were used
        enc_out = self.reduce_streams(enc_out) # Identity if num_streams=1

        # --- Output Head (MOMENT Style) ---
        # Reshape back to [B, C, NumPatches, d_model] for MOMENT heads
        enc_out = rearrange(enc_out, '(b c) n d -> b c n d', b=batch_size, c=n_channels)

        # Pass through the appropriate head
        # Heads expect different inputs, handle accordingly
        head_output = None
        embeddings_output = None
        anomaly_scores = None
        forecast_output = None
        reconstruction_output = None
        labels_output = None

        if self.task_name in {TASKS.PRETRAINING, TASKS.IMPUTATION, TASKS.ANOMALY_DETECTION} or \
           (self.task_name == TASKS.SHORT_HORIZON_FORECASTING and self.configs.getattr("finetuning_mode", "linear-probing") == "zero-shot"):
            # PretrainHead expects [B, C, Np, D] -> [B, C, SeqLen]
            dec_out = self.head(enc_out)
            # Denormalize reconstruction/forecast
            dec_out = self.normalizer(x=dec_out, mode="denorm")
            reconstruction_output = dec_out

            if self.task_name == TASKS.ANOMALY_DETECTION:
                 # Calculate anomaly scores (using original unnormalized input)
                 # Need original x_enc before normalization
                 # Recompute or pass original x_enc if needed. Assuming self.reconstruct handles this.
                 # For simplicity, let's calculate here. Need original x_enc.
                 # We'll skip anomaly score calculation here, assuming it's done post-forward if needed,
                 # or requires passing original x_enc.
                 pass # anomaly_scores = calculate_anomaly(original_x_enc, dec_out)
            if self.task_name == TASKS.SHORT_HORIZON_FORECASTING:
                 # Extract forecast part from reconstruction (as in MOMENT's short_forecast)
                 forecast_horizon = kwargs.get('forecast_horizon', self.configs.getattr('forecast_horizon', 1)) # Get horizon
                 num_masked_patches = ceil(forecast_horizon / self.patch_len)
                 num_masked_timesteps = num_masked_patches * self.patch_len
                 end = -num_masked_timesteps + forecast_horizon
                 end = None if end == 0 else end
                 forecast_output = dec_out[:, :, -num_masked_timesteps:end]


        elif self.task_name == TASKS.CLASSIFICATION:
            # ClassificationHead expects [B, C, Np, D] -> [B, n_classes]
            # It handles pooling internally
            labels_output = self.head(enc_out) # Direct output is logits/labels

        elif self.task_name == TASKS.LONG_HORIZON_FORECASTING or \
             (self.task_name == TASKS.SHORT_HORIZON_FORECASTING and self.configs.getattr("finetuning_mode", "linear-probing") != "zero-shot"):
             # ForecastingHead expects [B, C, Np, D] -> [B, C, Horizon]
             dec_out = self.head(enc_out)
             # Denormalize
             dec_out = self.normalizer(x=dec_out, mode="denorm")
             forecast_output = dec_out

        # Collect embeddings (e.g., mean over patches before head?)
        # Let's provide the final output of the MAC layers before the head
        # Pooling might be needed depending on embedding use case
        # Output: [B, C, NumPatches, d_model]
        # Embeddings can often be detached and moved to CPU/numpy for inspection/storage
        embeddings_output = enc_out.detach().cpu().numpy()

        # Convert masks to numpy as the pretraining class might expect this for final output collection
        # (Though it might be better to keep them tensors too if used in masking operations directly)
        # Let's keep them numpy for now based on the validation code snippet provided earlier.
        # input_mask_np = input_mask.cpu().numpy()
        # Ensure mask exists before converting
        # pretrain_mask_np = mask.cpu().numpy() if exists(mask) else None

        # Keep outputs needed for loss calculation as Tensors on the correct device
        # Do NOT detach or convert reconstruction/forecast/labels if they are used for loss/metrics
        # These should remain on the same device as the model/input tensors

        # Helper function ONLY for non-gradient-requiring outputs or final storage
        # to_numpy_safe = lambda x: x.detach().cpu().numpy() if exists(x) and isinstance(x, torch.Tensor) else x if exists(x) else None

        return TimeseriesOutputs(
            # --- Keep as Tensors ---
            forecast=forecast_output,            # Keep as Tensor
            reconstruction=reconstruction_output, # Keep as Tensor
            labels=labels_output,                # Keep as Tensor (if applicable)

            # --- Convert Masks (as per previous analysis of validation code) ---
            input_mask=input_mask,
            pretrain_mask=mask,

            # --- Anomaly Scores (depends, assume numpy for now) ---
            anomaly_scores=anomaly_scores,

            # --- Embeddings (often fine as numpy) ---
            embeddings=embeddings_output,

            metadata={},
            illegal_output=False
        )

    # Add other methods from MOMENT if needed (e.g., _check_model_weights_for_illegal_values)
    # Or specific methods for MAC components