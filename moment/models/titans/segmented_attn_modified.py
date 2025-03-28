from __future__ import annotations
from typing import Callable

from math import ceil
from copy import deepcopy
from functools import partial
from collections import namedtuple

import tqdm

import torch
from torch import nn, stack, cat
import torch.nn.functional as F
from torch.nn import Module, ModuleList, Linear

# flex attention
# https://pytorch.org/blog/flexattention/

flex_attention = None

try:
    from torch.nn.attention.flex_attention import flex_attention, create_block_mask
    if torch.cuda.is_available():
        flex_attention = torch.compile(flex_attention)
except ImportError:
    pass

def create_mac_block_mask(seq_len, window_size, persist_mem_len, sliding = False):

    def create_mac_mask(_, __, q_idx, kv_idx):
        is_persist_mem = kv_idx < persist_mem_len
        kv_without_mem = kv_idx - persist_mem_len
        causal_mask = q_idx >= kv_without_mem

        if not sliding:
            block_diagonal = (q_idx // window_size) == (kv_without_mem // window_size)
            causal_mask = causal_mask & block_diagonal
        else:
            sliding_mask = (q_idx - kv_without_mem) <= window_size
            causal_mask = causal_mask & sliding_mask

        return is_persist_mem | (~is_persist_mem & causal_mask)

    block_mask = create_block_mask(create_mac_mask, B = None, H = None, Q_LEN = seq_len, KV_LEN = seq_len + persist_mem_len, _compile = True)
    return block_mask

# einstein notation related

from einops import repeat, rearrange, pack, unpack, einsum
from einops.layers.torch import Rearrange

# b - batch
# n - sequence
# h - heads
# d - feature dimension

# absolute and relative positions

from axial_positional_embedding import ContinuousAxialPositionalEmbedding
from rotary_embedding_torch import RotaryEmbedding

# hyper connections / attend from x-transformers, which handles different queries and key lengths better

from x_transformers.attend import Attend

from hyper_connections import get_init_and_expand_reduce_stream_functions

# proposed neural memory

from titans_pytorch.neural_memory import NeuralMemory

# constants

LinearNoBias = partial(Linear, bias = False)

AttnIntermediates = namedtuple('AttnIntermediates', ('value_residual', 'cached_key_values'))

# helpers

def exists(v):
    return v is not None

def default(v, d):
    return v if exists(v) else d

def identity(t):
    return t

def divisible_by(num, den):
    return (num % den) == 0

def round_up_multiple(seq, mult):
    return ceil(seq / mult) * mult

def round_down_multiple(seq, mult):
    return seq // mult * mult

def pack_with_inverse(t, pattern):
    packed, packed_shape = pack(t, pattern)

    def inverse(out, inv_pattern = None):
        return unpack(out, packed_shape, default(inv_pattern, pattern))

    return packed, inverse

def pad_at_dim(t, pad, dim = -1, value = 0.):
    dims_from_right = (- dim - 1) if dim < 0 else (t.ndim - dim - 1)
    zeros = ((0, 0) * dims_from_right)
    return F.pad(t, (*zeros, *pad), value = value)

def pad_and_segment_with_inverse(
    seq,
    segment_len,
    fold_into_batch = True,
    inverse_remove_pad = True
):
    batch, seq_len = seq.shape[:2]
    next_seq_len_mult = round_up_multiple(seq_len, segment_len)

    padding = next_seq_len_mult - seq_len
    needs_pad = padding > 0

    if needs_pad:
        seq = F.pad(seq, (0, 0, 0, padding))

    if fold_into_batch:
        seq = rearrange(seq, 'b (w n) d -> (b w) n d', n = segment_len)

    def inverse(out):

        if fold_into_batch:
            out = rearrange(out, '(b w) ... n d -> b ... (w n) d', b = batch)

        if needs_pad and inverse_remove_pad:
            out = out[..., :-padding, :]

        return out

    return seq, inverse

# sampling related

def log(t, eps = 1e-20):
    return torch.log(t.clamp(min = eps))

def gumbel_noise(t):
    noise = torch.rand_like(t)
    return -log(-log(noise))

def gumbel_sample(t, temperature = 1.):
    if temperature > 0.:
        t = t / temperature + gumbel_noise(t)
    return t.argmax(dim = -1, keepdim = True)

# min_p
# https://arxiv.org/abs/2407.01082

def min_p_filter(logits, min_p = 0.1):
    probs = logits.softmax(dim = -1)
    max_probs = probs.amax(dim = -1, keepdim = True)
    limit = min_p * max_probs
    return torch.where(probs < limit, float('-inf'), logits)

# feedforward and attention

class GEGLU(Module):
    def forward(self, x):
        x, gate = x.chunk(2, dim = -1)
        return F.silu(gate) * x

def FeedForward(dim, mult = 4):
    dim_inner = int(dim * mult * 2 / 3)

    return nn.Sequential(
        nn.RMSNorm(dim),
        nn.Linear(dim, dim_inner * 2),
        GEGLU(),
        nn.Linear(dim_inner, dim)
    )

class SegmentedAttention(Module):
    def __init__(
        self,
        dim,
        segment_len,
        num_persist_mem_tokens = 0,
        num_longterm_mem_tokens = 0,
        dim_head = 64,
        heads = 8,
        sliding = False,
        accept_value_residual = False,
        attend_kwargs: dict = dict(),
        use_flex_attn = False
    ):
        super().__init__()
        self.norm = nn.RMSNorm(dim)

        dim_inner = dim_head * heads

        self.rotary_emb = RotaryEmbedding(dim_head)

        self.attend = Attend(causal = True, **attend_kwargs)

        self.to_qkv = LinearNoBias(dim, dim_inner * 3)
        self.to_out = LinearNoBias(dim_inner, dim)

        self.to_learned_v_mix = nn.Sequential(
            nn.Linear(dim, heads),
            Rearrange('b n h -> b h n 1'),
            nn.Sigmoid()
        ) if accept_value_residual else None

        self.segment_len = segment_len
        self.num_longterm_mem_tokens = num_longterm_mem_tokens

        total_segment_len = segment_len + num_longterm_mem_tokens
        self.total_segment_len = total_segment_len

        self.sliding = sliding # sliding window attn - doubt their non-sliding results being the best. local attention with overlapping windows is very strong

        self.split_heads = Rearrange('b n (h d) -> b h n d', h = heads)
        self.merge_heads = Rearrange('b h n d -> b n (h d)')

        self.persistent_memory = nn.Parameter(torch.zeros(2, heads, num_persist_mem_tokens, dim_head))

        # flex attn related

        assert not (use_flex_attn and not exists(flex_attention)), 'you need to be on the latest pytorch with a cuda device available'
        self.use_flex_attn = use_flex_attn

        self.segment_len = segment_len
        self.num_persist_mem_tokens = num_persist_mem_tokens

    def forward_inference(
        self,
        token,
        cache,
        value_residual = None,
        output_gating = None,
    ):
        batch = token.shape[0]

        # attention

        token = self.norm(token)

        q, k, v = self.to_qkv(token).chunk(3, dim = -1)
        q, k, v = map(self.split_heads, (q, k, v))

        # value residual

        orig_v = v

        if exists(self.to_learned_v_mix):
            mix = self.to_learned_v_mix(token)
            v = v.lerp(value_residual, mix)

        # caching

        ck, cv = cache
        k = cat((ck, k), dim = -2)
        v = cat((cv, v), dim = -2)

        next_cache = (k, v)

        # relative positions

        q, k = self.rotary_emb.rotate_queries_with_cached_keys(q, k)

        # fold

        q, k, v = tuple(rearrange(t, 'b h n d -> b h n d') for t in (q, k, v))

        # take care of persistent memory key / values

        pmk, pmv = repeat(self.persistent_memory, 'kv ... -> kv b ...', b = k.shape[0])

        # persistent memory

        k = cat((pmk, k), dim = -2)
        v = cat((pmv, v), dim = -2)

        # attention

        out, _ = self.attend(q, k, v)

        out = self.merge_heads(out)

        out = self.to_out(out)

        if exists(output_gating):
            out = out * output_gating

        return out, AttnIntermediates(orig_v, next_cache)

    def forward_flex(
        self,
        seq,
        value_residual = None,
        flex_attn_fn: Callable | None = None,
        output_gating = None,
        cache = None
    ):

        assert not (exists(value_residual) ^ exists(self.to_learned_v_mix))

        batch, seq_len = seq.shape[:2]

        # attention

        seq = self.norm(seq)

        q, k, v = self.to_qkv(seq).chunk(3, dim = -1)
        q, k, v = map(self.split_heads, (q, k, v))

        # value residual

        orig_v = v

        if exists(self.to_learned_v_mix):
            mix = self.to_learned_v_mix(seq)
            v = v.lerp(value_residual, mix)

        # caching

        next_cache = (k, v)

        # take care of persistent memory key / values

        pmk, pmv = repeat(self.persistent_memory, 'kv h n d -> kv b h n d', b = batch)

        # relative positions

        q, k = self.rotary_emb.rotate_queries_with_cached_keys(q, k)

        # persistent memory

        k = cat((pmk, k), dim = -2)
        v = cat((pmv, v), dim = -2)

        # prep flex attention

        if not exists(flex_attn_fn):
            block_mask = create_mac_block_mask(seq_len, self.total_segment_len, self.num_persist_mem_tokens, self.sliding)

            flex_attn_fn = partial(flex_attention, block_mask = block_mask)

        
        # --- FIX: Ensure k and v match q's dtype ---
        k = k.to(q.dtype)
        v = v.to(q.dtype)
        # --- END FIX ---
        
        
        # attention

        out = flex_attn_fn(q, k, v)

        out = self.merge_heads(out)

        out = self.to_out(out)

        if exists(output_gating):
            out = out * output_gating

        return out, AttnIntermediates(orig_v, next_cache)

    # --- MODIFICATION 1: Add mask to signature ---
    def forward(
        self,
        seq,
        value_residual = None,
        flex_attn_fn: Callable | None = None,
        disable_flex_attn = False,
        output_gating = None,
        cache = None,
        mask = None # ADDED mask=None argument
    ):
        is_inferencing = exists(cache)

        if is_inferencing:
            # Inference path doesn't use the explicit mask argument here
            return self.forward_inference(seq, cache, value_residual, output_gating = output_gating)

        if seq.is_cuda and self.use_flex_attn and not disable_flex_attn:
            # Flex path uses flex_attn_fn which has block_mask, doesn't need explicit mask here
            return self.forward_flex(seq, value_residual, flex_attn_fn, output_gating = output_gating, cache = cache)

        # --- Non-flex, non-inference path below ---

        assert not (exists(value_residual) ^ exists(self.to_learned_v_mix))

        segment_len, num_longterm_mem_tokens = self.segment_len, self.num_longterm_mem_tokens
        total_segment_len = segment_len + num_longterm_mem_tokens

        batch, seq_len = seq.shape[:2]

        seq, inverse_segment = pad_and_segment_with_inverse(seq, total_segment_len, fold_into_batch = False)

        seq = self.norm(seq)

        q, k, v = self.to_qkv(seq).chunk(3, dim = -1)
        q, k, v = map(self.split_heads, (q, k, v))

        orig_v = v
        if exists(self.to_learned_v_mix):
            mix = self.to_learned_v_mix(seq)
            v = v.lerp(value_residual, mix)

        next_cache = tuple(map(inverse_segment, (k, v)))
        q, k = self.rotary_emb.rotate_queries_with_cached_keys(q, k)
        q, k, v = tuple(rearrange(t, 'b h (w n) d -> (b w) h n d', n = total_segment_len) for t in (q, k, v))

        # --- MODIFICATION 2: Pass received `mask` to self.attend ---
        # Remove or comment out the internal attend_kwargs mask logic for clarity,
        # as the passed `mask` (padding) should be sufficient.
        # attend_kwargs = dict()
        # if self.sliding:
        #    ... calculate sliding_mask ...
        #    attend_kwargs.update(mask = sliding_mask) # COMMENT OUT or REMOVE

        pmk, pmv = repeat(self.persistent_memory, 'kv ... -> kv b ...', b = k.shape[0])
        k = cat((pmk, k), dim = -2)
        v = cat((pmv, v), dim = -2)

        # Pass the `mask` argument received by this forward method directly to self.attend
        out, _ = self.attend(q, k, v, mask = mask) # Use the passed mask argument here
        # --- END MODIFICATION 2 ---


        out = self.merge_heads(out)
        out = self.to_out(out)
        out = rearrange(out, '(b w) n d -> b (w n) d', b = batch)
        out = inverse_segment(out)

        if exists(output_gating):
            out = out * output_gating

        return out, AttnIntermediates(orig_v, next_cache)