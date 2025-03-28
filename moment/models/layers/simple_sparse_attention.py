import torch
import torch.nn as nn
import torch.nn.functional as F
import logging
import math


class SimpleSparseAttention(nn.Module):
    """
    A simplified implementation of sparse attention using sliding windows and top-k selection.
    This serves as a fallback when the native_sparse_attention_pytorch module isn't available
    or is incompatible with the current PyTorch version.
    """
    def __init__(
        self,
        dim,
        dim_head=64,
        heads=8,
        sliding_window_size=16,
        top_k_ratio=0.1,
        dropout=0.0,
    ):
        super().__init__()
        self.dim = dim
        self.dim_head = dim_head
        self.heads = heads
        self.sliding_window_size = sliding_window_size
        self.top_k_ratio = top_k_ratio
        self.scale = dim_head ** -0.5
        
        inner_dim = dim_head * heads
        
        self.to_qkv = nn.Linear(dim, inner_dim * 3, bias=False)
        self.to_out = nn.Sequential(
            nn.Linear(inner_dim, dim),
            nn.Dropout(dropout)
        )
        
        logging.info(f"Initialized SimpleSparseAttention with dim={dim}, heads={heads}, dim_head={dim_head}, window={sliding_window_size}")
        
    def forward(self, x, mask=None):
        """
        Args:
            x: Input tensor of shape [batch_size, seq_len, dim]
            mask: Attention mask of shape [batch_size, seq_len]
        
        Returns:
            Tensor of shape [batch_size, seq_len, dim]
        """
        b, n, d = x.shape
        h = self.heads
        
        # Generate QKV
        qkv = self.to_qkv(x).chunk(3, dim=-1)
        q, k, v = map(lambda t: t.reshape(b, n, h, self.dim_head).transpose(1, 2), qkv)
        
        # Scaled dot-product
        dots = torch.matmul(q, k.transpose(-1, -2)) * self.scale
        
        # Apply sliding window attention - restrict attention to nearby tokens
        if self.sliding_window_size > 0:
            window_mask = self._get_sliding_window_mask(n, self.sliding_window_size, dots.device)
            window_mask = window_mask.unsqueeze(0).unsqueeze(0)  # [1, 1, seq_len, seq_len]
            dots = dots.masked_fill(~window_mask, -torch.finfo(dots.dtype).max)
        
        # Apply attention mask if provided
        if mask is not None:
            mask = mask.unsqueeze(1).unsqueeze(1)  # [batch_size, 1, 1, seq_len]
            dots = dots.masked_fill(~mask, -torch.finfo(dots.dtype).max)
        
        # Apply top-k sparsity if specified
        if 0 < self.top_k_ratio < 1:
            k_value = max(1, int(n * self.top_k_ratio))
            top_k = torch.topk(dots, k=k_value, dim=-1)[0]
            threshold = top_k[..., -1, None]
            top_k_mask = dots >= threshold
            dots = dots.masked_fill(~top_k_mask, -torch.finfo(dots.dtype).max)
        
        # Apply softmax
        attn = F.softmax(dots, dim=-1)
        
        # Apply attention to values
        out = torch.matmul(attn, v)
        out = out.transpose(1, 2).reshape(b, n, h * self.dim_head)
        
        return self.to_out(out)
    
    def _get_sliding_window_mask(self, seq_len, window_size, device):
        """
        Generate a mask for sliding window attention.
        
        Args:
            seq_len: Length of the sequence
            window_size: Size of the sliding window (one-sided)
            device: Device to create the mask on
            
        Returns:
            Boolean mask of shape [seq_len, seq_len] where True indicates valid attention
        """
        mask = torch.zeros(seq_len, seq_len, dtype=torch.bool, device=device)
        for i in range(seq_len):
            start = max(0, i - window_size)
            end = min(seq_len, i + window_size + 1)
            mask[i, start:end] = True
        return mask 