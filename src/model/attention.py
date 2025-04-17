"""
Attention mechanisms for transformer models.

This module implements the Scaled Dot-Product Attention and Multi-Head Self-Attention
mechanisms as described in the paper "Attention Is All You Need".
"""

import torch
import torch.nn as nn
import torch.nn.functional as F
import math
from typing import Optional, Tuple


class ScaledDotProductAttention(nn.Module):
    """
    Scaled Dot-Product Attention mechanism.

    Computes attention weights using dot products between query and key vectors,
    scaled by the square root of the key dimension. These weights are then used
    to compute a weighted sum of the value vectors.

    Formula: Attention(Q, K, V) = softmax(QK^T / sqrt(d_k)) * V
    """

    def __init__(self, dropout: float = 0.1):
        """
        Initialize the Scaled Dot-Product Attention.

        Args:
            dropout: Dropout probability for attention weights
        """
        super().__init__()
        self.dropout = nn.Dropout(p=dropout)

    def forward(
        self,
        query: torch.Tensor,
        key: torch.Tensor,
        value: torch.Tensor,
        mask: Optional[torch.Tensor] = None,
        attention_scale: Optional[float] = None,
    ) -> tuple[torch.Tensor, torch.Tensor]:
        """
        Compute the attention output and attention weights.

        Args:
            query: Query tensor of shape (batch_size, num_queries, d_k)
            key: Key tensor of shape (batch_size, num_keys, d_k)
            value: Value tensor of shape (batch_size, num_keys, d_v)
            mask: Optional mask tensor of shape (batch_size, num_queries, num_keys)
                  or (batch_size, 1, num_keys) or (batch_size, 1, 1)
                  or (1, num_queries, num_keys) or broadcastable to the attention weights.
                  Values of True or 1 indicate positions to mask (set to -inf before softmax).
            attention_scale: Optional custom scale factor for attention scores.
                            If None, uses 1/sqrt(d_k).

        Returns:
            tuple containing:
                - output: Attention output of shape (batch_size, num_queries, d_v)
                - attention_weights: Attention weights of shape (batch_size, num_queries, num_keys)
        """
        # Get dimensions
        d_k = query.size(-1)

        # Compute attention scores: (batch_size, num_queries, num_keys)
        # matmul: (batch_size, num_queries, d_k) x (batch_size, d_k, num_keys)
        attention_scores = torch.matmul(query, key.transpose(-2, -1))

        # Scale attention scores
        scale = attention_scale if attention_scale is not None else 1.0 / math.sqrt(d_k)
        attention_scores = attention_scores * scale

        # Apply mask if provided
        if mask is not None:
            # Convert boolean mask to float mask where True/1 values become -inf
            if mask.dtype == torch.bool:
                attention_scores = attention_scores.masked_fill(mask, -1e9)
            elif mask.dtype == torch.int64 or mask.dtype == torch.int32 or mask.dtype == torch.int16 or mask.dtype == torch.int8:
                attention_scores = attention_scores.masked_fill(mask == 1, -1e9)

        # Apply softmax to get attention weights
        attention_weights = F.softmax(attention_scores, dim=-1)

        # Apply dropout
        attention_weights = self.dropout(attention_weights)

        # Compute output: (batch_size, num_queries, d_v)
        # matmul: (batch_size, num_queries, num_keys) x (batch_size, num_keys, d_v)
        output = torch.matmul(attention_weights, value)

        return output, attention_weights


def create_causal_mask(seq_length: int, device: torch.device = None) -> torch.Tensor:
    """
    Create a causal (triangular) mask for autoregressive decoding.

    This mask ensures that each position can only attend to previous positions
    and itself, preventing information flow from future positions.

    Args:
        seq_length: Length of the sequence
        device: Device to create the mask on

    Returns:
        Boolean mask of shape (1, seq_length, seq_length) where True values
        indicate positions to mask out (set to -inf in attention)
    """
    # Create a square matrix where each row i has 1s in columns i+1 and beyond
    # This represents positions that should be masked out
    mask = torch.triu(torch.ones(seq_length, seq_length, device=device), diagonal=1).bool()

    # Add batch dimension for broadcasting
    mask = mask.unsqueeze(0)  # Shape: (1, seq_length, seq_length)

    return mask


def create_padding_mask(src: torch.Tensor, pad_idx: int) -> torch.Tensor:
    """
    Create a mask for padding tokens.

    This mask ensures that padding tokens are not considered in attention calculations.

    Args:
        src: Input tensor of token indices of shape (batch_size, seq_length)
        pad_idx: Index of the padding token

    Returns:
        Boolean mask of shape (batch_size, 1, seq_length) where True values
        indicate padding positions to mask out (set to -inf in attention)
    """
    # Create mask where padding tokens are True
    mask = (src == pad_idx).bool()

    # Add a dimension for broadcasting across attention heads
    mask = mask.unsqueeze(1)  # Shape: (batch_size, 1, seq_length)

    return mask


class MultiHeadSelfAttention(nn.Module):
    """
    Multi-Head Self-Attention mechanism.

    This splits the input into multiple heads, applies scaled dot-product attention
    to each head independently, and then concatenates the results and projects them
    back to the original dimension.

    Formula: MultiHead(Q, K, V) = Concat(head_1, ..., head_h) * W_o
             where head_i = Attention(Q * W_q_i, K * W_k_i, V * W_v_i)
    """

    def __init__(self, d_model: int, num_heads: int, dropout: float = 0.1):
        """
        Initialize the Multi-Head Self-Attention.

        Args:
            d_model: Model dimension (must be divisible by num_heads)
            num_heads: Number of attention heads
            dropout: Dropout probability for attention weights
        """
        super().__init__()

        if d_model % num_heads != 0:
            raise ValueError(f"d_model ({d_model}) must be divisible by num_heads ({num_heads})")

        self.d_model = d_model
        self.num_heads = num_heads
        self.d_k = d_model // num_heads  # Dimension of each head

        # Linear projections for Q, K, V, and output
        self.W_q = nn.Linear(d_model, d_model)
        self.W_k = nn.Linear(d_model, d_model)
        self.W_v = nn.Linear(d_model, d_model)
        self.W_o = nn.Linear(d_model, d_model)

        # Scaled dot-product attention with dropout
        self.attention = ScaledDotProductAttention(dropout=dropout)

        # Output dropout
        self.dropout = nn.Dropout(p=dropout)

        # Initialize weights
        self._init_weights()

    def _init_weights(self):
        """
        Initialize the weights using Xavier uniform initialization.
        """
        gain = nn.init.calculate_gain('linear')
        for module in [self.W_q, self.W_k, self.W_v, self.W_o]:
            nn.init.xavier_uniform_(module.weight, gain=gain)
            nn.init.zeros_(module.bias)

    def split_heads(self, x: torch.Tensor) -> torch.Tensor:
        """
        Split the last dimension into (num_heads, d_k).

        Args:
            x: Tensor of shape (batch_size, seq_length, d_model)

        Returns:
            Tensor of shape (batch_size, num_heads, seq_length, d_k)
        """
        batch_size, seq_length, _ = x.size()

        # Reshape to (batch_size, seq_length, num_heads, d_k)
        x = x.view(batch_size, seq_length, self.num_heads, self.d_k)

        # Transpose to (batch_size, num_heads, seq_length, d_k)
        return x.transpose(1, 2)

    def combine_heads(self, x: torch.Tensor) -> torch.Tensor:
        """
        Combine the heads back into the original shape.

        Args:
            x: Tensor of shape (batch_size, num_heads, seq_length, d_k)

        Returns:
            Tensor of shape (batch_size, seq_length, d_model)
        """
        batch_size, _, seq_length, _ = x.size()

        # Transpose to (batch_size, seq_length, num_heads, d_k)
        x = x.transpose(1, 2)

        # Reshape to (batch_size, seq_length, d_model)
        return x.contiguous().view(batch_size, seq_length, self.d_model)

    def forward(
        self,
        query: torch.Tensor,
        key: torch.Tensor,
        value: torch.Tensor,
        mask: Optional[torch.Tensor] = None,
    ) -> Tuple[torch.Tensor, torch.Tensor]:
        """
        Compute the multi-head self-attention.

        Args:
            query: Query tensor of shape (batch_size, seq_length, d_model)
            key: Key tensor of shape (batch_size, seq_length, d_model)
            value: Value tensor of shape (batch_size, seq_length, d_model)
            mask: Optional mask tensor of shape (batch_size, 1, seq_length) or
                  (batch_size, seq_length, seq_length) or broadcastable to the
                  attention weights. Values of True or 1 indicate positions to mask.

        Returns:
            tuple containing:
                - output: Attention output of shape (batch_size, seq_length, d_model)
                - attention_weights: Average attention weights across heads of shape
                                    (batch_size, seq_length, seq_length)
        """

        # Linear projections
        q = self.W_q(query)  # (batch_size, seq_length, d_model)
        k = self.W_k(key)    # (batch_size, seq_length, d_model)
        v = self.W_v(value)  # (batch_size, seq_length, d_model)

        # Split into multiple heads
        q = self.split_heads(q)  # (batch_size, num_heads, seq_length, d_k)
        k = self.split_heads(k)  # (batch_size, num_heads, seq_length, d_k)
        v = self.split_heads(v)  # (batch_size, num_heads, seq_length, d_k)

        # Apply scaled dot-product attention to all heads at once
        # Reshape q, k, v for batch processing of all heads
        batch_size, num_heads, seq_length, d_k = q.size()

        # Reshape for batch processing: (batch_size * num_heads, seq_length, d_k)
        q_reshaped = q.contiguous().view(batch_size * num_heads, seq_length, d_k)
        k_reshaped = k.contiguous().view(batch_size * num_heads, seq_length, d_k)
        v_reshaped = v.contiguous().view(batch_size * num_heads, seq_length, d_k)

        # Prepare mask for batch processing if provided
        if mask is not None:
            # Handle different mask shapes
            if mask.dim() == 3:  # (batch_size, 1, seq_length) or (batch_size, seq_length, seq_length)
                if mask.size(1) == 1:
                    # Expand to (batch_size, 1, 1, seq_length) for broadcasting
                    mask = mask.unsqueeze(1)
                    # Repeat for each head: (batch_size, num_heads, 1, seq_length)
                    mask = mask.expand(-1, num_heads, -1, -1)
                    # Reshape: (batch_size * num_heads, 1, seq_length)
                    mask = mask.contiguous().view(batch_size * num_heads, 1, seq_length)
                else:  # (batch_size, seq_length, seq_length)
                    # Repeat for each head: (batch_size, num_heads, seq_length, seq_length)
                    mask = mask.unsqueeze(1).expand(-1, num_heads, -1, -1)
                    # Reshape: (batch_size * num_heads, seq_length, seq_length)
                    mask = mask.contiguous().view(batch_size * num_heads, seq_length, seq_length)

        # Apply attention to all heads at once
        attn_output, attn_weights = self.attention(q_reshaped, k_reshaped, v_reshaped, mask)

        # Reshape attention output back to separate heads
        # (batch_size * num_heads, seq_length, d_k) -> (batch_size, num_heads, seq_length, d_k)
        attn_output = attn_output.view(batch_size, num_heads, seq_length, d_k)

        # Reshape attention weights back to separate heads
        # (batch_size * num_heads, seq_length, seq_length) -> (batch_size, num_heads, seq_length, seq_length)
        attn_weights = attn_weights.view(batch_size, num_heads, seq_length, seq_length)

        # Combine heads
        combined_output = self.combine_heads(attn_output)  # (batch_size, seq_length, d_model)

        # Final linear projection
        output = self.W_o(combined_output)  # (batch_size, seq_length, d_model)

        # Apply dropout to the output
        output = self.dropout(output)

        # Average attention weights across heads for visualization/analysis
        avg_attention_weights = attn_weights.mean(dim=1)  # (batch_size, seq_length, seq_length)

        return output, avg_attention_weights
