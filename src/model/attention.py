"""
Attention mechanisms for transformer models.

This module implements the Scaled Dot-Product Attention mechanism
as described in the paper "Attention Is All You Need".
"""

import torch
import torch.nn as nn
import torch.nn.functional as F
import math
from typing import Optional


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
        batch_size, num_queries, d_k = query.shape
        _, num_keys, d_v = value.shape
        
        # Compute attention scores: (batch_size, num_queries, num_keys)
        # matmul: (batch_size, num_queries, d_k) x (batch_size, d_k, num_keys)
        attention_scores = torch.bmm(query, key.transpose(1, 2))
        
        # Scale attention scores
        scale = attention_scale if attention_scale is not None else 1.0 / math.sqrt(d_k)
        attention_scores = attention_scores * scale
        
        # Apply mask if provided
        if mask is not None:
            # Convert boolean mask to float mask where True/1 values become -inf
            if mask.dtype == torch.bool:
                float_mask = torch.zeros_like(mask, dtype=torch.float)
                float_mask.masked_fill_(mask, float('-inf'))
                mask = float_mask
            elif mask.dtype == torch.int64 or mask.dtype == torch.int32 or mask.dtype == torch.int16 or mask.dtype == torch.int8:
                float_mask = torch.zeros_like(mask, dtype=torch.float)
                float_mask.masked_fill_(mask == 1, float('-inf'))
                mask = float_mask
            
            # Apply mask to attention scores
            attention_scores = attention_scores + mask
        
        # Apply softmax to get attention weights
        attention_weights = F.softmax(attention_scores, dim=-1)
        
        # Apply dropout
        attention_weights = self.dropout(attention_weights)
        
        # Compute output: (batch_size, num_queries, d_v)
        # matmul: (batch_size, num_queries, num_keys) x (batch_size, num_keys, d_v)
        output = torch.bmm(attention_weights, value)
        
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
