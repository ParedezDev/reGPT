"""
Positional encoding implementations for transformer models.

This module provides different positional encoding strategies:
1. Sinusoidal: Fixed encoding based on sine/cosine functions (from original Transformer paper)
2. Learned: Trainable positional embeddings
"""

import torch
import torch.nn as nn
import math
from typing import Optional


class SinusoidalPositionalEncoding(nn.Module):
    """
    Sinusoidal positional encoding from the paper "Attention Is All You Need".
    
    This encoding uses a fixed pattern of sine and cosine functions of different frequencies
    to encode position information. It doesn't require training and has the benefit of
    being able to extrapolate to sequence lengths not seen during training.
    """
    
    def __init__(self, embedding_dim: int, max_seq_length: int = 5000, dropout: float = 0.1):
        """
        Initialize the sinusoidal positional encoding.
        
        Args:
            embedding_dim: Dimension of the embedding vectors (must be even)
            max_seq_length: Maximum sequence length to pre-compute
            dropout: Dropout probability
        """
        super().__init__()
        
        if embedding_dim % 2 != 0:
            raise ValueError(f"Embedding dimension must be even, got {embedding_dim}")
        
        self.embedding_dim = embedding_dim
        self.max_seq_length = max_seq_length
        self.dropout = nn.Dropout(p=dropout)
        
        # Create positional encoding matrix
        # Shape: (max_seq_length, embedding_dim)
        position = torch.arange(max_seq_length).unsqueeze(1)
        div_term = torch.exp(torch.arange(0, embedding_dim, 2) * (-math.log(10000.0) / embedding_dim))
        
        pe = torch.zeros(max_seq_length, embedding_dim)
        pe[:, 0::2] = torch.sin(position * div_term)  # Even indices
        pe[:, 1::2] = torch.cos(position * div_term)  # Odd indices
        
        # Register buffer (persistent state that's not a parameter)
        self.register_buffer('pe', pe)
    
    def forward(self, x: torch.Tensor) -> torch.Tensor:
        """
        Add positional encoding to the input embeddings.
        
        Args:
            x: Tensor of token embeddings of shape (batch_size, seq_length, embedding_dim)
            
        Returns:
            Tensor with positional encoding added, same shape as input
        """
        seq_length = x.size(1)
        if seq_length > self.max_seq_length:
            raise ValueError(
                f"Sequence length ({seq_length}) exceeds maximum length ({self.max_seq_length})"
            )
        
        # Add positional encoding to input embeddings
        # x: (batch_size, seq_length, embedding_dim)
        # pe[:seq_length]: (seq_length, embedding_dim)
        x = x + self.pe[:seq_length]
        
        return self.dropout(x)


class LearnedPositionalEncoding(nn.Module):
    """
    Learned positional encoding that uses trainable embeddings for each position.
    
    This approach learns the optimal positional encoding during training, but may not
    generalize well to sequences longer than those seen during training.
    """
    
    def __init__(self, embedding_dim: int, max_seq_length: int = 5000, dropout: float = 0.1):
        """
        Initialize the learned positional encoding.
        
        Args:
            embedding_dim: Dimension of the embedding vectors
            max_seq_length: Maximum sequence length to support
            dropout: Dropout probability
        """
        super().__init__()
        
        self.embedding_dim = embedding_dim
        self.max_seq_length = max_seq_length
        
        # Create learnable position embeddings
        self.position_embeddings = nn.Parameter(torch.zeros(1, max_seq_length, embedding_dim))
        self.dropout = nn.Dropout(p=dropout)
        
        # Initialize weights
        self._init_weights()
    
    def _init_weights(self):
        """Initialize the position embedding weights."""
        nn.init.normal_(self.position_embeddings, mean=0, std=0.02)
    
    def forward(self, x: torch.Tensor) -> torch.Tensor:
        """
        Add positional encoding to the input embeddings.
        
        Args:
            x: Tensor of token embeddings of shape (batch_size, seq_length, embedding_dim)
            
        Returns:
            Tensor with positional encoding added, same shape as input
        """
        seq_length = x.size(1)
        if seq_length > self.max_seq_length:
            raise ValueError(
                f"Sequence length ({seq_length}) exceeds maximum length ({self.max_seq_length})"
            )
        
        # Add positional encoding to input embeddings
        # x: (batch_size, seq_length, embedding_dim)
        # position_embeddings[:, :seq_length, :]: (1, seq_length, embedding_dim)
        x = x + self.position_embeddings[:, :seq_length, :]
        
        return self.dropout(x)
