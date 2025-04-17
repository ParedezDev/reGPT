"""
Embedding layers for the transformer model.
"""

import torch
import torch.nn as nn
import math
from typing import Optional


class InputEmbeddings(nn.Module):
    """
    Input embeddings layer that converts token indices to embeddings.
    
    This layer maps token indices to dense vectors of fixed size.
    """
    
    def __init__(self, vocab_size: int, embedding_dim: int, pad_token_id: Optional[int] = None):
        """
        Initialize the input embeddings layer.
        
        Args:
            vocab_size: Size of the vocabulary
            embedding_dim: Dimension of the embedding vectors
            pad_token_id: Optional token ID used for padding. If provided, the embedding for this ID will be initialized to zeros.
        """
        super().__init__()
        self.vocab_size = vocab_size
        self.embedding_dim = embedding_dim
        self.embedding = nn.Embedding(vocab_size, embedding_dim, padding_idx=pad_token_id)
        
        # Initialize weights using standard approach for transformers
        self._init_weights()
    
    def _init_weights(self):
        """Initialize the embedding weights."""
        nn.init.normal_(self.embedding.weight, mean=0, std=0.02)
        
        # If padding_idx is not None, zero the embedding vector for padding token
        if self.embedding.padding_idx is not None:
            with torch.no_grad():
                self.embedding.weight[self.embedding.padding_idx].zero_()
    
    def forward(self, x: torch.Tensor) -> torch.Tensor:
        """
        Convert token indices to embeddings.
        
        Args:
            x: Tensor of token indices of shape (batch_size, sequence_length)
            
        Returns:
            Tensor of embeddings of shape (batch_size, sequence_length, embedding_dim)
        """
        # x: (batch_size, sequence_length)
        # output: (batch_size, sequence_length, embedding_dim)
        return self.embedding(x) * math.sqrt(self.embedding_dim)
