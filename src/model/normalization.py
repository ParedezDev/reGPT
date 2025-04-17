"""
Normalization layers for transformer models.

This module provides normalization layers used in transformer architectures,
primarily Layer Normalization as described in the paper "Layer Normalization"
by Ba, Kiros, and Hinton.
"""

import torch
import torch.nn as nn
from typing import Optional, Union, List, Tuple


class LayerNorm(nn.Module):
    """
    Layer Normalization implementation using PyTorch's nn.LayerNorm.
    
    Normalizes the activations of the previous layer for each token,
    applying an affine transformation (scale and shift) after normalization.
    
    Formula: y = (x - E[x]) / sqrt(Var[x] + eps) * gamma + beta
    where gamma and beta are learnable parameters.
    """
    
    def __init__(
        self,
        normalized_shape: Union[int, List[int], torch.Size],
        eps: float = 1e-5,
        elementwise_affine: bool = True,
    ):
        """
        Initialize the Layer Normalization.
        
        Args:
            normalized_shape: Input shape from an expected input of size.
                If a single integer is used, it is treated as a singleton list [normalized_shape].
            eps: A small constant added to the denominator for numerical stability.
            elementwise_affine: Whether to use learnable affine parameters (gamma, beta).
        """
        super().__init__()
        
        # Use PyTorch's built-in LayerNorm
        self.layer_norm = nn.LayerNorm(
            normalized_shape=normalized_shape,
            eps=eps,
            elementwise_affine=elementwise_affine
        )
    
    def forward(self, x: torch.Tensor) -> torch.Tensor:
        """
        Apply layer normalization to the input.
        
        Args:
            x: Input tensor to be normalized.
                For a 3D input of shape (batch_size, seq_length, hidden_size),
                normalization is applied along the last dimension.
                
        Returns:
            Normalized tensor of the same shape as the input.
        """
        return self.layer_norm(x)


class TransformerLayerNorm(nn.Module):
    """
    Layer Normalization specifically designed for transformer models.
    
    This is a convenience wrapper around LayerNorm that defaults to
    normalizing along the model dimension (last dimension) as is common
    in transformer architectures.
    """
    
    def __init__(self, d_model: int, eps: float = 1e-5):
        """
        Initialize the Transformer Layer Normalization.
        
        Args:
            d_model: The model dimension (hidden size).
            eps: A small constant added to the denominator for numerical stability.
        """
        super().__init__()
        
        self.layer_norm = LayerNorm(normalized_shape=d_model, eps=eps)
    
    def forward(self, x: torch.Tensor) -> torch.Tensor:
        """
        Apply layer normalization to the input.
        
        Args:
            x: Input tensor of shape (batch_size, seq_length, d_model).
                
        Returns:
            Normalized tensor of the same shape as the input.
        """
        return self.layer_norm(x)
