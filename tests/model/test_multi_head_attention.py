"""
Tests for the Multi-Head Self-Attention mechanism.
"""

import unittest
import torch
from src.model.attention import MultiHeadSelfAttention


class TestMultiHeadSelfAttention(unittest.TestCase):
    """Test cases for the MultiHeadSelfAttention class."""
    
    def test_output_shape(self):
        """Test that the output shape is correct."""
        batch_size = 2
        seq_length = 10
        d_model = 64
        num_heads = 8
        
        # Create input tensors
        x = torch.randn(batch_size, seq_length, d_model)
        
        # Create multi-head attention module
        mha = MultiHeadSelfAttention(d_model, num_heads)
        
        # Compute attention
        output, attention_weights = mha(x, x, x)
        
        # Check output shape
        self.assertEqual(output.shape, (batch_size, seq_length, d_model))
        
        # Check attention weights shape
        self.assertEqual(attention_weights.shape, (batch_size, seq_length, seq_length))
    
    def test_invalid_dimensions(self):
        """Test that an error is raised if d_model is not divisible by num_heads."""
        d_model = 65  # Not divisible by 8
        num_heads = 8
        
        # Check that an error is raised
        with self.assertRaises(ValueError):
            MultiHeadSelfAttention(d_model, num_heads)
    
    def test_mask_application(self):
        """Test that masking is applied correctly."""
        batch_size = 2
        seq_length = 6
        d_model = 64
        num_heads = 8
        
        # Create input tensors
        x = torch.randn(batch_size, seq_length, d_model)
        
        # Create a mask where the first token can't attend to the last token
        mask = torch.zeros(batch_size, 1, seq_length, dtype=torch.bool)
        mask[:, :, -1] = True  # Mask out the last position
        
        # Create multi-head attention module
        mha = MultiHeadSelfAttention(d_model, num_heads, dropout=0.0)  # Disable dropout for deterministic results
        
        # Compute attention with mask
        _, attention_weights = mha(x, x, x, mask=mask)
        
        # Check that the masked positions have zero attention weight
        # Since we're averaging across heads, we check that the weights are very small
        self.assertTrue(torch.all(attention_weights[:, :, -1] < 1e-5))
    
    def test_head_splitting_combining(self):
        """Test that splitting and combining heads works correctly."""
        batch_size = 2
        seq_length = 4
        d_model = 64
        num_heads = 8
        d_k = d_model // num_heads
        
        # Create multi-head attention module
        mha = MultiHeadSelfAttention(d_model, num_heads)
        
        # Create a test tensor
        x = torch.randn(batch_size, seq_length, d_model)
        
        # Split heads
        split = mha.split_heads(x)
        
        # Check split shape
        self.assertEqual(split.shape, (batch_size, num_heads, seq_length, d_k))
        
        # Combine heads
        combined = mha.combine_heads(split)
        
        # Check combined shape
        self.assertEqual(combined.shape, (batch_size, seq_length, d_model))
        
        # Check that the content is preserved (approximately)
        self.assertTrue(torch.allclose(x, combined, rtol=1e-5, atol=1e-5))
    
    def test_projection_matrices(self):
        """Test that the projection matrices have the correct shapes."""
        d_model = 64
        num_heads = 8
        
        # Create multi-head attention module
        mha = MultiHeadSelfAttention(d_model, num_heads)
        
        # Check shapes of projection matrices
        self.assertEqual(mha.W_q.weight.shape, (d_model, d_model))
        self.assertEqual(mha.W_k.weight.shape, (d_model, d_model))
        self.assertEqual(mha.W_v.weight.shape, (d_model, d_model))
        self.assertEqual(mha.W_o.weight.shape, (d_model, d_model))
    
    def test_self_attention(self):
        """Test that self-attention works correctly (query = key = value)."""
        batch_size = 2
        seq_length = 4
        d_model = 64
        num_heads = 8
        
        # Create input tensor
        x = torch.randn(batch_size, seq_length, d_model)
        
        # Create multi-head attention module
        mha = MultiHeadSelfAttention(d_model, num_heads, dropout=0.0)  # Disable dropout for deterministic results
        
        # Compute self-attention
        output, _ = mha(x, x, x)
        
        # Check that the output has the same shape as the input
        self.assertEqual(output.shape, x.shape)
        
        # Check that the output is different from the input (transformation has occurred)
        self.assertFalse(torch.allclose(output, x))


if __name__ == "__main__":
    unittest.main()
