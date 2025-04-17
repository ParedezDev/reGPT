"""
Tests for the attention mechanisms.
"""

import unittest
import torch
import math
from src.model.attention import ScaledDotProductAttention, create_causal_mask, create_padding_mask


class TestScaledDotProductAttention(unittest.TestCase):
    """Test cases for the ScaledDotProductAttention class."""
    
    def test_output_shape(self):
        """Test that the output shape is correct."""
        batch_size = 2
        num_queries = 4
        num_keys = 6
        d_k = 8
        d_v = 10
        
        # Create input tensors
        query = torch.randn(batch_size, num_queries, d_k)
        key = torch.randn(batch_size, num_keys, d_k)
        value = torch.randn(batch_size, num_keys, d_v)
        
        # Create attention module
        attention = ScaledDotProductAttention()
        
        # Compute attention
        output, attention_weights = attention(query, key, value)
        
        # Check output shape
        self.assertEqual(output.shape, (batch_size, num_queries, d_v))
        
        # Check attention weights shape
        self.assertEqual(attention_weights.shape, (batch_size, num_queries, num_keys))
    
    def test_scaling(self):
        """Test that the scaling is applied correctly."""
        batch_size = 2
        num_queries = 4
        num_keys = 4
        d_k = 64
        
        # Create identical query and key tensors for easy testing
        query = torch.ones(batch_size, num_queries, d_k)
        key = torch.ones(batch_size, num_keys, d_k)
        value = torch.ones(batch_size, num_keys, d_k)
        
        # Create attention module
        attention = ScaledDotProductAttention()
        
        # Compute attention with default scaling
        _, attention_weights = attention(query, key, value)
        
        # Since query and key are all ones, the dot product is d_k
        # After scaling by 1/sqrt(d_k) and applying softmax (which normalizes),
        # all attention weights should be equal (1/num_keys)
        expected_weight = 1.0 / num_keys
        self.assertTrue(torch.allclose(attention_weights, torch.full_like(attention_weights, expected_weight)))
        
        # Test with custom scaling
        custom_scale = 0.5
        _, attention_weights_custom = attention(query, key, value, attention_scale=custom_scale)
        
        # The result should still be uniform due to softmax normalization
        self.assertTrue(torch.allclose(attention_weights_custom, torch.full_like(attention_weights_custom, expected_weight)))
    
    def test_mask_boolean(self):
        """Test that boolean masks are applied correctly."""
        batch_size = 2
        seq_length = 4
        d_k = 8
        
        # Create input tensors
        query = torch.randn(batch_size, seq_length, d_k)
        key = torch.randn(batch_size, seq_length, d_k)
        value = torch.randn(batch_size, seq_length, d_k)
        
        # Create a mask where the first token can't attend to the last token
        mask = torch.zeros(batch_size, seq_length, seq_length, dtype=torch.bool)
        mask[:, 0, -1] = True  # Mask out position (0, -1) in each batch
        
        # Create attention module
        attention = ScaledDotProductAttention()
        
        # Compute attention with mask
        _, attention_weights = attention(query, key, value, mask=mask)
        
        # Check that the masked positions have zero attention weight
        self.assertTrue(torch.all(attention_weights[:, 0, -1] == 0))
    
    def test_mask_integer(self):
        """Test that integer masks are applied correctly."""
        batch_size = 2
        seq_length = 4
        d_k = 8
        
        # Create input tensors
        query = torch.randn(batch_size, seq_length, d_k)
        key = torch.randn(batch_size, seq_length, d_k)
        value = torch.randn(batch_size, seq_length, d_k)
        
        # Create a mask where the first token can't attend to the last token
        mask = torch.zeros(batch_size, seq_length, seq_length, dtype=torch.int64)
        mask[:, 0, -1] = 1  # Mask out position (0, -1) in each batch
        
        # Create attention module
        attention = ScaledDotProductAttention()
        
        # Compute attention with mask
        _, attention_weights = attention(query, key, value, mask=mask)
        
        # Check that the masked positions have zero attention weight
        self.assertTrue(torch.all(attention_weights[:, 0, -1] == 0))
    
    def test_softmax_normalization(self):
        """Test that softmax normalization is applied correctly."""
        batch_size = 2
        seq_length = 4
        d_k = 8
        
        # Create input tensors
        query = torch.randn(batch_size, seq_length, d_k)
        key = torch.randn(batch_size, seq_length, d_k)
        value = torch.randn(batch_size, seq_length, d_k)
        
        # Create attention module
        attention = ScaledDotProductAttention()
        
        # Compute attention
        _, attention_weights = attention(query, key, value)
        
        # Check that attention weights sum to 1 along the key dimension
        sums = attention_weights.sum(dim=-1)
        self.assertTrue(torch.allclose(sums, torch.ones_like(sums)))


class TestMaskFunctions(unittest.TestCase):
    """Test cases for the mask creation functions."""
    
    def test_causal_mask(self):
        """Test that the causal mask is created correctly."""
        seq_length = 5
        
        # Create causal mask
        mask = create_causal_mask(seq_length)
        
        # Check shape
        self.assertEqual(mask.shape, (1, seq_length, seq_length))
        
        # Check that the mask is upper triangular (excluding the diagonal)
        expected_mask = torch.triu(torch.ones(seq_length, seq_length), diagonal=1).bool().unsqueeze(0)
        self.assertTrue(torch.all(mask == expected_mask))
        
        # Check specific values
        # - Diagonal and below should be False (not masked)
        # - Above diagonal should be True (masked)
        for i in range(seq_length):
            for j in range(seq_length):
                if j <= i:  # On or below diagonal
                    self.assertFalse(mask[0, i, j].item())
                else:  # Above diagonal
                    self.assertTrue(mask[0, i, j].item())
    
    def test_padding_mask(self):
        """Test that the padding mask is created correctly."""
        batch_size = 3
        seq_length = 5
        pad_idx = 0
        
        # Create input tensor with some padding
        src = torch.randint(1, 10, (batch_size, seq_length))
        src[0, -2:] = pad_idx  # Add padding to first sequence
        src[1, -1] = pad_idx   # Add padding to second sequence
        
        # Create padding mask
        mask = create_padding_mask(src, pad_idx)
        
        # Check shape
        self.assertEqual(mask.shape, (batch_size, 1, seq_length))
        
        # Check specific values
        # - Padding positions should be True (masked)
        # - Non-padding positions should be False (not masked)
        for b in range(batch_size):
            for j in range(seq_length):
                if src[b, j].item() == pad_idx:
                    self.assertTrue(mask[b, 0, j].item())
                else:
                    self.assertFalse(mask[b, 0, j].item())


if __name__ == "__main__":
    unittest.main()
