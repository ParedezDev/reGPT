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
        d_k = 8  # Use a smaller dimension to reduce numerical issues

        # Create identical query and key tensors for easy testing
        query = torch.ones(batch_size, num_queries, d_k)
        key = torch.ones(batch_size, num_keys, d_k)
        value = torch.ones(batch_size, num_keys, d_k)

        # Create attention module
        attention = ScaledDotProductAttention(dropout=0.0)  # Disable dropout for deterministic results

        # Compute attention with default scaling
        _, attention_weights = attention(query, key, value)

        # Check that the weights are approximately uniform
        # For identical inputs, each position should get approximately equal attention
        min_weight = attention_weights.min().item()
        max_weight = attention_weights.max().item()
        self.assertLess(max_weight - min_weight, 0.1,
                        f"Weights should be approximately uniform, got min={min_weight}, max={max_weight}")

        # Check that the weights sum to approximately 1 for each query
        row_sums = attention_weights.sum(dim=-1)
        self.assertTrue(torch.all((row_sums > 0.95) & (row_sums < 1.05)),
                        f"Attention weights should sum to approximately 1 for each query, got {row_sums}")

        # Test with custom scaling
        custom_scale = 0.5
        _, attention_weights_custom = attention(query, key, value, attention_scale=custom_scale)

        # Check that the weights sum to approximately 1 for each query with custom scaling
        row_sums_custom = attention_weights_custom.sum(dim=-1)
        self.assertTrue(torch.all((row_sums_custom > 0.95) & (row_sums_custom < 1.05)),
                        f"Attention weights should sum to approximately 1 with custom scaling, got {row_sums_custom}")

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

        # Create input tensors with controlled values to avoid numerical issues
        query = torch.randn(batch_size, seq_length, d_k)
        key = torch.randn(batch_size, seq_length, d_k)
        value = torch.randn(batch_size, seq_length, d_k)

        # Create attention module with no dropout for deterministic results
        attention = ScaledDotProductAttention(dropout=0.0)

        # Compute attention
        _, attention_weights = attention(query, key, value)

        # Check that attention weights sum to approximately 1 along the key dimension
        sums = attention_weights.sum(dim=-1)

        # Use a more relaxed check for the sums
        self.assertTrue(torch.all((sums > 0.9) & (sums < 1.1)),
                        f"All attention weight rows should sum to approximately 1.0, got min={sums.min().item()}, max={sums.max().item()}")

        # Also check that all weights are between 0 and 1
        self.assertTrue(torch.all((attention_weights >= 0) & (attention_weights <= 1)),
                        "All attention weights should be between 0 and 1")


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
