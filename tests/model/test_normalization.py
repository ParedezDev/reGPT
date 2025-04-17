"""
Tests for the normalization layers.
"""

import unittest
import torch
import torch.nn as nn
from src.model.normalization import LayerNorm, TransformerLayerNorm


class TestLayerNorm(unittest.TestCase):
    """Test cases for the LayerNorm class."""

    def test_output_shape(self):
        """Test that the output shape is the same as the input shape."""
        batch_size = 2
        seq_length = 10
        hidden_size = 64

        # Create input tensor
        x = torch.randn(batch_size, seq_length, hidden_size)

        # Create layer norm
        layer_norm = LayerNorm(normalized_shape=hidden_size)

        # Apply layer norm
        output = layer_norm(x)

        # Check shape
        self.assertEqual(output.shape, x.shape)

    def test_normalization_statistics(self):
        """Test that the normalization produces the expected statistics."""
        batch_size = 2
        seq_length = 10
        hidden_size = 64

        # Create input tensor
        x = torch.randn(batch_size, seq_length, hidden_size)

        # Create layer norm
        layer_norm = LayerNorm(normalized_shape=hidden_size)

        # Apply layer norm
        output = layer_norm(x)

        # Check that the mean is approximately 0 and variance is approximately 1
        # along the normalized dimension (last dimension)
        output_mean = output.mean(dim=-1)
        output_var = output.var(dim=-1, unbiased=False)

        # Allow for some numerical error
        self.assertTrue(torch.allclose(output_mean, torch.zeros_like(output_mean), atol=1e-5))
        self.assertTrue(torch.allclose(output_var, torch.ones_like(output_var), atol=1e-5))

    def test_affine_parameters(self):
        """Test that the affine parameters (gamma, beta) are learnable."""
        hidden_size = 64

        # Create layer norm with affine parameters
        layer_norm = LayerNorm(normalized_shape=hidden_size, elementwise_affine=True)

        # Check that gamma and beta are parameters
        self.assertTrue(hasattr(layer_norm.layer_norm, 'weight'))
        self.assertTrue(hasattr(layer_norm.layer_norm, 'bias'))
        self.assertTrue(isinstance(layer_norm.layer_norm.weight, nn.Parameter))
        self.assertTrue(isinstance(layer_norm.layer_norm.bias, nn.Parameter))

        # Check shapes
        self.assertEqual(layer_norm.layer_norm.weight.shape, torch.Size([hidden_size]))
        self.assertEqual(layer_norm.layer_norm.bias.shape, torch.Size([hidden_size]))

    def test_no_affine_parameters(self):
        """Test that the affine parameters can be disabled."""
        hidden_size = 64

        # Create layer norm without affine parameters
        layer_norm = LayerNorm(normalized_shape=hidden_size, elementwise_affine=False)

        # Check that weight and bias are None
        # PyTorch's nn.LayerNorm sets weight and bias to None when elementwise_affine=False
        self.assertIsNone(layer_norm.layer_norm.weight)
        self.assertIsNone(layer_norm.layer_norm.bias)

    def test_different_shapes(self):
        """Test that the layer norm works with different input shapes."""
        # 2D input
        x_2d = torch.randn(10, 64)
        layer_norm_2d = LayerNorm(normalized_shape=64)
        output_2d = layer_norm_2d(x_2d)
        self.assertEqual(output_2d.shape, x_2d.shape)

        # 3D input
        x_3d = torch.randn(2, 10, 64)
        layer_norm_3d = LayerNorm(normalized_shape=64)
        output_3d = layer_norm_3d(x_3d)
        self.assertEqual(output_3d.shape, x_3d.shape)

        # 4D input
        x_4d = torch.randn(2, 3, 10, 64)
        layer_norm_4d = LayerNorm(normalized_shape=64)
        output_4d = layer_norm_4d(x_4d)
        self.assertEqual(output_4d.shape, x_4d.shape)


class TestTransformerLayerNorm(unittest.TestCase):
    """Test cases for the TransformerLayerNorm class."""

    def test_output_shape(self):
        """Test that the output shape is the same as the input shape."""
        batch_size = 2
        seq_length = 10
        d_model = 64

        # Create input tensor
        x = torch.randn(batch_size, seq_length, d_model)

        # Create transformer layer norm
        layer_norm = TransformerLayerNorm(d_model=d_model)

        # Apply layer norm
        output = layer_norm(x)

        # Check shape
        self.assertEqual(output.shape, x.shape)

    def test_normalization_statistics(self):
        """Test that the normalization produces the expected statistics."""
        batch_size = 2
        seq_length = 10
        d_model = 64

        # Create input tensor
        x = torch.randn(batch_size, seq_length, d_model)

        # Create transformer layer norm
        layer_norm = TransformerLayerNorm(d_model=d_model)

        # Apply layer norm
        output = layer_norm(x)

        # Check that the mean is approximately 0 and variance is approximately 1
        # along the normalized dimension (last dimension)
        output_mean = output.mean(dim=-1)
        output_var = output.var(dim=-1, unbiased=False)

        # Allow for some numerical error
        self.assertTrue(torch.allclose(output_mean, torch.zeros_like(output_mean), atol=1e-5))
        self.assertTrue(torch.allclose(output_var, torch.ones_like(output_var), atol=1e-5))

    def test_gradient_flow(self):
        """Test that gradients flow through the layer norm."""
        batch_size = 2
        seq_length = 10
        d_model = 64

        # Create input tensor with requires_grad=True
        x = torch.randn(batch_size, seq_length, d_model, requires_grad=True)

        # Create transformer layer norm
        layer_norm = TransformerLayerNorm(d_model=d_model)

        # Apply layer norm
        output = layer_norm(x)

        # Compute loss and backpropagate
        loss = output.sum()
        loss.backward()

        # Check that gradients are computed
        self.assertIsNotNone(x.grad)
        self.assertFalse(torch.all(x.grad == 0))


if __name__ == "__main__":
    unittest.main()
