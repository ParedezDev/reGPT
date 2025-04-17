"""
Tests for the positional encoding implementations.
"""

import unittest
import torch
import math
from src.model.positional_encoding import SinusoidalPositionalEncoding, LearnedPositionalEncoding


class TestSinusoidalPositionalEncoding(unittest.TestCase):
    """Test cases for the SinusoidalPositionalEncoding class."""
    
    def test_output_shape(self):
        """Test that the output shape is the same as the input shape."""
        batch_size = 2
        seq_length = 10
        embedding_dim = 64
        
        # Create input tensor
        x = torch.randn(batch_size, seq_length, embedding_dim)
        
        # Create positional encoding
        pos_encoding = SinusoidalPositionalEncoding(embedding_dim)
        
        # Apply positional encoding
        output = pos_encoding(x)
        
        # Check shape
        self.assertEqual(output.shape, (batch_size, seq_length, embedding_dim))
    
    def test_sinusoidal_pattern(self):
        """Test that the sinusoidal pattern is correctly implemented."""
        embedding_dim = 64
        max_seq_length = 100
        
        # Create positional encoding
        pos_encoding = SinusoidalPositionalEncoding(embedding_dim, max_seq_length)
        
        # Get the positional encoding matrix
        pe = pos_encoding.pe
        
        # Check shape of the positional encoding matrix
        self.assertEqual(pe.shape, (max_seq_length, embedding_dim))
        
        # Check that the pattern follows the formula from the paper
        # For even indices: sin(pos / 10000^(2i/d_model))
        # For odd indices: cos(pos / 10000^(2i/d_model))
        for pos in range(0, max_seq_length, 10):  # Check every 10th position
            for i in range(0, embedding_dim, 2):  # Check every dimension pair
                div_term = 10000 ** (i / embedding_dim)
                
                # Check sine value for even indices
                expected_sin = math.sin(pos / div_term)
                self.assertAlmostEqual(pe[pos, i].item(), expected_sin, places=5)
                
                # Check cosine value for odd indices
                expected_cos = math.cos(pos / div_term)
                self.assertAlmostEqual(pe[pos, i+1].item(), expected_cos, places=5)
    
    def test_max_length_constraint(self):
        """Test that an error is raised if sequence length exceeds max_seq_length."""
        embedding_dim = 64
        max_seq_length = 50
        
        # Create positional encoding with limited max_seq_length
        pos_encoding = SinusoidalPositionalEncoding(embedding_dim, max_seq_length)
        
        # Create input tensor with sequence length > max_seq_length
        x = torch.randn(2, max_seq_length + 10, embedding_dim)
        
        # Check that an error is raised
        with self.assertRaises(ValueError):
            pos_encoding(x)
    
    def test_even_dimension_constraint(self):
        """Test that an error is raised if embedding_dim is not even."""
        # Try to create positional encoding with odd embedding_dim
        with self.assertRaises(ValueError):
            SinusoidalPositionalEncoding(embedding_dim=63)


class TestLearnedPositionalEncoding(unittest.TestCase):
    """Test cases for the LearnedPositionalEncoding class."""
    
    def test_output_shape(self):
        """Test that the output shape is the same as the input shape."""
        batch_size = 2
        seq_length = 10
        embedding_dim = 64
        
        # Create input tensor
        x = torch.randn(batch_size, seq_length, embedding_dim)
        
        # Create positional encoding
        pos_encoding = LearnedPositionalEncoding(embedding_dim)
        
        # Apply positional encoding
        output = pos_encoding(x)
        
        # Check shape
        self.assertEqual(output.shape, (batch_size, seq_length, embedding_dim))
    
    def test_learned_weights(self):
        """Test that the position embeddings are learnable parameters."""
        embedding_dim = 64
        max_seq_length = 100
        
        # Create positional encoding
        pos_encoding = LearnedPositionalEncoding(embedding_dim, max_seq_length)
        
        # Check that position_embeddings is a Parameter
        self.assertTrue(isinstance(pos_encoding.position_embeddings, torch.nn.Parameter))
        
        # Check shape of the position embeddings
        self.assertEqual(pos_encoding.position_embeddings.shape, (1, max_seq_length, embedding_dim))
    
    def test_max_length_constraint(self):
        """Test that an error is raised if sequence length exceeds max_seq_length."""
        embedding_dim = 64
        max_seq_length = 50
        
        # Create positional encoding with limited max_seq_length
        pos_encoding = LearnedPositionalEncoding(embedding_dim, max_seq_length)
        
        # Create input tensor with sequence length > max_seq_length
        x = torch.randn(2, max_seq_length + 10, embedding_dim)
        
        # Check that an error is raised
        with self.assertRaises(ValueError):
            pos_encoding(x)
    
    def test_gradient_flow(self):
        """Test that gradients flow through the learned positional encoding."""
        batch_size = 2
        seq_length = 10
        embedding_dim = 64
        
        # Create input tensor
        x = torch.randn(batch_size, seq_length, embedding_dim, requires_grad=True)
        
        # Create positional encoding
        pos_encoding = LearnedPositionalEncoding(embedding_dim)
        
        # Apply positional encoding
        output = pos_encoding(x)
        
        # Compute loss and backpropagate
        loss = output.sum()
        loss.backward()
        
        # Check that gradients are computed for position_embeddings
        self.assertIsNotNone(pos_encoding.position_embeddings.grad)
        self.assertFalse(torch.all(pos_encoding.position_embeddings.grad == 0))


if __name__ == "__main__":
    unittest.main()
