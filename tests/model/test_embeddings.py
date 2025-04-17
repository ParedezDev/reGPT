"""
Tests for the embedding layers.
"""

import unittest
import torch
from src.model.embeddings import InputEmbeddings


class TestInputEmbeddings(unittest.TestCase):
    """Test cases for the InputEmbeddings class."""
    
    def test_embedding_shape(self):
        """Test that the output shape is correct."""
        batch_size = 2
        seq_length = 10
        vocab_size = 1000
        embedding_dim = 64
        
        # Create input tensor
        x = torch.randint(0, vocab_size, (batch_size, seq_length))
        
        # Create embedding layer
        embedding_layer = InputEmbeddings(vocab_size, embedding_dim)
        
        # Get embeddings
        embeddings = embedding_layer(x)
        
        # Check shape
        self.assertEqual(embeddings.shape, (batch_size, seq_length, embedding_dim))
    
    def test_padding_token(self):
        """Test that padding token embeddings are zero."""
        batch_size = 2
        seq_length = 10
        vocab_size = 1000
        embedding_dim = 64
        pad_token_id = 0
        
        # Create embedding layer with padding
        embedding_layer = InputEmbeddings(vocab_size, embedding_dim, pad_token_id=pad_token_id)
        
        # Create input with padding tokens
        x = torch.ones((batch_size, seq_length), dtype=torch.long)
        x[:, 0] = pad_token_id  # Set first token of each sequence to padding
        
        # Get embeddings
        embeddings = embedding_layer(x)
        
        # Check that padding token embeddings are zero
        padding_embeddings = embeddings[:, 0, :]
        self.assertTrue(torch.all(padding_embeddings == 0))
        
        # Check that non-padding token embeddings are non-zero
        non_padding_embeddings = embeddings[:, 1:, :]
        self.assertFalse(torch.all(non_padding_embeddings == 0))
    
    def test_embedding_scaling(self):
        """Test that embeddings are scaled by sqrt(embedding_dim)."""
        batch_size = 2
        seq_length = 10
        vocab_size = 1000
        embedding_dim = 64
        
        # Create input tensor with a single token ID
        token_id = 42
        x = torch.full((batch_size, seq_length), token_id, dtype=torch.long)
        
        # Create embedding layer
        embedding_layer = InputEmbeddings(vocab_size, embedding_dim)
        
        # Get raw embedding for the token (without scaling)
        with torch.no_grad():
            raw_embedding = embedding_layer.embedding.weight[token_id]
        
        # Get scaled embeddings from the forward pass
        scaled_embeddings = embedding_layer(x)
        
        # Check that the scaling is applied correctly
        expected_scaling = torch.sqrt(torch.tensor(embedding_dim, dtype=torch.float))
        for i in range(batch_size):
            for j in range(seq_length):
                self.assertTrue(torch.allclose(
                    scaled_embeddings[i, j], 
                    raw_embedding * expected_scaling
                ))


if __name__ == "__main__":
    unittest.main()
