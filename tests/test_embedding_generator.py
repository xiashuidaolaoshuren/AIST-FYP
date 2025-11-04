"""
Unit tests for embedding generator.
"""

import pytest
import numpy as np
import tempfile
from pathlib import Path

from src.data_processing import EmbeddingGenerator


class TestEmbeddingGenerator:
    """Tests for EmbeddingGenerator class."""
    
    def test_create_generator(self):
        """Test creating an EmbeddingGenerator instance."""
        generator = EmbeddingGenerator(
            model_name='sentence-transformers/all-MiniLM-L6-v2',
            batch_size=16,
            device='cuda',
            use_fp16=True
        )
        
        assert generator.model_name == 'sentence-transformers/all-MiniLM-L6-v2'
        assert generator.batch_size == 16
        assert generator.device == 'cuda'
        assert generator.embedding_dim == 384
    
    def test_generate_embeddings(self):
        """Test embedding generation for simple chunks."""
        generator = EmbeddingGenerator(
            model_name='sentence-transformers/all-MiniLM-L6-v2',
            batch_size=2,
            device='cuda',
            use_fp16=True
        )
        
        chunks = [
            {'text': 'This is the first sentence.'},
            {'text': 'This is the second sentence.'},
            {'text': 'This is the third sentence.'}
        ]
        
        embeddings = generator.generate_embeddings(chunks)
        
        assert embeddings.shape == (3, 384)
        # FP16 mode returns float16 embeddings
        assert embeddings.dtype in (np.float16, np.float32)
    
    def test_embeddings_normalized(self):
        """Test that embeddings are L2 normalized."""
        generator = EmbeddingGenerator(
            model_name='sentence-transformers/all-MiniLM-L6-v2',
            batch_size=2,
            device='cuda',
            use_fp16=True
        )
        
        chunks = [
            {'text': 'Test sentence one.'},
            {'text': 'Test sentence two.'}
        ]
        
        embeddings = generator.generate_embeddings(chunks)
        
        norms = np.linalg.norm(embeddings, axis=1)
        # Allow small tolerance due to FP16 precision
        assert np.allclose(norms, 1.0, atol=1e-3)
    
    def test_checkpoint_save_and_load(self, tmp_path):
        """Test checkpoint saving and loading."""
        checkpoint_file = tmp_path / "test_checkpoint.pkl"
        
        generator = EmbeddingGenerator(
            model_name='sentence-transformers/all-MiniLM-L6-v2',
            batch_size=2,
            device='cuda',
            use_fp16=True
        )
        
        # Create more chunks than checkpoint interval to test checkpointing
        chunks = [{'text': f'Sentence number {i} for testing.'} for i in range(5)]
        
        # Generate embeddings with checkpoint (interval=3 means checkpoint after 3 chunks)
        embeddings1 = generator.generate_embeddings(
            chunks,
            checkpoint_path=str(checkpoint_file),
            checkpoint_interval=3
        )
        
        # Checkpoint should be cleaned up after successful completion
        assert not checkpoint_file.exists()
        assert embeddings1.shape == (5, 384)
    
    def test_encode_single(self):
        """Test single text encoding."""
        generator = EmbeddingGenerator(
            model_name='sentence-transformers/all-MiniLM-L6-v2',
            device='cuda',
            use_fp16=True
        )
        
        embedding = generator.encode_single("Test sentence.")
        
        assert embedding.shape == (384,)
        assert np.isclose(np.linalg.norm(embedding), 1.0, atol=1e-3)
    
    def test_different_batch_sizes(self):
        """Test that different batch sizes produce same results."""
        generator = EmbeddingGenerator(
            model_name='sentence-transformers/all-MiniLM-L6-v2',
            device='cuda',
            use_fp16=True
        )
        
        chunks = [{'text': f'Sentence {i}'} for i in range(10)]
        
        # Generate with batch size 5
        generator.batch_size = 5
        embeddings1 = generator.generate_embeddings(chunks)
        
        # Generate with batch size 2
        generator.batch_size = 2
        embeddings2 = generator.generate_embeddings(chunks)
        
        # Should produce same embeddings (within FP16 tolerance)
        assert np.allclose(embeddings1, embeddings2, atol=1e-3)
    
    def test_get_embedding_dimension(self):
        """Test getting embedding dimension."""
        generator = EmbeddingGenerator(
            model_name='sentence-transformers/all-MiniLM-L6-v2',
            device='cuda'
        )
        
        dim = generator.get_embedding_dimension()
        assert dim == 384
