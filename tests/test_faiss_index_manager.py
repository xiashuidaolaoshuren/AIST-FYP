"""
Unit tests for the FAISS Index Manager.

Tests the FAISSIndexManager class for building, saving, loading, and searching
FAISS indices with different index types.
"""

import pytest
import numpy as np
import tempfile
from pathlib import Path
import shutil

from src.retrieval import FAISSIndexManager


@pytest.fixture
def sample_embeddings():
    """Create sample embeddings for testing."""
    np.random.seed(42)
    # Create 100 normalized vectors of dimension 384
    embeddings = np.random.randn(100, 384).astype(np.float32)
    # L2 normalize
    norms = np.linalg.norm(embeddings, axis=1, keepdims=True)
    embeddings = embeddings / norms
    return embeddings


@pytest.fixture
def sample_metadata():
    """Create sample metadata for testing."""
    return [
        {'doc_id': f'doc_{i}', 'text': f'Sample text {i}'}
        for i in range(100)
    ]


@pytest.fixture
def temp_dir():
    """Create a temporary directory for testing."""
    temp_path = tempfile.mkdtemp()
    yield temp_path
    shutil.rmtree(temp_path)


class TestFAISSIndexManager:
    """Test suite for FAISSIndexManager class."""
    
    def test_init_valid_index_type(self):
        """Test initialization with valid index types."""
        for index_type in ['FLAT', 'IVFFLAT', 'HNSW']:
            manager = FAISSIndexManager(dimension=384, index_type=index_type)
            assert manager.dimension == 384
            assert manager.index_type == index_type
    
    def test_init_invalid_index_type(self):
        """Test initialization with invalid index type."""
        with pytest.raises(ValueError, match="Unsupported index type"):
            FAISSIndexManager(dimension=384, index_type='INVALID')
    
    def test_build_flat_index(self, sample_embeddings):
        """Test building a FLAT index."""
        manager = FAISSIndexManager(dimension=384, index_type='FLAT')
        index = manager.build_index(sample_embeddings)
        
        assert index.ntotal == len(sample_embeddings)
        assert index.d == 384
    
    def test_build_ivfflat_index(self, sample_embeddings):
        """Test building an IVFFLAT index."""
        manager = FAISSIndexManager(dimension=384, index_type='IVFFLAT')
        # Use small nlist for testing
        index = manager.build_index(sample_embeddings, nlist=10)
        
        assert index.ntotal == len(sample_embeddings)
        assert index.d == 384
    
    def test_build_hnsw_index(self, sample_embeddings):
        """Test building an HNSW index."""
        manager = FAISSIndexManager(dimension=384, index_type='HNSW')
        index = manager.build_index(sample_embeddings, hnsw_m=16)
        
        assert index.ntotal == len(sample_embeddings)
        assert index.d == 384
    
    def test_build_index_wrong_dimension(self, sample_embeddings):
        """Test building index with wrong embedding dimension."""
        manager = FAISSIndexManager(dimension=512, index_type='FLAT')
        
        with pytest.raises(ValueError, match="doesn't match expected dimension"):
            manager.build_index(sample_embeddings)
    
    def test_save_and_load_index(self, sample_embeddings, sample_metadata, temp_dir):
        """Test saving and loading index with metadata."""
        manager = FAISSIndexManager(dimension=384, index_type='FLAT')
        index = manager.build_index(sample_embeddings)
        
        # Save index
        manager.save_index(index, sample_metadata, temp_dir)
        
        # Verify files exist
        assert (Path(temp_dir) / 'faiss.index').exists()
        assert (Path(temp_dir) / 'metadata.pkl').exists()
        assert (Path(temp_dir) / 'index_config.json').exists()
        
        # Load index
        loaded_index, loaded_metadata = manager.load_index(temp_dir)
        
        assert loaded_index.ntotal == index.ntotal
        assert len(loaded_metadata) == len(sample_metadata)
        assert loaded_metadata[0]['doc_id'] == 'doc_0'
    
    def test_save_index_metadata_mismatch(self, sample_embeddings, sample_metadata, temp_dir):
        """Test saving index with mismatched metadata length."""
        manager = FAISSIndexManager(dimension=384, index_type='FLAT')
        index = manager.build_index(sample_embeddings)
        
        # Use wrong number of metadata entries
        wrong_metadata = sample_metadata[:50]
        
        with pytest.raises(ValueError, match="doesn't match index size"):
            manager.save_index(index, wrong_metadata, temp_dir)
    
    def test_load_index_missing_files(self, temp_dir):
        """Test loading index from directory with missing files."""
        manager = FAISSIndexManager(dimension=384, index_type='FLAT')
        
        with pytest.raises(FileNotFoundError, match="FAISS index not found"):
            manager.load_index(temp_dir)
    
    def test_search(self, sample_embeddings):
        """Test searching the index."""
        manager = FAISSIndexManager(dimension=384, index_type='FLAT')
        index = manager.build_index(sample_embeddings)
        
        # Use first embedding as query
        query = sample_embeddings[0:1]
        distances, indices = manager.search(index, query, top_k=5)
        
        # Check shapes
        assert distances.shape == (1, 5)
        assert indices.shape == (1, 5)
        
        # First result should be the query itself
        assert indices[0, 0] == 0
        # Score should be very close to 1.0 (normalized vectors)
        assert distances[0, 0] > 0.99
    
    def test_search_multiple_queries(self, sample_embeddings):
        """Test searching with multiple queries."""
        manager = FAISSIndexManager(dimension=384, index_type='FLAT')
        index = manager.build_index(sample_embeddings)
        
        # Use first 3 embeddings as queries
        queries = sample_embeddings[:3]
        distances, indices = manager.search(index, queries, top_k=5)
        
        # Check shapes
        assert distances.shape == (3, 5)
        assert indices.shape == (3, 5)
        
        # Each query's first result should be itself
        for i in range(3):
            assert indices[i, 0] == i
            assert distances[i, 0] > 0.99
    
    def test_search_single_vector(self, sample_embeddings):
        """Test searching with a single vector (1D array)."""
        manager = FAISSIndexManager(dimension=384, index_type='FLAT')
        index = manager.build_index(sample_embeddings)
        
        # Use 1D array as query
        query = sample_embeddings[0]
        distances, indices = manager.search(index, query, top_k=5)
        
        # Should automatically reshape to (1, dim)
        assert distances.shape == (1, 5)
        assert indices.shape == (1, 5)
    
    def test_ivfflat_nprobe_setting(self, sample_embeddings):
        """Test that nprobe is correctly set for IVFFLAT index."""
        manager = FAISSIndexManager(dimension=384, index_type='IVFFLAT')
        index = manager.build_index(sample_embeddings, nlist=10, nprobe=5)
        
        # IVFFLAT index should have nprobe attribute
        assert hasattr(index, 'nprobe')
        assert index.nprobe == 5
    
    def test_embedding_dtype_conversion(self):
        """Test that embeddings are converted to float32 if needed."""
        # Create float64 embeddings
        embeddings = np.random.randn(50, 384).astype(np.float64)
        norms = np.linalg.norm(embeddings, axis=1, keepdims=True)
        embeddings = embeddings / norms
        
        manager = FAISSIndexManager(dimension=384, index_type='FLAT')
        index = manager.build_index(embeddings)
        
        # Should work despite float64 input
        assert index.ntotal == len(embeddings)


if __name__ == '__main__':
    pytest.main([__file__, '-v'])
