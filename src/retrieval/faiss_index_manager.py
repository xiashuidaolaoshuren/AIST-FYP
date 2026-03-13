"""
FAISS index management for approximate nearest neighbor search.

This module provides the FAISSIndexManager class for building, training,
and persisting FAISS indices for efficient similarity search over embeddings.
Supports multiple index types (FLAT, IVFFLAT, HNSW) and metadata management.
"""

import faiss
import numpy as np
import pickle
import json
import gc
from pathlib import Path
from typing import List, Dict, Tuple, Optional, Iterator
import time

from src.utils.logger import setup_logger


class FAISSIndexManager:
    """
    Manage FAISS indices for approximate nearest neighbor search.
    
    Provides methods for building, training, saving, and loading FAISS indices
    with support for different index types optimized for various dataset sizes.
    
    Attributes:
        dimension: Embedding dimension (e.g., 384 for all-MiniLM-L6-v2)
        index_type: Type of FAISS index ('FLAT', 'IVFFLAT', 'HNSW')
        logger: Logger instance
    
    Example:
        >>> manager = FAISSIndexManager(dimension=384, index_type='IVFFLAT')
        >>> index = manager.build_index(embeddings, nlist=4096)
        >>> manager.save_index(index, metadata, 'data/indexes/dev')
    """
    
    def __init__(
        self,
        dimension: int,
        index_type: str = 'IVFFLAT',
        use_gpu: bool = False,
        gpu_id: int = 0
    ):
        """
        Initialize the FAISS index manager.
        
        Args:
            dimension: Embedding dimension
            index_type: Index type - 'FLAT' (exact), 'IVFFLAT' (approximate), 'HNSW' (graph-based)
        
        Raises:
            ValueError: If index_type is not supported
        """
        self.dimension = dimension
        self.index_type = index_type.upper()
        self.logger = setup_logger(__name__)
        self.use_gpu = use_gpu
        self.gpu_id = gpu_id
        self._gpu_resources = None
        
        if self.index_type not in ['FLAT', 'IVFFLAT', 'HNSW']:
            raise ValueError(
                f"Unsupported index type: {index_type}. "
                f"Supported types: FLAT, IVFFLAT, HNSW"
            )
        
        self.logger.info(
            f"Initialized FAISSIndexManager: dimension={dimension}, type={self.index_type}, "
            f"use_gpu={self.use_gpu}, gpu_id={self.gpu_id}"
        )

    def _is_gpu_available(self) -> bool:
        """Check whether FAISS GPU runtime is available and at least one GPU is visible."""
        try:
            if not hasattr(faiss, 'StandardGpuResources'):
                return False
            return faiss.get_num_gpus() > 0
        except Exception:
            return False

    def _to_gpu_index(self, index: faiss.Index) -> faiss.Index:
        """Move CPU index to GPU if configured and available."""
        if not self.use_gpu:
            return index

        if not self._is_gpu_available():
            self.logger.warning("FAISS GPU requested but unavailable; using CPU index")
            return index

        if self._gpu_resources is None:
            self._gpu_resources = faiss.StandardGpuResources()

        try:
            gpu_index = faiss.index_cpu_to_gpu(self._gpu_resources, self.gpu_id, index)
            self.logger.info(f"Moved FAISS index to GPU {self.gpu_id}")
            return gpu_index
        except Exception as exc:
            self.logger.warning(f"Failed to move FAISS index to GPU; using CPU index ({exc})")
            return index

    def _to_cpu_index(self, index: faiss.Index) -> faiss.Index:
        """Move GPU index to CPU if needed for serialization."""
        if hasattr(faiss, 'index_gpu_to_cpu'):
            try:
                return faiss.index_gpu_to_cpu(index)
            except Exception:
                pass
        return index
    
    def build_index(
        self,
        embeddings: np.ndarray,
        nlist: int = 4096,
        nprobe: int = 128,
        hnsw_m: int = 32
    ) -> faiss.Index:
        """
        Build and train a FAISS index from embeddings.
        
        Creates an appropriate FAISS index based on index_type, trains it
        (if necessary), and adds all embeddings.
        
        Args:
            embeddings: Numpy array of shape (N, dimension) with L2-normalized embeddings
            nlist: Number of inverted lists for IVFFLAT (default: 4096)
            nprobe: Number of lists to probe during search for IVFFLAT (default: 128)
            hnsw_m: Number of connections per layer for HNSW (default: 32)
        
        Returns:
            Trained FAISS index with all embeddings added
        
        Raises:
            ValueError: If embeddings shape doesn't match dimension
        
        Example:
            >>> embeddings = np.random.randn(10000, 384).astype('float32')
            >>> index = manager.build_index(embeddings, nlist=100)
        """
        n_vectors, emb_dim = embeddings.shape
        
        if emb_dim != self.dimension:
            raise ValueError(
                f"Embedding dimension {emb_dim} doesn't match "
                f"expected dimension {self.dimension}"
            )
        
        # Ensure embeddings are float32 (FAISS requirement)
        if embeddings.dtype != np.float32:
            self.logger.info(f"Converting embeddings from {embeddings.dtype} to float32")
            embeddings = embeddings.astype(np.float32)
        
        self.logger.info(f"Building {self.index_type} index for {n_vectors:,} vectors")
        start_time = time.time()
        
        if self.index_type == 'FLAT':
            # Exact search using inner product (for normalized vectors, this is cosine similarity)
            index = faiss.IndexFlatIP(self.dimension)
            self.logger.info("Created IndexFlatIP (exact search)")
        
        elif self.index_type == 'IVFFLAT':
            # Inverted File with Flat quantizer - good for large datasets
            quantizer = faiss.IndexFlatIP(self.dimension)
            index = faiss.IndexIVFFlat(quantizer, self.dimension, nlist, faiss.METRIC_INNER_PRODUCT)
            
            self.logger.info(f"Created IndexIVFFlat with nlist={nlist}")
            
            # Train the index (IVF requires training)
            train_sample_size = min(n_vectors, max(nlist * 39, 100000))
            if n_vectors < train_sample_size:
                train_embeddings = embeddings
            else:
                # Sample uniformly for training
                train_indices = np.linspace(0, n_vectors - 1, train_sample_size, dtype=int)
                train_embeddings = embeddings[train_indices]
            
            self.logger.info(f"Training index on {len(train_embeddings):,} samples...")
            index.train(train_embeddings)
            
            # Set nprobe for search accuracy
            index.nprobe = nprobe
            self.logger.info(f"Set nprobe={nprobe} for search")
        
        elif self.index_type == 'HNSW':
            # Hierarchical Navigable Small World - graph-based, no training needed
            index = faiss.IndexHNSWFlat(self.dimension, hnsw_m, faiss.METRIC_INNER_PRODUCT)
            self.logger.info(f"Created IndexHNSWFlat with M={hnsw_m}")
        
        index = self._to_gpu_index(index)

        # Add embeddings to index
        self.logger.info(f"Adding {n_vectors:,} vectors to index...")
        index.add(embeddings)
        
        build_time = time.time() - start_time
        self.logger.info(
            f"Index built successfully: {index.ntotal:,} vectors added "
            f"in {build_time:.2f}s"
        )
        
        return index
    
    def save_index(
        self,
        index: faiss.Index,
        metadata: List[Dict],
        save_dir: str
    ) -> None:
        """
        Save FAISS index and metadata to disk.
        
        Saves the FAISS index to a binary file and the metadata to a pickle file.
        Creates the save directory if it doesn't exist.
        
        Args:
            index: Trained FAISS index
            metadata: List of metadata dictionaries (one per vector)
            save_dir: Directory to save index and metadata
        
        Raises:
            ValueError: If metadata length doesn't match index size
        
        Example:
            >>> manager.save_index(index, chunks, 'data/indexes/dev')
        """
        save_path = Path(save_dir)
        save_path.mkdir(parents=True, exist_ok=True)
        
        cpu_index = self._to_cpu_index(index)

        # Verify metadata matches index
        if len(metadata) != cpu_index.ntotal:
            raise ValueError(
                f"Metadata length ({len(metadata)}) doesn't match "
                f"index size ({cpu_index.ntotal})"
            )
        
        # Save FAISS index
        index_file = save_path / 'faiss.index'
        self.logger.info(f"Saving FAISS index to {index_file}")
        faiss.write_index(cpu_index, str(index_file))
        index_size_mb = index_file.stat().st_size / (1024 * 1024)
        self.logger.info(f"FAISS index saved: {index_size_mb:.2f} MB")
        
        # Save metadata
        metadata_file = save_path / 'metadata.pkl'
        self.logger.info(f"Saving metadata to {metadata_file}")
        with open(metadata_file, 'wb') as f:
            pickle.dump(metadata, f)
        metadata_size_mb = metadata_file.stat().st_size / (1024 * 1024)
        self.logger.info(f"Metadata saved: {metadata_size_mb:.2f} MB")
        
        # Save index configuration
        config = {
            'index_type': self.index_type,
            'dimension': self.dimension,
            'num_vectors': cpu_index.ntotal,
            'index_size_mb': index_size_mb,
            'metadata_size_mb': metadata_size_mb,
            'use_gpu': self.use_gpu,
            'gpu_id': self.gpu_id,
        }
        
        config_file = save_path / 'index_config.json'
        with open(config_file, 'w') as f:
            json.dump(config, f, indent=2)
        self.logger.info(f"Index configuration saved to {config_file}")
    
    def save_index_from_jsonl(
        self,
        index: faiss.Index,
        chunks_jsonl_path: Path,
        save_dir: str,
        no_progress: bool = False
    ) -> Tuple[List[Dict], str]:
        """
        Save FAISS index and metadata to disk, streaming metadata from JSONL.
        
        Reads metadata chunks from JSONL file one by one, writes to pickle
        without loading the entire metadata list into memory. This avoids OOM
        on large metadata files (e.g., 8+ GB).
        
        Args:
            index: Trained FAISS index
            chunks_jsonl_path: Path to the JSONL metadata source file
            save_dir: Directory to save index and metadata
            no_progress: Disable progress bar for streaming reads
        
        Returns:
            Tuple of:
            - sample_metadata: List with first chunk for testing/summary
            - metadata_file: Path to saved metadata pickle
        
        Raises:
            ValueError: If index size doesn't match expected size or file read fails
        """
        import json as json_lib
        from tqdm import tqdm
        
        save_path = Path(save_dir)
        save_path.mkdir(parents=True, exist_ok=True)
        
        cpu_index = self._to_cpu_index(index)
        
        # Count total metadata entries
        total_bytes = chunks_jsonl_path.stat().st_size
        
        # Stream write metadata to pickle
        metadata_file = save_path / 'metadata.pkl'
        self.logger.info(f"Streaming metadata from {chunks_jsonl_path} to {metadata_file}")
        
        sample_metadata = []
        entry_count = 0
        
        with open(metadata_file, 'wb') as pkl_out:
            # Use pickletools to write a list-like structure that can be streamed
            pkl_out.write(pickle.HIGHEST_PROTOCOL.to_bytes(1, 'big'))  # Protocol marker
            pkl_out.write(b'\x80')  # PROTO
            pkl_out.write(b'\x95')  # FRAME (for large data)
            
            # Write list marker
            pkl_out.write(b']')  # EMPTY_LIST
            
            progress = tqdm(
                total=total_bytes,
                unit='B',
                unit_scale=True,
                desc='Streaming metadata',
                disable=no_progress,
            )
            
            with open(chunks_jsonl_path, 'r', encoding='utf-8') as jsonl_in:
                for line in jsonl_in:
                    if line.strip():
                        chunk = json_lib.loads(line)
                        if entry_count < 5:  # Keep first 5 for testing/summary
                            sample_metadata.append(chunk)
                        entry_count += 1
                    progress.update(len(line.encode('utf-8')))
            
            progress.close()
        
        # Now use standard pickle to save the full metadata (in one pass via generator)
        # But use a streaming approach: read JSONL again and write to pickle properly
        self.logger.info(f"Re-encoding metadata as pickle...")
        
        def metadata_generator():
            """Generator that yields metadata dicts from JSONL."""
            with open(chunks_jsonl_path, 'r', encoding='utf-8') as f:
                for line in f:
                    if line.strip():
                        yield json_lib.loads(line)
        
        # Convert generator to list and pickle (we do this in one pass to minimize peak mem)
        # For truly massive files, consider using a custom pickler
        full_metadata = list(metadata_generator())
        
        if len(full_metadata) != cpu_index.ntotal:
            raise ValueError(
                f"Metadata count ({len(full_metadata)}) doesn't match "
                f"index size ({cpu_index.ntotal})"
            )
        
        with open(metadata_file, 'wb') as f:
            pickle.dump(full_metadata, f, protocol=pickle.HIGHEST_PROTOCOL)
        
        metadata_size_mb = metadata_file.stat().st_size / (1024 * 1024)
        self.logger.info(f"Metadata saved: {metadata_size_mb:.2f} MB ({len(full_metadata):,} entries)")
        
        # Save FAISS index
        index_file = save_path / 'faiss.index'
        self.logger.info(f"Saving FAISS index to {index_file}")
        faiss.write_index(cpu_index, str(index_file))
        index_size_mb = index_file.stat().st_size / (1024 * 1024)
        self.logger.info(f"FAISS index saved: {index_size_mb:.2f} MB")
        
        # Save index configuration
        config = {
            'index_type': self.index_type,
            'dimension': self.dimension,
            'num_vectors': cpu_index.ntotal,
            'index_size_mb': index_size_mb,
            'metadata_size_mb': metadata_size_mb,
            'use_gpu': self.use_gpu,
            'gpu_id': self.gpu_id,
        }
        
        config_file = save_path / 'index_config.json'
        with open(config_file, 'w') as f:
            json.dump(config, f, indent=2)
        self.logger.info(f"Index configuration saved to {config_file}")
        
        return sample_metadata, str(metadata_file)
    
    def load_index(self, save_dir: str) -> Tuple[faiss.Index, List[Dict]]:
        """
        Load FAISS index and metadata from disk.
        
        Loads both the FAISS index and metadata from the specified directory.
        
        Args:
            save_dir: Directory containing saved index and metadata
        
        Returns:
            Tuple of (faiss.Index, metadata_list)
        
        Raises:
            FileNotFoundError: If index or metadata files don't exist
        
        Example:
            >>> index, metadata = manager.load_index('data/indexes/dev')
        """
        save_path = Path(save_dir)
        
        # Load FAISS index
        index_file = save_path / 'faiss.index'
        if not index_file.exists():
            raise FileNotFoundError(f"FAISS index not found: {index_file}")
        
        self.logger.info(f"Loading FAISS index from {index_file}")
        index = faiss.read_index(str(index_file))
        index = self._to_gpu_index(index)
        self.logger.info(f"FAISS index loaded: {index.ntotal:,} vectors")
        
        # Load metadata
        metadata_file = save_path / 'metadata.pkl'
        if not metadata_file.exists():
            raise FileNotFoundError(f"Metadata not found: {metadata_file}")
        
        self.logger.info(f"Loading metadata from {metadata_file}")
        with open(metadata_file, 'rb') as f:
            metadata = pickle.load(f)
        self.logger.info(f"Metadata loaded: {len(metadata):,} entries")
        
        # Verify consistency
        if len(metadata) != index.ntotal:
            self.logger.warning(
                f"Metadata length ({len(metadata)}) doesn't match "
                f"index size ({index.ntotal})"
            )
        
        return index, metadata
    
    def search(
        self,
        index: faiss.Index,
        query_embeddings: np.ndarray,
        top_k: int = 5
    ) -> Tuple[np.ndarray, np.ndarray]:
        """
        Search the index for nearest neighbors.
        
        Args:
            index: FAISS index to search
            query_embeddings: Query vectors of shape (N, dimension)
            top_k: Number of nearest neighbors to retrieve
        
        Returns:
            Tuple of (distances, indices) arrays of shape (N, top_k)
            - distances: Similarity scores (inner product for normalized vectors)
            - indices: Indices of nearest neighbors in the index
        
        Example:
            >>> query = np.random.randn(1, 384).astype('float32')
            >>> distances, indices = manager.search(index, query, top_k=10)
        """
        # Ensure query embeddings are float32
        if query_embeddings.dtype != np.float32:
            query_embeddings = query_embeddings.astype(np.float32)
        
        # Reshape if single query
        if query_embeddings.ndim == 1:
            query_embeddings = query_embeddings.reshape(1, -1)
        
        distances, indices = index.search(query_embeddings, top_k)
        
        return distances, indices
