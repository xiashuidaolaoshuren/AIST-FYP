"""
Embedding generator with GPU-accelerated batch processing.

This module provides the EmbeddingGenerator class for creating dense embeddings
from text chunks using sentence-transformers models. Supports GPU acceleration,
FP16 precision, checkpointing for long-running jobs, and progress tracking.
"""

import numpy as np
import torch
from sentence_transformers import SentenceTransformer
from typing import List, Dict, Optional, Tuple
from tqdm import tqdm
import pickle
from pathlib import Path
import time

from src.utils.logger import setup_logger


class EmbeddingGenerator:
    """
    Generate dense embeddings from text using sentence-transformers.
    
    Supports GPU acceleration, FP16 precision for memory efficiency,
    checkpointing for resume capability, and L2 normalization for
    cosine similarity with inner product in FAISS.
    
    Attributes:
        model: SentenceTransformer model
        batch_size: Number of texts to process per batch
        device: Device to run on ('cuda' or 'cpu')
        use_fp16: Whether to use FP16 precision (GPU only)
        logger: Logger instance
    
    Example:
        >>> generator = EmbeddingGenerator('sentence-transformers/all-MiniLM-L6-v2')
        >>> chunks = [{'text': 'Hello world'}, {'text': 'Test sentence'}]
        >>> embeddings = generator.generate_embeddings(chunks)
        >>> print(embeddings.shape)  # (2, 384)
    """
    
    def __init__(
        self,
        model_name: str,
        batch_size: int = 16,
        device: str = 'cuda',
        use_fp16: bool = True
    ):
        """
        Initialize the embedding generator.
        
        Args:
            model_name: Name or path of the sentence-transformers model
            batch_size: Batch size for processing (default: 16 for 8GB VRAM)
            device: Device to use ('cuda' or 'cpu', default: 'cuda')
            use_fp16: Use FP16 precision for 2x speedup and 50% memory reduction (default: True)
        """
        self.model_name = model_name
        self.batch_size = batch_size
        self.device = device
        self.use_fp16 = use_fp16 and device == 'cuda'
        self.logger = setup_logger(__name__)
        
        self.logger.info(f"Loading model: {model_name}")
        self.logger.info(f"Device: {device}, Batch size: {batch_size}, FP16: {self.use_fp16}")
        
        # Load model
        self.model = SentenceTransformer(model_name, device=device)
        
        # Enable FP16 if requested and on CUDA
        if self.use_fp16:
            self.model.half()
            self.logger.info("Enabled FP16 precision for faster inference")
        
        self.embedding_dim = self.model.get_sentence_embedding_dimension()
        self.logger.info(f"Model loaded successfully. Embedding dimension: {self.embedding_dim}")
    
    def generate_embeddings(
        self,
        chunks: List[Dict[str, any]],
        checkpoint_path: Optional[str] = None,
        checkpoint_interval: int = 10000
    ) -> np.ndarray:
        """
        Generate embeddings for a list of text chunks with checkpointing.
        
        Processes chunks in batches, saves checkpoints periodically for resume
        capability, and returns L2-normalized embeddings.
        
        Args:
            chunks: List of chunk dictionaries with 'text' field
            checkpoint_path: Path to save/load checkpoints (optional)
            checkpoint_interval: Save checkpoint every N chunks (default: 10000)
        
        Returns:
            numpy array of shape (N, embedding_dim) with L2-normalized embeddings
        
        Example:
            >>> chunks = [{'text': 'sentence 1'}, {'text': 'sentence 2'}]
            >>> embeddings = generator.generate_embeddings(chunks, 'checkpoint.pkl')
        """
        n_chunks = len(chunks)
        self.logger.info(f"Generating embeddings for {n_chunks} chunks")
        
        # Check for existing checkpoint
        start_idx = 0
        embeddings_list = []
        
        if checkpoint_path and Path(checkpoint_path).exists():
            start_idx, embeddings_list = self._load_checkpoint(checkpoint_path)
            self.logger.info(f"Resumed from checkpoint: {start_idx}/{n_chunks} chunks processed")
        
        # Extract texts
        texts = [chunk['text'] for chunk in chunks]
        
        # Process remaining chunks
        start_time = time.time()
        
        with tqdm(total=n_chunks, initial=start_idx, desc="Generating embeddings") as pbar:
            for i in range(start_idx, n_chunks, self.batch_size):
                batch_end = min(i + self.batch_size, n_chunks)
                batch_texts = texts[i:batch_end]
                
                try:
                    # Generate embeddings for batch
                    batch_embeddings = self.model.encode(
                        batch_texts,
                        batch_size=self.batch_size,
                        show_progress_bar=False,
                        convert_to_numpy=True,
                        normalize_embeddings=True  # L2 normalization
                    )
                    
                    embeddings_list.append(batch_embeddings)
                    pbar.update(len(batch_texts))
                    
                    # Save checkpoint periodically
                    if checkpoint_path and (batch_end % checkpoint_interval == 0 or batch_end == n_chunks):
                        self._save_checkpoint(checkpoint_path, batch_end, embeddings_list)
                        self.logger.info(f"Checkpoint saved at {batch_end}/{n_chunks} chunks")
                
                except Exception as e:
                    self.logger.error(f"Error processing batch {i}-{batch_end}: {e}")
                    raise
        
        # Combine all embeddings
        embeddings = np.vstack(embeddings_list)
        
        elapsed_time = time.time() - start_time
        chunks_per_sec = n_chunks / elapsed_time if elapsed_time > 0 else 0
        
        self.logger.info(
            f"Embedding generation complete: {n_chunks} chunks, "
            f"{elapsed_time:.2f}s ({chunks_per_sec:.2f} chunks/sec)"
        )
        
        # Verify L2 normalization
        norms = np.linalg.norm(embeddings, axis=1)
        if not np.allclose(norms, 1.0, atol=1e-5):
            self.logger.warning("Embeddings may not be properly normalized")
        else:
            self.logger.info("Embeddings are L2 normalized")
        
        # Clean up checkpoint if successful
        if checkpoint_path and Path(checkpoint_path).exists():
            Path(checkpoint_path).unlink()
            self.logger.info("Checkpoint file removed after successful completion")
        
        return embeddings
    
    def _save_checkpoint(
        self,
        checkpoint_path: str,
        processed_count: int,
        embeddings_list: List[np.ndarray]
    ) -> None:
        """
        Save checkpoint for resume capability.
        
        Args:
            checkpoint_path: Path to save checkpoint
            processed_count: Number of chunks processed so far
            embeddings_list: List of embedding arrays
        """
        checkpoint_data = {
            'processed_count': processed_count,
            'embeddings_list': embeddings_list,
            'model_name': self.model_name,
            'embedding_dim': self.embedding_dim
        }
        
        with open(checkpoint_path, 'wb') as f:
            pickle.dump(checkpoint_data, f)
    
    def _load_checkpoint(self, checkpoint_path: str) -> Tuple[int, List[np.ndarray]]:
        """
        Load checkpoint to resume processing.
        
        Args:
            checkpoint_path: Path to checkpoint file
        
        Returns:
            Tuple of (processed_count, embeddings_list)
        
        Raises:
            ValueError: If checkpoint model doesn't match current model
        """
        with open(checkpoint_path, 'rb') as f:
            checkpoint_data = pickle.load(f)
        
        # Verify checkpoint compatibility
        if checkpoint_data['model_name'] != self.model_name:
            raise ValueError(
                f"Checkpoint model ({checkpoint_data['model_name']}) "
                f"doesn't match current model ({self.model_name})"
            )
        
        if checkpoint_data['embedding_dim'] != self.embedding_dim:
            raise ValueError(
                f"Checkpoint embedding dimension ({checkpoint_data['embedding_dim']}) "
                f"doesn't match current model dimension ({self.embedding_dim})"
            )
        
        return checkpoint_data['processed_count'], checkpoint_data['embeddings_list']
    
    def get_embedding_dimension(self) -> int:
        """
        Get the embedding dimension of the model.
        
        Returns:
            Embedding dimension (e.g., 384 for all-MiniLM-L6-v2)
        """
        return self.embedding_dim
    
    def encode_single(self, text: str) -> np.ndarray:
        """
        Encode a single text into an embedding.
        
        Convenience method for single-text encoding.
        
        Args:
            text: Text to encode
        
        Returns:
            L2-normalized embedding vector
        
        Example:
            >>> embedding = generator.encode_single("Hello world")
            >>> print(embedding.shape)  # (384,)
        """
        embedding = self.model.encode(
            [text],
            convert_to_numpy=True,
            normalize_embeddings=True
        )[0]
        
        return embedding
