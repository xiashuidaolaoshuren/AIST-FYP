"""
Embedding generator with GPU-accelerated batch processing.

This module provides the EmbeddingGenerator class for creating dense embeddings
from text chunks using sentence-transformers models. Supports GPU acceleration,
FP16 precision, checkpointing for long-running jobs, and progress tracking.
"""

import numpy as np
import torch
from sentence_transformers import SentenceTransformer
from typing import Any, Iterable, List, Dict, Optional, Tuple
from tqdm import tqdm
import pickle
from pathlib import Path
import time

from src.utils.logger import setup_logger
from src.utils.checkpoint_utils import (
    CHECKPOINT_SCHEMA_VERSION,
    ensure_manifest_compatible,
    load_manifest,
    save_manifest,
)


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
        checkpoint_manifest_path: Optional[str] = None,
        checkpoint_metadata: Optional[Dict[str, Any]] = None,
        checkpoint_interval: int = 10000
    ) -> np.ndarray:
        """
        Generate embeddings for a list of text chunks with checkpointing.
        
        Processes chunks in batches, saves checkpoints periodically for resume
        capability, and returns L2-normalized embeddings.
        
        Args:
            chunks: List of chunk dictionaries with 'text' field
            checkpoint_path: Path to save/load checkpoint payload (pickle, optional)
            checkpoint_manifest_path: Path to checkpoint manifest (json, optional)
            checkpoint_metadata: Additional strict compatibility fields
            checkpoint_interval: Save checkpoint every N chunks (default: 10000)
        
        Returns:
            numpy array of shape (N, embedding_dim) with L2-normalized embeddings
        
        Example:
            >>> chunks = [{'text': 'sentence 1'}, {'text': 'sentence 2'}]
            >>> embeddings = generator.generate_embeddings(chunks, 'checkpoint.pkl')
        """
        n_chunks = len(chunks)
        self.logger.info(f"Generating embeddings for {n_chunks} chunks")

        if checkpoint_interval <= 0:
            raise ValueError("checkpoint_interval must be > 0")

        manifest_path = Path(checkpoint_manifest_path) if checkpoint_manifest_path else None
        payload_path = Path(checkpoint_path) if checkpoint_path else None
        metadata = checkpoint_metadata or {}

        if payload_path and manifest_path is None:
            manifest_path = payload_path.with_suffix('.manifest.json')

        if manifest_path and payload_path is None:
            payload_path = manifest_path.with_suffix('.payload.pkl')

        if manifest_path and payload_path:
            if manifest_path.exists() != payload_path.exists():
                raise ValueError(
                    "Incomplete embedding checkpoint state detected (manifest/payload mismatch). "
                    "Reset checkpoint and retry."
                )
        
        # Check for existing checkpoint
        start_idx = 0
        embeddings_list = []

        expected_manifest = {
            'schema_version': CHECKPOINT_SCHEMA_VERSION,
            'model_name': self.model_name,
            'embedding_dim': self.embedding_dim,
            'total_chunks': n_chunks,
            **metadata,
        }

        if manifest_path and payload_path and manifest_path.exists() and payload_path.exists():
            start_idx, embeddings_list = self._load_checkpoint(
                payload_path,
                manifest_path,
                expected_manifest,
            )
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
                    if payload_path and manifest_path and (batch_end % checkpoint_interval == 0 or batch_end == n_chunks):
                        self._save_checkpoint(
                            payload_path=payload_path,
                            manifest_path=manifest_path,
                            processed_count=batch_end,
                            embeddings_list=embeddings_list,
                            manifest_fields=expected_manifest,
                        )
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
        if payload_path and payload_path.exists():
            payload_path.unlink()
            self.logger.info("Checkpoint payload removed after successful completion")

        if manifest_path and manifest_path.exists():
            manifest_path.unlink()
            self.logger.info("Checkpoint manifest removed after successful completion")
        
        return embeddings

    def generate_embeddings_streaming(
        self,
        text_batches: Iterable[List[str]],
        total_chunks: int,
        output_path: str,
        checkpoint_manifest_path: Optional[str] = None,
        checkpoint_metadata: Optional[Dict[str, Any]] = None,
        checkpoint_interval: int = 10000,
    ) -> np.ndarray:
        """
        Generate embeddings from streamed text batches and write directly to disk-backed .npy.

        This method avoids materializing all chunks/texts/embeddings in memory and supports
        resume via manifest-only checkpointing.

        Args:
            text_batches: Iterable yielding lists of text strings.
            total_chunks: Total number of chunks expected from the stream.
            output_path: Output .npy path.
            checkpoint_manifest_path: Path to checkpoint manifest (json, optional).
            checkpoint_metadata: Additional strict compatibility fields.
            checkpoint_interval: Save checkpoint every N processed chunks.

        Returns:
            Memory-mapped numpy array loaded from output_path.
        """
        if total_chunks < 0:
            raise ValueError("total_chunks must be >= 0")
        if checkpoint_interval <= 0:
            raise ValueError("checkpoint_interval must be > 0")

        if total_chunks == 0:
            raise ValueError("total_chunks must be > 0")

        output_file = Path(output_path)
        output_file.parent.mkdir(parents=True, exist_ok=True)

        manifest_path = Path(checkpoint_manifest_path) if checkpoint_manifest_path else output_file.with_suffix('.manifest.json')
        metadata = checkpoint_metadata or {}

        expected_manifest = {
            'schema_version': CHECKPOINT_SCHEMA_VERSION,
            'model_name': self.model_name,
            'embedding_dim': self.embedding_dim,
            'total_chunks': int(total_chunks),
            'output_path': str(output_file.resolve()),
            **metadata,
        }

        start_idx = 0
        if manifest_path.exists():
            manifest = load_manifest(manifest_path)
            ensure_manifest_compatible(
                manifest,
                expected_manifest,
                required_keys=['schema_version', 'model_name', 'embedding_dim', 'total_chunks', 'processed_count', 'output_path'],
            )
            start_idx = int(manifest.get('processed_count', 0))
            self.logger.info(f"Resumed streaming checkpoint: {start_idx}/{total_chunks} chunks processed")

        if output_file.exists():
            embeddings_mmap = np.load(output_file, mmap_mode='r+')
            if embeddings_mmap.shape != (total_chunks, self.embedding_dim):
                raise ValueError(
                    f"Existing output shape {embeddings_mmap.shape} does not match expected "
                    f"({total_chunks}, {self.embedding_dim})"
                )
        else:
            embeddings_mmap = np.lib.format.open_memmap(
                output_file,
                mode='w+',
                dtype=np.float32,
                shape=(total_chunks, self.embedding_dim),
            )

        stream_position = 0
        processed_count = start_idx
        start_time = time.time()

        with tqdm(total=total_chunks, initial=start_idx, desc="Generating embeddings") as pbar:
            for batch_texts in text_batches:
                batch_size = len(batch_texts)
                if batch_size == 0:
                    continue

                batch_start = stream_position
                batch_end = stream_position + batch_size
                stream_position = batch_end

                if batch_start >= total_chunks:
                    break

                if batch_end <= start_idx:
                    continue

                if batch_start < start_idx:
                    skip = start_idx - batch_start
                    batch_texts = batch_texts[skip:]
                    batch_start = start_idx

                if batch_start + len(batch_texts) > total_chunks:
                    batch_texts = batch_texts[: total_chunks - batch_start]

                if not batch_texts:
                    continue

                try:
                    batch_embeddings = self.model.encode(
                        batch_texts,
                        batch_size=self.batch_size,
                        show_progress_bar=False,
                        convert_to_numpy=True,
                        normalize_embeddings=True,
                    )
                    batch_embeddings = np.asarray(batch_embeddings, dtype=np.float32)
                    embeddings_mmap[batch_start:batch_start + len(batch_texts)] = batch_embeddings
                    processed_count = batch_start + len(batch_texts)
                    pbar.update(len(batch_texts))

                    if processed_count % checkpoint_interval == 0 or processed_count == total_chunks:
                        embeddings_mmap.flush()
                        save_manifest(
                            manifest_path,
                            {
                                **expected_manifest,
                                'processed_count': int(processed_count),
                            },
                        )
                        self.logger.info(f"Checkpoint saved at {processed_count}/{total_chunks} chunks")
                except Exception as e:
                    self.logger.error(f"Error processing streamed batch at offset {batch_start}: {e}")
                    raise

                if processed_count >= total_chunks:
                    break

        if processed_count != total_chunks:
            raise ValueError(
                f"Stream ended early: processed {processed_count} chunks, expected {total_chunks}."
            )

        embeddings_mmap.flush()
        del embeddings_mmap

        elapsed_time = time.time() - start_time
        chunks_per_sec = total_chunks / elapsed_time if elapsed_time > 0 else 0
        self.logger.info(
            f"Streaming embedding generation complete: {total_chunks} chunks, "
            f"{elapsed_time:.2f}s ({chunks_per_sec:.2f} chunks/sec)"
        )

        if manifest_path.exists():
            manifest_path.unlink()
            self.logger.info("Checkpoint manifest removed after successful completion")

        embeddings = np.load(output_file, mmap_mode='r')
        return embeddings
    
    def _save_checkpoint(
        self,
        payload_path: Path,
        manifest_path: Path,
        processed_count: int,
        embeddings_list: List[np.ndarray],
        manifest_fields: Dict[str, Any],
    ) -> None:
        """
        Save checkpoint for resume capability.
        
        Args:
            payload_path: Path to save checkpoint payload
            manifest_path: Path to save checkpoint manifest
            processed_count: Number of chunks processed so far
            embeddings_list: List of embedding arrays
        """
        payload_data = {
            'processed_count': processed_count,
            'embeddings_list': embeddings_list,
            'model_name': self.model_name,
            'embedding_dim': self.embedding_dim
        }
        
        payload_path.parent.mkdir(parents=True, exist_ok=True)
        with open(payload_path, 'wb') as f:
            pickle.dump(payload_data, f)

        save_manifest(
            manifest_path,
            {
                **manifest_fields,
                'processed_count': int(processed_count),
                'payload_path': str(payload_path),
            }
        )
    
    def _load_checkpoint(
        self,
        payload_path: Path,
        manifest_path: Path,
        expected_manifest: Dict[str, Any],
    ) -> Tuple[int, List[np.ndarray]]:
        """
        Load checkpoint to resume processing.
        
        Args:
            payload_path: Path to checkpoint payload file
            manifest_path: Path to checkpoint manifest file
            expected_manifest: Expected compatibility fields
        
        Returns:
            Tuple of (processed_count, embeddings_list)
        
        Raises:
            ValueError: If checkpoint model doesn't match current model
        """
        manifest = load_manifest(manifest_path)
        ensure_manifest_compatible(
            manifest,
            expected_manifest,
            required_keys=['schema_version', 'model_name', 'embedding_dim', 'total_chunks', 'processed_count']
        )

        with open(payload_path, 'rb') as f:
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

        processed_count = checkpoint_data['processed_count']
        if processed_count != manifest['processed_count']:
            raise ValueError(
                f"Checkpoint mismatch: payload processed_count={processed_count} "
                f"!= manifest processed_count={manifest['processed_count']}"
            )
        
        return processed_count, checkpoint_data['embeddings_list']
    
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
