"""
Build FAISS index from embeddings.

This script loads pre-computed embeddings and builds a FAISS index for
efficient similarity search. Supports different strategies (development, validation, production)
and index types (FLAT, IVFFLAT, HNSW).

Usage:
    python scripts/build_faiss_index.py --strategy development
    python scripts/build_faiss_index.py --strategy validation
    python scripts/build_faiss_index.py --strategy production --index-type IVFFLAT --nlist 8192
"""

import argparse
import gc
import json
import sys
from pathlib import Path
import numpy as np
import faiss
from tqdm import tqdm

# Add project root to path
sys.path.insert(0, str(Path(__file__).parent.parent))

from src.retrieval import FAISSIndexManager
from src.utils.config import Config
from src.utils.logger import setup_logger
from src.utils.checkpoint_utils import (
    CHECKPOINT_SCHEMA_VERSION,
    ensure_manifest_compatible,
    file_fingerprint,
    load_manifest,
    save_manifest,
)


def load_embeddings(embeddings_path: Path) -> np.ndarray:
    """Load embeddings from .npy file."""
    logger = setup_logger(__name__)
    logger.info(f"Loading embeddings from {embeddings_path}")
    
    embeddings = np.load(embeddings_path, mmap_mode='r')
    logger.info(f"Loaded embeddings: shape={embeddings.shape}, dtype={embeddings.dtype}")
    
    return embeddings


def load_chunk_metadata(chunks_path: Path, no_progress: bool = False) -> list:
    """Load chunk metadata from .jsonl file."""
    logger = setup_logger(__name__)
    logger.info(f"Loading chunk metadata from {chunks_path}")
    total_size_bytes = chunks_path.stat().st_size

    metadata = []
    with open(chunks_path, 'r', encoding='utf-8') as f:
        progress_bar = tqdm(
            total=total_size_bytes,
            unit='B',
            unit_scale=True,
            desc='Loading metadata',
            disable=no_progress,
        )
        for line in f:
            progress_bar.update(len(line.encode('utf-8')))
            metadata.append(json.loads(line))
        progress_bar.close()
    
    logger.info(f"Loaded {len(metadata):,} chunk metadata entries")
    
    return metadata


def count_chunk_metadata(chunks_path: Path, no_progress: bool = False) -> int:
    """Count metadata rows in .jsonl file without loading all entries in RAM."""
    logger = setup_logger(__name__)
    logger.info(f"Counting chunk metadata rows in {chunks_path}")
    total_size_bytes = chunks_path.stat().st_size

    count = 0
    with open(chunks_path, 'r', encoding='utf-8') as f:
        progress_bar = tqdm(
            total=total_size_bytes,
            unit='B',
            unit_scale=True,
            desc='Counting metadata',
            disable=no_progress,
        )
        for line in f:
            progress_bar.update(len(line.encode('utf-8')))
            if line.strip():
                count += 1
        progress_bar.close()

    logger.info(f"Counted {count:,} metadata rows")
    return count


def create_faiss_index(
    embeddings: np.ndarray,
    index_type: str,
    dimension: int,
    nlist: int,
    nprobe: int,
    hnsw_m: int,
    logger,
):
    """Create and train (if required) FAISS index, without adding vectors."""
    index_type = index_type.upper()

    if index_type == 'FLAT':
        return faiss.IndexFlatIP(dimension)

    if index_type == 'HNSW':
        return faiss.IndexHNSWFlat(dimension, hnsw_m, faiss.METRIC_INNER_PRODUCT)

    if index_type == 'IVFFLAT':
        quantizer = faiss.IndexFlatIP(dimension)
        index = faiss.IndexIVFFlat(quantizer, dimension, nlist, faiss.METRIC_INNER_PRODUCT)

        n_vectors = len(embeddings)
        train_sample_size = min(n_vectors, max(nlist * 39, 100000))

        if n_vectors <= train_sample_size:
            train_embeddings = np.asarray(embeddings, dtype=np.float32)
        else:
            train_indices = np.linspace(0, n_vectors - 1, train_sample_size, dtype=int)
            train_embeddings = np.asarray(embeddings[train_indices], dtype=np.float32)

        logger.info(f"Training IVFFLAT index on {len(train_embeddings):,} samples...")
        index.train(train_embeddings)
        index.nprobe = nprobe
        return index

    raise ValueError(f"Unsupported index type: {index_type}")


def faiss_gpu_available() -> bool:
    """Return True if FAISS GPU runtime is available and a CUDA device is visible."""
    try:
        if not hasattr(faiss, 'StandardGpuResources'):
            return False
        return faiss.get_num_gpus() > 0
    except Exception:
        return False


def cpu_to_gpu_index(index: faiss.Index, gpu_id: int, logger):
    """Move CPU index to GPU if possible; otherwise return original index."""
    if not faiss_gpu_available():
        logger.warning("FAISS GPU requested but unavailable; continuing with CPU index")
        return index, None

    try:
        resources = faiss.StandardGpuResources()
        gpu_index = faiss.index_cpu_to_gpu(resources, gpu_id, index)
        logger.info(f"Moved FAISS index to GPU {gpu_id}")
        return gpu_index, resources
    except Exception as exc:
        logger.warning(f"Failed to move FAISS index to GPU; using CPU index ({exc})")
        return index, None


def to_cpu_index(index: faiss.Index):
    """Convert index to CPU form when needed for serialization."""
    if hasattr(faiss, 'index_gpu_to_cpu'):
        try:
            return faiss.index_gpu_to_cpu(index)
        except Exception:
            pass
    return index


def build_checkpoint_paths(checkpoint_dir: Path, strategy: str):
    checkpoint_dir.mkdir(parents=True, exist_ok=True)
    manifest_path = checkpoint_dir / f"faiss_build_{strategy}.json"
    partial_index_path = checkpoint_dir / f"faiss_build_{strategy}.partial.index"
    return manifest_path, partial_index_path


def commit_faiss_checkpoint(
    index: faiss.Index,
    partial_index_path: Path,
    manifest_path: Path,
    expected_manifest: dict,
    added_count: int,
    logger,
):
    """Write FAISS checkpoint data first, then manifest as commit marker."""
    checkpoint_index = to_cpu_index(index)
    tmp_index_path = partial_index_path.with_suffix(partial_index_path.suffix + '.tmp')
    faiss.write_index(checkpoint_index, str(tmp_index_path))
    tmp_index_path.replace(partial_index_path)
    save_manifest(
        manifest_path,
        {
            **expected_manifest,
            'added_count': int(added_count),
        },
    )
    logger.info(f"Checkpoint saved at {added_count:,}/{expected_manifest['n_vectors']:,} vectors")


def main():
    parser = argparse.ArgumentParser(description='Build FAISS index from embeddings')
    parser.add_argument(
        '--strategy',
        type=str,
        choices=['development', 'validation', 'production'],
        default='development',
        help='Build strategy (development, validation, or production)'
    )
    parser.add_argument(
        '--index-type',
        type=str,
        choices=['FLAT', 'IVFFLAT', 'HNSW'],
        default='IVFFLAT',
        help='FAISS index type (default: IVFFLAT)'
    )
    parser.add_argument(
        '--nlist',
        type=int,
        default=4096,
        help='Number of inverted lists for IVFFLAT (default: 4096)'
    )
    parser.add_argument(
        '--nprobe',
        type=int,
        default=128,
        help='Number of lists to probe during search for IVFFLAT (default: 128)'
    )
    parser.add_argument(
        '--hnsw-m',
        type=int,
        default=32,
        help='Number of connections per layer for HNSW (default: 32)'
    )
    parser.add_argument(
        '--config',
        type=str,
        default='config.yaml',
        help='Path to configuration file (default: config.yaml)'
    )
    parser.add_argument(
        '--resume',
        action='store_true',
        help='Resume from existing FAISS build checkpoint if available'
    )
    parser.add_argument(
        '--no-resume',
        action='store_true',
        help='Disable checkpoint loading and build from scratch'
    )
    parser.add_argument(
        '--reset-checkpoint',
        action='store_true',
        help='Delete FAISS build checkpoint before starting'
    )
    parser.add_argument(
        '--checkpoint-dir',
        type=str,
        default=None,
        help='Directory for FAISS checkpoints (default: from config)'
    )
    parser.add_argument(
        '--checkpoint-interval',
        type=int,
        default=None,
        help='Save checkpoint every N added vectors (default: from config)'
    )
    parser.add_argument(
        '--add-batch-size',
        type=int,
        default=None,
        help='Vectors to add per batch during index construction (default: from config)'
    )
    parser.add_argument(
        '--use-gpu',
        action=argparse.BooleanOptionalAction,
        default=None,
        help='Enable/disable FAISS GPU index build path (default: from config retrieval.faiss.use_gpu)'
    )
    parser.add_argument(
        '--gpu-id',
        type=int,
        default=None,
        help='GPU device ID for FAISS GPU index build (default: from config retrieval.faiss.gpu_id)'
    )
    parser.add_argument(
        '--no-progress',
        action='store_true',
        help='Disable tqdm progress bar during FAISS vector add phase'
    )
    
    args = parser.parse_args()
    
    logger = setup_logger(__name__)
    logger.info(f"Starting FAISS index build: strategy={args.strategy}, type={args.index_type}")

    config = Config(args.config)
    checkpoint_enabled = config.get('checkpointing.faiss.enabled', True)
    default_resume = config.get('checkpointing.faiss.resume_by_default', True)
    strict_compatibility = config.get('checkpointing.strict_compatibility', True)
    checkpoint_interval = args.checkpoint_interval or config.get('checkpointing.faiss.checkpoint_interval', 200000)
    add_batch_size = args.add_batch_size or config.get('checkpointing.faiss.add_batch_size', 50000)
    checkpoint_dir = Path(args.checkpoint_dir or config.get('checkpointing.faiss.checkpoint_dir', 'data/checkpoints/faiss/'))

    if args.no_resume:
        resume = False
    elif args.resume:
        resume = True
    else:
        resume = default_resume
    
    # Set up paths based on strategy
    embeddings_template = config.get('data.embeddings', 'data/embeddings/wiki_embeddings_{strategy}.npy')
    chunks_template = config.get('data.processed_chunks', 'data/processed/wiki_chunks_{strategy}.jsonl')
    faiss_template = config.get('data.faiss_index', 'data/indexes/{strategy}/faiss.index')

    embeddings_path = Path(embeddings_template.format(strategy=args.strategy))
    chunks_path = Path(chunks_template.format(strategy=args.strategy))
    output_dir = Path(faiss_template.format(strategy=args.strategy)).parent
    manifest_path, partial_index_path = build_checkpoint_paths(checkpoint_dir, args.strategy)

    if args.reset_checkpoint:
        if manifest_path.exists():
            manifest_path.unlink()
            logger.info(f"Removed checkpoint manifest: {manifest_path}")
        if partial_index_path.exists():
            partial_index_path.unlink()
            logger.info(f"Removed partial index: {partial_index_path}")
    
    # Verify input files exist
    if not embeddings_path.exists():
        logger.error(f"Embeddings file not found: {embeddings_path}")
        logger.error("Run generate_embeddings.py first!")
        sys.exit(1)
    
    if not chunks_path.exists():
        logger.error(f"Chunks file not found: {chunks_path}")
        logger.error("Run prepare_wikipedia_chunks.py first!")
        sys.exit(1)
    
    # Load data
    embeddings = load_embeddings(embeddings_path)
    metadata_count = count_chunk_metadata(chunks_path, no_progress=args.no_progress)
    
    # Verify data consistency
    if len(embeddings) != metadata_count:
        logger.error(
            f"Mismatch: {len(embeddings)} embeddings but {metadata_count} metadata entries"
        )
        sys.exit(1)
    
    logger.info(f"Data verified: {len(embeddings):,} vectors ready for indexing")
    prep_progress = tqdm(
        total=5,
        desc="FAISS prep",
        unit="stage",
        disable=args.no_progress,
    )
    prep_progress.set_postfix_str("data consistency verified")
    prep_progress.update(1)
    
    # Adjust parameters based on dataset size
    n_vectors = len(embeddings)
    adjusted_nlist = args.nlist
    adjusted_index_type = args.index_type
    
    if args.index_type == 'IVFFLAT':
        # IVFFLAT requires at least nlist training points
        if n_vectors < args.nlist:
            # For small datasets, use FLAT instead or adjust nlist
            if n_vectors < 100:
                logger.warning(
                    f"Dataset too small ({n_vectors} vectors) for IVFFLAT. "
                    f"Switching to FLAT index."
                )
                adjusted_index_type = 'FLAT'
            else:
                # Use smaller nlist (roughly sqrt(n_vectors))
                adjusted_nlist = int(np.sqrt(n_vectors))
                logger.warning(
                    f"Adjusting nlist from {args.nlist} to {adjusted_nlist} "
                    f"for dataset with {n_vectors} vectors"
                )
    
    if add_batch_size <= 0:
        raise ValueError("add_batch_size must be > 0")

    config_use_gpu = bool(config.get('retrieval.faiss.use_gpu', False))
    config_gpu_id = int(config.get('retrieval.faiss.gpu_id', 0))

    use_gpu = config_use_gpu if args.use_gpu is None else args.use_gpu
    gpu_id = config_gpu_id if args.gpu_id is None else args.gpu_id
    prep_progress.set_postfix_str("parameter/GPU resolution completed")
    prep_progress.update(1)

    # Create FAISS index manager
    embedding_dim = embeddings.shape[1]
    manager = FAISSIndexManager(
        dimension=embedding_dim,
        index_type=adjusted_index_type,
        use_gpu=use_gpu,
        gpu_id=gpu_id,
    )

    expected_manifest = {
        'schema_version': CHECKPOINT_SCHEMA_VERSION,
        'strategy': args.strategy,
        'index_type': adjusted_index_type,
        'dimension': int(embedding_dim),
        'nlist': int(adjusted_nlist),
        'nprobe': int(args.nprobe),
        'hnsw_m': int(args.hnsw_m),
        'input_embeddings': file_fingerprint(embeddings_path),
        'input_chunks': file_fingerprint(chunks_path),
        'n_vectors': int(n_vectors),
    }
    prep_progress.set_postfix_str("manifest/setup prepared")
    prep_progress.update(1)

    start_idx = 0
    index = None
    gpu_resources = None

    if checkpoint_enabled and resume and manifest_path.exists():
        manifest = load_manifest(manifest_path)
        if strict_compatibility:
            ensure_manifest_compatible(
                manifest,
                expected_manifest,
                required_keys=[
                    'schema_version', 'strategy', 'index_type', 'dimension',
                    'nlist', 'nprobe', 'hnsw_m', 'input_embeddings',
                    'input_chunks', 'n_vectors', 'added_count',
                ],
            )

        if not partial_index_path.exists():
            raise FileNotFoundError(
                f"Checkpoint manifest found but partial FAISS index missing: {partial_index_path}. "
                "Use --reset-checkpoint to rebuild."
            )

        index = faiss.read_index(str(partial_index_path))
        if use_gpu:
            index, gpu_resources = cpu_to_gpu_index(index, gpu_id, logger)
        start_idx = int(manifest.get('added_count', 0))

        if index.ntotal != start_idx:
            # On network-backed filesystems (e.g., Colab Google Drive mount),
            # the manifest can be persisted before the large FAISS index file.
            # Recover by trusting the actual index size when manifest is ahead.
            if index.ntotal < start_idx:
                logger.warning(
                    "Checkpoint mismatch detected: index.ntotal=%s, added_count=%s. "
                    "Recovering by resuming from index.ntotal to avoid data loss.",
                    f"{index.ntotal:,}",
                    f"{start_idx:,}",
                )
                start_idx = int(index.ntotal)
            else:
                raise ValueError(
                    "Checkpoint mismatch: index contains more vectors than manifest "
                    f"(index.ntotal={index.ntotal}, added_count={start_idx}). "
                    "Use --reset-checkpoint to rebuild."
                )

        logger.info(f"Resumed FAISS build from checkpoint at {start_idx:,}/{n_vectors:,} vectors")

    if index is None:
        logger.info("Initializing new FAISS index...")
        index = create_faiss_index(
            embeddings=embeddings,
            index_type=adjusted_index_type,
            dimension=embedding_dim,
            nlist=adjusted_nlist,
            nprobe=args.nprobe,
            hnsw_m=args.hnsw_m,
            logger=logger,
        )
        if use_gpu:
            index, gpu_resources = cpu_to_gpu_index(index, gpu_id, logger)

    prep_progress.set_postfix_str("index prepared")
    prep_progress.update(1)

    if adjusted_index_type == 'IVFFLAT' and hasattr(index, 'nprobe'):
        index.nprobe = args.nprobe

    logger.info("Adding vectors to FAISS index in batches...")
    batch_iterator = range(start_idx, n_vectors, add_batch_size)
    if not args.no_progress:
        total_batches = (max(n_vectors - start_idx, 0) + add_batch_size - 1) // add_batch_size
        batch_iterator = tqdm(
            batch_iterator,
            total=total_batches,
            desc="FAISS add",
            unit="batch",
        )

    prep_progress.set_postfix_str("add-loop configured")
    prep_progress.update(1)
    prep_progress.close()

    try:
        for i in batch_iterator:
            batch_end = min(i + add_batch_size, n_vectors)
            batch_vectors = np.asarray(embeddings[i:batch_end], dtype=np.float32)
            index.add(batch_vectors)

            if checkpoint_enabled and checkpoint_interval > 0 and (
                batch_end % checkpoint_interval == 0 or batch_end == n_vectors
            ):
                commit_faiss_checkpoint(
                    index=index,
                    partial_index_path=partial_index_path,
                    manifest_path=manifest_path,
                    expected_manifest=expected_manifest,
                    added_count=batch_end,
                    logger=logger,
                )
    except KeyboardInterrupt:
        logger.warning("FAISS build interrupted by user; attempting to persist latest checkpoint...")
        if checkpoint_enabled:
            committed_count = int(index.ntotal)
            commit_faiss_checkpoint(
                index=index,
                partial_index_path=partial_index_path,
                manifest_path=manifest_path,
                expected_manifest=expected_manifest,
                added_count=committed_count,
                logger=logger,
            )
            logger.warning(
                "Saved interrupt-safe FAISS checkpoint. Resume with --resume or reset with --reset-checkpoint."
            )
        raise

    test_query = np.asarray(embeddings[0:1], dtype=np.float32) if n_vectors > 0 else None
    del embeddings
    gc.collect()

    logger.info("Saving index and streaming metadata to disk...")
    try:
        sample_metadata, metadata_file = manager.save_index_from_jsonl(
            index=index,
            chunks_jsonl_path=chunks_path,
            save_dir=str(output_dir),
            no_progress=args.no_progress,
        )
    except Exception as e:
        logger.error(f"Failed to save index: {e}")
        raise

    if checkpoint_enabled:
        if manifest_path.exists():
            manifest_path.unlink()
            logger.info(f"Removed checkpoint manifest: {manifest_path}")
        if partial_index_path.exists():
            partial_index_path.unlink()
            logger.info(f"Removed partial index checkpoint: {partial_index_path}")
    
    # Test search with sample query (using sample metadata)
    logger.info("\n" + "="*60)
    logger.info("Testing index with sample query...")
    logger.info("="*60)

    if test_query is None:
        logger.warning("No vectors in index; skipping sample search test")
        distances, indices = [], []
    else:
        distances, indices = manager.search(index, test_query, top_k=5)
    
    if test_query is not None and sample_metadata:
        logger.info("\nTest query (first embedding in dataset):")
        logger.info("Top 5 results:")
        for i, (idx, score) in enumerate(zip(indices[0], distances[0]), 1):
            if idx < len(sample_metadata):
                chunk = sample_metadata[idx]
                logger.info(f"\n{i}. Score: {score:.4f}, Index: {idx}")
                logger.info(f"   Doc ID: {chunk.get('doc_id', 'N/A')}")
                logger.info(f"   Text: {chunk['text'][:100]}...")
            else:
                logger.info(f"\n{i}. Score: {score:.4f}, Index: {idx} (outside sample range, use full metadata.pkl for details)")
    
    # Print summary statistics (from sample_metadata)
    logger.info("\n" + "="*60)
    logger.info("Index Build Summary")
    logger.info("="*60)
    corpus_source = sample_metadata[0].get('source', 'unknown') if sample_metadata else 'unknown'
    corpus_version = sample_metadata[0].get('version', 'unknown') if sample_metadata else 'unknown'
    logger.info(f"Strategy: {args.strategy}")
    logger.info(f"Corpus Source: {corpus_source}")
    logger.info(f"Corpus Version: {corpus_version}")
    logger.info(f"Index Type: {adjusted_index_type}")
    logger.info(f"Embedding Dimension: {embedding_dim}")
    logger.info(f"Total Vectors: {index.ntotal:,}")
    logger.info(f"Output Directory: {output_dir}")
    logger.info(f"FAISS Build Device: {'GPU' if use_gpu and gpu_resources is not None else 'CPU'}")
    logger.info(f"FAISS GPU ID: {gpu_id}")
    logger.info(f"Metadata file: {metadata_file}")
    
    if adjusted_index_type == 'IVFFLAT':
        logger.info(f"nlist (clusters): {adjusted_nlist}")
        logger.info(f"nprobe (search): {args.nprobe}")
    elif adjusted_index_type == 'HNSW':
        logger.info(f"HNSW M: {args.hnsw_m}")
    
    logger.info("\n✓ FAISS index built successfully!")


if __name__ == '__main__':
    main()
