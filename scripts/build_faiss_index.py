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
import json
import sys
from pathlib import Path
import numpy as np

# Add project root to path
sys.path.insert(0, str(Path(__file__).parent.parent))

from src.retrieval import FAISSIndexManager
from src.utils.config import Config
from src.utils.logger import setup_logger


def load_embeddings(embeddings_path: Path) -> np.ndarray:
    """Load embeddings from .npy file."""
    logger = setup_logger(__name__)
    logger.info(f"Loading embeddings from {embeddings_path}")
    
    embeddings = np.load(embeddings_path)
    logger.info(f"Loaded embeddings: shape={embeddings.shape}, dtype={embeddings.dtype}")
    
    return embeddings


def load_chunk_metadata(chunks_path: Path) -> list:
    """Load chunk metadata from .jsonl file."""
    logger = setup_logger(__name__)
    logger.info(f"Loading chunk metadata from {chunks_path}")
    
    metadata = []
    with open(chunks_path, 'r', encoding='utf-8') as f:
        for line in f:
            metadata.append(json.loads(line))
    
    logger.info(f"Loaded {len(metadata):,} chunk metadata entries")
    
    return metadata


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
    
    args = parser.parse_args()
    
    logger = setup_logger(__name__)
    logger.info(f"Starting FAISS index build: strategy={args.strategy}, type={args.index_type}")

    config = Config(args.config)
    
    # Set up paths based on strategy
    embeddings_template = config.get('data.embeddings', 'data/embeddings/wiki_embeddings_{strategy}.npy')
    chunks_template = config.get('data.processed_chunks', 'data/processed/wiki_chunks_{strategy}.jsonl')
    faiss_template = config.get('data.faiss_index', 'data/indexes/{strategy}/faiss.index')

    embeddings_path = Path(embeddings_template.format(strategy=args.strategy))
    chunks_path = Path(chunks_template.format(strategy=args.strategy))
    output_dir = Path(faiss_template.format(strategy=args.strategy)).parent
    
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
    metadata = load_chunk_metadata(chunks_path)
    
    # Verify data consistency
    if len(embeddings) != len(metadata):
        logger.error(
            f"Mismatch: {len(embeddings)} embeddings but {len(metadata)} metadata entries"
        )
        sys.exit(1)
    
    logger.info(f"Data verified: {len(embeddings):,} vectors ready for indexing")
    
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
    
    # Create FAISS index manager
    embedding_dim = embeddings.shape[1]
    manager = FAISSIndexManager(dimension=embedding_dim, index_type=adjusted_index_type)
    
    # Build index
    logger.info("Building FAISS index...")
    index = manager.build_index(
        embeddings,
        nlist=adjusted_nlist,
        nprobe=args.nprobe,
        hnsw_m=args.hnsw_m
    )
    
    # Save index and metadata
    logger.info(f"Saving index to {output_dir}")
    manager.save_index(index, metadata, str(output_dir))
    
    # Test search with sample query
    logger.info("\n" + "="*60)
    logger.info("Testing index with sample query...")
    logger.info("="*60)
    
    # Use first embedding as test query
    test_query = embeddings[0:1]  # Shape (1, dimension)
    distances, indices = manager.search(index, test_query, top_k=5)
    
    logger.info(f"\nTest query (first embedding in dataset):")
    logger.info(f"Top 5 results:")
    for i, (idx, score) in enumerate(zip(indices[0], distances[0]), 1):
        chunk = metadata[idx]
        logger.info(f"\n{i}. Score: {score:.4f}, Index: {idx}")
        logger.info(f"   Doc ID: {chunk.get('doc_id', 'N/A')}")
        logger.info(f"   Text: {chunk['text'][:100]}...")
    
    # Print summary statistics
    logger.info("\n" + "="*60)
    logger.info("Index Build Summary")
    logger.info("="*60)
    corpus_source = metadata[0].get('source', 'unknown') if metadata else 'unknown'
    corpus_version = metadata[0].get('version', 'unknown') if metadata else 'unknown'
    logger.info(f"Strategy: {args.strategy}")
    logger.info(f"Corpus Source: {corpus_source}")
    logger.info(f"Corpus Version: {corpus_version}")
    logger.info(f"Index Type: {adjusted_index_type}")
    logger.info(f"Embedding Dimension: {embedding_dim}")
    logger.info(f"Total Vectors: {index.ntotal:,}")
    logger.info(f"Output Directory: {output_dir}")
    
    if adjusted_index_type == 'IVFFLAT':
        logger.info(f"nlist (clusters): {adjusted_nlist}")
        logger.info(f"nprobe (search): {args.nprobe}")
    elif adjusted_index_type == 'HNSW':
        logger.info(f"HNSW M: {args.hnsw_m}")
    
    logger.info("\n✓ FAISS index built successfully!")


if __name__ == '__main__':
    main()
