"""
Embedding generation script for Wikipedia chunks.

This script generates dense embeddings for text chunks using sentence-transformers
with GPU acceleration, FP16 precision, and checkpointing support.

Usage:
    python scripts/generate_embeddings.py --strategy development
    python scripts/generate_embeddings.py --strategy validation --model custom-model
"""

import argparse
import json
import sys
import time
from pathlib import Path
import numpy as np

# Add project root to path
project_root = Path(__file__).parent.parent
sys.path.insert(0, str(project_root))

from src.data_processing import EmbeddingGenerator
from src.utils import Config, setup_logger


def load_chunks(chunks_file: Path) -> list:
    """
    Load chunks from JSONL file.
    
    Args:
        chunks_file: Path to JSONL file with chunks
    
    Returns:
        List of chunk dictionaries
    """
    chunks = []
    with open(chunks_file, 'r', encoding='utf-8') as f:
        for line in f:
            chunks.append(json.loads(line.strip()))
    return chunks


def main():
    """Main embedding generation function."""
    parser = argparse.ArgumentParser(
        description='Generate embeddings for Wikipedia chunks',
        formatter_class=argparse.RawDescriptionHelpFormatter,
        epilog="""
Examples:
  # Development mode (10k articles)
  python scripts/generate_embeddings.py --strategy development

  # Validation mode with custom model
  python scripts/generate_embeddings.py --strategy validation --model custom-model

  # Production mode without FP16
  python scripts/generate_embeddings.py --strategy production --no-fp16
        """
    )
    
    parser.add_argument(
        '--strategy',
        type=str,
        required=True,
        choices=['development', 'validation', 'production'],
        help='Data processing strategy'
    )
    
    parser.add_argument(
        '--model',
        type=str,
        default=None,
        help='Model name or path (default: from config.yaml)'
    )
    
    parser.add_argument(
        '--batch-size',
        type=int,
        default=None,
        help='Batch size for processing (default: from config.yaml)'
    )
    
    parser.add_argument(
        '--device',
        type=str,
        default=None,
        choices=['cuda', 'cpu'],
        help='Device to use (default: from config.yaml)'
    )
    
    parser.add_argument(
        '--no-fp16',
        action='store_true',
        help='Disable FP16 precision (use full FP32)'
    )
    
    parser.add_argument(
        '--config',
        type=str,
        default='config.yaml',
        help='Path to configuration file (default: config.yaml)'
    )
    
    parser.add_argument(
        '--checkpoint-interval',
        type=int,
        default=10000,
        help='Save checkpoint every N chunks (default: 10000)'
    )
    
    args = parser.parse_args()
    
    # Setup logging
    logger = setup_logger(__name__, log_file='logs/month2.log')
    logger.info(f"Starting embedding generation with strategy: {args.strategy}")
    
    # Load configuration
    try:
        config = Config(args.config)
        logger.info(f"Loaded configuration from {args.config}")
    except Exception as e:
        logger.error(f"Failed to load configuration: {e}")
        sys.exit(1)
    
    # Determine parameters
    model_name = args.model or config.models.sentence_transformer
    batch_size = args.batch_size or config.processing.batch_size
    device = args.device or config.processing.device
    use_fp16 = config.processing.use_fp16 and not args.no_fp16
    
    logger.info(f"Model: {model_name}")
    logger.info(f"Batch size: {batch_size}, Device: {device}, FP16: {use_fp16}")
    
    # Determine input/output paths
    processed_dir = Path(config.get('paths.processed', 'data/processed'))
    embeddings_dir = Path(config.get('paths.embeddings', 'data/embeddings'))
    embeddings_dir.mkdir(parents=True, exist_ok=True)
    
    chunks_file = processed_dir / f"wiki_chunks_{args.strategy}.jsonl"
    embeddings_file = embeddings_dir / f"wiki_embeddings_{args.strategy}.npy"
    metadata_file = embeddings_dir / f"metadata_{args.strategy}.json"
    checkpoint_file = embeddings_dir / f"checkpoint_{args.strategy}.pkl"
    
    if not chunks_file.exists():
        logger.error(
            f"Chunks file not found: {chunks_file}\n"
            f"Please run prepare_wikipedia_chunks.py first with --strategy {args.strategy}"
        )
        sys.exit(1)
    
    logger.info(f"Input chunks: {chunks_file}")
    logger.info(f"Output embeddings: {embeddings_file}")
    logger.info(f"Output metadata: {metadata_file}")
    
    # Load chunks
    logger.info("Loading chunks...")
    start_load = time.time()
    chunks = load_chunks(chunks_file)
    load_time = time.time() - start_load
    logger.info(f"Loaded {len(chunks)} chunks in {load_time:.2f}s")
    
    if len(chunks) == 0:
        logger.error("No chunks loaded. Exiting.")
        sys.exit(1)
    
    # Initialize embedding generator
    try:
        logger.info("Initializing embedding generator...")
        generator = EmbeddingGenerator(
            model_name=model_name,
            batch_size=batch_size,
            device=device,
            use_fp16=use_fp16
        )
        embedding_dim = generator.get_embedding_dimension()
        logger.info(f"Embedding dimension: {embedding_dim}")
    except Exception as e:
        logger.error(f"Failed to initialize embedding generator: {e}")
        raise
    
    # Generate embeddings
    try:
        logger.info("Generating embeddings...")
        start_gen = time.time()
        
        embeddings = generator.generate_embeddings(
            chunks,
            checkpoint_path=str(checkpoint_file),
            checkpoint_interval=args.checkpoint_interval
        )
        
        gen_time = time.time() - start_gen
        chunks_per_sec = len(chunks) / gen_time if gen_time > 0 else 0
        
        logger.info(f"Generation complete in {gen_time:.2f}s ({chunks_per_sec:.2f} chunks/sec)")
        logger.info(f"Embeddings shape: {embeddings.shape}")
        
    except Exception as e:
        logger.error(f"Embedding generation failed: {e}")
        raise
    
    # Save embeddings
    try:
        logger.info(f"Saving embeddings to {embeddings_file}...")
        np.save(embeddings_file, embeddings)
        
        # Save metadata
        metadata = {
            'model_name': model_name,
            'embedding_dimension': embedding_dim,
            'num_chunks': len(chunks),
            'strategy': args.strategy,
            'batch_size': batch_size,
            'device': device,
            'use_fp16': use_fp16,
            'generation_time_seconds': gen_time,
            'chunks_per_second': chunks_per_sec
        }
        
        with open(metadata_file, 'w', encoding='utf-8') as f:
            json.dump(metadata, f, indent=2)
        
        logger.info(f"Metadata saved to {metadata_file}")
        
    except Exception as e:
        logger.error(f"Failed to save embeddings: {e}")
        raise
    
    # Print summary
    embeddings_size_mb = embeddings.nbytes / (1024 * 1024)
    
    summary = f"""
{'='*60}
Embedding Generation Complete!
{'='*60}
Strategy:              {args.strategy}
Model:                 {model_name}
Input Chunks:          {chunks_file}
Output Embeddings:     {embeddings_file}
Total Chunks:          {len(chunks):,}
Embedding Dimension:   {embedding_dim}
Embeddings Shape:      {embeddings.shape}
Embeddings Size:       {embeddings_size_mb:.2f} MB
Generation Time:       {gen_time:.2f}s
Processing Speed:      {chunks_per_sec:.2f} chunks/sec
Device:                {device}
FP16 Enabled:          {use_fp16}
Batch Size:            {batch_size}
{'='*60}
    """
    
    print(summary)
    logger.info(summary)
    logger.info("Embedding generation completed successfully")


if __name__ == '__main__':
    main()
