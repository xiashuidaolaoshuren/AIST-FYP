"""
Build and cache BM25 index for faster loading.

This script pre-builds BM25 indexes for all dataset strategies (dev, validation,
production) and caches them to disk. This allows the BM25Retriever to load
instantly instead of rebuilding the index each time.

Usage:
    python scripts/build_bm25_index.py --strategy development
    python scripts/build_bm25_index.py --strategy validation
    python scripts/build_bm25_index.py --strategy production
    python scripts/build_bm25_index.py --all  # Build all strategies
"""

import argparse
import sys
from pathlib import Path

# Add project root to path
project_root = Path(__file__).parent.parent
sys.path.insert(0, str(project_root))

from src.retrieval.bm25_retriever import BM25Retriever
from src.utils.config import Config
from src.utils.logger import setup_logger


def build_index_for_strategy(strategy: str, config_path: str = 'config.yaml'):
    """
    Build and cache BM25 index for a specific dataset strategy.
    
    Args:
        strategy: Dataset strategy (dev, validation, or production)
        config_path: Path to configuration file
    """
    logger = setup_logger('build_bm25_index')
    
    # Load configuration
    config = Config(config_path)
    
    logger.info(f"Building BM25 index for strategy: {strategy}")
    
    # Get paths
    chunks_template = config.get('data.processed_chunks', 'data/processed/wiki_chunks_{strategy}.jsonl')
    bm25_template = config.get('data.bm25_index', 'data/indexes/{strategy}/bm25_index.pkl')
    corpus_path = Path(chunks_template.format(strategy=strategy))
    index_path = Path(bm25_template.format(strategy=strategy))
    
    if not corpus_path.exists():
        logger.error(f"Corpus file not found: {corpus_path}")
        logger.error("Please run preprocessing first:")
        logger.error(f"  python scripts/prepare_wikipedia_chunks.py --strategy {strategy}")
        return False
    
    # Get BM25 parameters from config (with defaults)
    k1 = config.retrieval.get('bm25', {}).get('k1', 1.5)
    b = config.retrieval.get('bm25', {}).get('b', 0.75)
    
    logger.info(f"BM25 parameters: k1={k1}, b={b}")
    logger.info(f"Corpus path: {corpus_path}")
    logger.info(f"Index will be saved to: {index_path}")
    
    # Build index (this will automatically cache it)
    try:
        retriever = BM25Retriever(
            corpus_path=str(corpus_path),
            index_path=str(index_path),
            k1=k1,
            b=b
        )
        
        logger.info(f"Successfully built and cached BM25 index for {strategy}")
        logger.info(f"Index contains {len(retriever.chunks)} chunks")
        return True
        
    except Exception as e:
        logger.error(f"Failed to build BM25 index: {e}")
        import traceback
        traceback.print_exc()
        return False


def main():
    parser = argparse.ArgumentParser(
        description='Build and cache BM25 indexes for retrieval'
    )
    parser.add_argument(
        '--strategy',
        type=str,
        choices=['development', 'validation', 'production'],
        help='Dataset strategy to build index for'
    )
    parser.add_argument(
        '--all',
        action='store_true',
        help='Build indexes for all strategies'
    )
    parser.add_argument(
        '--config',
        type=str,
        default='config.yaml',
        help='Path to configuration file (default: config.yaml)'
    )
    
    args = parser.parse_args()
    
    # Validate arguments
    if not args.strategy and not args.all:
        parser.error("Either --strategy or --all must be specified")
    
    if args.strategy and args.all:
        parser.error("Cannot specify both --strategy and --all")
    
    # Build indexes
    if args.all:
        strategies = ['development', 'validation', 'production']
        success_count = 0
        
        for strategy in strategies:
            success = build_index_for_strategy(strategy, args.config)
            if success:
                success_count += 1
            print()  # Blank line between strategies
        
        print(f"\nCompleted: {success_count}/{len(strategies)} indexes built successfully")
        
        if success_count < len(strategies):
            sys.exit(1)
    else:
        success = build_index_for_strategy(args.strategy, args.config)
        if not success:
            sys.exit(1)


if __name__ == '__main__':
    main()
