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
import json
import pickle
import sys
from pathlib import Path
from typing import List
from rank_bm25 import BM25Okapi
from tqdm import tqdm

# Add project root to path
project_root = Path(__file__).parent.parent
sys.path.insert(0, str(project_root))

from src.utils.config import Config
from src.utils.logger import setup_logger
from src.utils.nlp_utils import get_spacy_model
from src.utils.checkpoint_utils import (
    CHECKPOINT_SCHEMA_VERSION,
    ensure_manifest_compatible,
    file_fingerprint,
    load_manifest,
    save_manifest,
)


def _tokenize(nlp, text: str) -> List[str]:
    doc = nlp(text)
    return [token.text.lower() for token in doc if not token.is_space]


def build_index_for_strategy(
    strategy: str,
    config_path: str = 'config.yaml',
    resume: bool = True,
    reset_checkpoint: bool = False,
    checkpoint_dir_override: str = None,
    checkpoint_interval_override: int = None,
):
    """
    Build and cache BM25 index for a specific dataset strategy.
    
    Args:
        strategy: Dataset strategy (dev, validation, or production)
        config_path: Path to configuration file
    """
    logger = setup_logger('build_bm25_index')
    
    # Load configuration
    config = Config(config_path)
    checkpoint_enabled = config.get('checkpointing.bm25.enabled', True)
    strict_compatibility = config.get('checkpointing.strict_compatibility', True)
    checkpoint_interval = checkpoint_interval_override or config.get('checkpointing.bm25.checkpoint_interval', 5000)
    checkpoint_dir = Path(
        checkpoint_dir_override or config.get('checkpointing.bm25.checkpoint_dir', 'data/checkpoints/bm25/')
    )
    
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

    spacy_model = config.get('verification.spacy_model', 'en_core_web_sm')
    nlp = get_spacy_model(spacy_model)

    checkpoint_dir.mkdir(parents=True, exist_ok=True)
    checkpoint_manifest = checkpoint_dir / f'bm25_build_{strategy}.json'
    checkpoint_tokens = checkpoint_dir / f'bm25_tokens_{strategy}.pkl'

    if reset_checkpoint:
        if checkpoint_manifest.exists():
            checkpoint_manifest.unlink()
            logger.info(f"Removed BM25 checkpoint manifest: {checkpoint_manifest}")
        if checkpoint_tokens.exists():
            checkpoint_tokens.unlink()
            logger.info(f"Removed BM25 token checkpoint: {checkpoint_tokens}")

    expected_manifest = {
        'schema_version': CHECKPOINT_SCHEMA_VERSION,
        'strategy': strategy,
        'corpus_fingerprint': file_fingerprint(corpus_path),
        'k1': float(k1),
        'b': float(b),
        'spacy_model': spacy_model,
    }

    # Build index with tokenization-progress checkpointing
    try:
        chunks = []
        with open(corpus_path, 'r', encoding='utf-8') as f:
            for line in f:
                chunks.append(json.loads(line))

        logger.info(f"Loaded {len(chunks):,} chunks from corpus")

        processed_count = 0
        tokenized_corpus = []

        if checkpoint_enabled and resume and checkpoint_manifest.exists() != checkpoint_tokens.exists():
            raise ValueError(
                "Incomplete BM25 checkpoint state detected (manifest/token file mismatch). "
                "Use --reset-checkpoint to rebuild."
            )

        if checkpoint_enabled and resume and checkpoint_manifest.exists() and checkpoint_tokens.exists():
            manifest = load_manifest(checkpoint_manifest)
            if strict_compatibility:
                ensure_manifest_compatible(
                    manifest,
                    expected_manifest,
                    required_keys=[
                        'schema_version',
                        'strategy',
                        'corpus_fingerprint',
                        'k1',
                        'b',
                        'spacy_model',
                        'processed_count',
                    ],
                )

            with open(checkpoint_tokens, 'rb') as f:
                checkpoint_data = pickle.load(f)

            tokenized_corpus = checkpoint_data.get('tokenized_corpus', [])
            processed_count = int(manifest.get('processed_count', len(tokenized_corpus)))

            if processed_count != len(tokenized_corpus):
                raise ValueError(
                    f"BM25 checkpoint mismatch: processed_count={processed_count}, "
                    f"tokenized_entries={len(tokenized_corpus)}"
                )

            logger.info(f"Resuming BM25 tokenization from {processed_count:,}/{len(chunks):,}")

        for idx in tqdm(range(processed_count, len(chunks)), desc='Tokenizing', unit='chunk'):
            tokenized_corpus.append(_tokenize(nlp, chunks[idx]['text']))
            processed_count = idx + 1

            if checkpoint_enabled and checkpoint_interval > 0 and (
                processed_count % checkpoint_interval == 0 or processed_count == len(chunks)
            ):
                save_manifest(
                    checkpoint_manifest,
                    {
                        **expected_manifest,
                        'processed_count': processed_count,
                    }
                )

                with open(checkpoint_tokens, 'wb') as f:
                    pickle.dump({'tokenized_corpus': tokenized_corpus}, f)

                logger.info(f"Checkpoint saved at {processed_count:,}/{len(chunks):,} chunks")

        logger.info("Building BM25 index from tokenized corpus...")
        bm25 = BM25Okapi(tokenized_corpus, k1=k1, b=b)

        index_path.parent.mkdir(parents=True, exist_ok=True)
        with open(index_path, 'wb') as f:
            pickle.dump(
                {
                    'bm25': bm25,
                    'chunks': chunks,
                    'k1': k1,
                    'b': b,
                    'corpus_fingerprint': expected_manifest['corpus_fingerprint'],
                    'spacy_model': spacy_model,
                    'schema_version': CHECKPOINT_SCHEMA_VERSION,
                },
                f,
            )

        if checkpoint_enabled:
            if checkpoint_manifest.exists():
                checkpoint_manifest.unlink()
            if checkpoint_tokens.exists():
                checkpoint_tokens.unlink()

        logger.info(f"Successfully built and cached BM25 index for {strategy}")
        logger.info(f"Index contains {len(chunks)} chunks")
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
    parser.add_argument(
        '--resume',
        action='store_true',
        help='Resume BM25 tokenization from checkpoint if available'
    )
    parser.add_argument(
        '--no-resume',
        action='store_true',
        help='Disable checkpoint loading and build from scratch'
    )
    parser.add_argument(
        '--reset-checkpoint',
        action='store_true',
        help='Delete existing BM25 checkpoints before building'
    )
    parser.add_argument(
        '--checkpoint-dir',
        type=str,
        default=None,
        help='Directory for BM25 checkpoints (default: from config)'
    )
    parser.add_argument(
        '--checkpoint-interval',
        type=int,
        default=None,
        help='Save checkpoint every N tokenized chunks (default: from config)'
    )
    
    args = parser.parse_args()
    
    # Validate arguments
    if not args.strategy and not args.all:
        parser.error("Either --strategy or --all must be specified")
    
    if args.strategy and args.all:
        parser.error("Cannot specify both --strategy and --all")
    
    # Build indexes
    config = Config(args.config)
    default_resume = config.get('checkpointing.bm25.resume_by_default', True)
    if args.no_resume:
        resume = False
    elif args.resume:
        resume = True
    else:
        resume = default_resume

    if args.all:
        strategies = ['development', 'validation', 'production']
        success_count = 0
        
        for strategy in strategies:
            success = build_index_for_strategy(
                strategy,
                args.config,
                resume=resume,
                reset_checkpoint=args.reset_checkpoint,
                checkpoint_dir_override=args.checkpoint_dir,
                checkpoint_interval_override=args.checkpoint_interval,
            )
            if success:
                success_count += 1
            print()  # Blank line between strategies
        
        print(f"\nCompleted: {success_count}/{len(strategies)} indexes built successfully")
        
        if success_count < len(strategies):
            sys.exit(1)
    else:
        success = build_index_for_strategy(
            args.strategy,
            args.config,
            resume=resume,
            reset_checkpoint=args.reset_checkpoint,
            checkpoint_dir_override=args.checkpoint_dir,
            checkpoint_interval_override=args.checkpoint_interval,
        )
        if not success:
            sys.exit(1)


if __name__ == '__main__':
    main()
