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
    write_pickle_atomic,
)


def _tokens_from_doc(doc) -> List[str]:
    return [token.text.lower() for token in doc if not token.is_space]


def _count_jsonl_rows(path: Path, no_progress: bool = False) -> int:
    total_size_bytes = path.stat().st_size
    count = 0
    with open(path, 'r', encoding='utf-8') as f:
        progress_bar = tqdm(
            total=total_size_bytes,
            unit='B',
            unit_scale=True,
            desc='Counting corpus rows',
            disable=no_progress,
        )
        for line in f:
            progress_bar.update(len(line.encode('utf-8')))
            if line.strip():
                count += 1
        progress_bar.close()
    return count


def build_index_for_strategy(
    strategy: str,
    config_path: str = 'config.yaml',
    resume: bool = True,
    reset_checkpoint: bool = False,
    checkpoint_dir_override: str = None,
    checkpoint_interval_override: int = None,
    tokenize_batch_size_override: int = None,
    spacy_pipe_batch_size_override: int = None,
    no_progress: bool = False,
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
    tokenize_batch_size = tokenize_batch_size_override or config.get('retrieval.bm25.tokenize_batch_size', 2048)
    spacy_pipe_batch_size = spacy_pipe_batch_size_override or config.get('retrieval.bm25.spacy_pipe_batch_size', 256)
    checkpoint_dir = Path(
        checkpoint_dir_override or config.get('checkpointing.bm25.checkpoint_dir', 'data/checkpoints/bm25/')
    )

    if tokenize_batch_size <= 0:
        raise ValueError('tokenize_batch_size must be > 0')
    if spacy_pipe_batch_size <= 0:
        raise ValueError('spacy_pipe_batch_size must be > 0')
    
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

    processed_count = 0
    tokenized_corpus = []

    # Build index with tokenization-progress checkpointing
    try:
        total_chunks = _count_jsonl_rows(corpus_path, no_progress=no_progress)
        logger.info(f"Corpus size: {total_chunks:,} chunks")
        logger.info(
            f"Tokenization batches: text_batch={tokenize_batch_size}, "
            f"spacy_pipe_batch={spacy_pipe_batch_size}"
        )

        chunks = []

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

            logger.info(f"Resuming BM25 tokenization from {processed_count:,}/{total_chunks:,}")

        if processed_count > total_chunks:
            raise ValueError(
                f"Invalid BM25 checkpoint: processed_count={processed_count} exceeds corpus rows={total_chunks}"
            )

        pending_texts = []

        def flush_pending_texts(progress_bar):
            nonlocal processed_count
            if not pending_texts:
                return

            for doc in nlp.pipe(pending_texts, batch_size=spacy_pipe_batch_size):
                tokenized_corpus.append(_tokens_from_doc(doc))
                processed_count += 1
                progress_bar.update(1)

                if checkpoint_enabled and checkpoint_interval > 0 and (
                    processed_count % checkpoint_interval == 0 or processed_count == total_chunks
                ):
                    save_manifest(
                        checkpoint_manifest,
                        {
                            **expected_manifest,
                            'processed_count': processed_count,
                        }
                    )
                    write_pickle_atomic(
                        checkpoint_tokens,
                        {'tokenized_corpus': tokenized_corpus},
                    )

                    logger.info(f"Checkpoint saved at {processed_count:,}/{total_chunks:,} chunks")

            pending_texts.clear()

        remaining_chunks = total_chunks - processed_count
        with tqdm(
            total=remaining_chunks,
            desc='Tokenizing',
            unit='chunk',
            disable=no_progress,
        ) as progress_bar, tqdm(
            total=total_chunks,
            desc='Loading corpus',
            unit='chunk',
            disable=no_progress,
        ) as loading_bar:
            with open(corpus_path, 'r', encoding='utf-8') as f:
                chunk_idx = 0
                for line in f:
                    if not line.strip():
                        continue

                    chunk = json.loads(line)
                    chunks.append(chunk)
                    loading_bar.update(1)

                    if chunk_idx < processed_count:
                        chunk_idx += 1
                        continue

                    pending_texts.append(chunk['text'])
                    if len(pending_texts) >= tokenize_batch_size:
                        flush_pending_texts(progress_bar)

                    chunk_idx += 1

            flush_pending_texts(progress_bar)

        if len(tokenized_corpus) != len(chunks):
            raise ValueError(
                f"Tokenized corpus length ({len(tokenized_corpus)}) does not match chunk count ({len(chunks)})"
            )

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
    except KeyboardInterrupt:
        logger.warning("BM25 build interrupted by user; attempting to persist latest checkpoint...")
        if checkpoint_enabled:
            write_pickle_atomic(
                checkpoint_tokens,
                {'tokenized_corpus': tokenized_corpus},
            )
            save_manifest(
                checkpoint_manifest,
                {
                    **expected_manifest,
                    'processed_count': int(processed_count),
                },
            )
            logger.warning(
                "Saved interrupt-safe BM25 checkpoint. Resume with --resume or reset with --reset-checkpoint."
            )
        raise
        
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
    parser.add_argument(
        '--tokenize-batch-size',
        type=int,
        default=None,
        help='Number of texts buffered before calling spaCy pipe (default: from config retrieval.bm25.tokenize_batch_size)'
    )
    parser.add_argument(
        '--spacy-pipe-batch-size',
        type=int,
        default=None,
        help='spaCy nlp.pipe batch size (default: from config retrieval.bm25.spacy_pipe_batch_size)'
    )
    parser.add_argument(
        '--no-progress',
        action='store_true',
        help='Disable tqdm progress bar during BM25 tokenization'
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
                tokenize_batch_size_override=args.tokenize_batch_size,
                spacy_pipe_batch_size_override=args.spacy_pipe_batch_size,
                no_progress=args.no_progress,
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
            tokenize_batch_size_override=args.tokenize_batch_size,
            spacy_pipe_batch_size_override=args.spacy_pipe_batch_size,
            no_progress=args.no_progress,
        )
        if not success:
            sys.exit(1)


if __name__ == '__main__':
    main()
