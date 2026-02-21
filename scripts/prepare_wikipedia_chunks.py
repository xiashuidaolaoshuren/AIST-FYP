"""
Wikipedia chunk preparation script.

This script processes Wikipedia XML dumps and creates sentence-level chunks
for the RAG retrieval system. Supports different data strategies for
development, validation, and production.

Usage:
    python scripts/prepare_wikipedia_chunks.py --strategy development
    python scripts/prepare_wikipedia_chunks.py --strategy validation
    python scripts/prepare_wikipedia_chunks.py --strategy production --dump path/to/dump.xml
"""

import argparse
import json
import sys
from pathlib import Path
from tqdm import tqdm

# Add project root to path
project_root = Path(__file__).parent.parent
sys.path.insert(0, str(project_root))

from src.data_processing import WikipediaParser, TextChunker
from src.utils import Config, setup_logger
from src.utils.checkpoint_utils import (
    CHECKPOINT_SCHEMA_VERSION,
    ensure_manifest_compatible,
    file_fingerprint,
    load_manifest,
    save_manifest,
    truncate_to_last_complete_jsonl_line,
)


def _build_checkpoint_path(checkpoint_dir: Path, strategy: str) -> Path:
    return checkpoint_dir / f"prepare_chunks_{strategy}.json"


def _save_chunking_checkpoint(
    checkpoint_path: Path,
    strategy: str,
    dump_path: Path,
    output_file: Path,
    is_jsonl: bool,
    max_articles,
    total_articles: int,
    total_chunks: int,
    input_offset: int,
) -> None:
    save_manifest(
        checkpoint_path,
        {
            'schema_version': CHECKPOINT_SCHEMA_VERSION,
            'strategy': strategy,
            'input_fingerprint': file_fingerprint(dump_path),
            'output_file': str(output_file),
            'is_jsonl': is_jsonl,
            'max_articles': max_articles,
            'total_articles': total_articles,
            'total_chunks': total_chunks,
            'input_offset': input_offset,
        }
    )


def main():
    """Main processing function."""
    parser = argparse.ArgumentParser(
        description='Prepare Wikipedia chunks for RAG retrieval',
        formatter_class=argparse.RawDescriptionHelpFormatter,
        epilog="""
Examples:
  # Development mode (10k articles)
  python scripts/prepare_wikipedia_chunks.py --strategy development

  # Validation mode (100k articles)
  python scripts/prepare_wikipedia_chunks.py --strategy validation

  # Production mode (all articles)
  python scripts/prepare_wikipedia_chunks.py --strategy production --dump enwiki-latest.xml
        """
    )
    
    parser.add_argument(
        '--strategy',
        type=str,
        required=True,
        choices=['development', 'validation', 'production'],
        help='Data processing strategy (determines article limit)'
    )
    
    parser.add_argument(
        '--dump',
        type=str,
        default=None,
        help='Path to Wikipedia XML dump file (default: from config)'
    )
    
    parser.add_argument(
        '--config',
        type=str,
        default='config.yaml',
        help='Path to configuration file (default: config.yaml)'
    )
    
    parser.add_argument(
        '--output-dir',
        type=str,
        default=None,
        help='Output directory (default: data/processed from config)'
    )
    parser.add_argument(
        '--resume',
        action='store_true',
        help='Resume from existing checkpoint if available'
    )
    parser.add_argument(
        '--no-resume',
        action='store_true',
        help='Disable checkpoint loading and start from scratch'
    )
    parser.add_argument(
        '--reset-checkpoint',
        action='store_true',
        help='Delete existing checkpoint before processing'
    )
    parser.add_argument(
        '--checkpoint-dir',
        type=str,
        default=None,
        help='Directory for chunking checkpoints (default: from config)'
    )
    parser.add_argument(
        '--checkpoint-interval',
        type=int,
        default=None,
        help='Save checkpoint every N processed articles (default: from config)'
    )
    
    args = parser.parse_args()
    
    # Setup logging
    logger = setup_logger(__name__, log_file='logs/month2.log')
    logger.info(f"Starting Wikipedia chunk preparation with strategy: {args.strategy}")
    
    # Load configuration
    try:
        config = Config(args.config)
        logger.info(f"Loaded configuration from {args.config}")
    except Exception as e:
        logger.error(f"Failed to load configuration: {e}")
        sys.exit(1)
    
    # Determine max_articles based on strategy
    strategy_config = config.data_strategy[args.strategy]
    max_articles = strategy_config.get('max_articles')
    
    logger.info(
        f"Strategy '{args.strategy}': "
        f"max_articles = {max_articles if max_articles else 'unlimited'}"
    )
    
    # Determine Wikipedia dump path based on strategy
    if args.dump:
        dump_path = Path(args.dump)
    else:
        # For development/validation, use JSONL files from download_wikipedia.py
        # For production, use XML dump
        if args.strategy == 'development':
            dump_path = Path(config.get('data.wikipedia_sample_dev', 'data/raw/wiki_sample_development.jsonl'))
        elif args.strategy == 'validation':
            dump_path = Path(config.get('data.wikipedia_sample_val', 'data/raw/wiki_sample_validation.jsonl'))
        else:  # production
            dump_path = Path(config.get('data.wikipedia_dump', 'data/raw/enwiki-latest-pages-articles.xml.bz2'))
    
    if not dump_path.exists():
        logger.error(
            f"Wikipedia data not found: {dump_path}\n"
            f"Please run the download script first:\n"
            f"  python scripts/download_wikipedia.py --strategy {args.strategy}\n"
        )
        sys.exit(1)
    
    # Detect file type
    is_jsonl = dump_path.suffix == '.jsonl'
    logger.info(f"Using Wikipedia data: {dump_path} (format: {'JSONL' if is_jsonl else 'XML'})")
    
    # Determine output path
    if args.output_dir:
        output_dir = Path(args.output_dir)
        output_file = output_dir / f"wiki_chunks_{args.strategy}.jsonl"
    else:
        output_template = config.get('data.processed_chunks', 'data/processed/wiki_chunks_{strategy}.jsonl')
        output_file = Path(output_template.format(strategy=args.strategy))
        output_dir = output_file.parent
    
    output_dir.mkdir(parents=True, exist_ok=True)
    
    logger.info(f"Output will be saved to: {output_file}")

    checkpoint_enabled = config.get('checkpointing.chunking.enabled', True)
    default_resume = config.get('checkpointing.chunking.resume_by_default', True)
    strict_compatibility = config.get('checkpointing.strict_compatibility', True)
    checkpoint_interval = args.checkpoint_interval or config.get('checkpointing.chunking.checkpoint_interval', 1000)
    checkpoint_dir = Path(
        args.checkpoint_dir or config.get('checkpointing.chunking.checkpoint_dir', 'data/checkpoints/chunking/')
    )
    checkpoint_path = _build_checkpoint_path(checkpoint_dir, args.strategy)

    if args.no_resume:
        resume = False
    elif args.resume:
        resume = True
    else:
        resume = default_resume

    if args.reset_checkpoint and checkpoint_path.exists():
        checkpoint_path.unlink()
        logger.info(f"Removed existing checkpoint: {checkpoint_path}")
    
    # Initialize components
    try:
        text_chunker = TextChunker()
        logger.info("Initialized TextChunker")
        
        # For XML files, use WikipediaParser
        if not is_jsonl:
            wiki_parser = WikipediaParser(str(dump_path), max_articles=max_articles)
            logger.info("Initialized WikipediaParser for XML input")
    except Exception as e:
        logger.error(f"Failed to initialize components: {e}")
        sys.exit(1)
    
    # Process articles
    total_articles = 0
    total_chunks = 0
    input_offset = 0

    if checkpoint_enabled and resume and checkpoint_path.exists():
        checkpoint = load_manifest(checkpoint_path)
        if strict_compatibility:
            ensure_manifest_compatible(
                checkpoint,
                {
                    'schema_version': CHECKPOINT_SCHEMA_VERSION,
                    'strategy': args.strategy,
                    'input_fingerprint': file_fingerprint(dump_path),
                    'output_file': str(output_file),
                    'is_jsonl': is_jsonl,
                    'max_articles': max_articles,
                },
                required_keys=[
                    'schema_version',
                    'strategy',
                    'input_fingerprint',
                    'output_file',
                    'is_jsonl',
                    'max_articles',
                    'total_articles',
                    'total_chunks',
                    'input_offset',
                ],
            )

        total_articles = int(checkpoint.get('total_articles', 0))
        total_chunks = int(checkpoint.get('total_chunks', 0))
        input_offset = int(checkpoint.get('input_offset', 0))

        if output_file.exists():
            new_size = truncate_to_last_complete_jsonl_line(output_file)
            logger.info(f"Output recovery complete, file truncated to {new_size} bytes")
        else:
            raise FileNotFoundError(
                f"Checkpoint exists but output file is missing: {output_file}. "
                "Use --reset-checkpoint to restart cleanly."
            )

        logger.info(
            f"Resuming from checkpoint: articles={total_articles}, chunks={total_chunks}, "
            f"input_offset={input_offset}"
        )
    
    try:
        output_mode = 'a' if (checkpoint_enabled and resume and checkpoint_path.exists()) else 'w'
        with open(output_file, output_mode, encoding='utf-8') as f:
            logger.info("Starting article processing...")
            
            # Handle JSONL input (from download_wikipedia.py)
            if is_jsonl:
                logger.info("Processing JSONL input...")
                with open(dump_path, 'r', encoding='utf-8') as jsonl_file:
                    for line_num, line in enumerate(tqdm(jsonl_file, desc="Processing articles", unit=" articles")):
                        if line_num < input_offset:
                            continue

                        # Check max_articles limit
                        if max_articles and total_articles >= max_articles:
                            logger.info(f"Reached max_articles limit: {max_articles}")
                            break
                        
                        try:
                            article = json.loads(line.strip())
                            
                            # Chunk the article
                            chunks = text_chunker.chunk_article(article)
                            
                            # Write chunks to JSONL
                            for chunk in chunks:
                                f.write(json.dumps(chunk, ensure_ascii=False) + '\n')
                            
                            total_articles += 1
                            total_chunks += len(chunks)

                            input_offset = line_num + 1
                            if checkpoint_enabled and checkpoint_interval > 0 and total_articles % checkpoint_interval == 0:
                                _save_chunking_checkpoint(
                                    checkpoint_path=checkpoint_path,
                                    strategy=args.strategy,
                                    dump_path=dump_path,
                                    output_file=output_file,
                                    is_jsonl=is_jsonl,
                                    max_articles=max_articles,
                                    total_articles=total_articles,
                                    total_chunks=total_chunks,
                                    input_offset=input_offset,
                                )
                        
                        except json.JSONDecodeError as e:
                            logger.error(f"JSON decode error at line {line_num + 1}: {e}")
                            continue
                        except Exception as e:
                            logger.error(f"Error processing article at line {line_num + 1}: {e}")
                            continue
            
            # Handle XML input (for production)
            else:
                logger.info("Processing XML input...")
                skipped_processed_articles = 0
                for article in wiki_parser.extract_articles():
                    if skipped_processed_articles < total_articles:
                        skipped_processed_articles += 1
                        continue

                    if max_articles and total_articles >= max_articles:
                        logger.info(f"Reached max_articles limit: {max_articles}")
                        break

                    try:
                        # Chunk the article
                        chunks = text_chunker.chunk_article(article)
                        
                        # Write chunks to JSONL
                        for chunk in chunks:
                            f.write(json.dumps(chunk, ensure_ascii=False) + '\n')
                        
                        total_articles += 1
                        total_chunks += len(chunks)

                        if checkpoint_enabled and checkpoint_interval > 0 and total_articles % checkpoint_interval == 0:
                            _save_chunking_checkpoint(
                                checkpoint_path=checkpoint_path,
                                strategy=args.strategy,
                                dump_path=dump_path,
                                output_file=output_file,
                                is_jsonl=is_jsonl,
                                max_articles=max_articles,
                                total_articles=total_articles,
                                total_chunks=total_chunks,
                                input_offset=total_articles,
                            )
                    
                    except Exception as e:
                        logger.error(f"Error processing article {article.get('doc_id', 'unknown')}: {e}")
                        continue
        
        # Print summary
        avg_chunks_per_article = total_chunks / total_articles if total_articles > 0 else 0
        
        summary = f"""
{'='*60}
Processing Complete!
{'='*60}
Strategy:              {args.strategy}
Wikipedia Dump:        {dump_path}
Output File:           {output_file}
Total Articles:        {total_articles:,}
Total Chunks:          {total_chunks:,}
Avg Chunks/Article:    {avg_chunks_per_article:.1f}
Output File Size:      {output_file.stat().st_size / (1024*1024):.2f} MB
{'='*60}
        """
        
        print(summary)
        logger.info(summary)

        if checkpoint_enabled and checkpoint_path.exists():
            checkpoint_path.unlink()
            logger.info(f"Removed checkpoint after successful completion: {checkpoint_path}")

        logger.info("Wikipedia chunk preparation completed successfully")
    
    except KeyboardInterrupt:
        logger.warning("Processing interrupted by user")
        if checkpoint_enabled:
            _save_chunking_checkpoint(
                checkpoint_path=checkpoint_path,
                strategy=args.strategy,
                dump_path=dump_path,
                output_file=output_file,
                is_jsonl=is_jsonl,
                max_articles=max_articles,
                total_articles=total_articles,
                total_chunks=total_chunks,
                input_offset=input_offset if is_jsonl else total_articles,
            )
            logger.info(f"Checkpoint saved after interruption: {checkpoint_path}")
        print(f"\nProcessing interrupted. Partial results saved to: {output_file}")
        sys.exit(1)
    
    except Exception as e:
        logger.error(f"Processing failed: {e}")
        raise


if __name__ == '__main__':
    main()
