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


def _build_article_export_checkpoint_path(checkpoint_dir: Path, strategy: str) -> Path:
    return checkpoint_dir / f"prepare_articles_{strategy}.json"


def _default_article_jsonl_path(strategy: str) -> Path:
    return Path(f"data/processed/wiki_articles_{strategy}.jsonl")


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


def _save_article_export_checkpoint(
    checkpoint_path: Path,
    strategy: str,
    dump_path: Path,
    article_output_file: Path,
    max_articles,
    total_articles: int,
) -> None:
    save_manifest(
        checkpoint_path,
        {
            'schema_version': CHECKPOINT_SCHEMA_VERSION,
            'strategy': strategy,
            'input_fingerprint': file_fingerprint(dump_path),
            'article_output_file': str(article_output_file),
            'max_articles': max_articles,
            'total_articles': total_articles,
        }
    )


def _export_xml_articles_to_jsonl(
    logger,
    wiki_parser: WikipediaParser,
    dump_path: Path,
    article_output_file: Path,
    strategy: str,
    max_articles,
    checkpoint_enabled: bool,
    checkpoint_interval: int,
    checkpoint_dir: Path,
    resume: bool,
    strict_compatibility: bool,
    reset_checkpoint: bool,
) -> int:
    checkpoint_path = _build_article_export_checkpoint_path(checkpoint_dir, strategy)

    if reset_checkpoint and checkpoint_path.exists():
        checkpoint_path.unlink()
        logger.info(f"Removed existing article export checkpoint: {checkpoint_path}")

    total_articles = 0
    resumed_from_checkpoint = False

    if checkpoint_enabled and resume and checkpoint_path.exists():
        checkpoint = load_manifest(checkpoint_path)
        if strict_compatibility:
            ensure_manifest_compatible(
                checkpoint,
                {
                    'schema_version': CHECKPOINT_SCHEMA_VERSION,
                    'strategy': strategy,
                    'input_fingerprint': file_fingerprint(dump_path),
                    'article_output_file': str(article_output_file),
                    'max_articles': max_articles,
                },
                required_keys=[
                    'schema_version',
                    'strategy',
                    'input_fingerprint',
                    'article_output_file',
                    'max_articles',
                    'total_articles',
                ],
            )

        total_articles = int(checkpoint.get('total_articles', 0))
        resumed_from_checkpoint = True

        if article_output_file.exists():
            new_size = truncate_to_last_complete_jsonl_line(article_output_file)
            logger.info(f"Article export recovery complete, file truncated to {new_size} bytes")
        else:
            raise FileNotFoundError(
                f"Article export checkpoint exists but output file is missing: {article_output_file}. "
                "Use --reset-checkpoint to restart cleanly."
            )

        logger.info(f"Resuming article export from checkpoint: total_articles={total_articles}")

    output_mode = 'a' if (checkpoint_enabled and resume and checkpoint_path.exists()) else 'w'
    skipped_processed_articles = 0
    articles_to_skip = total_articles if resumed_from_checkpoint else 0

    with open(article_output_file, output_mode, encoding='utf-8') as article_file:
        logger.info(f"Exporting XML articles to JSONL: {article_output_file}")
        for article in wiki_parser.extract_articles():
            if skipped_processed_articles < articles_to_skip:
                skipped_processed_articles += 1
                continue

            if max_articles and total_articles >= max_articles:
                logger.info(f"Reached max_articles limit during article export: {max_articles}")
                break

            article_file.write(json.dumps(article, ensure_ascii=False) + '\n')
            total_articles += 1

            if checkpoint_enabled and checkpoint_interval > 0 and total_articles % checkpoint_interval == 0:
                _save_article_export_checkpoint(
                    checkpoint_path=checkpoint_path,
                    strategy=strategy,
                    dump_path=dump_path,
                    article_output_file=article_output_file,
                    max_articles=max_articles,
                    total_articles=total_articles,
                )

    if checkpoint_enabled and checkpoint_path.exists():
        checkpoint_path.unlink()
        logger.info(f"Removed article export checkpoint after successful completion: {checkpoint_path}")

    logger.info(f"Article export complete: {total_articles:,} articles written to {article_output_file}")
    return total_articles


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
        '--article-jsonl',
        type=str,
        default=None,
        help='Path to intermediate article JSONL file for two-stage XML processing'
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

    # Production shortcut: if intermediate article JSONL already exists,
    # allow stage-2 chunking without requiring raw XML dump.
    article_jsonl_path = Path(args.article_jsonl) if args.article_jsonl else _default_article_jsonl_path(args.strategy)
    reuse_intermediate_jsonl = (
        args.strategy == 'production'
        and article_jsonl_path.exists()
        and not args.reset_checkpoint
    )

    if reuse_intermediate_jsonl:
        dump_path = article_jsonl_path
        is_jsonl = True
        logger.info(f"Reusing existing intermediate article JSONL: {article_jsonl_path}")
        logger.info(f"Using Wikipedia data: {dump_path} (format: JSONL)")
    else:
        if not dump_path.exists():
            if args.strategy == 'production' and args.reset_checkpoint:
                logger.error(
                    "Raw production dump is required when using --reset-checkpoint.\n"
                    f"Expected dump path: {dump_path}\n"
                    f"Or remove --reset-checkpoint to reuse existing intermediate JSONL at: {article_jsonl_path}"
                )
            else:
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
    if checkpoint_enabled:
        checkpoint_dir.mkdir(parents=True, exist_ok=True)

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
    
    if not is_jsonl:
        article_jsonl_path.parent.mkdir(parents=True, exist_ok=True)

        logger.info(
            "Two-stage mode enabled for XML input: "
            f"Stage 1 export XML -> {article_jsonl_path}, Stage 2 chunk from JSONL"
        )

        if article_jsonl_path.exists() and not args.reset_checkpoint:
            logger.info(f"Reusing existing intermediate article JSONL: {article_jsonl_path}")
        else:
            _export_xml_articles_to_jsonl(
                logger=logger,
                wiki_parser=wiki_parser,
                dump_path=dump_path,
                article_output_file=article_jsonl_path,
                strategy=args.strategy,
                max_articles=max_articles,
                checkpoint_enabled=checkpoint_enabled,
                checkpoint_interval=checkpoint_interval,
                checkpoint_dir=checkpoint_dir,
                resume=resume,
                strict_compatibility=strict_compatibility,
                reset_checkpoint=args.reset_checkpoint,
            )

        dump_path = article_jsonl_path
        is_jsonl = True
        logger.info(f"Stage 1 finished. Continuing with Stage 2 chunking from: {dump_path}")

    # Process articles
    total_articles = 0
    total_chunks = 0
    run_articles_processed = 0
    run_articles_failed = 0
    input_offset = 0
    resumed_from_checkpoint = False

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
        resumed_from_checkpoint = True

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
    else:
        logger.info("Starting fresh chunking run (current-run throughput starts at 0)")
    
    try:
        output_mode = 'a' if (checkpoint_enabled and resume and checkpoint_path.exists()) else 'w'
        with open(output_file, output_mode, encoding='utf-8') as f:
            logger.info("Starting article processing...")
            
            # Handle JSONL input (from download_wikipedia.py)
            if is_jsonl:
                logger.info("Processing JSONL input...")
                with open(dump_path, 'r', encoding='utf-8') as jsonl_file:
                    skipped_lines = 0
                    while skipped_lines < input_offset:
                        if not jsonl_file.readline():
                            break
                        skipped_lines += 1

                    progress_bar = tqdm(desc="Chunking articles (current run)", unit=" articles")
                    progress_bar.set_postfix(total_articles=total_articles)
                    for line_num, line in enumerate(jsonl_file, start=skipped_lines):

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
                            run_articles_processed += 1
                            total_chunks += len(chunks)
                            progress_bar.update(1)
                            progress_bar.set_postfix(total_articles=total_articles)

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
                            run_articles_failed += 1
                            continue
                        except Exception as e:
                            logger.error(f"Error processing article at line {line_num + 1}: {e}")
                            run_articles_failed += 1
                            continue

                    progress_bar.close()
            
            # XML inputs are converted to JSONL in stage 1, so stage 2 always runs in JSONL mode
        
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
Current Run Articles:  {run_articles_processed:,}
Current Run Failures:  {run_articles_failed:,}
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
