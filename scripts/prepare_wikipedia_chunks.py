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
    else:
        output_dir = Path(config.get('paths.processed', 'data/processed'))
    
    output_dir.mkdir(parents=True, exist_ok=True)
    output_file = output_dir / f"wiki_chunks_{args.strategy}.jsonl"
    
    logger.info(f"Output will be saved to: {output_file}")
    
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
    
    try:
        with open(output_file, 'w', encoding='utf-8') as f:
            logger.info("Starting article processing...")
            
            # Handle JSONL input (from download_wikipedia.py)
            if is_jsonl:
                logger.info("Processing JSONL input...")
                with open(dump_path, 'r', encoding='utf-8') as jsonl_file:
                    for line_num, line in enumerate(tqdm(jsonl_file, desc="Processing articles", unit=" articles")):
                        # Check max_articles limit
                        if max_articles and line_num >= max_articles:
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
                        
                        except json.JSONDecodeError as e:
                            logger.error(f"JSON decode error at line {line_num + 1}: {e}")
                            continue
                        except Exception as e:
                            logger.error(f"Error processing article at line {line_num + 1}: {e}")
                            continue
            
            # Handle XML input (for production)
            else:
                logger.info("Processing XML input...")
                for article in wiki_parser.extract_articles():
                    try:
                        # Chunk the article
                        chunks = text_chunker.chunk_article(article)
                        
                        # Write chunks to JSONL
                        for chunk in chunks:
                            f.write(json.dumps(chunk, ensure_ascii=False) + '\n')
                        
                        total_articles += 1
                        total_chunks += len(chunks)
                    
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
        logger.info("Wikipedia chunk preparation completed successfully")
    
    except KeyboardInterrupt:
        logger.warning("Processing interrupted by user")
        print(f"\nProcessing interrupted. Partial results saved to: {output_file}")
        sys.exit(1)
    
    except Exception as e:
        logger.error(f"Processing failed: {e}")
        raise


if __name__ == '__main__':
    main()
