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
    
    # Determine Wikipedia dump path
    if args.dump:
        dump_path = Path(args.dump)
    else:
        # Check if dump path is in config
        dump_path = Path(config.get('paths.wikipedia_dump', 'data/raw/enwiki-sample.xml'))
    
    if not dump_path.exists():
        logger.error(
            f"Wikipedia dump not found: {dump_path}\n"
            f"Please either:\n"
            f"1. Provide --dump argument with path to Wikipedia XML dump\n"
            f"2. Download a Wikipedia dump and place it at {dump_path}\n"
            f"3. Create a sample XML file for testing"
        )
        sys.exit(1)
    
    logger.info(f"Using Wikipedia dump: {dump_path}")
    
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
        wiki_parser = WikipediaParser(str(dump_path), max_articles=max_articles)
        text_chunker = TextChunker()
        logger.info("Initialized WikipediaParser and TextChunker")
    except Exception as e:
        logger.error(f"Failed to initialize components: {e}")
        sys.exit(1)
    
    # Process articles
    total_articles = 0
    total_chunks = 0
    
    try:
        with open(output_file, 'w', encoding='utf-8') as f:
            logger.info("Starting article processing...")
            
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
