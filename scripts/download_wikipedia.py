#!/usr/bin/env python3
"""
Download Wikipedia Data Script

Supports multiple strategies:
1. Development: Uses HuggingFace datasets to download ~10k articles (fast, no XML parsing)
2. Validation: Downloads ~100k articles from HuggingFace
3. Production: Downloads full Wikipedia dump from dumps.wikimedia.org (~20GB)

Usage:
    python scripts/download_wikipedia.py --strategy development
    python scripts/download_wikipedia.py --strategy validation
    python scripts/download_wikipedia.py --strategy production --dump-date 20240101
"""

import argparse
import json
import logging
import os
import sys
from pathlib import Path
from typing import Dict, List, Optional
from datetime import datetime
from datasets import load_dataset

# Add project root to Python path
project_root = Path(__file__).parent.parent
sys.path.insert(0, str(project_root))

from src.utils.config import Config
from src.utils.logger import setup_logger

logger = logging.getLogger(__name__)


class WikipediaDownloader:
    """Download and prepare Wikipedia data for processing."""
    
    def __init__(self, config_path: str = "config.yaml"):
        """Initialize Wikipedia downloader.
        
        Args:
            config_path: Path to configuration file
        """
        self.config = Config(config_path)
        log_file = self.config.get("logging.log_file", "logs/month2.log")
        log_level_str = self.config.get("logging.log_level", "INFO")
        console_level_str = self.config.get("logging.console_level", "ERROR")
        
        # Convert string log levels to logging constants
        log_level = getattr(logging, log_level_str.upper(), logging.INFO)
        console_level = getattr(logging, console_level_str.upper(), logging.ERROR)
        
        self.logger = setup_logger("download_wikipedia", log_file, log_level, console_level)
        
    def download_development_data(self, max_articles: int = 10000) -> str:
        """Download development sample using HuggingFace datasets.
        
        Args:
            max_articles: Maximum number of articles to download
            
        Returns:
            Path to output JSONL file
        """
        
        self.logger.info(f"Downloading {max_articles} Wikipedia articles from HuggingFace...")
        
        # Load Wikipedia dataset from HuggingFace
        # Using wikimedia/wikipedia "20231101.en" (English Wikipedia from November 2023)
        # This is a Parquet-based dataset (no loading scripts)
        dataset = load_dataset(
            "wikimedia/wikipedia",
            "20231101.en",
            split=f"train[:{max_articles}]"
        )
        
        # Prepare output path
        output_path = self.config.get("data.wikipedia_sample_dev", "data/raw/wiki_sample_development.jsonl")
        output_file = Path(output_path)
        output_file.parent.mkdir(parents=True, exist_ok=True)
        
        # Convert to JSONL format
        self.logger.info(f"Converting to JSONL format and saving to {output_path}...")
        articles_written = 0
        total_chars = 0
        
        with open(output_file, 'w', encoding='utf-8') as f:
            for idx, article in enumerate(dataset):
                # Create JSONL entry
                entry = {
                    'doc_id': f"wiki_{idx:08d}",
                    'title': article['title'],
                    'text': article['text'],
                    'url': article.get('url', ''),
                    'timestamp': datetime.now().isoformat(),
                    'source': 'huggingface_wikipedia_20220301'
                }
                
                # Write to file
                f.write(json.dumps(entry, ensure_ascii=False) + '\n')
                articles_written += 1
                total_chars += len(article['text'])
                
                # Progress tracking
                if (idx + 1) % 1000 == 0:
                    self.logger.info(f"Progress: {idx + 1}/{max_articles} articles processed")
        
        # Calculate file size
        file_size_mb = output_file.stat().st_size / (1024 * 1024)
        avg_chars_per_article = total_chars / articles_written if articles_written > 0 else 0
        
        # Print summary
        self.logger.info("=" * 80)
        self.logger.info("Download Complete!")
        self.logger.info(f"Total articles: {articles_written}")
        self.logger.info(f"File size: {file_size_mb:.2f} MB")
        self.logger.info(f"Average characters per article: {avg_chars_per_article:.0f}")
        self.logger.info(f"Output file: {output_path}")
        self.logger.info("=" * 80)
        
        return str(output_file)
    
    def download_validation_data(self, max_articles: int = 100000) -> str:
        """Download validation sample using HuggingFace datasets.
        
        Args:
            max_articles: Maximum number of articles to download
            
        Returns:
            Path to output JSONL file
        """
        
        self.logger.info(f"Downloading {max_articles} Wikipedia articles from HuggingFace...")
        
        # Load Wikipedia dataset
        dataset = load_dataset(
            "wikimedia/wikipedia",
            "20231101.en",
            split=f"train[:{max_articles}]"
        )
        
        # Prepare output path
        output_path = self.config.get("data.wikipedia_sample_val", "data/raw/wiki_sample_validation.jsonl")
        output_file = Path(output_path)
        output_file.parent.mkdir(parents=True, exist_ok=True)
        
        # Convert to JSONL format
        self.logger.info(f"Converting to JSONL format and saving to {output_path}...")
        articles_written = 0
        total_chars = 0
        
        with open(output_file, 'w', encoding='utf-8') as f:
            for idx, article in enumerate(dataset):
                # Create JSONL entry
                entry = {
                    'doc_id': f"wiki_{idx:08d}",
                    'title': article['title'],
                    'text': article['text'],
                    'url': article.get('url', ''),
                    'timestamp': datetime.now().isoformat(),
                    'source': 'huggingface_wikipedia_20220301'
                }
                
                # Write to file
                f.write(json.dumps(entry, ensure_ascii=False) + '\n')
                articles_written += 1
                total_chars += len(article['text'])
                
                # Progress tracking
                if (idx + 1) % 5000 == 0:
                    self.logger.info(f"Progress: {idx + 1}/{max_articles} articles processed")
        
        # Calculate file size
        file_size_mb = output_file.stat().st_size / (1024 * 1024)
        avg_chars_per_article = total_chars / articles_written if articles_written > 0 else 0
        
        # Print summary
        self.logger.info("=" * 80)
        self.logger.info("Download Complete!")
        self.logger.info(f"Total articles: {articles_written}")
        self.logger.info(f"File size: {file_size_mb:.2f} MB")
        self.logger.info(f"Average characters per article: {avg_chars_per_article:.0f}")
        self.logger.info(f"Output file: {output_path}")
        self.logger.info("=" * 80)
        
        return str(output_file)
    
    def download_production_data(self, dump_date: Optional[str] = None) -> str:
        """Download full Wikipedia dump from dumps.wikimedia.org.
        
        Args:
            dump_date: Wikipedia dump date in YYYYMMDD format (e.g., 20240101)
                      If None, uses 'latest'
            
        Returns:
            Path to downloaded dump file
        """
        import requests
        from tqdm import tqdm
        
        # Use latest dump if no date specified
        if dump_date is None:
            dump_date = "latest"
        
        # Construct URL
        # Format: https://dumps.wikimedia.org/enwiki/YYYYMMDD/enwiki-YYYYMMDD-pages-articles.xml.bz2
        if dump_date == "latest":
            url = "https://dumps.wikimedia.org/enwiki/latest/enwiki-latest-pages-articles.xml.bz2"
        else:
            url = f"https://dumps.wikimedia.org/enwiki/{dump_date}/enwiki-{dump_date}-pages-articles.xml.bz2"
        
        # Prepare output path
        output_path = self.config.get("data.wikipedia_dump", "data/raw/enwiki-latest-pages-articles.xml.bz2")
        output_file = Path(output_path)
        output_file.parent.mkdir(parents=True, exist_ok=True)
        
        self.logger.info(f"Downloading Wikipedia dump from {url}...")
        self.logger.warning("This is a large file (~20GB) and may take a long time!")
        
        # Download with progress bar
        response = requests.get(url, stream=True)
        response.raise_for_status()
        
        total_size = int(response.headers.get('content-length', 0))
        
        with open(output_file, 'wb') as f, tqdm(
            total=total_size,
            unit='B',
            unit_scale=True,
            unit_divisor=1024,
            desc=f"Downloading to {output_file.name}"
        ) as pbar:
            for chunk in response.iter_content(chunk_size=8192):
                if chunk:
                    f.write(chunk)
                    pbar.update(len(chunk))
        
        # Calculate file size
        file_size_gb = output_file.stat().st_size / (1024 * 1024 * 1024)
        
        # Print summary
        self.logger.info("=" * 80)
        self.logger.info("Download Complete!")
        self.logger.info(f"File size: {file_size_gb:.2f} GB")
        self.logger.info(f"Output file: {output_path}")
        self.logger.info("=" * 80)
        self.logger.info("Note: This XML dump requires parsing with WikipediaParser")
        
        return str(output_file)


def main():
    """Main entry point for Wikipedia download script."""
    parser = argparse.ArgumentParser(
        description="Download Wikipedia data for AIST-FYP project",
        formatter_class=argparse.RawDescriptionHelpFormatter,
        epilog="""
Examples:
    # Download development sample (10k articles, ~5 min)
    python scripts/download_wikipedia.py --strategy development
    
    # Download validation sample (100k articles, ~30 min)
    python scripts/download_wikipedia.py --strategy validation
    
    # Download full production dump (~20GB, several hours)
    python scripts/download_wikipedia.py --strategy production
    
    # Download specific dump date
    python scripts/download_wikipedia.py --strategy production --dump-date 20240101
        """
    )
    
    parser.add_argument(
        '--strategy',
        type=str,
        required=True,
        choices=['development', 'validation', 'production'],
        help='Download strategy: development (10k articles), validation (100k), or production (full dump)'
    )
    
    parser.add_argument(
        '--dump-date',
        type=str,
        default=None,
        help='Wikipedia dump date in YYYYMMDD format (only for production strategy). Uses "latest" if not specified.'
    )
    
    parser.add_argument(
        '--config',
        type=str,
        default='config.yaml',
        help='Path to configuration file (default: config.yaml)'
    )
    
    parser.add_argument(
        '--max-articles',
        type=int,
        default=None,
        help='Override maximum number of articles (for development/validation strategies)'
    )
    
    args = parser.parse_args()
    
    # Validate dump-date format if provided
    if args.dump_date and args.dump_date != "latest":
        try:
            datetime.strptime(args.dump_date, "%Y%m%d")
        except ValueError:
            print(f"Error: Invalid dump-date format '{args.dump_date}'. Expected YYYYMMDD (e.g., 20240101)")
            sys.exit(1)
    
    # Initialize downloader
    try:
        downloader = WikipediaDownloader(args.config)
    except Exception as e:
        print(f"Error initializing downloader: {e}")
        sys.exit(1)
    
    # Execute download based on strategy
    try:
        start_time = datetime.now()
        
        if args.strategy == 'development':
            max_articles = args.max_articles or downloader.config.get("data_strategy.development.max_articles", 10000)
            output_file = downloader.download_development_data(max_articles)
        elif args.strategy == 'validation':
            max_articles = args.max_articles or downloader.config.get("data_strategy.validation.max_articles", 100000)
            output_file = downloader.download_validation_data(max_articles)
        elif args.strategy == 'production':
            output_file = downloader.download_production_data(args.dump_date)
        
        end_time = datetime.now()
        duration = end_time - start_time
        
        print(f"\n✓ Download completed successfully in {duration}")
        print(f"✓ Output file: {output_file}")
        
    except KeyboardInterrupt:
        print("\n\nDownload cancelled by user")
        sys.exit(1)
    except Exception as e:
        print(f"\n✗ Error during download: {e}")
        import traceback
        traceback.print_exc()
        sys.exit(1)


if __name__ == "__main__":
    main()
