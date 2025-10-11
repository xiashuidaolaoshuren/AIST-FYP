"""
Wikipedia XML parser for extracting article text.

This module uses wikiextractor to parse Wikipedia XML dumps and extract
clean article text with metadata. Supports limiting the number of articles
for development and testing.
"""

import re
import xml.etree.ElementTree as ET
from typing import Dict, Iterator, Optional
from pathlib import Path
import json
from tqdm import tqdm

from src.utils.logger import setup_logger


class WikipediaParser:
    """
    Parser for Wikipedia XML dumps.
    
    Extracts article text from Wikipedia XML dumps, filtering out redirects,
    disambiguation pages, and other non-content pages.
    
    Attributes:
        dump_path: Path to the Wikipedia XML dump file
        max_articles: Maximum number of articles to extract (None = all)
        logger: Logger instance for tracking progress and errors
    
    Example:
        >>> parser = WikipediaParser("enwiki-latest-pages-articles.xml", max_articles=10000)
        >>> for article in parser.extract_articles():
        ...     print(article['title'], len(article['text']))
    """
    
    def __init__(self, dump_path: str, max_articles: Optional[int] = None):
        """
        Initialize the Wikipedia parser.
        
        Args:
            dump_path: Path to the Wikipedia XML dump file
            max_articles: Maximum number of articles to extract (default: None for all)
        
        Raises:
            FileNotFoundError: If dump_path doesn't exist
        """
        self.dump_path = Path(dump_path)
        self.max_articles = max_articles
        self.logger = setup_logger(__name__)
        
        if not self.dump_path.exists():
            raise FileNotFoundError(f"Wikipedia dump not found: {self.dump_path}")
        
        self.logger.info(f"Initialized WikipediaParser with dump: {self.dump_path}")
        if self.max_articles:
            self.logger.info(f"Limiting extraction to {self.max_articles} articles")
    
    def extract_articles(self) -> Iterator[Dict[str, str]]:
        """
        Extract articles from the Wikipedia XML dump.
        
        Yields dictionaries containing article metadata:
        - doc_id: Unique identifier (format: "enwiki_{page_id}")
        - title: Article title
        - text: Clean article text (wikimarkup removed)
        - url: Wikipedia URL
        
        Yields:
            Dictionary with article data
        
        Example:
            >>> parser = WikipediaParser("dump.xml")
            >>> for article in parser.extract_articles():
            ...     process_article(article)
        """
        articles_processed = 0
        articles_skipped = 0
        
        # Use iterparse for memory-efficient XML parsing
        context = ET.iterparse(
            str(self.dump_path),
            events=('start', 'end')
        )
        
        # Track current page data
        current_page = {}
        in_page = False
        in_text = False
        current_element = None
        
        pbar = tqdm(desc="Parsing Wikipedia", unit=" articles")
        
        try:
            for event, elem in context:
                tag = elem.tag.split('}')[-1]  # Remove namespace
                
                if event == 'start':
                    if tag == 'page':
                        in_page = True
                        current_page = {}
                    elif tag == 'text':
                        in_text = True
                    
                elif event == 'end':
                    if tag == 'id' and in_page and 'id' not in current_page:
                        # First ID in page is the page ID
                        current_page['id'] = elem.text
                    
                    elif tag == 'title' and in_page:
                        current_page['title'] = elem.text or ""
                    
                    elif tag == 'text' and in_page and in_text:
                        current_page['text'] = elem.text or ""
                        in_text = False
                    
                    elif tag == 'page' and in_page:
                        # Process completed page
                        in_page = False
                        
                        if self._should_include_page(current_page):
                            article = self._process_page(current_page)
                            if article:
                                yield article
                                articles_processed += 1
                                pbar.update(1)
                                
                                if self.max_articles and articles_processed >= self.max_articles:
                                    self.logger.info(f"Reached max_articles limit: {self.max_articles}")
                                    break
                        else:
                            articles_skipped += 1
                        
                        # Clear element to free memory
                        elem.clear()
                        current_page = {}
                    
                    # Clear processed elements to save memory
                    if tag in ['id', 'title', 'text', 'page']:
                        elem.clear()
        
        except ET.ParseError as e:
            self.logger.error(f"XML parsing error: {e}")
            raise
        
        finally:
            pbar.close()
            self.logger.info(
                f"Extraction complete: {articles_processed} articles processed, "
                f"{articles_skipped} articles skipped"
            )
    
    def _should_include_page(self, page: Dict[str, str]) -> bool:
        """
        Check if a page should be included in extraction.
        
        Filters out:
        - Redirects
        - Disambiguation pages
        - Wikipedia meta pages (Wikipedia:, Help:, etc.)
        - Empty pages
        
        Args:
            page: Dictionary with page data
        
        Returns:
            True if page should be included, False otherwise
        """
        if not page.get('title') or not page.get('text'):
            return False
        
        title = page['title']
        text = page['text']
        
        # Skip redirects
        if text.strip().upper().startswith('#REDIRECT'):
            return False
        
        # Skip disambiguation pages
        if '(disambiguation)' in title.lower():
            return False
        
        # Skip meta pages
        meta_prefixes = ['Wikipedia:', 'Help:', 'Template:', 'Category:', 
                        'Portal:', 'File:', 'MediaWiki:', 'Module:']
        if any(title.startswith(prefix) for prefix in meta_prefixes):
            return False
        
        # Skip very short pages (likely stubs or malformed)
        if len(text.strip()) < 100:
            return False
        
        return True
    
    def _process_page(self, page: Dict[str, str]) -> Optional[Dict[str, str]]:
        """
        Process a Wikipedia page into clean article format.
        
        Removes wikimarkup and extracts clean text.
        
        Args:
            page: Dictionary with page data
        
        Returns:
            Dictionary with processed article data, or None if processing fails
        """
        try:
            doc_id = f"enwiki_{page['id']}"
            title = page['title']
            text = self._clean_wikitext(page['text'])
            url = f"https://en.wikipedia.org/wiki/{title.replace(' ', '_')}"
            
            return {
                'doc_id': doc_id,
                'title': title,
                'text': text,
                'url': url
            }
        
        except Exception as e:
            self.logger.error(f"Error processing page {page.get('title', 'unknown')}: {e}")
            return None
    
    def _clean_wikitext(self, wikitext: str) -> str:
        """
        Remove wikimarkup and extract clean text.
        
        This is a simplified cleaner. For production, consider using
        mwparserfromhell or similar libraries for more robust parsing.
        
        Args:
            wikitext: Raw wikimarkup text
        
        Returns:
            Cleaned plain text
        """
        # Remove comments
        text = re.sub(r'<!--.*?-->', '', wikitext, flags=re.DOTALL)
        
        # Remove templates (simplified - doesn't handle nested templates)
        text = re.sub(r'\{\{.*?\}\}', '', text, flags=re.DOTALL)
        
        # Remove references
        text = re.sub(r'<ref[^>]*>.*?</ref>', '', text, flags=re.DOTALL | re.IGNORECASE)
        text = re.sub(r'<ref[^>]*/>', '', text, flags=re.IGNORECASE)
        
        # Remove HTML tags
        text = re.sub(r'<[^>]+>', '', text)
        
        # Remove file/image links
        text = re.sub(r'\[\[File:.*?\]\]', '', text, flags=re.IGNORECASE)
        text = re.sub(r'\[\[Image:.*?\]\]', '', text, flags=re.IGNORECASE)
        
        # Convert wiki links: [[Link|Text]] -> Text, [[Link]] -> Link
        text = re.sub(r'\[\[([^|\]]+)\|([^\]]+)\]\]', r'\2', text)
        text = re.sub(r'\[\[([^\]]+)\]\]', r'\1', text)
        
        # Remove external links: [http://... Text] -> Text
        text = re.sub(r'\[https?://[^\s\]]+\s+([^\]]+)\]', r'\1', text)
        text = re.sub(r'\[https?://[^\s\]]+\]', '', text)
        
        # Remove wiki formatting
        text = re.sub(r"'''", '', text)  # Bold
        text = re.sub(r"''", '', text)   # Italic
        
        # Remove multiple newlines
        text = re.sub(r'\n{3,}', '\n\n', text)
        
        # Remove leading/trailing whitespace
        text = text.strip()
        
        return text
