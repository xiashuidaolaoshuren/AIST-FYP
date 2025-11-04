"""
Text chunker for splitting articles into sentence-level chunks.

This module uses spaCy to segment text into sentences and create
chunk dictionaries with metadata for retrieval and indexing.
"""

import spacy
from typing import Dict, List, Optional
from src.utils.logger import setup_logger


class TextChunker:
    """
    Sentence-level text chunker using spaCy.
    
    Segments article text into sentence-level chunks with character offsets
    and metadata. Optimized for retrieval use cases.
    
    Attributes:
        nlp: spaCy language model
        overlap_sentences: Number of overlapping sentences between chunks
        min_length: Minimum character length for a valid sentence
        logger: Logger instance
    
    Example:
        >>> chunker = TextChunker()
        >>> article = {'doc_id': 'enwiki_123', 'title': 'Example', 'text': 'Sentence 1. Sentence 2.'}
        >>> chunks = chunker.chunk_article(article)
        >>> print(chunks[0]['text'])
    """
    
    def __init__(self, overlap_sentences: int = 0, min_length: int = 10):
        """
        Initialize the text chunker.
        
        Args:
            overlap_sentences: Number of sentences to overlap between chunks (default: 0)
            min_length: Minimum character length for valid sentences (default: 10)
        
        Raises:
            OSError: If spaCy model 'en_core_web_sm' is not installed
        """
        self.overlap_sentences = overlap_sentences
        self.min_length = min_length
        self.logger = setup_logger(__name__)
        
        try:
            # Load spaCy model - using sentencizer for speed
            self.nlp = spacy.load('en_core_web_sm', exclude=['ner', 'parser'])
            # Enable sentencizer component (faster than full parser)
            if 'sentencizer' not in self.nlp.pipe_names:
                self.nlp.add_pipe('sentencizer')
            
            self.logger.info("Loaded spaCy model en_core_web_sm with sentencizer")
        
        except OSError as e:
            self.logger.error(
                "spaCy model 'en_core_web_sm' not found. "
                "Please run: python -m spacy download en_core_web_sm"
            )
            raise
    
    def chunk_article(self, article: Dict[str, str]) -> List[Dict[str, any]]:
        """
        Chunk an article into sentence-level fragments.
        
        Segments article text using spaCy sentencizer and creates chunk
        dictionaries with metadata matching the EvidenceChunk schema.
        
        Args:
            article: Dictionary with keys 'doc_id', 'title', 'text'
        
        Returns:
            List of chunk dictionaries, each containing:
            - doc_id: Document identifier
            - sent_id: Sentence index (0-based)
            - text: Sentence text
            - char_start: Character offset (start)
            - char_end: Character offset (end)
            - source: Source corpus name
            - version: Version identifier
        
        Example:
            >>> article = {'doc_id': 'enwiki_123', 'title': 'Test', 'text': 'Hello. World.'}
            >>> chunks = chunker.chunk_article(article)
            >>> len(chunks)
            2
        """
        doc_id = article['doc_id']
        text = article['text']
        
        # Process text with spaCy
        doc = self.nlp(text)
        
        chunks = []
        sent_id = 0
        
        for sent in doc.sents:
            sent_text = sent.text.strip()
            
            # Filter out empty or too-short sentences
            if not sent_text or len(sent_text) < self.min_length:
                continue
            
            # Create chunk dictionary
            chunk = {
                'doc_id': doc_id,
                'sent_id': sent_id,
                'text': sent_text,
                'char_start': sent.start_char,
                'char_end': sent.end_char,
                'source': 'wikipedia',
                'version': 'wiki_sent_v1'
            }
            
            chunks.append(chunk)
            sent_id += 1
        
        return chunks
    
    def chunk_text(self, text: str, doc_id: str = 'unknown') -> List[Dict[str, any]]:
        """
        Chunk raw text into sentence-level fragments.
        
        Convenience method for chunking text without full article metadata.
        
        Args:
            text: Raw text to chunk
            doc_id: Document identifier (default: 'unknown')
        
        Returns:
            List of chunk dictionaries
        
        Example:
            >>> chunks = chunker.chunk_text("Hello world. This is a test.", doc_id="test_123")
            >>> len(chunks)
            2
        """
        article = {
            'doc_id': doc_id,
            'title': doc_id,
            'text': text
        }
        return self.chunk_article(article)
