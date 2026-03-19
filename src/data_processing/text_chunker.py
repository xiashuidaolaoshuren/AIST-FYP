"""
Text chunker for splitting articles into sentence-level chunks.

This module uses spaCy to segment text into sentences and create
chunk dictionaries with metadata for retrieval and indexing.
"""

import spacy
from typing import Any, Dict, List, Optional
from src.utils.logger import setup_logger


def _format_yes_no(value: Any) -> str:
    """Format common boolean-like values into readable yes/no text."""
    if value is True:
        return "Yes"
    if value is False:
        return "No"
    if value is None:
        return "Unknown"
    return str(value)


def chunk_data2txt(source_info: Dict[str, Any]) -> List[str]:
    """
    Convert a RAGTruth Data2txt source_info dict into natural-language contexts.

    The output is a list of context strings that can be sentence-split and indexed
    for retrieval. This avoids indexing raw JSON fragments as evidence.
    """
    if not isinstance(source_info, dict):
        return [str(source_info)]

    contexts: List[str] = []

    name = source_info.get('name')
    address = source_info.get('address')
    city = source_info.get('city')
    state = source_info.get('state')
    categories = source_info.get('categories')
    stars = source_info.get('business_stars')

    parts: List[str] = []
    if name:
        parts.append(str(name))
    if categories:
        parts.append(f"is listed under {categories}")

    location_parts = [p for p in [address, city, state] if p]
    if location_parts:
        if parts:
            parts.append(f"located at {', '.join(str(p) for p in location_parts)}")
        else:
            parts.append(f"Located at {', '.join(str(p) for p in location_parts)}")

    if stars is not None:
        parts.append(f"with a business rating of {stars} stars")

    if parts:
        contexts.append(" ".join(parts) + ".")

    hours = source_info.get('hours')
    if isinstance(hours, dict) and hours:
        hour_parts = []
        for day, value in hours.items():
            if value is None:
                continue
            value_text = str(value).replace('-', ' to ')
            hour_parts.append(f"{day}: {value_text}")
        if hour_parts:
            contexts.append("Operating hours are " + "; ".join(hour_parts) + ".")

    attributes = source_info.get('attributes')
    if isinstance(attributes, dict) and attributes:
        attr_lines: List[str] = []

        reservations = attributes.get('RestaurantsReservations')
        if reservations is not None:
            attr_lines.append(f"Reservations: {_format_yes_no(reservations)}")

        outdoor = attributes.get('OutdoorSeating')
        if outdoor is not None:
            attr_lines.append(f"Outdoor seating: {_format_yes_no(outdoor)}")

        wifi = attributes.get('WiFi')
        if wifi is not None:
            attr_lines.append(f"WiFi: {wifi}")

        takeout = attributes.get('RestaurantsTakeOut')
        if takeout is not None:
            attr_lines.append(f"Takeout: {_format_yes_no(takeout)}")

        groups = attributes.get('RestaurantsGoodForGroups')
        if groups is not None:
            attr_lines.append(f"Good for groups: {_format_yes_no(groups)}")

        parking = attributes.get('BusinessParking')
        if isinstance(parking, dict):
            enabled = [k for k, v in parking.items() if v is True]
            if enabled:
                attr_lines.append("Parking options include " + ", ".join(enabled))
            elif any(v is False for v in parking.values()):
                attr_lines.append("No parking options are marked as available")

        ambience = attributes.get('Ambience')
        if isinstance(ambience, dict):
            ambience_flags = [k for k, v in ambience.items() if v is True]
            if ambience_flags:
                attr_lines.append("Ambience includes " + ", ".join(ambience_flags))

        if attr_lines:
            contexts.append("Business attributes: " + ". ".join(attr_lines) + ".")

    review_info = source_info.get('review_info')
    if isinstance(review_info, list):
        for review in review_info:
            if not isinstance(review, dict):
                continue
            review_text = str(review.get('review_text', '')).strip()
            if not review_text:
                continue

            prefix_bits: List[str] = []
            if review.get('review_stars') is not None:
                prefix_bits.append(f"Review rating: {review.get('review_stars')} stars")
            if review.get('review_date'):
                prefix_bits.append(f"Review date: {review.get('review_date')}")

            if prefix_bits:
                contexts.append(". ".join(prefix_bits) + ". " + review_text)
            else:
                contexts.append(review_text)

    # Safe fallback to avoid empty contexts.
    if not contexts:
        contexts.append(str(source_info))

    return contexts


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
        
        except OSError:
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
        source = article.get('source', 'wikipedia')
        version = article.get('version', 'wiki_sent_v1')
        
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
                'source': source,
                'version': version
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
