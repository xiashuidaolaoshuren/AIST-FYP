"""
BM25 retriever for sparse lexical search.

This module provides the BM25Retriever class that uses the BM25 ranking
function to retrieve documents based on exact term matching. It uses spaCy
for tokenization and supports index caching for faster loading.
"""

import json
import pickle
import numpy as np
from pathlib import Path
from typing import List, Optional
from rank_bm25 import BM25Okapi
from tqdm import tqdm
import spacy

from src.utils.data_structures import EvidenceChunk
from src.utils.nlp_utils import get_spacy_model
from src.utils.logger import setup_logger


class BM25Retriever:
    """
    BM25 retriever for sparse lexical retrieval.
    
    Uses the BM25 ranking function (BM25Okapi variant) to score and rank
    documents based on exact term matching. Tokenizes queries and documents
    using spaCy (without parser/NER for speed), and supports caching the
    BM25 index to disk for faster loading.
    
    Attributes:
        bm25: BM25Okapi instance for scoring
        chunks: List of chunk metadata dictionaries
        nlp: spaCy language model for tokenization
        k1: BM25 term frequency saturation parameter
        b: BM25 length normalization parameter
        logger: Logger instance
    
    Example:
        >>> retriever = BM25Retriever(
        ...     corpus_path='data/processed/dev/wiki_chunks_dev.jsonl',
        ...     index_path='data/indexes/dev/bm25_index.pkl',
        ...     k1=1.5,
        ...     b=0.75
        ... )
        >>> results = retriever.retrieve("What is machine learning?", top_k=5)
        >>> print(f"Top result: {results[0].text}")
    """
    
    def __init__(
        self,
        corpus_path: str,
        index_path: Optional[str] = None,
        k1: float = 1.5,
        b: float = 0.75,
        spacy_model: str = 'en_core_web_sm'
    ):
        """
        Initialize BM25 retriever.
        
        Args:
            corpus_path: Path to JSONL file containing processed chunks
            index_path: Optional path to cached BM25 index (pickled)
            k1: BM25 term frequency saturation parameter (default: 1.5)
            b: BM25 length normalization parameter (default: 0.75)
            spacy_model: spaCy model name for tokenization (default: 'en_core_web_sm')
        """
        self.logger = setup_logger(self.__class__.__name__)
        self.k1 = k1
        self.b = b
        self.corpus_path = Path(corpus_path)
        self.index_path = Path(index_path) if index_path else None
        
        # Load spaCy tokenizer (exclude parser and NER for speed)
        self.logger.info(f"Loading spaCy model: {spacy_model}")
        self.nlp = get_spacy_model(spacy_model)
        
        # Try to load cached index first
        if self.index_path and self.index_path.exists():
            self.logger.info(f"Loading cached BM25 index from {self.index_path}")
            self._load_index()
        else:
            self.logger.info(f"Building BM25 index from {self.corpus_path}")
            self._build_index()
            
            # Cache the index if path is provided
            if self.index_path:
                self.logger.info(f"Caching BM25 index to {self.index_path}")
                self._save_index()
        
        self.logger.info(f"BM25 retriever initialized with {len(self.chunks)} chunks")
    
    def _tokenize(self, text: str) -> List[str]:
        """
        Tokenize text using spaCy.
        
        Args:
            text: Input text to tokenize
            
        Returns:
            List of lowercase tokens
        """
        doc = self.nlp(text)
        return [token.text.lower() for token in doc if not token.is_space]
    
    def _build_index(self):
        """Build BM25 index from corpus file."""
        self.logger.info("Loading and tokenizing corpus...")
        
        # Load chunks from JSONL
        self.chunks = []
        with open(self.corpus_path, 'r', encoding='utf-8') as f:
            for line in f:
                self.chunks.append(json.loads(line))
        
        self.logger.info(f"Loaded {len(self.chunks)} chunks")
        
        # Tokenize corpus with progress bar
        self.logger.info("Tokenizing corpus with spaCy...")
        tokenized_corpus = []
        for chunk in tqdm(self.chunks, desc="Tokenizing", unit="chunk"):
            tokenized_corpus.append(self._tokenize(chunk['text']))
        
        # Build BM25 index
        self.logger.info("Building BM25 index...")
        self.bm25 = BM25Okapi(tokenized_corpus, k1=self.k1, b=self.b)
        self.logger.info("BM25 index built successfully")
    
    def _save_index(self):
        """Save BM25 index and chunks to disk."""
        self.index_path.parent.mkdir(parents=True, exist_ok=True)
        
        index_data = {
            'bm25': self.bm25,
            'chunks': self.chunks,
            'k1': self.k1,
            'b': self.b
        }
        
        with open(self.index_path, 'wb') as f:
            pickle.dump(index_data, f)
        
        self.logger.info(f"Saved BM25 index to {self.index_path}")
    
    def _load_index(self):
        """Load BM25 index and chunks from disk."""
        with open(self.index_path, 'rb') as f:
            index_data = pickle.load(f)
        
        self.bm25 = index_data['bm25']
        self.chunks = index_data['chunks']
        
        # Verify parameters match (warn if different)
        if index_data.get('k1') != self.k1:
            self.logger.warning(
                f"Cached index k1={index_data.get('k1')} differs from requested k1={self.k1}"
            )
        if index_data.get('b') != self.b:
            self.logger.warning(
                f"Cached index b={index_data.get('b')} differs from requested b={self.b}"
            )
        
        self.logger.info(f"Loaded BM25 index with {len(self.chunks)} chunks")
    
    def retrieve(self, query: str, top_k: int = 5) -> List[EvidenceChunk]:
        """
        Retrieve top-k most relevant chunks for a query using BM25.
        
        Args:
            query: Query string
            top_k: Number of top results to return
            
        Returns:
            List of EvidenceChunk objects ranked by BM25 score (highest first)
        """
        # Tokenize query
        tokenized_query = self._tokenize(query)
        
        if not tokenized_query:
            self.logger.warning(f"Empty tokenized query for: {query}")
            return []
        
        # Get BM25 scores
        scores = self.bm25.get_scores(tokenized_query)
        
        # Get top-k indices (argsort returns ascending, so reverse)
        top_k = min(top_k, len(scores))
        top_indices = np.argsort(scores)[-top_k:][::-1]
        
        # Build EvidenceChunk objects
        results = []
        for rank, idx in enumerate(top_indices, start=1):
            chunk_data = self.chunks[idx]
            chunk = EvidenceChunk(
                doc_id=chunk_data['doc_id'],
                sent_id=chunk_data['sent_id'],
                text=chunk_data['text'],
                char_start=chunk_data['char_start'],
                char_end=chunk_data['char_end'],
                score_dense=0.0,  # Not used for BM25-only retrieval
                score_bm25=float(scores[idx]),
                rank=rank,
                source=chunk_data.get('source', 'wikipedia'),
                version=chunk_data.get('version', 'wiki_sent_v1')
            )
            results.append(chunk)
        
        self.logger.debug(
            f"Retrieved {len(results)} chunks for query: {query[:50]}..."
        )
        
        return results
    
    def batch_retrieve(
        self,
        queries: List[str],
        top_k: int = 5
    ) -> List[List[EvidenceChunk]]:
        """
        Retrieve top-k chunks for multiple queries.
        
        Args:
            queries: List of query strings
            top_k: Number of top results per query
            
        Returns:
            List of lists of EvidenceChunk objects, one list per query
        """
        self.logger.info(f"Batch retrieving for {len(queries)} queries")
        
        results = []
        for i, query in enumerate(queries):
            if (i + 1) % 100 == 0:
                self.logger.info(f"Processed {i + 1}/{len(queries)} queries")
            results.append(self.retrieve(query, top_k))
        
        return results
