"""
Demo script comparing Dense, BM25, and Hybrid retrieval modes.

This script demonstrates the differences between dense (embedding-based),
sparse (BM25), and hybrid retrieval by running the same queries through
all three modes and displaying their results side-by-side.

Usage:
    python scripts/demo_hybrid_retrieval.py --strategy devlopment
    python scripts/demo_hybrid_retrieval.py --strategy devlopment --query "What is machine learning?"
"""

import argparse
import sys
import time
from pathlib import Path
from typing import List

# Add project root to path
project_root = Path(__file__).parent.parent
sys.path.insert(0, str(project_root))

from src.retrieval.dense_retriever import DenseRetriever
from src.retrieval.bm25_retriever import BM25Retriever
from src.retrieval.hybrid_retriever import HybridRetriever
from src.utils.config import Config
from src.utils.data_structures import EvidenceChunk
from src.utils.logger import setup_logger


def print_results(query: str, chunks: List[EvidenceChunk], mode: str, elapsed_time: float):
    """Print retrieval results in a formatted way."""
    print(f"\n{'=' * 80}")
    print(f"{mode.upper()} RETRIEVAL")
    print(f"{'=' * 80}")
    print(f"Query: {query}")
    print(f"Retrieved: {len(chunks)} chunks in {elapsed_time:.3f}s")
    print(f"{'-' * 80}")
    
    for i, chunk in enumerate(chunks, start=1):
        print(f"\n[{i}] Rank: {chunk.rank}")
        print(f"    Doc ID: {chunk.doc_id}, Sent ID: {chunk.sent_id}")
        
        # Show available scores
        scores = []
        if chunk.score_dense > 0:
            scores.append(f"Dense: {chunk.score_dense:.4f}")
        if chunk.score_bm25 and chunk.score_bm25 > 0:
            scores.append(f"BM25: {chunk.score_bm25:.4f}")
        if chunk.score_hybrid and chunk.score_hybrid > 0:
            scores.append(f"Hybrid: {chunk.score_hybrid:.4f}")
        
        print(f"    Scores: {', '.join(scores)}")
        
        # Truncate text for display
        text = chunk.text
        if len(text) > 200:
            text = text[:197] + "..."
        print(f"    Text: {text}")


def compare_retrievers(
    query: str,
    strategy: str,
    top_k: int = 5,
    config_path: str = 'config.yaml'
):
    """
    Compare dense, BM25, and hybrid retrievers on the same query.
    
    Args:
        query: Query string
        strategy: Dataset strategy ('dev', 'validation', or 'production')
        top_k: Number of results to retrieve
        config_path: Path to configuration file
    """
    logger = setup_logger('demo_hybrid_retrieval')
    config = Config(config_path)
    
    # Define paths
    index_path = Path('data/indexes') / strategy / 'faiss.index'
    metadata_path = Path('data/indexes') / strategy / 'metadata.pkl'
    corpus_path = Path('data/processed') / f'wiki_chunks_{strategy}.jsonl'
    bm25_index_path = Path('data/indexes') / strategy / 'bm25_index.pkl'
    
    # Check if required files exist
    if not index_path.exists():
        logger.error(f"FAISS index not found: {index_path}")
        logger.error("Run: python scripts/build_faiss_index.py --strategy {strategy}")
        return
    
    if not corpus_path.exists():
        logger.error(f"Corpus not found: {corpus_path}")
        logger.error("Run preprocessing first")
        return
    
    print(f"\n{'#' * 80}")
    print(f"HYBRID RETRIEVAL COMPARISON DEMO")
    print(f"{'#' * 80}")
    print(f"Strategy: {strategy}")
    print(f"Top-k: {top_k}")
    
    # 1. Dense Retrieval
    logger.info("Initializing Dense Retriever...")
    dense_retriever = DenseRetriever(
        index_path=str(index_path),
        metadata_path=str(metadata_path),
        encoder_model=config.models.sentence_transformer,
        device=config.processing.device
    )
    
    start_time = time.time()
    dense_results = dense_retriever.retrieve(query, top_k=top_k)
    dense_time = time.time() - start_time
    
    print_results(query, dense_results, "Dense (Embeddings)", dense_time)
    
    # 2. BM25 Retrieval
    logger.info("Initializing BM25 Retriever...")
    bm25_retriever = BM25Retriever(
        corpus_path=str(corpus_path),
        index_path=str(bm25_index_path),
        k1=config.retrieval.bm25.get('k1', 1.5),
        b=config.retrieval.bm25.get('b', 0.75)
    )
    
    start_time = time.time()
    bm25_results = bm25_retriever.retrieve(query, top_k=top_k)
    bm25_time = time.time() - start_time
    
    print_results(query, bm25_results, "BM25 (Sparse Lexical)", bm25_time)
    
    # 3. Hybrid Retrieval (Linear Fusion)
    logger.info("Initializing Hybrid Retriever (Linear)...")
    hybrid_linear = HybridRetriever(
        dense_retriever=dense_retriever,
        bm25_retriever=bm25_retriever,
        alpha=config.retrieval.hybrid.get('alpha', 0.5),
        fusion_method='linear'
    )
    
    start_time = time.time()
    hybrid_linear_results = hybrid_linear.retrieve(query, top_k=top_k)
    hybrid_linear_time = time.time() - start_time
    
    print_results(query, hybrid_linear_results, "Hybrid (Linear Fusion)", hybrid_linear_time)
    
    # 4. Hybrid Retrieval (RRF)
    logger.info("Initializing Hybrid Retriever (RRF)...")
    hybrid_rrf = HybridRetriever(
        dense_retriever=dense_retriever,
        bm25_retriever=bm25_retriever,
        alpha=0.5,  # Not used in RRF, but required
        fusion_method='rrf',
        rrf_k=config.retrieval.hybrid.get('rrf_k', 60)
    )
    
    start_time = time.time()
    hybrid_rrf_results = hybrid_rrf.retrieve(query, top_k=top_k)
    hybrid_rrf_time = time.time() - start_time
    
    print_results(query, hybrid_rrf_results, "Hybrid (RRF)", hybrid_rrf_time)
    
    # Summary
    print(f"\n{'=' * 80}")
    print(f"TIMING SUMMARY")
    print(f"{'=' * 80}")
    print(f"Dense:         {dense_time:.3f}s")
    print(f"BM25:          {bm25_time:.3f}s")
    print(f"Hybrid Linear: {hybrid_linear_time:.3f}s")
    print(f"Hybrid RRF:    {hybrid_rrf_time:.3f}s")
    print()


def main():
    parser = argparse.ArgumentParser(
        description='Compare Dense, BM25, and Hybrid retrieval modes'
    )
    parser.add_argument(
        '--strategy',
        type=str,
        default='development',
        choices=['development', 'validation', 'production'],
        help='Dataset strategy to use (default: development)'
    )
    parser.add_argument(
        '--query',
        type=str,
        help='Custom query to test. If not provided, uses default queries.'
    )
    parser.add_argument(
        '--top-k',
        type=int,
        default=5,
        help='Number of results to retrieve (default: 5)'
    )
    parser.add_argument(
        '--config',
        type=str,
        default='config.yaml',
        help='Path to configuration file (default: config.yaml)'
    )
    
    args = parser.parse_args()
    
    # Define test queries if not provided
    if args.query:
        queries = [args.query]
    else:
        queries = [
            "What is the capital of France?",
            "Who invented the telephone?",
            "When was the United Nations founded?",
            "What is machine learning?"
        ]
    
    # Run comparison for each query
    for i, query in enumerate(queries):
        if i > 0:
            print("\n" * 2)
        compare_retrievers(
            query=query,
            strategy=args.strategy,
            top_k=args.top_k,
            config_path=args.config
        )
    
    print(f"\n{'#' * 80}")
    print("Demo complete!")
    print(f"{'#' * 80}\n")


if __name__ == '__main__':
    main()
