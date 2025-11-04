"""
Demo script for Dense Retriever.

Demonstrates the DenseRetriever's ability to find relevant evidence chunks
for various queries using the development FAISS index.
"""

import sys
from pathlib import Path

sys.path.insert(0, str(Path(__file__).parent.parent))

from src.retrieval import DenseRetriever


def main():
    # Initialize retriever
    print("="*70)
    print("Dense Retriever Demo")
    print("="*70)
    
    project_root = Path(__file__).parent.parent
    index_dir = project_root / 'data' / 'indexes' / 'development'
    
    print("\nInitializing DenseRetriever...")
    retriever = DenseRetriever(
        index_path=str(index_dir / 'faiss.index'),
        metadata_path=str(index_dir / 'metadata.pkl'),
        encoder_model='sentence-transformers/all-MiniLM-L6-v2',
        device='cuda'
    )
    print(f"✓ Loaded index with {retriever.index.ntotal} vectors")
    
    # Test queries
    queries = [
        "What is artificial intelligence?",
        "How do machines learn from data?",
        "What is deep learning?",
        "What is natural language processing?"
    ]
    
    # Single query retrieval
    print("\n" + "="*70)
    print("Single Query Retrieval")
    print("="*70)
    
    for query in queries[:2]:
        print(f"\nQuery: \"{query}\"")
        print("-" * 70)
        
        results = retriever.retrieve(query, top_k=3)
        
        for chunk in results:
            print(f"\nRank {chunk.rank} | Score: {chunk.score_dense:.4f}")
            print(f"Doc: {chunk.doc_id} | Sent: {chunk.sent_id}")
            print(f"Text: {chunk.text[:150]}...")
    
    # Batch retrieval
    print("\n" + "="*70)
    print("Batch Query Retrieval")
    print("="*70)
    
    all_results = retriever.batch_retrieve(queries, top_k=3)
    
    for query, results in zip(queries, all_results):
        print(f"\nQuery: \"{query}\"")
        print(f"Retrieved {len(results)} results")
        if results:
            top_result = results[0]
            print(f"Top result (score: {top_result.score_dense:.4f}):")
            print(f"  {top_result.text[:100]}...")
    
    # Comparison of different queries
    print("\n" + "="*70)
    print("Query Comparison")
    print("="*70)
    
    compare_queries = [
        "machine learning algorithms",
        "deep neural networks"
    ]
    
    for query in compare_queries:
        results = retriever.retrieve(query, top_k=5)
        print(f"\nQuery: \"{query}\"")
        print(f"Top 5 scores: {[f'{r.score_dense:.3f}' for r in results]}")
    
    print("\n" + "="*70)
    print("✓ Demo completed successfully!")
    print("="*70)


if __name__ == '__main__':
    main()
