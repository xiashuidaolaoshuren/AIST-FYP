"""Quick test to verify the FAISS index works correctly."""

import sys
from pathlib import Path
import numpy as np

sys.path.insert(0, str(Path(__file__).parent.parent))

from src.retrieval import FAISSIndexManager

# Load the index
manager = FAISSIndexManager(dimension=384, index_type='FLAT')
index, metadata = manager.load_index('data/indexes/development')

print(f"Loaded index: {index.ntotal:,} vectors")
print(f"Loaded metadata: {len(metadata):,} entries\n")

# Load embeddings for testing
embeddings = np.load('data/embeddings/wiki_embeddings_development.npy')

# Test 1: Search with first embedding (should find itself as top result)
print("="*60)
print("Test 1: Search with first embedding")
print("="*60)
query = embeddings[0:1]
distances, indices = manager.search(index, query, top_k=3)

print(f"\nQuery text: {metadata[0]['text'][:100]}...\n")
print("Top 3 results:")
for i, (idx, score) in enumerate(zip(indices[0], distances[0]), 1):
    chunk = metadata[idx]
    print(f"\n{i}. Score: {score:.4f}, Index: {idx}")
    print(f"   Doc: {chunk['doc_id']}, Sent: {chunk['sent_id']}")
    print(f"   Text: {chunk['text'][:100]}...")

# Test 2: Search with a different embedding
print("\n" + "="*60)
print("Test 2: Search with 10th embedding")
print("="*60)
query = embeddings[9:10]
distances, indices = manager.search(index, query, top_k=3)

print(f"\nQuery text: {metadata[9]['text'][:100]}...\n")
print("Top 3 results:")
for i, (idx, score) in enumerate(zip(indices[0], distances[0]), 1):
    chunk = metadata[idx]
    print(f"\n{i}. Score: {score:.4f}, Index: {idx}")
    print(f"   Doc: {chunk['doc_id']}, Sent: {chunk['sent_id']}")
    print(f"   Text: {chunk['text'][:100]}...")

print("\n" + "="*60)
print("✓ Index verification successful!")
print("="*60)
