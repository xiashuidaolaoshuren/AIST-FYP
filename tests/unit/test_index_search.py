"""Quick test to verify the FAISS index works correctly."""

import os
import sys
from pathlib import Path
import numpy as np
import pytest

sys.path.insert(0, str(Path(__file__).parent.parent.parent))

from src.retrieval import FAISSIndexManager


def _get_strategy() -> str:
    return os.getenv("DATA_STRATEGY", "validation")


def _resolve_paths(strategy: str) -> tuple[str, str]:
    index_dir = f"data/indexes/{strategy}"
    embeddings_path = f"data/embeddings/wiki_embeddings_{strategy}.npy"
    return index_dir, embeddings_path


def _skip_if_missing(index_dir: str, embeddings_path: str) -> None:
    index_file = Path(index_dir) / "faiss.index"
    if not index_file.exists() or not Path(embeddings_path).exists():
        pytest.skip(
            f"FAISS index or embeddings not found for strategy '{_get_strategy()}'. "
            f"Expected: {index_file} and {embeddings_path}"
        )


def test_index_search_with_first_embedding():
    """Search with first embedding should return valid results."""
    strategy = _get_strategy()
    index_dir, embeddings_path = _resolve_paths(strategy)
    _skip_if_missing(index_dir, embeddings_path)

    manager = FAISSIndexManager(dimension=384, index_type='FLAT')
    index, metadata = manager.load_index(index_dir)
    embeddings = np.load(embeddings_path)

    query = embeddings[0:1]
    distances, indices = manager.search(index, query, top_k=3)

    assert distances.shape == (1, 3)
    assert indices.shape == (1, 3)
    assert 0 <= indices[0][0] < len(metadata)


def test_index_search_with_tenth_embedding():
    """Search with 10th embedding should return valid results."""
    strategy = _get_strategy()
    index_dir, embeddings_path = _resolve_paths(strategy)
    _skip_if_missing(index_dir, embeddings_path)

    manager = FAISSIndexManager(dimension=384, index_type='FLAT')
    index, metadata = manager.load_index(index_dir)
    embeddings = np.load(embeddings_path)

    query = embeddings[9:10]
    distances, indices = manager.search(index, query, top_k=3)

    assert distances.shape == (1, 3)
    assert indices.shape == (1, 3)
    assert 0 <= indices[0][0] < len(metadata)
