"""
EvidenceSentenceRetriever — sentence-level evidence retrieval for gold-context
evaluation modes (ragtruth_eval, gold_context_generation, CiteEval oracle).

Motivation
----------
NLI models are trained on (premise, hypothesis) pairs where the premise is a
single sentence or short passage (~1-3 sentences).  When passed an entire
gold context paragraph (often 200-500 tokens), the model silently truncates
to 512 tokens, which can cause the most relevant sentence to be cut off and
allows noisy "entailment" from unrelated parts of the text.

This retriever splits gold contexts into sentences and, for each claim,
retrieves only the top-k most semantically similar sentences.  This gives
the NLI and grounded detectors a clean, targeted premise without truncation.

Two usage patterns
------------------
a) Pre-built file-based index (fast; suited for repeated RAGTruth evaluation):

    retriever = EvidenceSentenceRetriever.from_index(
        index_dir="data/indexes/ragtruth_sentences/test",
        encoder_model="sentence-transformers/all-MiniLM-L6-v2",
    )
    chunks = retriever.retrieve(claim_text, sample_id="42", top_k=5)

b) On-the-fly in-memory index (suited for CiteEval oracle, per-row processing):

    retriever = EvidenceSentenceRetriever.from_encoder(
        encoder_model="sentence-transformers/all-MiniLM-L6-v2",
    )
    ctx_index = retriever.build_context_index_from_chunks(evidence_chunks)
    chunks = retriever.retrieve_from_index(claim_text, ctx_index, top_k=5)

Index files (produced by scripts/build_sentence_index.py)
----------------------------------------------------------
  {index_dir}/sentences.jsonl    — one JSON line per sentence
  {index_dir}/embeddings.npy     — float32 (N, dim), L2-normalised, row-aligned
  {index_dir}/sample_index.json  — {sample_id: [row_start, row_end]}
"""

from __future__ import annotations

import json
from dataclasses import dataclass
from pathlib import Path
from typing import List, Optional

import numpy as np
from sentence_transformers import SentenceTransformer

from src.utils.data_structures import EvidenceChunk
from src.utils.logger import setup_logger
from src.utils.nlp_utils import get_spacy_model


@dataclass
class ContextSentenceIndex:
    """In-memory sentence index for a single context (on-the-fly mode)."""

    sentences: List[dict]    # [{text, doc_id, sent_idx, passage_idx, char_start, char_end, ...}]
    embeddings: np.ndarray   # shape (n_sents, dim), L2-normalised float32


class EvidenceSentenceRetriever:
    """
    Retrieves the top-k most semantically similar sentences for a claim,
    scoped to the evidence passages for that sample.

    Supports two modes:
    - File-based pre-built index (fast, for RAGTruth repeated evaluation)
    - On-the-fly in-memory index (per-row, for CiteEval oracle evaluation)
    """

    _MIN_SENT_LENGTH = 10  # minimum characters for a valid sentence
    _NO_INDEX_ERROR = "No pre-built index loaded. Use from_index() or call _load_index()."

    def __init__(
        self,
        encoder_model: str,
        device: str = "cpu",
        *,
        index_dir: Optional[str] = None,
        spacy_model_name: str = "en_core_web_sm",
    ) -> None:
        self.logger = setup_logger(__name__)
        self._spacy_model_name = spacy_model_name

        self.logger.info(
            "Loading SentenceTransformer encoder: %s on %s", encoder_model, device
        )
        self._encoder = SentenceTransformer(encoder_model, device=device)

        # Pre-built index (optional; loaded when index_dir is provided)
        self._sentences: Optional[List[dict]] = None
        self._embeddings: Optional[np.ndarray] = None
        self._sample_index: Optional[dict] = None

        if index_dir is not None:
            self._load_index(index_dir)

    # ------------------------------------------------------------------
    # Constructors
    # ------------------------------------------------------------------

    @classmethod
    def from_index(
        cls,
        index_dir: str,
        encoder_model: str,
        device: str = "cpu",
        spacy_model_name: str = "en_core_web_sm",
    ) -> "EvidenceSentenceRetriever":
        """Load encoder + pre-built sentence index from disk (RAGTruth mode)."""
        return cls(
            encoder_model=encoder_model,
            device=device,
            index_dir=index_dir,
            spacy_model_name=spacy_model_name,
        )

    @classmethod
    def from_encoder(
        cls,
        encoder_model: str,
        device: str = "cpu",
        spacy_model_name: str = "en_core_web_sm",
    ) -> "EvidenceSentenceRetriever":
        """Load encoder only for on-the-fly retrieval (CiteEval oracle mode)."""
        return cls(
            encoder_model=encoder_model,
            device=device,
            index_dir=None,
            spacy_model_name=spacy_model_name,
        )

    # ------------------------------------------------------------------
    # Pre-built index loading
    # ------------------------------------------------------------------

    def _load_index(self, index_dir: str) -> None:
        base = Path(index_dir)
        sentences_path = base / "sentences.jsonl"
        embeddings_path = base / "embeddings.npy"
        sample_index_path = base / "sample_index.json"

        for p in (sentences_path, embeddings_path, sample_index_path):
            if not p.exists():
                raise FileNotFoundError(f"Sentence index file not found: {p}")

        self._sentences = []
        with sentences_path.open(encoding="utf-8") as f:
            for line in f:
                line = line.strip()
                if line:
                    self._sentences.append(json.loads(line))

        # Memory-map avoids loading the full matrix into RAM for large corpora
        self._embeddings = np.load(str(embeddings_path), mmap_mode="r")

        if len(self._sentences) != self._embeddings.shape[0]:
            raise ValueError(
                f"sentences.jsonl has {len(self._sentences)} entries but "
                f"embeddings.npy has {self._embeddings.shape[0]} rows"
            )

        with sample_index_path.open(encoding="utf-8") as f:
            self._sample_index = json.load(f)

        self.logger.info(
            "Sentence index loaded: %d sentences, %d samples from %s",
            len(self._sentences),
            len(self._sample_index),
            index_dir,
        )

    # ------------------------------------------------------------------
    # Pre-built index retrieval
    # ------------------------------------------------------------------

    def retrieve(
        self,
        query_text: str,
        sample_id: str,
        top_k: int = 5,
    ) -> List[EvidenceChunk]:
        """
        Retrieve top-k sentence EvidenceChunks for *query_text* scoped to
        *sample_id* using the pre-built file index.

        Returns an empty list if sample_id is not found in the index.
        """
        if (
            self._sample_index is None
            or self._sentences is None
            or self._embeddings is None
        ):
            raise RuntimeError(self._NO_INDEX_ERROR)

        key = str(sample_id)
        if key not in self._sample_index:
            self.logger.warning(
                "sample_id '%s' not in sentence index; returning empty.", key
            )
            return []

        row_start, row_end = self._sample_index[key]
        if row_end <= row_start:
            return []

        sample_embeddings = self._embeddings[row_start:row_end].astype(np.float32)
        return self._cosine_topk(
            query_text,
            sample_embeddings,
            self._sentences[row_start:row_end],
            top_k,
        )

    def retrieve_batch(
        self,
        query_texts: List[str],
        sample_id: str,
        top_k: int = 5,
    ) -> List[List[EvidenceChunk]]:
        """
        Retrieve top-k sentence EvidenceChunks for each query in *query_texts*
        scoped to *sample_id* using the pre-built file index.

        Returns one list per query text. If sample_id is missing, each query
        receives an empty evidence list.
        """
        if not query_texts:
            return []

        if (
            self._sample_index is None
            or self._sentences is None
            or self._embeddings is None
        ):
            raise RuntimeError(self._NO_INDEX_ERROR)

        key = str(sample_id)
        if key not in self._sample_index:
            self.logger.warning(
                "sample_id '%s' not in sentence index; returning empty for %d claims.",
                key,
                len(query_texts),
            )
            return [[] for _ in query_texts]

        row_start, row_end = self._sample_index[key]
        if row_end <= row_start:
            return [[] for _ in query_texts]

        sample_embeddings = self._embeddings[row_start:row_end].astype(np.float32)
        return self._cosine_topk_batch(
            query_texts,
            sample_embeddings,
            self._sentences[row_start:row_end],
            top_k,
        )

    def retrieve_all(
        self,
        sample_id: str,
        max_sentences: Optional[int] = None,
    ) -> List[EvidenceChunk]:
        """
        Return all indexed sentences for *sample_id* as EvidenceChunks.

        This supports evaluator-side all-sentence-pairs NLI experiments where
        each claim is verified against the full sentence pool instead of top-k.
        """
        if (
            self._sample_index is None
            or self._sentences is None
            or self._embeddings is None
        ):
            raise RuntimeError(self._NO_INDEX_ERROR)

        key = str(sample_id)
        if key not in self._sample_index:
            self.logger.warning(
                "sample_id '%s' not in sentence index; returning empty.", key
            )
            return []

        row_start, row_end = self._sample_index[key]
        if row_end <= row_start:
            return []

        sentences = self._sentences[row_start:row_end]
        if max_sentences is not None and max_sentences > 0:
            sentences = sentences[:max_sentences]

        return self._rows_to_evidence_chunks(sentences)

    def retrieve_all_ranked(
        self,
        query_text: str,
        sample_id: str,
        max_sentences: Optional[int] = None,
    ) -> List[EvidenceChunk]:
        """
        Return all indexed sentences for *sample_id* ranked by semantic
        similarity to *query_text*.

        Unlike retrieve_all(), this preserves meaningful dense scores and rank
        order so downstream contradiction suppression can reason about evidence
        quality while still seeing the full sentence pool.
        """
        if (
            self._sample_index is None
            or self._sentences is None
            or self._embeddings is None
        ):
            raise RuntimeError(self._NO_INDEX_ERROR)

        key = str(sample_id)
        if key not in self._sample_index:
            self.logger.warning(
                "sample_id '%s' not in sentence index; returning empty.", key
            )
            return []

        row_start, row_end = self._sample_index[key]
        if row_end <= row_start:
            return []

        sample_embeddings = self._embeddings[row_start:row_end].astype(np.float32)
        return self._cosine_ranked_all(
            query_text,
            sample_embeddings,
            self._sentences[row_start:row_end],
            max_sentences=max_sentences,
        )

    def retrieve_all_ranked_batch(
        self,
        query_texts: List[str],
        sample_id: str,
        max_sentences: Optional[int] = None,
    ) -> List[List[EvidenceChunk]]:
        """
        Batched variant of retrieve_all_ranked() for multiple claims.

        Returns one ranked full-pool evidence list per query.
        """
        if not query_texts:
            return []

        if (
            self._sample_index is None
            or self._sentences is None
            or self._embeddings is None
        ):
            raise RuntimeError(self._NO_INDEX_ERROR)

        key = str(sample_id)
        if key not in self._sample_index:
            self.logger.warning(
                "sample_id '%s' not in sentence index; returning empty for %d claims.",
                key,
                len(query_texts),
            )
            return [[] for _ in query_texts]

        row_start, row_end = self._sample_index[key]
        if row_end <= row_start:
            return [[] for _ in query_texts]

        sample_embeddings = self._embeddings[row_start:row_end].astype(np.float32)
        return self._cosine_ranked_all_batch(
            query_texts,
            sample_embeddings,
            self._sentences[row_start:row_end],
            max_sentences=max_sentences,
        )

    # ------------------------------------------------------------------
    # On-the-fly index building and retrieval
    # ------------------------------------------------------------------

    def build_context_index_from_chunks(
        self,
        evidence_chunks: List[EvidenceChunk],
    ) -> ContextSentenceIndex:
        """
        Split EvidenceChunks into sentences, preserving the original doc_id
        and source from each chunk. Returns an in-memory ContextSentenceIndex.

        Use this when you already have EvidenceChunk objects (e.g., from
        _build_evidence_from_contexts or _build_oracle_evidence_chunks).
        """
        sents = self._split_chunks_to_sentences(evidence_chunks)
        if not sents:
            return ContextSentenceIndex(
                sentences=[], embeddings=np.empty((0, 0), dtype=np.float32)
            )

        texts = [s["text"] for s in sents]
        embeddings = self._encoder.encode(
            texts,
            normalize_embeddings=True,
            convert_to_numpy=True,
            show_progress_bar=False,
        ).astype(np.float32)
        return ContextSentenceIndex(sentences=sents, embeddings=embeddings)

    def build_context_index(
        self,
        contexts: List[str],
        prefix: str = "ctx",
    ) -> ContextSentenceIndex:
        """
        Split raw context strings into sentences, encode them, and return an
        in-memory ContextSentenceIndex. Sentence doc_ids take the form
        "{prefix}_p{passage_idx}".
        """
        sents = self._split_contexts_to_sentences(contexts, prefix=prefix)
        if not sents:
            return ContextSentenceIndex(
                sentences=[], embeddings=np.empty((0, 0), dtype=np.float32)
            )

        texts = [s["text"] for s in sents]
        embeddings = self._encoder.encode(
            texts,
            normalize_embeddings=True,
            convert_to_numpy=True,
            show_progress_bar=False,
        ).astype(np.float32)
        return ContextSentenceIndex(sentences=sents, embeddings=embeddings)

    def retrieve_from_index(
        self,
        query_text: str,
        ctx_index: ContextSentenceIndex,
        top_k: int = 5,
    ) -> List[EvidenceChunk]:
        """
        Retrieve top-k EvidenceChunks from an in-memory ContextSentenceIndex.
        Used in on-the-fly mode (e.g., per-row CiteEval oracle evaluation).
        Returns an empty list if the index has no sentences.
        """
        if ctx_index.embeddings.shape[0] == 0:
            return []
        return self._cosine_topk(
            query_text, ctx_index.embeddings, ctx_index.sentences, top_k
        )

    def retrieve_from_index_batch(
        self,
        query_texts: List[str],
        ctx_index: ContextSentenceIndex,
        top_k: int = 5,
    ) -> List[List[EvidenceChunk]]:
        """
        Retrieve top-k EvidenceChunks for each query from an in-memory
        ContextSentenceIndex. Returns one list per query.
        """
        if not query_texts:
            return []
        if ctx_index.embeddings.shape[0] == 0:
            return [[] for _ in query_texts]
        return self._cosine_topk_batch(
            query_texts, ctx_index.embeddings, ctx_index.sentences, top_k
        )

    def retrieve_from_index_all(
        self,
        ctx_index: ContextSentenceIndex,
        max_sentences: Optional[int] = None,
    ) -> List[EvidenceChunk]:
        """
        Return all sentences from an in-memory context index as EvidenceChunks.

        This is the on-the-fly counterpart of retrieve_all() used in
        gold-context generation flows.
        """
        if ctx_index.embeddings.shape[0] == 0:
            return []

        sentences = ctx_index.sentences
        if max_sentences is not None and max_sentences > 0:
            sentences = sentences[:max_sentences]

        return self._rows_to_evidence_chunks(sentences)

    def retrieve_from_index_all_ranked(
        self,
        query_text: str,
        ctx_index: ContextSentenceIndex,
        max_sentences: Optional[int] = None,
    ) -> List[EvidenceChunk]:
        """
        Return all in-memory index sentences ranked by semantic similarity to
        *query_text*.
        """
        if ctx_index.embeddings.shape[0] == 0:
            return []
        return self._cosine_ranked_all(
            query_text,
            ctx_index.embeddings,
            ctx_index.sentences,
            max_sentences=max_sentences,
        )

    def retrieve_from_index_all_ranked_batch(
        self,
        query_texts: List[str],
        ctx_index: ContextSentenceIndex,
        max_sentences: Optional[int] = None,
    ) -> List[List[EvidenceChunk]]:
        """
        Batched variant of retrieve_from_index_all_ranked().
        """
        if not query_texts:
            return []
        if ctx_index.embeddings.shape[0] == 0:
            return [[] for _ in query_texts]
        return self._cosine_ranked_all_batch(
            query_texts,
            ctx_index.embeddings,
            ctx_index.sentences,
            max_sentences=max_sentences,
        )

    # ------------------------------------------------------------------
    # Internal helpers
    # ------------------------------------------------------------------

    def _cosine_topk(
        self,
        query_text: str,
        embeddings: np.ndarray,   # (n, dim), L2-normalised float32
        sentences: List[dict],
        top_k: int,
    ) -> List[EvidenceChunk]:
        """Encode query, compute cosine similarity, return top-k EvidenceChunks."""
        query_vec = self._encoder.encode(
            [query_text],
            normalize_embeddings=True,
            convert_to_numpy=True,
            show_progress_bar=False,
        ).astype(np.float32)[0]  # shape (dim,)

        scores = embeddings @ query_vec  # shape (n,)
        n = min(top_k, len(scores))
        top_idx = np.argpartition(scores, -n)[-n:]
        top_idx = top_idx[np.argsort(scores[top_idx])[::-1]]

        chunks: List[EvidenceChunk] = []
        for rank, local_i in enumerate(top_idx, start=1):
            local_i = int(local_i)
            sent = sentences[local_i]
            chunks.append(
                EvidenceChunk(
                    doc_id=sent.get("doc_id", f"sent_{local_i}"),
                    sent_id=int(sent.get("sent_idx", local_i)),
                    text=sent["text"],
                    char_start=int(sent.get("char_start", 0)),
                    char_end=int(sent.get("char_end", len(sent["text"]))),
                    score_dense=float(scores[local_i]),
                    rank=rank,
                    source=sent.get("source", "gold_context"),
                    version=sent.get("version", "sentence_v1"),
                )
            )
        return chunks

    def _cosine_topk_batch(
        self,
        query_texts: List[str],
        embeddings: np.ndarray,   # (n, dim), L2-normalised float32
        sentences: List[dict],
        top_k: int,
    ) -> List[List[EvidenceChunk]]:
        """
        Batched variant of _cosine_topk that encodes all queries in one pass.
        """
        if not query_texts:
            return []

        query_vecs = self._encoder.encode(
            query_texts,
            normalize_embeddings=True,
            convert_to_numpy=True,
            show_progress_bar=False,
        ).astype(np.float32)  # shape (num_queries, dim)

        scores_matrix = embeddings @ query_vecs.T  # shape (num_sents, num_queries)
        all_chunks: List[List[EvidenceChunk]] = []
        num_sents = scores_matrix.shape[0]

        for query_idx in range(scores_matrix.shape[1]):
            scores = scores_matrix[:, query_idx]
            n = min(top_k, num_sents)
            if n <= 0:
                all_chunks.append([])
                continue

            top_idx = np.argpartition(scores, -n)[-n:]
            top_idx = top_idx[np.argsort(scores[top_idx])[::-1]]

            chunks: List[EvidenceChunk] = []
            for rank, local_i in enumerate(top_idx, start=1):
                local_i = int(local_i)
                sent = sentences[local_i]
                chunks.append(
                    EvidenceChunk(
                        doc_id=sent.get("doc_id", f"sent_{local_i}"),
                        sent_id=int(sent.get("sent_idx", local_i)),
                        text=sent["text"],
                        char_start=int(sent.get("char_start", 0)),
                        char_end=int(sent.get("char_end", len(sent["text"]))),
                        score_dense=float(scores[local_i]),
                        rank=rank,
                        source=sent.get("source", "gold_context"),
                        version=sent.get("version", "sentence_v1"),
                    )
                )
            all_chunks.append(chunks)

        return all_chunks

    def _cosine_ranked_all(
        self,
        query_text: str,
        embeddings: np.ndarray,
        sentences: List[dict],
        max_sentences: Optional[int] = None,
    ) -> List[EvidenceChunk]:
        """
        Return all sentence chunks ranked by cosine similarity for one query.
        """
        query_vec = self._encoder.encode(
            [query_text],
            normalize_embeddings=True,
            convert_to_numpy=True,
            show_progress_bar=False,
        ).astype(np.float32)[0]

        scores = embeddings @ query_vec
        sorted_idx = np.argsort(scores)[::-1]
        if max_sentences is not None and max_sentences > 0:
            sorted_idx = sorted_idx[:max_sentences]

        chunks: List[EvidenceChunk] = []
        for rank, local_i in enumerate(sorted_idx, start=1):
            local_i = int(local_i)
            sent = sentences[local_i]
            chunks.append(
                EvidenceChunk(
                    doc_id=sent.get("doc_id", f"sent_{local_i}"),
                    sent_id=int(sent.get("sent_idx", local_i)),
                    text=sent["text"],
                    char_start=int(sent.get("char_start", 0)),
                    char_end=int(sent.get("char_end", len(sent["text"]))),
                    score_dense=float(scores[local_i]),
                    rank=rank,
                    source=sent.get("source", "gold_context"),
                    version=sent.get("version", "sentence_v1"),
                )
            )
        return chunks

    def _cosine_ranked_all_batch(
        self,
        query_texts: List[str],
        embeddings: np.ndarray,
        sentences: List[dict],
        max_sentences: Optional[int] = None,
    ) -> List[List[EvidenceChunk]]:
        """
        Return all sentence chunks ranked by cosine similarity for each query.
        """
        if not query_texts:
            return []

        query_vecs = self._encoder.encode(
            query_texts,
            normalize_embeddings=True,
            convert_to_numpy=True,
            show_progress_bar=False,
        ).astype(np.float32)

        scores_matrix = embeddings @ query_vecs.T
        all_chunks: List[List[EvidenceChunk]] = []

        for query_idx in range(scores_matrix.shape[1]):
            scores = scores_matrix[:, query_idx]
            sorted_idx = np.argsort(scores)[::-1]
            if max_sentences is not None and max_sentences > 0:
                sorted_idx = sorted_idx[:max_sentences]

            chunks: List[EvidenceChunk] = []
            for rank, local_i in enumerate(sorted_idx, start=1):
                local_i = int(local_i)
                sent = sentences[local_i]
                chunks.append(
                    EvidenceChunk(
                        doc_id=sent.get("doc_id", f"sent_{local_i}"),
                        sent_id=int(sent.get("sent_idx", local_i)),
                        text=sent["text"],
                        char_start=int(sent.get("char_start", 0)),
                        char_end=int(sent.get("char_end", len(sent["text"]))),
                        score_dense=float(scores[local_i]),
                        rank=rank,
                        source=sent.get("source", "gold_context"),
                        version=sent.get("version", "sentence_v1"),
                    )
                )
            all_chunks.append(chunks)

        return all_chunks

    def _split_chunks_to_sentences(
        self, chunks: List[EvidenceChunk]
    ) -> List[dict]:
        """
        Split EvidenceChunk texts into sentence dicts using spaCy.
        Preserves the original doc_id and source from each chunk.
        """
        nlp = get_spacy_model(self._spacy_model_name)
        sents: List[dict] = []
        global_sent_idx = 0

        for chunk in chunks:
            if not chunk.text or not chunk.text.strip():
                continue
            doc = nlp(chunk.text)
            for sent in doc.sents:
                text = sent.text.strip()
                if len(text) < self._MIN_SENT_LENGTH:
                    continue
                sents.append(
                    {
                        "text": text,
                        "doc_id": chunk.doc_id,
                        "passage_idx": chunk.rank,
                        "sent_idx": global_sent_idx,
                        "char_start": sent.start_char,
                        "char_end": sent.end_char,
                        "source": chunk.source,
                        "version": chunk.version or "sentence_v1",
                    }
                )
                global_sent_idx += 1

        return sents

    def _rows_to_evidence_chunks(self, sentences: List[dict]) -> List[EvidenceChunk]:
        """Convert indexed sentence rows into EvidenceChunk objects."""
        chunks: List[EvidenceChunk] = []
        for idx, sent in enumerate(sentences, start=1):
            text = sent.get("text", "")
            if not text:
                continue
            chunks.append(
                EvidenceChunk(
                    doc_id=sent.get("doc_id", f"sent_{idx}"),
                    sent_id=int(sent.get("sent_idx", idx - 1)),
                    text=text,
                    char_start=int(sent.get("char_start", 0)),
                    char_end=int(sent.get("char_end", len(text))),
                    # Keep these chunks eligible for contradiction-path logic.
                    score_dense=1.0,
                    rank=idx,
                    source=sent.get("source", "gold_context"),
                    version=sent.get("version", "sentence_v1"),
                )
            )
        return chunks

    def _split_contexts_to_sentences(
        self,
        contexts: List[str],
        prefix: str = "ctx",
    ) -> List[dict]:
        """
        Split raw context strings into sentence dicts using spaCy.
        Sentence doc_ids take the form "{prefix}_p{passage_idx}".
        """
        nlp = get_spacy_model(self._spacy_model_name)
        sents: List[dict] = []
        global_sent_idx = 0

        for passage_idx, context in enumerate(contexts):
            if not context or not context.strip():
                continue
            doc = nlp(context)
            for sent in doc.sents:
                text = sent.text.strip()
                if len(text) < self._MIN_SENT_LENGTH:
                    continue
                sents.append(
                    {
                        "text": text,
                        "doc_id": f"{prefix}_p{passage_idx}",
                        "passage_idx": passage_idx,
                        "sent_idx": global_sent_idx,
                        "char_start": sent.start_char,
                        "char_end": sent.end_char,
                        "source": "gold_context",
                        "version": "sentence_v1",
                    }
                )
                global_sent_idx += 1

        return sents
