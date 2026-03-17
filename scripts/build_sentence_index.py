"""
Build a sentence-level evidence index for RAGTruth or CiteEval oracle evaluation.

Reads gold contexts / oracle passages from the benchmark dataset, splits each
passage into sentences using spaCy, encodes them with the project's
sentence-transformer model, and writes three files to the output directory:

  sentences.jsonl    — one JSON line per sentence
  embeddings.npy     — float32 (N, embedding_dim), L2-normalised, row-aligned
  sample_index.json  — {sample_id: [row_start, row_end]} for O(1) per-sample slicing

The pre-built index is consumed by EvidenceSentenceRetriever in
src/retrieval/sentence_retriever.py.

Usage
-----
    # RAGTruth test split (default)
    python scripts/build_sentence_index.py --dataset ragtruth --split test

    # RAGTruth all splits (train + test) into a single combined index
    python scripts/build_sentence_index.py --dataset ragtruth --split all

    # CiteEval oracle ASQA dataset
    python scripts/build_sentence_index.py --dataset citeeval --oracle-dataset asqa

    # Custom oracle path
    python scripts/build_sentence_index.py --dataset citeeval \\
        --oracle-source benchmark/CiteEval/data/dev/asqa_oracle.dev.jsonl

    # Override encoder / device
    python scripts/build_sentence_index.py --dataset ragtruth --split test \\
        --encoder sentence-transformers/all-MiniLM-L6-v2 --device cuda
"""

from __future__ import annotations

import argparse
import json
import sys
from pathlib import Path

import numpy as np
from sentence_transformers import SentenceTransformer
from tqdm import tqdm

project_root = Path(__file__).resolve().parents[1]
sys.path.insert(0, str(project_root))

from src.utils.config import Config  # noqa: E402
from src.utils.logger import setup_logger  # noqa: E402
from src.utils.nlp_utils import get_spacy_model  # noqa: E402


logger = setup_logger(__name__)

MIN_SENT_LENGTH = 10  # minimum characters for a valid sentence

CITEEVAL_ORACLE_PRESETS: dict[str, str] = {
    "asqa": "benchmark/CiteEval/data/dev/asqa_oracle.dev.jsonl",
    "eli5": "benchmark/CiteEval/data/dev/eli5_oracle.dev.jsonl",
    "msmarco": "benchmark/CiteEval/data/dev/msmarco_oracle.dev.jsonl",
}


# ---------------------------------------------------------------------------
# Sentence splitting
# ---------------------------------------------------------------------------

def _split_to_sentences(
    nlp,
    contexts: list[str],
    prefix: str,
) -> list[dict]:
    """
    Split a list of context strings into sentence dicts using spaCy.

    Each sentence dict has the keys:
      text, doc_id, passage_idx, sent_idx, char_start, char_end, source, version
    """
    sents: list[dict] = []
    global_idx = 0
    for passage_idx, ctx in enumerate(contexts):
        if not ctx or not ctx.strip():
            continue
        doc = nlp(ctx)
        for sent in doc.sents:
            text = sent.text.strip()
            if len(text) < MIN_SENT_LENGTH:
                continue
            sents.append(
                {
                    "text": text,
                    "doc_id": f"{prefix}_p{passage_idx}",
                    "passage_idx": passage_idx,
                    "sent_idx": global_idx,
                    "char_start": sent.start_char,
                    "char_end": sent.end_char,
                    "source": "gold_context",
                    "version": "sentence_v1",
                }
            )
            global_idx += 1
    return sents


# ---------------------------------------------------------------------------
# Dataset loaders — return list of (sample_id, contexts) tuples
# ---------------------------------------------------------------------------

def _load_ragtruth_samples(
    dataset_dir: Path,
    splits: list[str],
) -> list[tuple[str, list[str]]]:
    """
    Load (sample_id, contexts) pairs from the RAGTruth benchmark dataset.

    Splits is a list of split names to include (e.g., ["test"] or ["train", "test"]).
    Only "good" quality responses are included (matching the evaluator logic).
    """
    source_info_path = dataset_dir / "source_info.jsonl"
    response_path = dataset_dir / "response.jsonl"

    if not source_info_path.exists():
        raise FileNotFoundError(f"source_info.jsonl not found: {source_info_path}")
    if not response_path.exists():
        raise FileNotFoundError(f"response.jsonl not found: {response_path}")

    source_map: dict[str, dict] = {}
    with source_info_path.open(encoding="utf-8") as f:
        for line in f:
            item = json.loads(line.strip())
            source_map[item["source_id"]] = item

    samples: list[tuple[str, list[str]]] = []
    seen_ids: set[str] = set()

    with response_path.open(encoding="utf-8") as f:
        for line in f:
            resp = json.loads(line.strip())
            if splits != ["all"] and resp.get("split") not in splits:
                continue
            if resp.get("quality") != "good":
                continue

            sample_id = str(resp["id"])
            if sample_id in seen_ids:
                continue
            seen_ids.add(sample_id)

            source = source_map.get(resp["source_id"])
            if source is None:
                continue

            task_type = source["task_type"]
            source_info = source["source_info"]

            if task_type == "QA":
                passages_text = source_info["passages"]
                contexts = []
                for passage in passages_text.split("\n\n"):
                    passage = passage.strip()
                    if passage.startswith("passage "):
                        passage = passage.split(":", 1)[1].strip()
                    if passage:
                        contexts.append(passage)
            elif task_type == "Summary":
                contexts = (
                    [source_info]
                    if isinstance(source_info, str)
                    else [json.dumps(source_info)]
                )
            else:  # Data2txt
                contexts = [json.dumps(source_info, indent=2)]

            samples.append((sample_id, contexts))

    return samples


def _load_citeeval_samples(oracle_path: Path) -> list[tuple[str, list[str]]]:
    """
    Load (sample_id, passage_texts) pairs from a CiteEval oracle JSONL file.

    Each line must be a JSON object with an optional "id" field and a "passages"
    list where each passage has a "text" field.
    """
    samples: list[tuple[str, list[str]]] = []
    with oracle_path.open(encoding="utf-8") as f:
        for line_num, line in enumerate(f, start=1):
            line = line.strip()
            if not line:
                continue
            row = json.loads(line)
            sample_id = str(row.get("id", f"row_{line_num}"))
            passages = row.get("passages", [])
            texts = [
                str(p.get("text", "")).strip()
                for p in passages
                if isinstance(p, dict) and str(p.get("text", "")).strip()
            ]
            if texts:
                samples.append((sample_id, texts))
    return samples


# ---------------------------------------------------------------------------
# Index builder
# ---------------------------------------------------------------------------

def build_index(
    samples: list[tuple[str, list[str]]],
    output_dir: Path,
    encoder_model: str,
    device: str,
    spacy_model: str,
    batch_size: int,
) -> None:
    """
    Build and write the sentence index files to *output_dir*.

    Args:
        samples:       List of (sample_id, context_strings) tuples.
        output_dir:    Target directory for the three index files.
        encoder_model: SentenceTransformer model name.
        device:        "cuda" or "cpu".
        spacy_model:   spaCy model name for sentence splitting.
        batch_size:    Encoding batch size.
    """
    output_dir.mkdir(parents=True, exist_ok=True)

    logger.info("Loading spaCy model: %s", spacy_model)
    nlp = get_spacy_model(spacy_model)

    logger.info("Loading SentenceTransformer: %s on %s", encoder_model, device)
    encoder = SentenceTransformer(encoder_model, device=device)

    all_sents: list[dict] = []
    sample_index: dict[str, list[int]] = {}

    logger.info("Splitting %d samples into sentences ...", len(samples))
    for sample_id, contexts in tqdm(samples, desc="Sentence splitting", unit="sample"):
        start = len(all_sents)
        sents = _split_to_sentences(nlp, contexts, prefix=sample_id)
        all_sents.extend(sents)
        end = len(all_sents)
        sample_index[sample_id] = [start, end]

    if not all_sents:
        logger.warning("No sentences found; aborting index build.")
        return

    total_sents = len(all_sents)
    logger.info("Total sentences: %d", total_sents)

    # Encode in batches with progress bar
    texts = [s["text"] for s in all_sents]
    logger.info("Encoding sentences (batch_size=%d) ...", batch_size)
    embeddings = encoder.encode(
        texts,
        batch_size=batch_size,
        normalize_embeddings=True,
        convert_to_numpy=True,
        show_progress_bar=True,
    ).astype(np.float32)

    # Write outputs
    sentences_path = output_dir / "sentences.jsonl"
    embeddings_path = output_dir / "embeddings.npy"
    sample_index_path = output_dir / "sample_index.json"

    logger.info("Writing sentences.jsonl (%d entries) ...", total_sents)
    with sentences_path.open("w", encoding="utf-8") as f:
        for sent in all_sents:
            f.write(json.dumps(sent, ensure_ascii=False) + "\n")

    logger.info("Writing embeddings.npy (shape %s) ...", embeddings.shape)
    np.save(str(embeddings_path), embeddings)

    logger.info("Writing sample_index.json (%d samples) ...", len(sample_index))
    with sample_index_path.open("w", encoding="utf-8") as f:
        json.dump(sample_index, f, ensure_ascii=False)

    logger.info(
        "✓ Sentence index built: %d sentences, %d samples → %s",
        total_sents,
        len(sample_index),
        output_dir,
    )


# ---------------------------------------------------------------------------
# CLI
# ---------------------------------------------------------------------------

def _build_parser() -> argparse.ArgumentParser:
    parser = argparse.ArgumentParser(
        description=(
            "Build a sentence-level evidence index for RAGTruth or CiteEval "
            "gold-context evaluation."
        )
    )
    parser.add_argument(
        "--config",
        type=str,
        default="config.yaml",
        help="Path to config file (default: config.yaml)",
    )
    parser.add_argument(
        "--dataset",
        type=str,
        required=True,
        choices=["ragtruth", "citeeval"],
        help="Dataset to build the index for",
    )
    parser.add_argument(
        "--split",
        type=str,
        default="test",
        help="RAGTruth split: train | test | all (ignored for citeeval)",
    )
    parser.add_argument(
        "--oracle-dataset",
        type=str,
        default="asqa",
        choices=list(CITEEVAL_ORACLE_PRESETS),
        help="CiteEval oracle dataset preset (citeeval mode only; default: asqa)",
    )
    parser.add_argument(
        "--oracle-source",
        type=str,
        default=None,
        help="Custom path to oracle JSONL, relative to project root (overrides --oracle-dataset)",
    )
    parser.add_argument(
        "--output-dir",
        type=str,
        default=None,
        help=(
            "Output directory "
            "(default: data/indexes/{dataset}_sentences/{split or oracle-dataset})"
        ),
    )
    parser.add_argument(
        "--encoder",
        type=str,
        default=None,
        help="SentenceTransformer model name (default: read from config.yaml models.sentence_transformer)",
    )
    parser.add_argument(
        "--device",
        type=str,
        default=None,
        help="Device: cuda or cpu (default: read from config.yaml processing.device)",
    )
    parser.add_argument(
        "--spacy-model",
        type=str,
        default="en_core_web_sm",
        help="spaCy model for sentence splitting (default: en_core_web_sm)",
    )
    parser.add_argument(
        "--batch-size",
        type=int,
        default=256,
        help="Encoding batch size (default: 256)",
    )
    return parser


def main() -> int:
    args = _build_parser().parse_args()

    config = Config(args.config)
    encoder_model = args.encoder or str(config.models.sentence_transformer)
    device_str = args.device or str(getattr(config.processing, "device", "cpu"))

    # Resolve output directory and load samples
    if args.dataset == "ragtruth":
        split_label = args.split
        if args.output_dir:
            output_dir = Path(args.output_dir)
        else:
            output_dir = project_root / "data" / "indexes" / "ragtruth_sentences" / split_label

        ragtruth_dir = project_root / "benchmark" / "RAGTruth" / "dataset"
        splits = ["train", "test"] if args.split == "all" else [args.split]
        samples = _load_ragtruth_samples(ragtruth_dir, splits)

    else:  # citeeval
        split_label = args.oracle_dataset if not args.oracle_source else "custom"
        if args.output_dir:
            output_dir = Path(args.output_dir)
        else:
            output_dir = project_root / "data" / "indexes" / "citeeval_sentences" / split_label

        if args.oracle_source:
            oracle_path = (project_root / args.oracle_source).resolve()
        else:
            oracle_path = project_root / CITEEVAL_ORACLE_PRESETS[args.oracle_dataset]

        if not oracle_path.exists():
            logger.error("Oracle source not found: %s", oracle_path)
            return 1

        samples = _load_citeeval_samples(oracle_path)

    if not samples:
        logger.error("No samples found; check dataset path and split name.")
        return 1

    logger.info(
        "Building sentence index: dataset=%s, split=%s, samples=%d, output=%s",
        args.dataset,
        split_label,
        len(samples),
        output_dir,
    )

    build_index(
        samples=samples,
        output_dir=output_dir,
        encoder_model=encoder_model,
        device=device_str,
        spacy_model=args.spacy_model,
        batch_size=args.batch_size,
    )
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
