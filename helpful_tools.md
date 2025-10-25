# Helpful Tools for Hallucination Detection Project

This document outlines tools and resources that can be beneficial for the AIST-FYP project on LLM hallucination.

## 1. Ragas: A Framework for Evaluating RAG Pipelines

Ragas is an open-source framework designed to evaluate Retrieval-Augmented Generation (RAG) pipelines. This is highly relevant to the project, as the system architecture is based on a RAG pipeline. Ragas provides a set of metrics to assess the performance of different components of the pipeline, which can be directly applied to the "Verifier" module to detect and quantify hallucinations.

### Key Benefits for the Project:

- **Standardized Evaluation:** Provides a standardized way to measure the quality of the system's outputs, making it easier to compare different approaches and track improvements.
- **Component-wise Metrics:** Offers metrics to evaluate not just the final answer, but also the performance of the retrieval and generation components separately.
- **Hallucination-focused Metrics:** Many of the core metrics are directly aimed at identifying hallucinations and ensuring factual consistency.

### Relevant Ragas Metrics:

- **`Faithfulness`**: Evaluates whether the generated answer is factually consistent with the information present in the retrieved context. A low faithfulness score indicates a potential hallucination.
- **`Answer Relevancy`**: Measures how relevant the generated answer is to the original prompt.
- **`Context Precision` and `Context Recall`**: These metrics evaluate the performance of the retriever.
- **`Factual Correctness`**: Assesses if the answer is factually correct with respect to a ground truth answer.

By integrating Ragas, the project can systematically evaluate its hallucination detection and mitigation strategies.

## 2. Hugging Face: Models and Datasets for a Trainless Approach

Hugging Face is an essential resource for this project, especially for the new **trainless** approach. It provides access to off-the-shelf models and tools that are crucial for building the "Verifier" module without requiring fine-tuning.

### How Hugging Face Will Be Used:

- **Zero-Shot NLI Models:** The verifier will use a pre-trained, multi-domain NLI model to check for contradictions. A strong candidate is `MoritzLaurer/DeBERTa-v3-base-mnli-fever-anli`, which is already fine-tuned on several fact-checking datasets (MNLI, FEVER, ANLI). This allows for immediate, high-quality contradiction detection without any training.
- **Cross-Encoder Models for Relevance:** A pre-trained cross-encoder, such as one from the `ms-marco` collection (e.g., `cross-encoder/ms-marco-MiniLM-L-6-v2`), can be used to score the semantic relevance between a generated claim and the retrieved evidence. A low relevance score can be a powerful heuristic for suspecting hallucination.
- **Transformers Library:** The `transformers` library from Hugging Face simplifies the process of downloading and using these pre-trained models for inference.
- **Datasets for Evaluation:** While the approach is trainless, datasets like `FEVER`, `TruthfulQA`, and `RAGTruth` remain critical for **evaluating** the effectiveness of the different zero-shot signals and the overall detector.

## 3. Month 2 Tools & Scripts

These utilities were developed during Month 2 to support the baseline RAG system implementation.

### Scripts

- **`scripts/download_wikipedia.py`**: Download Wikipedia datasets from Hugging Face Hub
  - **Usage:** `python scripts/download_wikipedia.py [--mode {development|validation|production}]`
  - **Modes:**
    - `development`: Download 100 articles (~10k chunks) for quick iteration
    - `validation`: Download 10,000 articles (~100k chunks) for validation
    - `production`: Download full 6.4M articles for production use
  - **Output:** JSONL file in `data/raw/wikipedia_en_{num_articles}.jsonl`
  - **Features:** Progress tracking, automatic retry on failure, configurable cache directory

### Data Processing Utilities

- **Wikipedia Parser** (`src/data_processing/wikipedia_parser.py`)
  - Parse Wikipedia JSONL files to sentence-level chunks
  - **Usage:** `parser.parse_file(input_file, output_file)`
  - **Output:** Chunked JSONL with metadata (title, URL, chunk index)

- **Embedding Generator** (`src/data_processing/embedding_generator.py`)
  - Generate embeddings with checkpoint resume capability
  - **Usage:** `generator.generate_embeddings(chunks_file, output_file, batch_size=16)`
  - **Features:** 
    - Automatic checkpointing every 10k chunks
    - Resume from last checkpoint on interruption
    - GPU batch processing support
    - Progress tracking with ETA

- **Text Chunker** (`src/data_processing/text_chunker.py`)
  - Sentence-level text chunking using spaCy
  - **Usage:** `chunker.chunk_text(text)`
  - **Config:** Max tokens (default: 128), overlap (default: 0)

### Retrieval Tools

- **FAISS Index Manager** (`src/retrieval/faiss_index_manager.py`)
  - Build and manage FAISS indexes
  - **Supported Index Types:**
    - `FLAT`: Exact search (development)
    - `IVFFLAT`: Fast approximate search (10k-1M docs)
    - `HNSW`: Graph-based search (>1M docs)
  - **Usage:** `manager.build_index(embeddings, index_type="IVFFLAT", nlist=4096)`

- **Dense Retriever** (`src/retrieval/dense_retriever.py`)
  - Query encoding and similarity search
  - **Usage:** `retriever.retrieve(query, top_k=5)`
  - **Features:** Configurable nprobe for IVFFLAT, batch query support

### Generation Tools

- **Generator Wrapper** (`src/generation/generator_wrapper.py`)
  - LLM wrapper with metadata capture for Month 3
  - **Captured Metadata:**
    - Token-level logits and probabilities
    - Average token probability (confidence proxy)
    - Sequence scores from beam search
    - Input prompt used
  - **Usage:** `generator.generate(prompt, max_new_tokens=100)`

- **Claim Extractor** (`src/generation/claim_extractor.py`)
  - Extract individual claims from generated text
  - **Usage:** `extractor.extract_claims(generated_text)`
  - **Method:** Sentence segmentation with spaCy

- **Baseline RAG Pipeline** (`src/generation/baseline_rag_pipeline.py`)
  - End-to-end RAG query processing
  - **Usage:** `pipeline.run(query)`
  - **Output:** Generated text, retrieved chunks, claims, claim-evidence pairs, metadata

### Testing Framework

- **pytest Configuration** (`pytest.ini`)
  - Test discovery, coverage reporting, verbose output
  - **Run Tests:** `pytest` (runs all 103 tests in ~98 seconds)
  - **Run with Coverage:** `pytest --cov=src --cov-report=html`

- **Test Fixtures** (`tests/fixtures/`)
  - `sample_chunks.jsonl`: 10 Wikipedia chunks for testing
  - `test_config.yaml`: Test-specific configuration
  - Shared fixtures in `tests/conftest.py`

### Configuration Management

- **Config Loader** (`src/utils/config.py`)
  - YAML configuration loading and validation
  - **Usage:** `config = load_config("config.yaml")`
  - **Structure:** Models, data paths, processing params, retrieval, generation

### Logging Utilities

- **Centralized Logger** (`src/utils/logger.py`)
  - Structured logging across all modules
  - **Usage:** `from src.utils.logger import get_logger; logger = get_logger(__name__)`
  - **Features:** File and console output, configurable log level

### Data Structures

- **Evidence Chunk** (`src/utils/data_structures.EvidenceChunk`)
  - Represents a chunk of text with metadata
  - **Fields:** chunk_id, text, source_doc_id, chunk_idx, metadata

- **Claim** (`src/utils/data_structures.Claim`)
  - Represents a factual claim
  - **Fields:** claim_id, text, source

- **Claim-Evidence Pair** (`src/utils/data_structures.ClaimEvidencePair`)
  - Pairs claims with supporting evidence
  - **Fields:** claim, evidence_chunks, retrieval_scores

### HuggingFace Cache Configuration

- **Environment Variable:** `HF_HOME`
  - Set to `D:\huggingface_cache` to avoid filling C: drive
  - **Setup:** Add `HF_HOME=D:\huggingface_cache` to system environment variables
  - **Documentation:** See `README_ENVIRONMENT.md` for details

### Performance Monitoring

- **Benchmarking Results** (Development Mode: 10k articles)
  - Wikipedia Download: 10 min 2 sec (650 articles/sec)
  - Parsing & Chunking: 25 seconds (400 articles/sec)
  - Embedding Generation: 8.5 seconds (1,176 chunks/sec)
  - FAISS Index Build: 5 seconds (2,000 vectors/sec)
  - Query Latency: ~860ms end-to-end (1.16 queries/sec)
  - **Bottleneck:** Text generation (98.8% of query time)
