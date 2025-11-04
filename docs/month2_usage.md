# Month 2 Baseline RAG System - Quick Start Guide

This guide helps you get started with the Month 2 Baseline RAG (Retrieval-Augmented Generation) system for the AIST-FYP hallucination detection project.

## Table of Contents

- [Overview](#overview)
- [Prerequisites](#prerequisites)
- [Quick Start](#quick-start)
- [Data Preparation](#data-preparation)
- [Running the Baseline RAG](#running-the-baseline-rag)
- [Example Queries](#example-queries)
- [Troubleshooting](#troubleshooting)

## Overview

The Month 2 Baseline RAG system is a retrieval-augmented generation pipeline that:
1. Retrieves relevant Wikipedia evidence chunks using dense retrieval (FAISS)
2. Generates answers using a language model (FLAN-T5-base)
3. Captures metadata (logits, scores) for Month 3 hallucination verification

**Key Components:**
- **Dense Retriever:** sentence-transformers/all-MiniLM-L6-v2 (384D embeddings)
- **Generator LLM:** google/flan-t5-base (250M parameters)
- **Index:** FAISS IVFFLAT (nlist=4096, nprobe=128)

## Prerequisites

Before you begin, ensure you have completed:

1. **Environment Setup** (Month 1):
   ```bash
   # Activate virtual environment
   .\.venv\Scripts\Activate.ps1  # Windows PowerShell
   source venv/bin/activate       # Linux/macOS
   
   # Verify GPU (optional but recommended)
   python verify_gpu.py
   ```

2. **HuggingFace Cache Configuration** (Important for Windows):
   ```powershell
   # Set cache to D: drive to avoid filling C:
   [System.Environment]::SetEnvironmentVariable('HF_HOME', 'D:\huggingface_cache', 'User')
   $env:HF_HOME = 'D:\huggingface_cache'
   ```

3. **Dependencies Installed**:
   ```bash
   pip install -r requirements.txt
   python -m spacy download en_core_web_sm
   ```

## Quick Start

### 1. Download Wikipedia Data

Choose a strategy based on your needs:

**Development (Recommended for testing):**
```bash
# Download 10k articles (~5 minutes, ~50MB)
python scripts/download_wikipedia.py --strategy development
```

**Validation (Medium-scale):**
```bash
# Download 100k articles (~30 minutes, ~500MB)
python scripts/download_wikipedia.py --strategy validation
```

**Production (Full Wikipedia):**
```bash
# Download full dump (~20GB, several hours)
python scripts/download_wikipedia.py --strategy production
```

### 2. Process Wikipedia Data

After downloading, process the data into chunks:

```bash
# Parse Wikipedia and create chunks
python -m src.data_processing.wikipedia_parser \
    --input data/raw/wiki_sample_development.jsonl \
    --output data/processed/wiki_chunks_development.jsonl \
    --config config.yaml
```

Expected output:
- Parsed articles with sentence-level chunks
- Metadata preserved (doc_id, sent_id, char_start, char_end)

### 3. Generate Embeddings

Generate embeddings for all chunks:

```bash
# Generate embeddings using sentence-transformers
python -m src.data_processing.embedding_generator \
    --input data/processed/wiki_chunks_development.jsonl \
    --output data/embeddings/wiki_embeddings_development.npy \
    --metadata data/embeddings/metadata_development.json \
    --config config.yaml \
    --batch-size 16
```

**GPU users:**
- Batch size 16-32 recommended for 8GB VRAM
- Processing ~10k chunks takes ~5-10 minutes on RTX 3070 Ti

**CPU users:**
- Use smaller batch size (8 or less)
- Processing will take longer (~30-60 minutes for 10k chunks)

Progress tracking shows:
```
Processing chunks: 100%|████████████| 10000/10000 [00:08<00:00, 1234.56it/s]
Embeddings saved to: data/embeddings/wiki_embeddings_development.npy
Metadata saved to: data/embeddings/metadata_development.json
```

### 4. Build FAISS Index

Create a FAISS index for efficient similarity search:

```bash
# Build FAISS index
python -m src.retrieval.faiss_index_manager \
    --embeddings data/embeddings/wiki_embeddings_development.npy \
    --metadata data/embeddings/metadata_development.json \
    --output data/indexes/development \
    --config config.yaml \
    --index-type IVFFLAT
```

Expected output:
```
Building FAISS index...
Training index on 10000 vectors...
Adding vectors to index...
Index saved to: data/indexes/development/faiss.index
Metadata saved to: data/indexes/development/metadata.pkl
```

### 5. Run the Baseline RAG Pipeline

Now you're ready to query the system:

```python
# Interactive Python session
from src.generation.baseline_rag_pipeline import BaselineRAGPipeline
from src.utils.config import Config

# Load configuration
config = Config("config.yaml")

# Initialize pipeline
pipeline = BaselineRAGPipeline(
    index_path="data/indexes/development/faiss.index",
    metadata_path="data/indexes/development/metadata.pkl",
    config=config
)

# Query the system
query = "What is machine learning?"
results = pipeline.query(query)

# Display results
for claim_evidence_pair in results:
    print(f"Claim: {claim_evidence_pair.claim.text}")
    print(f"Evidence: {claim_evidence_pair.evidence.text}")
    print(f"Confidence: {claim_evidence_pair.claim.metadata['avg_token_prob']}")
    print("-" * 80)
```

## Data Preparation

### Data Directory Structure

After completing all steps, your directory should look like:

```
data/
├── raw/
│   ├── wiki_sample_development.jsonl       # Downloaded Wikipedia (JSONL)
│   ├── wiki_sample_validation.jsonl        # (Optional) Larger sample
│   └── enwiki-latest-pages-articles.xml.bz2 # (Optional) Full dump
├── processed/
│   └── wiki_chunks_development.jsonl       # Parsed and chunked text
├── embeddings/
│   ├── wiki_embeddings_development.npy     # Embeddings (numpy array)
│   ├── metadata_development.json           # Chunk metadata
│   └── checkpoints/                        # (Optional) Resume capability
└── indexes/
    └── development/
        ├── faiss.index                     # FAISS index file
        └── metadata.pkl                    # Chunk text and metadata
```

### File Formats

**Wikipedia JSONL (`data/raw/`):**
```json
{
  "doc_id": "wiki_00000000",
  "title": "Machine Learning",
  "text": "Full article text...",
  "url": "https://en.wikipedia.org/wiki/Machine_Learning",
  "timestamp": "2025-10-23T22:58:10.904362",
  "source": "huggingface_wikipedia_20220301"
}
```

**Processed Chunks (`data/processed/`):**
```json
{
  "doc_id": "wiki_00000000",
  "sent_id": 0,
  "text": "Machine learning is a field of artificial intelligence.",
  "char_start": 0,
  "char_end": 55,
  "source": "Machine Learning",
  "version": "20220301.en"
}
```

## Running the Baseline RAG

### Command-Line Interface

You can also use the command-line interface:

```bash
# Single query
python -m src.generation.baseline_rag_pipeline \
    --query "What is machine learning?" \
    --config config.yaml \
    --index-path data/indexes/development/faiss.index \
    --metadata-path data/indexes/development/metadata.pkl \
    --output results.json
```

### Python Script

Create a script `run_rag.py`:

```python
#!/usr/bin/env python3
"""Run the baseline RAG pipeline."""

from src.generation.baseline_rag_pipeline import BaselineRAGPipeline
from src.utils.config import Config
import json

def main():
    # Initialize
    config = Config("config.yaml")
    pipeline = BaselineRAGPipeline(
        index_path="data/indexes/development/faiss.index",
        metadata_path="data/indexes/development/metadata.pkl",
        config=config
    )
    
    # Queries
    queries = [
        "What is machine learning?",
        "Who invented the telephone?",
        "What is the capital of France?"
    ]
    
    # Process queries
    all_results = {}
    for query in queries:
        print(f"\nQuery: {query}")
        print("=" * 80)
        
        results = pipeline.query(query)
        all_results[query] = [
            {
                "claim": pair.claim.text,
                "evidence": pair.evidence.text,
                "metadata": pair.claim.metadata
            }
            for pair in results
        ]
        
        # Display
        for i, pair in enumerate(results, 1):
            print(f"\n{i}. Claim: {pair.claim.text}")
            print(f"   Evidence: {pair.evidence.text[:200]}...")
            print(f"   Confidence: {pair.claim.metadata.get('avg_token_prob', 'N/A')}")
    
    # Save results
    with open("rag_results.json", "w") as f:
        json.dump(all_results, f, indent=2)
    print("\nResults saved to rag_results.json")

if __name__ == "__main__":
    main()
```

Run it:
```bash
python run_rag.py
```

## Example Queries

### Example 1: Factual Query

**Query:** "What is machine learning?"

**Expected Output:**
```
Claim: Machine learning is a field of artificial intelligence that uses statistical 
       techniques to give computers the ability to learn from data.
Evidence: Machine learning (ML) is a field of study in artificial intelligence 
          concerned with the development of algorithms that can learn from data...
Confidence: 0.85
```

### Example 2: Named Entity Query

**Query:** "Who invented the telephone?"

**Expected Output:**
```
Claim: Alexander Graham Bell invented the telephone in 1876.
Evidence: Alexander Graham Bell was a Scottish-born inventor, scientist, and 
          engineer who is credited with inventing and patenting the first 
          practical telephone...
Confidence: 0.92
```

### Example 3: Definition Query

**Query:** "What is photosynthesis?"

**Expected Output:**
```
Claim: Photosynthesis is the process by which plants convert light energy into 
       chemical energy.
Evidence: Photosynthesis is a process used by plants and other organisms to 
          convert light energy into chemical energy that can later be released...
Confidence: 0.88
```

## Troubleshooting

### Common Issues

#### 1. Out of Memory (GPU)

**Error:** `CUDA out of memory`

**Solutions:**
- Reduce batch size in `config.yaml`:
  ```yaml
  processing:
    batch_size: 8  # Reduce from 16
  ```
- Use FP16 precision:
  ```yaml
  processing:
    use_fp16: true
  ```
- Process in smaller batches with checkpointing

#### 2. Import Errors

**Error:** `ModuleNotFoundError: No module named 'src'`

**Solutions:**
- Ensure you're in the project root directory
- Add project root to PYTHONPATH:
  ```bash
  export PYTHONPATH="${PYTHONPATH}:$(pwd)"  # Linux/macOS
  $env:PYTHONPATH += ";$(pwd)"              # Windows PowerShell
  ```

#### 3. FAISS Index Not Found

**Error:** `FileNotFoundError: FAISS index not found`

**Solutions:**
- Ensure you've completed Step 4 (Build FAISS Index)
- Check the index path in your command/script
- Rebuild the index if corrupted:
  ```bash
  python -m src.retrieval.faiss_index_manager --embeddings ... --output ...
  ```

#### 4. Slow Retrieval

**Problem:** Retrieval takes too long

**Solutions:**
- Check FAISS index type (IVFFLAT is faster than FLAT)
- Adjust `nprobe` in `config.yaml` (lower = faster but less accurate):
  ```yaml
  retrieval:
    nprobe: 64  # Reduce from 128
  ```
- Use GPU FAISS if available:
  ```bash
  pip install faiss-gpu
  ```

#### 5. Poor Quality Responses

**Problem:** Generated answers are not relevant

**Possible Causes:**
- Not enough Wikipedia data (use validation or production strategy)
- Retrieval not finding relevant chunks (increase `top_k`)
- Generator model needs different prompting

**Solutions:**
- Increase retrieval top_k:
  ```yaml
  retrieval:
    top_k: 10  # Increase from 5
  ```
- Try different generation parameters:
  ```yaml
  generation:
    temperature: 0.5  # Lower for more deterministic
    max_new_tokens: 512  # Increase for longer answers
  ```

## Next Steps

After successfully running the Baseline RAG:

1. **Month 3:** Implement the Verifier Module
   - Intrinsic Uncertainty Detector (entropy, perplexity)
   - Retrieval-Grounded Heuristics (evidence coverage, citation integrity)

2. **Evaluation:** Test on benchmarks
   - TruthfulQA
   - RAGTruth
   - FEVER

3. **Documentation:** Refer to
   - `docs/architecture_month2.md` - Technical implementation details
   - `USAGE.md` - Complete API reference
   - `README_ENVIRONMENT.md` - Environment setup

## Support

For issues or questions:
- Check `TODO_List.md` for project progress
- Review `helpful_tools.md` for additional resources
- Consult `System_Architecture_Design.md` for architecture details

---

**Last Updated:** 2025-10-25  
**Month 2 Status:** Complete ✓  
**Next Milestone:** Month 3 - Verifier Module Implementation
