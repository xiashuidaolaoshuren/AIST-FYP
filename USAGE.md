# USAGE Guide - AIST-FYP Baseline RAG System

This document provides comprehensive API documentation, code examples, and usage patterns for the AIST-FYP Baseline RAG system.

## Table of Contents

- [Quick Start](#quick-start)
- [Configuration](#configuration)
- [Module-by-Module API](#module-by-module-api)
- [End-to-End Workflows](#end-to-end-workflows)
- [Integration Patterns](#integration-patterns)
- [Error Handling](#error-handling)
- [Best Practices](#best-practices)

## Quick Start

See [docs/month2_usage.md](docs/month2_usage.md) for a step-by-step quick start guide.

## Configuration

### Config File Structure

The system uses YAML configuration files. Default: `config.yaml`

```yaml
# Model Configuration
models:
  embedding_model: "sentence-transformers/all-MiniLM-L6-v2"
  generator_model: "google/flan-t5-base"
  spacy_model: "en_core_web_sm"

# Data Paths
data:
  raw_dir: "data/raw"
  processed_dir: "data/processed"
  embeddings_dir: "data/embeddings"
  indexes_dir: "data/indexes"

# Processing Parameters
processing:
  chunk_max_tokens: 128
  chunk_overlap: 0
  embedding_batch_size: 16
  checkpoint_frequency: 10000

# Retrieval Configuration
retrieval:
  index_type: "IVFFLAT"
  nlist: 4096
  nprobe: 128
  top_k: 5

# Generation Configuration
generation:
  max_new_tokens: 100
  temperature: 0.7
  num_beams: 1
```

### Loading Configuration

```python
from src.utils.config import load_config

# Load default config
config = load_config()

# Load custom config
config = load_config("path/to/custom_config.yaml")

# Access config values
embedding_model = config['models']['embedding_model']
top_k = config['retrieval']['top_k']
```

## Module-by-Module API

### 1. Data Structures (`src/utils/data_structures.py`)

#### EvidenceChunk

Represents a chunk of text with metadata.

```python
from src.utils.data_structures import EvidenceChunk

chunk = EvidenceChunk(
    chunk_id="wiki_123_chunk_5",
    text="Machine learning is a subset of artificial intelligence.",
    source_doc_id="wiki_123",
    chunk_idx=5,
    metadata={"title": "Machine Learning", "section": "Introduction"}
)

# Access properties
print(chunk.text)           # "Machine learning is..."
print(chunk.chunk_id)       # "wiki_123_chunk_5"
print(chunk.metadata)       # {"title": "Machine Learning", ...}

# Convert to dict for serialization
chunk_dict = chunk.to_dict()

# Create from dict
chunk = EvidenceChunk.from_dict(chunk_dict)
```

#### Claim

Represents a single factual claim extracted from generated text.

```python
from src.utils.data_structures import Claim

claim = Claim(
    claim_id="claim_001",
    text="The Eiffel Tower is located in Paris.",
    source="generated"
)

# Serialize
claim_dict = claim.to_dict()
```

#### ClaimEvidencePair

Pairs a claim with supporting evidence chunks.

```python
from src.utils.data_structures import ClaimEvidencePair

pair = ClaimEvidencePair(
    claim=claim,
    evidence_chunks=[chunk1, chunk2],
    retrieval_scores=[0.95, 0.87]
)

# Access components
print(pair.claim.text)              # "The Eiffel Tower..."
print(pair.evidence_chunks[0].text) # Evidence text
print(pair.retrieval_scores)        # [0.95, 0.87]
```

### 2. Data Processing

#### Wikipedia Parser (`src/data_processing/wikipedia_parser.py`)

Parse Wikipedia JSONL files into chunks.

```python
from src.data_processing.wikipedia_parser import WikipediaParser
from src.utils.config import load_config

config = load_config()
parser = WikipediaParser(config)

# Parse Wikipedia file to chunks
chunks = parser.parse_file(
    input_file="data/raw/wikipedia_en_100.jsonl",
    output_file="data/processed/chunks.jsonl"
)

print(f"Parsed {len(chunks)} chunks")
# Output: Parsed 10000 chunks
```

**Input Format (Wikipedia JSONL):**
```json
{"id": "12", "url": "https://en.wikipedia.org/wiki/Anarchism", "title": "Anarchism", "text": "Article text..."}
```

**Output Format (Chunks JSONL):**
```json
{"chunk_id": "12_chunk_0", "text": "Anarchism is a political philosophy...", "source_doc_id": "12", "chunk_idx": 0, "metadata": {"title": "Anarchism", "url": "https://..."}}
```

#### Text Chunker (`src/data_processing/text_chunker.py`)

Chunk text into sentence-level fragments.

```python
from src.data_processing.text_chunker import TextChunker

chunker = TextChunker(
    spacy_model="en_core_web_sm",
    max_tokens=128,
    overlap=0
)

# Chunk a single text
text = "This is sentence one. This is sentence two. This is sentence three."
chunks = chunker.chunk_text(text)

print(chunks)
# Output: ["This is sentence one.", "This is sentence two.", "This is sentence three."]

# Chunk with metadata
chunks_with_metadata = chunker.chunk_with_metadata(
    text=text,
    doc_id="doc_123",
    metadata={"title": "Example"}
)

# Returns list of EvidenceChunk objects
for chunk in chunks_with_metadata:
    print(chunk.chunk_id, chunk.text)
```

#### Embedding Generator (`src/data_processing/embedding_generator.py`)

Generate embeddings for chunks with checkpointing.

```python
from src.data_processing.embedding_generator import EmbeddingGenerator
from src.utils.config import load_config

config = load_config()
generator = EmbeddingGenerator(config)

# Generate embeddings for chunks
embeddings = generator.generate_embeddings(
    chunks_file="data/processed/chunks.jsonl",
    output_file="data/embeddings/embeddings.npy",
    batch_size=16,
    checkpoint_frequency=10000
)

print(f"Generated {len(embeddings)} embeddings")
# Output: Generated 10000 embeddings

# Embeddings are saved as NumPy arrays
# Shape: (num_chunks, embedding_dim) e.g., (10000, 384)
```

**Checkpoint Resume:**
```python
# If interrupted, automatically resumes from last checkpoint
# Checkpoint files: data/embeddings/embeddings_checkpoint_10000.npy
embeddings = generator.generate_embeddings(
    chunks_file="data/processed/chunks.jsonl",
    output_file="data/embeddings/embeddings.npy",
    batch_size=16
)
# Resumes from checkpoint if found
```

### 3. Retrieval

#### FAISS Index Manager (`src/retrieval/faiss_index_manager.py`)

Build and manage FAISS indexes.

```python
from src.retrieval.faiss_index_manager import FAISSIndexManager
from src.utils.config import load_config
import numpy as np

config = load_config()
manager = FAISSIndexManager(config)

# Load embeddings
embeddings = np.load("data/embeddings/embeddings.npy")

# Build IVFFLAT index
index = manager.build_index(
    embeddings=embeddings,
    index_type="IVFFLAT",
    nlist=4096
)

# Save index
manager.save_index(index, "data/indexes/wikipedia_ivfflat.index")

# Load index
loaded_index = manager.load_index("data/indexes/wikipedia_ivfflat.index")

# Search index
query_embedding = np.random.rand(1, 384).astype('float32')
distances, indices = loaded_index.search(query_embedding, k=5)

print(f"Top 5 indices: {indices[0]}")
print(f"Top 5 distances: {distances[0]}")
```

**Index Types:**

- **FLAT:** Exact search (slow for large datasets)
  ```python
  index = manager.build_index(embeddings, index_type="FLAT")
  ```

- **IVFFLAT:** Inverted file with flat quantizer (recommended for 10k-1M docs)
  ```python
  index = manager.build_index(embeddings, index_type="IVFFLAT", nlist=4096)
  ```

- **HNSW:** Hierarchical navigable small world (best for >1M docs)
  ```python
  index = manager.build_index(embeddings, index_type="HNSW", M=32)
  ```

#### Dense Retriever (`src/retrieval/dense_retriever.py`)

Encode queries and retrieve relevant chunks.

```python
from src.retrieval.dense_retriever import DenseRetriever
from src.utils.config import load_config

config = load_config()
retriever = DenseRetriever(config)

# Load index and chunks
retriever.load_index("data/indexes/wikipedia_ivfflat.index")
retriever.load_chunks("data/processed/chunks.jsonl")

# Retrieve relevant chunks
query = "What is machine learning?"
results = retriever.retrieve(query, top_k=5)

# Results is a list of (EvidenceChunk, score) tuples
for chunk, score in results:
    print(f"Score: {score:.3f}")
    print(f"Text: {chunk.text[:100]}...")
    print(f"Metadata: {chunk.metadata}")
    print("-" * 50)
```

**Advanced Retrieval:**

```python
# Set nprobe for IVFFLAT index (trade-off between speed and recall)
retriever.set_nprobe(128)  # Default: 128

# Retrieve with custom top_k
results = retriever.retrieve(query, top_k=10)

# Get only chunks (no scores)
chunks = [chunk for chunk, score in results]

# Get only scores
scores = [score for chunk, score in results]
```

### 4. Generation

#### Generator Wrapper (`src/generation/generator_wrapper.py`)

Generate text with metadata capture.

```python
from src.generation.generator_wrapper import GeneratorWrapper
from src.utils.config import load_config

config = load_config()
generator = GeneratorWrapper(config)

# Generate text from prompt
prompt = "Answer the question: What is the capital of France?"
result = generator.generate(
    prompt=prompt,
    max_new_tokens=50,
    temperature=0.7,
    num_beams=1
)

# Access generated text
print(result['generated_text'])
# Output: "The capital of France is Paris."

# Access metadata (for Month 3 hallucination detection)
print(result['avg_token_prob'])      # Average token probability
print(result['token_logits'])        # List of logit arrays
print(result['token_probs'])         # List of token probabilities
print(result['sequence_scores'])     # Beam search scores
```

**Batch Generation:**

```python
# Generate for multiple prompts
prompts = [
    "What is machine learning?",
    "What is the Eiffel Tower?",
    "Who invented the telephone?"
]

results = [generator.generate(prompt) for prompt in prompts]

for result in results:
    print(result['generated_text'])
```

**Generation Parameters:**

```python
result = generator.generate(
    prompt="What is artificial intelligence?",
    max_new_tokens=100,     # Maximum tokens to generate
    temperature=0.7,        # Sampling temperature (0.0 = greedy, 1.0 = random)
    num_beams=1,            # Number of beams for beam search (1 = greedy)
    do_sample=False,        # Whether to use sampling
    top_k=50,               # Top-k sampling
    top_p=0.95              # Nucleus sampling
)
```

#### Claim Extractor (`src/generation/claim_extractor.py`)

Extract individual claims from generated text.

```python
from src.generation.claim_extractor import ClaimExtractor

extractor = ClaimExtractor(spacy_model="en_core_web_sm")

# Extract claims from generated text
generated_text = "Paris is the capital of France. The Eiffel Tower is located in Paris."
claims = extractor.extract_claims(generated_text, source="generated")

# Returns list of Claim objects
for claim in claims:
    print(claim.claim_id)   # Auto-generated UUID
    print(claim.text)       # Individual sentence
    print(claim.source)     # "generated"
    print("-" * 50)

# Output:
# claim_12345
# Paris is the capital of France.
# generated
# --------------------------------------------------
# claim_67890
# The Eiffel Tower is located in Paris.
# generated
```

#### Baseline RAG Pipeline (`src/generation/baseline_rag_pipeline.py`)

End-to-end RAG pipeline.

```python
from src.generation.baseline_rag_pipeline import BaselineRAGPipeline
from src.utils.config import load_config

config = load_config()
pipeline = BaselineRAGPipeline(config)

# Initialize pipeline (loads models, index, chunks)
pipeline.initialize(
    index_path="data/indexes/wikipedia_ivfflat.index",
    chunks_path="data/processed/chunks.jsonl"
)

# Run RAG query
query = "What is machine learning?"
results = pipeline.run(query)

# Access results
print(results['query'])                     # Original query
print(results['generated_text'])            # Generated answer
print(results['retrieved_chunks'])          # List of EvidenceChunk objects
print(results['retrieval_scores'])          # List of retrieval scores
print(results['claims'])                    # List of Claim objects
print(results['claim_evidence_pairs'])      # List of ClaimEvidencePair objects

# Generation metadata
print(results['generation_metadata']['avg_token_prob'])
print(results['generation_metadata']['token_logits'])
```

**Full Example:**

```python
from src.generation.baseline_rag_pipeline import BaselineRAGPipeline
from src.utils.config import load_config

# Load config and initialize pipeline
config = load_config()
pipeline = BaselineRAGPipeline(config)
pipeline.initialize(
    index_path="data/indexes/wikipedia_ivfflat.index",
    chunks_path="data/processed/chunks.jsonl"
)

# Run query
query = "What is the capital of France?"
results = pipeline.run(query)

# Print results
print(f"Query: {results['query']}")
print(f"Answer: {results['generated_text']}")
print(f"\nRetrieved Evidence:")
for i, (chunk, score) in enumerate(zip(results['retrieved_chunks'], results['retrieval_scores']), 1):
    print(f"{i}. [{score:.3f}] {chunk.text[:100]}...")

print(f"\nExtracted Claims:")
for claim in results['claims']:
    print(f"- {claim.text}")

print(f"\nClaim-Evidence Pairs:")
for pair in results['claim_evidence_pairs']:
    print(f"Claim: {pair.claim.text}")
    print(f"Evidence: {len(pair.evidence_chunks)} chunks")
```

## End-to-End Workflows

### Workflow 1: From Wikipedia to RAG Query

```python
from src.data_processing.wikipedia_parser import WikipediaParser
from src.data_processing.embedding_generator import EmbeddingGenerator
from src.retrieval.faiss_index_manager import FAISSIndexManager
from src.generation.baseline_rag_pipeline import BaselineRAGPipeline
from src.utils.config import load_config
import numpy as np

# Load config
config = load_config()

# Step 1: Parse Wikipedia to chunks
parser = WikipediaParser(config)
chunks = parser.parse_file(
    input_file="data/raw/wikipedia_en_100.jsonl",
    output_file="data/processed/chunks.jsonl"
)

# Step 2: Generate embeddings
generator = EmbeddingGenerator(config)
embeddings = generator.generate_embeddings(
    chunks_file="data/processed/chunks.jsonl",
    output_file="data/embeddings/embeddings.npy",
    batch_size=16
)

# Step 3: Build FAISS index
manager = FAISSIndexManager(config)
index = manager.build_index(embeddings, index_type="IVFFLAT", nlist=4096)
manager.save_index(index, "data/indexes/wikipedia_ivfflat.index")

# Step 4: Run RAG query
pipeline = BaselineRAGPipeline(config)
pipeline.initialize(
    index_path="data/indexes/wikipedia_ivfflat.index",
    chunks_path="data/processed/chunks.jsonl"
)
results = pipeline.run("What is machine learning?")

print(results['generated_text'])
```

### Workflow 2: Batch Query Processing

```python
from src.generation.baseline_rag_pipeline import BaselineRAGPipeline
from src.utils.config import load_config
import json

# Initialize pipeline
config = load_config()
pipeline = BaselineRAGPipeline(config)
pipeline.initialize(
    index_path="data/indexes/wikipedia_ivfflat.index",
    chunks_path="data/processed/chunks.jsonl"
)

# Load queries
queries = [
    "What is machine learning?",
    "What is the Eiffel Tower?",
    "Who invented the telephone?",
    "What is photosynthesis?"
]

# Process batch
results_batch = []
for query in queries:
    result = pipeline.run(query)
    results_batch.append({
        'query': result['query'],
        'answer': result['generated_text'],
        'num_claims': len(result['claims']),
        'avg_retrieval_score': sum(result['retrieval_scores']) / len(result['retrieval_scores'])
    })

# Save results
with open("results/batch_queries.json", "w") as f:
    json.dump(results_batch, f, indent=2)
```

### Workflow 3: Custom Retrieval + Generation

```python
from src.retrieval.dense_retriever import DenseRetriever
from src.generation.generator_wrapper import GeneratorWrapper
from src.generation.claim_extractor import ClaimExtractor
from src.utils.config import load_config

# Load config and components
config = load_config()
retriever = DenseRetriever(config)
generator = GeneratorWrapper(config)
extractor = ClaimExtractor()

# Load index and chunks
retriever.load_index("data/indexes/wikipedia_ivfflat.index")
retriever.load_chunks("data/processed/chunks.jsonl")

# Retrieve evidence
query = "What is machine learning?"
retrieved = retriever.retrieve(query, top_k=5)

# Build custom prompt
evidence_texts = [chunk.text for chunk, score in retrieved]
prompt = f"Answer the question using the provided context.\n\n"
prompt += f"Context:\n" + "\n".join(evidence_texts) + "\n\n"
prompt += f"Question: {query}\n\nAnswer:"

# Generate
result = generator.generate(prompt, max_new_tokens=100)
generated_text = result['generated_text']

# Extract claims
claims = extractor.extract_claims(generated_text)

# Print results
print(f"Query: {query}")
print(f"Answer: {generated_text}")
print(f"Claims: {[claim.text for claim in claims]}")
```

## Integration Patterns

### Pattern 1: Plug into Existing Systems

```python
# Your existing system
class MyQASystem:
    def __init__(self):
        self.rag_pipeline = None
    
    def setup_rag(self, config_path="config.yaml"):
        from src.generation.baseline_rag_pipeline import BaselineRAGPipeline
        from src.utils.config import load_config
        
        config = load_config(config_path)
        self.rag_pipeline = BaselineRAGPipeline(config)
        self.rag_pipeline.initialize(
            index_path="data/indexes/wikipedia_ivfflat.index",
            chunks_path="data/processed/chunks.jsonl"
        )
    
    def answer_question(self, question):
        if not self.rag_pipeline:
            raise RuntimeError("RAG pipeline not initialized")
        
        result = self.rag_pipeline.run(question)
        return {
            'answer': result['generated_text'],
            'confidence': result['generation_metadata']['avg_token_prob'],
            'sources': [chunk.metadata.get('title') for chunk in result['retrieved_chunks']]
        }

# Usage
qa = MyQASystem()
qa.setup_rag()
answer = qa.answer_question("What is machine learning?")
print(answer)
```

### Pattern 2: Custom Evidence Sources

```python
from src.retrieval.dense_retriever import DenseRetriever
from src.utils.data_structures import EvidenceChunk
from src.utils.config import load_config
import numpy as np

# Load config and retriever
config = load_config()
retriever = DenseRetriever(config)

# Create custom chunks
custom_chunks = [
    EvidenceChunk("custom_1", "Custom evidence 1", "custom_source", 0, {}),
    EvidenceChunk("custom_2", "Custom evidence 2", "custom_source", 1, {}),
]

# Generate embeddings for custom chunks
from sentence_transformers import SentenceTransformer
model = SentenceTransformer(config['models']['embedding_model'])
texts = [chunk.text for chunk in custom_chunks]
embeddings = model.encode(texts, convert_to_numpy=True)

# Build custom index
from src.retrieval.faiss_index_manager import FAISSIndexManager
manager = FAISSIndexManager(config)
index = manager.build_index(embeddings, index_type="FLAT")

# Set custom index and chunks in retriever
retriever.index = index
retriever.chunks = custom_chunks

# Now retriever uses your custom evidence
results = retriever.retrieve("query about custom evidence", top_k=2)
```

### Pattern 3: Month 3 Verifier Integration

```python
# This is how Month 3 verifier will use the RAG pipeline

from src.generation.baseline_rag_pipeline import BaselineRAGPipeline
from src.utils.config import load_config

# Initialize RAG
config = load_config()
pipeline = BaselineRAGPipeline(config)
pipeline.initialize(
    index_path="data/indexes/wikipedia_ivfflat.index",
    chunks_path="data/processed/chunks.jsonl"
)

# Run query
results = pipeline.run("What is the capital of France?")

# Extract data for verification
claim_evidence_pairs = results['claim_evidence_pairs']
generation_metadata = results['generation_metadata']

# For each claim-evidence pair
for pair in claim_evidence_pairs:
    claim_text = pair.claim.text
    evidence_texts = [chunk.text for chunk in pair.evidence_chunks]
    
    # Month 3 verifier logic here
    # - Check factual consistency (SummaC)
    # - Calculate entropy from generation_metadata['token_logits']
    # - Perform self-agreement checks
    # - Apply claim-evidence verification
    
    # Example placeholder
    is_hallucinated = verify_claim(claim_text, evidence_texts, generation_metadata)
    print(f"Claim: {claim_text}")
    print(f"Hallucinated: {is_hallucinated}")
```

## Error Handling

### Common Errors and Solutions

```python
from src.generation.baseline_rag_pipeline import BaselineRAGPipeline
from src.utils.config import load_config
import logging

# Setup logging
logging.basicConfig(level=logging.INFO)
logger = logging.getLogger(__name__)

try:
    config = load_config()
    pipeline = BaselineRAGPipeline(config)
    pipeline.initialize(
        index_path="data/indexes/wikipedia_ivfflat.index",
        chunks_path="data/processed/chunks.jsonl"
    )
    results = pipeline.run("What is machine learning?")
    
except FileNotFoundError as e:
    logger.error(f"Required file not found: {e}")
    logger.info("Make sure you've run the data processing pipeline first")
    # Fallback to default behavior or exit gracefully
    
except RuntimeError as e:
    if "CUDA out of memory" in str(e):
        logger.error("GPU out of memory. Try reducing batch_size in config")
        # Retry with CPU
        config['device'] = 'cpu'
        pipeline = BaselineRAGPipeline(config)
        pipeline.initialize(...)
        results = pipeline.run("What is machine learning?")
    else:
        raise
        
except Exception as e:
    logger.error(f"Unexpected error: {e}", exc_info=True)
    raise
```

### Validation and Assertions

```python
from src.utils.data_structures import EvidenceChunk, Claim, ClaimEvidencePair

# Validate chunks
def validate_chunk(chunk: EvidenceChunk):
    assert chunk.text, "Chunk text cannot be empty"
    assert chunk.chunk_id, "Chunk ID cannot be empty"
    assert isinstance(chunk.metadata, dict), "Metadata must be a dict"

# Validate retrieval results
def validate_retrieval(results, expected_k=5):
    assert len(results) <= expected_k, f"Expected at most {expected_k} results"
    for chunk, score in results:
        assert isinstance(chunk, EvidenceChunk)
        assert 0 <= score <= 1, f"Score {score} out of range [0, 1]"

# Validate generation results
def validate_generation(result):
    assert 'generated_text' in result
    assert 'generation_metadata' in result
    assert 'avg_token_prob' in result['generation_metadata']
```

## Best Practices

### 1. Configuration Management

```python
# DO: Use config files for all parameters
config = load_config("config.yaml")
pipeline = BaselineRAGPipeline(config)

# DON'T: Hardcode parameters
# pipeline = BaselineRAGPipeline(
#     embedding_model="sentence-transformers/all-MiniLM-L6-v2",
#     generator_model="google/flan-t5-base",
#     ...
# )
```

### 2. Resource Management

```python
# DO: Use context managers or explicit cleanup
from src.generation.baseline_rag_pipeline import BaselineRAGPipeline

pipeline = BaselineRAGPipeline(config)
try:
    pipeline.initialize(...)
    results = pipeline.run(query)
finally:
    # Clean up if needed
    del pipeline.retriever.model
    del pipeline.generator.model

# DON'T: Let models stay in memory indefinitely
```

### 3. Batch Processing

```python
# DO: Process in batches with progress tracking
from tqdm import tqdm

queries = [...]  # Large list of queries
batch_size = 10

for i in tqdm(range(0, len(queries), batch_size)):
    batch = queries[i:i+batch_size]
    results = [pipeline.run(q) for q in batch]
    # Save intermediate results
    save_results(results, f"batch_{i}.json")

# DON'T: Process all at once without checkpointing
# results = [pipeline.run(q) for q in queries]  # Risky if queries is large
```

### 4. Logging

```python
# DO: Use structured logging
import logging

logger = logging.getLogger(__name__)
logger.setLevel(logging.INFO)

logger.info(f"Running query: {query}")
logger.debug(f"Retrieved {len(chunks)} chunks")
logger.warning(f"Low retrieval score: {score}")
logger.error(f"Generation failed: {error}")

# DON'T: Use print statements
# print(f"Running query: {query}")
```

### 5. Testing

```python
# DO: Write unit tests for each component
import pytest

def test_chunk_extraction():
    from src.data_processing.text_chunker import TextChunker
    chunker = TextChunker()
    text = "Sentence one. Sentence two."
    chunks = chunker.chunk_text(text)
    assert len(chunks) == 2
    assert chunks[0] == "Sentence one."

# DON'T: Skip testing
```

---

**Last Updated:** 2025-10-25  
**System Version:** Month 2 Baseline  
**Documentation Status:** Complete

For quick start instructions, see [docs/month2_usage.md](docs/month2_usage.md).  
For technical details, see [docs/architecture_month2.md](docs/architecture_month2.md).
