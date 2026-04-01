# Final Technical Report

**Date:** April 1, 2026
**Project:** Hallucination Detection & Mitigation for LLMs
**Author:** Technical Lead (Member 2)

---

## 1. System Design

Our project aims to address the critical issue of factual hallucinations in Large Language Models (LLMs), particularly within Retrieval-Augmented Generation (RAG) frameworks. Building on the progress from the first term, we have successfully developed a complete **four-stage end-to-end pipeline** that acts as a post-hoc safety layer, combining a **trainless verifier** with a proactive **mitigation module**.

![System Architecture](../../System_Architecture_Design%20_%20Mermaid%20Chart-2025-11-22-041757.png)

### Core Architecture: Generator-Retriever-Verifier-Mitigator
The system follows a structured pipeline designed to ensure that generating claims are grounded in retrieved evidence, verified for factuality, and automatically corrected if evidence proves contradictory:

1.  **Baseline RAG (Retrieval & Generation):** Fetches relevant documents from our knowledge base (Wikipedia) based on the user's query utilizing a **Hybrid Retriever**. The language model then generates a unified response while strictly capturing **token-level metadata** (logits/probabilities). 
2.  **Claim Extraction:** A decomposition utility breaks the generated response into atomic, independent claims, ensuring precise sentence-level processing.
3.  **Verification (Trainless Verifier):** The `VerifierHub` receives these claim-evidence pairs. This module evaluates each claim and assigns a multidimensional confidence score by aggregating several independent, zero-shot signals.
4.  **Mitigation:** Finally, the `MitigationOrchestrator` receives the verified claims and applies rule-based corrective actions—such as filtering unsupported claims, reprompting the generator, or reranking evidence—to construct a factually consistent final response.

### The Trainless Verifier and Mitigation Philosophy
Unlike approaches that require training massive, opaque "LLM-as-a-Judge" models, our verifier aggregates multiple zero-shot signals. This approach minimizes latency and maintains computational efficiency:

1.  **Intrinsic Uncertainty:** Measures the model's internal confidence using token-level statistics, specifically calculating **Shannon Entropy** ($H = -\sum p \log p$) over the vocabulary distribution. High entropy often strongly correlates with hallucination.
2.  **Retrieval-Grounded Heuristics:** Quantifies the overlap between the generated claim and the evidence, encompassing **Entity Coverage**, **Number Coverage**, and **Token Overlap** (ROUGE-L).
3.  **Zero-Shot NLI:** Uses an off-the-shelf Natural Language Inference model to classify the formal relationship (Entailment, Contradiction, Neutral) between a retrieved source document and an atomic claim.
4.  **Self-Agreement:** Validates factual consistency across multiple generated stochastic samples.

These signals converge into a **Rule-Based Aggregator** that issues a finalized verdict. The **Mitigation Module** then acts transparently on this verdict without retraining, relying on predetermined procedures (e.g., omitting contradicting sentences or replacing sections entirely) to "edit" the final answer.

---

## 2. Technical Methodology

### 2.1 Knowledge Base: The Wikipedia Corpus

Our system utilizes a curated subset of the English Wikipedia (`enwiki-sample.xml`) as its primary knowledge base. To ensure high-quality retrieval, we implemented a rigorous data ingestion pipeline:

*   **Parsing & Cleaning:** We developed a custom `WikipediaParser` to extract plain text from the XML dump, stripping out redirects, disambiguation pages, and complex wikitext elements (templates, HTML tags).
*   **Semantic Chunking:** The cleaned text is segmented using spaCy's sentencizer into atomic units (chunks). We apply a length filter to ensure each chunk contains enough semantic context for the retriever without overwhelming the generator's context window.
*   **Vector Embedding:** We utilize the `sentence-transformers/all-MiniLM-L6-v2` model to transform each text chunk into a 384-dimensional dense vector. This step captures the semantic essence of the information, allowing the system to match queries with evidence based on conceptual meaning rather than just shared keywords.
*   **Indexing:** To enable efficient retrieval over the large-scale Wikipedia corpus, we maintain two parallel indices:
    *   **Dense Index (FAISS):** The generated embeddings are stored in a **FAISS** `IndexFlatIP` (Inner Product) index. This enables high-speed, approximate nearest-neighbor searches in the latent vector space.
    *   **Sparse Index (BM25):** We construct a lexical index using the **BM25Okapi** algorithm. This involves pre-tokenizing the corpus and calculating the inverse document frequency (IDF) for all terms, ensuring high-precision retrieval for rare entities and exact keyword matches.

### 2.2 The RAG Module

The Baseline RAG Pipeline (`BaselineRAGPipeline`) is the critical starting point of our system. It is engineered to fulfill standard retrieval duties while remaining "verifier-aware" by storing state and detailed probabilistic metadata for downstream modules. 

#### A. Hybrid Retrieval Strategy

Our foundation relies on the synthesis of dense semantic matching and sparse lexical keyword search to minimize the "lexical gap" identified in the first term.

*   **Dense Semantic Search (FAISS):** We continue to use high-dimensional Sentence-Transformers encoding into an `IndexFlatIP`. This guarantees excellent recall for conceptual and synonymous relationships.
*   **Sparse Lexical Search (BM25):** We incorporated `BM25Okapi` indexing integrated within our `BM25Retriever`. The BM25 index targets specific noun phrases and rare acronyms that dense embeddings frequently obscure.

**Retrieval Data Flow and Fusion:**
We execute parallel searches across both indexes. To standardize the output, we calculate the combination of scores utilizing **Linear Scoring** or **Reciprocal Rank Fusion (RRF)**:
$$ S_{RRF} = \frac{1}{k + \text{rank}_{dense}} + \frac{1}{k + \text{rank}_{BM25}} $$
The top-$k$ returned segments are securely passed to the generator.

**Code Example (`src/retrieval/hybrid_retriever.py`):**
```python
def retrieve(self, query: str, top_k: int = 5) -> List[EvidenceChunk]:
    # Parallel retrieval execution
    dense_results = self.dense_retriever.retrieve(query, top_k)
    bm25_results = self.bm25_retriever.retrieve(query, top_k)
    
    # Score normalization and Reciprocal Rank Fusion
    dense_norm = self._normalize_scores(dense_results, 'score_dense')
    bm25_norm = self._normalize_scores(bm25_results, 'score_bm25')
    
    # Merge, rerank, and select top_k documents
    fused_results = self._apply_rrf(dense_norm, bm25_norm)
    return fused_results[:top_k]
```

#### B. Generation with Metadata Harvesting

The generator is responsible for drafting the preliminary response from the fused user query and context. Our custom wrapper (`GeneratorWrapper`) implements metadata harvesting during sequence decoding. 

*   **Logit Extraction:** By forcing `output_scores=True` inside standard `generate` calls, we intercept the unnormalized output logs (logits) iteratively. 
*   **Token Probabilities:** Our generator maintains exact indices and maps raw text characters to internal SubWord tokens, securing accurate references for verification algorithms. 

**Code Example (`src/generation/generator_wrapper.py`):**
```python
def generate_with_metadata(self, prompt: str, evidence_chunks: List) -> Dict:
    inputs = self.tokenizer(prompt, return_tensors='pt', truncation=True)
    
    # Collect generation and store logits to trace internal uncertainty
    outputs = self.model.generate(
        **inputs,
        output_scores=True,
        return_dict_in_generate=True,
        max_new_tokens=150
    )
    
    return {
        'text': self.tokenizer.decode(outputs.sequences[0], skip_special_tokens=True),
        'metadata': outputs.scores  # Vital dataset for the Intrinsic Uncertainty Detector
    }
```

#### C. Atomic Claim Extraction

Working with entire paragraphs drastically degrades verification reliability. Post-generation, the system employs spaCy Dependency Parsing to intelligently fragment raw LLM text outputs into `Claim` objects. 

Every sentence boundary is split, resulting in individual properties ("The Eiffel Tower was built in 1889", "It is located in Paris"). These claims are joined with the full array of retrieved `$k$` instances to assemble `ClaimEvidencePair` objects. This structural decomposition transforms unstructured responses into formal, verifiable propositions for the Trainless Verifier.
