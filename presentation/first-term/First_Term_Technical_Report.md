# First Term Technical Report

**Date:** November 20, 2025
**Project:** Hallucination Detection & Mitigation for LLMs
**Author:** Technical Lead (Member 2)

---

## 1. System Design

Our project aims to address the critical issue of factual hallucinations in Large Language Models (LLMs), particularly in Retrieval-Augmented Generation (RAG) scenarios. We have designed a modular, **trainless verifier system** that operates as a post-hoc safety layer.

![System Architecture](../../System_Architecture_Design%20_%20Mermaid%20Chart-2025-10-03-081957.png)

### Core Architecture: Generator-Retriever-Verifier
The system follows a three-stage pipeline designed to ensure that every generated claim is grounded in retrieved evidence:

1.  **Retrieval:** Fetches relevant documents from a knowledge base (Wikipedia) based on the user's query using a **Dense Retriever**. This component encodes the query into a vector space and performs an approximate nearest neighbor search to find semantically similar evidence chunks.
2.  **Generation:** An LLM generates a response using the retrieved context. Crucially, this stage also captures **token-level metadata** (logits/probabilities) which is essential for the downstream uncertainty analysis.
3.  **Verification:** A dedicated "Verifier Module" analyzes the generated claims against the retrieved evidence. This module decomposes the response into atomic claims and assigns a confidence score to each based on multiple signals.

### The Trainless Verifier
Unlike traditional approaches that require training massive "judge" models, our verifier aggregates multiple zero-shot signals to assess factuality without fine-tuning. The four key signals are:

1.  **Intrinsic Uncertainty:** Measures the model's internal confidence using token-level statistics. We specifically calculate **Shannon Entropy** ($H = -\sum p \log p$) over the probability distribution of the generated tokens. High entropy indicates the model was "unsure" about its output, which correlates with hallucination.
2.  **Retrieval-Grounded Heuristics:** Quantifies the overlap between the generated claim and the evidence using three sub-metrics:
    *   **Entity Coverage:** Checks if named entities (People, Organizations, Locations) in the claim are present in the evidence.
    *   **Number Coverage:** Verifies that numeric values in the claim match the source text.
    *   **Token Overlap:** Calculates the **ROUGE-L F1 score** (based on Longest Common Subsequence) to measure lexical similarity.
3.  **Zero-Shot NLI (Planned):** Uses an off-the-shelf Natural Language Inference model to classify the relationship between a claim and its evidence as "Entailment", "Contradiction", or "Neutral".
4.  **Self-Agreement (Planned):** Checks the consistency of the generated information across multiple stochastic samples (Self-Consistency).

These signals are combined via a **Rule-Based Aggregator** to produce a final verdict (Supported, Contradictory, or Low Confidence).

---

## 2. Implementation Progress

We have successfully completed the foundational phases of the project (Months 1 & 2) and have made significant progress on the Verifier Module (Month 3).

### A. Data Processing & Retrieval (Completed)

This component forms the foundation of our RAG system, responsible for ingesting the massive Wikipedia corpus and enabling real-time, semantic search. It transforms raw text into a searchable vector space, allowing the system to retrieve relevant evidence based on meaning rather than just keyword matching. We utilized a high-performance vector database (FAISS) to handle the scale of millions of document chunks, ensuring low-latency retrieval essential for an interactive system.

*   **Input:** A raw user query (e.g., "Who founded the FEVER dataset?").
*   **Method:**
    1.  **Encoding:** The query is encoded into a high-dimensional vector (384 dimensions) using the `sentence-transformers/all-MiniLM-L6-v2` model.
    2.  **Indexing:** We use **FAISS** (Facebook AI Similarity Search) with an `IndexFlatIP` (Inner Product) index. Since embeddings are normalized, this is equivalent to Cosine Similarity.
    3.  **Search:** The system performs an exact nearest neighbor search to find the top-k most similar chunks from the Wikipedia corpus.
*   **Output:** A ranked list of `EvidenceChunk` objects containing the text and metadata.

**Code Example (`src/retrieval/dense_retriever.py`):**
```python
def retrieve(self, query: str, top_k: int = 5) -> List[EvidenceChunk]:
    # 1. Encode query
    query_embedding = self.encoder.encode([query], normalize_embeddings=True)
    
    # 2. Search FAISS index
    scores, indices = self.index.search(query_embedding, top_k)
    
    # 3. Construct results
    results = []
    for idx, score in zip(indices[0], scores[0]):
        metadata = self.metadata[idx]
        results.append(EvidenceChunk(..., score_dense=float(score)))
    return results
```

### B. Baseline RAG Pipeline (Completed)

The Baseline RAG Pipeline integrates the retrieval mechanism with a generative Large Language Model (LLM) to produce grounded answers. Beyond standard generation, this pipeline is engineered to be "verifier-aware": it captures critical metadata during generation—specifically token-level logits—and structures the output into atomic "claim-evidence pairs". This structured output is the prerequisite for our downstream verification logic, enabling granular fact-checking at the sentence level.

*   **Input:** User query.
*   **Method:**
    1.  **Retrieval:** Calls the `DenseRetriever`.
    2.  **Generation:** Uses a `GeneratorWrapper` (e.g., Llama-3-8B) to generate a response. Crucially, we set `output_scores=True` to capture **logits** for uncertainty analysis.
    3.  **Claim Extraction:** We use **spaCy** to segment the generated response into sentences (atomic claims).
    4.  **Pairing:** Each claim is paired with the retrieved evidence.
*   **Output:** A dictionary containing the `draft_response` and a list of `ClaimEvidencePair` objects.

**Code Example (`src/pipelines/baseline_rag.py`):**
```python
def run(self, query: str, top_k: int = 5) -> Dict:
    # 1. Retrieve
    evidence_chunks = self.retriever.retrieve(query, top_k=top_k)
    
    # 2. Generate with metadata (logits)
    gen_output = self.generator.generate_with_metadata(
        prompt=query, evidence_chunks=evidence_chunks
    )
    
    # 3. Extract Claims
    claims = extract_claims(gen_output['text'])
    
    # 4. Pair & Return
    return {
        'draft_response': gen_output['text'],
        'claim_evidence_pairs': [
            ClaimEvidencePair(claim, evidence_chunks) for claim in claims
        ]
    }
```

**Result Demo:**
!

### C. Verifier Module - Part 1 (In Progress/Completed)

The Verifier Module is the core innovation of our project, designed to assess the factual reliability of generated claims without relying on expensive, black-box "LLM-as-a-Judge" calls. In this first phase, we have implemented two complementary signal detectors: one that looks "inward" at the model's own confidence (Intrinsic Uncertainty) and one that looks "outward" at the alignment between the claim and the source text (Retrieval-Grounded Heuristics). These signals provide the initial layers of our multi-signal verification strategy.

#### 1. Intrinsic Uncertainty Detector
*   **Input:** Extracted `Claim` object and the `generator_metadata` (containing token logits).
*   **Method:**
    1.  **Alignment:** Maps the claim's character span (e.g., chars 0-50) to the specific tokens generated by the LLM.
    2.  **Entropy Calculation:** Computes the Shannon Entropy ($H$) for each token's probability distribution.
    $$ H(x) = - \sum p(x) \log p(x) $$
    3.  **Aggregation:** Returns the mean entropy over the claim's tokens.
*   **Output:** A dictionary `{'mean_entropy': float}`.

**Code Example (`src/verification/intrinsic_uncertainty.py`):**
```python
def _calculate_entropy(self, logits: np.ndarray) -> float:
    # Softmax with log-sum-exp stability
    max_logit = np.max(logits)
    exp_logits = np.exp(logits - max_logit)
    probs = exp_logits / np.sum(exp_logits)
    
    # Shannon Entropy
    entropy = -np.sum(probs * np.log(probs + self.epsilon))
    return float(entropy)
```

#### 2. Retrieval-Grounded Heuristics
*   **Input:** `Claim` text and `EvidenceChunk` text.
*   **Method:**
    1.  **Entity Coverage:** Uses **spaCy NER** to find entities in the claim and checks if they exist in the evidence (fuzzy match).
    2.  **Token Overlap:** Computes the **Longest Common Subsequence (LCS)** to calculate the ROUGE-L F1 score.
*   **Output:** A dictionary with scores for `entities`, `numbers`, and `tokens_overlap`.

**Code Example (`src/verification/retrieval_grounded.py`):**
```python
def _calculate_entity_coverage(self, claim, evidence) -> float:
    # Extract entities
    doc_claim = self.nlp(claim.text)
    entities = [ent.text for ent in doc_claim.ents]
    
    # Check presence in evidence
    matched = sum(1 for e in entities if self._fuzzy_match(e, evidence.text))
    
    return matched / len(entities) if entities else 1.0
```

---

## 3. Difficulties and Limitations

### Technical Challenges
-   **Token Alignment Complexity:** Mapping the character offsets of a sentence (e.g., "The Eiffel Tower is tall") back to the specific tokens produced by the LLM (e.g., `["The", " E", "iffel", " Tower"]`) proved difficult due to sub-word tokenization and special characters (like SentencePiece's ` `). We had to implement a fuzzy matching logic with tolerance to ensure accurate logit extraction.
-   **Entity Normalization Challenge:** A significant limitation identified in the Retrieval-Grounded Detector is the "Entity Surface Form Variation" problem. The current system uses fuzzy substring matching, which fails to recognize that different textual representations (e.g., "USA" vs. "United States", "WHO" vs. "World Health Organization") refer to the same real-world entity. This leads to false negatives in the entity coverage metric, where valid evidence is rejected because the surface forms do not match. We plan to address this by implementing a tiered matching approach including acronym expansion and an alias dictionary.
-   **Resource Constraints:** Processing the entire English Wikipedia dump was highly resource-intensive. Generating embeddings for millions of chunks required significant GPU time. Our development environment (RTX 3070Ti) imposes limits on the batch size and the size of the LLM we can run locally, requiring careful memory management (e.g., `load_in_8bit`).

### System Limitations
-   **Lexical Gap in Retrieval:** We currently rely solely on dense retrieval. While effective for semantic matching, it sometimes struggles with exact keyword matching (the "lexical gap"), leading to missed evidence for specific proper nouns or rare terms.
-   **Lack of Reranking:** We have not yet implemented a reranker. The top-k retrieved documents are used directly, which means the most relevant evidence might sometimes be ranked lower than less relevant but semantically similar chunks.
-   **Verifier Latency:** Calculating intrinsic uncertainty requires access to the model's logits, and heuristic checks add computational overhead. This increases the end-to-end latency of the system compared to a standard RAG pipeline.

---

## 4. Future Works

The next phase of the project (Months 4-6) will focus on completing the verifier, evaluating the system, and refining the user experience.

### Immediate Next Steps (Month 4)
-   **Zero-Shot NLI:** Implement the NLI signal using the `MoritzLaurer/DeBERTa-v3-base-mnli-fever-anli` model. This will allow us to detect logical contradictions even when lexical overlap is low.
-   **Self-Agreement:** Implement the self-consistency check by sampling multiple responses (e.g., with `temperature > 0.7`) and measuring the semantic variance between them.


### Evaluation & Refinement (Month 5)
-   **Rule-Based Aggregation:** Develop the logic to combine the four signals (Entropy, Heuristics, NLI, Consistency) into a single confidence score. We will need to tune the thresholds for each signal based on validation data.
-   **Ragas Integration:** Integrate the **Ragas** framework to systematically evaluate `faithfulness` and `answer_relevancy`. This will provide an external benchmark to validate our custom verifier's performance.
-   **Benchmarking:** Run the full system on **CiteBench** and **RAGTruth** benchmarks to quantify detection accuracy, precision, and recall.
-   **Hallucination Mitigation:** Beyond detection, we aim to implement active mitigation strategies. This involves using the verifier's negative feedback to trigger corrective actions, such as re-ranking retrieved documents, re-prompting the LLM to self-correct, or filtering out unsupported claims from the final response.


### Optimization
-   **Hybrid Retrieval:** Investigate adding a **BM25 sparse retriever** to complement the dense retriever (Hybrid Search) to address the lexical gap.
-   **UI Development:** Build the "Confidence UI" to visualize the verifier's output for end-users, displaying the confidence score and the breakdown of the underlying signals.
