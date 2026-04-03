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

Our system is designed to be knowledge-agnostic, supporting both static large-scale corpora and dynamic user-provided context. While our architecture natively supports **user-input context** as a temporary, high-relevance knowledge base (an optimization for specific workflows), we utilize a curated subset of the **English Wikipedia** (`enwiki-sample.xml`) as our primary, comprehensive knowledge base for general-purpose retrieval.

To ensure high-quality retrieval across these sources, we implemented a rigorous data ingestion pipeline:

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

---

## 3. The Trainless Verifier Module

The Verifier Module is the core innovation of our project. It fundamentally shifts away from computationally expensive, black-box "LLM-as-a-Judge" frameworks towards a rigorous, transparent ensemble of zero-shot signals. By inspecting both the model's internal statistical confidence and the external linguistic overlap, the `VerifierHub` issues a highly contextualized verdict.

### 3.1 Core Data Structure: Claim-Evidence Pair
The verifier operates strictly on atomic `ClaimEvidencePair` objects. This ensures sentence-level precision rather than paragraph-level ambiguity.
*   **Input:** `ClaimEvidencePair(claim: Claim, evidence_candidates: List[EvidenceChunk], generator_metadata: dict)`
*   **Output:** `VerifiedClaim` (includes confidence scores of the 4 independent signals and a final categorical verdict: Supported, Contradictory, or Low Confidence).

### 3.2 Intrinsic Uncertainty Detector

**Core Idea & Inspiration:**  
This detector is inspired by the **SelfCheckGPT** framework, which posits that LLMs behave like "stochastic parrots" when they lack factual knowledge. The core hypothesis is that factual hallucinations are not random errors but are preceded by high internal model uncertainty. When a model "knows" a fact, it allocates nearly all probability mass to a single, correct token. Conversely, when fabricating information, the probability distribution becomes "flat" or "entropic" as the model essentially guesses between multiple plausible-sounding but incorrect tokens. By measuring this "intrinsic" signal, we can detect hallucinations even without external evidence.

*   **Technical Highlights:**
    *   **Sub-word Token Mapping:** Accurately aligning character-level claims to LLM tokens (e.g., Llama's SentencePiece) requires handling prefix spaces and special control tokens.
    *   **Logit Stability:** We utilize **Log-Sum-Exp** normalization to prevent numerical overflow when processing raw model outputs.
    *   **Shannon Entropy ($H$):** A robust measure from Information Theory; $H=0$ implies total certainty, while higher values indicate "flat" distributions typical of guessing.

*   **Input:** Extracted `Claim` object and the `generator_metadata` (containing token-level logits).
*   **Method:** Utilizing **Log-Sum-Exp** for numerical stability, the system maps the claim's character span back to the original **Sub-word Tokens**. It then calculates the mean **Shannon Entropy** ($H$) over the probability distribution of the tokens:
    $$ H(x) = - \sum p(x) \log p(x) $$
*   **Output:** A scalar entropy score `{'mean_entropy': float}`. High mean entropy strongly correlates with fabricated facts.

**Code Example (`src/verification/intrinsic_uncertainty.py`):**
```python
def _calculate_entropy(self, logits: np.ndarray) -> float:
    # Applying Softmax with log-sum-exp stability
    max_logit = np.max(logits)
    exp_logits = np.exp(logits - max_logit)
    probs = exp_logits / np.sum(exp_logits)
    
    # Calculate Shannon Entropy
    entropy = -np.sum(probs * np.log(probs + self.epsilon))
    return float(entropy)
```

### 3.3 Retrieval-Grounded Heuristics

**Core Idea & Inspiration:**  
Drawing inspiration from traditional fact-checking benchmarks like **FEVER**, this module operates on the principle of **lexical grounding**. The idea is that for a claim to be considered "faithful" to its source, it must preserve the core "anchors" of the evidence—specifically named entities and numeric values. Hallucinations often involve "entity-swapping" (mixing up names) or "numeric drift" (incorrect dates or quantities). By strictly enforcing coverage of these anchors, we provide a fast, interpretable heuristic that catches the most common types of RAG hallucinations where the model deviates from the provided context.

*   **Technical Highlights:**
    *   **NER Fuzzy Matching:** Since surface forms can vary (e.g., "U.S." vs "United States"), we use a small Levenshtein distance threshold for matching.
    *   **Anchor Point Analysis:** Numbers and Proper Nouns are treated as "Anchors"—non-negotiable factual units that must intersect with source text for a claim to be considered faithful.
    *   **ROUGE-L (Longest Common Subsequence):** Unlike simple n-gram overlap, ROUGE-L accounts for sentence structure by measuring the longest sequence of words appearing in both claim and evidence in the same relative order.

*   **Input:** `Claim` text and `EvidenceChunk` object (retrieved document snippet).
*   **Method:**
    *   **Anchor Point Analysis (Entity/Number):** Uses **spaCy NER** and rule-based extractors to identify anchor points. Validates their presence in the evidence via **Fuzzy Matching** (Levenshtein distance).
    *   **Structural Lexical Overlap:** Computes the **ROUGE-L F1** score based on the Longest Common Subsequence between the strings.
*   **Output:** A dictionary of grounding scores `{entities: float, numbers: float, tokens_overlap: float}`.

**Code Example (`src/verification/retrieval_grounded.py`):**
```python
def _calculate_entity_coverage(self, claim: Claim, evidence: EvidenceChunk) -> float:
    # 1. Extract Named Entities using spaCy Dependency
    doc_claim = self.nlp(claim.text)
    entities = [ent.text for ent in doc_claim.ents]
    
    # 2. Validate presence in evidence
    matched = sum(1 for e in entities if self._fuzzy_match(e, evidence.text))
    
    # 3. Calculate coverage percentage (defaulting to 1.0 if no entities exist)
    return matched / len(entities) if entities else 1.0
```

### 3.4 Zero-Shot Natural Language Inference (NLI)

**Core Idea & Inspiration:**  
While lexical overlap is a strong signal, it cannot capture logical contradictions (e.g., adding a "not" to a sentence preserves almost all tokens but flips the meaning). This module is inspired by the **SummaC** and **Self-RAG** research, which repurposes Natural Language Inference (NLI) for factuality verification. By treating the evidence as a "premise" and the claim as a "hypothesis," we can use a model trained on logical relationships to detect if the evidence *actually supports* the claim. This adds a critical layer of semantic understanding that simple word-matching lacks.

*   **Technical Highlights:**
    *   **Cross-Encoding:** Unlike "bi-encoders" which score vectors, a Cross-Encoder passes both strings into the transformer simultaneously. This allows the model to capture deep semantic interactions (e.g., negations or coreference).
    *   **Veto Principle:** In our rule-based aggregator, the `contradiction` score from the NLI model acts as a "hard veto" that can override high lexical overlap scores.
    *   **DeBERTa-v3 Architecture:** A state-of-the-art encoder with "Disentangled Attention," significantly better at logical reasoning than standard BERT/RoBERTa models.

*   **Input:** `Claim` text (Hypothesis) and `EvidenceChunk` text (Premise).
*   **Method:** Utilizing a **DeBERTa-v3 Cross-Encoder**, the system performs a zero-shot classification of the logical relationship (entailment/contradiction) between the evidence and the claim.
*   **Output:** A probability distribution over three classes: `{'entailment': float, 'contradiction': float, 'neutral': float}`.

### 3.5 Self-Agreement Detector

**Core Idea & Inspiration:**  
This detector is based on the **Self-Consistency (CoT-SC)** principle. The intuition is that for internal knowledge-based questions, an LLM that "knows" the answer will converge on the same factual claim regardless of slight variations in the prompt or decoding path. However, if the model is hallucinating (guessing), it will generate different, inconsistent stories across multiple independent runs. By checking if a claim "agrees" with other stochastic samples of the same model, we can verify its stability and reliability.

*   **Technical Highlights:**
    *   **Stochastic Sampling:** We adjust the model's **temperature** ($\tau > 0.7$) to generate multiple distinct paths, rather than a single greedy sequence.
    *   **Majority Vote Strategy:** We determine if the claim follows the "centroid" of the model's internal knowledge; a claim appearing in $<30\%$ of samples is flagged as highly unreliable.
    *   **Semantic Consistency:** Not restricted to exact string matching; semantic clustering is used to group similar claims before voting.

*   **Input:** Original `Claim` text and $N$ stochastic `response_samples` generated with high temperature.
*   **Method:** The module uses **Stochastic Sampling** to generate $N$ alternative responses. It determines the semantic consensus using a **Majority Vote Strategy** across clustered claims to verify the factual stability of the original response.
*   **Output:** An agreement score `{'agreement_ratio': float}` based on semantic consensus.

### 3.6 Rule-Based Aggregation
The four independent signals are passed to the `VerifierHub`'s aggregation engine. 

*   **Technical Highlights:**
    *   **Veto Logic:** Specific critical failure signals (e.g., NLI Contradiction or Extreme Uncertainty) can unilaterally override positive signals.
    *   **Signal Normalization:** Raw telemetry from heterogeneous detectors (probabilities, ratios, entropy) are scaled to a unified 0-1 range before fusion.

*   **Input:** All preceding detector outputs (`mean_entropy`, grounding scores, NLI distribution, agreement ratio).
*   **Method:** The engine normalizes the signals and applies **Veto Logic** to determine if any catastrophic failures exist. If no vetoes are triggered, it computes a weighted aggregation of the scores to issue a final verdict.
*   **Output:** A finalized `VerifiedClaim` with a categorical verdict: **Supported**, **Contradictory**, or **Low Confidence**.
