# Presentation Workflow: First Term Technical Report

**Project:** Hallucination Detection & Mitigation for LLMs
**Date:** November 2025

---

## Slide 1: Title Slide
- **Title:** Hallucination Detection & Mitigation for Large Language Models
- **Subtitle:** A Trainless, Multi-Signal Verification Approach
- **Team Members:** [Member 1], [Member 2]
- **Date:** November 2025

---

## Slide 2: Introduction & Problem Statement
- **The Problem:**
    - Large Language Models (LLMs) are prone to "hallucinations" (generating factually incorrect information).
    - Even in Retrieval-Augmented Generation (RAG), models can misinterpret or ignore retrieved evidence.
- **The Goal:**
    - Develop a **lightweight, trainless verifier system**.
    - Operate as a post-hoc safety layer to detect and mitigate hallucinations without fine-tuning massive "judge" models.

---

## Slide 3: System Architecture
- **Core Pipeline:** Generator-Retriever-Verifier
    1.  **Retrieval:** Fetch relevant evidence from Wikipedia.
    2.  **Generation:** LLM generates a response + captures token-level metadata (logits).
    3.  **Verification:** Decompose response into claims and verify each against evidence.
- **Key Innovation:**
    - **Trainless:** No expensive model training required.
    - **Multi-Signal:** Aggregates multiple zero-shot signals (Uncertainty, Heuristics, NLI, Consistency).
- **Visual:** [System Architecture Diagram]

---

## Slide 4: Implementation - Data Processing & Retrieval
- **Foundation:** English Wikipedia Corpus.
- **Method:**
    - **Encoding:** `sentence-transformers/all-MiniLM-L6-v2` (384 dim).
    - **Indexing:** FAISS (Facebook AI Similarity Search) for efficient similarity search.
    - **Retrieval:** Dense Retriever fetches top-k semantic matches.
- **Status:** Completed.

---

## Slide 5: Implementation - Baseline RAG Pipeline
- **"Verifier-Aware" Generation:**
    - Uses Llama-3-8B (or similar).
    - **Crucial Step:** Captures **token-level logits** during generation.
    - **Structure:** Decomposes output into "Claim-Evidence Pairs" using spaCy.
- **Why?** Enables granular, sentence-level verification.
- **Status:** Completed.

---

## Slide 6: Implementation - Verifier Module (Part 1)
- **Signal 1: Intrinsic Uncertainty**
    - **Logic:** If the model is "unsure" (high entropy), it's more likely to hallucinate.
    - **Metric:** Shannon Entropy ($H = -\sum p \log p$) over token probabilities.
- **Signal 2: Retrieval-Grounded Heuristics**
    - **Logic:** Does the claim match the evidence text?
    - **Metrics:**
        - **Entity Coverage:** Are names/places in the claim present in the evidence?
        - **Token Overlap:** ROUGE-L F1 score.
- **Status:** In Progress/Completed.

---

## Slide 7: Technical Challenges
- **Token Alignment:**
    - Mapping character offsets (e.g., "Eiffel Tower") back to specific LLM tokens is complex due to sub-word tokenization.
    - *Solution:* Implemented fuzzy matching logic.
- **Entity Normalization:**
    - "USA" vs. "United States" mismatch causes false negatives.
    - *Solution:* Planning a tiered matching approach (Acronyms, Alias Dictionary).
- **Resource Constraints:**
    - Processing millions of Wikipedia chunks requires significant GPU resources.
    - *Solution:* Optimized batch sizes and memory management (`load_in_8bit`).

---

## Slide 8: Future Work - Next Steps (Month 4)
- **New Signals:**
    - **Zero-Shot NLI:** Use DeBERTa to detect logical contradictions (Entailment/Contradiction).
    - **Self-Agreement:** Sample multiple responses to check for consistency.
- **Active Mitigation:**
    - **Re-ranking:** Adjust evidence order based on verification.
    - **Re-prompting:** Ask LLM to self-correct low-confidence claims.
    - **Filtering:** Suppress unsupported claims.

---

## Slide 9: Evaluation Plan (Month 5)
- **Benchmarks:**
    - **RAGTruth:** Specifically for RAG hallucinations.
    - **CiteBench:** For citation quality and attribution.
- **Metrics:**
    - Detection Accuracy, Precision, Recall.
    - Comparison with **Ragas** framework (Faithfulness, Answer Relevancy).
- **Aggregation:** Develop rule-based logic to combine all signals into a final verdict.

---

## Slide 10: Conclusion
- **Summary:**
    - Successfully built the RAG foundation and initial verifier signals.
    - Identified key challenges (Entity Normalization) and solutions.
- **Next Phase:**
    - Implement advanced signals (NLI, Consistency).
    - Focus on active mitigation and rigorous evaluation.
- **Q&A**
