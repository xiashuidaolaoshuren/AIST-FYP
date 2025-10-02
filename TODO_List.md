# Project To-Do List

This to-do list breaks down the 6-month project plan into actionable tasks, organized by month.

---

### **Month 1: Research & Planning**

-   [X] **Literature Review:**
    -   [X] Read and summarize key papers on trainless hallucination detection (e.g., SelfCheckGPT, CoVe).
    -   [X] Consolidate findings and identify the most promising signals for the verifier module.
-   [ ] **Data Sourcing:**
    -   [ ] Download the English Wikipedia corpus.
    -   [ ] Download evaluation benchmarks: `TruthfulQA`, `RAGTruth`, and `FEVER`.
    -   [ ] Organize all datasets in a structured project directory.
-   [X] **System Architecture:**
    -   [X] Review and confirm the finalized trainless architecture design.
    -   [X] Define the precise inputs and outputs for each signal detector in the verifier module.
-   [X] **Environment Setup:**
    -   [X] Create a dedicated Python environment (e.g., conda, venv).
    -   [X] Install core libraries: `transformers`, `faiss-cpu` (or `faiss-gpu`), `torch`, `datasets`.
    -   [X] Write a script to verify GPU access and CUDA setup.

---

### **Month 2: Baseline & Retrieval Module**

-   [ ] **Data Preparation:**
    -   [ ] Write and run a script to parse and clean the Wikipedia XML dump.
    -   [ ] Write a script to chunk the cleaned text into sentence-level fragments.
    -   [ ] Generate embeddings for all chunks using a pre-trained sentence-transformer model.
    -   [ ] Build and save the FAISS index for efficient similarity search.
-   [ ] **Retriever Implementation:**
    -   [ ] Implement a `DenseRetriever` class that takes a query and returns the top-k evidence chunks from the FAISS index.
-   [ ] **Baseline RAG Implementation:**
    -   [ ] Integrate a generator LLM (e.g., from Hugging Face) with the `DenseRetriever`.
    -   [ ] Create a simple pipeline that takes a user query and returns a generated answer with retrieved context.

---

### **Month 3: Verifier Module - Signal Implementation (Part 1)**

-   [ ] **Intrinsic Uncertainty Detector:**
    -   [ ] Implement a function to extract token-level logits/probabilities from the generator's output.
    -   [ ] Implement a module to calculate token-level entropy and length-normalized perplexity for each claim.
-   [ ] **Retrieval-Grounded Heuristics:**
    -   [ ] Implement an `evidence_coverage` function that calculates the percentage of named entities and noun phrases from a claim that appear in the evidence.
    -   [ ] Implement a `citation_span_integrity` function that measures the token overlap between a claim and its direct citation.
-   [ ] **Integration:**
    -   [ ] Integrate these two detectors into the main pipeline to process claims after generation.

---

### **Month 4: Verifier Module - Signal Implementation (Part 2)**

-   [ ] **Zero-Shot NLI Contradiction Detector:**
    -   [ ] Load the pre-trained `MoritzLaurer/DeBERTa-v3-base-mnli-fever-anli` model from Hugging Face.
    -   [ ] Implement a module that takes a (claim, evidence) pair and returns the probabilities for "entailment," "contradiction," and "neutral."
-   [ ] **Self-Agreement Detector:**
    -   [ ] Implement a function to generate `k` different responses for the same query using stochastic sampling (e.g., temperature > 0).
    -   [ ] Implement a module to measure the semantic consistency or claim variability across the `k` responses.
-   [ ] **Integration:**
    -   [ ] Add these two new detectors to the verifier module.

---

### **Month 5: Detector Evaluation & Mitigation (Optional)**

-   [ ] **Rule-Based Aggregation:**
    -   [ ] Design and implement a `RuleBasedAggregator` that combines the outputs of all four signal detectors.
    -   [ ] Define explicit rules and thresholds to classify each claim as "Supported," "Contradictory," or "Low Confidence."
-   [ ] **End-to-End Detector Evaluation:**
    -   [ ] Set up an evaluation harness to run the full system on the `TruthfulQA` and `RAGTruth` benchmarks.
    -   [ ] Run the evaluation and collect the results.
-   [ ] **Performance Analysis:**
    -   [ ] Calculate key metrics: detection accuracy, precision, recall, and F1-score for identifying hallucinations.
    -   [ ] Analyze the results to identify strengths and weaknesses of the detector.
-   [ ] **(Optional) Simple Mitigation:**
    -   [ ] If time permits, implement a simple `Flagging` module that adds warnings to low-confidence or contradictory claims in the final output.

---

### **Month 6: Finalization & Documentation**

-   [ ] **Ablation Study:**
    -   [ ] Design and run experiments to analyze the contribution of each trainless signal.
    -   [ ] Systematically disable each of the four detectors one by one and re-run the evaluation to measure the drop in performance.
-   [ ] **Final Report & Demo:**
    -   [ ] Write the final project report, including sections on architecture, methodology, results, ablation study, and conclusions.
    -   [ ] Create a presentation summarizing the project.
    -   [ ] Prepare a compelling live demo or a recorded video showcasing the system's ability to detect hallucinations.
-   [ ] **Code Cleanup & Handoff:**
    -   [ ] Refactor the codebase for clarity and readability.
    -   [ ] Add comprehensive comments and docstrings to all functions and classes.
    -   [ ] Create a `README.md` file with instructions on how to set up and run the project.
    -   [ ] Ensure all project artifacts are committed to the version control system.
