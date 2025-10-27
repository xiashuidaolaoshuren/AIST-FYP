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

-   [x] **Data Preparation:**
    -   [x] Write and run a script to parse and clean the Wikipedia XML dump.
    -   [x] Write a script to chunk the cleaned text into sentence-level fragments.
    -   [x] Generate embeddings for all chunks using a pre-trained sentence-transformer model.
    -   [x] Build and save the FAISS index for efficient similarity search.
-   [x] **Retriever Implementation:**
    -   [x] Implement a `DenseRetriever` class that takes a query and returns the top-k evidence chunks from the FAISS index.
-   [x] **Baseline RAG Implementation:**
    -   [x] Integrate a generator LLM (e.g., from Hugging Face) with the `DenseRetriever`.
    -   [x] Create a simple pipeline that takes a user query and returns a generated answer with retrieved context.

---

### **Month 3: Verifier Module (Part 1) & Presentation Preparation**

-   **Member 1 (Research & Presentation Focus):**
    -   [ ] Conduct a deep dive into the theoretical foundations of intrinsic uncertainty (entropy, perplexity) in LLMs.
    -   [ ] Research and document best practices for implementing retrieval-grounded heuristics, analyzing trade-offs between different overlap metrics (e.g., lexical vs. semantic).
    -   [ ] Begin drafting the methodology section of the first term report.
-   **Member 2 (Development & Experimentation Focus):**
    -   [ ] **Intrinsic Uncertainty Detector:**
        -   [ ] Implement a function to extract token-level logits/probabilities from the generator's output.
        -   [ ] Implement a module to calculate token-level entropy and length-normalized perplexity for each claim.
    -   [ ] **Retrieval-Grounded Heuristics:**
        -   [ ] Implement an `evidence_coverage` function that calculates the percentage of named entities and noun phrases from a claim that appear in the evidence.
        -   [ ] Implement a `citation_span_integrity` function that measures the token overlap between a claim and its direct citation.
    -   [ ] **Integration:**
        -   [ ] Integrate these two detectors into the main pipeline to process claims after generation.
        -   [ ] Run initial tests to ensure signals are being generated correctly.
-   **Team (End of Month):**
    -   [ ] **First Term Presentation Preparation:**
        -   [ ] Prepare presentation slides covering project introduction, literature review, system architecture, and progress to date.
        -   [ ] Draft and rehearse the presentation script.
        -   [ ] Finalize and submit the first term report.

---

### **Month 4: Verifier Module - Signal Implementation (Part 2)**

-   **Team (Beginning of Month):**
    -   [ ] Deliver First Term Presentation.

-   **Member 1 (Research & Presentation Focus):**
    -   [ ] Research advanced NLI models and their application in fact-checking beyond the baseline DeBERTa model.
    -   [ ] Analyze different approaches to self-agreement and consistency checking (e.g., SelfCheckGPT variants).
    -   [ ] Consolidate findings from all four signals and prepare for the integration analysis.
-   **Member 2 (Development & Experimentation Focus):**
    -   [ ] **Zero-Shot NLI Contradiction Detector:**
        -   [ ] Load the pre-trained `MoritzLaurer/DeBERTa-v3-base-mnli-fever-anli` model from Hugging Face.
        -   [ ] Implement a module that takes a (claim, evidence) pair and returns the probabilities for "entailment," "contradiction," and "neutral."
    -   [ ] **Self-Agreement Detector:**
        -   [ ] Implement a function to generate `k` different responses for the same query using stochastic sampling (e.g., temperature > 0).
        -   [ ] Implement a module to measure the semantic consistency or claim variability across the `k` responses.
    -   [ ] **Integration:**
        -   [ ] Add these two new detectors to the verifier module.

---

### **Month 5: Detector Evaluation & Mitigation**

-   **Member 1 (Research & Presentation Focus):**
    -   [ ] Analyze the results from the end-to-end evaluation, focusing on the performance of each individual signal.
    -   [ ] Correlate the custom verifier's scores with the Ragas framework's metrics (`faithfulness`, `answer_relevancy`) to produce detailed comparison charts.
    -   [ ] Draft the results and discussion sections of the final report.
-   **Member 2 (Development & Experimentation Focus):**
    -   [ ] **Rule-Based Aggregation:**
        -   [ ] Design and implement a `RuleBasedAggregator` that combines the outputs of all four signal detectors.
        -   [ ] Define explicit rules and thresholds to classify each claim as "Supported," "Contradictory," or "Low Confidence."
    -   [ ] **Confidence UI Display:**
        -   [ ] Implement a simple UI to visualize the confidence score for each generated claim.
    -   [ ] **Ragas Integration:**
        -   [ ] Integrate the Ragas framework into the evaluation pipeline.
        -   [ ] Configure Ragas to compute `faithfulness`, `answer_relevancy`, and other relevant metrics.
    -   [ ] **End-to-End Detector Evaluation:**
        -   [ ] Set up an evaluation harness to run the full system on the `TruthfulQA` and `RAGTruth` benchmarks.
        -   [ ] Run the evaluation and collect the results.
        -   [ ] Calculate key metrics for the verifier (e.g., detection accuracy, precision, recall, F1-score).
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
