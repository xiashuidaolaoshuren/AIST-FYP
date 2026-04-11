# Project To-Do List

This to-do list breaks down the 6-month project plan into actionable tasks, organized by month.

---

### **Month 1: Research & Planning**

-   [X] **Literature Review:**
    -   [X] Read and summarize key papers on trainless hallucination detection (e.g., SelfCheckGPT, CoVe).
    -   [X] Consolidate findings and identify the most promising signals for the verifier module.
-   [X] **Data Sourcing:**
    -   [X] Download the English Wikipedia corpus.
    -   [X] Download evaluation benchmarks: `TruthfulQA`, `RAGTruth`, and `FEVER`.
    -   [X] Organize all datasets in a structured project directory.
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
    -   [x] Conduct a deep dive into the theoretical foundations of intrinsic uncertainty (entropy, perplexity) in LLMs.
    -   [x] Research and document best practices for implementing retrieval-grounded heuristics, analyzing trade-offs between different overlap metrics (e.g., lexical vs. semantic).
    -   [x] Begin drafting the methodology section of the first term report.
-   **Member 2 (Development & Experimentation Focus):**
    -   [x] **Intrinsic Uncertainty Detector:**
        -   [x] Implement a function to extract token-level logits/probabilities from the generator's output.
        -   [x] Implement a module to calculate token-level entropy and length-normalized perplexity for each claim.
    -   [x] **Retrieval-Grounded Heuristics:**
        -   [x] Implement an `evidence_coverage` function that calculates the percentage of named entities and noun phrases from a claim that appear in the evidence.
        -   [x] Implement a `citation_span_integrity` function that measures the token overlap between a claim and its direct citation.
    -   [x] **Integration:**
        -   [x] Integrate these two detectors into the main pipeline to process claims after generation.
        -   [x] Run initial tests to ensure signals are being generated correctly.
-   **Team (End of Month):**
    -   [x] **First Term Presentation Preparation:**
        -   [x] Prepare presentation slides covering project introduction, literature review, system architecture, and progress to date.
        -   [x] Draft and rehearse the presentation script.
        -   [x] Finalize and submit the first term report.

---

### **Month 4: Verifier Module - Signal Implementation (Part 2)**

-   **Team (Beginning of Month):**
    -   [x] Deliver First Term Presentation.

-   **Member 1 (Research & Presentation Focus):**
    -   [ ] Research advanced NLI models and their application in fact-checking beyond the baseline DeBERTa model.
    -   [ ] Analyze different approaches to self-agreement and consistency checking (e.g., SelfCheckGPT variants).
    -   [ ] Consolidate findings from all four signals and prepare for the integration analysis.
-   **Member 2 (Development & Experimentation Focus):**
    -   [x] **Architecture Refactoring:**
        -   [x] Implement `VerifierHub` class in `src/verification/verifier_hub.py` to centralize all detector orchestration.
        -   [x] Refactor `baseline_rag.py` to use VerifierHub instead of calling detectors directly.
    -   [x] **Evidence Strategy Enhancement:**
        -   [x] Extend verification from top-ranked evidence only to verify each claim against ALL evidence chunks.
        -   [x] Update VerifierHub to support both strategies via configuration flag (e.g., `verification.verify_all_evidence: bool`).
        -   [x] Implement signal aggregation when multiple signals per claim exist (e.g., max, mean, or weighted average).
        -   [x] Add performance optimization to avoid redundant detector calls if needed.
        -   [x] **Note:** Month 3 uses top-ranked evidence for all claims - this task extends to comprehensive verification.
    -   [x] **Zero-Shot NLI Contradiction Detector:**
        -   [x] Load the pre-trained `MoritzLaurer/DeBERTa-v3-base-mnli-fever-anli` model from Hugging Face.
        -   [x] Implement a module that takes a (claim, evidence) pair and returns the probabilities for "entailment," "contradiction," and "neutral."
    -   [x] **Self-Agreement Detector:**
        -   [x] Implement a function to generate `k` different responses for the same query using stochastic sampling (e.g., temperature > 0).
        -   [x] Implement a module to measure the semantic consistency or claim variability across the `k` responses.
    -   [x] **Integration:**
        -   [x] Add NLI and self-consistency detectors to VerifierHub.
        -   [x] Update VerifierSignal construction to populate `nli` and `consistency` fields (currently None in Month 3).

---

### **Month 5: Detector Evaluation & Mitigation**

-   **Member 1 (Research & Presentation Focus):**
    -   [ ] Analyze the results from the end-to-end evaluation, focusing on the performance of each individual signal.
    -   [x] Draft the results and discussion sections of the final report.
-   **Member 2 (Development & Experimentation Focus):**
    -   [x] **Rule-Based Aggregation:**
        -   [x] Design and implement a `RuleBasedAggregator` that combines the outputs of all four signal detectors.
        -   [x] Define explicit rules and thresholds to classify each claim as "Supported," "Contradictory," or "Low Confidence."
    -   [x] **CitationFormatter (Enable CiteBench/CiteEval):**
        -   [x] Design a citation strategy that maps top-k retrieved evidence chunks to 1-based bracketed indices `[1]..[k]` in the answer.
        -   [x] Implement `CitationFormatter` to inject inline citations and return: formatted_text, citation_map (claim_id → [indices]), passage_list (ordered evidence for export).
        -   [x] Add a post-processor to align citation markers to claim character spans (uses `extract_claims` spans) and validate with `validate_claim_spans`.
        -   [x] Implement an exporter to produce CiteEval System Evaluation JSON: `{id, query, passages:[{text,title?}], pred}`.
        -   [x] Smoke test CiteEval in `Full` mode (no citations required), then `Cited` mode (with `[i]` markers).
        -   [x] Add unit tests: (a) single and multi-sentence answers; (b) missing punctuation; (c) redundant citations; (d) out-of-range indices (should not occur).
    -   [x] **End-to-End Detector Evaluation:**
        -   [x] Set up an evaluation harness to run the full system on the `RAGTruth` and `CiteBench` benchmarks.
        -   [ ] Run the evaluation and collect the results.
        -   [ ] Calculate key metrics for the verifier (e.g., detection accuracy, precision, recall, F1-score).
    -   [x] **Mitigation Strategies:**
        -   [x] If time permits, implement active mitigation logic based on verifier feedback.
        -   [x] **Re-ranking:** Implement logic to re-order retrieved documents based on verification scores.
        -   [x] **Re-prompting:** Implement a feedback loop to ask the LLM to self-correct when low confidence is detected.
        -   [x] **Filtering:** Implement a module to suppress or flag claims that are unsupported or contradictory.
    -   [x] **Confidence UI Display:**
        -   [x] Implement a lightweight web UI (e.g., Streamlit or Gradio) that runs on top of the existing demo pipeline.
        -   [x] Display the user query and final answer text, with inline color highlighting for each claim based on its verdict (e.g., green = Supported, yellow = Low Confidence, red = Contradictory).
        -   [x] Add a per-claim table view showing: short claim snippet, final verdict badge, overall confidence band, and key signal scores (entropy, entity/number coverage, NLI contradiction, self-agreement).
        -   [x] Provide a simple drill-down interaction (e.g., expandable row or details panel) to inspect full claim text, top evidence sentences, and the raw confidence_breakdown for debugging.
        -   [x] Normalize and bucket raw scores into interpretable ranges (e.g., 0–1 or 0–100% with High/Medium/Low labels) so the UI emphasizes qualitative bands rather than raw floats.

---

### **Month 6: Finalization & Documentation**

-   [ ] **Ablation Study:**
    -   [ ] Design and run experiments to analyze the contribution of each trainless signal.
    -   [ ] Systematically disable each of the four detectors one by one and re-run the evaluation to measure the drop in performance.
-   [x] **Final Report & Demo:**
    -   [ ] Write the final project report, including sections on architecture, methodology, results, ablation study, and conclusions.
    -   [ ] Create a presentation summarizing the project.
    -   [ ] Prepare a compelling live demo or a recorded video showcasing the system's ability to detect hallucinations.
-   [x] **Code Cleanup & Handoff:**
    -   [x] Refactor the codebase for clarity and readability.
    -   [x] Add comprehensive comments and docstrings to all functions and classes.
    -   [ ] Create a `README.md` file with instructions on how to set up and run the project.
    -   [ ] Ensure all project artifacts are committed to the version control system.
