# Comprehensive Evaluation Workflow: RAGTruth & CiteBench

This document provides an end-to-end overview of the verification and mitigation evaluation pipelines using the **RAGTruth** and **CiteBench (CiteEval)** frameworks. It is designed as a technical reference to assist in writing project reports, detailing how raw data flows to final evaluation metrics.

---

## 1. RAGTruth Evaluation Workflow

The RAGTruth pipeline evaluates the system's ability to detect factual hallucinations at the claim level.

### Phase 1: Data Loading & Preparation
- **Input:** `source_info.jsonl` (questions and contexts) and `response.jsonl` (gold responses and hallucination spans).
- **Process:** The system loads samples based on the configured split (train/test), applying quality filters. Context is parsed according to task type (e.g., extracting table chunks for `Data2txt`, parsing passages for `QA`, or using full documents for `Summary`).

### Phase 2: Response Generation & Claim Extraction
- **Generation:** A hybrid RAG pipeline (FAISS dense + BM25 sparse retrieval) fetches the top-$k$ evidence chunks and generates a response.
- **Extraction:** The generated response is passed through a `claim_extractor`, which tokenizes the text into atomic, verifiable claims (using sentence boundaries for QA/Data2txt and clause-level boundaries for Summary).

### Phase 3: Base Verification (VerifierHub)
Each claim is verified against the retrieved evidence using a multi-signal approach:
1. **Parallel Detection:** The system computes Intrinsic Uncertainty (entropy), Grounded Coverage (entity/number matching), Self-Agreement (stochastic sampling variance), and Natural Language Inference (NLI entailment/contradiction).
2. **NLI Batching:** Claims from across samples are batched for efficient processing through the DeBERTa NLI model.
3. **Aggregation:** A rule-based aggregator classifies each claim as `Supported`, `Contradictory`, or `Low Confidence` based on normalized confidence scores from the detectors.

### Phase 4: Metric Computation
The pipeline computes detection classification metrics:
- **Detection Metrics:** Standard Accuracy, Precision, Recall, and F1 score for sample-level hallucination detection.
- **Task-Specific Analysis:** Performance is broken down by task (QA, Data2txt, Summary) to analyze how domain-specific features (like list-based QA vs. prose Summary) affect verifier reliability.

---

## 2. CiteBench (CiteEval) Evaluation Workflow

The CiteBench pipeline assesses the quality, placement, and semantic correctness of citations and evaluates how verification signals can be used to mitigate citation failures.

### Stage 1: Data Conversion & Citation Injection
- **Input:** Raw RAG pipeline outputs (queries, answers, claims, and evidence maps).
- **Process:** The `CitationFormatter` maps generated claims to the global evidence list. It ranks passages by NLI entailment and dense retrieval scores, then injects bracketed citations (e.g., `[1][2]`) into the text adjacent to the supported claim.

### Stage 2: Verification-Aware Mitigation
Unlike RAGTruth, the CiteBench pipeline actively uses verification signals to refine the output before evaluation. This is governed by `scripts/evaluate_mitigation_citebench.py`:
1. **Evidence Re-ranking:** Uses verifier scores (NLI + Retrieval) to prioritize evidence chunks that actually support the generated claims, updating the citation indices based on the strongest source.
2. **Claim Filtering:** Removes claims explicitly flagged as `Contradictory` by the verifier aggregator before the response is submitted to CiteEval.
3. **Self-Correction:** In "Full Verifier" mode, claims with high `Intrinsic Uncertainty` are removed to ensure every remaining statement can be confidently cited.

#### Why Verifier Variants Must Include a Filter in CiteBench

This is a fundamental design difference from the RAGTruth pipeline that requires careful explanation.

**In RAGTruth**, the evaluator compares the verifier's *label* (`Supported` / `Contradictory` / `Low Confidence`) against the gold hallucination annotations. The actual text of the response is never modified — the benchmark only cares whether the verifier's decision matches the ground truth. Therefore, adding a filter would distort the evaluation by altering the sample before labeling.

**In CiteBench**, CiteEval receives the *text string itself* for scoring. It produces a Statement Rating (0–3) based on whether the cited passages genuinely support each sentence in the output. This distinction creates a critical problem:

> If the verifier labels a claim as hallucinated but does **not** physically remove it from the response, the response text submitted to CiteEval is **identical** to the baseline response. The verifier ran in the background, but its decisions had no effect on what CiteEval evaluated. The resulting CiteEval score would be the same as if no verification had occurred.

**The filter is the "actuator"** — it is the mechanism that translates the verifier's abstract labels into a tangible change in the submitted text. Only by *removing* the verified-bad claims does the response improve enough for CiteEval to detect a difference.

**Why filter and not re-ranking or re-prompting?**

| Mechanism | Effect | Problem for Citebench Ablation |
|---|---|---|
| **Re-ranking** | Reorders evidence for retrieval | Already-generated hallucinated claims remain in the response text unchanged |
| **Re-prompting** | Regenerates the entire response from scratch | The new text confounds which verifier signal caused the improvement |
| **Filter** | Removes only claims labeled bad by the verifier | Surgical and deterministic — isolates the contribution of each verifier module |

Because re-ranking and re-prompting cannot isolate the contribution of a single verifier signal, the ablation study pairs **each verifier variant with the same filter mechanism**. This holds the actuator constant across variants and varies only the detection signal:

| Variant Name | Verifier Signal | Actuator | Interpretation |
|---|---|---|---|
| `baseline` | None | None | Vanilla RAG score (no verification) |
| `verifier_nli_filter` | NLI only | Filter | What does NLI alone contribute to citation quality? |
| `verifier_grounded_filter` | Grounded Coverage only | Filter | What does entity/number matching alone contribute? |
| `verifier_intrinsic_filter` | Intrinsic Uncertainty only | Filter | What does entropy-based confidence alone contribute? |
| `verifier_self_agreement_filter` | Self-Agreement only | Filter | What does stochastic sampling variance alone contribute? |
| `full_verifier_filter` | All signals | Filter | What does the full multi-signal ensemble contribute? |

The **CiteEval score delta** between a variant and the baseline directly measures how much that verifier signal improves citation quality when its detections are acted upon.

### Stage 3: Auto-Evaluation Modules
The mitigated output is processed by four CiteEval automated modules:
1. **Context Attribution (CA):** Classifies the origin of each sentence (e.g., Query, Retrieval, Response model).
2. **Citation Editing (CE):** Detects formatting or placement errors in the citations.
3. **Citation Rating (IterCoE):** Provides an iterative assessment of how well the cited passage supports the statement.
4. **Citation Rating (EditDist):** Uses string-edit distance heuristics for accuracy scoring.

### Stage 4: Output Metrics
- **Statement Rating (0-3):** Measures citation quality at the sentence level (3 = Fully Supported, 0 = Hallucinated/Unsupported).
- **Response Rating:** An aggregated quality score for the entire response.
- **Mitigation Delta:** Compares `Baseline` (no verification) vs. `Full Verifier` CiteEval scores to quantify the impact of verification-led mitigation.

---

## 3. Cross-Framework Interpretation

Because RAGTruth and CiteBench measure fundamentally different aspects of system performance, they must be interpreted jointly to understand the true impact of the verifier.

| Framework | Evaluation Focus | Primary Signal of Improvement |
|---|---|---|
| **RAGTruth** | Claim Factual Accuracy | Higher Detection F1, Higher Precision/Recall | 
| **CiteBench** | Citation Placement & Support | Higher Statement/Response Rating |

### Interpreting Combined Outcomes

*   **Ideal Outcome:** CiteEval Ratings $\uparrow$ AND RAGTruth F1 $\uparrow$. The system correctly identifies hallucinations (RAGTruth) and uses that signal to clean up or correctly cite the remaining text (CiteEval).
*   **Precision Gap:** High RAGTruth F1 but Low CiteEval Rating. The verifier is good at labeling errors in its head, but the `CitationFormatter` is failing to translate those labels into accurate bracketed citations.
*   **Safety Trade-off:** RAGTruth Recall $\uparrow$ (Minimizing FN) leads to CiteEval Rating $\uparrow$. By prioritizing the removal of any potential hallucination (reducing False Negatives), the resulting response contains only highly-verifiable, well-cited claims.

---

## 4. Reference Appendix

### Key Scripts & Entry Points
*   **RAGTruth Evaluation:** `scripts/demo_ragtruth_eval.py`
*   **CiteBench Mitigation Eval:** `scripts/evaluate_mitigation_citebench.py`
*   **Benchmark Baseline:** `scripts/run_ragtruth_baseline.py`

### Core Implementation Modules
*   **RAGTruth Evaluator:** `src/evaluation/ragtruth_evaluator.py`
*   **Citation Formatter:** `src/citation/citation_formatter.py`
*   **Verifier Hub:** `src/verification/verifier_hub.py`

### Configuration (`config.yaml`)
*   `verification.*`: Controls the active detector modules (NLI, Intrinsic, etc.) and threshold behaviors.
*   `evaluation.citebench`: Configures the CiteEval module paths and oracle dataset presets (ASQA, ELI5, MSMARCO).