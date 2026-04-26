# Chapter: System Evaluation and Results

This chapter comprehensively evaluates the performance of the proposed trainless verifier and mitigation pipeline against established baselines. Moving beyond traditional evaluation, this research posits that detecting a hallucination is only half the problem; system evaluation must also capture the effectiveness of *correcting* these errors in the final user-facing output. Therefore, the evaluation is structured across three primary analytical dimensions:

1. **Factual Hallucination Detection (RAGTruth):** Assessing the pipeline's raw labeling capability to accurately flag factual errors across various generation tasks.
2. **Verifier Signal Evaluation (CiteBench/CiteEval):** Isolating and evaluating the structural and citation quality when relying on different detection signals (e.g., Intrinsic vs. NLI) paired with a rigid, deterministic filter.
3. **Mitigation Pipeline Evaluation (CiteBench/CiteEval):** Testing different downstream "actuators" (Filtering, Reranking, Reprompting) to determine the most effective strategy for repairing a detected hallucination.

The pipeline is baselined against **LettuceDetect** [3], a state-of-the-art fine-tuned hallucination detection model, to highlight the robust capabilities of our modular, zero-shot architecture.

---

## 1. Evaluation Methodology and Workflow

The evaluation framework purposefully separates the **detection of hallucinations** from the **mitigation of hallucinations**. RAGTruth evaluates pure factual accuracy and the intrinsic reliability of our verifier's labeling at the isolated claim level. Conversely, CiteBench is employed dynamically to assess how these verifier decisions actually restructure and repair the final text—specifically examining the quality, density, and semantic correctness of citations after applying various mitigation strategies. 

### 1.1 Factual Accuracy Verification (RAGTruth Pipeline)

The RAGTruth workflow consists of four systematic stages designed to test claim-level hallucination detection:
1. **Data Loading & Preparation**: Samples are loaded based on configured splits, and context is parsed according to task types (e.g., extracting table chunks for Data2txt, parsing passages for QA, or using full documents for Summary).
2. **Response Generation & Extraction**: A hybrid RAG pipeline (FAISS dense + BM25 sparse retrieval) fetches the top-k evidence chunks and generates a response. The `claim_extractor` then tokenizes the text into atomic, verifiable claims.
3. **Multi-Signal Verification (VerifierHub)**: Claims are processed in parallel through Intrinsic Uncertainty (entropy), Grounded Coverage (entity/number matching), Self-Agreement (stochastic sampling variance), and Natural Language Inference (NLI) modules. The NLI module processes batched claims via DeBERTa. An aggregator classifies the final claim status as `Supported`, `Contradictory`, or `Low Confidence`.
4. **Metric Computation**: System classifications are evaluated against ground truth annotations to determine Accuracy, Precision, Recall, and F1 scores, broken down by task to analyze domain-specific verifier reliability.

### 1.2 Citation Mitigation (CiteBench Pipeline)

Unlike RAGTruth, which strictly evaluates detection performance label-matching, CiteBench actively uses verification signals to mitigate errors before final output generation.
1. **Citation Injection**: Generated claims are mapped to global evidence lists. Passages are ranked by NLI and retrieval scores to inject bracketed citations (e.g., `[1][2]`).
2. **Verification-Aware Mitigation (The Filtering Actuator)**: To isolate the impact of different verifier signals, claims flagged as `Contradictory` are deterministically filtered. 
   > **Why Verifier Variants Must Include a Filter in CiteBench:** 
   > This is a fundamental design difference from the RAGTruth pipeline. In RAGTruth, the evaluator compares the verifier's *label* against ground truth annotations without altering the text. However, CiteEval scores the *final text string itself*. If a verifier merely labels a claim as hallucinated but does not physically remove it, the response text submitted to CiteEval remains identical to the unverified baseline, resulting in zero measurable difference. 
   > 
   > **Why filter and not re-rank or re-prompt during Verifier Ablation?** Filtering is surgical and deterministic. Re-ranking or re-prompting introduces complete generation variance (the LLM creates entirely new text), which severely confounds which underlying verifier signal (e.g., NLI vs. Intrinsic) actually caused the improvement. Thus, pairing each verifier signal with a constant rigid filter isolates the contribution of the detection mechanism perfectly.
3. **Auto-Evaluation Modules**: The mitigated output is processed iteratively using four CiteEval automated modules: Context Attribution (CA), Citation Evaluation (CE), and continuous Citation Recall mapping metrics (IterCoE and EditDist).

---

## 2. Hallucination Detection Performance (RAGTruth)

### 2.1 Overall Detection Metrics & Design Philosophy

The performance of the verifier variants across all tasks is summarized below.

**Table 1: Overall RAGTruth Detection Metrics**

| Variant | Accuracy | Precision | Recall | F1 | Total Claims | FP | FN |
| :--- | :---: | :---: | :---: | :---: | :---: | :---: | :---: |
| **LettuceDetect** (Baseline) | - | 0.7664 | 0.7550 | **0.7607** | - | - | - |
| **full_verifier** (Proposed) | 0.7361 | 0.5921 | **0.8196** | 0.6875 | 4931 | 144 | 46 |
| verifier_nli_only | 0.5819 | 0.4502 | 0.8157 | 0.5802 | 4931 | 254 | 47 |
| verifier_grounded_only | 0.5917 | 0.4593 | 0.8627 | 0.5995 | 4931 | 259 | 35 |
| verifier_intrinsic_only | 0.6458 | 0.0000 | 0.0000 | 0.0000 | 4931 | 0 | 255 |
| verifier_self_agreement | 0.6458 | 0.0000 | 0.0000 | 0.0000 | 4931 | 0 | 255 |

> **Explanation of Table 1:** LettuceDetect achieves a higher overall F1 score (0.7607) due to its nature as a fine-tuned model trained specifically on hallucination boundaries. However, our proposed `full_verifier` achieves a significantly higher recall (0.8196). In safety-critical RAG deployments, minimizing False Negatives (FN=46) is paramount. Missing a hallucination is dangerous, whereas False Positives (FP=144) only temporarily reduce fluency.
> 
> **Architectural Factors Driving the Difference:**
> - **Trained Model vs. Heuristics:** LettuceDetect learns the exact boundary between stylistic variations and factual errors. Our pipeline uses a trainless setup (off-the-shelf NLI + heuristics), which favors identifying potential issues (high recall) but lacks fine-tuned boundaries (causing lower precision/more false positives).
> - **Multi-Path Aggregation vs. Single Span:** Our pipeline aggregates claim-based evaluations using multiple trigger paths (e.g., `contradictory`, `low_confidence_coverage`). This aggressive "catch-all" approach increases recall but introduces compounding noise. LettuceDetect predicts directly on spans, minimizing structural noise.
> - **Ablation Insights:** NLI is the indispensable core signal driving the bulk of detection recall. Grounded coverage serves as a precision guard (reducing FP from 254 in NLI-only down to 144). Intrinsic and self-agreement signals yield zero detections independently, functioning solely to modulate confidence within the broader multi-signal array.

### 2.2 Per-Task Detection Metrics

Performance varied considerably depending on the input domain. Tables 2, 3, and 4 illustrate this breakdown.

**Table 2: Detection Metrics for Data2txt (Structured Data)**

| Variant | Accuracy | Precision | Recall | F1 | TP | TN | FP | FN |
| :--- | :---: | :---: | :---: | :---: | :---: | :---: | :---: | :---: |
| **LettuceDetect** | - | 0.8930 | 0.8653 | **0.8789** | - | - | - | - |
| **full_verifier** | 0.7583 | 0.7778 | 0.8693 | 0.8210 | 133 | 49 | 38 | 20 |
| verifier_nli_only | 0.6667 | 0.6714 | 0.9346 | 0.7814 | 143 | 17 | 70 | 10 |
| verifier_grounded_only | 0.6375 | 0.6375 | 1.0000 | 0.7786 | 153 | 0 | 87 | 0 |

> **Explanation of Table 2:** On structured table data (Data2txt), the pipeline shows strong performance (F1=0.8210) with high recall. Grounded entity matching (Recall=1.0000) perfectly catches structural fabrications but suffers severe precision issues if not guarded by NLI semantics.

**Table 3: Detection Metrics for QA (Short Answers)**

| Variant | Accuracy | Precision | Recall | F1 | TP | TN | FP | FN |
| :--- | :---: | :---: | :---: | :---: | :---: | :---: | :---: | :---: |
| **LettuceDetect** | - | 0.6064 | 0.7125 | **0.6552** | - | - | - | - |
| **full_verifier** | 0.8250 | 0.5000 | 0.5714 | 0.5333 | 24 | 174 | 24 | 18 |
| verifier_nli_only | 0.5792 | 0.2342 | 0.6190 | 0.3399 | 26 | 113 | 85 | 16 |
| verifier_grounded_only | 0.8500 | 0.7143 | 0.2381 | 0.3571 | 10 | 194 | 4 | 32 |

> **Explanation of Table 3:** QA tasks challenge the multi-signal approach, causing a lower F1 of 0.5333. Short, concise responses often lack sufficient context for the NLI model, which causes a spike in False Positives (24) on valid answers.

**Table 4: Detection Metrics for Summary (Long-form Text)**

| Variant | Accuracy | Precision | Recall | F1 | TP | TN | FP | FN |
| :--- | :---: | :---: | :---: | :---: | :---: | :---: | :---: | :---: |
| **LettuceDetect** | - | 0.5389 | 0.4755 | 0.5052 | - | - | - | - |
| **full_verifier** | 0.6250 | 0.3881 | 0.8667 | **0.5361** | 52 | 98 | 82 | 8 |
| verifier_nli_only | 0.5000 | 0.2826 | 0.6500 | 0.3939 | 39 | 81 | 99 | 21 |
| verifier_grounded_only | 0.2875 | 0.2533 | 0.9500 | 0.4000 | 57 | 12 | 168 | 3 |

> **Explanation of Table 4:** Our `full_verifier` actually outperforms the fine-tuned baseline in Summary tasks (F1=0.5361 vs 0.5052). Long dense texts distribute subtle hallucinations across multiple clauses. The pipeline's aggressive signal aggregation successfully captures these nuances (Recall=0.8667) that a direct model classification misses.

### 2.3 Task Performance Gap Analysis

An F1 gap of ~0.29 exists between structured constraints (Data2txt) and abstractive answering (QA/Summary). The performance summary across these tasks is highlighted below.

**Table 5: Task Performance Overview (full_verifier)**

| Task | Accuracy | Precision | Recall | F1 | TP | TN | FP | FN |
| :--- | :---: | :---: | :---: | :---: | :---: | :---: | :---: | :---: |
| **Data2txt** | 0.7583 | 0.7778 | 0.8693 | **0.8210** | 133 | 49 | 38 | 20 |
| **QA** | 0.8250 | 0.5000 | 0.5714 | **0.5333** | 24 | 174 | 24 | 18 |
| **Summary** | 0.6250 | 0.3881 | 0.8667 | **0.5361** | 52 | 98 | 82 | 8 |

> **Explanation of Table 5:** This table directly illustrates the performance drop causing the ~0.29 F1 score gap. Data2txt maintains a robust balance of Precision and Recall, while QA suffers from low Recall and Summary suffers from low Precision.

We identified several driving factors causing this gap, categorized by structural mismatches and signal failures.

#### A. Structural and Dataset Mismatches

1. **Hallucination Prevalence (Dataset Characteristics):** The base density of hallucinations in `Data2txt` test samples is extremely high (63.8%), providing a rich, dense signal. In contrast, `QA` only has a 17.5% prevalence and `Summary` 25.0%.
2. **Nature of Hallucinations:** `Data2txt` errors are typically explicit structural or numerical contradictions that are unambiguously contradicted by the source. `QA` and `Summary` errors often involve contextual mixing, abstractive distortions, or procedural mix-ups that are harder to falsify using a single context window.
3. **Claim Extraction Granularity Mismatch:** `Summary` tasks use aggressive clause-level splitting, producing fragmented claims that lose surrounding syntactic context. When fed to NLI, this stripping of context generates artificial uncertainty (driving down precision). `QA` and `Data2txt` use more robust sentence-level boundaries.
4. **Evidence Quality and Retrieval Asymmetry:** `Data2txt` maps cleanly to pre-chunked structured table fields. `QA` and `Summary` rely on overlapping, multi-sentence passage chunks. Comparing an atomic claim or clause directly against a strict sentence boundary frequently results in misalignment, causing false contradictions and low-coverage flags.

#### B. NLI Signal Leakage

5. **Severe NLI Signal Leakage:** Ground truth evaluations reveal the foundational NLI signal confidently misclassifies a large portion of actual hallucinated claims in QA as "Supported".

**Table 6: NLI Signal Quality - Gold Overlap Breakdown**

| Task | Total Gold Claims | Contradictory | Low Confidence | **Leakage (Supported)** |
|:---|:---:|:---:|:---:|:---:|
|**Data2txt** | 250 | 43 (17.2%) | 201 (80.4%) | **6 (2.4%)** |
|**QA** | 114 | 17 (14.9%) | 67 (58.8%) | **30 (26.3%)** |
|**Summary**	 | 93 | 17 (18.3%) | 61 (65.6%) | **15 (16.1%)** |

> **Explanation of Table 6:** This table analyzes "leakage," defined as actual ground-truth hallucinations that the NLI model incorrectly passed as "Supported". QA features severe leakage (26.3%). Because over a quarter of hallucinations semantically bypass the NLI module, the maximum possible recall ceiling for QA is inherently capped at ~0.57.

#### C. Rule Trigger Efficiency

6. **Detection Trigger Path Efficiency:** The specific heuristic rules triggered vary wildly in internal precision across tasks. The `contradictory` NLI path is highly reliable in Data2txt but misfires heavily in QA and Summary.

**Table 7: Detection Trigger Path Analysis**

| Primary Trigger Path | Data2txt Hits | Data2txt FP | QA Hits | QA FP | Summary Hits | Summary FP |
| :--- | :---: | :---: | :---: | :---: | :---: | :---: |
| `contradictory` | 139 | 28 (20%) | 77 | 32 (42%) | 153 | 62 (41%) |
| `*_low_confidence` | 76 | 19 (25%) | 7 | 2 (28%) | 67 | 27 (40%) |
| `none` (Did not fire) | 67 | - | 190 | - | 104 | - |

> **Explanation of Table 7:** This breakdown details which aggregation rule caught the errors. The NLI `contradictory` path proves extremely reliable in Data2txt (20% FP rate internally). However, it misfires heavily in QA and Summary (>40% FP rates). This is driven by evaluating clause-level semantic fragments against much larger multi-sentence context boundaries, creating artificial alignment contradictions.

---

## 3. Mitigation and Citation Quality (CiteBench)

316 queries from the ASQA Oracle dataset were processed via CiteBench to evaluate actionable mitigation protocols.

### 3.1 Metric Definitions (CiteEval)

To properly interpret the mitigation results, CiteBench utilizes a strict module-based structure to systematically assess citation quality:
- **Citation Attribution (CA):** Evaluates if a sentence requires a citation. It maps text to **Retrieval** (supported by context), **Model** (relies entirely on hallucinated/parametric knowledge), **Response** (logical derivations), or **Query**. *A high CA Retrieval ratio indicates strong factual grounding.*
- **Citation Evaluation (CE):** Manually assesses the relevance and accuracy of the citations provided on a 1-5 integer scale (5 = Excellent, 1 = Unacceptable).
- **Citation Recall (CR):** Computed only for statements requiring citations. **IterCoE** scales the 1-5 evaluation tags into a continuous 0.0-1.0 scoring range. **EditDist** tracks structural operations (Deletes/Adds) required to repair the generated citations and converts it into a structural similarity score.
- **Generation Stats:** Provides the Statement Rating (overall response faithfulness), Density (ratio of sentences with citations), and Length.

### 3.2 Verifier Signals Performance (Signal Ablation)

**Table 8: Overall CiteEval Metrics**

| Variant | Statement Rating | Density | CA Retrieval Ratio | CE Mean Sent Rating | CE Sent Coverage | CR IterCoE | CR EditDist |
| :--- | :---: | :---: | :---: | :---: | :---: | :---: | :---: |
| **LettuceDetect** | - | - | - | 1.7029 | **0.7563** | 0.9241 | 0.9364 |
| **full_verifier_filter** | 0.5934 | 0.8290 | 0.8693 | 2.7504 | **2.0791** | 0.5798 | 0.7849 |
| **verifier_nli_filter** | **0.8046** | 0.8677 | 0.8627 | **4.0090** | 2.1203 | 0.7701 | 0.8358 |
| verifier_grounded_filter | 0.6156 | 0.8616 | 0.8552 | 2.9053 | 2.1392 | 0.6062 | 0.7788 |
| verifier_intrinsic_filter | 0.6096 | 0.8869 | 0.8766 | 2.8561 | 2.1772 | 0.5833 | 0.7688 |
| verifier_self_agreement_filter | 0.6037 | 0.8841 | 0.8625 | 2.8379 | 2.1867 | 0.5752 | 0.7652 |

> **Explanation of Table 8:** This table evaluates the structural citation generation capacity. LettuceDetect functions as a post-hoc detection tool rather than an active filter; consequently, its CE Sentence Coverage is abysmal (0.7563, evaluating barely ~9.5% of total sentences) and its CE Mean Rating is categorized as "Poor" (1.7029). By contrast, our integrated pipeline generates and filters structurally. The `verifier_nli_filter` dramatically improves citation quality, achieving an exceptional CE Mean Rating of **4.0090 (Good)** across extensive sentence coverage (2.1203).

**LettuceDetect vs. Full Verifier Pipeline Analysis:**
While LettuceDetect is a post-hoc detection tool for tagging spans as hallucinated, our Full Verifier Pipeline integrates active evidence mitigation and automated citation injection. Due to these structural differences, a direct comparison across all CiteEval metrics is not fully balanced; however, certain metrics allow for meaningful comparative evaluation of citation quality:
- **Citation Evaluation (CE) Quality**: Comparing the **Mean Sentence Rating**, LettuceDetect achieves a score of **1.7029** (categorized as "Poor"), while our Full Verifier scores **2.7504** (approaching "Fair") and the best variant `verifier_nli_filter` achieves **4.0090** ("Good"). This highlights that our integrated pipeline generates citeable content of significantly higher quality and relevance than simply applying detection over baseline responses.
- **Citation Recall (CR) and Coverage**: LettuceDetect exhibits superficially high **Answer Ratings** (0.9241) but suffers from extremely low **Sentence Coverage** (0.0949), representing only ~9.5% of total sentences evaluated. In contrast, the Full Verifier achieves a coverage of **2.0791**, demonstrating that it provides nearly 20x more verified, grounded evidence per response than the detection baseline.
- **Attribution Reliability**: Our pipeline demonstrates structural grounding through the CA module, ensuring the majority of logic remains aligned with retrieved contexts, whereas LettuceDetect remains largely unaligned with the retrieval set (mapping mostly to parametric Model knowledge).

### 3.3 Verifier Filtering Statistics

**Table 9: Filtering Effectiveness**

| Variant | Filtered Claims (out of ~700) | Avg NLI Entailment | Avg Entropy | Avg Token F1 |
| :--- | :---: | :---: | :---: | :---: |
| full_verifier_filter | 22 | 0.8395 | 0.0760 | 0.2896 |
| **verifier_nli_filter** | **31** | 0.8306 | 0.0000 | 0.3130 |
| verifier_grounded_filter | 14 | 0.0000 | 0.0000 | 0.3150 |
| verifier_intrinsic_filter | 0 | 0.0000 | 0.0125 | 0.3144 |
| verifier_self_agreement_filter | 0 | 0.0000 | 0.0000 | 0.3146 |

> **Explanation of Table 9:** This isolates exactly which signal executed the mitigations. The NLI filter efficiently removed the most corrupted statements (31 claims). Intrinsic Entropy metrics filtered exactly *zero* claims, showing that in heavily grounded generation contexts, parametric probability distributions converge tightly, rendering uncertainty measures ineffective. Forcing these redundant signals into the `full_verifier_filter` unfortunately diluted the effective NLI responses, reducing the filter amount to 22.

**In-Depth Analysis of Verifier Variants:**
1. **`verifier_nli_filter` (Best Performance)**:
   - **Performance**: Statement Rating 0.8046, CE Mean Rating 4.009 (nearly "Good"), CR IterCoE 0.7701.
   - **Signal Analysis**: This variant utilizes a DeBERTa semantic entailment model. Its high scores prove that **Semantic Entailment** is the most effective signal for CiteEval's criteria. By accurately flagging 31 ungrounded statements for the filter to remove, it ensures that the remaining text is strictly supported by the evidence, aligning perfectly with human-like citation requirements.
2. **`verifier_grounded_filter` (Moderate Performance)**:
   - **Performance**: Statement Rating 0.6156, CE Mean Rating 2.905.
   - **Signal Analysis**: This signal relies on heuristic exact-matches (Entities/Numbers). While it successfully caught 14 surface-level hallucinations, its "Moderate" performance reveals that **lexical matching alone is insufficient**. It misses subtle semantic contradictions that don't involve entity errors, resulting in lower groundedness scores compared to NLI.
3. **`verifier_intrinsic_filter` & `verifier_self_agreement_filter` (No Effect)**:
   - **Performance**: Identical metrics to unfiltered (0 detected/filtered).
   - **Signal Analysis**: These variants represent **Uncertainty-based signals** (Entropy and Stochastic Sampling). In the context of a RAG pipeline where the generator is heavily anchored by provided evidence, the model's output probability distributions become effectively flat. This "collapsing" of uncertainty means these signals fail to trigger, proving that **internal model confidence is a poor hallucination proxy** when strong external evidence is present.
4. **`full_verifier_filter` (Sub-Par Aggregation Effect)**:
   - **Performance**: Statement Rating 0.5934, lower than NLI or Grounded variants independently.
   - **Signal Analysis**: The RuleBasedAggregator's current logic **dilutes the superior NLI signal**. By attempting to balance multiple signals, the ensemble's hierarchical gating or permissive thresholds allowed 9 ungrounded claims (which NLI alone would have caught) to leak into the final output. This indicates that a "weighted sum" or "majority vote" approach is less effective than a strict "NLI-first" strategy for CiteBench.

### 3.4 Detailed Verifier Module Interpretations

**Table 10: Citation Attribution (CA)**

| Variant | Classified Sentences | Type: Retrieval | Type: Model | Type: Response | Type: Query |
| :--- | :---: | :---: | :---: | :---: | :---: |
| **LettuceDetect** | 316 | 9 | 293 | 14 | - |
| **full_verifier_filter** | 658 | 572 | 39 | 45 | 2 |
| verifier_grounded_filter | 677 | 579 | 45 | 53 | - |
| verifier_intrinsic_filter | 689 | 604 | 37 | 48 | - |
| **verifier_nli_filter** | 670 | 578 | 44 | 48 | - |
| verifier_self_agreement_filter | 691 | 596 | 31 | 64 | - |

> **Explanation of Table 10:** The Context Attribution (CA) module decides what type of knowledge a sentence requires. LettuceDetect largely relied on hallucinatory "Model" knowledge (293 sentences), where the context could not support the answer. The pipeline overwhelmingly forces structural reliance mapped back to "Retrieval" (570+ sentences).

**Table 11: Citation Evaluation (CE)**

| Variant | Sentence Ratings | Mean Sentence Rating | Sentence Coverage |
| :--- | :---: | :---: | :---: |
| LettuceDetect | 239 | 1.7029 | 0.7563 |
| full_verifier_filter | 657 | 2.7504 | 2.0791 |
| verifier_grounded_filter | 676 | 2.9053 | 2.1392 |
| verifier_intrinsic_filter | 688 | 2.8561 | 2.1772 |
| **verifier_nli_filter** | 670 | **4.0090** | 2.1203 |
| verifier_self_agreement_filter | 691 | 2.8379 | 2.1867 |

> **Explanation of Table 11:** The CE metric assesses manual alignment relevance on a 1-5 scale. Post-filtering, the pure NLI logic mapped sentences directly back to source documents creating a highly qualitative output near a 4.0 'Good' rating scale, dwarfing the LettuceDetect baseline.

**Table 12: Citation Recall — Iterative CoE (CR IterCoE)**

| Variant | Answer Ratings | Mean Answer Rating | Sent Ratings | Mean Sent Rating | Sent Coverage |
| :--- | :---: | :---: | :---: | :---: | :---: |
| LettuceDetect | 316 | 0.9241 | 30 | 0.2000 | 0.0949 |
| full_verifier_filter | 316 | 0.5798 | 602 | 0.4186 | 1.9051 |
| verifier_grounded_filter | 316 | 0.6062 | 624 | 0.4663 | 1.9747 |
| verifier_intrinsic_filter | 316 | 0.5833 | 645 | 0.4593 | 2.0411 |
| **verifier_nli_filter** | 316 | **0.7701** | 612 | 0.7606 | 1.9367 |
| verifier_self_agreement_filter | 316 | 0.5752 | 647 | 0.4490 | 2.0475 |

> **Explanation of Table 12:** Iterative Chain-of-Evaluation scores the answer using logic extraction steps scaled 0-1. LettuceDetect's deceptively high 0.92 is merely an artifact of evaluating only 30 sentences globally. NLI filtering achieves 0.7701 recall while validating ~20x the number of generated logical steps compared to the baseline.

**Table 13: Citation Recall — Edit Distance (CR EditDist)**

| Variant | Answer Ratings | Mean Answer Rating | Sent Ratings | Mean Sent Rating | Sent Coverage |
| :--- | :---: | :---: | :---: | :---: | :---: |
| LettuceDetect | 316 | 0.9364 | 30 | 0.3304 | 0.0949 |
| full_verifier_filter | 316 | 0.7849 | 602 | 0.7153 | 1.9051 |
| verifier_grounded_filter | 316 | 0.7788 | 624 | 0.7221 | 1.9747 |
| verifier_intrinsic_filter | 316 | 0.7688 | 645 | 0.7257 | 2.0411 |
| **verifier_nli_filter** | 316 | **0.8358** | 612 | 0.8404 | 1.9367 |
| verifier_self_agreement_filter | 316 | 0.7652 | 647 | 0.7233 | 2.0475 |

> **Explanation of Table 13:** Edit distance measures the quantitative structural deletes/adds required to repair citation logic. The robust initial generation backed by the NLI-mitigation layer ensures minimal post-generation surgical editing is required, scoring a highly accurate 0.8358.

### 3.5 Mitigation Pipeline Performance (Actuator Ablation)

While the previous sections evaluated the quality of individual verifier *signals* (using a fixed filter), this section evaluates the effectiveness of different mitigation *actuators* (Filtering, Reranking, Reprompting) when using the `full_verifier` signal suite.

**Table 14: Mitigation Actuator Overall Metrics**

| Variant | Statement Rating | Density | CA Retrieval Ratio | CE Mean Sent Rating | CR IterCoE (Answer) | CR EditDist (Answer) |
| :--- | :---: | :---: | :---: | :---: | :---: | :---: |
| **full_verifier_filter** (Baseline) | 0.5934 | 0.8290 | 0.8693 | 2.7504 | 0.5798 | 0.7849 |
| **mitigation_filter_only** | 0.8104 | 0.8508 | 0.8612 | 3.9836 | 0.7829 | 0.8498 |
| **mitigation_rerank_only** | 0.8112 | 0.8836 | 0.8623 | 3.9986 | 0.7698 | 0.8400 |
| **mitigation_reprompt_only** | **0.8216** | 0.8686 | **0.8707** | **4.0640** | **0.7899** | **0.8489** |
| **mitigation_all** | 0.8158 | **0.8788** | 0.8666 | 4.0402 | 0.7824 | 0.8457 |

> **Explanation of Table 14:** `mitigation_reprompt_only` is the strongest mitigation variant on the main CiteEval quality metrics. It leads on Statement Rating (0.8216), CE Mean Sentence Rating (4.0640), and Citation Recall IterCoE (0.7899). All four active mitigation variants heavily outperform the baseline `full_verifier_filter`.

**Table 15: Mitigation Actuator Filtering Statistics**

| Variant | Total Claims | Filtered Claims | Filter Rate | CE Sent Coverage |
| :--- | :---: | :---: | :---: | :---: |
| **full_verifier_filter** | 685 | 22 | 0.0321 | 2.0791 |
| **mitigation_filter_only** | 727 | 30 | 0.0413 | 2.1203 |
| **mitigation_rerank_only** | 738 | 0 | 0.0000 | 2.2057 |
| **mitigation_reprompt_only** | 757 | 0 | 0.0000 | **2.2247** |
| **mitigation_all** | 736 | 24 | 0.0326 | 2.2025 |

> **Explanation of Table 15:** `mitigation_filter_only` employs the most aggressive deletion (30 claims), but this subtraction does not translate into the best overall CiteEval quality. `mitigation_reprompt_only` and `mitigation_rerank_only` do not remove claims directly (filter rate 0.0) but achieve strong scores by improving the generated response or evidence alignment respectively.

**Key Analytical Takeaways for Mitigation Strategies:**

1. **Detection is Not Correction:** Verifier-only labeling and basic filtering (`full_verifier_filter`) are insufficient for high-quality citation generation. Both `full_verifier_filter` and `mitigation_all` delete a similar amount of content (~22-24 claims), yet the massive quality score increase in `mitigation_all` proves that active mitigation strategies are required to properly align citations and fix logic.
2. **Rewriting beats Deletion:** The `reprompt_only` strategy proved to be the most effective active mitigation. By asking the LLM to actively rewrite its answer dynamically using the verification conflict signals, it reconstructs strongly-supported narratives seamlessly rather than abruptly cutting sentences out natively.
3. **Ensemble Interference:** Stacking all mitigations (`mitigation_all`) yields extremely strong results, but slightly underperforms `reprompt_only`. This suggests that running filtering, reranking, and reprompting concurrently on the same text introduces slight redundancies or over-corrections (e.g., aggressively filtering a claim that the reprompt module could have otherwise gracefully rewritten). Future iterations may benefit from dynamic, sequential routing rather than a static parallel pipeline.

### 3.6 Detailed Mitigation Module Interpretations

**Table 16: Mitigation Module - Citation Attribution (CA)**

| Variant | Classified Sentences | Retrieval | Model | Response | Query |
| :--- | :---: | :---: | :---: | :---: | :---: |
| **full_verifier_filter** | 658 | 572 | 39 | 45 | 2 |
| **mitigation_filter_only** | 670 | 577 | 46 | 47 | 0 |
| **mitigation_rerank_only** | 697 | 601 | 39 | 57 | 0 |
| **mitigation_reprompt_only** | 704 | 613 | 34 | 57 | 0 |
| **mitigation_all** | 697 | 604 | 35 | 58 | 0 |

> **Explanation of Table 16:** The `reprompt_only` and `mitigation_all` variants generated the highest absolute number of retrieval-grounded sentences (>600). This indicates these active mitigation steps allow the LLM to synthesize more substantiated material rather than merely deleting faulty segments.

**Table 17: Mitigation Module - Citation Evaluation (CE)**

| Variant | Sentence Ratings | Mean Sentence Rating | Sentence Coverage |
| :--- | :---: | :---: | :---: |
| **full_verifier_filter** | 657 | 2.7504 | 2.0791 |
| **mitigation_filter_only** | 670 | 3.9836 | 2.1203 |
| **mitigation_rerank_only** | 697 | 3.9986 | 2.2057 |
| **mitigation_reprompt_only** | 703 | **4.0640** | **2.2247** |
| **mitigation_all** | 696 | 4.0402 | 2.2025 |

> **Explanation of Table 17:** This compares the raw 1-5 scale rating for how well cited each sentence is. `mitigation_reprompt_only` achieves the highest average (4.0640) and covers the most sentences, proving rewriting directly facilitates better citation structure and mapping.

**Table 18: Mitigation Module - Citation Recall (IterCoE)**

| Variant | Answer Ratings | Mean Answer Rating | Sent Ratings | Mean Sent Rating | Sent Coverage |
| :--- | :---: | :---: | :---: | :---: | :---: |
| **full_verifier_filter** | 316 | 0.5798 | 602 | 0.4186 | 1.9051 |
| **mitigation_filter_only** | 316 | 0.7829 | 610 | 0.7623 | 1.9304 |
| **mitigation_rerank_only** | 316 | 0.7698 | 652 | 0.7592 | 2.0633 |
| **mitigation_reprompt_only** | 316 | **0.7899** | 654 | **0.7737** | **2.0696** |
| **mitigation_all** | 316 | 0.7824 | 649 | 0.7720 | 2.0538 |

> **Explanation of Table 18:** IterCoE evaluates logic preservation. Actively rewriting (`reprompt_only`) preserves the most factual logic while adhering to citation boundaries optimally, leading to the highest Mean Answer Rating.

**Table 19: Mitigation Module - Citation Recall (Edit Distance)**

| Variant | Answer Ratings | Mean Answer Rating | Sent Ratings | Mean Sent Rating | Sent Coverage |
| :--- | :---: | :---: | :---: | :---: | :---: |
| **full_verifier_filter** | 316 | 0.7849 | 602 | 0.7153 | 1.9051 |
| **mitigation_filter_only** | 316 | **0.8498** | 610 | 0.8446 | 1.9304 |
| **mitigation_rerank_only** | 316 | 0.8400 | 652 | 0.8443 | 2.0633 |
| **mitigation_reprompt_only** | 316 | 0.8489 | 654 | **0.8505** | **2.0696** |
| **mitigation_all** | 316 | 0.8457 | 649 | 0.8473 | 2.0538 |

> **Explanation of Table 19:** Edit distance measures how much structural surgery is required post-generation to fix citations. All active mitigation variants dramatically cut down the required edits compared to baseline tracking at ~0.84+, indicating a highly robust structural form.


---

## 4. Conclusion

This evaluation substantiates the hypothesis that a trainless, zero-shot verifier architecture functions as a highly sensitive, safety-centric hallucination detector. Across both factual detection (RAGTruth) and citation mitigation (CiteBench) evaluations, deep semantic verification (NLI) consistently emerges as the champion mechanism for intercepting RAG-based factual fabrications.

The proposed architecture successfully trades absolute precision thresholds for system transparency and a dramatic reduction in false negatives. This renders the system highly appropriate for safety-critical deployments—especially within long-form summative tasks where hallucinated fragments are subtly dispersed. Future iterations of this architecture should reformulate the RuleBasedAggregator's weights, explicitly weighting NLI conflict signals more aggressively over redundant intrinsic metrics during mitigation filtering.

---

## References
[1] C. Niu et al., "RAGTruth: A Hallucination Corpus for Developing Trustworthy Retrieval-Augmented Language Models," *arXiv preprint arXiv:2401.00396*, 2024.
[2] Y. Xu et al., "CiteEval: Principle-Driven Citation Evaluation for Source Attribution," *arXiv preprint arXiv:2506.01829*, 2025.
[3] Á. Kovács and G. Recski, "LettuceDetect: A Hallucination Detection Framework for RAG Applications," *arXiv preprint arXiv:2502.17125*, 2025.
