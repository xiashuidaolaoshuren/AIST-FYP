# RAGTruth Task Performance Gap Analysis

## Executive Summary
In the recent RAGTruth evaluation using the `full_verifier` variant, a significant performance gap was observed between task types:
*   **Data2txt**: F1 = **0.8210** (Strongest performance, balanced precision/recall)
*   **QA**: F1 = **0.5333** (Low recall, low precision)
*   **Summary**: F1 = **0.5361** (High recall, but very low precision)

This document outlines seven key factors—spanning dataset characteristics, claim extraction, evidence retrieval, signal quality, and threshold architecture—that contribute to the ~0.29 F1 discrepancy.

---

## 1. Performance Overview

| Task | Accuracy | Precision | Recall | F1 | TP | TN | FP | FN |
|---|---:|---:|---:|---:|---:|---:|---:|---:|
| **Data2txt** | 0.7583 | 0.7778 | 0.8693 | 0.8210 | 133 | 49 | 38 | 20 |
| **QA** | 0.8250 | 0.5000 | 0.5714 | 0.5333 | 24 | 174 | 24 | 18 |
| **Summary** | 0.6250 | 0.3881 | 0.8667 | 0.5361 | 52 | 98 | 82 | 8 |

*Metrics based on the 240-sample test split per task (`ragtruth_full_verifier.json`).*

---

## 2. Factor 1: Hallucination Prevalence (Dataset Characteristics)

The distribution of hallucinations in the underlying data dramatically shifts the base detection difficulty. 
*   **Data2txt** has a very high hallucination density. In the test split, **63.8%** (153/240) of Data2txt samples are hallucinated, compared to only **17.5%** (42/240) for QA and **25.0%** (60/240) for Summary.
*   Corpus-wide RAGTruth statistics reflect this: Data2txt responses contain ~1.5 hallucinated spans per response (69% hallucinated overall), whereas QA and Summary average ~0.4 spans. 
*   **Impact**: More hallucinations per sample in Data2txt provide a denser, louder signal for both the NLI and heuristic thresholds to detect.

---

## 3. Factor 2: Nature of Hallucinations

Hallucinations manifest differently across the tasks, affecting NLI ease-of-detection:
*   **Data2txt**: Errors are typically fabricated structural attributes (e.g., a "rating: 5.0" claim when the table says "stars: 3.5"). These are atomic and unambiguously contradicted by the source.
*   **QA**: Errors are often contextual or procedural (e.g., mixing up steps in a recipe or referencing an incorrect passage). These are dispersed and harder to falsify using a single context window.
*   **Summary**: Errors often involve abstractive distortions, over-generalization, or added stylistic nuance. These are not always "factually contradicted," leading to ambiguous entailment/neutral scores from the NLI model.

---

## 4. Factor 3: Claim Extraction Granularity Mismatch

There is a fundamental difference in how claims are parsed prior to verification (`src/generation/claim_extractor.py`):
*   **Summary** uses **clause-level splitting** (`apply_clause_split` is triggered by conjunctions like `,`, `;`, `but`, `however`). This produces atomic, but fragmented claims (averaging 5.62 claims/sample).
*   **QA and Data2txt** use **sentence-level semantic boundaries** via spaCy (averaging 5.81 and 9.11 claims/sample, respectively).
*   **Impact**: For Summary, clause fragments frequently lose their syntactic context. When fed to the NLI model alongside full sentences, the limited context causes systematic confidence drops and artificial "Low Coverage" signals across many valid claims.

---

## 5. Factor 4: Evidence Quality and Retrieval Asymmetry

The quality of evidence fed to the NLI module varies by task:
*   **Data2txt**: Pre-chunked, structured table fields align perfectly with factual claims. (High-quality signal).
*   **QA**: Relies on multi-sentence passage chunks. A specific QA claim may reference an entity that is not explicitly detailed in the provided chunk, causing NLI to yield neutral/ambiguous results.
*   **Summary**: A **sentence retriever is mandatory** and utilized to fetch the top-k sentences. Because Summary extracts *clause-level* claims, comparing a clause fragment directly against a strictly bounded sentence often results in alignment mismatches, exacerbating false contradictions and low-coverage flags.

---

## 6. Factor 5: NLI Signal Quality — Gold Overlap Breakdown

The accuracy of the foundational NLI signal reveals massive "leakage" in QA and Summary. Looking at how claims overlapping with *known gold hallucinations* were classified:

| Task | Total Gold Claims | Classified as Contradictory | Classified as Low Confidence | **Classified as Supported (Leakage)** |
|---|---|---|---|---|
| **Data2txt** | 250 | 43 (17.2%) | 201 (80.4%) | **6 (2.4%)** |
| **QA** | 114 | 17 (14.9%) | 67 (58.8%) | **30 (26.3%)** |
| **Summary** | 93 | 17 (18.3%) | 61 (65.6%) | **15 (16.1%)** |

*   **Impact**: A staggering **26.3% of actual hallucinated claims in QA are confidently being classified as "Supported"**. This actively forces False Negatives and inherently caps QA recall at around ~0.57. Summary also suffers a notable 16.1% leakage. Data2txt's leakage is a negligible 2.4%.

---

## 7. Factor 6: Detection Trigger Path Analysis

Examining the exact internal paths that triggered detections highlights the internal precision of the rules (`ragtruth_full_verifier.json`):

| Primary Trigger Path | Data2txt Hits | Data2txt FP | QA Hits | QA FP | Summary Hits | Summary FP |
|---|---|---|---|---|---|---|
| `contradictory` | 139 | 28 (20%) | 77 | 32 (42%) | 153 | 62 (41%) |
| `*_low_confidence` | 76 | 19 (25%) | 7 | 2 (28%) | 67 | 27 (40%) |
| `none` (Did not fire) | 67 | N/A | 190 | N/A | 104 | N/A |

*   Data2txt's `contradictory` path triggers are highly reliable (80% precision).
*   QA's and Summary's `contradictory` paths frequently misfire (only ~58-59% internal precision).

---