# CiteBench Verifier Variant Evaluation Metrics

Sources:
- Verifier variants: `c:/Users/admin/Desktop/citebench/verification`
- LettuceDetect: `c:/Users/admin/Desktop/citebench/lettucedetect`

Dataset: ASQA Oracle (`asqa_oracle.dev.jsonl`), 316 queries  
Evaluator: CiteEval (`citeeval-auto-12272024`, provider: `deepseek-chat`)  
Modules: `ca`, `ce`, `cr_itercoe`, `cr_editdist`

---

## Overall CiteEval Metrics

> **Note:** LettuceDetect is a post-hoc detection method applied to baseline responses; it does not generate new responses, so response-level generation metrics (Statement Rating, Length, Density, etc.) are not applicable.

| Variant | Run Folder | Samples | Statement Rating | Avg Length | Density | CA Retrieval Ratio | CE Mean Sent Rating | CE Sent Coverage | CR IterCoE (Answer) | CR EditDist (Answer) |
| --- | --- | ---: | ---: | ---: | ---: | ---: | ---: | ---: | ---: | ---: |
| LettuceDetect | lettucedetect_eval | 316 | — | — | — | — | 1.7029 | 0.7563 | 0.9241 | 0.9364 |
| full_verifier_filter | citebench_verifier_full_verifier_filter_20260329_113436 | 316 | 0.5934 | 2.0823 | 0.8290 | 0.8693 | 2.7504 | 2.0791 | 0.5798 | 0.7849 |
| verifier_grounded_filter | citebench_verifier_verifier_grounded_filter_20260407_144241 | 316 | 0.6156 | 2.1424 | 0.8616 | 0.8552 | 2.9053 | 2.1392 | 0.6062 | 0.7788 |
| verifier_intrinsic_filter | citebench_verifier_verifier_intrinsic_filter_20260409_105853 | 316 | 0.6096 | 2.1804 | 0.8869 | 0.8766 | 2.8561 | 2.1772 | 0.5833 | 0.7688 |
| verifier_nli_filter | citebench_verifier_verifier_nli_filter_20260409_120024 | 316 | 0.8046 | 2.1203 | 0.8677 | 0.8627 | 4.0090 | 2.1203 | 0.7701 | 0.8358 |
| verifier_self_agreement_filter | citebench_verifier_verifier_self_agreement_filter_20260409_152024 | 316 | 0.6037 | 2.1867 | 0.8841 | 0.8625 | 2.8379 | 2.1867 | 0.5752 | 0.7652 |

**Column descriptions:**
- **Statement Rating**: Overall faithfulness/quality of the generated response (=Response Rating)
- **Avg Length**: Average number of cited sentences per response
- **Density**: Average citation density
- **CA Retrieval Ratio**: Proportion of sentences classified as retrieval-attributed (Citation Attribution module)
- **CE Mean Sent Rating**: Mean citation quality rating per sentence (Citation Evaluation module)
- **CE Sent Coverage**: Coverage of evaluated sentences (Citation Evaluation module)
- **CR IterCoE (Answer)**: Mean answer-level citation recall via iterative CoE (Citation Recall module)
- **CR EditDist (Answer)**: Mean answer-level citation recall via edit distance (Citation Recall module)

---

## Verifier Filtering Statistics

| Variant | Run Folder | Total Claims | Filtered Claims | Filter Rate | Avg NLI Entailment | Avg Entropy | Avg Token F1 | Avg Recall |
| --- | --- | ---: | ---: | ---: | ---: | ---: | ---: | ---: |
| full_verifier_filter | citebench_verifier_full_verifier_filter_20260329_113436 | 685 | 22 | 0.0321 | 0.8395 | 0.0760 | 0.2896 | 0.2347 |
| verifier_grounded_filter | citebench_verifier_verifier_grounded_filter_20260407_144241 | 727 | 14 | 0.0193 | 0.0000 | 0.0000 | 0.3150 | 0.2542 |
| verifier_intrinsic_filter | citebench_verifier_verifier_intrinsic_filter_20260409_105853 | 727 | 0 | 0.0000 | 0.0000 | 0.0125 | 0.3144 | 0.2539 |
| verifier_nli_filter | citebench_verifier_verifier_nli_filter_20260409_120024 | 715 | 31 | 0.0434 | 0.8306 | 0.0000 | 0.3130 | 0.2518 |
| verifier_self_agreement_filter | citebench_verifier_verifier_self_agreement_filter_20260409_152024 | 731 | 0 | 0.0000 | 0.0000 | 0.0000 | 0.3146 | 0.2557 |

---

## Module Metrics: Citation Attribution (CA)

| Variant | Classified Sentences | Type: Retrieval | Type: Model | Type: Response | Type: Query |
| --- | ---: | ---: | ---: | ---: | ---: |
| LettuceDetect | 316 | 9 | 293 | 14 | — |
| full_verifier_filter | 658 | 572 | 39 | 45 | 2 |
| verifier_grounded_filter | 677 | 579 | 45 | 53 | — |
| verifier_intrinsic_filter | 689 | 604 | 37 | 48 | — |
| verifier_nli_filter | 670 | 578 | 44 | 48 | — |
| verifier_self_agreement_filter | 691 | 596 | 31 | 64 | — |

---

## Module Metrics: Citation Evaluation (CE)

| Variant | Sentence Ratings | Mean Sentence Rating | Sentence Coverage |
| --- | ---: | ---: | ---: |
| LettuceDetect | 239 | 1.7029 | 0.7563 |
| full_verifier_filter | 657 | 2.7504 | 2.0791 |
| verifier_grounded_filter | 676 | 2.9053 | 2.1392 |
| verifier_intrinsic_filter | 688 | 2.8561 | 2.1772 |
| verifier_nli_filter | 670 | 4.0090 | 2.1203 |
| verifier_self_agreement_filter | 691 | 2.8379 | 2.1867 |

---

## Module Metrics: Citation Recall — Iterative CoE (CR IterCoE)

| Variant | Answer Ratings | Mean Answer Rating | Sentence Ratings | Mean Sentence Rating | Sentence Coverage |
| --- | ---: | ---: | ---: | ---: | ---: |
| LettuceDetect | 316 | 0.9241 | 30 | 0.2000 | 0.0949 |
| full_verifier_filter | 316 | 0.5798 | 602 | 0.4186 | 1.9051 |
| verifier_grounded_filter | 316 | 0.6062 | 624 | 0.4663 | 1.9747 |
| verifier_intrinsic_filter | 316 | 0.5833 | 645 | 0.4593 | 2.0411 |
| verifier_nli_filter | 316 | 0.7701 | 612 | 0.7606 | 1.9367 |
| verifier_self_agreement_filter | 316 | 0.5752 | 647 | 0.4490 | 2.0475 |

---

## Module Metrics: Citation Recall — Edit Distance (CR EditDist)

| Variant | Answer Ratings | Mean Answer Rating | Sentence Ratings | Mean Sentence Rating | Sentence Coverage |
| --- | ---: | ---: | ---: | ---: | ---: |
| LettuceDetect | 316 | 0.9364 | 30 | 0.3304 | 0.0949 |
| full_verifier_filter | 316 | 0.7849 | 602 | 0.7153 | 1.9051 |
| verifier_grounded_filter | 316 | 0.7788 | 624 | 0.7221 | 1.9747 |
| verifier_intrinsic_filter | 316 | 0.7688 | 645 | 0.7257 | 2.0411 |
| verifier_nli_filter | 316 | 0.8358 | 612 | 0.8404 | 1.9367 |
| verifier_self_agreement_filter | 316 | 0.7652 | 647 | 0.7233 | 2.0475 |
