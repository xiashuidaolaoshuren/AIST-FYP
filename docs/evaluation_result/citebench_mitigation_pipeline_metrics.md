# CiteBench Mitigation Pipeline Evaluation Metrics

Sources:
- Mitigation runs: `c:/Users/admin/Desktop/citebench/mitigation`
- `full_verifier_filter` comparator: existing report in `docs/evaluation_result/citebench_verifier_variant_evaluation_metrics.md`

Dataset: ASQA Oracle (`asqa_oracle.dev.jsonl`), 316 queries  
Evaluator: CiteEval (`citeeval-auto-12272024`, provider: `deepseek-chat`)  
Modules: `ca`, `ce`, `cr_itercoe`, `cr_editdist`

> Note: `full_verifier_filter` is not part of the mitigation-only run batch. It is included here as a requested comparator because it represents the verifier-ablation setting where all verifier signals are enabled and only the filter actuator is applied.

---

## Overall CiteEval Metrics

| Variant | Run Folder / Source | Samples | Statement Rating | Response Rating | Avg Length | Density | CA Retrieval Ratio | CE Mean Sent Rating | CE Sent Coverage | CR IterCoE (Answer) | CR EditDist (Answer) |
| --- | --- | ---: | ---: | ---: | ---: | ---: | ---: | ---: | ---: | ---: | ---: |
| mitigation_all | citebench_mitigation_mitigation_all_20260409_091149 | 316 | 0.8158 | 0.8158 | 2.2057 | 0.8788 | 0.8666 | 4.0402 | 2.2025 | 0.7824 | 0.8457 |
| mitigation_filter_only | citebench_mitigation_mitigation_filter_only_20260407_162715 | 316 | 0.8104 | 0.8104 | 2.1203 | 0.8508 | 0.8612 | 3.9836 | 2.1203 | 0.7829 | 0.8498 |
| mitigation_rerank_only | citebench_mitigation_mitigation_rerank_only_20260407_064408 | 316 | 0.8112 | 0.8112 | 2.2057 | 0.8836 | 0.8623 | 3.9986 | 2.2057 | 0.7698 | 0.8400 |
| mitigation_reprompt_only | citebench_mitigation_mitigation_reprompt_only_20260407_095422 | 316 | 0.8216 | 0.8216 | 2.2278 | 0.8686 | 0.8707 | 4.0640 | 2.2247 | 0.7899 | 0.8489 |
| full_verifier_filter | citebench_verifier_full_verifier_filter_20260329_113436 | 316 | 0.5934 | 0.5934 | 2.0823 | 0.8290 | 0.8693 | 2.7504 | 2.0791 | 0.5798 | 0.7849 |

### Quick Read

- `mitigation_reprompt_only` is the strongest mitigation variant on the main CiteEval quality metrics: Statement Rating `0.8216`, CE Mean Sentence Rating `4.0640`, and CR IterCoE `0.7899`.
- `mitigation_filter_only` applies the most aggressive filtering: `30` filtered claims and a filter rate of `0.0413`.
- All four mitigation variants outperform `full_verifier_filter` by a large margin on Statement Rating, CE Mean Sentence Rating, and both citation recall metrics.

---

## Verifier / Filtering Statistics

| Variant | Total Claims | Filtered Claims | Filter Rate | Avg NLI Entailment | Avg Entropy | Avg Token F1 | Avg Recall |
| --- | ---: | ---: | ---: | ---: | ---: | ---: | ---: |
| mitigation_all | 736 | 24 | 0.0326 | 0.8382 | 0.0122 | 0.3141 | 0.2531 |
| mitigation_filter_only | 727 | 30 | 0.0413 | 0.8344 | 0.0129 | 0.3117 | 0.2518 |
| mitigation_rerank_only | 738 | 0 | 0.0000 | 0.8373 | 0.0123 | 0.3140 | 0.2552 |
| mitigation_reprompt_only | 757 | 0 | 0.0000 | 0.8359 | 0.0136 | 0.3151 | 0.2557 |
| full_verifier_filter | 685 | 22 | 0.0321 | 0.8395 | 0.0760 | 0.2896 | 0.2347 |

### Interpretation

- `mitigation_filter_only` removes the most claims, but this does not translate into the best overall CiteEval quality.
- `mitigation_reprompt_only` and `mitigation_rerank_only` achieve strong CiteEval scores without removing claims, suggesting their gains come from improving the generated response or evidence alignment rather than direct deletion.
- `full_verifier_filter` has a similar filter rate to `mitigation_all`, but much worse downstream citation quality, indicating that mitigation-stage edits are more effective than verifier-only filtering.

---

## Module Metrics: Citation Attribution (CA)

| Variant | Classified Sentences | Retrieval | Model | Response | Query |
| --- | ---: | ---: | ---: | ---: | ---: |
| mitigation_all | 697 | 604 | 35 | 58 | 0 |
| mitigation_filter_only | 670 | 577 | 46 | 47 | 0 |
| mitigation_rerank_only | 697 | 601 | 39 | 57 | 0 |
| mitigation_reprompt_only | 704 | 613 | 34 | 57 | 0 |
| full_verifier_filter | 658 | 572 | 39 | 45 | 2 |

---

## Module Metrics: Citation Evaluation (CE)

| Variant | Sentence Ratings | Mean Sentence Rating | Sentence Coverage |
| --- | ---: | ---: | ---: |
| mitigation_all | 696 | 4.0402 | 2.2025 |
| mitigation_filter_only | 670 | 3.9836 | 2.1203 |
| mitigation_rerank_only | 697 | 3.9986 | 2.2057 |
| mitigation_reprompt_only | 703 | 4.0640 | 2.2247 |
| full_verifier_filter | 657 | 2.7504 | 2.0791 |

---

## Module Metrics: Citation Recall - IterCoE

| Variant | Answer Ratings | Mean Answer Rating | Sentence Ratings | Mean Sentence Rating | Sentence Coverage |
| --- | ---: | ---: | ---: | ---: | ---: |
| mitigation_all | 316 | 0.7824 | 649 | 0.7720 | 2.0538 |
| mitigation_filter_only | 316 | 0.7829 | 610 | 0.7623 | 1.9304 |
| mitigation_rerank_only | 316 | 0.7698 | 652 | 0.7592 | 2.0633 |
| mitigation_reprompt_only | 316 | 0.7899 | 654 | 0.7737 | 2.0696 |
| full_verifier_filter | 316 | 0.5798 | 602 | 0.4186 | 1.9051 |

---

## Module Metrics: Citation Recall - Edit Distance

| Variant | Answer Ratings | Mean Answer Rating | Sentence Ratings | Mean Sentence Rating | Sentence Coverage |
| --- | ---: | ---: | ---: | ---: | ---: |
| mitigation_all | 316 | 0.8457 | 649 | 0.8473 | 2.0538 |
| mitigation_filter_only | 316 | 0.8498 | 610 | 0.8446 | 1.9304 |
| mitigation_rerank_only | 316 | 0.8400 | 652 | 0.8443 | 2.0633 |
| mitigation_reprompt_only | 316 | 0.8489 | 654 | 0.8505 | 2.0696 |
| full_verifier_filter | 316 | 0.7849 | 602 | 0.7153 | 1.9051 |

---

## Key Takeaways

1. `mitigation_reprompt_only` is the best single mitigation variant overall. It leads on Statement Rating, CE Mean Sentence Rating, CR IterCoE, and CA Retrieval Ratio.
2. `mitigation_filter_only` is the strongest deletion-based variant. It filters the most claims, but its quality gains are smaller than `mitigation_reprompt_only`.
3. `mitigation_rerank_only` improves citation quality over `full_verifier_filter`, but it trails the other mitigation variants on answer-level citation recall.
4. `mitigation_all` is competitive, but it does not beat `mitigation_reprompt_only`, which suggests the combined mitigation stack does not add enough complementary benefit over reprompting alone in this run.
5. All mitigation variants are substantially stronger than `full_verifier_filter`, so verifier-only filtering is not sufficient to maximize CiteBench quality; the mitigation-stage intervention matters.