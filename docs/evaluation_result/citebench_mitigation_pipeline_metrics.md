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

## Analysis

### Full Verifier vs Mitigation All

When comparing the `full_verifier_filter` baseline (which relies strictly on the verification stage to filter unsupported claims) to the `mitigation_all` variant (which employs the full suite of filtering, reranking, and reprompting), there is a stark performance gap across all major quality metrics. 

- **Statement Rating:** `mitigation_all` achieves **0.8158**, compared to just **0.5934** for `full_verifier_filter`.
- **CE Mean Sentence Rating:** `mitigation_all` reaches **4.0402**, far surpassing the baseline's **2.7504**.
- **Citation Recall (IterCoE / EditDist):** `mitigation_all` leads with **0.7824 / 0.8457**, whereas the baseline struggles at **0.5798 / 0.7849**.

Interestingly, both variants delete a similar amount of content (`mitigation_all` filters 24 claims; `full_verifier_filter` filters 22). This indicates that the massive quality increase in `mitigation_all` does not come from simply deleting more hallucinations. Instead, it proves that **active mitigation strategies (like reprompting and reranking) are required to repair the final generated text** and properly align the citations, solving issues that a purely subtractive verification filter cannot fix.

### Mitigation Variant Analysis

A closer look at the isolated mitigation modules reveals how each intervention shapes the final response:

1. **`mitigation_reprompt_only` (The Top Performer):** 
   This variant achieves the highest overall quality (Statement Rating: **0.8216**, CE Mean Sentence Rating: **4.0640**, CR IterCoE: **0.7899**). Because it asks the LLM to actively rewrite the answer using the verification signals, it can gracefully correct weakly-supported statements and re-weave the narrative, rather than abruptly cutting sentences out or just shifting evidence.
2. **`mitigation_filter_only` (The Aggressive Subtractor):** 
   This is the most aggressive variant, filtering 30 claims (a 0.0413 filter rate). While it successfully removes bad claims (improving citation quality over the baseline), it naturally reduces answer length and coverage. It cannot repair surviving weak claims, making it fundamentally limited compared to reprompting.
3. **`mitigation_rerank_only` (The Context Organizer):** 
   This variant improves evidence alignment and citation density (**0.8836**) by reordering the context provided to the model. However, because it doesn't remove unsupported content or actively rewrite the model's past mistakes, its gains in raw citation recall are moderate (CR IterCoE: **0.7698**).
4. **`mitigation_all` (The Ensemble):** 
   While extremely strong, it slightly underperforms `reprompt_only` on key metrics. This suggests that combining all interventions concurrently might introduce slight redundancies or over-corrections (e.g., aggressively filtering a claim that the reprompt module might have otherwise successfully rewritten and salvaged).

### Brief Conclusion for Report Writing

For the FYP report, the following key narratives can be elaborated by the team:

- **Detection is Not Correction:** Verifier-only labeling and basic filtering (`full_verifier_filter`) are insufficient for high-quality citation generation. CiteBench strictly evaluates the final text, verifying that an active *mitigation* phase is mandatory.
- **Rewriting beats Deletion:** The `reprompt_only` strategy proved to be the most effective mitigation technique. Allowing the LLM to holistically rewrite its answer based on verification signals yields more coherent and better-cited text than simply deleting (`filter_only`) isolated bad claims.
- **Ensemble Interference:** Stacking all mitigations (`mitigation_all`) doesn't strictly yield the best result. The interplay between aggressive filtering and reprompting can cause slight degradation compared to relying on reprompting alone. Future work may require dynamic routing (e.g., only using filtering if reprompting fails) rather than a static pipeline.