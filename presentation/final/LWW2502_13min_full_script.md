# LWW2502 Project Presentation (13 Minutes)
## Scope: Introduction, Evaluation, Limitation, Conclusion

Use this file as a slide-by-slide content script. Each slide includes:
- On-slide text: what to place on the slide
- Speaker notes: what to say
- Time: suggested speaking time

---

## Slide 4 - Evaluation Design and Benchmarks (1:00)

### On-slide text
**Three evaluation dimensions (from final report)**
1. **Factual Hallucination Detection (RAGTruth)**
	- claim-level labeling quality (Accuracy / Precision / Recall / F1)
2. **Verifier Signal Evaluation (CiteBench/CiteEval)**
	- compare verifier signals under the same deterministic filter
3. **Mitigation Pipeline Evaluation (CiteBench/CiteEval)**
	- compare actuators: filter / rerank / reprompt

**Core principle**
- Separate detection quality from mitigation quality
- Evaluate both label correctness and final citation quality

**Refs:** [8], [10], [18]

### Speaker notes
Based on the final report, our evaluation is organized into three analytical dimensions. First, RAGTruth measures pure claim-level hallucination detection quality. Second, on CiteBench with CiteEval, we isolate verifier signal contribution by fixing the actuator to deterministic filtering. Third, we compare mitigation actuators such as filtering, reranking, and reprompting. The key idea is to separate detection quality from mitigation quality, because accurate labels alone do not guarantee better final citation outputs.

---

## Slide 5 - Evaluation Workflow: RAGTruth (0:50)

### On-slide text
**RAGTruth: claim-level factual detection pipeline**
1. Data preparation by task type:
	- Data2txt: table-structured context
	- QA: passage context
	- Summary: document context
2. Hybrid generation:
	- Dense retrieval (FAISS) + sparse retrieval (BM25)
	- Generate answer, then split into atomic claims
3. Multi-signal verification per claim:
	- Entropy, grounded coverage, self-agreement, NLI
4. Final labeling and scoring:
	- Supported / Contradictory / Low Confidence
	- Accuracy, Precision, Recall, F1

**Refs:** [10]

### Speaker notes
For RAGTruth, we evaluate pure detection quality only. The key design is claim-level decomposition and signal fusion. We compare predicted claim labels with ground truth and report standard classification metrics. This isolates verifier quality before any mitigation rewrites happen.

---

## Slide 6 - Evaluation Workflow: CiteBench/CiteEval (0:55)

### On-slide text
**CiteBench: mitigation-aware citation quality pipeline**
1. Citation injection:
	- Map claims to ranked evidence
	- Insert bracket citations like [1], [2]
2. Verification-aware action (for verifier ablation):
	- Use a fixed deterministic filter to remove contradictory claims
3. CiteEval scoring on final text string:
	- **CA (Context Attribution)**: Classifies whether each sentence requires a citation from: Retrieval, Model (parametric), Response (logic), or Query. Higher % Retrieval = stronger grounding.
	- **CE (Citation Evaluation)**: Human raters score citation relevance on 1–5 scale. We compute mean sentence rating (higher = better citation quality, ~4.0 is "Good").
	- **CR IterCoE**: Iterative Chain-of-Evaluation; scores logical reasoning steps 0–1 normalized scale. Measures how well citation logic chains together.
	- **CR EditDist**: Structural edit distance; counts delete/add operations needed to repair citations, converted to 0–1 similarity. Higher = fewer edits needed.

**Refs:** [8]

### Speaker notes
CiteEval scoring happens on the final response string with citations, so this workflow includes citation injection, verification-aware action, and final module-based scoring. The detailed rationale for using a fixed filter in verifier ablation is shown on the next slide.

---

## Slide 7 - Why We Use Filter in Verifier Ablation (0:40)

### On-slide text
**Filter is chosen for measurement fairness, not because it is the best mitigation**
- **CiteEval scores the final submitted string**: if a verifier only flags a claim but text is unchanged, score change is near zero.
- **Filter converts labels into concrete edits**: contradictory claims are deterministically removed, making verifier impact observable.
- **Deterministic and surgical control**: same actuator across variants, with minimal rewrite side effects and limited structural drift.
- **Empirical intervention is small**: highest observed filter rate is about **4.3%** (e.g., 31/715), so most wording stays intact.

**Why not rerank/reprompt in this ablation?**
- They regenerate new text, introducing model-generation variance beyond verifier signal quality.
- That mixes "detector quality" with "generator behavior", making cross-signal comparison not strictly comparable.

**Refs:** [8]

### Speaker notes
This is the key experimental control from the final report. In verifier ablation, our target is to compare detection signals, not generation creativity. Because CiteEval evaluates final text, detection must be acted upon to become measurable. A fixed deterministic filter provides that action with minimal disruption, so differences in CiteEval outcomes can be attributed more directly to signal quality. If we used reranking or reprompting here, newly generated content would add variance and confound the attribution.

---

## Slide 8 - Evaluation Setup Details (0:45)

### On-slide text
**Reproducible open-model setup**
- **Framework stack**: PyTorch + Hugging Face Transformers + Sentence-Transformers
- **Text generation**: Qwen3-4B-Instruct (8-bit quantized)
- **Dense retrieval / self-agreement semantic clustering**: all-MiniLM-L6-v2
- **Indexing / retrieval runtime**: FAISS (with BM25 in hybrid retrieval)
- **Zero-shot NLI signal**: DeBERTa-v3-large-mnli-fever-anli-ling-wanli
- **NER and sentence boundary detection**: spaCy en_core_web_sm (v3.7.1)
- **CiteEval citation scoring sub-modules**: DeepSeek-V3 API (deepseek-chat)

**LettuceDetect baseline (report appendix setting)**
- Model: KRLabsOrg/lettucedect-base-modernbert-en-v1
- Role: token-level hallucination detector used as comparison baseline in evaluation workflow

**Evaluation protocol details**
- Baseline comparison: LettuceDetect vs our trainless variants
- Signal ablation: nli-only, grounded-only, intrinsic-only, self-agreement-only, full verifier
- Mitigation ablation: filter-only, rerank-only, reprompt-only, mitigation-all

**Refs:** [18], [20], [21], [22], [23], [24], [27], [28], [31], [32]

### Speaker notes
This slide now matches the report setup details: we use only open-weight components, with Qwen for generation, MiniLM plus FAISS/BM25 for retrieval, DeBERTa-v3-large-mnli-fever-anli-ling-wanli as the zero-shot NLI verifier, spaCy for NER and sentence boundaries, and DeepSeek-V3 for CiteEval scoring modules. For baseline, we follow the report appendix workflow using KRLabsOrg/lettucedect-base-modernbert-en-v1. We then run signal and mitigation ablations under the same protocol.

---

## Slide 9 - Evaluation Part I: Verifier on RAGTruth (0:20)

### On-slide text
**Part I: Verifier evaluated on RAGTruth**
- Goal: factual hallucination detection quality at claim level
- Output labels: Supported / Contradictory / Low Confidence
- Structure: Detailed tables -> Part summary table -> Analysis

**Refs:** [10], [18]

### Speaker notes
The first evaluation part is pure verifier detection on RAGTruth. We first present detailed tables, then a part-level summary table slide, and finally the in-depth analysis.

---




## Slide 10 - RAGTruth Overall Metrics

### On-slide title
RAGTruth Overall Detection Metrics (All Variants)

### On-slide table
| Variant | Accuracy | Precision | Recall | F1 | Samples | TP | TN | FP | FN |
| --- | ---: | ---: | ---: | ---: | ---: | ---: | ---: | ---: | ---: |
| LettuceDetect | - | 0.7664 | 0.7550 | 0.7607 | - | - | - | - | - |
| full_verifier | 0.7361 | 0.5921 | 0.8196 | 0.6875 | 2880 | 836 | 1284 | 576 | 184 |
| verifier_nli_only | 0.5819 | 0.4502 | 0.8157 | 0.5802 | 2880 | 832 | 844 | 1016 | 188 |
| verifier_grounded_only | 0.5917 | 0.4593 | 0.8627 | 0.5995 | 2880 | 880 | 824 | 1036 | 140 |
| verifier_intrinsic_only | 0.6458 | 0.0000 | 0.0000 | 0.0000 | 2880 | 0 | 1860 | 0 | 900 |
| verifier_self_agreement_only | 0.6458 | 0.0000 | 0.0000 | 0.0000 | 2880 | 0 | 1860 | 0 | 1020 |

### Speaker notes
This table shows the complete Part I overall comparison. NLI is the strongest single detection signal, while intrinsic and self-agreement cannot detect hallucinations independently.

---

## Slide 11 - RAGTruth Per-Task (Data2txt)

### On-slide title
RAGTruth Data2txt Metrics

### On-slide table
| Variant | Accuracy | Precision | Recall | F1 | TP | TN | FP | FN |
| --- | ---: | ---: | ---: | ---: | ---: | ---: | ---: | ---: |
| LettuceDetect | - | 0.8930 | 0.8653 | 0.8789 | - | - | - | - |
| full_verifier | 0.7583 | 0.7778 | 0.8693 | 0.8210 | 532 | 196 | 152 | 80 |
| verifier_nli_only | 0.6667 | 0.6714 | 0.9346 | 0.7814 | 572 | 68 | 280 | 40 |
| verifier_grounded_only | 0.6375 | 0.6375 | 1.0000 | 0.7786 | 612 | 0 | 348 | 0 |
| verifier_intrinsic_only | 0.3625 | 0.0000 | 0.0000 | 0.0000 | 0 | 348 | 0 | 612 |
| verifier_self_agreement_only | 0.3625 | 0.0000 | 0.0000 | 0.0000 | 0 | 348 | 0 | 612 |

### Speaker notes
On structured data, contradictions are explicit, so both full verifier and NLI-only perform strongly. Grounded-only reaches perfect recall but creates many false positives.

---

## Slide 12 - RAGTruth Per-Task (QA)

### On-slide title
RAGTruth QA Metrics

### On-slide table
| Variant | Accuracy | Precision | Recall | F1 | TP | TN | FP | FN |
| --- | ---: | ---: | ---: | ---: | ---: | ---: | ---: | ---: |
| LettuceDetect | - | 0.6064 | 0.7125 | 0.6552 | - | - | - | - |
| full_verifier | 0.8250 | 0.5000 | 0.5714 | 0.5333 | 96 | 696 | 96 | 72 |
| verifier_nli_only | 0.5792 | 0.2342 | 0.6190 | 0.3399 | 104 | 452 | 340 | 64 |
| verifier_grounded_only | 0.8500 | 0.7143 | 0.2381 | 0.3571 | 40 | 776 | 16 | 128 |
| verifier_intrinsic_only | 0.8250 | 0.0000 | 0.0000 | 0.0000 | 0 | 792 | 0 | 168 |
| verifier_self_agreement_only | 0.8250 | 0.0000 | 0.0000 | 0.0000 | 0 | 792 | 0 | 168 |

### Speaker notes
QA is the hardest split for the trainless pipeline. The main bottleneck is context mismatch and NLI leakage, which lowers both precision and recall ceiling.

---

## Slide 13 - RAGTruth Per-Task (Summary)

### On-slide title
RAGTruth Summary Metrics

### On-slide table
| Variant | Accuracy | Precision | Recall | F1 | TP | TN | FP | FN |
| --- | ---: | ---: | ---: | ---: | ---: | ---: | ---: | ---: |
| LettuceDetect | - | 0.5389 | 0.4755 | 0.5052 | - | - | - | - |
| full_verifier | 0.6250 | 0.3881 | 0.8667 | 0.5361 | 208 | 392 | 328 | 32 |
| verifier_nli_only | 0.5000 | 0.2826 | 0.6500 | 0.3939 | 156 | 324 | 396 | 84 |
| verifier_grounded_only | 0.2875 | 0.2533 | 0.9500 | 0.4000 | 228 | 48 | 672 | 12 |
| verifier_intrinsic_only | 0.7500 | 0.0000 | 0.0000 | 0.0000 | 0 | 720 | 0 | 240 |
| verifier_self_agreement_only | 0.7500 | 0.0000 | 0.0000 | 0.0000 | 0 | 720 | 0 | 240 |

### Speaker notes
On long summaries, the full verifier beats LettuceDetect in F1 by recovering many hallucinations through high recall, with a higher false positive trade-off.

---

## Slide 14 - Part I Tables: RAGTruth Detection Results (1:20)

### On-slide text
**Overall (Table 1)**
- LettuceDetect: P 0.7664, R 0.7550, F1 0.7607
- full_verifier: P 0.5921, **R 0.8196**, F1 0.6875
- full_verifier error profile: **FP 576, FN 184** (safety-first)

**Per-task (Tables 2-4, full_verifier F1)**
- Data2txt: **0.8210**
- QA: 0.5333
- Summary: 0.5361

**Signal ablation highlights**
- nli_only: R 0.8157, but FP 1016
- grounded_only: R 0.8627, but FP 1036
- intrinsic_only / self_agreement_only: 0 TP
- NLI + coverage guard: FP **1016 -> 576** with only **4 TP** loss

**Refs:** [10], [18]

### Speaker notes
These tables show a clear safety-first profile. Compared with the fine-tuned baseline, our full verifier trades precision for higher recall, ending at recall 0.8196 with 184 false negatives and 576 false positives. In our design, missing hallucinations is the more severe failure mode, so this trade-off is intentional. The per-task breakdown also shows a large domain gap: structured Data2txt is much easier than QA and Summary. Ablation further confirms that NLI is the indispensable detection core, and grounded coverage acts as a precision guard by cutting false positives from 1016 to 576 with only 4 true-positive losses.

---

## Slide 15 - Part I In-Depth Analysis: RAGTruth (0:55)

### On-slide text
**In-depth analysis from report**
- Design philosophy: minimize FN (safety-first), accept higher FP
- NLI is the indispensable core detection signal
- Grounded coverage works as precision guard with NLI
- QA/Summary gap causes:
	- structural/dataset mismatch
	- claim granularity mismatch (clause split)
	- evidence alignment asymmetry
	- NLI leakage (QA leakage **26.3%**)
- Trigger-path efficiency gap:
	- contradictory path FP rate: Data2txt **20%** vs QA **42%** / Summary **41%**

**Refs:** [10], [18]

### Speaker notes
The in-depth analysis explains both the design choice and the bottlenecks. We deliberately optimize for low false negatives, so we accept higher false alarms. But the QA and Summary drop is not random noise. It comes from dataset structure mismatch, clause-level claim splitting, and weaker evidence alignment. The strongest bottleneck is NLI leakage: in QA, 26.3% of gold hallucinations are incorrectly passed as supported, which creates a hard recall ceiling. Trigger-path analysis also shows reliability drift across tasks: the contradictory rule is much cleaner on Data2txt than on QA and Summary.

---

## Slide 16 - Evaluation Part II: Verifier on CiteEval (0:20)

### On-slide text
**Part II: Verifier evaluated on CiteEval**
- Goal: citation quality impact of verifier signals
- Setting: each verifier variant paired with the same filter actuator
- Structure: Detailed tables -> Part summary table -> Analysis

**Refs:** [8], [18]

### Speaker notes
The second part evaluates verifier quality in CiteEval terms. We use the same sequence: detailed tables first, then the part summary table slide, then detailed analysis.

---




## Slide 17 - CiteEval Verifier Overall (Table 8)

### On-slide title
Verifier Signals on CiteEval (Overall)

### On-slide table
| Variant | Statement Rating | Density | CA Retrieval Ratio | CE Mean Sent Rating | CE Sent Coverage | CR IterCoE | CR EditDist |
| --- | ---: | ---: | ---: | ---: | ---: | ---: | ---: |
| LettuceDetect | - | - | - | 1.7029 | 0.7563 | 0.9241 | 0.9364 |
| full_verifier_filter | 0.5934 | 0.8290 | 0.8693 | 2.7504 | 2.0791 | 0.5798 | 0.7849 |
| verifier_nli_filter | **0.8046** | 0.8677 | 0.8627 | **4.0090** | 2.1203 | 0.7701 | 0.8358 |
| verifier_grounded_filter | 0.6156 | 0.8616 | 0.8552 | 2.9053 | 2.1392 | 0.6062 | 0.7788 |
| verifier_intrinsic_filter | 0.6096 | 0.8869 | 0.8766 | 2.8561 | 2.1772 | 0.5833 | 0.7688 |
| verifier_self_agreement_filter | 0.6037 | 0.8841 | 0.8625 | 2.8379 | 2.1867 | 0.5752 | 0.7652 |

### Speaker notes
This table is the main verifier-on-CiteEval comparison. NLI filter clearly dominates citation quality metrics.

---

## Slide 18 - Verifier Filtering Statistics (Table 9)

### On-slide title
Verifier Filtering Statistics

### On-slide table
| Variant | Filtered Claims (out of ~700) | Avg NLI Entailment | Avg Entropy | Avg Token F1 |
| --- | ---: | ---: | ---: | ---: |
| full_verifier_filter | 22 | 0.8395 | 0.0760 | 0.2896 |
| verifier_nli_filter | **31** | 0.8306 | 0.0000 | 0.3130 |
| verifier_grounded_filter | 14 | 0.0000 | 0.0000 | 0.3150 |
| verifier_intrinsic_filter | 0 | 0.0000 | 0.0125 | 0.3144 |
| verifier_self_agreement_filter | 0 | 0.0000 | 0.0000 | 0.3146 |

### Speaker notes
Filter is a controlled actuator in verifier ablation: minimal edits, measurable impact, and fair comparison across signals.

---

## Slide 19 - CiteEval Module Breakdown (Tables 10-11)

### On-slide title
CiteEval Module Breakdown (Verifier Variants)

### On-slide table
| Variant | CA Retrieval (count) | CA Model (count) | CE Mean Sentence Rating |
| --- | ---: | ---: | ---: |
| LettuceDetect | 9 | 293 | 1.7029 |
| full_verifier_filter | 572 | 39 | 2.7504 |
| verifier_nli_filter | 578 | 44 | **4.0090** |
| verifier_grounded_filter | 579 | 45 | 2.9053 |

### Speaker notes
The module-level tables show the structural grounding shift: compared with LettuceDetect, our pipeline maps far more sentences to retrieval-backed attribution.

---

## Slide 20 - Part II Tables: Verifier on CiteEval (1:20)

### On-slide text
**Verifier signal ablation (Table 8)**
- verifier_nli_filter: Statement 0.8046, **CE mean 4.0090**, CR IterCoE 0.7701
- full_verifier_filter: Statement 0.5934, CE mean 2.7504, CR IterCoE 0.5798
- LettuceDetect CE mean: **1.7029** (far below NLI-filter)

**Filtering statistics (Table 9)**
- nli_filter: **31** filtered claims (highest)
- full_verifier_filter: 22
- intrinsic/self_agreement filter: 0
- max intervention rate is still low: about **4.3%** (31/715)

**Module interpretation (Tables 10-13)**
- NLI-filter has strongest CE quality among verifier variants
- Retrieval-grounded attribution remains high in our pipeline
- full_verifier_filter under NLI-filter indicates aggregation dilution

**Refs:** [8], [18]

### Speaker notes
These results show that semantic entailment is the dominant signal for citation quality. NLI-filter reaches CE mean 4.0090 and strongly improves statement quality, while full_verifier_filter is notably lower. Even with this gain, intervention remains surgical: only around 4.3% of claims are filtered at maximum. Relative to LettuceDetect, our CiteBench pipeline also improves CE quality substantially. The key interpretation is that fixed filtering lets us measure signal contribution directly, and it reveals that current aggregation logic can dilute the best NLI signal.

---

## Slide 21 - Part II In-Depth Analysis: Verifier on CiteEval (0:55)

### On-slide text
**In-depth analysis from report**
- nli_filter is best: semantic entailment aligns best with citation criteria
- grounded_filter is moderate: catches surface errors, misses subtle contradiction
- intrinsic/self-agreement near no effect in grounded RAG outputs
- full_verifier_filter underperforms due to aggregation dilution of NLI signal
- module evidence:
	- CA shifts from Model-heavy baseline to Retrieval-heavy outputs (500+ retrieval-tagged sentences)
	- LettuceDetect high CR answer score is coverage artifact (very low sentence coverage)

**Design implication**
- Prefer NLI-first policy when optimizing CiteEval quality

**Refs:** [8]

### Speaker notes
The report conclusion is that not all verifier signals are equally useful for CiteEval objectives. NLI aligns best with semantic citation correctness, while lexical grounded checks are only partial. Intrinsic uncertainty signals barely trigger in this evidence-anchored RAG setup. Module-level tables also explain why: attribution shifts strongly toward retrieval-backed sentences, and some baseline CR scores are inflated by low evaluated sentence coverage. So for citation quality optimization, an NLI-first policy is more reliable than a broad but permissive ensemble.

---

## Slide 22 - Evaluation Part III: Mitigation on CiteEval (0:20)

### On-slide text
**Part III: Mitigation evaluated on CiteEval**
- Goal: compare mitigation actuators after verification
- Actuators: filter-only, rerank-only, reprompt-only, mitigation-all
- Structure: Detailed tables -> Part summary table -> Analysis

**Refs:** [8]

### Speaker notes
The third part shifts from detection quality to correction quality. Again, we present detailed tables first, then the part summary table slide, then mechanism-level analysis.

---




## Slide 23 - Mitigation Overall (Table 14)

### On-slide title
CiteBench/CiteEval Overall Mitigation Metrics

### On-slide table
| Variant | Statement Rating | Density | CA Retrieval Ratio | CE Mean Sent Rating | CR IterCoE (Answer) | CR EditDist (Answer) |
| --- | ---: | ---: | ---: | ---: | ---: | ---: |
| full_verifier_filter (Baseline) | 0.5934 | 0.8290 | 0.8693 | 2.7504 | 0.5798 | 0.7849 |
| mitigation_filter_only | 0.8104 | 0.8508 | 0.8612 | 3.9836 | 0.7829 | 0.8498 |
| mitigation_rerank_only | 0.8112 | 0.8836 | 0.8623 | 3.9986 | 0.7698 | 0.8400 |
| mitigation_reprompt_only | **0.8216** | 0.8686 | **0.8707** | **4.0640** | **0.7899** | **0.8489** |
| mitigation_all | 0.8158 | **0.8788** | 0.8666 | 4.0402 | 0.7824 | 0.8457 |

### Speaker notes
This table shows active mitigation is essential, and reprompt-only is the strongest single actuator on key quality metrics.

---

## Slide 24 - Mitigation Filtering Statistics (Table 15)

### On-slide title
Mitigation Filtering Statistics

### On-slide table
| Variant | Total Claims | Filtered Claims | Filter Rate | CE Sent Coverage |
| --- | ---: | ---: | ---: | ---: |
| full_verifier_filter | 685 | 22 | 0.0321 | 2.0791 |
| mitigation_filter_only | 727 | 30 | 0.0413 | 2.1203 |
| mitigation_rerank_only | 738 | 0 | 0.0000 | 2.2057 |
| mitigation_reprompt_only | 757 | 0 | 0.0000 | **2.2247** |
| mitigation_all | 736 | 24 | 0.0326 | 2.2025 |

### Speaker notes
Deleting more claims is not equal to better quality. Rerank and reprompt improve citation quality without direct deletion.

---

## Slide 25 - Part III Table: Mitigation on CiteEval (1:20)

### On-slide text
**Mitigation overall metrics (Table 14)**
- reprompt_only: **Statement 0.8216**, **CE mean 4.0640**, **CR IterCoE 0.7899**
- mitigation_all: Statement 0.8158, CE mean 4.0402
- full_verifier_filter baseline: Statement 0.5934, CE mean 2.7504
- all active mitigation variants clearly outperform verifier-only baseline

**Filtering behavior (Table 15)**
- filter_only removes most claims (30), but not best final quality
- rerank_only / reprompt_only filter rate 0, still strong quality
- mitigation_all filters similar amount as baseline (24 vs 22) but much better quality

**Refs:** [8]

### Speaker notes
This table confirms that detection alone is insufficient for final quality. Reprompt-only is the strongest single actuator across statement quality, CE, and reasoning recall. Importantly, quality gains are not explained by deleting more content: mitigation_all filters about as many claims as the verifier-only baseline, yet quality is much higher. That indicates active repair, not subtraction volume, is the key mechanism.

---

## Slide 26 - Part III In-Depth Analysis: Mitigation on CiteEval (0:55)

### On-slide text
**In-depth analysis from report**
- Detection is not correction: labeling + basic filter is insufficient
- Rewriting beats deletion: reprompt reconstructs grounded narrative better
- Ensemble interference: mitigation_all is strong but slightly below reprompt_only
- module evidence:
	- reprompt_only has highest CE coverage and best IterCoE answer rating
	- rerank improves alignment without direct deletion

**Actionable direction**
- Use adaptive sequential routing instead of static parallel stacking

**Refs:** [8]

### Speaker notes
The detailed analysis shows three mechanisms. First, detection is not correction: verifier labeling with basic filtering cannot fully repair citation logic. Second, rewriting is more effective than deletion; reprompting rebuilds coherent evidence-grounded narratives and achieves the best CE and IterCoE outcomes. Third, stacking all modules can introduce slight interference, where early deletion may remove content that reprompting could have repaired better. This motivates adaptive sequential routing rather than static parallel stacking.

---

## Slide 27 - Evaluation Summary Across 3 Parts (0:50)

### On-slide text
**Part I: RAGTruth (Detection Quality)**
- full_verifier recall **0.8196**, with safety-first FN minimization
- NLI is core detector; coverage guard cuts FP (**1016 -> 576**)
- QA bottleneck explained by NLI leakage (**26.3%**)

**Part II: CiteBench/CiteEval (Verifier Signal Quality)**
- nli_filter leads verifier variants: CE mean **4.0090**
- fixed-filter ablation isolates signal quality with low intervention (~**4.3%**)
- full ensemble can dilute strongest NLI signal under current aggregation

**Part III: CiteBench/CiteEval (Mitigation Quality)**
- reprompt_only is best single actuator: CE mean **4.0640**, IterCoE **0.7899**
- detection-only filtering is insufficient for final citation quality
- active repair (rerank/reprompt) outperforms pure deletion

**Refs:** [8], [10], [18]

### Speaker notes
Together, the three-part evaluation gives a closed-loop story. Part I shows high-recall detection behavior and explains where the recall/precision trade-off comes from. Part II isolates verifier signal value and shows semantic NLI is the strongest driver for citation quality under controlled filtering. Part III shows downstream correction quality depends on active mitigation, with reprompting as the strongest practical actuator. This means we should optimize detector policy and mitigation policy jointly rather than treating detection alone as the final objective.

---

## Slide 28 - Limitations (1:30)

### On-slide text
**Limitation 1: Decontextualized chunking**
- Sentence-level chunking with no overlap removes discourse context
- Pronouns/entities lose antecedents, so NLI premise becomes ambiguous
- Report evidence: contextual retrieval failure rate improves from 5.7% to 2.9% when context is restored

**Limitation 2: Sentence-level NLI ceiling**
- Single premise-hypothesis scoring misses cross-sentence reasoning
- In QA, model cannot fully use original question + long evidence jointly
- Report evidence: QA false-negative rate 83.6% (46/55 missed)

**Limitation 3: Rule-based aggregation rigidity**
- Boolean thresholds assume clean separability that does not hold in practice
- Signal distributions overlap between TP and FP on summarization
- Report evidence: FP:TP ratio reached 2.3:1 in structural analysis

**Refs:** [7], [19]

### Speaker notes
From the report, limitation one is not just "less context"; it is a grounding failure mode. With overlap_sentences=0, many chunks become decontextualized and ambiguous before NLI even starts. Limitation two is architectural: sentence-level NLI is forced to decide in isolation, so QA cases requiring multi-sentence and question-conditioned reasoning are often missed, which is why QA FN is very high. Limitation three is decision-layer brittleness: cascading Boolean cutoffs cannot separate overlapping TP/FP signal distributions, especially on summarization. These are the main root causes of the precision gap and unstable behavior across tasks.

---

## Slide 29 - Limitation 4: Mitigation Correctness Gap (0:45)

### On-slide text
**Limitation 4: No unified benchmark for mitigation correctness**
- CiteEval is strong for citation faithfulness/grounding, but does not directly verify whether a correction is factually right against hidden gold truth
- RAGTruth is strong for detection label correctness, but does not evaluate if mitigation actions repaired text correctly
- Existing options (e.g., FActScore-style final factual precision) treat mitigation as a black box

**Practical consequence**
- Current evaluation can show quality improvement, but cannot fully audit internal mitigation logic or collateral damage claim-by-claim

**Refs:** [8], [10], [25]

### Speaker notes
This is an evaluation limitation rather than a model-only limitation. Today, we can measure detection correctness and citation faithfulness well, but we still lack an accepted benchmark that scores the internal correctness of each mitigation action end-to-end. So our conclusions on mitigation are strong but still partially proxy-based.

---

## Slide 30 - Limitations Summary

### On-slide title
Technical and Architectural Limitations (Summary)

### On-slide table
| Limitation | Manifestation in Experiments | Evidence | Planned Fix |
| --- | --- | --- | --- |
| Decontextualized chunking | Pronouns/entities lose antecedents; NLI premise becomes ambiguous | Contextual retrieval literature: failure rate reduced from 5.7% to 2.9% when context restored | Add context prepending and overlap-aware chunking |
| Sentence-level NLI ceiling | Cannot reason across multi-sentence evidence and question conditioning | QA false-negative rate reached 83.6% in structural analysis | Passage-level NLI and question-conditioned hypothesis |
| Rule-based threshold rigidity | Overlapping TP/FP signal distributions cause high FP leakage | Summarization analysis showed severe overlap; FP:TP ratio up to 2.3:1 | Replace static thresholds with lightweight learned fusion |

**Refs:** [7], [19], [25]

### Speaker notes
This table maps each limitation directly to observed experimental symptoms and concrete engineering fixes.

---

## Slide 31 - Future Improvements (0:30)

### On-slide text
**Planned fixes**
- Contextual retrieval or overlap-aware chunking
- Passage-level or question-conditioned NLI
- Lightweight learned aggregator (e.g., logistic regression)

**Refs:** [7], [19], [25]

### Speaker notes
Future work will add context-aware retrieval chunks, move from sentence-level to passage-level or question-conditioned NLI, and replace hard threshold logic with lightweight learned fusion to better balance recall and precision.

---

## Slide 32 - Conclusion (1:00)

### On-slide text
**Conclusion**
- We built a trainless, modular verifier-mitigator pipeline
- The system achieves strong hallucination recall on RAGTruth
- Mitigation, especially re-prompting, greatly improves citation quality
- Practical path toward reliable and interpretable RAG safety layers

**Refs:** [8], [10], [18]

### Speaker notes
To conclude, our work demonstrates that a trainless and modular approach can provide strong practical value. We can catch hallucinations with high recall and meaningfully improve citation quality through mitigation. This gives a transparent and computationally efficient direction for safer RAG deployment.

---

## Backup Slide - Q&A Support (Optional)

### On-slide text
**If asked: why not just use LettuceDetect?**
- Fine-tuned baseline has higher F1 overall
- Our method has higher recall and no verifier training cost
- Better transparency and module-level interpretability

### Speaker notes
If someone asks why we do not directly use the fine-tuned baseline, the key answer is design goal. We prioritize trainless deployment, interpretability, and reducing missed hallucinations, while still achieving competitive performance and stronger downstream mitigation behavior.

---
## Optional Slide - 13-Minute Fast Version (If Time Is Tight)

### On-slide text
If only 13 minutes are available, prioritize:
1. Slide 2 (Motivation)
2. Slide 4 (Evaluation Design)
3. Slide 14 / 10 (Part I summary + overall table)
4. Slide 20 / 17 (Part II summary + overall table)
5. Slide 25 / 23 (Part III summary + overall table)
6. Slide 32 (Conclusion)

### Speaker notes
This reduced sequence still preserves the full story: problem, method, detector evidence, mitigation evidence, limitation honesty, and final conclusion.

