# LWW2502 Project Presentation — Direct Speaking Script (13 Minutes)

This script is designed to be spoken naturally, sentence by sentence. Each slide includes timing and natural language flow.

---
## Slide 4 - Evaluation Design and Benchmarks (1:00)

Our evaluation is organized into three analytical dimensions, based on the final report.

**First dimension: Factual Hallucination Detection.** We use RAGTruth to measure pure claim-level hallucination detection quality with standard metrics: Accuracy, Precision, Recall, F1.

**Second dimension: Verifier Signal Evaluation.** On CiteBench with CiteEval, we isolate verifier signal contribution by fixing the actuator to deterministic filtering. This lets us compare signals fairly.

**Third dimension: Mitigation Pipeline Evaluation.** We compare mitigation actuators — filter, rerank, reprompt — and measure their downstream citation quality.

The core principle is to separate detection quality from mitigation quality. Why? Because accurate labels alone do not guarantee better final citation outputs. A verifier might label correctly, but if the mitigation is weak, the final output might still be poor. Similarly, a good mitigation can't fix bad labels. So we study both independently.

---

## Slide 5 - Evaluation Workflow: RAGTruth (0:50)

For RAGTruth, we evaluate pure detection quality only. The pipeline has four steps:

First, **data preparation by task type.** RAGTruth has three tasks: Data2txt with table-structured context, QA with passage context, and Summary with document context. We prepare data separately for each.

Second, **hybrid generation.** We use dense retrieval with FAISS plus sparse retrieval with BM25. We generate an answer, then split it into atomic claims.

Third, **multi-signal verification per claim.** We compute four signals: entropy (intrinsic uncertainty), grounded coverage (lexical overlap with evidence), self-agreement (semantic similarity), and NLI (zero-shot entailment).

Fourth, **final labeling and scoring.** We label each claim as Supported, Contradictory, or Low Confidence, then compare with ground truth and compute standard classification metrics.

This isolates verifier quality before any mitigation rewrites happen.

---

## Slide 6 - Evaluation Workflow: CiteBench/CiteEval (0:55)

For CiteBench and CiteEval, the workflow includes three steps:

**Step one: Citation injection.** We map claims to ranked evidence and insert bracket citations like [1], [2] into the text.

**Step two: Verification-aware action.** For verifier ablation, we use a fixed deterministic filter to remove contradictory claims. This ensures all variants use the same actuator, so we can isolate signal quality.

**Step three: CiteEval scoring on final text string.** CiteEval is a multi-module scoring system. Let me break it down:

**CA, or Context Attribution**, classifies whether each sentence requires a citation from retrieval-backed sources, parametric knowledge, reasoning logic, or the query. Higher retrieval percentage means stronger grounding.

**CE, or Citation Evaluation**, has human raters score citation relevance on a 1–5 scale. We compute mean sentence rating, where around 4.0 is "Good" and higher is better.

**CR IterCoE**, iterative chain-of-evaluation, scores logical reasoning steps on a 0–1 scale. It measures how well citation logic chains together.

**CR EditDist**, structural edit distance, counts delete-and-add operations needed to repair citations, converted to a 0–1 similarity score. Higher means fewer edits needed.

CiteEval scoring happens on the final response string with citations, so this workflow includes citation injection, a fixed verification action, and final module-based scoring. The detailed rationale for the fixed filter is on the next slide.

---

## Slide 7 - Why We Use Filter in Verifier Ablation (0:40)

This is the key experimental control from the final report. Let me explain why we choose a fixed deterministic filter for verifier ablation.

The first reason: **CiteEval scores the final submitted string.** If a verifier only flags a claim but the text is unchanged, the score change is near zero. So detection must be acted upon to become measurable.

The second reason: **Filter converts labels into concrete edits.** A deterministic filter removes contradictory claims, making verifier impact observable in the final text.

The third reason: **Deterministic and surgical control.** We use the same actuator across all variants, with minimal rewrite side effects and limited structural drift. This keeps comparisons fair.

The fourth reason: **Empirical intervention is small.** The highest observed filter rate is about 4.3%, meaning 31 out of 715 claims. So most wording stays intact.

Why don't we use rerank or reprompt in this ablation? Because they regenerate new text, introducing model-generation variance beyond verifier signal quality. That would mix detector quality with generator behavior, making cross-signal comparison unreliable.

In verifier ablation, our target is to compare detection signals, not generation creativity. A fixed deterministic filter provides that measurement with minimal disruption, so differences in CiteEval outcomes can be attributed more directly to signal quality.

---

## Slide 8 - Evaluation Setup Details (0:45)

Now let me detail our reproducible open-model setup. We prioritize transparency and reproducibility.

For the **framework stack**, we use PyTorch, Hugging Face Transformers, and Sentence-Transformers.

For **text generation**, we use Qwen3-4B-Instruct quantized to 8-bit.

For **dense retrieval and self-agreement semantic clustering**, we use all-MiniLM-L6-v2.

For **indexing and retrieval runtime**, we use FAISS with BM25 for hybrid retrieval.

For **zero-shot NLI signal**, we use DeBERTa-v3-large-mnli-fever-anli-ling-wanli.

For **NER and sentence boundary detection**, we use spaCy version 3.7.1 with the English web model.

For **CiteEval citation scoring sub-modules**, we use the DeepSeek-V3 API.

We also include a **LettuceDetect baseline**, which is KRLabsOrg/lettucedect-base-modernbert-en-v1. This is a token-level hallucination detector used as a comparison baseline in our evaluation workflow.

Finally, our **evaluation protocol** runs three types of comparisons: baseline comparison between LettuceDetect and our trainless variants, signal ablation spanning nli-only, grounded-only, intrinsic-only, self-agreement-only, and the full verifier, and mitigation ablation comparing filter-only, rerank-only, reprompt-only, and mitigation-all.

---

## Slide 9 - Evaluation Part I: Verifier on RAGTruth (0:20)

Now we move to results. **Part I focuses on the verifier evaluated on RAGTruth.**

The goal is factual hallucination detection quality at the claim level.

Our output labels are: Supported, Contradictory, or Low Confidence.

The structure is: detailed tables first, then a part-level summary table slide, then in-depth analysis. This gives you both granular data and high-level insights.

---

## Slide 10 - RAGTruth Overall Metrics

Here's the overall detection metrics table for all variants on RAGTruth.

**LettuceDetect**, the fine-tuned baseline, achieves precision 0.7664, recall 0.7550, F1 0.7607.

**Our full_verifier** achieves precision 0.5921 but **recall 0.8196** and F1 0.6875. This is our safety-first design: we trade precision for higher recall.

**verifier_nli_only** — using only NLI — reaches recall 0.8157 but has 1016 false positives.

**verifier_grounded_only** reaches near-perfect recall 0.8627 but creates 1036 false positives.

**verifier_intrinsic_only and verifier_self_agreement_only** cannot detect hallucinations independently — zero TP.

The takeaway: NLI is the strongest single detection signal. Intrinsic and self-agreement alone cannot detect hallucinations. When combined with NLI, grounded coverage acts as a precision guard.

---

## Slide 11 - RAGTruth Per-Task (Data2txt)

On structured data, like Data2txt, contradictions are explicit. So both full verifier and NLI-only perform strongly. Our full verifier F1 is 0.8210.

Grounded-only reaches perfect recall 1.0 but creates many false positives due to strict lexical matching.

---

## Slide 12 - RAGTruth Per-Task (QA)

On QA, performance drops significantly. Our full verifier F1 is only 0.5333 — the hardest task. The main bottleneck is context mismatch and NLI leakage. Both precision and recall have low ceilings on this split.

---

## Slide 13 - RAGTruth Per-Task (Summary)

On long summaries, the full verifier F1 is 0.5361. Our full verifier beats LettuceDetect by recovering many hallucinations through high recall, but we accept a higher false positive trade-off.

---

## Slide 14 - Part I Tables: RAGTruth Detection Results (1:20)

Let me synthesize the Part I findings.

**For overall metrics:** LettuceDetect achieves P 0.7664, R 0.7550, F1 0.7607. Our full_verifier achieves P 0.5921 but **R 0.8196**, F1 0.6875. The error profile shows **FP 576, FN 184** — this is intentional safety-first design.

**For per-task F1 on full_verifier:** Data2txt is **0.8210**, QA is 0.5333, Summary is 0.5361. There's a large domain gap.

**For signal ablation:** nli_only reaches R 0.8157 but FP 1016. Grounded_only reaches R 0.8627 but FP 1036. Intrinsic_only and self_agreement_only produce zero TP.

**Key insight:** NLI plus coverage guard cuts false positives from 1016 to 576 with only 4 TP loss. This is a very efficient trade-off.

---

## Slide 15 - Part I In-Depth Analysis: RAGTruth (0:55)

Let me explain the bottlenecks. Our design philosophy is to minimize false negatives — that is, minimize missed hallucinations. We accept higher false positives. Why? Because in a safety system, missing hallucinations is the more severe failure.

**NLI is the indispensable core detection signal.** Without it, we get zero detection.

**Grounded coverage works as a precision guard alongside NLI.** It cuts false positives significantly.

**But QA and Summary show a large drop.** Why? Several factors:
- Structural and dataset mismatch between tasks
- Claim granularity mismatch — we split clauses but the evidence is structured differently
- Evidence alignment asymmetry
- **NLI leakage at 26.3% in QA** — meaning 26.3% of gold hallucinations slip through as "supported"

**Additionally, trigger-path efficiency varies:** The contradictory rule works cleanly on Data2txt with FP rate 20%, but on QA and Summary it's much noisier at 42% and 41%. This explains the task-level variability.

These are the root causes of our recall drop on QA and Summary.

---

## Slide 16 - Evaluation Part II: Verifier on CiteEval (0:20)

**Part II evaluates verifier quality in CiteEval terms.**

The goal is to measure citation quality impact of different verifier signals.

The setting is controlled: each verifier variant is paired with the same filter actuator.

The structure again follows: detailed tables first, then the part summary table slide, then detailed analysis.

---

## Slide 17 - CiteEval Verifier Overall (Table 8)

This is the main verifier-on-CiteEval comparison.

**verifier_nli_filter** clearly dominates. It achieves:
- Statement Rating **0.8046** (highest among variants)
- **CE Mean Sent Rating 4.0090** (excellent citation quality)
- CR IterCoE 0.7701

**full_verifier_filter** achieves Statement 0.5934, CE mean 2.7504, CR IterCoE 0.5798 — noticeably lower.

**LettuceDetect** baseline has CE mean only 1.7029 — far below our methods.

The key result: NLI filter clearly dominates citation quality metrics.

---

## Slide 18 - Verifier Filtering Statistics (Table 9)

**verifier_nli_filter** filters **31 claims** out of approximately 700 — the highest rate.

**full_verifier_filter** filters 22.

**verifier_grounded_filter** filters 14.

**verifier_intrinsic_filter and verifier_self_agreement_filter** filter 0.

The maximum intervention rate is still low — about 4.3% of claims. So even with NLI providing the strongest signal, we're being surgical, not aggressive.

---

## Slide 19 - CiteEval Module Breakdown (Tables 10-11)

The module-level tables show the structural grounding shift.

**LettuceDetect** has only 9 retrieval-backed sentences and 293 model-backed sentences, with CE mean 1.7029.

**Our variants** have 570+ retrieval-backed sentences, with CE means around 2.75–4.00.

This means compared with LettuceDetect, our pipeline maps far more sentences to retrieval-backed attribution — making the output more grounded.

---

## Slide 20 - Part II Tables: Verifier on CiteEval (1:20)

Let me synthesize Part II findings.

**For verifier signal ablation (Table 8):**
- verifier_nli_filter: Statement 0.8046, **CE mean 4.0090**, CR IterCoE 0.7701
- full_verifier_filter: Statement 0.5934, CE mean 2.7504, CR IterCoE 0.5798
- LettuceDetect CE mean: **1.7029** — far below our NLI-filter

**For filtering statistics (Table 9):**
- nli_filter: **31** filtered claims (highest)
- full_verifier_filter: 22
- intrinsic/self_agreement filter: 0
- max intervention rate: about **4.3%** (31/715)

**For module interpretation (Tables 10-13):**
- NLI-filter has strongest CE quality among all verifier variants
- Our pipeline maintains high retrieval-grounded attribution
- full_verifier_filter underperforms due to aggregation dilution of the NLI signal

These results show that semantic entailment is the dominant signal for citation quality. NLI-filter reaches CE mean 4.0090 and strongly improves statement quality. Even with this gain, intervention remains surgical — only around 4.3% maximum.

The key interpretation: fixed filtering lets us measure signal contribution directly, and it reveals that current aggregation logic can dilute the best NLI signal.

---

## Slide 21 - Part II In-Depth Analysis: Verifier on CiteEval (0:55)

Not all verifier signals are equally useful for CiteEval objectives. Here's why:

**nli_filter is best:** semantic entailment aligns best with citation correctness criteria.

**grounded_filter is moderate:** it catches surface lexical errors but misses subtle contradictions.

**intrinsic and self-agreement near zero effect** in evidence-anchored RAG outputs.

**full_verifier_filter underperforms** due to aggregation dilution of the NLI signal.

**Module evidence:** context attribution shifts strongly from model-heavy (baseline) to retrieval-heavy (our methods) — 500+ retrieval-tagged sentences. Some baseline reasoning chain scores are inflated by very low sentence coverage.

**Design implication:** when optimizing for CiteEval quality, prefer an NLI-first policy rather than a broad but permissive ensemble.

---

## Slide 22 - Evaluation Part III: Mitigation on CiteEval (0:20)

**Part III shifts from detection quality to correction quality.**

The goal is to compare mitigation actuators after verification.

We test four actuators: filter-only, rerank-only, reprompt-only, and mitigation-all.

The structure again: detailed tables, then part-level summary, then mechanism-level analysis.

---

## Slide 23 - Mitigation Overall (Table 14)

Here are the overall mitigation metrics on CiteEval.

**The baseline** (full_verifier_filter) achieves Statement 0.5934, CE mean 2.7504, CR IterCoE 0.5798.

**mitigation_filter_only** improves to Statement 0.8104, CE mean 3.9836.

**mitigation_rerank_only** achieves Statement 0.8112, CE mean 3.9986.

**mitigation_reprompt_only** is strongest: **Statement 0.8216**, **CE mean 4.0640**, **CR IterCoE 0.7899**.

**mitigation_all** achieves Statement 0.8158, CE mean 4.0402.

Key takeaway: active mitigation is essential, and reprompt-only is the strongest single actuator on key quality metrics.

---

## Slide 24 - Mitigation Filtering Statistics (Table 15)

**full_verifier_filter** filters 22 claims out of 685 total — 3.21% rate, CE coverage 2.0791.

**mitigation_filter_only** filters 30 — 4.13% rate, CE coverage 2.1203.

**mitigation_rerank_only** filters 0 — 0% rate, but CE coverage improves to 2.2057.

**mitigation_reprompt_only** filters 0 — 0% rate, and CE coverage is highest: **2.2247**.

**mitigation_all** filters 24 — 3.26% rate, CE coverage 2.2025.

The key insight: deleting more claims is not equal to better quality. Rerank and reprompt improve citation quality without direct deletion. This shows active repair is more important than aggressive filtering.

---

## Slide 25 - Part III Table: Mitigation on CiteEval (1:20)

Synthesizing Part III findings:

**For overall metrics (Table 14):**
- reprompt_only: **Statement 0.8216**, **CE mean 4.0640**, **CR IterCoE 0.7899**
- mitigation_all: Statement 0.8158, CE mean 4.0402
- full_verifier_filter baseline: Statement 0.5934, CE mean 2.7504
- All active mitigation variants clearly outperform verifier-only baseline

**For filtering behavior (Table 15):**
- filter_only removes most claims (30), but doesn't achieve best final quality
- rerank_only and reprompt_only filter rate 0, yet still achieve strong quality
- mitigation_all filters about as many claims as baseline (24 vs 22) but much better quality

Quality gains are not explained by deleting more content. That indicates active repair, not subtraction volume, is the key mechanism.

---

## Slide 26 - Part III In-Depth Analysis: Mitigation on CiteEval (0:55)

Three key mechanisms emerge:

**Detection is not correction.** Verifier labeling with basic filtering cannot fully repair citation logic. You need downstream repair.

**Rewriting beats deletion.** Reprompting reconstructs grounded evidence-based narratives and achieves the best CE and IterCoE outcomes. Active repair outperforms pure subtraction.

**Ensemble interference.** Mitigation_all is strong but slightly below reprompt_only alone. This suggests that stacking all modules creates slight interference. Early deletion may remove content that reprompting could have repaired better.

**Module evidence:** reprompt_only has highest CE coverage and best answer-level IterCoE rating. Rerank improves alignment without direct deletion.

**Actionable direction:** use adaptive sequential routing instead of static parallel stacking. Choose actions based on what the verifier actually detected, rather than applying all transformations uniformly.

---

## Slide 27 - Evaluation Summary Across 3 Parts (0:50)

Let me tie together all three parts into a coherent story.

**Part I: RAGTruth — Detection Quality.**
- Our full_verifier achieves recall **0.8196** with safety-first FN minimization.
- NLI is the core detector; coverage guard cuts FP from **1016 to 576** with only 4 TP loss.
- QA bottleneck is explained by NLI leakage of **26.3%**.

**Part II: CiteBench/CiteEval — Verifier Signal Quality.**
- nli_filter leads all verifier variants: CE mean **4.0090**
- Fixed-filter ablation isolates signal quality with low intervention (~**4.3%**)
- Full ensemble can dilute strongest NLI signal under current aggregation logic

**Part III: CiteBench/CiteEval — Mitigation Quality.**
- reprompt_only is best single actuator: CE mean **4.0640**, IterCoE **0.7899**
- Detection-only filtering is insufficient for final citation quality
- Active repair (rerank/reprompt) outperforms pure deletion

Together, these three parts give a complete closed-loop story. Part I shows high-recall detection behavior. Part II isolates verifier signal value in citation terms. Part III shows downstream correction quality depends on active mitigation. The implication: optimize detector policy and mitigation policy jointly, not detection alone.

---

## Slide 28 - Limitations (1:30)

Now let's be honest about limitations.

**Limitation 1: Decontextualized chunking.** Our sentence-level chunking with no overlap removes discourse context. Pronouns and entities lose their antecedents, so NLI premises become ambiguous. The report shows: contextual retrieval failure rate improves from 5.7% to 2.9% when context is restored. This is not minor; it's a real grounding failure mode.

**Limitation 2: Sentence-level NLI ceiling.** Our single premise-hypothesis scoring misses cross-sentence reasoning. In QA, the model cannot fully use the original question plus long evidence jointly. Report evidence: QA false-negative rate reached 83.6%. This architectural limitation is the main reason QA performance drops.

**Limitation 3: Rule-based aggregation rigidity.** Our Boolean thresholds assume clean separability that does not hold in practice. Signal distributions overlap between true positives and false positives on summarization. Report evidence: FP:TP ratio reached 2.3:1 in structural analysis. These cascading cutoffs create brittleness, especially on long texts.

These are the main root causes of the precision gap and unstable behavior across tasks.

---

## Slide 29 - Limitation 4: Mitigation Correctness Gap (0:45)

There's also an evaluation-level limitation.

**No unified benchmark for mitigation correctness.** CiteEval is strong for citation faithfulness and grounding, but it doesn't directly verify whether a correction is factually right against hidden gold truth. RAGTruth is strong for detection label correctness, but doesn't evaluate if mitigation actions repaired text correctly. Existing options treat mitigation as a black box.

**Practical consequence:** We can show quality improvement, but cannot fully audit internal mitigation logic or collateral damage claim-by-claim.

This is an evaluation limitation, not a model-only limitation. We have good benchmarks for detection and citation, but we lack an accepted metric for mitigation correctness across the board.

---

## Slide 30 - Limitations Summary

Let me summarize each limitation, its experimental evidence, and planned fix.

**Decontextualized chunking → Pronouns/entities lose antecedents; NLI premise ambiguous.**
- Evidence: contextual retrieval failure rate reduced from 5.7% to 2.9% when context restored
- Fix: Add context prepending and overlap-aware chunking

**Sentence-level NLI ceiling → Cannot reason across multi-sentence evidence and question conditioning.**
- Evidence: QA false-negative rate reached 83.6%
- Fix: Passage-level NLI and question-conditioned hypothesis

**Rule-based threshold rigidity → Overlapping TP/FP signal distributions cause high FP leakage.**
- Evidence: Summarization analysis showed severe overlap; FP:TP ratio up to 2.3:1
- Fix: Replace static thresholds with lightweight learned fusion

Each limitation maps directly to observed experimental symptoms and concrete engineering fixes.

---

## Slide 31 - Future Improvements (0:30)

Our planned fixes are concrete:

First, add **contextual retrieval or overlap-aware chunking** to restore discourse context.

Second, move from sentence-level to **passage-level or question-conditioned NLI** for better reasoning.

Third, replace hard threshold logic with a **lightweight learned aggregator** — for example, logistic regression — to better balance recall and precision.

These are all practical engineering improvements that directly address the root causes we identified.

---

## Slide 32 - Conclusion (1:00)

To conclude: We built a trainless, modular verifier-mitigator pipeline. The system achieves strong hallucination recall on RAGTruth — 0.8196 — with safety-first design. Mitigation, especially re-prompting, greatly improves citation quality, with CE mean reaching 4.0640 versus 2.7504 baseline. Together, this provides a practical path toward reliable and interpretable RAG safety layers.

Our work demonstrates that a trainless and modular approach can provide strong practical value. We can catch hallucinations with high recall and meaningfully improve citation quality through mitigation. This gives a transparent and computationally efficient direction for safer RAG deployment, without expensive model training.

Thank you.

---

## Optional — Q&A Support

**If asked: Why not just use LettuceDetect?**

The fine-tuned baseline has higher overall F1. But our method has higher recall and zero verifier training cost. More importantly, it offers better transparency and module-level interpretability. We can see exactly which signal contributes, and we can fix each component independently.

