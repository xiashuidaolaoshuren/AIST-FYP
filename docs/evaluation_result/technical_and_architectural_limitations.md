# Technical and Architectural Limitations

This section details the top three architecturally significant limitations of the trainless verifier pipeline, grounded in evaluation data and cited literature.

## 1. Decontextualized Chunking Destroys NLI Grounding

**Limitation:** Sentence-level chunking without overlap causes a severe loss of discourse context. Chunks with unresolved pronouns (e.g., "it", "he", "she") or orphaned entity references lack an antecedent within the same chunk. When this chunk is embedded and retrieved, the semantic representation is weakened. Crucially, when the DeBERTa NLI model receives this chunk as a premise, it cannot resolve what the subject refers to, making the entailment or contradiction judgment highly unreliable.

**Where it manifests:**
- `src/data_processing/text_chunker.py` — The text chunker uses `overlap_sentences=0` by default, producing sentence-level fragments.

**Evidence:**
- Anthropic's September 2024 paper on Contextual Retrieval demonstrates this exact problem (e.g., a chunk stating "The company's revenue grew by 3% over the previous quarter" loses both the company name and time period). 
- Traditional RAG retrieval without context showed a failure rate of 5.7%, which improved dramatically (to 2.9%, a 49% reduction) when context was restored.

**Mitigation Strategy:**
- **Contextual Retrieval (Anthropic):** Prepend a short (50–100 token) LLM-generated context summary to each chunk before embedding and BM25 indexing. This situates the chunk within its source document, resolving pronouns and ambiguous entities before the NLI model processes them.

## 2. Sentence-Level NLI Cannot Reason Across Multi-Sentence Evidence

**Limitation:** The DeBERTa-v3 NLI model (trained on MNLI/FEVER/ANLI) evaluates a single `(premise, hypothesis)` sentence pair in isolation. In Question-Answering (QA) tasks, a hallucination often spans a question, a multi-sentence answer, and an evidence passage. Because the model never sees the original question context when scoring a claim, the premise is semantically incomplete, leading to signal blindness.

**Where it manifests:**
- The limitation of sentence-level decomposition is fundamentally embedded in how claims are extracted and scored iteratively in the Verifier Module (`docs/month4_verifier_part2.md`).

**Evidence:**
- **Evaluation 24 Structural Analysis:** The current approach yields an 83.6% false-negative rate on QA tasks (46 out of 55 gold-positive QA samples went undetected). 
- There is a severe "signal inversion" on QA tasks: False Positive (FP) samples actually exhibit a *higher* mean `max_contradict_prob` (0.917) than True Positive (TP) samples (0.881).
- The *SummaC: Re-Visiting NLI-based Models for Inconsistency Detection in Summarization* paper explicitly notes that sentence-level decomposition "may miss inconsistencies that require reasoning across multiple sentences from the source document."

**Mitigation Strategy:**
- **Passage-level NLI or Question-Conditioned NLI:** Expand the premise to include the full evidence paragraph, or explicitly prepend the original query to the hypothesis before scoring.

## 3. Rule-Based Aggregator Cannot Separate Overlapping Signal Distributions

**Limitation:** The current Rule-Based Aggregator relies on Boolean threshold cascades (veto logic) that assume each detection signal (e.g., NLI contradiction, coverage) is highly discriminative. In practice, the signal distributions for True Positives and False Positives overlap significantly, meaning that hard thresholds cannot cleanly separate hallucinations from valid claims.

**Where it manifests:**
- `outputs/eval_analysis/eval24_structural_root_cause.md` — The cascading if-else logic fails to isolate the true hallucinations without catching valid summaries.

**Evidence:**
- **Evaluation 24 Analysis:** On summary tasks, False Positive samples appeared *better grounded* than True Positives. The FP mean for `avg_coverage` was 0.764 (compared to the TP mean of 0.745).
- The FP mean `max_contradict_prob` (0.574) was too close to the TP mean (0.645) for static thresholds to work.
- Even after applying all guard cascades, the system leaked 124 FP samples against only 54 TP samples (a 2.3:1 FP:TP ratio).

**Mitigation Strategy:**
- **Learned Aggregator:** Replace the cascaded if-else thresholds with a lightweight learned aggregator (such as logistic regression over the signal feature vector). This composite scoring approach can intelligently weigh the interaction between signals rather than relying on brittle boolean logic.
