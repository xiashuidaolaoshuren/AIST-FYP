# Technical and Architectural Limitations

## 1. Decontextualized Chunking Destroys NLI Grounding

### Limitation and Manifestation
A significant technical limitation of the current pipeline is that decontextualized chunking destroys the grounding required for Natural Language Inference (NLI). The text chunker currently employs sentence-level chunking without overlap (i.e., `overlap_sentences=0`). This approach causes a severe loss of discourse context; chunks containing unresolved pronouns (e.g., "it", "he", "she") or orphaned entity references lack an antecedent within the same chunk. Consequently, when the DeBERTa NLI model processes such a chunk as a premise, it cannot reliably resolve the subjects, making entailment or contradiction judgments highly uncertain.

### Evidence
This issue aligns with recent findings on contextual retrieval [1], which demonstrated that traditional retrieval without context exhibits a failure rate of 5.7%. However, when context is restored, this failure rate significantly improves to 2.9% (a 49% reduction).

### Mitigation Strategy
To mitigate this limitation, future implementations should adopt contextual retrieval strategies—such as prepending a short (50–100 token) LLM-generated context summary to each chunk before embedding and indexing—to situate the chunk within its source document and resolve ambiguities before NLI processing.

## 2. Sentence-Level NLI Cannot Reason Across Multi-Sentence Evidence

### Limitation and Manifestation
A second architectural limitation is that sentence-level NLI cannot effectively reason across multi-sentence evidence. The DeBERTa-v3 NLI model—trained on MNLI, FEVER, and ANLI—evaluates a single premise-hypothesis sentence pair in isolation. However, in complex Question-Answering (QA) tasks, a hallucination often spans the original question, a multi-sentence answer, and an extended evidence passage. Because the model never observes the broader query context when scoring a single claim, the premise remains semantically incomplete, leading to signal blindness.

### Evidence
This limitation was starkly evident in our Evaluation 24 structural analysis, where the current approach yielded an 83.6% false-negative rate on QA tasks (failing to detect 46 out of 55 gold-positive QA samples). Furthermore, a severe "signal inversion" was observed in QA tasks: False Positive (FP) samples exhibited a higher mean maximum contradiction probability (0.917) than True Positive (TP) samples (0.881). As explicitly noted in the development of SummaC [2], sentence-level decomposition is prone to missing inconsistencies that require reasoning across multiple sentences from the source document.

### Mitigation Strategy
A viable mitigation strategy would involve expanding the premise to passage-level NLI or explicitly prepending the original query to the hypothesis (Question-Conditioned NLI) before scoring.

## 3. Rule-Based Aggregator Cannot Separate Overlapping Signal Distributions

### Limitation and Manifestation
Finally, the rule-based aggregator fails to cleanly separate overlapping signal distributions. The current aggregator module relies on cascading Boolean thresholds (veto logic), which assumes that each detection signal (e.g., NLI contradiction probability, entailment coverage) acts as a highly discriminative boundary. In practice, the detection signal distributions for True Positives and False Positives overlap significantly, meaning that hard static thresholds cannot cleanly separate valid claims from true hallucinations.

### Evidence
During the Evaluation 24 analysis on summarization tasks, False Positive samples frequently appeared better grounded than True Positives; the FP mean for average coverage was 0.764, compared to the TP mean of 0.745. Similarly, the FP mean maximum contradiction probability (0.574) was too close to the TP mean (0.645) for static thresholds to be strictly effective. Consequently, even after applying all cascaded guardrails, the system leaked 124 FP samples against only 54 TP samples, yielding a problematic 2.3:1 FP-to-TP ratio.

### Mitigation Strategy
To address this, the brittle Boolean threshold logic should be replaced with a lightweight learned aggregator, such as a logistic regression model applied over the signal feature vector. Such a composite scoring approach can intelligently weigh the interactions between signals rather than relying on strict boolean cutoffs.

### References

[1] Anthropic, "Contextual Retrieval," Anthropic, Sep. 2024. [Online]. Available: https://www.anthropic.com/news/contextual-retrieval
[2] P. Laban, T. Schnabel, P. N. Bennett, and M. A. Hearst, "SummaC: Re-Visiting NLI-based Models for Inconsistency Detection in Summarization," *arXiv preprint arXiv:2111.09525*, 2022.