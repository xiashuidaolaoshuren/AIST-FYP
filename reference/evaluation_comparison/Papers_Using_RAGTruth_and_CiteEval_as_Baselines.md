# Papers using RAGTruth and CiteEval as baselines (as of 2026-02-10)

This list focuses on papers that **explicitly evaluate on RAGTruth** (hallucination detection in RAG), and papers that **introduce or use CiteEval/CiteBench** for citation/source-attribution evaluation.

> Note: **CiteEval is very recent (2025)**, and (based on arXiv + OpenAlex indexing) there are currently **few follow-up papers that explicitly report results on CiteEval/CiteBench**.

## RAGTruth (used as an evaluation benchmark)

- **ORION Grounded in Context: Retrieval-Based Method for Hallucination Detection** (2025)
  - arXiv: https://arxiv.org/abs/2504.15771
  - Evidence (from abstract): reports an F1 score on *“RAGTruth's response-level classification task”*.
  - **Evaluation Metrics**: F1 score of 0.83 in RAGTruth's response-level classification task.

- **LettuceDetect: A Hallucination Detection Framework for RAG Applications** (2025)
  - arXiv: https://arxiv.org/abs/2502.17125
  - Evidence (from abstract): trained on the RAGTruth benchmark and reports evaluation F1 on the RAGTruth corpus.
  - **Evaluation Metrics**: F1 score of 79.22% for example-level detection on the RAGTruth corpus.

- **Detecting Hallucinations in Retrieval-Augmented Generation via Semantic-level Internal Reasoning Graph** (2026)
  - arXiv: https://arxiv.org/abs/2601.03052
  - Evidence (from abstract): states it achieves better performance than baselines on *“RAGTruth and Dolly-15k”*.
  - **Evaluation Metrics**: Achieves better overall performance compared to state-of-the-art baselines. On RAGTruth with Llama-7B, it achieves a Precision of 73.64%, Recall of 79.83%, and F1-score of 76.61%.

## CiteEval / CiteBench (citation & attribution evaluation)

- **CiteEval: Principle-Driven Citation Evaluation for Source Attribution** (2025)
  - arXiv: https://arxiv.org/abs/2506.01829
  - Evidence (from abstract): introduces **CiteEval**, constructs **CiteBench**, and proposes **CiteEval-Auto** metrics.

- **L-CiteEval: Do Long-Context Models Truly Leverage Context for Responding?** (2024)
  - arXiv: https://arxiv.org/abs/2410.02115
  - Note: This work introduces **L-CiteEval** (a long-context citation benchmark). It is strongly related to citation-faithfulness evaluation, but it is a **different benchmark** from CiteEval/CiteBench.

## How this list was compiled

- Queried **OpenAlex** for works mentioning “RAGTruth” and “CiteEval”, then cross-checked entries using the corresponding arXiv landing pages.
- Queried the **arXiv API** for “RAGTruth” and “CiteEval” to catch newer items not fully indexed elsewhere.
