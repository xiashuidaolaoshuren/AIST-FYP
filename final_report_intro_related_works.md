# Introduction and Related Works

## 1. Introduction

Large Language Models (LLMs) have demonstrated remarkable capabilities in generating human-like text across various domains. However, a critical challenge hindering their deployment in high-stakes applications is their tendency to "hallucinate"—generating information that is nonsensical, factually incorrect, or unfaithful to the provided source content. This issue is particularly problematic in scenarios where the LLM is expected to cite references to support its claims, as hallucinatory citations can mislead users and erode trust in the system. 

Existing mitigation strategies often rely on massive "judge" models to evaluate and correct generations, which can be computationally expensive, prone to biases [1], and difficult to deploy in resource-constrained environments. To address these limitations, this project proposes a novel approach to detect and mitigate factual hallucinations in LLM outputs. The core objective is to develop a lightweight, plug-and-play, and **trainless** verifier module ensemble. Instead of relying on a single monolithic judge model, our system synergistically combines multiple zero-shot signals:
1. **Intrinsic Uncertainty**: Utilizing decoder statistics such as token-level entropy.
2. **Self-Agreement Methods**: Analyzing response variability across multiple generations (Self-Consistency).
3. **Retrieval-Grounded Heuristics**: Calculating evidence coverage and lexical overlap against a retrieved knowledge base.
4. **Zero-Shot Natural Language Inference (NLI)**: Employing off-the-shelf NLI models to rigorously check for entailment and contradiction between the LLM's claims and the retrieved evidence (e.g., against Wikipedia corpora).

By integrating a generator-retriever-verifier pipeline, this project aims to provide a robust, interpretable (via confidence UI and citation formatting), and efficient solution for ensuring factual consistency and faithful source attribution in Retrieval-Augmented Generation (RAG) systems.

---

## 2. Related Works

The phenomenon of hallucination in LLMs has attracted significant research attention. Our work builds upon several key strands of existing literature, ranging from foundational fact-checking frameworks to advanced mitigation techniques.

### 2.1 Overviews and Foundational Frameworks
A comprehensive overview of hallucination types, causes, and state-of-the-art detection techniques is provided by Zhang et al. [2]. Foundational to our claim-by-claim verification approach is the **FEVER** dataset [3], which pioneered the framework of classifying claim-evidence pairs into "Supported," "Refuted," or "NotEnoughInfo." Similarly, the **KILT** benchmark [4] established the necessity of source attribution in knowledge-intensive tasks, emphasizing traceability to specific paragraphs or pages.

### 2.2 Automated Factuality Evaluation
Evaluating the factual consistency of generated text, particularly in summarization and QA, has seen rapid development. Kryscinski et al. [5] introduced models to classify claim-evidence pairs for factual consistency, providing a blueprint for modern verifier modules. Fabbri et al. [6] developed **QAFactEval**, a QA-based method that evaluates consistency by generating questions from claims and checking answers against source documents. Furthermore, **SummaC** [7] demonstrated the effectiveness of using Natural Language Inference (NLI) models for fine-grained, sentence-level alignment to detect contradictions. Most recently, **CiteEval** [8] proposed a principle-driven framework for evaluating citation quality in RAG systems, which directly informs our system's citation formatting and verification design.

### 2.3 Hallucination Benchmarking
To accurately measure the performance of hallucination detectors, specialized benchmarks are essential. **TruthfulQA** [9] highlighted that models frequently generate plausible but false answers, underscoring the need for uncertainty estimation. More specific to our RAG context, the **RAGTruth** benchmark [10] revealed that "cited hallucinations"—where models hallucinate despite having retrieved evidence—are common, thereby necessitating the fine-grained discrimination mechanisms we propose.

### 2.4 Hallucination Detection and Mitigation Techniques
Several advanced techniques have been proposed to detect and mitigate hallucinations without extensive retraining. Manakul et al. [11] introduced **SelfCheckGPT**, a zero-resource paradigm that leverages self-consistency sampling to flag unreliable statements. For active mitigation during generation, **Self-RAG** [12] allows the model to retrieve evidence and self-reflect to improve attribution. Additionally, the **Chain-of-Verification** method [13] employs an "answer-then-verify" procedure where the model automatically checks and self-corrects its initial responses, significantly boosting factuality. 

### 2.5 Critiques of LLM-as-a-Judge
While using large LLMs as evaluators has become popular, Zheng et al. [1] demonstrated that the "LLM-as-a-judge" paradigm is highly susceptible to biases, such as position, verbosity, and self-preference. These findings validate our architectural decision to eschew a single, massive LLM judge in favor of a robust, multi-signal trainless verifier that combines explicit retrieval heuristics with off-the-shelf NLI logic.

---

## References

[1] L. Zheng et al., "Judging LLM-as-a-Judge with MT-Bench and Chatbot Arena," *arXiv preprint arXiv:2306.05685*, 2023.
[2] Y. Zhang et al., "A Survey on Hallucination in Large Language Models," *arXiv preprint arXiv:2311.05232*, 2023.
[3] J. Thorne et al., "FEVER: a Large-scale Dataset for Fact Extraction and VERification," *arXiv preprint arXiv:1803.05355*, 2018.
[4] F. Petroni et al., "KILT: a Benchmark for Knowledge Intensive Language Tasks," *arXiv preprint arXiv:2009.02252*, 2021.
[5] W. Kryscinski et al., "Evaluating the Factual Consistency of Abstractive Text Summarization," *arXiv preprint arXiv:1910.12840*, 2019.
[6] A. R. Fabbri et al., "QAFactEval: Improved QA-Based Factual Consistency Evaluation for Summarization," *arXiv preprint arXiv:2112.08542*, 2021.
[7] P. Laban et al., "SummaC: Re-Visiting NLI-based Models for Inconsistency Detection in Summarization," *arXiv preprint arXiv:2111.09525*, 2022.
[8] W. Xu et al., "CiteEval: Principle-Driven Citation Evaluation for Source Attribution," *arXiv preprint arXiv:2506.01829*, 2025.
[9] S. Lin et al., "TruthfulQA: Measuring How Models Mimic Human Falsehoods," *arXiv preprint arXiv:2109.07958*, 2021.
[10] T. Vu et al., "RAGTruth: A Hallucination Benchmark for Retrieval-Augmented Generation," *arXiv preprint arXiv:2401.00396*, 2023.
[11] P. Manakul et al., "SelfCheckGPT: Zero-Resource Black-Box Hallucination Detection for Generative Large Language Models," *arXiv preprint arXiv:2303.08896*, 2023.
[12] A. Asai et al., "Self-RAG: Learning to Retrieve, Generate, and Critique through Self-Reflection," *arXiv preprint arXiv:2310.11511*, 2023.
[13] L. Gao et al., "Chain-of-Verification Reduces Hallucination in Large Language Models," *arXiv preprint arXiv:2309.11495*, 2023.
