# Helpful Tools for Hallucination Detection Project

This document outlines tools and resources that can be beneficial for the AIST-FYP project on LLM hallucination.

## 1. Ragas: A Framework for Evaluating RAG Pipelines

Ragas is an open-source framework designed to evaluate Retrieval-Augmented Generation (RAG) pipelines. This is highly relevant to the project, as the system architecture is based on a RAG pipeline. Ragas provides a set of metrics to assess the performance of different components of the pipeline, which can be directly applied to the "Verifier" module to detect and quantify hallucinations.

### Key Benefits for the Project:

- **Standardized Evaluation:** Provides a standardized way to measure the quality of the system's outputs, making it easier to compare different approaches and track improvements.
- **Component-wise Metrics:** Offers metrics to evaluate not just the final answer, but also the performance of the retrieval and generation components separately.
- **Hallucination-focused Metrics:** Many of the core metrics are directly aimed at identifying hallucinations and ensuring factual consistency.

### Relevant Ragas Metrics:

- **`Faithfulness`**: Evaluates whether the generated answer is factually consistent with the information present in the retrieved context. A low faithfulness score indicates a potential hallucination.
- **`Answer Relevancy`**: Measures how relevant the generated answer is to the original prompt.
- **`Context Precision` and `Context Recall`**: These metrics evaluate the performance of the retriever.
- **`Factual Correctness`**: Assesses if the answer is factually correct with respect to a ground truth answer.

By integrating Ragas, the project can systematically evaluate its hallucination detection and mitigation strategies.

## 2. Hugging Face: Models and Datasets for a Trainless Approach

Hugging Face is an essential resource for this project, especially for the new **trainless** approach. It provides access to off-the-shelf models and tools that are crucial for building the "Verifier" module without requiring fine-tuning.

### How Hugging Face Will Be Used:

- **Zero-Shot NLI Models:** The verifier will use a pre-trained, multi-domain NLI model to check for contradictions. A strong candidate is `MoritzLaurer/DeBERTa-v3-base-mnli-fever-anli`, which is already fine-tuned on several fact-checking datasets (MNLI, FEVER, ANLI). This allows for immediate, high-quality contradiction detection without any training.
- **Cross-Encoder Models for Relevance:** A pre-trained cross-encoder, such as one from the `ms-marco` collection (e.g., `cross-encoder/ms-marco-MiniLM-L-6-v2`), can be used to score the semantic relevance between a generated claim and the retrieved evidence. A low relevance score can be a powerful heuristic for suspecting hallucination.
- **Transformers Library:** The `transformers` library from Hugging Face simplifies the process of downloading and using these pre-trained models for inference.
- **Datasets for Evaluation:** While the approach is trainless, datasets like `FEVER`, `TruthfulQA`, and `RAGTruth` remain critical for **evaluating** the effectiveness of the different zero-shot signals and the overall detector.
