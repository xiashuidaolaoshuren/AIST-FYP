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

## 2. Hugging Face: Models and Datasets for Hallucination Detection

Hugging Face is an essential resource for this project. It provides access to a vast collection of pre-trained models and datasets that are crucial for building the "Verifier" module, which relies on tasks like Natural Language Inference (NLI) and fact-checking.

### How Hugging Face Can Be Used:

- **Pre-trained Models for NLI:** The "Verifier" in the architecture uses NLI to check for contradictions between the generated response and the source documents. The Hugging Face Hub contains many pre-trained NLI models that can be used for this purpose. A "contradiction" classification from one of these models is a strong signal of a hallucination.
- **Datasets for Fine-tuning:** The Hub hosts numerous NLI and fact-checking datasets (like SNLI, MNLI, and FEVER) that can be used to fine-tune models for better performance on specific domains.
- **Transformers Library:** The `transformers` library from Hugging Face simplifies the process of downloading and using these pre-trained models.
