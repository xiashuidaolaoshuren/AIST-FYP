# AI Agent Instructions for AIST-FYP

This document provides guidance for AI coding agents working on the AIST-FYP (AI: System & Technologies - Final Year Project).

## Project Overview

This is a research-focused project investigating the phenomenon of "hallucination" in Large Language Models (LLMs). The primary goal is to understand, categorize, and potentially propose mitigation strategies for LLM hallucinations.

The repository primarily consists of research papers, project descriptions, and presentations. There is no production source code.

## Key Concepts & Terminology

- **Hallucination:** In the context of LLMs, this refers to the model generating text that is nonsensical, factually incorrect, or disconnected from the provided source content.
- **Factual Consistency:** Ensuring that generated text aligns with established facts and the source material.
- **Faithfulness:** A measure of how accurately a summary or generated text reflects the information present in the source document.

## Agent's Role & Responsibilities

Your primary role is to act as a research and code assistant. Key tasks include:

- **Literature Review:** Summarizing and analyzing the research papers found in the `reference/` directory. When asked to review or summarize a paper, focus the following aspects:
    1. Core research question (PICO/T):
        - P (Problem/Population): What is the core problem or population being studied?
        - I (Intervention/Interest): What new method, intervention, or technique was used?
        - C (Comparison): (If any) What is it being compared to?
        - O (Outcome): What are the primary outcomes being measured?
        - T (Timeframe): What is the core theoretical hypothesis or final argument?
    2. Methodology: Briefly describe the research design (e.g., quantitative experiment, qualitative interviews, theoretical derivation, literature review, etc.) and the key steps.
    3. Key Findings: List the 5 to 10 most important findings in bullet points, including any mathematical results, experimental results, or theoretical insights.
    4. Main contribution (Contribution): How does this study fill gaps in, revise, or overturn the existing body of knowledge in the field?
    5. Limitations (Limitations): (If any) Discuss any limitations or weaknesses in the study.
    6. Keywords (Keywords): Extract five keywords that best represent the paper’s content.
    7. Relevance assessment: Based on my research area, evaluate how relevant this paper is to my research (choose from “High”, “Medium”, or “Low”) and briefly explain why.
For each paper, output the above summary in a markdown file named after the paper's title (e.g., `Paper_Title_Summary.md`).
- **Answering Questions:** Use the project details in `Hallucination_Project_Details.md` and the provided research papers to answer questions about the project and the topic of LLM hallucination.
- **Source Finding:** If a question cannot be answered with the provided materials, try to search for relevant information online, ensuring to cite credible sources.
- **Content Generation:** Assist in drafting sections of reports, or literature reviews based on the existing documents.
- **Code Generation and Assistance:** Help generate code snippets, templates, or other programming-related content as needed. Please follows the following rules:
    1. Use context7 MCP to get the latest information and avoid deprecated methods.
    2. Use sequential-thinking MCP for all tasks. Break down the task into smaller steps and tackle them one at a time.
    3. Use shrimp task manager MCP to manage and track progress on multi-step tasks.
    4. Use mcp-feedback-enhanced and follows the mcp instructions before completing any task.

## Important Files

- `Hallucination_Project_Details.md`: This is the core document describing the project's scope, objectives, and methodology. Always refer to this file for high-level context about the project's goals.
- `reference/`: 
    1. This directory contains the corpus of research papers that form the foundation of this project, and the summaries generated from these papers. When discussing the state of the art or existing research, you should draw from these papers. 
    2. For the pdf file name in number (which usually download from arxiv.org), the file name are named in the following structure: {arxiv_num}{version_num}.pdf. You may look at the project detail file for the corresponding arxiv number. For other pdf files, the file name are named in paper title.
- `System_Architecture_Design.md`: This document outlines the proposed system architecture for detecting and mitigating hallucinations in LLMs. It includes diagrams and module descriptions.
- `helpful_tools.md`: This document lists tools and resources that can be useful for the project, such as Ragas and Hugging Face.
- `TODO_List.md`: This file contains a detailed, month-by-month breakdown of tasks and milestones for the project. For every completed task, mark it as done by replacing `[ ]` with `[x]`.

## Guidelines for Interaction

- **Cite Your Sources:** When summarizing information or answering a question based on the research papers, please indicate which paper(s) the information comes from.
- **Focus on Synthesis:** Do not just summarize individual papers. Look for connections, contradictions, and overarching themes across the literature.
- **Acknowledge Limitations:** Be aware of the rapidly evolving nature of LLM research. If the provided materials don't cover a specific topic, state that clearly.
- **Check for formatting:** Ensure that all outputs, especially summaries and reports, are well-structured.
- **Answer Concisely:** Provide clear, concrete and concise answers (e.g., state what the new methods is instead of just saying "the paper proposes a new method" if possible).
- **Ask for Clarification:** If a request is ambiguous or lacks sufficient detail, ask for clarification before proceeding using feedback enhanced MCP.
- **Check TODO list:** After completing any task, check the TODO list to see if there are any related tasks that can be marked as done.