# Summary of "CiteEval: Principle-Driven Citation Evaluation for Source Attribution"

## Core Research Question (PICO/T)

- **P (Problem/Population):** The core problem is the inadequate and often misleading evaluation of citation quality in Retrieval-Augmented Generation (RAG) systems. Existing evaluation frameworks primarily rely on Natural Language Inference (NLI) to check if a statement is supported by its *cited* sources, which is a suboptimal proxy. This approach suffers from "context insufficiency" as it ignores the full set of retrieved documents, the user's query, and the model's own parametric knowledge, leading to inaccurate assessments of citation utility and trustworthiness.
- **I (Intervention/Interest):** The paper introduces `CiteEval`, a new principle-driven framework for evaluating citations. This framework is built on three core principles: 1) evaluating citations against the *full* set of retrieved sources, 2) considering contexts *beyond* retrieval (user query, response history, parametric knowledge), and 3) using *fine-grained criteria* beyond simple supportiveness (e.g., credibility, redundancy). Based on this framework, the authors developed `CiteBench`, a human-annotated benchmark, and `CiteEval-Auto`, a suite of automated, model-based metrics designed to mirror human judgment.
- **C (Comparison):** The performance of `CiteEval-Auto` is compared against existing automated, NLI-based citation evaluation metrics, including `AutoAIS`, `AttriScore`, and `LQAC`.
- **O (Outcome):** The primary outcome measured is the correlation (Pearson, Spearman, Kendall-Tau) between the automated metrics and fine-grained human judgments on citation quality. The framework is also used to benchmark the citation quality of various prominent LLMs.
- **T (Theoretical Hypothesis):** The central argument is that robust citation evaluation must move beyond binary NLI-based supportiveness. A more accurate and holistic evaluation requires assessing citations within the full context of information available to the RAG system and employing a richer, multi-faceted definition of citation quality that aligns better with human notions of trust and utility.

## Methodology

The research follows a multi-stage process:
1.  **Framework Development:** The authors first established a set of principles for what constitutes a good citation evaluation, critiquing the limitations of existing NLI-based approaches.
2.  **Benchmark Creation (`CiteBench`):** A high-quality benchmark was constructed using queries from diverse Long-Form QA datasets. Responses with citations were generated using a variety of proprietary and open-source LLMs.
3.  **Principled Human Annotation:** A three-step annotation process was designed:
    - **Context Attribution:** Each generated sentence is attributed to a source: retrieval, user query, response context, or parametric model knowledge.
    - **Critical Editing:** Annotators perform `add` or `delete` actions on citations for explicit reasons (e.g., `add-evidence`, `delete-misleading`, `add-credibility`).
    - **Likert Scale Rating:** The original citations are rated on a 1-5 scale based on fine-grained guidelines.
4.  **Automated Metric Development (`CiteEval-Auto`):** An automated pipeline was created to mimic the human annotation process. It uses an LLM (GPT-4o) to perform context attribution and then rates citations through two methods: `Iterative Chain of Edits` (reasoning through edits before rating) and `Edit Distance` (a regression model that weighs different edit types).
5.  **Experimental Validation:** The `CiteEval-Auto` metric was validated by measuring its correlation with human judgments on the `CiteBench` test set and comparing it to other metrics. It was then used to benchmark the citation performance of various LLMs.

## Key Findings

1.  **Superior Human Correlation:** `CiteEval-Auto` substantially outperforms existing state-of-the-art metrics (AutoAIS, AttriScore, LQAC) in correlating with human judgments on citation quality at both the statement and response levels.
2.  **Context Attribution is Crucial:** Explicitly identifying and excluding statements not attributable to the retrieval context (e.g., paraphrasing the query, performing reasoning, or stating parametric facts) is critical for accurate citation evaluation. Removing this step significantly degrades metric performance.
3.  **Fine-Grained Editing Improves Evaluation:** The process of generating explicit edit actions (e.g., "delete misleading citation") before rating leads to better performance than directly rating or using a simple chain-of-thought prompt.
4.  **Missing Citations are a Key Failure Mode:** The study found a strong positive correlation between response length and the ratio of missing citations, indicating that models struggle to maintain citation completeness in longer, more complex answers.
5.  **Better Retrieval Recall Improves Citations:** Using re-ranked retrieval contexts with higher recall of relevant passages generally leads to improved citation quality. However, simply improving precision by filtering out irrelevant documents did not show a benefit.
6.  **Iterative Improvement is Possible:** Using the edit actions generated by `CiteEval-Auto`, the framework can be used iteratively to automatically critique and improve the citation quality of a given response.
7.  **Llama-3-70b Outperforms GPT-4o in `Full` Scenario:** While GPT-4o was the top performer in the `Cited` scenario (evaluating only sentences with citations), Llama-3-70b performed best in the `Full` scenario (evaluating all citable sentences), largely because it produced more concise responses with fewer uncited statements.

## Main Contribution

This study fundamentally revises the approach to citation evaluation in RAG systems. It moves the field beyond simplistic, NLI-based "supportiveness" checks and introduces a more holistic, principle-driven framework (`CiteEval`) that is more aligned with human-centric qualities like trust, verifiability, and source credibility. The main contributions are the `CiteBench` benchmark and the `CiteEval-Auto` metric, which provide researchers and developers with a more reliable and nuanced tool for measuring and improving the source attribution capabilities of LLMs.

## Limitations

1.  **Limited Context Types:** The context attribution focuses on typical RAG contexts and could be expanded to include others, such as user demographic information for personalization.
2.  **Extrinsic Evaluation Needed:** The work focuses on intrinsic evaluation (correlation with human judgments). Future user studies are needed to measure the extrinsic, real-world impact on user trust and task success.
3.  **Sentence-Level Granularity:** The evaluation is performed at the sentence level, but a finer-grained (chunk-level) analysis could be beneficial.
4.  **Dependence on Large LLMs:** The `CiteEval-Auto` metric relies on powerful proprietary models like GPT-4o, which can be costly and slow. Distillation to smaller models is a potential area for future work.

## Keywords

- Citation Evaluation
- Retrieval-Augmented Generation (RAG)
- Source Attribution
- Factual Consistency
- Automated Metrics

## Relevance Assessment

- **Relevance:** High
- **Reasoning:** This paper is highly relevant to the project. The project's core is to build a "verifier" module to assess the factual consistency and hallucination of a RAG system's output. `CiteEval` provides a state-of-the-art, principle-driven methodology for exactly this kind of evaluation. The concepts of context attribution (identifying what *should* be cited) and fine-grained quality criteria (going beyond simple NLI) directly inform the design of our verifier. The `CiteBench` dataset and `CiteEval-Auto` metric could serve as an excellent baseline or evaluation framework for our own verifier module.
