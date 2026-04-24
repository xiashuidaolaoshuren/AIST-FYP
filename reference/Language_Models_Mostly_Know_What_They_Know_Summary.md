# Summary of "Language Models (Mostly) Know What They Know" (Kadavath et al., 2022)

## 1. Core Research Question (PICO/T)
- **P (Problem/Population):** Large Language Models (LLMs) frequently hallucinate or confidently generate incorrect information, making it difficult for users to know when to trust their outputs.
- **I (Intervention/Interest):** The study investigates whether LLMs have an internal, well-calibrated sense of their own uncertainty. Specifically, it explores prompting the model to evaluate the probability that its own proposed answer is correct (e.g., asking "Is the following claim: [claim] (A) True or (B) False?" and measuring the probability of the "True" token), a method sometimes referred to as predicting $P(\text{True})$.
- **C (Comparison):** The study compares the calibration (the alignment between the model's predicted probability of correctness and its actual accuracy) across different model sizes (ranging from small to 52B parameters) and across multiple diverse datasets (e.g., TriviaQA, MMLU, TruthfulQA).
- **O (Outcome):** The primary outcome is the quality of the model's calibration, measured using Brier scores and calibration curves. The goal is to see if models assigning an 80% confidence to an answer are actually correct 80% of the time.
- **T (Timeframe/Theory):** The core hypothesis is that as LLMs scale, they inherently develop better internal representations of their own knowledge boundaries, and this "meta-knowledge" can be surfaced simply by asking the model to evaluate its own outputs using multiple-choice probabilities.

## 2. Methodology
The research follows a quantitative experimental design.
1.  **Task Formatting:** The authors format various question-answering and reasoning datasets into multiple-choice formats (often True/False or A/B/C/D).
2.  **Probability Extraction:** For a given question and proposed answer, the model is prompted to select whether the answer is true or false. The authors extract the log-probability assigned to the token corresponding to the "True" option or the correct letter option.
3.  **Self-Evaluation:** They test a setup where the model first generates an answer to a free-form question, and then is prompted in a second step to evaluate the likelihood that its generated answer is correct.
4.  **Calibration Measurement:** They group the model's predicted probabilities into bins and compare them against the actual empirical accuracy within each bin to create calibration curves and compute Brier scores.

## 3. Key Findings
1.  Larger language models are generally well-calibrated out-of-the-box on a variety of multiple-choice and True/False QA tasks, meaning their predicted confidence closely matches their actual likelihood of being right.
2.  Models are capable of evaluating the correctness of their own generated answers (self-evaluation), showing that they "know what they know."
3.  Calibration improves consistently with model scale; larger models are not just more accurate, but also better at knowing when they are likely to be wrong.
4.  The models tend to be slightly overconfident (sycophantic) when evaluating their own free-text generations compared to evaluating external, ground-truth options, demonstrating a bias toward confirming their own outputs.
5.  Calibration is notably worse on certain types of tasks, such as complex mathematical reasoning or tasks that are highly counter-intuitive (like some adversarial questions in TruthfulQA).

## 4. Main Contribution
The core contribution is demonstrating that large language models inherently possess well-calibrated representations of their own uncertainty and knowledge limits. It shows that simple prompting techniques (like asking for $P(\text{True})$) combined with token probability extraction can serve as a highly effective, zero-shot method for uncertainty estimation and hallucination detection without any additional training or external tools.

## 5. Limitations
- The primary limitation for practical application is that extracting $P(\text{True})$ requires white-box or grey-box access to the model's token log-probabilities, a feature often unavailable or heavily restricted in many commercial black-box API services.
- The method's effectiveness relies heavily on specific prompt formatting.
- The noted "sycophancy" bias means the model is more likely to validate a hallucination if it was the one that originally generated it.

## 6. Keywords
- Uncertainty Estimation
- Calibration
- P(True)
- Self-Evaluation
- Hallucination Detection

## 7. Relevance Assessment
- **Relevance:** High
- **Justification:** This paper is highly relevant for the related work section discussing alternative zero-shot verifier signals. Validating claims via $P(\text{True})$ is a prominent approach in the literature for detecting hallucination. However, it requires log-probability access (which breaks black-box compatibility) and suffers from self-confirmation bias, providing strong, evidence-backed reasons for why this project chose other methods (like NLI and Entropy) for its verifier module.