with open('docs/evaluation_result/Comprehensive_Final_Evaluation_Report.md', 'r') as f:
    text = f.read()

old_opening = """This section comprehensively evaluates the performance of the proposed trainless verifier pipeline against established baselines. The evaluation is focused on two complementary dimensions: the system's ability to accurately detect factual hallucinations (assessed via the RAGTruth benchmark [1]), and its effectiveness in mitigating hallucinations while generating high-quality citations (assessed via the CiteBench framework using CiteEval [2]). The pipeline is compared against **LettuceDetect** [3], a state-of-the-art fine-tuned hallucination detection model."""

new_opening = """This chapter comprehensively evaluates the performance of the proposed trainless verifier and mitigation pipeline against established baselines. Moving beyond traditional evaluation, this research posits that detecting a hallucination is only half the problem; system evaluation must also capture the effectiveness of *correcting* these errors in the final user-facing output. Therefore, the evaluation is structured across three primary analytical dimensions:

1. **Factual Hallucination Detection (RAGTruth):** Assessing the pipeline's raw labeling capability to accurately flag factual errors across various generation tasks.
2. **Verifier Signal Evaluation (CiteBench/CiteEval):** Isolating and evaluating the structural and citation quality when relying on different detection signals (e.g., Intrinsic vs. NLI) paired with a rigid, deterministic filter.
3. **Mitigation Pipeline Evaluation (CiteBench/CiteEval):** Testing different downstream "actuators" (Filtering, Reranking, Reprompting) to determine the most effective strategy for repairing a detected hallucination.

The pipeline is baselined against **LettuceDetect** [3], a state-of-the-art fine-tuned hallucination detection model, to highlight the robust capabilities of our modular, zero-shot architecture."""

text = text.replace(old_opening, new_opening)

old_part1_desc = """The evaluation framework is bifurcated to measure fundamentally different aspects of system performance. RAGTruth evaluates factual accuracy at the claim level, while CiteBench assesses the quality, placement, and semantic correctness of citations."""

new_part1_desc = """The evaluation framework purposefully separates the **detection of hallucinations** from the **mitigation of hallucinations**. RAGTruth evaluates pure factual accuracy and the intrinsic reliability of our verifier's labeling at the isolated claim level. Conversely, CiteBench is employed dynamically to assess how these verifier decisions actually restructure and repair the final text—specifically examining the quality, density, and semantic correctness of citations after applying various mitigation strategies."""

text = text.replace(old_part1_desc, new_part1_desc)

with open('docs/evaluation_result/Comprehensive_Final_Evaluation_Report.md', 'w') as f:
    f.write(text)

print("Opening and Part 1 updated.")
