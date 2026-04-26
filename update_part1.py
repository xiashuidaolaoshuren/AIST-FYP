with open('docs/evaluation_result/Comprehensive_Final_Evaluation_Report.md', 'r') as f:
    content = f.read()

part1_old = """2. **Verification-Aware Mitigation (The Filtering Actuator)**: To isolate the impact of different verifier signals, claims flagged as `Contradictory` are deterministically filtered. **Why use a filter?** If the verifier labels a claim as hallucinated but does not physically remove it, the response text submitted to CiteEval remains identical to the baseline. Filtering is surgical and deterministic, isolating the contribution of each individual verifier module (e.g., NLI vs. Grounded) without the confounding variables introduced by complete response re-generation."""

part1_new = """2. **Verification-Aware Mitigation (The Filtering Actuator)**: To isolate the impact of different verifier signals, claims flagged as `Contradictory` are deterministically filtered. 
   > **Why Verifier Variants Must Include a Filter in CiteBench:** 
   > This is a fundamental design difference from the RAGTruth pipeline. In RAGTruth, the evaluator compares the verifier's *label* against ground truth annotations without altering the text. However, CiteEval scores the *final text string itself*. If a verifier merely labels a claim as hallucinated but does not physically remove it, the response text submitted to CiteEval remains identical to the unverified baseline, resulting in zero measurable difference. 
   > 
   > **Why filter and not re-rank or re-prompt during Verifier Ablation?** Filtering is surgical and deterministic. Re-ranking or re-prompting introduces complete generation variance (the LLM creates entirely new text), which severely confounds which underlying verifier signal (e.g., NLI vs. Intrinsic) actually caused the improvement. Thus, pairing each verifier signal with a constant rigid filter isolates the contribution of the detection mechanism perfectly."""

content = content.replace(part1_old, part1_new)

header32_old = "### 3.2 Overall CiteEval Mitigation Performance"
header32_new = "### 3.2 Verifier Signals Performance (Signal Ablation)"

content = content.replace(header32_old, header32_new)

header33_old = "### 3.3 Filtering Statistics and Variant Ablation"
header33_new = "### 3.3 Verifier Filtering Statistics"

content = content.replace(header33_old, header33_new)

header34_old = "### 3.4 Detailed Module Interpretations"
header34_new = "### 3.4 Detailed Verifier Module Interpretations"

content = content.replace(header34_old, header34_new)

with open('docs/evaluation_result/Comprehensive_Final_Evaluation_Report.md', 'w') as f:
    f.write(content)

print("Part 1 and Headers updated.")
