# Citation Number Fixes and Reference Deduplication

## Scope
- File updated: `presentation/final/LWW2502_Final-report.md`
- Constraint followed: only citation bracket numbers (`[n]`) and duplicated references were modified.

## Reference List Cleanup
- Removed the first duplicated `References` block (`[1]` to `[21]`) that appeared before the final reference list.
- Kept the later complete reference list (`[1]` to `[24]`) as the single canonical list.

## In-text Citation Renumbering
The following in-text citation numbers were updated to match the kept canonical reference list:

- Related Work 2.1
  - `[1] -> [2]` (survey)
  - `[2] -> [3]` (FEVER)
  - `[3] -> [4]` (KILT)

- Related Work 2.2
  - `[4] -> [5]` (Kryscinski)
  - `[5] -> [6]` (QAFactEval)
  - `[6] -> [7]` (SummaC)
  - `[7] -> [8]` (CiteEval)

- Related Work 2.3
  - `[8] -> [9]` (TruthfulQA)
  - `[9] -> [10]` (RAGTruth)

- Related Work 2.4
  - `[10] -> [11]` (SelfCheckGPT)
  - `[11] -> [12]` (Self-RAG)
  - `[12] -> [13]` (CoVe)

- System Design intro sentence
  - `[1] -> [2]` (survey citation)

- LLM-as-a-Judge mention
  - `[9] -> [14]`

- Generation module inspiration
  - `[3] -> [11]` (SelfCheckGPT)

- Retrieval-grounded heuristics inspiration
  - `[4] -> [3]` (FEVER)

- Zero-shot NLI inspiration sentence
  - `SummaC [6] -> [7]`
  - `Self-RAG [7] -> [12]`

- Self-agreement inspiration sentence
  - `[8] -> [16]` (Self-Consistency)

- Re-prompter inspiration sentence
  - `CoVe [12] -> [13]`
  - `Self-RAG [7] -> [12]`

- Pipeline Demonstration paragraph
  - `[15] -> [18]` (LettuceDetect)

- Evaluation bullets and baseline comparison
  - `RAGTruth [1] -> [10]`
  - `CiteEval [2] -> [8]` (both occurrences)
  - `LettuceDetect [3] -> [18]`

- Open-source model configuration
  - `Qwen [17] -> [20]`
  - `all-MiniLM-L6-v2 [18] -> [21]`
  - `DeBERTa [19] -> [22]`
  - `spaCy [120] -> [23]` (fixed invalid citation number)
  - `DeepSeek [21] -> [24]`

- Limitations section
  - `Contextual Retrieval [16] -> [19]`
  - `SummaC [6] -> [7]`

## Notes
- Generic example placeholders in text such as `(e.g. [1], [2])` were intentionally left unchanged because they are example markers, not direct source attribution statements.

## Additional Alignment Update
- Added missing in-text citation for `[15]`3. System Design:
  - System Design opening sentence now cites RAG frameworks as `[15]`.
- Added missing in-text citation for `[17]`：2.2 Automated Factuality Evaluation:
  - Automated factuality evaluation opening sentence now cites broader factuality evaluation context as `[17]`.
- Re-checked number set consistency:
  - All reference entries `[1]` to `[24]` are now cited in body text.
  - The only unmatched token in body extraction is `[0]`, which appears in code examples and is not a bibliography citation.

## Identifier Correction Update
- Corrected reference entry `[2]` to match the actual survey paper record:
  - arXiv ID: `2311.03687 -> 2311.05232`
  - Author lead: `Y. Zhang -> L. Huang`
  - Title expanded to the official arXiv title:
    - `A Survey on Hallucination in Large Language Models: Principles, Taxonomy, Challenges, and Open Questions`
