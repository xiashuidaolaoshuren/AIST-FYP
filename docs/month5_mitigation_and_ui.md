# Month 5 Implementation: Multi-Signal Aggregation, Mitigation, and UI

This document details the components implemented in Month 5 of the AIST-FYP project. The focus of this phase was to integrate individual verification signals into a coherent decision-making system, provide active mitigation for hallucinations, and visualize the results through a user-friendly interface.

## 1. Overview

The Month 5 implementation introduces the **Full Hallucination Detection Pipeline**, which connects the following components:

1.  **RuleBasedAggregator**: A logic layer that combines signals (Entropy, NLI, Coverage, Self-Consistency) to classify claims.
2.  **Mitigation Module**: Active strategies to improve output quality (Evidence Re-Ranking and Claim Filtering).
3.  **CitationFormatter**: Automated injection of evidence citations into generated answers.
4.  **ConfidenceUI**: A generic web interface (Gradio) for visualizing claim verification results.

![System Architecture](https://via.placeholder.com/800x400?text=Month+5+System+Architecture) 
*(Note: Placeholder for architecture diagram)*

---

## 2. Rule-Based Aggregator

The `RuleBasedAggregator` is the core decision engine. It normalizes heterogeneous signals onto a 0-1 confidence scale and applies a hierarchical rule set to determine the final verdict for each claim.

### 2.1 Decision Hierarchy

The aggregator evaluates claims in the following order of precedence:

| Priority | Detector | Condition | Verdict | Meaning |
| :--- | :--- | :--- | :--- | :--- |
| 1 | **NLI** | `contradiction_score > threshold` | **Contradictory** | The claim directly contradicts retrieved evidence. |
| 2 | **Coverage** | `coverage_score < threshold` | **Low Confidence** | The claim relies on entities/numbers not found in evidence. |
| 3 | **Self-Check** | `consistency_score < threshold` | **Low Confidence** | The model's stochastic samples disagree with the claim. |
| 4 | **Uncertainty** | `entropy_confidence < threshold` | **Low Confidence** | The model's token-level probability is too low (high uncertainty). |
| 5 | **Default** | *(None of above)* | **Supported** | The claim is consistent, grounded, and confident. |

### 2.2 Signal Normalization

Raw signals are transformed to a uniform [0, 1] confidence scale:

*   **Entropy**: Sigmoid transformation. High entropy (uncertainty) $\to$ Low confidence.
*   **Consistency**: Exponential decay. High variance $\to$ Low consistency.
*   **Coverage**: Weighted sum of Entity (0.4), Number (0.3), and Token (0.3) overlap.

### 2.3 Configuration (`config.yaml`)

```yaml
aggregator:
  contradiction_threshold: 0.5  # NLI Contradiction > 0.5 -> Contradictory
  low_coverage_threshold: 0.3   # Coverage < 0.3 -> Low Confidence
  consistency_confidence_threshold: 0.4
  entropy_confidence_threshold: 0.4
```

---

## 3. Mitigation Strategies

Two active mitigation strategies were implemented to address detected hallucinations.

### 3.1 Evidence Re-Ranker

Re-orders retrieved evidence chunks before generation or verification to prioritize high-quality evidence.

*   **Logic**: Combines semantic similarity (Retrieval) with verification feedback (Verification).
*   **Formula**: 
    $$ Score_{final} = \alpha \times Score_{retrieval} + \beta \times Score_{verification} $$
    $$ Score_{verification} = \frac{Coverage_{entities} + NLI_{entailment}}{2} $$
*   **Default Weights**: $\alpha=0.6$, $\beta=0.4$

### 3.2 Claim Filter

Removes claims flagged as **Contradictory** from the final answer to prevent misinformation.

*   **Mechanism**: Replaces the character span of the contradictory claim with a placeholder.
*   **Safety**: Processes claims in reverse order (last to first) to prevent index shifting issues.
*   **Placeholder**: `[Claim removed: Contradictory]` (Configurable).

### 3.3 Configuration (`config.yaml`)

```yaml
mitigation:
  enabled: true
  reranker:
    enabled: true
    alpha: 0.6  # Retrieval importance
    beta: 0.4   # Verification importance
  filter:
    enabled: true
    placeholder: "[Claim removed: Contradictory]"
```

---

## 4. Citation Formatter

The `CitationFormatter` adds transparency by linking claims to their supporting evidence.

*   **Format**: Adds `[i]` markers to claims, corresponding to the rank of the supporting evidence chunk.
*   **Limit**: Configurable max citations per claim (Default: 3).
*   **Output**: Returns decorated text and a structured citation map for evaluation (CiteEval/CiteBench compatible).

---

## 5. Confidence UI (Gradio Demo)

The `ConfidenceUI` provides a visual dashboard for the entire pipeline.

### 5.1 Features

*   **Interactive Chat**: Query the RAG system in real-time.
*   **Color-Coded Analysis**:
    *   <span style="color:green">**Green**</span>: Supported (High Confidence)
    *   <span style="color:red">**Red**</span>: Contradictory (Potential Hallucination)
    *   <span style="color:gold">**Yellow**</span>: Low Confidence (Uncertain/Ungrounded)
*   **Evidence Inspector**: Expander to view retrieved docs and individual signal scores.
*   **Metrics Panel**: Displays raw scores (Entropy, NLI, etc.) for debugging.
*   **Mitigation Toggle**: Toggle re-ranking and filtering on/off via config (requires restart).

---

## 6. Usage Guide

### 6.1 Running the Full Pipeline Demo

To launch the integrated system with the Gradio UI:

```bash
python scripts/demo_full_pipeline.py
```

*   **URL**: `http://localhost:7860` (or `http://0.0.0.0:7860`)
*   **Note**: Ensure `config.yaml` has `verification.enabled: true`.

### 6.2 Example Output

**Query**: "What is the capital of Australia?"

**Answer**: 
> Canberra is the capital of Australia <span style="color:green">[Supported]</span>. It was selected in 1908. Sydney is the large city <span style="color:green">[Supported]</span>.

**Mitigated Output** (if "Sydney is capital" was generated):
> Canberra is the capital of Australia. [Claim removed: Contradictory].

---

## 7. Integration Status

| Component | Status | Source Location | Tests |
| :--- | :--- | :--- | :--- |
| RuleBasedAggregator | ✅ Complete | `src/verification/rule_based_aggregator.py` | `tests/unit/test_rule_based_aggregator.py` |
| CitationFormatter | ✅ Complete | `src/verification/citation_formatter.py` | `tests/unit/test_citation_formatter.py` |
| EvidenceReRanker | ✅ Complete | `src/mitigation/re_ranker.py` | `tests/unit/test_re_ranker.py` |
| ClaimFilter | ✅ Complete | `src/mitigation/claim_filter.py` | `tests/unit/test_claim_filter.py` |
| ConfidenceUI | ✅ Complete | `src/ui/confidence_ui.py` | N/A (Integration) |
