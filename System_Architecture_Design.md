# System Architecture: Trainless Verifier & Simplified UI

This document outlines a redesigned architecture focused on a **trainless, multi-signal verifier** for hallucination detection. The design prioritizes modular, zero-shot techniques and defers complex training and UI elements.

## High-Level Pipeline Flowchart

This diagram illustrates the updated data flow, emphasizing the parallel, trainless signals within the Verifier and the simplified mitigation and UI components.

```mermaid
graph TD
    A[User Query] --> B;
    subgraph B[Baseline RAG Module]
        direction LR
        B1{Retriever} --> B2{Generator};
    end
    B --> C["Draft Response + Claim-Evidence Pairs"];
    C --> D;
    subgraph D["Verifier Module (Trainless Signals)"]
        direction TB
        D1["Intrinsic Uncertainty (Entropy)"]
        D2["Self-Agreement (Consistency)"]
        D3["Retrieval Heuristics (Coverage)"]
        D4["Zero-Shot NLI (Contradiction)"]
        D_Aggregator{"Rule-Based Aggregator"}
        D1 --> D_Aggregator;
        D2 --> D_Aggregator;
        D3 --> D_Aggregator;
        D4 --> D_Aggregator;
    end
    D --> E["Verified Claims with Confidence Breakdown"];
    E --> F["Flagging & Suppression Module"];
    E --> G["Minimal Confidence UI (Table/Badges)"];
    F --> H[Final Verified Response];
    G --> I((Final Output));
    H --> I;

    style B fill:#f9f,stroke:#333,stroke-width:2px
    style D fill:#ccf,stroke:#333,stroke-width:2px
    style F fill:#fcf,stroke:#333,stroke-width:2px
    style G fill:#ffc,stroke:#333,stroke-width:2px
    style I fill:#bdf,stroke:#333,stroke-width:4px
```

---

## Module-by-Module Design

### 1. Baseline RAG Module
*(This module's core function remains the same, but it now passes token-level metadata to the Verifier.)*

-   **Inputs:**
    -   `user_query`: (string) The input prompt from the user.
-   **Process:**
    1.  **Retrieve:** The retriever fetches relevant documents.
    2.  **Generate:** The generator LLM produces a draft response, while capturing token-level metadata (e.g., logits for entropy calculation).
    3.  **Decompose & Pair:** The draft is decomposed into atomic claims, creating direct `(claim, evidence)` pairs.
-   **Outputs:**
    -   `draft_response`: (string) The full, unverified draft response.
    -   `claim_evidence_pairs`: (List[dict]) A list where each dictionary contains the `claim`, the `evidence` document, and `generator_metadata`.

### 2. Verifier Module (Trainless Signal Hub)

This module is redesigned as a hub for calculating multiple, parallel, trainless confidence signals.

-   **Inputs:**
    -   `claim_evidence_pairs`: (List[dict]) The output from the Baseline RAG Module.

-   **Process & Sub-components:**
    1.  For each `(claim, evidence)` pair, the following sub-components run in parallel:
        -   **Intrinsic Uncertainty:** Analyzes `generator_metadata`.
            -   **Output:** `entropy_score` (e.g., length-normalized negative log-likelihood).
        -   **Self-Agreement (Conditional):** If enabled, generates `k` response samples.
            -   **Output:** `consistency_score` (e.g., variance or disagreement across samples).
        -   **Retrieval-Grounded Heuristics:**
            -   **Process:** Calculates `evidence_coverage` (percentage of claim entities in evidence) and `citation_integrity` (token overlap for cited spans).
            -   **Output:** A dictionary of heuristic scores.
        -   **Zero-Shot NLI:** Uses an off-the-shelf NLI model.
            -   **Process:** Labels each `(claim, evidence_sentence)` pair as Entail/Contradict/Neutral.
            -   **Output:** Aggregated NLI scores (e.g., `max_contradiction_prob`, `entailment_ratio`).
    2.  **Rule-Based Aggregator:** Gathers all signals into a structured breakdown using explicit rules. No trainable fusion logic is used at this stage.

-   **Outputs:**
    -   `verified_claims`: (List[dict]) A list where each dictionary contains:
        -   `claim`: (string) The original atomic claim.
        -   `evidence`: (dict) The associated evidence.
        -   `confidence_breakdown`: (dict) A structured dictionary containing all raw signals (e.g., `entropy_score`, `nli_results`, `coverage_score`).
        -   `final_verdict`: (string) A final verdict (e.g., "Supported", "Contradictory", "Low Confidence") derived from the rule-based aggregator.

### 3. Flagging & Suppression Module (Simplified Mitigation)

This module applies simple, rule-based actions based on the final verdict from the verifier.

-   **Inputs:**
    -   `draft_response`: (string) The original draft.
    -   `verified_claims`: (List[dict]) The output from the Verifier Module.

-   **Process:**
    1.  Iterates through the `verified_claims` and assembles the final response.
    2.  Applies simple rules based on the `final_verdict`:
        -   **If "Contradictory"**: The claim is suppressed or replaced with a warning (e.g., "[Warning: The following claim contradicts the source]"). No complex rewrite loop is implemented.
        -   **If "Low Confidence"**: The claim is flagged with a visual indicator.
        -   **If "Supported"**: The claim is included as is, with its citation.

-   **Outputs:**
    -   `final_response`: (string) The final text with flags and (optional) suppressions.

### 4. Minimal Confidence UI (Table/Badges)

This module provides a simple, transparent view of the confidence signals.

-   **Inputs:**
    -   `verified_claims`: (List[dict]) The output from the Verifier Module.

-   **Process:**
    1.  Displays the final response with clear visual cues (e.g., colored highlights or badges) for each claim based on its `final_verdict`.
    2.  On hover or click, a simple table or list appears, showing the raw values from the `confidence_breakdown` for that claim (e.g., "Entropy: 0.85", "NLI Contradiction: 0.92", "Coverage: 0.65").
    3.  No complex visualizations like radial or donut charts are used at this stage.

-   **Outputs:**
    -   An interactive user interface that allows users to inspect the raw, multi-dimensional evidence behind each claim's confidence level.

---
*Interface for Future Training:* The `Verifier Module` is designed to be extensible. Each trainless signal component can be replaced by a trained model in the future. The `Rule-Based Aggregator` can be swapped with a trainable `Ensemble Fusion Logic` that learns to weigh the signals optimally, without changing the overall architecture.
