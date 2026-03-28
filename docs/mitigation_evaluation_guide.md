# Mitigation Strategy Evaluation Guide

This guide explains how to run **paired baseline-vs-mitigation** evaluation for the pipeline using RAGTruth with **HRR/FRR** as primary metrics.

## What this evaluates

The script compares mitigation variants on identical settings/samples:
- `verifier_only`: mitigation disabled baseline
- `mitigation_all`: rerank + filter + reprompt enabled
- optional ablations: `filter_only`, `rerank_only`, `reprompt_only`

The comparison reports:
- **HRR** (Hallucination Reduction Rate)
- **FRR** (False Removal Rate)
- Gold/non-gold claim removal counts
- Reference detection metrics (Accuracy / Precision / Recall / F1)
- Delta metrics vs baseline

## Command

```powershell
python scripts/evaluate_mitigation_ragtruth.py --max-samples 30
```

Full matrix (all variants):

```powershell
python scripts/evaluate_mitigation_ragtruth.py --variants verifier_only mitigation_all filter_only rerank_only reprompt_only
```

Use full test split with saved outputs in a custom folder:

```powershell
python scripts/evaluate_mitigation_ragtruth.py --split test --output-dir outputs/mitigation_hrr_eval/final_run
```

## Important flags

- `--ragtruth-eval-mode ragtruth_eval|normal`
  - `ragtruth_eval`: use benchmark responses
  - `normal`: use pipeline-generated responses
- `--strategy development|validation|production`: retrieval index strategy
- `--max-samples N`: quick smoke test before full run

For verifier signal ablation (Precision/Recall/F1-focused), use:

```powershell
python scripts/evaluate_verifier_signals.py --max-samples 30
```

## Output artifacts

Each run writes to `outputs/mitigation_hrr_eval/<timestamp>/` (or `--output-dir`):

- `configs/config_<variant>.yaml`: generated per-variant config
- `ragtruth/ragtruth_<variant>.json`: raw RAGTruth result per variant
- `summary.json`: machine-readable summary of all variant metrics
- `summary.md`: human-readable comparison table with deltas

## Recommended workflow

1. Run quick smoke test (`--max-samples 10` or `30`).
2. Verify no variant failures and inspect `summary.md`.
3. Run full split with fixed settings.
4. Compare `HRR` (higher is better) and `FRR` (lower is better) against baseline.
5. Pick best trade-off variant, then tune thresholds in `config.yaml`.

## Cross-Evaluation With CiteBench/CiteEval

Use RAGTruth and CiteBench together for mitigation research:

- RAGTruth answers: "Did mitigation reduce hallucinations without over-filtering?" (HRR/FRR)
- CiteEval answers: "Did the final response have high citation quality?" (`statement_rating`, `response_rating`)

Interpretation guide:

- CiteEval up, HRR up, FRR flat/down: strong improvement (better citation quality and better mitigation behavior)
- CiteEval up, HRR flat/down: likely citation formatting/selection gains without meaningful hallucination reduction
- CiteEval flat/down, HRR up but FRR up: mitigation may be over-filtering useful content
- Both down: regression in both faithfulness and citation quality

Recommended reporting table columns (same variant order):

- `statement_rating`, `response_rating` (CiteEval)
- `hrr`, `frr` (RAGTruth mitigation metrics)
- `precision`, `recall`, `f1` (RAGTruth reference detection metrics)
- `filtered_claim_count` (internal mitigation diagnostics)
