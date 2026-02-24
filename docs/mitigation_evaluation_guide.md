# Mitigation Strategy Evaluation Guide

This guide explains how to run **paired baseline-vs-mitigation** evaluation for the pipeline using RAGTruth.

## What this evaluates

The script compares mitigation variants on identical settings/samples:
- `baseline`: mitigation disabled
- `mitigation_all`: rerank + filter + reprompt enabled
- optional ablations: `filter_only`, `rerank_only`, `reprompt_only`

The comparison reports:
- Accuracy / Precision / Recall / F1
- Confusion matrix counts (TP/TN/FP/FN)
- Delta metrics vs baseline

## Command

```powershell
python scripts/evaluate_mitigation_strategy.py --max-samples 30
```

Full matrix (all variants):

```powershell
python scripts/evaluate_mitigation_strategy.py --variants baseline mitigation_all filter_only rerank_only reprompt_only
```

Use full test split with saved outputs in a custom folder:

```powershell
python scripts/evaluate_mitigation_strategy.py --split test --output-dir outputs/mitigation_eval/final_run
```

## Important flags

- `--ragtruth-eval-mode ragtruth_eval|normal`
  - `ragtruth_eval`: use benchmark responses
  - `normal`: use pipeline-generated responses
- `--strategy development|validation|production`: retrieval index strategy
- `--max-samples N`: quick smoke test before full run

## Output artifacts

Each run writes to `outputs/mitigation_eval/<timestamp>/` (or `--output-dir`):

- `configs/config_<variant>.yaml`: generated per-variant config
- `ragtruth/ragtruth_<variant>.json`: raw RAGTruth result per variant
- `summary.json`: machine-readable summary of all variant metrics
- `summary.md`: human-readable comparison table with deltas

## Recommended workflow

1. Run quick smoke test (`--max-samples 10` or `30`).
2. Verify no variant failures and inspect `summary.md`.
3. Run full split with fixed settings.
4. Compare `ΔF1`, `ΔRecall`, and `ΔPrecision` against baseline.
5. Pick best trade-off variant, then tune thresholds in `config.yaml`.
