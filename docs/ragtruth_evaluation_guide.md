# RAGTruth Evaluation Harness - Usage Guide

## Overview

The RAGTruth evaluation harness assesses the hallucination detection system's performance on the RAGTruth benchmark dataset. It runs the full RAG pipeline, verifies generated claims, and compares detection results against gold-standard hallucination annotations.

## Quick Start

### 1. Prerequisites

Ensure you have:
- RAGTruth dataset in `benchmark/RAGTruth/dataset/`
- FAISS index built (`data/indexes/`)
- Verification components enabled in `config.yaml`

### 2. Basic Usage

```bash
# Test with 10 samples
python scripts/demo_ragtruth_eval.py --max-samples 10

# Full test set evaluation
python scripts/demo_ragtruth_eval.py --split test

# Full evaluation with results export
python scripts/demo_ragtruth_eval.py --split test --save-results

# Resume with strict compatibility checks (default)
python scripts/demo_ragtruth_eval.py --split test --save-results --resume

# Resume but auto-start fresh if selection changed
python scripts/demo_ragtruth_eval.py --split test --save-results --resume --resume-policy fresh-on-mismatch
```

### 3. Programmatic Usage

```python
from src.utils.config import Config
from src.pipelines.baseline_rag import BaselineRAGPipeline
from src.verification.verifier_hub import VerifierHub
from src.verification.rule_based_aggregator import RuleBasedAggregator
from src.evaluation.ragtruth_evaluator import RAGTruthEvaluator

# Initialize components
config = Config('config.yaml')
pipeline = BaselineRAGPipeline.from_config(config)
verifier = VerifierHub(config, pipeline.generator)
aggregator = RuleBasedAggregator(config)

# Create evaluator
evaluator = RAGTruthEvaluator(config, pipeline, verifier, aggregator)

# Run evaluation
metrics = evaluator.run_evaluation(
    split='test',
    max_samples=50,
    batch_size=10,
    save_results=True
)

# Access metrics
print(f"Detection F1: {metrics['overall']['f1']:.3f}")
print(f"Precision: {metrics['overall']['precision']:.3f}")
print(f"Recall: {metrics['overall']['recall']:.3f}")
```

## Configuration

### config.yaml Settings

```yaml
evaluation:
  benchmarks:
    ragtruth:
      dataset_path: "benchmark/RAGTruth/dataset"
      output_dir: "outputs/ragtruth_eval/"
      batch_size: 10
      ragtruth_eval_mode: "ragtruth_eval"  # ragtruth_eval | normal
      teacher_forced_intrinsic: true
      low_confidence_ratio_threshold: 0.6
      low_coverage_ratio_threshold: 0.3
```

  Summary-policy note:

  - In `ragtruth_eval` mode, Summary samples are evaluated with strict sentence-index evidence.
  - If the sentence retriever/index is unavailable, evaluation fails fast instead of falling back to full-document gold context.

Disable multi-question splitting (recommended for RAGTruth evaluation stability):

```yaml
processing:
  query_split:
    enabled: false
```

### Verification Settings

Ensure verification is enabled:

```yaml
verification:
  enabled: true
  modules:
    intrinsic: true
    grounded: true
    nli: true
    self_agreement: true
```

### Independent Module Evaluation (Verifier Signals)

Run all key variants in one command:

```bash
python scripts/evaluate_verifier_signals.py \
  --split test \
  --strategy validation \
  --variants baseline verifier_intrinsic_only verifier_grounded_only verifier_nli_only verifier_self_agreement_only
```

## Baseline Training & Serving

The RAGTruth baseline involves training a hallucination detector and evaluating it using a specific prompt format. An automation runner is provided in `scripts/run_ragtruth_baseline.py`.

### 1. Local/Hybrid Workflow

Use the unified runner script from the project root:

```powershell
# Step 1: Prepare train/dev/test datasets
python scripts/run_ragtruth_baseline.py prepare

# Step 2: Train the baseline detector
# Use --profile single-gpu for local setups or --profile exact for FSDP multi-GPU
python scripts/run_ragtruth_baseline.py train --profile single-gpu --model-name baseline

# Step 3: Serve the model (Docker required for TGI)
# This command prints the docker run command to execute in your shell
python scripts/run_ragtruth_baseline.py serve-cmd --model-subdir baseline --port 8300

# Step 4: Run evaluation against the TGI endpoint
python scripts/run_ragtruth_baseline.py evaluate --model-name baseline --port 8300
```

### 2. Google Colab Workflow

For environments without Docker or high-VRAM local GPUs, use the specialized notebook:

- Path: [colab/notebooks/colab_ragtruth_baseline.ipynb](colab/notebooks/colab_ragtruth_baseline.ipynb)
- **Features**: Automates dataset prep, performs local model inference (transformers), and calculates RAGTruth case-level metrics (Precision/Recall/F1).

### 3. CLI Options Reference

| Subcommand | Purpose | Key Flags |
| --- | --- | --- |
| `prepare` | Splits `response.jsonl` into `train/dev/test` | `--baseline-dir` |
| `train` | Fine-tunes Llama-2-13B (or other) | `--profile`, `--model-path`, `--report-to-wandb` |
| `serve-cmd` | Generates TGI Docker command | `--gpu-device`, `--port` |
| `evaluate` | Hits endpoint and computes metrics | `--tokenizer`, `--output-file` |
| `all` | Runs prepare + train + evaluate sequence | `--run-evaluate` |

Quick smoke test:

```bash
python scripts/evaluate_verifier_signals.py --max-samples 20
```

Artifacts are saved under `outputs/verifier_eval/<timestamp>/`:
- `summary.json`: machine-readable aggregate metrics by variant
- `summary.md`: human-readable comparison table with deltas vs baseline

## Understanding the Metrics

### Overall Metrics

- **Accuracy**: Percentage of samples where hallucination was correctly detected/not detected
- **Precision**: Of samples where we detected hallucinations, how many actually had them
- **Recall**: Of samples with gold hallucinations, how many did we detect
- **F1 Score**: Harmonic mean of precision and recall (balanced metric)

### Confusion Matrix

- **True Positives (TP)**: Correctly detected hallucinations
- **True Negatives (TN)**: Correctly identified clean responses
- **False Positives (FP)**: Flagged clean responses as hallucinations (false alarms)
- **False Negatives (FN)**: Missed actual hallucinations

### Statistics

- **Total Samples**: Number of samples evaluated
- **Gold Hallucinations**: Samples with annotated hallucinations
- **Detected Hallucinations**: Samples where system flagged hallucinations
- **Correct Detections**: True positives
- **Missed Hallucinations**: False negatives
- **False Alarms**: False positives

## Output Format

### Console Output

```
======================================================================
RAGTruth Evaluation Summary
======================================================================

📊 Overall Metrics:
  Accuracy:  0.850
  Precision: 0.820
  Recall:    0.780
  F1 Score:  0.800

📈 Statistics:
  Total Samples:          100
  Gold Hallucinations:    50
  Detected Hallucinations: 48
  Correct Detections:     39
  Missed Hallucinations:  11
  False Alarms:           9

🎯 Confusion Matrix:
  True Negatives:  41
  False Positives: 9
  False Negatives: 11
  True Positives:  39
```

### JSON Export

When using `--save-results`, outputs detailed JSON file:

```json
{
  "metrics": {
    "overall": {
      "accuracy": 0.85,
      "precision": 0.82,
      "recall": 0.78,
      "f1": 0.80,
      "num_samples": 100
    },
    "per_task": {
      "QA": {
        "accuracy": 0.88,
        "precision": 0.85,
        "recall": 0.82,
        "f1": 0.83,
        "num_samples": 40,
        "confusion_matrix": {
          "true_negatives": 12,
          "false_positives": 2,
          "false_negatives": 4,
          "true_positives": 22
        },
        "statistics": {
          "total_samples": 40,
          "gold_hallucinations": 26,
          "detected_hallucinations": 24,
          "correct_detections": 22,
          "missed_hallucinations": 4,
          "false_alarms": 2,
          "total_claims": 180,
          "detected_claim_hallucinations": 30,
          "detected_low_confidence_claims": 12,
          "avg_claims_per_sample": 4.5,
          "avg_claim_hallucinations_per_sample": 0.75
        }
      }
    },
    "confusion_matrix": {
      "true_negatives": 41,
      "false_positives": 9,
      "false_negatives": 11,
      "true_positives": 39
    },
    "statistics": {
      "total_samples": 100,
      "gold_hallucinations": 50,
      "detected_hallucinations": 48,
      "correct_detections": 39,
      "missed_hallucinations": 11,
      "false_alarms": 9
    }
  },
  "sample_results": [
    {
      "sample_id": "resp_1",
      "task_type": "QA",
      "task_id": "src_1",
      "question": "What is the capital of France?",
      "generated_response": "The capital is Paris.",
      "num_claims": 1,
      "predictions": ["Supported"],
      "gold_has_hallucination": false,
      "detected_hallucination": false,
      "claim_results": [...]
    },
    ...
  ],
  "metadata": {
    "evaluator": "RAGTruthEvaluator",
    "num_samples": 100,
    "benchmark": "RAGTruth",
    "ragtruth_eval_mode": "ragtruth_eval",
    "unique_tasks": ["Data2txt", "QA", "Summary"],
    "selection_fingerprint": {
      "split": "test",
      "max_samples": 100,
      "samples_per_task": null,
      "ragtruth_eval_mode": "ragtruth_eval",
      "dataset_path": ".../benchmark/RAGTruth/dataset"
    }
  }
}
```

## Resume Troubleshooting

- Error `Resume mismatch: sample_id 'X' not found in current dataset selection` means your existing output file was created with a different selection configuration.
- Selection compatibility now uses `metadata.selection_fingerprint` (split, max_samples, samples_per_task, ragtruth_eval_mode, dataset_path).
- If you intentionally changed sampling settings, either:
  - run without `--resume`, or
  - use `--resume --resume-policy fresh-on-mismatch` to start a clean run automatically.

## Evaluation Pipeline

### Step-by-Step Process

1. **Dataset Loading**
   - Loads `source_info.jsonl` (questions + contexts)
   - Loads `response.jsonl` (gold hallucination annotations)
   - Filters by split (train/test) and quality
   - Joins source info with responses by `source_id`

2. **Sample Evaluation** (for each sample)
  - Extract question and contexts from source info
  - If `ragtruth_eval_mode: ragtruth_eval`: use dataset response + gold contexts
  - For Summary in `ragtruth_eval`, require sentence-index evidence and do not fallback to full-context verification
  - If `teacher_forced_intrinsic: true` in `ragtruth_eval`, score the gold response with teacher forcing to compute intrinsic uncertainty (without replacing response text)
  - If `ragtruth_eval_mode: normal`: run RAG pipeline to generate responses
  - Verify each claim using VerifierHub (all detectors)
  - Aggregate signals into claim decisions
  - Escalate sample-level hallucination when contradiction exists, or when low-confidence ratio is high and weak-evidence ratio corroborates it
  - Check if claims overlap with gold hallucination spans
  - Record predictions and ground truth

3. **Metrics Computation**
   - Convert per-sample results to binary labels
   - Calculate accuracy, precision, recall, F1
   - Generate confusion matrix
   - Compute detailed statistics

4. **Results Export**
   - Save metrics and detailed predictions to JSON
   - Print formatted summary to console

## Task Types

RAGTruth includes three task types:

### QA (Question Answering)
- Question + passages → answer
- Example: "What is the capital of France?" with Wikipedia passages

### Summary
- Document → summary
- Example: Summarize a CNN news article

### Data2txt
- Structured data → natural language description
- Example: Generate description from Yelp business data

The evaluator handles all three types automatically.

## Performance Considerations

### Memory Usage

- **Batch Processing**: Use `batch_size` parameter to control memory
- **Max Samples**: Test with small subsets before full evaluation
- **Progress Tracking**: tqdm shows real-time progress

### Timing

Approximate times (with RTX 3070Ti):
- **Per sample**: ~10-30 seconds (depends on verification enabled)
- **10 samples**: ~2-5 minutes
- **100 samples**: ~20-50 minutes
- **Full test set (2,700)**: ~15-45 hours

### Recommendations

1. **Development**: Use `max_samples=10-20` for quick validation
2. **Validation**: Use `max_samples=50-100` for representative results
3. **Production**: Run full test set overnight or in batches

## Interpreting Results

### High Precision, Low Recall

System is **conservative** - flags few hallucinations but most are correct.
- Fewer false alarms
- Misses some actual hallucinations
- **Action**: Lower thresholds in aggregator to increase sensitivity

### Low Precision, High Recall

System is **aggressive** - flags many hallucinations but includes false alarms.
- Catches most hallucinations
- Many false positives
- **Action**: Raise thresholds in aggregator to reduce false alarms

### Balanced F1 Score

System has good **trade-off** between precision and recall.
- Optimal for most use cases
- **Target**: F1 > 0.75 for production systems

## Troubleshooting

### Dataset Not Found

```
FileNotFoundError: RAGTruth benchmark directory not found
```

**Solution**: Download RAGTruth to `benchmark/RAGTruth/dataset/`

### Verification Disabled

```
WARNING: VerifierHub initialized but verification is disabled
```

**Solution**: Enable verification in `config.yaml`:
```yaml
verification:
  enabled: true
```

### Out of Memory

```
RuntimeError: CUDA out of memory
```

**Solution**: 
- Reduce `batch_size` parameter
- Use smaller `max_samples`
- Close other GPU applications

### Slow Evaluation

**Solutions**:
- Disable unnecessary detectors temporarily
- Reduce `top_k` in retrieval
- Use smaller model for verification
- Process in smaller batches

## Advanced Usage

### Custom Metrics

Add custom metric computation in `_compute_metrics()`:

```python
# Custom metric: detection rate by task type
def compute_by_task_type(self, results):
    by_type = defaultdict(list)
    for r in results:
        task_type = r['task_type']
        by_type[task_type].append(r['detected_hallucination'])
    return {t: np.mean(v) for t, v in by_type.items()}
```

### Filtering Samples

Filter by task type or other criteria:

```python
# Only evaluate QA samples
samples = evaluator._load_dataset(split='test')
qa_samples = [s for s in samples if s['task_type'] == 'QA']

# Run evaluation on filtered samples
for sample in qa_samples:
    result = evaluator._evaluate_sample(sample)
```

### Custom Aggregation

Override aggregator behavior:

```python
# Use custom decision thresholds
config.verification.aggregator.contradiction_threshold = 0.6
config.verification.aggregator.entailment_threshold = 0.8

aggregator = RuleBasedAggregator(config)
evaluator = RAGTruthEvaluator(config, pipeline, verifier, aggregator)
```

## References

- **RAGTruth Paper**: https://arxiv.org/abs/2401.00396
- **RAGTruth GitHub**: https://github.com/CodingLL/RAGTruth
- **Project Documentation**: `docs/month4_verifier_part2.md`
- **Configuration Guide**: `USAGE.md`
