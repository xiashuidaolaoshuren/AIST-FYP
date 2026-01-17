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
```

### Verification Settings

Ensure verification is enabled:

```yaml
verification:
  enabled: true
  intrinsic_uncertainty:
    enabled: true
  retrieval_grounded:
    enabled: true
  nli:
    enabled: true
  self_agreement:
    enabled: true
```

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
    "benchmark": "RAGTruth"
  }
}
```

## Evaluation Pipeline

### Step-by-Step Process

1. **Dataset Loading**
   - Loads `source_info.jsonl` (questions + contexts)
   - Loads `response.jsonl` (gold hallucination annotations)
   - Filters by split (train/test) and quality
   - Joins source info with responses by `source_id`

2. **Sample Evaluation** (for each sample)
   - Extract question and contexts from source info
   - Run RAG pipeline: retrieve → generate → extract claims
   - Verify each claim using VerifierHub (all detectors)
   - Aggregate signals into claim decisions
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

## Integration with Ragas

Compare RAGTruth results with Ragas metrics:

```python
from src.evaluation.ragas_evaluator import RagasEvaluator

# Run RAGTruth evaluation
ragtruth_metrics = ragtruth_evaluator.run_evaluation(...)

# Run Ragas evaluation on same samples
ragas_evaluator = RagasEvaluator(config)
rag_results = [...]  # From RAG pipeline
ragas_df = ragas_evaluator.evaluate_rag_outputs(rag_results)

# Compare metrics
print(f"RAGTruth F1: {ragtruth_metrics['overall']['f1']:.3f}")
print(f"Ragas Faithfulness: {ragas_df['faithfulness'].mean():.3f}")
```

## References

- **RAGTruth Paper**: https://arxiv.org/abs/2401.00396
- **RAGTruth GitHub**: https://github.com/CodingLL/RAGTruth
- **Project Documentation**: `docs/month4_verifier_part2.md`
- **Configuration Guide**: `USAGE.md`
