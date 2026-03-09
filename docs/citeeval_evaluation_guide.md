# CiteEval (CiteBench) Evaluation Guide

This guide describes how to evaluate the citation quality and factual consistency of our RAG pipeline using the [CiteEval](https://github.com/amazon-science/CiteEval) framework and the **CiteBench** benchmark.

## 1. Overview

CiteEval focuses on fine-grained citation assessment, evaluating not just whether a claim is supported, but how accurately citations are used within the context.

- **CiteBench**: The multi-domain benchmark dataset.
- **CiteEval-Auto**: A suite of model-based metrics (AutoAIS, etc.) used for evaluation.

## 2. Prerequisites

Ensure you have completed the setup in [docs/evaluation_setup_guide.md](docs/evaluation_setup_guide.md):
- **CiteEval repository with DeepSeek modifications**: Use the modified CiteBench version provided by your project lead. Extract and place in `benchmark/CiteEval/`. (See "Using the Custom CiteBench Version" in the setup guide)
- CiteBench data placed in `benchmark/CiteEval/data/`.
- Required environment variables set in `.env` file at project root (see `.env placement` in the setup guide)
- Environment variables (see `benchmark/CiteEval/README.md`)

### Provider Configuration (OpenAI or DeepSeek)

You can configure the evaluator with a project-root [.env](.env) file. The code auto-loads this file if `python-dotenv` is installed.

**OpenAI (default)**
```
CITEEVAL_PROVIDER=openai
OPENAI_API_KEY=your_openai_key
```

**DeepSeek (OpenAI-compatible) - RECOMMENDED**

This evaluation framework has been modified to support DeepSeek API for cost-effective evaluation. Configure your `.env` file at the project root with:
```
CITEEVAL_PROVIDER=deepseek
DEEPSEEK_API_KEY=your_deepseek_key
DEEPSEEK_BASE_URL=https://api.deepseek.com
```

The `.env` file should be placed in the project root: `AIST-FYP/.env`

## 3. Data Formatting for Evaluation

To run CiteEval on our system, we must format the pipeline's output into the CiteEval **System Evaluation** JSON format.

### Using the Conversion Script

We provide a helper script `scripts/convert_to_citeeval.py` that uses the `CitationFormatter` class to convert pipeline outputs to CiteEval format. **This script should be run using the Main Project Environment (`.venv`).**

**Option 1: Convert existing pipeline output file**
```bash
python scripts/convert_to_citeeval.py \
    --input outputs/full_pipeline_queries_20260201_173435.json \
    --output benchmark/CiteEval/data/system_eval/my_pipeline_results.json
```

**Option 2: Run pipeline and convert in one step**
```bash
python scripts/convert_to_citeeval.py \
    --run-pipeline \
    --queries "What is artificial intelligence?" "What is deep learning?" \
    --output benchmark/CiteEval/data/system_eval/my_pipeline_results.json \
    --strategy validation
```

If multi-question splitting disrupts evaluation, disable it in config:

```yaml
processing:
    query_split:
        enabled: false
```

### Required JSON Structure
Each item in your evaluation file should follow this structure:
```json
{
    "id": "unique_query_id",
    "query": "The user's question",
    "passages": [
        {
            "id": "1",
            "title": "Wikipedia Article Title",
            "text": "Content of the retrieved passage..."
        }
    ],
    "pred": "The generated answer with citations in [1] format."
}
```

**Note:** The `CitationFormatter` class (`src/citation/citation_formatter.py`) automatically handles:
- Citation injection with proper bracket format `[1][2][3]`
- Passage deduplication and ordering
- Punctuation-aware citation placement
- Export to CiteEval-compatible JSON structure

### Manual Conversion (Alternative)
If your output is in the standard pipeline format (e.g., `outputs/full_pipeline_queries_*.json`), you may need to use a conversion script or manually map the `metadata` and `full_response` to the `pred` field.

## 4. Running Evaluation Metrics

Use two separate tracks depending on your goal.

### Track A: Official Metric Evaluation (Meta-Eval)

This track evaluates metric-human correlation using official CiteBench splits with human labels.

- Input splits: `benchmark/CiteEval/data/metric_eval/metric_dev`, `benchmark/CiteEval/data/metric_eval/metric_test`
- Human labels: `citebench.metric_*.human.out`

1. Ensure dependencies are synced via uv:
```powershell
uv sync
```

2. Run CiteEval metric generation (inside CiteEval repo):
```bash
cd benchmark/CiteEval/src
sh run_citeeval.sh
```

3. Run human-correlation scoring:
```bash
cd benchmark/CiteEval/src
sh run_metric_eval.sh
```

### Track B: System Evaluation (Our Pipeline Output)

This track evaluates our own RAG outputs without human annotations.

1. Prepare CiteEval-formatted system output:
```bash
python scripts/convert_to_citeeval.py \
    --input outputs/full_pipeline_queries_20260201_173435.json \
    --output benchmark/CiteEval/data/system_eval/my_pipeline_results.json
```

2. Convert to `.citeeval` format expected by CiteEval system scripts:
```bash
cd benchmark/CiteEval
python src/data/convert_to_citeeval_format.py \
    --system_output_file data/system_eval/my_pipeline_results.json
```

3. Run CiteEval-Auto on the converted file (direct CLI):
```bash
cd benchmark/CiteEval
export PYTHONPATH=$PYTHONPATH:$(pwd):$(pwd)/src
python src/scripts/run_citeeval.py \
    --response_output_file data/system_eval/my_pipeline_results.citeeval \
    --eval_output_dir data/system_eval_outputs/ \
    --modules ca,ce,cr_itercoe,cr_editdist \
    --version citeeval-auto-12272024 \
    --model_name deepseek-chat
```
*(Use `--model_name gpt-4o` with `CITEEVAL_PROVIDER=openai`.)*

4. Summarize system-level results:
```bash
cd benchmark/CiteEval/src
sh run_system_eval.sh
```

### Notes on Data Layout

- Some upstream scripts use `data/citebench/metric_eval/...`, while this project keeps files in `data/metric_eval/...`.
- Local wrappers in `benchmark/CiteEval/src/run_citeeval.sh` and `benchmark/CiteEval/src/run_metric_eval.sh` now auto-detect either layout.

### One-Command Runner (Windows-Friendly)

You can run preflight + evaluation from project root with:

```powershell
python scripts/run_citebench_eval.py --track both --metric-split test
```

When `.env` has `CITEEVAL_PROVIDER=deepseek`, the runner auto-selects `deepseek-chat`.

Quick smoke test with ~10 examples:

```powershell
python scripts/run_citebench_eval.py --track both --metric-split test --max-examples 10
```

Force provider/model explicitly (optional):

```powershell
python scripts/run_citebench_eval.py --track both --provider deepseek --model-name deepseek-chat --max-examples 10
```

Useful variants:

```powershell
# Metric track only
python scripts/run_citebench_eval.py --track metric --metric-split dev

# System track only with your own output file
python scripts/run_citebench_eval.py --track system --system-input benchmark/CiteEval/data/system_eval/my_pipeline_results.json

# Print commands without executing
python scripts/run_citebench_eval.py --track both --dry-run
```

### Module-Level Evaluation (System Track)

To compare module impact directly on CiteBench system evaluation (same query set, multiple variants):

```powershell
python scripts/evaluate_mitigation_citebench.py --max-samples 10 --provider deepseek --model-name deepseek-chat
```

This runner will:
- Create temporary variant configs for selected module variants
- Generate per-variant system inputs from identical queries
- Run CiteEval system-track scoring for each variant
- Write `summary.json` and `summary.md` under `outputs/mitigation_eval_citebench/<timestamp>/`

Optional full variant arguments:

```powershell
python scripts/evaluate_mitigation_citebench.py --variants baseline full_pipeline mitigation_filter_only mitigation_rerank_only mitigation_reprompt_only --strategy validation --system-source benchmark/CiteEval/data/system_eval/system_eval_examples.json
```

## 6. Two-Method Comparison Workflow (RAGTruth Baseline vs LettuceDetect)

This workflow is designed for your method comparison requirement:

1. Run or collect official RAGTruth baseline output.
2. Run LettuceDetect pretrained inference and collect raw output.
3. Convert both outputs into the same CiteEval system-eval JSON schema.
4. Evaluate both with identical CiteEval settings and save JSON summaries.

### Step A: Convert official RAGTruth output to CiteEval format

If you have baseline prediction output from `benchmark/RAGTruth/baseline/predict_and_evaluate.py`
or `outputs/ragtruth_eval/*.json`, convert with:

```powershell
python scripts/convert_ragtruth_baseline_to_citeeval.py \
    --input outputs/ragtruth_eval/ragtruth_eval_test_10.json \
    --output benchmark/CiteEval/data/system_eval/ragtruth_baseline_system_eval.json \
    --report-json outputs/method_comparison/ragtruth_conversion_report.json
```

Use `--strict` to drop rows that have empty `pred` or `passages`.

### Step B: Convert LettuceDetect pretrained output to CiteEval format

The LettuceDetect adapter is field-configurable because upstream output structures can vary.

```powershell
python scripts/convert_lettucedetect_to_citeeval.py \
    --input outputs/lettucedetect/raw_predictions.json \
    --output benchmark/CiteEval/data/system_eval/lettucedetect_system_eval.json \
    --id-key id \
    --query-key query \
    --pred-key response \
    --passages-key passages \
    --report-json outputs/method_comparison/lettucedetect_conversion_report.json
```

If your raw fields differ, change the dotted key paths (for example `result.answer_text`).

### Step C: Run controlled comparison and save metrics JSON

```powershell
python scripts/compare_citebench_methods.py \
    --ragtruth-input benchmark/CiteEval/data/system_eval/ragtruth_baseline_system_eval.json \
    --lettuce-input benchmark/CiteEval/data/system_eval/lettucedetect_system_eval.json \
    --provider deepseek \
    --model-name deepseek-chat \
    --context-source oracle \
    --max-samples 30
```

The script writes a timestamped run directory under `outputs/method_comparison/` with:

- `ragtruth_aligned.json`
- `lettucedetect_aligned.json`
- `aligned_ids.json`
- `summary.json` (method metrics and deltas)
- `run_logs.json` (stdout/stderr for both evaluation runs)

### Output Contract for Verifier Comparison

Use `summary.json` as the canonical machine-readable artifact for downstream comparison
against your verifier outputs.

Minimum expected fields:

- `run.aligned_count`
- `method_metrics.ragtruth`
- `method_metrics.lettucedetect`
- `delta` (lettucedetect - ragtruth for available mean ratings)

### Upstream Conversion: CiteBench metric_eval -> Model Input Formats

If you want both pipelines to consume CiteBench metric_eval data directly, run
the upstream converters first.

#### A) Convert CiteBench metric_eval to RAGTruth baseline-style input

This converter writes JSONL records with fixed `task_type=QA` and prefilled
`response` from CiteBench `prediction`.

```powershell
python scripts/convert_citebench_metric_to_ragtruth.py \
    --input benchmark/CiteEval/data/metric_eval/metric_test/citebench.metric_test \
    --output outputs/citebench_converted/ragtruth_metric_test.jsonl \
    --split test \
    --strict \
    --aligned-ids-output outputs/citebench_converted/aligned_ids.metric_test.json \
    --report-json outputs/citebench_converted/ragtruth_metric_test.report.json
```

#### B) Convert CiteBench metric_eval to LettuceDetect input

This converter preserves `id/query/response/passages` and can optionally add a
flattened `context` string.

```powershell
python scripts/convert_citebench_metric_to_lettucedetect.py \
    --input benchmark/CiteEval/data/metric_eval/metric_test/citebench.metric_test \
    --output outputs/citebench_converted/lettucedetect_metric_test.json \
    --output-format json \
    --include-flat-context \
    --strict \
    --aligned-ids-output outputs/citebench_converted/aligned_ids.metric_test.json \
    --report-json outputs/citebench_converted/lettucedetect_metric_test.report.json
```

#### C) Continue with downstream adaptation and comparison

After method-specific inference is complete, convert each method output to
CiteEval system format and run the comparator (Section 6, Step A/B/C).

Recommended smoke run option:

```powershell
--max-samples 10
```

### One-Command LettuceDetect Pipeline (Converters Linked)

Use the orchestrator script to run:

1. CiteBench metric_eval -> LettuceDetect input conversion
2. LettuceDetect pretrained inference (span output)
3. LettuceDetect raw output -> CiteEval system JSON conversion
4. (Optional) method comparison call

Smoke run (no compare):

```powershell
python scripts/run_lettucedetect_pipeline.py \
    --source-metric-file benchmark/CiteEval/data/metric_eval/metric_test/citebench.metric_test \
    --metric-split test \
    --model-path KRLabsOrg/lettucedect-base-modernbert-en-v1 \
    --max-samples 10 \
    --strict \
    --include-flat-context
```

Run and compare immediately (requires RAGTruth system input):

```powershell
python scripts/run_lettucedetect_pipeline.py \
    --source-metric-file benchmark/CiteEval/data/metric_eval/metric_test/citebench.metric_test \
    --metric-split test \
    --model-path KRLabsOrg/lettucedect-base-modernbert-en-v1 \
    --max-samples 30 \
    --strict \
    --include-flat-context \
    --run-compare \
    --ragtruth-input benchmark/CiteEval/data/system_eval/ragtruth_baseline_system_eval.json \
    --provider deepseek \
    --eval-model-name deepseek-chat \
    --context-source oracle
```

The run directory (`outputs/lettucedetect_pipeline/<timestamp>/`) includes:

- `lettucedetect_input.json`
- `lettucedetect_raw_predictions.json`
- `lettucedetect_system_eval.json`
- `upstream_conversion_report.json`
- `downstream_conversion_report.json`
- `run_manifest.json`

## 7. Integrating with Verifier Signals

Our custom verifier signals (NLI, Entropy, Coverage) can be compared against CiteEval scores to validate their effectiveness as "trainless" hallucination detectors.

For more details on specific CiteEval metrics, refer to the [CiteEval README](benchmark/CiteEval/README.md).
