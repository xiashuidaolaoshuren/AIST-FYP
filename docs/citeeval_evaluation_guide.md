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

1. Activate the CiteBench virtual environment:
```powershell
.\.venv_citeeval\Scripts\Activate.ps1
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

## 5. Integrating with Verifier Signals

Our custom verifier signals (NLI, Entropy, Coverage) can be compared against CiteEval scores to validate their effectiveness as "trainless" hallucination detectors.

For more details on specific CiteEval metrics, refer to the [CiteEval README](benchmark/CiteEval/README.md).
