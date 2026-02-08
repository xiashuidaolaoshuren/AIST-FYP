# CiteEval (CiteBench) Evaluation Guide

This guide describes how to evaluate the citation quality and factual consistency of our RAG pipeline using the [CiteEval](https://github.com/amazon-science/CiteEval) framework and the **CiteBench** benchmark.

## 1. Overview

CiteEval focuses on fine-grained citation assessment, evaluating not just whether a claim is supported, but how accurately citations are used within the context.

- **CiteBench**: The multi-domain benchmark dataset.
- **CiteEval-Auto**: A suite of model-based metrics (AutoAIS, etc.) used for evaluation.

## 2. Prerequisites

Ensure you have completed the setup in [docs/evaluation_setup_guide.md](docs/evaluation_setup_guide.md):
- CiteEval repository downloaded to `benchmark/CiteEval/`.
- CiteBench data placed in `benchmark/CiteEval/data/`.
- Required environment variables set (see `benchmark/CiteEval/README.md`).

### Provider Configuration (OpenAI or DeepSeek)

You can configure the evaluator with a project-root [.env](.env) file. The code auto-loads this file if `python-dotenv` is installed.

**OpenAI (default)**
```
CITEEVAL_PROVIDER=openai
OPENAI_API_KEY=your_openai_key
```

**DeepSeek (OpenAI-compatible)**
```
CITEEVAL_PROVIDER=deepseek
DEEPSEEK_API_KEY=your_deepseek_key
DEEPSEEK_BASE_URL=https://api.deepseek.com
```

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

CiteEval provides several metrics. The most relevant for our project is the **System Evaluation** suite.

### Step 1: Prepare System Output
Use the conversion script to generate CiteEval-formatted JSON:
```bash
python scripts/convert_to_citeeval.py \
    --input outputs/full_pipeline_queries_20260201_173435.json \
    --output benchmark/CiteEval/data/system_eval/my_pipeline_results.json
```

### Step 2: Run CiteEval-Auto

1. Activate the CiteBench virtual environment:
```powershell
.\.venv_citeeval\Scripts\Activate.ps1
```

2. Run the evaluation script from within the CiteEval directory:
```bash
cd benchmark/CiteEval
export PYTHONPATH=$PYTHONPATH:$(pwd):$(pwd)/src
python src/scripts/run_citeeval.py \
    --input_file data/system_eval/my_pipeline_results.json \
    --output_dir data/system_eval_outputs/ \
    --model_name deepseek-chat
```
*(Note: Use `--model_name gpt-4o` with `CITEEVAL_PROVIDER=openai`.)*

### Step 3: Analyze Results
The evaluation will produce scores for:
- **Cite-Precision**: Accuracy of citations.
- **Cite-Recall**: Coverage of citations for supported claims.
- **AutoAIS**: Overall factual alignment score.

## 5. Integrating with Verifier Signals

Our custom verifier signals (NLI, Entropy, Coverage) can be compared against CiteEval scores to validate their effectiveness as "trainless" hallucination detectors.

For more details on specific CiteEval metrics, refer to the [CiteEval README](benchmark/CiteEval/README.md).
