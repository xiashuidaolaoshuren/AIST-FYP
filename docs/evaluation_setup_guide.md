# Evaluation Data & Benchmark Setup Guide

This guide provides instructions for preparing the Wikipedia corpus and downloading the benchmarks required for evaluating the hallucination detection pipeline.

## 0. Virtual Environment Setup

The evaluation pipeline uses a single project-root virtual environment (`.venv`) for both the main RAG system and CiteEval benchmark workflows.

### Main Project Environment (Single Environment)
Used for: Wikipedia preparation, running the RAG pipeline, RAGTruth evaluation, and CiteEval/CiteBench evaluation.
```bash
# Root directory of AIST-FYP
python -m venv .venv
.\.venv\Scripts\Activate.ps1  # Windows PowerShell
pip install -r requirements.txt
python -m spacy download en_core_web_sm
```

---

## 0.5 Environment Configuration

### Setting Up the .env File

The project requires a `.env` file to configure API keys and environment variables. Follow these steps:

1. **Obtain the .env file** from your project lead (Felix)
2. **Place it in the project root directory**: `AIST-FYP/.env`
   - This is the same directory where `config.yaml`, `README.md`, and `requirements.txt` are located
   - The path should be: `d:\Felix_stuff\AIST-FYP\.env` (Windows) or the equivalent on your system
3. **Do NOT commit this file to Git** - it should remain on your local machine only

The `.env` file contains sensitive credentials (API keys) and is automatically loaded by the pipeline through the `python-dotenv` package. Key configurations include:

**For DeepSeek API (Citation Evaluation):**
```env
CITEEVAL_PROVIDER=deepseek
DEEPSEEK_API_KEY=your_deepseek_key
DEEPSEEK_BASE_URL=https://api.deepseek.com
```

**For CiteEval System Path:**
```env
CITEEVAL_ROOT="benchmark/CiteEval"
PYTHONPATH="${PYTHONPATH}:benchmark/CiteEval:benchmark/CiteEval/src"
```

---

## 1. Wikipedia Corpus Preparation

The evaluation pipeline uses a Wikipedia corpus as the knowledge base. For evaluation, we use the **strategy validation** to ensure a sufficient scale for reliable results.

Follow these steps to download and prepare the Wikipedia chunks and indexes:

### Step 1: Download Wikipedia Content
Download the validation set (~100k articles).
```bash
python scripts/download_wikipedia.py --strategy validation
```

### Step 2: Prepare Chunks
Parse the downloaded Wikipedia articles into sentence-level chunks.
```bash
python scripts/prepare_wikipedia_chunks.py --strategy validation
```

### Step 3: Generate Embedding Index (for Dense Retrieval)
Generate vector embeddings and build the FAISS index.
```bash
# Generate embeddings
python scripts/generate_embeddings.py --strategy validation

# Build FAISS index
python scripts/build_faiss_index.py --strategy validation
```

### Step 4: Build BM25 Index (for Hybrid Retrieval)
Build and cache the BM25 index for faster loading during hybrid retrieval.
```bash
python scripts/build_bm25_index.py --strategy validation
```

---

## 2. Benchmark Datasets

The evaluation uses two primary benchmark datasets: **RAGTruth** and **CiteEval**. These should be downloaded from their respective repositories and placed in the `./benchmark` directory.

### Download RAGTruth
1. Clone or download the RAGTruth repository: [https://github.com/ParticleMedia/RAGTruth](https://github.com/ParticleMedia/RAGTruth)
2. Place the contents in `benchmark/RAGTruth/`.

### Download CiteEval
1. Clone or download the CiteEval repository: [https://github.com/amazon-science/CiteEval](https://github.com/amazon-science/CiteEval)
2. Place the contents in `benchmark/CiteEval/`.

### Using the Custom CiteBench Version (Modified for DeepSeek Integration)

A modified version of the CiteBench benchmark (with DeepSeek API integration) has been prepared. Follow these steps:

1. **Obtain the CiteBench zip file** from your project lead (Felix)
2. **Extract the zip file** to a temporary location
3. **Copy the extracted files** directly into `benchmark/CiteEval/`:
   ```bash
   # After extracting the zip file
   # Copy the contents (maintaining directory structure) to:
   # benchmark/CiteEval/
   #
   # The directory structure should look like:
   # benchmark/CiteEval/
   # ├── data/
   # ├── src/
   # ├── scripts/
   # └── ... (other files)
   ```
4. **Ensure the modified files are in place** before running CiteEval evaluation scripts

#### Configuration
Set the required environment variables in your project-root `.env` file to ensure the CiteEval scripts can resolve their internal modules:
```env
# .env in AIST-FYP/
CITEEVAL_ROOT="benchmark/CiteEval"
PYTHONPATH="${PYTHONPATH}:benchmark/CiteEval:benchmark/CiteEval/src"
```

#### Dataset Preparation (CiteBench)
1. Ensure you are using the **Main Project Environment** (`.venv`) for data management.
2. Create necessary data directories:
```bash
cd benchmark/CiteEval
mkdir -p data/metric_eval data/metric_eval_outputs data/system_eval_outputs data/system_eval
```
3. Download the CiteBench dataset from [Google Drive](https://drive.google.com/drive/folders/12Evj0f92wKz_7OGuuwq3KShTdSM8eu4v?usp=drive_link).
4. Extract and place the dataset folders (e.g., `metric_eval`, `dev`, `test`) under `benchmark/CiteEval/data/`.

### Directory Structure After Setup
Your `./benchmark` folder should look like this:
```
benchmark/
├── CiteEval/
│   ├── data/
│   │   ├── metric_eval/
│   │   ├── system_eval/
│   │   └── ...
│   ├── src/
│   └── ...
└── RAGTruth/
    ├── dataset/
    ├── baseline/
    └── ...
```

---

## 3. Running Evaluation

Once the data is prepared, you can run evaluation scripts such as:
```bash
python scripts/demo_ragtruth_eval.py
```
Refer to the following guides for detailed evaluation metrics and procedures:
- [RAGTruth Evaluation Guide](docs/ragtruth_evaluation_guide.md)
- [CiteEval (CiteBench) Evaluation Guide](docs/citeeval_evaluation_guide.md)
