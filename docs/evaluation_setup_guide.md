# Evaluation Data & Benchmark Setup Guide

This guide provides instructions for preparing the Wikipedia corpus and downloading the benchmarks required for evaluating the hallucination detection pipeline.

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

#### Environment Setup
Install CiteEval-specific dependencies:
```bash
cd benchmark/CiteEval
pip install -r requirements.txt
```

Set the required environment variables. You can add these to your project-root `.env` file:
```env
# .env in AIST-FYP/
CITEEVAL_ROOT="benchmark/CiteEval"
PYTHONPATH="${PYTHONPATH}:benchmark/CiteEval:benchmark/CiteEval/src"
```

#### Dataset Preparation (CiteBench)
1. Create necessary data directories:
```bash
cd benchmark/CiteEval
mkdir -p data/metric_eval data/metric_eval_outputs data/system_eval_outputs data/system_eval
```
2. Download the CiteBench dataset from [Google Drive](https://drive.google.com/drive/folders/12Evj0f92wKz_7OGuuwq3KShTdSM8eu4v?usp=drive_link).
3. Extract and place the dataset folders (e.g., `metric_eval`, `dev`, `test`) under `benchmark/CiteEval/data/`.

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
