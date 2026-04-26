# AIST-FYP Colab Evaluation Guide

This comprehensive guide explains how to run the complete hallucination detection evaluation pipeline on Google Colab.

## Table of Contents

1. [Overview](#overview)
2. [Prerequisites](#prerequisites)
3. [Stage 0: One-Time Setup](#stage-0-one-time-setup)
4. [Stage 1: Wikipedia Preprocessing](#stage-1-wikipedia-preprocessing)
5. [Stage 2: Verifier Evaluation](#stage-2-verifier-evaluation)
6. [Stage 3: Mitigation Evaluation](#stage-3-mitigation-evaluation)
7. [Stage 4: Baseline Comparison](#stage-4-baseline-comparison)
8. [Troubleshooting](#troubleshooting)
9. [Quick Reference](#quick-reference)

---

## Overview

The AIST-FYP evaluation pipeline consists of **4 main stages**:

| Stage | Notebook | Duration | Purpose |
|-------|----------|----------|---------|
| **1** | `colab_wikipedia_preprocessing.ipynb` | 2-4 hours | Build knowledge base (FAISS index, chunks, embeddings) |
| **2** | `colab_verifier_eval_citebench.ipynb` or `colab_verifier_eval_ragtruth.ipynb` | 1-3 hours | Test hallucination detection signals (Entropy, NLI, Coverage, Self-Consistency) |
| **3** | `colab_mitigation_eval_citebench.ipynb` or `colab_mitigation_eval_ragtruth.ipynb` | 1-2 hours | Test claim filtering and evidence re-ranking strategies |
| **4** | `colab_citebench_dual_pipeline_eval.ipynb` | 1-2 hours | Compare your verifier against LettuceDetect baseline |

**Total time:** ~6-8 hours (or 1-2 hours with smoke tests)

### System Architecture

```
User Query
    ↓
Baseline RAG Module
    ├─ Retriever (FAISS/BM25)
    └─ Generator (Qwen LLM)
    ↓
Verifier Module (Trainless Signals)
    ├─ Entropy (Token uncertainty)
    ├─ NLI (Contradiction detection)
    ├─ Coverage (Entity overlap with evidence)
    └─ Self-Agreement (Consistency across samples)
    ├─ Rule-Based Aggregator → Confidence Score
    ↓
Mitigation Module
    ├─ Evidence Re-Ranking
    └─ Claim Filtering
    ↓
Final Output with Citations
```

---

## Prerequisites

### System Requirements
- **GPU:** Google Colab with GPU enabled (Runtime → Change runtime type → GPU)
- **Storage:** ~100GB on Google Drive for artifacts and outputs
- **Time:** 6-8 hours for full pipeline

### What You Need Prepared

1. **Google Drive account** with sufficient quota
2. **API Keys** (stored securely in Colab Secrets):
   - `DEEPSEEK_API_KEY` (required for citation evaluation)
   - `OPENAI_API_KEY` (optional, alternative to DeepSeek)
   - `HUGGINGFACE_TOKEN` (optional, for gated models)
3. **Benchmark datasets** (CiteEval, RAGTruth) — can be downloaded automatically

---

## Stage 0: One-Time Setup

### Step 0.1: Create Google Drive Folder Structure

Create this structure in your Google Drive root (`/content/drive/MyDrive/`):

```
AIST-FYP-colab-preprocess/          ← Wikipedia preprocessing outputs
AIST-FYP-colab-outputs/             ← Evaluation outputs (metrics, results)
AIST-FYP/                           ← Project repository & benchmarks
  ├─ benchmark/
  │   ├─ CiteEval/                 ← Citation evaluation benchmark
  │   └─ RAGTruth/                 ← RAG hallucination benchmark
  └─ ...
data/                               ← Shared artifacts (if not in AIST-FYP/)
  ├─ indexes/                       ← FAISS and BM25 indexes
  │   └─ validation/
  │       ├─ faiss.index
  │       ├─ metadata.pkl
  │       └─ bm25_index.pkl
  ├─ processed/                     ← Wikipedia chunks
  │   └─ wiki_chunks_validation.jsonl
  └─ embeddings/                    ← Embedding vectors
      └─ embeddings_validation.npy
```

### Step 0.2: Add Colab Secrets

1. Open any Colab notebook
2. Click the **🔑 Secrets** icon in the left sidebar
3. Add these secrets and toggle **"Notebook access" ON**:
   - `DEEPSEEK_API_KEY` → Get from [deepseek.com](https://platform.deepseek.com)
   - `OPENAI_API_KEY` → Get from [platform.openai.com](https://platform.openai.com)
   - `HUGGINGFACE_TOKEN` → Get from [huggingface.co](https://huggingface.co/settings/tokens)

### Step 0.3: Upload Benchmarks (Optional)

If you have benchmarks locally, upload to Google Drive:

```
/content/drive/MyDrive/AIST-FYP/benchmark/
├─ CiteEval/
│  └─ data/
│     └─ metric_eval/
│        └─ metric_test/
│           └─ citebench.metric_test
└─ RAGTruth/
   └─ dataset/
      └─ ...
```

If not provided, notebooks will download automatically.

---

## Stage 1: Wikipedia Preprocessing

**Notebook:** `colab_wikipedia_preprocessing.ipynb`  
**Duration:** 2-4 hours (for production strategy) or 30 min (for validation)  
**Outputs:** FAISS index, chunks, embeddings for RAG retrieval

### Workflow

```
Download Wikipedia
    ↓
Parse into Chunks (sentence-level)
    ↓
Generate Embeddings (vector representations)
    ↓
Build FAISS Index (dense vector search)
    ↓
Build BM25 Index (sparse keyword search)
    ↓
Export to Google Drive
```

### Configuration

Open the **"Parameters"** cell and customize:

```python
# Processing strategy (choose one)
STRATEGY = "validation"  # Fast, 100K articles (good for testing)
                        # "production" = full corpus (slow, 1.5M articles)
                        # "development" = tiny, 5K articles (quickest)

# Google Drive locations
DRIVE_OUTPUT_ROOT = "/content/drive/MyDrive/AIST-FYP-colab-preprocess"
LOCAL_WORK_ROOT = f"{DRIVE_OUTPUT_ROOT}/work"  # Auto-synced to Drive

# Build targets
BUILD_FAISS = True      # Enable dense vector index
BUILD_BM25 = True       # Enable sparse keyword index

# For quick testing, limit articles
MAX_ARTICLES_OVERRIDE = None  # None = use default for strategy
                              # 5000 = use only 5K articles (testing)

# Resume from interruptions
RESUME = True           # Resume from last checkpoint
RESET_CHECKPOINT = False  # True = start from scratch
```

### Step-by-Step Execution

| Step | Cell Name | What It Does |
|------|-----------|--------------|
| 1 | Setup API Keys | Load secrets from Colab Secrets |
| 2 | Mount Drive & Clone Repo | Connect to Drive, clone AIST-FYP repository |
| 3 | Install Dependencies | Setup uv virtual environment, install packages |
| 4 | Download Wikipedia | Run `scripts/download_wikipedia.py` |
| 5 | Prepare Chunks | Run `scripts/prepare_wikipedia_chunks.py` |
| 6 | Generate Embeddings | Run `scripts/generate_embeddings.py` (streaming) |
| 7 | Build FAISS Index | Run `scripts/build_faiss_index.py` with checkpoints |
| 8 | Build BM25 Index | Run `scripts/build_bm25_index.py` with checkpoints |
| 9 | Export to Drive | Copy artifacts to Drive (if using local temp storage) |

### Monitoring Progress

- Each cell displays **progress bars** for long-running tasks
- **Checkpoints are auto-saved** — if Colab disconnects, restart the notebook to resume
- Check Drive folder for completed artifacts: `/content/drive/MyDrive/AIST-FYP-colab-preprocess/work/`

### Output Verification

After completion, verify these files exist:

```bash
# Check FAISS index
ls -lh /content/AIST-FYP/data/indexes/validation/
  faiss.index                       (1-5 GB)
  metadata.pkl                      (10-100 MB)
  bm25_index.pkl                    (100-500 MB)

# Check chunks
ls -lh /content/AIST-FYP/data/processed/
  wiki_chunks_validation.jsonl      (500 MB - 2 GB)

# Check embeddings
ls -lh /content/AIST-FYP/data/embeddings/
  embeddings_validation.npy         (500 MB - 1 GB)
```

---

## Stage 2: Verifier Evaluation

**Notebook:** `colab_verifier_eval_citebench.ipynb` (CiteBench) or `colab_verifier_eval_ragtruth.ipynb` (RAGTruth)  
**Duration:** 1-3 hours (full) or 15-30 min (smoke test)  
**Purpose:** Evaluate individual hallucination detection signals

### Prerequisites Checklist

Before running, verify these artifacts exist:

```
✓ /content/AIST-FYP/data/indexes/validation/
  ├─ faiss.index
  └─ metadata.pkl
✓ /content/AIST-FYP/data/processed/
  └─ wiki_chunks_validation.jsonl
✓ /content/AIST-FYP/benchmark/CiteEval/
  └─ (should have data/metric_eval/ subdirectories)
```

If missing, copy them to `/content/AIST-FYP/` after cloning.

### What Gets Tested

The notebook evaluates **5 verifier variants** on citation accuracy:

| Variant | Signals Used | Purpose |
|---------|--------------|---------|
| `full_verifier_filter` | All 4 signals | Baseline (all together) |
| `verifier_intrinsic_filter` | Entropy only | Measure uncertainty contribution |
| `verifier_grounded_filter` | Coverage only | Measure evidence matching |
| `verifier_nli_filter` | NLI only | Measure contradiction detection |
| `verifier_self_agreement_filter` | Consistency only | Measure claim stability |

### Configuration

```python
# ========== Evaluation Dataset ==========
RUN_CITEEVAL_VERIFIER_MODULE_EVAL = True   # Enable CiteBench
CITEEVAL_MAX_SAMPLES = 10                  # Change to None for FULL evaluation
CITEEVAL_ORACLE_DATASET = "asqa"           # Options: asqa | eli5 | msmarco
CITEEVAL_METRIC_SPLIT = "test"             # "test" or "dev"

# ========== LLM Settings ==========
GENERATOR_MODEL = "Qwen/Qwen3-4B-Instruct-2507"  # 4B parameter model
GENERATION_MAX_NEW_TOKENS = 256            # Max tokens per response

# ========== Verifier Signal Settings ==========
SELF_AGREEMENT_K_SAMPLES = 2               # Generate N responses for consistency
SELF_AGREEMENT_TEMPERATURE = 1.0           # Sampling temperature (1.0 = max variety)
NLI_BATCH_SIZE = 32                        # Batch size for NLI model

# ========== Evaluation Provider ==========
CITEEVAL_PROVIDER = "deepseek"             # Citation scorer (requires API key)
CITEEVAL_MODEL_NAME = ""                   # Use default (deepseek-chat)

# ========== Output Location ==========
DRIVE_OUTPUT_DIR = "/content/drive/MyDrive/AIST-FYP-colab-outputs"
STRATEGY = "validation"                    # Must match preprocessing strategy
```

### Execution Steps

1. **Setup** → Load API keys and mount Drive
2. **Clone & Install** → Clone repo, setup environment
3. **Validate Artifacts** → Check FAISS index, chunks, CiteEval paths
4. **Dry-Run** (optional) → Test setup without full evaluation
5. **Run Verifier** → Evaluate all 5 variants on dataset
6. **Generate Metrics** → Score with CiteEval
7. **Export** → Copy results to Drive

### Understanding Results

After completion, find metrics in:
```
/content/drive/MyDrive/AIST-FYP-colab-outputs/work_eval/citeeval/metric_eval_outputs/metric_test/
```

**Key Metrics:**
- **CE** (Citation Exact): Exact match of cited passages (0-1, higher is better)
- **CA** (Citation Accuracy): Quality of cited evidence (0-1)
- **CR** (Citation Recall): Coverage of important citations (0-1)

**Example Output:**
```
full_verifier_filter:           CE=0.68, CA=0.75, CR=0.70
verifier_nli_filter:            CE=0.62, CA=0.69, CR=0.65
verifier_intrinsic_filter:      CE=0.54, CA=0.61, CR=0.57
verifier_grounded_filter:       CE=0.59, CA=0.66, CR=0.60
verifier_self_agreement_filter: CE=0.55, CA=0.62, CR=0.58
```

**Interpretation:**
- `full_verifier_filter` performs best (combines all signals)
- NLI signal has strongest individual impact
- Entropy/coverage have weaker but complementary signals

---

## Stage 3: Mitigation Evaluation

**Notebook:** `colab_mitigation_eval_citebench.ipynb` (CiteBench) or `colab_mitigation_eval_ragtruth.ipynb` (RAGTruth)  
**Duration:** 1-2 hours (full) or 15-30 min (smoke test)  
**Purpose:** Test claim filtering and evidence re-ranking strategies

### What Gets Tested

```
Original RAG Output
    ↓
Apply Mitigation Strategies
    ├─ Evidence Re-ranking
    │  └─ Re-order evidence by confidence score
    └─ Claim Filtering
       └─ Remove contradictory claims
    ↓
Mitigated Output (improved quality)
    ↓
Evaluate Metrics (compare before/after)
```

### Configuration

```python
# ========== Mitigation Strategies ==========
RUN_EVIDENCE_RERANKING = True              # Enable evidence re-ranking
RUN_CLAIM_FILTERING = True                 # Enable claim filtering
RUN_REPROMPTING = False                    # Advanced: self-correction (slow)

# ========== Mitigation Thresholds ==========
RERANKER_ALPHA = 0.6                       # Weight for retrieval relevance
RERANKER_BETA = 0.4                        # Weight for verification confidence
FILTER_CONTRADICTION_THRESHOLD = 0.5       # NLI threshold (0-1)

# ========== Keep Other Settings Same ==========
CITEEVAL_MAX_SAMPLES = 10                  # Smoke test setting
CITEEVAL_DRY_RUN_FIRST = True
STRATEGY = "validation"
```

### Expected Improvements

**Before Mitigation:**
```
CE=0.68, CA=0.75, CR=0.70
```

**After Evidence Re-ranking:**
```
CE=0.70, CA=0.77, CR=0.72  (↑2-3 points)
```

**After Claim Filtering:**
```
CE=0.72, CA=0.78, CR=0.71  (↑4-5 points)
```

The goal is to **reduce hallucinations** while maintaining citation coverage.

---

## Stage 4: Baseline Comparison

**Notebook:** `colab_citebench_dual_pipeline_eval.ipynb`  
**Duration:** 1-2 hours  
**Purpose:** Compare your verifier against LettuceDetect baseline

### Why "Oracle Mode"?

Both systems see:
- **Same LLM responses** (generated by Qwen)
- **Same gold passages** (from CiteEval dataset)

Only difference:
- **Your system:** Uses multi-signal verifier (Entropy + NLI + Coverage + Self-Agreement)
- **LettuceDetect:** Uses pre-trained classifier on citations

This is a **fair, apples-to-apples comparison**.

### Configuration

```python
# ========== Data ==========
METRIC_SPLIT = 'test'                      # "test" or "dev"
MAX_SAMPLES = 10                           # Change to None for full

# ========== LettuceDetect Baseline ==========
LETTUCE_MODEL_PATH = 'KRLabsOrg/lettucedect-base-modernbert-en-v1'
LETTUCE_USE_SPAN_CITATIONS = True
LETTUCE_CONFIDENCE_THRESHOLD = 0.5

# ========== Your Pipeline ==========
PIPELINE_VARIANT = 'full_verifier'         # Which verifier to compare
CITEEVAL_PROVIDER = 'deepseek'
```

### Expected Output

**Head-to-Head Comparison:**

```
┌──────────────────┬────┬────┬────┐
│ Method           │ CE │ CA │ CR │
├──────────────────┼────┼────┼────┤
│ LettuceDetect    │0.58│0.65│0.62│
│ Your Verifier    │0.68│0.75│0.70│
│ Relative Gain    │+17%│+15%│+13%│
└──────────────────┴────┴────┴────┘
```

---

## Troubleshooting

### Issue: "FAISS index not found"

**Error:** `FileNotFoundError: data/indexes/validation/faiss.index`

**Solution:**
1. Verify preprocessing completed successfully
2. Copy from Drive manually:
   ```bash
   cp -r /content/drive/MyDrive/AIST-FYP-colab-preprocess/work/data/indexes \
         /content/AIST-FYP/data/
   ```
3. Check file exists: `ls -lh /content/AIST-FYP/data/indexes/validation/`

---

### Issue: "API Key not found" (DeepSeek)

**Error:** `openai.error.AuthenticationError` or "DEEPSEEK_API_KEY not set"

**Solution:**
1. Open Colab → Click 🔑 **Secrets** in left sidebar
2. Click **"+ New Secret"** → Add `DEEPSEEK_API_KEY`
3. Paste your key from [deepseek.com](https://platform.deepseek.com)
4. Toggle **"Notebook access"** to **ON**
5. Re-run the API key loading cell

---

### Issue: GPU Out-of-Memory (OOM)

**Error:** `CUDA out of memory` or `RuntimeError: CUDA error: out of memory`

**Solution 1: Reduce Batch Sizes**
```python
EMBED_BATCH_SIZE_OVERRIDE = 64         # Smaller than default 128
BM25_TOKENIZE_BATCH_SIZE = 1024        # Smaller than default 2048
FAISS_ADD_BATCH_SIZE = 10000           # Smaller than default 50000
NLI_BATCH_SIZE = 16                    # Smaller than default 32
```

**Solution 2: Use Lower Precision**
```python
GENERATOR_DTYPE = "fp32"               # Instead of bf16 (uses more memory but sometimes more stable)
DISABLE_FP16 = True
```

**Solution 3: Clear Cache & Restart**
```python
# In a new cell:
import torch
torch.cuda.empty_cache()
# Then: Runtime → Restart runtime
```

---

### Issue: Colab Runtime Disconnected

**Problem:** Notebook was running long and got disconnected mid-evaluation

**Solution:** The notebook auto-saves checkpoints to Drive, so:
1. **Restart the runtime** (`Runtime → Restart runtime`)
2. **Re-run from the top** of the notebook
3. The notebook will **automatically resume** from the last checkpoint
   - Set `RESUME = True` (default)
   - If you want a fresh start: Set `RESET_CHECKPOINT = True`

---

### Issue: "CiteEval benchmark not found"

**Error:** `FileNotFoundError: benchmark/CiteEval/data/metric_eval/...`

**Solution:**
1. Check if benchmark exists: `ls -la /content/AIST-FYP/benchmark/`
2. If missing, download from Drive:
   ```bash
   cp -r /content/drive/MyDrive/AIST-FYP/benchmark/CiteEval \
         /content/AIST-FYP/benchmark/
   ```
3. If not on Drive, the notebook will download it (slow)

---

### Issue: "Smoke test passes, but full evaluation hangs"

**Problem:** Full evaluation (e.g., `MAX_SAMPLES = None`) takes too long or times out

**Solution:**
1. **Check if it's actually running:**
   - Look for progress bars in the cell output
   - Check `/content/drive/MyDrive/...` for new output files

2. **Increase the timeout:**
   - Colab cells have ~1 hour timeout by default
   - Full evaluation may take 2+ hours
   - Best practice: Check status every 30 min, don't leave unattended

3. **Split into multiple notebooks:**
   - Run verifier eval in one notebook
   - Run mitigation in another
   - Less risk of timeout

---

## Quick Reference

### Smoke Test (15-30 min per notebook)

```python
# Preprocessing
STRATEGY = "validation"
MAX_ARTICLES_OVERRIDE = 5000

# Verifier Evaluation
CITEEVAL_MAX_SAMPLES = 10
CITEEVAL_DRY_RUN_FIRST = True

# Mitigation Evaluation
CITEEVAL_MAX_SAMPLES = 10

# Baseline Comparison
MAX_SAMPLES = 10
```

### Full Evaluation (2-8 hours)

```python
# Preprocessing
STRATEGY = "validation"  # or "production" for ultimate accuracy
MAX_ARTICLES_OVERRIDE = None

# Verifier Evaluation
CITEEVAL_MAX_SAMPLES = None  # All samples
CITEEVAL_DRY_RUN_FIRST = False

# Mitigation Evaluation
CITEEVAL_MAX_SAMPLES = None

# Baseline Comparison
MAX_SAMPLES = None
```

### Check Outputs Location

```
/content/drive/MyDrive/AIST-FYP-colab-outputs/work_eval/
├─ outputs/
│  ├─ verifier_eval_citebench/     ← Verifier signal results
│  └─ verifier_eval_ragtruth/      ← RAGTruth results (if enabled)
└─ citeeval/
   ├─ system_eval_outputs/         ← System outputs before scoring
   └─ metric_eval_outputs/         ← Final metrics (CE, CA, CR)
      └─ metric_test/
         └─ citeeval_result.json   ← Main results file
```

### Key Files to Check

| File | Location | Content |
|------|----------|---------|
| Metrics | `.../metric_test/citeeval_result.json` | CE, CA, CR scores for each variant |
| System Output | `.../system_eval_outputs/` | Generated answers + citations |
| Logs | `.../work_eval/logs/` | Debug info from each evaluation |
| Config | `.../config.colab.yaml` | Auto-generated configuration used |

---

## Next Steps

1. **Run Smoke Test First**
   - Verify setup works with small dataset
   - Takes 15-30 min per notebook
   - Catches configuration errors early

2. **Review Results**
   - Check output metrics in JSON files
   - Compare against baseline
   - Identify which signals work best

3. **Run Full Evaluation**
   - After smoke test passes
   - Allocate 6-8 hours
   - Schedule for when you can monitor

4. **Analyze & Iterate**
   - Modify verifier parameters
   - Re-run to measure impact
   - Document improvements

---

## Additional Resources

- **Project Details:** See [Hallucination_Project_Details.md](../Hallucination_Project_Details.md)
- **System Architecture:** See [System_Architecture_Design.md](../System_Architecture_Design.md)
- **Month 5 Implementation:** See [docs/month5_mitigation_and_ui.md](../docs/month5_mitigation_and_ui.md)
- **CiteEval Docs:** See [docs/citeeval_evaluation_guide.md](../docs/citeeval_evaluation_guide.md)

---

**Last Updated:** March 29, 2026  
**For Questions:** Refer to project documentation or contact the team
