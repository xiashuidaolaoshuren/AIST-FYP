# AIST-FYP Colab Evaluation Notebook Workflow

此文件用來保存「在 Google Colab 跑完整評估流程時，Notebook 的執行順序與每一步重點」。

## 0. 先完成的前置工作（只做一次）

1. 在 Google Drive 建立資料夾：
   - `AIST-FYP-colab-preprocess/`
   - `AIST-FYP-colab-outputs/`
   - `AIST-FYP/benchmark/CiteEval/`
   - `AIST-FYP/benchmark/RAGTruth/`
   - `data/indexes/`, `data/processed/`, `data/embeddings/`
2. 在 Colab Secrets 新增：
   - `DEEPSEEK_API_KEY`（建議必備）
   - `OPENAI_API_KEY`（可選）
   - `HUGGINGFACE_TOKEN`（可選）
3. 在 Colab 開啟 GPU：
   - Runtime -> Change runtime type -> GPU

---

## 1. Notebook 執行順序（建議）

### Step 1: 資料預處理（必要）
Notebook：`colab/notebooks/colab_wikipedia_preprocessing.ipynb`

目的：建立後續評估需要的檢索資產（chunks、embeddings、FAISS/BM25 index）。

建議設定：
- 先做 smoke test：
  - `STRATEGY = "validation"`
  - `MAX_ARTICLES_OVERRIDE = 5000`
- 確認流程 OK 後做完整版：
  - `MAX_ARTICLES_OVERRIDE = None`

完成後要確認有以下檔案：
- `data/processed/wiki_chunks_validation.jsonl`
- `data/embeddings/embeddings_validation.npy`
- `data/indexes/validation/faiss.index`
- `data/indexes/validation/metadata.pkl`
- （若啟用 BM25）`data/indexes/validation/bm25_index.pkl`

---

### Step 2: Verifier 評估（核心）
Notebook：`colab/notebooks/colab_verifier_eval_citebench.ipynb`

目的：評估 verifier 訊號組合與各子模組表現（例如 full verifier、NLI-only、coverage-only）。

建議設定：
- smoke test：`CITEEVAL_MAX_SAMPLES = 10`
- 完整版：`CITEEVAL_MAX_SAMPLES = None`
- `STRATEGY` 要和 Step 1 一致（通常 `validation`）

主要輸出：
- CiteEval 指標：`CE / CA / CR`
- 各 verifier variant 的比較結果

可選：
- 若要額外看 RAGTruth verifier 結果，再跑：
  - `colab/notebooks/colab_verifier_eval_ragtruth.ipynb`

---

### Step 3: Mitigation 評估（進階）
Notebook：`colab/notebooks/colab_mitigation_eval_citebench.ipynb`

目的：評估 mitigation 策略（例如 evidence reranking、claim filtering）對 CE/CA/CR 的改善。

建議設定：
- smoke test：`CITEEVAL_MAX_SAMPLES = 10`
- 完整版：`CITEEVAL_MAX_SAMPLES = None`

可選：
- 若要額外看 RAGTruth mitigation 結果，再跑：
  - `colab/notebooks/colab_mitigation_eval_ragtruth.ipynb`

---

### Step 4: Baseline 對照（最終比較）
Notebook：`colab/notebooks/colab_citebench_dual_pipeline_eval.ipynb`

目的：把你的 pipeline 和 LettuceDetect baseline 放在同一設定下做公平比較（oracle mode）。

建議設定：
- smoke test：`MAX_SAMPLES = 10`
- 完整版：`MAX_SAMPLES = None`

主要輸出：
- 兩條 pipeline 的 CE/CA/CR 對照表
- 最後可用於報告或簡報的 benchmark 結果

---

## 2. 最短可行流程（第一次建議）

1. 跑 `colab_wikipedia_preprocessing.ipynb`（先 smoke test）
2. 跑 `colab_verifier_eval_citebench.ipynb`（先 smoke test）
3. 跑 `colab_mitigation_eval_citebench.ipynb`（先 smoke test）
4. 跑 `colab_citebench_dual_pipeline_eval.ipynb`（先 smoke test）
5. 全部 smoke test 通過後，把 samples 改為 `None` 跑完整版

---

## 3. 產出位置（Drive）

常見輸出根目錄：
- `MyDrive/AIST-FYP-colab-preprocess/`
- `MyDrive/AIST-FYP-colab-outputs/`
- `MyDrive/AIST-FYP-colab-evals/`

建議每次跑完都記錄：
- 執行日期
- 使用 notebook 名稱
- 主要參數（sample 數量、dataset、variant）
- 最終 CE/CA/CR

---

## 4. 注意事項

1. 先 smoke test 再 full run，能大幅降低長時間跑到一半失敗的機率。
2. Colab 中斷後，優先使用 notebook 的 `RESUME` 相關設定接續。
3. `STRATEGY`（development/validation/production）要在各 notebook 保持一致，避免資產對不上。
4. 若 benchmark 或 index 路徑錯誤，先檢查 Drive 掛載與 symlink/path 設定。
