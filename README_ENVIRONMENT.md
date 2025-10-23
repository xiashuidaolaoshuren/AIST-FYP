# AIST-FYP Environment Setup

This document provides instructions for setting up the development environment for the AIST-FYP (AI: System & Technologies - Final Year Project) on LLM Hallucination Detection.

## System Requirements

- **Python**: 3.12+ (Tested with Python 3.12.9)
- **GPU**: NVIDIA GPU with CUDA support (Recommended)
  - Tested on: NVIDIA GeForce RTX 3070 Ti (8GB VRAM)
  - CUDA 12.1+ compatible
- **OS**: Windows, Linux, or macOS
- **RAM**: 16GB+ recommended

## Quick Start

### 1. Clone the Repository

```bash
git clone https://github.com/xiashuidaolaoshuren/AIST-FYP.git
cd AIST-FYP
```

### 2. Create Virtual Environment

```bash
# Create venv
python -m venv venv

# Activate venv
# On Windows (PowerShell):
.\venv\Scripts\Activate.ps1

# On Windows (CMD):
.\venv\Scripts\activate.bat

# On Linux/macOS:
source venv/bin/activate
```

### 3. Install PyTorch with CUDA Support

**For GPU (CUDA 12.1):**
```bash
pip install torch torchvision torchaudio --index-url https://download.pytorch.org/whl/cu121
```

**For CPU-only:**
```bash
pip install torch torchvision torchaudio --index-url https://download.pytorch.org/whl/cpu
```

### 4. Install Other Dependencies

```bash
pip install transformers datasets sentence-transformers faiss-cpu numpy pandas tqdm scikit-learn matplotlib seaborn
```

Or use the requirements file (after installing PyTorch):
```bash
pip install -r requirements.txt
```

**Additional dependencies for Month 2:**
```bash
# spaCy for sentence segmentation
pip install spacy
python -m spacy download en_core_web_sm

# For production Wikipedia dump processing (optional)
pip install requests tqdm

# For GPU acceleration (already in requirements)
pip install accelerate
```

### 5. Verify Installation

```bash
python verify_gpu.py
```

Expected output should show:
- ✓ CUDA Available: True
- ✓ GPU detected and working
- ✓ All libraries installed correctly

## Installed Packages

### Core ML Libraries
- **transformers** (4.56.2): Hugging Face Transformers for LLMs
- **torch** (2.5.1+cu121): PyTorch with CUDA 12.1 support
- **datasets** (4.1.1): Hugging Face Datasets
- **sentence-transformers** (5.1.1): Sentence embeddings

### Retrieval & Search
- **faiss-cpu** (1.12.0): Efficient similarity search

### Data Processing
- **numpy** (2.3.3): Numerical computing
- **pandas** (2.3.3): Data manipulation
- **scikit-learn** (1.7.2): Machine learning utilities

### Utilities
- **tqdm** (4.67.1): Progress bars
- **matplotlib** (3.10.6): Plotting
- **seaborn** (0.13.2): Statistical visualization

## Project Structure

```
AIST-FYP/
├── venv/                          # Virtual environment (git-ignored)
├── reference/                     # Research papers and summaries
├── src/                           # Source code (Month 2+)
│   ├── utils/                     # Utilities (config, logger, data structures)
│   ├── data_processing/           # Wikipedia parser, chunker, embeddings
│   ├── retrieval/                 # Dense retriever, FAISS index
│   └── generation/                # Generator wrapper, claim extractor
├── scripts/                       # Utility scripts
│   └── download_wikipedia.py      # Wikipedia data download script
├── tests/                         # Unit and integration tests
├── data/                          # Data directory (git-ignored)
│   ├── raw/                       # Raw Wikipedia data
│   ├── processed/                 # Processed chunks
│   ├── embeddings/                # Embeddings and checkpoints
│   └── indexes/                   # FAISS indexes
├── logs/                          # Log files (git-ignored)
├── config.yaml                    # Configuration file
├── requirements.txt               # Python dependencies
├── verify_gpu.py                  # GPU verification script
├── Hallucination_Project_Details.txt
├── System_Architecture_Design.md
├── TODO_List.md
└── README_ENVIRONMENT.md          # This file
```

## Troubleshooting

### CUDA Not Available

If `verify_gpu.py` shows "CUDA Available: False":

1. **Check GPU**: Run `nvidia-smi` to verify GPU is detected
2. **Update Drivers**: Ensure NVIDIA drivers are up to date
3. **Reinstall PyTorch**: Uninstall and reinstall with correct CUDA version
   ```bash
   pip uninstall torch torchvision torchaudio
   pip install torch torchvision torchaudio --index-url https://download.pytorch.org/whl/cu121
   ```

### Out of Memory Errors

If you encounter GPU memory errors:
- Reduce batch size
- Enable gradient checkpointing
- Use CPU for large models: `device='cpu'`

### Import Errors

If you get import errors:
```bash
pip install --upgrade transformers datasets
```

## Development Workflow

1. **Activate environment** before working:
   ```bash
   .\venv\Scripts\Activate.ps1  # Windows
   source venv/bin/activate      # Linux/macOS
   ```

2. **Check GPU status** periodically:
   ```bash
   python verify_gpu.py
   ```

3. **Install new packages** as needed:
   ```bash
   pip install <package_name>
   ```

## Next Steps

After completing the environment setup, proceed with:

### Month 2 - Baseline RAG System

#### 0. Configure HuggingFace Cache (Important for Windows Users)

**Problem:** By default, HuggingFace datasets cache to `C:\Users\{username}\.cache\huggingface`, which can quickly fill up your C: drive (especially during Wikipedia dataset downloads which can require ~15GB).

**Solution:** Set the `HF_HOME` environment variable to use D: drive (or any drive with sufficient space):

**PowerShell (Recommended):**
```powershell
# Set environment variable permanently for current user
[System.Environment]::SetEnvironmentVariable('HF_HOME', 'D:\huggingface_cache', 'User')

# Set for current session
$env:HF_HOME = 'D:\huggingface_cache'

# Verify
echo $env:HF_HOME
```

**Command Prompt:**
```cmd
setx HF_HOME "D:\huggingface_cache"
```

**Note:** After setting the environment variable, restart your terminal/PowerShell for it to take effect. This will redirect all HuggingFace dataset downloads to D: drive.

#### 1. Download Wikipedia Data

The project supports three download strategies:

**Development (Recommended for testing):**
```bash
# Download 10k articles (~5 minutes, ~50MB)
python scripts/download_wikipedia.py --strategy development
```

**Validation (Medium-scale testing):**
```bash
# Download 100k articles (~30 minutes, ~500MB)
python scripts/download_wikipedia.py --strategy validation
```

**Production (Full Wikipedia):**
```bash
# Download full Wikipedia dump (~20GB, several hours)
python scripts/download_wikipedia.py --strategy production

# Or specify a specific dump date
python scripts/download_wikipedia.py --strategy production --dump-date 20240101
```

#### 2. Process Wikipedia Data

After downloading, process the data:

```bash
# Process downloaded data into chunks (development strategy)
python -m src.data_processing.wikipedia_parser \
    --input data/raw/wiki_sample_development.jsonl \
    --output data/processed/wiki_chunks_development.jsonl

# Generate embeddings
python -m src.data_processing.embedding_generator \
    --input data/processed/wiki_chunks_development.jsonl \
    --output data/embeddings/wiki_embeddings_development.npy \
    --config config.yaml

# Build FAISS index
python -m src.retrieval.faiss_index_manager \
    --embeddings data/embeddings/wiki_embeddings_development.npy \
    --output data/indexes/development/faiss.index \
    --config config.yaml
```

#### 3. Run Baseline RAG Pipeline

```bash
# Run the baseline RAG pipeline
python -m src.generation.baseline_rag_pipeline \
    --query "What is machine learning?" \
    --config config.yaml \
    --strategy development
```

For detailed usage instructions, see `docs/month2_usage.md` (after Task 11 completion).

### Month 1 Tasks (If not completed)

1. **Data Sourcing**
   - Download Wikipedia corpus (see above)
   - Download benchmarks: TruthfulQA, RAGTruth, FEVER

2. **Baseline & Retrieval Module** (Month 2)
   - Implement dense retriever ✓
   - Build FAISS index ✓
   - Create baseline RAG system ✓

Refer to `TODO_List.md` for the complete project timeline.

## Resources

- [Project Details](Hallucination_Project_Details.txt)
- [System Architecture](System_Architecture_Design.md)
- [TODO List](TODO_List.md)
- [PyTorch Documentation](https://pytorch.org/docs/stable/index.html)
- [Hugging Face Documentation](https://huggingface.co/docs)

## Contact

For questions or issues, refer to the project documentation or contact the project supervisor.

---

**Last Updated**: 2025-10-23 (Month 2 complete)
**Python Version**: 3.12.9
**PyTorch Version**: 2.5.1+cu121
**CUDA Version**: 12.1
