# Development Guidelines for AIST-FYP

**Document Purpose:** Operational rules for AI agents working on the AIST-FYP hallucination detection research project.

---

## 1. Project Context & Scope

### Project Identity
- **Focus:** Research on LLM hallucination detection using trainless verifier methods
- **Duration:** 6-month timeline (Month 1-6 with specific milestones)
- **Architecture:** RAG baseline → Multi-signal Verifier → Mitigation → UI
- **Approach:** Trainless (zero-shot signals, no fine-tuning in initial phase)

### Core Technical Components
- **Baseline RAG Module:** Retriever + Generator
- **Verifier Module (4 Signals):**
  1. Intrinsic Uncertainty (entropy)
  2. Self-Agreement (consistency)
  3. Retrieval-Grounded Heuristics (coverage)
  4. Zero-Shot NLI (contradiction detection)
- **Mitigation Module:** Flagging & suppression
- **UI Module:** Minimal confidence visualization

### Technology Stack
- **Language:** Python 3.12+
- **GPU:** NVIDIA RTX 3070Ti (CUDA 12.1+)
- **Core Libraries:** transformers, torch, faiss, sentence-transformers
- **Key Model:** `MoritzLaurer/DeBERTa-v3-base-mnli-fever-anli` (zero-shot NLI)

---

## 2. File System Rules

### Directory Structure (READ-ONLY vs WRITABLE)

**READ-ONLY (DO NOT MODIFY):**
- `.github/copilot-instructions.md` - Master AI agent instructions
- `.venv/` - Virtual environment (managed by pip)
- `.git/` - Git internals
- `reference/*.pdf` - Original research papers
- `System_Architecture_Design _ Mermaid Chart-2025-10-03-081957.png` - Architecture diagram

**WRITABLE (CAN MODIFY/CREATE):**
- `reference/*_Summary.txt` - Paper summaries (follow naming convention)
- `TODO_List.md` - Task tracking (MUST update checkboxes)
- `Hallucination_Project_Details.md` - Project documentation
- `System_Architecture_Design.md` - Architecture specification
- `helpful_tools.md` - Tool documentation
- `README_ENVIRONMENT.md` - Environment setup guide
- `requirements.txt` - Python dependencies
- `verify_gpu.py` - GPU verification script
- `shrimp_data/` - Task manager data
- Python implementation files (when created in future months)

### File Naming Conventions

**Research Papers (PDF):**
- **ArXiv papers:** `{arxiv_number}{version}.pdf` (e.g., `1803.05355v3.pdf`, `2310.11511v1.pdf`)
- **Other papers:** `{Paper_Title_With_Underscores}.pdf` (e.g., `A survey in hallucination of large language model.pdf`)
- **Lookup:** Check `Hallucination_Project_Details.md` Section 4 for arxiv number mappings

**Paper Summaries:**
- **Format:** `{Paper_Title_With_Underscores}_Summary.txt`
- **Example:** `SelfCheckGPT_Zero-Resource_Black-Box_Hallucination_Detection_for_Generative_Large_Language_Models_Summary.txt`
- **Location:** `reference/` directory
- **Encoding:** UTF-8

**Python Files (Future):**
- Use snake_case: `dense_retriever.py`, `nli_detector.py`, `evidence_coverage.py`

---

## 3. Documentation Workflow

### Paper Summarization Process

**WHEN ASKED TO REVIEW/SUMMARIZE A PAPER:**

1. **MUST use the PICO/T framework** specified in `.github/copilot-instructions.md`:
   - **P (Problem/Population):** Core problem or population studied
   - **I (Intervention/Interest):** New method, intervention, or technique
   - **C (Comparison):** What is it being compared to (if any)
   - **O (Outcome):** Primary outcomes measured
   - **T (Timeframe/Theory):** Core theoretical hypothesis or final argument

2. **MUST include these sections:**
   ```
   1. Core Research Question (PICO/T)
   2. Methodology
   3. Key Findings (5-10 bullet points)
   4. Main Contribution
   5. Limitations (if any)
   6. Keywords (5 keywords)
   7. Relevance Assessment (High/Medium/Low with explanation)
   ```

3. **Output format:**
   - Save as `.txt` file in `reference/` directory
   - Use clear section headers with `##` markdown
   - Keep findings concise and specific (NOT "the paper proposes a method" BUT "proposes Chain-of-Verification: LLM generates answer → generates verification questions → answers questions → revises original answer")

**DO:**
- ✓ Extract specific method names, algorithms, and mathematical formulations
- ✓ Include dataset names, model architectures, and evaluation metrics
- ✓ State concrete results (e.g., "achieves 87.3% accuracy on FEVER benchmark")
- ✓ Identify connections to other papers in `reference/` directory

**DON'T:**
- ✗ Provide vague summaries like "introduces a novel approach"
- ✗ Omit quantitative results
- ✗ Skip the relevance assessment

### TODO List Management

**CRITICAL RULE: MUST update `TODO_List.md` when completing tasks**

**Markdown Checkbox Syntax:**
- Incomplete: `- [ ] Task description`
- Complete: `- [x] Task description`

**Update Process:**
1. After completing ANY task mentioned in `TODO_List.md`
2. Search for the task description in the file
3. Replace `[ ]` with `[x]`
4. DO NOT modify task descriptions or add new content without explicit request

**Example:**
```markdown
# Before
-   [ ] Read and summarize key papers on trainless hallucination detection

# After  
-   [x] Read and summarize key papers on trainless hallucination detection
```

### Multi-File Synchronization Requirements

**When modifying documentation, check if related files need updates:**

| Primary File Modified | Must Check/Update |
|----------------------|-------------------|
| `Hallucination_Project_Details.md` | `System_Architecture_Design.md`, `TODO_List.md` |
| `System_Architecture_Design.md` | `helpful_tools.md` (if tools/libraries change) |
| `requirements.txt` | `README_ENVIRONMENT.md` (installation instructions) |
| Any research implementation | `TODO_List.md` (mark milestones complete) |

---

## 4. MCP Tool Usage Mandates

### REQUIRED Tools for Code Generation

**MUST ALWAYS USE (in this order):**

1. **context7 MCP (`context7`):**
   - **When:** Before implementing ANY library/framework code
   - **Why:** Get latest documentation, avoid deprecated methods
   - **Example:** Before using transformers API, call `mcp_context7_resolve-library-id` then `mcp_context7_get-library-docs`

2. **sequential-thinking MCP (`sequentialthinking`):**
   - **When:** For ALL tasks requiring multi-step reasoning
   - **Why:** Break down complex tasks, verify logic
   - **Example:** Planning verifier module implementation, analyzing paper connections

3. **shrimp-task-manager MCP (`shrimp-task-manager`):**
   - **When:** For multi-step implementation tasks
   - **Why:** Track progress, manage dependencies, ensure systematic completion
   - **Example:** Implementing Month 3 verifier signals (intrinsic uncertainty + heuristics)

4. **mcp-feedback MCP (`mcp-feedback_enhanced`):**
   - **When:** Before completing ANY task
   - **Why:** Validate work, get user confirmation, ensure requirements met
   - **Example:** After generating paper summary, before marking task complete

### Decision Tree: Which Tool When?

```
User Request
├─ "Review/summarize paper"
│  └─ Read PDF → PICO/T analysis → Create *_Summary.txt → Update TODO → mcp-feedback
│
├─ "Implement module/feature"
│  └─ sequential-thinking → shrimp-task-manager (plan) → context7 (docs) → Code → mcp-feedback
│
├─ "Answer question about project"
│  └─ Read Hallucination_Project_Details.md → Check reference/ summaries → Answer (NO SPECULATION)
│
├─ "Find information on topic X"
│  └─ Check reference/ summaries FIRST → If not found, web search → Cite sources
│
└─ "Setup/environment issue"
   └─ Read README_ENVIRONMENT.md → Check requirements.txt → run verify_gpu.py → Troubleshoot
```

### Hugging Face Tools Usage

**WHEN TO USE `mcp_huggingface_*` tools:**
- Searching for models: `mcp_huggingface_model_search`
- Searching for datasets: `mcp_huggingface_dataset_search`
- Searching for papers: `mcp_huggingface_paper_search`
- Documentation lookup: `mcp_huggingface_hf_doc_search`

**EXAMPLE (Finding NLI model):**
```
1. mcp_huggingface_model_search with query "NLI MNLI FEVER"
2. Filter for task="text-classification"
3. Check model card for zero-shot capabilities
4. Document in helpful_tools.md if relevant
```

---

## 5. Information Gathering Hierarchy

### DECISION TREE: Where to Find Information

**STEP 1: Check Local Project Files (ALWAYS FIRST)**
```
Question about...                  → Check file...
├─ Project goals/timeline          → Hallucination_Project_Details.md
├─ System architecture             → System_Architecture_Design.md
├─ What to do next                 → TODO_List.md
├─ Environment setup               → README_ENVIRONMENT.md
├─ Python dependencies             → requirements.txt
├─ GPU/CUDA status                 → Run verify_gpu.py
├─ Available tools                 → helpful_tools.md
└─ Research context                → reference/*_Summary.txt files
```

**STEP 2: If Not Found Locally → Check Research Papers**
```
1. Search reference/*_Summary.txt files (faster)
2. If insufficient, read relevant reference/*.pdf directly
3. ALWAYS cite specific paper when answering
```

**STEP 3: If Not Found in Papers → Web Search**
```
1. Use web search ONLY when local sources exhausted
2. Verify information against multiple sources
3. ALWAYS cite URLs
```

**PROHIBITED:**
- ✗ Guessing or speculating when information is unavailable
- ✗ Providing outdated library syntax without checking context7
- ✗ Inventing paper citations or results
- ✗ Ignoring existing summaries in reference/ directory

---

## 6. Code Implementation Standards

### Environment Management

**Before ANY Python code execution:**
1. **MUST verify environment:** Run `verify_gpu.py` if GPU-related work
2. **MUST check dependencies:** Read `requirements.txt` for available libraries
3. **MUST use venv:** Assume `.venv` is active (installation uses venv paths)

**GPU-Specific Rules:**
- **ALWAYS check CUDA availability** before GPU operations: `torch.cuda.is_available()`
- **Default device:** Set `device = 'cuda' if torch.cuda.is_available() else 'cpu'`
- **Memory management:** For RTX 3070Ti (8GB VRAM), use batch_size ≤ 16 for transformers models

### Library Version Constraints

**CRITICAL: These versions are FIXED in requirements.txt:**
```python
transformers == 4.56.2
torch == 2.5.1+cu121
datasets == 4.1.1
sentence-transformers == 5.1.1
faiss-cpu == 1.12.0
```

**Before using any method/class:**
1. Use `mcp_context7_get-library-docs` to verify API for EXACT version
2. DO NOT assume API from generic LLM knowledge (may be outdated)

### Code Structure for Verifier Signals

**When implementing detector modules (Month 3-4), MUST follow:**

```python
# DO: Modular signal classes
class IntrinsicUncertaintyDetector:
    """Calculates entropy-based uncertainty from generator output."""
    def __init__(self, config: dict):
        self.config = config
    
    def compute_signal(self, claim: str, evidence: str, metadata: dict) -> dict:
        """Returns: {'entropy_score': float, 'perplexity': float}"""
        pass

# DON'T: Monolithic functions
def calculate_everything(claim, evidence):  # ✗ Too broad
    pass
```

**Interface Requirements:**
- Each detector MUST return `dict` with named scores
- Each detector MUST accept `(claim, evidence, metadata)` parameters
- Each detector MUST include docstring with input/output specs

### Testing & Verification

**Before considering implementation complete:**
1. **Test on sample data** from benchmarks (TruthfulQA/RAGTruth/FEVER)
2. **Verify output format** matches `System_Architecture_Design.md` data structures
3. **Check GPU memory usage** if using transformers models
4. **Run through mcp-feedback** for validation

---

## 7. Prohibited Actions

### STRICTLY FORBIDDEN

**1. Speculation & Assumptions**
- ✗ Guessing paper results without reading
- ✗ Assuming dataset characteristics without verification
- ✗ Inventing method names or technical details
- ✗ Providing "typical" implementation without project context

**CORRECT APPROACH:**
- ✓ "Let me check the paper in reference/ directory"
- ✓ "I'll search for this information using available tools"
- ✓ "Based on Hallucination_Project_Details.md Section X..."

**2. Including General Development Knowledge**
- ✗ Explaining what "transformers" library does
- ✗ Describing general Python virtual environment concepts
- ✗ Defining what "hallucination" means in LLMs (project docs already cover this)
- ✗ Generic "best practices" not specific to this project

**CORRECT APPROACH:**
- ✓ "For THIS project's NLI model (`MoritzLaurer/DeBERTa-v3-base-mnli-fever-anli`), use..."
- ✓ "According to System_Architecture_Design.md, the verifier MUST..."
- ✓ "The project's trainless approach requires..."

**3. Ignoring Project Timeline**
- ✗ Suggesting Month 4 implementations when in Month 1
- ✗ Proposing trained models when project is in "trainless phase"
- ✗ Implementing features not in TODO_List.md without approval

**CORRECT APPROACH:**
- ✓ Check TODO_List.md for current phase
- ✓ "This is a Month 5 task, currently we're in Month X"
- ✓ "The trainless phase doesn't include fine-tuning"

**4. Breaking File System Rules**
- ✗ Modifying `.github/copilot-instructions.md` without explicit request
- ✗ Editing PDF files in reference/
- ✗ Renaming project structure files
- ✗ Creating files outside designated writable locations

**5. Skipping Required MCP Tools**
- ✗ Generating code without checking context7 for latest API
- ✗ Completing tasks without mcp-feedback validation
- ✗ Implementing complex features without sequential-thinking breakdown
- ✗ Starting multi-step work without shrimp-task-manager planning

---

## 8. Quality Checkpoints

### Before Marking ANY Task Complete

**MANDATORY VERIFICATION CHECKLIST:**

- [ ] **Primary deliverable created** (summary file, code module, documentation)
- [ ] **TODO_List.md updated** (checkbox marked if applicable)
- [ ] **Multi-file consistency checked** (related docs updated if needed)
- [ ] **Naming conventions followed** (file names match project standards)
- [ ] **MCP tools used correctly** (context7 for APIs, feedback for validation)
- [ ] **No speculation or assumptions** (all information sourced from project files or verified externally)
- [ ] **Citations provided** (papers, URLs, or file paths referenced)
- [ ] **mcp-feedback-enhanced called** (user validation obtained)

### Self-Review Questions

**Before submitting work, ask:**
1. "Did I check the project documentation FIRST before searching externally?"
2. "Did I use the EXACT library versions specified in requirements.txt?"
3. "Did I update TODO_List.md if I completed a listed task?"
4. "Did I provide SPECIFIC information (method names, metrics, citations) instead of vague descriptions?"
5. "Did I use imperative language in documentation (MUST/DO/DON'T) instead of suggestions?"
6. "Did I follow the PICO/T format for paper summaries?"
7. "Did I call mcp-feedback before claiming task completion?"

---

## 9. Research-Specific Workflows

### Literature Review Process (Month 1)

**WHEN ASKED: "Review papers on [topic]"**

**STEP-BY-STEP:**
1. **Identify relevant papers** in `reference/` directory
   - Check `Hallucination_Project_Details.md` Section 4 for paper list
   - Search by arxiv number or title keywords
2. **Read existing summaries** (*_Summary.txt) if available
3. **If summary missing:** Create new summary using PICO/T format
4. **Cross-reference findings** with System_Architecture_Design.md
5. **Update TODO_List.md** (mark literature review tasks complete)
6. **Call mcp-feedback** with summary of findings

### Implementation Planning (Month 2-6)

**WHEN ASKED: "Implement [feature/module]"**

**STEP-BY-STEP:**
1. **Check current phase:** Read TODO_List.md to verify this is the right month
2. **Use sequential-thinking:** Break down implementation into steps
3. **Use shrimp-task-manager:** Create task plan with dependencies
4. **Check architecture:** Verify design in System_Architecture_Design.md
5. **Get latest API docs:** Use context7 for transformers/torch/faiss APIs
6. **Implement with verification:** Test with sample data after each component
7. **Update documentation:** Add implementation notes to relevant .md files
8. **Update TODO_List.md:** Mark implementation tasks complete
9. **Call mcp-feedback:** Validate implementation before moving forward

### Data Sourcing Process (Month 1-2)

**WHEN ASKED: "Download [dataset]"**

**STEP-BY-STEP:**
1. **Verify dataset requirement:** Check Hallucination_Project_Details.md Section 3
2. **Use Hugging Face tools:** `mcp_huggingface_dataset_search` for datasets on HF Hub
3. **Check storage location:** Confirm where data should be stored (future: likely `data/` directory)
4. **Document dataset:** Update helpful_tools.md with dataset info if significant
5. **Update TODO_List.md:** Mark data sourcing task complete

---

## 10. Examples: DO vs DON'T

### Example 1: Answering Questions About Papers

**USER: "What method does SelfCheckGPT use?"**

**DON'T:**
```
SelfCheckGPT uses a self-consistency approach to detect hallucinations.
```
(Too vague, no citation, no specifics)

**DO:**
```
According to `SelfCheckGPT_Zero-Resource_Black-Box_Hallucination_Detection_for_Generative_Large_Language_Models_Summary.txt`:

SelfCheckGPT uses **self-consistency sampling**: generates k different responses to the same prompt with stochastic sampling (temperature > 0), then measures semantic consistency across responses. High variance indicates potential hallucination. The method includes three variants:
1. BERTScore-based consistency
2. Question-answering consistency  
3. N-gram overlap consistency

This is relevant to our Month 4 implementation of the Self-Agreement detector module.
```

### Example 2: Implementing Code

**USER: "Create the intrinsic uncertainty detector"**

**DON'T:**
```python
# Just use transformers like this:
from transformers import AutoModel
model = AutoModel.from_pretrained('bert-base')
# ... (generic code without project context)
```

**DO:**
```
I'll implement this following the Month 3 TODO and System_Architecture_Design.md specs.

First, let me verify the current transformers API for version 4.56.2:
[Call mcp_context7_get-library-docs with transformers]

Then break down implementation:
[Call mcp_sequentialthi_sequentialthinking to plan steps]

Implementation will:
1. Accept (claim, evidence, generator_metadata) as input
2. Calculate token-level entropy from logits in metadata
3. Return {'entropy_score': float, 'perplexity': float} dict
4. Follow the modular class structure specified in shrimp-rules.md Section 6

[Provide actual implementation]
[Test with sample data]
[Call mcp-feedback for validation]
```

### Example 3: Task Completion

**USER: "Did you finish summarizing the FEVER paper?"**

**DON'T:**
```
Yes, I've finished the summary.
```
(No evidence, no verification, TODO not updated)

**DO:**
```
Yes, I've completed the FEVER paper summary:

1. ✓ Created `FEVER_a_large-scale_dataset_for_Fact_Extraction_and_VERification_Summary.txt` in `reference/` directory
2. ✓ Used PICO/T framework as specified in .github/copilot-instructions.md
3. ✓ Included 7 key findings with specific metrics (e.g., 185,000 claims annotated)
4. ✓ Assessed relevance as "High" - directly applicable to Month 4 NLI detector
5. ✓ Updated TODO_List.md: Marked "Read and summarize key papers..." as [x]

Let me verify the work is complete using mcp-feedback:
[Call mcp_mcp-feedback-_interactive_feedback]
```

---

## 11. Project-Specific Terminology

**USE THESE EXACT TERMS (consistency with project docs):**

| Term | Meaning in THIS Project | DON'T Say |
|------|------------------------|-----------|
| **Trainless** | Zero-shot, no fine-tuning approach | "Unsupervised" or "Pre-trained" |
| **Verifier Module** | The 4-signal hallucination detector | "Validator" or "Checker" |
| **Signal** | Individual detector output (entropy, NLI, etc.) | "Feature" or "Metric" |
| **Claim** | Atomic verifiable statement from LLM output | "Sentence" or "Statement" |
| **Evidence** | Retrieved document/chunk from knowledge corpus | "Context" or "Document" |
| **Generator** | LLM that produces draft responses | "Model" or "LLM" (be specific) |
| **Retriever** | Dense retriever for evidence fetching | "Search" or "IR system" |
| **Baseline RAG** | Simple retriever + generator pipeline | "RAG system" (this is THE baseline) |
| **Mitigation** | Flagging & suppression of low-confidence claims | "Post-processing" or "Correction" |

---

## 12. Integration with Existing Instructions

**RELATIONSHIP WITH `.github/copilot-instructions.md`:**

- `.github/copilot-instructions.md` → **Master instructions** (general AI agent guidance)
- `shrimp-rules.md` (THIS FILE) → **Operational rules** (project-specific workflows)

**CONFLICT RESOLUTION:**
- If instructions conflict, `.github/copilot-instructions.md` takes precedence
- For implementation details not covered in `.github/copilot-instructions.md`, follow `shrimp-rules.md`
- For architecture/design questions, check `System_Architecture_Design.md` FIRST

**ALWAYS REFERENCE BOTH:**
- When starting work session: Read `.github/copilot-instructions.md` for context
- When executing specific tasks: Follow workflows in `shrimp-rules.md`
- When in doubt: Check both files + relevant project documentation

---

## Document Version

- **Created:** 2025-10-08
- **Last Updated:** 2025-10-08
- **Version:** 1.0
- **Target Agent:** AI coding assistants (GitHub Copilot, Claude, etc.)
- **Maintenance:** Update when project structure or workflows change

---

**END OF DEVELOPMENT GUIDELINES**
