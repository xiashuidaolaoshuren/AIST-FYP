# Test Suite Organization

This directory contains all tests for the AIST-FYP project, organized into **unit tests** and **integration tests**.

## Directory Structure

```
tests/
├── unit/                       # Unit tests (fast, isolated)
│   ├── test_intrinsic_uncertainty.py
│   ├── test_retrieval_grounded.py
│   ├── test_retrieval.py
│   ├── test_generation.py
│   ├── test_utils.py
│   ├── test_data_processing.py
│   ├── test_embedding_generator.py
│   ├── test_faiss_index_manager.py
│   └── test_index_search.py
├── integration/                # Integration tests (slower, requires external dependencies)
│   ├── test_verifier_integration.py
│   ├── test_end_to_end.py
│   └── test_pipeline.py
└── fixtures/                   # Shared test fixtures and data
```

---

## Unit Tests (`tests/unit/`)

**Purpose:** Test individual components in isolation with minimal dependencies.

**Characteristics:**
- ✅ Fast execution (< 1 second per test)
- ✅ No external dependencies (FAISS index, API keys, etc.)
- ✅ Use mocked data and fixtures
- ✅ Focus on single function/class behavior
- ✅ Can run independently in any order

**Files:**

### Month 3 Verifier Module
- **`test_intrinsic_uncertainty.py`** (361 lines, 12 tests, 81% coverage)
  - Tests `IntrinsicUncertaintyDetector` class
  - Entropy calculations, token-claim alignment, edge cases
  - Numerical stability (log-sum-exp trick, epsilon handling)

- **`test_retrieval_grounded.py`** (342 lines, 14 tests, 84% coverage)
  - Tests `RetrievalGroundedDetector` class
  - Entity extraction (spaCy NER), fuzzy matching, number coverage
  - Token overlap (ROUGE-L), spaCy model singleton pattern

### Month 2 Core Components
- **`test_retrieval.py`** (232 lines)
  - Tests `DenseRetriever` class
  - Query encoding, FAISS search, EvidenceChunk creation
  - Ranking and scoring logic

- **`test_generation.py`** (339 lines)
  - Tests `GeneratorWrapper` and claim extraction
  - `extract_claims()`, `extract_claims_spacy()`, `extract_claims_regex()`
  - Claim validation and span checking

- **`test_utils.py`**
  - Tests utility functions and data structures
  - Configuration loading, helper functions

### Data Processing Pipeline
- **`test_data_processing.py`**
  - Tests Wikipedia data processing
  - Text cleaning, chunking, preprocessing

- **`test_embedding_generator.py`**
  - Tests embedding generation
  - Sentence transformer integration, batch processing

- **`test_faiss_index_manager.py`**
  - Tests FAISS index building and management
  - Index creation, saving, loading

- **`test_index_search.py`**
  - Tests FAISS search functionality
  - Query encoding, similarity search

---

## Integration Tests (`tests/integration/`)

**Purpose:** Validate that multiple components work together correctly in realistic scenarios.

**Characteristics:**
- ⚠️ Slower execution (1-10 seconds per test)
- ⚠️ Requires external dependencies (FAISS index, models, etc.)
- ⚠️ Tests component interactions
- ⚠️ May require specific setup/teardown
- ✅ Validates end-to-end workflows

**Files:**

- **`test_verifier_integration.py`** (341 lines, 10 tests)
  - **Purpose:** End-to-end verifier pipeline integration in `baseline_rag.py`
  - **Tests:**
    - Verifier enabled/disabled (backward compatibility)
    - VerifierSignal structure and format
    - Performance overhead (<100ms requirement)
    - Multiple claims handling
    - Error handling with malformed data
  - **Requirements:** Config with verification settings, spaCy model

- **`test_end_to_end.py`** (366 lines)
  - **Purpose:** Complete RAG system integration
  - **Tests:**
    - Full pipeline from query to answer with verification
    - Claim extraction integration with generated text
    - Evidence retrieval → Generation → Verification flow
    - Output format validation for Month 3 verifier
  - **Requirements:** FAISS index, generator model, retriever

- **`test_pipeline.py`** (181 lines)
  - **Purpose:** Baseline RAG pipeline integration
  - **Tests:**
    - Retriever + Generator integration
    - Query processing and answer generation
    - Evidence formatting and ranking
  - **Requirements:** FAISS index, generator model

---

## Running Tests

### Run All Tests
```bash
pytest tests/
```

### Run Unit Tests Only (Fast)
```bash
pytest tests/unit/
```

### Run Integration Tests Only
```bash
pytest tests/integration/
```

### Run Specific Test File
```bash
# Unit test
pytest tests/unit/test_intrinsic_uncertainty.py

# Integration test
pytest tests/integration/test_verifier_integration.py
```

### Run with Coverage
```bash
# All tests with coverage
pytest tests/ --cov=src --cov-report=html

# Unit tests with coverage
pytest tests/unit/ --cov=src --cov-report=term-missing
```

### Run with Markers
```bash
# Run only integration tests (if marked with @pytest.mark.integration)
pytest -m integration

# Skip integration tests
pytest -m "not integration"
```

---

## Test Statistics

### Month 3 Test Coverage Summary

| Component | Tests | Coverage | Lines |
|-----------|-------|----------|-------|
| IntrinsicUncertaintyDetector | 12 | 81% | 361 |
| RetrievalGroundedDetector | 14 | 84% | 342 |
| Verifier Integration | 10 | 100% | 341 |
| **Total Month 3** | **36** | **83%** | **1044** |

### Overall Test Metrics

- **Total Test Files:** 12 (9 unit + 3 integration)
- **Total Test Cases:** 60+ (estimated)
- **Unit Test Execution Time:** ~5 seconds
- **Integration Test Execution Time:** ~15 seconds
- **Code Coverage:** 83% (Month 3 components)

---

## Writing New Tests

### Unit Test Template

```python
"""
Unit tests for [Component Name].

Brief description of what is being tested.
"""

import pytest
import sys
from pathlib import Path

sys.path.insert(0, str(Path(__file__).parent.parent.parent))

from src.module.component import ComponentClass


class TestComponentClass:
    """Test suite for ComponentClass."""
    
    @pytest.fixture
    def sample_config(self):
        """Create test configuration."""
        # Setup code
        return config
    
    def test_basic_functionality(self, sample_config):
        """Test basic functionality."""
        # Arrange
        component = ComponentClass(sample_config)
        
        # Act
        result = component.method(input_data)
        
        # Assert
        assert result == expected_output
```

### Integration Test Template

```python
"""
Integration tests for [System Component].

Tests multi-component interactions and end-to-end workflows.
"""

import pytest
import sys
from pathlib import Path

sys.path.insert(0, str(Path(__file__).parent.parent.parent))


@pytest.mark.integration
class TestSystemIntegration:
    """Test suite for system integration."""
    
    @pytest.fixture(scope="class")
    def system(self):
        """Create system instance with real dependencies."""
        # May require FAISS index, models, etc.
        try:
            system = System.from_config("config.yaml")
            return system
        except FileNotFoundError:
            pytest.skip("Required dependencies not found")
    
    def test_end_to_end_workflow(self, system):
        """Test complete workflow."""
        # Arrange
        input_data = "test query"
        
        # Act
        result = system.run(input_data)
        
        # Assert
        assert result is not None
        assert result.has_expected_structure()
```

---

## Best Practices

### Unit Tests
1. ✅ **Isolate dependencies** - Mock external services, use fixtures
2. ✅ **Test one thing** - Each test should verify a single behavior
3. ✅ **Fast execution** - Should complete in milliseconds
4. ✅ **Deterministic** - Same input = same output, always
5. ✅ **Clear naming** - `test_method_scenario_expectedOutcome`

### Integration Tests
1. ✅ **Test realistic scenarios** - Use actual components, not mocks
2. ✅ **Document requirements** - Note what external dependencies are needed
3. ✅ **Graceful failures** - Skip if dependencies unavailable (`pytest.skip()`)
4. ✅ **Performance checks** - Validate system meets performance requirements
5. ✅ **Clean up** - Ensure tests don't leave artifacts

### General Guidelines
1. ✅ **Use pytest fixtures** - Share setup code efficiently
2. ✅ **Parametrize tests** - Test multiple scenarios with `@pytest.mark.parametrize`
3. ✅ **Document edge cases** - Explain why specific test cases exist
4. ✅ **Measure coverage** - Aim for >80% on critical components
5. ✅ **Run before commit** - Always run tests before pushing code

---

## Troubleshooting

### Common Issues

**Issue:** `ModuleNotFoundError: No module named 'src'`
```bash
# Solution: Ensure project root is in PYTHONPATH
export PYTHONPATH="${PYTHONPATH}:$(pwd)"
# Or use the sys.path.insert() pattern in test files
```

**Issue:** Integration tests fail with "FAISS index not found"
```bash
# Solution: Build the FAISS index first
python -m src.data_processing.build_index --config config.yaml
```

**Issue:** spaCy model not found
```bash
# Solution: Download the required spaCy model
python -m spacy download en_core_web_sm
```

**Issue:** GPU/CUDA errors
```bash
# Solution: Set device to CPU in config or use CPU-only dependencies
# In config.yaml: device: 'cpu'
```

---

## CI/CD Integration

### GitHub Actions Example

```yaml
name: Tests

on: [push, pull_request]

jobs:
  test:
    runs-on: ubuntu-latest
    
    steps:
    - uses: actions/checkout@v2
    
    - name: Set up Python
      uses: actions/setup-python@v2
      with:
        python-version: '3.9'
    
    - name: Install dependencies
      run: |
        pip install -r requirements.txt
        python -m spacy download en_core_web_sm
    
    - name: Run unit tests
      run: pytest tests/unit/ --cov=src --cov-report=xml
    
    - name: Upload coverage
      uses: codecov/codecov-action@v2
      with:
        file: ./coverage.xml
```

---

## Contributing

When adding new tests:

1. **Determine test type:** Is it a unit test (isolated) or integration test (multi-component)?
2. **Place in correct directory:** `tests/unit/` or `tests/integration/`
3. **Follow naming convention:** `test_<component>_<scenario>.py`
4. **Add docstrings:** Explain what the test validates
5. **Update this README:** Add entry to the appropriate section
6. **Run coverage:** Ensure new code is adequately tested

---

## Related Documentation

- **Month 3 Verifier Documentation:** `docs/month3_verifier_part1.md`
- **Entity Normalization Challenge:** `docs/entity_normalization_challenge.md`
- **Project TODO List:** `TODO_List.md`
- **System Architecture:** `System_Architecture_Design.md`

---

**Last Updated:** 2025-11-23  
**Test Suite Version:** Month 3 Complete (v1.0)
