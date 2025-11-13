"""
Smoke test for verifier integration into baseline RAG pipeline.

Tests both enabled and disabled verification modes to ensure backward compatibility
and proper verifier signal computation.
"""

import sys
import os
import json
from pathlib import Path

# Add project root to path
sys.path.insert(0, os.path.abspath(os.path.join(os.path.dirname(__file__), '..')))

from src.utils.config import Config
from src.utils.data_structures import Claim, EvidenceChunk, VerifierSignal
from src.pipelines.baseline_rag import BaselineRAGPipeline
from src.retrieval.dense_retriever import DenseRetriever
from src.generation.generator_wrapper import GeneratorWrapper
from unittest.mock import Mock, MagicMock


def create_mock_retriever():
    """Create a mock retriever that returns sample evidence."""
    mock_retriever = Mock(spec=DenseRetriever)
    
    def mock_retrieve(query, top_k=5):
        """Return mock evidence chunks."""
        return [
            EvidenceChunk(
                doc_id="test_doc_1",
                sent_id=0,
                text="Barack Obama was the 44th president of the United States.",
                char_start=0,
                char_end=59,
                score_dense=0.95,
                rank=1
            ),
            EvidenceChunk(
                doc_id="test_doc_1",
                sent_id=1,
                text="He served from 2009 to 2017.",
                char_start=60,
                char_end=88,
                score_dense=0.88,
                rank=2
            )
        ]
    
    mock_retriever.retrieve = mock_retrieve
    return mock_retriever


def create_mock_generator():
    """Create a mock generator that returns sample output."""
    mock_generator = Mock(spec=GeneratorWrapper)
    
    def mock_generate(prompt, evidence_chunks, **kwargs):
        """Return mock generation output with metadata."""
        text = "Barack Obama was president of the United States."
        tokens = ["Barack", "Obama", "was", "president", "of", "the", "United", "States", "."]
        
        # Mock logits (9 tokens, vocab_size=100)
        import numpy as np
        logits = []
        for i in range(len(tokens)):
            # Create peaked distribution (confident predictions)
            token_logits = np.random.randn(100) - 5.0  # Low baseline
            token_logits[i * 10] = 5.0  # Peak at different positions
            logits.append(token_logits.tolist())
        
        return {
            'text': text,
            'tokens': tokens,
            'logits': logits,
            'token_scores': [0.99] * len(tokens)
        }
    
    mock_generator.generate_with_metadata = mock_generate
    return mock_generator


def test_pipeline_verification_disabled():
    """Test pipeline with verification disabled (backward compatibility)."""
    print("\n" + "="*60)
    print("Test 1: Verification Disabled (Backward Compatibility)")
    print("="*60)
    
    # Create config with verification disabled
    config = Config()
    config.verification.enabled = False
    
    # Create pipeline with mocks
    mock_retriever = create_mock_retriever()
    mock_generator = create_mock_generator()
    
    pipeline = BaselineRAGPipeline(
        retriever=mock_retriever,
        generator=mock_generator,
        config=config
    )
    
    # Verify detectors not initialized
    assert pipeline.verifier_enabled == False, "verifier_enabled should be False"
    assert not hasattr(pipeline, 'uncertainty_detector') or pipeline.uncertainty_detector is None
    print("✓ Detectors not initialized when disabled")
    
    # Run pipeline
    result = pipeline.run("Who was Barack Obama?", top_k=2)
    
    # Verify output structure (Month 2 format)
    assert 'query' in result
    assert 'draft_response' in result
    assert 'claim_evidence_pairs' in result
    assert 'generator_metadata' in result
    assert 'retrieval_metadata' in result
    print("✓ Standard output fields present")
    
    # Verify NO verifier_signals in output
    assert 'verifier_signals' not in result, "verifier_signals should not be in output when disabled"
    print("✓ No verifier_signals in output (backward compatible)")
    
    print("\n✓ Test 1 PASSED: Backward compatibility maintained")


def test_pipeline_verification_enabled():
    """Test pipeline with verification enabled."""
    print("\n" + "="*60)
    print("Test 2: Verification Enabled")
    print("="*60)
    
    # Create config and enable verification
    config = Config()
    # Modify internal config dict to enable verification
    config._config['verification']['enabled'] = True
    
    # Create pipeline with mocks
    mock_retriever = create_mock_retriever()
    mock_generator = create_mock_generator()
    
    pipeline = BaselineRAGPipeline(
        retriever=mock_retriever,
        generator=mock_generator,
        config=config
    )
    
    # Verify detectors initialized
    assert pipeline.verifier_enabled == True, "verifier_enabled should be True"
    assert hasattr(pipeline, 'uncertainty_detector'), "uncertainty_detector should exist"
    assert hasattr(pipeline, 'grounded_detector'), "grounded_detector should exist"
    print("✓ Detectors initialized when enabled")
    
    # Run pipeline
    result = pipeline.run("Who was Barack Obama?", top_k=2)
    
    # Verify standard output fields
    assert 'query' in result
    assert 'draft_response' in result
    assert 'claim_evidence_pairs' in result
    print("✓ Standard output fields present")
    
    # Verify verifier_signals in output
    assert 'verifier_signals' in result, "verifier_signals should be in output when enabled"
    verifier_signals = result['verifier_signals']
    
    # Verify signals structure
    assert isinstance(verifier_signals, list), "verifier_signals should be a list"
    assert len(verifier_signals) > 0, "Should have at least one verifier signal"
    print(f"✓ Generated {len(verifier_signals)} verifier signal(s)")
    
    # Verify signal format
    signal = verifier_signals[0]
    required_fields = [
        'claim_id', 'doc_id', 'sent_id', 'nli', 'coverage',
        'uncertainty', 'consistency', 'citation_span_match', 'numeric_check'
    ]
    
    for field in required_fields:
        assert field in signal, f"Signal missing required field: {field}"
    print("✓ All required fields present in VerifierSignal")
    
    # Verify field types and values
    assert isinstance(signal['claim_id'], str)
    assert isinstance(signal['doc_id'], str)
    assert isinstance(signal['sent_id'], int)
    assert signal['nli'] is None, "nli should be None for Month 3"
    assert isinstance(signal['coverage'], dict)
    assert isinstance(signal['uncertainty'], dict)
    assert signal['consistency'] == {'variance': None}, "consistency should be {'variance': None} for Month 3"
    assert isinstance(signal['citation_span_match'], float)
    assert isinstance(signal['numeric_check'], bool)
    print("✓ Signal field types and values correct")
    
    # Verify coverage fields
    assert 'entities' in signal['coverage']
    assert 'numbers' in signal['coverage']
    assert 'tokens_overlap' in signal['coverage']
    print("✓ Coverage signal has required fields")
    
    # Verify uncertainty fields
    assert 'mean_entropy' in signal['uncertainty']
    print("✓ Uncertainty signal has required fields")
    
    # Print sample signal for inspection
    print("\n--- Sample VerifierSignal ---")
    print(json.dumps(signal, indent=2))
    
    print("\n✓ Test 2 PASSED: Verification integration working")


def test_pipeline_no_evidence():
    """Test pipeline with verification enabled but no evidence retrieved."""
    print("\n" + "="*60)
    print("Test 3: Verification Enabled with No Evidence")
    print("="*60)
    
    # Create config and enable verification
    config = Config()
    config._config['verification']['enabled'] = True
    
    # Create mock retriever that returns empty evidence
    mock_retriever = Mock(spec=DenseRetriever)
    mock_retriever.retrieve = lambda query, top_k=5: []
    
    mock_generator = create_mock_generator()
    
    pipeline = BaselineRAGPipeline(
        retriever=mock_retriever,
        generator=mock_generator,
        config=config
    )
    
    # Run pipeline
    result = pipeline.run("Test query", top_k=2)
    
    # Should have standard fields
    assert 'query' in result
    assert 'draft_response' in result
    
    # Should NOT have verifier_signals (no evidence to verify against)
    # or should have empty list
    if 'verifier_signals' in result:
        assert len(result['verifier_signals']) == 0, "verifier_signals should be empty with no evidence"
    
    print("✓ Handled no evidence case gracefully")
    print("\n✓ Test 3 PASSED: Edge case handled correctly")


if __name__ == "__main__":
    print("="*60)
    print("Baseline RAG Pipeline Integration Smoke Test")
    print("="*60)
    
    try:
        test_pipeline_verification_disabled()
        test_pipeline_verification_enabled()
        test_pipeline_no_evidence()
        
        print("\n" + "="*60)
        print("✓ ALL SMOKE TESTS PASSED!")
        print("="*60)
    except AssertionError as e:
        print(f"\n✗ Test failed: {e}")
        import traceback
        traceback.print_exc()
        sys.exit(1)
    except Exception as e:
        print(f"\n✗ Unexpected error: {e}")
        import traceback
        traceback.print_exc()
        sys.exit(1)
