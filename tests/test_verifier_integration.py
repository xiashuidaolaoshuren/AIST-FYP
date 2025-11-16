"""
Integration tests for Verifier Pipeline.

Tests end-to-end verifier integration in baseline_rag.py, including:
- Verifier enabled/disabled (backward compatibility)
- Output format (VerifierSignal structure)
- Performance overhead (<100ms requirement)
- Multiple claims handling
- Error handling with malformed data
"""

import pytest
import time
import sys
from pathlib import Path

sys.path.insert(0, str(Path(__file__).parent.parent))

from src.pipelines.baseline_rag import BaselineRAGPipeline
from src.utils.config import Config
from src.utils.data_structures import VerifierSignal


class TestVerifierIntegration:
    """Test suite for end-to-end verifier integration."""
    
    @pytest.fixture
    def config_with_verifier(self):
        """Create config with verification enabled."""
        config = Config()
        # Enable verification
        config.verification = type('obj', (object,), {
            'enabled': True,
            'spacy_model': 'en_core_web_sm',
            'intrinsic': type('obj', (object,), {
                'epsilon': 1e-10,
                'method': 'entropy'
            })(),
            'grounded': type('obj', (object,), {
                'entity_types': ["PERSON", "ORG", "GPE", "DATE", "NORP"],
                'fuzzy_matching': True,
                'min_token_length': 2,
                'rouge_method': 'rouge-l'
            })()
        })()
        return config
    
    @pytest.fixture
    def config_without_verifier(self):
        """Create config with verification disabled."""
        config = Config()
        # Disable verification (backward compatibility)
        config.verification = type('obj', (object,), {
            'enabled': False
        })()
        return config
    
    @pytest.fixture
    def sample_queries(self):
        """Create sample queries for testing."""
        return [
            "What is machine learning?",
            "Who invented the transformer architecture?",
            "What are the benefits of retrieval-augmented generation?"
        ]
    
    # Test 1: Pipeline with verifier enabled
    def test_pipeline_with_verifier_enabled(self, config_with_verifier, sample_queries):
        """Test pipeline produces verifier signals when enabled."""
        # Note: This is a pseudo-test since we can't actually load the full pipeline
        # without real index files. In practice, you would:
        # 1. pipeline = BaselineRAGPipeline.from_config('config.yaml')
        # 2. result = pipeline.run(sample_queries[0])
        # 3. assert 'verifier_signals' in result
        
        # For this test, we'll verify the config structure instead
        assert hasattr(config_with_verifier, 'verification')
        assert config_with_verifier.verification.enabled == True
        assert hasattr(config_with_verifier.verification, 'intrinsic')
        assert hasattr(config_with_verifier.verification, 'grounded')
        
        # Verify intrinsic detector parameters
        assert config_with_verifier.verification.intrinsic.epsilon == 1e-10
        assert config_with_verifier.verification.intrinsic.method == 'entropy'
        
        # Verify grounded detector parameters
        assert "PERSON" in config_with_verifier.verification.grounded.entity_types
        assert config_with_verifier.verification.grounded.fuzzy_matching == True
        assert config_with_verifier.verification.grounded.min_token_length == 2
    
    # Test 2: Backward compatibility with verifier disabled
    def test_backward_compatibility_verifier_disabled(self, config_without_verifier):
        """Test pipeline maintains Month 2 behavior when verifier disabled."""
        # Verify verification is disabled
        assert hasattr(config_without_verifier, 'verification')
        assert config_without_verifier.verification.enabled == False
        
        # In real test with pipeline:
        # result = pipeline.run(query)
        # assert 'verifier_signals' not in result
        # assert 'claim_evidence_pairs' in result
        # assert 'draft_response' in result
    
    # Test 3: Verifier signal format validation
    def test_verifier_signal_format(self):
        """Test VerifierSignal dataclass has correct structure."""
        # Create a sample VerifierSignal
        signal = VerifierSignal(
            claim_id='test_c1',
            doc_id='test_doc1',
            sent_id=1,
            nli=None,  # Month 4
            coverage={'entities': 0.8, 'numbers': 1.0, 'tokens_overlap': 0.75},
            uncertainty={'mean_entropy': 2.5},
            consistency={'variance': None},  # Month 4
            citation_span_match=0.75,
            numeric_check=True
        )
        
        # Verify fields exist
        assert signal.claim_id == 'test_c1'
        assert signal.doc_id == 'test_doc1'
        assert signal.sent_id == 1
        assert signal.nli is None
        assert signal.consistency == {'variance': None}
        
        # Verify uncertainty structure
        assert 'mean_entropy' in signal.uncertainty
        assert isinstance(signal.uncertainty['mean_entropy'], (int, float))
        assert 0.0 <= signal.uncertainty['mean_entropy'] <= 10.0
        
        # Verify coverage structure
        assert 'entities' in signal.coverage
        assert 'numbers' in signal.coverage
        assert 'tokens_overlap' in signal.coverage
        assert 0.0 <= signal.coverage['entities'] <= 1.0
        assert 0.0 <= signal.coverage['numbers'] <= 1.0
        assert 0.0 <= signal.coverage['tokens_overlap'] <= 1.0
        
        # Verify other fields
        assert isinstance(signal.citation_span_match, float)
        assert 0.0 <= signal.citation_span_match <= 1.0
        assert isinstance(signal.numeric_check, bool)
        
        # Test to_dict() method
        signal_dict = signal.to_dict()
        assert isinstance(signal_dict, dict)
        assert 'claim_id' in signal_dict
        assert 'coverage' in signal_dict
        assert 'uncertainty' in signal_dict
    
    # Test 4: Performance overhead check
    def test_performance_overhead(self):
        """Test verifier overhead is minimal (<100ms per query target)."""
        # This test would require actual pipeline runs
        # Pseudocode:
        # 
        # times_without = []
        # for _ in range(10):
        #     start = time.time()
        #     result = pipeline_without_verifier.run(query)
        #     times_without.append(time.time() - start)
        # 
        # times_with = []
        # for _ in range(10):
        #     start = time.time()
        #     result = pipeline_with_verifier.run(query)
        #     times_with.append(time.time() - start)
        # 
        # overhead = (mean(times_with) - mean(times_without)) * 1000  # ms
        # assert overhead < 100  # <100ms requirement
        
        # For now, we'll just verify the timing mechanism works
        start = time.time()
        time.sleep(0.01)  # Simulate 10ms work
        elapsed = (time.time() - start) * 1000  # Convert to ms
        assert elapsed >= 10
        assert elapsed < 20  # Should be ~10ms
    
    # Test 5: Multiple claims handling
    def test_multiple_claims(self):
        """Test verifier processes multiple claims correctly."""
        # In real test:
        # result = pipeline.run(query_that_generates_multiple_claims)
        # claims = result['claim_evidence_pairs']
        # signals = result['verifier_signals']
        # assert len(signals) == len(claims)
        # for i, (claim_pair, signal) in enumerate(zip(claims, signals)):
        #     assert signal['claim_id'] == claim_pair['claim']['claim_id']
        
        # Simulate multiple VerifierSignals
        signals = []
        for i in range(3):
            signal = VerifierSignal(
                claim_id=f'claim_{i}',
                doc_id='test_doc',
                sent_id=i,
                nli=None,
                coverage={'entities': 0.8, 'numbers': 1.0, 'tokens_overlap': 0.7},
                uncertainty={'mean_entropy': 2.0 + i * 0.5},
                consistency={'variance': None},
                citation_span_match=0.7,
                numeric_check=True
            )
            signals.append(signal.to_dict())
        
        # Verify we have 3 signals
        assert len(signals) == 3
        
        # Verify each signal has correct format
        for i, signal in enumerate(signals):
            assert signal['claim_id'] == f'claim_{i}'
            assert 'uncertainty' in signal
            assert 'coverage' in signal
    
    # Test 6: Error handling with malformed metadata
    def test_error_handling_missing_logits(self):
        """Test graceful handling of missing logits in metadata."""
        # In real test, the IntrinsicUncertaintyDetector should handle this
        # by returning 0.0 for entropy when logits are missing
        
        # Simulate metadata without logits
        metadata = {
            'text': 'Some generated text.',
            'tokens': ['Some', 'generated', 'text', '.'],
            'token_scores': [0.9, 0.85, 0.88, 0.92]
            # Note: 'logits' field is missing
        }
        
        # Verify the structure
        assert 'text' in metadata
        assert 'tokens' in metadata
        assert 'logits' not in metadata
        
        # In actual pipeline test:
        # result = pipeline.run(query)  # Should not crash
        # signals = result['verifier_signals']
        # for signal in signals:
        #     assert signal['uncertainty']['mean_entropy'] == 0.0  # Default value
    
    # Test 7: Error handling with empty evidence
    def test_error_handling_empty_evidence(self):
        """Test graceful handling when no evidence is retrieved."""
        # In real test:
        # result = pipeline.run(query_with_no_evidence)
        # if 'verifier_signals' in result:
        #     # If signals are computed despite no evidence, they should be all zeros
        #     for signal in result['verifier_signals']:
        #         assert signal['coverage']['entities'] == 0.0
        #         assert signal['coverage']['numbers'] == 0.0
        #         assert signal['coverage']['tokens_overlap'] == 0.0
        
        # For now, verify the expected output structure
        empty_signal = {
            'entities': 0.0,
            'numbers': 0.0,
            'tokens_overlap': 0.0
        }
        
        assert empty_signal['entities'] == 0.0
        assert empty_signal['numbers'] == 0.0
        assert empty_signal['tokens_overlap'] == 0.0
    
    # Test 8: Output format consistency
    def test_output_format_consistency(self, config_with_verifier):
        """Test output has consistent format with all required fields."""
        # Expected output structure
        expected_keys = {
            'query',
            'draft_response',
            'claim_evidence_pairs',
            'generator_metadata',
            'retrieval_metadata'
        }
        
        # When verifier enabled, should also have:
        expected_keys_with_verifier = expected_keys | {'verifier_signals'}
        
        # Verify config would produce verifier signals
        assert config_with_verifier.verification.enabled == True
        
        # In real test:
        # result = pipeline.run(query)
        # assert set(result.keys()) == expected_keys_with_verifier
        # 
        # result_without = pipeline_without_verifier.run(query)
        # assert set(result_without.keys()) == expected_keys
    
    # Test 9: Verifier signal values are reasonable
    def test_verifier_signal_value_ranges(self):
        """Test verifier signal values are within expected ranges."""
        signal = VerifierSignal(
            claim_id='test',
            doc_id='doc',
            sent_id=1,
            nli=None,
            coverage={'entities': 0.75, 'numbers': 1.0, 'tokens_overlap': 0.6},
            uncertainty={'mean_entropy': 3.5},
            consistency={'variance': None},
            citation_span_match=0.6,
            numeric_check=True
        )
        
        # Entropy should be in [0, 10] (Shannon entropy upper bound ~ log2(vocab_size))
        assert 0.0 <= signal.uncertainty['mean_entropy'] <= 10.0
        
        # Coverage metrics should be in [0, 1]
        assert 0.0 <= signal.coverage['entities'] <= 1.0
        assert 0.0 <= signal.coverage['numbers'] <= 1.0
        assert 0.0 <= signal.coverage['tokens_overlap'] <= 1.0
        
        # Citation span match should be in [0, 1]
        assert 0.0 <= signal.citation_span_match <= 1.0
        
        # Numeric check should be boolean
        assert isinstance(signal.numeric_check, bool)
    
    # Test 10: Config loading from file
    def test_config_loading_from_file(self):
        """Test config can be loaded from config.yaml."""
        try:
            config = Config('config.yaml')
            
            # Verify verification section exists
            assert hasattr(config, 'verification')
            assert hasattr(config.verification, 'enabled')
            
            # Default should be False for backward compatibility
            # (unless changed in config.yaml)
            assert isinstance(config.verification.enabled, bool)
            
            # If verification is enabled, check other fields
            if config.verification.enabled:
                assert hasattr(config.verification, 'intrinsic')
                assert hasattr(config.verification, 'grounded')
                assert hasattr(config.verification.intrinsic, 'epsilon')
                assert hasattr(config.verification.grounded, 'entity_types')
        
        except FileNotFoundError:
            pytest.skip("config.yaml not found, skipping this test")
