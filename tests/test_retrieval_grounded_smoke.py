"""
Smoke test for RetrievalGroundedDetector to verify basic functionality.

This is a quick validation test to ensure the detector can be instantiated
and run without errors. Full unit tests will be in test_retrieval_grounded.py.
"""

import sys
import os

# Add project root to path
sys.path.insert(0, os.path.abspath(os.path.join(os.path.dirname(__file__), '..')))

from src.utils.config import Config
from src.utils.data_structures import Claim, EvidenceChunk
from src.verification.retrieval_grounded import RetrievalGroundedDetector


def test_detector_initialization():
    """Test that detector can be initialized."""
    print("Testing detector initialization...")
    config = Config()
    detector = RetrievalGroundedDetector(config)
    assert detector is not None
    assert detector.nlp is not None
    print("✓ Detector initialized successfully")


def test_entity_coverage():
    """Test entity coverage calculation."""
    print("\nTesting entity coverage...")
    config = Config()
    detector = RetrievalGroundedDetector(config)
    
    # Create test claim with entity
    claim = Claim(
        claim_id="test_c1",
        answer_id="test_a1",
        text="Barack Obama was the 44th president of the United States.",
        answer_char_span=[0, 60]
    )
    
    # Create evidence with matching entities
    evidence = EvidenceChunk(
        doc_id="test_doc",
        sent_id=1,
        text="Barack Obama was the 44th president of the United States from 2009 to 2017.",
        char_start=0,
        char_end=77,
        score_dense=0.95,
        rank=1
    )
    
    signal = detector.compute_signal(claim, evidence, {})
    
    print(f"  Entities: {signal['entities']:.2f}")
    print(f"  Numbers: {signal['numbers']:.2f}")
    print(f"  Token overlap: {signal['tokens_overlap']:.2f}")
    
    # Entity "Barack Obama" should be found
    assert signal['entities'] > 0.5, f"Expected high entity coverage, got {signal['entities']}"
    print("✓ Entity coverage test passed")


def test_number_coverage():
    """Test number coverage calculation."""
    print("\nTesting number coverage...")
    config = Config()
    detector = RetrievalGroundedDetector(config)
    
    # Create test claim with number
    claim = Claim(
        claim_id="test_c2",
        answer_id="test_a2",
        text="The Eiffel Tower is 324 meters tall.",
        answer_char_span=[0, 37]
    )
    
    # Create evidence with matching number
    evidence = EvidenceChunk(
        doc_id="test_doc",
        sent_id=2,
        text="Built in 1889, the tower stands at 324 meters in height.",
        char_start=0,
        char_end=57,
        score_dense=0.92,
        rank=1
    )
    
    signal = detector.compute_signal(claim, evidence, {})
    
    print(f"  Entities: {signal['entities']:.2f}")
    print(f"  Numbers: {signal['numbers']:.2f}")
    print(f"  Token overlap: {signal['tokens_overlap']:.2f}")
    
    # Number "324" should be found
    assert signal['numbers'] > 0.9, f"Expected high number coverage, got {signal['numbers']}"
    print("✓ Number coverage test passed")


def test_token_overlap():
    """Test token overlap (ROUGE-L) calculation."""
    print("\nTesting token overlap...")
    config = Config()
    detector = RetrievalGroundedDetector(config)
    
    # Create test claim
    claim = Claim(
        claim_id="test_c3",
        answer_id="test_a3",
        text="The capital of France is Paris.",
        answer_char_span=[0, 32]
    )
    
    # Create evidence with high overlap
    evidence = EvidenceChunk(
        doc_id="test_doc",
        sent_id=3,
        text="Paris is the capital of France.",
        char_start=0,
        char_end=32,
        score_dense=0.98,
        rank=1
    )
    
    signal = detector.compute_signal(claim, evidence, {})
    
    print(f"  Entities: {signal['entities']:.2f}")
    print(f"  Numbers: {signal['numbers']:.2f}")
    print(f"  Token overlap: {signal['tokens_overlap']:.2f}")
    
    # Should have high token overlap (same words, different order)
    assert signal['tokens_overlap'] > 0.6, f"Expected high token overlap, got {signal['tokens_overlap']}"
    print("✓ Token overlap test passed")


def test_edge_case_empty_claim():
    """Test edge case: empty claim."""
    print("\nTesting edge case: empty claim...")
    config = Config()
    detector = RetrievalGroundedDetector(config)
    
    claim = Claim(
        claim_id="test_c4",
        answer_id="test_a4",
        text="",
        answer_char_span=[0, 0]
    )
    
    evidence = EvidenceChunk(
        doc_id="test_doc",
        sent_id=4,
        text="Some evidence text.",
        char_start=0,
        char_end=19,
        score_dense=0.5,
        rank=1
    )
    
    signal = detector.compute_signal(claim, evidence, {})
    
    # Should return zeros for empty claim
    assert signal['entities'] == 0.0
    assert signal['numbers'] == 0.0
    assert signal['tokens_overlap'] == 0.0
    print("✓ Empty claim test passed")


def test_edge_case_no_entities():
    """Test edge case: claim with no entities."""
    print("\nTesting edge case: no entities...")
    config = Config()
    detector = RetrievalGroundedDetector(config)
    
    claim = Claim(
        claim_id="test_c5",
        answer_id="test_a5",
        text="The sky is blue and the grass is green.",
        answer_char_span=[0, 41]
    )
    
    evidence = EvidenceChunk(
        doc_id="test_doc",
        sent_id=5,
        text="Colors in nature are beautiful.",
        char_start=0,
        char_end=31,
        score_dense=0.6,
        rank=1
    )
    
    signal = detector.compute_signal(claim, evidence, {})
    
    # Should return 1.0 for entities (trivially satisfied)
    assert signal['entities'] == 1.0
    print("✓ No entities test passed")


if __name__ == "__main__":
    print("=" * 60)
    print("RetrievalGroundedDetector Smoke Test")
    print("=" * 60)
    
    try:
        test_detector_initialization()
        test_entity_coverage()
        test_number_coverage()
        test_token_overlap()
        test_edge_case_empty_claim()
        test_edge_case_no_entities()
        
        print("\n" + "=" * 60)
        print("✓ All smoke tests passed!")
        print("=" * 60)
    except Exception as e:
        print(f"\n✗ Test failed: {e}")
        import traceback
        traceback.print_exc()
        sys.exit(1)
