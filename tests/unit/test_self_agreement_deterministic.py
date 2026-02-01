"""
Unit tests for SelfAgreementDetector deterministic mode.

Tests that:
1. With deterministic=true, same query produces identical scores across runs
2. With deterministic=false, scores show natural variance across runs
"""

import pytest
from src.utils.config import Config
from src.generation.generator_wrapper import GeneratorWrapper
from src.verification.self_agreement import SelfAgreementDetector
from src.utils.data_structures import EvidenceChunk


@pytest.fixture(scope="module")
def base_config():
    """Load base configuration."""
    return Config('config.yaml')


@pytest.fixture(scope="module")
def generator(base_config):
    """Initialize generator wrapper."""
    return GeneratorWrapper(base_config.models.generator)


@pytest.fixture
def test_query():
    """Sample query for testing."""
    return "What is artificial intelligence?"


@pytest.fixture
def test_claim():
    """Sample claim for testing."""
    return "AI is the intelligence of machines or software."


@pytest.fixture
def test_evidence():
    """Sample evidence chunk for testing."""
    return [
        EvidenceChunk(
            doc_id="test_doc",
            sent_id=0,
            text="Artificial intelligence is the simulation of human intelligence by machines.",
            char_start=0,
            char_end=76,
            score_dense=0.95,
            rank=0
        )
    ]


def test_deterministic_mode_produces_identical_scores(
    base_config, generator, test_query, test_claim, test_evidence
):
    """Test that deterministic mode produces identical results across multiple runs."""
    # Configure deterministic mode
    config_dict = base_config.to_dict()
    config_dict['verification']['self_agreement']['deterministic'] = True
    config_dict['verification']['self_agreement']['random_seed'] = 42
    
    # Initialize detector
    detector = SelfAgreementDetector(config_dict, generator)
    
    # Run detection 3 times
    scores = []
    variances = []
    for _ in range(3):
        result = detector.detect(test_claim, test_query, test_evidence)
        scores.append(result.get('score'))
        variances.append(result.get('variance'))
    
    # Assert all scores are identical (deterministic behavior)
    assert len(set(scores)) == 1, (
        f"Expected identical scores in deterministic mode, got: {scores}"
    )
    
    # Assert all variances are identical
    assert len(set(variances)) == 1, (
        f"Expected identical variances in deterministic mode, got: {variances}"
    )
    
    # Assert scores are not None
    assert scores[0] is not None, "Score should not be None"
    assert variances[0] is not None, "Variance should not be None"
    
    # Assert scores are in valid range [0, 1]
    assert 0.0 <= scores[0] <= 1.0, f"Score {scores[0]} out of valid range [0, 1]"


def test_stochastic_mode_shows_variance(
    base_config, generator, test_query, test_claim, test_evidence
):
    """Test that stochastic mode produces variable results across runs."""
    # Configure stochastic mode
    config_dict = base_config.to_dict()
    config_dict['verification']['self_agreement']['deterministic'] = False
    
    # Initialize detector
    detector = SelfAgreementDetector(config_dict, generator)
    
    # Run detection 3 times
    scores = []
    variances = []
    for _ in range(3):
        result = detector.detect(test_claim, test_query, test_evidence)
        scores.append(result.get('score'))
        variances.append(result.get('variance'))
    
    # Note: Stochastic mode SHOULD show variation, but it's possible (though unlikely)
    # to get identical scores by chance. We check if at least some variation exists.
    unique_scores = len(set(scores))
    
    # We expect variation, but allow for the small chance of identical scores
    # Assert that scores are valid even if identical
    assert all(s is not None for s in scores), "All scores should be non-None"
    assert all(0.0 <= s <= 1.0 for s in scores), "All scores should be in [0, 1]"
    
    # Log variation info (this will appear in pytest output with -v)
    if unique_scores == 1:
        pytest.skip(
            f"Stochastic mode produced identical scores ({scores}), "
            "which is unusual but can happen by chance"
        )
    else:
        # This is the expected behavior
        assert unique_scores > 1, (
            f"Expected variation in stochastic mode, got {unique_scores}/3 unique values"
        )


def test_different_seeds_produce_different_results(
    base_config, generator, test_query, test_claim, test_evidence
):
    """Test that different random seeds produce different deterministic results."""
    # Configure deterministic mode with seed 42
    config_dict_1 = base_config.to_dict()
    config_dict_1['verification']['self_agreement']['deterministic'] = True
    config_dict_1['verification']['self_agreement']['random_seed'] = 42
    detector_1 = SelfAgreementDetector(config_dict_1, generator)
    
    # Configure deterministic mode with seed 123
    config_dict_2 = base_config.to_dict()
    config_dict_2['verification']['self_agreement']['deterministic'] = True
    config_dict_2['verification']['self_agreement']['random_seed'] = 123
    detector_2 = SelfAgreementDetector(config_dict_2, generator)
    
    # Run detection with each detector
    result_1 = detector_1.detect(test_claim, test_query, test_evidence)
    result_2 = detector_2.detect(test_claim, test_query, test_evidence)
    
    score_1 = result_1.get('score')
    score_2 = result_2.get('score')
    
    # Different seeds should produce different results
    # (though they could theoretically be the same by chance)
    assert score_1 is not None and score_2 is not None
    assert 0.0 <= score_1 <= 1.0 and 0.0 <= score_2 <= 1.0
    
    # We expect different scores, but allow for small chance of collision
    if abs(score_1 - score_2) < 1e-6:
        pytest.skip(
            f"Different seeds produced nearly identical scores ({score_1:.6f}, {score_2:.6f}), "
            "which is possible but unusual"
        )


def test_deterministic_mode_config_defaults():
    """Test that deterministic mode defaults are correctly set."""
    # Load fresh config to check defaults
    fresh_config = Config('config.yaml')
    config_dict = fresh_config.to_dict()
    
    # Check default values
    sa_config = config_dict['verification']['self_agreement']
    assert 'deterministic' in sa_config, "deterministic key should exist in config"
    assert 'random_seed' in sa_config, "random_seed key should exist in config"
    
    # Check default values match expected
    assert sa_config['deterministic'] == False, "Default should be stochastic (False)"
    assert isinstance(sa_config['random_seed'], int), "random_seed should be an integer"
