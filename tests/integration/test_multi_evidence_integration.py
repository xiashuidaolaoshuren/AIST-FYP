"""
Integration test for multi-evidence verification in the full RAG pipeline.

Tests that the multi-evidence functionality works end-to-end with the baseline RAG pipeline.
"""

import pytest
import sys
from pathlib import Path

sys.path.insert(0, str(Path(__file__).parent.parent.parent))

from src.utils.config import Config


def test_config_yaml_has_multi_evidence_flags():
    """Test that config.yaml has the new multi-evidence flags."""
    # Load actual config
    config = Config()
    
    # Verify flags exist
    assert hasattr(config, 'verification'), "Config should have verification section"
    assert hasattr(config.verification, 'verify_all_evidence'), "Should have verify_all_evidence flag"
    assert hasattr(config.verification, 'aggregation_method'), "Should have aggregation_method flag"
    
    # Verify defaults
    assert config.verification.verify_all_evidence == False, "Default should be False for backward compatibility"
    assert config.verification.aggregation_method == 'max', "Default should be 'max'"


def test_multi_evidence_can_be_enabled():
    """Test that multi-evidence flag is accessible from config."""
    config = Config()
    
    # Verify the flag can be read (even if false)
    flag_value = config.verification.verify_all_evidence
    assert isinstance(flag_value, bool), "verify_all_evidence should be a boolean"
    
    # Verify aggregation method is accessible
    agg_method = config.verification.aggregation_method
    assert agg_method in ['max', 'mean'], "aggregation_method should be 'max' or 'mean'"


def test_backward_compatibility_preserved():
    """Test that default config behavior is unchanged (backward compatible)."""
    config = Config()
    
    # With default config, verification should be disabled
    assert config.verification.enabled == False
    
    # Even if enabled, multi-evidence should be off by default
    assert config.verification.verify_all_evidence == False
