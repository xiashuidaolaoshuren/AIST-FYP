"""
Unit tests for the utilities module.

Tests data structures, configuration loader, and logging functionality.
"""

import pytest
import tempfile
import yaml
from pathlib import Path
import logging

from src.utils.data_structures import (
    EvidenceChunk, Claim, ClaimEvidencePair,
    VerifierSignal, ClaimDecision, AnnotatedAnswer
)
from src.utils.config import Config
from src.utils.logger import setup_logger, get_logger


class TestEvidenceChunk:
    """Tests for EvidenceChunk dataclass."""
    
    def test_create_evidence_chunk(self):
        """Test creating an EvidenceChunk with required fields."""
        chunk = EvidenceChunk(
            doc_id="enwiki_12345",
            sent_id=17,
            text="The FEVER dataset was introduced in 2018.",
            char_start=210,
            char_end=265,
            score_dense=0.62,
            rank=3
        )
        
        assert chunk.doc_id == "enwiki_12345"
        assert chunk.sent_id == 17
        assert chunk.source == "wikipedia"
        assert chunk.version == "wiki_sent_v1"
        assert chunk.score_bm25 is None
    
    def test_evidence_chunk_with_optional_fields(self):
        """Test EvidenceChunk with optional BM25 and hybrid scores."""
        chunk = EvidenceChunk(
            doc_id="enwiki_12345",
            sent_id=17,
            text="Test text",
            char_start=0,
            char_end=9,
            score_dense=0.62,
            rank=3,
            score_bm25=7.43,
            score_hybrid=0.69
        )
        
        assert chunk.score_bm25 == 7.43
        assert chunk.score_hybrid == 0.69
    
    def test_evidence_chunk_to_dict(self):
        """Test converting EvidenceChunk to dictionary."""
        chunk = EvidenceChunk(
            doc_id="enwiki_12345",
            sent_id=17,
            text="Test text",
            char_start=0,
            char_end=9,
            score_dense=0.62,
            rank=3
        )
        
        chunk_dict = chunk.to_dict()
        assert isinstance(chunk_dict, dict)
        assert chunk_dict['doc_id'] == "enwiki_12345"
        assert chunk_dict['sent_id'] == 17


class TestClaim:
    """Tests for Claim dataclass."""
    
    def test_create_claim(self):
        """Test creating a Claim with valid char_span."""
        claim = Claim(
            claim_id="c_0007",
            answer_id="ans_001",
            text="The FEVER dataset was introduced in 2018.",
            answer_char_span=[134, 175]
        )
        
        assert claim.claim_id == "c_0007"
        assert claim.answer_id == "ans_001"
        assert claim.extraction_method == "rule_sentence_split_v1"
        assert len(claim.answer_char_span) == 2
    
    def test_claim_invalid_char_span(self):
        """Test that Claim raises error with invalid char_span."""
        with pytest.raises(ValueError, match="must have exactly 2 elements"):
            Claim(
                claim_id="c_0007",
                answer_id="ans_001",
                text="Test claim",
                answer_char_span=[134]  # Only 1 element
            )
    
    def test_claim_to_dict(self):
        """Test converting Claim to dictionary."""
        claim = Claim(
            claim_id="c_0007",
            answer_id="ans_001",
            text="Test claim",
            answer_char_span=[134, 175]
        )
        
        claim_dict = claim.to_dict()
        assert isinstance(claim_dict, dict)
        assert claim_dict['claim_id'] == "c_0007"


class TestClaimEvidencePair:
    """Tests for ClaimEvidencePair dataclass."""
    
    def test_create_claim_evidence_pair(self):
        """Test creating a ClaimEvidencePair."""
        pair = ClaimEvidencePair(
            claim_id="c_0007",
            evidence_candidates=["enwiki_12345#17", "enwiki_77889#04"],
            top_evidence="enwiki_12345#17",
            evidence_spans=[
                {
                    "doc_id": "enwiki_12345",
                    "sent_id": 17,
                    "text": "The FEVER dataset was introduced in 2018...",
                    "rank": 3
                }
            ]
        )
        
        assert pair.claim_id == "c_0007"
        assert len(pair.evidence_candidates) == 2
        assert pair.top_evidence == "enwiki_12345#17"
        assert len(pair.evidence_spans) == 1


class TestConfig:
    """Tests for Config class."""
    
    def test_config_load_valid_file(self, tmp_path):
        """Test loading a valid configuration file."""
        config_file = tmp_path / "test_config.yaml"
        config_data = {
            'models': {
                'sentence_transformer': 'test-model',
                'generator': 'test-generator'
            },
            'data_strategy': {
                'development': {'max_articles': 100}
            },
            'processing': {
                'device': 'cpu',
                'batch_size': 16
            },
            'retrieval': {
                'top_k': 5
            }
        }
        
        with open(config_file, 'w') as f:
            yaml.dump(config_data, f)
        
        config = Config(str(config_file))
        assert config.models.sentence_transformer == 'test-model'
        assert config['processing']['batch_size'] == 16
    
    def test_config_missing_file(self):
        """Test that Config raises error for missing file."""
        with pytest.raises(FileNotFoundError):
            Config("nonexistent_config.yaml")
    
    def test_config_missing_required_field(self, tmp_path):
        """Test that Config raises error for missing required fields."""
        config_file = tmp_path / "incomplete_config.yaml"
        config_data = {
            'models': {
                'sentence_transformer': 'test-model'
                # Missing 'generator'
            }
        }
        
        with open(config_file, 'w') as f:
            yaml.dump(config_data, f)
        
        with pytest.raises(ValueError, match="Missing required configuration fields"):
            Config(str(config_file))
    
    def test_config_dot_notation_access(self, tmp_path):
        """Test dot notation access to nested config values."""
        config_file = tmp_path / "test_config.yaml"
        config_data = {
            'models': {
                'sentence_transformer': 'test-model',
                'generator': 'test-generator'
            },
            'data_strategy': {
                'development': {'max_articles': 100}
            },
            'processing': {
                'device': 'cpu',
                'batch_size': 16
            },
            'retrieval': {
                'top_k': 5
            }
        }
        
        with open(config_file, 'w') as f:
            yaml.dump(config_data, f)
        
        config = Config(str(config_file))
        assert config.models.sentence_transformer == 'test-model'
        assert config.processing.batch_size == 16
    
    def test_config_get_method(self, tmp_path):
        """Test get method with default values."""
        config_file = tmp_path / "test_config.yaml"
        config_data = {
            'models': {
                'sentence_transformer': 'test-model',
                'generator': 'test-generator'
            },
            'data_strategy': {
                'development': {'max_articles': 100}
            },
            'processing': {
                'device': 'cpu',
                'batch_size': 16
            },
            'retrieval': {
                'top_k': 5
            }
        }
        
        with open(config_file, 'w') as f:
            yaml.dump(config_data, f)
        
        config = Config(str(config_file))
        assert config.get('models.sentence_transformer') == 'test-model'
        assert config.get('nonexistent.key', 'default') == 'default'


class TestLogger:
    """Tests for logger functionality."""
    
    def test_setup_logger(self, tmp_path):
        """Test setting up a logger with file and console handlers."""
        log_file = tmp_path / "test.log"
        logger = setup_logger('test_logger', str(log_file))
        
        assert isinstance(logger, logging.Logger)
        assert logger.name == 'test_logger'
        assert len(logger.handlers) == 2  # File + Console
        
        # Test logging
        logger.info("Test info message")
        logger.error("Test error message")
        
        # Check log file was created
        assert log_file.exists()
        
        # Check log file contains the info message
        with open(log_file, 'r') as f:
            content = f.read()
            assert "Test info message" in content
            assert "Test error message" in content
    
    def test_logger_creates_directory(self, tmp_path):
        """Test that logger creates log directory if it doesn't exist."""
        log_file = tmp_path / "subdir" / "nested" / "test.log"
        logger = setup_logger('test_logger_nested', str(log_file))
        
        logger.info("Test message")
        
        assert log_file.exists()
        assert log_file.parent.exists()
    
    def test_get_existing_logger(self, tmp_path):
        """Test retrieving an existing logger."""
        log_file = tmp_path / "test.log"
        logger1 = setup_logger('test_logger_2', str(log_file))
        logger2 = get_logger('test_logger_2')
        
        assert logger1 is logger2


class TestVerifierSignal:
    """Tests for VerifierSignal dataclass."""
    
    def test_create_verifier_signal(self):
        """Test creating a VerifierSignal."""
        signal = VerifierSignal(
            claim_id="c_0007",
            doc_id="enwiki_12345",
            sent_id=17,
            nli={"entail": 0.81, "contradict": 0.03, "neutral": 0.16},
            coverage={"entities": 0.83, "numbers": 1.0, "tokens_overlap": 0.74},
            uncertainty={"mean_entropy": 1.12},
            consistency={"variance": None},
            citation_span_match=0.9,
            numeric_check=True
        )
        
        assert signal.claim_id == "c_0007"
        assert signal.nli["entail"] == 0.81
        assert signal.numeric_check is True


class TestClaimDecision:
    """Tests for ClaimDecision dataclass."""
    
    def test_create_claim_decision(self):
        """Test creating a ClaimDecision."""
        decision = ClaimDecision(
            claim_id="c_0007",
            status="Supported",
            rationale="High entail prob, good entity coverage",
            primary_evidence="enwiki_12345#17",
            signals_ref=["sig_c_0007_17"],
            confidence={
                "support_prob": 0.81,
                "contradict_prob": 0.03,
                "overall_confidence": 0.74,
                "band": "High"
            }
        )
        
        assert decision.claim_id == "c_0007"
        assert decision.status == "Supported"
        assert decision.confidence["band"] == "High"


class TestAnnotatedAnswer:
    """Tests for AnnotatedAnswer dataclass."""
    
    def test_create_annotated_answer(self):
        """Test creating an AnnotatedAnswer."""
        answer = AnnotatedAnswer(
            answer_id="ans_001",
            query_id="q_20250201_001",
            raw_answer="The FEVER dataset was introduced in 2018.",
            claims=[
                {
                    "claim_id": "c_0007",
                    "status": "Supported"
                }
            ],
            summary_stats={
                "claims_total": 1,
                "supported_high": 1
            },
            mitigation_actions=["removed_contradicted_claims"]
        )
        
        assert answer.answer_id == "ans_001"
        assert answer.version == "pipeline_v0.3"
        assert len(answer.claims) == 1
