"""
Unit tests for CitationFormatter.

This module tests citation injection, deduplication, CiteEval export,
and edge case handling for the CitationFormatter class.
"""

import pytest
from pathlib import Path
from src.citation.citation_formatter import CitationFormatter
from src.utils.data_structures import Claim, EvidenceChunk
from src.utils.config import Config


@pytest.fixture
def mock_config():
    """Create a Config object with actual config.yaml."""
    config_path = Path(__file__).parent.parent.parent / "config.yaml"
    config = Config(str(config_path))
    return config


@pytest.fixture
def formatter(mock_config):
    """Create a CitationFormatter instance with mock config."""
    return CitationFormatter(mock_config)


@pytest.fixture
def sample_claims():
    """Create sample claims for testing."""
    return [
        Claim(
            claim_id='c1',
            answer_id='ans1',
            text='The cat sat on the mat.',
            answer_char_span=(0, 23)
        ),
        Claim(
            claim_id='c2',
            answer_id='ans1',
            text='It was very comfortable',
            answer_char_span=(24, 47)
        )
    ]


@pytest.fixture
def sample_evidence():
    """Create sample evidence chunks for testing."""
    return {
        'c1': [
            EvidenceChunk(
                doc_id='enwiki_001',
                sent_id=1,
                text='Cats often sit on mats.',
                char_start=0,
                char_end=23,
                score_dense=0.95,
                rank=1
            ),
            EvidenceChunk(
                doc_id='enwiki_002',
                sent_id=3,
                text='A mat is a comfortable surface.',
                char_start=100,
                char_end=132,
                score_dense=0.88,
                rank=2
            ),
            EvidenceChunk(
                doc_id='enwiki_003',
                sent_id=2,
                text='Felines prefer soft surfaces.',
                char_start=50,
                char_end=79,
                score_dense=0.82,
                rank=3
            )
        ],
        'c2': [
            EvidenceChunk(
                doc_id='enwiki_004',
                sent_id=1,
                text='Comfort is important for cats.',
                char_start=0,
                char_end=31,
                score_dense=0.91,
                rank=1
            ),
            EvidenceChunk(
                doc_id='enwiki_005',
                sent_id=2,
                text='Soft materials provide comfort.',
                char_start=50,
                char_end=81,
                score_dense=0.85,
                rank=2
            )
        ]
    }


class TestCitationFormatterInit:
    """Test CitationFormatter initialization."""
    
    def test_init_with_config_value(self, mock_config):
        """Test initialization with configured max_citations_per_claim."""
        # Add citation config if not exists
        if not hasattr(mock_config, 'citation'):
            mock_config.citation = type('CitationConfig', (), {})()
        mock_config.citation.max_citations_per_claim = 5
        formatter = CitationFormatter(mock_config)
        assert formatter.max_citations_per_claim == 5
    
    def test_init_with_default_value(self):
        """Test initialization falls back to default value when not in config."""
        config_path = Path(__file__).parent.parent.parent / "config.yaml"
        config = Config(str(config_path))
        formatter = CitationFormatter(config)
        # Should use default value of 3 since config.yaml doesn't have citation section
        assert formatter.max_citations_per_claim == 3


class TestFormatWithCitations:
    """Test main citation formatting functionality."""
    
    def test_single_claim_citation(self, formatter):
        """Test citation injection for a single claim."""
        answer_text = "The cat sat on the mat."
        claims = [
            Claim(
                claim_id='c1',
                answer_id='ans1',
                text='The cat sat on the mat.',
                answer_char_span=(0, 23)
            )
        ]
        evidence_map = {
            'c1': [
                EvidenceChunk(
                    doc_id='enwiki_001',
                    sent_id=1,
                    text='Cats sit on mats.',
                    char_start=0,
                    char_end=17,
                    score_dense=0.95,
                    rank=1
                )
            ]
        }
        
        result = formatter.format_with_citations(answer_text, claims, evidence_map)
        
        # Citation should be inserted before period
        assert result['formatted_text'] == "The cat sat on the mat[1]."
        assert result['citation_map'] == {'c1': [1]}
        assert len(result['passage_list']) == 1
        assert result['passage_list'][0]['text'] == 'Cats sit on mats.'
    
    def test_multiple_claims_citation(self, formatter, sample_claims, sample_evidence):
        """Test citation injection for multiple claims."""
        answer_text = "The cat sat on the mat. It was very comfortable."
        
        result = formatter.format_with_citations(answer_text, sample_claims, sample_evidence)
        
        # Should have citations after each claim (indices will vary based on score_dense sorting)
        # Check that both claims have citations with brackets
        assert '[' in result['formatted_text'] and ']' in result['formatted_text']
        
        # Check citation map exists for both claims
        assert 'c1' in result['citation_map']
        assert 'c2' in result['citation_map']
        
        # Check that each claim has exactly 3 and 2 citations respectively
        assert len(result['citation_map']['c1']) == 3
        assert len(result['citation_map']['c2']) == 2
        
        # Check passage list has all unique evidence
        assert len(result['passage_list']) == 5  # 3 + 2 evidence chunks
    
    def test_citation_format_no_spaces(self, formatter):
        """Test that citation format has no spaces between brackets."""
        answer_text = "Test sentence."
        claims = [
            Claim(
                claim_id='c1',
                answer_id='ans1',
                text='Test sentence.',
                answer_char_span=(0, 14)
            )
        ]
        evidence_map = {
            'c1': [
                EvidenceChunk(doc_id='doc1', sent_id=1, text='E1', char_start=0, char_end=2, score_dense=0.9, rank=1),
                EvidenceChunk(doc_id='doc2', sent_id=1, text='E2', char_start=0, char_end=2, score_dense=0.8, rank=2),
                EvidenceChunk(doc_id='doc3', sent_id=1, text='E3', char_start=0, char_end=2, score_dense=0.7, rank=3)
            ]
        }
        
        result = formatter.format_with_citations(answer_text, claims, evidence_map)
        
        # Check format is [1][2][3] with NO spaces
        assert result['formatted_text'] == "Test sentence[1][2][3]."
        assert '[1] [2]' not in result['formatted_text']
    
    def test_punctuation_handling_period(self, formatter):
        """Test citation insertion before period."""
        answer_text = "Cats are mammals."
        claims = [
            Claim(
                claim_id='c1',
                answer_id='ans1',
                text='Cats are mammals.',
                answer_char_span=(0, 17)
            )
        ]
        evidence_map = {
            'c1': [
                EvidenceChunk(doc_id='doc1', sent_id=1, text='E1', char_start=0, char_end=2, score_dense=0.9, rank=1)
            ]
        }
        
        result = formatter.format_with_citations(answer_text, claims, evidence_map)
        assert result['formatted_text'] == "Cats are mammals[1]."
    
    def test_punctuation_handling_exclamation(self, formatter):
        """Test citation insertion before exclamation mark."""
        answer_text = "Cats are amazing!"
        claims = [
            Claim(
                claim_id='c1',
                answer_id='ans1',
                text='Cats are amazing!',
                answer_char_span=(0, 17)
            )
        ]
        evidence_map = {
            'c1': [
                EvidenceChunk(doc_id='doc1', sent_id=1, text='E1', char_start=0, char_end=2, score_dense=0.9, rank=1)
            ]
        }
        
        result = formatter.format_with_citations(answer_text, claims, evidence_map)
        assert result['formatted_text'] == "Cats are amazing[1]!"
    
    def test_punctuation_handling_question(self, formatter):
        """Test citation insertion before question mark."""
        answer_text = "Are cats mammals?"
        claims = [
            Claim(
                claim_id='c1',
                answer_id='ans1',
                text='Are cats mammals?',
                answer_char_span=(0, 17)
            )
        ]
        evidence_map = {
            'c1': [
                EvidenceChunk(doc_id='doc1', sent_id=1, text='E1', char_start=0, char_end=2, score_dense=0.9, rank=1)
            ]
        }
        
        result = formatter.format_with_citations(answer_text, claims, evidence_map)
        assert result['formatted_text'] == "Are cats mammals[1]?"
    
    def test_no_punctuation_append(self, formatter):
        """Test citation appended at end when no punctuation."""
        answer_text = "Cats are mammals"
        claims = [
            Claim(
                claim_id='c1',
                answer_id='ans1',
                text='Cats are mammals',
                answer_char_span=(0, 16)
            )
        ]
        evidence_map = {
            'c1': [
                EvidenceChunk(doc_id='doc1', sent_id=1, text='E1', char_start=0, char_end=2, score_dense=0.9, rank=1)
            ]
        }
        
        result = formatter.format_with_citations(answer_text, claims, evidence_map)
        assert result['formatted_text'] == "Cats are mammals[1]"
    
    def test_max_citations_limit(self):
        """Test that only max_citations_per_claim are used."""
        config_path = Path(__file__).parent.parent.parent / "config.yaml"
        config = Config(str(config_path))
        # Add citation config
        if not hasattr(config, 'citation'):
            config.citation = type('CitationConfig', (), {})()
        config.citation.max_citations_per_claim = 2  # Set to 2
        formatter = CitationFormatter(config)
        
        answer_text = "Test."
        claims = [
            Claim(claim_id='c1', answer_id='ans1', text='Test.', answer_char_span=(0, 5))
        ]
        evidence_map = {
            'c1': [
                EvidenceChunk(doc_id='doc1', sent_id=1, text='E1', char_start=0, char_end=2, score_dense=0.9, rank=1),
                EvidenceChunk(doc_id='doc2', sent_id=1, text='E2', char_start=0, char_end=2, score_dense=0.8, rank=2),
                EvidenceChunk(doc_id='doc3', sent_id=1, text='E3', char_start=0, char_end=2, score_dense=0.7, rank=3),
                EvidenceChunk(doc_id='doc4', sent_id=1, text='E4', char_start=0, char_end=2, score_dense=0.6, rank=4)
            ]
        }
        
        result = formatter.format_with_citations(answer_text, claims, evidence_map)
        
        # Should only have [1][2], not [1][2][3][4]
        assert result['formatted_text'] == "Test[1][2]."
        assert len(result['citation_map']['c1']) == 2
    
    def test_empty_evidence_handling(self, formatter):
        """Test handling of claims with no evidence (should skip and log warning)."""
        answer_text = "The cat sat. It was happy."
        claims = [
            Claim(claim_id='c1', answer_id='ans1', text='The cat sat.', answer_char_span=(0, 12)),
            Claim(claim_id='c2', answer_id='ans1', text='It was happy.', answer_char_span=(13, 26))
        ]
        evidence_map = {
            'c1': [
                EvidenceChunk(doc_id='doc1', sent_id=1, text='E1', char_start=0, char_end=2, score_dense=0.9, rank=1)
            ],
            'c2': []  # Empty evidence
        }
        
        result = formatter.format_with_citations(answer_text, claims, evidence_map)
        
        # c1 should have citation, c2 should not
        assert 'c1' in result['citation_map']
        assert 'c2' not in result['citation_map']
        assert result['formatted_text'] == "The cat sat[1]. It was happy."
    
    def test_missing_claim_in_evidence_map(self, formatter):
        """Test handling of claims not in evidence_map (should skip and log warning)."""
        answer_text = "Test sentence."
        claims = [
            Claim(claim_id='c1', answer_id='ans1', text='Test sentence.', answer_char_span=(0, 14))
        ]
        evidence_map = {}  # No evidence for c1
        
        result = formatter.format_with_citations(answer_text, claims, evidence_map)
        
        # Should skip citation
        assert result['formatted_text'] == answer_text
        assert len(result['citation_map']) == 0
    
    def test_text_integrity_no_corruption(self, formatter, sample_claims, sample_evidence):
        """Test that original text is preserved except for citations."""
        answer_text = "The cat sat on the mat. It was very comfortable."
        
        result = formatter.format_with_citations(answer_text, sample_claims, sample_evidence)
        
        # Remove all citation brackets and verify original text
        cleaned_text = result['formatted_text']
        for i in range(1, 10):
            cleaned_text = cleaned_text.replace(f'[{i}]', '')
        
        assert cleaned_text == answer_text


class TestBuildPassageList:
    """Test passage list building and deduplication."""
    
    def test_passage_list_deduplication(self, formatter):
        """Test that duplicate evidence (same doc_id#sent_id) is removed."""
        evidence_map = {
            'c1': [
                EvidenceChunk(doc_id='doc1', sent_id=1, text='E1', char_start=0, char_end=2, score_dense=0.9, rank=1),
                EvidenceChunk(doc_id='doc2', sent_id=1, text='E2', char_start=0, char_end=2, score_dense=0.8, rank=2)
            ],
            'c2': [
                EvidenceChunk(doc_id='doc1', sent_id=1, text='E1', char_start=0, char_end=2, score_dense=0.9, rank=1),  # Duplicate
                EvidenceChunk(doc_id='doc3', sent_id=1, text='E3', char_start=0, char_end=2, score_dense=0.7, rank=3)
            ]
        }
        
        passage_list, doc_to_index = formatter._build_passage_list(evidence_map)
        
        # Should have 3 unique passages (doc1#1, doc2#1, doc3#1)
        assert len(passage_list) == 3
        assert len(doc_to_index) == 3
        
        # Check doc_to_index mapping
        assert 'doc1#1' in doc_to_index
        assert 'doc2#1' in doc_to_index
        assert 'doc3#1' in doc_to_index
    
    def test_passage_list_sorting_by_score(self, formatter):
        """Test that passages are sorted by score_dense in descending order."""
        evidence_map = {
            'c1': [
                EvidenceChunk(doc_id='doc1', sent_id=1, text='E1', char_start=0, char_end=2, score_dense=0.7, rank=3),
                EvidenceChunk(doc_id='doc2', sent_id=1, text='E2', char_start=0, char_end=2, score_dense=0.9, rank=1),
                EvidenceChunk(doc_id='doc3', sent_id=1, text='E3', char_start=0, char_end=2, score_dense=0.8, rank=2)
            ]
        }
        
        passage_list, doc_to_index = formatter._build_passage_list(evidence_map)
        
        # Check first passage has highest score (doc2)
        assert passage_list[0]['title'] == 'doc2'
        assert passage_list[1]['title'] == 'doc3'
        assert passage_list[2]['title'] == 'doc1'
        
        # Check indices are assigned in sorted order
        assert doc_to_index['doc2#1'] == 1
        assert doc_to_index['doc3#1'] == 2
        assert doc_to_index['doc1#1'] == 3
    
    def test_passage_list_format(self, formatter):
        """Test that passages have correct format {text, title}."""
        evidence_map = {
            'c1': [
                EvidenceChunk(
                    doc_id='enwiki_12345',
                    sent_id=2,
                    text='Cats are mammals.',
                    char_start=100,
                    char_end=117,
                    score_dense=0.9,
                    rank=1
                )
            ]
        }
        
        passage_list, _ = formatter._build_passage_list(evidence_map)
        
        assert len(passage_list) == 1
        passage = passage_list[0]
        assert 'text' in passage
        assert 'title' in passage
        assert passage['text'] == 'Cats are mammals.'
        assert passage['title'] == 'enwiki_12345'
    
    def test_empty_evidence_map(self, formatter):
        """Test handling of empty evidence map."""
        passage_list, doc_to_index = formatter._build_passage_list({})
        
        assert passage_list == []
        assert doc_to_index == {}
    
    def test_one_based_indexing(self, formatter):
        """Test that passage indices start from 1, not 0."""
        evidence_map = {
            'c1': [
                EvidenceChunk(doc_id='doc1', sent_id=1, text='E1', char_start=0, char_end=2, score_dense=0.9, rank=1)
            ]
        }
        
        _, doc_to_index = formatter._build_passage_list(evidence_map)
        
        # First passage should have index 1
        assert doc_to_index['doc1#1'] == 1


class TestExportCiteEvalFormat:
    """Test CiteEval format export."""
    
    def test_citeeval_format_structure(self, formatter):
        """Test that output has correct CiteEval format structure."""
        query = "What is a cat?"
        formatted_output = {
            'formatted_text': 'A cat is a mammal[1].',
            'citation_map': {'c1': [1]},
            'passage_list': [
                {'text': 'Cats are mammals.', 'title': 'enwiki_001'}
            ]
        }
        answer_id = 'ans_001'
        
        result = formatter.export_citeeval_format(query, formatted_output, answer_id)
        
        # Check all required keys exist
        assert 'id' in result
        assert 'query' in result
        assert 'passages' in result
        assert 'pred' in result
        
        # Check values
        assert result['id'] == 'ans_001'
        assert result['query'] == 'What is a cat?'
        assert result['passages'] == formatted_output['passage_list']
        assert result['pred'] == 'A cat is a mammal[1].'
    
    def test_citeeval_format_multiple_passages(self, formatter):
        """Test CiteEval export with multiple passages."""
        query = "Tell me about cats."
        formatted_output = {
            'formatted_text': 'Cats are mammals[1][2]. They are popular pets[3].',
            'citation_map': {'c1': [1, 2], 'c2': [3]},
            'passage_list': [
                {'text': 'Passage 1', 'title': 'doc1'},
                {'text': 'Passage 2', 'title': 'doc2'},
                {'text': 'Passage 3', 'title': 'doc3'}
            ]
        }
        answer_id = 'ans_002'
        
        result = formatter.export_citeeval_format(query, formatted_output, answer_id)
        
        assert len(result['passages']) == 3
        assert result['pred'] == 'Cats are mammals[1][2]. They are popular pets[3].'


class TestOverlappingSpans:
    """Test overlapping span detection and handling."""
    
    def test_overlapping_spans_warning(self, formatter, caplog):
        """Test that overlapping spans trigger a warning log."""
        import logging
        
        # Set up logging capture
        with caplog.at_level(logging.WARNING):
            answer_text = "The quick brown fox jumps."
            claims = [
                # These spans overlap: [0, 15] and [10, 26]
                Claim(claim_id='c1', answer_id='ans1', text='The quick brown', answer_char_span=(0, 15)),
                Claim(claim_id='c2', answer_id='ans1', text='brown fox jumps.', answer_char_span=(10, 26))
            ]
            evidence_map = {
                'c1': [EvidenceChunk(doc_id='doc1', sent_id=1, text='E1', char_start=0, char_end=2, score_dense=0.9, rank=1)],
                'c2': [EvidenceChunk(doc_id='doc2', sent_id=1, text='E2', char_start=0, char_end=2, score_dense=0.8, rank=1)]
            }
            
            formatter.format_with_citations(answer_text, claims, evidence_map)
            
            # Check that warning was logged
            assert any('Overlapping spans detected' in record.message for record in caplog.records)
    
    def test_non_overlapping_spans_no_warning(self, formatter, caplog):
        """Test that non-overlapping spans don't trigger warning."""
        import logging
        
        with caplog.at_level(logging.WARNING):
            answer_text = "The cat sat. It was happy."
            claims = [
                Claim(claim_id='c1', answer_id='ans1', text='The cat sat.', answer_char_span=(0, 12)),
                Claim(claim_id='c2', answer_id='ans1', text='It was happy.', answer_char_span=(13, 26))
            ]
            evidence_map = {
                'c1': [EvidenceChunk(doc_id='doc1', sent_id=1, text='E1', char_start=0, char_end=2, score_dense=0.9, rank=1)],
                'c2': [EvidenceChunk(doc_id='doc2', sent_id=1, text='E2', char_start=0, char_end=2, score_dense=0.8, rank=1)]
            }
            
            formatter.format_with_citations(answer_text, claims, evidence_map)
            
            # Check that no overlapping warning was logged
            assert not any('Overlapping spans detected' in record.message for record in caplog.records)


class TestEdgeCases:
    """Test edge cases and boundary conditions."""
    
    def test_empty_answer_text(self, formatter):
        """Test handling of empty answer text."""
        result = formatter.format_with_citations("", [], {})
        assert result['formatted_text'] == ""
        assert result['citation_map'] == {}
        assert result['passage_list'] == []
    
    def test_span_exceeds_text_length(self, formatter, caplog):
        """Test handling of invalid span that exceeds text length."""
        import logging
        
        with caplog.at_level(logging.ERROR):
            answer_text = "Short text"
            claims = [
                Claim(claim_id='c1', answer_id='ans1', text='Invalid span', answer_char_span=(0, 100))  # Exceeds length
            ]
            evidence_map = {
                'c1': [EvidenceChunk(doc_id='doc1', sent_id=1, text='E1', char_start=0, char_end=2, score_dense=0.9, rank=1)]
            }
            
            result = formatter.format_with_citations(answer_text, claims, evidence_map)
            
            # Should skip this claim and log error
            assert result['formatted_text'] == answer_text
            assert 'c1' not in result['citation_map']
            assert any('exceeds text length' in record.message for record in caplog.records)
    
    def test_single_character_claim(self, formatter):
        """Test handling of single-character claim."""
        answer_text = "A."
        claims = [
            Claim(claim_id='c1', answer_id='ans1', text='A.', answer_char_span=(0, 2))
        ]
        evidence_map = {
            'c1': [EvidenceChunk(doc_id='doc1', sent_id=1, text='E1', char_start=0, char_end=2, score_dense=0.9, rank=1)]
        }
        
        result = formatter.format_with_citations(answer_text, claims, evidence_map)
        assert result['formatted_text'] == "A[1]."
    
    def test_claim_at_end_of_text(self, formatter):
        """Test citation insertion for claim at the very end of text."""
        answer_text = "The cat sat on the mat."
        claims = [
            Claim(claim_id='c1', answer_id='ans1', text='mat.', answer_char_span=(19, 23))
        ]
        evidence_map = {
            'c1': [EvidenceChunk(doc_id='doc1', sent_id=1, text='E1', char_start=0, char_end=2, score_dense=0.9, rank=1)]
        }
        
        result = formatter.format_with_citations(answer_text, claims, evidence_map)
        assert result['formatted_text'] == "The cat sat on the mat[1]."
