"""
Unit tests for RagasEvaluator.

Tests the Ragas evaluation framework wrapper, including:
- Initialization and configuration loading
- Dataset format conversion
- Error handling
- Metric management

Note: These are unit tests with mocked Ragas API calls.
      See test_ragas_integration.py for integration tests with real API calls.
"""

import os
import pytest
from unittest.mock import Mock, patch, MagicMock
import pandas as pd
from datasets import Dataset

from src.evaluation.ragas_evaluator import RagasEvaluator
from src.utils.config import Config


@pytest.fixture
def mock_env_openai_key(monkeypatch):
    """Mock OPENAI_API_KEY environment variable."""
    monkeypatch.setenv('OPENAI_API_KEY', 'sk-test-key-123')


@pytest.fixture
def config():
    """Create test configuration."""
    config = Mock(spec=Config)
    config.get = Mock(side_effect=lambda key, default=None: {
        'evaluation.ragas.model': 'gpt-4o-mini',
        'evaluation.ragas.temperature': 0,
        'evaluation.ragas.metrics': ['faithfulness', 'answer_relevancy']
    }.get(key, default))
    return config


@pytest.fixture
def sample_rag_results():
    """Sample RAG results for testing."""
    return [
        {
            'question': 'What is a cat?',
            'answer': 'A cat is a small carnivorous mammal.',
            'contexts': [
                'Cats are mammals that belong to the family Felidae.',
                'They are carnivorous animals with retractable claws.'
            ],
            'ground_truth': 'A cat is a domesticated carnivorous mammal of the family Felidae.'
        },
        {
            'question': 'What is the capital of France?',
            'answer': 'The capital of France is Paris.',
            'contexts': [
                'Paris is the capital and largest city of France.',
                'It is located in the north of France.'
            ],
            'ground_truth': 'Paris is the capital of France.'
        }
    ]


class TestRagasEvaluatorInitialization:
    """Test RagasEvaluator initialization."""
    
    @patch('src.evaluation.ragas_evaluator.ChatOpenAI')
    @patch('src.evaluation.ragas_evaluator.LangchainLLMWrapper')
    @patch('src.evaluation.ragas_evaluator.Faithfulness')
    @patch('src.evaluation.ragas_evaluator.AnswerRelevancy')
    def test_init_success(
        self,
        mock_answer_relevancy,
        mock_faithfulness,
        mock_llm_wrapper,
        mock_chat_openai,
        config,
        mock_env_openai_key
    ):
        """Test successful initialization."""
        # Setup mocks
        mock_llm = Mock()
        mock_chat_openai.return_value = mock_llm
        mock_wrapped_llm = Mock()
        mock_llm_wrapper.return_value = mock_wrapped_llm
        
        # Initialize
        evaluator = RagasEvaluator(config)
        
        # Verify
        assert evaluator.model_name == 'gpt-4o-mini'
        assert evaluator.temperature == 0
        assert len(evaluator.metrics) == 2
        assert evaluator.metric_names == ['faithfulness', 'answer_relevancy']
        
        # Verify LLM initialization
        mock_chat_openai.assert_called_once_with(
            model='gpt-4o-mini',
            temperature=0
        )
        mock_llm_wrapper.assert_called_once_with(mock_llm)
        
        # Verify metrics initialization
        mock_faithfulness.assert_called_once_with(llm=mock_wrapped_llm)
        mock_answer_relevancy.assert_called_once_with(llm=mock_wrapped_llm)
    
    def test_init_missing_api_key(self, config):
        """Test initialization fails without OPENAI_API_KEY."""
        with pytest.raises(ValueError, match="OPENAI_API_KEY environment variable is required"):
            RagasEvaluator(config)
    
    @patch('src.evaluation.ragas_evaluator.ChatOpenAI')
    @patch('src.evaluation.ragas_evaluator.LangchainLLMWrapper')
    def test_init_no_metrics_configured(
        self,
        mock_llm_wrapper,
        mock_chat_openai,
        config,
        mock_env_openai_key
    ):
        """Test initialization fails with no valid metrics."""
        # Mock config to return empty metrics list
        config.get = Mock(side_effect=lambda key, default=None: {
            'evaluation.ragas.model': 'gpt-4o-mini',
            'evaluation.ragas.temperature': 0,
            'evaluation.ragas.metrics': []
        }.get(key, default))
        
        with pytest.raises(ValueError, match="No valid metrics configured"):
            RagasEvaluator(config)
    
    @patch('src.evaluation.ragas_evaluator.ChatOpenAI')
    @patch('src.evaluation.ragas_evaluator.LangchainLLMWrapper')
    @patch('src.evaluation.ragas_evaluator.Faithfulness')
    def test_init_unknown_metric_skipped(
        self,
        mock_faithfulness,
        mock_llm_wrapper,
        mock_chat_openai,
        config,
        mock_env_openai_key,
        caplog
    ):
        """Test unknown metrics are skipped with warning."""
        # Mock config with unknown metric
        config.get = Mock(side_effect=lambda key, default=None: {
            'evaluation.ragas.model': 'gpt-4o-mini',
            'evaluation.ragas.temperature': 0,
            'evaluation.ragas.metrics': ['faithfulness', 'unknown_metric']
        }.get(key, default))
        
        evaluator = RagasEvaluator(config)
        
        # Should only have 1 valid metric
        assert len(evaluator.metrics) == 1
        assert evaluator.metric_names == ['faithfulness']
        
        # Should log warning
        assert "Unknown metric 'unknown_metric'" in caplog.text


class TestDatasetConversion:
    """Test RAG results to Ragas dataset conversion."""
    
    @patch('src.evaluation.ragas_evaluator.ChatOpenAI')
    @patch('src.evaluation.ragas_evaluator.LangchainLLMWrapper')
    @patch('src.evaluation.ragas_evaluator.Faithfulness')
    @patch('src.evaluation.ragas_evaluator.AnswerRelevancy')
    def test_convert_to_ragas_format_success(
        self,
        mock_answer_relevancy,
        mock_faithfulness,
        mock_llm_wrapper,
        mock_chat_openai,
        config,
        sample_rag_results,
        mock_env_openai_key
    ):
        """Test successful conversion to Ragas format."""
        evaluator = RagasEvaluator(config)
        
        dataset = evaluator._convert_to_ragas_format(sample_rag_results)
        
        # Verify dataset structure
        assert isinstance(dataset, Dataset)
        assert len(dataset) == 2
        assert set(dataset.column_names) == {'question', 'answer', 'contexts', 'ground_truth'}
        
        # Verify first sample
        assert dataset[0]['question'] == 'What is a cat?'
        assert dataset[0]['answer'] == 'A cat is a small carnivorous mammal.'
        assert len(dataset[0]['contexts']) == 2
        assert 'Cats are mammals' in dataset[0]['contexts'][0]
    
    @patch('src.evaluation.ragas_evaluator.ChatOpenAI')
    @patch('src.evaluation.ragas_evaluator.LangchainLLMWrapper')
    @patch('src.evaluation.ragas_evaluator.Faithfulness')
    def test_convert_contexts_string_to_list(
        self,
        mock_faithfulness,
        mock_llm_wrapper,
        mock_chat_openai,
        config,
        mock_env_openai_key,
        caplog
    ):
        """Test contexts as string is converted to list with warning."""
        evaluator = RagasEvaluator(config)
        
        # RAG result with contexts as string instead of list
        rag_results = [{
            'question': 'What is a cat?',
            'answer': 'A cat is a mammal.',
            'contexts': 'Cats are mammals.',  # String, not list!
            'ground_truth': 'A cat is a mammal.'
        }]
        
        dataset = evaluator._convert_to_ragas_format(rag_results)
        
        # Should convert to list
        assert isinstance(dataset[0]['contexts'], list)
        assert len(dataset[0]['contexts']) == 1
        
        # Should log warning
        assert "contexts was a string, converted to list" in caplog.text
    
    @patch('src.evaluation.ragas_evaluator.ChatOpenAI')
    @patch('src.evaluation.ragas_evaluator.LangchainLLMWrapper')
    @patch('src.evaluation.ragas_evaluator.Faithfulness')
    def test_convert_missing_question_raises_error(
        self,
        mock_faithfulness,
        mock_llm_wrapper,
        mock_chat_openai,
        config,
        mock_env_openai_key
    ):
        """Test missing question field raises KeyError."""
        evaluator = RagasEvaluator(config)
        
        rag_results = [{
            'answer': 'A cat is a mammal.',
            'contexts': ['Cats are mammals.'],
            # Missing 'question'
        }]
        
        with pytest.raises(KeyError, match="Missing required field 'question'"):
            evaluator._convert_to_ragas_format(rag_results)
    
    @patch('src.evaluation.ragas_evaluator.ChatOpenAI')
    @patch('src.evaluation.ragas_evaluator.LangchainLLMWrapper')
    @patch('src.evaluation.ragas_evaluator.Faithfulness')
    def test_convert_missing_ground_truth_uses_empty(
        self,
        mock_faithfulness,
        mock_llm_wrapper,
        mock_chat_openai,
        config,
        mock_env_openai_key
    ):
        """Test missing ground_truth uses empty string."""
        evaluator = RagasEvaluator(config)
        
        rag_results = [{
            'question': 'What is a cat?',
            'answer': 'A cat is a mammal.',
            'contexts': ['Cats are mammals.'],
            # No ground_truth
        }]
        
        dataset = evaluator._convert_to_ragas_format(rag_results)
        
        # Should use empty string
        assert dataset[0]['ground_truth'] == ''


class TestEvaluateRagOutputs:
    """Test RAG evaluation functionality."""
    
    @patch('src.evaluation.ragas_evaluator.evaluate')
    @patch('src.evaluation.ragas_evaluator.ChatOpenAI')
    @patch('src.evaluation.ragas_evaluator.LangchainLLMWrapper')
    @patch('src.evaluation.ragas_evaluator.Faithfulness')
    @patch('src.evaluation.ragas_evaluator.AnswerRelevancy')
    def test_evaluate_success(
        self,
        mock_answer_relevancy,
        mock_faithfulness,
        mock_llm_wrapper,
        mock_chat_openai,
        mock_evaluate,
        config,
        sample_rag_results,
        mock_env_openai_key
    ):
        """Test successful evaluation."""
        evaluator = RagasEvaluator(config)
        
        # Mock Ragas evaluate result
        mock_result = Mock()
        mock_df = pd.DataFrame({
            'question': ['What is a cat?', 'What is the capital of France?'],
            'answer': ['A cat is a mammal.', 'Paris.'],
            'contexts': [['ctx1', 'ctx2'], ['ctx3']],
            'ground_truth': ['gt1', 'gt2'],
            'faithfulness': [0.9, 0.85],
            'answer_relevancy': [0.88, 0.92]
        })
        mock_result.to_pandas.return_value = mock_df
        mock_evaluate.return_value = mock_result
        
        # Evaluate
        df = evaluator.evaluate_rag_outputs(sample_rag_results)
        
        # Verify result
        assert isinstance(df, pd.DataFrame)
        assert len(df) == 2
        assert 'faithfulness' in df.columns
        assert 'answer_relevancy' in df.columns
        assert df['faithfulness'].mean() == pytest.approx(0.875)
        
        # Verify evaluate was called
        mock_evaluate.assert_called_once()
    
    @patch('src.evaluation.ragas_evaluator.ChatOpenAI')
    @patch('src.evaluation.ragas_evaluator.LangchainLLMWrapper')
    @patch('src.evaluation.ragas_evaluator.Faithfulness')
    def test_evaluate_empty_results_raises_error(
        self,
        mock_faithfulness,
        mock_llm_wrapper,
        mock_chat_openai,
        config,
        mock_env_openai_key
    ):
        """Test empty rag_results raises ValueError."""
        evaluator = RagasEvaluator(config)
        
        with pytest.raises(ValueError, match="rag_results cannot be empty"):
            evaluator.evaluate_rag_outputs([])


class TestMetricManagement:
    """Test metric addition and retrieval."""
    
    @patch('src.evaluation.ragas_evaluator.ChatOpenAI')
    @patch('src.evaluation.ragas_evaluator.LangchainLLMWrapper')
    @patch('src.evaluation.ragas_evaluator.Faithfulness')
    @patch('src.evaluation.ragas_evaluator.ContextRecall')
    def test_add_metric_success(
        self,
        mock_context_recall,
        mock_faithfulness,
        mock_llm_wrapper,
        mock_chat_openai,
        config,
        mock_env_openai_key
    ):
        """Test successfully adding a new metric."""
        evaluator = RagasEvaluator(config)
        
        # Initially only has faithfulness
        assert 'context_recall' not in evaluator.metric_names
        
        # Add metric
        evaluator.add_metric('context_recall')
        
        # Verify added
        assert 'context_recall' in evaluator.metric_names
        assert len(evaluator.metrics) == 2
        mock_context_recall.assert_called_once()
    
    @patch('src.evaluation.ragas_evaluator.ChatOpenAI')
    @patch('src.evaluation.ragas_evaluator.LangchainLLMWrapper')
    @patch('src.evaluation.ragas_evaluator.Faithfulness')
    def test_add_metric_already_exists(
        self,
        mock_faithfulness,
        mock_llm_wrapper,
        mock_chat_openai,
        config,
        mock_env_openai_key
    ):
        """Test adding existing metric raises error."""
        evaluator = RagasEvaluator(config)
        
        with pytest.raises(ValueError, match="already exists"):
            evaluator.add_metric('faithfulness')
    
    @patch('src.evaluation.ragas_evaluator.ChatOpenAI')
    @patch('src.evaluation.ragas_evaluator.LangchainLLMWrapper')
    @patch('src.evaluation.ragas_evaluator.Faithfulness')
    def test_add_metric_unknown(
        self,
        mock_faithfulness,
        mock_llm_wrapper,
        mock_chat_openai,
        config,
        mock_env_openai_key
    ):
        """Test adding unknown metric raises error."""
        evaluator = RagasEvaluator(config)
        
        with pytest.raises(ValueError, match="Unknown metric"):
            evaluator.add_metric('nonexistent_metric')
    
    @patch('src.evaluation.ragas_evaluator.ChatOpenAI')
    @patch('src.evaluation.ragas_evaluator.LangchainLLMWrapper')
    @patch('src.evaluation.ragas_evaluator.Faithfulness')
    def test_get_available_metrics(
        self,
        mock_faithfulness,
        mock_llm_wrapper,
        mock_chat_openai,
        config,
        mock_env_openai_key
    ):
        """Test retrieving available metrics."""
        evaluator = RagasEvaluator(config)
        
        metrics = evaluator.get_available_metrics()
        
        assert isinstance(metrics, list)
        assert 'faithfulness' in metrics
        assert 'answer_relevancy' in metrics
        assert 'context_precision' in metrics
        assert 'context_recall' in metrics
