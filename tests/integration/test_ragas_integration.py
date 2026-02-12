"""
Integration test for RagasEvaluator.

This test makes REAL API calls to OpenAI and is NOT run as part of the
regular test suite. It's designed for manual testing after setting up
your OPENAI_API_KEY in the .env file.

Usage:
    1. Create .env file with OPENAI_API_KEY=sk-your-key-here
    2. Run: python -m pytest tests/integration/test_ragas_integration.py -v

Cost Warning:
    This test will make OpenAI API calls and incur costs.
    With 3 samples and 3 metrics, expect ~9 API calls.
    Estimated cost with gpt-4o-mini: ~$0.01-0.02 per run.
    
    Monitor your usage at: https://platform.openai.com/usage

Notes:
    - Tests can be slow (30-60 seconds) due to API calls
    - Tests may fail if OpenAI API is down or rate-limited
    - Results may vary slightly due to LLM non-determinism (even at temp=0)
"""

import os
import pytest
from pathlib import Path

# Load .env file if it exists
try:
    from dotenv import load_dotenv
    env_path = Path(__file__).parent.parent.parent / '.env'
    if env_path.exists():
        load_dotenv(env_path)
        print(f"✓ Loaded environment from {env_path}")
    else:
        print(f"⚠ No .env file found at {env_path}")
        print("  Please create one with OPENAI_API_KEY=sk-your-key-here")
except ImportError:
    print("⚠ python-dotenv not installed, skipping .env file loading")

from src.evaluation.ragas_evaluator import RagasEvaluator
from src.utils.config import Config


# Skip entire module if no API key
pytestmark = pytest.mark.skipif(
    not os.getenv('OPENAI_API_KEY'),
    reason="OPENAI_API_KEY not found. Set it in .env file to run integration tests."
)


@pytest.fixture(scope='module')
def config():
    """Load actual config from config.yaml."""
    config_path = Path(__file__).parent.parent.parent / 'config.yaml'
    if not config_path.exists():
        pytest.skip(f"config.yaml not found at {config_path}")
    return Config(str(config_path))


@pytest.fixture(scope='module')
def evaluator(config):
    """Create RagasEvaluator with real configuration."""
    print("\n" + "=" * 60)
    print("Initializing RagasEvaluator for integration test")
    print("This will make OpenAI API calls and incur costs")
    print("=" * 60)
    return RagasEvaluator(config)


@pytest.fixture
def sample_rag_results():
    """
    Sample RAG results for testing.
    
    These are carefully crafted to test different scenarios:
    1. High quality answer with accurate contexts
    2. Partially correct answer with mixed contexts
    3. Answer missing some information from ground truth
    """
    return [
        {
            'question': 'What is the capital of France?',
            'answer': 'The capital of France is Paris.',
            'contexts': [
                'Paris is the capital and most populous city of France.',
                'Located in northern France, Paris is a global center for art, fashion, and culture.'
            ],
            'ground_truth': 'Paris is the capital of France.'
        },
        {
            'question': 'What is photosynthesis?',
            'answer': 'Photosynthesis is the process by which plants convert sunlight into energy.',
            'contexts': [
                'Photosynthesis is a process used by plants to convert light energy into chemical energy.',
                'During photosynthesis, plants use carbon dioxide and water to produce glucose and oxygen.',
                'Chlorophyll in plant cells captures light energy for photosynthesis.'
            ],
            'ground_truth': 'Photosynthesis is the process by which plants and other organisms convert light energy into chemical energy stored in glucose, using carbon dioxide and water, and releasing oxygen as a byproduct.'
        },
        {
            'question': 'Who wrote Romeo and Juliet?',
            'answer': 'William Shakespeare wrote Romeo and Juliet.',
            'contexts': [
                'Romeo and Juliet is a tragedy written by William Shakespeare.',
                'The play was first published in 1597.'
            ],
            'ground_truth': 'William Shakespeare wrote Romeo and Juliet.'
        }
    ]


class TestRagasEvaluatorIntegration:
    """Integration tests with real API calls."""
    
    def test_evaluate_sample_rag_results(self, evaluator, sample_rag_results):
        """
        Test evaluation on sample RAG results.
        
        This makes real OpenAI API calls and verifies:
        1. Evaluation completes without errors
        2. Returns DataFrame with expected structure
        3. Metric scores are in reasonable range [0, 1]
        4. All samples have scores
        """
        print("\n" + "=" * 60)
        print("Starting Ragas evaluation (this may take 30-60 seconds)")
        print(f"Evaluating {len(sample_rag_results)} samples")
        print(f"Metrics: {evaluator.metric_names}")
        print("=" * 60)
        
        # Run evaluation
        df = evaluator.evaluate_rag_outputs(sample_rag_results)
        
        # Verify DataFrame structure
        assert len(df) == 3, f"Expected 3 rows, got {len(df)}"
        
        required_columns = {'question', 'answer', 'contexts', 'ground_truth'}
        assert required_columns.issubset(df.columns), \
            f"Missing required columns. Got: {df.columns}"
        
        # Verify metric columns exist
        for metric_name in evaluator.metric_names:
            assert metric_name in df.columns, \
                f"Metric '{metric_name}' not in results"
        
        # Verify scores are in valid range [0, 1]
        for metric_name in evaluator.metric_names:
            scores = df[metric_name]
            assert scores.min() >= 0, \
                f"{metric_name} has score < 0: {scores.min()}"
            assert scores.max() <= 1, \
                f"{metric_name} has score > 1: {scores.max()}"
            
            # No NaN values
            assert not scores.isna().any(), \
                f"{metric_name} has NaN values"
        
        # Print results for manual inspection
        print("\n" + "=" * 60)
        print("Evaluation Results")
        print("=" * 60)
        
        for idx, row in df.iterrows():
            print(f"\nSample {idx + 1}:")
            print(f"  Question: {row['question'][:60]}...")
            for metric_name in evaluator.metric_names:
                score = row[metric_name]
                print(f"  {metric_name:20s}: {score:.3f}")
        
        print("\n" + "=" * 60)
        print("Summary Statistics")
        print("=" * 60)
        for metric_name in evaluator.metric_names:
            mean_score = df[metric_name].mean()
            std_score = df[metric_name].std()
            print(f"{metric_name:20s}: mean={mean_score:.3f}, std={std_score:.3f}")
        print("=" * 60)
        
        # Basic sanity checks on scores
        # High-quality samples should generally score > 0.5
        assert df['faithfulness'].mean() > 0.5, \
            "Mean faithfulness unexpectedly low (<0.5)"
        assert df['answer_relevancy'].mean() > 0.5, \
            "Mean answer_relevancy unexpectedly low (<0.5)"
    
    def test_evaluate_with_missing_ground_truth(self, evaluator):
        """
        Test evaluation when ground_truth is missing.
        
        Some metrics (faithfulness, answer_relevancy) don't require ground_truth.
        Others (context_precision) do require it.
        """
        rag_results = [{
            'question': 'What is AI?',
            'answer': 'AI is artificial intelligence.',
            'contexts': ['Artificial intelligence is the simulation of human intelligence by machines.'],
            # No ground_truth
        }]
        
        print("\n" + "=" * 60)
        print("Testing evaluation without ground_truth")
        print("=" * 60)
        
        # Should work for metrics that don't need ground_truth
        df = evaluator.evaluate_rag_outputs(rag_results)
        
        assert len(df) == 1
        assert 'faithfulness' in df.columns or 'answer_relevancy' in df.columns
        
        print(f"✓ Evaluation succeeded without ground_truth")
        print(f"  Metrics: {list(df.columns)}")
    
    def test_add_metric_and_evaluate(self, evaluator, sample_rag_results):
        """
        Test dynamically adding a metric and evaluating.
        
        This tests the extensibility of the evaluator.
        """
        # Get current metrics
        original_metrics = evaluator.metric_names.copy()
        print(f"\nOriginal metrics: {original_metrics}")
        
        # Add new metric if not already present
        if 'context_recall' not in original_metrics:
            print("Adding 'context_recall' metric...")
            evaluator.add_metric('context_recall')
            assert 'context_recall' in evaluator.metric_names
            print(f"✓ Metric added. New metrics: {evaluator.metric_names}")
        
        # Evaluate with new metric
        print("\nEvaluating with updated metrics...")
        df = evaluator.evaluate_rag_outputs(sample_rag_results[:1])  # Just 1 sample
        
        # Verify new metric is in results
        assert 'context_recall' in df.columns
        assert not df['context_recall'].isna().any()
        
        print(f"✓ Evaluation with new metric succeeded")
        print(f"  context_recall score: {df['context_recall'].iloc[0]:.3f}")


if __name__ == '__main__':
    """
    Run integration tests directly.
    
    Usage:
        python tests/integration/test_ragas_integration.py
    """
    print("\n" + "=" * 60)
    print("Ragas Integration Test")
    print("=" * 60)
    print("\n⚠  WARNING: This will make OpenAI API calls and incur costs!")
    print("   Estimated cost: ~$0.01-0.02 per run")
    print("   Monitor usage: https://platform.openai.com/usage\n")
    
    # Check for API key
    if not os.getenv('OPENAI_API_KEY'):
        print("❌ ERROR: OPENAI_API_KEY not found in environment")
        print("\nPlease create a .env file with:")
        print("  OPENAI_API_KEY=sk-your-key-here")
        print("\nOr set it in your environment:")
        print("  export OPENAI_API_KEY=sk-your-key-here  # Linux/Mac")
        print("  set OPENAI_API_KEY=sk-your-key-here     # Windows")
        exit(1)
    
    response = input("Continue with integration test? (y/N): ")
    if response.lower() != 'y':
        print("Test cancelled.")
        exit(0)
    
    print("\nRunning pytest...")
    pytest.main([__file__, '-v', '-s'])
