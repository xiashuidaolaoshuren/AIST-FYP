"""
Unit tests for RAGTruthEvaluator.

Tests the RAGTruth evaluation harness with mock data to validate:
- Dataset loading and parsing
- Sample evaluation pipeline
- Metrics computation
- Results export
"""

import unittest
from unittest.mock import Mock, patch, MagicMock
import tempfile
import json
from pathlib import Path
import sys

# Add src to path
sys.path.insert(0, str(Path(__file__).parent.parent.parent))

from src.evaluation.ragtruth_evaluator import RAGTruthEvaluator
from src.utils.config import Config
from src.utils.data_structures import Claim, EvidenceChunk, VerifierSignal, ClaimDecision


class TestRAGTruthEvaluator(unittest.TestCase):
    """Test suite for RAGTruthEvaluator class."""
    
    def setUp(self):
        """Set up test fixtures."""
        # Create mock config
        self.config = Mock(spec=Config)
        self.config.evaluation = Mock()
        self.config.evaluation.benchmarks = Mock()
        self.config.evaluation.benchmarks.ragtruth = Mock()
        
        # Create temp directory for mock dataset
        self.temp_dir = tempfile.mkdtemp()
        self.dataset_path = Path(self.temp_dir) / 'dataset'
        self.dataset_path.mkdir(parents=True)
        
        self.config.evaluation.benchmarks.ragtruth.dataset_path = str(self.dataset_path)
        
        # Create mock components
        self.rag_pipeline = Mock()
        self.verifier_hub = Mock()
        self.verifier_hub.enabled = True
        self.aggregator = Mock()
        
        # Create mock dataset files
        self._create_mock_dataset()
    
    def tearDown(self):
        """Clean up test fixtures."""
        import shutil
        shutil.rmtree(self.temp_dir)
    
    def _create_mock_dataset(self):
        """Create mock RAGTruth dataset files."""
        # Create source_info.jsonl
        source_info = [
            {
                'source_id': 'src_1',
                'task_type': 'QA',
                'source': 'TEST',
                'source_info': {
                    'question': 'What is the capital of France?',
                    'passages': 'passage 1:Paris is the capital of France.\n\npassage 2:It is located in northern France.'
                },
                'prompt': 'Answer the question based on passages...'
            },
            {
                'source_id': 'src_2',
                'task_type': 'Summary',
                'source': 'TEST',
                'source_info': 'The Eiffel Tower is a famous landmark in Paris.',
                'prompt': 'Summarize the following...'
            }
        ]
        
        source_info_path = self.dataset_path / 'source_info.jsonl'
        with open(source_info_path, 'w', encoding='utf-8') as f:
            for item in source_info:
                f.write(json.dumps(item) + '\n')
        
        # Create response.jsonl
        responses = [
            {
                'id': 'resp_1',
                'source_id': 'src_1',
                'model': 'test-model',
                'temperature': 0.7,
                'labels': [
                    {
                        'start': 20,
                        'end': 26,
                        'text': 'Berlin',
                        'label_type': 'Evident Contradiction'
                    }
                ],
                'split': 'test',
                'quality': 'good',
                'response': 'The capital is Berlin and it is beautiful.'
            },
            {
                'id': 'resp_2',
                'source_id': 'src_2',
                'model': 'test-model',
                'temperature': 0.7,
                'labels': [],  # No hallucinations
                'split': 'test',
                'quality': 'good',
                'response': 'The Eiffel Tower is in Paris.'
            },
            {
                'id': 'resp_3',
                'source_id': 'src_1',
                'model': 'test-model',
                'temperature': 0.7,
                'labels': [],
                'split': 'train',
                'quality': 'good',
                'response': 'Paris is the capital.'
            }
        ]
        
        response_path = self.dataset_path / 'response.jsonl'
        with open(response_path, 'w', encoding='utf-8') as f:
            for item in responses:
                f.write(json.dumps(item) + '\n')
    
    def test_initialization(self):
        """Test RAGTruthEvaluator initialization."""
        evaluator = RAGTruthEvaluator(
            self.config,
            self.rag_pipeline,
            self.verifier_hub,
            self.aggregator
        )
        
        self.assertIsNotNone(evaluator)
        self.assertEqual(evaluator.config, self.config)
        self.assertEqual(evaluator.rag_pipeline, self.rag_pipeline)
        self.assertEqual(evaluator.verifier_hub, self.verifier_hub)
        self.assertEqual(evaluator.aggregator, self.aggregator)
    
    def test_initialization_missing_dataset(self):
        """Test initialization fails if dataset directory not found."""
        self.config.evaluation.benchmarks.ragtruth.dataset_path = '/nonexistent/path'
        
        with self.assertRaises(FileNotFoundError):
            RAGTruthEvaluator(
                self.config,
                self.rag_pipeline,
                self.verifier_hub,
                self.aggregator
            )
    
    def test_load_dataset_test_split(self):
        """Test loading test split from dataset."""
        evaluator = RAGTruthEvaluator(
            self.config,
            self.rag_pipeline,
            self.verifier_hub,
            self.aggregator
        )
        
        samples = evaluator._load_dataset(split='test', max_samples=None)
        
        # Should have 2 test samples
        self.assertEqual(len(samples), 2)
        
        # Check first sample (with hallucination)
        sample1 = samples[0]
        self.assertEqual(sample1['id'], 'resp_1')
        self.assertEqual(sample1['task_type'], 'QA')
        self.assertIn('capital of France', sample1['question'])
        self.assertEqual(len(sample1['gold_labels']), 1)
        
        # Check second sample (no hallucination)
        sample2 = samples[1]
        self.assertEqual(sample2['id'], 'resp_2')
        self.assertEqual(len(sample2['gold_labels']), 0)
    
    def test_load_dataset_train_split(self):
        """Test loading train split from dataset."""
        evaluator = RAGTruthEvaluator(
            self.config,
            self.rag_pipeline,
            self.verifier_hub,
            self.aggregator
        )
        
        samples = evaluator._load_dataset(split='train', max_samples=None)
        
        # Should have 1 train sample
        self.assertEqual(len(samples), 1)
        self.assertEqual(samples[0]['id'], 'resp_3')
    
    def test_load_dataset_max_samples(self):
        """Test max_samples parameter limits loaded samples."""
        evaluator = RAGTruthEvaluator(
            self.config,
            self.rag_pipeline,
            self.verifier_hub,
            self.aggregator
        )
        
        samples = evaluator._load_dataset(split='test', max_samples=1)
        
        # Should have only 1 sample
        self.assertEqual(len(samples), 1)
    
    def test_extract_question_and_contexts_qa(self):
        """Test extracting question and contexts for QA task."""
        evaluator = RAGTruthEvaluator(
            self.config,
            self.rag_pipeline,
            self.verifier_hub,
            self.aggregator
        )
        
        source = {
            'task_type': 'QA',
            'source_info': {
                'question': 'What is AI?',
                'passages': 'passage 1:AI is artificial intelligence.\n\npassage 2:It involves machine learning.'
            }
        }
        
        question, contexts = evaluator._extract_question_and_contexts(source)
        
        self.assertEqual(question, 'What is AI?')
        self.assertEqual(len(contexts), 2)
        self.assertIn('artificial intelligence', contexts[0])
        self.assertIn('machine learning', contexts[1])
    
    def test_extract_question_and_contexts_summary(self):
        """Test extracting question and contexts for Summary task."""
        evaluator = RAGTruthEvaluator(
            self.config,
            self.rag_pipeline,
            self.verifier_hub,
            self.aggregator
        )
        
        source = {
            'task_type': 'Summary',
            'source_info': 'This is a document to summarize.'
        }
        
        question, contexts = evaluator._extract_question_and_contexts(source)
        
        self.assertIn('Summarize', question)
        self.assertEqual(len(contexts), 1)
        self.assertEqual(contexts[0], 'This is a document to summarize.')
    
    def test_check_overlap_with_gold_true(self):
        """Test overlap detection with gold hallucination span."""
        evaluator = RAGTruthEvaluator(
            self.config,
            self.rag_pipeline,
            self.verifier_hub,
            self.aggregator
        )
        
        claim_text = 'capital is Berlin'
        full_response = 'The capital is Berlin and it is beautiful.'
        gold_labels = [
            {'start': 20, 'end': 26, 'text': 'Berlin'}
        ]
        
        overlaps = evaluator._check_overlap_with_gold(
            claim_text,
            full_response,
            gold_labels
        )
        
        self.assertTrue(overlaps)
    
    def test_check_overlap_with_gold_false(self):
        """Test no overlap when claim doesn't overlap gold span."""
        evaluator = RAGTruthEvaluator(
            self.config,
            self.rag_pipeline,
            self.verifier_hub,
            self.aggregator
        )
        
        claim_text = 'it is beautiful'
        full_response = 'The capital is Berlin and it is beautiful.'
        gold_labels = [
            {'start': 20, 'end': 26, 'text': 'Berlin'}
        ]
        
        overlaps = evaluator._check_overlap_with_gold(
            claim_text,
            full_response,
            gold_labels
        )
        
        self.assertFalse(overlaps)
    
    def test_compute_metrics_perfect_detection(self):
        """Test metrics computation with perfect detection."""
        evaluator = RAGTruthEvaluator(
            self.config,
            self.rag_pipeline,
            self.verifier_hub,
            self.aggregator
        )
        
        # Mock results: all correct
        results = [
            {'gold_has_hallucination': True, 'detected_hallucination': True},   # TP
            {'gold_has_hallucination': False, 'detected_hallucination': False}, # TN
            {'gold_has_hallucination': True, 'detected_hallucination': True},   # TP
            {'gold_has_hallucination': False, 'detected_hallucination': False}  # TN
        ]
        
        metrics = evaluator._compute_metrics(results)
        
        self.assertEqual(metrics['overall']['accuracy'], 1.0)
        self.assertEqual(metrics['overall']['precision'], 1.0)
        self.assertEqual(metrics['overall']['recall'], 1.0)
        self.assertEqual(metrics['overall']['f1'], 1.0)
        self.assertEqual(metrics['confusion_matrix']['true_positives'], 2)
        self.assertEqual(metrics['confusion_matrix']['true_negatives'], 2)
        self.assertEqual(metrics['confusion_matrix']['false_positives'], 0)
        self.assertEqual(metrics['confusion_matrix']['false_negatives'], 0)
    
    def test_compute_metrics_with_errors(self):
        """Test metrics computation with detection errors."""
        evaluator = RAGTruthEvaluator(
            self.config,
            self.rag_pipeline,
            self.verifier_hub,
            self.aggregator
        )
        
        results = [
            {'gold_has_hallucination': True, 'detected_hallucination': True},   # TP
            {'gold_has_hallucination': False, 'detected_hallucination': True},  # FP
            {'gold_has_hallucination': True, 'detected_hallucination': False},  # FN
            {'gold_has_hallucination': False, 'detected_hallucination': False}  # TN
        ]
        
        metrics = evaluator._compute_metrics(results)
        
        self.assertEqual(metrics['overall']['accuracy'], 0.5)
        self.assertEqual(metrics['overall']['precision'], 0.5)  # 1/(1+1)
        self.assertEqual(metrics['overall']['recall'], 0.5)     # 1/(1+1)
        self.assertEqual(metrics['confusion_matrix']['true_positives'], 1)
        self.assertEqual(metrics['confusion_matrix']['false_positives'], 1)
        self.assertEqual(metrics['confusion_matrix']['false_negatives'], 1)
        self.assertEqual(metrics['confusion_matrix']['true_negatives'], 1)
    
    def test_save_results(self):
        """Test saving evaluation results to JSON file."""
        evaluator = RAGTruthEvaluator(
            self.config,
            self.rag_pipeline,
            self.verifier_hub,
            self.aggregator
        )
        
        metrics = {
            'overall': {'accuracy': 0.9, 'f1': 0.85},
            'confusion_matrix': {'true_positives': 10},
            'statistics': {'total_samples': 20}
        }
        
        results = [
            {'sample_id': 'test_1', 'detected_hallucination': True}
        ]
        
        output_path = Path(self.temp_dir) / 'results.json'
        evaluator._save_results(metrics, results, str(output_path))
        
        # Verify file was created
        self.assertTrue(output_path.exists())
        
        # Verify content
        with open(output_path, 'r', encoding='utf-8') as f:
            saved = json.load(f)
        
        self.assertIn('metrics', saved)
        self.assertIn('sample_results', saved)
        self.assertIn('metadata', saved)
        self.assertEqual(saved['metrics']['overall']['accuracy'], 0.9)


if __name__ == '__main__':
    unittest.main()
