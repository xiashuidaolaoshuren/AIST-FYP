"""
Demo script for RAGTruth evaluation harness.

This script demonstrates how to use the RAGTruthEvaluator to evaluate
the hallucination detection system on the RAGTruth benchmark.

Usage:
    # Quick test with 10 samples
    python scripts/demo_ragtruth_eval.py --max-samples 10
    
    # Full test split evaluation
    python scripts/demo_ragtruth_eval.py --split test
    
    # Full test with results export
    python scripts/demo_ragtruth_eval.py --split test --save-results
"""

import sys
from pathlib import Path

# Add project root to Python path
project_root = Path(__file__).parent.parent
sys.path.insert(0, str(project_root))

import argparse
from tqdm import tqdm
from src.utils.config import Config
from src.utils.logger import setup_logger
from src.pipelines.baseline_rag import BaselineRAGPipeline
from src.verification.verifier_hub import VerifierHub
from src.verification.rule_based_aggregator import RuleBasedAggregator
from src.evaluation.ragtruth_evaluator import RAGTruthEvaluator


def main():
    """Run RAGTruth evaluation with command-line arguments."""
    # Parse arguments
    parser = argparse.ArgumentParser(
        description='Evaluate hallucination detection on RAGTruth benchmark'
    )
    parser.add_argument(
        '--config',
        type=str,
        default='config.yaml',
        help='Path to configuration file (default: config.yaml)'
    )
    parser.add_argument(
        '--split',
        type=str,
        default='test',
        choices=['train', 'test'],
        help='Dataset split to evaluate (default: test)'
    )
    parser.add_argument(
        '--max-samples',
        type=int,
        default=None,
        help='Maximum number of samples to evaluate (default: all)'
    )
    parser.add_argument(
        '--batch-size',
        type=int,
        default=10,
        help='Batch size for processing (default: 10)'
    )
    parser.add_argument(
        '--strategy',
        type=str,
        default='validation',
        choices=['development', 'validation', 'production'],
        help='Data strategy for retrieval indexes (default: validation)'
    )
    parser.add_argument(
        '--save-results',
        action='store_true',
        help='Save detailed results to JSON file'
    )
    parser.add_argument(
        '--output-path',
        type=str,
        default=None,
        help='Custom output path for results (default: auto-generated)'
    )
    parser.add_argument(
        '--ragtruth-eval-mode',
        type=str,
        default='ragtruth_eval',
        choices=['ragtruth_eval', 'normal'],
        help='Evaluation mode: ragtruth_eval uses dataset responses; normal uses pipeline responses'
    )
    
    args = parser.parse_args()
    
    # Setup logger
    logger = setup_logger(__name__)
    
    logger.info("=" * 70)
    logger.info("RAGTruth Evaluation Demo")
    logger.info("=" * 70)
    logger.info(f"Configuration: {args.config}")
    logger.info(f"Split: {args.split}")
    logger.info(f"Max samples: {args.max_samples or 'all'}")
    logger.info(f"Batch size: {args.batch_size}")
    logger.info(f"Data strategy: {args.strategy}")
    logger.info(f"Save results: {args.save_results}")
    logger.info(f"RAGTruth eval mode: {args.ragtruth_eval_mode}")
    
    # Load configuration
    setup_steps = [
        "Load configuration",
        "Initialize RAG pipeline",
        "Initialize VerifierHub",
        "Initialize RuleBasedAggregator",
        "Initialize RAGTruthEvaluator"
    ]
    setup_bar = tqdm(total=len(setup_steps), desc="Setup", unit="step")

    logger.info("\n📋 Loading configuration...")
    config = Config(args.config)
    config._config.setdefault('evaluation', {})
    config._config['evaluation'].setdefault('benchmarks', {})
    config._config['evaluation']['benchmarks'].setdefault('ragtruth', {})
    config._config['evaluation']['benchmarks']['ragtruth']['ragtruth_eval_mode'] = args.ragtruth_eval_mode
    setup_bar.set_postfix_str(setup_steps[0])
    setup_bar.update(1)
    
    # Initialize RAG pipeline
    logger.info("🔧 Initializing RAG pipeline...")
    rag_pipeline = BaselineRAGPipeline.from_config(
        args.config,
        strategy=args.strategy
    )
    logger.info("✓ RAG pipeline initialized")
    setup_bar.set_postfix_str(setup_steps[1])
    setup_bar.update(1)
    
    # Initialize VerifierHub
    logger.info("🔧 Initializing VerifierHub...")
    verifier_hub = VerifierHub(config, rag_pipeline.generator)
    logger.info("✓ VerifierHub initialized")
    setup_bar.set_postfix_str(setup_steps[2])
    setup_bar.update(1)
    
    # Initialize RuleBasedAggregator
    logger.info("🔧 Initializing RuleBasedAggregator...")
    aggregator = RuleBasedAggregator(config)
    logger.info("✓ RuleBasedAggregator initialized")
    setup_bar.set_postfix_str(setup_steps[3])
    setup_bar.update(1)
    
    # Initialize RAGTruthEvaluator
    logger.info("🔧 Initializing RAGTruthEvaluator...")
    evaluator = RAGTruthEvaluator(
        config=config,
        rag_pipeline=rag_pipeline,
        verifier_hub=verifier_hub,
        aggregator=aggregator
    )
    logger.info("✓ RAGTruthEvaluator initialized")
    setup_bar.set_postfix_str(setup_steps[4])
    setup_bar.update(1)
    setup_bar.close()
    
    # Run evaluation
    logger.info("\n🚀 Starting evaluation...")
    logger.info("=" * 70)
    
    try:
        metrics = evaluator.run_evaluation(
            split=args.split,
            max_samples=args.max_samples,
            batch_size=args.batch_size,
            save_results=args.save_results,
            output_path=args.output_path
        )
        
        logger.info("\n✅ Evaluation completed successfully!")
        
        # Print quick summary
        overall = metrics['overall']
        logger.info("\n📊 Quick Summary:")
        logger.info(f"  Samples evaluated: {overall['num_samples']}")
        logger.info(f"  Accuracy:  {overall['accuracy']:.3f}")
        logger.info(f"  Precision: {overall['precision']:.3f}")
        logger.info(f"  Recall:    {overall['recall']:.3f}")
        logger.info(f"  F1 Score:  {overall['f1']:.3f}")
        
        return 0
        
    except Exception as e:
        logger.error(f"\n❌ Evaluation failed: {str(e)}")
        import traceback
        logger.error(traceback.format_exc())
        return 1


if __name__ == '__main__':
    sys.exit(main())
