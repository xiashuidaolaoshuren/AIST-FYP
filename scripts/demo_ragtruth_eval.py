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
import uuid

# Add project root to Python path
project_root = Path(__file__).parent.parent
sys.path.insert(0, str(project_root))

import argparse
from tqdm import tqdm
import yaml
from src.utils.config import Config
from src.utils.logger import setup_logger
from src.pipelines.baseline_rag import BaselineRAGPipeline
from src.verification.verifier_hub import VerifierHub
from src.verification.rule_based_aggregator import RuleBasedAggregator
from src.evaluation.ragtruth_evaluator import RAGTruthEvaluator
from src.retrieval.sentence_retriever import EvidenceSentenceRetriever


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
        '--model-name',
        type=str,
        default=None,
        help='Override models.generator at runtime'
    )
    parser.add_argument(
        '--max-input-tokens',
        type=int,
        default=None,
        help='Override generation.max_input_tokens at runtime'
    )
    parser.add_argument(
        '--max-new-tokens',
        type=int,
        default=None,
        help='Override generation.max_new_tokens at runtime'
    )
    parser.add_argument(
        '--strict-logits',
        choices=['true', 'false'],
        default=None,
        help='Override verification.intrinsic.strict_logits at runtime'
    )
    parser.add_argument(
        '--temperature',
        type=float,
        default=None,
        help='Override generation.temperature at runtime'
    )
    parser.add_argument(
        '--top-p',
        type=float,
        default=None,
        help='Override generation.top_p at runtime'
    )
    parser.add_argument(
        '--do-sample',
        choices=['true', 'false'],
        default=None,
        help='Override generation.do_sample at runtime'
    )
    parser.add_argument(
        '--repetition-penalty',
        type=float,
        default=None,
        help='Override generation.repetition_penalty at runtime'
    )
    parser.add_argument(
        '--no-repeat-ngram-size',
        type=int,
        default=None,
        help='Override generation.no_repeat_ngram_size at runtime'
    )
    parser.add_argument(
        '--sanitize-meta-text',
        choices=['true', 'false'],
        default=None,
        help='Override generation.sanitize_meta_text at runtime'
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
        '--samples-per-task',
        type=int,
        default=None,
        help='Maximum number of samples per task type (overrides --max-samples when set)'
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
        '--resume',
        action='store_true',
        help='Resume from existing output JSON when available'
    )
    parser.add_argument(
        '--resume-policy',
        type=str,
        default='strict',
        choices=['strict', 'fresh-on-mismatch'],
        help='How to handle incompatible resume files (default: strict)'
    )
    parser.add_argument(
        '--ragtruth-eval-mode',
        type=str,
        default='ragtruth_eval',
        choices=['ragtruth_eval', 'normal', 'gold_context_generation'],
        help=(
            'Evaluation mode: ragtruth_eval uses benchmark responses; '
            'normal uses pipeline responses with local retrieval; '
            'gold_context_generation generates responses from benchmark gold contexts'
        )
    )
    
    args = parser.parse_args()

    # Build a runtime config so users can keep one canonical config file
    # and switch model/token settings via CLI overrides.
    with open(args.config, 'r', encoding='utf-8') as f:
        runtime_config = yaml.safe_load(f)

    if args.model_name:
        runtime_config.setdefault('models', {})
        runtime_config['models']['generator'] = args.model_name

    if args.max_input_tokens is not None:
        runtime_config.setdefault('generation', {})
        runtime_config['generation']['max_input_tokens'] = int(args.max_input_tokens)

    if args.max_new_tokens is not None:
        runtime_config.setdefault('generation', {})
        runtime_config['generation']['max_new_tokens'] = int(args.max_new_tokens)

    if args.temperature is not None:
        runtime_config.setdefault('generation', {})
        runtime_config['generation']['temperature'] = float(args.temperature)

    if args.top_p is not None:
        runtime_config.setdefault('generation', {})
        runtime_config['generation']['top_p'] = float(args.top_p)

    if args.do_sample is not None:
        runtime_config.setdefault('generation', {})
        runtime_config['generation']['do_sample'] = (args.do_sample == 'true')

    if args.repetition_penalty is not None:
        runtime_config.setdefault('generation', {})
        runtime_config['generation']['repetition_penalty'] = float(args.repetition_penalty)

    if args.no_repeat_ngram_size is not None:
        runtime_config.setdefault('generation', {})
        runtime_config['generation']['no_repeat_ngram_size'] = int(args.no_repeat_ngram_size)

    if args.sanitize_meta_text is not None:
        runtime_config.setdefault('generation', {})
        runtime_config['generation']['sanitize_meta_text'] = (args.sanitize_meta_text == 'true')

    if args.strict_logits is not None:
        runtime_config.setdefault('verification', {})
        runtime_config['verification'].setdefault('intrinsic', {})
        runtime_config['verification']['intrinsic']['strict_logits'] = (args.strict_logits == 'true')

    runtime_config_dir = Path('outputs')
    runtime_config_dir.mkdir(exist_ok=True)
    runtime_config_path = runtime_config_dir / f"runtime_config_eval_{uuid.uuid4().hex[:8]}.yaml"
    with open(runtime_config_path, 'w', encoding='utf-8') as f:
        yaml.safe_dump(runtime_config, f, sort_keys=False, allow_unicode=False)

    args.config = str(runtime_config_path)
    
    # Setup logger
    logger = setup_logger(__name__)
    
    logger.info("=" * 70)
    logger.info("RAGTruth Evaluation Demo")
    logger.info("=" * 70)
    logger.info(f"Configuration: {args.config}")
    logger.info(f"Split: {args.split}")
    logger.info(f"Max samples: {args.max_samples or 'all'}")
    logger.info(f"Samples per task: {args.samples_per_task or 'off'}")
    logger.info(f"Batch size: {args.batch_size}")
    logger.info(f"Data strategy: {args.strategy}")
    logger.info(f"Save results: {args.save_results}")
    logger.info(f"RAGTruth eval mode: {args.ragtruth_eval_mode}")
    logger.info(f"Resume policy: {args.resume_policy}")

    effective_output_path = args.output_path
    if args.save_results and effective_output_path is None:
        default_output_dir = Path('outputs/ragtruth_eval')
        default_output_dir.mkdir(parents=True, exist_ok=True)
        effective_output_path = str(default_output_dir / f'ragtruth_eval_{args.split}.json')

    resume_source = effective_output_path if args.resume else None
    if args.resume:
        logger.info(f"Resume enabled: source={resume_source}")
    
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
    selected_model = str(config.models.generator)
    configured_input_tokens = int(config.generation.get('max_input_tokens', 0) or 0)
    configured_new_tokens = int(config.generation.get('max_new_tokens', 0) or 0)
    logger.info(f"Generator model: {selected_model}")
    logger.info(f"Generation max_input_tokens: {configured_input_tokens}")
    logger.info(f"Generation max_new_tokens: {configured_new_tokens}")
    if 'Qwen/' in selected_model and configured_input_tokens < 4096:
        raise ValueError(
            "Evaluation constraint violation: Qwen models require "
            "generation.max_input_tokens >= 4096 for this project setup."
        )
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
    
    # Initialize EvidenceSentenceRetriever (optional, controlled by config)
    sr_cfg = config.get('verification', {}).get('sentence_retrieval', {})
    if not isinstance(sr_cfg, dict):
        sr_cfg = {}
    sentence_retriever = None
    sr_enabled = bool(sr_cfg.get('enabled', False))
    if args.ragtruth_eval_mode == 'ragtruth_eval' and not sr_enabled:
        raise ValueError(
            "RAGTruth strict Summary policy requires sentence retrieval in ragtruth_eval mode. "
            "Enable verification.sentence_retrieval.enabled in config or use --ragtruth-eval-mode normal."
        )
    if sr_enabled:
        index_dir_tpl = sr_cfg.get('index_dir', 'data/indexes/{dataset}_sentences/{split}')
        index_dir = index_dir_tpl.format(dataset='ragtruth', split=args.split)
        logger.info(f"🔧 Loading sentence index from {index_dir}...")
        sentence_retriever = EvidenceSentenceRetriever.from_index(
            index_dir=index_dir,
            encoder_model=str(config.models.sentence_transformer),
            device=str(getattr(config.processing, 'device', 'cpu')),
        )
        logger.info("✓ EvidenceSentenceRetriever loaded")

    # Initialize RAGTruthEvaluator
    logger.info("🔧 Initializing RAGTruthEvaluator...")
    evaluator = RAGTruthEvaluator(
        config=config,
        rag_pipeline=rag_pipeline,
        verifier_hub=verifier_hub,
        aggregator=aggregator,
        sentence_retriever=sentence_retriever,
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
            samples_per_task=args.samples_per_task,
            batch_size=args.batch_size,
            save_results=args.save_results,
            output_path=effective_output_path,
            resume_from_output=resume_source,
            resume_policy=args.resume_policy,
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
    finally:
        try:
            runtime_config_path.unlink(missing_ok=True)
        except Exception:
            pass


if __name__ == '__main__':
    sys.exit(main())
