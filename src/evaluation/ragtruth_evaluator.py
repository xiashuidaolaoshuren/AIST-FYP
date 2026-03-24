"""
RAGTruthEvaluator - Evaluation harness for RAGTruth benchmark.

This module implements an end-to-end evaluation pipeline that:
1. Loads RAGTruth dataset (questions + contexts + gold hallucination annotations)
2. Runs the full RAG pipeline to generate responses
3. Extracts and verifies claims using the VerifierHub
4. Compares detected hallucinations against gold annotations
5. Computes detection metrics (precision, recall, F1)

RAGTruth Structure:
- response.jsonl: Generated responses with annotated hallucination spans
- source_info.jsonl: Source contexts and prompts for each query

Evaluation Strategy:
- Generate responses with our RAG pipeline
- Extract atomic claims from generated text
- Verify each claim against retrieved evidence
- Label claims as Supported/Contradictory/Low Confidence
- Compare with gold hallucination span annotations
- Compute claim-level detection metrics

Example:
    >>> from src.utils.config import Config
    >>> from src.pipelines.baseline_rag import BaselineRAGPipeline
    >>> from src.verification.verifier_hub import VerifierHub
    >>> from src.verification.rule_based_aggregator import RuleBasedAggregator
    >>> 
    >>> config = Config('config.yaml')
    >>> pipeline = BaselineRAGPipeline.from_config(config)
    >>> verifier = VerifierHub(config, pipeline.generator)
    >>> aggregator = RuleBasedAggregator(config)
    >>> 
    >>> evaluator = RAGTruthEvaluator(config, pipeline, verifier, aggregator)
    >>> metrics = evaluator.run_evaluation(split='test', max_samples=100)
    >>> print(f"Detection F1: {metrics['overall']['f1']:.3f}")
    >>> print(f"Precision: {metrics['overall']['precision']:.3f}")
    >>> print(f"Recall: {metrics['overall']['recall']:.3f}")

References:
    - RAGTruth Paper: https://arxiv.org/abs/2401.00396
    - RAGTruth GitHub: https://github.com/CodingLL/RAGTruth
"""

import json
import os
import re
from types import SimpleNamespace
from pathlib import Path
from typing import Dict, List, Optional, Any, Tuple
from collections import defaultdict
from tqdm.auto import tqdm
import numpy as np
from sklearn.metrics import (
    accuracy_score,
    precision_recall_fscore_support,
    classification_report,
    confusion_matrix
)

from src.utils.config import Config
from src.utils.logger import setup_logger
from src.utils.data_structures import ClaimDecision, Claim, EvidenceChunk
from src.generation.claim_extractor import extract_claims
from src.mitigation.orchestrator import MitigationOrchestrator
from src.data_processing.text_chunker import chunk_data2txt


QA_EPISTEMIC_HEDGE_PATTERN = re.compile(
    r"\b(unable to|cannot|can't|not possible to|it is unclear|"
    r"no information|insufficient|based on the (provided|given)|"
    r"the (passage|context|document) does not|i don't know|"
    r"i cannot determine|not mentioned|unable to answer)\b",
    re.IGNORECASE,
)


class RAGTruthEvaluator:
    """
    Evaluation harness for RAGTruth hallucination benchmark.
    
    Runs the complete RAG + verification pipeline on RAGTruth dataset and
    evaluates the verifier's ability to detect hallucinations by comparing
    against gold-standard annotations.
    
    The evaluator performs claim-level evaluation:
    1. Generates responses using our RAG pipeline
    2. Extracts atomic claims from generated text
    3. Verifies claims using VerifierHub (all detectors)
    4. Aggregates signals into final decisions
    5. Compares decisions with gold hallucination spans
    6. Computes precision, recall, F1 for hallucination detection
    
    Attributes:
        config: Configuration object
        rag_pipeline: BaselineRAGPipeline instance for generation
        verifier_hub: VerifierHub instance for claim verification
        aggregator: RuleBasedAggregator instance for decision making
        logger: Logger instance
        benchmark_dir: Path to RAGTruth benchmark directory
        
    Example:
        >>> evaluator = RAGTruthEvaluator(config, pipeline, verifier, aggregator)
        >>> 
        >>> # Quick test with 50 samples
        >>> metrics = evaluator.run_evaluation(split='test', max_samples=50)
        >>> 
        >>> # Full evaluation with batching
        >>> metrics = evaluator.run_evaluation(
        ...     split='test',
        ...     batch_size=10,
        ...     save_results=True,
        ...     output_path='outputs/ragtruth_eval.json'
        ... )
    """
    
    def __init__(
        self,
        config: Config,
        rag_pipeline,
        verifier_hub,
        aggregator,
        sentence_retriever=None,
    ):
        """
        Initialize RAGTruthEvaluator with pipeline components.
        
        Args:
            config: Configuration object with evaluation settings
            rag_pipeline: BaselineRAGPipeline instance for end-to-end generation
            verifier_hub: VerifierHub instance for claim verification
            aggregator: RuleBasedAggregator instance for signal aggregation
            sentence_retriever: Optional EvidenceSentenceRetriever.  When set,
                each claim is verified against the top-k most similar sentences
                from the gold context instead of full passage chunks.
            
        Raises:
            ValueError: If config is missing required fields
            FileNotFoundError: If RAGTruth benchmark directory not found
        """
        self.config = config
        self.rag_pipeline = rag_pipeline
        self.verifier_hub = verifier_hub
        self.aggregator = aggregator
        self.sentence_retriever = sentence_retriever
        self.logger = setup_logger(__name__)

        # Read sentence retrieval top-k from config
        sr_cfg = self.config.get('verification', {}) if hasattr(self.config, 'get') else {}
        if not isinstance(sr_cfg, dict):
            sr_cfg = {}
        sr_sub = sr_cfg.get('sentence_retrieval', {})
        if not isinstance(sr_sub, dict):
            sr_sub = {}
        self.sentence_retrieval_top_k = int(sr_sub.get('top_k', 5))

        mitigation_config = self.config.get('mitigation', {})
        if not isinstance(mitigation_config, dict):
            mitigation_config = {}

        self.mitigation_enabled = bool(mitigation_config.get('enabled', False))
        self.mitigation_module_flags = self._extract_module_flags(
            mitigation_config,
            ('reranker', 'filter', 'reprompt')
        )
        verification_config = self.config.get('verification', {})
        if not isinstance(verification_config, dict):
            verification_config = {}
        self.verification_enabled = bool(verification_config.get('enabled', True))
        self.verification_module_flags = self._extract_module_flags(
            verification_config.get('modules', {}),
            ('intrinsic', 'grounded', 'nli', 'self_agreement')
        )
        self.mitigation_orchestrator = None
        
        # Get benchmark directory from config
        if hasattr(config, 'evaluation') and hasattr(config.evaluation, 'benchmarks'):
            if hasattr(config.evaluation.benchmarks, 'ragtruth'):
                benchmark_config = config.evaluation.benchmarks.ragtruth
                self.benchmark_dir = Path(getattr(benchmark_config, 'dataset_path', 'benchmark/RAGTruth/dataset'))
                self.ragtruth_eval_mode = getattr(benchmark_config, 'ragtruth_eval_mode', 'ragtruth_eval')
                self.teacher_forced_intrinsic = bool(
                    getattr(benchmark_config, 'teacher_forced_intrinsic', True)
                )
                raw_threshold = getattr(benchmark_config, 'low_confidence_ratio_threshold', 0.5)
                try:
                    self.low_confidence_ratio_threshold = float(raw_threshold)
                except (TypeError, ValueError):
                    self.low_confidence_ratio_threshold = 0.5
                raw_low_coverage_ratio = getattr(benchmark_config, 'low_coverage_ratio_threshold', 0.3)
                try:
                    self.low_coverage_ratio_threshold = float(raw_low_coverage_ratio)
                except (TypeError, ValueError):
                    self.low_coverage_ratio_threshold = 0.3
            else:
                self.benchmark_dir = Path('benchmark/RAGTruth/dataset')
                self.ragtruth_eval_mode = 'ragtruth_eval'
                self.teacher_forced_intrinsic = True
                self.low_confidence_ratio_threshold = 0.5
                self.low_coverage_ratio_threshold = 0.3
        else:
            self.benchmark_dir = Path('benchmark/RAGTruth/dataset')
            self.ragtruth_eval_mode = 'ragtruth_eval'
            self.teacher_forced_intrinsic = True
            self.low_confidence_ratio_threshold = 0.5
            self.low_coverage_ratio_threshold = 0.3

        # Per-task minimum contradictory claims to flag a sample as hallucinated.
        try:
            _bc = config.evaluation.benchmarks.ragtruth
            self.min_contradictory_count = int(
                getattr(_bc, 'min_contradictory_count_for_detection', 1)
            )
            _raw_per_task = getattr(_bc, 'per_task_min_contradictory', None)
            if isinstance(_raw_per_task, dict):
                self.per_task_min_contradictory = {
                    k: int(v) for k, v in _raw_per_task.items()
                }
            else:
                self.per_task_min_contradictory = {}
        except AttributeError:
            self.min_contradictory_count = 1
            self.per_task_min_contradictory = {}

        if isinstance(self.ragtruth_eval_mode, str):
            self.ragtruth_eval_mode = self.ragtruth_eval_mode.strip().lower()
        else:
            self.ragtruth_eval_mode = 'ragtruth_eval'

        valid_modes = {'ragtruth_eval', 'normal', 'gold_context_generation'}
        if self.ragtruth_eval_mode not in valid_modes:
            self.logger.warning(
                "Invalid ragtruth_eval_mode '%s' (expected one of %s); "
                "defaulting to 'ragtruth_eval'",
                self.ragtruth_eval_mode,
                sorted(valid_modes)
            )
            self.ragtruth_eval_mode = 'ragtruth_eval'

        self.low_confidence_ratio_threshold = float(
            np.clip(self.low_confidence_ratio_threshold, 0.0, 1.0)
        )
        self.low_coverage_ratio_threshold = float(
            np.clip(self.low_coverage_ratio_threshold, 0.0, 1.0)
        )
            
        # Validate benchmark directory exists
        if not self.benchmark_dir.exists():
            raise FileNotFoundError(
                f"RAGTruth benchmark directory not found: {self.benchmark_dir}. "
                "Please download RAGTruth dataset to benchmark/RAGTruth/dataset/"
            )
            
        self.logger.info(
            "Initialized RAGTruthEvaluator with benchmark: %s (ragtruth_eval_mode=%s, teacher_forced_intrinsic=%s, low_confidence_ratio_threshold=%.2f, low_coverage_ratio_threshold=%.2f)",
            self.benchmark_dir,
            self.ragtruth_eval_mode,
            self.teacher_forced_intrinsic,
            self.low_confidence_ratio_threshold,
            self.low_coverage_ratio_threshold
        )
        self.logger.info(
            "Effective verification config: enabled=%s, modules=%s",
            self.verification_enabled,
            self.verification_module_flags,
        )
        self.logger.info(
            "Effective mitigation config: enabled=%s, modules=%s",
            self.mitigation_enabled,
            self.mitigation_module_flags,
        )

        if self.mitigation_enabled:
            try:
                generator = getattr(self.rag_pipeline, 'generator', None)
                self.mitigation_orchestrator = MitigationOrchestrator(
                    config=config,
                    verifier_hub=self.verifier_hub,
                    aggregator=self.aggregator,
                    generator=generator,
                )
            except Exception as exc:
                self.logger.warning("Failed to initialize MitigationOrchestrator: %s", exc)

            self.logger.info(
                "Mitigation enabled for evaluator (orchestrator=%s)",
                bool(self.mitigation_orchestrator and self.mitigation_orchestrator.enabled)
            )
    
    def run_evaluation(
        self,
        split: str = 'test',
        max_samples: Optional[int] = None,
        samples_per_task: Optional[int] = None,
        max_saved_samples: Optional[int] = None,
        batch_size: int = 10,
        save_results: bool = True,
        output_path: Optional[str] = None,
        resume_from_output: Optional[str] = None,
        resume_policy: str = 'strict',
    ) -> Dict[str, Any]:
        """
        Run complete evaluation on RAGTruth benchmark.
        
        Main evaluation pipeline:
        1. Load RAGTruth dataset (source_info + responses)
        2. For each sample in batches:
           a. Extract question and context from source_info
           b. Run RAG pipeline to generate response
           c. Extract atomic claims from response
           d. Verify each claim using VerifierHub
           e. Aggregate signals into decisions
           f. Compare with gold hallucination spans
        3. Compute overall and per-class metrics
        4. Save detailed results if requested
        
        Args:
            split: Dataset split to evaluate ('train' or 'test')
            max_samples: Maximum samples to evaluate (None = all)
            samples_per_task: Maximum samples per task type; when set, takes precedence over max_samples
            max_saved_samples: Maximum sample results to persist in output JSON
            batch_size: Process samples in batches for memory efficiency
            save_results: Whether to save detailed results to file
            output_path: Path to save results JSON (auto-generated if None)
            resume_from_output: Existing output JSON path to resume from
            resume_policy: Resume behavior when existing output is incompatible:
                - strict: raise mismatch error
                - fresh-on-mismatch: log warning and start a fresh run
            
        Returns:
            Dictionary containing:
                - overall: {accuracy, precision, recall, f1}
                - per_class: {Supported: {...}, Contradictory: {...}, ...}
                - confusion_matrix: 2D array
                - sample_results: List of per-sample predictions and ground truth
                - metadata: Evaluation configuration and timing
                
        Example:
            >>> # Quick validation run
            >>> metrics = evaluator.run_evaluation(split='test', max_samples=20)
            >>> 
            >>> # Full evaluation with results export
            >>> metrics = evaluator.run_evaluation(
            ...     split='test',
            ...     batch_size=10,
            ...     save_results=True
            ... )
            >>> print(f"Hallucination Detection F1: {metrics['overall']['f1']:.3f}")
        """
        self.logger.info("=" * 70)
        self.logger.info("Starting RAGTruth Evaluation")
        self.logger.info("=" * 70)
        self.logger.info(f"Split: {split}")
        self.logger.info(f"Max samples: {max_samples or 'all'}")
        self.logger.info(f"Samples per task: {samples_per_task or 'off'}")
        self.logger.info(f"Max saved samples: {max_saved_samples or 'all'}")
        self.logger.info(f"Batch size: {batch_size}")
        self.logger.info(f"RAGTruth eval mode: {self.ragtruth_eval_mode}")
        self.logger.info(f"Teacher-forced intrinsic: {self.teacher_forced_intrinsic}")
        self.logger.info(f"Resume policy: {resume_policy}")
        if samples_per_task is not None and max_samples is not None:
            self.logger.info("Sampling precedence: using samples_per_task and ignoring max_samples")
        if resume_from_output:
            self.logger.info(f"Resume source: {resume_from_output}")

        allowed_resume_policies = {'strict', 'fresh-on-mismatch'}
        if resume_policy not in allowed_resume_policies:
            raise ValueError(
                f"Invalid resume_policy '{resume_policy}' (expected one of {sorted(allowed_resume_policies)})"
            )

        run_context = {
            'split': split,
            'max_samples': max_samples,
            'samples_per_task': samples_per_task,
            'ragtruth_eval_mode': self.ragtruth_eval_mode,
            'dataset_path': str(self.benchmark_dir.resolve()),
            'selection_fingerprint': self._build_selection_fingerprint(
                split=split,
                max_samples=max_samples,
                samples_per_task=samples_per_task,
            ),
        }
        
        # Step 1: Load dataset
        self.logger.info("\nStep 1: Loading RAGTruth dataset...")
        samples = self._load_dataset(
            split=split,
            max_samples=max_samples,
            samples_per_task=samples_per_task,
        )
        self.logger.info(f"Loaded {len(samples)} samples from RAGTruth {split} split")
        
        # Step 2: Run evaluation in batches
        self.logger.info("\nStep 2: Running evaluation pipeline...")
        all_results: List[Dict[str, Any]] = []
        sample_id_universe = {str(sample['id']) for sample in samples}
        resume_count = 0

        if resume_from_output:
            resume_path = Path(resume_from_output)
            if resume_path.exists():
                try:
                    with open(resume_path, 'r', encoding='utf-8') as f:
                        resume_payload = json.load(f)

                    resume_meta = resume_payload.get('metadata', {})
                    if not isinstance(resume_meta, dict):
                        resume_meta = {}

                    existing_fingerprint = resume_meta.get('selection_fingerprint')
                    if existing_fingerprint is not None:
                        if not isinstance(existing_fingerprint, dict):
                            raise ValueError(
                                "Resume mismatch: metadata.selection_fingerprint is not an object"
                            )
                        if existing_fingerprint != run_context['selection_fingerprint']:
                            raise ValueError(
                                "Resume mismatch: selection fingerprint differs from current run. "
                                f"existing={existing_fingerprint}, current={run_context['selection_fingerprint']}"
                            )

                    existing_results = resume_payload.get('sample_results', [])
                    if not isinstance(existing_results, list):
                        raise ValueError(f"Invalid resume file format (sample_results must be a list): {resume_path}")

                    seen_resume_ids: set[str] = set()
                    for result in existing_results:
                        sample_id = str(result.get('sample_id', ''))
                        if not sample_id:
                            raise ValueError(f"Resume file contains sample without sample_id: {resume_path}")
                        if sample_id not in sample_id_universe:
                            raise ValueError(
                                f"Resume mismatch: sample_id '{sample_id}' not found in current dataset selection"
                            )
                        if sample_id in seen_resume_ids:
                            raise ValueError(f"Resume file contains duplicate sample_id '{sample_id}': {resume_path}")
                        seen_resume_ids.add(sample_id)

                    if max_samples is not None and len(existing_results) > len(samples):
                        raise ValueError(
                            f"Resume mismatch: existing results ({len(existing_results)}) exceed current sample limit ({len(samples)})"
                        )

                    all_results.extend(existing_results)
                    resume_count = len(existing_results)
                    self.logger.info(
                        "Loaded %d existing sample results for resume from %s",
                        resume_count,
                        resume_path,
                    )
                except ValueError as exc:
                    if resume_policy == 'fresh-on-mismatch':
                        self.logger.warning(
                            "Resume payload at %s is incompatible (%s). Starting fresh run due to resume_policy=%s.",
                            resume_path,
                            str(exc),
                            resume_policy,
                        )
                        all_results = []
                        resume_count = 0
                    else:
                        raise
            else:
                self.logger.info("Resume file not found at %s, starting fresh", resume_path)

        processed_ids = {str(result['sample_id']) for result in all_results}
        samples_to_process = [sample for sample in samples if str(sample['id']) not in processed_ids]

        if resume_count:
            self.logger.info(
                "Resume progress: %d already processed, %d remaining",
                resume_count,
                len(samples_to_process),
            )
        
        # Process in batches to manage memory, with one persistent progress bar
        total_batches = (len(samples_to_process) - 1) // batch_size + 1 if samples_to_process else 0
        with tqdm(total=len(samples_to_process), desc="Evaluating samples", unit="sample") as sample_progress:
            for batch_start in range(0, len(samples_to_process), batch_size):
                batch_end = min(batch_start + batch_size, len(samples_to_process))
                batch = samples_to_process[batch_start:batch_end]

                self.logger.info(
                    f"\nProcessing batch {batch_start//batch_size + 1}/{total_batches} "
                    f"(samples {batch_start+1}-{batch_end})"
                )

                batch_results = self._evaluate_batch(batch)
                for result in batch_results:
                    all_results.append(result)
                    if save_results and output_path is not None:
                        interim_metrics = self._compute_metrics(all_results)
                        self._save_results(
                            interim_metrics,
                            all_results,
                            output_path,
                            run_context=run_context,
                            max_saved_samples=max_saved_samples,
                        )
                sample_progress.update(len(batch))
        
        # Step 3: Compute metrics
        self.logger.info("\nStep 3: Computing evaluation metrics...")
        metrics = self._compute_metrics(all_results)
        
        # Step 4: Save results if requested
        if save_results:
            if output_path is None:
                output_dir = Path('outputs/ragtruth_eval')
                output_dir.mkdir(parents=True, exist_ok=True)
                output_path = output_dir / f'ragtruth_eval_{split}.json'
            
            self._save_results(
                metrics,
                all_results,
                output_path,
                run_context=run_context,
                max_saved_samples=max_saved_samples,
            )
            self.logger.info(f"Results saved to: {output_path}")
        
        # Print summary
        self._print_summary(metrics)
        
        return metrics
    
    def _load_dataset(
        self,
        split: str = 'test',
        max_samples: Optional[int] = None,
        samples_per_task: Optional[int] = None,
    ) -> List[Dict[str, Any]]:
        """
        Load RAGTruth dataset samples with source info and gold labels.
        
        Loads both source_info.jsonl (questions + contexts) and response.jsonl
        (gold hallucination annotations), then joins them by source_id.
        
        Args:
            split: Dataset split ('train' or 'test')
            max_samples: Maximum number of samples to load (None = all)
            samples_per_task: Maximum samples per task type; when set, takes precedence over max_samples
            
        Returns:
            List of sample dictionaries containing:
                - id: Response ID
                - source_id: Source information ID
                - task_type: QA, Summary, or Data2txt
                - question: Query string (extracted from prompt)
                - contexts: List of context passages
                - gold_labels: List of hallucination spans from gold annotations
                - split: train or test
                
        Raises:
            FileNotFoundError: If dataset files not found
            ValueError: If dataset format is invalid
        """
        self.logger.debug(f"Loading RAGTruth {split} split...")
        if samples_per_task is not None and samples_per_task <= 0:
            raise ValueError("samples_per_task must be a positive integer")
        
        # Load source info (questions + contexts)
        source_info_path = self.benchmark_dir / 'source_info.jsonl'
        if not source_info_path.exists():
            raise FileNotFoundError(f"source_info.jsonl not found: {source_info_path}")
        
        source_info_map = {}
        with open(source_info_path, 'r', encoding='utf-8') as f:
            for line in f:
                item = json.loads(line)
                source_info_map[item['source_id']] = item
        
        self.logger.debug(f"Loaded {len(source_info_map)} source info records")
        
        # Load responses with gold annotations
        response_path = self.benchmark_dir / 'response.jsonl'
        if not response_path.exists():
            raise FileNotFoundError(f"response.jsonl not found: {response_path}")
        
        samples = []
        task_counts: Dict[str, int] = defaultdict(int)
        with open(response_path, 'r', encoding='utf-8') as f:
            for line in f:
                response = json.loads(line)
                
                # Filter by split
                if response['split'] != split:
                    continue
                
                # Skip low-quality samples
                if response.get('quality') != 'good':
                    continue
                
                # Get source info
                source_id = response['source_id']
                if source_id not in source_info_map:
                    self.logger.warning(f"Source ID {source_id} not found in source_info")
                    continue
                
                source = source_info_map[source_id]
                
                # Extract question and contexts based on task type
                task_type = source['task_type']
                question, contexts = self._extract_question_and_contexts(source)

                if samples_per_task is not None and task_counts[task_type] >= samples_per_task:
                    continue
                
                # Create sample
                sample = {
                    'id': response['id'],
                    'source_id': source_id,
                    'task_type': task_type,
                    'question': question,
                    'dataset_prompt': source.get('prompt', ''),
                    'contexts': contexts,
                    'gold_labels': response['labels'],  # List of hallucination spans
                    'split': split,
                    'gold_response': response['response']  # For reference
                }
                
                samples.append(sample)
                task_counts[task_type] += 1
                
                # Check if we've reached max_samples
                if samples_per_task is None and max_samples and len(samples) >= max_samples:
                    break
        
        self.logger.debug(f"Loaded {len(samples)} samples from {split} split")
        return samples
    
    def _extract_question_and_contexts(
        self,
        source: Dict[str, Any]
    ) -> Tuple[str, List[str]]:
        """
        Extract question and context passages from source info.
        
        Handles different task types (QA, Summary, Data2txt) with different
        source_info formats.
        
        Args:
            source: Source info dictionary from source_info.jsonl
            
        Returns:
            Tuple of (question_string, list_of_context_passages)
        """
        task_type = source['task_type']
        source_info = source['source_info']
        
        if task_type == 'QA':
            # QA tasks have question and passages
            question = source_info['question']
            
            # Parse passages (format: "passage 1:...\n\npassage 2:...")
            passages_text = source_info['passages']
            passages = []
            
            # Split by passage markers
            for passage in passages_text.split('\n\n'):
                passage = passage.strip()
                if passage and passage.startswith('passage '):
                    # Remove "passage N:" prefix
                    passage = passage.split(':', 1)[1].strip()
                # Remove line-leading step ordinals (e.g., "1  Preheat ...").
                if passage:
                    passage = '\n'.join(
                        re.sub(r'^\s*\d+\s+', '', line) for line in passage.splitlines()
                    ).strip()
                if passage:
                    passages.append(passage)
            
            return question, passages
        
        elif task_type == 'Summary':
            # Summary tasks: summarize a document
            question = "Summarize the following document."
            contexts = [source_info]  # Full document as single context
            return question, contexts
        
        elif task_type == 'Data2txt':
            # Data2txt: generate text from structured data
            question = "Generate a description from the following data."
            contexts = chunk_data2txt(source_info)
            return question, contexts
        
        else:
            raise ValueError(f"Unknown task type: {task_type}")
    
    def _evaluate_sample(self, sample: Dict[str, Any]) -> Dict[str, Any]:
        """
        Evaluate a single RAGTruth sample through the full pipeline.
        
        Pipeline:
        1. Run RAG with question + contexts → generate response
        2. Extract claims from generated response
        3. Verify each claim using VerifierHub
        4. Aggregate signals into decisions
        5. Determine if each claim overlaps with gold hallucination spans
        6. Compare predicted labels with gold labels
        
        Args:
            sample: Sample dictionary with question, contexts, gold_labels
            
        Returns:
            Result dictionary containing:
                - sample_id: Sample ID
                - question: Original question
                - generated_response: Our generated text
                - claims: List of extracted claims
                - predictions: List of claim decisions
                - gold_has_hallucination: Whether gold annotation has hallucinations
                - detected_hallucination: Whether we detected any hallucinations
                - claim_results: Detailed per-claim results
        """
        prepared = self._prepare_sample_for_verification(sample)
        resolved_pairs = prepared['resolved_pairs']
        default_generation_metadata = {}

        claim_decisions = []
        claim_signals = []
        verified_pairs = []
        if self.verifier_hub and self.verifier_hub.enabled and resolved_pairs:
            batch_records = []
            for pair in resolved_pairs:
                batch_records.append({
                    'claim': pair['claim'],
                    'evidence': pair['evidence'],
                    'metadata': pair.get('metadata') or default_generation_metadata,
                })

            batch_signals = self.verifier_hub.verify_claims_batch(batch_records)
            try:
                iter(batch_signals)
            except TypeError:
                batch_signals = [
                    self.verifier_hub.verify_claim(
                        pair['claim'],
                        pair['evidence'],
                        pair.get('metadata') or default_generation_metadata,
                    )
                    for pair in resolved_pairs
                ]

            for pair, signal in zip(resolved_pairs, batch_signals):
                if signal is None:
                    continue
                verified_pairs.append(pair)
                claim_signals.append(signal)
                decision = self.aggregator.aggregate(signal)
                claim_decisions.append(decision)
        else:
            verified_pairs = resolved_pairs

        return self._finalize_sample_result(
            prepared,
            claim_decisions,
            claim_signals,
            verified_pairs,
        )

    def _evaluate_batch(self, batch: List[Dict[str, Any]]) -> List[Dict[str, Any]]:
        """Evaluate one sample batch with cross-sample NLI batching."""
        prepared_samples: List[Dict[str, Any]] = []
        for sample in batch:
            try:
                prepared_samples.append(self._prepare_sample_for_verification(sample))
            except Exception as e:
                self.logger.error(f"Error preparing sample {sample.get('id')}: {str(e)}")

        all_pending_nli: List[Tuple[int, str, str]] = []
        for prepared in prepared_samples:
            resolved_pairs = prepared['resolved_pairs']
            if self.verifier_hub and self.verifier_hub.enabled and resolved_pairs:
                state = self.verifier_hub.prepare_verification_collect_nli([
                    {
                        'claim': pair['claim'],
                        'evidence': pair['evidence'],
                        'metadata': pair.get('metadata') or {},
                    }
                    for pair in resolved_pairs
                ])
                prepared['_verifier_state'] = state
                prepared['_verifier_pending_count'] = len(state.nli_pending)
                all_pending_nli.extend(state.nli_pending)
            else:
                prepared['_verifier_state'] = None
                prepared['_verifier_pending_count'] = 0

        nli_scores: List[Dict[str, float]] = []
        if self.verifier_hub and self.verifier_hub.nli_detector is not None and all_pending_nli:
            nli_scores = self.verifier_hub.nli_detector.detect_batch(
                [item[1] for item in all_pending_nli],
                [item[2] for item in all_pending_nli],
            )

        results: List[Dict[str, Any]] = []
        score_offset = 0
        for prepared in prepared_samples:
            resolved_pairs = prepared['resolved_pairs']
            state = prepared.get('_verifier_state')
            pending_count = int(prepared.get('_verifier_pending_count', 0))
            claim_decisions: List[ClaimDecision] = []
            claim_signals: List[Any] = []
            verified_pairs: List[Dict[str, Any]] = []

            if state is not None:
                sample_scores = nli_scores[score_offset:score_offset + pending_count]
                score_offset += pending_count
                batch_signals = self.verifier_hub.finalize_from_nli_scores(state, sample_scores)
                for pair, signal in zip(resolved_pairs, batch_signals):
                    if signal is None:
                        continue
                    verified_pairs.append(pair)
                    claim_signals.append(signal)
                    claim_decisions.append(self.aggregator.aggregate(signal))
            else:
                verified_pairs = resolved_pairs

            results.append(
                self._finalize_sample_result(
                    prepared,
                    claim_decisions,
                    claim_signals,
                    verified_pairs,
                )
            )

        return results

    @staticmethod
    def _is_qa_epistemic_claim(claim_text: str) -> bool:
        """Return True for QA meta/hedge claims that are not factual assertions."""
        return bool(QA_EPISTEMIC_HEDGE_PATTERN.search((claim_text or '').strip()))

    def _prepare_sample_for_verification(self, sample: Dict[str, Any]) -> Dict[str, Any]:
        """Prepare model outputs and claim-evidence pairs before verification."""
        sample_id = sample['id']
        task_type = sample.get('task_type', 'Unknown')
        task_id = sample.get('source_id')
        question = sample['question']
        dataset_prompt = sample.get('dataset_prompt', '')
        gold_labels = sample['gold_labels']
        hallucination_gold_labels = [
            label for label in gold_labels if self._is_hallucination_label(label)
        ]

        resolved_pairs: List[Dict[str, Any]] = []
        generated_response = None
        context_source = 'rag_db'
        evaluation_track = 'mitigation'

        if self.ragtruth_eval_mode == 'ragtruth_eval':
            context_source = 'gold_context'
            evaluation_track = 'verifier'
            generated_response = sample.get('gold_response', '')
            if task_type == 'QA':
                # Strip leading ordinal step numbers (e.g. "1. ", "2) ") from each line
                # so DeBERTa doesn't read the numbering as contradictory to evidence.
                generated_response = '\n'.join(
                    re.sub(r'^\s*\d+[\.\)]\s*', '', line)
                    for line in generated_response.splitlines()
                ).strip()
            evidence_chunks = self._build_evidence_from_contexts(sample.get('contexts', []))
            claims = extract_claims(
                text=generated_response,
                answer_id=str(sample_id),
                method='auto'
            )
            if task_type == 'QA' and claims:
                original_count = len(claims)
                claims = [
                    claim for claim in claims
                    if not self._is_qa_epistemic_claim(claim.text)
                ]
                filtered_count = original_count - len(claims)
                if filtered_count > 0:
                    self.logger.debug(
                        "Filtered %d QA epistemic/meta claim(s) for sample %s",
                        filtered_count,
                        sample_id,
                    )
            metadata = {
                'text': generated_response,
                'original_query': question,
                'tokens': [],
                'scores': [],
                'disable_intrinsic_uncertainty': True
            }

            if (
                self.teacher_forced_intrinsic
                and hasattr(self.rag_pipeline, 'generator')
                and hasattr(self.rag_pipeline.generator, 'score_target_with_metadata')
            ):
                try:
                    scored_metadata = self.rag_pipeline.generator.score_target_with_metadata(
                        prompt=dataset_prompt or question,
                        target_text=generated_response,
                        evidence_chunks=[] if dataset_prompt else evidence_chunks
                    )
                    scored_metadata['original_query'] = question
                    scored_metadata['disable_intrinsic_uncertainty'] = False
                    metadata = scored_metadata
                except Exception as e:
                    self.logger.warning(
                        "Teacher-forced intrinsic scoring failed for sample %s: %s. "
                        "Falling back to intrinsic-disabled metadata.",
                        sample_id,
                        str(e)
                    )

            if task_type == 'Summary' and self.sentence_retriever is None:
                raise RuntimeError(
                    "Summary samples in ragtruth_eval mode require sentence retrieval evidence. "
                    "Configure and pass a sentence retriever (prebuilt index) before running evaluation."
                )

            if self.sentence_retriever is not None and evidence_chunks:
                claim_texts = [claim.text for claim in claims]
                if hasattr(self.sentence_retriever, 'retrieve_batch'):
                    per_claim_evidence = self.sentence_retriever.retrieve_batch(
                        claim_texts,
                        str(sample_id),
                        self.sentence_retrieval_top_k,
                    )
                else:
                    per_claim_evidence = [
                        self.sentence_retriever.retrieve(
                            claim_text, str(sample_id), self.sentence_retrieval_top_k
                        )
                        for claim_text in claim_texts
                    ]

                for idx, claim in enumerate(claims):
                    per_claim_ev = (
                        per_claim_evidence[idx] if idx < len(per_claim_evidence) else []
                    )
                    if task_type == 'Summary' and not per_claim_ev:
                        raise RuntimeError(
                            f"Summary sample {sample_id} returned no sentence evidence for claim '{claim.text[:120]}'. "
                            "Strict index-only policy forbids fallback to full gold context."
                        )
                    resolved_pairs.append({
                        'claim': claim,
                        'evidence': per_claim_ev if per_claim_ev else evidence_chunks,
                        'metadata': metadata,
                    })
            else:
                for claim in claims:
                    if not evidence_chunks:
                        continue
                    resolved_pairs.append({
                        'claim': claim,
                        'evidence': evidence_chunks,
                        'metadata': metadata,
                    })
        else:
            if self.ragtruth_eval_mode == 'normal':
                context_source = 'rag_db'
                rag_result = self.rag_pipeline.run(query=question, top_k=5)
            elif self.ragtruth_eval_mode == 'gold_context_generation':
                context_source = 'gold_context'
                rag_result = self._run_gold_context_generation(sample)
            else:
                raise ValueError(f"Unsupported ragtruth_eval_mode: {self.ragtruth_eval_mode}")

            generated_response = rag_result['draft_response']
            claim_evidence_pairs = rag_result.get('claim_evidence_pairs', [])

            claim_map = {}
            for entry in rag_result.get('claims_by_sub_answer', []):
                for claim in entry.get('claims', []):
                    claim_obj = Claim(**claim) if isinstance(claim, dict) else claim
                    if claim_obj is not None:
                        claim_map[claim_obj.claim_id] = claim_obj

            sub_answer_metadata = []
            if isinstance(rag_result.get('generator_metadata'), dict):
                sub_answer_metadata = rag_result['generator_metadata'].get('sub_answer_metadata', [])
            for pair in claim_evidence_pairs:
                claim = pair.get('claim') if isinstance(pair, dict) else None
                if claim is None and isinstance(pair, dict):
                    claim = claim_map.get(pair.get('claim_id'))
                if isinstance(claim, dict):
                    claim = Claim(**claim)

                evidence = None
                if isinstance(pair, dict) and 'evidence' in pair:
                    evidence = pair.get('evidence')
                elif isinstance(pair, dict):
                    evidence_spans = pair.get('evidence_spans', [])
                    evidence_list = []
                    for span in evidence_spans:
                        if isinstance(span, EvidenceChunk):
                            evidence_list.append(span)
                        elif isinstance(span, dict):
                            evidence_list.append(EvidenceChunk(**span))
                    evidence = evidence_list

                metadata = None
                if claim is not None:
                    for entry in sub_answer_metadata:
                        span = entry.get('char_span')
                        if (
                            isinstance(span, list)
                            and len(span) == 2
                            and claim.answer_char_span[0] >= span[0]
                            and claim.answer_char_span[1] <= span[1]
                        ):
                            metadata = entry.get('metadata')
                            break

                if claim is None or not evidence:
                    self.logger.warning("Skipping claim-evidence pair due to missing claim or evidence")
                    continue

                resolved_pairs.append({'claim': claim, 'evidence': evidence, 'metadata': metadata})

        return {
            'sample': sample,
            'sample_id': sample_id,
            'task_type': task_type,
            'task_id': task_id,
            'question': question,
            'generated_response': generated_response,
            'hallucination_gold_labels': hallucination_gold_labels,
            'context_source': context_source,
            'evaluation_track': evaluation_track,
            'resolved_pairs': resolved_pairs,
        }

    def _finalize_sample_result(
        self,
        prepared: Dict[str, Any],
        claim_decisions: List[ClaimDecision],
        claim_signals: List[Any],
        verified_pairs: List[Dict[str, Any]],
    ) -> Dict[str, Any]:
        """Finalize sample outputs and metric labels after verification."""
        sample_id = prepared['sample_id']
        task_type = prepared.get('task_type', 'Unknown')
        task_id = prepared.get('task_id')
        question = prepared['question']
        generated_response = prepared['generated_response']
        resolved_pairs = prepared['resolved_pairs']
        hallucination_gold_labels = prepared['hallucination_gold_labels']
        context_source = prepared['context_source']
        evaluation_track = prepared['evaluation_track']

        mitigation_actions = []
        filtered_response = generated_response
        removed_count = 0
        mitigation_runtime_enabled = bool(
            self.mitigation_orchestrator and self.mitigation_orchestrator.enabled
        )
        mitigation_applied = False
        if mitigation_runtime_enabled and resolved_pairs:
            mitigation_result = self.mitigation_orchestrator.apply(
                query=question,
                answer_text=generated_response,
                claim_records=resolved_pairs,
                objective_override='ragtruth',
                precomputed_verification=(claim_signals, claim_decisions),
            )
            mitigation_actions = mitigation_result.get('actions', [])
            filtered_response = mitigation_result.get('final_answer', generated_response)
            removed_count = mitigation_result.get('filtered_claim_count', 0)
            resolved_pairs = mitigation_result.get('claim_records', resolved_pairs)
            claim_decisions = mitigation_result.get('decisions', claim_decisions)
            claim_signals = mitigation_result.get('signals', claim_signals)
            verified_pairs = resolved_pairs
            mitigation_applied = bool(mitigation_actions) or removed_count > 0 or filtered_response != generated_response
        
        # Step 3: Compare with gold annotations
        gold_has_hallucination = len(hallucination_gold_labels) > 0
        
        # Determine if we detected any hallucinations
        contradictory_count = len([
            d for d in claim_decisions
            if d.status == 'Contradictory'
        ])
        low_confidence_count = len([
            d for d in claim_decisions
            if d.status == 'Low Confidence'
        ])
        low_confidence_ratio = (
            low_confidence_count / len(claim_decisions)
            if claim_decisions else 0.0
        )
        low_coverage_count = len([
            d for d in claim_decisions
            if d.confidence.get('coverage_score', 1.0) < 0.5
        ])
        low_coverage_ratio = (
            low_coverage_count / len(claim_decisions)
            if claim_decisions else 0.0
        )
        detected_hallucination = (
            contradictory_count >= self.per_task_min_contradictory.get(
                task_type, self.min_contradictory_count
            )
            or (
                low_confidence_ratio >= self.low_confidence_ratio_threshold
                and low_coverage_ratio >= self.low_coverage_ratio_threshold
            )
        )
        
        # Detailed per-claim analysis
        claim_results = []
        for idx, decision in enumerate(claim_decisions):
            if idx >= len(verified_pairs):
                break
            claim_text = verified_pairs[idx]['claim'].text
            evidence_items = verified_pairs[idx].get('evidence', [])
            
            # Check if this claim overlaps with any gold hallucination span
            overlaps_gold = self._check_overlap_with_gold(
                claim_text,
                generated_response,
                hallucination_gold_labels
            )
            
            claim_results.append({
                'claim_text': claim_text,
                'predicted_status': decision.status,
                'confidence': decision.confidence,
                'overlaps_gold_hallucination': overlaps_gold,
                'top_k_evidences': self._serialize_evidences(evidence_items)
            })
        
        return {
            'sample_id': sample_id,
            'task_type': task_type,
            'task_id': task_id,
            'question': question,
            'generated_response': generated_response,
            'response_after_mitigation': filtered_response,
            'ragtruth_eval_mode': self.ragtruth_eval_mode,
            'evaluation_track': evaluation_track,
            'context_source': context_source,
            'num_claims': len(claim_decisions),
            'predictions': [d.status for d in claim_decisions],
            'gold_has_hallucination': gold_has_hallucination,
            'detected_hallucination': detected_hallucination,
            'contradictory_count': contradictory_count,
            'low_confidence_count': low_confidence_count,
            'low_confidence_ratio': low_confidence_ratio,
            'low_coverage_count': low_coverage_count,
            'low_coverage_ratio': low_coverage_ratio,
            'mitigation_enabled': mitigation_runtime_enabled,
            'mitigation_applied': mitigation_applied,
            'mitigation_actions': mitigation_actions,
            'filtered_claim_count': removed_count,
            'claim_results': claim_results
        }

    def _run_gold_context_generation(self, sample: Dict[str, Any]) -> Dict[str, Any]:
        """
        Generate a response using benchmark-provided gold contexts as evidence.

        This path bypasses retrieval so mitigation quality can be evaluated
        independently from local index coverage.
        """
        question = sample.get('question', '')
        evidence_chunks = self._build_evidence_from_contexts(sample.get('contexts', []))
        generator = getattr(self.rag_pipeline, 'generator', None)
        if generator is None:
            raise ValueError("RAG pipeline has no generator; cannot run gold_context_generation mode")

        generation_params = {
            'max_new_tokens': 256,
            'temperature': 0.7,
            'top_p': 0.9,
            'do_sample': True,
        }

        pipeline_config = getattr(self.rag_pipeline, 'config', None)
        if pipeline_config is not None:
            generation_config = getattr(pipeline_config, 'generation', None)
            if generation_config is not None:
                generation_params = {
                    'max_new_tokens': getattr(generation_config, 'max_new_tokens', 256),
                    'temperature': getattr(generation_config, 'temperature', 0.7),
                    'top_p': getattr(generation_config, 'top_p', 0.9),
                    'do_sample': getattr(generation_config, 'do_sample', True),
                }

        split_enabled = True
        processing_config = getattr(pipeline_config, 'processing', None)
        query_split_config = getattr(processing_config, 'query_split', None) if processing_config is not None else None
        if query_split_config is not None:
            split_enabled = bool(getattr(query_split_config, 'enabled', True))

        if split_enabled and hasattr(self.rag_pipeline, '_split_query_by_questions'):
            sub_queries = self.rag_pipeline._split_query_by_questions(question)
        else:
            sub_queries = [{'text': question.strip() if question else '', 'sub_query_id': 0}]

        if not sub_queries:
            sub_queries = [{'text': question.strip() if question else '', 'sub_query_id': 0}]

        if not evidence_chunks:
            self.logger.warning(
                "Sample %s has no gold contexts; generating without evidence.",
                sample.get('id')
            )

        combined_response_parts: List[str] = []
        all_claims = []
        claims_by_sub_answer = []
        sub_answer_metadata = []

        for sub_query_data in sub_queries:
            sub_query_text = sub_query_data.get('text', '')
            sub_query_id = sub_query_data.get('sub_query_id', 0)

            generation_output = generator.generate_with_metadata(
                prompt=sub_query_text,
                evidence_chunks=evidence_chunks,
                **generation_params
            )
            generation_output['original_query'] = sub_query_text

            generated_text = generation_output.get('text', '')
            sub_claims = extract_claims(text=generated_text, method='auto')

            char_start = len(' '.join(combined_response_parts) + (' ' if combined_response_parts else ''))
            combined_response_parts.append(generated_text)
            char_end = len(' '.join(combined_response_parts))

            for claim in sub_claims:
                original_span = claim.answer_char_span
                claim.answer_char_span = [
                    original_span[0] + char_start,
                    original_span[1] + char_start,
                ]

            claims_by_sub_answer.append({
                'sub_answer_id': sub_query_id,
                'sub_text': generated_text,
                'sub_query': sub_query_text,
                'claims': sub_claims,
            })
            sub_answer_metadata.append({
                'sub_answer_id': sub_query_id,
                'char_span': [char_start, char_end],
                'sub_query': sub_query_text,
                'metadata': generation_output,
            })
            all_claims.extend(sub_claims)

        combined_response = ' '.join(combined_response_parts)
        if self.sentence_retriever is not None and evidence_chunks:
            ctx_index = self.sentence_retriever.build_context_index_from_chunks(
                evidence_chunks
            )
            claim_evidence_pairs = []
            claim_texts = [claim.text for claim in all_claims]
            if hasattr(self.sentence_retriever, 'retrieve_from_index_batch'):
                per_claim_evidence = self.sentence_retriever.retrieve_from_index_batch(
                    claim_texts,
                    ctx_index,
                    self.sentence_retrieval_top_k,
                )
            else:
                per_claim_evidence = [
                    self.sentence_retriever.retrieve_from_index(
                        claim_text, ctx_index, self.sentence_retrieval_top_k
                    )
                    for claim_text in claim_texts
                ]

            for idx, claim in enumerate(all_claims):
                per_claim_ev = (
                    per_claim_evidence[idx] if idx < len(per_claim_evidence) else []
                )
                claim_evidence_pairs.append({
                    'claim': claim,
                    'evidence': per_claim_ev if per_claim_ev else evidence_chunks,
                })
        else:
            claim_evidence_pairs = [
                {
                    'claim': claim,
                    'evidence': evidence_chunks,
                }
                for claim in all_claims
                if evidence_chunks
            ]

        return {
            'query': question,
            'draft_response': combined_response,
            'sub_answers': [{'text': text} for text in combined_response_parts],
            'claims_by_sub_answer': claims_by_sub_answer,
            'claim_evidence_pairs': claim_evidence_pairs,
            'generator_metadata': {
                'sub_answer_metadata': sub_answer_metadata,
            },
            'retrieval_metadata': {
                'context_source': 'gold_context',
                'retrieved_count': len(evidence_chunks),
            },
        }

    def _build_rerank_signal_map(
        self,
        signal: Any,
        evidence_items: List[EvidenceChunk]
    ) -> Dict[str, Any]:
        """
        Build doc_id#sent_id -> signal mapping for EvidenceReRanker.

        Supports both per-chunk verifier output and aggregate-only signals.
        """
        if signal is None:
            return {}

        signal_map: Dict[str, Any] = {}
        per_chunk_signals = getattr(signal, 'per_chunk_signals', None) or []

        for item in per_chunk_signals:
            if not isinstance(item, dict):
                continue
            doc_id = item.get('doc_id')
            sent_id = item.get('sent_id')
            if doc_id is None or sent_id is None:
                continue
            nli = item.get('nli', {}) or {}
            coverage = item.get('coverage', {}) or {}
            if 'entailment' not in nli and 'entail' in nli:
                nli = {**nli, 'entailment': nli.get('entail', 0.0)}
            signal_map[f"{doc_id}#{sent_id}"] = SimpleNamespace(
                nli=nli,
                coverage=coverage
            )

        if signal_map:
            return signal_map

        base_nli = getattr(signal, 'nli', {}) or {}
        if 'entailment' not in base_nli and 'entail' in base_nli:
            base_nli = {**base_nli, 'entailment': base_nli.get('entail', 0.0)}
        base_coverage = getattr(signal, 'coverage', {}) or {}

        for evidence in evidence_items:
            signal_map[f"{evidence.doc_id}#{evidence.sent_id}"] = SimpleNamespace(
                nli=base_nli,
                coverage=base_coverage
            )

        return signal_map

    def _build_evidence_from_contexts(self, contexts: List[str]) -> List[EvidenceChunk]:
        """
        Convert RAGTruth contexts into EvidenceChunk objects.
        """
        evidence_chunks = []
        for idx, context in enumerate(contexts):
            if not context:
                continue
            evidence_chunks.append(
                EvidenceChunk(
                    doc_id=f"ragtruth_context_{idx}",
                    sent_id=idx,
                    text=context,
                    char_start=0,
                    char_end=len(context),
                    score_dense=1.0,
                    rank=idx
                )
            )
        return evidence_chunks

    def _serialize_evidences(self, evidence_items: List[Any]) -> List[Dict[str, Any]]:
        """Serialize evidence items for JSON export in claim-level results."""
        serialized = []
        for evidence in evidence_items:
            if isinstance(evidence, EvidenceChunk):
                serialized.append({
                    'doc_id': evidence.doc_id,
                    'sent_id': evidence.sent_id,
                    'text': evidence.text,
                    'rank': evidence.rank,
                    'score_dense': evidence.score_dense,
                    'score_bm25': evidence.score_bm25,
                    'score_hybrid': evidence.score_hybrid,
                    'source': evidence.source,
                    'version': evidence.version
                })
            elif isinstance(evidence, dict):
                serialized.append({
                    'doc_id': evidence.get('doc_id', ''),
                    'sent_id': evidence.get('sent_id', -1),
                    'text': evidence.get('text', ''),
                    'rank': evidence.get('rank', None),
                    'score_dense': evidence.get('score_dense', None),
                    'score_bm25': evidence.get('score_bm25', None),
                    'score_hybrid': evidence.get('score_hybrid', None),
                    'source': evidence.get('source', None),
                    'version': evidence.get('version', None)
                })
        return serialized
    
    def _check_overlap_with_gold(
        self,
        claim_text: str,
        full_response: str,
        gold_labels: List[Dict[str, Any]]
    ) -> bool:
        """
        Check if a claim overlaps with any gold hallucination span.
        
        Finds the claim's position in the full response, then checks if
        it overlaps with any annotated hallucination span.
        
        Args:
            claim_text: Text of the claim
            full_response: Complete generated response
            gold_labels: List of gold hallucination spans with start/end positions
            
        Returns:
            True if claim overlaps with any gold hallucination span
        """
        # Find claim position in response
        claim_start = full_response.find(claim_text)
        if claim_start == -1:
            # Claim not found in response (should not happen)
            return False
        
        claim_end = claim_start + len(claim_text)
        
        # Check overlap with each gold span
        for label in gold_labels:
            gold_start = label['start']
            gold_end = label['end']
            
            # Check if ranges overlap
            # Overlap occurs if: claim_start < gold_end AND claim_end > gold_start
            if claim_start < gold_end and claim_end > gold_start:
                return True
        
        return False

    def _is_hallucination_label(self, label: Dict[str, Any]) -> bool:
        """
        Decide whether a gold label should count as hallucination.

        `implicit_true=True` labels are factually correct statements not mentioned
        in context; those are excluded from hallucination-positive targets.
        """
        if not isinstance(label, dict):
            return False
        implicit_true = label.get('implicit_true', False)
        if isinstance(implicit_true, str):
            implicit_true = implicit_true.strip().lower() == 'true'
        return not bool(implicit_true)
    
    def _compute_metrics(
        self,
        results: List[Dict[str, Any]]
    ) -> Dict[str, Any]:
        """
        Compute evaluation metrics from sample results.
        
        Calculates:
        - Overall accuracy: Correct hallucination detection rate
        - Precision: Of detected hallucinations, how many were correct
        - Recall: Of gold hallucinations, how many did we detect
        - F1: Harmonic mean of precision and recall
        
        Args:
            results: List of evaluation results from _evaluate_sample
            
        Returns:
            Metrics dictionary with overall and per-class statistics
        """
        # Extract predictions and ground truth
        y_true = [r['gold_has_hallucination'] for r in results]
        y_pred = [r['detected_hallucination'] for r in results]
        
        # Convert boolean to binary labels
        y_true_binary = [1 if x else 0 for x in y_true]
        y_pred_binary = [1 if x else 0 for x in y_pred]
        
        # Compute binary classification metrics
        accuracy = accuracy_score(y_true_binary, y_pred_binary)
        precision, recall, f1, _ = precision_recall_fscore_support(
            y_true_binary,
            y_pred_binary,
            average='binary',
            zero_division=0
        )
        
        # Confusion matrix (force labels to ensure 2x2 output even if one class present)
        cm = confusion_matrix(y_true_binary, y_pred_binary, labels=[0, 1])

        # Per-class metrics (tn, fp, fn, tp)
        tn, fp, fn, tp = cm.ravel()
        
        total_claims = sum(int(r.get('num_claims', 0)) for r in results)
        total_claim_hallucinations = sum(int(r.get('contradictory_count', 0)) for r in results)
        total_low_confidence_claims = sum(int(r.get('low_confidence_count', 0)) for r in results)

        per_task: Dict[str, Any] = {}
        task_types = sorted({str(r.get('task_type', 'Unknown')) for r in results})
        for task_type in task_types:
            task_results = [r for r in results if str(r.get('task_type', 'Unknown')) == task_type]
            if not task_results:
                continue

            y_true_task = [1 if r['gold_has_hallucination'] else 0 for r in task_results]
            y_pred_task = [1 if r['detected_hallucination'] else 0 for r in task_results]

            task_accuracy = accuracy_score(y_true_task, y_pred_task)
            task_precision, task_recall, task_f1, _ = precision_recall_fscore_support(
                y_true_task,
                y_pred_task,
                average='binary',
                zero_division=0
            )
            task_cm = confusion_matrix(y_true_task, y_pred_task, labels=[0, 1])
            task_tn, task_fp, task_fn, task_tp = task_cm.ravel()

            task_total_claims = sum(int(r.get('num_claims', 0)) for r in task_results)
            task_detected_claim_hallucinations = sum(
                int(r.get('contradictory_count', 0)) for r in task_results
            )
            task_detected_low_confidence_claims = sum(
                int(r.get('low_confidence_count', 0)) for r in task_results
            )

            per_task[task_type] = {
                'accuracy': float(task_accuracy),
                'precision': float(task_precision),
                'recall': float(task_recall),
                'f1': float(task_f1),
                'num_samples': len(task_results),
                'confusion_matrix': {
                    'true_negatives': int(task_tn),
                    'false_positives': int(task_fp),
                    'false_negatives': int(task_fn),
                    'true_positives': int(task_tp)
                },
                'statistics': {
                    'total_samples': len(task_results),
                    'gold_hallucinations': int(sum(y_true_task)),
                    'detected_hallucinations': int(sum(y_pred_task)),
                    'correct_detections': int(task_tp),
                    'missed_hallucinations': int(task_fn),
                    'false_alarms': int(task_fp),
                    'total_claims': int(task_total_claims),
                    'detected_claim_hallucinations': int(task_detected_claim_hallucinations),
                    'detected_low_confidence_claims': int(task_detected_low_confidence_claims),
                    'avg_claims_per_sample': float(task_total_claims / len(task_results)) if task_results else 0.0,
                    'avg_claim_hallucinations_per_sample': (
                        float(task_detected_claim_hallucinations / len(task_results)) if task_results else 0.0
                    )
                }
            }

        metrics = {
            'overall': {
                'accuracy': float(accuracy),
                'precision': float(precision),
                'recall': float(recall),
                'f1': float(f1),
                'num_samples': len(results)
            },
            'per_task': per_task,
            'confusion_matrix': {
                'true_negatives': int(tn),
                'false_positives': int(fp),
                'false_negatives': int(fn),
                'true_positives': int(tp)
            },
            'statistics': {
                'total_samples': len(results),
                'gold_hallucinations': sum(y_true_binary),
                'detected_hallucinations': sum(y_pred_binary),
                'correct_detections': int(tp),
                'missed_hallucinations': int(fn),
                'false_alarms': int(fp),
                'total_claims': int(total_claims),
                'detected_claim_hallucinations': int(total_claim_hallucinations),
                'detected_low_confidence_claims': int(total_low_confidence_claims),
                'avg_claims_per_sample': float(total_claims / len(results)) if results else 0.0,
                'avg_claim_hallucinations_per_sample': float(total_claim_hallucinations / len(results)) if results else 0.0
            },
            'definitions': {
                'sample_hallucination': (
                    'Sample is hallucinated when at least one Contradictory claim exists, '
                    'or when low_confidence_ratio >= threshold and low_coverage_ratio >= threshold.'
                ),
                'claim_hallucination': 'Claim is counted as hallucinated when predicted status is Contradictory.'
            }
        }
        
        return metrics
    
    def _save_results(
        self,
        metrics: Dict[str, Any],
        results: List[Dict[str, Any]],
        output_path: str,
        run_context: Optional[Dict[str, Any]] = None,
        max_saved_samples: Optional[int] = None,
    ) -> None:
        """
        Save evaluation metrics and detailed results to JSON file.
        
        Args:
            metrics: Computed metrics dictionary
            results: List of per-sample results
            output_path: Path to output JSON file
        """
        if max_saved_samples is not None:
            max_saved_samples = max(0, int(max_saved_samples))
            saved_results = results[:max_saved_samples]
        else:
            saved_results = results
        was_truncated = len(saved_results) < len(results)

        output = {
            'metrics': metrics,
            'sample_results': saved_results,
            'metadata': {
                'evaluator': 'RAGTruthEvaluator',
                'num_samples': len(results),
                'benchmark': 'RAGTruth',
                'ragtruth_eval_mode': self.ragtruth_eval_mode,
                'unique_tasks': sorted({str(r.get('task_type', 'Unknown')) for r in results})
            }
        }
        if max_saved_samples is not None:
            output['metadata']['sample_results_truncated'] = was_truncated
            output['metadata']['sample_results_limit'] = max_saved_samples
        if isinstance(run_context, dict):
            output['metadata']['selection_fingerprint'] = run_context.get('selection_fingerprint')
            output['metadata']['split'] = run_context.get('split')
            output['metadata']['max_samples'] = run_context.get('max_samples')
            output['metadata']['samples_per_task'] = run_context.get('samples_per_task')
            output['metadata']['dataset_path'] = run_context.get('dataset_path')
        
        output_path = Path(output_path)
        output_path.parent.mkdir(parents=True, exist_ok=True)
        
        with open(output_path, 'w', encoding='utf-8') as f:
            json.dump(output, f, indent=2, ensure_ascii=False)
        
        self.logger.info(f"Saved evaluation results to: {output_path}")

    def _build_selection_fingerprint(
        self,
        split: str,
        max_samples: Optional[int],
        samples_per_task: Optional[int],
    ) -> Dict[str, Any]:
        """Build a deterministic selection fingerprint for resume compatibility checks."""
        return {
            'split': split,
            'max_samples': max_samples,
            'samples_per_task': samples_per_task,
            'ragtruth_eval_mode': self.ragtruth_eval_mode,
            'dataset_path': str(self.benchmark_dir.resolve()),
            'verification_enabled': self.verification_enabled,
            'verification_modules': self.verification_module_flags,
            'mitigation_enabled': self.mitigation_enabled,
            'mitigation_modules': self.mitigation_module_flags,
        }

    @staticmethod
    def _extract_module_flags(module_config: Any, module_names: tuple[str, ...]) -> Dict[str, bool]:
        """Extract deterministic enabled/disabled flags for configured modules."""
        if not isinstance(module_config, dict):
            module_config = {}

        flags: Dict[str, bool] = {}
        for name in module_names:
            raw_value = module_config.get(name, False)
            if isinstance(raw_value, dict):
                flags[name] = bool(raw_value.get('enabled', False))
            else:
                flags[name] = bool(raw_value)
        return flags
    
    def _print_summary(self, metrics: Dict[str, Any]) -> None:
        """
        Print evaluation summary to console.
        
        Args:
            metrics: Computed metrics dictionary
        """
        overall = metrics['overall']
        stats = metrics['statistics']
        cm = metrics['confusion_matrix']
        
        self.logger.info("\n" + "=" * 70)
        self.logger.info("RAGTruth Evaluation Summary")
        self.logger.info("=" * 70)
        
        self.logger.info(f"\n📊 Overall Metrics:")
        self.logger.info(f"  Accuracy:  {overall['accuracy']:.3f}")
        self.logger.info(f"  Precision: {overall['precision']:.3f}")
        self.logger.info(f"  Recall:    {overall['recall']:.3f}")
        self.logger.info(f"  F1 Score:  {overall['f1']:.3f}")
        
        self.logger.info(f"\n📈 Statistics:")
        self.logger.info(f"  Total Samples:          {stats['total_samples']}")
        self.logger.info(f"  Gold Hallucinations:    {stats['gold_hallucinations']}")
        self.logger.info(f"  Detected Hallucinations: {stats['detected_hallucinations']}")
        self.logger.info(f"  Correct Detections:     {stats['correct_detections']}")
        self.logger.info(f"  Missed Hallucinations:  {stats['missed_hallucinations']}")
        self.logger.info(f"  False Alarms:           {stats['false_alarms']}")
        
        self.logger.info(f"\n🎯 Confusion Matrix:")
        self.logger.info(f"  True Negatives:  {cm['true_negatives']}")
        self.logger.info(f"  False Positives: {cm['false_positives']}")
        self.logger.info(f"  False Negatives: {cm['false_negatives']}")
        self.logger.info(f"  True Positives:  {cm['true_positives']}")
        
        self.logger.info("\n" + "=" * 70)
