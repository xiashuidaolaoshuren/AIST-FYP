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
from pathlib import Path
from typing import Dict, List, Optional, Any, Tuple
from collections import defaultdict
from tqdm import tqdm
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
        aggregator
    ):
        """
        Initialize RAGTruthEvaluator with pipeline components.
        
        Args:
            config: Configuration object with evaluation settings
            rag_pipeline: BaselineRAGPipeline instance for end-to-end generation
            verifier_hub: VerifierHub instance for claim verification
            aggregator: RuleBasedAggregator instance for signal aggregation
            
        Raises:
            ValueError: If config is missing required fields
            FileNotFoundError: If RAGTruth benchmark directory not found
        """
        self.config = config
        self.rag_pipeline = rag_pipeline
        self.verifier_hub = verifier_hub
        self.aggregator = aggregator
        self.logger = setup_logger(__name__)
        
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

        if isinstance(self.ragtruth_eval_mode, str):
            self.ragtruth_eval_mode = self.ragtruth_eval_mode.strip().lower()
        else:
            self.ragtruth_eval_mode = 'ragtruth_eval'

        if self.ragtruth_eval_mode not in {'ragtruth_eval', 'normal'}:
            self.logger.warning(
                "Invalid ragtruth_eval_mode '%s' (expected 'ragtruth_eval' or 'normal'); "
                "defaulting to 'ragtruth_eval'",
                self.ragtruth_eval_mode
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
    
    def run_evaluation(
        self,
        split: str = 'test',
        max_samples: Optional[int] = None,
        batch_size: int = 10,
        save_results: bool = True,
        output_path: Optional[str] = None
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
            batch_size: Process samples in batches for memory efficiency
            save_results: Whether to save detailed results to file
            output_path: Path to save results JSON (auto-generated if None)
            
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
        self.logger.info(f"Batch size: {batch_size}")
        self.logger.info(f"RAGTruth eval mode: {self.ragtruth_eval_mode}")
        self.logger.info(f"Teacher-forced intrinsic: {self.teacher_forced_intrinsic}")
        
        # Step 1: Load dataset
        self.logger.info("\nStep 1: Loading RAGTruth dataset...")
        samples = self._load_dataset(split=split, max_samples=max_samples)
        self.logger.info(f"Loaded {len(samples)} samples from RAGTruth {split} split")
        
        # Step 2: Run evaluation in batches
        self.logger.info("\nStep 2: Running evaluation pipeline...")
        all_results = []
        
        # Process in batches to manage memory, with one persistent progress bar
        total_batches = (len(samples) - 1) // batch_size + 1 if samples else 0
        with tqdm(total=len(samples), desc="Evaluating samples", unit="sample") as sample_progress:
            for batch_start in range(0, len(samples), batch_size):
                batch_end = min(batch_start + batch_size, len(samples))
                batch = samples[batch_start:batch_end]

                self.logger.info(
                    f"\nProcessing batch {batch_start//batch_size + 1}/{total_batches} "
                    f"(samples {batch_start+1}-{batch_end})"
                )

                # Evaluate each sample in batch
                for sample in batch:
                    try:
                        result = self._evaluate_sample(sample)
                        all_results.append(result)
                    except Exception as e:
                        self.logger.error(f"Error evaluating sample {sample['id']}: {str(e)}")
                        # Continue with next sample
                    finally:
                        sample_progress.update(1)
        
        # Step 3: Compute metrics
        self.logger.info("\nStep 3: Computing evaluation metrics...")
        metrics = self._compute_metrics(all_results)
        
        # Step 4: Save results if requested
        if save_results:
            if output_path is None:
                output_dir = Path('outputs/ragtruth_eval')
                output_dir.mkdir(parents=True, exist_ok=True)
                output_path = output_dir / f'ragtruth_eval_{split}.json'
            
            self._save_results(metrics, all_results, output_path)
            self.logger.info(f"Results saved to: {output_path}")
        
        # Print summary
        self._print_summary(metrics)
        
        return metrics
    
    def _load_dataset(
        self,
        split: str = 'test',
        max_samples: Optional[int] = None
    ) -> List[Dict[str, Any]]:
        """
        Load RAGTruth dataset samples with source info and gold labels.
        
        Loads both source_info.jsonl (questions + contexts) and response.jsonl
        (gold hallucination annotations), then joins them by source_id.
        
        Args:
            split: Dataset split ('train' or 'test')
            max_samples: Maximum number of samples to load (None = all)
            
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
                
                # Check if we've reached max_samples
                if max_samples and len(samples) >= max_samples:
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
            # Convert structured data to string representation
            contexts = [json.dumps(source_info, indent=2)]
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
        sample_id = sample['id']
        question = sample['question']
        dataset_prompt = sample.get('dataset_prompt', '')
        gold_labels = sample['gold_labels']
        hallucination_gold_labels = [
            label for label in gold_labels if self._is_hallucination_label(label)
        ]
        default_generation_metadata = {}
        
        resolved_pairs = []
        generated_response = None

        if self.ragtruth_eval_mode == 'ragtruth_eval':
            generated_response = sample.get('gold_response', '')
            evidence_chunks = self._build_evidence_from_contexts(sample.get('contexts', []))
            claims = extract_claims(
                text=generated_response,
                answer_id=str(sample_id),
                method='auto'
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

            for claim in claims:
                if not evidence_chunks:
                    continue
                resolved_pairs.append({
                    'claim': claim,
                    'evidence': evidence_chunks,
                    'metadata': metadata
                })
        else:
            # Step 1: Run RAG pipeline
            # Note: RAGTruth contexts are gold passages, but we'll use retrieval for realism
            rag_result = self.rag_pipeline.run(
                query=question,
                top_k=5  # Retrieve top 5 passages
            )
            
            generated_response = rag_result['draft_response']
            claim_evidence_pairs = rag_result.get('claim_evidence_pairs', [])

            # Build claim lookup from pipeline output for claim_id -> Claim
            claim_map = {}
            for entry in rag_result.get('claims_by_sub_answer', []):
                for claim in entry.get('claims', []):
                    claim_obj = Claim(**claim) if isinstance(claim, dict) else claim
                    if claim_obj is not None:
                        claim_map[claim_obj.claim_id] = claim_obj

            # Normalize claim/evidence pairs for verification
            sub_answer_metadata = []
            if isinstance(rag_result.get('generator_metadata'), dict):
                sub_answer_metadata = rag_result['generator_metadata'].get('sub_answer_metadata', [])
            for pair in claim_evidence_pairs:
                claim = pair.get('claim') if isinstance(pair, dict) else None
                if claim is None and isinstance(pair, dict):
                    claim_id = pair.get('claim_id')
                    claim = claim_map.get(claim_id)

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
                    self.logger.warning(
                        "Skipping claim-evidence pair due to missing claim or evidence"
                    )
                    continue

                resolved_pairs.append({
                    'claim': claim,
                    'evidence': evidence,
                    'metadata': metadata
                })
        
        # Step 2: Verify claims if verifier is enabled
        claim_decisions = []
        if self.verifier_hub and self.verifier_hub.enabled and resolved_pairs:
            for pair in resolved_pairs:
                claim = pair['claim']
                evidence = pair['evidence']
                metadata = pair.get('metadata') or default_generation_metadata
                
                # Verify claim
                signal = self.verifier_hub.verify_claim(claim, evidence, metadata)
                
                # Aggregate into decision
                decision = self.aggregator.aggregate(signal)
                claim_decisions.append(decision)
        
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
            contradictory_count > 0
            or (
                low_confidence_ratio >= self.low_confidence_ratio_threshold
                and low_coverage_ratio >= self.low_coverage_ratio_threshold
            )
        )
        
        # Detailed per-claim analysis
        claim_results = []
        for idx, decision in enumerate(claim_decisions):
            claim_text = resolved_pairs[idx]['claim'].text
            evidence_items = resolved_pairs[idx].get('evidence', [])
            
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
            'question': question,
            'generated_response': generated_response,
            'num_claims': len(claim_decisions),
            'predictions': [d.status for d in claim_decisions],
            'gold_has_hallucination': gold_has_hallucination,
            'detected_hallucination': detected_hallucination,
            'contradictory_count': contradictory_count,
            'low_confidence_count': low_confidence_count,
            'low_confidence_ratio': low_confidence_ratio,
            'low_coverage_count': low_coverage_count,
            'low_coverage_ratio': low_coverage_ratio,
            'claim_results': claim_results
        }

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
        
        metrics = {
            'overall': {
                'accuracy': float(accuracy),
                'precision': float(precision),
                'recall': float(recall),
                'f1': float(f1),
                'num_samples': len(results)
            },
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
                'false_alarms': int(fp)
            }
        }
        
        return metrics
    
    def _save_results(
        self,
        metrics: Dict[str, Any],
        results: List[Dict[str, Any]],
        output_path: str
    ) -> None:
        """
        Save evaluation metrics and detailed results to JSON file.
        
        Args:
            metrics: Computed metrics dictionary
            results: List of per-sample results
            output_path: Path to output JSON file
        """
        output = {
            'metrics': metrics,
            'sample_results': results,
            'metadata': {
                'evaluator': 'RAGTruthEvaluator',
                'num_samples': len(results),
                'benchmark': 'RAGTruth'
            }
        }
        
        output_path = Path(output_path)
        output_path.parent.mkdir(parents=True, exist_ok=True)
        
        with open(output_path, 'w', encoding='utf-8') as f:
            json.dump(output, f, indent=2, ensure_ascii=False)
        
        self.logger.info(f"Saved evaluation results to: {output_path}")
    
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
