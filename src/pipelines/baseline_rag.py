"""
Baseline RAG Pipeline integrating retrieval and generation.

This module implements the core end-to-end RAG pipeline that retrieves
evidence, generates responses, and creates claim-evidence pairs for
downstream verification in Month 3-5.
"""

from typing import Dict, List, Optional, Any, Tuple
from pathlib import Path

from src.retrieval.dense_retriever import DenseRetriever
from src.retrieval.bm25_retriever import BM25Retriever
from src.retrieval.hybrid_retriever import HybridRetriever
from src.generation.generator_wrapper import GeneratorWrapper
from src.generation.claim_extractor import extract_claims
from src.utils.data_structures import ClaimEvidencePair, EvidenceChunk, Claim, VerifierSignal
from src.utils.config import Config
from src.utils.logger import setup_logger
from src.verification.verifier_hub import VerifierHub
from src.verification.rule_based_aggregator import RuleBasedAggregator
from src.mitigation.orchestrator import MitigationOrchestrator


class BaselineRAGPipeline:
    """
    End-to-end baseline RAG pipeline.
    
    Integrates DenseRetriever and GeneratorWrapper to create a complete
    retrieval-augmented generation system. Takes a query, retrieves evidence,
    generates a response, extracts claims, and pairs them with evidence.
    
    This baseline implementation pairs each claim with all retrieved evidence.
    More sophisticated claim-evidence matching will be implemented in Month 3.
    
    Attributes:
        retriever: DenseRetriever instance for evidence retrieval
        generator: GeneratorWrapper instance for text generation
        config: Configuration object (optional)
        logger: Logger instance
    
    Example:
        >>> # Load from config
        >>> pipeline = BaselineRAGPipeline.from_config("config.yaml")
        >>> 
        >>> # Run query
        >>> result = pipeline.run("Who founded the FEVER dataset?")
        >>> print(result['draft_response'])
        >>> print(f"Found {len(result['claim_evidence_pairs'])} claims")
    """
    
    def __init__(
        self,
        retriever: DenseRetriever,
        generator: GeneratorWrapper,
        config: Optional[Config] = None
    ):
        """
        Initialize the baseline RAG pipeline.
        
        Args:
            retriever: DenseRetriever instance for evidence retrieval
            generator: GeneratorWrapper instance for text generation
            config: Configuration object (optional, for accessing generation params)
        """
        self.retriever = retriever
        self.generator = generator
        self.config = config
        self.logger = setup_logger(__name__)
        
        # Initialize VerifierHub if enabled (Month 3+)
        # Hub manages all verification detectors (Intrinsic, Grounded, and future NLI/Self-Agreement)
        if config and hasattr(config, 'verification') and hasattr(config.verification, 'enabled') and config.verification.enabled:
            try:
                self.verifier_hub = VerifierHub(config, generator)
                self.aggregator = RuleBasedAggregator(config)
                self.verifier_enabled = True
                self.logger.info("BaselineRAGPipeline initialized with VerifierHub enabled")
            except Exception as e:
                self.logger.error(f"Failed to initialize VerifierHub: {str(e)}")
                self.logger.warning("Continuing without verification")
                self.aggregator = None
                self.verifier_enabled = False
        else:
            self.verifier_hub = None
            self.aggregator = None
            self.verifier_enabled = False
            self.logger.info("BaselineRAGPipeline initialized (verification disabled)")

        self.mitigation_orchestrator = None
        if config and self.verifier_enabled and self.verifier_hub is not None and self.aggregator is not None:
            try:
                self.mitigation_orchestrator = MitigationOrchestrator(
                    config=config,
                    verifier_hub=self.verifier_hub,
                    aggregator=self.aggregator,
                    generator=generator,
                )
                self.logger.info(
                    "Mitigation orchestrator initialized (enabled=%s)",
                    self.mitigation_orchestrator.enabled
                )
            except Exception as exc:
                self.logger.warning("Failed to initialize mitigation orchestrator: %s", exc)
                self.mitigation_orchestrator = None
    
    def _split_query_by_questions(self, query: str) -> List[Dict[str, Any]]:
        """
        Split input query by question marks into sub-questions.
        
        Each question mark followed by optional whitespace is treated as a boundary.
        Returns a list of sub-question dictionaries.
        
        Args:
            query: User's input query (potentially multi-question)
        
        Returns:
            List of dicts with keys:
                - text: Sub-question text
                - sub_query_id: Sequential ID (0, 1, 2, ...)
        
        Example:
            >>> query = "What is AI? How does it work?"
            >>> result = self._split_query_by_questions(query)
            >>> len(result)
            2
            >>> result[0]['text']
            'What is AI?'
        """
        import re
        
        if not query or not query.strip():
            return [{'text': '', 'sub_query_id': 0}]
        
        # Pattern: match text ending with '?' (optionally followed by whitespace)
        pattern = r'[^?]+\?'
        matches = list(re.finditer(pattern, query))
        
        if not matches:
            # No question marks found, return entire query as single sub-question
            return [{
                'text': query.strip(),
                'sub_query_id': 0
            }]
        
        sub_queries = []
        for idx, match in enumerate(matches):
            sub_text = match.group(0).strip()
            if sub_text:
                sub_queries.append({
                    'text': sub_text,
                    'sub_query_id': idx
                })
        
        # If no valid sub-queries extracted, return whole query
        if not sub_queries:
            sub_queries = [{
                'text': query.strip(),
                'sub_query_id': 0
            }]
        
        self.logger.debug(f"Split query into {len(sub_queries)} sub-questions")
        return sub_queries

    def _get_retrieval_mode(self) -> str:
        """Return configured retrieval mode or dense fallback."""
        if self.config and hasattr(self.config, 'retrieval'):
            return str(self.config.retrieval.get('mode', 'dense'))
        return 'dense'

    @staticmethod
    def _score_metric_for_mode(retrieval_mode: str) -> str:
        """Map retrieval mode to its primary score field name."""
        mode = (retrieval_mode or 'dense').lower()
        if mode == 'hybrid':
            return 'score_hybrid'
        if mode == 'bm25':
            return 'score_bm25'
        return 'score_dense'

    def _chunk_primary_score(self, chunk: EvidenceChunk, retrieval_mode: str) -> float:
        """Extract the primary retrieval score for a chunk based on retrieval mode."""
        metric = self._score_metric_for_mode(retrieval_mode)
        score = getattr(chunk, metric, None)
        if score is not None:
            return float(score)
        # Fallback in case a retriever does not populate the preferred field.
        if chunk.score_hybrid is not None:
            return float(chunk.score_hybrid)
        if chunk.score_bm25 is not None:
            return float(chunk.score_bm25)
        return float(chunk.score_dense)

    def _retrieve_with_guardrail(
        self,
        sub_query_text: str,
        top_k: int,
        retrieval_mode: str
    ) -> Tuple[List[EvidenceChunk], Dict[str, Any]]:
        """Retrieve evidence with optional low-score fallback expansion."""
        evidence_chunks = self.retriever.retrieve(sub_query_text, top_k=top_k)
        metric = self._score_metric_for_mode(retrieval_mode)
        initial_top_score = (
            self._chunk_primary_score(evidence_chunks[0], retrieval_mode)
            if evidence_chunks else None
        )

        diagnostics: Dict[str, Any] = {
            'score_metric': metric,
            'initial_count': len(evidence_chunks),
            'initial_top_score': initial_top_score,
            'fallback_triggered': False,
            'fallback_reason': None,
            'fallback_top_k': None,
            'fallback_count': None,
            'fallback_top_score': None,
        }

        guardrails_cfg = None
        if self.config and hasattr(self.config, 'retrieval'):
            guardrails_cfg = self.config.retrieval.get('guardrails', None)

        if not isinstance(guardrails_cfg, dict):
            return evidence_chunks, diagnostics

        if not bool(guardrails_cfg.get('enabled', False)):
            return evidence_chunks, diagnostics

        thresholds = guardrails_cfg.get('low_score_thresholds', {})
        if isinstance(thresholds, dict):
            low_score_threshold = thresholds.get(retrieval_mode, thresholds.get('default', None))
        else:
            low_score_threshold = None

        fallback_multiplier = int(guardrails_cfg.get('fallback_top_k_multiplier', 2) or 2)
        fallback_multiplier = max(1, fallback_multiplier)
        fallback_top_k = max(top_k, int(top_k * fallback_multiplier))
        fallback_top_k = min(fallback_top_k, int(guardrails_cfg.get('max_fallback_top_k', 20) or 20))

        low_score_hit = (
            low_score_threshold is not None
            and initial_top_score is not None
            and float(initial_top_score) < float(low_score_threshold)
        )
        low_count_hit = len(evidence_chunks) < top_k

        if not (low_score_hit or low_count_hit) or fallback_top_k <= top_k:
            return evidence_chunks, diagnostics

        diagnostics['fallback_triggered'] = True
        diagnostics['fallback_reason'] = 'low_score' if low_score_hit else 'insufficient_results'
        diagnostics['fallback_top_k'] = fallback_top_k

        self.logger.info(
            f"Retrieval guardrail triggered ({diagnostics['fallback_reason']}): "
            f"metric={metric}, initial_top={initial_top_score}, threshold={low_score_threshold}, "
            f"expanding top_k {top_k}->{fallback_top_k}"
        )
        self.logger.info(
            "rag_retrieval_guardrail_triggered",
            extra={
                "event": "rag_retrieval_guardrail_triggered",
                "data": {
                    "reason": diagnostics['fallback_reason'],
                    "score_metric": metric,
                    "initial_top_score": initial_top_score,
                    "threshold": low_score_threshold,
                    "top_k": top_k,
                    "fallback_top_k": fallback_top_k
                }
            }
        )

        expanded_chunks = self.retriever.retrieve(sub_query_text, top_k=fallback_top_k)
        if expanded_chunks:
            expanded_chunks = sorted(
                expanded_chunks,
                key=lambda c: self._chunk_primary_score(c, retrieval_mode),
                reverse=True
            )
            for rank, chunk in enumerate(expanded_chunks, start=1):
                chunk.rank = rank

        fallback_selection = expanded_chunks[:top_k] if expanded_chunks else evidence_chunks
        fallback_top_score = (
            self._chunk_primary_score(fallback_selection[0], retrieval_mode)
            if fallback_selection else None
        )

        diagnostics['fallback_count'] = len(expanded_chunks)
        diagnostics['fallback_top_score'] = fallback_top_score

        return fallback_selection, diagnostics
    
    def run(
        self,
        query: str,
        top_k: int = 5,
        max_new_tokens: Optional[int] = None,
        temperature: Optional[float] = None,
        top_p: Optional[float] = None,
        do_sample: Optional[bool] = None,
        repetition_penalty: Optional[float] = None,
        no_repeat_ngram_size: Optional[int] = None,
        sanitize_meta_text: Optional[bool] = None,
    ) -> Dict[str, Any]:
        """
        Run the complete RAG pipeline on a query.
        
        Executes the following steps:
        1. Split query into sub-questions (if multiple questions present)
        2. For each sub-question:
           a. Retrieve top-k evidence chunks using DenseRetriever
           b. Generate response with metadata using GeneratorWrapper
           c. Extract atomic claims from generated text
        3. Create ClaimEvidencePair objects pairing claims with evidence
        4. Format output matching System_Architecture_Design.md specification
        
        Args:
            query: User's input question (can be multiple questions)
            top_k: Number of evidence chunks to retrieve (default: 5)
            max_new_tokens: Max tokens to generate (uses config if None)
            temperature: Sampling temperature (uses config if None)
            top_p: Nucleus sampling threshold (uses config if None)
            do_sample: Whether to use sampling (uses config if None)
        
        Returns:
            Dictionary containing:
                - query: Original query string
                - draft_response: Combined generated text response
                - sub_answers: List of sub-answer dicts
                - claims_by_sub_answer: Claims grouped by sub-answer
                - claim_evidence_pairs: List of ClaimEvidencePair dicts
                - generator_metadata: Full metadata from generation (for Month 3)
                - retrieval_metadata: Metadata about retrieved evidence
        
        Example:
            >>> result = pipeline.run("What is machine learning? How does it work?", top_k=3)
            >>> print(result['draft_response'])
            >>> print(f"Sub-answers: {len(result['sub_answers'])}")
        """
        self.logger.info(f"Running RAG pipeline for query: {query[:50]}...")
        self.logger.info(
            "rag_run_start",
            extra={
                "event": "rag_run_start",
                "data": {
                    "query_length": len(query) if query else 0,
                    "top_k": top_k
                }
            }
        )
        
        # Step 0: Split query into sub-questions (optional)
        split_enabled = True
        if self.config and hasattr(self.config, 'processing'):
            split_enabled = bool(
                getattr(getattr(self.config.processing, 'query_split', None), 'enabled', True)
            )

        if split_enabled:
            sub_queries = self._split_query_by_questions(query)
        else:
            sub_queries = [{
                'text': query.strip() if query else '',
                'sub_query_id': 0
            }]

        self.logger.info(
            f"Split into {len(sub_queries)} sub-question(s) (enabled={split_enabled})"
        )
        self.logger.info(
            "rag_query_split",
            extra={
                "event": "rag_query_split",
                "data": {
                    "num_sub_questions": len(sub_queries),
                    "enabled": split_enabled
                }
            }
        )
        
        # Prepare generation parameters
        if self.config:
            gen_params = {
                'max_new_tokens': max_new_tokens or self.config.generation.max_new_tokens,
                'temperature': temperature if temperature is not None else self.config.generation.temperature,
                'top_p': top_p if top_p is not None else self.config.generation.top_p,
                'do_sample': do_sample if do_sample is not None else self.config.generation.do_sample,
                'repetition_penalty': (
                    repetition_penalty if repetition_penalty is not None
                    else self.config.generation.get('repetition_penalty', None)
                ),
                'no_repeat_ngram_size': (
                    no_repeat_ngram_size if no_repeat_ngram_size is not None
                    else self.config.generation.get('no_repeat_ngram_size', None)
                ),
                'sanitize_meta_text': (
                    sanitize_meta_text if sanitize_meta_text is not None
                    else bool(self.config.generation.get('sanitize_meta_text', False))
                ),
            }
        else:
            gen_params = {
                'max_new_tokens': max_new_tokens or 256,
                'temperature': temperature if temperature is not None else 0.7,
                'top_p': top_p if top_p is not None else 0.9,
                'do_sample': do_sample if do_sample is not None else True,
                'repetition_penalty': repetition_penalty,
                'no_repeat_ngram_size': no_repeat_ngram_size,
                'sanitize_meta_text': bool(sanitize_meta_text) if sanitize_meta_text is not None else False,
            }
        
        # Process each sub-question separately
        all_sub_answers = []
        all_claims_by_sub_answer = []
        all_claims = []
        combined_response_parts = []
        all_evidence_chunks = []  # Track all evidence for final metadata
        all_generation_metadata = []  # Track per-sub-answer generation metadata
        retrieval_diagnostics = []  # Track per-sub-question retrieval guardrail info
        retrieval_mode = self._get_retrieval_mode()
        
        for sub_query_data in sub_queries:
            sub_query_text = sub_query_data['text']
            sub_query_id = sub_query_data['sub_query_id']
            
            self.logger.info(f"Processing sub-question {sub_query_id + 1}: {sub_query_text[:50]}...")
            
            # Step 1: Retrieve evidence for this sub-question
            self.logger.debug(f"Retrieving top-{top_k} evidence chunks")
            evidence_chunks, retrieval_diag = self._retrieve_with_guardrail(
                sub_query_text=sub_query_text,
                top_k=top_k,
                retrieval_mode=retrieval_mode
            )
            retrieval_diag['sub_query_id'] = sub_query_id
            retrieval_diagnostics.append(retrieval_diag)
            
            if not evidence_chunks:
                self.logger.warning(f"No evidence retrieved for sub-question {sub_query_id}")
            else:
                self.logger.info(
                    f"Retrieved {len(evidence_chunks)} evidence chunks, "
                    f"top score: {self._chunk_primary_score(evidence_chunks[0], retrieval_mode):.4f}"
                )

            self.logger.info(
                "rag_retrieval_result",
                extra={
                    "event": "rag_retrieval_result",
                    "data": {
                        "sub_query_id": sub_query_id,
                        "retrieved_count": len(evidence_chunks),
                        "top_score": (
                            self._chunk_primary_score(evidence_chunks[0], retrieval_mode)
                            if evidence_chunks else None
                        ),
                        "score_metric": self._score_metric_for_mode(retrieval_mode),
                        "guardrail": retrieval_diag,
                    }
                }
            )
            
            # Track evidence for metadata
            all_evidence_chunks.extend(evidence_chunks)
            
            # Step 2: Generate response for this sub-question
            self.logger.debug(f"Generating response with params: {gen_params}")
            generation_output = self.generator.generate_with_metadata(
                prompt=sub_query_text,
                evidence_chunks=evidence_chunks,
                **gen_params
            )
            
            # Add original sub-query to metadata
            generation_output['original_query'] = sub_query_text
            
            generated_text = generation_output['text']
            self.logger.info(
                f"Generated response for sub-question {sub_query_id + 1}: "
                f"{len(generated_text)} chars, {len(generation_output['tokens'])} tokens"
            )
            self.logger.info(
                "rag_generation_result",
                extra={
                    "event": "rag_generation_result",
                    "data": {
                        "sub_query_id": sub_query_id,
                        "output_chars": len(generated_text),
                        "token_count": len(generation_output.get('tokens', []))
                    }
                }
            )
            
            # Step 3: Extract claims from this sub-answer
            self.logger.debug(f"Extracting claims from sub-answer {sub_query_id}")
            sub_claims = extract_claims(
                text=generated_text,
                method='auto'
            )
            
            self.logger.info(f"Extracted {len(sub_claims)} claims from sub-answer {sub_query_id}")
            self.logger.info(
                "rag_claims_extracted",
                extra={
                    "event": "rag_claims_extracted",
                    "data": {
                        "sub_query_id": sub_query_id,
                        "claims_count": len(sub_claims)
                    }
                }
            )
            
            # Calculate char span for this sub-answer in combined response
            char_start = len(' '.join(combined_response_parts) + (' ' if combined_response_parts else ''))
            combined_response_parts.append(generated_text)
            char_end = len(' '.join(combined_response_parts))
            
            # Adjust claim char spans to be relative to combined response
            for claim in sub_claims:
                original_span = claim.answer_char_span
                claim.answer_char_span = [
                    original_span[0] + char_start,
                    original_span[1] + char_start
                ]
            
            # Store sub-answer data
            sub_answer_dict = {
                'text': generated_text,
                'char_span': [char_start, char_end],
                'sub_answer_id': sub_query_id,
                'sub_query': sub_query_text
            }
            all_sub_answers.append(sub_answer_dict)

            # Store generation metadata for this sub-answer
            all_generation_metadata.append({
                'sub_answer_id': sub_query_id,
                'char_span': [char_start, char_end],
                'sub_query': sub_query_text,
                'metadata': generation_output
            })
            
            # Store claims for this sub-answer
            claims_by_sub_answer_dict = {
                'sub_answer_id': sub_query_id,
                'sub_text': generated_text,
                'sub_query': sub_query_text,
                'claims': sub_claims
            }
            all_claims_by_sub_answer.append(claims_by_sub_answer_dict)
            
            all_claims.extend(sub_claims)
        
        # Combine all responses
        combined_response = ' '.join(combined_response_parts)
        
        self.logger.info(
            f"Combined response: {len(combined_response)} chars, "
            f"{len(all_sub_answers)} sub-answers, {len(all_claims)} total claims"
        )
        self.logger.info(
            f"Combined response: {len(combined_response)} chars, "
            f"{len(all_sub_answers)} sub-answers, {len(all_claims)} total claims"
        )
        
        # Step 4: Create claim-evidence pairs
        # For multi-question: use evidence from the corresponding sub-question
        # For now, use all evidence chunks combined (can be refined later)
        claim_evidence_pairs = []
        
        for claim in all_claims:
            # Create evidence candidate IDs from all evidence
            evidence_candidates = [
                f"{chunk.doc_id}#{chunk.sent_id}"
                for chunk in all_evidence_chunks[:top_k]  # Use top_k from first retrieval
            ]
            
            # Top evidence is the first chunk
            top_evidence = evidence_candidates[0] if evidence_candidates else ""
            
            # Convert evidence chunks to dicts for serialization
            evidence_spans = [chunk.to_dict() for chunk in all_evidence_chunks[:top_k]]
            
            # Create ClaimEvidencePair
            pair = ClaimEvidencePair(
                claim_id=claim.claim_id,
                evidence_candidates=evidence_candidates,
                top_evidence=top_evidence,
                evidence_spans=evidence_spans
            )
            
            claim_evidence_pairs.append(pair)
            self.logger.debug(
                f"Paired claim {claim.claim_id} with {len(evidence_candidates)} evidence chunks"
            )
        
        # Step 4.5: Compute verifier signals (Month 3+ functionality)
        # For multi-question, verify claims with their corresponding evidence
        verifier_signals = []
        if self.verifier_enabled and all_evidence_chunks:
            self.logger.debug("Computing verifier signals via VerifierHub")
            
            # Read verify_all_evidence flag from hub configuration
            verify_all = self.verifier_hub.verify_all_evidence
            
            # Use evidence from first sub-question for now (can be refined)
            evidence_for_verification = all_evidence_chunks[:top_k]
            
            for claim, pair in zip(all_claims, claim_evidence_pairs):
                # Choose evidence: all chunks or top-1 based on config
                if verify_all:
                    evidence_input = evidence_for_verification
                    self.logger.debug(
                        f"Multi-evidence verification for claim {claim.claim_id}: "
                        f"{len(evidence_for_verification)} chunks"
                    )
                else:
                    evidence_input = evidence_for_verification[0] if evidence_for_verification else None
                    self.logger.debug(f"Single-chunk verification for claim {claim.claim_id}")
                
                if evidence_input:
                    # Select generation metadata that matches the claim span
                    verification_metadata = None
                    for entry in all_generation_metadata:
                        span = entry['char_span']
                        if (
                            claim.answer_char_span[0] >= span[0]
                            and claim.answer_char_span[1] <= span[1]
                        ):
                            verification_metadata = dict(entry['metadata'])
                            verification_metadata.setdefault('original_query', entry['sub_query'])
                            break

                    if verification_metadata is None:
                        self.logger.warning(
                            f"No generation metadata found for claim {claim.claim_id}; "
                            f"using fallback metadata"
                        )
                        verification_metadata = {
                            'text': combined_response,
                            'original_query': query,
                            'tokens': [],
                            'scores': []
                        }
                    
                    # Call VerifierHub to compute all signals
                    signal = self.verifier_hub.verify_claim(
                        claim, evidence_input, verification_metadata
                    )
                    
                    if signal:
                        verifier_signals.append(signal.to_dict())
                    else:
                        self.logger.warning(
                            f"VerifierHub returned None for claim {claim.claim_id}, skipping signal"
                        )
            
            self.logger.info(f"Computed {len(verifier_signals)} verifier signals via VerifierHub")
            self.logger.info(
                "rag_verifier_signals",
                extra={
                    "event": "rag_verifier_signals",
                    "data": {
                        "signals_count": len(verifier_signals)
                    }
                }
            )
        elif self.verifier_enabled and not all_evidence_chunks:
            self.logger.warning(
                "Verification enabled but no evidence retrieved - skipping verifier signals"
            )
        
        # Step 5: Format output
        # Get unique evidence doc_ids for metadata
        unique_evidence_doc_ids = list(dict.fromkeys([chunk.doc_id for chunk in all_evidence_chunks[:top_k * len(sub_queries)]]))

        mitigation_result = {
            'final_answer': combined_response,
            'actions': [],
            'filtered_claim_count': 0,
            'claim_records': []
        }
        if self.mitigation_orchestrator and self.mitigation_orchestrator.enabled:
            claim_records = []
            evidence_for_mitigation = all_evidence_chunks[:top_k]
            for claim in all_claims:
                verification_metadata = None
                for entry in all_generation_metadata:
                    span = entry['char_span']
                    if (
                        claim.answer_char_span[0] >= span[0]
                        and claim.answer_char_span[1] <= span[1]
                    ):
                        verification_metadata = dict(entry['metadata'])
                        verification_metadata.setdefault('original_query', entry['sub_query'])
                        break

                if verification_metadata is None:
                    verification_metadata = {
                        'text': combined_response,
                        'original_query': query,
                        'tokens': [],
                        'scores': []
                    }

                claim_records.append({
                    'claim': claim,
                    'evidence': evidence_for_mitigation,
                    'metadata': verification_metadata
                })

            mitigation_result = self.mitigation_orchestrator.apply(
                query=query,
                answer_text=combined_response,
                claim_records=claim_records,
            )
        
        output = {
            'query': query,
            'draft_response': combined_response,
            'response_after_mitigation': mitigation_result['final_answer'],
            'mitigation_actions': mitigation_result['actions'],
            'filtered_claim_count': mitigation_result['filtered_claim_count'],
            'sub_answers': all_sub_answers,
            'claims_by_sub_answer': all_claims_by_sub_answer,
            'claim_evidence_pairs': [pair.to_dict() for pair in claim_evidence_pairs],
            'generator_metadata': {
                'text': combined_response,
                'sub_answers': all_sub_answers,
                'sub_answer_metadata': all_generation_metadata,
                'original_query': query,
                'num_sub_questions': len(sub_queries)
            },
            'retrieval_metadata': {
                'top_k': top_k,
                'num_retrieved': len(all_evidence_chunks[:top_k * len(sub_queries)]),
                'top_score': (
                    max(
                        [self._chunk_primary_score(chunk, retrieval_mode) for chunk in all_evidence_chunks],
                        default=0.0
                    )
                ),
                'score_metric': self._score_metric_for_mode(retrieval_mode),
                'guardrail_diagnostics': retrieval_diagnostics,
                'evidence_doc_ids': unique_evidence_doc_ids[:10]  # Limit to top 10 for display
            }
        }

        if mitigation_result.get('claim_records'):
            output['mitigation_claims'] = [
                record['claim'].to_dict() for record in mitigation_result['claim_records']
            ]
            output['mitigation_evidence_map'] = {
                record['claim'].claim_id: [chunk.to_dict() for chunk in record.get('evidence', [])]
                for record in mitigation_result['claim_records']
            }
        
        # Add verifier signals if computed (Month 3)
        if verifier_signals:
            output['verifier_signals'] = verifier_signals
        
        self.logger.info(
            f"Pipeline complete: {len(all_claims)} claims from {len(all_sub_answers)} sub-answers, "
            f"{len(all_evidence_chunks)} evidence chunks"
        )
        self.logger.info(
            "rag_run_complete",
            extra={
                "event": "rag_run_complete",
                "data": {
                    "claims_count": len(all_claims),
                    "sub_answers_count": len(all_sub_answers),
                    "evidence_chunks_count": len(all_evidence_chunks)
                }
            }
        )
        
        return output
    
    @classmethod
    def from_config(
        cls,
        config_path: str = "config.yaml",
        strategy: str = "development"
    ) -> "BaselineRAGPipeline":
        """
        Create a BaselineRAGPipeline from a configuration file.
        
        Loads the config, initializes the retriever and generator with
        the specified settings, and returns a ready-to-use pipeline.
        
        Args:
            config_path: Path to config.yaml file (default: "config.yaml")
            strategy: Data strategy to use: "development", "validation", or "production"
                     (default: "development")
        
        Returns:
            Initialized BaselineRAGPipeline instance
        
        Raises:
            FileNotFoundError: If config file or index files not found
            ValueError: If configuration is invalid
        
        Example:
            >>> # Load with development dataset
            >>> pipeline = BaselineRAGPipeline.from_config()
            >>> 
            >>> # Load with validation dataset
            >>> pipeline = BaselineRAGPipeline.from_config(strategy="validation")
        """
        logger = setup_logger(__name__)
        logger.info(f"Loading pipeline from config: {config_path}")
        logger.info(f"Using data strategy: {strategy}")
        
        # Load configuration
        config = Config(config_path)
        
        # Get paths with strategy substitution
        index_path = Path(config.data.faiss_index.format(strategy=strategy))
        metadata_path = Path(config.data.index_metadata.format(strategy=strategy))
        
        # Check if index exists
        if not index_path.exists():
            raise FileNotFoundError(
                f"FAISS index not found at {index_path}. "
                f"Please run the data processing pipeline first:\n"
                f"1. python scripts/process_wikipedia.py\n"
                f"2. python scripts/generate_embeddings.py\n"
                f"3. python scripts/build_faiss_index.py"
            )
        
        # Check if metadata exists
        if not metadata_path.exists():
            raise FileNotFoundError(
                f"Metadata not found at {metadata_path}. "
                f"Please run the data processing pipeline first."
            )
        
        # Get retrieval mode from config
        retrieval_mode = config.retrieval.get('mode', 'dense')
        logger.info(f"Retrieval mode: {retrieval_mode}")
        
        # Initialize retriever based on mode
        if retrieval_mode == 'dense':
            # Dense retrieval only (default)
            logger.info(f"Initializing DenseRetriever with {config.models.sentence_transformer}")
            retriever = DenseRetriever(
                index_path=str(index_path),
                metadata_path=str(metadata_path),
                encoder_model=config.models.sentence_transformer,
                device=config.processing.device
            )
        
        elif retrieval_mode == 'bm25':
            # BM25 sparse retrieval only
            corpus_path = Path('data/processed') / f'wiki_chunks_{strategy}.jsonl'
            bm25_index_path = Path('data/indexes') / strategy / 'bm25_index.pkl'
            
            logger.info(f"Initializing BM25Retriever")
            retriever = BM25Retriever(
                corpus_path=str(corpus_path),
                index_path=str(bm25_index_path),
                k1=config.retrieval.bm25.get('k1', 1.5),
                b=config.retrieval.bm25.get('b', 0.75)
            )
        
        elif retrieval_mode == 'hybrid':
            # Hybrid: combine dense and BM25
            # Initialize dense retriever
            logger.info(f"Initializing DenseRetriever for hybrid mode")
            dense_retriever = DenseRetriever(
                index_path=str(index_path),
                metadata_path=str(metadata_path),
                encoder_model=config.models.sentence_transformer,
                device=config.processing.device
            )
            
            # Initialize BM25 retriever
            corpus_path = Path('data/processed') / f'wiki_chunks_{strategy}.jsonl'
            bm25_index_path = Path('data/indexes') / strategy / 'bm25_index.pkl'
            
            logger.info(f"Initializing BM25Retriever for hybrid mode")
            bm25_retriever = BM25Retriever(
                corpus_path=str(corpus_path),
                index_path=str(bm25_index_path),
                k1=config.retrieval.bm25.get('k1', 1.5),
                b=config.retrieval.bm25.get('b', 0.75)
            )
            
            # Initialize hybrid retriever
            logger.info(f"Initializing HybridRetriever")
            retriever = HybridRetriever(
                dense_retriever=dense_retriever,
                bm25_retriever=bm25_retriever,
                alpha=config.retrieval.hybrid.get('alpha', 0.5),
                fusion_method=config.retrieval.hybrid.get('fusion_method', 'rrf'),
                rrf_k=config.retrieval.hybrid.get('rrf_k', 60)
            )
        
        else:
            raise ValueError(
                f"Invalid retrieval mode: {retrieval_mode}. "
                f"Must be 'dense', 'bm25', or 'hybrid'"
            )
        
        # Initialize GeneratorWrapper
        logger.info(f"Initializing GeneratorWrapper with {config.models.generator}")
        generator = GeneratorWrapper(
            model_name=config.models.generator,
            device=config.processing.device,
            load_in_8bit=config.generation.load_in_8bit,
            max_input_tokens=config.generation.get('max_input_tokens', None)
        )
        
        logger.info("Pipeline initialization complete")
        
        # Create and return pipeline
        return cls(
            retriever=retriever,
            generator=generator,
            config=config
        )
