"""
Gradio-based confidence visualization UI for hallucination detection.

This module provides a web interface for visualizing claim-level confidence
scores and verification results. Claims are color-coded based on their
verification status (Supported/Contradictory/Low Confidence).
"""

import gradio as gr
import pandas as pd
from typing import List, Tuple, Dict, Any, Optional

from src.utils.config import Config
from src.utils.logger import setup_logger
from src.pipelines.baseline_rag import BaselineRAGPipeline
from src.verification.verifier_hub import VerifierHub
from src.verification.rule_based_aggregator import RuleBasedAggregator
from src.utils.data_structures import Claim, ClaimDecision


class ConfidenceUI:
    """
    Gradio interface for visualizing claim verification with confidence scores.
    
    Displays generated answers with color-coded claims and a detailed table
    showing per-claim verification metrics (NLI scores, coverage, entropy, etc.).
    
    Color Mapping:
        - Green (#28a745): Supported - High confidence, well-grounded in evidence
        - Yellow (#ffc107): Low Confidence - Uncertain or insufficient evidence
        - Red (#dc3545): Contradictory - Conflicts with evidence or numeric mismatch
    
    Attributes:
        rag_pipeline: BaselineRAGPipeline instance for query processing
        verifier_hub: VerifierHub instance for claim verification
        aggregator: RuleBasedAggregator instance for final decisions
        color_map: Mapping from status to hex colors
        logger: Logger instance
    
    Example:
        >>> config = Config.from_yaml("config.yaml")
        >>> pipeline = BaselineRAGPipeline.from_config("config.yaml")
        >>> verifier_hub = VerifierHub(config, pipeline.generator)
        >>> aggregator = RuleBasedAggregator(config)
        >>> 
        >>> ui = ConfidenceUI(pipeline, verifier_hub, aggregator)
        >>> demo = ui.create_interface()
        >>> demo.launch(share=False, server_port=7860)
    """
    
    def __init__(
        self,
        rag_pipeline: BaselineRAGPipeline,
        verifier_hub: VerifierHub,
        aggregator: RuleBasedAggregator
    ):
        """
        Initialize the confidence visualization UI.
        
        Args:
            rag_pipeline: RAG pipeline for generating answers
            verifier_hub: Verification hub for computing detector signals
            aggregator: Rule-based aggregator for final claim decisions
        """
        self.rag_pipeline = rag_pipeline
        self.verifier_hub = verifier_hub
        self.aggregator = aggregator
        self.logger = setup_logger(__name__)
        
        # Define color mapping for claim statuses
        self.color_map = {
            'Supported': '#28a745',        # Green
            'Contradictory': '#dc3545',    # Red
            'Low Confidence': '#ffc107'    # Yellow
        }
        
        self.logger.info("ConfidenceUI initialized with color mapping: %s", self.color_map)
    
    def create_interface(self) -> gr.Interface:
        """
        Create the Gradio interface for claim visualization.
        
        The interface includes:
        - Input: Textbox for entering queries
        - Output 1: HighlightedText showing color-coded claims in the answer
        - Output 2: DataFrame showing detailed claim-level metrics
        
        Returns:
            Gradio Interface object ready to launch
        
        Example:
            >>> ui = ConfidenceUI(pipeline, hub, aggregator)
            >>> demo = ui.create_interface()
            >>> demo.launch()
        """
        def process_query(query: str) -> Tuple[List[Tuple[str, Optional[str]]], pd.DataFrame]:
            """
            Process a query and return color-coded output and details table.
            
            Args:
                query: User's input question
            
            Returns:
                Tuple of (highlighted_text, details_dataframe):
                - highlighted_text: List of (text, label) tuples for HighlightedText
                - details_dataframe: DataFrame with per-claim verification metrics
            """
            if not query.strip():
                return [], pd.DataFrame()
            
            try:
                self.logger.info(f"Processing query: {query[:100]}")
                
                # Step 1: Run RAG pipeline to generate answer and extract claims
                result = self.rag_pipeline.run(query, top_k=5)
                
                answer_text = result['draft_response']
                sub_answers = result.get('sub_answers', [])
                claims_by_sub_answer = result.get('claims_by_sub_answer', [])
                claim_evidence_pairs = result.get('claim_evidence_pairs', [])
                metadata = result.get('generator_metadata', {})
                
                if not claim_evidence_pairs:
                    self.logger.warning("No claims extracted from answer")
                    # Return empty DataFrame with proper columns
                    return [(answer_text, None)], self._build_details_table([], [])
                
                # Extract evidence chunks from the first claim_evidence_pair
                # All claims share the same evidence in baseline implementation
                evidence_chunks = []
                if claim_evidence_pairs:
                    from src.utils.data_structures import EvidenceChunk
                    evidence_spans = claim_evidence_pairs[0].get('evidence_spans', [])
                    for span in evidence_spans:
                        evidence_chunks.append(EvidenceChunk(**span))
                
                self.logger.info(
                    f"Extracted {len(claim_evidence_pairs)} claims from "
                    f"{len(sub_answers)} sub-answers"
                )
                
                # Step 2: Collect all claims from claims_by_sub_answer
                all_claims = []
                for sub_ans_data in claims_by_sub_answer:
                    all_claims.extend(sub_ans_data['claims'])
                
                self.logger.info(f"Collected {len(all_claims)} claims for verification")
                
                if len(all_claims) != len(claim_evidence_pairs):
                    self.logger.warning(
                        f"Mismatch in claim counts: Pipeline collected {len(all_claims)}, "
                        f"claim_evidence_pairs has {len(claim_evidence_pairs)}. "
                        "Alignment may be incorrect."
                    )
                
                # Step 3: Verify and aggregate each claim
                decisions = []
                for i, (claim, pair) in enumerate(zip(all_claims, claim_evidence_pairs)):
                    # Verify claim using VerifierHub
                    signal = self.verifier_hub.verify_claim(
                        claim,
                        evidence_chunks,
                        metadata
                    )
                    
                    if signal:
                        # Aggregate signals into final decision
                        decision = self.aggregator.aggregate(signal)
                        decisions.append(decision)
                        
                        self.logger.debug(
                            f"Claim {claim.claim_id}: {decision.status} "
                            f"(confidence: {decision.confidence.get('overall_confidence', 0):.1f}%)"
                        )
                    else:
                        self.logger.warning(f"No signal for claim {claim.claim_id}")
                
                if not decisions:
                    self.logger.warning("No decisions generated")
                    return [(answer_text, None)], self._build_details_table([], [])
                
                # Step 4: Build highlighted output with sub-answer headers
                highlighted_text = self._build_highlighted_output_with_headers(
                    answer_text, sub_answers, claims_by_sub_answer, decisions
                )
                
                # Step 5: Build details table
                details_df = self._build_details_table(all_claims, decisions)
                
                self.logger.info(f"Successfully processed query with {len(decisions)} decisions")
                
                return highlighted_text, details_df
                
            except Exception as e:
                self.logger.error(f"Error processing query: {str(e)}", exc_info=True)
                error_msg = f"Error: {str(e)}"
                return [(error_msg, None)], pd.DataFrame()
        
        # Create Gradio interface
        demo = gr.Interface(
            fn=process_query,
            inputs=gr.Textbox(
                label='Query',
                placeholder='Ask a question... (e.g., "What is artificial intelligence?")',
                lines=2
            ),
            outputs=[
                gr.HighlightedText(
                    label='Answer with Confidence Highlighting',
                    combine_adjacent=False,
                    show_legend=True,
                    color_map=self.color_map
                ),
                gr.Dataframe(
                    label='Claim-Level Details',
                    wrap=True
                )
            ],
            title='🔍 Hallucination Detection Demo',
            description=(
                '**Color Coding:**\n'
                '- 🟢 **Green (Supported)**: High confidence, well-grounded in evidence\n'
                '- 🟡 **Yellow (Low Confidence)**: Uncertain or insufficient evidence\n'
                '- 🔴 **Red (Contradictory)**: Conflicts with evidence or numeric mismatch\n\n'
                'Enter a question to see the answer with color-coded claims and detailed verification metrics.'
            ),
            examples=[
                ["What is artificial intelligence?"],
                ["How do machines learn from data?"],
                ["What is deep learning?"]
            ],
            cache_examples=False
        )
        
        self.logger.info("Gradio interface created successfully")
        return demo
    
    def _build_highlighted_output_with_headers(
        self,
        answer_text: str,
        sub_answers: List[Dict],
        claims_by_sub_answer: List[Dict],
        decisions: List[ClaimDecision]
    ) -> List[Tuple[str, Optional[str]]]:
        """
        Build highlighted text output with sub-answer headers.
        
        Inserts headers like "Sub-Answer 1:" before each sub-answer's claims,
        making it clear which claims belong to which part of a multi-question response.
        
        Args:
            answer_text: The full generated answer text
            sub_answers: List of sub-answer dicts with text and char_span
            claims_by_sub_answer: List of dicts with sub_answer_id and claims
            decisions: List of ClaimDecision objects with statuses
        
        Returns:
            List of (text, label) tuples for Gradio HighlightedText component
        """
        # Create a mapping from claim_id to decision
        decision_map = {d.claim_id: d for d in decisions}
        
        # Build tokens with headers per sub-answer
        tokens = []
        
        for idx, sub_ans_data in enumerate(claims_by_sub_answer):
            sub_id = sub_ans_data['sub_answer_id']
            sub_text = sub_ans_data['sub_text']
            sub_claims = sub_ans_data['claims']
            
            # Add header for this sub-answer (if multiple sub-answers)
            if len(claims_by_sub_answer) > 1:
                header = f"[Sub-Answer {sub_id + 1}] "
                tokens.append((header, None))
            
            # Sort claims within this sub-answer by position
            sorted_items = []
            for claim in sub_claims:
                if claim.claim_id in decision_map:
                    sorted_items.append((
                        claim.answer_char_span[0],
                        claim.answer_char_span[1],
                        decision_map[claim.claim_id].status
                    ))
            
            sorted_items.sort(key=lambda x: x[0])
            
            # Build highlighted segments for this sub-answer
            if not sorted_items:
                # No claims with decisions, just add the text unlabeled
                tokens.append((sub_text, None))
            else:
                # Get the char_span of the sub-answer relative to full text
                sub_start = sub_answers[idx]['char_span'][0]
                sub_end = sub_answers[idx]['char_span'][1]
                
                current_pos = sub_start
                
                for start, end, status in sorted_items:
                    # Add text before the claim (unlabeled)
                    if current_pos < start:
                        tokens.append((answer_text[current_pos:start], None))
                    
                    # Add the claim text with its status label
                    claim_text = answer_text[start:end]
                    tokens.append((claim_text, status))
                    
                    current_pos = end
                
                # Add any remaining text in this sub-answer
                if current_pos < sub_end:
                    tokens.append((answer_text[current_pos:sub_end], None))
            
            # Add spacing between sub-answers
            if idx < len(claims_by_sub_answer) - 1:
                tokens.append((" ", None))
        
        self.logger.debug(f"Built highlighted output with {len(tokens)} segments and headers")
        return tokens
    
    def _build_highlighted_output(
        self,
        answer_text: str,
        claims: List[Any],
        decisions: List[ClaimDecision]
    ) -> List[Tuple[str, Optional[str]]]:
        """
        Build highlighted text output from claims and decisions.
        
        Converts the answer text into a list of (text, label) tuples where
        each claim is labeled with its verification status.
        
        Args:
            answer_text: The full generated answer text
            claims: List of Claim objects with char spans
            decisions: List of ClaimDecision objects with statuses
        
        Returns:
            List of (text, label) tuples for Gradio HighlightedText component
            
        Example:
            >>> output = [
            ...     ("Paris is the capital of France", "Supported"),
            ...     (" and it has a population of ", None),
            ...     ("over 2 million", "Low Confidence"),
            ...     (".", None)
            ... ]
        """
        # Create a mapping from claim_id to decision
        decision_map = {d.claim_id: d for d in decisions}
        
        # Sort claims by their position in the answer text
        sorted_items = []
        for claim in claims:
            # Extract claim object attributes
            if isinstance(claim, dict):
                claim_id = claim['claim_id']
                char_span = claim['answer_char_span']
            else:
                claim_id = claim.claim_id
                char_span = claim.answer_char_span
            
            if claim_id in decision_map:
                sorted_items.append((char_span[0], char_span[1], decision_map[claim_id].status))
        
        # Sort by start position
        sorted_items.sort(key=lambda x: x[0])
        
        # Build the highlighted text tokens
        tokens = []
        current_pos = 0
        
        for start, end, status in sorted_items:
            # Add text before the claim (unlabeled)
            if current_pos < start:
                tokens.append((answer_text[current_pos:start], None))
            
            # Add the claim text with its status label
            claim_text = answer_text[start:end]
            tokens.append((claim_text, status))
            
            current_pos = end
        
        # Add any remaining text after the last claim
        if current_pos < len(answer_text):
            tokens.append((answer_text[current_pos:], None))
        
        self.logger.debug(f"Built highlighted output with {len(tokens)} segments")
        return tokens
    
    def _build_details_table(
        self,
        claims: List[Any],
        decisions: List[ClaimDecision]
    ) -> pd.DataFrame:
        """
        Build a DataFrame with detailed per-claim verification metrics.
        
        Creates a table showing claim text (truncated), verdict, confidence scores,
        NLI probabilities, coverage metrics, and entropy scores.
        
        Args:
            claims: List of Claim objects
            decisions: List of ClaimDecision objects
        
        Returns:
            DataFrame with columns: Claim, Verdict, Confidence, Band,
            NLI Entailment, NLI Contradiction, Coverage, Entropy Conf
        
        Example:
            >>> df = pd.DataFrame([{
            ...     'Claim': 'Paris is the capital...',
            ...     'Verdict': 'Supported',
            ...     'Confidence': '85.2%',
            ...     'Band': 'High',
            ...     'NLI Entailment': '0.92',
            ...     'NLI Contradiction': '0.03',
            ...     'Coverage': '0.78',
            ...     'Entropy Conf': '0.65'
            ... }])
        """
        rows = []
        
        # Create mapping from claim_id to claim
        claim_map = {}
        for claim in claims:
            if isinstance(claim, dict):
                claim_map[claim['claim_id']] = claim['text']
            else:
                claim_map[claim.claim_id] = claim.text
        
        for decision in decisions:
            claim_text = claim_map.get(decision.claim_id, 'N/A')
            
            # Truncate long claims for display
            display_text = claim_text[:80] + '...' if len(claim_text) > 80 else claim_text
            
            # Extract confidence metrics (with safe defaults)
            conf = decision.confidence
            overall_conf = conf.get('overall_confidence', 0)
            band = conf.get('band', 'Unknown')
            support_prob = conf.get('support_prob', 0.0)
            contradict_prob = conf.get('contradict_prob', 0.0)
            coverage_score = conf.get('coverage_score', 0.0)
            entropy_conf = conf.get('entropy_conf', 0.0)
            
            row = {
                'Claim': display_text,
                'Verdict': decision.status,
                'Confidence': f"{overall_conf:.1f}%",
                'Band': band,
                'NLI Entailment': f"{support_prob:.2f}",
                'NLI Contradiction': f"{contradict_prob:.2f}",
                'Coverage': f"{coverage_score:.2f}",
                'Entropy Conf': f"{entropy_conf:.2f}"
            }
            
            rows.append(row)
        
        # Define columns explicitely to ensure headers are always present
        columns = [
            'Claim', 'Verdict', 'Confidence', 'Band', 
            'NLI Entailment', 'NLI Contradiction', 'Coverage', 'Entropy Conf'
        ]
        
        if not rows:
            return pd.DataFrame(columns=columns)
            
        df = pd.DataFrame(rows, columns=columns)
        self.logger.debug(f"Built details table with {len(rows)} rows")
        
        return df
