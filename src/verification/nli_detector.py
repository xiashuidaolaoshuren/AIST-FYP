"""
NLI (Natural Language Inference) Detector for Hallucination Detection.

This module implements a zero-shot NLI detector using a pre-trained DeBERTa model
fine-tuned on MNLI, FEVER, and ANLI datasets. It classifies whether a claim is
entailed by, contradicted by, or neutral with respect to the evidence.

Key concepts:
- Entailment: Evidence logically implies the claim (high confidence)
- Contradiction: Evidence contradicts the claim (potential hallucination)
- Neutral: Evidence neither supports nor contradicts (insufficient information)

Reference:
- DeBERTa-v3: He et al., "DeBERTaV3: Improving DeBERTa using ELECTRA-Style Pre-Training"
- MNLI: Multi-Genre Natural Language Inference dataset
- FEVER: Fact Extraction and VERification dataset
- ANLI: Adversarial Natural Language Inference dataset
"""

import torch
import torch.nn.functional as F
from transformers import AutoTokenizer, AutoModelForSequenceClassification
from typing import Dict, Optional
import warnings

from src.utils.data_structures import Claim, EvidenceChunk
from src.utils.logger import setup_logger
from src.utils.config import Config


class NLIDetector:
    """
    Zero-shot NLI detector for claim-evidence relationship classification.
    
    Uses a pre-trained DeBERTa-v3 model fine-tuned on NLI datasets to classify
    the relationship between evidence (premise) and claim (hypothesis) as:
    - Entailment: Evidence supports the claim
    - Contradiction: Evidence contradicts the claim (hallucination signal)
    - Neutral: Evidence is unrelated to the claim
    
    The detector:
    1. Tokenizes evidence-claim pair (evidence as premise, claim as hypothesis)
    2. Runs transformer inference to get classification logits
    3. Applies softmax to obtain probability distribution
    4. Returns scores for all three categories
    
    Attributes:
        config: Configuration object
        model_name: Hugging Face model identifier
        device: Computation device (cuda/cpu)
        model: Pre-trained sequence classification model
        tokenizer: Corresponding tokenizer
        label_mapping: Dict mapping label indices to category names
        logger: Logger instance
    
    Example:
        >>> config = Config()
        >>> detector = NLIDetector(config)
        >>> scores = detector.detect(
        ...     claim_text="Einstein won a Nobel Prize.",
        ...     evidence_text="Albert Einstein received the Nobel Prize in Physics in 1921."
        ... )
        >>> print(f"Entailment: {scores['entailment']:.2f}")
    """
    
    def __init__(self, config: Config):
        """
        Initialize the NLI detector and load the pre-trained model.
        
        Args:
            config: Configuration object with verification settings
        
        Raises:
            RuntimeError: If model loading fails
        """
        self.config = config
        self.logger = setup_logger(__name__)
        
        # Read configuration
        if hasattr(config, 'verification') and hasattr(config.verification, 'nli'):
            self.model_name = getattr(
                config.verification.nli, 
                'model_name', 
                'MoritzLaurer/DeBERTa-v3-base-mnli-fever-anli'
            )
            device_config = getattr(config.verification.nli, 'device', 'cuda')
            self._nli_batch_size = max(1, int(getattr(config.verification.nli, 'batch_size', 32)))
        else:
            self.model_name = 'MoritzLaurer/DeBERTa-v3-base-mnli-fever-anli'
            device_config = 'cuda'
            self._nli_batch_size = 32
        
        # Determine device (strictly use config value as per Q1: Option B)
        if device_config == 'cuda':
            if torch.cuda.is_available():
                self.device = torch.device('cuda')
                self.logger.info("Using CUDA for NLI detector")
            else:
                self.logger.warning("CUDA requested but not available, falling back to CPU")
                self.device = torch.device('cpu')
        else:
            self.device = torch.device('cpu')
            self.logger.info("Using CPU for NLI detector")
        
        try:
            self.logger.info(f"Loading NLI model: {self.model_name}")
            
            # Load tokenizer
            self.tokenizer = AutoTokenizer.from_pretrained(self.model_name)
            
            # Load model
            self.model = AutoModelForSequenceClassification.from_pretrained(self.model_name)
            self.model.to(self.device)
            self.model.eval()  # Set to evaluation mode
            
            # Get label mapping from model config
            # Standard NLI models use: 0=contradiction, 1=neutral, 2=entailment OR
            # 0=entailment, 1=neutral, 2=contradiction (depends on model)
            # We'll check the model's id2label mapping
            if hasattr(self.model.config, 'id2label'):
                id2label = self.model.config.id2label
                self.label_mapping = {
                    'entailment': None,
                    'neutral': None,
                    'contradiction': None
                }
                # Find the index for each label
                for idx, label in id2label.items():
                    label_lower = label.lower()
                    if 'entail' in label_lower:
                        self.label_mapping['entailment'] = int(idx)
                    elif 'neutral' in label_lower:
                        self.label_mapping['neutral'] = int(idx)
                    elif 'contradiction' in label_lower or 'contradict' in label_lower:
                        self.label_mapping['contradiction'] = int(idx)
                
                self.logger.info(f"Label mapping: {self.label_mapping}")
            else:
                # Default mapping (most common)
                self.label_mapping = {
                    'contradiction': 0,
                    'neutral': 1,
                    'entailment': 2
                }
                self.logger.warning(
                    f"No id2label in model config, using default mapping: {self.label_mapping}"
                )
            
            # Verify all labels are mapped
            if None in self.label_mapping.values():
                raise ValueError(f"Incomplete label mapping: {self.label_mapping}")
            
            self.logger.info(
                f"NLIDetector initialized successfully "
                f"(device={self.device}, model={self.model_name}, batch_size={self._nli_batch_size})"
            )
            
        except Exception as e:
            error_msg = f"Failed to initialize NLIDetector: {str(e)}"
            self.logger.error(error_msg)
            raise RuntimeError(error_msg) from e
    
    def detect(
        self,
        claim_text: str,
        evidence_text: str
    ) -> Dict[str, float]:
        """
        Classify the NLI relationship between claim and evidence.
        
        Performs zero-shot NLI classification to determine if the evidence
        entails, contradicts, or is neutral with respect to the claim.
        
        Args:
            claim_text: The claim to verify (hypothesis in NLI terminology)
            evidence_text: The evidence to check against (premise in NLI terminology)
        
        Returns:
            Dictionary with three probability scores:
            {
                'entailment': float (0-1),
                'neutral': float (0-1),
                'contradiction': float (0-1)
            }
            Scores sum to approximately 1.0
        
        Raises:
            ValueError: If inputs are invalid (empty strings)
        
        Example:
            >>> scores = detector.detect(
            ...     "Einstein won a Nobel Prize",
            ...     "Albert Einstein received the Nobel Prize in Physics"
            ... )
            >>> print(scores)
            {'entailment': 0.95, 'neutral': 0.03, 'contradiction': 0.02}
        
        Note:
            - Evidence is treated as the premise
            - Claim is treated as the hypothesis
            - High contradiction score indicates potential hallucination
        """
        # Validate inputs
        if not claim_text or not claim_text.strip():
            self.logger.error("detect() called with empty claim_text")
            raise ValueError("claim_text cannot be empty")
        
        if not evidence_text or not evidence_text.strip():
            self.logger.error("detect() called with empty evidence_text")
            raise ValueError("evidence_text cannot be empty")
        
        try:
            # Tokenize: evidence (premise) first, then claim (hypothesis)
            # This is the standard input format for NLI models
            inputs = self.tokenizer(
                evidence_text,
                claim_text,
                return_tensors="pt",
                truncation=True,
                max_length=512,  # Standard max length for most transformers
                padding=True
            )
            
            # Move inputs to device
            inputs = {k: v.to(self.device) for k, v in inputs.items()}
            
            # Run inference (no gradients needed)
            with torch.no_grad():
                outputs = self.model(**inputs)
                logits = outputs.logits  # Shape: (batch_size, num_labels)
            
            # Apply softmax to get probabilities
            probabilities = F.softmax(logits, dim=-1)
            probs = probabilities[0].cpu().numpy()  # Get first (and only) batch item
            
            # Map probabilities to labels
            scores = {
                'entailment': float(probs[self.label_mapping['entailment']]),
                'neutral': float(probs[self.label_mapping['neutral']]),
                'contradiction': float(probs[self.label_mapping['contradiction']])
            }
            
            self.logger.debug(
                f"NLI scores - entailment: {scores['entailment']:.3f}, "
                f"neutral: {scores['neutral']:.3f}, "
                f"contradiction: {scores['contradiction']:.3f}"
            )

            self.logger.info(
                "detector_nli",
                extra={
                    "event": "detector_nli",
                    "data": {
                        "entailment": scores['entailment'],
                        "neutral": scores['neutral'],
                        "contradiction": scores['contradiction']
                    }
                }
            )
            
            return scores
            
        except Exception as e:
            error_msg = f"NLI detection failed: {str(e)}"
            self.logger.error(error_msg)
            # Return neutral scores as fallback
            self.logger.warning("Returning neutral fallback scores due to error")
            return {
                'entailment': 0.33,
                'neutral': 0.34,
                'contradiction': 0.33
            }
    
    def detect_batch(
        self,
        claim_texts: list[str],
        evidence_texts: list[str]
    ) -> list[Dict[str, float]]:
        """
        Classify NLI relationship for multiple claim-evidence pairs (batch processing).
        
        This method is more efficient than calling detect() multiple times
        when processing many pairs, as it leverages batch inference.
        
        Args:
            claim_texts: List of claims to verify
            evidence_texts: List of evidence texts (must match length of claim_texts)
        
        Returns:
            List of score dictionaries, one per input pair
        
        Raises:
            ValueError: If input lists have different lengths or are empty
        
        Example:
            >>> scores_list = detector.detect_batch(
            ...     ["Claim 1", "Claim 2"],
            ...     ["Evidence 1", "Evidence 2"]
            ... )
        
        Note:
            Uses micro-batch chunking with OOM auto-tune. On CUDA OOM, the
            effective batch size is halved and retried until successful.
        """
        # Validate inputs
        if len(claim_texts) != len(evidence_texts):
            raise ValueError(
                f"Length mismatch: {len(claim_texts)} claims vs {len(evidence_texts)} evidence"
            )
        
        if len(claim_texts) == 0:
            raise ValueError("Empty input lists")
        
        try:
            self.logger.debug(
                "Processing %d claim-evidence pairs (batched, micro_batch=%d)",
                len(claim_texts),
                self._nli_batch_size,
            )
            results: list[Dict[str, float]] = []

            idx = 0
            while idx < len(claim_texts):
                chunk_size = min(self._nli_batch_size, len(claim_texts) - idx)

                while True:
                    claim_chunk = claim_texts[idx:idx + chunk_size]
                    evidence_chunk = evidence_texts[idx:idx + chunk_size]
                    try:
                        inputs = self.tokenizer(
                            evidence_chunk,
                            claim_chunk,
                            return_tensors="pt",
                            truncation=True,
                            max_length=512,
                            padding=True
                        )
                        inputs = {k: v.to(self.device) for k, v in inputs.items()}

                        with torch.no_grad():
                            outputs = self.model(**inputs)
                            logits = outputs.logits

                        probabilities = F.softmax(logits, dim=-1).cpu().numpy()
                        for probs in probabilities:
                            results.append({
                                'entailment': float(probs[self.label_mapping['entailment']]),
                                'neutral': float(probs[self.label_mapping['neutral']]),
                                'contradiction': float(probs[self.label_mapping['contradiction']])
                            })
                        idx += chunk_size
                        break
                    except torch.cuda.OutOfMemoryError:
                        if self.device.type != 'cuda':
                            raise
                        if self._nli_batch_size <= 1:
                            self.logger.warning(
                                "NLI OOM at micro_batch=1; falling back to sequential detect() for remaining %d pairs",
                                len(claim_texts) - idx,
                            )
                            raise

                        old_size = self._nli_batch_size
                        self._nli_batch_size = max(1, self._nli_batch_size // 2)
                        chunk_size = min(self._nli_batch_size, len(claim_texts) - idx)
                        self.logger.warning(
                            "NLI CUDA OOM. Reducing micro_batch from %d to %d and retrying",
                            old_size,
                            self._nli_batch_size,
                        )
                        torch.cuda.empty_cache()

            return results

        except Exception as e:
            self.logger.error(f"Batch NLI detection failed: {str(e)}")
            self.logger.warning("Falling back to sequential detect() calls")
            results = []
            for claim, evidence in zip(claim_texts, evidence_texts):
                scores = self.detect(claim, evidence)
                results.append(scores)
            return results

    def detect_bidirectional(
        self,
        claim_text: str,
        evidence_text: str,
    ) -> Dict[str, float]:
        """
        Run bidirectional NLI and return forward scores plus reverse entailment.

        Forward direction uses (evidence -> claim), reverse uses (claim -> evidence).
        The returned dict extends forward scores with:
            - reverse_entailment: entailment probability in reverse direction
            - reverse_neutral: neutral probability in reverse direction
            - reverse_contradiction: contradiction probability in reverse direction
        """
        batch_scores = self.detect_batch_bidirectional([claim_text], [evidence_text])
        return batch_scores[0] if batch_scores else {
            'entailment': 0.33,
            'neutral': 0.34,
            'contradiction': 0.33,
            'reverse_entailment': 0.0,
            'reverse_neutral': 0.0,
            'reverse_contradiction': 0.0,
        }

    def detect_batch_bidirectional(
        self,
        claim_texts: list[str],
        evidence_texts: list[str],
    ) -> list[Dict[str, float]]:
        """
        Run bidirectional NLI for multiple pairs efficiently using one doubled batch.

        For each (claim, evidence) pair, we score:
            1) forward  : (evidence -> claim)
            2) reverse  : (claim -> evidence)
        """
        if len(claim_texts) != len(evidence_texts):
            raise ValueError(
                f"Length mismatch: {len(claim_texts)} claims vs {len(evidence_texts)} evidence"
            )

        if len(claim_texts) == 0:
            return []

        forward_claims = list(claim_texts)
        forward_evidence = list(evidence_texts)

        reverse_claims = list(evidence_texts)
        reverse_evidence = list(claim_texts)

        merged_claims = forward_claims + reverse_claims
        merged_evidence = forward_evidence + reverse_evidence

        merged_scores = self.detect_batch(merged_claims, merged_evidence)
        n = len(claim_texts)
        forward_scores = merged_scores[:n]
        reverse_scores = merged_scores[n:]

        bidirectional_scores: list[Dict[str, float]] = []
        for forward, reverse in zip(forward_scores, reverse_scores):
            bidirectional_scores.append({
                'entailment': float(forward.get('entailment', 0.33)),
                'neutral': float(forward.get('neutral', 0.34)),
                'contradiction': float(forward.get('contradiction', 0.33)),
                'reverse_entailment': float(reverse.get('entailment', 0.0)),
                'reverse_neutral': float(reverse.get('neutral', 0.0)),
                'reverse_contradiction': float(reverse.get('contradiction', 0.0)),
            })

        return bidirectional_scores
