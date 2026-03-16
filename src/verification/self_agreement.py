"""
Self-Agreement Detector for consistency-based hallucination detection.

This module implements the SelfAgreementDetector which measures the consistency
of LLM outputs by generating multiple stochastic responses to the same query
and measuring their semantic similarity. Low consistency indicates high
uncertainty and potential hallucination.

Based on the self-consistency method from:
- Wang et al. (2022): "Self-Consistency Improves Chain of Thought Reasoning"
"""

import torch
import numpy as np
from typing import List, Dict, Optional, Tuple
from sentence_transformers import SentenceTransformer

from src.utils.data_structures import EvidenceChunk
from src.utils.logger import setup_logger


class SelfAgreementDetector:
    """
    Detector that measures consistency by generating multiple stochastic
    responses and comparing their semantic similarity.
    
    This detector helps identify hallucinations by exposing inconsistencies
    in model outputs. When a model is uncertain about its answer, multiple
    stochastic generations will produce diverse (inconsistent) responses.
    Conversely, confident correct answers tend to be consistent across samples.
    
    Attributes:
        config: Configuration dictionary
        generator: GeneratorWrapper instance for text generation
        similarity_model: SentenceTransformer for embedding-based similarity
        k_samples: Number of stochastic samples to generate
        temperature: Sampling temperature for generation
        device: Device for model inference
        logger: Logger instance
    
    Example:
        >>> from src.generation.generator_wrapper import GeneratorWrapper
        >>> from src.config import Config
        >>> 
        >>> config = Config.from_yaml('config.yaml')
        >>> generator = GeneratorWrapper(config.models.generator)
        >>> detector = SelfAgreementDetector(config, generator)
        >>> 
        >>> result = detector.detect(
        ...     claim_text="Machine learning is a subset of AI",
        ...     query="What is machine learning?",
        ...     evidence_chunks=[chunk1, chunk2]
        ... )
        >>> print(f"Consistency score: {result['consistency_score']:.3f}")
    """
    
    def __init__(self, config: Dict, generator):
        """
        Initialize the Self-Agreement Detector.
        
        Args:
            config: Configuration dictionary with self_agreement settings
            generator: GeneratorWrapper instance for text generation
        
        Raises:
            ValueError: If configuration is invalid or model loading fails
        """
        self.config = config
        self.generator = generator
        self.logger = setup_logger(__name__)
        
        # Extract configuration
        sa_config = config.get('verification', {}).get('self_agreement', {})
        self.model_name = sa_config.get('model_name', 'sentence-transformers/all-MiniLM-L6-v2')
        self.k_samples = sa_config.get('k_samples', 5)
        self.temperature = sa_config.get('temperature', 1.5)
        self.device = sa_config.get('device', 'cuda' if torch.cuda.is_available() else 'cpu')
        self.deterministic = sa_config.get('deterministic', False)
        self.random_seed = sa_config.get('random_seed', 42)
        
        self.logger.info(f"Initializing SelfAgreementDetector...")
        self.logger.info(f"Model: {self.model_name}")
        self.logger.info(f"k_samples: {self.k_samples}, temperature: {self.temperature}")
        self.logger.info(f"Device: {self.device}")
        self.logger.info(f"Deterministic mode: {self.deterministic} (seed={self.random_seed})")
        
        # Load sentence-transformer model for similarity measurement
        try:
            self.similarity_model = SentenceTransformer(self.model_name, device=self.device)
            self.logger.info(f"Similarity model loaded successfully")
        except Exception as e:
            self.logger.error(f"Failed to load similarity model: {e}")
            raise ValueError(f"Failed to load similarity model: {e}")
        
        self.logger.info("SelfAgreementDetector initialized successfully")
    
    def generate_samples(
        self,
        query: str,
        evidence_chunks: Optional[List[EvidenceChunk]] = None,
        k: Optional[int] = None
    ) -> List[str]:
        """
        Generate k stochastic responses for the same query.
        
        Uses the generator with do_sample=True and higher temperature to
        produce diverse responses. Each response is generated independently
        to capture natural variation in model uncertainty.
        
        Args:
            query: User's original query/question
            evidence_chunks: List of evidence chunks to condition on
            k: Number of samples to generate (defaults to self.k_samples)
        
        Returns:
            List of k generated response texts
        
        Raises:
            ValueError: If query is empty or None
            RuntimeError: If generation fails
        
        Example:
            >>> samples = detector.generate_samples(
            ...     query="What is machine learning?",
            ...     evidence_chunks=[chunk1, chunk2],
            ...     k=5
            ... )
            >>> print(f"Generated {len(samples)} samples")
        """
        if not query or not query.strip():
            raise ValueError("query cannot be empty")
        
        if k is None:
            k = self.k_samples
        
        if evidence_chunks is None:
            evidence_chunks = []
        
        # Set random seed for deterministic mode
        if self.deterministic:
            torch.manual_seed(self.random_seed)
            if torch.cuda.is_available():
                torch.cuda.manual_seed_all(self.random_seed)
            np.random.seed(self.random_seed)
            self.logger.debug(f"Deterministic mode: seed set to {self.random_seed}")
        
        self.logger.debug(f"Generating {k} samples for query: {query[:50]}...")

        samples = []
        try:
            if hasattr(self.generator, 'generate_n_samples'):
                batch_samples = self.generator.generate_n_samples(
                    prompt=query,
                    evidence_chunks=evidence_chunks,
                    num_samples=k,
                    max_new_tokens=256,
                    temperature=self.temperature,
                    top_p=0.95,
                    do_sample=True,
                    sanitize_meta_text=True,
                )
                for i, generated_text in enumerate(batch_samples):
                    if generated_text and generated_text.strip():
                        samples.append(generated_text)
                        self.logger.debug(f"Sample {i+1}/{k}: {generated_text[:50]}...")
                    else:
                        self.logger.warning(f"Sample {i+1}/{k} is empty, skipping")
            else:
                raise AttributeError("Generator does not implement generate_n_samples")
        except Exception as e:
            self.logger.warning(
                "Batched sample generation unavailable, falling back to sequential mode: %s",
                e
            )
            for i in range(k):
                try:
                    result = self.generator.generate_with_metadata(
                        prompt=query,
                        evidence_chunks=evidence_chunks,
                        max_new_tokens=256,
                        temperature=self.temperature,
                        top_p=0.95,
                        do_sample=True
                    )

                    generated_text = result['text']

                    if generated_text and generated_text.strip():
                        samples.append(generated_text)
                        self.logger.debug(f"Sample {i+1}/{k}: {generated_text[:50]}...")
                    else:
                        self.logger.warning(f"Sample {i+1}/{k} is empty, skipping")

                except Exception as inner_e:
                    self.logger.error(f"Failed to generate sample {i+1}/{k}: {inner_e}")
                    continue
        
        # Validate we have at least some samples
        if len(samples) == 0:
            raise RuntimeError(f"All {k} generation attempts produced empty samples")
        
        self.logger.debug(f"Successfully generated {len(samples)}/{k} valid samples")
        return samples
    
    def measure_consistency(
        self,
        original_answer: str,
        samples: List[str]
    ) -> Dict[str, float]:
        """
        Measure semantic consistency between original answer and samples.
        
        Computes embeddings for all texts and calculates pairwise cosine
        similarities. Returns the mean similarity as the consistency score,
        along with variance and individual similarities for analysis.
        
        High consistency (score ~1.0) indicates confident, consistent outputs.
        Low consistency (score <0.5) indicates uncertainty and potential
        hallucination.
        
        Args:
            original_answer: The original generated answer to compare against
            samples: List of stochastically generated samples
        
        Returns:
            Dictionary containing:
                - consistency_score: Mean cosine similarity (0.0 to 1.0)
                - variance: Variance of similarities
                - individual_scores: List of similarity scores with each sample
                - min_score: Minimum similarity
                - max_score: Maximum similarity
        
        Raises:
            ValueError: If inputs are empty or invalid
        
        Example:
            >>> result = detector.measure_consistency(
            ...     original_answer="ML is a subset of AI",
            ...     samples=["ML is part of AI", "ML belongs to AI", "ML is AI"]
            ... )
            >>> print(f"Consistency: {result['consistency_score']:.3f}")
        """
        if not original_answer or not original_answer.strip():
            raise ValueError("original_answer cannot be empty")
        
        if not samples or len(samples) == 0:
            raise ValueError("samples list cannot be empty")
        
        self.logger.debug(f"Measuring consistency between original and {len(samples)} samples")
        
        try:
            # Encode all texts into embeddings
            all_texts = [original_answer] + samples
            embeddings = self.similarity_model.encode(
                all_texts,
                convert_to_tensor=True,
                show_progress_bar=False
            )
            
            # Original answer embedding (first one)
            original_embedding = embeddings[0:1]  # Keep dimension (1, D)
            
            # Sample embeddings (rest)
            sample_embeddings = embeddings[1:]  # Shape: (k, D)
            
            # Compute cosine similarities between original and each sample
            similarities = self.similarity_model.similarity(
                original_embedding,
                sample_embeddings
            )
            
            # Convert to list of float scores
            similarities = similarities[0].cpu().numpy().tolist()
            
            # Calculate statistics
            consistency_score = float(np.mean(similarities))
            variance = float(np.var(similarities))
            min_score = float(np.min(similarities))
            max_score = float(np.max(similarities))
            
            self.logger.debug(
                f"Consistency: mean={consistency_score:.3f}, "
                f"var={variance:.3f}, min={min_score:.3f}, max={max_score:.3f}"
            )
            
            return {
                'consistency_score': consistency_score,
                'variance': variance,
                'individual_scores': similarities,
                'min_score': min_score,
                'max_score': max_score
            }
            
        except Exception as e:
            self.logger.error(f"Failed to measure consistency: {e}")
            # Return fallback neutral scores
            return {
                'consistency_score': 0.5,
                'variance': 0.0,
                'individual_scores': [0.5] * len(samples),
                'min_score': 0.5,
                'max_score': 0.5
            }
    
    def detect(
        self,
        claim_text: str,
        query: str,
        evidence_chunks: Optional[List[EvidenceChunk]] = None
    ) -> Dict[str, float]:
        """
        Main detection method: measure self-agreement consistency.
        
        Generates k stochastic samples for the query and measures how
        consistent they are with the original claim. This is the primary
        interface called by VerifierHub.
        
        Args:
            claim_text: The original generated claim/answer to verify
            query: The query/question that generated the claim
            evidence_chunks: Evidence chunks used for generation
        
        Returns:
            Dictionary containing:
                - variance: Consistency variance (None if error)
                - score: Consistency score (0.0 to 1.0, higher = more consistent)
                - samples_generated: Number of samples successfully generated
        
        Raises:
            ValueError: If inputs are invalid
        
        Example:
            >>> result = detector.detect(
            ...     claim_text="Machine learning is a subset of AI",
            ...     query="What is machine learning?",
            ...     evidence_chunks=[chunk1, chunk2]
            ... )
            >>> if result['score'] < 0.5:
            ...     print("Low consistency - potential hallucination!")
        """
        if not claim_text or not claim_text.strip():
            raise ValueError("claim_text cannot be empty")
        
        if not query or not query.strip():
            raise ValueError("query cannot be empty")
        
        self.logger.debug(f"Detecting self-agreement for claim: {claim_text[:50]}...")
        
        try:
            # Generate k stochastic samples
            samples = self.generate_samples(query, evidence_chunks)
            
            # Measure consistency
            consistency_result = self.measure_consistency(claim_text, samples)

            self.logger.info(
                "detector_self_agreement",
                extra={
                    "event": "detector_self_agreement",
                    "data": {
                        "score": consistency_result.get('consistency_score', None),
                        "variance": consistency_result.get('variance', None),
                        "samples_generated": len(samples)
                    }
                }
            )
            
            # Return formatted result for VerifierHub
            return {
                'variance': consistency_result['variance'],
                'score': consistency_result['consistency_score'],
                'samples_generated': len(samples)
            }
            
        except Exception as e:
            self.logger.error(f"Self-agreement detection failed: {e}")
            # Return fallback indicating detection failure
            return {
                'variance': None,
                'score': None,
                'samples_generated': 0
            }
