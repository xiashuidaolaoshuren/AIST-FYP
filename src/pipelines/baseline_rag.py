"""
Baseline RAG Pipeline integrating retrieval and generation.

This module implements the core end-to-end RAG pipeline that retrieves
evidence, generates responses, and creates claim-evidence pairs for
downstream verification in Month 3-5.
"""

from typing import Dict, List, Optional, Any
from pathlib import Path

from src.retrieval.dense_retriever import DenseRetriever
from src.generation.generator_wrapper import GeneratorWrapper
from src.generation.claim_extractor import extract_claims
from src.utils.data_structures import ClaimEvidencePair, EvidenceChunk, Claim
from src.utils.config import Config
from src.utils.logger import setup_logger


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
        
        self.logger.info("BaselineRAGPipeline initialized")
    
    def run(
        self,
        query: str,
        top_k: int = 5,
        max_new_tokens: Optional[int] = None,
        temperature: Optional[float] = None,
        top_p: Optional[float] = None,
        do_sample: Optional[bool] = None
    ) -> Dict[str, Any]:
        """
        Run the complete RAG pipeline on a query.
        
        Executes the following steps:
        1. Retrieve top-k evidence chunks using DenseRetriever
        2. Generate response with metadata using GeneratorWrapper
        3. Extract atomic claims from generated text
        4. Create ClaimEvidencePair objects pairing claims with evidence
        5. Format output matching System_Architecture_Design.md specification
        
        Args:
            query: User's input question
            top_k: Number of evidence chunks to retrieve (default: 5)
            max_new_tokens: Max tokens to generate (uses config if None)
            temperature: Sampling temperature (uses config if None)
            top_p: Nucleus sampling threshold (uses config if None)
            do_sample: Whether to use sampling (uses config if None)
        
        Returns:
            Dictionary containing:
                - query: Original query string
                - draft_response: Generated text response
                - claim_evidence_pairs: List of ClaimEvidencePair dicts
                - generator_metadata: Full metadata from generation (for Month 3)
                - retrieval_metadata: Metadata about retrieved evidence
        
        Example:
            >>> result = pipeline.run("What is machine learning?", top_k=3)
            >>> print(result['draft_response'])
            >>> for pair in result['claim_evidence_pairs']:
            ...     print(f"Claim: {pair['claim_id']}")
            ...     print(f"Evidence: {pair['top_evidence']}")
        """
        self.logger.info(f"Running RAG pipeline for query: {query[:50]}...")
        
        # Step 1: Retrieve evidence
        self.logger.debug(f"Retrieving top-{top_k} evidence chunks")
        evidence_chunks = self.retriever.retrieve(query, top_k=top_k)
        
        if not evidence_chunks:
            self.logger.warning("No evidence retrieved for query")
        else:
            self.logger.info(
                f"Retrieved {len(evidence_chunks)} evidence chunks, "
                f"top score: {evidence_chunks[0].score_dense:.4f}"
            )
        
        # Step 2: Generate response with metadata
        # Use config values if not explicitly provided
        gen_params = {}
        if self.config:
            gen_params = {
                'max_new_tokens': max_new_tokens or self.config.generation.max_new_tokens,
                'temperature': temperature if temperature is not None else self.config.generation.temperature,
                'top_p': top_p if top_p is not None else self.config.generation.top_p,
                'do_sample': do_sample if do_sample is not None else self.config.generation.do_sample
            }
        else:
            gen_params = {
                'max_new_tokens': max_new_tokens or 256,
                'temperature': temperature if temperature is not None else 0.7,
                'top_p': top_p if top_p is not None else 0.9,
                'do_sample': do_sample if do_sample is not None else True
            }
        
        self.logger.debug(f"Generating response with params: {gen_params}")
        generation_output = self.generator.generate_with_metadata(
            prompt=query,
            evidence_chunks=evidence_chunks,
            **gen_params
        )
        
        self.logger.info(
            f"Generated response: {len(generation_output['text'])} chars, "
            f"{len(generation_output['tokens'])} tokens"
        )
        
        # Step 3: Extract claims from generated text
        self.logger.debug("Extracting claims from generated text")
        claims = extract_claims(
            text=generation_output['text'],
            method='auto'  # Use auto method selection (spaCy if available, else regex)
        )
        
        self.logger.info(f"Extracted {len(claims)} claims")
        
        # Step 4: Create claim-evidence pairs
        # Baseline: Pair each claim with all retrieved evidence
        # Month 3 verifier will do more sophisticated claim-evidence matching
        claim_evidence_pairs = []
        
        for claim in claims:
            # Create evidence candidate IDs in format "doc_id#sent_id"
            evidence_candidates = [
                f"{chunk.doc_id}#{chunk.sent_id}"
                for chunk in evidence_chunks
            ]
            
            # Top evidence is the first (highest-ranked) chunk
            top_evidence = evidence_candidates[0] if evidence_candidates else ""
            
            # Convert evidence chunks to dicts for serialization
            evidence_spans = [chunk.to_dict() for chunk in evidence_chunks]
            
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
        
        # Step 5: Format output
        output = {
            'query': query,
            'draft_response': generation_output['text'],
            'claim_evidence_pairs': [pair.to_dict() for pair in claim_evidence_pairs],
            'generator_metadata': generation_output,
            'retrieval_metadata': {
                'top_k': top_k,
                'num_retrieved': len(evidence_chunks),
                'top_score': evidence_chunks[0].score_dense if evidence_chunks else 0.0,
                'evidence_doc_ids': [chunk.doc_id for chunk in evidence_chunks]
            }
        }
        
        self.logger.info(
            f"Pipeline complete: {len(claims)} claims, "
            f"{len(evidence_chunks)} evidence chunks"
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
        
        # Initialize DenseRetriever
        logger.info(f"Initializing DenseRetriever with {config.models.sentence_transformer}")
        retriever = DenseRetriever(
            index_path=str(index_path),
            metadata_path=str(metadata_path),
            encoder_model=config.models.sentence_transformer,
            device=config.processing.device
        )
        
        # Initialize GeneratorWrapper
        logger.info(f"Initializing GeneratorWrapper with {config.models.generator}")
        generator = GeneratorWrapper(
            model_name=config.models.generator,
            device=config.processing.device,
            load_in_8bit=config.generation.load_in_8bit
        )
        
        logger.info("Pipeline initialization complete")
        
        # Create and return pipeline
        return cls(
            retriever=retriever,
            generator=generator,
            config=config
        )
