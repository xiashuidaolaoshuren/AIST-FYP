"""
Generator wrapper for LLM text generation with metadata capture.

This module provides the GeneratorWrapper class that uses transformer-based
sequence-to-sequence models (e.g., FLAN-T5) to generate responses while
capturing token-level logits and scores for downstream verifier modules.
"""

import torch
from transformers import AutoTokenizer, AutoModelForSeq2SeqLM
from typing import List, Dict, Optional, Tuple
import numpy as np

from src.utils.data_structures import EvidenceChunk
from src.utils.logger import setup_logger


class GeneratorWrapper:
    """
    Wrapper for seq2seq LLM generation with metadata capture.
    
    Loads a transformer-based seq2seq model (e.g., FLAN-T5, T5, mT5) and
    generates text responses while capturing token-level metadata including
    logits, scores, and evidence usage for hallucination detection.
    
    Attributes:
        model_name: Name of the pretrained model
        tokenizer: Loaded tokenizer
        model: Loaded model
        device: Device for inference ('cuda' or 'cpu')
        logger: Logger instance
    
    Example:
        >>> generator = GeneratorWrapper('google/flan-t5-base')
        >>> evidence = [EvidenceChunk(...), ...]
        >>> result = generator.generate_with_metadata(
        ...     prompt="What is machine learning?",
        ...     evidence_chunks=evidence
        ... )
        >>> print(result['text'])
    """
    
    def __init__(
        self,
        model_name: str = 'google/flan-t5-base',
        device: str = 'cuda',
        load_in_8bit: bool = False,
        dtype: Optional[torch.dtype] = torch.float16
    ):
        """
        Initialize the generator wrapper.
        
        Loads the tokenizer and model with optional 8-bit quantization
        for memory efficiency on larger models.
        
        Args:
            model_name: HuggingFace model name (default: google/flan-t5-base)
            device: Device to run on ('cuda' or 'cpu')
            load_in_8bit: Whether to use 8-bit quantization (for models >1GB)
            dtype: Data type for model weights (default: float16 for GPU)
        
        Raises:
            ValueError: If model loading fails
        """
        self.model_name = model_name
        self.device = device
        self.logger = setup_logger(__name__)
        
        self.logger.info(f"Loading model: {model_name}")
        self.logger.info(f"Device: {device}, 8-bit: {load_in_8bit}, dtype: {dtype}")

        # Determine whether we should forward the dtype argument based on device
        is_cuda_device = isinstance(device, str) and device.startswith('cuda')
        
        # Load tokenizer
        try:
            self.tokenizer = AutoTokenizer.from_pretrained(model_name)
            self.logger.info("Tokenizer loaded successfully")
        except Exception as e:
            raise ValueError(f"Failed to load tokenizer: {e}")
        
        # Load model with appropriate settings
        try:
            if load_in_8bit:
                # 8-bit quantization for memory efficiency
                self.model = AutoModelForSeq2SeqLM.from_pretrained(
                    model_name,
                    load_in_8bit=True,
                    device_map='auto'
                )
                self.logger.info("Model loaded with 8-bit quantization")
            else:
                # Standard loading
                if is_cuda_device and torch.cuda.is_available():
                    model_kwargs = {'device_map': 'auto'}
                    if dtype is not None:
                        model_kwargs['dtype'] = dtype
                    self.model = AutoModelForSeq2SeqLM.from_pretrained(
                        model_name,
                        **model_kwargs
                    )
                else:
                    cpu_kwargs = {}
                    if dtype is not None and not is_cuda_device:
                        cpu_kwargs['dtype'] = dtype
                    cpu_model = AutoModelForSeq2SeqLM.from_pretrained(
                        model_name,
                        **cpu_kwargs
                    )
                    self.model = cpu_model.to(device)
                
                self.logger.info(f"Model loaded successfully on {self.model.device}")
            
            # Get model memory footprint
            if hasattr(self.model, 'get_memory_footprint'):
                memory_mb = self.model.get_memory_footprint() / (1024 ** 2)
                self.logger.info(f"Model memory footprint: {memory_mb:.2f} MB")
        
        except Exception as e:
            raise ValueError(f"Failed to load model: {e}")
    
    def _format_prompt(
        self,
        prompt: str,
        evidence_chunks: List[EvidenceChunk]
    ) -> str:
        """
        Format the prompt with evidence context.
        
        Creates a structured prompt that includes relevant evidence chunks
        followed by the user's question.
        
        Args:
            prompt: User's query/question
            evidence_chunks: List of relevant evidence chunks
        
        Returns:
            Formatted prompt string with context and question
        
        Example:
            Context: [Evidence 1] [Evidence 2] ...
            
            Question: What is machine learning?
            
            Answer:
        """
        if not evidence_chunks:
            # No evidence provided, just use the question
            return f"Question: {prompt}\n\nAnswer:"
        
        # Format evidence context
        evidence_texts = []
        for i, chunk in enumerate(evidence_chunks, 1):
            evidence_texts.append(f"[{i}] {chunk.text}")
        
        evidence_context = " ".join(evidence_texts)
        
        # Create structured prompt
        formatted_prompt = (
            f"Context: {evidence_context}\n\n"
            f"Question: {prompt}\n\n"
            f"Answer:"
        )
        
        return formatted_prompt
    
    def generate_with_metadata(
        self,
        prompt: str,
        evidence_chunks: Optional[List[EvidenceChunk]] = None,
        max_new_tokens: int = 256,
        temperature: float = 1.0,
        top_p: float = 1.0,
        num_beams: int = 1,
        do_sample: bool = False
    ) -> Dict:
        """
        Generate text response with comprehensive metadata capture.
        
        Produces a text response to the prompt using the provided evidence,
        while capturing token-level logits, scores, and other metadata
        needed for hallucination detection in Month 3.
        
        Args:
            prompt: User's query/question
            evidence_chunks: List of relevant evidence chunks (optional)
            max_new_tokens: Maximum number of tokens to generate
            temperature: Sampling temperature (1.0 = neutral, <1 = focused)
            top_p: Nucleus sampling threshold
            num_beams: Number of beams for beam search (1 = greedy)
            do_sample: Whether to use sampling (vs greedy/beam search)
        
        Returns:
            Dictionary containing:
                - text: Generated response text
                - prompt_text: Formatted input prompt
                - tokens: List of generated token strings
                - token_ids: List of generated token IDs
                - logits: List of logit tensors for each generated token
                - scores: List of probability scores for each token
                - evidence_used: List of doc_ids from evidence chunks
                - generation_config: Dict of generation parameters used
        
        Example:
            >>> result = generator.generate_with_metadata(
            ...     prompt="What is AI?",
            ...     evidence_chunks=[chunk1, chunk2],
            ...     max_new_tokens=128,
            ...     temperature=0.7
            ... )
            >>> print(result['text'])
            >>> print(f"Used {len(result['tokens'])} tokens")
        """
        if evidence_chunks is None:
            evidence_chunks = []
        
        # Format prompt with evidence
        formatted_prompt = self._format_prompt(prompt, evidence_chunks)
        self.logger.debug(f"Formatted prompt length: {len(formatted_prompt)} chars")
        
        # Tokenize input
        inputs = self.tokenizer(
            formatted_prompt,
            return_tensors='pt',
            truncation=True,
            max_length=512
        ).to(self.model.device)
        
        # Generate with metadata capture
        self.logger.debug(
            f"Generating with max_new_tokens={max_new_tokens}, "
            f"temp={temperature}, top_p={top_p}"
        )
        
        with torch.no_grad():
            outputs = self.model.generate(
                **inputs,
                max_new_tokens=max_new_tokens,
                temperature=temperature,
                top_p=top_p,
                num_beams=num_beams,
                do_sample=do_sample,
                output_scores=True,
                return_dict_in_generate=True
            )
        
        # Extract generated sequence
        generated_ids = outputs.sequences[0]  # Remove batch dimension
        
        # Decode generated text (skip input tokens for seq2seq)
        generated_text = self.tokenizer.decode(
            generated_ids,
            skip_special_tokens=True
        )
        
        # Extract token-level information
        # For seq2seq models, we need to decode each token individually
        generated_tokens = []
        for token_id in generated_ids:
            token_str = self.tokenizer.decode([token_id], skip_special_tokens=False)
            generated_tokens.append(token_str)
        
        # Convert token IDs to list
        token_ids = generated_ids.cpu().numpy().tolist()
        
        # Extract scores (logits for each position)
        # outputs.scores is a tuple of tensors, one per generated token
        logits_list = []
        scores_list = []
        
        if hasattr(outputs, 'scores') and outputs.scores:
            for score_tensor in outputs.scores:
                # score_tensor shape: (batch_size, vocab_size)
                # Take first batch item
                logits = score_tensor[0].cpu().numpy()
                logits_list.append(logits)
                
                # Compute probabilities
                probs = torch.softmax(score_tensor[0], dim=-1).cpu().numpy()
                # Get probability of the selected token
                selected_token_idx = len(scores_list)
                if selected_token_idx < len(generated_ids):
                    selected_token_id = generated_ids[selected_token_idx].item()
                    selected_prob = probs[selected_token_id]
                    scores_list.append(float(selected_prob))
        
        # Extract evidence usage
        evidence_used = [chunk.doc_id for chunk in evidence_chunks]
        
        # Create metadata dictionary
        metadata = {
            'text': generated_text,
            'prompt_text': formatted_prompt,
            'tokens': generated_tokens,
            'token_ids': token_ids,
            'logits': logits_list,  # List of numpy arrays
            'scores': scores_list,  # List of floats (probabilities)
            'evidence_used': evidence_used,
            'generation_config': {
                'max_new_tokens': max_new_tokens,
                'temperature': temperature,
                'top_p': top_p,
                'num_beams': num_beams,
                'do_sample': do_sample,
                'model_name': self.model_name
            }
        }
        
        self.logger.info(
            f"Generated {len(generated_tokens)} tokens, "
            f"text length: {len(generated_text)} chars"
        )
        
        return metadata
    
    def generate_batch(
        self,
        prompts: List[str],
        evidence_chunks_list: List[List[EvidenceChunk]],
        max_new_tokens: int = 256,
        **generation_kwargs
    ) -> List[Dict]:
        """
        Generate responses for multiple prompts in batch.
        
        Note: Batch generation without metadata capture for efficiency.
        For metadata, use generate_with_metadata() in a loop.
        
        Args:
            prompts: List of user queries
            evidence_chunks_list: List of evidence lists (one per prompt)
            max_new_tokens: Maximum tokens to generate per prompt
            **generation_kwargs: Additional generation parameters
        
        Returns:
            List of metadata dictionaries, one per prompt
        """
        results = []
        for prompt, evidence_chunks in zip(prompts, evidence_chunks_list):
            result = self.generate_with_metadata(
                prompt=prompt,
                evidence_chunks=evidence_chunks,
                max_new_tokens=max_new_tokens,
                **generation_kwargs
            )
            results.append(result)
        
        return results
