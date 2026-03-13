"""
Generator wrapper for LLM text generation with metadata capture.

This module provides the GeneratorWrapper class that uses transformer-based
sequence-to-sequence models (e.g., FLAN-T5) to generate responses while
capturing token-level logits and scores for downstream verifier modules.
"""

import torch
from transformers import AutoTokenizer, AutoModelForSeq2SeqLM
from typing import List, Dict, Optional, Tuple, Any
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
        dtype: Optional[torch.dtype] = torch.float16,
        max_input_tokens: Optional[int] = None
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
            max_input_tokens: Optional tokenizer truncation limit override
        
        Raises:
            ValueError: If model loading fails
        """
        self.model_name = model_name
        self.device = device
        self.max_input_tokens = max_input_tokens
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
            # Prefer safetensors to avoid torch.load CVE guard for .bin checkpoints
            # in environments pinned below torch 2.6.
            safe_tensor_kwargs = {'use_safetensors': True}
            loading_info: Dict[str, Any] = {}

            if load_in_8bit:
                # 8-bit quantization for memory efficiency
                try:
                    self.model, loading_info = AutoModelForSeq2SeqLM.from_pretrained(
                        model_name,
                        output_loading_info=True,
                        load_in_8bit=True,
                        device_map='auto',
                        **safe_tensor_kwargs
                    )
                except Exception as load_err:
                    self.logger.warning(
                        "Safetensors loading failed for %s: %s",
                        model_name,
                        load_err
                    )
                    self.model, loading_info = AutoModelForSeq2SeqLM.from_pretrained(
                        model_name,
                        output_loading_info=True,
                        load_in_8bit=True,
                        device_map='auto'
                    )
                self.logger.info("Model loaded with 8-bit quantization")
            else:
                # Standard loading
                if is_cuda_device and torch.cuda.is_available():
                    model_kwargs = {'device_map': 'auto', **safe_tensor_kwargs}
                    if dtype is not None:
                        model_kwargs['dtype'] = dtype
                    try:
                        self.model, loading_info = AutoModelForSeq2SeqLM.from_pretrained(
                            model_name,
                            output_loading_info=True,
                            **model_kwargs
                        )
                    except Exception as load_err:
                        self.logger.warning(
                            "Safetensors loading failed for %s: %s",
                            model_name,
                            load_err
                        )
                        model_kwargs.pop('use_safetensors', None)
                        self.model, loading_info = AutoModelForSeq2SeqLM.from_pretrained(
                            model_name,
                            output_loading_info=True,
                            **model_kwargs
                        )
                else:
                    cpu_kwargs = {**safe_tensor_kwargs}
                    if dtype is not None and not is_cuda_device:
                        cpu_kwargs['dtype'] = torch.float32 if dtype == torch.float16 else dtype
                    try:
                        cpu_model, loading_info = AutoModelForSeq2SeqLM.from_pretrained(
                            model_name,
                            output_loading_info=True,
                            **cpu_kwargs
                        )
                    except Exception as load_err:
                        self.logger.warning(
                            "Safetensors loading failed for %s: %s",
                            model_name,
                            load_err
                        )
                        cpu_kwargs.pop('use_safetensors', None)
                        cpu_model, loading_info = AutoModelForSeq2SeqLM.from_pretrained(
                            model_name,
                            output_loading_info=True,
                            **cpu_kwargs
                        )
                    self.model = cpu_model.to(device)

                self.logger.info(f"Model loaded successfully on {self.model.device}")

            repair_applied = self._repair_missing_embeddings_if_needed(loading_info)
            tie_applied = self._enforce_tied_embeddings(force_full_sync=repair_applied)
            if repair_applied or tie_applied:
                self.logger.info(
                    "Embedding fixes applied: repair=%s, tie=%s",
                    repair_applied,
                    tie_applied
                )
            
            # Get model memory footprint
            if hasattr(self.model, 'get_memory_footprint'):
                memory_mb = self.model.get_memory_footprint() / (1024 ** 2)
                self.logger.info(f"Model memory footprint: {memory_mb:.2f} MB")
        
        except Exception as e:
            raise ValueError(f"Failed to load model: {e}")

    def _repair_missing_embeddings_if_needed(self, loading_info: Dict[str, Any]) -> bool:
        """Repair missing LongT5 encoder/decoder embeddings from shared weights."""
        if not isinstance(loading_info, dict):
            return False

        missing_keys = set(loading_info.get('missing_keys') or [])
        required_keys = {
            'encoder.embed_tokens.weight',
            'decoder.embed_tokens.weight',
        }
        if not required_keys.issubset(missing_keys):
            return False

        shared = getattr(self.model, 'shared', None)
        encoder = getattr(self.model, 'encoder', None)
        decoder = getattr(self.model, 'decoder', None)
        if shared is None or encoder is None or decoder is None:
            return False
        if getattr(shared, 'weight', None) is None:
            return False
        if getattr(encoder, 'embed_tokens', None) is None or getattr(decoder, 'embed_tokens', None) is None:
            return False

        with torch.no_grad():
            encoder.embed_tokens.weight.copy_(shared.weight)
            decoder.embed_tokens.weight.copy_(shared.weight)

        self.logger.warning(
            "Detected missing encoder/decoder embeddings during load; initialized them from shared.weight"
        )
        return True

    def _enforce_tied_embeddings(self, force_full_sync: bool = False) -> bool:
        """
        Ensure seq2seq embedding matrices are tied consistently after loading.

        When `force_full_sync` is True (used only after missing-key repair),
        encoder/decoder token embeddings are copied from `shared.weight`.
        For normal checkpoints, only model-native `tie_weights()` is used.
        """
        shared = getattr(self.model, 'shared', None)
        encoder = getattr(self.model, 'encoder', None)
        decoder = getattr(self.model, 'decoder', None)
        lm_head = getattr(self.model, 'lm_head', None)

        if shared is None or getattr(shared, 'weight', None) is None:
            return False

        if force_full_sync:
            with torch.no_grad():
                if encoder is not None and getattr(encoder, 'embed_tokens', None) is not None:
                    encoder.embed_tokens.weight.copy_(shared.weight)
                if decoder is not None and getattr(decoder, 'embed_tokens', None) is not None:
                    decoder.embed_tokens.weight.copy_(shared.weight)

        # Re-run model-native tie logic to ensure parameter sharing references are consistent.
        if hasattr(self.model, 'tie_weights'):
            self.model.tie_weights()
        return force_full_sync or (lm_head is not None)

    def _resolve_max_input_tokens(self) -> int:
        """
        Resolve tokenizer truncation length for encoder inputs.

        Priority:
        1) Explicit `max_input_tokens` from config
        2) Model/tokenizer declared max lengths
        3) Safe fallback (512)
        """
        if isinstance(self.max_input_tokens, int) and self.max_input_tokens > 0:
            return int(self.max_input_tokens)

        candidates = []

        tokenizer_max = getattr(self.tokenizer, 'model_max_length', None)
        if isinstance(tokenizer_max, int) and 0 < tokenizer_max < 1_000_000:
            candidates.append(tokenizer_max)

        model_config = getattr(self.model, 'config', None)
        for attr_name in ('n_positions', 'max_position_embeddings', 'max_encoder_position_embeddings'):
            value = getattr(model_config, attr_name, None)
            if isinstance(value, int) and value > 0:
                candidates.append(value)

        if candidates:
            return int(min(candidates))

        return 512
    
    def _format_prompt(
        self,
        prompt: str,
        evidence_chunks: List[EvidenceChunk]
    ) -> str:
        """
        Format the prompt with evidence context.
        
        Creates a structured prompt that includes relevant evidence chunks
        followed by the user's question. Uses "Passage N:" format to avoid
        confusion with citation-style markers.
        
        Args:
            prompt: User's query/question
            evidence_chunks: List of relevant evidence chunks
        
        Returns:
            Formatted prompt string with context and question
        
        Example:
            Context: Passage 1: Deep learning is a type of machine learning...
            
            Passage 2: Neural networks use multiple layers...
            
            Question: What is machine learning?
            
            Answer:
        
        Note:
            Previously used [1] [2] [3] citation markers, but changed to
            "Passage N:" format to prevent FLAN-T5 from generating citation
            references like "[1]" instead of actual answers.
        """
        if not evidence_chunks:
            # No evidence provided, just use the question
            return f"Question: {prompt}\n\nAnswer:"
        
        # Format evidence context without citation numbers to avoid confusion
        # FLAN-T5 sometimes interprets [1] [2] [3] as citation references
        # Instead, use "Passage N:" format which is less ambiguous
        evidence_texts = []
        for i, chunk in enumerate(evidence_chunks, 1):
            evidence_texts.append(f"Passage {i}: {chunk.text}")
        
        # Join with double newlines for clear separation
        evidence_context = "\n\n".join(evidence_texts)
        
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
        max_input_tokens = self._resolve_max_input_tokens()
        inputs = self.tokenizer(
            formatted_prompt,
            return_tensors='pt',
            truncation=True,
            max_length=max_input_tokens
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

        # Detect severe decoding collapse early (e.g., same token repeated).
        if outputs.sequences is not None and outputs.sequences.shape[1] > 8:
            gen_ids = outputs.sequences[0].tolist()
            tail = gen_ids[1:] if len(gen_ids) > 1 else gen_ids
            if tail:
                dominant_ratio = max(tail.count(tid) for tid in set(tail)) / len(tail)
                if dominant_ratio >= 0.8:
                    self.logger.warning(
                        "Detected degenerate decoding (dominant_token_ratio=%.2f). "
                        "Model '%s' may be unsuitable for instruction QA prompts.",
                        dominant_ratio,
                        self.model_name
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

    def score_target_with_metadata(
        self,
        prompt: str,
        target_text: str,
        evidence_chunks: Optional[List[EvidenceChunk]] = None
    ) -> Dict:
        """
        Score a provided target text with teacher forcing and return token-level metadata.

        This method is designed for benchmark settings where response text is fixed
        (e.g., RAGTruth gold responses) but intrinsic uncertainty signals are still
        required. It computes per-token entropies over the decoder distribution for
        the forced target sequence.

        Args:
            prompt: Input prompt/query used for conditioning
            target_text: Target response text to score
            evidence_chunks: Optional evidence chunks used to format the prompt

        Returns:
            Metadata dictionary containing target text, tokens, token entropies,
            and per-token probabilities/log-probabilities for the forced sequence.
        """
        if evidence_chunks is None:
            evidence_chunks = []

        formatted_prompt = self._format_prompt(prompt, evidence_chunks)

        model_inputs = self.tokenizer(
            formatted_prompt,
            text_target=target_text,
            return_tensors='pt',
            truncation=True,
            max_length=self._resolve_max_input_tokens()
        ).to(self.model.device)

        labels = model_inputs.get('labels')
        if labels is None:
            raise ValueError("Tokenizer did not return labels for teacher-forcing scoring")

        with torch.no_grad():
            outputs = self.model(
                input_ids=model_inputs['input_ids'],
                attention_mask=model_inputs.get('attention_mask'),
                labels=labels
            )

        logits = outputs.logits[0]  # (target_len, vocab_size)
        label_ids = labels[0]

        token_ids = []
        tokens = []
        token_entropies = []
        token_probs = []
        token_logprobs = []

        for idx in range(label_ids.shape[0]):
            token_id = int(label_ids[idx].item())
            if token_id < 0:
                continue

            step_logits = logits[idx]
            step_probs = torch.softmax(step_logits, dim=-1)
            step_log_probs = torch.log(step_probs + 1e-12)
            entropy = -torch.sum(step_probs * step_log_probs)

            token_prob = float(step_probs[token_id].item())
            token_logprob = float(step_log_probs[token_id].item())

            token_ids.append(token_id)
            tokens.append(self.tokenizer.decode([token_id], skip_special_tokens=False))
            token_entropies.append(float(entropy.item()))
            token_probs.append(token_prob)
            token_logprobs.append(token_logprob)

        evidence_used = [chunk.doc_id for chunk in evidence_chunks]

        return {
            'text': target_text,
            'prompt_text': formatted_prompt,
            'tokens': tokens,
            'token_ids': token_ids,
            'token_entropies': token_entropies,
            'scores': token_probs,
            'token_logprobs': token_logprobs,
            'evidence_used': evidence_used,
            'generation_config': {
                'mode': 'teacher_forcing',
                'model_name': self.model_name
            }
        }
