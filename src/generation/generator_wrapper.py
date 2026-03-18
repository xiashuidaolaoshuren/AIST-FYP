"""
Generator wrapper for LLM text generation with metadata capture.

This module provides the GeneratorWrapper class that uses transformer-based
sequence-to-sequence models (e.g., FLAN-T5) to generate responses while
capturing token-level logits and scores for downstream verifier modules.
"""

import torch
import re
from transformers import (
    AutoConfig,
    AutoTokenizer,
    AutoModelForSeq2SeqLM,
    AutoModelForCausalLM,
)
from typing import List, Dict, Optional, Tuple, Any, Union
import numpy as np

from src.utils.data_structures import EvidenceChunk
from src.utils.logger import setup_logger


class GeneratorWrapper:
    """
    Wrapper for seq2seq/causal LLM generation with metadata capture.
    
    Loads a transformer-based model and generates text responses while
    capturing token-level metadata including logits, scores, and evidence
    usage for hallucination detection.
    
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
        dtype: Union[str, torch.dtype, None] = 'bf16',
        max_input_tokens: Optional[int] = None,
        enable_thinking: bool = True,
    ):
        """
        Initialize the generator wrapper.
        
        Loads the tokenizer and model with optional 8-bit quantization
        for memory efficiency on larger models.
        
        Args:
            model_name: HuggingFace model name (default: google/flan-t5-base)
            device: Device to run on ('cuda' or 'cpu')
            dtype: Precision mode ('bf16'|'fp16'|'fp32'|'8bit') or torch dtype
            max_input_tokens: Optional tokenizer truncation limit override
            enable_thinking: Whether to enable reasoning mode in supported chat templates
        
        Raises:
            ValueError: If model loading fails
        """
        self.model_name = model_name
        self.device = device
        self.max_input_tokens = max_input_tokens
        self.enable_thinking = bool(enable_thinking)
        self.model_family = 'seq2seq'
        self.logger = setup_logger(__name__)
        
        self.logger.info(f"Loading model: {model_name}")
        normalized_dtype_mode = self._normalize_dtype_mode(dtype)
        is_8bit_mode = normalized_dtype_mode == '8bit'
        selected_torch_dtype = self._resolve_torch_dtype(normalized_dtype_mode)

        self.logger.info(
            "Device: %s, dtype_mode: %s, torch_dtype: %s",
            device,
            normalized_dtype_mode,
            selected_torch_dtype,
        )

        # Determine whether we should forward the dtype argument based on device
        is_cuda_device = isinstance(device, str) and device.startswith('cuda')
        if not is_cuda_device and selected_torch_dtype in (torch.float16, torch.bfloat16):
            # CPU float16/bfloat16 support is not reliable across transformer stacks.
            selected_torch_dtype = torch.float32
            self.logger.info("CPU device detected, upgrading dtype to float32")
        
        # Load tokenizer
        try:
            self.tokenizer = AutoTokenizer.from_pretrained(model_name)
            if self.tokenizer.pad_token_id is None and self.tokenizer.eos_token_id is not None:
                # Causal chat models often don't define pad_token; set a safe fallback.
                self.tokenizer.pad_token = self.tokenizer.eos_token
            self.logger.info("Tokenizer loaded successfully")
        except Exception as e:
            raise ValueError(f"Failed to load tokenizer: {e}")

        # Detect architecture family first so loading and generation logic can branch safely.
        try:
            model_config = AutoConfig.from_pretrained(model_name)
            self.model_family = 'seq2seq' if bool(getattr(model_config, 'is_encoder_decoder', False)) else 'causal'
            self.logger.info("Detected generator family: %s", self.model_family)
        except Exception as e:
            raise ValueError(f"Failed to inspect model architecture: {e}")
        
        # Load model with appropriate settings
        try:
            # Prefer safetensors to avoid torch.load CVE guard for .bin checkpoints
            # in environments pinned below torch 2.6.
            safe_tensor_kwargs = {'use_safetensors': True}
            loading_info: Dict[str, Any] = {}
            model_cls = AutoModelForSeq2SeqLM if self.model_family == 'seq2seq' else AutoModelForCausalLM

            quantization_kwargs: Dict[str, Any] = {}
            if is_8bit_mode:
                try:
                    from transformers import BitsAndBytesConfig
                    quantization_kwargs['quantization_config'] = BitsAndBytesConfig(load_in_8bit=True)
                except Exception as quant_err:
                    self.logger.warning(
                        "8-bit requested but quantization setup is unavailable; "
                        "falling back to standard precision: %s",
                        quant_err
                    )
                    is_8bit_mode = False

            if is_8bit_mode:
                # 8-bit quantization for memory efficiency
                try:
                    self.model, loading_info = model_cls.from_pretrained(
                        model_name,
                        output_loading_info=True,
                        device_map='auto',
                        **quantization_kwargs,
                        **safe_tensor_kwargs
                    )
                except Exception as load_err:
                    self.logger.warning(
                        "8-bit loading failed for %s; falling back to standard precision: %s",
                        model_name,
                        load_err
                    )
                    is_8bit_mode = False

                if is_8bit_mode:
                    self.logger.info("Model loaded with 8-bit quantization")

            if not is_8bit_mode:
                # Standard loading
                if is_cuda_device and torch.cuda.is_available():
                    model_kwargs = {'device_map': 'auto', **safe_tensor_kwargs}
                    if selected_torch_dtype is not None:
                        model_kwargs['torch_dtype'] = selected_torch_dtype
                    try:
                        self.model, loading_info = model_cls.from_pretrained(
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
                        self.model, loading_info = model_cls.from_pretrained(
                            model_name,
                            output_loading_info=True,
                            **model_kwargs
                        )
                else:
                    cpu_kwargs = {**safe_tensor_kwargs}
                    if selected_torch_dtype is not None and not is_cuda_device:
                        cpu_kwargs['torch_dtype'] = selected_torch_dtype
                    try:
                        cpu_model, loading_info = model_cls.from_pretrained(
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
                        cpu_model, loading_info = model_cls.from_pretrained(
                            model_name,
                            output_loading_info=True,
                            **cpu_kwargs
                        )
                    self.model = cpu_model.to(device)

                self.logger.info(f"Model loaded successfully on {self.model.device}")

            repair_applied = False
            tie_applied = False
            if self.model_family == 'seq2seq':
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

    def _normalize_dtype_mode(self, dtype: Union[str, torch.dtype, None]) -> str:
        """Normalize precision mode to one of: bf16, fp16, fp32, 8bit."""
        if dtype is None:
            return 'bf16'

        if isinstance(dtype, torch.dtype):
            mapping = {
                torch.bfloat16: 'bf16',
                torch.float16: 'fp16',
                torch.float32: 'fp32',
            }
            if dtype in mapping:
                return mapping[dtype]
            raise ValueError(f"Unsupported torch dtype: {dtype}")

        mode = str(dtype).strip().lower()
        aliases = {
            'bf16': 'bf16',
            'bfloat16': 'bf16',
            'fp16': 'fp16',
            'float16': 'fp16',
            'fp32': 'fp32',
            'float32': 'fp32',
            '8bit': '8bit',
            'int8': '8bit',
        }
        if mode not in aliases:
            raise ValueError(
                f"Unsupported dtype mode '{dtype}'. Expected one of: bf16, fp16, fp32, 8bit"
            )
        return aliases[mode]

    def _resolve_torch_dtype(self, mode: str) -> Optional[torch.dtype]:
        """Resolve torch dtype for a normalized precision mode."""
        if mode == 'bf16':
            return torch.bfloat16
        if mode == 'fp16':
            return torch.float16
        if mode == 'fp32':
            return torch.float32
        if mode == '8bit':
            return None
        raise ValueError(f"Unknown dtype mode: {mode}")

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
        
        Creates a structured prompt that optionally incorporates the tokenizer's
        chat formatting via `apply_chat_template` if supported by the model,
        or gracefully degrades to raw concatenation for non-chat models.
        """
        system_instruction = (
            "You are a factual assistant. If context is provided, use the provided passages to "
            "answer the question. Answer directly and concisely in plain prose. "
            "Do not include meta commentary, self-evaluation, note sections, or statements "
            "about passage selection. Respond in English only."
        )

        user_content = ""
        if not evidence_chunks:
            user_content = f"Question: {prompt}"
        else:
            evidence_texts = [f"Passage {i}: {chunk.text}" for i, chunk in enumerate(evidence_chunks, 1)]
            evidence_context = "\n\n".join(evidence_texts)
            user_content = f"Context:\n{evidence_context}\n\nQuestion: {prompt}"

        # 1. Structure the message
        messages = [
            {"role": "system", "content": system_instruction},
            {"role": "user", "content": user_content}
        ]

        # 2. Try the built-in tokenizer apply_chat_template natively
        if hasattr(self.tokenizer, "apply_chat_template"):
            # Ensure the tokenizer has a template initialized
            if getattr(self.tokenizer, "chat_template", None) is not None or getattr(self.tokenizer, "default_chat_template", None) is not None:
                try:
                    formatted_prompt = self.tokenizer.apply_chat_template(
                        messages,
                        tokenize=False,
                        add_generation_prompt=True,
                        enable_thinking=self.enable_thinking,
                    )
                    return formatted_prompt
                except Exception as e:
                    self.logger.warning(f"apply_chat_template failed: {e}. Falling back to manual formatting.")

        # 3. Fallbacks 
        if self.model_family == 'causal':
            return f"{system_instruction}\n\n{user_content}\n\nAnswer:"
        
        # Seq2Seq fallback (e.g. FLAN-T5 ignores system roles traditionally)
        return f"{user_content}\n\nAnswer:"

    def _sanitize_generated_text(self, text: str) -> str:
        """Remove common meta-commentary tails from generated responses."""
        if not text:
            return text

        cleaned = text.strip()
        # Remove trailing explicit note sections.
        cleaned = re.sub(r"\n\s*Note:\s.*$", "", cleaned, flags=re.IGNORECASE | re.DOTALL)

        lines = cleaned.splitlines()
        if not lines:
            return cleaned

        meta_starts = (
            "you are correct",
            "to answer your question more directly",
            "i've included elements",
            "however, the passage numbers",
        )

        kept = []
        for line in lines:
            stripped = line.strip()
            lowered = stripped.lower()
            if stripped and lowered.startswith(meta_starts):
                break
            kept.append(line)

        cleaned = "\n".join(kept).strip()
        return cleaned if cleaned else text.strip()
    
    def generate_with_metadata(
        self,
        prompt: str,
        evidence_chunks: Optional[List[EvidenceChunk]] = None,
        max_new_tokens: int = 256,
        temperature: float = 1.0,
        top_p: float = 1.0,
        num_beams: int = 1,
        do_sample: bool = False,
        repetition_penalty: Optional[float] = None,
        no_repeat_ngram_size: Optional[int] = None,
        sanitize_meta_text: bool = False,
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
        
        generate_kwargs = {
            'max_new_tokens': max_new_tokens,
            'temperature': temperature,
            'top_p': top_p,
            'num_beams': num_beams,
            'do_sample': do_sample,
            'output_scores': True,
            'return_dict_in_generate': True,
            'pad_token_id': self.tokenizer.pad_token_id,
        }

        if repetition_penalty is not None and repetition_penalty > 0:
            generate_kwargs['repetition_penalty'] = repetition_penalty
        if no_repeat_ngram_size is not None and no_repeat_ngram_size > 0:
            generate_kwargs['no_repeat_ngram_size'] = int(no_repeat_ngram_size)

        with torch.no_grad():
            outputs = self.model.generate(
                **inputs,
                **generate_kwargs,
            )

        # Detect severe decoding collapse early (e.g., same token repeated).
        if self.model_family == 'seq2seq':
            generated_ids = outputs.sequences[0]
        else:
            prompt_len = int(inputs['input_ids'].shape[1])
            generated_ids = outputs.sequences[0][prompt_len:]

        if generated_ids is not None and len(generated_ids) > 8:
            gen_ids = generated_ids.tolist()
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
        
        # Decode generated text
        generated_text = self.tokenizer.decode(
            generated_ids,
            skip_special_tokens=True
        )
        if sanitize_meta_text:
            generated_text = self._sanitize_generated_text(generated_text)
        
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
                'repetition_penalty': repetition_penalty,
                'no_repeat_ngram_size': no_repeat_ngram_size,
                'sanitize_meta_text': sanitize_meta_text,
                'model_name': self.model_name,
                'model_family': self.model_family
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

    def generate_n_samples(
        self,
        prompt: str,
        evidence_chunks: Optional[List[EvidenceChunk]] = None,
        num_samples: int = 1,
        max_new_tokens: int = 256,
        temperature: float = 1.0,
        top_p: float = 1.0,
        do_sample: bool = True,
        repetition_penalty: Optional[float] = None,
        no_repeat_ngram_size: Optional[int] = None,
        sanitize_meta_text: bool = False,
    ) -> List[str]:
        """
        Generate multiple samples for a single prompt in one model call.

        This is optimized for self-agreement style sampling, where the same
        prompt/evidence pair is decoded multiple times stochastically.
        """
        if num_samples <= 0:
            raise ValueError("num_samples must be > 0")

        if evidence_chunks is None:
            evidence_chunks = []

        formatted_prompt = self._format_prompt(prompt, evidence_chunks)
        max_input_tokens = self._resolve_max_input_tokens()

        inputs = self.tokenizer(
            formatted_prompt,
            return_tensors='pt',
            truncation=True,
            max_length=max_input_tokens
        ).to(self.model.device)

        generate_kwargs = {
            'max_new_tokens': max_new_tokens,
            'temperature': temperature,
            'top_p': top_p,
            'do_sample': do_sample,
            'num_return_sequences': int(num_samples),
            'pad_token_id': self.tokenizer.pad_token_id,
        }

        if repetition_penalty is not None and repetition_penalty > 0:
            generate_kwargs['repetition_penalty'] = repetition_penalty
        if no_repeat_ngram_size is not None and no_repeat_ngram_size > 0:
            generate_kwargs['no_repeat_ngram_size'] = int(no_repeat_ngram_size)

        with torch.no_grad():
            outputs = self.model.generate(
                **inputs,
                **generate_kwargs,
            )

        if isinstance(outputs, torch.Tensor):
            sequences = outputs
        else:
            sequences = outputs.sequences

        if self.model_family == 'seq2seq':
            generated_sequences = sequences
        else:
            prompt_len = int(inputs['input_ids'].shape[1])
            generated_sequences = sequences[:, prompt_len:]

        samples = []
        for seq in generated_sequences:
            text = self.tokenizer.decode(seq, skip_special_tokens=True)
            if sanitize_meta_text:
                text = self._sanitize_generated_text(text)
            samples.append(text)

        self.logger.debug(
            "Generated %d samples in one call (model=%s)",
            len(samples),
            self.model_name
        )
        return samples

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

        if self.model_family == 'seq2seq':
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
            target_position_map = list(range(label_ids.shape[0]))
        else:
            max_input_tokens = self._resolve_max_input_tokens()
            prompt_ids = self.tokenizer(
                formatted_prompt,
                add_special_tokens=True,
                truncation=True,
                max_length=max_input_tokens,
            )['input_ids']
            target_ids = self.tokenizer(
                target_text,
                add_special_tokens=False,
            )['input_ids']

            if not target_ids:
                return {
                    'text': target_text,
                    'prompt_text': formatted_prompt,
                    'tokens': [],
                    'token_ids': [],
                    'token_entropies': [],
                    'scores': [],
                    'token_logprobs': [],
                    'evidence_used': [chunk.doc_id for chunk in evidence_chunks],
                    'generation_config': {
                        'mode': 'teacher_forcing',
                        'model_name': self.model_name,
                        'model_family': self.model_family,
                    }
                }

            total_len = len(prompt_ids) + len(target_ids)
            if total_len > max_input_tokens:
                keep_prompt_len = max(1, max_input_tokens - len(target_ids))
                prompt_ids = prompt_ids[-keep_prompt_len:]

            input_ids = prompt_ids + target_ids
            input_tensor = torch.tensor([input_ids], dtype=torch.long, device=self.model.device)
            attention_mask = torch.ones_like(input_tensor, device=self.model.device)

            with torch.no_grad():
                outputs = self.model(
                    input_ids=input_tensor,
                    attention_mask=attention_mask,
                )

            logits = outputs.logits[0]
            label_ids = torch.tensor(target_ids, dtype=torch.long, device=self.model.device)
            # Causal LM predicts token at position t from logits[t-1].
            target_start = len(prompt_ids)
            target_position_map = [target_start + i - 1 for i in range(len(target_ids))]

        token_ids = []
        tokens = []
        token_entropies = []
        token_probs = []
        token_logprobs = []

        for idx in range(label_ids.shape[0]):
            token_id = int(label_ids[idx].item())
            if token_id < 0:
                continue

            logits_idx = target_position_map[idx]
            if logits_idx < 0 or logits_idx >= logits.shape[0]:
                continue

            step_logits = logits[logits_idx]
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
                'model_name': self.model_name,
                'model_family': self.model_family
            }
        }
