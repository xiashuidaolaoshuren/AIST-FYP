"""
Claim extraction from generated text.

This module provides functions to extract atomic claims from generated answers.
Uses simple sentence splitting for baseline implementation, with room for
improvement in Month 5 with more sophisticated NLP techniques.
"""

import re
import uuid
from typing import List, Optional, Tuple

from src.utils.data_structures import Claim
from src.utils.logger import setup_logger
from src.utils.nlp_utils import get_spacy_model


CLAUSE_SPLIT_PATTERN = re.compile(
    r'\s*[,;]?\s*(?:however|but|while|although),?\s*',
    re.IGNORECASE,
)


def _has_clause_verb(nlp, text: str) -> bool:
    """Heuristic: fragment is claim-like only if it contains a verb/aux."""
    if not text or not text.strip():
        return False
    doc = nlp(text)
    return any(tok.pos_ in {'VERB', 'AUX'} for tok in doc)


def _split_claim_text_with_spans(nlp, claim_text: str, claim_start: int) -> List[Tuple[str, int, int]]:
    """Split compound claim text into clause-level segments with absolute spans."""
    matches = list(CLAUSE_SPLIT_PATTERN.finditer(claim_text))
    if not matches:
        return [(claim_text, claim_start, claim_start + len(claim_text))]

    parts: List[Tuple[str, int, int]] = []
    cursor = 0
    for match in matches:
        raw = claim_text[cursor:match.start()]
        raw_start = claim_start + cursor
        raw_end = claim_start + match.start()
        trimmed = raw.strip()
        if trimmed:
            left_ws = len(raw) - len(raw.lstrip())
            right_ws = len(raw) - len(raw.rstrip())
            part_start = raw_start + left_ws
            part_end = raw_end - right_ws
            parts.append((trimmed, part_start, part_end))
        cursor = match.end()

    tail_raw = claim_text[cursor:]
    tail_start = claim_start + cursor
    tail_end = claim_start + len(claim_text)
    tail_trimmed = tail_raw.strip()
    if tail_trimmed:
        left_ws = len(tail_raw) - len(tail_raw.lstrip())
        right_ws = len(tail_raw) - len(tail_raw.rstrip())
        part_start = tail_start + left_ws
        part_end = tail_end - right_ws
        parts.append((tail_trimmed, part_start, part_end))

    if len(parts) <= 1:
        return [(claim_text, claim_start, claim_start + len(claim_text))]

    # Keep only reasonably atomic, verbal clauses; otherwise keep original sentence.
    valid_parts = []
    for text, start, end in parts:
        if len(text.split()) < 5:
            continue
        if not _has_clause_verb(nlp, text):
            continue
        valid_parts.append((text, start, end))

    if len(valid_parts) < 2:
        return [(claim_text, claim_start, claim_start + len(claim_text))]

    return valid_parts


def extract_claims_spacy(
    text: str,
    answer_id: Optional[str] = None,
    task_type: Optional[str] = None,
) -> List[Claim]:
    """
    Extract claims using spaCy sentence segmentation.
    
    Uses spaCy's sentence boundary detection to split the text into
    atomic claims. Each sentence becomes one claim with proper char spans.
    
    Args:
        text: Generated answer text to extract claims from
        answer_id: Optional answer ID to associate with claims
    
    Returns:
        List of Claim objects, one per sentence
    
    Example:
        >>> text = "AI is intelligence by machines. It includes ML and NLP."
        >>> claims = extract_claims_spacy(text)
        >>> len(claims)
        2
        >>> claims[0].text
        'AI is intelligence by machines.'
    """
    logger = setup_logger(__name__)
    
    if not text or not text.strip():
        logger.warning("Empty text provided, returning empty claim list")
        return []
    
    # Generate answer_id if not provided
    if answer_id is None:
        answer_id = str(uuid.uuid4())
    
    # Load spaCy model
    try:
        nlp = get_spacy_model()
    except OSError as e:
        logger.error(f"Failed to load spaCy model: {e}")
        logger.info("Falling back to regex-based sentence splitting")
        return extract_claims_regex(text, answer_id)
    
    # Process text with spaCy
    doc = nlp(text)
    
    apply_clause_split = (task_type or '').strip().lower() == 'summary'

    # Extract sentences
    claims = []
    for sent in doc.sents:
        # Get sentence text and character span
        sent_text = sent.text.strip()
        
        if not sent_text:
            continue
        
        # Calculate character span in original text
        char_start = sent.start_char
        char_end = sent.end_char
        
        split_parts = _split_claim_text_with_spans(nlp, sent_text, char_start) if apply_clause_split else [
            (sent_text, char_start, char_end)
        ]
        extraction_method = 'spacy_clause_split_v1' if apply_clause_split else 'spacy_sent_v1'
        for part_text, part_start, part_end in split_parts:
            claim = Claim(
                claim_id=str(uuid.uuid4()),
                answer_id=answer_id,
                text=part_text,
                answer_char_span=[part_start, part_end],
                extraction_method=extraction_method
            )
            claims.append(claim)
    
    # Fallback: If no claims extracted but text is not empty, treat whole text as one claim
    if not claims and text.strip():
        logger.info("No sentences detected by spaCy, falling back to whole text as claim")
        claim = Claim(
            claim_id=str(uuid.uuid4()),
            answer_id=answer_id,
            text=text.strip(),
            answer_char_span=[0, len(text)],
            extraction_method='spacy_fallback'
        )
        claims.append(claim)
    
    logger.info(f"Extracted {len(claims)} claims using spaCy from {len(text)} chars")
    
    return claims


def extract_claims_regex(
    text: str,
    answer_id: Optional[str] = None,
    task_type: Optional[str] = None,
) -> List[Claim]:
    """
    Extract claims using regex-based sentence splitting.
    
    Fallback method using simple regex patterns to detect sentence
    boundaries. Less accurate than spaCy but doesn't require model download.
    
    Args:
        text: Generated answer text to extract claims from
        answer_id: Optional answer ID to associate with claims
    
    Returns:
        List of Claim objects, one per detected sentence
    
    Example:
        >>> text = "AI is growing. It includes many subfields."
        >>> claims = extract_claims_regex(text)
        >>> len(claims)
        2
    """
    logger = setup_logger(__name__)
    
    if not text or not text.strip():
        return []

    _ = task_type
    
    # Generate answer_id if not provided
    if answer_id is None:
        answer_id = str(uuid.uuid4())
    
    # Simple sentence boundary regex
    # Matches: . ! ? followed by space and uppercase, or end of string
    sentence_pattern = r'([^.!?]+[.!?]+(?:\s+|$))'
    
    # Find all sentences with their positions
    claims = []
    for match in re.finditer(sentence_pattern, text):
        sent_text = match.group(1).strip()
        
        if not sent_text:
            continue
        
        # Get character span
        char_start = match.start()
        char_end = match.end()
        
        # Create Claim object
        claim = Claim(
            claim_id=str(uuid.uuid4()),
            answer_id=answer_id,
            text=sent_text,
            answer_char_span=[char_start, char_end],
            extraction_method='rule_sentence_split_v1'
        )
        
        claims.append(claim)
    
    # Handle remaining text if no sentence boundary at end
    if claims and claims[-1].answer_char_span[1] < len(text):
        remaining_text = text[claims[-1].answer_char_span[1]:].strip()
        if remaining_text:
            claim = Claim(
                claim_id=str(uuid.uuid4()),
                answer_id=answer_id,
                text=remaining_text,
                answer_char_span=[claims[-1].answer_char_span[1], len(text)],
                extraction_method='rule_sentence_split_v1'
            )
            claims.append(claim)
    elif not claims and text.strip():
        # No sentence boundaries found, treat entire text as one claim
        claim = Claim(
            claim_id=str(uuid.uuid4()),
            answer_id=answer_id,
            text=text.strip(),
            answer_char_span=[0, len(text)],
            extraction_method='rule_sentence_split_v1'
        )
        claims.append(claim)
    
    logger.info(f"Extracted {len(claims)} claims using regex from {len(text)} chars")
    
    return claims


def extract_claims(
    text: str,
    answer_id: Optional[str] = None,
    method: str = 'auto',
    task_type: Optional[str] = None,
) -> List[Claim]:
    """
    Extract atomic claims from generated text.
    
    Main entry point for claim extraction. Supports multiple methods:
    - 'spacy': Use spaCy sentence segmentation (most accurate)
    - 'regex': Use regex-based splitting (fallback)
    - 'auto': Try spaCy, fall back to regex if unavailable
    
    Args:
        text: Generated answer text to extract claims from
        answer_id: Optional answer ID to associate with claims
        method: Extraction method ('spacy', 'regex', or 'auto')
    
    Returns:
        List of Claim objects with unique IDs and char spans
    
    Raises:
        ValueError: If method is not recognized
    
    Example:
        >>> text = "Machine learning is a subset of AI. It uses data to learn patterns."
        >>> claims = extract_claims(text)
        >>> for claim in claims:
        ...     print(f"{claim.claim_id[:8]}... {claim.text}")
    """
    logger = setup_logger(__name__)
    
    if method == 'spacy':
        return extract_claims_spacy(text, answer_id, task_type=task_type)
    elif method == 'regex':
        return extract_claims_regex(text, answer_id, task_type=task_type)
    elif method == 'auto':
        # Try spaCy first, fall back to regex
        try:
            return extract_claims_spacy(text, answer_id, task_type=task_type)
        except Exception as e:
            logger.warning(f"spaCy extraction failed: {e}, using regex fallback")
            return extract_claims_regex(text, answer_id, task_type=task_type)
    else:
        raise ValueError(
            f"Unknown extraction method: {method}. "
            f"Use 'spacy', 'regex', or 'auto'"
        )


def validate_claim_spans(claims: List[Claim], original_text: str) -> bool:
    """
    Validate that claim char spans match the original text.
    
    Checks that each claim's char span correctly extracts its text
    from the original answer text.
    
    Args:
        claims: List of claims to validate
        original_text: Original answer text
    
    Returns:
        True if all claims valid, False otherwise
    """
    logger = setup_logger(__name__)
    
    for i, claim in enumerate(claims):
        start, end = claim.answer_char_span
        extracted_text = original_text[start:end].strip()
        claim_text = claim.text.strip()
        
        if extracted_text != claim_text:
            logger.error(
                f"Claim {i} span mismatch:\n"
                f"  Expected: '{claim_text}'\n"
                f"  Extracted: '{extracted_text}'\n"
                f"  Span: [{start}, {end}]"
            )
            return False
    
    logger.debug(f"All {len(claims)} claim spans validated successfully")
    return True
