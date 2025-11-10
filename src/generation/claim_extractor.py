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


def extract_claims_spacy(
    text: str,
    answer_id: Optional[str] = None
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
        
        # Create Claim object
        claim = Claim(
            claim_id=str(uuid.uuid4()),
            answer_id=answer_id,
            text=sent_text,
            answer_char_span=[char_start, char_end],
            extraction_method='spacy_sent_v1'
        )
        
        claims.append(claim)
    
    logger.info(f"Extracted {len(claims)} claims using spaCy from {len(text)} chars")
    
    return claims


def extract_claims_regex(
    text: str,
    answer_id: Optional[str] = None
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
    
    # Generate answer_id if not provided
    if answer_id is None:
        answer_id = str(uuid.uuid4())
    
    # Simple sentence boundary regex
    # Matches: . ! ? followed by space and uppercase, or end of string
    sentence_pattern = r'([^.!?]+[.!?]+(?:\s+|$))'
    
    # Find all sentences with their positions
    claims = []
    current_pos = 0
    
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
    method: str = 'auto'
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
        return extract_claims_spacy(text, answer_id)
    elif method == 'regex':
        return extract_claims_regex(text, answer_id)
    elif method == 'auto':
        # Try spaCy first, fall back to regex
        try:
            return extract_claims_spacy(text, answer_id)
        except Exception as e:
            logger.warning(f"spaCy extraction failed: {e}, using regex fallback")
            return extract_claims_regex(text, answer_id)
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
