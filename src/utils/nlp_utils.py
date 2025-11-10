"""
Shared NLP utilities for the AIST-FYP project.

This module provides centralized loading and caching of NLP models (e.g., spaCy)
to prevent redundant model loading across different modules. Uses a singleton
pattern to ensure only one instance of each model is loaded in memory.

"""

import spacy
from typing import Optional

from src.utils.logger import setup_logger

# Global variable for cached spaCy model
_nlp_model = None

logger = setup_logger(__name__)


def get_spacy_model(model_name: str = 'en_core_web_sm'):
    """
    Get or load the spaCy model with singleton pattern.
    
    Loads the spaCy model only once and caches it globally. Subsequent calls
    return the cached instance, preventing redundant model loading.
    
    This function is thread-safe for reading the cached model, but concurrent
    first-time loads should be avoided (call this during initialization).
    
    Args:
        model_name: Name of the spaCy model to load (default: 'en_core_web_sm')
    
    Returns:
        Loaded spaCy Language model instance
    
    Raises:
        OSError: If the specified spaCy model is not found
    
    Example:
        >>> nlp = get_spacy_model()
        >>> doc = nlp("This is a test sentence.")
        >>> sentences = list(doc.sents)
    
    Note:
        If the model is not installed, install it with:
        python -m spacy download en_core_web_sm
    """
    global _nlp_model
    
    if _nlp_model is None:
        try:
            logger.info(f"Loading spaCy model: {model_name}")
            _nlp_model = spacy.load(model_name)
            logger.info(f"spaCy model '{model_name}' loaded successfully")
        except OSError as e:
            logger.error(
                f"Failed to load spaCy model '{model_name}'. "
                f"Install with: python -m spacy download {model_name}"
            )
            raise OSError(
                f"spaCy model '{model_name}' not found. "
                f"Install with: python -m spacy download {model_name}"
            ) from e
    
    return _nlp_model


def reset_spacy_model():
    """
    Reset the cached spaCy model (mainly for testing purposes).
    
    Forces the next call to get_spacy_model() to reload the model.
    Use with caution in production code.
    """
    global _nlp_model
    _nlp_model = None
    logger.debug("spaCy model cache reset")
