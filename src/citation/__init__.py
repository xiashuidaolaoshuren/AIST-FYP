"""
Citation module for CiteEval-compatible citation formatting.

This module provides functionality to inject bracketed citations into answer text
and export to CiteEval benchmark format.
"""

from .citation_formatter import CitationFormatter

__all__ = ['CitationFormatter']
