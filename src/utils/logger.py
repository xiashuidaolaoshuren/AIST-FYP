"""
Centralized logging setup for the Month 2 Baseline RAG Module.

This module provides a setup_logger function that configures logging with
both file and console handlers, suitable for long-running operations.
"""

import logging
from pathlib import Path
from typing import Optional


def setup_logger(
    name: str,
    log_file: str = 'logs/month2.log',
    level: int = logging.INFO,
    console_level: int = logging.ERROR
) -> logging.Logger:
    """
    Set up and configure a logger with file and console handlers.
    
    Creates a logger that outputs INFO-level messages to a file and
    ERROR-level messages to the console. The log file directory is
    created automatically if it doesn't exist.
    
    Args:
        name: Name for the logger (typically __name__ of the calling module)
        log_file: Path to the log file (default: 'logs/month2.log')
        level: Logging level for the file handler (default: INFO)
        console_level: Logging level for the console handler (default: ERROR)
    
    Returns:
        Configured logger instance
    
    Example:
        >>> logger = setup_logger(__name__)
        >>> logger.info("Processing started")
        >>> logger.error("An error occurred")
    """
    # Create logger
    logger = logging.getLogger(name)
    logger.setLevel(logging.DEBUG)  # Capture all levels, handlers will filter
    
    # Avoid adding duplicate handlers if logger already configured
    if logger.handlers:
        return logger
    
    # Create logs directory if it doesn't exist
    log_path = Path(log_file)
    log_path.parent.mkdir(parents=True, exist_ok=True)
    
    # Create formatters
    formatter = logging.Formatter(
        '%(asctime)s - %(name)s - %(levelname)s - %(message)s',
        datefmt='%Y-%m-%d %H:%M:%S'
    )
    
    # File handler (INFO level)
    file_handler = logging.FileHandler(log_file, encoding='utf-8')
    file_handler.setLevel(level)
    file_handler.setFormatter(formatter)
    logger.addHandler(file_handler)
    
    # Console handler (ERROR level)
    console_handler = logging.StreamHandler()
    console_handler.setLevel(console_level)
    console_handler.setFormatter(formatter)
    logger.addHandler(console_handler)
    
    return logger


def get_logger(name: str) -> logging.Logger:
    """
    Get an existing logger by name.
    
    This is a convenience function to retrieve a logger that was
    previously configured with setup_logger.
    
    Args:
        name: Name of the logger to retrieve
    
    Returns:
        Logger instance
    
    Example:
        >>> logger = get_logger(__name__)
    """
    return logging.getLogger(name)


def set_log_level(logger: logging.Logger, level: int) -> None:
    """
    Change the logging level of an existing logger.
    
    Args:
        logger: Logger instance to modify
        level: New logging level (e.g., logging.DEBUG, logging.INFO)
    
    Example:
        >>> logger = setup_logger(__name__)
        >>> set_log_level(logger, logging.DEBUG)
    """
    logger.setLevel(level)
    for handler in logger.handlers:
        if isinstance(handler, logging.FileHandler):
            handler.setLevel(level)
