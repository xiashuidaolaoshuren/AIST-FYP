"""
Centralized logging setup for the Month 2 Baseline RAG Module.

This module provides a setup_logger function that configures logging with
both file and console handlers, suitable for long-running operations.
"""

import json
import logging
import os
import sys
from datetime import datetime
from pathlib import Path
from typing import Optional


_STANDARD_LOG_RECORD_ATTRS = {
    'name', 'msg', 'args', 'levelname', 'levelno', 'pathname', 'filename',
    'module', 'exc_info', 'exc_text', 'stack_info', 'lineno', 'funcName',
    'created', 'msecs', 'relativeCreated', 'thread', 'threadName', 'processName',
    'process', 'message'
}


class JsonFormatter(logging.Formatter):
    """Format log records as JSON lines for structured logging."""

    def format(self, record: logging.LogRecord) -> str:
        base = {
            'timestamp': datetime.fromtimestamp(record.created).isoformat(),
            'level': record.levelname,
            'logger': record.name,
            'message': record.getMessage()
        }

        if record.exc_info:
            base['exc_info'] = self.formatException(record.exc_info)

        extras = {
            k: v for k, v in record.__dict__.items()
            if k not in _STANDARD_LOG_RECORD_ATTRS
        }
        if extras:
            base['extra'] = extras

        return json.dumps(base, ensure_ascii=False, default=str)


def setup_logger(
    name: str,
    log_file: str = 'logs/month2.log',
    level: int = logging.INFO,
    console_level: int = logging.ERROR,
    json_log_file: Optional[str] = 'logs/full_pipeline_events.jsonl',
    json_level: int = logging.INFO
) -> logging.Logger:
    """
    Set up and configure a logger with file and console handlers.
    
    Creates a logger that outputs INFO-level messages to a file and
    ERROR-level messages to the console. Also supports structured JSON
    logging to a .jsonl file for module-level tracing. The log file
    directory is created automatically if it doesn't exist.
    
    Args:
        name: Name for the logger (typically __name__ of the calling module)
        log_file: Path to the log file (default: 'logs/month2.log')
        level: Logging level for the file handler (default: INFO)
        console_level: Logging level for the console handler (default: ERROR)
        json_log_file: Path to the JSONL log file (default: logs/full_pipeline_events.jsonl)
        json_level: Logging level for JSONL handler (default: INFO)
    
    Returns:
        Configured logger instance
    
    Example:
        >>> logger = setup_logger(__name__)
        >>> logger.info("Processing started")
        >>> logger.error("An error occurred")
    """
    # Support env override so notebook launchers can surface logs without code changes.
    env_console_level = os.getenv('AIST_STDOUT_LOG_LEVEL', '').strip().upper()
    if env_console_level:
        level_map = {
            'CRITICAL': logging.CRITICAL,
            'ERROR': logging.ERROR,
            'WARNING': logging.WARNING,
            'INFO': logging.INFO,
            'DEBUG': logging.DEBUG,
        }
        console_level = level_map.get(env_console_level, console_level)

    # Create logger
    logger = logging.getLogger(name)
    logger.setLevel(logging.DEBUG)  # Capture all levels, handlers will filter
    
    def has_json_handler() -> bool:
        return any(getattr(h, '_is_json_handler', False) for h in logger.handlers)

    def has_file_handler(file_path: Path) -> bool:
        for handler in logger.handlers:
            if isinstance(handler, logging.FileHandler):
                try:
                    if Path(handler.baseFilename).resolve() == file_path.resolve():
                        return True
                except Exception:
                    continue
        return False

    def has_console_handler() -> bool:
        return any(
            isinstance(h, logging.StreamHandler) and not isinstance(h, logging.FileHandler)
            for h in logger.handlers
        )

    # Create logs directory if it doesn't exist
    log_path = Path(log_file)
    log_path.parent.mkdir(parents=True, exist_ok=True)
    
    # Create formatters
    formatter = logging.Formatter(
        '%(asctime)s - %(name)s - %(levelname)s - %(message)s',
        datefmt='%Y-%m-%d %H:%M:%S'
    )
    
    # File handler (INFO level)
    if not has_file_handler(log_path):
        file_handler = logging.FileHandler(log_file, encoding='utf-8')
        file_handler.setLevel(level)
        file_handler.setFormatter(formatter)
        logger.addHandler(file_handler)

    # Console handler (defaults to stderr in stdlib; force stdout for Colab visibility)
    if not has_console_handler():
        console_handler = logging.StreamHandler(stream=sys.stdout)
        console_handler.setLevel(console_level)
        console_handler.setFormatter(formatter)
        logger.addHandler(console_handler)

    # JSONL handler (INFO level)
    if json_log_file and not has_json_handler():
        json_path = Path(json_log_file)
        json_path.parent.mkdir(parents=True, exist_ok=True)
        json_handler = logging.FileHandler(json_log_file, encoding='utf-8')
        json_handler.setLevel(json_level)
        json_handler.setFormatter(JsonFormatter())
        json_handler._is_json_handler = True
        logger.addHandler(json_handler)
    
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
