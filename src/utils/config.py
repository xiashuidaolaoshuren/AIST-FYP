"""
Configuration loader with validation for the Month 2 Baseline RAG Module.

This module provides a Config class that loads and validates the config.yaml file,
ensuring all required fields are present and the GPU configuration is correct.
"""

import yaml
from pathlib import Path
from typing import Any, Dict, Optional
import sys


class Config:
    """
    Configuration loader and validator.
    
    Loads YAML configuration files and provides dot-notation access to
    nested configuration values. Validates required fields and GPU availability.
    
    Attributes:
        config_path: Path to the configuration YAML file
        _config: Internal dictionary storing the configuration
    
    Example:
        >>> config = Config("config.yaml")
        >>> print(config.models.sentence_transformer)
        >>> print(config["processing"]["batch_size"])
    """
    
    def __init__(self, config_path: str = "config.yaml"):
        """
        Initialize the Config object and load the configuration file.
        
        Args:
            config_path: Path to the YAML configuration file (default: "config.yaml")
        
        Raises:
            FileNotFoundError: If the configuration file doesn't exist
            ValueError: If validation fails
        """
        self.config_path = Path(config_path)
        
        if not self.config_path.exists():
            raise FileNotFoundError(
                f"Configuration file not found: {self.config_path.absolute()}"
            )
        
        # Load YAML configuration
        with open(self.config_path, 'r', encoding='utf-8') as f:
            self._config: Dict[str, Any] = yaml.safe_load(f)
        
        # Validate configuration
        self._validate()
    
    def _validate(self) -> None:
        """
        Validate the configuration file.
        
        Checks for required fields and validates GPU availability if device='cuda'.
        
        Raises:
            ValueError: If required fields are missing or GPU validation fails
        """
        # Define required top-level fields
        required_fields = ['models', 'data_strategy', 'processing', 'retrieval']
        
        missing_fields = [field for field in required_fields if field not in self._config]
        if missing_fields:
            raise ValueError(
                f"Missing required configuration fields: {', '.join(missing_fields)}"
            )
        
        # Validate models section
        if 'sentence_transformer' not in self._config.get('models', {}):
            raise ValueError("Missing 'sentence_transformer' in 'models' configuration")
        
        if 'generator' not in self._config.get('models', {}):
            raise ValueError("Missing 'generator' in 'models' configuration")
        
        # Validate processing section
        processing = self._config.get('processing', {})
        if 'device' not in processing:
            raise ValueError("Missing 'device' in 'processing' configuration")
        
        if 'batch_size' not in processing:
            raise ValueError("Missing 'batch_size' in 'processing' configuration")
        
        # Validate GPU availability if device is set to 'cuda'
        if processing['device'] == 'cuda':
            self._check_gpu()
    
    def _check_gpu(self) -> None:
        """
        Check if CUDA GPU is available when device='cuda'.
        
        Raises:
            RuntimeError: If CUDA is not available but device='cuda'
        """
        try:
            # Direct torch check (simpler and doesn't have output issues)
            import torch
            if not torch.cuda.is_available():
                raise RuntimeError(
                    "CUDA is not available, but device='cuda' in configuration. "
                    "Please set device='cpu' in config.yaml or ensure CUDA is properly installed."
                )
        except ImportError:
            raise RuntimeError(
                "Cannot verify CUDA availability (torch not installed). "
                "Please ensure PyTorch with CUDA support is installed."
            )
        except RuntimeError:
            # Re-raise RuntimeError as-is
            raise
        except Exception as e:
            # Re-raise other exceptions as RuntimeError with context
            raise RuntimeError(f"GPU validation failed: {str(e)}")
    
    def get(self, key: str, default: Any = None) -> Any:
        """
        Get a configuration value using dot notation or dict access.
        
        Args:
            key: Configuration key (supports nested keys with dots, e.g., "models.sentence_transformer")
            default: Default value if key doesn't exist
        
        Returns:
            Configuration value or default
        
        Example:
            >>> config.get("models.sentence_transformer")
            >>> config.get("nonexistent.key", "default_value")
        """
        keys = key.split('.')
        value = self._config
        
        for k in keys:
            if isinstance(value, dict) and k in value:
                value = value[k]
            else:
                return default
        
        return value
    
    def __getitem__(self, key: str) -> Any:
        """
        Dict-style access to configuration values.
        
        Args:
            key: Configuration key
        
        Returns:
            Configuration value
        
        Raises:
            KeyError: If key doesn't exist
        """
        if key not in self._config:
            raise KeyError(f"Configuration key not found: {key}")
        return self._config[key]
    
    def __getattr__(self, name: str) -> Any:
        """
        Dot-notation access to configuration values.
        
        Args:
            name: Configuration key
        
        Returns:
            Configuration value or ConfigSection for nested dicts
        
        Raises:
            AttributeError: If key doesn't exist
        """
        if name.startswith('_'):
            # Avoid infinite recursion for private attributes
            raise AttributeError(f"'{self.__class__.__name__}' object has no attribute '{name}'")
        
        if name in self._config:
            value = self._config[name]
            if isinstance(value, dict):
                return ConfigSection(value)
            return value
        
        raise AttributeError(f"Configuration key not found: {name}")
    
    def to_dict(self) -> Dict[str, Any]:
        """
        Get the entire configuration as a dictionary.
        
        Returns:
            Complete configuration dictionary
        """
        return self._config.copy()


class ConfigSection:
    """
    Helper class for nested configuration access with dot notation.
    
    Allows chaining dot-notation access for nested configuration values.
    """
    
    def __init__(self, config_dict: Dict[str, Any]):
        """
        Initialize ConfigSection with a dictionary.
        
        Args:
            config_dict: Dictionary representing a configuration section
        """
        self._config = config_dict
    
    def __getattr__(self, name: str) -> Any:
        """
        Dot-notation access to nested configuration values.
        
        Args:
            name: Configuration key
        
        Returns:
            Configuration value or ConfigSection for nested dicts
        
        Raises:
            AttributeError: If key doesn't exist
        """
        if name.startswith('_'):
            raise AttributeError(f"'{self.__class__.__name__}' object has no attribute '{name}'")
        
        if name in self._config:
            value = self._config[name]
            if isinstance(value, dict):
                return ConfigSection(value)
            return value
        
        raise AttributeError(f"Configuration key not found: {name}")
    
    def __getitem__(self, key: str) -> Any:
        """
        Dict-style access to configuration values.
        
        Args:
            key: Configuration key
        
        Returns:
            Configuration value
        
        Raises:
            KeyError: If key doesn't exist
        """
        if key not in self._config:
            raise KeyError(f"Configuration key not found: {key}")
        return self._config[key]
    
    def get(self, key: str, default: Any = None) -> Any:
        """
        Get a configuration value with a default fallback.
        
        Args:
            key: Configuration key
            default: Default value if key doesn't exist
        
        Returns:
            Configuration value or default
        """
        return self._config.get(key, default)
    
    def to_dict(self) -> Dict[str, Any]:
        """
        Get the section as a dictionary.
        
        Returns:
            Configuration section dictionary
        """
        return self._config.copy()
