"""
RagasEvaluator: Wrapper for Ragas framework RAG evaluation.

This module provides a clean interface to the Ragas framework for evaluating
RAG (Retrieval-Augmented Generation) systems using metrics like Faithfulness,
AnswerRelevancy, and ContextPrecision.

Ragas (Retrieval-Augmented Generation Assessment) is an independent evaluation
framework that uses LLMs to assess RAG system performance across multiple dimensions.

Example:
    >>> from src.utils.config import Config
    >>> from src.evaluation.ragas_evaluator import RagasEvaluator
    >>> 
    >>> config = Config('config.yaml')
    >>> evaluator = RagasEvaluator(config)
    >>> 
    >>> rag_results = [
    ...     {
    ...         'question': 'What is a cat?',
    ...         'answer': 'A cat is a small carnivorous mammal.',
    ...         'contexts': ['Cats are mammals...', 'They are carnivores...'],
    ...         'ground_truth': 'A cat is a domesticated carnivorous mammal.'
    ...     },
    ...     # ... more samples
    ... ]
    >>> 
    >>> df = evaluator.evaluate_rag_outputs(rag_results)
    >>> print(df[['question', 'faithfulness', 'answer_relevancy', 'context_precision']])
    >>> print(f"Mean Faithfulness: {df['faithfulness'].mean():.3f}")

References:
    - Ragas GitHub: https://github.com/explodinggradients/ragas
    - Ragas Paper: https://arxiv.org/abs/2309.15217
    - Ragas Docs: https://docs.ragas.io/
"""

import os
from typing import List, Dict, Any, Optional
import pandas as pd
from datasets import Dataset

from src.utils.config import Config
from src.utils.logger import setup_logger

# Import Ragas components
try:
    from ragas import evaluate
    from ragas.metrics import (
        Faithfulness,
        AnswerRelevancy,
        ContextPrecision,
        ContextRecall,
        ContextRelevance,
        AnswerCorrectness
    )
    from ragas.llms import LangchainLLMWrapper
    from langchain_openai import ChatOpenAI
except ImportError as e:
    raise ImportError(
        "Ragas dependencies not installed. "
        "Please run: pip install ragas>=0.1.0 langchain-openai>=0.1.0"
    ) from e


class RagasEvaluator:
    """
    Wrapper for Ragas framework to evaluate RAG system outputs.
    
    This class provides a simplified interface to Ragas evaluation metrics,
    handling dataset format conversion, LLM initialization, and result processing.
    
    Supported Metrics:
        - Faithfulness: Measures factual consistency between answer and contexts
        - AnswerRelevancy: Measures how relevant the answer is to the question
        - ContextPrecision: Measures how well retrieved contexts match ground truth
        - ContextRecall: Measures how much of ground truth is covered by contexts
        - ContextRelevance: Measures relevance of retrieved contexts to question
        - AnswerCorrectness: Measures semantic and factual similarity to ground truth
    
    Configuration:
        Requires config.yaml section:
        ```yaml
        evaluation:
          ragas:
            model: "gpt-4o-mini"
            temperature: 0
            metrics:
              - faithfulness
              - answer_relevancy
              - context_precision
        ```
    
    Environment Variables:
        OPENAI_API_KEY: Required for OpenAI API access
        OPENAI_ORG_ID: Optional organization ID
    
    Attributes:
        config: Configuration object
        model_name: OpenAI model name for evaluation
        temperature: LLM temperature (0 = deterministic)
        evaluator_llm: Wrapped LLM for Ragas metrics
        metrics: List of initialized Ragas metrics
        metric_names: List of metric names for logging
        logger: Logger instance
    
    Cost Warning:
        Ragas evaluation makes OpenAI API calls for each sample and metric.
        Costs can accumulate quickly on large datasets. Use sparingly on
        development/test sets. Monitor your OpenAI usage dashboard.
    """
    
    # Mapping of metric names to Ragas metric classes
    AVAILABLE_METRICS = {
        'faithfulness': Faithfulness,
        'answer_relevancy': AnswerRelevancy,
        'context_precision': ContextPrecision,
        'context_recall': ContextRecall,
        'context_relevance': ContextRelevance,
        'answer_correctness': AnswerCorrectness
    }
    
    def __init__(self, config: Config):
        """
        Initialize RagasEvaluator with configuration.
        
        Sets up the OpenAI LLM, wraps it for Ragas, and initializes
        the specified metrics from configuration.
        
        Args:
            config: Configuration object with evaluation.ragas section
        
        Raises:
            ValueError: If OPENAI_API_KEY not found in environment
            KeyError: If required config sections are missing
            ImportError: If ragas or langchain-openai not installed
        
        Example:
            >>> config = Config('config.yaml')
            >>> evaluator = RagasEvaluator(config)
            INFO: RagasEvaluator initialized with model: gpt-4o-mini
            INFO: Loaded 3 metrics: faithfulness, answer_relevancy, context_precision
        """
        self.config = config
        self.logger = setup_logger(__name__)
        
        # Check for OpenAI API key
        if not os.getenv('OPENAI_API_KEY'):
            self.logger.error(
                "OPENAI_API_KEY not found in environment. "
                "Ragas requires OpenAI API access. Please set OPENAI_API_KEY "
                "in your .env file or environment variables."
            )
            raise ValueError(
                "OPENAI_API_KEY environment variable is required for Ragas evaluation"
            )
        
        # Load configuration
        self.model_name = config.get('evaluation.ragas.model', 'gpt-4o-mini')
        self.temperature = config.get('evaluation.ragas.temperature', 0)
        metric_names = config.get('evaluation.ragas.metrics', [
            'faithfulness', 'answer_relevancy', 'context_precision'
        ])
        
        self.logger.info(
            f"Initializing RagasEvaluator with model: {self.model_name}, "
            f"temperature: {self.temperature}"
        )
        
        # Initialize OpenAI LLM
        try:
            llm = ChatOpenAI(
                model=self.model_name,
                temperature=self.temperature
            )
            self.evaluator_llm = LangchainLLMWrapper(llm)
            self.logger.info("LLM wrapper initialized successfully")
        except Exception as e:
            self.logger.error(f"Failed to initialize OpenAI LLM: {str(e)}")
            raise
        
        # Initialize metrics
        self.metrics = []
        self.metric_names = []
        
        for metric_name in metric_names:
            if metric_name not in self.AVAILABLE_METRICS:
                self.logger.warning(
                    f"Unknown metric '{metric_name}', skipping. "
                    f"Available metrics: {list(self.AVAILABLE_METRICS.keys())}"
                )
                continue
            
            try:
                metric_class = self.AVAILABLE_METRICS[metric_name]
                metric = metric_class(llm=self.evaluator_llm)
                self.metrics.append(metric)
                self.metric_names.append(metric_name)
                self.logger.debug(f"Initialized metric: {metric_name}")
            except Exception as e:
                self.logger.error(f"Failed to initialize metric '{metric_name}': {str(e)}")
                raise
        
        if not self.metrics:
            raise ValueError(
                "No valid metrics configured. Please check evaluation.ragas.metrics in config.yaml"
            )
        
        self.logger.info(
            f"RagasEvaluator initialized successfully with {len(self.metrics)} metrics: "
            f"{', '.join(self.metric_names)}"
        )
    
    def evaluate_rag_outputs(
        self,
        rag_results: List[Dict[str, Any]],
        show_progress: bool = True
    ) -> pd.DataFrame:
        """
        Evaluate RAG system outputs using Ragas metrics.
        
        Takes a list of RAG results (question, answer, contexts, ground_truth)
        and computes Ragas metrics for each sample. Returns a pandas DataFrame
        with per-sample scores.
        
        Args:
            rag_results: List of dicts with keys:
                - question (str): The input question
                - answer (str): Generated answer from RAG system
                - contexts (List[str]): Retrieved context passages
                - ground_truth (str): Reference answer (optional for some metrics)
            show_progress: Whether to show progress bar during evaluation
        
        Returns:
            pd.DataFrame: Results with columns:
                - question: Input question
                - answer: Generated answer
                - contexts: Retrieved contexts (as list)
                - ground_truth: Reference answer
                - <metric_name>: Score for each metric (float, typically 0-1)
        
        Raises:
            ValueError: If rag_results is empty or has invalid format
            KeyError: If required fields are missing from rag_results
        
        Example:
            >>> rag_results = [
            ...     {
            ...         'question': 'What is a cat?',
            ...         'answer': 'A cat is a mammal.',
            ...         'contexts': ['Cats are mammals.', 'They have fur.'],
            ...         'ground_truth': 'A cat is a small carnivorous mammal.'
            ...     }
            ... ]
            >>> df = evaluator.evaluate_rag_outputs(rag_results)
            >>> print(df['faithfulness'].mean())
            0.875
        
        Notes:
            - Ragas evaluation can be slow (multiple API calls per sample)
            - Costs scale with: num_samples * num_metrics * API_cost
            - Some metrics require ground_truth, others don't
            - Contexts should be list of strings, not single string
        """
        if not rag_results:
            raise ValueError("rag_results cannot be empty")
        
        self.logger.info(f"Starting Ragas evaluation on {len(rag_results)} samples")
        
        # Validate and convert to Ragas dataset format
        try:
            ragas_dataset = self._convert_to_ragas_format(rag_results)
            self.logger.debug(f"Converted dataset format successfully")
        except Exception as e:
            self.logger.error(f"Failed to convert dataset format: {str(e)}")
            raise
        
        # Run Ragas evaluation
        try:
            self.logger.info(
                f"Running Ragas evaluation with metrics: {', '.join(self.metric_names)}"
            )
            self.logger.warning(
                "This will make OpenAI API calls. Monitor your usage at "
                "https://platform.openai.com/usage"
            )
            
            result = evaluate(
                dataset=ragas_dataset,
                metrics=self.metrics,
                llm=self.evaluator_llm
            )
            
            self.logger.info("Ragas evaluation completed successfully")
        except Exception as e:
            self.logger.error(f"Ragas evaluation failed: {str(e)}")
            raise
        
        # Convert results to DataFrame
        try:
            df = result.to_pandas()
            self.logger.debug(f"Converted results to DataFrame: {df.shape}")
        except Exception as e:
            self.logger.error(f"Failed to convert results to DataFrame: {str(e)}")
            raise
        
        # Log summary statistics
        self._log_summary_statistics(df)
        
        return df
    
    def _convert_to_ragas_format(self, rag_results: List[Dict[str, Any]]) -> Dataset:
        """
        Convert RAG results to Ragas dataset format.
        
        Ragas expects a Hugging Face Dataset with specific structure:
        - question: List[str]
        - answer: List[str]
        - contexts: List[List[str]]  # Note: list of lists!
        - ground_truth: List[str]
        
        Args:
            rag_results: List of dicts with question, answer, contexts, ground_truth
        
        Returns:
            Dataset: Hugging Face Dataset in Ragas format
        
        Raises:
            KeyError: If required fields are missing
            ValueError: If data format is invalid
        """
        # Validate required fields
        required_fields = ['question', 'answer', 'contexts']
        for i, result in enumerate(rag_results):
            for field in required_fields:
                if field not in result:
                    raise KeyError(
                        f"Missing required field '{field}' in rag_results[{i}]"
                    )
        
        # Extract and validate data
        data_dict = {
            'question': [],
            'answer': [],
            'contexts': [],
            'ground_truth': []
        }
        
        for i, result in enumerate(rag_results):
            # Question
            data_dict['question'].append(str(result['question']))
            
            # Answer
            data_dict['answer'].append(str(result['answer']))
            
            # Contexts - must be list of strings
            contexts = result['contexts']
            if isinstance(contexts, str):
                # If single string, wrap in list
                contexts = [contexts]
                self.logger.warning(
                    f"Sample {i}: contexts was a string, converted to list. "
                    "Expected List[str]."
                )
            elif not isinstance(contexts, list):
                raise ValueError(
                    f"Sample {i}: contexts must be List[str], got {type(contexts)}"
                )
            
            # Ensure all context items are strings
            contexts = [str(c) for c in contexts]
            data_dict['contexts'].append(contexts)
            
            # Ground truth (optional for some metrics)
            ground_truth = result.get('ground_truth', '')
            data_dict['ground_truth'].append(str(ground_truth) if ground_truth else '')
        
        # Create Hugging Face Dataset
        try:
            dataset = Dataset.from_dict(data_dict)
            self.logger.debug(
                f"Created Ragas dataset: {len(dataset)} samples, "
                f"fields: {list(data_dict.keys())}"
            )
            return dataset
        except Exception as e:
            self.logger.error(f"Failed to create Ragas dataset: {str(e)}")
            raise
    
    def _log_summary_statistics(self, df: pd.DataFrame) -> None:
        """
        Log summary statistics for evaluation results.
        
        Computes and logs mean scores for each metric, helping with
        quick assessment of RAG system performance.
        
        Args:
            df: DataFrame with evaluation results
        """
        self.logger.info("=" * 60)
        self.logger.info("Ragas Evaluation Summary")
        self.logger.info("=" * 60)
        
        for metric_name in self.metric_names:
            if metric_name in df.columns:
                mean_score = df[metric_name].mean()
                std_score = df[metric_name].std()
                min_score = df[metric_name].min()
                max_score = df[metric_name].max()
                
                self.logger.info(
                    f"{metric_name.capitalize():20s}: "
                    f"mean={mean_score:.3f}, std={std_score:.3f}, "
                    f"min={min_score:.3f}, max={max_score:.3f}"
                )
        
        self.logger.info("=" * 60)
        self.logger.info(
            f"Evaluated {len(df)} samples with {len(self.metrics)} metrics"
        )
        self.logger.info("=" * 60)
    
    def add_metric(self, metric_name: str) -> None:
        """
        Add an additional metric to the evaluator.
        
        Allows dynamic addition of metrics after initialization.
        Useful for experimenting with different evaluation approaches.
        
        Args:
            metric_name: Name of metric to add (must be in AVAILABLE_METRICS)
        
        Raises:
            ValueError: If metric_name is not recognized or already exists
        
        Example:
            >>> evaluator.add_metric('context_recall')
            INFO: Added metric: context_recall
        """
        if metric_name in self.metric_names:
            raise ValueError(f"Metric '{metric_name}' already exists")
        
        if metric_name not in self.AVAILABLE_METRICS:
            raise ValueError(
                f"Unknown metric '{metric_name}'. "
                f"Available: {list(self.AVAILABLE_METRICS.keys())}"
            )
        
        try:
            metric_class = self.AVAILABLE_METRICS[metric_name]
            metric = metric_class(llm=self.evaluator_llm)
            self.metrics.append(metric)
            self.metric_names.append(metric_name)
            self.logger.info(f"Added metric: {metric_name}")
        except Exception as e:
            self.logger.error(f"Failed to add metric '{metric_name}': {str(e)}")
            raise
    
    def get_available_metrics(self) -> List[str]:
        """
        Get list of all available Ragas metrics.
        
        Returns:
            List[str]: Names of all supported metrics
        
        Example:
            >>> evaluator.get_available_metrics()
            ['faithfulness', 'answer_relevancy', 'context_precision', ...]
        """
        return list(self.AVAILABLE_METRICS.keys())
