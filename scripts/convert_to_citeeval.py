"""
Convert pipeline outputs to CiteEval/CiteBench format.

This script takes the full pipeline output JSON (from demo_full_pipeline.py)
or baseline RAG outputs and converts them to the CiteEval benchmark format
for evaluation.

Usage:
    # Convert a single pipeline output file
    python scripts/convert_to_citeeval.py --input outputs/full_pipeline_queries_20260201_173435.json --output citeeval_input.json
    
    # Run the full pipeline and convert to CiteEval format in one step
    python scripts/convert_to_citeeval.py --run-pipeline --queries "What is AI?" "What is deep learning?" --output citeeval_input.json
    
    # Convert with custom strategy
    python scripts/convert_to_citeeval.py --input outputs/my_results.json --output citeeval_input.json --strategy validation
"""

import sys
import json
import argparse
from pathlib import Path
from typing import List, Dict, Any

sys.path.insert(0, str(Path(__file__).parent.parent))

from src.pipelines import BaselineRAGPipeline
from src.verification.verifier_hub import VerifierHub
from src.verification.rule_based_aggregator import RuleBasedAggregator
from src.citation.citation_formatter import CitationFormatter
from src.utils.config import Config
from src.utils.logger import setup_logger


def convert_pipeline_output_to_citeeval(
    pipeline_output: Dict[str, Any],
    citation_formatter: CitationFormatter,
    sample_id: str
) -> Dict[str, Any]:
    """
    Convert a single pipeline output to CiteEval format.
    
    Args:
        pipeline_output: Output from BaselineRAGPipeline with claims, evidence, etc.
        citation_formatter: CitationFormatter instance
        sample_id: Unique identifier for this sample
    
    Returns:
        Dict in CiteEval format with keys: id, query, passages, pred
    """
    query = pipeline_output.get('query', '')
    answer_text = pipeline_output.get('answer', '')
    claims = pipeline_output.get('claims', [])
    evidence_map = pipeline_output.get('evidence_map', {})
    
    # Format with citations
    formatted_output = citation_formatter.format_with_citations(
        answer_text=answer_text,
        claims=claims,
        evidence_map=evidence_map
    )
    
    # Export to CiteEval format
    citeeval_sample = citation_formatter.export_citeeval_format(
        query=query,
        formatted_output=formatted_output,
        answer_id=sample_id
    )
    
    return citeeval_sample


def convert_file_to_citeeval(
    input_file: str,
    output_file: str,
    logger
) -> None:
    """
    Convert a JSON file of pipeline outputs to CiteEval format.
    
    Args:
        input_file: Path to input JSON file with pipeline outputs
        output_file: Path to output JSON file in CiteEval format
        logger: Logger instance
    """
    logger.info(f"Loading pipeline outputs from: {input_file}")
    
    with open(input_file, 'r', encoding='utf-8') as f:
        pipeline_outputs = json.load(f)
    
    if not isinstance(pipeline_outputs, list):
        pipeline_outputs = [pipeline_outputs]
    
    logger.info(f"Found {len(pipeline_outputs)} samples to convert")
    
    # Initialize citation formatter
    config = Config()
    citation_formatter = CitationFormatter(config)
    
    # Convert each sample
    citeeval_samples = []
    for idx, output in enumerate(pipeline_outputs):
        sample_id = output.get('id', f'sample_{idx+1}')
        
        try:
            citeeval_sample = convert_pipeline_output_to_citeeval(
                pipeline_output=output,
                citation_formatter=citation_formatter,
                sample_id=sample_id
            )
            citeeval_samples.append(citeeval_sample)
            logger.info(f"Converted sample {sample_id}")
        except Exception as e:
            logger.error(f"Failed to convert sample {sample_id}: {e}")
            continue
    
    # Save to output file
    logger.info(f"Saving {len(citeeval_samples)} samples to: {output_file}")
    
    output_path = Path(output_file)
    output_path.parent.mkdir(parents=True, exist_ok=True)
    
    with open(output_file, 'w', encoding='utf-8') as f:
        json.dump(citeeval_samples, f, indent=2, ensure_ascii=False)
    
    logger.info(f"Successfully saved CiteEval format output to {output_file}")


def run_pipeline_and_convert(
    queries: List[str],
    output_file: str,
    strategy: str,
    logger
) -> None:
    """
    Run the full pipeline on queries and convert results to CiteEval format.
    
    Args:
        queries: List of query strings to process
        output_file: Path to output JSON file in CiteEval format
        strategy: Dataset strategy (development, validation, production)
        logger: Logger instance
    """
    logger.info(f"Running pipeline with strategy: {strategy}")
    logger.info(f"Processing {len(queries)} queries")
    
    # Initialize components
    config = Config()
    pipeline = BaselineRAGPipeline.from_config(config, strategy=strategy)
    verifier_hub = VerifierHub(config, pipeline.generator)
    aggregator = RuleBasedAggregator(config)
    citation_formatter = CitationFormatter(config)
    
    # Process each query
    citeeval_samples = []
    for idx, query in enumerate(queries):
        sample_id = f"query_{idx+1}"
        logger.info(f"Processing {sample_id}: {query}")
        
        try:
            # Run pipeline
            result = pipeline.retrieve_and_generate(query)
            
            # Verify claims
            claims_with_scores = verifier_hub.verify_claims(
                claims=result['claims'],
                evidence_chunks=result['evidence_chunks'],
                query=query
            )
            
            # Build evidence map (map claim_id to evidence)
            evidence_map = {}
            for claim in result['claims']:
                # Get evidence for this claim (top-k from retrieved evidence)
                evidence_map[claim.claim_id] = result['evidence_chunks'][:5]
            
            # Prepare pipeline output
            pipeline_output = {
                'id': sample_id,
                'query': query,
                'answer': result['answer'],
                'claims': result['claims'],
                'evidence_map': evidence_map
            }
            
            # Convert to CiteEval format
            citeeval_sample = convert_pipeline_output_to_citeeval(
                pipeline_output=pipeline_output,
                citation_formatter=citation_formatter,
                sample_id=sample_id
            )
            citeeval_samples.append(citeeval_sample)
            logger.info(f"Successfully processed {sample_id}")
            
        except Exception as e:
            logger.error(f"Failed to process {sample_id}: {e}")
            continue
    
    # Save to output file
    logger.info(f"Saving {len(citeeval_samples)} samples to: {output_file}")
    
    output_path = Path(output_file)
    output_path.parent.mkdir(parents=True, exist_ok=True)
    
    with open(output_file, 'w', encoding='utf-8') as f:
        json.dump(citeeval_samples, f, indent=2, ensure_ascii=False)
    
    logger.info(f"Successfully saved CiteEval format output to {output_file}")


def main():
    parser = argparse.ArgumentParser(
        description='Convert pipeline outputs to CiteEval/CiteBench format'
    )
    
    # Mode selection
    mode_group = parser.add_mutually_exclusive_group(required=True)
    mode_group.add_argument(
        '--input',
        type=str,
        help='Path to input JSON file with pipeline outputs'
    )
    mode_group.add_argument(
        '--run-pipeline',
        action='store_true',
        help='Run the pipeline and convert results directly'
    )
    
    # Common arguments
    parser.add_argument(
        '--output',
        type=str,
        required=True,
        help='Path to output JSON file in CiteEval format'
    )
    parser.add_argument(
        '--strategy',
        type=str,
        choices=['development', 'validation', 'production'],
        default='development',
        help='Dataset strategy (default: development)'
    )
    
    # Pipeline mode arguments
    parser.add_argument(
        '--queries',
        type=str,
        nargs='+',
        help='List of queries to process (required with --run-pipeline)'
    )
    
    args = parser.parse_args()
    
    # Setup logger
    logger = setup_logger('convert_to_citeeval')
    
    # Validate arguments
    if args.run_pipeline and not args.queries:
        parser.error("--queries is required when using --run-pipeline")
    
    # Execute conversion
    if args.input:
        convert_file_to_citeeval(
            input_file=args.input,
            output_file=args.output,
            logger=logger
        )
    else:
        run_pipeline_and_convert(
            queries=args.queries,
            output_file=args.output,
            strategy=args.strategy,
            logger=logger
        )


if __name__ == '__main__':
    main()
