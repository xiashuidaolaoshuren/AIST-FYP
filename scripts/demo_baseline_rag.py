"""
Demo script for Baseline RAG Pipeline.

Demonstrates the end-to-end RAG pipeline with sample queries,
showing retrieval, generation, claim extraction, and claim-evidence pairing.
Supports multiple data strategies (development, validation, production).
"""

import sys
import json
from pathlib import Path
from datetime import datetime

sys.path.insert(0, str(Path(__file__).parent.parent))

import numpy as np

from src.pipelines import BaselineRAGPipeline
from src.utils.logger import setup_logger


def make_json_serializable(obj):
    """
    Recursively convert numpy arrays and other non-JSON-serializable objects.
    
    Args:
        obj: Object to convert (dict, list, numpy array, etc.)
        
    Returns:
        JSON-serializable version of the object
    """
    if isinstance(obj, dict):
        return {key: make_json_serializable(value) for key, value in obj.items()}
    elif isinstance(obj, (list, tuple)):
        return [make_json_serializable(item) for item in obj]
    elif isinstance(obj, np.ndarray):
        return obj.tolist()
    elif isinstance(obj, (np.integer, np.floating)):
        return obj.item()
    elif isinstance(obj, np.bool_):
        return bool(obj)
    else:
        return obj


def detect_available_strategies() -> list:
    """
    Detect which data strategies have available FAISS indices.
    
    Returns:
        List of strategy names that have valid FAISS indices
    """
    strategies = ['development', 'validation', 'production']
    available = []
    
    for strategy in strategies:
        index_path = Path(f"data/indexes/{strategy}/faiss.index")
        if index_path.exists():
            available.append(strategy)
            print(f"  ✓ {strategy.upper()}: FAISS index found at {index_path}")
        else:
            print(f"  ✗ {strategy.upper()}: No FAISS index at {index_path}")
    
    return available


def ask_user_to_choose_strategy(strategies: list) -> str:
    """
    Ask user to choose a strategy from available options.
    
    Args:
        strategies: List of available strategy names
    
    Returns:
        Chosen strategy name
    """
    print("\n" + "=" * 80)
    print("STRATEGY SELECTION")
    print("=" * 80)
    print(f"\nAvailable strategies: {', '.join([s.upper() for s in strategies])}\n")
    
    for i, strategy in enumerate(strategies, 1):
        print(f"  {i}. {strategy.upper()}")
    
    while True:
        try:
            choice = input("\nChoose a strategy (enter number): ").strip()
            choice_idx = int(choice) - 1
            
            if 0 <= choice_idx < len(strategies):
                chosen = strategies[choice_idx]
                print(f"\n✓ Selected: {chosen.upper()}")
                return chosen
            else:
                print(f"Invalid choice. Please enter a number between 1 and {len(strategies)}")
        except ValueError:
            print("Invalid input. Please enter a valid number.")
        except KeyboardInterrupt:
            print("\n\nInterrupted by user")
            sys.exit(0)


def print_section(title: str, char: str = "="):
    """Print a formatted section header."""
    print(f"\n{char * 80}")
    print(title)
    print(f"{char * 80}\n")


def display_evidence(evidence_chunks: list, max_display: int = 3):
    """Display retrieved evidence chunks."""
    for i, evidence in enumerate(evidence_chunks[:max_display], 1):
        print(f"  [{i}] (Score: {evidence.get('score_dense', 0):.4f}, Rank: {evidence.get('rank', 0)})")
        print(f"      Doc: {evidence.get('doc_id', 'N/A')}#{evidence.get('sent_id', 0)}")
        print(f"      Text: {evidence.get('text', '')[:150]}...")
        print()


def display_claims(claim_evidence_pairs: list):
    """Display extracted claims with their evidence."""
    for i, pair in enumerate(claim_evidence_pairs, 1):
        print(f"  Claim {i}:")
        print(f"    ID: {pair['claim_id']}")
        print(f"    Top Evidence: {pair['top_evidence']}")
        print(f"    Evidence Candidates: {len(pair['evidence_candidates'])} chunks")
        print(f"    Evidence Spans Available: {len(pair['evidence_spans'])} chunks")
        print()


def display_sub_answers(sub_answers: list, claims_by_sub_answer: list):
    """Display extracted sub-answers and their claims."""
    if not sub_answers or len(sub_answers) <= 1:
        return  # Skip display if only one answer
    
    print("🔀 Multi-Question Analysis:")
    print(f"   Total Sub-Answers: {len(sub_answers)}\n")
    
    for sub_ans_data in claims_by_sub_answer:
        sub_id = sub_ans_data['sub_answer_id']
        sub_text = sub_ans_data['sub_text']
        sub_claims = sub_ans_data['claims']
        
        print(f"   [Sub-Answer {sub_id + 1}]")
        print(f"      Text: {sub_text[:100]}...")
        print(f"      Claims Extracted: {len(sub_claims)}")
        for claim in sub_claims:
            print(f"        - {claim.text[:80]}")
        print()


def run_demo():
    """Run the baseline RAG pipeline demo."""
    print_section("BASELINE RAG PIPELINE DEMO", "=")
    
    logger = setup_logger(__name__)
    
    # Step 1: Detect available strategies
    print("🔍 Detecting available FAISS indices...\n")
    available_strategies = detect_available_strategies()
    
    if not available_strategies:
        print("\n❌ ERROR: No FAISS indices found!")
        print("\nPlease run the data processing pipeline first:")
        print("  1. python scripts/prepare_wikipedia_chunks.py")
        print("  2. python scripts/generate_embeddings.py")
        print("  3. python scripts/build_faiss_index.py")
        return
    
    # Step 1.5: Choose strategy (auto-select if only one, ask if multiple)
    if len(available_strategies) == 1:
        chosen_strategy = available_strategies[0]
        print(f"\n✓ Only one strategy available, using: {chosen_strategy.upper()}")
    else:
        chosen_strategy = ask_user_to_choose_strategy(available_strategies)
    
    # Step 2: Initialize pipeline
    print("\n🔧 Initializing Pipeline...")
    print(f"   Loading from config.yaml ({chosen_strategy} strategy)")
    
    try:
        pipeline = BaselineRAGPipeline.from_config(
            config_path="config.yaml",
            strategy=chosen_strategy
        )
        print("✓ Pipeline initialized successfully\n")
    except FileNotFoundError as e:
        print(f"\n❌ ERROR: {e}")
        print("\nPlease ensure you have:")
        print("  1. Processed Wikipedia data (scripts/process_wikipedia.py)")
        print("  2. Generated embeddings (scripts/generate_embeddings.py)")
        print("  3. Built FAISS index (scripts/build_faiss_index.py)")
        return
    except Exception as e:
        print(f"\n❌ ERROR initializing pipeline: {e}")
        import traceback
        traceback.print_exc()
        return
    
    # Define sample queries
    sample_queries = [
        "What is artificial intelligence?",
        "How do machines learn from data?",
        "What is deep learning?",
        "What is natural language processing?",
        # Multi-question queries to test the new sub-answer handling
        "What is machine learning? How does it differ from traditional programming?",
        "Explain neural networks? What are the main types? How do they work?"
    ]
    
    # Store all results for JSON output
    all_results = []
    
    # Run each query
    for query_idx, query in enumerate(sample_queries, 1):
        print_section(f"Query {query_idx}: {query}", "-")
        
        try:
            # Run pipeline
            print("🔍 Running pipeline...")
            result = pipeline.run(query, top_k=5)
            
            # Display results
            print(f"\n✓ Pipeline completed successfully\n")
            
            # Show draft response
            print("📝 Generated Response:")
            print(f"   {result['draft_response']}\n")
            
            # Show sub-answers if multiple questions were asked
            sub_answers = result.get('sub_answers', [])
            claims_by_sub_answer = result.get('claims_by_sub_answer', [])
            if sub_answers and len(sub_answers) > 1:
                display_sub_answers(sub_answers, claims_by_sub_answer)
            
            # Show retrieval metadata
            print("📊 Retrieval Metadata:")
            retrieval = result['retrieval_metadata']
            print(f"   Retrieved: {retrieval['num_retrieved']} chunks")
            print(f"   Top Score: {retrieval['top_score']:.4f}")
            print(f"   Documents: {', '.join(retrieval['evidence_doc_ids'][:3])}...")
            print()
            
            # Show top evidence
            if result['claim_evidence_pairs']:
                print("📚 Top Retrieved Evidence:")
                first_pair = result['claim_evidence_pairs'][0]
                display_evidence(first_pair['evidence_spans'], max_display=3)
            
            # Show claims
            print(f"🔖 Extracted Claims: {len(result['claim_evidence_pairs'])}")
            if result['claim_evidence_pairs']:
                display_claims(result['claim_evidence_pairs'])
            else:
                print("   (No claims extracted)")
            
            # Show generator metadata summary
            print("🤖 Generator Metadata:")
            gen_meta = result['generator_metadata']
            print(f"   Tokens Generated: {len(gen_meta.get('tokens', []))}")
            print(f"   Sub-Answers Generated: {len(gen_meta.get('sub_answers', []))}")
            print(f"   Logits Captured: {len(gen_meta.get('logits', []))} positions")
            print(f"   Scores Captured: {len(gen_meta.get('scores', []))} probabilities")
            print(f"   Evidence Used: {len(gen_meta.get('evidence_used', []))} chunks")
            print()
            
            # Add to results
            all_results.append({
                'query_index': query_idx,
                'query': query,
                'result': result,
                'timestamp': datetime.now().isoformat()
            })
            
        except Exception as e:
            print(f"\n❌ ERROR running query: {e}")
            import traceback
            traceback.print_exc()
            continue
    
    # Save results to JSON
    print_section("Saving Results", "=")
    
    output_dir = Path("outputs")
    output_dir.mkdir(exist_ok=True)
    
    timestamp = datetime.now().strftime("%Y%m%d_%H%M%S")
    output_file = output_dir / f"rag_demo_results_{timestamp}.json"
    
    try:
        # Prepare JSON output (remove non-serializable numpy arrays)
        json_results = []
        for result_obj in all_results:
            result = result_obj['result'].copy()
            
            # Remove logits (numpy arrays) for JSON serialization
            if 'generator_metadata' in result:
                gen_meta = result['generator_metadata'].copy()
                if 'logits' in gen_meta:
                    # Replace logits with shape info
                    logits_info = [
                        {'shape': list(logit.shape), 'dtype': str(logit.dtype)}
                        for logit in gen_meta['logits']
                    ]
                    gen_meta['logits'] = logits_info
                result['generator_metadata'] = gen_meta
            
            # Convert Claim objects to dictionaries for JSON serialization
            if 'claims_by_sub_answer' in result:
                claims_by_sub_answer_serialized = []
                for sub_ans_data in result['claims_by_sub_answer']:
                    sub_data_copy = sub_ans_data.copy()
                    # Convert Claim objects to dicts
                    if 'claims' in sub_data_copy:
                        sub_data_copy['claims'] = [
                            {
                                'claim_id': claim.claim_id,
                                'answer_id': claim.answer_id,
                                'text': claim.text,
                                'answer_char_span': claim.answer_char_span,
                                'extraction_method': claim.extraction_method
                            }
                            for claim in sub_data_copy['claims']
                        ]
                    claims_by_sub_answer_serialized.append(sub_data_copy)
                result['claims_by_sub_answer'] = claims_by_sub_answer_serialized
            
            # Apply recursive serialization to handle any remaining numpy arrays
            result = make_json_serializable(result)
            
            json_results.append({
                'query_index': result_obj['query_index'],
                'query': result_obj['query'],
                'timestamp': result_obj['timestamp'],
                'result': result
            })
        
        # Write to file
        with open(output_file, 'w', encoding='utf-8') as f:
            json.dump(json_results, f, indent=2, ensure_ascii=False)
        
        print(f"✓ Results saved to: {output_file}")
        print(f"  File size: {output_file.stat().st_size / 1024:.2f} KB")
        
    except Exception as e:
        print(f"❌ ERROR saving results: {e}")
        import traceback
        traceback.print_exc()
    
    # Summary
    print_section("Demo Summary", "=")
    print(f"✓ Queries processed: {len(all_results)}/{len(sample_queries)}")
    print(f"✓ Total claims extracted: {sum(len(r['result']['claim_evidence_pairs']) for r in all_results)}")
    print(f"✓ Pipeline components:")
    print(f"   - Retriever: DenseRetriever")
    print(f"   - Generator: GeneratorWrapper")
    print(f"   - Claim Extractor: ClaimExtractor (auto method)")
    print(f"\n✓ Demo completed successfully!")
    print("="*80)


def main():
    """Main entry point."""
    try:
        run_demo()
    except KeyboardInterrupt:
        print("\n\n⚠️  Demo interrupted by user")
        sys.exit(0)
    except Exception as e:
        print(f"\n\n❌ FATAL ERROR: {e}")
        import traceback
        traceback.print_exc()
        sys.exit(1)


if __name__ == '__main__':
    main()
