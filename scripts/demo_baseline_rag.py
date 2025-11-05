"""
Demo script for Baseline RAG Pipeline.

Demonstrates the end-to-end RAG pipeline with sample queries,
showing retrieval, generation, claim extraction, and claim-evidence pairing.
"""

import sys
import json
from pathlib import Path
from datetime import datetime

sys.path.insert(0, str(Path(__file__).parent.parent))

from src.pipelines import BaselineRAGPipeline
from src.utils.logger import setup_logger


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


def run_demo():
    """Run the baseline RAG pipeline demo."""
    print_section("BASELINE RAG PIPELINE DEMO", "=")
    
    logger = setup_logger(__name__)
    
    # Initialize pipeline
    print("🔧 Initializing Pipeline...")
    print("   Loading from config.yaml (development strategy)")
    
    try:
        pipeline = BaselineRAGPipeline.from_config(
            config_path="config.yaml",
            strategy="development"
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
        "What is natural language processing?"
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
