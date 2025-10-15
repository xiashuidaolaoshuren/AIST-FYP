"""
Demo script for Generator Wrapper with Metadata Capture.

Demonstrates the GeneratorWrapper's ability to generate text responses
with comprehensive metadata capture for hallucination detection.
"""

import sys
from pathlib import Path

sys.path.insert(0, str(Path(__file__).parent.parent))

from src.generation import GeneratorWrapper, extract_claims
from src.utils.data_structures import EvidenceChunk


def main():
    print("="*70)
    print("Generator Wrapper Demo")
    print("="*70)
    
    # Initialize generator
    print("\nInitializing GeneratorWrapper with FLAN-T5-base...")
    generator = GeneratorWrapper(
        model_name='google/flan-t5-base',
        device='cuda'
    )
    print("✓ Model loaded successfully")
    
    # Create sample evidence
    evidence = [
        EvidenceChunk(
            doc_id='ml_doc',
            sent_id=0,
            text='Machine learning is a subset of artificial intelligence.',
            char_start=0,
            char_end=57,
            source='textbook',
            version='v1'
        ),
        EvidenceChunk(
            doc_id='ml_doc',
            sent_id=1,
            text='It uses data to learn patterns and make predictions.',
            char_start=58,
            char_end=111,
            source='textbook',
            version='v1'
        )
    ]
    
    # Test generation with metadata
    print("\n" + "="*70)
    print("Test 1: Generation with Evidence")
    print("="*70)
    
    prompt = "What is machine learning?"
    print(f"\nPrompt: {prompt}")
    print(f"Evidence: {len(evidence)} chunks")
    
    result = generator.generate_with_metadata(
        prompt=prompt,
        evidence_chunks=evidence,
        max_new_tokens=50
    )
    
    print(f"\n✓ Generated text: {result['text']}")
    print(f"✓ Tokens generated: {len(result['tokens'])}")
    print(f"✓ Logits captured: {len(result['logits'])} positions")
    print(f"✓ Scores captured: {len(result['scores'])} probabilities")
    print(f"✓ Evidence used: {result['evidence_used']}")
    
    # Extract claims from generated text
    print("\n" + "="*70)
    print("Test 2: Claim Extraction")
    print("="*70)
    
    claims = extract_claims(result['text'])
    print(f"\n✓ Extracted {len(claims)} claims:")
    for i, claim in enumerate(claims, 1):
        print(f"\n{i}. {claim.text}")
        print(f"   ID: {claim.claim_id[:8]}...")
        print(f"   Char span: {claim.answer_char_span}")
        print(f"   Method: {claim.extraction_method}")
    
    # Test generation without evidence
    print("\n" + "="*70)
    print("Test 3: Generation without Evidence")
    print("="*70)
    
    prompt2 = "What is 2 + 2?"
    result2 = generator.generate_with_metadata(
        prompt=prompt2,
        max_new_tokens=20
    )
    
    print(f"\nPrompt: {prompt2}")
    print(f"Generated: {result2['text']}")
    print(f"Tokens: {len(result2['tokens'])}")
    
    print("\n" + "="*70)
    print("✓ Demo completed successfully!")
    print("="*70)


if __name__ == '__main__':
    main()
