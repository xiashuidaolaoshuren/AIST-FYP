"""Quick verification that the fix resolves the [1] citation issue."""

import sys
from pathlib import Path

sys.path.insert(0, str(Path(__file__).parent))

from src.generation.generator_wrapper import GeneratorWrapper
from src.utils.data_structures import EvidenceChunk

print("=" * 80)
print("VERIFYING FIX FOR [1] CITATION ISSUE")
print("=" * 80)

# Initialize generator
generator = GeneratorWrapper("google/flan-t5-base")

# Create the exact same evidence that caused the issue
evidence_chunks = [
    EvidenceChunk(
        doc_id="wiki_00000278",
        sent_id=59,
        text="Deep learning is a type of machine learning that runs inputs through biologically inspired artificial neural networks for all of these types of learning.",
        char_start=0,
        char_end=160,
        score_dense=0.78,
        rank=1
    ),
    EvidenceChunk(
        doc_id="wiki_00000278",
        sent_id=135,
        text="Deep learning uses several layers of neurons between the network's inputs and outputs.",
        char_start=0,
        char_end=90,
        score_dense=0.74,
        rank=2
    ),
    EvidenceChunk(
        doc_id="wiki_00000278",
        sent_id=138,
        text="Deep learning has drastically improved the performance of programs in many important subfields of artificial intelligence, including computer vision, speech recognition, image classification and others.",
        char_start=0,
        char_end=200,
        score_dense=0.66,
        rank=3
    ),
]

query = "What is deep learning?"

print(f"\nQuery: {query}")
print(f"Evidence chunks: {len(evidence_chunks)}")

# Generate 5 times to check for stochastic behavior
print("\nGenerating 5 responses to check consistency...")
print("-" * 80)

for i in range(5):
    result = generator.generate_with_metadata(query, evidence_chunks)
    response = result['text'].strip()
    
    if response in ['[1]', '[2]', '[3]']:
        print(f"  Run {i+1}: ❌ '{response}' (CITATION REFERENCE - BAD)")
    else:
        print(f"  Run {i+1}: ✓ '{response[:60]}...' (PROPER ANSWER)")

print("\n" * 2)
print("=" * 80)
print("Sample Prompt Format (with fix):")
print("=" * 80)
result = generator.generate_with_metadata(query, evidence_chunks)
print(result['prompt_text'])
print("\n" * 2)
print("=" * 80)
print("CONCLUSION")
print("=" * 80)
print("If you see mostly ✓ marks above, the fix is working!")
print("If you still see ❌ marks, the issue persists and needs further investigation.")
