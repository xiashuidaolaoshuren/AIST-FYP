"""
Demo script for Full Pipeline with Gradio UI.

Launches the complete hallucination detection system with a web interface
for interactive visualization of claim verification results.
Supports multiple data strategies (development, validation, production).
"""

import sys
from pathlib import Path

sys.path.insert(0, str(Path(__file__).parent.parent))

from src.pipelines import BaselineRAGPipeline
from src.verification.verifier_hub import VerifierHub
from src.verification.rule_based_aggregator import RuleBasedAggregator
from src.ui import ConfidenceUI
from src.utils.config import Config
from src.utils.logger import setup_logger


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


def main():
    """
    Initialize all components and launch the Gradio UI.
    
    This script:
    1. Loads configuration from config.yaml
    2. Initializes RAG pipeline with retriever and generator
    3. Initializes VerifierHub with all detectors
    4. Initializes RuleBasedAggregator for claim classification
    5. Creates and launches the Gradio UI
    """
    print("=" * 80)
    print("HALLUCINATION DETECTION - FULL PIPELINE DEMO")
    print("=" * 80)
    print()
    
    # Step 0: Detect available strategies
    print("🔍 Detecting available FAISS indices...\n")
    available_strategies = detect_available_strategies()
    
    if not available_strategies:
        print("\n❌ ERROR: No FAISS indices found!")
        print("\nPlease ensure you have created indices for at least one strategy:")
        print("  1. python scripts/prepare_wikipedia_chunks.py")
        print("  2. python scripts/generate_embeddings.py")
        print("  3. python scripts/build_faiss_index.py")
        return
    
    # Step 0.5: Ask user to choose strategy if multiple available
    if len(available_strategies) == 1:
        strategy = available_strategies[0]
        print(f"\n✓ Only one strategy available, using: {strategy.upper()}")
    else:
        strategy = ask_user_to_choose_strategy(available_strategies)
    
    # Step 1: Load configuration
    print("\n📋 Loading configuration...")
    try:
        config = Config("config.yaml")
        print("✓ Configuration loaded successfully\n")
        
        if 'aggregator' not in config.get('verification', {}):
            print("\n⚠️  WARNING: verification.aggregator section missing in config.yaml")
            print("   The full pipeline requires aggregator configuration.")
            return
        
        print("✓ Verification configuration validated\n")
    except Exception as e:
        print(f"❌ ERROR loading configuration: {e}")
        print("\nPlease ensure config.yaml exists in the project root.")
        return
    
    # Step 2: Initialize RAG pipeline
    print("🔧 Initializing RAG Pipeline...")
    print("   This includes:")
    print("   - DenseRetriever (loading FAISS index and embeddings)")
    print("   - GeneratorWrapper (loading language model)")
    print("   - VerifierHub (loading all detectors)")
    print()
    
    try:
        pipeline = BaselineRAGPipeline.from_config(
            config_path="config.yaml",
            strategy=strategy
        )
        print("✓ RAG Pipeline initialized successfully\n")
    except FileNotFoundError as e:
        print(f"❌ ERROR: {e}")
        print("\nPlease ensure you have:")
        print("  1. Processed Wikipedia data (scripts/prepare_wikipedia_chunks.py)")
        print("  2. Generated embeddings (scripts/generate_embeddings.py)")
        print("  3. Built FAISS index (scripts/build_faiss_index.py)")
        return
    except Exception as e:
        print(f"❌ ERROR initializing pipeline: {e}")
        import traceback
        traceback.print_exc()
        return
    
    # Step 3: Initialize VerifierHub (if not already initialized in pipeline)
    print("🔍 Initializing VerifierHub...")
    try:
        if not hasattr(pipeline, 'verifier_hub') or pipeline.verifier_hub is None:
            verifier_hub = VerifierHub(config, pipeline.generator)
            print("✓ VerifierHub initialized successfully\n")
        else:
            verifier_hub = pipeline.verifier_hub
            print("✓ Using VerifierHub from pipeline\n")
    except Exception as e:
        print(f"❌ ERROR initializing VerifierHub: {e}")
        import traceback
        traceback.print_exc()
        return
    
    # Step 4: Initialize RuleBasedAggregator
    print("⚙️ Initializing RuleBasedAggregator...")
    try:
        aggregator = RuleBasedAggregator(config)
        print("✓ RuleBasedAggregator initialized successfully\n")
    except Exception as e:
        print(f"❌ ERROR initializing RuleBasedAggregator: {e}")
        import traceback
        traceback.print_exc()
        return
    
    # Step 5: Create ConfidenceUI
    print("🎨 Creating Gradio UI...")
    try:
        ui = ConfidenceUI(
            rag_pipeline=pipeline,
            verifier_hub=verifier_hub,
            aggregator=aggregator
        )
        demo = ui.create_interface()
        print("✓ Gradio UI created successfully\n")
    except Exception as e:
        print(f"❌ ERROR creating UI: {e}")
        import traceback
        traceback.print_exc()
        return
    
    # Step 6: Launch the interface
    print("=" * 80)
    print("🚀 LAUNCHING WEB INTERFACE")
    print("=" * 80)
    print()
    print(f"Data Strategy: {strategy.upper()}")
    print()
    print("The interface will open in your browser at:")
    print("   http://localhost:7860")
    print()
    print("Color Coding:")
    print("   🟢 Green   = Supported (high confidence)")
    print("   🟡 Yellow  = Low Confidence (uncertain)")
    print("   🔴 Red     = Contradictory (conflicts with evidence)")
    print()
    print("Press Ctrl+C to stop the server")
    print("=" * 80)
    print()
    
    try:
        demo.launch(
            share=False,
            server_port=7860,
            server_name="127.0.0.1",
            show_error=True
        )
    except KeyboardInterrupt:
        print("\n\n👋 Shutting down...")
    except Exception as e:
        print(f"\n❌ ERROR launching UI: {e}")
        import traceback
        traceback.print_exc()


if __name__ == "__main__":
    main()
