"""
Demo script for Full Pipeline with Gradio UI.

Launches the complete hallucination detection system with a web interface
for interactive visualization of claim verification results.
Supports multiple data strategies (development, validation, production).
"""

import sys
import json
from datetime import datetime
from pathlib import Path

sys.path.insert(0, str(Path(__file__).parent.parent))

from src.pipelines import BaselineRAGPipeline
from src.verification.verifier_hub import VerifierHub
from src.verification.rule_based_aggregator import RuleBasedAggregator
from src.mitigation.reprompt import RePrompter
from src.ui import ConfidenceUI
from src.utils.config import Config
from src.utils.logger import setup_logger


class LoggingUIWrapper:
    """
    Wrapper around ConfidenceUI to log all queries and results to JSON.
    """
    def __init__(self, ui: ConfidenceUI, log_filepath: str):
        self.ui = ui
        self.log_filepath = log_filepath
        self.query_logs = []
        
    def create_interface(self):
        """Create Gradio interface with logging wrapper."""
        demo = self.ui.create_interface()
        
        # Wrap the process_query function to intercept calls
        original_fn = demo.fns[0].fn  # Get the original process_query function
        
        def logged_process_query(query, *args, **kwargs):
            # Call original function
            result = original_fn(query, *args, **kwargs)
            
            # Log the query and metadata only (exclude HTML output)
            log_entry = {
                "timestamp": datetime.now().isoformat(),
                "query": query,
                "metadata": result[1] if len(result) > 1 else None
            }
            self.query_logs.append(log_entry)
            
            return result
        
        # Replace the function
        demo.fns[0].fn = logged_process_query
        return demo
    
    def _serialize_obj(self, obj):
        """Convert non-serializable objects to serializable format."""
        import pandas as pd
        
        if isinstance(obj, pd.DataFrame):
            return obj.to_dict(orient='records')
        elif isinstance(obj, pd.Series):
            return obj.to_dict()
        elif hasattr(obj, '__dict__'):
            return str(obj)
        elif isinstance(obj, dict):
            return {k: self._serialize_obj(v) for k, v in obj.items()}
        elif isinstance(obj, (list, tuple)):
            return [self._serialize_obj(item) for item in obj]
        else:
            return obj
    
    def save_logs(self):
        """Save accumulated logs to JSON file."""
        try:
            output_dir = Path("outputs")
            output_dir.mkdir(exist_ok=True)
            
            # Serialize all objects in logs to JSON-compatible format
            serialized_logs = []
            for log_entry in self.query_logs:
                serialized_entry = {
                    "timestamp": log_entry["timestamp"],
                    "query": log_entry["query"],
                    "metadata": self._serialize_obj(log_entry["metadata"])
                }
                serialized_logs.append(serialized_entry)
            
            with open(self.log_filepath, 'w', encoding='utf-8') as f:
                json.dump(serialized_logs, f, indent=2, ensure_ascii=False)
            
            print(f"\n✓ Query logs saved to: {self.log_filepath}")
            print(f"   Total queries logged: {len(self.query_logs)}")
        except Exception as e:
            print(f"\n⚠️  Warning: Could not save logs: {e}")


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
    
    # Step 4.5: Initialize RePrompter (if enabled in config)
    repromptr = None
    reprompt_config = config.get('mitigation', {}).get('reprompt', {})
    if reprompt_config.get('enabled', False):
        print("🔄 Initializing RePrompter for hallucination mitigation...")
        try:
            repromptr = RePrompter(config, pipeline.generator)
            print("✓ RePrompter initialized successfully")
            print(f"   - Threshold: {repromptr.threshold:.2%}")
            print(f"   - Max iterations: {repromptr.max_iterations}")
            print(f"   - Strategy: {repromptr.strategy}\n")
        except Exception as e:
            print(f"⚠️  WARNING: Could not initialize RePrompter: {e}")
            print("   Continuing without re-prompting...\n")
            repromptr = None
    else:
        print("ℹ️  Re-prompting disabled in config (mitigation.reprompt.enabled: false)\n")
    
    # Step 5: Create ConfidenceUI with logging
    print("🎨 Creating Gradio UI with logging...")
    try:
        ui = ConfidenceUI(
            rag_pipeline=pipeline,
            verifier_hub=verifier_hub,
            aggregator=aggregator,
            repromptr=repromptr
        )
        
        # Wrap UI with logging functionality
        timestamp = datetime.now().strftime("%Y%m%d_%H%M%S")
        log_filepath = f"outputs/full_pipeline_queries_{timestamp}.json"
        logging_ui = LoggingUIWrapper(ui, log_filepath)
        demo = logging_ui.create_interface()
        
        print("✓ Gradio UI created successfully")
        print(f"✓ Query logging enabled -> {log_filepath}\n")
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
        logging_ui.save_logs()
    except Exception as e:
        print(f"\n❌ ERROR launching UI: {e}")
        import traceback
        traceback.print_exc()
    finally:
        # Save logs on any exit
        if 'logging_ui' in locals():
            logging_ui.save_logs()


if __name__ == "__main__":
    main()
