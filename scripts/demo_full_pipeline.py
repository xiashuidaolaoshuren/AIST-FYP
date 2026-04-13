"""
Demo script for Full Pipeline with Gradio UI.

Launches the complete hallucination detection system with a web interface
for interactive visualization of claim verification results.
Supports multiple data strategies (development, validation, production).
"""

import argparse
import os
import sys
import json
from datetime import datetime
from pathlib import Path
from tqdm import tqdm

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
    def __init__(self, ui, log_filepath: str):
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


def is_running_in_colab() -> bool:
    """Return True when running inside a Google Colab runtime."""
    runtime_hint = (os.getenv("AIST_RUNTIME") or "").strip().lower()
    if runtime_hint == "colab":
        return True

    try:
        import google.colab  # type: ignore  # noqa: F401
        return True
    except Exception:
        return False


def parse_args() -> argparse.Namespace:
    """Parse CLI args while tolerating unknown notebook-injected flags."""
    parser = argparse.ArgumentParser(
        description="Launch the full hallucination-detection demo UI."
    )
    parser.add_argument(
        "--config",
        default="config.yaml",
        help="Path to runtime config YAML (default: config.yaml).",
    )
    parser.add_argument(
        "--strategy",
        choices=["development", "validation", "production"],
        help="Data strategy to use. If omitted, script auto-selects.",
    )
    parser.add_argument(
        "--share",
        action="store_true",
        help="Enable Gradio public share link (recommended for Colab).",
    )
    parser.add_argument(
        "--server-name",
        default=None,
        help="Gradio server_name. Defaults to 127.0.0.1 locally, 0.0.0.0 on Colab.",
    )
    parser.add_argument(
        "--server-port",
        type=int,
        default=7860,
        help="Gradio server port (default: 7860).",
    )
    parser.add_argument(
        "--force-non-interactive",
        action="store_true",
        help="Disable terminal prompts and auto-pick the first available strategy.",
    )
    parser.add_argument(
        "--ui-mode",
        choices=["controlled", "legacy"],
        default="controlled",
        help="UI mode: controlled (generate/edit/verify flow) or legacy (single-step).",
    )
    args, _ = parser.parse_known_args()
    return args


def main():
    """
    Initialize all components and launch the Gradio UI.
    
    This script:
    1. Loads configuration from the configured YAML path
    2. Initializes RAG pipeline with retriever and generator
    3. Initializes VerifierHub with all detectors
    4. Initializes RuleBasedAggregator for claim classification
    5. Creates and launches the Gradio UI
    """
    args = parse_args()
    in_colab = is_running_in_colab()

    # Default to DEBUG when unset so claim-level diagnostics surface in streamed stdout.
    if not os.getenv("AIST_STDOUT_LOG_LEVEL"):
        os.environ["AIST_STDOUT_LOG_LEVEL"] = "DEBUG"

    controlled_ui_available = True

    try:
        from src.ui import ControlledPipelineUI
    except Exception as exc:
        controlled_ui_available = False
        if args.ui_mode == "controlled":
            print("⚠️  Controlled UI dependencies are unavailable; falling back to legacy UI mode.")
            print(f"   Root cause: {exc}")
            args.ui_mode = "legacy"

    print("=" * 80)
    print("HALLUCINATION DETECTION - FULL PIPELINE DEMO")
    print("=" * 80)
    print()
    print(f"Runtime detected as Colab: {in_colab}")
    print(f"AIST_STDOUT_LOG_LEVEL: {os.getenv('AIST_STDOUT_LOG_LEVEL', 'unset')}")
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
    non_interactive_mode = in_colab or args.force_non_interactive or (not sys.stdin.isatty())

    if args.strategy:
        if args.strategy not in available_strategies:
            print(
                f"\n❌ ERROR: Requested strategy '{args.strategy}' is not available in this environment."
            )
            print(f"Available strategies: {', '.join(available_strategies)}")
            return
        strategy = args.strategy
        print(f"\n✓ Strategy provided by CLI: {strategy.upper()}")
    elif len(available_strategies) == 1:
        strategy = available_strategies[0]
        print(f"\n✓ Only one strategy available, using: {strategy.upper()}")
    elif non_interactive_mode:
        strategy = available_strategies[0]
        print(
            "\n✓ Non-interactive mode detected; "
            f"auto-selected first available strategy: {strategy.upper()}"
        )
    else:
        strategy = ask_user_to_choose_strategy(available_strategies)
    
    # Step 1: Load configuration
    setup_steps = [
        "Loading configuration",
        "Initializing RAG Pipeline",
        "Initializing VerifierHub",
        "Initializing RuleBasedAggregator",
        "Setting up RePrompter",
        "Creating Gradio UI",
    ]
    pbar = tqdm(total=len(setup_steps), desc="🔧 Setup progress", ascii=False, unit="step")
    
    try:
        config = Config(args.config)
        pbar.update(1)
        pbar.set_description_str("🔧 Setup progress | ✓ Config loaded")
        
        if 'aggregator' not in config.get('verification', {}):
            print(f"\n⚠️  WARNING: verification.aggregator section missing in {args.config}")
            print("   The full pipeline requires aggregator configuration.")
            pbar.close()
            return
    except Exception as e:
        pbar.close()
        print(f"❌ ERROR loading configuration: {e}")
        print(f"\nPlease ensure {args.config} exists and is readable.")
        return
    
    # Step 2: Initialize RAG pipeline
    pbar.set_description_str("🔧 Setup progress | Loading RAG Pipeline")
    try:
        pipeline = BaselineRAGPipeline.from_config(
            config_path=args.config,
            strategy=strategy
        )
        pbar.update(1)
        pbar.set_description_str("🔧 Setup progress | ✓ RAG Pipeline loaded")
        generator = getattr(pipeline, "generator", None)
        if generator is not None and getattr(generator, "model_family", None) == "seq2seq":
            print("\n⚠️  GENERATOR WARNING: Seq2Seq generator detected.")
            print(
                f"   model={getattr(generator, 'model_name', 'unknown')} | "
                "Seq2Seq models may ignore chat-style system instructions and produce short QA outputs."
            )
            print("   For full-sentence reprompt testing, use an instruction-tuned causal model (for example Qwen3).")
    except FileNotFoundError as e:
        pbar.close()
        print(f"❌ ERROR: {e}")
        print("\nPlease ensure you have:")
        print("  1. Processed Wikipedia data (scripts/prepare_wikipedia_chunks.py)")
        print("  2. Generated embeddings (scripts/generate_embeddings.py)")
        print("  3. Built FAISS index (scripts/build_faiss_index.py)")
        return
    except Exception as e:
        pbar.close()
        print(f"❌ ERROR initializing pipeline: {e}")
        import traceback
        traceback.print_exc()
        return
    
    # Step 3: Initialize VerifierHub (if not already initialized in pipeline)
    pbar.set_description_str("🔧 Setup progress | Loading VerifierHub")
    try:
        if not hasattr(pipeline, 'verifier_hub') or pipeline.verifier_hub is None:
            verifier_hub = VerifierHub(config, pipeline.generator)
        else:
            verifier_hub = pipeline.verifier_hub
        pbar.update(1)
        pbar.set_description_str("🔧 Setup progress | ✓ VerifierHub loaded")
    except Exception as e:
        pbar.close()
        print(f"❌ ERROR initializing VerifierHub: {e}")
        import traceback
        traceback.print_exc()
        return
    
    # Step 4: Initialize RuleBasedAggregator
    pbar.set_description_str("🔧 Setup progress | Setting up Aggregator")
    try:
        aggregator = RuleBasedAggregator(config)
        pbar.update(1)
        pbar.set_description_str("🔧 Setup progress | ✓ Aggregator loaded")
    except Exception as e:
        pbar.close()
        print(f"❌ ERROR initializing RuleBasedAggregator: {e}")
        import traceback
        traceback.print_exc()
        return
    
    # Step 4.5: Initialize RePrompter (if enabled in config)
    pbar.set_description_str("🔧 Setup progress | Setting up RePrompter")
    repromptr = None
    reprompt_config = config.get('mitigation', {}).get('reprompt', {})
    if reprompt_config.get('enabled', False):
        try:
            repromptr = RePrompter(config, pipeline.generator)
            pbar.update(1)
            pbar.set_description_str("🔧 Setup progress | ✓ RePrompter loaded")
        except Exception as e:
            pbar.update(1)
            print(f"⚠️  WARNING: Could not initialize RePrompter: {e}")
            print("   Continuing without re-prompting...")
            repromptr = None
    else:
        pbar.update(1)
        pbar.set_description_str("🔧 Setup progress | ℹ️  RePrompter skipped")
    
    # Step 5: Create selected UI mode with logging
    pbar.set_description_str("🔧 Setup progress | Creating Gradio UI")
    try:
        if args.ui_mode == "legacy":
            ui = ConfidenceUI(
                rag_pipeline=pipeline,
                verifier_hub=verifier_hub,
                aggregator=aggregator,
                repromptr=repromptr
            )
        else:
            if not controlled_ui_available:
                raise RuntimeError("Controlled UI requested but unavailable in this environment")
            ui = ControlledPipelineUI(
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
        
        pbar.update(1)
        pbar.close()
        print("\n" + "=" * 80)
        print(f"✓ Gradio UI created successfully (mode={args.ui_mode})")
        print(f"✓ Query logging enabled -> {log_filepath}")
        print("=" * 80 + "\n")
    except Exception as e:
        pbar.close()
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
    launch_share = args.share or in_colab
    launch_server_name = args.server_name or ("0.0.0.0" if in_colab else "127.0.0.1")

    if in_colab:
        print("Runtime: Google Colab (share link enabled by default)")
    else:
        print("Runtime: Local")

    print(f"Gradio Launch Settings: share={launch_share}, server_name={launch_server_name}, server_port={args.server_port}")
    print()
    if launch_share:
        print("The interface will print a public Gradio URL after launch.")
    else:
        print("The interface will open in your browser at:")
        print(f"   http://{launch_server_name}:{args.server_port}")
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
            share=launch_share,
            server_port=args.server_port,
            server_name=launch_server_name,
            show_error=True,
            inline=in_colab,
            inbrowser=(not in_colab),
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
