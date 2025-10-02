"""
GPU and CUDA Verification Script
This script checks the availability of CUDA and GPU resources for the project.
"""

import sys
import torch

def check_pytorch():
    """Check PyTorch installation and version."""
    print("=" * 60)
    print("PyTorch Information")
    print("=" * 60)
    print(f"PyTorch Version: {torch.__version__}")
    print()

def check_cuda():
    """Check CUDA availability and configuration."""
    print("=" * 60)
    print("CUDA Information")
    print("=" * 60)
    
    cuda_available = torch.cuda.is_available()
    print(f"CUDA Available: {cuda_available}")
    
    if cuda_available:
        print(f"CUDA Version: {torch.version.cuda}")
        print(f"cuDNN Version: {torch.backends.cudnn.version()}")
        print(f"Number of GPUs: {torch.cuda.device_count()}")
        print()
        
        # List all available GPUs
        for i in range(torch.cuda.device_count()):
            print(f"GPU {i}: {torch.cuda.get_device_name(i)}")
            print(f"  Memory Allocated: {torch.cuda.memory_allocated(i) / 1024**3:.2f} GB")
            print(f"  Memory Reserved: {torch.cuda.memory_reserved(i) / 1024**3:.2f} GB")
            print(f"  Total Memory: {torch.cuda.get_device_properties(i).total_memory / 1024**3:.2f} GB")
            print()
    else:
        print("CUDA is NOT available. The model will run on CPU.")
        print("To use GPU acceleration, ensure you have:")
        print("  1. A CUDA-compatible GPU")
        print("  2. NVIDIA drivers installed")
        print("  3. CUDA toolkit installed")
        print("  4. PyTorch with CUDA support installed")
        print()
        print("You can install PyTorch with CUDA support using:")
        print("  pip install torch --index-url https://download.pytorch.org/whl/cu118")
        print()

def test_gpu_computation():
    """Test basic GPU computation."""
    print("=" * 60)
    print("GPU Computation Test")
    print("=" * 60)
    
    if torch.cuda.is_available():
        try:
            # Create a small tensor and move it to GPU
            x = torch.randn(1000, 1000)
            x_gpu = x.cuda()
            
            # Perform a simple computation
            result = torch.matmul(x_gpu, x_gpu)
            
            print("✓ Successfully performed GPU computation")
            print(f"  Tensor shape: {result.shape}")
            print(f"  Device: {result.device}")
            print()
        except Exception as e:
            print(f"✗ GPU computation failed: {str(e)}")
            print()
    else:
        print("Skipping GPU test (CUDA not available)")
        print()

def check_other_libraries():
    """Check other important libraries."""
    print("=" * 60)
    print("Other Libraries")
    print("=" * 60)
    
    try:
        import transformers
        print(f"✓ Transformers: {transformers.__version__}")
    except ImportError:
        print("✗ Transformers: NOT INSTALLED")
    
    try:
        import datasets
        print(f"✓ Datasets: {datasets.__version__}")
    except ImportError:
        print("✗ Datasets: NOT INSTALLED")
    
    try:
        import faiss
        print(f"✓ FAISS: {faiss.__version__}")
    except ImportError:
        print("✗ FAISS: NOT INSTALLED")
    
    try:
        import sentence_transformers
        print(f"✓ Sentence-Transformers: {sentence_transformers.__version__}")
    except ImportError:
        print("✗ Sentence-Transformers: NOT INSTALLED")
    
    print()

def main():
    """Main function to run all checks."""
    print("\n")
    print("=" * 60)
    print("Environment Verification for AIST-FYP")
    print("=" * 60)
    print()
    
    check_pytorch()
    check_cuda()
    test_gpu_computation()
    check_other_libraries()
    
    print("=" * 60)
    print("Verification Complete")
    print("=" * 60)
    print()
    
    # Summary
    if torch.cuda.is_available():
        print("✓ Your environment is ready for GPU-accelerated training!")
    else:
        print("⚠ Your environment will use CPU. Consider setting up CUDA for faster training.")
    print()

if __name__ == "__main__":
    main()
