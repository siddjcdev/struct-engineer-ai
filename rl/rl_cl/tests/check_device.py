"""Check available compute devices"""
import torch

print("PyTorch version:", torch.__version__)
print("\nAvailable devices:")
print(f"  CUDA available: {torch.cuda.is_available()}")
print(f"  MPS available: {torch.backends.mps.is_available()}")
print(f"  CPU: Always available")

if torch.backends.mps.is_available():
    print("\n✅ Apple Silicon GPU (MPS) detected!")
    print("   Recommended device: 'mps'")
elif torch.cuda.is_available():
    print("\n✅ NVIDIA GPU (CUDA) detected!")
    print("   Recommended device: 'cuda'")
else:
    print("\n⚠️  No GPU detected, will use CPU")
    print("   Device: 'cpu'")

# Test actual device selection
print(f"\n'auto' resolves to: {torch.device('cuda' if torch.cuda.is_available() else 'cpu')}")
print("Note: 'auto' doesn't detect MPS on Apple Silicon - must specify 'mps' explicitly")
