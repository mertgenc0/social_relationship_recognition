import sys

print("🍎 Mac Installation Test")
print("=" * 60)

# PyTorch
try:
    import torch

    print(f"✅ PyTorch: {torch.__version__}")

    # Mac MPS (Metal Performance Shaders) support check
    if torch.backends.mps.is_available():
        print(f"✅ Apple Silicon GPU (MPS) Available!")
        device = torch.device("mps")
        x = torch.randn(2, 3).to(device)
        print(f"✅ MPS test successful: {x.device}")
    else:
        print(f"⚠️  MPS not available, using CPU")
        device = torch.device("cpu")

except ImportError as e:
    print(f"❌ PyTorch: {e}")
    sys.exit(1)

# Transformers
try:
    import transformers

    print(f"✅ Transformers: {transformers.__version__}")
except ImportError as e:
    print(f"❌ Transformers: {e}")

# OpenCV
try:
    import cv2

    print(f"✅ OpenCV: {cv2.__version__}")
except ImportError as e:
    print(f"❌ OpenCV: {e}")

# NumPy
try:
    import numpy

    print(f"✅ NumPy: {numpy.__version__}")
except ImportError as e:
    print(f"❌ NumPy: {e}")

# Pandas
try:
    import pandas

    print(f"✅ Pandas: {pandas.__version__}")
except ImportError as e:
    print(f"❌ Pandas: {e}")

# Matplotlib
try:
    import matplotlib

    print(f"✅ Matplotlib: {matplotlib.__version__}")
except ImportError as e:
    print(f"❌ Matplotlib: {e}")

# PIL (Pillow)
try:
    from PIL import Image

    print(f"✅ Pillow installed")
except ImportError as e:
    print(f"❌ Pillow: {e}")

print("=" * 60)

# Simple PyTorch test
print("\n🧪 Running PyTorch Operations Test...")
x = torch.randn(3, 4)
y = x * 2 + 1
print(f"✅ CPU Tensor operation: {y.shape}")

# MPS test if available
if torch.backends.mps.is_available():
    x_mps = torch.randn(3, 4, device='mps')
    y_mps = x_mps * 2 + 1
    print(f"✅ MPS (GPU) Tensor operation: {y_mps.shape}")

print("\n🎉 All installations successful!")
print("=" * 60)


"""
Mac Installation Test
============================================================
✅ PyTorch: 2.9.1
✅ Apple Silicon GPU (MPS) Available!
✅ MPS test successful: mps:0
✅ Transformers: 4.57.1
✅ OpenCV: 4.12.0
✅ NumPy: 2.2.6
✅ Pandas: 2.3.3
✅ Matplotlib: 3.10.7
✅ Pillow installed
============================================================

🧪 Running PyTorch Operations Test...
✅ CPU Tensor operation: torch.Size([3, 4])
✅ MPS (GPU) Tensor operation: torch.Size([3, 4])

🎉 All installations successful!
============================================================

Process finished with exit code 0
"""