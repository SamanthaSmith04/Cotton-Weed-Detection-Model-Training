# Import required packages
import torch
import tlc
from pathlib import Path
import pandas as pd
from IPython.display import display

# Check environment
print("Environment Check:")
print("=" * 50)
print(f"PyTorch version: {torch.__version__}")
print(f"3LC version: {tlc.__version__}")
print(f"CUDA available: {torch.cuda.is_available()}")

if torch.cuda.is_available():
    print(f"GPU: {torch.cuda.get_device_name(0)}")
    print(
        f"GPU Memory: {torch.cuda.get_device_properties(0).total_memory / 1024**3:.1f} GB"
    )
else:
    print("!!! No GPU detected - training will be slower on CPU")

print("\n All systems ready! Let's begin.")