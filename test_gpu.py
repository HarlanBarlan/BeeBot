"""Smoke test: is PyTorch installed correctly and does CUDA work?"""
import torch

print(f"PyTorch version: {torch.__version__}")
print(f"CUDA available:  {torch.cuda.is_available()}")
if torch.cuda.is_available():
    print(f"CUDA version:    {torch.version.cuda}")
    print(f"Device count:    {torch.cuda.device_count()}")
    print(f"Device name:     {torch.cuda.get_device_name(0)}")
    # Actually run a tiny op on the GPU to confirm it works
    x = torch.randn(1024, 1024, device="cuda")
    y = x @ x
    print(f"Matmul on GPU:   ok, output sum = {y.sum().item():.2f}")
else:
    print("!! CUDA not available — training will fall back to CPU (~10-50x slower)")
