import platform
import sys
from pathlib import Path

import torch


def bytes_to_gb(num_bytes: int) -> float:
    return num_bytes / 1024**3


print("=" * 50)
print("TokenScope-ViT Environment Check")
print("=" * 50)

print("Working directory:", Path.cwd())
print("Operating system:", platform.platform())
print("Python executable:", sys.executable)
print("Python version:", sys.version.split()[0])
print("PyTorch version:", torch.__version__)
print("CUDA available:", torch.cuda.is_available())
print("PyTorch CUDA version:", torch.version.cuda)

if torch.cuda.is_available():
    device = torch.device("cuda")
    properties = torch.cuda.get_device_properties(0)

    print("GPU name:", properties.name)
    print("GPU memory: {:.2f} GB".format(
        bytes_to_gb(properties.total_memory)
    ))
    print("CUDA capability:", properties.major, properties.minor)
    print("cuDNN version:", torch.backends.cudnn.version())
else:
    device = torch.device("cpu")
    print("GPU is unavailable; using CPU.")

# 创建两个矩阵并执行一次矩阵乘法，
# 验证 PyTorch 不仅能识别 GPU，还能真正执行 CUDA 运算。
x = torch.randn(512, 512, device=device)
y = torch.randn(512, 512, device=device)
z = x @ y

print("-" * 50)
print("Input shape:", x.shape)
print("Output shape:", z.shape)
print("Output device:", z.device)
print("All values finite:", torch.isfinite(z).all().item())
print("Environment check passed.")