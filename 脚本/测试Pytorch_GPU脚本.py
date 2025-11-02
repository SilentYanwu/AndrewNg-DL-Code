import torch
import subprocess
import sys

print("=== 环境检查 ===")
print("Python版本:", sys.version)
print("PyTorch版本:", torch.__version__)
print("CUDA可用:", torch.cuda.is_available())

if torch.cuda.is_available():
    print("CUDA版本:", torch.version.cuda)
    print("GPU设备:", torch.cuda.get_device_name(0))
    print("GPU数量:", torch.cuda.device_count())
    
    # 测试 GPU 计算
    device = torch.device("cuda")
    x = torch.randn(1000, 1000).to(device)
    y = torch.randn(1000, 1000).to(device)
    z = torch.matmul(x, y)
    print("GPU计算测试成功!")
else:
    print("❌ 仍然无法使用GPU")