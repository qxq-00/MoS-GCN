import torch

print(torch.__version__)       # PyTorch 版本
print(torch.version.cuda)      # 当前 PyTorch 使用的 CUDA 版本
print(torch.cuda.is_available())  # 检查 CUDA 是否可用
