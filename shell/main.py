import time
import numpy as np
import torch

# 检查CUDA是否可用
device = torch.device('cuda' if torch.cuda.is_available() else 'cpu')
print(f"Using device: {device}")

# 创建随机数组（NumPy）
a_np = np.random.rand(100, 300, 500).astype(np.float32)  # 使用float32加速计算
b_np = np.random.rand(100, 500, 300).astype(np.float32)

# 转换为PyTorch张量并移动到GPU
a = torch.from_numpy(a_np).to(device)
b = torch.from_numpy(b_np).to(device)

# 预热（避免第一次运行的CUDA初始化开销）
_ = torch.matmul(a, b)
torch.cuda.synchronize()  # 确保CUDA操作完成

# 正式计算并计时
start = time.perf_counter()
c = torch.matmul(a, b)  # 或者 c = a @ b
torch.cuda.synchronize()  # 确保CUDA操作完成
end = time.perf_counter()

print(f"结果形状: {c.shape}")  # 输出 (100, 300, 300)

execution_time_ms = (end - start) * 1000
print(f"PyTorch (CUDA) 执行时间: {execution_time_ms:.4f} ms")
