import sys
import torch
print("python 版本：",sys.version)
print("python 程序位置",sys.executable)
print("pytoch版本",torch.__version__)

gpu_available = torch.cuda.is_available()
print("GPU是否可用：",gpu_available)

gpu_numbers = torch.tensor([1.0,2.0,3.0],device="cuda")
doubled_numbers = gpu_numbers *2
print("gpu计算结果：",doubled_numbers)