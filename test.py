import torch
import time

size = 1024 * 1024 * 256

x = torch.empty(
    size,
    dtype=torch.float32,
    pin_memory=True
)

# 预热
for _ in range(3):
    y = x.cuda()
    torch.cuda.synchronize()

times = []

for _ in range(10):
    torch.cuda.synchronize()
    t0 = time.perf_counter()

    y = x.cuda()

    torch.cuda.synchronize()
    t1 = time.perf_counter()

    times.append(t1 - t0)

size_gb = x.numel() * x.element_size() / 1024**3

avg = sum(times) / len(times)

print("平均时间:", avg, "s")
print("平均带宽:", size_gb / avg, "GB/s")