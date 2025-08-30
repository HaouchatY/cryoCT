import time
import torch
import torch.nn.functional as F

device = 'cuda:0'
conv = torch.nn.Conv3d(64, 64, 13, padding=6, bias=False).to(device)
weight = conv.weight

start = time.time()
for k in range(10):
    test = torch.randn(1, 64, 128, 128, 128, device=device)
    y = F.conv3d(test, weight=weight, padding=6)
print(time.time() - start)