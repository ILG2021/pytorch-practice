import torch
from torch.functional import F

x = torch.rand(1, 4,4)
print(F.avg_pool2d(x, 2, stride=2))