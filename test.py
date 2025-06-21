import torch
from torch import nn

conv = nn.Conv2d(1, 16, 7, 7)
x = torch.randn(1, 1, 28, 28)
print(conv(x).shape)
