import torch

# x = torch.rand(4, 3)
# print(x.view(12))
# print(torch.tensor(5))
# 广播，行列不匹配的时候自动扩展
x = torch.arange(1, 3).view(1, 2)
y = torch.arange(1, 4).view(3, 1)
print(x)
print(y)
print(x + y)
