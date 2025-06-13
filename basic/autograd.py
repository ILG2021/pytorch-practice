import torch

x = torch.tensor([1.0, 2.0], requires_grad=True)
y = x * x
y = y.sum()
y.backward()
print(x.grad)
# print(y.grad)
