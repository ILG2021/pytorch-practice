import torch

# x = torch.tensor(1.0, requires_grad=True)
# y = x * x
# y.backward()
# print(x.grad)

x = torch.tensor([1.0, 2.0, 3.0], requires_grad=True)
y = torch.tensor([4.0, 5.0, 6.0], requires_grad=True)
z = x ** 2 + y ** 3
print(z)
z = z.sum()
z.backward()
print(x.grad)
