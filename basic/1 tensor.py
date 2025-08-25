import torch

# x = torch.rand(4, 3)
# print(x.view(12))
# print(torch.tensor(5))
# 广播，行列不匹配的时候自动扩展
# x = torch.arange(1, 3).view(1, 2)
# y = torch.arange(1, 4).view(3, 1)
# print(x)
# print(y)
# print(x + y)

# print(torch.ones(1,3))
# a = torch.tensor([1, 2])
# b = torch.tensor([3, 4])
# print(a * b)
# a = torch.randn(3, 4)
# b = torch.randn(3, 4)
# print(a.T)
# print(a@b.T)
# 广播
# a = torch.tensor([[1],
#                   [2],
#                   [3]])
# b = torch.tensor([1, 2, 3])
# print(a + b)
# batch_size = 10 10个矩阵相乘
# a = torch.randn(batch_size, 3, 4)
# b = torch.randn(batch_size, 4, 5)
# c = a@b
# print(c.shape)
# # d的第一维 batch size会传播
# d = torch.randn(4,5)
# e = a@d
# print(e.shape)
a = torch.randn(2, 3, 4)
a_flat = a.reshape(-1)
print(a_flat.shape)
# a_transpose_1 = a.transpose(0, 1)
# a_transpose_2 = a.transpose(1, 2)
# print(a_transpose_1.shape)
# print(a_transpose_2.shape)
# a_permute = a.permute(1, 2, 0)
# print(a_permute.shape)

# a = torch.ones(1,5,5).to(torch.int32)
# b = torch.ones(1,5).to(torch.int32)
# print(a & b)
