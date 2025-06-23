import torch
from torch import tensor

from asr.model import TranscribeModel

# x = torch.tensor([[1, 1, 1, 1, 1],
#                   [2, 2, 2, 2, 2],
#                   [3, 3, 3, 3, 3]])
# print(x.transpose(-2, -1))

# x = tensor([[0.1000, 0.2000, 0.3000],
#             [0.4000, 0.5000, 0.6000]])
# y = tensor([[0.0832, -0.0327, 0.0692],
#             [-0.0873, 0.0568, -0.0210],
#             [0.0933, 0.0343, 0.0597],
#             [-0.0792, -0.0680, 0.0640],
#             [0.0540, 0.0914, -0.0955]])
# distances = torch.cdist(x, y, p=2)
# print(distances)
# print(torch.argmin(distances, dim=1))

model = TranscribeModel(num_codebooks=3,
                        codebook_size=64,
                        embedding_dim=64,
                        vocab_size=30,
                        strides=[6,8,4,2],
                        max_seq_length=2000,
                        num_transformer_layers=2,
                        initial_mean_pooling_kernel_size=4)
x = torch.randn(4, 237680)
out, loss = model(x)
print(out.shape)
