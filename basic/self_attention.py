import math
import torch
from torch import nn


class SelfAttention(nn.Module):
    def __init__(self, hidden_dim, *args, **kwargs):
        super().__init__(*args, **kwargs)
        self.hidden_dim = hidden_dim
        self.Q = nn.Linear(hidden_dim, hidden_dim)
        self.K = nn.Linear(hidden_dim, hidden_dim)
        self.V = nn.Linear(hidden_dim, hidden_dim)

    def forward(self, x):  # batch, seq_len, dim
        q = self.Q(x)
        k = self.K(x)
        v = self.V(x)
        attention_value = torch.matmul(q, k.transpose(-1, -2))
        attention_weight = torch.softmax(attention_value / math.sqrt(self.hidden_dim), dim=-1)
        print(attention_weight)
        output = torch.matmul(attention_weight, v)
        return output


if __name__ == '__main__':
    x = torch.randn(3, 2, 4)
    self_attention = SelfAttention(hidden_dim=4)
    print(self_attention(x))
