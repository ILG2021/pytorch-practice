import math

import torch
from torch import nn


class MultiHeadAttention(nn.Module):
    def __init__(self, hidden, num_heads):
        super().__init__()
        self.hidden = hidden
        self.num_heads = num_heads
        self.head_dim = hidden // num_heads
        self.q_proj = nn.Linear(hidden, hidden)
        self.k_proj = nn.Linear(hidden, hidden)
        self.v_proj = nn.Linear(hidden, hidden)
        self.out_proj = nn.Linear(hidden, hidden)
        self.dropout = nn.Dropout(0.1)

    def forward(self, x, attention_mask):
        batch_size, seq_len, _ = x.shape
        q = self.q_proj(x)
        k = self.k_proj(x)
        v = self.v_proj(x)
        # b hn s hd
        q_state = q.view(batch_size, seq_len, self.num_heads, self.head_dim).transpose(1, 2)
        k_state = k.view(batch_size, seq_len, self.num_heads, self.head_dim).transpose(1, 2)
        v_state = v.view(batch_size, seq_len, self.num_heads, self.head_dim).transpose(1, 2)
        attention_weight = torch.matmul(q_state, k_state.transpose(-1, -2)) / math.sqrt(self.head_dim)
        if attention_mask is not None:
            attention_weight = torch.masked_fill(attention_weight, attention_mask == 0, value=float('-inf'))
        attention_weight = torch.softmax(attention_weight, -1)
        attention_weight = self.dropout(attention_weight)
        output = torch.matmul(attention_weight, v_state).transpose(1, 2).reshape(batch_size, seq_len, self.hidden)
        output = self.out_proj(output)
        return output


if __name__ == '__main__':
    x = torch.randn(2, 3, 16)
    attention = MultiHeadAttention(16, 4)
    attention_mask = torch.tensor([
        [1, 0, 0],
        [1, 1, 0]
    ])
    # b hn s s
    attention_mask = attention_mask.unsqueeze(1).unsqueeze(2)
    output = attention(x, attention_mask)
    print(output)
