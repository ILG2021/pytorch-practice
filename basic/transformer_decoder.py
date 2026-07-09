import math
from torch import nn
import torch

class DecoderLayer(nn.Module):
    def __init__(self, hidden, num_heads, dropout_rate = 0.1):
        super().__init__()
        self.hidden = hidden  # 修复 1：保存 hidden 维度
        # mha
        self.q_proj = nn.Linear(hidden, hidden)
        self.k_proj = nn.Linear(hidden, hidden)
        self.v_proj = nn.Linear(hidden, hidden)
        self.dropout = nn.Dropout(dropout_rate)
        self.out_proj = nn.Linear(hidden, hidden)
        self.attn_ln = nn.LayerNorm(hidden, eps=1e-6)
        self.head_dim = hidden // num_heads
        self.num_heads = num_heads
        
        # ffn
        self.up_proj = nn.Linear(hidden, hidden * 4)
        self.act_fn = nn.GELU()
        self.ffn_ln = nn.LayerNorm(hidden, eps=1e-6)
        self.down_proj = nn.Linear(hidden * 4, hidden)
        self.ffn_dropout = nn.Dropout(dropout_rate)

    def mha(self, x, attention_mask):
        q = self.q_proj(x)
        k = self.k_proj(x)
        v = self.v_proj(x)
        batch_size, seq_len, _ = x.shape
        q = q.view(batch_size, seq_len, self.num_heads, self.head_dim).transpose(1, 2)
        k = k.view(batch_size, seq_len, self.num_heads, self.head_dim).transpose(1, 2)
        v = v.view(batch_size, seq_len, self.num_heads, self.head_dim).transpose(1, 2)
        attention_weight = torch.matmul(q, k.transpose(-1, -2)) / math.sqrt(self.head_dim)
        if attention_mask is not None:
            attention_mask = attention_mask.tril()
        else:
            attention_mask = torch.ones_like(attention_weight).tril()
        attention_weight = torch.masked_fill(attention_weight, attention_mask == 0, value=float('-inf'))
        attention_weight = torch.softmax(attention_weight, -1)
        attention_weight = self.dropout(attention_weight)
        output = torch.matmul(attention_weight, v).transpose(1, 2).reshape(batch_size, seq_len, self.hidden)
        output = self.out_proj(output)
        return self.attn_ln(x + output)

    def ffn(self, x):
        output = self.up_proj(x)
        output = self.act_fn(output)
        output = self.down_proj(output)
        output = self.ffn_dropout(output)
        return self.ffn_ln(x + output)


    def forward(self, x, attention_mask):
        x = self.mha(x, attention_mask)
        x = self.ffn(x)
        return x

class Decoder(nn.Module):
    def __init__(self):
        super().__init__()
        self.emb= nn.Embedding(12, 64)
        self.out_proj = nn.Linear(64, 12)
        self.decoder_layers = nn.ModuleList([DecoderLayer(64, 8) for _ in range(5)])

    def forward(self, x, attention_mask):
        x = self.emb(x)
        for decoder_layer in self.decoder_layers:
            x = decoder_layer(x, attention_mask)
        x = self.out_proj(x)
        return torch.softmax(x, dim=-1)

x = torch.randint(0, 12, (3, 4))
attn_mask = torch.tensor([
    [1,1,1,0],
    [1,1,0,0],
    [1,0,0,0]
])
attn_mask = attn_mask.unsqueeze(1).unsqueeze(2).repeat(1, 8, 4, 1) # b 1 1 seq_len
decoder = Decoder()
print(decoder(x, attn_mask).shape)