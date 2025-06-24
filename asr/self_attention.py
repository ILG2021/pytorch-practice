import math
import torch
import torch.nn.functional as F
from torch import nn


def calculate_attention(query: torch.Tensor, keys: torch.Tensor, values: torch.Tensor, mask: torch.Tensor = None):
    attention_scores = torch.matmul(query, keys.transpose(-2, -1))  # Q * Kt
    attention_scores = attention_scores / math.sqrt(keys.shape[-1])  # 最后一维是embedding维度
    if mask is not None:
        attention_scores = torch.where(mask == 0, torch.tensor(-1e9), attention_scores)
    attention_scores = F.softmax(attention_scores, dim=-1)
    attention = torch.matmul(attention_scores, values)
    return attention, attention_scores


class AttentionLayer(nn.Module):
    def __init__(self, embed_size: int):
        super().__init__()
        self.embed_size = embed_size
        self.key_dense = nn.Linear(embed_size, embed_size)
        self.query_dense = nn.Linear(embed_size, embed_size)
        self.value_dense = nn.Linear(embed_size, embed_size)

    # embedding batch, seq_len, dim
    def forward(self, embeddings: torch.Tensor, mask):
        seq_len = embeddings.shape[1]
        query = self.query_dense(embeddings)  # 自注意力，如果query是别的embedding，是交叉注意力
        key = self.key_dense(embeddings)
        value = self.value_dense(embeddings)
        if mask:
            right_triangular_mask = torch.tril(torch.ones(1, seq_len, seq_len)).to(embeddings.device)
        else:
            right_triangular_mask = None
        attention, attention_scores = calculate_attention(query, key, value, right_triangular_mask)
        return attention, attention_scores


class FeedForward(nn.Module):
    def __init__(self, embed_size):
        super().__init__()
        self.layer1 = nn.Linear(embed_size, embed_size)
        self.layer2 = nn.Linear(embed_size, embed_size)

    def forward(self, x):
        x = self.layer1(x)
        x = F.gelu(x)
        x = self.layer2(x)
        return x


class TransformerBlock(nn.Module):
    def __init__(self, embed_size):
        super().__init__()
        self.attention_layer = AttentionLayer(embed_size)
        self.norm1 = nn.LayerNorm(embed_size)
        self.norm2 = nn.LayerNorm(embed_size)
        self.feed_forward = FeedForward(embed_size)

    def forward(self, x, mask):
        # Pre-LN Attention
        norm_x = self.norm1(x)
        attention_output, attention_scores = self.attention_layer(norm_x, mask)
        x = x + attention_output  # Add

        # Pre-LN FeedForward
        norm_x = self.norm2(x)
        ff_output = self.feed_forward(norm_x)
        output = x + ff_output  # Add

        return output, attention_scores


class Transformer(nn.Module):
    def __init__(self, embed_size, num_layers):
        super().__init__()
        self.transformer_blocks = nn.ModuleList([TransformerBlock(embed_size) for _ in range(num_layers)])

    def forward(self, x, mask=False):
        attention_scores = []
        for transformer_block in self.transformer_blocks:
            x, score = transformer_block(x, mask)
            attention_scores.append(score)
        return x, attention_scores


class SinPositionEncoding(nn.Module):
    def __init__(self, embed_size, max_seq_len):
        super().__init__()
        position = torch.arange(max_seq_len).unsqueeze(1)
        div_term = torch.exp(torch.arange(0, embed_size, 2) * (-math.log(10000.0) / embed_size))
        pe = torch.zeros(max_seq_len, embed_size)
        pe[:, 0::2] = torch.sin(position * div_term)
        pe[:, 1::2] = torch.cos(position * div_term)
        self.register_buffer("positional_embedding", pe)

    def forward(self, x):
        return x + self.positional_embedding[:x.size(1), :]