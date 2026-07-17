import torch
from torch import nn
import torch.nn.functional as F
import math

class MultiHeadAttention(nn.Module):
    def __init__(self, hidden_dim, heads, dropout_rate = 0.1):
        super().__init__()
        self.hidden_dim = hidden_dim
        self.heads = heads
        self.head_dim = hidden_dim // heads
        self.dropout_rate = dropout_rate
        
        self.q_proj = nn.Linear(hidden_dim, hidden_dim)
        self.k_proj = nn.Linear(hidden_dim, hidden_dim)
        self.v_proj = nn.Linear(hidden_dim, hidden_dim)
        self.dropout = nn.Dropout(dropout_rate)
        self.out_proj = nn.Linear(hidden_dim, hidden_dim)

    def forward(self, x, attention_mask):
        batch_size, seq_len, _ = x.shape
        q = self.q_proj(x)
        k = self.k_proj(x)
        v = self.v_proj(x)
        q = q.view(batch_size, seq_len, self.heads, self.head_dim).transpose(1, 2)
        k = k.view(batch_size, seq_len, self.heads, self.head_dim).transpose(1, 2)
        v = v.view(batch_size, seq_len, self.heads, self.head_dim).transpose(1, 2)

        attn_weight = q@k.transpose(-1,-2)/math.sqrt(self.head_dim)

        if attention_mask is not None:
            attn_weight = attn_weight.masked_fill(attention_mask == 0, float('-inf'))
        attn_weight = torch.softmax(attn_weight, dim=-1)
        attn_weight = self.dropout(attn_weight)
        out = attn_weight@v
        out = out.transpose(1, 2).contiguous().view(batch_size, seq_len, self.hidden_dim)
        out = self.out_proj(out)
        return out
        
class MLP(nn.Module):
    def __init__(self,hidden_dim):
        super().__init__()
        self.hidden_dim = hidden_dim
        self.ffn = nn.Sequential(
            nn.Linear(hidden_dim, hidden_dim*4),
            nn.GELU(),
            nn.Linear(hidden_dim*4, hidden_dim),
        )

    def forward(self, x):
        return self.ffn(x)

class TransformerBlock(nn.Module):
    def __init__(self,hidden_dim, heads):
        super().__init__()
        self.mha = MultiHeadAttention(hidden_dim, heads)
        self.layernorm1 = nn.LayerNorm(hidden_dim)
        self.layernorm2 = nn.LayerNorm(hidden_dim)
        self.mlp = MLP(hidden_dim)
    
    def forward(self, x, attention_mask):
        x = x + self.mha(self.layernorm1(x), attention_mask)
        x = x + self.mlp(self.layernorm2(x))
        return x

class GPT2(nn.Module):
    def __init__(self, vocab_size=50257, hidden_dim=768, heads=12, layers=12, max_seq_length=512, dropout_rate = 0.1):
        super().__init__()
        self.max_seq_length = max_seq_length
        self.token_embedding = nn.Embedding(vocab_size, hidden_dim)
        self.position_embedding = nn.Embedding(max_seq_length, hidden_dim)
        self.dropout = nn.Dropout(dropout_rate)
        self.layers = nn.ModuleList([TransformerBlock(hidden_dim, heads) for _ in range(layers)])
        self.layernorm = nn.LayerNorm(hidden_dim)
        self.lm_head = nn.Linear(hidden_dim, vocab_size, bias = False)
        # tie weights between token_embedding and lm_head
        self.lm_head.weight = self.token_embedding.weight
        self.register_buffer('attention_mask', torch.tril(torch.ones(max_seq_length, max_seq_length)))
    
    def forward(self, x, targets=None):
        x = self.token_embedding(x) + self.position_embedding(torch.arange(x.size(1), device = x.device))
        x = self.dropout(x)
        for layer in self.layers:
            x = layer(x, self.attention_mask[:x.size(1), :x.size(1)])
        x = self.layernorm(x)
        logits = self.lm_head(x)
        if targets is None:
            return logits
        else:
            batch_size, seq_len, vocab_size = logits.shape
            logits = logits.view(batch_size * seq_len, vocab_size)
            targets = targets.view(batch_size * seq_len)
            loss = F.cross_entropy(logits, targets)
            return logits, loss

    def generate(self, input_str, tokenizer):
        with torch.no_grad():
            device = next(self.parameters()).device
            input_ids = tokenizer.encode(input_str)
            generated_ids = []
            eos = tokenizer.encode("<|endoftext|>", allowed_special={"<|endoftext|>"})[0]
            for i in range(256): # 最多512个token
                logits = self.forward(torch.tensor(input_ids, device=device).unsqueeze(0))
                next_token = torch.argmax(logits[:,-1, :], -1)
                token_id = next_token[0].item()
                if token_id == eos:
                    break
                generated_ids.append(token_id)
                input_ids.append(token_id)
            return tokenizer.decode(generated_ids)
