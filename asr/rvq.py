import torch
from torch import nn


class VectorQuantizer(nn.Module):
    def __init__(self, num_embeddings, embedding_dim, commitment_cost=0.25):
        super().__init__()
        self.embedding = nn.Embedding(num_embeddings, embedding_dim)
        nn.init.uniform_(self.embedding.weight, -0.1, 0.1)
        self.commitment_cost = commitment_cost

    def forward(self, x):
        batch_size, seq_len, embedding_dim = x.shape
        flat_x = x.reshape(batch_size * seq_len, embedding_dim)
        distances = torch.cdist(flat_x, self.embedding.weight, p=2)  # 欧式距离，维度是 batch_size*seq_len, num_embeddings
        encoding_indices = torch.argmin(distances, dim=1)  # (batch_size * seq_len,)
        quantized = self.embedding(encoding_indices).view(batch_size, seq_len, embedding_dim)
        e_latent_loss = torch.mean((quantized.detach() - x) ** 2)  # detach之后相当于常数
        q_latent_loss = torch.mean((quantized - x.detach()) ** 2)
        loss = q_latent_loss + self.commitment_cost * e_latent_loss
        quantized = x + (quantized - x).detach()
        return quantized, loss


class ResidualVectorQuantizer(nn.Module):
    def __init__(self, num_codebooks, codebook_size, embedding_dim):
        super().__init__()
        self.codebooks = nn.ModuleList([
            VectorQuantizer(codebook_size, embedding_dim) for _ in range(num_codebooks)
        ])

    def forward(self, x):
        out = 0
        total_loss = 0
        for codebook in self.codebooks:
            this_output, this_loss = codebook(x)
            x = x - this_output
            out += this_output
            total_loss += this_loss
        return out, total_loss
