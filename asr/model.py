import torch
from torch import nn
from torch.nn import ModuleList
from vector_quantize_pytorch import ResidualVQ

from cfm_tts.modules import SinusPositionEmbedding
from llm.model import Transformer, SinPositionEncoding


class ResidualDownsamplingBlock(nn.Module):
    def __init__(self, in_channel, out_channel, stride, kernel_size=4):
        super().__init__()
        self.conv1 = nn.Conv1d(in_channel, out_channel, kernel_size, padding='same')
        self.bn = nn.BatchNorm1d(out_channel)  # 输入归一化
        self.relu = nn.ReLU()
        # stride是步长，用来下采样
        self.conv2 = nn.Conv1d(out_channel, out_channel, kernel_size, stride=stride)

    def forward(self, x):
        y = self.conv1(x)
        y = self.bn(y)
        y = self.relu(y) + x
        y = self.conv2(y)
        return y


class DownsamplingModal(nn.Module):
    def __init__(self, out_channel, hidden_dim, strides=[6, 6, 8, 4, 2], initial_mean_pooling_kernel_size=2):
        super().__init__()
        self.layers = ModuleList()
        self.meaning_pool = nn.MaxPool1d(kernel_size=initial_mean_pooling_kernel_size)  # 下采样，没有学习权重
        for i in range(len(strides)):
            self.layers.append(
                ResidualDownsamplingBlock(hidden_dim if i > 0 else 1, hidden_dim, strides[i], kernel_size=8))
        self.final_conv = nn.Conv1d(hidden_dim, out_channel, 4, padding='same')

    def forward(self, x):
        x = self.meaning_pool(x)
        for layer in self.layers:
            x = layer(x)
        x = self.final_conv(x)
        x = x.transpose(1, 2)  # batch dim seq_len  transformer的维度
        return x


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


class TranscribeModel(nn.Module):
    def __init__(self, num_codebooks, codebook_size, embedding_dim, vocab_size, strides, num_transformer_layers,
                 initial_mean_pooling_kernel_size, max_seq_length=2000):
        super().__init__()
        self.options = {
            "num_codebooks": num_codebooks,
            "codebook_size": codebook_size,
            "vocab_size": vocab_size,
            "strides": strides,
            "num_transformer_layers": num_transformer_layers,
            "max_seq_length": max_seq_length
        }
        self.downsampling_model = DownsamplingModal(embedding_dim, embedding_dim // 2, strides,
                                                    initial_mean_pooling_kernel_size)
        self.pos_encoding = SinPositionEncoding(embedding_dim, max_seq_length)
        self.transformer_model = Transformer(embedding_dim, num_transformer_layers)
        self.rvq = ResidualVectorQuantizer(num_codebooks, codebook_size, embedding_dim)
        self.output_layer = nn.Linear(embedding_dim, vocab_size)

    def forward(self, x: torch.Tensor):
        loss = torch.tensor(0.0)
        x = x.unsqueeze(1)
        x = self.downsampling_model(x)
        x = self.pos_encoding(x)
        x, _ = self.transformer_model(x)
        x, loss = self.rvq(x)
        x = self.output_layer(x)
        x = torch.log_softmax(x, dim=-1)
        return x, loss

    def save(self, path):
        torch.save({
            "model": self.state_dict(),
            "options": self.options
        }, path)

    @staticmethod
    def load(path):
        weight = torch.load(path)
        model = TranscribeModel(**weight['options'])
        model.load_state_dict(weight["model"])
