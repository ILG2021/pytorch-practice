import torch
from torch import nn
from torch.nn import ModuleList
from vector_quantize_pytorch import ResidualVQ


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
    def __init__(self, out_channel, hidden_dim, strides=[6, 6, 8, 4, 2]):
        super().__init__()
        self.layers = ModuleList()
        self.meaning_pool = nn.AvgPool1d(kernel_size=2)  # 下采样，没有学习权重
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


class TranscribeModel(nn.Module):
    def __init__(self, num_codebooks, codebook_size, embedding_dim, vocab_size, strides, num_transformer_layers,
                 max_seq_length=2000):
        super().__init__()
        self.options = {
            "num_codebooks": num_codebooks,
            "codebook_size": codebook_size,
            "vocab_size": vocab_size,
            "strides": strides,
            "num_transformer_layers": num_transformer_layers,
            "max_seq_length": max_seq_length
        }
        self.downsampling_model = DownsamplingModal(embedding_dim, embedding_dim // 2, strides)
        self.transformer_model = nn.TransformerEncoder(nn.TransformerEncoderLayer(
            d_model=embedding_dim,
            nhead=1,
            dim_feedforward=embedding_dim,
            dropout=0.1,
            batch_first=True  # 输入形状为 (batch_size, seq_len, d_model)
        ), 2)
        self.rvq = ResidualVQ(dim=embedding_dim, num_quantizers=num_codebooks, codebook_size=codebook_size)
        self.output_layer = nn.Linear(embedding_dim, vocab_size)

    def forward(self, x):
        x = self.downsampling_model(x)
        x = self.transformer_model(x)
        x, _, loss = self.rvq(x)
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
