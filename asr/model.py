import sys

import torch
from torch import nn

sys.path.append(".")

from asr.downsampling import DownsamplingModal
from asr.rvq import ResidualVectorQuantizer

from asr.transformer import SinPositionEncoding, Transformer


class TranscribeModel(nn.Module):
    def __init__(self, num_codebooks, codebook_size, embedding_dim, vocab_size, strides, num_transformer_layers,
                 initial_mean_pooling_kernel_size, max_seq_length=2000):
        super().__init__()
        self.options = {
            "num_codebooks": num_codebooks,
            "codebook_size": codebook_size,
            "embedding_dim": embedding_dim,
            "vocab_size": vocab_size,
            "strides": strides,
            "num_transformer_layers": num_transformer_layers,
            "max_seq_length": max_seq_length,
            "initial_mean_pooling_kernel_size": initial_mean_pooling_kernel_size
        }
        self.downsampling_model = DownsamplingModal(embedding_dim, embedding_dim // 2, strides,
                                                    initial_mean_pooling_kernel_size)
        self.pos_encoding = SinPositionEncoding(embedding_dim, max_seq_length)
        self.transformer_model = Transformer(embedding_dim, num_transformer_layers)
        self.rvq = ResidualVectorQuantizer(num_codebooks, codebook_size, embedding_dim)
        self.output_layer = nn.Linear(embedding_dim, vocab_size)

    def forward(self, x: torch.Tensor):
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
        return model
