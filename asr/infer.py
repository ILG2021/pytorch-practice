import sys

sys.path.append(".")

import datasets
import torch

from asr.dataset import CustomDataset
from asr.model import TranscribeModel

device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
dataset = CustomDataset()
model = TranscribeModel(num_codebooks=2, codebook_size=32, embedding_dim=16, num_transformer_layers=2,
                        vocab_size=len(dataset.tokenizer.get_vocab()), strides=[6, 6, 6],
                        initial_mean_pooling_kernel_size=4, max_seq_length=400).to(device)
model.load_state_dict(torch.load("asr/model_last.pth")["model"])
dataset = datasets.load_dataset("m-aliabbas/idrak_timit_subsample1", split="test")
data = next(iter(dataset))
print(data)
output, _ = model(torch.from_numpy(data["audio"]["array"]).float().unsqueeze(0).to(device))
print(torch.exp(output))
