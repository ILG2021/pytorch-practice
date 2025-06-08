import sys

from torch.utils.data import DataLoader
sys.path.append(".")
from asr.dataset import CustomDataset, collate_fn

dataset = CustomDataset()
dataloader = DataLoader(dataset, batch_size=4, collate_fn=collate_fn)
print("dataloader", next(iter(dataloader)))