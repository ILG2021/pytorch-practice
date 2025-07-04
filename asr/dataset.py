import datasets
import torch
from torch.functional import F
from torch.utils.data import Dataset

from asr.tokenizer import my_tokenizer


class CustomDataset(Dataset):

    def __init__(self):
        self.dataset = datasets.load_dataset("m-aliabbas/idrak_timit_subsample1", split="train")

    def __getitem__(self, idx):
        item = self.dataset[idx]
        audio = torch.from_numpy(item["audio"]["array"]).float()
        input_ids = my_tokenizer.encode(item["transcription"].upper()).ids

        return {
            "audio": audio,
            "input_ids": input_ids
        }

    def __len__(self):
        return 100


def collate_fn(batch):
    max_audio_len = max([item['audio'].shape[0] for item in batch])
    max_ids_len = max([len(item['input_ids']) for item in batch])
    return {
        # F.pad的第二个参数是个元组，分别表示左边和右边pad多少，pad值默认是0
        "audio": torch.stack([F.pad(item['audio'], (0, max_audio_len - item['audio'].shape[0])) for item in batch]),
        "input_ids": torch.stack([F.pad(torch.tensor(item['input_ids']), (0, max_ids_len - len(item['input_ids']))) for item in batch])
    }