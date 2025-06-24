import sys
sys.path.append(".")

from asr.tokenizer import my_tokenizer, blank_id
import datasets
import torch

from asr.dataset import CustomDataset
from asr.model import TranscribeModel

device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
dataset = CustomDataset()
model = TranscribeModel.load("asr/model_last.pth").to(device)
dataset2 = datasets.load_dataset("m-aliabbas/idrak_timit_subsample1", split="train")
data = next(iter(dataset2))
print(data)
def ctc_decode(log_probs, blank_token=0):
    decoded = []
    prev = None
    for t in torch.argmax(log_probs, dim=-1).cpu().tolist():
        if t != blank_token and t != prev:
            decoded.append(t)
        prev = t
    return decoded

output, _ = model(torch.from_numpy(data["audio"]["array"]).float().unsqueeze(0).to(device))
decoded = ctc_decode(output[0], blank_id)
print(my_tokenizer.decode(decoded))
