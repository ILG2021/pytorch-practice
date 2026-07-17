# 使用https://huggingface.co/datasets/ej2/seq-monkey-data/resolve/main/%E5%8F%A4%E8%AF%97%E4%BB%8A%E8%AF%91.tar.bz2训练
import math
import tiktoken
import torch
from datasets import load_dataset
from torch.optim import lr_scheduler
from torch.utils.data import Dataset, DataLoader
from torch.utils.tensorboard import SummaryWriter

from model import GPT2


class MyDataset(Dataset):
    def __init__(self):
        super().__init__()
        self.encoded_data = []
        dataset = load_dataset("dirtycomputer/THUCNews")
        dataset = dataset["train"]
        all_encoded = []
        tokenizer = tiktoken.get_encoding("gpt2")
        eos = tokenizer.encode("<|endoftext|>", allowed_special={"<|endoftext|>"})[0]
        for item in dataset:
            text = item["text"]
            all_encoded.extend(tokenizer.encode(text) + [eos])

        max_seq_len = 512
        for i in range(0, len(all_encoded), max_seq_len):
            chunk = all_encoded[i:i + max_seq_len + 1]
            if len(chunk) < max_seq_len:
                chunk += [eos] * (max_seq_len + 1 - len(chunk))
            self.encoded_data.append(chunk)

    def __len__(self):
        return min(len(self.encoded_data), 1000)

    def __getitem__(self, idx):
        item = self.encoded_data[idx]
        x = torch.tensor(item[:-1], dtype=torch.long)
        y = torch.tensor(item[1:], dtype=torch.long)
        return x, y


device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
dataloader = DataLoader(dataset=MyDataset(), batch_size=4, shuffle=True)
model = GPT2()
model = model.to(device)
total_param = sum(p.numel() for p in model.parameters())
print(f"总参数：{total_param / 1e6}MB")
optimizer = torch.optim.AdamW(model.parameters(), lr=3e-4)
scheduler = lr_scheduler.CosineAnnealingLR(optimizer, T_max=len(dataloader)*3)

model.train()
writer = SummaryWriter(log_dir="runs/gpt2")

global_step = 0
for epoch in range(3):
    total_loss = 0
    for x, y in dataloader:
        x = x.to(device)
        y = y.to(device)
        optimizer.zero_grad()
        logits, loss = model(x, y)
        loss.backward()
        optimizer.step()
        scheduler.step()
        if global_step % 10 == 0 and global_step != 0:
            print("steps", global_step, "loss:", loss.item())
            writer.add_scalar("loss",loss.item(), global_step)
        global_step += 1

torch.save(model.state_dict(), "gpt2/gpt2.pth")
