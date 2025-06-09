import sys

from torch import nn

sys.path.append(".")

import torch
from torch.utils.data import DataLoader
from asr.model import TranscribeModel
from asr.dataset import CustomDataset, collate_fn


def run_loss_function(log_probs, target, blank_token):
    loss_function = nn.CTCLoss(blank=blank_token)
    input_lengths = tuple(log_probs.shape[1] for _ in range(log_probs.shape[0]))
    target_lengths = (target != blank_token).sum(dim=1)
    target_lengths = tuple(t.item() for t in target_lengths)
    input_seq_first = log_probs.premute(1, 0, 2)
    return loss_function(input_seq_first, target, input_lengths, target_lengths)



vq_initial_loss_weight = 10
vq_warmup_steps = 1000
vq_final_loss_weight = 0.5

dataset = CustomDataset()
dataloader = DataLoader(dataset, batch_size=4, collate_fn=collate_fn)
print("dataloader", next(iter(dataloader)))

device = torch.device("cuda" if torch.cuda.is_available() else "cpu")

model = TranscribeModel(num_codebooks=2, codebook_size=32, embedding_dim=16, num_transformer_layers=2,
                        vocab_size=len(dataset.tokenizer.get_vocab()), strides=[6, 6, 6])
num_epochs = 100
steps = 0
optimizer = torch.optim.Adam(model.parameters(), lr=0.005)
for i in range(num_epochs):
    for idx, batch in enumerate(dataloader):
        audio = batch["audio"]
        target = batch["input_ids"]
        audio = audio.to(device)
        target = target.to(device)
        optimizer.zero_grad()
        output, vq_loss = model(audio)
        ctc_loss = run_loss_function(output, target, blank_token=0)
        vq_loss_weight = max(vq_final_loss_weight,
                             vq_initial_loss_weight - (vq_initial_loss_weight - vq_final_loss_weight) * (
                                         steps / vq_warmup_steps))
        if vq_loss is None:
            loss = ctc_loss
        else:
            loss = ctc_loss + vq_loss_weight * vq_loss
        if torch.isinf(loss):
            continue
        loss.backward()
        torch.nn.utils.clip_grad_norm(model.parameters(), max_norm=10)
        optimizer.step()
        steps = steps + 1
    if i % 20 == 0:
        model.save("model_Latest.pth")
