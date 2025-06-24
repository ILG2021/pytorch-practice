import sys
sys.path.append(".")

from torch import nn
from asr.tokenizer import my_tokenizer, blank_id
import torch
from torch.utils.data import DataLoader
from asr.model import TranscribeModel
from asr.dataset import CustomDataset, collate_fn


def run_loss_function(log_probs, target, blank_token):
    loss_function = nn.CTCLoss(blank=blank_token, zero_infinity=True)
    input_lengths = tuple(log_probs.shape[1] for _ in range(log_probs.shape[0]))  # batch_size个seq_len
    target_lengths = (target != blank_token).sum(dim=1)  # (batch_size,)
    target_lengths = tuple(t.item() for t in target_lengths)
    input_seq_first = log_probs.permute(1, 0, 2) # nn.CTCLoss 要求输入张量的第一个维度是时间步（sequence_length），因此需要通过 permute(1, 0, 2) 重新排列维度。
    return loss_function(input_seq_first, target, input_lengths, target_lengths)

if __name__ == '__main__':
    vq_initial_loss_weight = 10
    vq_warmup_steps = 1000
    vq_final_loss_weight = 0.5

    dataset = CustomDataset()
    dataloader = DataLoader(dataset, batch_size=64, collate_fn=collate_fn)

    device = torch.device("cuda" if torch.cuda.is_available() else "cpu")

    model = TranscribeModel(num_codebooks=2, codebook_size=32, embedding_dim=16, num_transformer_layers=2,
                            vocab_size=len(my_tokenizer.get_vocab()), strides=[6, 6, 6],
                            initial_mean_pooling_kernel_size=4, max_seq_length=400).to(device)
    num_epochs = 1000
    steps = 0
    optimizer = torch.optim.Adam(model.parameters(), lr=0.005)
    ctc_losses = []
    vq_losses = []

    for i in range(num_epochs):
        for idx, batch in enumerate(dataloader):
            audio = batch["audio"]
            target = batch["input_ids"]
            audio = audio.to(device)
            target = target.to(device)
            optimizer.zero_grad()
            output, vq_loss = model(audio)
            ctc_loss = run_loss_function(output, target, blank_token=blank_id)
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
            torch.nn.utils.clip_grad_norm_(model.parameters(), max_norm=10)
            optimizer.step()
            vq_losses.append(vq_loss.item())
            ctc_losses.append(ctc_loss.item())
            steps = steps + 1
        if i % 20 == 0:
            avg_ctc_loss = sum(ctc_losses) / len(ctc_losses)
            avg_vq_loss = sum(vq_losses) / len(vq_losses)
            avg_loss = avg_ctc_loss + vq_loss_weight * avg_vq_loss
            print(
                f"Epoch: {i} Steps: {steps} avg_loss: {avg_loss} avg_ctc_loss: {avg_ctc_loss} avg_vq_loss: {avg_vq_loss}")
            ctc_losses = []
            vq_losses = []
            model.save("asr/model_last.pth")