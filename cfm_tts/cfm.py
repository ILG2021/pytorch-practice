from random import random

import torch

from dit import DiT
from utils import exists, lens_to_mask, mask_from_frac_lengths
import torch.nn.functional as F


def forward(inp, text):  # mel or raw wave  # noqa: F722
    transformer = DiT(dim=1024,
                      depth=22,
                      heads=16,
                      mel_dim=3,
                      ff_mult=2,
                      text_dim=4,
                      conv_layers=4)
    audio_drop_prob = 0.3
    cond_drop_prob = 0.2
    # 星号是解包元组
    batch, seq_len, dtype = *inp.shape[:2], inp.dtype
    frac_lengths_mask = (0.7, 1.0)
    # lens and mask
    # [5, 5]
    lens = torch.full((batch,), seq_len)

    mask = lens_to_mask(lens, length=seq_len)  # useless here, as collate_fn will pad to max length in batch
    # [[True, True, True, True, True],
    #  [True, True, True, True, True]]
    # get a random span to mask out for training conditionally
    frac_lengths = torch.zeros((batch,)).float().uniform_(*frac_lengths_mask)
    rand_span_mask = mask_from_frac_lengths(lens, frac_lengths)

    if exists(mask):
        rand_span_mask &= mask
    # rand_span_mask维度是inp的前两维
    # mel is x1
    x1 = inp

    # x0 is gaussian noise
    x0 = torch.randn_like(x1)

    # time step
    time = torch.rand((batch,), dtype=dtype)
    # TODO. noise_scheduler

    # sample xt (φ_t(x) in the paper)
    t = time.unsqueeze(-1).unsqueeze(-1)
    φ = (1 - t) * x0 + t * x1
    flow = x1 - x0

    # only predict what is within the random mask span for infilling
    cond = torch.where(rand_span_mask[..., None], torch.zeros_like(x1), x1)

    # transformer and cfg training with a drop rate
    drop_audio_cond = random() < audio_drop_prob  # p_drop in voicebox paper
    if random() < cond_drop_prob:  # p_uncond in voicebox paper
        drop_audio_cond = True
        drop_text = True
    else:
        drop_text = False

    # if want rigorously mask out padding, record in collate_fn in dataset.py, and pass in here
    # adding mask will use more memory, thus also need to adjust batchsampler with scaled down threshold for long sequences
    pred = transformer(
        x=φ, cond=cond, text=text, time=time, drop_audio_cond=drop_audio_cond, drop_text=drop_text
    )

    # flow matching loss
    loss = F.mse_loss(pred, flow, reduction="none")
    loss = loss[rand_span_mask]

    return loss.mean(), cond, pred


if __name__ == '__main__':
    # 2 5 3
    audio = torch.tensor([[[1.0, 2.0, 3.0],
                           [1.0, 2.0, 3.0],
                           [1.0, 2.0, 3.0],
                           [1.0, 2.0, 3.0],
                           [1.0, 2.0, 3.0]],

                          [[4.0, 5.0, 6.0],
                           [4.0, 5.0, 6.0],
                           [4.0, 5.0, 6.0],
                           [4.0, 5.0, 6.0],
                           [4.0, 5.0, 6.0]]]
                         )
    # 2 2, 字符索引
    t = torch.tensor([[1, 2, 3],

                      [3, 4, 5]])
    forward(audio, t)
