import os
import sys

import torch
from torchvision.utils import save_image

sys.path.append(".")
from vae.model import VAE

model = VAE(input_dim=784, hidden_dim=400, latent_dim=20)

checkpoint = torch.load("vae/model.pth")
model.load_state_dict(checkpoint)

model.eval()
with torch.no_grad():
    z = torch.randn(64, 20)  # 随机采样潜在变量
    samples = model.decode(z).view(-1, 1, 28, 28)  # 重构图像
    # 可视化 samples
    # 创建一个目录来保存结果
    os.makedirs("results", exist_ok=True)

    # samples 的 shape 是 [64, 1, 28, 28]
    save_image(samples, "results/generated_samples.png", nrow=8)  # nrow=8 表示每行显示8张图
    print("Generated samples saved to results/generated_samples.png")