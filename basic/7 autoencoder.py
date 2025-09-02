import torch
from torch import nn


class AutoEncoder(nn.Module):
    def __init__(self, *args, **kwargs):
        super().__init__(*args, **kwargs)
        self.encoder = nn.Sequential(
            nn.Conv2d(1, 16, 3, padding=1),  # 16x28x28
            nn.ReLU(),
            nn.MaxPool2d(2, 2),  # 16x14x14
            nn.Conv2d(16, 8, 3, padding=1),  # 8x14x14
            nn.ReLU(),
            nn.MaxPool2d(2, 2),  # 8x7x7
            nn.Conv2d(8, 4, 3, padding=1),  # 4x7x7
            nn.ReLU()
        )

        self.decoder = nn.Sequential(
            # Hout = (Hin - 1)*stride - 2*padding + kernel_size + output_padding
            nn.ConvTranspose2d(4, 8, 2, 2),  # 8x14x14
            nn.ReLU(),
            nn.ConvTranspose2d(8, 16, 2, 2),  # 16x28x28
            nn.ReLU(),
            nn.Conv2d(16, 1, 3, padding=1),
            nn.Sigmoid()
        )

    def forward(self, x):
        latent = self.encoder(x)
        reconstructed = self.decoder(latent)
        return reconstructed, latent


model = AutoEncoder()
dummy_input = torch.randn(10, 1, 28, 28)
reconstructed, latent = model(dummy_input)
print(f"latent shape: {latent.shape} output shape: {reconstructed.shape}")
