import sys

import torch
from torch import nn
from torch.optim import Adam
from torchvision import transforms, datasets
sys.path.append(".")
from vae.model import VAE


def loss_function(recon_x, x, mu, log_var):
    BCE = nn.functional.binary_cross_entropy(recon_x, x, reduction='sum')
    KLD = -0.5 * torch.sum(1 + log_var - mu.pow(2) - log_var.exp())
    return BCE + KLD


transform = transforms.ToTensor()
dataset = datasets.MNIST('data', train=True, download=True, transform=transform)
dataloader = torch.utils.data.DataLoader(dataset, batch_size=128, shuffle=True)

model = VAE(input_dim=784, hidden_dim=400, latent_dim=20)
optimizer = Adam(model.parameters(), lr=1e-3)

model.train()
for epoch in range(50):
    for data, _ in dataloader:
        data = data.view(-1, 784)
        optimizer.zero_grad()
        recon, mu, log_var = model(data)
        loss = loss_function(recon, data, mu, log_var)
        loss.backward()
        optimizer.step()
    print(f'Epoch {epoch + 1}, Loss: {loss.item():.4f}')

torch.save(model.state_dict(), "vae/model.pth")
