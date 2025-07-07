import sys

import torch.optim
from torch import nn

sys.path.append(".")
from vit.dataset import get_dataloader
from vit.model import VITClassifier

device = "cuda" if torch.cuda.is_available() else "cpu"
num_epochs = 10

model = VITClassifier(1, 28, 7, 16, 4, 10)
model = model.to(device)
optimizer = torch.optim.Adam(model.parameters(), lr=1e-3)
loss_fn = nn.CrossEntropyLoss()

for epoch in range(num_epochs):
    for x, y in get_dataloader(True, 16):
        x = x.to(device)
        y = y.to(device)
        output = model(x)
        loss = loss_fn(output, y)
        optimizer.zero_grad()
        loss.backward()
        optimizer.step()
    print(f"epoch: {epoch} loss: {loss.item()}")
    torch.save(model.state_dict(), "vit/model_last.pth")

