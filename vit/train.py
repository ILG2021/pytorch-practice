import sys

import torch.optim
from torch import nn
from torch.utils.data import Subset

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

# epoch: 0 loss: 0.6325286626815796
# epoch: 1 loss: 0.3657349944114685
# epoch: 2 loss: 0.22916299104690552
# epoch: 3 loss: 0.15107090771198273
# epoch: 4 loss: 0.11748348921537399
# epoch: 5 loss: 0.0838257223367691
# epoch: 6 loss: 0.06997260451316833
# epoch: 7 loss: 0.057984743267297745
# epoch: 8 loss: 0.04824843257665634
# epoch: 9 loss: 0.049388304352760315