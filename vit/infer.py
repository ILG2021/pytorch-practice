import sys

import torch

sys.path.append(".")
from vit.dataset import get_dataloader
from vit.model import VITClassifier

device = "cuda" if torch.cuda.is_available() else "cpu"

model = VITClassifier(1, 28, 7, 16, 4, 10)
model.load_state_dict(torch.load("vit/model_last.pth"))
model = model.to(device)
test_loader = get_dataloader(False, 5)
x, y = next(iter(test_loader))
print(y)
x = x.to(device)
output, attention_scores = model(x, True)
print(torch.argmax(torch.softmax(output, dim=1), dim=1))
