import torch

try:
    from gpt2.model import GPT2
except ModuleNotFoundError:
    from model import GPT2

# GPT2 expects token indices (integers) as input rather than continuous features
x = torch.randint(0, 50257, (1, 4))
model = GPT2()
print("Input shape:", x.shape)
print("Output shape:", model(x).shape)
