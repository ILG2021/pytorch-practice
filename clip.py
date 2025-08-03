import torch
from transformers import pipeline

clip = pipeline(
   task="zero-shot-image-classification",
   model="openai/clip-vit-base-patch32",
   torch_dtype=torch.bfloat16,
   device=0
)
labels = ["a photo of a cat", "a photo of a dog", "a photo of a car"]
print(clip("data/1888_Flocken_Elektrowagen.jpg", candidate_labels=labels))