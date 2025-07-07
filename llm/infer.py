import torch

from train import vocab_to_index, decode
from model import CausalLM
from train import encode, embed_size, vocab_size, num_layers, device

model = CausalLM(embed_size, vocab_size, num_layers).to(device)
model.load_state_dict(torch.load("llm/model_last.pth")["state_dict"])


def generate(prefix, max_length=18):
    input_tokens = torch.tensor([vocab_to_index["<start>"]] + encode(prefix)).to(device).unsqueeze(0)
    for _ in range(max_length):
        with torch.no_grad():
            logits = model(input_tokens)
            last_token_logits = logits[0, -1:].argmax(dim=-1, keepdim=True)
            input_tokens = torch.cat([input_tokens, last_token_logits], dim=1)
        if input_tokens[0][-1] == vocab_to_index['<end>']:
            break
    return decode(input_tokens[0].tolist())

print(generate("machine"))
