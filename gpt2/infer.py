import tiktoken
import torch

from model import GPT2

device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
model = GPT2()
model = model.to(device)
model.load_state_dict(torch.load("gpt2/gpt2.pth", map_location=device))
model.eval()
tokenizer = tiktoken.get_encoding("gpt2")
while True:
    user_input = input("你: ")
    if user_input.lower() in ["quit", "exit", "退出"]:
        break
    response = model.generate(user_input, tokenizer)
    print("GPT2:", response)
