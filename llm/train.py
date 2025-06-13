import torch
import torch.nn.functional as F

from model import CausalLM


def encode(text):
    return [vocab_to_index[word] for word in text.split()]


def encode_batch(sentences):
    encode_sentences = [[vocab_to_index['<start>']] + encode(s) + [vocab_to_index['<end>']] for s in sentences]
    max_len = max([len(s) for s in encode_sentences])
    encode_sentences = [s + (max_len - len(s)) * [vocab_to_index["<pad>"]] for s in encode_sentences]
    return encode_sentences


def decode(tokens):
    return " ".join([vocab[token] for token in tokens])


dataset = [
    "dont forget to like share and subscribe",
    "dont forget machine learning is fun",
    "machine learning is fun and awesome",
    "if you like machine learning i like you",
    "i like you more than machine learning"
]

vocab = []
special_token = ["<pad>", "<start>", "<end>"]
for sentence in dataset:
    for word in sentence.split():
        if word not in vocab:
            vocab.append(word)
vocab = special_token + list(vocab)
vocab_to_index = {word: index for index, word in enumerate(vocab)}

vocab_size = len(vocab)
embed_size = 6
num_layers = 2
device = 'cuda'


def train():
    tokenize_dataset = torch.tensor(encode_batch(dataset))

    input_tokens = tokenize_dataset[:, :-1]
    target_tokens = tokenize_dataset[:, 1:]
    num_epochs = 1000
    input_tokens = input_tokens.to(device)
    target_tokens = target_tokens.to(device)

    model = CausalLM(embed_size, vocab_size, num_layers).to(device)
    optimizer = torch.optim.Adam(model.parameters(), lr=2e-3)

    for i in range(num_epochs):
        logits = model(input_tokens)
        loss = F.cross_entropy(logits.view(-1, logits.shape[-1]), target_tokens.view(-1))
        loss.backward()
        optimizer.step()
        optimizer.zero_grad()
        if i % 10 == 0:
            print(f"Epoch {i} Loss: {loss.item()}")
            torch.save({
                "state_dict": model.state_dict(),
            }, "llm/model_last.pth")


if __name__ == '__main__':
    train()
