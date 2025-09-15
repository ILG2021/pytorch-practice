import traceback
from random import random

import click
import spacy
import torch
from datasets import load_dataset
from torch import nn, optim
from torch.nn.utils.rnn import pad_sequence
from torch.utils.data import DataLoader
from torchtext.vocab import build_vocab_from_iterator


class Encoder(nn.Module):
    def __init__(self, vocab_size, emb_dim, hidden_dim, n_layers, dropout_rate, *args, **kwargs):
        super().__init__(*args, **kwargs)
        self.embedding = nn.Embedding(vocab_size, emb_dim)
        self.rnn = nn.GRU(emb_dim, hidden_dim, n_layers, dropout=dropout_rate, batch_first=True)
        self.dropout = nn.Dropout(dropout_rate)

    def forward(self, x):
        # x: [batch, seq_len] embedding: [batch, seq_len, emb_dim] hidden: [n_layers, batch, hidden_dim]
        # nn.GRU batch_first只影响输入和输出的维度顺序，不影响hidden
        embedding = self.dropout(self.embedding(x))
        output, hidden = self.rnn(embedding)
        return hidden


class Decoder(nn.Module):
    def __init__(self, vocab_size, emb_dim, hidden_dim, n_layers, dropout_rate, *args, **kwargs):
        super().__init__(*args, **kwargs)
        self.embedding = nn.Embedding(vocab_size, emb_dim)
        self.rnn = nn.GRU(emb_dim, hidden_dim, n_layers, dropout=dropout_rate, batch_first=True)
        self.dropout = nn.Dropout(dropout_rate)
        self.fc = nn.Linear(hidden_dim, vocab_size)
        self.vocab_size = vocab_size

    def forward(self, x, hidden):  # 解码器输入的是只有一个token的序列
        x = x.unsqueeze(1)  # [batch] -> [batch, 1]
        embedding = self.dropout(self.embedding(x))
        output, hidden = self.rnn(embedding, hidden)
        output = self.fc(output.squeeze(1))  # output.squeeze: [batch, 1, hidden_dim] -> [batch, hidden_dim]
        return output, hidden


class Seq2Seq(nn.Module):
    def __init__(self, encoder, decoder, device, *args, **kwargs):
        super().__init__(*args, **kwargs)
        self.encoder = encoder
        self.decoder = decoder
        self.device = device

    def forward(self, src, tgt, teacher_force_rate=0.5):
        hidden = self.encoder(src)
        outputs = torch.zeros(tgt.shape[0], tgt.shape[1], self.decoder.vocab_size).to(self.device)

        token0 = tgt[:, 0]
        x = token0
        for t in range(1, tgt.shape[1]):
            output, hidden = self.decoder(x, hidden)
            outputs[:, t, :] = output
            teacher_force = random() < teacher_force_rate
            x = tgt[:, t] if teacher_force else torch.argmax(output, -1)
        return outputs


trainset = load_dataset("Aye10032/zh-en-translate-20k")["train"]
tokenizer_zh = spacy.load('zh_core_web_sm')
tokenizer_en = spacy.load('en_core_web_sm')

sentences_zh = []
sentences_en = []
for item in trainset:
    sentences_zh.append(item['chinese'])
    sentences_en.append(item['english'])
tokens_zh = [[token.text for token in tokenizer_zh.tokenizer(sentence)] for sentence in sentences_zh]
tokens_en = [[token.text for token in tokenizer_en.tokenizer(sentence)] for sentence in sentences_en]

vocab_zh = build_vocab_from_iterator(
    tokens_zh,
    min_freq=2,
    specials=['<unk>', '<pad>', '<bos>', '<eos>'],
    max_tokens=10000
)

vocab_en = build_vocab_from_iterator(
    tokens_en,
    min_freq=2,
    specials=['<unk>', '<pad>', '<bos>', '<eos>'],
    max_tokens=10000
)

vocab_zh.set_default_index(vocab_zh['<unk>'])
vocab_en.set_default_index(vocab_en['<unk>'])

EMB_DIM = 256
HID_DIM = 512
DROPOUT = 0.5
N_LAYERS = 2

device = "cuda" if torch.cuda.is_available() else 'cpu'
encoder = Encoder(len(vocab_zh), EMB_DIM, HID_DIM, N_LAYERS, DROPOUT)
decoder = Decoder(len(vocab_en), EMB_DIM, HID_DIM, N_LAYERS, DROPOUT)
model = Seq2Seq(encoder, decoder, device).to(device)


def train():
    trainset = []
    for i, tokens in enumerate(tokens_zh):
        trainset.append((torch.tensor([vocab_zh['<bos>']] + [vocab_zh[token] for token in tokens] + [vocab_zh['<eos>']],
                                      dtype=torch.long),
                         torch.tensor(
                             [vocab_en['<bos>']] + [vocab_en[token] for token in tokens_en[i]] + [vocab_en['<eos>']],
                             dtype=torch.long)))

    def collate_fn(batch):
        batch_zh, batch_en = [], []
        for zh, en in batch:
            batch_zh.append(zh)
            batch_en.append(en)
        batch_zh = pad_sequence(batch_zh, True, vocab_zh['<pad>'])
        batch_en = pad_sequence(batch_en, True, vocab_en['<pad>'])
        return batch_zh, batch_en

    dataloader = DataLoader(trainset, 32, shuffle=True, collate_fn=collate_fn)

    optimizer = optim.Adam(model.parameters())
    criterion = nn.CrossEntropyLoss(ignore_index=vocab_en['<pad>'])
    model.train()

    for epoch in range(100):
        for batch_zh, batch_en in dataloader:
            batch_zh, batch_en = batch_zh.to(device), batch_en.to(device)
            predict = model(batch_zh, batch_en)
            # 忽略第一个token，不预测
            predict = predict[:, 1:, :].contiguous().view(-1, predict.shape[-1])
            batch_en = batch_en[:, 1:].contiguous().view(-1)
            loss = criterion(predict, batch_en)
            loss.backward()
            optimizer.step()
            optimizer.zero_grad()
        print(f"epoch: {epoch} loss: {loss}")
    torch.save(model.state_dict(), "basic/seq2seq.pth")


def translate_sentence(sentence):
    tokens = ['<bos>'] + [token.text for token in tokenizer_zh.tokenizer(sentence)] + ['<eos>']
    token_ids = torch.tensor([vocab_zh[token] for token in tokens]).unsqueeze(0).to(device)
    with torch.no_grad():
        hidden = model.encoder(token_ids)
    tokens = torch.tensor([vocab_en['<bos>']]).to(device)
    max_len = 50
    translated = ''
    for i in range(max_len):
        with torch.no_grad():
            output, hidden = model.decoder(tokens, hidden)  # 扩充一维batch
        pred = torch.argmax(output, -1)  # output: [batch, vocab], pred: (batch,) batch是1
        tokens = pred
        translated += vocab_en.lookup_token(pred[0].item()) + ' '
        if pred[0].item() == vocab_en['<eos>']:
            break
    return translated


def infer():
    model.load_state_dict(torch.load("basic/seq2seq.pth"))
    model.eval()
    print("Enter a Chinese sentence to translate (or 'quit' to exit):")
    while True:
        sentence = input("> ").strip()
        if sentence.lower() == "quit":
            print("Exiting...")
            break
        if not sentence:
            print("Please enter a non-empty sentence.")
            continue
        translated = translate_sentence(sentence)
        print(f"Translated: {translated}")
        print("\nEnter another sentence (or 'quit' to exit):")


@click.command
@click.option("--is_train", default=False)
def main(is_train):
    if is_train:
        train()
    else:
        infer()


if __name__ == '__main__':
    main()
