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
        batch_size = src.shape[0]
        tgt_len = tgt.shape[1]
        tgt_vocab_size = self.decoder.vocab_size

        # 初始化输出张量
        outputs = torch.zeros(batch_size, tgt_len, tgt_vocab_size).to(self.device)

        # 编码器获取隐藏状态
        hidden = self.encoder(src)

        # 解码器的第一个输入是<bos>
        input_token = tgt[:, 0]

        for t in range(1, tgt_len):
            # 前向传播
            output, hidden = self.decoder(input_token, hidden)
            outputs[:, t, :] = output

            # Teacher forcing决策
            teacher_force = random() < teacher_force_rate
            input_token = tgt[:, t] if teacher_force else torch.argmax(output, dim=-1)

        return outputs


# 数据加载和预处理
print("Loading dataset...")
trainset = load_dataset("Aye10032/zh-en-translate-20k")["train"]
tokenizer_zh = spacy.load('zh_core_web_sm')
tokenizer_en = spacy.load('en_core_web_sm')

print("Tokenizing sentences...")
sentences_zh = []
sentences_en = []
for item in trainset:
    sentences_zh.append(item['chinese'])
    sentences_en.append(item['english'])

# 改进的tokenization，过滤掉太长的句子
MAX_LEN = 50
tokens_zh = []
tokens_en = []
filtered_indices = []

for i, (zh_sentence, en_sentence) in enumerate(zip(sentences_zh, sentences_en)):
    zh_tokens = [token.text for token in tokenizer_zh.tokenizer(zh_sentence)]
    en_tokens = [token.text for token in tokenizer_en.tokenizer(en_sentence)]

    # 过滤太长的句子
    if len(zh_tokens) <= MAX_LEN and len(en_tokens) <= MAX_LEN:
        tokens_zh.append(zh_tokens)
        tokens_en.append(en_tokens)
        filtered_indices.append(i)

print(f"Filtered dataset size: {len(tokens_zh)} (original: {len(sentences_zh)})")

# 构建词汇表
print("Building vocabularies...")
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

print(f"Chinese vocab size: {len(vocab_zh)}")
print(f"English vocab size: {len(vocab_en)}")

# 模型参数
EMB_DIM = 256
HID_DIM = 512
DROPOUT = 0.5
N_LAYERS = 2

device = "cuda" if torch.cuda.is_available() else 'cpu'
print(f"Using device: {device}")

encoder = Encoder(len(vocab_zh), EMB_DIM, HID_DIM, N_LAYERS, DROPOUT)
decoder = Decoder(len(vocab_en), EMB_DIM, HID_DIM, N_LAYERS, DROPOUT)
model = Seq2Seq(encoder, decoder, device).to(device)


# 参数初始化
def init_weights(m):
    for name, param in m.named_parameters():
        if 'weight' in name:
            nn.init.normal_(param.data, mean=0, std=0.01)
        else:
            nn.init.constant_(param.data, 0)


model.apply(init_weights)


def train():
    print("Preparing training data...")
    trainset = []
    for i, tokens in enumerate(tokens_zh):
        src_tokens = [vocab_zh['<bos>']] + [vocab_zh[token] for token in tokens] + [vocab_zh['<eos>']]
        tgt_tokens = [vocab_en['<bos>']] + [vocab_en[token] for token in tokens_en[i]] + [vocab_en['<eos>']]

        trainset.append((
            torch.tensor(src_tokens, dtype=torch.long),
            torch.tensor(tgt_tokens, dtype=torch.long)
        ))

    def collate_fn(batch):
        batch_zh, batch_en = [], []
        for zh, en in batch:
            batch_zh.append(zh)
            batch_en.append(en)
        batch_zh = pad_sequence(batch_zh, True, vocab_zh['<pad>'])
        batch_en = pad_sequence(batch_en, True, vocab_en['<pad>'])
        return batch_zh, batch_en

    dataloader = DataLoader(trainset, 32, shuffle=True, collate_fn=collate_fn)

    # 优化器和学习率调度
    optimizer = optim.Adam(model.parameters(), lr=0.001)
    scheduler = optim.lr_scheduler.StepLR(optimizer, step_size=10, gamma=0.5)
    criterion = nn.CrossEntropyLoss(ignore_index=vocab_en['<pad>'])

    model.train()
    print("Starting training...")

    for epoch in range(50):  # 减少epoch数量
        total_loss = 0
        num_batches = 0

        for batch_zh, batch_en in dataloader:
            batch_zh, batch_en = batch_zh.to(device), batch_en.to(device)

            optimizer.zero_grad()

            # Teacher forcing rate随训练降低
            teacher_force_rate = max(0.5, 1.0 - epoch * 0.01)
            predict = model(batch_zh, batch_en, teacher_force_rate)

            # 计算损失，忽略第一个token(<bos>)
            predict = predict[:, 1:, :].contiguous().view(-1, predict.shape[-1])
            target = batch_en[:, 1:].contiguous().view(-1)

            loss = criterion(predict, target)
            loss.backward()

            # 梯度裁剪
            torch.nn.utils.clip_grad_norm_(model.parameters(), 1.0)

            optimizer.step()

            total_loss += loss.item()
            num_batches += 1

        avg_loss = total_loss / num_batches
        scheduler.step()

        print(f"Epoch: {epoch + 1:2d} | Loss: {avg_loss:.4f} | LR: {scheduler.get_last_lr()[0]:.6f}")

        # 每10个epoch测试一下
        if (epoch + 1) % 10 == 0:
            test_translation()

    torch.save(model.state_dict(), "basic/seq2seq.pth")
    print("Training completed and model saved!")


def test_translation():
    """测试翻译效果"""
    model.eval()
    test_sentences = [
        "你好世界",
        "我爱你",
        "今天天气很好"
    ]

    print("\n--- Translation Test ---")
    for sentence in test_sentences:
        try:
            translated = translate_sentence(sentence)
            print(f"中文: {sentence}")
            print(f"英文: {translated.strip()}")
            print()
        except Exception as e:
            print(f"Translation error for '{sentence}': {e}")
    print("--- End Test ---\n")
    model.train()


def translate_sentence(sentence):
    tokens = ['<bos>'] + [token.text for token in tokenizer_zh.tokenizer(sentence)] + ['<eos>']
    token_ids = torch.tensor([vocab_zh[token] for token in tokens]).unsqueeze(0).to(device)

    with torch.no_grad():
        hidden = model.encoder(token_ids)

    tokens = torch.tensor([vocab_en['<bos>']]).to(device)
    max_len = 50
    translated = []

    for i in range(max_len):
        with torch.no_grad():
            output, hidden = model.decoder(tokens, hidden)

        pred = torch.argmax(output, dim=-1)
        tokens = pred

        pred_token = vocab_en.lookup_token(pred[0].item())
        if pred_token == '<eos>':
            break
        if pred_token not in ['<bos>', '<pad>', '<unk>']:
            translated.append(pred_token)

    return ' '.join(translated)


def infer():
    try:
        model.load_state_dict(torch.load("basic/seq2seq.pth", map_location=device))
        model.eval()
        print("Model loaded successfully!")
    except FileNotFoundError:
        print("Model file not found. Please train the model first.")
        return

    print("Enter a Chinese sentence to translate (or 'quit' to exit):")
    while True:
        sentence = input("> ").strip()
        if sentence.lower() in ["quit", "exit", "q"]:
            print("Exiting...")
            break
        if not sentence:
            print("Please enter a non-empty sentence.")
            continue

        try:
            translated = translate_sentence(sentence)
            print(f"Translated: {translated}")
        except Exception as e:
            print(f"Translation error: {e}")
        print("\nEnter another sentence (or 'quit' to exit):")


@click.command()
@click.option("--is_train", default=False, help="Train the model")
def main(is_train):
    if is_train:
        train()
    else:
        infer()


if __name__ == '__main__':
    main()