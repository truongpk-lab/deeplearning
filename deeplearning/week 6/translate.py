import torch
import torch.nn as nn
import torch.nn.functional as F
import math
import random
from torch.utils.data import Dataset, DataLoader
from torch.nn.utils.rnn import pad_sequence

# Dummy dataset (you should replace with actual data loading)
en_sents = ["hello", "how are you", "my name is John"]
vi_sents = ["xin chào", "bạn khỏe không", "tôi tên là John"]

# Tokenizer class
class Tokenizer:
    def __init__(self, texts):
        self.word2idx = {"<pad>": 0, "<sos>": 1, "<eos>": 2, "<unk>": 3}
        self.idx2word = ["<pad>", "<sos>", "<eos>", "<unk>"]
        self.build_vocab(texts)

    def build_vocab(self, texts):
        for sentence in texts:
            for word in sentence.lower().split():
                if word not in self.word2idx:
                    self.word2idx[word] = len(self.idx2word)
                    self.idx2word.append(word)

    def encode(self, sentence):
        return [1] + [self.word2idx.get(w, 3) for w in sentence.lower().split()] + [2]

    def decode(self, tokens):
        return ' '.join([self.idx2word[t] for t in tokens if t not in (0, 1, 2)])

    def __len__(self):
        return len(self.idx2word)

# Dataset
class TranslationDataset(Dataset):
    def __init__(self, src_sents, trg_sents, src_tokenizer, trg_tokenizer):
        self.src_data = [torch.tensor(src_tokenizer.encode(sent)) for sent in src_sents]
        self.trg_data = [torch.tensor(trg_tokenizer.encode(sent)) for sent in trg_sents]

    def __len__(self):
        return len(self.src_data)

    def __getitem__(self, idx):
        return self.src_data[idx], self.trg_data[idx]

# Collate function
pad_idx = 0

def collate_fn(batch):
    src_batch, trg_batch = zip(*batch)
    src_batch = pad_sequence(src_batch, padding_value=pad_idx)
    trg_batch = pad_sequence(trg_batch, padding_value=pad_idx)
    return src_batch, trg_batch

# Positional Encoding
class PositionalEncoding(nn.Module):
    def __init__(self, d_model, max_len=5000):
        super().__init__()
        pe = torch.zeros(max_len, d_model)
        position = torch.arange(0, max_len).unsqueeze(1)
        div_term = torch.exp(torch.arange(0, d_model, 2) * -(math.log(10000.0) / d_model))
        pe[:, 0::2] = torch.sin(position * div_term)
        pe[:, 1::2] = torch.cos(position * div_term)
        pe = pe.unsqueeze(1)
        self.register_buffer('pe', pe)

    def forward(self, x):
        x = x + self.pe[:x.size(0)]
        return x

# Transformer Seq2Seq Model
class TransformerSeq2Seq(nn.Module):
    def __init__(self, src_vocab_size, trg_vocab_size, emb_dim=128, nhead=8, nhid=256, nlayers=3, dropout=0.1):
        super().__init__()
        self.src_tok_emb = nn.Embedding(src_vocab_size, emb_dim)
        self.trg_tok_emb = nn.Embedding(trg_vocab_size, emb_dim)
        self.pos_encoder = PositionalEncoding(emb_dim)
        self.transformer = nn.Transformer(d_model=emb_dim, nhead=nhead, num_encoder_layers=nlayers, 
                                          num_decoder_layers=nlayers, dim_feedforward=nhid, dropout=dropout)
        self.fc_out = nn.Linear(emb_dim, trg_vocab_size)

    def make_src_mask(self, src):
        return (src == pad_idx).transpose(0, 1)

    def make_trg_mask(self, trg):
        tgt_len = trg.shape[0]
        tgt_mask = torch.triu(torch.ones(tgt_len, tgt_len) == 1).transpose(0, 1)
        tgt_mask = tgt_mask.float().masked_fill(tgt_mask == 0, float('-inf')).masked_fill(tgt_mask == 1, float(0.0))
        return tgt_mask

    def forward(self, src, trg):
        src_emb = self.pos_encoder(self.src_tok_emb(src))
        trg_emb = self.pos_encoder(self.trg_tok_emb(trg))
        src_mask = self.make_src_mask(src)
        trg_mask = self.make_trg_mask(trg)
        out = self.transformer(src_emb, trg_emb, src_key_padding_mask=src_mask, tgt_mask=trg_mask)
        return self.fc_out(out)

# Training loop
def train(model, iterator, optimizer, criterion):
    model.train()
    total_loss = 0
    for src, trg in iterator:
        optimizer.zero_grad()
        output = model(src, trg[:-1])
        output_dim = output.shape[-1]
        output = output.view(-1, output_dim)
        trg = trg[1:].reshape(-1)
        loss = criterion(output, trg)
        loss.backward()
        optimizer.step()
        total_loss += loss.item()
    return total_loss / len(iterator)

# Main execution
src_tokenizer = Tokenizer(en_sents)
trg_tokenizer = Tokenizer(vi_sents)
dataset = TranslationDataset(en_sents, vi_sents, src_tokenizer, trg_tokenizer)
dataloader = DataLoader(dataset, batch_size=2, collate_fn=collate_fn)

model = TransformerSeq2Seq(len(src_tokenizer), len(trg_tokenizer))
optimizer = torch.optim.Adam(model.parameters(), lr=0.0005)
criterion = nn.CrossEntropyLoss(ignore_index=pad_idx)

for epoch in range(10):
    loss = train(model, dataloader, optimizer, criterion)
    print(f"Epoch {epoch+1}, Loss: {loss:.4f}")

# Example inference
def translate_sentence(model, sentence, src_tokenizer, trg_tokenizer, max_len=20):
    model.eval()
    tokens = torch.tensor(src_tokenizer.encode(sentence)).unsqueeze(1)
    output = torch.tensor([1]).unsqueeze(1)  # <sos>
    for _ in range(max_len):
        with torch.no_grad():
            out = model(tokens, output)
        next_token = out.argmax(2)[-1, 0].item()
        output = torch.cat((output, torch.tensor([[next_token]])), dim=0)
        if next_token == 2:
            break
    return trg_tokenizer.decode(output.squeeze().tolist())

print(translate_sentence(model, "how are you", src_tokenizer, trg_tokenizer))