"""Stage 3: Fine-tune the pretrained SELFIES-Transformer on curated HIV-1 protease inhibitors.

Adapted from original notebook cell 011.
"""
import json, random, torch, selfies as sf, pandas as pd
from torch import nn
from torch.utils.data import Dataset, DataLoader

device = torch.device("cuda" if torch.cuda.is_available() else "cpu")

# load checkpoint
ck = torch.load("selfies_transformer_pretrained.pt", map_location=device)
vocab, max_len = ck["vocab"], ck["max_len"]
ivocab = {i: t for t, i in vocab.items()}

# reload model definition
class GPT(nn.Module):
    def __init__(self, V, D=512, H=8, L=8, drop=0.1, M=256):
        super().__init__()
        self.tok = nn.Embedding(V, D)
        self.pos = nn.Embedding(M, D)
        layer = nn.TransformerEncoderLayer(D, H, 4 * D, drop, activation="gelu", batch_first=True)
        self.enc = nn.TransformerEncoder(layer, L)
        self.lm = nn.Linear(D, V)
        self.M = M
    def mask(self, T, dev):
        m = torch.full((T, T), float("-inf"), device=dev)
        return torch.triu(m, 1)
    def forward(self, x):
        b, t = x.size()
        p = torch.arange(t, device=x.device).unsqueeze(0).expand(b, t)
        h = self.tok(x) + self.pos(p)
        h = self.enc(h, mask=self.mask(t, x.device))
        return self.lm(h)

model = GPT(len(vocab), M=max_len).to(device)
model.load_state_dict(ck["model"])

# load data
df = pd.read_csv("chembl_hiv_protease_selfies.csv")
tok = [list(sf.split_selfies(s)) for s in df["selfies"].tolist()]
def encode(seq):
    ids = [vocab["<bos>"]] + [vocab[t] for t in seq if t in vocab] + [vocab["<eos>"]]
    return ids[:max_len] if len(ids) > max_len else ids

enc = [encode(s) for s in tok if len(s) > 0]
random.shuffle(enc)
split = int(0.9 * len(enc))
train, valid = enc[:split], enc[split:]

class DS(Dataset):
    def __init__(self, data): self.data = data
    def __len__(self): return len(self.data)
    def __getitem__(self, i):
        x = self.data[i] + [vocab["<pad>"]] * (max_len - len(self.data[i]))
        return torch.tensor(x[:-1]), torch.tensor(x[1:])

train_dl = DataLoader(DS(train), batch_size=64, shuffle=True)
valid_dl = DataLoader(DS(valid), batch_size=64)

opt = torch.optim.AdamW(model.parameters(), lr=1e-4)
lossf = nn.CrossEntropyLoss(ignore_index=vocab["<pad>"])

best = float("inf")
for ep in range(1, 10):
    model.train(); total = 0
    for xb, yb in train_dl:
        xb, yb = xb.to(device), yb.to(device)
        opt.zero_grad()
        out = model(xb)
        loss = lossf(out.view(-1, len(vocab)), yb.view(-1))
        loss.backward()
        torch.nn.utils.clip_grad_norm_(model.parameters(), 1.0)
        opt.step()
        total += loss.item()
    model.eval(); val = 0; c = 0
    with torch.no_grad():
        for xb, yb in valid_dl:
            xb, yb = xb.to(device), yb.to(device)
            val += lossf(model(xb).view(-1, len(vocab)), yb.view(-1)).item()
            c += 1
    val /= c
    print(f"epoch {ep}: val_loss={val:.4f}")
    if val < best:
        best = val
        torch.save({"model": model.state_dict(), "vocab": vocab, "max_len": max_len}, "selfies_transformer_finetuned_hiv.pt")

print("Fine-tuned model saved → selfies_transformer_finetuned_hiv.pt")
