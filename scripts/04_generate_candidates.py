"""Stage 4: Load the fine-tuned checkpoint, run sampling sweep, generate molecules, and score/filter candidates.

Adapted from original notebook cells 013, 015, and 017.
"""
# Notebook shell command removed for script execution: !pip -q install rdkit-pypi selfies
import pandas as pd,numpy as np,torch,selfies as sf
from rdkit import Chem
from rdkit.Chem import QED,Descriptors,rdMolDescriptors as rdMD
from rdkit.Chem.Scaffolds import MurckoScaffold
from torch import nn

device=torch.device("cuda" if torch.cuda.is_available() else "cpu")

ck=torch.load("selfies_transformer_finetuned_hiv.pt",map_location=device)
vocab,max_len=ck["vocab"],ck["max_len"];ivocab={i:t for t,i in vocab.items()}
class GPT(nn.Module):
    def __init__(self,V,D=512,H=8,L=8,drop=0.1,M=256):
        super().__init__();self.tok=nn.Embedding(V,D);self.pos=nn.Embedding(M,D)
        layer=nn.TransformerEncoderLayer(D,H,4*D,drop,activation="gelu",batch_first=True)
        self.enc=nn.TransformerEncoder(layer,L);self.lm=nn.Linear(D,V);self.M=M
    def mask(self,T):m=torch.full((T,T),float("-inf"),device=device);return torch.triu(m,1)
    def forward(self,x):
        b,t=x.size();p=torch.arange(t,device=x.device).unsqueeze(0).expand(b,t)
        h=self.tok(x)+self.pos(p);h=self.enc(h,mask=self.mask(t));return self.lm(h)
model=GPT(len(vocab),M=max_len).to(device);model.load_state_dict(ck["model"]);model.eval()

train=pd.read_csv("chembl_hiv_protease_clean.csv")
train_smiles=set(train["smiles"].astype(str))


import math

def topk_filter(logits, k):
    if not k:
        return logits
    k = min(k, logits.size(-1))  # clamp to vocab size
    v, _ = torch.topk(logits, k)
    thr = v[..., -1, None]
    return torch.where(logits < thr, torch.full_like(logits, -1e9), logits)

@torch.no_grad()
def generate(n, temperature=0.8, top_k=80, max_new_tokens=128):
    out = []
    for _ in range(n):
        x = torch.full((1, max_len), vocab["<pad>"], dtype=torch.long, device=device)
        x[0, 0] = vocab["<bos>"]
        pos = 1
        tok = []
        for _ in range(max_new_tokens):
            logits = model(x[:, :pos])[:, -1, :] / max(1e-8, temperature)
            logits = topk_filter(logits, top_k)
            p = torch.softmax(logits, dim=-1)
            nxt = torch.multinomial(p, 1).item()
            if nxt == vocab["<eos>"] or pos >= max_len:
                break
            x[0, pos] = nxt
            pos += 1
            if nxt != vocab["<pad>"]:
                tok.append(nxt)
        try:
            out.append(sf.decoder("".join(ivocab[i] for i in tok)))
        except:
            out.append("")
    return out

def validity_unique(smiles_list):
    val = [s for s in smiles_list if isinstance(s, str) and len(s) > 3 and Chem.MolFromSmiles(s)]
    return (
        100 * len(val) / max(1, len(smiles_list)),
        100 * len(set(val)) / max(1, len(val)),
        len(val)
    )

grid = [(t, k) for t in [0.7, 0.8, 0.9, 1.0] for k in [40, 60, 80, 120]]
records = []

for t, k in grid:
    print(f"Running temp={t}, top_k={k}...")
    s = generate(400, temperature=t, top_k=k, max_new_tokens=128)
    v, u, n = validity_unique(s)
    records.append({"temperature": t, "top_k": k, "valid%": v, "unique%": u, "n_valid": n})

sweep = pd.DataFrame(records).sort_values(["valid%", "unique%", "n_valid"], ascending=[False, False, False])
sweep.to_csv("sampling_sweep.csv", index=False)
print(sweep.head(8))

best_t = float(sweep.iloc[0]["temperature"])
best_k = int(sweep.iloc[0]["top_k"])
print(f"best settings → temperature={best_t}, top_k={best_k}")


def sa_score(m):
    try:return float(rdMD.CalcSyntheticAccessibilityScore(m))
    except:
        nr=rdMD.CalcNumRings(m);nh=sum(a.GetAtomicNum() not in (1,6) for a in m.GetAtoms())
        return 3.0+0.15*nr+0.1*nh

s=generate(5000,temperature=best_t,top_k=best_k,max_new_tokens=128)
valid=[Chem.MolFromSmiles(x) for x in s if isinstance(x,str) and Chem.MolFromSmiles(x)]
rows=[]
for m in valid:
    try:
        rows.append({"smiles":Chem.MolToSmiles(m,canonical=True,isomericSmiles=True),
                     "qed":QED.qed(m),"sa":sa_score(m),
                     "molwt":Descriptors.MolWt(m),"clogp":Descriptors.MolLogP(m)})
    except: pass
gen=pd.DataFrame(rows).drop_duplicates(subset=["smiles"]).reset_index(drop=True)
gen.to_csv("hiv_gen_scored.csv",index=False)
filt=gen.query("qed>=0.5 and sa<=5.0 and 200<=molwt<=800").copy()
filt.to_csv("hiv_gen_candidates.csv",index=False)
print(len(s),"generated;",len(valid),"valid;",len(gen),"unique valid;",len(filt),"passed filter")
