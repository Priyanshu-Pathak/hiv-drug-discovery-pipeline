# Extracted from the original Kaggle notebook.
# Notebook shell commands starting with ! are preserved as comments here.

from rdkit import Chem
import pandas as pd, torch, selfies as sf

def top_k_filter(logits, k=0):
    if k and k < logits.size(-1):
        v,_=torch.topk(logits,k)
        thresh=v[..., -1, None]
        logits=torch.where(logits<thresh, torch.full_like(logits, -1e9), logits)
    return logits

def gen(n=500, temperature=0.8, max_new_tokens=128, top_k=0):
    model.eval(); res=[]
    for _ in range(n):
        # init sequence: [<bos>, <pad>, ..., <pad>]
        x=torch.full((1, max_len), vocab["<pad>"], dtype=torch.long, device=device)
        x[0,0]=vocab["<bos>"]
        pos=1
        out_ids=[]
        for _ in range(max_new_tokens):
            # run model only on the prefix actually filled
            logits=model(x[:,:pos])[:,-1,:] / max(1e-8, temperature)
            logits=top_k_filter(logits, top_k)
            probs=torch.softmax(logits, dim=-1)
            nxt=torch.multinomial(probs, 1).item()
            if nxt==vocab["<eos>"] or pos>=max_len: break
            x[0,pos]=nxt
            if nxt!=vocab["<pad>"]: out_ids.append(nxt)
            pos+=1
        toks=[ivocab[i] for i in out_ids]
        try:
            res.append(sf.decoder("".join(toks)))
        except:
            res.append("")
    return res

s=gen(500, temperature=0.8, max_new_tokens=128, top_k=50)  # top_k optional for cleaner samples
pd.Series(s).to_csv("pretrain_samples_500.csv", index=False)
print("Generated 500 molecules → pretrain_samples_500.csv")
