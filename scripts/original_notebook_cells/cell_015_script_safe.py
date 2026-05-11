# Extracted from the original Kaggle notebook.
# Notebook shell commands starting with ! are preserved as comments here.

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
