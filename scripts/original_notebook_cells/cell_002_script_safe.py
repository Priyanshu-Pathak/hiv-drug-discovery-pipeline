# Extracted from the original Kaggle notebook.
# Notebook shell commands starting with ! are preserved as comments here.


# NOTEBOOK SHELL: pip -q install selfies rdkit-pypi openpyxl
import os,json,random,torch,pandas as pd,numpy as np
from rdkit import Chem
import selfies as sf
from torch import nn
from torch.utils.data import Dataset,DataLoader
from tqdm import tqdm
import warnings;warnings.filterwarnings("ignore")
device=torch.device("cuda" if torch.cuda.is_available() else "cpu")

# -------- load Excel --------
# NOTE: The user provided a file, but the code uses a hardcoded path.
# We'll try the hardcoded path first, then search /kaggle/input if it fails.
path="/kaggle/input/zinc-dataset-capstone/ZINC_dataset.xlsx"
if not os.path.exists(path):
    # Fallback search if the specific path doesn't exist
    found = False
    for d in os.listdir("/kaggle/input"):
        try:
            for f in os.listdir(os.path.join("/kaggle/input",d)):
                if f.lower().endswith(".xlsx"):
                    path=os.path.join("/kaggle/input",d,f)
                    found = True
                    break
        except NotADirectoryError:
            continue # Skip files in /kaggle/input
        if found: break
    if not found:
        # As a final fallback, use the user-uploaded file's accessible name
        # This assumes the user-uploaded file is the correct one.
        path = "ZINC_dataset.xlsx - 250k_rndm_zinc_drugs_clean_3.csv"
        # Since the accessible name ends in .csv, we must treat it as CSV.
        # This contradicts the original code's .xlsx logic, but is necessary
        # if the .xlsx files are truly missing and the user's file is the input.
        
        # We will attempt to read as Excel first as the original code intended
        # and only switch to CSV if the extension is .csv
        if path.lower().endswith(".csv"):
             df=pd.read_csv(path)
        else:
             try:
                 df=pd.read_excel(path,engine="openpyxl")
             except Exception as e:
                 print(f"Failed to read Excel {path}: {e}. Trying as CSV.")
                 try:
                     # Try reading the user-uploaded CSV file by its accessible name
                     df = pd.read_csv("ZINC_dataset.xlsx - 250k_rndm_zinc_drugs_clean_3.csv")
                     print("Successfully loaded CSV file.")
                 except Exception as e_csv:
                     print(f"Failed to read CSV as well: {e_csv}")
                     raise FileNotFoundError("Could not find or read the input data file.")
else:
    # This block executes if the original hardcoded path was found
    try:
        df=pd.read_excel(path,engine="openpyxl")
    except Exception as e:
        print(f"Error reading {path}: {e}")
        # Add fallback to user's CSV if Excel read fails
        try:
            print("Trying user-uploaded CSV file as fallback...")
            df = pd.read_csv("ZINC_dataset.xlsx - 250k_rndm_zinc_drugs_clean_3.csv")
            print("Successfully loaded CSV file.")
        except Exception as e_csv:
            print(f"Failed to read CSV as well: {e_csv}")
            raise e # Re-raise the original Excel error

smicol=[c for c in df.columns if "smiles" in c.lower()][0]
raw_smiles=df[smicol].astype(str).str.strip()

# -------- canonicalize with RDKit --------
def canon(s):
    try:
        m=Chem.MolFromSmiles(s,sanitize=True)
        if m is None:return None
        return Chem.MolToSmiles(m,canonical=True,isomericSmiles=True)
    except:return None

smiles=[canon(s) for s in raw_smiles.tolist()]
smiles=[s for s in smiles if s]
print(f"After RDKit canonicalization: {len(smiles)}")

# -------- robust SELFIES encoder --------
def to_selfies(s):
    try:
        x=sf.encoder(s)
        if isinstance(x,str) and len(x)>0:return x
    except:pass
    try:
        x=sf.encoder(s,strict=False)
        if isinstance(x,str) and len(x)>0:return x
    except:pass
    return None

selfies=[to_selfies(s) for s in smiles]
bad_count=sum(1 for x in selfies if x is None)
selfies=[x for x in selfies if x]
print(f"SELFIES ok: {len(selfies)}   failed: {bad_count}")

if len(selfies)==0 and len(smiles) > 0:
    print("Examples causing failure (first 10):")
    for s in smiles[:10]:
        try:print(s, "->", sf.encoder(s))
        except Exception as e:print(s, "-> ERR:",repr(e))
    raise ValueError("All SMILES→SELFIES failed. Check environment and input.")
elif len(selfies) == 0:
    raise ValueError("No valid SMILES strings found to convert to SELFIES.")

# -------- tokenize + vocab --------
tok=[list(sf.split_selfies(s)) for s in selfies]
bos,eos,pad="<bos>","<eos>","<pad>"
vocab={pad:0,bos:1,eos:2}
for seq in tok:
    for t in seq:
        if t not in vocab:vocab[t]=len(vocab)
ivocab={i:t for t,i in vocab.items()}
lens=[len(s) for s in tok]
max_len=int(np.clip(np.percentile(lens,95)+2,32,256)) if lens else 128
print(f"max_len: {max_len} vocab_size: {len(vocab)}")

def enc(seq):
    ids=[vocab[bos]]+[vocab[t] for t in seq]+[vocab[eos]]
    return ids[:max_len] if len(ids)>max_len else ids
encd=[enc(s) for s in tok]

# -------- split + loaders --------
random.seed(42)
idx=list(range(len(encd)));random.shuffle(idx)
n=len(idx);tr=int(0.95*n);train_idx,valid_idx=idx[:tr],idx[tr:]

class SeqDS(Dataset):
    def __init__(self,ids):self.data=[encd[i] for i in ids]
    def __len__(self):return len(self.data)
    def __getitem__(self,i):
        x=self.data[i]+[vocab[pad]]*(max_len-len(self.data[i]))
        return torch.tensor(x[:-1]).long(),torch.tensor(x[1:]).long()

bs=128
train_dl=DataLoader(SeqDS(train_idx),batch_size=bs,shuffle=True,num_workers=2,pin_memory=True)
valid_dl=DataLoader(SeqDS(valid_idx),batch_size=bs,shuffle=False,num_workers=2,pin_memory=True)

# -------- model --------
class GPT(nn.Module):
    def __init__(self,vocab_size,d_model=512,nhead=8,depth=8,drop=0.1,maxlen=256):
        super().__init__()
        self.tok=nn.Embedding(vocab_size,d_model)
        self.pos=nn.Embedding(maxlen,d_model)
        layer=nn.TransformerEncoderLayer(d_model=d_model,nhead=nhead,dim_feedforward=4*d_model,dropout=drop,activation="gelu",batch_first=True)
        self.enc=nn.TransformerEncoder(layer,num_layers=depth)
        self.lm=nn.Linear(d_model,vocab_size)
        self.maxlen=maxlen
    def causal_mask(self,sz,dev):
        m=torch.full((sz,sz),float("-inf"),device=dev)
        return torch.triu(m,1)
    def forward(self,x):
        b,t=x.size()
        pos=torch.arange(t,device=x.device).unsqueeze(0).expand(b,t)
        h=self.tok(x)+self.pos(pos)
        h=self.enc(h,mask=self.causal_mask(t,x.device))
        return self.lm(h)

vsz=len(vocab)
model=GPT(vsz,maxlen=max_len).to(device)
opt=torch.optim.AdamW(model.parameters(),lr=3e-4,weight_decay=0.01)
sched=torch.optim.lr_scheduler.CosineAnnealingLR(opt,T_max=10)
lossf=nn.CrossEntropyLoss(ignore_index=vocab[pad])

# -------- train --------
best=float("inf");epochs=5
for ep in range(1,epochs+1):
    model.train();tr_loss=0
    # Wrap train_dl with tqdm for a progress bar
    for xb,yb in tqdm(train_dl,leave=False,desc=f"Epoch {ep} Train"):
        xb,yb=xb.to(device),yb.to(device);opt.zero_grad()
        y=model(xb);loss=lossf(y.view(-1,vsz),yb.view(-1))
        loss.backward();nn.utils.clip_grad_norm_(model.parameters(),1.0);opt.step()
        tr_loss+=loss.item()
    sched.step()
    model.eval();vl=0;c=0
    with torch.no_grad():
        for xb,yb in valid_dl:
            xb,yb=xb.to(device),yb.to(device)
            y=model(xb);vl+=lossf(y.view(-1,vsz),yb.view(-1)).item();c+=1
    vl/=max(1,c)
    print(f"epoch {ep} train_loss {tr_loss/len(train_dl):.4f} valid_loss {vl:.4f}")
    if vl<best:
        best=vl
        torch.save({"model":model.state_dict(),"vocab":vocab,"max_len":max_len},"selfies_transformer_pretrained.pt")

# -------- sampling --------
def sample(n=5,temperature=0.9,max_new_tokens=128):
    model.eval();res=[]
    for _ in range(n):
        x=torch.full((1,max_len),vocab[pad],dtype=torch.long,device=device);x[0,0]=vocab[bos]
        out=[]
        # Corrected loop: ensure i does not go out of bounds
        for i in range(min(max_new_tokens, max_len - 1)):
            logits=model(x)[:,i,:]/temperature
            probs=torch.softmax(logits,dim=-1)
            nxt=torch.multinomial(probs,1).item()
            if nxt==vocab[eos]:break
            # This check is technically redundant now but safe to keep
            if i+1<max_len:x[0,i+1]=nxt
            if nxt!=vocab[pad]:out.append(nxt)
        toks=[ivocab[i] for i in out]
        try:res.append(sf.decoder("".join(toks)))
        except:res.append("")
    return res

# --- Execute sampling and save outputs ---
try:
    samples_list = sample()
    if samples_list:
        pd.Series(samples_list).to_csv("pretrain_samples.csv",index=False, header=False)
    else:
        print("Sampling returned no results.")
        # Create an empty file to signal completion
        pd.Series([]).to_csv("pretrain_samples.csv",index=False, header=False)
        
    with open("selfies_vocab.json","w") as f:json.dump(vocab,f)
    print("saved: selfies_transformer_pretrained.pt, selfies_vocab.json, pretrain_samples.csv")

except Exception as e:
    print(f"An error occurred during sampling or saving: {e}")
    # Still try to save vocab
    try:
        with open("selfies_vocab.json","w") as f:json.dump(vocab,f)
        print("Warning: Sampling failed, but selfies_vocab.json was saved.")
    except Exception as e_json:
        print(f"Failed to save vocab file as well: {e_json}")
