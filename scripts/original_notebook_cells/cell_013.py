!pip -q install rdkit-pypi selfies
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
