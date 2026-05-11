import torch
from torch import nn

class GPT(nn.Module):
    """Decoder-only Transformer used in the original notebook."""
    def __init__(self,V,D=512,H=8,L=8,drop=0.1,M=256):
        super().__init__()
        self.tok=nn.Embedding(V,D)
        self.pos=nn.Embedding(M,D)
        layer=nn.TransformerEncoderLayer(D,H,4*D,drop,activation="gelu",batch_first=True)
        self.enc=nn.TransformerEncoder(layer,L)
        self.lm=nn.Linear(D,V)
        self.M=M
    def mask(self,T,device):
        m=torch.full((T,T),float("-inf"),device=device)
        return torch.triu(m,1)
    def forward(self,x):
        b,t=x.size()
        p=torch.arange(t,device=x.device).unsqueeze(0).expand(b,t)
        h=self.tok(x)+self.pos(p)
        h=self.enc(h,mask=self.mask(t,x.device))
        return self.lm(h)
