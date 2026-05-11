import torch
import selfies as sf

def topk_filter(logits,k):
    if not k:
        return logits
    k=min(k,logits.size(-1))
    v,_=torch.topk(logits,k)
    thr=v[...,-1,None]
    return torch.where(logits<thr,torch.full_like(logits,-1e9),logits)

@torch.no_grad()
def generate(model,vocab,ivocab,max_len,device,n,temperature=0.8,top_k=80,max_new_tokens=128):
    out=[]
    model.eval()
    for _ in range(n):
        x=torch.full((1,max_len),vocab["<pad>"],dtype=torch.long,device=device)
        x[0,0]=vocab["<bos>"]
        pos=1
        tok=[]
        for _ in range(max_new_tokens):
            logits=model(x[:,:pos])[:,-1,:]/max(1e-8,temperature)
            logits=topk_filter(logits,top_k)
            p=torch.softmax(logits,dim=-1)
            nxt=torch.multinomial(p,1).item()
            if nxt==vocab["<eos>"] or pos>=max_len:
                break
            x[0,pos]=nxt
            pos+=1
            if nxt!=vocab["<pad>"]:
                tok.append(nxt)
        try:
            out.append(sf.decoder("".join(ivocab[i] for i in tok)))
        except Exception:
            out.append("")
    return out
