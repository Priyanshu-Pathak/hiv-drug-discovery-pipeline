import pandas as pd
from rdkit import Chem
from rdkit.Chem.Scaffolds import MurckoScaffold

train=pd.read_csv("chembl_hiv_protease_clean.csv")
train_smiles=set(train["smiles"].astype(str))

gen=pd.read_csv("hiv_gen_scored.csv")
gen["novel"]=~gen["smiles"].isin(train_smiles)

def scaffold(sm):
    m=Chem.MolFromSmiles(sm)
    return MurckoScaffold.MurckoScaffoldSmiles(mol=m,includeChirality=True) if m else None

gen["scaffold"]=gen["smiles"].map(scaffold)
novel_rate=100*gen["novel"].mean()
n_scaf=gen["scaffold"].dropna().nunique()
print(f"novel%: {novel_rate:.2f} | unique scaffolds: {n_scaf}")
gen.to_csv("hiv_gen_scored_novelty.csv",index=False)
