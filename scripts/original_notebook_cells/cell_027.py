import pandas as pd
from rdkit import Chem
from rdkit.Chem.Scaffolds import MurckoScaffold

gen=pd.read_csv("hiv_gen_scored.csv")
cand=pd.read_csv("hiv_gen_candidates_clean.csv")
top=pd.read_csv("hiv_top200_diverse.csv")

def scaffold(s):
    m=Chem.MolFromSmiles(s); 
    return MurckoScaffold.MurckoScaffoldSmiles(mol=m,includeChirality=True) if m else None

gen["scaffold"]=gen["smiles"].map(scaffold)
cand["scaffold"]=cand["smiles"].map(scaffold)
top["scaffold"]=top["smiles"].map(scaffold)

print("Generated valid unique:",len(gen))
print("Filtered candidates:",len(cand))
print("Top diverse:",len(top))
print("Scaffolds (gen/cand/top):",gen["scaffold"].nunique(),cand["scaffold"].nunique(),top["scaffold"].nunique())
print("QED mean (gen/cand/top):",gen["qed"].mean(),cand["qed"].mean(),top["qed"].mean())
print("SA  mean (gen/cand/top):",gen["sa"].mean(),cand["sa"].mean(),top["sa"].mean())
