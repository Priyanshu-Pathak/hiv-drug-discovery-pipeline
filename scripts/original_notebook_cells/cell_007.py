!pip -q install rdkit-pypi openpyxl
import os,re,pandas as pd,numpy as np
from rdkit import Chem
from rdkit.Chem import rdMolDescriptors as rdMD
from rdkit.Chem.inchi import MolToInchiKey

p="/kaggle/input/finetuningthemodel/chembl_9000_hiv _entries.xlsx"
df=pd.read_excel(p,engine="openpyxl")
df.columns=[str(c).strip() for c in df.columns]

def find(df,keys):
    n=lambda s:re.sub(r'[^a-z0-9]+','',s.lower())
    nk=[n(k) for k in keys]
    for c in df.columns:
        if n(str(c)) in nk:return c
    for c in df.columns:
        if any(k in n(str(c)) for k in nk):return c
    return None

col_mid =find(df,["Molecule ChEMBL ID"])
col_smi =find(df,["Smiles","Canonical Smiles"])
col_pch =find(df,["pChEMBL Value"])
col_val =find(df,["Standard Value"])
col_uni =find(df,["Standard Units"])
col_tn  =find(df,["Target Name"])
col_tid =find(df,["Target ChEMBL ID"])

X=df[[c for c in [col_mid,col_smi,col_pch,col_val,col_uni,col_tn,col_tid] if c]].copy()
X.columns=["chembl_id","smiles","pchembl","std_value","std_units","target_name","target_id"][:X.shape[1]]

# 1) Filter to HIV sensibly
if "target_id" in X.columns and X["target_id"].notna().any():
    m=X["target_id"].astype(str).str.upper().str.contains("CHEMBL243")
else:
    tn=X.get("target_name",pd.Series("",index=X.index)).astype(str)
    m=tn.str.contains("hiv",case=False,na=False)&tn.str.contains("protease",case=False,na=False)
    if m.sum()==0:m=tn.str.contains("hiv",case=False,na=False)
X=X[m].copy()
print("rows after HIV filter:",len(X))

# 2) Sanitize SMILES: strip quotes/whitespace; drop true NaNs and literal "nan"/"None"
def clean_smiles(s):
    if pd.isna(s):return None
    s=str(s).strip().strip('"').strip("'").strip()
    if s=="" or s.lower() in {"nan","none","null"}:return None
    # remove non-printables
    s=re.sub(r'[\x00-\x1f\x7f]','',s)
    return s

X["smiles"]=X["smiles"].map(clean_smiles)
X=X.dropna(subset=["smiles"]).copy()
print("rows after SMILES clean:",len(X))

# 3) Canonicalize
def canon(s):
    try:
        m=Chem.MolFromSmiles(s)
        return Chem.MolToSmiles(m,canonical=True,isomericSmiles=True) if m else None
    except:return None
X["smiles"]=X["smiles"].map(canon)
X=X.dropna(subset=["smiles"]).copy()
print("rows after RDKit parse:",len(X))

# 4) pChEMBL numeric and backfill from value+units
X["pchembl"]=pd.to_numeric(X.get("pchembl"),errors="coerce")
X["std_value"]=pd.to_numeric(X.get("std_value"),errors="coerce")
def to_pchembl(v,u):
    if pd.isna(v) or pd.isna(u):return np.nan
    try:v=float(v)
    except:return np.nan
    u=str(u).lower()
    if "nm" in u:return 9-np.log10(v)
    if ("µm" in u) or ("um" in u) or ("μm" in u):return 9-np.log10(v*1e3)
    if "mm" in u:return 9-np.log10(v*1e6)
    if u in ("m","mol/l"):return 9-np.log10(v*1e9)
    return np.nan
if "std_value" in X.columns and "std_units" in X.columns:
    miss=X["pchembl"].isna() & X["std_value"].notna() & X["std_units"].notna()
    if miss.any():
        X.loc[miss,"pchembl"]=X.loc[miss,["std_value","std_units"]].apply(lambda r:to_pchembl(r["std_value"],r["std_units"]),axis=1)
print("pChEMBL present:",X["pchembl"].notna().sum(),"/",len(X))

# 5) InChIKey using RDKit InChI (more robust than rdMD CalcInchiKey in some builds)
def ik(sm):
    try:
        m=Chem.MolFromSmiles(sm)
        return MolToInchiKey(m) if m else None
    except:return None

X["inchikey"]=X["smiles"].map(ik)
X=X.dropna(subset=["inchikey"]).copy()
print("rows with inchikey:",len(X))

# 6) Dedupe by structure; keep best potency if available
X=X.sort_values("pchembl",ascending=False,na_position="last")\
     .groupby("inchikey",as_index=False)\
     .agg({"smiles":"first","pchembl":"max","chembl_id":"first"})
print("rows after dedupe:",len(X))
if X["pchembl"].notna().any():print(X["pchembl"].describe())

X.to_csv("chembl_hiv_protease_clean.csv",index=False)
print("saved -> chembl_hiv_protease_clean.csv")
