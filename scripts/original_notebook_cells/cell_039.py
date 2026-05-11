import os,glob,subprocess
import pandas as pd
from rdkit import Chem
from rdkit.Chem import AllChem, rdMolDescriptors as rdMD

df=pd.read_csv(cand_csv_path).dropna(subset=["smiles"]).copy()

SDF_DIR=f"{LIG_DIR}_sdf";os.makedirs(SDF_DIR,exist_ok=True)

def largest_fragment(m):
    frags=Chem.GetMolFrags(m,asMols=True,sanitizeFrags=True)
    frags=[f for f in frags if f.GetNumAtoms()>0]
    if not frags:return None
    return max(frags,key=lambda x: x.GetNumHeavyAtoms())

def rdkit_3d(sm):
    m=Chem.MolFromSmiles(sm)
    if not m:return None
    m=Chem.AddHs(m)
    # embed
    if AllChem.EmbedMolecule(m,useRandomCoords=True,randomSeed=17)!=0:
        p=AllChem.ETKDGv3();p.randomSeed=17
        if AllChem.EmbedMolecule(m,p)!=0:return None
    # optimize
    try:
        if AllChem.MMFFHasAllMoleculeParams(m):
            AllChem.MMFFOptimizeMolecule(m,mmffVariant="MMFF94s",maxIters=200)
        else:
            AllChem.UFFOptimizeMolecule(m,maxIters=200)
    except: pass
    # keep largest fragment only
    m=largest_fragment(m)
    return m

# 1) Write one SDF per ligand with explicit Hs and 3D
sdf_paths=[]
for i,sm in enumerate(df["smiles"].tolist(),1):
    tag=f"lig_{i:04d}"
    m=rdkit_3d(sm)
    if not m:continue
    m.SetProp("_Name",tag)
    sdf_path=os.path.join(SDF_DIR,f"{tag}.sdf")
    w=Chem.SDWriter(sdf_path);w.write(m);w.close()
    sdf_paths.append((tag,sm,sdf_path))
print(f"SDF written: {len(sdf_paths)} / {len(df)}")

# 2) Convert SDF → PDBQT with OpenBabel (adds charges, hydrogens handled)
lig_files=[]
for tag,sm,sdf in sdf_paths:
    out_pdbqt=os.path.join(LIG_DIR,f"{tag}.pdbqt")
    cmd=f'obabel -isdf "{sdf}" -opdbqt -O "{out_pdbqt}" -p 7.4 --partialcharge gasteiger'
    r=subprocess.run(cmd,shell=True,stdout=subprocess.PIPE,stderr=subprocess.PIPE)
    if os.path.exists(out_pdbqt): lig_files.append((tag,sm,out_pdbqt))
print(f"PDBQT prepared: {len(lig_files)} / {len(sdf_paths)}")
