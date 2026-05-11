"""Stage 8: Prepare receptor, ligands, grid center, SDF/PDBQT files for docking.

Adapted from original notebook cells 033, 035, 037, and 039.
The original notebook used shell magics for Open Babel. This script uses subprocess for the same command.
"""
import os,glob,shutil,math,json,random
import pandas as pd
from rdkit import Chem
from rdkit.Chem import AllChem
from rdkit.Chem import rdMolAlign
from meeko import MoleculePreparation, PDBQTMolecule
from vina import Vina

# === REQUIRED USER FILES (upload to /kaggle/input/<dataset>/) ===
# receptor PDB (protein only, no ligand/waters if possible). Example filename:
RECEPTOR_PDB = "1HVR.pdb"          # e.g., 1HVR protein cleaned
# optional reference ligand to auto-center box (PDB or SDF). If None, set box manually:
REFERENCE_LIG = "1hvr_C_XK2.sdf"           # or "ref_ligand.pdb" or None

# Grid box size in Angstroms (will be used with auto-centered or manual center):
BOX_SIZE = (20.0,20.0,20.0)

# If no reference ligand, set manual box center (x,y,z):
MANUAL_CENTER = (0.0,0.0,0.0)              # replace if REFERENCE_LIG=None

# Candidates file from previous step:
CAND_FILE = "hiv_top200_diverse.csv"       # must contain a 'smiles' column

# Work dirs
WORK="dockrun"; os.makedirs(WORK,exist_ok=True)
LIG_DIR=f"{WORK}/ligands";os.makedirs(LIG_DIR,exist_ok=True)
OUT_DIR=f"{WORK}/out";os.makedirs(OUT_DIR,exist_ok=True)

# Locate uploaded files inside /kaggle/input/*
def find_in_kaggle(name):
    for d in glob.glob("/kaggle/input/*"):
        p=os.path.join(d,name)
        if os.path.exists(p):return p
    return name if os.path.exists(name) else None

receptor_pdb_path=find_in_kaggle(RECEPTOR_PDB)
ref_lig_path=find_in_kaggle(REFERENCE_LIG) if REFERENCE_LIG else None
cand_csv_path=find_in_kaggle(CAND_FILE)
assert receptor_pdb_path, "Receptor PDB not found."
assert cand_csv_path, "Candidates CSV not found."
print("Receptor:",receptor_pdb_path)
print("Ref ligand:",ref_lig_path)
print("Candidates:",cand_csv_path)


# --- Clean receptor and convert PDB to PDBQT ---
import os, subprocess
print(f"Original receptor PDB: {receptor_pdb_path}")
CHAINS_TO_KEEP={'A','B'}
print(f"Will keep only chains: {CHAINS_TO_KEEP}")
clean_pdb_renumbered=os.path.join(WORK,"receptor_protein_AB_renumbered.pdb")
atom_counter=1
with open(receptor_pdb_path) as f_in, open(clean_pdb_renumbered,"w") as f_out:
    for line in f_in:
        if line.startswith("ATOM"):
            chain_id=line[21]
            if chain_id in CHAINS_TO_KEEP:
                new_atom_line=line[:6]+str(atom_counter).rjust(5)+line[11:]
                f_out.write(new_atom_line)
                atom_counter+=1
        elif line.startswith("TER"):
            chain_id=line[21]
            if chain_id in CHAINS_TO_KEEP:
                f_out.write(line)
        elif line.startswith("END"):
            f_out.write(line)
print(f"Wrote new clean, renumbered PDB with {atom_counter-1} atoms: {clean_pdb_renumbered}")
receptor_pdbqt=os.path.join(WORK,"receptor.pdbqt")
cmd=["obabel","-ipdb",clean_pdb_renumbered,"-opdbqt","-O",receptor_pdbqt,"-xr","-p","7.4","--partialcharge","gasteiger"]
print("Running:"," ".join(cmd))
subprocess.run(cmd,check=True)
assert os.path.exists(receptor_pdbqt),"PDBQT file was not created!"
print(f"Wrote clean receptor PDBQT: {receptor_pdbqt}")
from rdkit import Chem

def center_from_ref(path):
    mol=None
    if path and path.lower().endswith(".sdf"):
        sup=Chem.SDMolSupplier(path,removeHs=False)
        mol=sup[0] if len(sup)>0 else None
    elif path and path.lower().endswith(".pdb"):
        mol=Chem.MolFromPDBFile(path,removeHs=False)
    if not mol:return None
    c=mol.GetConformer()
    xs=[c.GetAtomPosition(i).x for i in range(mol.GetNumAtoms())]
    ys=[c.GetAtomPosition(i).y for i in range(mol.GetNumAtoms())]
    zs=[c.GetAtomPosition(i).z for i in range(mol.GetNumAtoms())]
    return (float(sum(xs)/len(xs)),float(sum(ys)/len(ys)),float(sum(zs)/len(zs)))

ctr=center_from_ref(ref_lig_path) if ref_lig_path else None
if ctr is None: ctr=MANUAL_CENTER
print("Grid center:",ctr," Box size:",BOX_SIZE)


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
