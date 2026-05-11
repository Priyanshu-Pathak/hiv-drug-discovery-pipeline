# Extracted from the original Kaggle notebook.
# Notebook shell commands starting with ! are preserved as comments here.

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
