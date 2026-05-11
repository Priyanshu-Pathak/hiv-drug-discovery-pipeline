# Extracted from the original Kaggle notebook.
# Notebook shell commands starting with ! are preserved as comments here.



from rdkit import Chem
from rdkit.Chem import AllChem
import pandas as pd

# This line requires the file 'hiv_top200_diverse.csv'
# Make sure you have uploaded it or it is in the correct path.
try:
    df=pd.read_csv("hiv_top200_diverse.csv")
except FileNotFoundError:
    print("Error: 'hiv_top200_diverse.csv' not found. Please upload the file.")
    # We'll create an empty DataFrame to avoid crashing the rest of the script
    # but no output will be generated.
    df = pd.DataFrame(columns=["smiles", "qed", "sa", "molwt", "clogp"])


# --- Write SMI file ---
with open("hiv_top200_diverse.smi","w") as f:
    for sm in df["smiles"]:f.write(sm+"\n")

# --- Write SDF file with 3D coordinates ---
w=Chem.SDWriter("hiv_top200_diverse.sdf")
skipped_count = 0
for _,r in df.iterrows():
    m=Chem.MolFromSmiles(r["smiles"])
    if not m:continue
    
    m=Chem.AddHs(m)
    
    # --- FIX ---
    # Check the return value of EmbedMolecule.
    # It returns -1 on failure.
    embed_success = AllChem.EmbedMolecule(m,randomSeed=17)
    
    if embed_success == -1:
        # print(f"Warning: Could not embed molecule: {r['smiles']}")
        skipped_count += 1
        continue # Skip to the next molecule if embedding failed
        
    # Only optimize if embedding was successful
    # We also check the return value of optimization, 0 means success
    opt_success = AllChem.UFFOptimizeMolecule(m,maxIters=200)
    
    # Set properties and write to file
    m.SetProp("qed",str(r["qed"]))
    m.SetProp("sa",str(r["sa"]))
    m.SetProp("molwt",str(r["molwt"]))
    m.SetProp("clogp",str(r["clogp"]))
    
    # It's good practice to write the molecule only if optimization also succeeded
    if opt_success == 0:
        w.write(m)
    else:
        # print(f"Warning: Optimization failed for: {r['smiles']}")
        skipped_count += 1

w.close()

if skipped_count > 0:
    print(f"Warning: Skipped {skipped_count} molecules that failed to embed or optimize.")

print("Wrote hiv_top200_diverse.smi and hiv_top200_diverse.sdf")
