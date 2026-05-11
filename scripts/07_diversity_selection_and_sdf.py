"""Stage 7: Diversity selection using Morgan fingerprints + Butina clustering and export SMI/SDF.

Adapted from original notebook cells 023, 025, and 027.
"""
from rdkit import Chem
from rdkit.Chem import AllChem, DataStructs
from rdkit.ML.Cluster import Butina
import pandas as pd

def mfp(sm):
    m = Chem.MolFromSmiles(sm)
    return AllChem.GetMorganFingerprintAsBitVect(m, 2, 2048) if m else None

cand = pd.read_csv("hiv_gen_candidates_clean.csv")
cand["fp"] = cand["smiles"].map(mfp)

fps = [fp for fp in cand["fp"] if fp is not None]
smiles_seq = [s for s, fp in zip(cand["smiles"], cand["fp"]) if fp is not None]

# compute distance data for Butina
dists = []
for i in range(1, len(fps)):
    sims = DataStructs.BulkTanimotoSimilarity(fps[i], fps[:i])
    dists.extend([1 - x for x in sims])

# use positional args for compatibility
clusters = Butina.ClusterData(dists, len(fps), 0.3, True)

# pick one representative per cluster (prioritize high QED, low SA)
ranked = cand.set_index("smiles").loc[smiles_seq].reset_index()
ranked = ranked.sort_values(["qed", "sa"], ascending=[False, True])

seen = set()
reps = []
for cl in clusters:
    for sm in ranked["smiles"]:
        if sm in seen:
            continue
        idx = ranked.index[ranked["smiles"] == sm][0]
        if idx in cl:
            reps.append(sm)
            seen.add(sm)
            break

top_diverse = ranked[ranked["smiles"].isin(reps)].head(200).copy()
top_diverse.to_csv("hiv_top200_diverse.csv", index=False)
print("diverse representatives:", len(top_diverse))


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
