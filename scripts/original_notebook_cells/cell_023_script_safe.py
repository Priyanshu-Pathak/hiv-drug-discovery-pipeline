# Extracted from the original Kaggle notebook.
# Notebook shell commands starting with ! are preserved as comments here.

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
