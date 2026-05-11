# Extracted from the original Kaggle notebook.
# Notebook shell commands starting with ! are preserved as comments here.

# NOTEBOOK SHELL: pip -q install selfies
import selfies as sf, pandas as pd

df = pd.read_csv("chembl_hiv_protease_clean.csv")
def to_self(s):
    try:
        return sf.encoder(s)
    except:
        return None

df["selfies"] = df["smiles"].apply(to_self)
df = df.dropna(subset=["selfies"]).reset_index(drop=True)
print("Valid SELFIES:", len(df))
df.to_csv("chembl_hiv_protease_selfies.csv", index=False)
print("Saved -> chembl_hiv_protease_selfies.csv")
