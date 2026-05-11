from rdkit import Chem
from rdkit.Chem import rdMolDescriptors as rdMD

def canon(s):
    try:
        m=Chem.MolFromSmiles(s,sanitize=True)
        if m is None:return None
        return Chem.MolToSmiles(m,canonical=True,isomericSmiles=True)
    except Exception:
        return None

def sa_score(m):
    try:
        return float(rdMD.CalcSyntheticAccessibilityScore(m))
    except Exception:
        nr=rdMD.CalcNumRings(m)
        nh=sum(a.GetAtomicNum() not in (1,6) for a in m.GetAtoms())
        return 3.0+0.15*nr+0.1*nh
