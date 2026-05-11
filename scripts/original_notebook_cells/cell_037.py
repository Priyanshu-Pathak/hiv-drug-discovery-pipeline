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
