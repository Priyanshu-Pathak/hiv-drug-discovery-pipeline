from rdkit.Chem import FilterCatalog

params=FilterCatalog.FilterCatalogParams()
params.AddCatalog(FilterCatalog.FilterCatalogParams.FilterCatalogs.PAINS_A)
params.AddCatalog(FilterCatalog.FilterCatalogParams.FilterCatalogs.PAINS_B)
params.AddCatalog(FilterCatalog.FilterCatalogParams.FilterCatalogs.PAINS_C)
params.AddCatalog(FilterCatalog.FilterCatalogParams.FilterCatalogs.BRENK)
cat=FilterCatalog.FilterCatalog(params)

def flagged(sm):
    m=Chem.MolFromSmiles(sm)
    return cat.GetFirstMatch(m) is not None if m else True

filt=pd.read_csv("hiv_gen_candidates.csv")
filt["flag"]=filt["smiles"].map(flagged)
clean=filt[~filt["flag"]].drop(columns=["flag"]).reset_index(drop=True)
print("candidates:",len(filt),"→ after PAINS/reactive removal:",len(clean))
clean.to_csv("hiv_gen_candidates_clean.csv",index=False)
