def sa_score(m):
    try:return float(rdMD.CalcSyntheticAccessibilityScore(m))
    except:
        nr=rdMD.CalcNumRings(m);nh=sum(a.GetAtomicNum() not in (1,6) for a in m.GetAtoms())
        return 3.0+0.15*nr+0.1*nh

s=generate(5000,temperature=best_t,top_k=best_k,max_new_tokens=128)
valid=[Chem.MolFromSmiles(x) for x in s if isinstance(x,str) and Chem.MolFromSmiles(x)]
rows=[]
for m in valid:
    try:
        rows.append({"smiles":Chem.MolToSmiles(m,canonical=True,isomericSmiles=True),
                     "qed":QED.qed(m),"sa":sa_score(m),
                     "molwt":Descriptors.MolWt(m),"clogp":Descriptors.MolLogP(m)})
    except: pass
gen=pd.DataFrame(rows).drop_duplicates(subset=["smiles"]).reset_index(drop=True)
gen.to_csv("hiv_gen_scored.csv",index=False)
filt=gen.query("qed>=0.5 and sa<=5.0 and 200<=molwt<=800").copy()
filt.to_csv("hiv_gen_candidates.csv",index=False)
print(len(s),"generated;",len(valid),"valid;",len(gen),"unique valid;",len(filt),"passed filter")
