"""Quickly summarize the checked-in result CSV files without rerunning training or docking."""
from pathlib import Path
import pandas as pd

base=Path(__file__).resolve().parents[1]
paths={
    "generated_scored":base/"results/candidates/hiv_gen_scored.csv",
    "novelty":base/"results/candidates/hiv_gen_scored_novelty.csv",
    "clean_candidates":base/"results/candidates/hiv_gen_candidates_clean.csv",
    "top200":base/"results/candidates/hiv_top200_diverse.csv",
    "vina":base/"results/docking/vina_scores_sorted.csv",
}
for k,p in paths.items():
    if p.exists():
        df=pd.read_csv(p)
        print(f"{k}: {len(df)} rows | {p.relative_to(base)}")
    else:
        print(f"missing: {p.relative_to(base)}")
if paths["vina"].exists():
    v=pd.read_csv(paths["vina"]).dropna(subset=["affinity"]).sort_values("affinity")
    print("\nTop docking hits:")
    print(v.head(10).to_string(index=False))
    print("\nAffinity summary:")
    print(v["affinity"].describe().round(3).to_string())
