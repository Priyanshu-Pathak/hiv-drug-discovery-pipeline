import pandas as pd
import matplotlib.pyplot as plt
import seaborn as sns
import numpy as np

# Set a plotting style
sns.set(style="whitegrid")

# 1. Load your DE NOVO (AI-generated) valid scores
denovo_df = pd.read_csv("/kaggle/input/baseline-model/vina_scores_sorted.csv") # This is your 84.5-min run

# 2. Load the KNOWN DRUGS (your fine-tuning set)
baseline_df = pd.read_csv("/kaggle/input/baseline-model/chembl_hiv_protease_clean.csv")

# 3. Create the comparison plot
plt.figure(figsize=(10, 6))

# Plot a density plot (KDE) for your AI-generated scores
sns.kdeplot(denovo_df['affinity'], label="De Novo AI Molecules (Predicted Affinity)", 
            color='blue', fill=True)

# Vina scores for known drugs are often in a similar range.
# We can't plot the baseline's 'pchembl' on the same axis as 'affinity' 
# because the units are different.
# So, we just analyze the stats for our paper.

plt.title('Distribution of Predicted Binding Affinities for De Novo Molecules')
plt.xlabel('Vina Docking Score (kcal/mol)')
plt.ylabel('Density')
plt.legend()
plt.show()

# 4. Print the "Baseline" statistics for your paper
print("--- De Novo (AI) Results ---")
print(denovo_df['affinity'].describe().round(3))

print("\n--- Baseline (Known Drugs) Results ---")
print(baseline_df['pchembl'].describe().round(3))