# Note

These datasets and structural files are included for research and educational purposes only. 

# Raw Data Files

This folder contains the raw datasets and structural files required to reproduce the HIV-1 protease inhibitor discovery pipeline.


## Files

### 1HVR.pdb
HIV-1 protease receptor structure used for molecular docking.

### 1hvr_C_XK2.sdf
Reference co-crystallized ligand extracted from the 1HVR complex.

### 250k_rndm_zinc_drugs_clean_3.csv
Subset of ZINC molecules used for Transformer pretraining and molecular grammar learning.

### chembl_9000_entries.csv
Curated ChEMBL molecular records.

### chembl_9000_hiv_entries.xlsx
Filtered HIV-1 protease inhibitor dataset used for HIV-specific fine-tuning.

---

## Data Sources

### ZINC Dataset
Kaggle mirror:
https://www.kaggle.com/datasets/basu369victor/zinc250k

### PDB Structure
RCSB PDB:
https://www.rcsb.org/structure/1HVR

### ChEMBL Query
Target ChEMBL ID used for HIV-1 protease retrieval:

```json
{
  "query": {
    "bool": {
      "must": [
        {
          "query_string": {
            "query": "target_chembl_id:CHEMBL243"
          }
        }
      ]
    }
  }
}