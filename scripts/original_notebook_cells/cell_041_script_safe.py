# Extracted from the original Kaggle notebook.
# Notebook shell commands starting with ! are preserved as comments here.

import os, gc, csv, time, traceback
import pandas as pd
import numpy as np
from vina import Vina
from concurrent.futures import ProcessPoolExecutor, as_completed
from tqdm import tqdm

# --- These functions MUST be defined at the top level ---

def make_vina():
    """Creates, configures, and returns a Vina object with maps."""
    # receptor_pdbqt, ctr, and BOX_SIZE are global
    v = Vina(sf_name='vina', seed=17)
    try: v.set_verbosity(0)  # Silence Vina's internal progress
    except: pass
    v.set_receptor(receptor_pdbqt)
    v.compute_vina_maps(center=ctr, box_size=BOX_SIZE)
    return v

def safe_affinity(v):
    """Safely extracts the affinity score from Vina's output."""
    try:
        sc = v.score()
        arr = np.array(sc, dtype=float).ravel()
        return float(arr[0]) if arr.size else np.nan
    except Exception:
        return np.nan

def run_docking_task(job_data):
    """
    This is the function that each of our 4 parallel workers will run.
    It docks one ligand and returns the results.
    """
    # 1. Set thread limits *inside* the worker
    os.environ.update({
        "OMP_NUM_THREADS": "1",
        "OPENBLAS_NUM_THREADS": "1",
        "MKL_NUM_THREADS": "1",
        "NUMEXPR_NUM_THREADS": "1"
    })
    
    tag, sm, pq = job_data
    
    try:
        # 2. Create a Vina instance (computes maps)
        v = make_vina()
        
        # 3. Define output path
        out_best = os.path.join(OUT_DIR, f"{tag}_best.pdbqt")

        # 4. Run docking
        v.set_ligand_from_file(pq)
        v.dock(exhaustiveness=EXH, n_poses=NMODES)
        v.write_poses(out_best, n_poses=1, overwrite=True)

        # 5. Get score
        aff = safe_affinity(v)
        
        # 6. Return success
        return {"tag": tag, "smiles": sm, "affinity": aff, "pose": out_best, "status": "success"}

    except Exception as e:
        # 7. Return failure
        return {"tag": tag, "smiles": sm, "affinity": np.nan, "pose": "", "status": "failed", "error": str(e)}

# --- This 'if' block is REQUIRED for multiprocessing in a notebook ---
if __name__ == "__main__":
    
    # --- 1. Configuration ---
    N_WORKERS = 4      # Number of parallel jobs. (4 * 3GB = 12GB RAM)
    EXH = 8            # Exhaustiveness (8 is good, 4 is faster)
    NMODES = 1
    out_csv = "vina_scores_fixed.csv"

    # --- 2. Assertions (make sure variables from other cells exist) ---
    try:
        assert os.path.exists(receptor_pdbqt)
        assert isinstance(ctr, tuple) and len(ctr) == 3
        assert len(lig_files) > 0
    except NameError:
        print("ERROR: 'receptor_pdbqt', 'ctr', or 'lig_files' not defined.")
        print("Please re-run the previous setup cells.")
        # This will stop the script if run in a fresh kernel
        # You can remove this 'try/except' if you are sure you re-ran them
    
    print(f"Starting parallel docking for {len(lig_files)} ligands using {N_WORKERS} workers...")
    start_time = time.time()
    
    # --- 3. Write CSV Header ---
    with open(out_csv, "w", newline="") as f:
        csv.writer(f).writerow(["tag", "smiles", "affinity", "pose"])

    # --- 4. Run Parallel Docking with Progress Bar ---
    # We use ProcessPoolExecutor for true CPU parallelism
    with ProcessPoolExecutor(max_workers=N_WORKERS) as executor:
        
        # Submit all jobs to the pool
        futures = [executor.submit(run_docking_task, job) for job in lig_files]
        
        # Use tqdm to create a progress bar as jobs are completed
        for future in tqdm(as_completed(futures), total=len(lig_files), desc="Docking Ligands"):
            
            result = future.result()
            
            # Write the result to the CSV *immediately* as it finishes
            with open(out_csv, "a", newline="") as f:
                csv.writer(f).writerow([
                    result["tag"], 
                    result["smiles"], 
                    result["affinity"], 
                    result["pose"]
                ])
                
            if result["status"] == "failed":
                print(f"[skip] {result['tag']}: {result['error']}")

    elapsed = (time.time() - start_time) / 60
    print(f"\nDocked {len(lig_files)} ligands in {elapsed:.1f} min → {out_csv}")

    # --- 5. Summarize Results ---
    try:
        dock_df = pd.read_csv(out_csv).dropna(subset=["affinity"]).sort_values("affinity")
        dock_df.to_csv("vina_scores_sorted.csv", index=False)
        print("\nAffinity summary (kcal/mol):")
        print(dock_df["affinity"].describe().round(3))
        print("\nTop-10 hits:")
        print(dock_df.head(10))
    except Exception as e:
        print(f"\nCould not summarize results: {e}")
        print("Please check the 'vina_scores_fixed.csv' file.")
