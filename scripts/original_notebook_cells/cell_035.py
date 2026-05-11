# --- 0. INSTALL THE REQUIRED TOOL ---
!pip install openbabel-wheel

import os
import re

print(f"Original receptor PDB: {receptor_pdb_path}")

# --- 1. SET THE CHAINS YOU WANT TO KEEP ---
CHAINS_TO_KEEP = {'A', 'B'}
print(f"Will keep only chains: {CHAINS_TO_KEEP}")

# --- 2. CREATE A CLEAN, RENUMBERED PDB (The Definitive Python Fix) ---
# This script reads the PDB, keeps only chains A & B,
# and manually renumbers all atoms sequentially to prevent all errors.

clean_pdb_renumbered = os.path.join(WORK, "receptor_protein_AB_renumbered.pdb")
atom_counter = 1

with open(receptor_pdb_path) as f_in, open(clean_pdb_renumbered, "w") as f_out:
    for line in f_in:
        # Keep ATOM lines from the correct chains
        if line.startswith("ATOM"):
            chain_id = line[21]
            if chain_id in CHAINS_TO_KEEP:
                # This is the fix: re-number the atom serial (cols 7-11)
                # with our new, sequential atom_counter
                new_atom_line = line[:6] + str(atom_counter).rjust(5) + line[11:]
                f_out.write(new_atom_line)
                atom_counter += 1 # Increment our master counter
        
        # Keep TER lines from the correct chains
        elif line.startswith("TER"):
            chain_id = line[21]
            if chain_id in CHAINS_TO_KEEP:
                f_out.write(line)
        
        # Keep the END line
        elif line.startswith("END"):
            f_out.write(line)

print(f"Wrote new clean, renumbered PDB with {atom_counter - 1} atoms: {clean_pdb_renumbered}")

# --- 3. CONVERT THE PERFECTLY CLEAN PDB TO PDBQT ---
# This command is now simple: it just converts. All the complex logic is done.
receptor_pdbqt = os.path.join(WORK, "receptor.pdbqt")

!obabel -ipdb "{clean_pdb_renumbered}" -opdbqt -O "{receptor_pdbqt}" -xr -p 7.4 --partialcharge gasteiger

assert os.path.exists(receptor_pdbqt), "PDBQT file was not created!"
print(f"Wrote clean receptor PDBQT: {receptor_pdbqt}")

# --- 4. VERIFY IT IS CLEAN AND *NOT EMPTY* ---
# 'ls -lh' will show the file size. It should be > 200K, not 0.
!ls -lh "{receptor_pdbqt}"
!grep -E "ROOT|BRANCH" "{receptor_pdbqt}" || echo "Verification OK: File is clean."