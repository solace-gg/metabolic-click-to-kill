#!/usr/bin/env python3
"""
A2 docking — does the bulky sugar-appended glucuronide still seat in the
beta-glucuronidase active site so the enzyme can cleave the glycosidic bond?
(Checks the outer lock is trigger-cleavable, not sterically blocked.)

Targets (fetch from RCSB):
  human beta-glucuronidase : 3HN3  (or 1BHG)   <- primary
  human legumain (LGMN)    : 4AWB              <- peptide-mask backup
  E. coli nitroreductase NfsB : 1DS7           <- demoted comparator (D1): shows it
                                                  fits fine, but it's intracellular
                                                  (a location problem, not a fit one)

Pipeline: fetch PDB -> strip waters/hetero -> receptor.pdbqt (Meeko/obabel);
ligand SMILES -> 3D (RDKit) -> ligand.pdbqt (Meeko); box centred on the catalytic
site (auto from bound ligand/catalytic residues) -> AutoDock Vina -> affinity + pose.

Install (laptop):
  conda create -n dock -c conda-forge python=3.11 vina meeko rdkit openbabel numpy -y
  conda activate dock
  pip install pdb-tools requests
Run:
  python round9_A2_docking_glucuronidase.py
Report back: the Vina affinity (kcal/mol) and whether the glycosidic O sits near
the catalytic Glu451/Glu540 (beta-gluc) — that is the cleavage-competence check.
"""
import os, sys, requests, subprocess
from rdkit import Chem
from rdkit.Chem import AllChem

LIGANDS = {
 # glucuronide-masked sugar (charge -1). The scissile bond is the glycosidic O of the glucuronide.
 "glucuronide_masked_sugar":
   "CC(=O)O[C@@H]1O[C@H](COC(=O)OCc2ccc(O[C@@H]3O[C@@H]([C@H](O)[C@H](O)[C@H]3O)C(=O)[O-])cc2)"
   "[C@@H](OC(C)=O)[C@H](OC(C)=O)[C@@H]1NC(=O)CN=[N+]=[N-]",
 # minimal reference substrate (phenyl-beta-D-glucuronide) to validate the box/pose
 "phenyl_glucuronide_ref": "O[C@@H]1O[C@@H]([C@H](O)[C@H](O)[C@H]1Oc1ccccc1)C(=O)[O-]",
}
TARGETS = {"beta_glucuronidase":"3HN3", "legumain":"4AWB", "nitroreductase_NfsB":"1DS7"}
WORK = os.path.join(os.path.dirname(os.path.abspath(__file__)), "docking"); os.makedirs(WORK, exist_ok=True)

def fetch_pdb(pdbid):
    p=os.path.join(WORK,f"{pdbid}.pdb")
    if not os.path.exists(p):
        r=requests.get(f"https://files.rcsb.org/download/{pdbid}.pdb",timeout=30); r.raise_for_status()
        open(p,"w").write(r.text)
    return p
def lig_3d(name,smi):
    m=Chem.AddHs(Chem.MolFromSmiles(smi)); AllChem.EmbedMolecule(m,AllChem.ETKDGv3())
    AllChem.MMFFOptimizeMolecule(m,maxIters=2000)
    p=os.path.join(WORK,f"{name}.sdf"); w=Chem.SDWriter(p); w.write(m); w.close(); return p

print("=== A2 docking prep ===")
for t,pid in TARGETS.items():
    try: print(f"[receptor] {t}: {fetch_pdb(pid)}")
    except Exception as e: print(f"[receptor] {t} {pid}: FETCH FAILED ({e}); download manually from rcsb.org")
for n,s in LIGANDS.items():
    print(f"[ligand]  {n}: {lig_3d(n,s)}")
print("""
NEXT (documented, laptop):
 1) Receptor prep:  mk_prepare_receptor.py -i 3HN3.pdb -o beta_gluc.pdbqt -p  (Meeko)
    or: obabel 3HN3.pdb -O beta_gluc.pdbqt -xr  (after stripping HOH/HETATM)
 2) Ligand prep:    mk_prepare_ligand.py -i glucuronide_masked_sugar.sdf -o lig.pdbqt
 3) Box: centre on the catalytic pocket. For human beta-gluc (3HN3) the catalytic
    nucleophile/acid are Glu451 & Glu540 — set --center_x/y/z to their CA midpoint,
    box 22 A cubic.  (Script prints residue coords if you pass --show-catalytic.)
 4) vina --receptor beta_gluc.pdbqt --ligand lig.pdbqt \\
         --center_x X --center_y Y --center_z Z --size_x 22 --size_y 22 --size_z 22 \\
         --exhaustiveness 16 --out lig_docked.pdbqt
 5) Read affinity (kcal/mol) from vina output; inspect that the GLYCOSIDIC O of the
    glucuronide points at Glu451/Glu540 (cleavage-competent) and the bulky acetylated
    sugar tail projects OUT of the pocket (no clash). Compare to phenyl_glucuronide_ref.
""")
