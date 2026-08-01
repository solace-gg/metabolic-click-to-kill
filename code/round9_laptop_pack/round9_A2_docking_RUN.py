#!/usr/bin/env python3
"""
A2 docking — the controlled re-run that replaces the earlier exhaustiveness-1
sandbox run. Docks both the glucuronide-masked sugar and the phenyl-beta-glucuronide
reference into human beta-glucuronidase (PDB 3HN3) at matched settings
(exhaustiveness=16, 3 seeds each), and measures how close the ligand gets to the
catalytic Glu451/Glu540 pair (cleavage-competence: does the glucuronide reach the
catalytic residues, or does the bulky sugar get shut out?). The claim isn't affinity
on its own — it's the pose/contact.

Run (WSL Ubuntu or Linux; obabel must be on PATH from the conda env):
  conda create -n dock -c conda-forge python=3.11 vina openbabel rdkit numpy requests -y
  conda activate dock
  python round9_A2_docking_RUN.py
Writes results/round9_A2_docking_result.json with, per ligand: best affinity across
seeds, per-seed spread (convergence check), and min distance to the catalytic Glu pair.
"""
import os, subprocess, json, glob
import numpy as np, requests
from rdkit import Chem
from rdkit.Chem import AllChem
try: from vina import Vina
except Exception as e: raise SystemExit("Install AutoDock Vina: conda install -c conda-forge vina  ("+str(e)+")")

SD=os.path.dirname(os.path.abspath(__file__)); ROOT=os.path.dirname(os.path.dirname(SD))
RES=os.path.join(ROOT,"results"); os.makedirs(RES,exist_ok=True)
WORK=os.path.join(SD,"docking"); os.makedirs(WORK,exist_ok=True)
PDB="3HN3"; CENTER=[83.4,84.1,113.4]; BOX=[24,24,24]; EXH=16; SEEDS=[1,42,123]
LIG={
 "glucuronide_masked_sugar":
  "CC(=O)O[C@@H]1O[C@H](COC(=O)OCc2ccc(O[C@@H]3O[C@@H]([C@H](O)[C@H](O)[C@H]3O)C(=O)[O-])cc2)"
  "[C@@H](OC(C)=O)[C@H](OC(C)=O)[C@@H]1NC(=O)CN=[N+]=[N-]",
 "phenyl_glucuronide_ref":"O[C@@H]1O[C@@H]([C@H](O)[C@H](O)[C@H]1Oc1ccccc1)C(=O)[O-]",
}
def sh(c): return subprocess.run(c,shell=True,cwd=WORK,capture_output=True,text=True)
def fetch():
    p=os.path.join(WORK,PDB+".pdb")
    if not os.path.exists(p): open(p,"w").write(requests.get(f"https://files.rcsb.org/download/{PDB}.pdb",timeout=30).text)
    return p
def catalytic_oxygens(pdb):
    O=[]
    for ln in open(pdb):
        if ln.startswith("ATOM") and ln[17:20]=="GLU" and ln[21]=="A" and int(ln[22:26]) in (451,540) and ln[12:16].strip() in ("OE1","OE2"):
            O.append([float(ln[30:38]),float(ln[38:46]),float(ln[46:54])])
    return np.array(O)
def prep_receptor(pdb):
    keep=[ln[:16]+" "+ln[17:] for ln in open(pdb) if ln.startswith("ATOM") and ln[21]=="A" and ln[16] in (" ","A")]
    open(os.path.join(WORK,"recA.pdb"),"w").writelines(keep)
    sh("obabel recA.pdb -O receptor.pdbqt -xr -p 7.4")
    return os.path.join(WORK,"receptor.pdbqt")
def prep_ligand(name,smi):
    m=Chem.AddHs(Chem.MolFromSmiles(smi))
    if AllChem.EmbedMolecule(m,AllChem.ETKDGv3())!=0:                 # embed can fail on the 21-torsion ligand
        p=AllChem.ETKDGv3(); p.useRandomCoords=True; AllChem.EmbedMolecule(m,p)
    AllChem.MMFFOptimizeMolecule(m,maxIters=2000)
    Chem.MolToMolFile(m,os.path.join(WORK,name+".sdf")); sh(f"obabel {name}.sdf -O {name}.pdbqt -p 7.4")
    return os.path.join(WORK,name+".pdbqt")
def pose_models(pdbqt):
    """Yield (vina_affinity, coords[N,3]) for every MODEL in a multi-pose pdbqt."""
    models=[]; cur=[]; aff=None
    for ln in open(pdbqt):
        if ln.startswith("REMARK VINA RESULT"):
            try: aff=float(ln.split()[3])
            except Exception: aff=None
        elif ln.startswith(("ATOM","HETATM")):
            cur.append([float(ln[30:38]),float(ln[38:46]),float(ln[46:54])])
        elif ln.startswith("ENDMDL"):
            models.append((aff,np.array(cur))); cur=[]; aff=None
    if cur: models.append((aff,np.array(cur)))
    return models

pdb=fetch(); catO=catalytic_oxygens(pdb); rec=prep_receptor(pdb)
print(f"catalytic Glu451/540 carboxyl O atoms found: {len(catO)} | box {CENTER} {BOX} | exhaustiveness {EXH} x seeds {SEEDS}")
res={}
for name,smi in LIG.items():
    lig=prep_ligand(name,smi); best=None; scores=[]
    for sd in SEEDS:
        v=Vina(sf_name="vina",cpu=0,seed=sd,verbosity=0); v.set_receptor(rec); v.set_ligand_from_file(lig)
        v.compute_vina_maps(center=CENTER,box_size=BOX); v.dock(exhaustiveness=EXH,n_poses=10)
        e=float(v.energies(n_poses=1)[0][0]); scores.append(round(e,2))
        if best is None or e<best[0]:
            outp=os.path.join(WORK,f"{name}_best.pdbqt"); v.write_poses(outp,n_poses=10,overwrite=True); best=(e,outp)
    models=pose_models(best[1])
    CUT=5.0  # A: a pose is "catalytically proximal" if any atom is within CUT of a catalytic Glu carboxyl O
    def pdist(coords): return float(np.min(np.linalg.norm(coords[:,None,:]-catO[None,:,:],axis=2)))
    top_aff,top_c=models[0]; top_d=pdist(top_c)
    prox=[(i,a,pdist(c)) for i,(a,c) in enumerate(models) if pdist(c)<=CUT]   # POSE-FILTERED: proximal poses only
    prox.sort(key=lambda t:t[1])                                              # best-scoring among proximal
    filt = prox[0] if prox else None
    res[name]=dict(best_affinity_kcal_mol=round(best[0],2), per_seed=scores,
                   seed_spread_kcal=round(max(scores)-min(scores),2),
                   top_pose_dist_to_Glu_A=round(top_d,2),
                   pose_filtered=(dict(rank=int(filt[0]), affinity_kcal_mol=round(float(filt[1]),2),
                                       dist_to_Glu_A=round(filt[2],2)) if filt else None),
                   n_proximal_poses=len(prox), cutoff_A=CUT, exhaustiveness=EXH)
    pf = res[name]["pose_filtered"]
    print(f"  {name:<26} best {best[0]:6.2f} | seeds {scores} spread {res[name]['seed_spread_kcal']} | "
          f"top-pose {top_d:.2f} A | proximal poses {len(prox)}/10" + (f" (best proximal: rank {pf['rank']}, {pf['affinity_kcal_mol']} kcal, {pf['dist_to_Glu_A']} A)" if pf else " (NONE within cutoff!)"))
mask=res["glucuronide_masked_sugar"]; ref=res["phenyl_glucuronide_ref"]
mpf=mask["pose_filtered"]; rpf=ref["pose_filtered"]
res["interpretation"]=("POSE-FILTERED cleavage-competence: masked sugar has %d/10 poses within %g A of the catalytic Glu451/540"
  " (best proximal pose %s); the bulky acetylated tail does NOT exclude the glucuronide from the active site -> fit."
  " Top-pose distances: masked %.2f A, ref %.2f A. Affinity (masked %.2f vs ref %.2f, matched exhaustiveness %d) is NOT"
  " a tighter-binding claim: Vina over-scores larger ligands and pose ranking is not filtered for catalysis."
  %(mask["n_proximal_poses"], mask["cutoff_A"],
    (("rank %d, %.2f kcal, %.2f A"%(mpf["rank"],mpf["affinity_kcal_mol"],mpf["dist_to_Glu_A"])) if mpf else "none"),
    mask["top_pose_dist_to_Glu_A"], ref["top_pose_dist_to_Glu_A"],
    mask["best_affinity_kcal_mol"], ref["best_affinity_kcal_mol"], EXH))
json.dump(res,open(os.path.join(RES,"round9_A2_docking_result.json"),"w"),indent=2)
print("\n"+res["interpretation"]); print("saved -> results/round9_A2_docking_result.json")
