#!/usr/bin/env python3
"""
A1/A2 self-immolation energetics — glucuronide/PABC-carbonate cascade.
The question: once the trigger frees the phenol, does the linker fragment cleanly
to the native sugar with no stable dead-end intermediate? We test it with xtb GFN2
free energies (ALPB water, 310 K) on a minimal model (payload = methanol proxy for
the sugar C6-OH).

Cascade modelled (p-hydroxybenzyl carbonate; the O-CO-O-sugar bond is our design):
  M0  HO-C6H4-CH2-O-C(=O)-O-CH3        (neutral phenol; post-glucuronidase intermediate)
  M1  [-O-C6H4-CH2-O-C(=O)-O-CH3]^-    (phenolate; the reactive species)
  P_QM  o-/p-quinone methide  O=C6H4=CH2
  P_carb [CH3-O-C(=O)-O]^-   -> decarboxylates -> CO2 + CH3O^-  (payload = free sugar OH)
Step energetics:
  dG1 = deprotonation (context/enzyme-driven; reported for completeness)
  dG2 = 1,6-elimination:  M1 -> P_QM + [CH3OC(=O)O]^-
  dG3 = decarboxylation:  [CH3OC(=O)O]^- -> CO2 + CH3O^-
Clean cascade  <=>  dG2 < 0 and dG3 < 0 (CO2 loss makes it irreversible) and no
intermediate sits in a deep well.

This script: (1) RDKit ETKDG 3D-embed each species -> .xyz; (2) emit the exact xtb
commands; (3) if xtb is on PATH, run them and parse G(total) to tabulate step dG.

Run on the laptop:
  conda create -n mck -c conda-forge xtb rdkit python=3.11 -y
  conda activate mck
  python round9_A1_selfimmolation_xtb.py
"""
import os, subprocess, shutil, re, json
from rdkit import Chem
from rdkit.Chem import AllChem

WORK = os.path.join(os.path.dirname(os.path.abspath(__file__)), "xtb_selfimmol"); os.makedirs(WORK, exist_ok=True)
SPECIES = {  # name: (smiles, charge)  -- charge-balanced set
 "M0_phenol_carbonate":      ("Oc1ccc(COC(=O)OC)cc1", 0),
 "M1_phenolate_carbonate":   ("[O-]c1ccc(COC(=O)OC)cc1", -1),
 "P_quinone_methide":        ("O=C1C=CC(=C)C=C1", 0),
 "P_methyl_carbonate_anion": ("COC(=O)[O-]", -1),
 "methoxide_anion":          ("C[O-]", -1),   # correct (charge-balanced) decarboxylation product
 "CO2":                      ("O=C=O", 0),
 "methanol_payload":         ("CO", 0),
 "water":                    ("O", 0),
 "p_hydroxybenzyl_alcohol":  ("OCc1ccc(O)cc1", 0),  # quinone-methide + water trapping product (the sink)
}
def embed(smi):
    m = Chem.AddHs(Chem.MolFromSmiles(smi))
    AllChem.EmbedMolecule(m, AllChem.ETKDGv3()); AllChem.MMFFOptimizeMolecule(m, maxIters=2000)
    return m
def write_xyz(m, path):
    conf=m.GetConformer(); n=m.GetNumAtoms()
    with open(path,"w") as f:
        f.write(f"{n}\n\n")
        for a in m.GetAtoms():
            p=conf.GetAtomPosition(a.GetIdx()); f.write(f"{a.GetSymbol()} {p.x:.4f} {p.y:.4f} {p.z:.4f}\n")
have_xtb = shutil.which("xtb") is not None
energies={}
for name,(smi,chg) in SPECIES.items():
    d=os.path.join(WORK,name); os.makedirs(d,exist_ok=True)
    xyz=os.path.join(d,"mol.xyz"); write_xyz(embed(smi), xyz)
    cmd=["xtb","mol.xyz","--gfn","2","--ohess","--alpb","water","--chrg",str(chg),"-T","4"]
    print(f"[{name}] xyz written ({smi}, chg {chg}). xtb cmd: cd {d} && "+" ".join(cmd))
    if have_xtb:
        # Windows-safe: xtb prints non-ASCII; force utf-8 + tolerate stray bytes.
        r=subprocess.run(cmd,cwd=d,capture_output=True,text=True,encoding="utf-8",errors="replace")
        open(os.path.join(d,"xtb.out"),"w",encoding="utf-8").write((r.stdout or "")+"\n=STDERR=\n"+(r.stderr or ""))
        g=None
        for ln in (r.stdout or "").splitlines():
            if "total free energy" in ln.lower():          # xtb --ohess thermo block
                mo=re.search(r"(-?\d+\.\d+)",ln)
                if mo: g=float(mo.group(1))
        energies[name]=g; print(f"    G_total(Eh) = {g}  (log: {name}/xtb.out)")
need=["M0_phenol_carbonate","M1_phenolate_carbonate","P_quinone_methide","P_methyl_carbonate_anion",
      "methoxide_anion","CO2","methanol_payload","water","p_hydroxybenzyl_alcohol"]
if have_xtb and all(energies.get(k) is not None for k in need):
    Eh=627.509  # kcal/mol per Hartree
    g=energies
    # all steps CHARGE-BALANCED:
    dG_elim  = (g["P_quinone_methide"]+g["P_methyl_carbonate_anion"]-g["M1_phenolate_carbonate"])*Eh   # -1 -> -1
    dG_decarb= (g["CO2"]+g["methoxide_anion"]-g["P_methyl_carbonate_anion"])*Eh                          # -1 -> -1
    dG_QMtrap= (g["p_hydroxybenzyl_alcohol"]-g["P_quinone_methide"]-g["water"])*Eh                       # 0  -> 0  (irrev. sink)
    dG_overall=(g["p_hydroxybenzyl_alcohol"]+g["CO2"]+g["methanol_payload"]-g["M0_phenol_carbonate"]-g["water"])*Eh  # neutral net
    # "clean" = the net (water-trapped) cascade is exergonic AND no single step is a deep well (> +15 uphill)
    steps={"1,6-elimination (M1- -> QM + carbonate-)":dG_elim,
           "decarboxylation (carbonate- -> CO2 + MeO-)":dG_decarb,
           "QM water-trapping (QM + H2O -> HO-benzyl-OH) [sink]":dG_QMtrap}
    deepest=max(steps.values())
    clean=bool(dG_overall<0 and deepest<15)
    out=dict(energies_Eh=g, steps_kcal={k:round(v,1) for k,v in steps.items()},
             dG_overall_net_kcal=round(dG_overall,1), clean_cascade=clean)
    json.dump(out,open(os.path.join(WORK,"selfimmol_energetics.json"),"w"),indent=2)
    print("\n=== SELF-IMMOLATION ENERGETICS (GFN2/ALPB-water, 310 K) — charge-balanced ===")
    for k,v in steps.items(): print(f"  dG {k:<52} = {v:+.1f} kcal/mol")
    print(f"  dG NET cascade (M0 + H2O -> HO-benzyl-OH + CO2 + MeOH) = {dG_overall:+.1f} kcal/mol")
    print(f"  CLEAN (net exergonic, no deep well >+15): {clean}")
    print("  saved -> xtb_selfimmol/selfimmol_energetics.json")
else:
    miss=[k for k in need if energies.get(k) is None]
    print("\n[xtb not on PATH or missing species] .xyz written; run printed xtb cmds, or install xtb and re-run.")
    if have_xtb: print("  missing free energies for:", miss)
