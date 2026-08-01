#!/usr/bin/env python3
"""A1 uptake-gating: does the outer mask block passive uptake, and does clean
reversion give us back the permeable native sugar? RDKit descriptors + charge.
The honest read: charge is decisive for the masked species; neutral species get
judged against the empirically permeable native Ac4ManNAz envelope, since
absolute TPSA mis-rates sugars."""
import os, json, csv
from rdkit import Chem
from rdkit.Chem import Descriptors, Crippen, rdMolDescriptors
RES = os.path.join(os.path.dirname(os.path.abspath(__file__)), "results"); os.makedirs(RES, exist_ok=True)

PANEL = {
 "native_Ac4ManNAz (permeable ref)":
   "CC(=O)O[C@@H]1O[C@H](COC(C)=O)[C@@H](OC(C)=O)[C@H](OC(C)=O)[C@@H]1NC(=O)CN=[N+]=[N-]",
 "free_ManNAz (deacetylated, cytosolic)":
   "OC[C@H]1O[C@@H](O)[C@@H](NC(=O)CN=[N+]=[N-])[C@@H](O)[C@@H]1O",
 "3G-AAM glucuronide mask (COO-)":
   "CC(=O)O[C@@H]1O[C@H](COC(=O)OCc2ccc(O[C@@H]3O[C@@H]([C@H](O)[C@H](O)[C@H]3O)C(=O)[O-])cc2)"
   "[C@@H](OC(C)=O)[C@H](OC(C)=O)[C@@H]1NC(=O)CN=[N+]=[N-]",
 "3G-AAM sulfonate mask (SO3-)":
   "CC(=O)O[C@@H]1O[C@H](COC(=O)OCc2ccc(CCS(=O)(=O)[O-])cc2)"
   "[C@@H](OC(C)=O)[C@H](OC(C)=O)[C@@H]1NC(=O)CN=[N+]=[N-]",
 "3G-AAM PEG3 mask (neutral)":
   "CC(=O)O[C@@H]1O[C@H](COC(=O)NCCOCCOCCOC)"
   "[C@@H](OC(C)=O)[C@H](OC(C)=O)[C@@H]1NC(=O)CN=[N+]=[N-]",
 "3G-AAM PEG12 mask (neutral)":
   "CC(=O)O[C@@H]1O[C@H](COC(=O)NCCOCCOCCOCCOCCOCCOCCOCCOCCOCCOCCOCCOC)"
   "[C@@H](OC(C)=O)[C@H](OC(C)=O)[C@@H]1NC(=O)CN=[N+]=[N-]",
 "3G-AAM phosphate mask (PO4 2-) [BAD: phosphatase-labile]":
   "CC(=O)O[C@@H]1O[C@H](COP(=O)([O-])[O-])"
   "[C@@H](OC(C)=O)[C@H](OC(C)=O)[C@@H]1NC(=O)CN=[N+]=[N-]",
 "cascade1: after glucuronidase (phenol-PABC-carbonate)":
   "CC(=O)O[C@@H]1O[C@H](COC(=O)OCc2ccc(O)cc2)"
   "[C@@H](OC(C)=O)[C@H](OC(C)=O)[C@@H]1NC(=O)CN=[N+]=[N-]",
 "cascade2: aza/quinone-methide byproduct":
   "O=C1C=CC(=C)C=C1",
 "cascade3: reverted native sugar (free C6-OH)":
   "CC(=O)O[C@@H]1O[C@H](CO)[C@@H](OC(C)=O)[C@H](OC(C)=O)[C@@H]1NC(=O)CN=[N+]=[N-]",
}
def props(smi):
    m = Chem.MolFromSmiles(smi)
    if m is None: return None
    return dict(MolWt=round(Descriptors.MolWt(m),1), cLogP=round(Crippen.MolLogP(m),2),
        TPSA=round(rdMolDescriptors.CalcTPSA(m),1), HBD=rdMolDescriptors.CalcNumHBD(m),
        HBA=rdMolDescriptors.CalcNumHBA(m), RotB=rdMolDescriptors.CalcNumRotatableBonds(m),
        charge=Chem.GetFormalCharge(m))
NAT = props(PANEL["native_Ac4ManNAz (permeable ref)"])
def verdict(p):
    if p["charge"] != 0: return "BLOCKED (charged: no passive bilayer crossing)"
    dT = p["TPSA"] - NAT["TPSA"]
    if p["MolWt"] > 700: return f"reduced (size-limited, MW {p['MolWt']:.0f} >> native {NAT['MolWt']:.0f})"
    if dT <= 30 and p["MolWt"] <= 500 and p["HBD"] <= NAT["HBD"]+1: return "~permeable (within native envelope)"
    if dT > 30: return f"reduced vs native (+{dT:.0f} TPSA)"
    return "borderline"
rows=[]
for name,smi in PANEL.items():
    p=props(smi)
    if p is None: rows.append(dict(name=name,error="parse fail")); continue
    p["name"]=name; p["verdict"]=verdict(p); rows.append(p)
json.dump(rows, open(os.path.join(RES,"round9_A1_permeability.json"),"w"), indent=2)
with open(os.path.join(RES,"round9_A1_permeability.csv"),"w",newline="") as f:
    w=csv.DictWriter(f,fieldnames=["name","MolWt","cLogP","TPSA","HBD","HBA","RotB","charge","verdict"]); w.writeheader()
    for r in rows:
        if "error" in r: continue
        w.writerow({k:r[k] for k in w.fieldnames})
print("="*116)
print("A1 UPTAKE-GATING  (native ref: MW %.0f TPSA %.0f HBD %d, empirically permeable)"%(NAT["MolWt"],NAT["TPSA"],NAT["HBD"]))
print("="*116)
print(f"{'molecule':<50}{'MW':>7}{'cLogP':>7}{'TPSA':>7}{'HBD':>5}{'chg':>5}  verdict"); print("-"*116)
for r in rows:
    if "error" in r: print(f"{r['name']:<50}  !!parse fail"); continue
    print(f"{r['name']:<50}{r['MolWt']:>7}{r['cLogP']:>7}{r['TPSA']:>7}{r['HBD']:>5}{r['charge']:>5}  {r['verdict']}")
print("-"*116); print("Saved -> results/round9_A1_permeability.{json,csv}")
