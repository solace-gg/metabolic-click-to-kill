#!/usr/bin/env python3
"""
A5 — 310 K stability MD (OpenMM, GPU) of the reverted native sugar and the
glucuronide-masked 3G-AAM in explicit water.
What MD can tell us here: thermal/conformational stability at body temperature
(bounded RMSD, no spontaneous unfolding/collapse), solvation behaviour, conformer
ensemble. What it can't: bond hydrolysis / pH cleavage / photostability - a fixed
topology can't break or form bonds, so those go to xtb + reasoning (D8).

Run on the laptop GPU (RTX 5070 Ti):
  conda create -n md -c conda-forge openmm openmmforcefields openff-toolkit rdkit numpy -y
  conda activate md
  python round9_A5_stability_md.py
Reports per molecule: mean/max all-heavy-atom Kabsch-aligned RMSD (nm) and radius of
gyration over a 1 ns 310 K trajectory (a small sugar has no 'backbone'). A large
all-atom RMSD on a floppy molecule is just pendant-arm flexibility, not instability -
so judge stability by whether the RMSD/Rg stays bounded (doesn't drift), not by the
arbitrary 0.5 nm flag.
"""
import numpy as np
from openff.toolkit import Molecule
from openmm import LangevinMiddleIntegrator, Platform, unit
from openmm.app import Simulation, PDBFile, Modeller, ForceField, PME, HBonds
from openmmforcefields.generators import SystemGenerator
from rdkit import Chem
from rdkit.Chem import AllChem

MOLS = {
 "reverted_native_sugar": "CC(=O)O[C@@H]1O[C@H](CO)[C@@H](OC(C)=O)[C@H](OC(C)=O)[C@@H]1NC(=O)CN=[N+]=[N-]",
 "glucuronide_masked_3GAAM":
   "CC(=O)O[C@@H]1O[C@H](COC(=O)OCc2ccc(O[C@@H]3O[C@@H]([C@H](O)[C@H](O)[C@H]3O)C(=O)[O-])cc2)"
   "[C@@H](OC(C)=O)[C@H](OC(C)=O)[C@@H]1NC(=O)CN=[N+]=[N-]",
}
def kabsch_rmsd(P, Q):
    # RMSD after optimal superposition (removes rigid-body rotation/translation)
    Pc = P - P.mean(0); Qc = Q - Q.mean(0)
    V, S, Wt = np.linalg.svd(Pc.T @ Qc)
    d = np.sign(np.linalg.det(Wt.T @ V.T))
    R = Wt.T @ np.diag([1, 1, d]) @ V.T
    return float(np.sqrt(((Pc @ R.T - Qc)**2).sum(1).mean()))

def build_sim(smi):
    off = Molecule.from_smiles(smi, allow_undefined_stereo=True); off.generate_conformers(n_conformers=1)
    rd = off.to_rdkit(); conf = rd.GetConformer()
    top = off.to_topology().to_openmm()
    pos = off.conformers[0].to_openmm()
    sg = SystemGenerator(small_molecule_forcefield="gaff-2.11",
                         forcefields=["amber/tip3p_standard.xml"],
                         molecules=[off], periodic_forcefield_kwargs={"nonbondedMethod":PME})
    modeller = Modeller(top, pos)
    modeller.addSolvent(sg.forcefield, model="tip3p", padding=0.9*unit.nanometer, ionicStrength=0.15*unit.molar)
    system = sg.create_system(modeller.topology)
    # Try GPU platforms, but actually BUILD the context so a CUDA/PTX mismatch
    # (e.g. a very new GPU vs the conda CUDA build) falls back cleanly to CPU.
    sim = None
    for pname in ["CUDA", "OpenCL", "CPU"]:
        try:
            plat = Platform.getPlatformByName(pname)
            integ = LangevinMiddleIntegrator(310*unit.kelvin, 1.0/unit.picosecond, 0.002*unit.picoseconds)
            sim = Simulation(modeller.topology, system, integ, plat)
            sim.context.setPositions(modeller.positions)
            sim.context.getState(getEnergy=True)   # forces CUDA module load -> triggers PTX error here if any
            print("    platform:", pname); break
        except Exception as e:
            print(f"    {pname} unavailable ({str(e)[:55]}...); trying next")
            sim = None
    if sim is None:
        raise RuntimeError("no working OpenMM platform (CUDA/OpenCL/CPU all failed)")
    return sim, off.n_atoms
def run(name, smi, ns=1.0):
    sim, nsolute = build_sim(smi)
    sim.minimizeEnergy()
    sim.context.setVelocitiesToTemperature(310*unit.kelvin)
    steps = int(ns*1000/0.002/1000)*1000  # ns -> 2 fs steps
    solute = list(range(nsolute))
    ref = sim.context.getState(getPositions=True).getPositions(asNumpy=True).value_in_unit(unit.nanometer)[solute]
    rmsds=[]; rgs=[]
    for i in range(50):
        sim.step(max(1,steps//50))
        p = sim.context.getState(getPositions=True).getPositions(asNumpy=True).value_in_unit(unit.nanometer)[solute]
        rmsds.append(kabsch_rmsd(p, ref))
        c = p-p.mean(0); rgs.append(float(np.sqrt((c**2).sum(1).mean())))
    print(f"  {name:<28} RMSD mean {np.mean(rmsds):.3f} max {np.max(rmsds):.3f} nm | Rg {np.mean(rgs):.3f} nm | "
          f"{'bounded (stable)' if (np.max(rmsds)-np.mean(rmsds))<0.25 else 'drifting? inspect'}")
    return dict(name=name, rmsd_mean=round(float(np.mean(rmsds)),3), rmsd_max=round(float(np.max(rmsds)),3),
                rg_mean=round(float(np.mean(rgs)),3))
if __name__=="__main__":
    print("A5 310 K stability MD (OpenMM/GAFF2, TIP3P, 0.15 M NaCl, 1 ns; aligned RMSD):")
    import json,os
    res=[run(n,s) for n,s in MOLS.items()]
    os.makedirs("results",exist_ok=True); json.dump(res,open("results/round9_A5_stability_md.json","w"),indent=2)
    print("saved -> results/round9_A5_stability_md.json")
