# Heavy-compute pack

These scripts run the optional heavier checks that back the third-gate design
(Chapter 7). Each is self-contained; run it in the matching environment and the
result is written to `results/` as JSON. None is required to reproduce the main
therapeutic-index, reach, or killing numbers (those come from the scripts in
`code/`); these establish the supporting chemistry claims.

## Self-immolation energetics (xtb GFN2)  [CPU, ~minutes]
Confirms the self-immolative cascade fragments cleanly to the C6-unmasked sugar
with no thermodynamic dead-end.
```
conda create -n mck -c conda-forge xtb rdkit python=3.11 -y
conda activate mck
python round9_A1_selfimmolation_xtb.py     # prints the +2.7 / -23.7 / +1.5 kcal/mol steps
python round9_A1_selfimmol_verdict.py      # writes the verdict JSON + profile figure
```
Honest verdict: `net_exergonic` is **false** and `no_thermodynamic_dead_end` is
**true**. The cascade has no stable well; completion is driven by quinone-methide
hydration and CO2 loss, not by a net-exergonic single step.

## beta-glucuronidase docking (AutoDock Vina)  [CPU, ~10-30 min]
Applies to the ENDOGENOUS glucuronide route only. Confirms the bulky
sugar-appended glucuronide still seats in the enzyme so the trigger can cleave it
(cleavage-competence, not a binding-affinity claim).
```
conda create -n dock -c conda-forge python=3.11 vina meeko rdkit openbabel numpy requests -y
conda activate dock
python round9_A2_docking_RUN.py            # both ligands, exhaustiveness 16, 3 seeds, + distance to catalytic Glu
```
The claim kept is "fits / cleavage-competent"; an affinity comparison only
survives if the gap exceeds Vina noise (~1 kcal/mol) at matched settings.

## 310 K stability MD (OpenMM, GPU)  [~minutes]
Conformational stability of the reverted sugar and the masked prodrug. MD tests
conformation only; bond hydrolysis / pH cleavage are covered by the xtb work.
```
conda create -n md -c conda-forge openmm openmmforcefields openff-toolkit rdkit numpy -y
conda activate md
python round9_A5_stability_md.py           # reports bounded-vs-drifting RMSD and Rg per molecule
```

## Analytical supporting scripts (any python + rdkit env)
```
python round9_A2_bystander_radius.py       # reaction-diffusion labelling radius (40-350 um)
python round9_S4_mucinase_audit.py         # StcE motif scan; sialic-acid-sparing anchor survival
```
