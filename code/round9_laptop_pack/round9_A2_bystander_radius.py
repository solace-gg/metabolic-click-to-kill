#!/usr/bin/env python3
"""
Reviewer point: with an extracellular trigger, the mask comes off in the interstitium
and the released native Ac4ManNAz then diffuses and labels a neighbourhood before it is
taken up or cleared -> a 'bystander' penumbra. Here we quantify that radius.

Reaction-diffusion length:  lambda = sqrt( D / (k_uptake + k_clearance) )
 D            effective diffusion of a ~450 Da sugar in tumour interstitium
 k_uptake     first-order cellular uptake rate of the released sugar
 k_clearance  first-order interstitial washout (blood/lymph convection)
Everything analytical (laptop-free).
"""
import numpy as np, json, os
ROOT=os.path.dirname(os.path.dirname(os.path.dirname(os.path.abspath(__file__)))); RES=os.path.join(ROOT,"results"); os.makedirs(RES,exist_ok=True)
# D in um^2/s : free small molecule ~200-600; tissue/tortuosity-reduced ~10-60
D_cases={"tissue (tortuous) D=20":20.0,"intermediate D=80":80.0,"near-free D=300":300.0}
# combined escape rate k = k_up + k_clear (1/s). Interstitial washout t1/2 ~ 1-30 min;
# metabolic-sugar uptake is slow (min-hr). Bracket by total escape half-life.
thalf_min=[1,5,15,60]   # minutes for the sugar to leave the interstitial point (uptake OR clearance)
rows=[]
for dn,D in D_cases.items():
    for th in thalf_min:
        k=np.log(2)/(th*60.0)           # 1/s
        lam=np.sqrt(D/k)                 # um
        rows.append((dn,th,round(lam,0)))
print("BYSTANDER / PENUMBRA RADIUS  lambda = sqrt(D/k)   [um]")
print(f"{'D case':<26}{'escape t1/2 (min)':>18}{'radius (um)':>14}")
for dn,th,lam in rows: print(f"{dn:<26}{th:>18}{lam:>14}")
lam_all=[r[2] for r in rows]
summary=dict(model="lambda=sqrt(D/(k_up+k_clear))",D_um2_s=D_cases,escape_thalf_min=thalf_min,
  radius_um_min=min(lam_all),radius_um_max=max(lam_all),
  interpretation=("Released native sugar labels a penumbra of ~tens to a few hundred um around "
   "high-extracellular-beta-gluc foci. This penumbra is still filtered by the HDAC/CTSL enzyme gate "
   "(gate 2), so it is a labelling LEAK, not free killing. Distant tissue (mm-cm, no beta-gluc) is "
   "unaffected -> the beta-gluc gate protects DISTANT tissue like a spatial gate, not the local margin."))
json.dump(summary,open(os.path.join(RES,"round9_A2_bystander_radius.json"),"w"),indent=2)
print(f"\nrange: {min(lam_all):.0f}-{max(lam_all):.0f} um. saved -> results/round9_A2_bystander_radius.json")
