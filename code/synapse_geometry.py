#!/usr/bin/env python3
"""
synapse_geometry.py - Killing-geometry model (nanoscale synapse geometry).
This handles the "can it actually kill?" geometry question the ABM never asked. A
tight NK lytic synapse wants the membranes at ~<=15 nm (for perforin delivery and
kinetic segregation), but reach needs a long tether (~30 nm) to get across the
coat. No single fixed tether length does both: a short tether kills fine but barely
reaches, and a long tether reaches but holds the membranes too far apart to kill.
That reach-vs-kill conflict is one of the two findings that retire the covalent
tether in favour of the engager architecture (Chapter 6/8).

Length/force-balance model (nanoscale synapse geometry):
  reach forms if   2*R_F(PEG) + L_reach >= h_nk          (azide at outer face, Ch 4)
  kill-competent if effective membrane distance d = max(L_final, coat_residual) <= d_lytic
"""
import os, sys, json
import numpy as np
SD=os.path.dirname(os.path.abspath(__file__)); ROOT=os.path.dirname(SD)

def R_F(peg_mw): return 0.35*(peg_mw/44.0)**0.588

STRATEGIES = {  # (L_reach nm, L_final nm)
 "fixed short (15 nm)": (15, 15),   # kill-competent, but reach fails
 "fixed long (30 nm)":  (30, 30),   # reaches, but too far to kill
}

def run(N=20000, seed=42, d_lytic_lo=12, d_lytic_hi=18):
    rng=np.random.default_rng(seed)
    out={"N":N, "d_lytic_range":[d_lytic_lo,d_lytic_hi], "strategies":{}}
    # shared samples
    peg=rng.uniform(2000,5000,N); h_nk=rng.uniform(10,40,N); h_t=rng.uniform(30,500,N)
    muc_res=rng.uniform(0.03,0.30,N)          # residual tumour-coat fraction after mucinase
    d_lytic=rng.uniform(d_lytic_lo,d_lytic_hi,N)
    rf=R_F(peg)
    coat_residual=h_t*muc_res                 # spacer left between membranes after mucinase
    for name,(Lr,Lf) in STRATEGIES.items():
        reach_ok = (2*rf + Lr) >= h_nk
        d = np.maximum(Lf, coat_residual)     # conservative: residual coat is a hard spacer
        kill_ok = d <= d_lytic
        # optimistic: a tight synapse excludes bulky mucin (kinetic segregation) once mucinase applied
        kill_ok_excl = (Lf <= d_lytic) & (coat_residual <= 40)  # coat thin enough to be squeezed out
        joint = reach_ok & kill_ok
        out["strategies"][name]=dict(
            reach_ok=round(float(reach_ok.mean())*100,1),
            kill_ok_if_reached=round(float(kill_ok[reach_ok].mean())*100,1) if reach_ok.any() else 0.0,
            joint_reach_and_kill=round(float(joint.mean())*100,1),
            joint_with_synaptic_exclusion=round(float((reach_ok & kill_ok_excl).mean())*100,1))
    return out

if __name__=="__main__":
    R=run(); os.makedirs(os.path.join(ROOT,"results"),exist_ok=True)
    json.dump(R,open(os.path.join(ROOT,"results","round8_synapse_geometry.json"),"w"),indent=2)
    print("="*72); print("KILLING GEOMETRY: reach-vs-kill tether conflict (d_lytic ~15 nm)"); print("="*72)
    print(f"{'tether strategy':<26} {'reach%':>7} {'kill|reach%':>12} {'JOINT%':>8} {'JOINT+excl%':>10}")
    for n,s in R["strategies"].items():
        print(f"  {n:<24} {s['reach_ok']:>7} {s['kill_ok_if_reached']:>12} {s['joint_reach_and_kill']:>8} {s['joint_with_synaptic_exclusion']:>8}")
    print("\nSaved -> results/round8_synapse_geometry.json")
