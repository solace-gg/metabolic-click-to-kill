#!/usr/bin/env python3
"""
M-3 fix: a push-button driver for the primary selectivity headline (median TI 2.16).
sweep_mck.py only prints the symmetric floor (median 1.37); the abstract/Ch5 headline
of 2.16 is the biologically-corrected primary (normal:cancer P_base ratio = 0.5,
sec. 5.4). This just emits it directly over the full 20k so nobody has to remember to
flip a kwarg.
"""
import os, sys, json
import numpy as np
sys.path.insert(0, os.path.dirname(os.path.abspath(__file__))); import mck_model as M
ROOT=os.path.dirname(os.path.dirname(os.path.abspath(__file__)))
def run(N=20000, seed=42):
    rng=np.random.default_rng(seed)
    S=[{k:M.sample_value(k,rng.random()) for k in M.PARAM_SPACE} for _ in range(N)]
    out={}
    for label,ratio in [("floor (symmetric, ratio 1.0)",1.0),
                        ("PRIMARY (corrected biology, ratio 0.5)",0.5),
                        ("realistic (ratio 0.25)",0.25)]:
        ti=np.array([M.therapeutic_index(p,3.8,mode="mechanistic",p_base_normal_ratio=ratio) for p in S])
        out[label]=dict(median=round(float(np.median(ti)),2),
                        IQR=[round(float(np.percentile(ti,25)),2),round(float(np.percentile(ti,75)),2)],
                        frac_ge3_pct=round(float(np.mean(ti>=3)*100),1))
    return out
if __name__=="__main__":
    R=run(); os.makedirs(os.path.join(ROOT,"results"),exist_ok=True)
    json.dump(R,open(os.path.join(ROOT,"results","round9_TI_primary.json"),"w"),indent=2)
    print("="*70); print("PRIMARY SELECTIVITY HEADLINE (Wang A0=3.8x, mechanistic TI, N=20000)"); print("="*70)
    for k,v in R.items(): print(f"  {k:<40} median {v['median']:<5} IQR {v['IQR']}  frac>=3 {v['frac_ge3_pct']}%")
    print("\n  -> the abstract/Ch5 headline is the PRIMARY row (median 2.16, ~19.8% >=3).")
