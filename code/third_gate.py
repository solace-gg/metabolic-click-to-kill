#!/usr/bin/env python3
"""
third_gate.py - Third-gate selectivity rescue (2026-07-06).
Selectivity from independent gates multiplies: A0_combined = A0_enzyme x S3. So we
pin A0_enzyme = 3.8 (Wang measured) and try out candidate third gates S3 to see
which one recovers the therapeutic index best. Mechanistic TI throughout.
Outputs: TI vs S3 curve, the required-S3 design target, a per-trigger comparison,
and a JSON dump.
"""
import os, sys, json
import numpy as np
SD = os.path.dirname(os.path.abspath(__file__)); ROOT = os.path.dirname(SD)
sys.path.insert(0, SD); import mck_model as M
WANG = 3.8; RES = os.path.join(ROOT, "results")

# candidate third gates: (name, S3_low, S3_central, S3_high, kind, note)
TRIGGERS = [
 ("Focused ultrasound (spatial, distant)", 20, 140, 1000, "spatial",
  "near-binary; cavitation threshold; distant normal tissue protected"),
 ("NIR two-photon (spatial, distant)", 10, 45, 200, "spatial",
  "two-photon I^2 depth confinement; ~cm depth"),
 ("Hypoxia / bioreductive (endogenous)", 1.5, 2.7, 5.0, "endogenous",
  "systemic incl. metastases; graded; no clinician control"),
 ("Acidity pH (endogenous)", 1.5, 2.1, 3.0, "endogenous", "systemic; modest"),
 ("ROS (endogenous)", 1.5, 2.4, 4.0, "endogenous", "systemic; modest"),
 ("Local margin (in-beam, any spatial)", 1.0, 1.0, 1.0, "reference",
  "spatial gate gives NO gain vs peritumoural tissue-of-origin inside focus"),
]

def ti_stats(samples, S3, ratio=1.0):
    a0 = WANG * S3
    ti = np.array([M.therapeutic_index(p, a0, mode="mechanistic", p_base_normal_ratio=ratio) for p in samples])
    return dict(median=float(np.median(ti)), iqr=[float(np.percentile(ti,25)), float(np.percentile(ti,75))],
                frac_ge3=float(np.mean(ti>=3)*100))

def run(N=20000, seed=42):
    rng = np.random.default_rng(seed)
    S = [{k: M.sample_value(k, rng.random()) for k in M.PARAM_SPACE} for _ in range(N)]
    base = ti_stats(S, 1.0)
    # required-S3 curve
    s3_grid = np.geomspace(1, 500, 90)
    curve = [(float(s), ti_stats(S, s)) for s in s3_grid]
    curve_corr = [(float(s), ti_stats(S, s, ratio=0.5)) for s in s3_grid]
    med = np.array([c[1]["median"] for c in curve]); fr = np.array([c[1]["frac_ge3"] for c in curve])
    def crossing(y, target):
        idx = np.where(y >= target)[0]
        return float(s3_grid[idx[0]]) if len(idx) else None
    # asymptotic (S3 -> very large) stats
    asym = ti_stats(S, 1e6)
    req = {"S3_for_median_TI_3": crossing(med, 3.0),
           "S3_for_frac_ge3_50pct": crossing(fr, 50.0),
           "S3_for_frac_ge3_80pct": crossing(fr, 80.0),
           "asymptote_median": asym["median"], "asymptote_frac_ge3": asym["frac_ge3"]}
    trig = []
    for name, lo, cen, hi, kind, note in TRIGGERS:
        trig.append(dict(name=name, kind=kind, note=note, S3=[lo, cen, hi],
                         TI_low=ti_stats(S, lo), TI_central=ti_stats(S, cen), TI_high=ti_stats(S, hi)))
    return dict(N=N, A0_enzyme=WANG, baseline_no_third_gate=base,
                curve_corrected_ratio0p5=[[c[0], c[1]["median"], c[1]["frac_ge3"]] for c in curve_corr],
                required_S3=req, triggers=trig,
                curve=[[c[0], c[1]["median"], c[1]["frac_ge3"]] for c in curve])

if __name__ == "__main__":
    R = run()
    os.makedirs(RES, exist_ok=True)
    json.dump(R, open(os.path.join(RES, "round7_third_gate.json"), "w"), indent=2)
    b = R["baseline_no_third_gate"]
    print("="*74); print("THIRD-GATE SELECTIVITY RESCUE (enzyme A0=3.8x anchored)"); print("="*74)
    print(f"\nBaseline (no third gate, S3=1): median TI {b['median']:.2f}, frac>=3 {b['frac_ge3']:.1f}%")
    print(f"\nREQUIRED third-gate selectivity (design targets):")
    rq=R['required_S3']
    def fmt(x): return f"{x:.1f}x" if x else "not reachable (P_base-capped)"
    print(f"  median TI >= 3  needs  S3 >= {fmt(rq['S3_for_median_TI_3'])}")
    print(f"  50%% of space >=3 needs S3 >= {fmt(rq['S3_for_frac_ge3_50pct'])}")
    print(f"  80%% of space >=3 needs S3 >= {fmt(rq['S3_for_frac_ge3_80pct'])}")
    print(f"  asymptote (S3 huge): median TI {rq['asymptote_median']:.2f}, frac>=3 {rq['asymptote_frac_ge3']:.1f}%  <- P_base ceiling")
    print(f"\n{'Third gate':<40} {'S3(cen)':>8} {'medianTI':>9} {'frac>=3':>8}")
    for t in R["triggers"]:
        print(f"  {t['name']:<38} {t['S3'][1]:>7}x {t['TI_central']['median']:>9.2f} {t['TI_central']['frac_ge3']:>7.1f}%")
    print("\nSaved -> results/round7_third_gate.json")
