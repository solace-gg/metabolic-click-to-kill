#!/usr/bin/env python3
"""
joint_feasibility.py - compound feasibility across the necessary conditions on the
same parameter samples (one correlated sweep), instead of multiplying fractions
from separate sweeps.

For each sampled "world" i we check, on shared draws of the physical params (coat
gc_thickness, NK coat gc_nk, peg_mw), so the correlations are captured exactly:
  REACH        : reach_margin(p, +30 nm reach-extending stalk) > 0          (Ch 4)
  SELECTIVITY  : therapeutic_index(p, ...) >= 3 under a chosen scenario      (Ch 5)
  KILL-GEOMETRY: a fixed short tether (~15 nm) can seat the lytic synapse    (Ch 6)

The reach-vs-kill geometry is reported for one fixed tether length: a long tether
reaches the outer-face azide but holds the membranes too far apart to kill, while a
short tether kills fine but barely reaches. No single fixed covalent tether does
both, which is one of the findings that retire the covalent-tether architecture in
favour of the engager (see synapse_geometry.py and Chapter 6/8).
"""
import os, sys, json
import numpy as np
sys.path.insert(0, os.path.dirname(os.path.abspath(__file__)))
import mck_model as M
ROOT = os.path.dirname(os.path.dirname(os.path.abspath(__file__)))


def run(N=20000, seed=42, S3_endogenous=2.7, S3_spatial=45.0):
    rng = np.random.default_rng(seed)
    # one full-parameter sample per world (same samples used for reach + selectivity)
    S = [{k: M.sample_value(k, rng.random()) for k in M.PARAM_SPACE} for _ in range(N)]
    # extra geometry nuisance draws (as in synapse_geometry.py), same seed stream
    muc_res = rng.uniform(0.03, 0.30, N)      # residual coat fraction after mucinase
    d_lytic = rng.uniform(12, 18, N)          # lytic-competent membrane distance (nm)

    # reach: a 30 nm reach-extending stalk clears the NK coat (azide at outer face)
    reach = np.array([M.reach_margin(p, 30.0) > 0 for p in S])
    coat = np.array([p["gc_thickness"] for p in S])
    coat_residual = coat * muc_res
    # kill-geometry for a FIXED short (15 nm) tether: residual coat is the hard spacer
    kill_short = (np.maximum(15.0, coat_residual) <= d_lytic)

    def sel(ratio, a0=3.8):
        return np.array([M.therapeutic_index(p, a0, mode="mechanistic", p_base_normal_ratio=ratio) >= 3 for p in S])

    scen = {
        "floor (symmetric P_base, no 3rd gate)":         sel(1.0, 3.8),
        "corrected biology (ratio 0.5, no 3rd gate)":    sel(0.5, 3.8),
        "corrected + endogenous 3rd gate (S3=2.7)":      sel(0.5, 3.8 * S3_endogenous),
        "corrected + spatial 3rd gate (S3=45, NIR/FUS)": sel(0.5, 3.8 * S3_spatial),
    }
    out = {"N": N, "S3_endogenous": S3_endogenous, "S3_spatial": S3_spatial,
           "marginals": {"reach_pct": round(float(reach.mean() * 100), 1),
                         "kill_geom_fixed_short_pct": round(float(kill_short.mean() * 100), 1)},
           "scenarios": {}}
    for name, s in scen.items():
        out["scenarios"][name] = {
            "selective_pct": round(float(s.mean() * 100), 1),
            "joint_reach_and_selective_pct": round(float((reach & s).mean() * 100), 2),
            "joint_reach_selective_killshort_pct": round(float((reach & s & kill_short).mean() * 100), 2),
        }
    return out


if __name__ == "__main__":
    R = run(); os.makedirs(os.path.join(ROOT, "results"), exist_ok=True)
    json.dump(R, open(os.path.join(ROOT, "results", "joint_feasibility.json"), "w"), indent=2)
    m = R["marginals"]
    print("=" * 84)
    print("COMPOUND FEASIBILITY  (reach AND selective [AND fixed-short kill-geometry], N=%d)" % R["N"])
    print("=" * 84)
    print(f"marginals: reach {m['reach_pct']}%  |  fixed-short kill-geometry {m['kill_geom_fixed_short_pct']}%")
    print(f"\n{'scenario':<48}{'select%':>8}{'reach&sel%':>12}{'+killshort%':>13}")
    for n, s in R["scenarios"].items():
        print(f"{n:<48}{s['selective_pct']:>8}{s['joint_reach_and_selective_pct']:>12}{s['joint_reach_selective_killshort_pct']:>13}")
    print("\nSaved -> results/joint_feasibility.json")
