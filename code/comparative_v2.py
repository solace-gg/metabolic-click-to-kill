#!/usr/bin/env python3
"""
comparative_v2.py - adjunct modularity story (Ch 5/8), v2 driver (2026-07-06)
=============================================================================
Rebuilt on the v2 model core. This backs up MCK's platform-modularity claim: the
two candidate adjuncts act on different necessary conditions, so they're
complementary rather than redundant. The twist is where the azide sits, which
changes an important detail compared to a naive comparative grid:

  * the reach-extender (rigid stalk / XTEN, +20-30 nm) acts on reach. It takes the
    reach-feasible fraction from ~1% to ~97% - the big reach lever.
  * mucinase acts on the kill step, not reach. With the azide at the tumour outer
    face (Mockl 2020) the coat height h_t cancels out of the reach balance, so
    chewing up the coat doesn't change reach feasibility at all. What it buys you
    is relief of the coat steric gate on lytic-granule delivery, which the ABM
    (abm_ch6.py) quantifies. We cross-reference that here rather than
    double-counting it in reach.

So the honest modularity table is: reach-extender fixes reach, mucinase fixes
kill, and together they cover two different conditions. Neither adjunct touches
selectivity (the enzyme gate).

Labelling anchored to Wang 3.8x, mechanistic TI. Monte-Carlo over the full space.
Usage: python notebooks_v2/comparative_v2.py --n 20000
"""
import os, sys, json, argparse
import numpy as np
SCRIPT_DIR = os.path.dirname(os.path.abspath(__file__))
if SCRIPT_DIR not in sys.path: sys.path.insert(0, SCRIPT_DIR)
import mck_model as M

WANG_A0 = 3.8
CONFIGS = {   # reach_bonus_nm, mucinase acts on kill (reach-inert, azide at outer face)
    "MCK_alone":            {"reach_bonus": 0.0,  "mucinase": False},
    "MCK+reach_extender":   {"reach_bonus": 30.0, "mucinase": False},
    "MCK+mucinase":         {"reach_bonus": 0.0,  "mucinase": True},
    "MCK+both":             {"reach_bonus": 30.0, "mucinase": True},
}


def run(N, seed=42):
    rng = np.random.default_rng(seed)
    # draw one shared sample set so every config is compared on the same draws
    P = [{k: M.sample_value(k, rng.random()) for k in M.PARAM_SPACE} for _ in range(N)]
    ti = np.array([M.therapeutic_index(p, WANG_A0, mode="mechanistic") for p in P])
    ti_ok = ti >= 3.0
    out = {"n": N, "labelling_sel_anchor": WANG_A0, "configs": {}}
    for name, c in CONFIGS.items():
        reach = np.array([M.reach_margin(p, c["reach_bonus"]) > 0 for p in P])
        joint = float(np.mean(reach & ti_ok))
        out["configs"][name] = {
            "reach_feasible_frac": float(np.mean(reach)),
            "TI_ge3_frac": float(np.mean(ti_ok)),                 # gate-only, so the same for every adjunct
            "JOINT_reach_and_TIge3": joint,
            "mucinase_role": ("KILL-step (reach-inert; see abm_ch6.py)"
                              if c["mucinase"] else "n/a"),
        }
    return out


if __name__ == "__main__":
    ap = argparse.ArgumentParser(); ap.add_argument("--n", type=int, default=20000)
    ap.add_argument("--seed", type=int, default=42); ap.add_argument("--out", default=None)
    a = ap.parse_args()
    S = run(a.n, a.seed)
    resdir = os.path.join(os.path.dirname(SCRIPT_DIR), "results")
    os.makedirs(resdir, exist_ok=True)
    out = a.out or os.path.join(resdir, f"round6_comparative_n{a.n}.json")
    with open(out, "w") as f: json.dump(S, f, indent=2)

    print("=" * 72); print(f"Adjunct modularity (N={a.n}, Wang 3.8x, mechanistic)"); print("=" * 72)
    print(f"{'config':>20} | {'reach-feas':>10} {'TI>=3':>7} {'JOINT':>7}  mucinase")
    for name, r in S["configs"].items():
        print(f"{name:>20} | {r['reach_feasible_frac']:>9.1%} {r['TI_ge3_frac']:>7.1%} "
              f"{r['JOINT_reach_and_TIge3']:>7.1%}  {r['mucinase_role']}")
    print("\n  reach-extender -> REACH; mucinase -> KILL (complementary, not redundant).")
    print("  Selectivity (enzyme gate) is untouched by either adjunct.")
    print(f"\nSaved -> {out}")
