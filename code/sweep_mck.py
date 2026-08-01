#!/usr/bin/env python3
"""
sweep_mck.py - MCK Phase 0 production Monte-Carlo driver (rebuild, 2026-07-06)
=============================================================================
A thin driver on top of the model core (mck_model.py). It produces the numbers
that fill the Chapter 4-6 [rerun] blanks and every figure. Nothing deprecated is
touched here.

Everything anchors labelling selectivity to Wang's measured 3.8x as the headline;
the model-idealised A0 only shows up as a ceiling. What it reports:
  * TI distribution (mechanistic): median, IQR, fraction >= 3.
  * Reach-feasible fractions (azide at outer face): naive / XTEN +20nm / rigid stalk +30nm.
  * Joint feasibility: reach-feasible AND TI >= 3 - this is the honest headline.
  * P_base response curve (deterministic; the two-axis / unmasking result).
  * ctsl_E0 one-at-a-time tornado on model A0 (the dominant lever).
  * Borrowed-multiplier sensitivity (folded/layered), so we can see how much any
    'TI>=3' claim would lean on the Niu multiplier we're not allowed to use.

Usage:
  python notebooks_v2/sweep_mck.py --n 500      # smoke test (fast)
  python notebooks_v2/sweep_mck.py --n 20000    # production (run on laptop)
"""
import os, sys, json, time, argparse
import numpy as np

SCRIPT_DIR = os.path.dirname(os.path.abspath(__file__))
if SCRIPT_DIR not in sys.path: sys.path.insert(0, SCRIPT_DIR)
import mck_model as M

WANG_A0 = 3.8   # Wang 2017 measured cancer/normal labelling selectivity (our anchor)


def _pct(x, q): return float(np.percentile(x, q)) if len(x) else float("nan")


def run_sweep(N, seed=42, compute_model_a0=True):
    """Monte-Carlo over the full parameter space. Returns dict of arrays."""
    rng = np.random.default_rng(seed)
    ti_wang, ti_model, a0_model = [], [], []
    r_naive, r_xten, r_rigid = [], [], []
    n_a0_fail = 0
    for _ in range(N):
        p = {k: M.sample_value(k, rng.random()) for k in M.PARAM_SPACE}
        # reach (cheap)
        r_naive.append(M.reach_margin(p, 0.0))
        r_xten.append(M.reach_margin(p, 20.0))
        r_rigid.append(M.reach_margin(p, 30.0))
        # TI anchored to Wang measured (headline) - independent of model A0
        ti_wang.append(M.therapeutic_index(p, WANG_A0, mode="mechanistic"))
        # model-idealised ceiling (needs the ODE; optional/subsample for speed)
        if compute_model_a0:
            a0 = M.a0_selectivity(p)
            if a0 is None or not np.isfinite(a0):
                n_a0_fail += 1
            else:
                a0_model.append(a0)
                ti_model.append(M.therapeutic_index(p, a0, mode="mechanistic"))
    return dict(
        ti_wang=np.array(ti_wang), ti_model=np.array(ti_model),
        a0_model=np.array(a0_model),
        r_naive=np.array(r_naive), r_xten=np.array(r_xten), r_rigid=np.array(r_rigid),
        n=N, n_a0_fail=n_a0_fail, seed=seed,
    )


def pbase_curve(pts=(0.5, 0.30, 0.20, 0.10, 0.05, 0.02, 0.01)):
    """Deterministic TI vs baseline NK killability at Wang 3.8x (central else)."""
    pc = {k: M.central(k) for k in M.PARAM_SPACE}
    out = []
    for pb in pts:
        q = dict(pc); q["P_base"] = pb
        out.append((pb, M.therapeutic_index(q, WANG_A0, mode="mechanistic")))
    return out


def pbase_asymmetry_curve(ratios=(1.0, 0.75, 0.5, 0.25, 0.1)):
    """Appendix sensitivity: normal cells MORE NK-protected than cancer.
    normal P_base = ratio * cancer P_base. ratio=1 is the conservative headline.
    Implemented by scaling the normal arm only (siglec_mod applies to cancer;
    here we recompute the ratio directly for transparency)."""
    pc = {k: M.central(k) for k in M.PARAM_SPACE}
    K = pc["K_half"]; A_c = min(1e7, pc["azide_density"]); Pb = pc["P_base"]
    hill = lambda A: A**2 / (K**2 + A**2)
    pk = lambda A, Pbb: Pbb + (1 - Pbb) * hill(A) * 0.9
    out = []
    for r in ratios:
        pkc = pk(A_c, Pb)
        pkn = pk(A_c / WANG_A0, Pb * r)     # normal arm less killable at baseline
        out.append((r, pkc / max(pkn, 1e-20)))
    return out


def ctsl_tornado(lo_hi=None):
    """One-at-a-time swing of model A0 as ctsl_E0 spans its range (dominant lever)."""
    pc = {k: M.central(k) for k in M.PARAM_SPACE}
    lo, hi = M.PARAM_SPACE["ctsl_E0"]
    out = {}
    for lbl, val in [("ctsl_E0_low", lo), ("ctsl_E0_central", M.central("ctsl_E0")),
                     ("ctsl_E0_high", hi)]:
        q = dict(pc); q["ctsl_E0"] = val
        a0 = M.a0_selectivity(q)
        out[lbl] = None if a0 is None else round(float(a0), 2)
    return out


def summarize(R):
    tw = R["ti_wang"]
    joint = np.mean((R["r_rigid"] > 0) & (tw >= 3.0))
    s = {
        "n": R["n"], "n_a0_fail": R["n_a0_fail"], "seed": R["seed"],
        "TI_wang_measured_3.8x": {
            "median": _pct(tw, 50), "IQR": [_pct(tw, 25), _pct(tw, 75)],
            "frac_ge_3": float(np.mean(tw >= 3.0)), "frac_ge_2": float(np.mean(tw >= 2.0)),
        },
        "reach_feasible_frac": {
            "naive": float(np.mean(R["r_naive"] > 0)),
            "XTEN_+20nm": float(np.mean(R["r_xten"] > 0)),
            "rigid_+30nm": float(np.mean(R["r_rigid"] > 0)),
        },
        "JOINT_reach_and_TIge3_rigidstalk": float(joint),
        "P_base_curve_TI": [[round(pb, 3), round(ti, 3)] for pb, ti in pbase_curve()],
        "P_base_asymmetry_appendix": [[r, round(ti, 3)] for r, ti in pbase_asymmetry_curve()],
        "ctsl_E0_tornado_on_model_A0": ctsl_tornado(),
    }
    if len(R["ti_model"]):
        tm = R["ti_model"]; a0 = R["a0_model"]
        s["TI_model_idealised_ceiling"] = {
            "median": _pct(tm, 50), "IQR": [_pct(tm, 25), _pct(tm, 75)],
            "frac_ge_3": float(np.mean(tm >= 3.0)),
        }
        s["A0_model_idealised"] = {"median": _pct(a0, 50), "IQR": [_pct(a0, 25), _pct(a0, 75)]}
    return s


if __name__ == "__main__":
    ap = argparse.ArgumentParser()
    ap.add_argument("--n", type=int, default=500)
    ap.add_argument("--seed", type=int, default=42)
    ap.add_argument("--no-model-a0", action="store_true",
                    help="skip per-sample ODE A0 (much faster; keeps Wang-anchored headline)")
    ap.add_argument("--out", default=None)
    a = ap.parse_args()

    t0 = time.time()
    R = run_sweep(a.n, seed=a.seed, compute_model_a0=not a.no_model_a0)
    S = summarize(R)
    S["runtime_s"] = round(time.time() - t0, 1)

    resdir = os.path.join(os.path.dirname(SCRIPT_DIR), "results")
    os.makedirs(resdir, exist_ok=True)
    out = a.out or os.path.join(resdir, f"round6_sweep_n{a.n}.json")
    with open(out, "w") as f: json.dump(S, f, indent=2)

    print("=" * 68); print(f"MCK production sweep  N={a.n}  ({S['runtime_s']}s)"); print("=" * 68)
    tw = S["TI_wang_measured_3.8x"]
    print(f"\nTI (Wang measured 3.8x, mechanistic): median {tw['median']:.2f}  "
          f"IQR [{tw['IQR'][0]:.2f}, {tw['IQR'][1]:.2f}]  frac>=3 {tw['frac_ge_3']:.1%}")
    rf = S["reach_feasible_frac"]
    print(f"Reach feasible: naive {rf['naive']:.1%} | XTEN {rf['XTEN_+20nm']:.1%} | "
          f"rigid {rf['rigid_+30nm']:.1%}")
    print(f"JOINT (rigid reach AND TI>=3): {S['JOINT_reach_and_TIge3_rigidstalk']:.1%}")
    print("\nP_base curve (unmasking result):")
    for pb, ti in S["P_base_curve_TI"]:
        print(f"  P_base={pb:<5} -> TI {ti:.2f}")
    print("\nctsl_E0 tornado on model A0 (dominant lever):", S["ctsl_E0_tornado_on_model_A0"])
    if "A0_model_idealised" in S:
        am = S["A0_model_idealised"]
        print(f"\nModel-idealised A0 ceiling: median {am['median']:.1f}x  "
              f"IQR [{am['IQR'][0]:.1f}, {am['IQR'][1]:.1f}]  (headline uses Wang 3.8x, NOT this)")
    print(f"\nSaved -> {out}")
