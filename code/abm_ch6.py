#!/usr/bin/env python3
"""abm_ch6.py - MCK Phase 0 killing summary (Ch 6) v2. A killing ladder across Niu's
E:T ratios using the calibrated params. fully_pure_mck (bond mechanics only) is the
headline; Niu_upper_bound (engage_boost=2.5, polymannose receptor frequency) is a
ceiling only. Same call as the mechanistic-TI decision: don't credit MCK with a
mechanism it doesn't have."""
import os, sys, json, time, csv, argparse
import numpy as np
SCRIPT_DIR = os.path.dirname(os.path.abspath(__file__))
ROOT = os.path.dirname(SCRIPT_DIR); NB = os.path.join(ROOT, "notebooks")
for p in (SCRIPT_DIR, NB):
    if p not in sys.path: sys.path.insert(0, p)
os.environ.setdefault("MPLBACKEND", "Agg")
import ABM_fast as ABM
NIU = {"NK": {1: 8.1, 5: 22.8, 10: 43.2}, "pMNK": {1: 26.7, 5: 47.8, 10: 72.5}}
ET_RATIOS = [1, 5, 10]; NIU_N = 80
def load_calib():
    d = {}
    for row in csv.DictReader(open(os.path.join(ROOT, "data", "abm_calibrated_params.csv"))):
        d[row["parameter"]] = row["value"]
    return dict(p_engage=float(d["p_engage"]), p_kill=float(d["p_kill_per_contact"]),
        speed=float(d["nk_speed_mean"]), engage_boost=float(d["engage_boost_pmnk"]),
        p_click_bonus=float(d["p_click_bonus"]), bond_thresh=int(float(d["bond_threshold_for_bonus"])))
def arm(et, reps, seed, C, **over):
    kw = dict(p_engage=C["p_engage"], p_kill=C["p_kill"], speed=C["speed"],
              p_click_bonus=C["p_click_bonus"], bond_thresh=C["bond_thresh"]); kw.update(over)
    vals = [ABM.run_abm(NIU_N, max(1, int(NIU_N*et)), seed=seed+r, **kw)["lysis"] for r in range(reps)]
    sd = float(np.std(vals, ddof=1)) if len(vals) > 1 else 0.0
    return float(np.mean(vals)), sd
MODES = {"NK_baseline": dict(click_enhanced=False),
    "fully_pure_mck": dict(click_enhanced=True, engage_boost=1.0, fully_pure_mck=True),
    "MCK_pure_dur": dict(click_enhanced=True, engage_boost=1.0, fully_pure_mck=False),
    "Niu_upper_bound": dict(click_enhanced=True, engage_boost=2.5, fully_pure_mck=False)}
def run(reps=4, seed=42):
    C = load_calib(); out = {"calibrated_params": C, "reps": reps, "n_tumour": NIU_N, "by_et": {}}
    for et in ET_RATIOS:
        row = {}
        for name, over in MODES.items():
            m, s = arm(et, reps, seed, C, **over)
            row[name] = {"lysis_pct": round(m,2), "sd": round(s,2), "se": round(s/(reps**0.5),2)}
        base = row["NK_baseline"]["lysis_pct"]
        for name in ("fully_pure_mck", "MCK_pure_dur", "Niu_upper_bound"):
            row[name]["fold_vs_baseline"] = round(row[name]["lysis_pct"]/base, 2) if base > 1e-6 else None
        row["niu_target_NK"] = NIU["NK"][et]; row["niu_target_pMNK"] = NIU["pMNK"][et]
        out["by_et"][et] = row
    folds = [out["by_et"][et]["fully_pure_mck"]["fold_vs_baseline"] for et in ET_RATIOS
             if out["by_et"][et]["fully_pure_mck"]["fold_vs_baseline"]]
    out["HEADLINE_fully_pure_mck_fold_median"] = round(float(np.median(folds)), 2) if folds else None
    return out
if __name__ == "__main__":
    ap = argparse.ArgumentParser(); ap.add_argument("--reps", type=int, default=4)
    ap.add_argument("--seed", type=int, default=42); ap.add_argument("--out", default=None)
    a = ap.parse_args()
    t0 = time.time(); S = run(a.reps, a.seed); S["runtime_s"] = round(time.time()-t0, 1)
    resdir = os.path.join(ROOT, "results"); os.makedirs(resdir, exist_ok=True)
    out = a.out or os.path.join(resdir, f"round6_abm_ch6_reps{a.reps}.json")
    with open(out, "w") as f: json.dump(S, f, indent=2)
    print("="*74); print(f"ABM Ch6 killing ladder (reps={a.reps}, calibrated, {S['runtime_s']}s)"); print("="*74)
    print(f"{'E:T':>4} | {'NK(Niu)':>11} {'pureMCK':>12} {'+dur':>10} {'Niu-up(Niu)':>15}")
    for et in ET_RATIOS:
        r = S["by_et"][et]
        print(f"{et:>3}:1 | {r['NK_baseline']['lysis_pct']:>5}({r['niu_target_NK']:>4}) "
              f"{r['fully_pure_mck']['lysis_pct']:>5}[{r['fully_pure_mck']['fold_vs_baseline']}x] "
              f"{r['MCK_pure_dur']['lysis_pct']:>6}[{r['MCK_pure_dur']['fold_vs_baseline']}x] "
              f"{r['Niu_upper_bound']['lysis_pct']:>5}[{r['Niu_upper_bound']['fold_vs_baseline']}x]({r['niu_target_pMNK']:>4})")
    print(f"\nHEADLINE (fully_pure_mck median fold): {S['HEADLINE_fully_pure_mck_fold_median']}x")
    print(f"Saved -> {out}")
