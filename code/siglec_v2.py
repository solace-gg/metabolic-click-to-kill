#!/usr/bin/env python3
"""
siglec_v2.py - Siglec-relief sweep (novel addition), v2 driver (2026-07-06)
===========================================================================
Rebuilt on the v2 model core (mck_model.py) instead of the old comparative
wrapper. This is where I test the thesis's new hypothesis: MCK's azide label, by
swapping out a fraction of the surface sialic acid, makes the Siglec ligands
worse and so eases off the Siglec inhibitory brake on NK killing of the cancer
cell. That's a cancer-arm-only effect.

Two-parameter grid (the relative formula, cancer arm only):
  p = siglec_penalty        [0..0.7]  baseline Siglec brake strength (estimated)
  q = azide_ligand_quality  [0..1]    q=1 -> label does nothing (sanity: mod=1)
                                       q=0 -> label fully kills Siglec binding
  f = fraction_labelled_sia [fixed 0.5, ~SiaNAz 51% from Sci Rep 2022]

  mod(p,q,f) = (1 - p*(1 - f*(1-q))) / (1 - p)      # cancer P_base multiplier
  sanity: mod(p, q=1, f) == 1 for all p.

The relief widens the therapeutic window by making the cancer more killable while
leaving the normal arm alone. Reported as the TI matrix plus the relief factor
TI(q)/TI(q=1). Labelling anchored to Wang 3.8x, mechanistic mode.

Usage: python notebooks_v2/siglec_v2.py
"""
import os, sys, json
SCRIPT_DIR = os.path.dirname(os.path.abspath(__file__))
if SCRIPT_DIR not in sys.path: sys.path.insert(0, SCRIPT_DIR)
import mck_model as M

WANG_A0 = 3.8
GRID_P = [0.0, 0.3, 0.5, 0.7]
GRID_Q = [0.0, 0.3, 0.5, 0.7, 1.0]
F = 0.5


def mod_factor(p, q, f):
    if p >= 1.0: raise ValueError("p must be < 1")
    return (1 - p * (1 - f * (1 - q))) / (1 - p)


def build():
    pc = {k: M.central(k) for k in M.PARAM_SPACE}
    ti_noSiglec = M.therapeutic_index(pc, WANG_A0, siglec_mod=1.0, mode="mechanistic")
    matrix, relief = {}, {}
    for p in GRID_P:
        row = {}
        ti_q1 = M.therapeutic_index(pc, WANG_A0, siglec_mod=mod_factor(p, 1.0, F), mode="mechanistic")
        for q in GRID_Q:
            m = mod_factor(p, q, F)
            ti = M.therapeutic_index(pc, WANG_A0, siglec_mod=m, mode="mechanistic")
            row[q] = {"mod": round(m, 3), "TI": round(ti, 3)}
        matrix[p] = row
        # relief = window with the label doing its full thing (q=0) vs doing nothing (q=1)
        relief[p] = round(row[0.0]["TI"] / ti_q1, 3) if ti_q1 > 1e-9 else None
    return {"f_fraction_labelled_sia": F, "labelling_sel_anchor": WANG_A0,
            "TI_no_siglec_reference": round(ti_noSiglec, 3),
            "grid": matrix, "relief_factor_q0_vs_q1": relief,
            "sanity_q1_mod_is_1": all(abs(mod_factor(p, 1.0, F) - 1.0) < 1e-12 for p in GRID_P)}


if __name__ == "__main__":
    S = build()
    resdir = os.path.join(os.path.dirname(SCRIPT_DIR), "results")
    os.makedirs(resdir, exist_ok=True)
    out = os.path.join(resdir, "round6_siglec_v2.json")
    with open(out, "w") as f: json.dump(S, f, indent=2)

    print("=" * 62); print("Siglec-relief sweep (novel addition) - TI matrix"); print("=" * 62)
    print(f"Reference TI (no Siglec brake): {S['TI_no_siglec_reference']}   f={F}, A0=3.8x")
    print(f"Sanity (q=1 -> mod=1): {S['sanity_q1_mod_is_1']}\n")
    hdr = "  p\\q  " + "".join(f"{q:>8}" for q in GRID_Q)
    print(hdr); print("  " + "-" * (len(hdr) - 2))
    for p in GRID_P:
        cells = "".join(f"{S['grid'][p][q]['TI']:>8.2f}" for q in GRID_Q)
        print(f"  {p:<4}{cells}   relief x{S['relief_factor_q0_vs_q1'][p]}")
    print("\n(columns = azide ligand quality q; q=1 no effect, q=0 full relief)")
    print(f"\nSaved -> {out}")
