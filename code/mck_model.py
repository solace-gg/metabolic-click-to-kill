#!/usr/bin/env python3
"""
mck_model.py - MCK Phase 0 model core (rebuild, 2026-07-06)
===========================================================
This is the one place the MCK model actually lives. I pulled the logic out of the
7 separate sweep scripts it used to be copy-pasted across - that duplication is how
the azide_density-stays-inert bug wound up in every file at once. The drivers all
just import from here now.

What changed in this rebuild (settled 2026-07-06):
  * The primary reach model puts the azide at the tumour outer face, following
    Mockl 2020 (sialic acid measured out at the distal edge on primary human
    tumours). I don't sample the buried-azide version anymore - it was really an
    artifact of treating azide height as if it were independent of coat
    thickness. So reach comes down to 2*R_F + bonus >= h_nk.
  * gc_thickness now runs up to 500 nm for the mucin-high indication (was 30-150;
    MUC4 can hit ~2 um). With the azide at the outer face this cancels out of
    reach - it drives kill/decoupling instead.
  * Therapeutic index is mechanistic by default (from the 2026-07-06 critical
    eval). MCK's selectivity is entirely a function of where the azide sits: the
    armed NK clicks in proportion to azide density, so the click's selectivity
    just is the labelling selectivity carried through the kill Hill. The old
    1.44-1.70x "click-attributable selectivity" came from Niu's polymannose
    receptor-avidity system, a different mechanism MCK doesn't have - so it's out
    of the headline and kept only as a labelled sensitivity
    (mode='layered'/'folded').
  * azide_density feeds the cancer kill signal directly (A_cancer = azide_density),
    no accessibility factor (matches the A3 convention; K_half on total azide).
  * Sampling classification (log vs linear) follows Anchor 3.3.
  * Selectivity is anchored to real cancer/normal enzyme ratios (DepMap x GTEx),
    with Wang's measured 3.8x reported next to the model-idealised value.

No deprecated scripts or data feed into this. Brush compression comes from the
polydisperse module.
"""
import os, sys, math
import numpy as np
from scipy.integrate import solve_ivp

SCRIPT_DIR = os.path.dirname(os.path.abspath(__file__))
NB = SCRIPT_DIR   # polydisperse_brush.py lives alongside this file
if NB not in sys.path: sys.path.insert(0, NB)

# ---------------------------------------------------------------------------
# 1. Parameter space + sampling classification (Anchor 3.3)
# ---------------------------------------------------------------------------
PARAM_SPACE = {
    # enzyme / A0 (LOG: rates, concentrations, folds)
    "hdac_kcat": (0.05, 2.0), "hdac_expr": (2, 20), "hdac_E0": (20e-9, 100e-9),
    "ctsl_kcat": (5, 25.8), "ctsl_expr": (50, 200), "ctsl_E0": (0.05e-9, 0.5e-9),
    "k_clear_prodrug": (5e-5, 3e-4), "metabolic_factor": (1.5, 4.0),
    "K_half": (2e5, 1e6), "azide_density": (1e5, 1e7),
    # bounded / dimensional (LINEAR)
    "ctsl_pH": (0.25, 0.5), "P_base": (0.01, 0.5),
    "click_attributable_selectivity": (1.0, 2.5),
    "gc_thickness": (30, 500),    # extended for the mucin-high indication (was 30-150)
    "gc_nk": (10, 40), "nk_force_pN": (10.0, 100.0), "peg_mw": (2000, 5000),
    "contact_area_um2": (1.0, 5.0),
}
LOG_PARAMS = {"hdac_kcat","hdac_expr","hdac_E0","ctsl_kcat","ctsl_expr","ctsl_E0",
              "k_clear_prodrug","metabolic_factor","K_half","azide_density"}

def sample_value(name, u):
    """Inverse-CDF map of uniform u in [0,1] to the parameter, log or linear."""
    lo, hi = PARAM_SPACE[name]
    if name in LOG_PARAMS:
        return math.exp(math.log(lo) + (math.log(hi) - math.log(lo)) * u)
    return lo + (hi - lo) * u

def central(name):
    lo, hi = PARAM_SPACE[name]
    return math.sqrt(lo*hi) if name in LOG_PARAMS else 0.5*(lo+hi)

# ---------------------------------------------------------------------------
# 2. A0 dual-lock selectivity (the ODE; this is the enzymatic labelling selectivity)
# ---------------------------------------------------------------------------
def _dual_lock_ode(t, y, hk, hKm, hE, ck, cKm, cE, cpH, kpro, kint, kact):
    S, I, A = y
    v1 = hk*hE*max(S,0)/(hKm+max(S,0))
    v2 = ck*cE*cpH*max(I,0)/(cKm+max(I,0))
    return [-v1-kpro*S, v1-v2-kint*I, v2-kact*A]

def a0_selectivity(p):
    """Cancer/normal azide-labelling ratio from the HDAC x CTSL dual-lock ODE."""
    PRO, Kmh, Kmc, ts = 10e-6, 50e-6, 2.2e-6, (0, 24*3600)
    try:
        sc = solve_ivp(_dual_lock_ode, ts, [PRO,0,0], method="LSODA",
            args=(p["hdac_kcat"],Kmh,p["hdac_E0"]*p["hdac_expr"],p["ctsl_kcat"],Kmc,
                  p["ctsl_E0"]*p["ctsl_expr"],p["ctsl_pH"],p["k_clear_prodrug"],2e-4,5e-5),
            rtol=1e-6, atol=1e-12)
        sn = solve_ivp(_dual_lock_ode, ts, [PRO,0,0], method="LSODA",
            args=(p["hdac_kcat"],Kmh,p["hdac_E0"],p["ctsl_kcat"],Kmc,p["ctsl_E0"],0.05,
                  p["k_clear_prodrug"],2e-4,5e-5), rtol=1e-6, atol=1e-12)
        if not (sc.success and sn.success): return None
        Ac = sc.y[2,-1]*p["metabolic_factor"]; An = sn.y[2,-1]
        return Ac/An if An > 1e-25 else float("inf")
    except Exception:
        return None

# ---------------------------------------------------------------------------
# 3. Reach - primary model: azide at outer face (Mockl 2020). h_t cancels.
# ---------------------------------------------------------------------------
try:
    from polydisperse_brush import compressed_height_polydisperse
    _HAVE_BRUSH = True
except Exception:
    _HAVE_BRUSH = False

def _h_nk_compressed(p, D=1.4):
    if _HAVE_BRUSH:
        return compressed_height_polydisperse(p["nk_force_pN"], p["gc_nk"],
                    polydispersity_D=D, contact_area_um2=p["contact_area_um2"])
    return p["gc_nk"] * 0.99   # fallback: ~1% compression (force-comp finding)

def reach_margin(p, reach_bonus_nm=0.0):
    """Azide at the tumour outer face, so h_t cancels and feasibility is just
    2*R_F(PEG) + reach_bonus >= h_nk_compressed. Mucinase only scales h_t, so it
    does nothing for reach here - that falls straight out of the outer-face
    assumption (see Mockl 2020)."""
    R_F = 0.35 * (p["peg_mw"]/44.0)**0.588
    return 2*R_F + reach_bonus_nm - _h_nk_compressed(p)

# ---------------------------------------------------------------------------
# 4. Therapeutic index - mechanistic (no borrowed click multiplier)
# ---------------------------------------------------------------------------
def therapeutic_index(p, a0_sel, click_sel=1.0, n_hill=2.0, mode="mechanistic", p_base_normal_ratio=1.0,
                      siglec_mod=1.0):
    """
    A_cancer = azide_density (no accessibility factor; K_half on total azide).
    P_kill = P_base + (1-P_base)*Hill(A)*0.9. Cancer P_base modified by Siglec.

    mode='mechanistic' (the primary one): MCK's selectivity is entirely set by
        where the azide is - the armed NK clicks in proportion to azide density,
        so the click's selectivity just is the labelling selectivity carried
        through the kill Hill. There's no separate click-attributable multiplier
        for MCK's covalent mechanism (the 1.44-1.70x was borrowed from Niu's
        polymannose receptor-avidity system, a different mechanism).
        A_normal = A_cancer/a0_sel; TI = pkill_c/pkill_n. click_sel is ignored.
    mode='layered'/'folded': kept only to show what happens if you (wrongly)
        import Niu's borrowed multiplier - a sensitivity, not a headline.
    """
    K = p["K_half"]; A_c = min(1e7, p["azide_density"])
    Pb = p["P_base"]; Pb_c = min(0.98, Pb*siglec_mod); Pb_n = Pb * p_base_normal_ratio
    hill = lambda A: A**n_hill/(K**n_hill + A**n_hill)
    pkill = lambda A, Pbb: Pbb + (1-Pbb)*hill(A)*0.9
    if mode == "folded":
        A_n = A_c/(a0_sel*click_sel)
        return pkill(A_c, Pb_c)/max(pkill(A_n, Pb_n), 1e-20)
    A_n = A_c/a0_sel
    ti_label = pkill(A_c, Pb_c)/max(pkill(A_n, Pb_n), 1e-20)
    return ti_label * click_sel if mode == "layered" else ti_label  # mechanistic

# ---------------------------------------------------------------------------
# 5. Smoke test / before-after (run: python notebooks_v2/mck_model.py)
# ---------------------------------------------------------------------------
if __name__ == "__main__":
    print("="*70); print("MCK model core - smoke test (2026-07-06 rebuild)"); print("="*70)
    pc = {k: central(k) for k in PARAM_SPACE}
    print("\nCentral-trajectory parameters (log geomean / linear mid):")
    for k in ["azide_density","K_half","P_base","hdac_expr","ctsl_expr","gc_nk","peg_mw"]:
        print(f"  {k:22} = {pc[k]:.4g}")

    a0 = a0_selectivity(pc)
    print(f"\nA0 model idealised selectivity (central): {a0:.2f}x   (Wang MEASURED: 3.8x)")

    print("\n--- THERAPEUTIC INDEX: mechanistic (PRIMARY) vs borrowed-multiplier sensitivity ---")
    print(f"{'A0 labelling sel':>18} | {'mechanistic':>12} {'folded':>8} {'layered':>8}")
    for a0sel, lbl in [(3.8,"Wang MEASURED"),(a0,"model ideal.")]:
        tm = therapeutic_index(pc, a0sel, mode="mechanistic")
        tf = therapeutic_index(pc, a0sel, 1.66, mode="folded")
        tl = therapeutic_index(pc, a0sel, 1.66, mode="layered")
        print(f"{lbl:>13} {a0sel:>4.1f}x | {tm:>12.2f} {tf:>8.2f} {tl:>8.2f}")
    print("  -> MECHANISTIC is the honest MCK number (no borrowed multiplier).")
    print("  -> At Wang measured 3.8x labelling, mechanistic TI ~1.9 (<3): borderline.")

    print("\n  P_base (NK-baseline killability) sweep at Wang 3.8x, mechanistic:")
    for pb in (0.5, 0.255, 0.10, 0.05, 0.02):
        pp = dict(pc); pp["P_base"] = pb
        print(f"    P_base={pb:<6} -> TI {therapeutic_index(pp, 3.8, mode='mechanistic'):.2f}")
    print("    (lower baseline kill -> enzyme/azide selectivity shows through undiluted)")

    print("\n--- REACH (h_t cancels; mucinase inert; only reach-extension matters) ---")
    rng = np.random.default_rng(42); N = 20000; feas = {0:0, 20:0, 30:0}
    for _ in range(N):
        p = {k: sample_value(k, rng.random()) for k in PARAM_SPACE}
        for bonus in feas:
            if reach_margin(p, bonus) > 0: feas[bonus] += 1
    print(f"  naive (bonus=0):      {feas[0]/N:.2%} reach-feasible")
    print(f"  XTEN (bonus=20nm):    {feas[20]/N:.2%}")
    print(f"  rigid stalk (30nm):   {feas[30]/N:.2%}  <- flagship reach lever")
    print("\nSmoke test complete. Brush module loaded:", _HAVE_BRUSH)
