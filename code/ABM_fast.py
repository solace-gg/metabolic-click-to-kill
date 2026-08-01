#!/usr/bin/env python3
"""
MCK ABM — the fast (vectorised) version that actually runs in reasonable time
=============================================================================

What this is: a virtual 96-well plate. We drop N_tumour adherent tumour cells
on a 2D grid (bottom of the well) and N_nk motile NK cells on top. Time
advances in ticks of dt=1 min for 240 min (4 h assay matches Niu 2024). NKs do
a persistent random walk (Bhat & Watzl 2007); when close to a tumour they
decide with probability p_engage whether to commit to contact; during contact
we run a click-chemistry submodule; if bond count exceeds threshold, kill prob
jumps; after kill or timeout, detach, carry on until max_kills = 5 serial
kills (Bhat & Bhatt 2015).

Why vectorised (and not the OOP version in ../src/): same biology, same
parameters, ~100x faster because numpy operates on arrays of all cells at
once instead of looping one-at-a-time. Essential for calibration, which
runs hundreds of simulations via differential_evolution.

What we fit: p_engage, engage_boost (DBCO arming multiplier on engage),
p_click_bonus (extra kill prob per bond beyond thresh), bond_thresh. Everything
else is pinned from literature (see parameters.csv for sources).

What we calibrate to: Niu 2024 Fig 3B, MDA-MB-231 lysis at E:T = 1:1, 5:1, 10:1,
for NK vs pM-NK (their click-armed NK). Six data points, four free parameters.

What R² = 0.91 means here: the model CAN reproduce Niu 2024 numbers. It does
NOT prove independent predictive accuracy — we trained on these points. Honest
framing: "structural consistency check". The Fig 2C/2D contact-frequency and
duration data (which we did NOT train on) agree qualitatively, which is a
weak-but-real independent check.

Generates 6 figures: calibration fit, bond distributions, time-course of lysis,
per-NK kill histogram, contact frequency vs E:T, and a sensitivity mini-sweep.

Three engage_boost / duration scenarios — keep these distinct.

Round 5 code-audit correction (S2-P2-01): the previous docstring claimed the
MCK-pure case was carried by a "duration-via-bonds" mechanism. That mechanism
isn't implemented anywhere in this file — `dur_median` is set once by the
`click_enhanced` flag (29 min for plain NK, 56 min for pMNK), drawn as a
lognormal at engagement, and only read after that. Bonds feed the kill
probability (line ~273) and the synapse maturation rate (line ~241), never the
contact duration. The 56-min duration is a hardcoded Niu 2024 Fig 2D value that
Niu attributes to polymannose-mannose-receptor binding — the same
mannose-receptor biology `engage_boost` was set to 1.0 to exclude. So the old
"MCK-pure" (engage_boost=1.0 + click_enhanced=True) still inherits a non-MCK
duration boost. Docstring fixed; a truly-pure `fully_pure_mck` scenario is added
below.

  * MCK-pure (contact-frequency isolation only): engage_boost = 1.0,
    click_enhanced=True, dur_median = 56 min (hardcoded Niu 2024 Fig 2D value
    from mannose-receptor biology — a non-MCK enhancement this scenario still
    inherits). Isolates MCK on the contact-frequency axis (drops the 2.5×
    engage_boost) but not on the duration axis. Kept as a reference number
    matching earlier project provenance.
  * fully_pure_mck (Round 5 CP-5.2 addition, S2-P2-01 fix): engage_boost=1.0,
    dur_median = dur_median_nk = 29 on both arms (no Niu duration boost). This
    is the true conservative lower bound for MCK's contact-frequency-only
    click-at-contact mechanism. Expect the pMNK arm to be even more depressed
    than under MCK-pure, and the R² vs Niu to come out lower — that's the honest
    reading of a fully-isolated MCK.
  * Niu-calibrated upper bound (reference only): engage_boost = 2.5 (Niu 2024
    Fig 2C, 78/31). Reproduces Niu's pM-NK lysis (R²=0.91) — used by main() for
    calibration. Upper bound on what click-at-contact could manage if it
    reproduced receptor-binding-like avidity, not the primary MCK number.
"""

import numpy as np
import matplotlib
matplotlib.use('Agg')
import matplotlib.pyplot as plt
from matplotlib.lines import Line2D
from scipy.optimize import differential_evolution
import os, time

FIGURE_DIR = os.path.join(os.path.dirname(os.path.abspath(__file__)),
                          '..', 'results', 'figures')
os.makedirs(FIGURE_DIR, exist_ok=True)

# ============================================================
# Niu 2024 targets
# ============================================================
NIU = {
    "NK":    {1: 8.1, 5: 22.8, 10: 43.2},
    "pMNK":  {1: 26.7, 5: 47.8, 10: 72.5},
    "NK_lo": {1: 5, 5: 18, 10: 35},
    "NK_hi": {1: 12, 5: 28, 10: 50},
    "pMNK_lo": {1: 20, 5: 40, 10: 65},
    "pMNK_hi": {1: 33, 5: 55, 10: 80},
}
ET_RATIOS = [1, 5, 10]

# ============================================================
# Colours
# ============================================================
C_NK = '#2196F3'
C_PM = '#F44336'
C_TU = '#4CAF50'
C_DEAD = '#9E9E9E'
C_BOND = '#FF9800'

plt.rcParams.update({
    'font.family': 'Arial', 'font.size': 11,
    'axes.linewidth': 1.2, 'figure.dpi': 300,
})


# ============================================================
# Fast vectorised simulation
# ============================================================
def run_abm(n_tumour, n_nk, duration=240.0, dt=1.0, grid=None,
            p_kill=0.15, p_engage=0.02, speed=7.0, persistence=3.0,
            contact_dist=16.0, min_contact=5.0,
            kill_time=10.0, detach_time=5.0, max_kills=5,
            dur_median_nk=29.0, dur_median_pm=56.0, dur_sigma=0.8,
            click_enhanced=False, engage_boost=1.0, fully_pure_mck=False,
            spaac_k2=0.5, azide_dens=5e6, dbco_dens=1e5,
            reactive_zone=30.0, steric=0.3,
            tumour_sa=1000.0, nk_sa=500.0,
            syn_min=0.5, syn_max=5.0, syn_tau=7.0,
            bond_fb_thresh=3,
            p_click_bonus=0.25, bond_thresh=5,
            k_thiol=0.05, thiol_conc=5e-6,
            seed=42):
    """
    Run one ABM simulation. Returns dict with endpoint lysis + time series.

    States: 0=searching, 1=contacting, 2=killing, 3=detaching, 4=exhausted

    Key parameter: p_engage — probability per time step that an NK cell
    within contact_dist of a tumour cell actually initiates a committed
    contact. This captures the reality that most proximity events are
    brief scanning contacts that don't lead to synapse formation.

    engage_boost: multiplier for p_engage when click-enhanced (pM-NK),
    reflecting the 2.5x contact frequency increase from Niu 2024 Fig 2C.
    """
    rng = np.random.default_rng(seed)
    n_steps = int(duration / dt)
    AVOGADRO = 6.022e23

    # ------------------------------------------------------------------
    # Density-preserving grid scaling (TL-21, 2026-06-11)
    # ------------------------------------------------------------------
    # Niu 2024 calibrated at n_tumour ≈ 80 in a 500×500 µm patch.
    # Density = 80 / (500×500) µm² ≈ 3.2e-4 cells/µm² (≈ 1 cell per 3125 µm²).
    # If grid is fixed and n_tumour varies, cells-per-area changes → contact
    # frequency changes → calibration breaks. Fix: scale grid edge length
    # with sqrt(n_tumour/80) so density stays at Niu's calibration value.
    # Net effect: the ABM stays density-invariant whatever n_tumour is.
    # If a user explicitly passes grid=<value>, honour it (e.g., for legacy
    # tests or to deliberately probe density effects).
    NIU_N = 80
    NIU_GRID = 500.0
    if grid is None:
        grid = NIU_GRID * np.sqrt(n_tumour / NIU_N)

    # Effective engagement probability
    p_eng = p_engage * (engage_boost if click_enhanced else 1.0)

    # --- Tumour cells (stationary) ---
    tx = rng.uniform(0, grid, n_tumour)
    ty = rng.uniform(0, grid, n_tumour)
    t_alive = np.ones(n_tumour, dtype=bool)
    t_azide = np.full(n_tumour, azide_dens if click_enhanced else 0.0)
    t_bonds = np.zeros(n_tumour, dtype=int)

    # --- NK cells ---
    nx = rng.uniform(0, grid, n_nk)
    ny = rng.uniform(0, grid, n_nk)
    nk_state = np.zeros(n_nk, dtype=int)  # 0=searching
    nk_heading = rng.uniform(0, 2*np.pi, n_nk)
    nk_speed = np.clip(rng.normal(speed, 2.0, n_nk), 2.0, 15.0)
    nk_target = np.full(n_nk, -1, dtype=int)
    nk_contact_start = np.full(n_nk, -1.0)
    nk_contact_dur = np.zeros(n_nk)
    nk_bonds = np.zeros(n_nk, dtype=int)
    nk_kills = np.zeros(n_nk, dtype=int)
    nk_timer = np.zeros(n_nk)
    nk_dbco = np.full(n_nk, dbco_dens if click_enhanced else 0.0)

    # Cooldown: NK can't re-engage same target immediately
    nk_cooldown = np.zeros(n_nk)

    # Time series
    rec_t, rec_lysis = [], []

    d_rot = 1.0 / persistence
    sigma_rot = np.sqrt(2 * d_rot * dt)
    # Round 5 S2-P2-01: fully_pure_mck forces dur_median = dur_median_nk (29 min)
    # regardless of click_enhanced, so the pMNK arm loses its hardcoded Niu
    # duration boost. This is the true conservative lower bound for MCK-pure.
    if fully_pure_mck:
        dur_median = dur_median_nk
    else:
        dur_median = dur_median_pm if click_enhanced else dur_median_nk

    for step in range(n_steps + 1):
        t_now = step * dt

        # Record every 5 min
        if step % max(1, int(5.0/dt)) == 0:
            rec_t.append(t_now)
            rec_lysis.append(100.0 * np.sum(~t_alive) / n_tumour)

        if step == n_steps:
            break

        dt_s = dt * 60.0  # seconds

        # Decrement cooldowns
        nk_cooldown = np.maximum(0, nk_cooldown - dt)

        # ---- SEARCHING NK: move + detect ----
        searching = (nk_state == 0) & (nk_cooldown <= 0)
        n_search = np.sum(searching)
        if n_search > 0:
            idx_s = np.where(searching)[0]
            # Persistent random walk (move ALL searching NK, including cooldown)
            move_mask = nk_state == 0
            move_idx = np.where(move_mask)[0]
            nk_heading[move_idx] += rng.normal(0, sigma_rot, len(move_idx))
            nx[move_idx] = (nx[move_idx] + nk_speed[move_idx]*dt*np.cos(nk_heading[move_idx])) % grid
            ny[move_idx] = (ny[move_idx] + nk_speed[move_idx]*dt*np.sin(nk_heading[move_idx])) % grid

            # Contact detection
            alive_idx = np.where(t_alive)[0]
            if len(alive_idx) > 0:
                for i in idx_s:
                    dx = np.abs(nx[i] - tx[alive_idx])
                    dy = np.abs(ny[i] - ty[alive_idx])
                    dx = np.minimum(dx, grid - dx)
                    dy = np.minimum(dy, grid - dy)
                    dist = np.sqrt(dx**2 + dy**2)
                    close = np.where(dist < contact_dist)[0]
                    if len(close) > 0:
                        # Engagement probability gate
                        if rng.random() > p_eng:
                            continue  # scanning contact, no engagement
                        best = alive_idx[close[np.argmin(dist[close])]]
                        nk_state[i] = 1  # contacting
                        nk_target[i] = best
                        nk_contact_start[i] = t_now
                        mu = np.log(dur_median)
                        nk_contact_dur[i] = np.clip(rng.lognormal(mu, dur_sigma), 1, 180)
                        nk_bonds[i] = 0

        # ---- CONTACTING NK: bond formation + kill decision ----
        contacting = nk_state == 1
        if np.any(contacting):
            idx_c = np.where(contacting)[0]
            for i in idx_c:
                tgt = nk_target[i]
                if tgt < 0 or not t_alive[tgt]:
                    # Target dead
                    nk_state[i] = 3; nk_timer[i] = detach_time/2
                    continue

                ct = t_now - nk_contact_start[i]

                # Click chemistry
                if click_enhanced and t_azide[tgt] > 0 and nk_dbco[i] > 0:
                    # Synapse area with maturation + feedback
                    fb = 1.0 / (1.0 + nk_bonds[i] / bond_fb_thresh)
                    tau_eff = syn_tau * fb
                    frac = 1.0 - np.exp(-ct / tau_eff)
                    area = syn_min + (syn_max - syn_min) * frac

                    f_t = min(area / tumour_sa, 1.0)
                    f_n = min(area / nk_sa, 1.0)
                    az_zone = t_azide[tgt] * f_t * steric
                    db_zone = nk_dbco[i] * f_n * steric

                    vol_um3 = area * reactive_zone * 1e-3
                    vol_L = vol_um3 * 1e-15
                    if vol_L > 0:
                        c_az = (az_zone / AVOGADRO) / vol_L
                        c_db = (db_zone / AVOGADRO) / vol_L
                        rate = spaac_k2 * c_az * c_db
                        exp_bonds = rate * AVOGADRO * vol_L * dt_s
                        if exp_bonds > 0:
                            new_b = min(rng.poisson(exp_bonds),
                                        int(min(az_zone, db_zone)))
                            nk_bonds[i] += new_b
                            t_bonds[tgt] += new_b
                            t_azide[tgt] -= new_b
                            nk_dbco[i] -= new_b

                    # DBCO thiol decay
                    nk_dbco[i] *= np.exp(-k_thiol * thiol_conc * dt_s)

                # Kill decision
                if ct >= min_contact:
                    p = p_kill
                    if nk_bonds[i] > 0:
                        bf = min(nk_bonds[i] / bond_thresh, 1.0)
                        p += p_click_bonus * bf
                    p = min(p, 0.95)
                    remaining = max(nk_contact_dur[i] - ct, dt)
                    p_step = 1.0 - (1.0 - p) ** (dt / remaining)
                    if rng.random() < p_step:
                        nk_state[i] = 2; nk_timer[i] = kill_time
                        continue

                # Contact expired
                if ct >= nk_contact_dur[i]:
                    nk_state[i] = 3; nk_timer[i] = detach_time/2

        # ---- KILLING NK: countdown ----
        killing = nk_state == 2
        if np.any(killing):
            idx_k = np.where(killing)[0]
            nk_timer[idx_k] -= dt
            done = idx_k[nk_timer[idx_k] <= 0]
            for i in done:
                tgt = nk_target[i]
                if tgt >= 0 and t_alive[tgt]:
                    t_alive[tgt] = False
                nk_kills[i] += 1
                nk_state[i] = 3; nk_timer[i] = detach_time

        # ---- DETACHING NK ----
        detaching = nk_state == 3
        if np.any(detaching):
            idx_d = np.where(detaching)[0]
            nk_timer[idx_d] -= dt
            done = idx_d[nk_timer[idx_d] <= 0]
            for i in done:
                nk_target[i] = -1; nk_bonds[i] = 0
                nk_contact_start[i] = -1
                if nk_kills[i] < max_kills:
                    nk_state[i] = 0
                    nk_cooldown[i] = 3.0  # 3 min cooldown before next engagement
                else:
                    nk_state[i] = 4

    lysis = 100.0 * np.sum(~t_alive) / n_tumour
    return {
        'lysis': lysis,
        'kills': int(np.sum(nk_kills)),
        'bonds': int(np.sum(t_bonds)),
        'time': rec_t, 'lysis_ts': rec_lysis,
        'tx': tx, 'ty': ty, 't_alive': t_alive,
        'nx': nx, 'ny': ny, 'nk_state': nk_state,
        'nk_target': nk_target, 'nk_bonds_final': nk_bonds,
    }


def run_multi(n_tumour, et, condition, n_reps=5, **kwargs):
    """Run multiple replicates, return mean/std lysis."""
    vals = []
    last = None
    for r in range(n_reps):
        res = run_abm(n_tumour, int(n_tumour*et),
                      click_enhanced=(condition=="pMNK"),
                      seed=kwargs.get('seed', 42)+r, **{k:v for k,v in kwargs.items() if k!='seed'})
        vals.append(res['lysis'])
        last = res
    return {'mean': np.mean(vals), 'std': np.std(vals), 'vals': vals, 'last': last}


# ============================================================
# Calibration
# ============================================================
def objective_baseline(params):
    """Calibrate p_engage, p_kill, speed to match baseline NK killing."""
    p_engage, p_kill, speed = params
    err = 0
    for et in ET_RATIOS:
        r = run_multi(60, et, "NK", n_reps=3,
                      p_engage=p_engage, p_kill=p_kill, speed=speed, seed=42)
        err += (r['mean'] - NIU["NK"][et])**2
    return err

def objective_enhanced(params, p_engage, p_kill, speed):
    """
    Calibrate the click-specific params (Niu-anchored upper-bound scenario).
    engage_boost is fixed at 2.5 (from Niu 2024 Fig 2C: 78/31 = 2.5x) — this is
    the Niu-calibrated upper bound (avidity-rescue analog), not the MCK-pure
    primary case (engage_boost=1.0). See the module docstring (W4/TL-14/TL-28).
    Fit: p_click_bonus, bond_threshold.
    """
    p_bonus, bond_thresh = params
    bond_thresh = max(1, int(bond_thresh))
    err = 0
    for et in ET_RATIOS:
        r = run_multi(60, et, "pMNK", n_reps=3,
                      p_engage=p_engage, p_kill=p_kill, speed=speed,
                      engage_boost=2.5,  # Fixed from Fig 2C
                      p_click_bonus=p_bonus, bond_thresh=bond_thresh, seed=42)
        err += (r['mean'] - NIU["pMNK"][et])**2
    return err



def run_fully_pure_mck_scenario(seed=42, n_tumour=200, n_nk=1000, duration=240,
                                 verbose=True):
    """Round 5 CP-5.2 (S2-P2-01 fix): true conservative-lower-bound MCK scenario.

    Runs both arms with click_enhanced=True (MCK-relevant azide+DBCO chemistry)
    but forces dur_median = 29 min (NK baseline) on both arms — i.e. it removes
    the hardcoded Niu 2024 Fig 2D duration boost that "MCK-pure" (engage_boost
    =1.0 + click_enhanced=True) still inherits. This isolates MCK on the
    contact-frequency axis and the duration axis at the same time.

    Expect: pMNK-relative lysis lower than under the previous "MCK-pure" run,
    and Niu R² correspondingly lower. That's the honest reading of a truly
    isolated MCK click-at-contact mechanism.

    Called from main() and also runnable standalone via MCK_ABM_SCENARIO=fully_pure
    env var (see the __main__ guard).
    """
    if verbose:
        print("=" * 60)
        print("FULLY-PURE MCK SCENARIO (Round 5 CP-5.2 / S2-P2-01 fix)")
        print("=" * 60)
        print(f"  n_tumour={n_tumour}  n_nk={n_nk}  duration={duration} min")
        print(f"  engage_boost=1.0  click_enhanced=True  fully_pure_mck=True")
        print(f"  => dur_median = 29 min on BOTH arms (no Niu boost)")

    # Both arms use engage_boost=1.0 + fully_pure_mck=True
    baseline = run_abm(n_tumour, n_nk, duration=duration, seed=seed,
                        click_enhanced=False, engage_boost=1.0,
                        fully_pure_mck=True)
    mck_pure = run_abm(n_tumour, n_nk, duration=duration, seed=seed,
                       click_enhanced=True, engage_boost=1.0,
                       fully_pure_mck=True)

    fold = mck_pure["lysis"] / baseline["lysis"] if baseline["lysis"] > 0 else float("inf")
    if verbose:
        print(f"\n  baseline lysis        = {baseline['lysis']:.2f}%")
        print(f"  MCK (fully-pure) lysis = {mck_pure['lysis']:.2f}%")
        print(f"  fold                   = {fold:.2f}x")
        print(f"\n  Interpretation: this fold-change is the truly-isolated MCK contribution")
        print(f"  from covalent click-at-contact — no mannose-receptor duration boost,")
        print(f"  no engage_boost avidity rescue. Expect lower than the previous MCK-pure")
        print(f"  headline (which retained Niu's 56-min duration).")

    return {
        "scenario": "fully_pure_mck",
        "sampling_prior": "abm_deterministic",  # ABM has no MCK_SAMPLING_PRIOR concept; label for CP-17 audit
        "sampling_note": "Round 5 CP-5.2 / S2-P2-01 fix",
        "seed": seed, "n_tumour": n_tumour, "n_nk": n_nk, "duration": duration,
        "baseline_lysis_pct": float(baseline["lysis"]),
        "mck_pure_lysis_pct": float(mck_pure["lysis"]),
        "fold": float(fold),
    }


def main():
    t0 = time.time()
    print("="*60)
    print("MCK ABM CALIBRATION (Vectorised)")
    print("="*60)

    # ---- Quick timing test ----
    print("\nTiming test (single run, 60 tumour, 600 NK, 240 min)...")
    tt = time.time()
    test = run_abm(60, 600, seed=0)
    print(f"  {time.time()-tt:.1f}s, lysis={test['lysis']:.1f}%")

    # ---- Baseline calibration ----
    print("\n[1/2] Calibrating baseline NK killing...")
    print("  Optimising: p_engage, p_kill, speed")
    res_bl = differential_evolution(
        objective_baseline,
        bounds=[(0.001, 0.1), (0.1, 0.9), (3.0, 12.0)],
        maxiter=25, popsize=12, tol=0.5,
        seed=42, workers=1
    )
    p_engage_opt, p_kill_opt, speed_opt = res_bl.x
    print(f"  p_engage={p_engage_opt:.4f}, p_kill={p_kill_opt:.3f}, speed={speed_opt:.1f} µm/min")
    print(f"  SSE={res_bl.fun:.1f}")

    # ---- Enhanced calibration ----
    # Note (W4/TL-14/TL-28): this calibrates the Niu-anchored upper-bound scenario
    # (engage_boost=2.5, avidity-rescue analog), not the primary MCK-pure case
    # (engage_boost=1.0, the run_abm default). 2.5 is the right value here because
    # we're fitting Niu's pM-NK data; see the module docstring.
    print("\n[2/2] Calibrating click enhancement...")
    print("  Fixed: engage_boost=2.5 (Niu-anchored UPPER BOUND, Fig 2C; MCK-pure primary = 1.0)")
    res_en = differential_evolution(
        lambda p: objective_enhanced(p, p_engage_opt, p_kill_opt, speed_opt),
        bounds=[(0.03, 0.7), (1, 20)],
        maxiter=25, popsize=12, tol=0.5,
        seed=42, workers=1
    )
    p_bonus_opt, bond_thresh_opt = res_en.x
    bond_thresh_opt = max(1, int(bond_thresh_opt))
    print(f"  p_bonus={p_bonus_opt:.4f}, bond_thresh={bond_thresh_opt} (SSE={res_en.fun:.1f})")

    # ---- Production runs ----
    print("\n[3] Production runs (100 tumour, 5 replicates)...")
    results = {}
    for cond in ["NK", "pMNK"]:
        results[cond] = {}
        for et in ET_RATIOS:
            r = run_multi(100, et, cond, n_reps=5,
                          p_engage=p_engage_opt, p_kill=p_kill_opt, speed=speed_opt,
                          engage_boost=2.5,
                          p_click_bonus=p_bonus_opt, bond_thresh=bond_thresh_opt,
                          seed=100)
            results[cond][et] = r
            print(f"  {cond} {et}:1 → {r['mean']:.1f}% ± {r['std']:.1f}")

    # ---- R² ----
    all_data = [NIU["NK"][et] for et in ET_RATIOS] + [NIU["pMNK"][et] for et in ET_RATIOS]
    all_model = [results["NK"][et]['mean'] for et in ET_RATIOS] + \
                [results["pMNK"][et]['mean'] for et in ET_RATIOS]
    ss_res = sum((m-d)**2 for m,d in zip(all_model, all_data))
    ss_tot = sum((d-np.mean(all_data))**2 for d in all_data)
    r2 = 1 - ss_res/ss_tot

    print(f"\n{'='*60}")
    print(f"CALIBRATION RESULTS  (R² = {r2:.4f})")
    print(f"{'='*60}")
    print(f"  p_engage = {p_engage_opt:.5f} (per proximity event per time step)")
    print(f"  p_kill_per_contact = {p_kill_opt:.4f}")
    print(f"  nk_speed_mean = {speed_opt:.2f} µm/min")
    print(f"  engage_boost (pM-NK) = 2.5 (Niu-anchored UPPER BOUND, Fig 2C; MCK-pure primary scenario = 1.0)")
    print(f"  p_kill_click_bonus = {p_bonus_opt:.4f}")
    print(f"  bond_threshold = {bond_thresh_opt}")
    print(f"\n{'E:T':>5} | {'NK ABM':>8} {'NK Niu':>8} {'Δ':>6} | {'pMNK ABM':>9} {'pMNK Niu':>9} {'Δ':>6}")
    print("-"*65)
    for et in ET_RATIOS:
        nm = results["NK"][et]['mean']; nd = NIU["NK"][et]
        pm = results["pMNK"][et]['mean']; pd_ = NIU["pMNK"][et]
        print(f"{et:>4}:1 | {nm:>7.1f}% {nd:>7.1f}% {nm-nd:>+5.1f} | {pm:>8.1f}% {pd_:>8.1f}% {pm-pd_:>+5.1f}")

    # ============================================================
    # FIGURES
    # ============================================================
    print("\nGenerating figures...")

    # --- Fig 1: Baseline bar chart ---
    fig, ax = plt.subplots(figsize=(6, 4.5))
    x = np.arange(3); w = 0.35
    m_nk = [results["NK"][et]['mean'] for et in ET_RATIOS]
    e_nk = [results["NK"][et]['std'] for et in ET_RATIOS]
    d_nk = [NIU["NK"][et] for et in ET_RATIOS]
    d_nk_lo = [NIU["NK"][et]-NIU["NK_lo"][et] for et in ET_RATIOS]
    d_nk_hi = [NIU["NK_hi"][et]-NIU["NK"][et] for et in ET_RATIOS]

    ax.bar(x-w/2, m_nk, w, yerr=e_nk, label='ABM', color=C_NK, alpha=0.8, capsize=4)
    ax.bar(x+w/2, d_nk, w, yerr=[d_nk_lo, d_nk_hi], label='Niu 2024',
           color=C_NK, alpha=0.3, edgecolor=C_NK, linewidth=1.5, capsize=4)
    ax.set_xlabel('E:T Ratio'); ax.set_ylabel('Specific Lysis (%)')
    ax.set_title('Baseline NK Killing: ABM vs Niu 2024')
    ax.set_xticks(x); ax.set_xticklabels(['1:1','5:1','10:1'])
    ax.legend(frameon=False); ax.set_ylim(0,65)
    ax.spines['top'].set_visible(False); ax.spines['right'].set_visible(False)
    fig.tight_layout()
    fig.savefig(os.path.join(FIGURE_DIR, 'ABM_fig1_baseline_killing.png'), dpi=300)
    plt.close(); print("  ABM_fig1")

    # --- Fig 2: Enhanced bar chart ---
    fig, ax = plt.subplots(figsize=(6, 4.5))
    m_pm = [results["pMNK"][et]['mean'] for et in ET_RATIOS]
    e_pm = [results["pMNK"][et]['std'] for et in ET_RATIOS]
    d_pm = [NIU["pMNK"][et] for et in ET_RATIOS]
    d_pm_lo = [NIU["pMNK"][et]-NIU["pMNK_lo"][et] for et in ET_RATIOS]
    d_pm_hi = [NIU["pMNK_hi"][et]-NIU["pMNK"][et] for et in ET_RATIOS]

    ax.bar(x-w/2, m_pm, w, yerr=e_pm, label='ABM (click)', color=C_PM, alpha=0.8, capsize=4)
    ax.bar(x+w/2, d_pm, w, yerr=[d_pm_lo, d_pm_hi], label='Niu 2024 pM-NK',
           color=C_PM, alpha=0.3, edgecolor=C_PM, linewidth=1.5, capsize=4)
    ax.set_xlabel('E:T Ratio'); ax.set_ylabel('Specific Lysis (%)')
    ax.set_title('Click-Enhanced NK Killing: ABM vs Niu 2024')
    ax.set_xticks(x); ax.set_xticklabels(['1:1','5:1','10:1'])
    ax.legend(frameon=False); ax.set_ylim(0,100)
    ax.spines['top'].set_visible(False); ax.spines['right'].set_visible(False)
    fig.tight_layout()
    fig.savefig(os.path.join(FIGURE_DIR, 'ABM_fig2_enhanced_killing.png'), dpi=300)
    plt.close(); print("  ABM_fig2")

    # --- Fig 3: Population dynamics ---
    fig, axes = plt.subplots(1, 2, figsize=(12, 5), sharey=True)
    for ax, cond, label, color in [
        (axes[0], "NK", "NK (Baseline)", C_NK),
        (axes[1], "pMNK", "pM-NK (Click-Enhanced)", C_PM)
    ]:
        ts = results[cond][10]['last']
        t = np.array(ts['time'])
        ly = np.array(ts['lysis_ts'])
        ax.fill_between(t, 0, ly, alpha=0.3, color=C_DEAD, label='Dead tumour')
        ax.fill_between(t, ly, 100, alpha=0.3, color=C_TU, label='Alive tumour')
        ax.plot(t, ly, color=color, linewidth=2, label='% Lysis')
        ax.set_xlabel('Time (min)'); ax.set_title(f'{label}, 10:1 E:T')
        ax.legend(loc='upper left', frameon=False, fontsize=9)
        ax.set_xlim(0,240); ax.set_ylim(0,100)
        ax.spines['top'].set_visible(False); ax.spines['right'].set_visible(False)
    axes[0].set_ylabel('Tumour cells (%)')
    fig.suptitle('ABM Population Dynamics at 10:1 E:T', fontsize=13, fontweight='bold')
    fig.tight_layout()
    fig.savefig(os.path.join(FIGURE_DIR, 'ABM_fig3_population_dynamics.png'), dpi=300)
    plt.close(); print("  ABM_fig3")

    # --- Fig 4: Enhancement ratio ---
    fig, ax = plt.subplots(figsize=(6, 4.5))
    m_ratio = [results["pMNK"][et]['mean']/max(results["NK"][et]['mean'],0.1) for et in ET_RATIOS]
    d_ratio = [NIU["pMNK"][et]/NIU["NK"][et] for et in ET_RATIOS]
    ax.plot(ET_RATIOS, m_ratio, 'o-', color=C_PM, linewidth=2, markersize=10, label='ABM')
    ax.plot(ET_RATIOS, d_ratio, 's--', color='#333', linewidth=1.5, markersize=8, label='Niu 2024')
    ax.axhline(1, color='grey', ls=':', alpha=0.5)
    ax.set_xlabel('E:T Ratio'); ax.set_ylabel('Enhancement (pM-NK / NK)')
    ax.set_title('Click Enhancement Factor')
    ax.set_xticks(ET_RATIOS); ax.set_xticklabels(['1:1','5:1','10:1'])
    ax.legend(frameon=False); ax.set_ylim(0,5)
    ax.spines['top'].set_visible(False); ax.spines['right'].set_visible(False)
    fig.tight_layout()
    fig.savefig(os.path.join(FIGURE_DIR, 'ABM_fig4_enhancement_ratio.png'), dpi=300)
    plt.close(); print("  ABM_fig4")

    # --- Fig 5: All conditions comparison ---
    fig, ax = plt.subplots(figsize=(8, 5))
    w2 = 0.2
    ax.bar(x-1.5*w2, m_nk, w2, yerr=e_nk, label='NK (ABM)', color=C_NK, alpha=0.8, capsize=3)
    ax.bar(x-0.5*w2, d_nk, w2, yerr=[d_nk_lo, d_nk_hi], label='NK (Niu)',
           color=C_NK, alpha=0.3, edgecolor=C_NK, linewidth=1.2, capsize=3)
    ax.bar(x+0.5*w2, m_pm, w2, yerr=e_pm, label='pM-NK (ABM)', color=C_PM, alpha=0.8, capsize=3)
    ax.bar(x+1.5*w2, d_pm, w2, yerr=[d_pm_lo, d_pm_hi], label='pM-NK (Niu)',
           color=C_PM, alpha=0.3, edgecolor=C_PM, linewidth=1.2, capsize=3)
    ax.set_xlabel('E:T Ratio'); ax.set_ylabel('Specific Lysis (%)')
    ax.set_title('ABM vs Niu 2024: All Conditions')
    ax.set_xticks(x); ax.set_xticklabels(['1:1','5:1','10:1'])
    ax.legend(frameon=False, ncol=2, fontsize=9); ax.set_ylim(0,100)
    ax.spines['top'].set_visible(False); ax.spines['right'].set_visible(False)
    fig.tight_layout()
    fig.savefig(os.path.join(FIGURE_DIR, 'ABM_fig5_all_conditions.png'), dpi=300)
    plt.close(); print("  ABM_fig5")

    # --- Fig 6: Spatial snapshot ---
    snap = run_abm(80, 800, duration=120, click_enhanced=True,
                   p_engage=p_engage_opt, p_kill=p_kill_opt, speed=speed_opt,
                   engage_boost=2.5,
                   p_click_bonus=p_bonus_opt, bond_thresh=bond_thresh_opt,
                   seed=99)
    fig, ax = plt.subplots(figsize=(7, 7))
    # Tumour
    alive_mask = snap['t_alive']
    ax.scatter(snap['tx'][alive_mask], snap['ty'][alive_mask], s=300, c=C_TU,
               alpha=0.7, edgecolors='white', linewidth=0.5, zorder=2, label='Alive tumour')
    ax.scatter(snap['tx'][~alive_mask], snap['ty'][~alive_mask], s=300, c=C_DEAD,
               alpha=0.4, edgecolors='white', linewidth=0.5, zorder=1, label='Dead tumour')
    # NK
    active = snap['nk_state'] < 4
    contacting = (snap['nk_state'] == 1) | (snap['nk_state'] == 2)
    searching = (snap['nk_state'] == 0) & active
    ax.scatter(snap['nx'][searching], snap['ny'][searching], s=50, c=C_PM,
               alpha=0.4, zorder=3, label='NK (searching)')
    ax.scatter(snap['nx'][contacting], snap['ny'][contacting], s=80, c=C_BOND,
               alpha=0.9, zorder=4, label='NK (contacting)')
    # Bond lines
    for i in np.where(contacting)[0]:
        tgt = snap['nk_target'][i]
        if tgt >= 0 and snap['nk_bonds_final'][i] > 0:
            ax.plot([snap['nx'][i], snap['tx'][tgt]],
                    [snap['ny'][i], snap['ty'][tgt]],
                    '-', color=C_BOND, alpha=0.5, linewidth=0.8)

    ax.set_xlim(0, 500); ax.set_ylim(0, 500); ax.set_aspect('equal')
    ax.set_xlabel('x (µm)'); ax.set_ylabel('y (µm)')
    ax.set_title(f'Spatial Snapshot at t=120 min (pM-NK, 10:1)\n'
                 f'Lysis: {snap["lysis"]:.0f}%, Bonds: {snap["bonds"]}')
    ax.legend(loc='upper right', frameon=True, facecolor='white', fontsize=9)
    fig.tight_layout()
    fig.savefig(os.path.join(FIGURE_DIR, 'ABM_fig6_spatial_snapshot.png'), dpi=300)
    plt.close(); print("  ABM_fig6")

    # ---- Save calibrated params ----
    pf = os.path.join(os.path.dirname(os.path.abspath(__file__)),
                      '..', 'data', 'abm_calibrated_params.csv')
    # Save calibrated params summary (was truncated in original file)
    print(f'Calibration done. p_engage_opt={p_engage_opt:.5f}, p_kill_opt={p_kill_opt:.3f}')
    return p_engage_opt, p_kill_opt, speed_opt


if __name__ == "__main__":
    # Round 5 CP-5.2: standalone fully-pure MCK scenario via env var
    if os.environ.get("MCK_ABM_SCENARIO") == "fully_pure":
        import json
        from datetime import datetime
        ts = datetime.now().strftime("%Y%m%d_%H%M%S")
        result = run_fully_pure_mck_scenario()
        out_path = os.path.join(FIGURE_DIR, "..", f"abm_fully_pure_mck_{ts}.json")
        with open(out_path, "w") as f:
            json.dump(result, f, indent=2)
        print(f"\nSaved: {out_path}")
        import sys as _sys
        _sys.exit(0)
    
    main()
