#!/usr/bin/env python
"""Measure the run's OWN residual power per WDM layer against its noise model.

    python scripts/noise/residual_kappa_probe.py            # env supplies the config

This is the decisive test for ``kappa``: the broadband component of the
instrument-PSD bias in the full fit.

WHAT IS ALREADY ESTABLISHED (do not re-derive):
  * noise-only and noise+galfor runs recover the injection to 0.05% on the
    same grid, band and noise model -> the noise model itself is correct.
  * the full fit sits at S_oms 1.3879x / S_tm 1.6017x = 1.93x / 2.57x in POWER.
  * that decomposes as ``resid = (1+kappa)*noise + eps*(resolvable source
    power)`` with eps=0.150 (subtraction residual, drives S_tm) and
    kappa=0.840 (broadband, drives S_oms).
  * every LOW-frequency mechanism tried pushes S_oms the WRONG WAY, so kappa
    has to be a genuine high-frequency excess. Above 22 mHz the mojito bricks
    carry EXACTLY zero signal power and only ~14 leaves/walker live above
    12 mHz -- so the residual up there should be pure instrument noise.

THE MEASUREMENT. Rebuild the residual for the stored state (the run's own last
sample, mostly-but-not-entirely subtracted) with the run's own machinery
(``GlobalFit.prepare_main`` -- see the note below on why setup_acs alone is not
enough), then per WDM layer form the exact multivariate whitened statistic over
the 3 TDI channels

    q = w^T C^-1 w ,    E[q] = 3 under a correct model

and report ``q/3`` against TWO covariances:

    q_fit / 3   C at the run's own fitted psd+galfor. ~1 by construction if the
                chain converged -- a sanity check that the probe is wired right.
    q_true / 3  C at the INJECTION [1.5e-11, 3e-15]. THIS IS KAPPA. If the
                residual above ~18 mHz comes back at ~1.84 the broadband excess
                is real and measured, and it is a bookkeeping/normalization
                problem in how the residual is assembled -- not physics. If it
                comes back ~1.0 then the run's stored state does NOT reproduce
                its own fitted noise, which points at state handling instead.

WHY prepare_main AND NOT setup_acs ALONE. ``setup_acs(rebuild_residuals=True)``
does NOT subtract gb or vgb -- neither registers a params-based ``signal_gen``,
so both are skipped with a warning (the PRODUCTION run logs the same skip).
GB templates come off in the LEGACY block inside ``build_gb_moves``
(recipe.py, guarded on ``signal_gen is None``), which runs only when the recipe
is materialized. Calling setup_acs by itself leaves the entire galaxy in the
array, so q would measure raw-data/model and look like a perfectly good number.
A subtraction gate below refuses to report if that happens.

SAFETY: point FILE_STORE_DIR at a COPY of the store. Building a
``GlobalFitSetup`` opens the HDF backend, and the production run may be live.
"""

from __future__ import annotations

import os
import sys

import numpy as np

REPO = os.path.dirname(os.path.dirname(os.path.dirname(os.path.abspath(__file__))))
sys.path.insert(0, os.path.join(REPO, "scripts", "fstat_proposal"))

INJECTION = [1.5e-11, 3e-15]


def _np(x):
    """Host numpy view of a possibly-cupy array (the run builds on GPU)."""
    if x is None:
        return None
    get = getattr(x, "get", None)
    if callable(get) and type(x).__module__.startswith("cupy"):
        return np.asarray(get())
    return np.asarray(x)


def per_layer_q(W, C):
    """``q = w^T C^-1 w`` averaged over time columns, per layer. E[q] = 3."""
    Cp = np.transpose(C, (2, 3, 0, 1))          # (layer, time, ch, ch)
    Wp = np.transpose(W, (1, 2, 0))             # (layer, time, ch)
    nl = Wp.shape[0]
    out = np.full(nl, np.nan)
    for i in range(nl):
        M, w = Cp[i], Wp[i]
        ok = np.isfinite(M).all(axis=(1, 2)) & np.isfinite(w).all(axis=1)
        ok &= np.abs(np.linalg.det(M)) > 0
        if not ok.any():
            continue
        try:
            sol = np.linalg.solve(M[ok], w[ok][..., None])[..., 0]
        except np.linalg.LinAlgError:
            continue
        out[i] = np.real(np.einsum("tc,tc->t", np.conj(w[ok]), sol)).mean()
    return out


def main() -> int:
    from mpi4py import MPI

    from lisatools.globalfit.run import GlobalFit
    from run_combined_staged import build_fit

    print("[probe] building the production fit config ...", flush=True)
    fit = build_fit()
    if os.environ.get("TOBS_TARGET", "").strip():
        fit.general.nf = None
        fit.general.nt = None
    curr = fit.build()
    print("[probe] build done", flush=True)

    gi = curr.general_info
    # The pristine input data, BEFORE any template subtraction. Kept so the
    # gate below can prove the residual is actually a residual.
    raw_dom = getattr(gi, "input_data_residual_array", None)
    W_raw = _np(getattr(raw_dom, "arr", raw_dom))

    gf = GlobalFit(curr, MPI.COMM_WORLD)

    # ⚠ prepare_main(), NOT the light priors->load_info->setup_acs path.
    # `setup_acs(rebuild_residuals=True)` does NOT subtract gb or vgb: neither
    # branch registers a params-based signal_gen, so both are skipped with a
    # warning (verified in the PRODUCTION log too, slurm_stdout_438:382-383).
    # The GB templates are subtracted by the LEGACY path inside
    # build_gb_moves (recipe.py ~2420, guarded on `signal_gen is None`), which
    # only runs when the recipe is materialized. Taking the light path leaves
    # the full galaxy in the array and would silently measure raw data --
    # a confident, wrong answer. prepare_main runs the setup_function, so the
    # subtraction happens. It also builds the engine; the F-stat epoch cache in
    # the copied store is loaded rather than refitted (GB_FSTAT_REFIT_EVERY is
    # pinned huge by the submit script).
    gf.prepare_main()
    state, acs = gf.state, gf.acs
    print("[probe] prepare_main done: state + residual ready", flush=True)

    ac = acs.acs[0]                                   # cold chain, walker 0
    data = getattr(ac, "data", None)
    dom = getattr(data, "data_res_arr", data)
    W = _np(getattr(dom, "arr", dom))
    print(f"[probe] residual array {W.shape} {W.dtype}", flush=True)

    # ---- SUBTRACTION GATE ---------------------------------------------------
    # Prove templates were removed. If the residual still equals the raw data
    # the measurement is meaningless, and the failure is SILENT otherwise.
    if W_raw is not None and W_raw.shape == W.shape:
        pr, pw = np.nansum(np.abs(W_raw) ** 2), np.nansum(np.abs(W) ** 2)
        frac = pw / pr if pr > 0 else np.nan
        print(f"[probe] total power  raw={pr:.6e}  residual={pw:.6e}  "
              f"residual/raw={frac:.6f}", flush=True)
        if not np.isfinite(frac) or frac > 0.999:
            print("[probe] REFUSING: the residual is indistinguishable from the "
                  "raw data -- no templates were subtracted, so q would just be "
                  "data/model. Check that the recipe materialized and that "
                  "build_gb_moves ran its legacy subtraction block.")
            return 2
    else:
        print("[probe] WARNING: could not compare against the raw data "
              f"(raw={None if W_raw is None else W_raw.shape}); the "
              "subtraction gate is NOT active.", flush=True)
    nlv = int(_np(state.branches["gb"].inds)[0].sum())
    print(f"[probe] gb leaves alive in the loaded state (cold): {nlv}", flush=True)
    # WHICH branches actually got subtracted. `setup_acs(rebuild_residuals=True)`
    # only subtracts branches carrying a params-based signal_gen; anything else
    # is skipped with a warning and normally subtracts inside its own move,
    # which this probe never runs. vgb is the known case. It does NOT touch the
    # verdict -- VGB power is <=8% of the noise below 8 mHz and EXACTLY zero
    # above 12 mHz -- but it does inflate the low-f rows, so say so.
    print("[probe] NOTE gb/vgb are skipped by rebuild_residuals (no signal_gen) "
          "and come off in their build_*_moves legacy blocks instead; the "
          "residual/raw ratio above is the authority on what actually got "
          "subtracted. Any branch still in the residual inflates the LOW-f "
          "rows only -- the 18-25 mHz verdict row has zero GB/VGB power.",
          flush=True)

    # --- the two covariances -------------------------------------------------
    psd_fit = gal_fit = None
    for br, tgt in (("psd", "psd"), ("galfor", "galfor")):
        try:
            coords = state.branches[br].coords
            v = _np(coords)[0, 0, 0]                  # cold, walker 0, leaf 0
            if tgt == "psd":
                psd_fit = list(v)
            else:
                gal_fit = list(v)
        except Exception as e:
            print(f"[probe] no {br} coords ({e})")
    print(f"[probe] fitted psd={psd_fit}  galfor={gal_fit}", flush=True)

    sb = gi.sensitivity_backend
    C_fit = _np(sb("probe_fit", psd_fit, gal_fit).sens_mat) if psd_fit else None
    C_true = _np(sb("probe_true", INJECTION, gal_fit).sens_mat)

    # ---- frequency axis ----------------------------------------------------
    # Job 457 silently fell back to bare layer INDEX here (printing "4000 mHz"
    # and NaN band medians) because Nf/dt were not readable off dom.settings
    # under those names. Try several sources, say which one won, and dump the
    # candidate attributes if they all fail so the next run is fixable.
    dom_s = getattr(dom, "settings", None)
    gsb = curr.general_info

    def _first(obj, names):
        for n in names:
            v = getattr(obj, n, None)
            if v:
                return float(v), n
        return None, None

    nf, nf_src = _first(dom_s, ("Nf", "nf", "N_f"))
    dt, dt_src = _first(dom_s, ("dt", "delta_t", "DT"))
    src = f"dom.settings.{nf_src}/{dt_src}"
    if nf is None or dt is None:
        nf2, n2 = _first(gsb, ("nf", "Nf"))
        dt2, d2 = _first(gsb, ("dt", "delta_t"))
        if nf2 and dt2:
            nf, dt, src = nf2, dt2, f"general_info.{n2}/{d2}"
    if nf and dt:
        df = 1.0 / (2 * nf * dt)
    else:
        df = None
        print("[probe] WARNING: could not resolve Nf/dt for the frequency axis. "
              f"dom.settings type={type(dom_s).__name__}; candidates present: "
              f"{[a for a in dir(dom_s) if not a.startswith('__')][:40]}")
    imin, _ = _first(dom_s, ("_ind_min_f", "ind_min_f"))
    imin = int(imin) if imin else 1
    if df:
        lay = (np.arange(W.shape[1]) + imin) * df
        print(f"[probe] frequency axis from {src}: Nf={nf:.0f} dt={dt} "
              f"layer_df={df:.6e} Hz, first layer index={imin} -> "
              f"{lay[0]*1e3:.4f} .. {lay[-1]*1e3:.3f} mHz", flush=True)
    else:
        lay = np.arange(W.shape[1], dtype=float)
        print("[probe] frequency axis UNRESOLVED -- 'f [mHz]' below is the bare "
              "LAYER INDEX and the band medians will be NaN. The q arrays in "
              "the npz are still correct; rebuild the axis offline as "
              "(k + ind_min_f) / (2*Nf*dt).", flush=True)

    q_true = per_layer_q(W, C_true) / 3.0
    q_fit = per_layer_q(W, C_fit) / 3.0 if C_fit is not None else None

    print("\n  f [mHz] |  q_true/3  |  q_fit/3   <- q_true/3 IS kappa+1")
    for i in range(0, len(lay), max(1, len(lay) // 40)):
        a = f"{q_fit[i]:9.4f}" if q_fit is not None else "     n/a"
        print(f" {lay[i]*1e3:8.3f} | {q_true[i]:9.4f}  | {a}")

    def band(v, lo, hi):
        m = (lay >= lo) & (lay < hi) & np.isfinite(v)
        return np.median(v[m]) if m.any() else np.nan

    print("\n BAND MEDIANS (the verdict is the 18-25 mHz row):")
    print("  band [mHz]   |  q_true/3  |  q_fit/3")
    for lo, hi in [(0.4e-3, 1.5e-3), (1.5e-3, 3e-3), (3e-3, 5e-3), (5e-3, 8e-3),
                   (8e-3, 12e-3), (12e-3, 18e-3), (18e-3, 25e-3)]:
        a = f"{band(q_fit, lo, hi):9.4f}" if q_fit is not None else "     n/a"
        print(f"  {lo*1e3:5.1f}-{hi*1e3:5.1f}  | {band(q_true, lo, hi):9.4f}  | {a}")
    print("\n  EXPECT if kappa is real : q_true/3 ~ 1.84 at 18-25 mHz")
    print("  EXPECT if kappa is not  : q_true/3 ~ 1.00 at 18-25 mHz")
    # ========================================================================
    # WHICH ANSWER DOES THE RUN'S OWN LIKELIHOOD PREFER?
    # ========================================================================
    # The residual measured above is CORRECT (q_true/3 -> 1.00 at 12-25 mHz),
    # and an honest ML on it lands near S_oms ~ 1.0 -- yet the run sits at
    # 1.39. So ask the code directly: evaluate ITS likelihood on THIS residual
    # along the line through the injection, and see where it peaks.
    #
    #   peak at alpha ~ 1.39  -> the LIKELIHOOD is mis-weighted. The sampler is
    #                            doing its job on a wrong objective; the bug is
    #                            in how chi^2, logdet and pixel counts combine.
    #   peak at alpha ~ 1.00  -> the likelihood is fine and the sampler never
    #                            reached its own optimum, OR the psd move scores
    #                            a different array than the one measured here.
    #
    # The source_only / noise_only split is the diagnostic: if the chi^2 term
    # alone prefers the injection while the TOTAL prefers the fitted params,
    # the noise-normalization term is carrying the wrong weight.
    print("\n" + "=" * 72)
    print(" lnL SCAN along psd = alpha * injection, on THIS residual")
    print(" (walker 0; galfor held at the run's fitted values)")
    print("=" * 72)
    saved_sens = ac.sens_mat
    alphas = [0.85, 0.95, 1.0, 1.05, 1.15, 1.25, 1.3879, 1.5, 1.7]
    rows = []
    try:
        for a in alphas:
            p = [INJECTION[0] * a, INJECTION[1] * a]
            ac.sens_mat = sb(f"lnl_a{a}", p, gal_fit)
            tot = float(np.real(_np(ac.likelihood(source_only=False))))
            src = float(np.real(_np(ac.likelihood(source_only=True))))
            rows.append((a, tot, src, tot - src))
        # the run's ACTUAL fitted point (not necessarily on the alpha line)
        ac.sens_mat = sb("lnl_fitted", psd_fit, gal_fit)
        tot_f = float(np.real(_np(ac.likelihood(source_only=False))))
        src_f = float(np.real(_np(ac.likelihood(source_only=True))))
    finally:
        ac.sens_mat = saved_sens

    base = max(r[1] for r in rows)
    print("  alpha | S_oms      | lnL_total - max | chi2 term (-<r|r>/2) | noise-norm term")
    for a, tot, src, nz in rows:
        mark = "  <- alpha=1 (INJECTION)" if a == 1.0 else (
               "  <- the run's S_oms ratio" if abs(a - 1.3879) < 1e-6 else "")
        print(f" {a:6.4f} | {INJECTION[0]*a:.4e} | {tot-base:15.2f} | "
              f"{src:20.6e} | {nz:15.6e}{mark}")
    print(f"\n  the run's ACTUAL fitted psd {psd_fit}:")
    print(f"     lnL_total = {tot_f:.6e}   (relative to scan max: {tot_f-base:+.2f})")
    print(f"     chi2 term = {src_f:.6e}   noise-norm = {tot_f-src_f:.6e}")
    best = max(rows, key=lambda r: r[1])
    print(f"\n  >>> the code's likelihood PEAKS at alpha = {best[0]:.4f} <<<")
    if abs(best[0] - 1.0) < 0.1:
        print("  => likelihood is FINE at alpha~1; the fitted params do NOT maximize")
        print("     it, so this is a SAMPLER or wrong-array problem, not weighting.")
    elif best[0] > 1.2:
        print("  => the likelihood ITSELF prefers the biased params: the objective")
        print("     is mis-weighted (chi^2 vs logdet vs pixel counts).")
    print("=" * 72)

    np.savez(os.environ.get("KAPPA_OUT", "kappa_probe.npz"),
             lnl_alphas=np.asarray([r[0] for r in rows]),
             lnl_total=np.asarray([r[1] for r in rows]),
             lnl_chi2=np.asarray([r[2] for r in rows]),
             lnl_noisenorm=np.asarray([r[3] for r in rows]),
             lnl_fitted_total=tot_f, lnl_fitted_chi2=src_f,
             lay=lay, q_true=q_true, q_fit=q_fit if q_fit is not None else [],
             psd_fit=np.asarray(psd_fit if psd_fit else []),
             gal_fit=np.asarray(gal_fit if gal_fit else []))
    print("\n[probe] saved", os.environ.get("KAPPA_OUT", "kappa_probe.npz"), flush=True)
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
