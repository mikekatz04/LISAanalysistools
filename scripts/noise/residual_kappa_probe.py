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
(``setup_acs(rebuild_residuals=True)``), then per WDM layer form the exact
multivariate whitened statistic over the 3 TDI channels

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


def _periodic_free_priors(gf):
    """Priors dict, mirroring the loop in ``GlobalFit.prepare_main``.

    Only needed so ``load_info`` can fall back to a prior draw; on a resume it
    returns ``backend.get_last_sample()`` and these are not consulted.
    """
    from lisatools.globalfit.engine import Setup

    priors = {}
    for name in gf.engine_info.branch_names:
        info = gf.curr.source_info.get(name)
        if info is None:
            continue
        if isinstance(info, dict):
            for k, v in info["priors"].items():
                priors[k] = v
        elif isinstance(info, Setup):
            for k, v in info.priors.items():
                priors[k] = v
    return priors


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

    gf = GlobalFit(curr, MPI.COMM_WORLD)
    priors = _periodic_free_priors(gf)
    state = gf.load_info(priors)
    print("[probe] state loaded", flush=True)

    acs = gf.setup_acs(state, rebuild_residuals=True)
    print("[probe] residual rebuilt", flush=True)

    gi = curr.general_info
    ac = acs.acs[0]                                   # cold chain, walker 0
    data = getattr(ac, "data", None)
    dom = getattr(data, "data_res_arr", data)
    W = np.asarray(getattr(dom, "arr", dom))
    print(f"[probe] residual array {W.shape} {W.dtype}", flush=True)

    # --- the two covariances -------------------------------------------------
    psd_fit = gal_fit = None
    for br, tgt in (("psd", "psd"), ("galfor", "galfor")):
        try:
            coords = state.branches[br].coords
            v = np.asarray(coords[0, 0, 0])           # cold, walker 0, leaf 0
            if tgt == "psd":
                psd_fit = list(v)
            else:
                gal_fit = list(v)
        except Exception as e:
            print(f"[probe] no {br} coords ({e})")
    print(f"[probe] fitted psd={psd_fit}  galfor={gal_fit}", flush=True)

    sb = gi.sensitivity_backend
    C_fit = np.asarray(sb("probe_fit", psd_fit, gal_fit).sens_mat) if psd_fit else None
    C_true = np.asarray(sb("probe_true", INJECTION, gal_fit).sens_mat)

    dom_s = getattr(dom, "settings", None)
    nf = getattr(dom_s, "Nf", None)
    dt = getattr(dom_s, "dt", None)
    df = 1.0 / (2 * nf * dt) if (nf and dt) else None
    imin = int(getattr(dom_s, "_ind_min_f", 1) or 1)
    lay = (np.arange(W.shape[1]) + imin) * df if df else np.arange(W.shape[1])

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
    np.savez(os.environ.get("KAPPA_OUT", "kappa_probe.npz"),
             lay=lay, q_true=q_true, q_fit=q_fit if q_fit is not None else [],
             psd_fit=np.asarray(psd_fit if psd_fit else []),
             gal_fit=np.asarray(gal_fit if gal_fit else []))
    print("\n[probe] saved", os.environ.get("KAPPA_OUT", "kappa_probe.npz"), flush=True)
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
