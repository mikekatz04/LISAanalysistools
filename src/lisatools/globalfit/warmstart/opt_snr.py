"""Optimal-SNR gate for warm-start clusters (ruling 2026-09-10, limit 8).

Annotates every fitted component with its OPTIMAL SNR under the run's own
fitted sensitivity — psd + galfor of the BEST-logL cold walker at the LAST
valid stored sample — and drops components below ``--limit`` (default 8,
matching GB_OPT_SNR_LIMIT_SEARCH): sub-threshold clusters are
noise-fitting fodder that should not seed warm births.

Nothing waveform- or noise-side is re-derived: the sensitivity grids, the
FD waveform sizing and the SNR sum are IMPORTED from
``scripts/diagnostics/build_truth.py`` (the frozen truth-set route), and
the sampled-basis -> physical transform is the run's own stock container
(the referee's pattern). The only new choices here are the walker
selector (best-logL, per the ruling, vs build_truth's cold median) and the
gate itself.

Pipeline slot: fit -> **snr gate** -> referee -> apply. The output keeps
the exact fitter writer schema (plus an ``opt_snr`` column and gate
provenance in meta), so the referee, the apply step and
``WarmStartComponents.from_npz`` consume it unchanged. That slot is
OPTIONAL and by hand only: neither the automatic build
(:func:`.build.ensure_warm_start_components`), the self refit nor the
9mo / 12mo launchers run the gate. The one function on the launchers' path
is :func:`best_logl_noise`, which :mod:`.noise_pin` uses for the noise pin.

Laptop rules: OMP_NUM_THREADS=1 VECLIB_MAXIMUM_THREADS=1
OPENBLAS_NUM_THREADS=1 (set as defaults at import); peak RSS ~1 GB (orbit +
waveform batches).

Warning:
    The gate's SNRs are not computed on the production sky. Its templates
    use :class:`~lisatools.detector.DefaultOrbits` on the CPU with
    ``frame="icrs"`` -- the packaged EQUAL-ARM orbit file, whose positions
    are SSB ECLIPTIC although the object is labelled ICRS -- and are fed the
    run's ICRS ``alpha`` / ``sin_delta``; GBGPU reads the sky angles in the
    frame of the orbit positions, so every component is evaluated at a sky
    rotated by the obliquity (and with equal-arm orbits, not the run's
    unequal-arm L1 orbits). See :func:`component_opt_snr`.

Example:
    By hand (needs the LAT source checkout for
    ``scripts/diagnostics/build_truth.py``):

    .. code-block:: bash

        python -m lisatools.globalfit.warmstart.opt_snr --fit fit.npz \\
            --store prev_run.h5 --out gated.npz --limit 8 --batch 20000
"""
from __future__ import annotations

import argparse
import importlib.util
import json
import os
import sys
from pathlib import Path

os.environ.setdefault("OMP_NUM_THREADS", "1")
os.environ.setdefault("VECLIB_MAXIMUM_THREADS", "1")
os.environ.setdefault("OPENBLAS_NUM_THREADS", "1")

import h5py
import numpy as np

# TODO check: keep/remove/document -- opt-SNR gate helper; no build step or launcher runs the gate
def _build_truth_path() -> Path:
    """scripts/diagnostics/build_truth.py, resolved from the repo checkout
    (this module moved into the installed package 2026-09-14, so
    ``__file__``-relative no longer reaches scripts/)."""
    import lisatools

    repo = Path(lisatools.__file__).resolve().parents[2]
    for base in (repo, Path.cwd()):
        cand = base / "scripts" / "diagnostics" / "build_truth.py"
        if cand.is_file():
            return cand
    raise FileNotFoundError(
        "scripts/diagnostics/build_truth.py not found -- the opt-snr gate "
        "needs the LAT repo checkout, not just the installed package."
    )


_BT = None  # resolved lazily by _build_truth()


# TODO check: keep/remove/document -- opt-SNR gate helper; no build step or launcher runs the gate
def _build_truth():
    """Load ``scripts/diagnostics/build_truth.py`` (a shim of the monitor one) as a module."""
    spec = importlib.util.spec_from_file_location(
        "build_truth", _build_truth_path())
    mod = importlib.util.module_from_spec(spec)
    spec.loader.exec_module(mod)
    return mod


def best_logl_noise(store: str) -> dict:
    """psd + galfor of the best-logL cold walker at the last VALID row.

    Valid = log_like written AND the gb coord slab nonzero (the
    keep-window/torn-tail guards of the fitter, applied to the row this
    gate reads its noise from); only the newest 8 written rows are tried.
    Raises on all-zero params — the build_truth unwritten-row trap surfaces
    loudly here. The noise pin of the 9mo / 12mo launchers
    (:func:`.noise_pin.noise_pin_from_store`) is this point.

    Args:
        store: The store h5. Needs the gf_format_version-2 layout (a
            ``global_fit`` group with an nsamplers axis; sampler 0, rung 0,
            leaf 0 of the ``psd`` / ``galfor`` chains are read).

    Returns:
        dict: ``psd_params`` / ``galfor_params`` (lists, in the SOURCE run's
        sampling basis, possibly log), ``iteration`` (the stored row),
        ``walker`` and its cold ``log_like``.

    Raises:
        RuntimeError: When none of the newest 8 written rows has a nonzero
            gb slab, or the psd parameters are zero / non-finite.
    """
    with h5py.File(store, "r") as f:
        g = f["global_fit"]
        llv = g["log_like"][:, 0, 0, :]
        written = np.flatnonzero(np.any(llv != 0.0, axis=-1))
        chain_gb = g["chain"]["gb"]
        row = None
        for it in written[::-1][:8]:
            if chain_gb[it, 0, 0][:8].any():
                row = int(it)
                break
        if row is None:
            raise RuntimeError(f"{store}: no valid coord row found")
        w = int(np.argmax(llv[row]))
        psd = np.asarray(g["chain"]["psd"][row, 0, 0, w, 0, :], float)
        gal = np.asarray(g["chain"]["galfor"][row, 0, 0, w, 0, :], float)
    if not (np.all(np.isfinite(psd)) and np.any(psd != 0.0)):
        raise RuntimeError(
            f"{store}: psd params at row {row} walker {w} are zero/NaN "
            f"({psd}) — unwritten row?")
    return dict(psd_params=psd.tolist(), galfor_params=gal.tolist(),
                iteration=row, walker=w, log_like=float(llv[row, w]))


# TODO check: keep/remove/document -- opt-SNR gate only (CLI, tests); no build/launcher runs it
def boxed_means(means: np.ndarray, meta: dict) -> np.ndarray:
    """Clip bounded columns to their meta box before waveform transforms.

    Truncated-MLE fits (2026-09-11) legitimately place a rail-piled
    component's mean at/beyond the box edge; the physical transform
    (arccos etc.) needs in-box values. No bounded_cols meta = identity."""
    bc = dict(meta.get("bounded_cols", {}) or {})
    if not bc:
        return means
    out = np.array(means, copy=True)
    for c, (blo, bhi) in bc.items():
        c = int(c)
        out[:, c] = np.clip(out[:, c], float(blo), float(bhi))
    return out


# TODO check: keep/remove/document -- opt-SNR gate only; no build step or launcher runs it
def component_opt_snr(means: np.ndarray, store: str, noise: dict,
                      batch: int = 20000) -> np.ndarray:
    """Optimal SNR of each component mean under the fitted sensitivity.

    Warning:
        The orbits are :class:`~lisatools.detector.DefaultOrbits` (the
        packaged equal-arm file) built with ``frame="icrs"``, but that
        file's positions are SSB ecliptic; GBGPU reads the sky in the frame
        of the positions, so the ICRS ``alpha`` / ``sin_delta`` of the means
        are evaluated as ecliptic angles. The returned SNRs are therefore
        those of a sky rotated by the obliquity (with equal-arm, not the
        run's unequal-arm L1, orbits), not of the components' true sky.
    """
    bt = _build_truth()
    from gbgpu.gbgpu import GBGPU
    from lisatools import detector as lisa_models
    from lisatools.globalfit.stock.erebor.transforms import (
        make_gb_transform_container,
    )
    from lisatools.globalfit.stock.erebor.variants.gb_no_fg import (
        GB_MOJITO_T_REF,
    )

    tobs = bt.store_tobs(store)
    nw = bt.nw_for(tobs)
    df = 1.0 / tobs
    sa, se = bt.sens_grids(np.asarray(noise["psd_params"]),
                           np.asarray(noise["galfor_params"]), df)
    orb = lisa_models.DefaultOrbits(force_backend="cpu", frame="icrs")
    gbw = GBGPU(force_backend="cpu", orbits=orb, t0=float(GB_MOJITO_T_REF))
    tc = make_gb_transform_container(
        use_chirp_mass=True, use_fdot_astro=True, use_distance=True)
    phys = np.asarray(tc.both_transforms(np.atleast_2d(means)), float)
    return bt.opt_snr(phys, sa, se, gbw, df, tobs, nw, batch=batch)


# TODO check: keep/remove/document -- opt-SNR gate helper; no build step or launcher runs the gate
def _gate_observable(arrs, snr, keep):
    """Gate a MIXTURE set: drop components, renormalise, drop empty clusters.

    The SNR is per COMPONENT (each mixture component's mean is a template),
    but ``p``/``mult``/``n_members``/``island_id`` are per CLUSTER, so one
    mask cannot serve both. A cluster survives if any of its components
    does; the survivors' weights are renormalised so the cluster still sums
    to one.
    """
    ncomp = np.asarray(arrs["gmm_ncomp"], dtype=int)
    splits = np.cumsum(ncomp)[:-1]
    keep_by = np.split(keep, splits)
    w_by = np.split(np.asarray(arrs["gmm_weights"], dtype=float), splits)
    cl_keep = np.array([bool(k.any()) for k in keep_by])
    out = {
        "gmm_ncomp": np.array([int(k.sum()) for k, c in zip(keep_by, cl_keep)
                               if c], dtype=int),
        "gmm_weights": np.concatenate(
            [w[k] / w[k].sum() for w, k, c in zip(w_by, keep_by, cl_keep)
             if c]) if cl_keep.any() else np.zeros(0),
    }
    for k in ("gmm_means", "gmm_covs", "gmm_invcovs", "gmm_dets"):
        out[k] = np.asarray(arrs[k])[keep]
    for k in ("gmm_mins", "gmm_maxs"):
        out[k] = np.asarray(arrs[k])[cl_keep]
    for k, v in arrs.items():
        if k.startswith("gmm_") or k == "f0_window_edges":
            continue
        out[k] = (v[cl_keep] if getattr(v, "ndim", 0) >= 1
                  and len(v) == len(cl_keep) else v)
    out["f0_window_edges"] = arrs["f0_window_edges"]
    out["opt_snr"] = snr[keep]
    return out


# TODO check: keep/remove/document -- opt-SNR gate only; no build step or launcher runs it
def gate(fit_npz: str, out: str, limit: float, snr: np.ndarray,
         noise: dict) -> str:
    """Write the gated npz: opt_snr column + comps below ``limit`` dropped."""
    with np.load(fit_npz, allow_pickle=False) as d:
        arrs = {k: np.array(d[k]) for k in d.files if k != "meta"}
        meta = json.loads(str(d["meta"]))
    observable = str(meta.get("basis", "sampling")) == "observable"
    n = len(arrs["gmm_means"] if observable else arrs["means"])
    snr = np.asarray(snr, float)
    if snr.shape != (n,):
        raise ValueError(f"snr has shape {snr.shape}, fit has {n} comps")
    keep = snr >= float(limit)
    if observable:
        out_arrs = _gate_observable(arrs, snr, keep)
    else:
        per_comp = {k for k, v in arrs.items()
                    if v.ndim >= 1 and len(v) == n and k != "f0_window_edges"}
        out_arrs = {k: (v[keep] if k in per_comp else v)
                    for k, v in arrs.items()}
        out_arrs["opt_snr"] = snr[keep]
    meta.update(
        opt_snr_limit=float(limit),
        opt_snr_dropped=int((~keep).sum()),
        opt_snr_noise=noise,
    )
    np.savez_compressed(out, **out_arrs, meta=json.dumps(meta))
    dropped_p = ("" if observable else
                 f"; dropped-p sum {arrs['p'][~keep].sum():.1f} of "
                 f"{arrs['p'].sum():.1f}")
    print(f"gated: {n} -> {int(keep.sum())} comps at opt SNR >= {limit:g} "
          f"(dropped {int((~keep).sum())}{dropped_p}) | "
          f"noise: it {noise['iteration']} walker {noise['walker']} "
          f"logL {noise['log_like']:.1f} | wrote {out}")
    return out


# TODO check: keep/remove/document -- by-hand gate CLI (+ scripts/gb shim); no launcher runs it
def main(argv=None):
    """CLI of the optional opt-SNR gate (see the module docstring and its Warning)."""
    ap = argparse.ArgumentParser(description=__doc__.split("\n")[0])
    ap.add_argument("--fit", required=True, help="fitter components npz")
    ap.add_argument("--store", required=True, help="run store h5 (noise "
                    "params + Tobs come from here)")
    ap.add_argument("--out", required=True, help="gated output npz")
    ap.add_argument("--limit", type=float, default=8.0,
                    help="optimal-SNR gate (default 8 = "
                         "GB_OPT_SNR_LIMIT_SEARCH); <= 0 annotates only")
    ap.add_argument("--batch", type=int, default=20000)
    args = ap.parse_args(argv)
    noise = best_logl_noise(args.store)
    print(f"noise: it {noise['iteration']} walker {noise['walker']} "
          f"logL {noise['log_like']:.1f} psd={noise['psd_params']} "
          f"galfor={np.round(noise['galfor_params'], 4)}")
    with np.load(args.fit, allow_pickle=False) as d:
        fit_meta = json.loads(str(d["meta"]))
        # WAVEFORM BOUNDARY: opt_snr builds real templates, so it needs
        # astro columns whatever basis the fit ran in.
        from . import basis as wb

        means, _obs_map = wb.component_means_sampling(d, fit_meta)
    snr = component_opt_snr(boxed_means(means, fit_meta), args.store,
                            noise, batch=args.batch)
    gate(args.fit, args.out, args.limit, snr, noise)
    return 0


if __name__ == "__main__":
    sys.exit(main())
