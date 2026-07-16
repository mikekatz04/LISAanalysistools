"""Zero-leaf GB search driven by the F-stat RJ-birth proposal.

Wires an :class:`~lisatools.sampling.fstat_proposal.FStatProposal4D` (rebuilt
instantly from a cached grid npz) into the stock ``gb_no_fg`` RJ-prior move
via ``GBSettings.rj_birth_distribution``, then runs the sampler in
``GB_MODE=search`` (zero-leaf start, progressive per-band leaf cap, analytic
phi0 maximization). The single ``rj_prior`` move carries the in-model repeat
blocks internally, so each accepted birth is refined toward its
maximum-likelihood point between RJ attempts.

Data: GB-injection-only (no noise realization); whitening: tabulated
empirical PSD from the mojito NOISE brick.

Environment knobs::

    FSTAT_TARGET       ("highest" [default] or "band75")
    FSTAT_PEAK_GRIDS   (comma-separated *_peak*.npz grid caches; REQUIRED)
    FSTAT_COMB_CACHE   (comb npz; sets the floor box f0 range [else band])
    FSTAT_FLOOR_EPS    (uniform-floor mixture weight; default 0.1)
    NITER              (engine iterations; default 20)
    NWALKERS / NTEMPS  (default 16 / 6)
    GB_N_LAYERS        (highest target only; default 3)

Outputs: run logs + diagnostics under FIT_DIR (default ./gf_runs_fstat_rj/):
leaf-count and logL traces, alive-leaf f0 traces vs the catalogue, and a
final-leaf parameter table.
"""

from __future__ import annotations

import os
import sys
import time

import numpy as np

os.environ.setdefault(
    "MOJITO_DATA_PATH",
    os.path.expanduser("~/.mojito_cache/brickmarket/mojito_light_v1_0_0/"),
)
os.environ.setdefault("DATA_PROCESSOR", "mojito")
os.environ.setdefault("MAKE_PLOTS", "0")
os.environ.setdefault("GB_MODE", "search")
os.environ.setdefault("GB_USE_CHIRP_MASS", "1")

MOJITO_HIGHEST_GB = {
    "ID": 7725228, "f0_mHz": 20.380377, "Mc_eff_Msol": 0.4658,
    "RA_rad": 4.0617, "Dec_rad": -0.9049,
}
MOJITO_BAND75_GB = {
    "ID": 1229636, "f0_mHz": 7.580260, "Mc_eff_Msol": 0.3355,
    "RA_rad": 4.9791, "Dec_rad": -0.0631,
}


def build_birth_distribution(fit, peak_grid_paths, comb_cache=None,
                             floor_eps=0.1, comb_weight=0.0):
    """Cached grids (+ optional comb component) -> floor mixture -> container.

    ``comb_weight > 0`` adds a :class:`CombIntrinsicProposal` built from the
    comb cache with linear-in-F weighting -- proportional mass on EVERY comb
    peak, which is what successive births need after the loudest source is
    born and subtracted (the peak grids alone only cover their own boxes).
    """
    from lisatools.sampling.fstat_proposal import (
        CombIntrinsicProposal,
        FStatProposal4D,
        MixtureProposal,
        UniformFloorMixture,
        make_gb_rj_birth_container,
    )

    mc_lims = list(fit.gb.m_chirp_lims) or [0.001, 1.0]

    comps, weights = [], []
    for path in peak_grid_paths:
        d = np.load(path)
        comps.append(FStatProposal4D.from_grid(
            (d["f0_ax"], d["Mc_ax"], d["alpha_ax"], d["sin_delta_ax"]),
            d["logp_grid"],
        ))
        weights.append(1.0)
        print(f"[birth] loaded grid {os.path.basename(path)}  "
              f"f0=[{d['f0_ax'][0]:.5f}, {d['f0_ax'][-1]:.5f}] mHz", flush=True)

    c = np.load(comb_cache) if comb_cache else None
    if c is not None and comb_weight > 0:
        comps.append(CombIntrinsicProposal(
            c["f0_nodes_mHz"], c["F_max"], mc_lims=mc_lims,
        ))
        peak_share = (1.0 - comb_weight) / max(1, len(weights))
        weights = [peak_share] * (len(comps) - 1) + [float(comb_weight)]
        print(f"[birth] comb component (linear-in-F) weight={comb_weight}",
              flush=True)

    base = comps[0] if len(comps) == 1 else MixtureProposal(comps, weights)

    # floor box: f0 across the whole scanned band (comb cache when given),
    # everything else the full prior ranges.
    if c is not None:
        f0_lo, f0_hi = float(c["f0_nodes_mHz"][0]), float(c["f0_nodes_mHz"][-1])
    else:
        f0_lo = min(float(np.load(p)["f0_ax"][0]) for p in peak_grid_paths)
        f0_hi = max(float(np.load(p)["f0_ax"][-1]) for p in peak_grid_paths)
    lo = [f0_lo, mc_lims[0], 0.0, -1.0]
    hi = [f0_hi, mc_lims[-1], 2.0 * np.pi, 1.0]
    mix = UniformFloorMixture(base, lo, hi, eps=float(floor_eps))
    print(f"[birth] floor box f0=[{f0_lo:.5f}, {f0_hi:.5f}] mHz  "
          f"eps={floor_eps}", flush=True)
    return make_gb_rj_birth_container(mix, fit.gb.A_lims, use_cupy=False)


def main():
    import matplotlib

    matplotlib.use("Agg")
    matplotlib.rcParams["text.usetex"] = False

    from lisatools.globalfit.stock import erebor
    from lisatools.sensitivity import MojitoNoiseEstimates
    from lisatools.globalfit.stock.erebor.noise import resolve_noise_file

    target = os.environ.get("FSTAT_TARGET", "highest").strip().lower()
    src = {"highest": MOJITO_HIGHEST_GB, "band75": MOJITO_BAND75_GB}[target]
    niter = int(os.environ.get("NITER", 20))
    nwalkers = int(os.environ.get("NWALKERS", 16))
    ntemps = int(os.environ.get("NTEMPS", 6))
    peak_grids = [p for p in os.environ.get("FSTAT_PEAK_GRIDS", "").split(",")
                  if p.strip()]
    if not peak_grids:
        sys.exit("FSTAT_PEAK_GRIDS is required (comma-separated npz paths)")
    comb_cache = os.environ.get("FSTAT_COMB_CACHE", "").strip() or None
    fit_dir = os.environ.get("FIT_DIR", f"./gf_runs_fstat_rj/{target}/")

    print(f"[main] target={target} (GB {src['ID']} at {src['f0_mHz']:.4f} mHz)"
          f"  niter={niter}  nwalkers={nwalkers} ntemps={ntemps}", flush=True)

    fit = erebor.gb_no_fg(nwalkers=nwalkers, ntemps=ntemps)
    fit.gb.mode = "search"
    fit.general.file_store_dir = fit_dir
    if target == "highest":
        fit.gb.center_freq = src["f0_mHz"] * 1e-3
        fit.gb.n_layers = int(os.environ.get("GB_N_LAYERS", 3))
    fit.gb.use_chirp_mass = True

    # empirical tabulated PSD; injection-only data (stock gb_no_fg mojito)
    noise_file = resolve_noise_file(fit.general.mojito_data_path)
    fit.general.fixed_psd_kwargs = dict(psd_params=None, galfor_params=None)
    fit.general.sensitivity_init_kwargs = dict(
        tdi_generation=fit.general.tdi_gen,
        extra_components=[MojitoNoiseEstimates(noise_file, which="xyz")],
    )

    # the birth proposal (from cached F-stat grids; no kernel sweep needed)
    fit.gb.rj_birth_distribution = build_birth_distribution(
        fit, peak_grids, comb_cache=comb_cache,
        floor_eps=float(os.environ.get("FSTAT_FLOOR_EPS", 0.1)),
        comb_weight=float(os.environ.get("FSTAT_COMB_WEIGHT", 0.0)),
    )

    fit.general.num_iterations = niter

    print("[main] fit.build() ...", flush=True)
    t0 = time.time()
    curr = fit.build()
    print(f"[main]   built in {time.time() - t0:.1f}s", flush=True)
    bp = curr.general_info.main_file_path
    if bp and os.path.exists(bp):
        os.remove(bp)
        print(f"[main] removed stale backend {bp}", flush=True)

    t0 = time.time()
    fit.run()
    print(f"[main] run finished in {time.time() - t0:.1f}s "
          f"(backend: {bp})", flush=True)


if __name__ == "__main__":
    main()
