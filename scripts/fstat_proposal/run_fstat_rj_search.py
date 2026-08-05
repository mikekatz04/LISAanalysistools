"""Zero-leaf GB search driven by the F-stat RJ-birth proposal.

Wires the F-stat birth distribution into the stock ``gb_no_fg`` RJ-prior
move via ``GBSettings.rj_birth_distribution``, then runs the sampler in
``GB_MODE=search`` (zero-leaf start, progressive per-band leaf cap, analytic
phi0 maximization). The single ``rj_prior`` move carries the in-model repeat
blocks internally, so each accepted birth is refined toward its
maximum-likelihood point between RJ attempts.

The birth distribution (DRAW side of the proposal)::

    floor( mixture[ stacked peak grids (per-sub-band F-stat boxes; box
                    weights w ~ F or equal),
                    comb (linear-in-F f0 x uniform Mc/sky) ],
           uniform floor eps over the whole f0_lims box )
    -> 8-column GB container (uniform lnA/phi0/cos_iota/psi)

The peak component's SAMPLING layer is the stacked grid itself
(``StackedFStatProposal4D`` -- exact rvs/logpdf, on-device for cupy runs).
``FSTAT_PEAK_SAMPLING=gmm`` selects the batched-GMM alternative
(``FullGaussianMixtureModel`` fitted FROM the grids; ~MBs instead of the
full grid stack) with a GMM-vs-grid fidelity gate logged at startup.
TODO(fstat-gmm-deprecation): the gmm option may be deprecated once the
grid path proves out at full-band GPU scale.

Band selection is driven by the stock band knobs (``GB_MIN_FREQ``/
``GB_MAX_FREQ``, ``GB_CENTER_FREQ``+``GB_N_LAYERS``,
``GB_HIGHEST_FREQUENCIES=N``) -- the SAME knobs the grid-prep script
honors, so one setting sizes prep + search identically.

Data: GB-injection-only (no noise realization); whitening: tabulated
empirical PSD from the mojito NOISE brick.

Environment knobs::

    FSTAT_GRID_DIR     (directory holding the *_peaks_stacked.npz [+
                        *_comb.npz] written by plot_fstat_proposal_mojito.py;
                        the standard grid source)
    FSTAT_PEAK_GRIDS   (TEST-ONLY override: comma-separated legacy per-box
                        *_peak<i>.npz caches; stacked + GMM-fitted on load)
    FSTAT_PEAK_WEIGHTING
                       ("fstat" [default]: peak-box weights w ~ F;
                        "equal": flat per-box weights)
    FSTAT_COMB_CACHE   (comb npz; overrides the FSTAT_GRID_DIR discovery)
    FSTAT_COMB_WEIGHT  (comb component weight in the mixture; default 0)
    FSTAT_FLOOR_EPS    (uniform-floor mixture weight; default 0.1)
    FSTAT_GRID_MEM_MB  (K-axis memory budget for the stacked grids)
    FSTAT_PEAK_SAMPLING
                       ("grid" [default]: the stacked grids themselves --
                        exact, production; "gmm": the batched-GMM
                        alternative, memory-light, gate-checked)
    GB_SUBTRACT_OUT_OF_BAND=1
                       (subtract all out-of-band catalogue sources from the
                        data as known signals; default 0 = keep them)
    NITER              (engine iterations; default 20)
    NWALKERS / NTEMPS  (default 16 / 6)
    FSTAT_RESUME       (=1: keep an existing backend and continue from its
                        last sample; pin TOBS_TARGET/NWALKERS/NTEMPS/
                        CHUNKED_* to the original run's values)

In GB_MODE=pe (injection-seeded debug) no grids are needed.

Outputs: run logs + diagnostics under FIT_DIR (default ./gf_runs_fstat_rj/):
leaf-count and logL traces, alive-leaf f0 traces vs the catalogue, and a
final-leaf parameter table.
"""

from __future__ import annotations

import glob
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


def _resolve_use_cupy(fit):
    """Match the birth container's array module to the run's compute backend
    (cupy on the CUDA path) -- resolve it exactly as the engine does so a
    GPU run doesn't hand a numpy container into a cupy-resident state."""
    from lisatools.globalfit.stock.erebor.common import resolve_compute

    _gpus, _ = resolve_compute(
        fit.general.use_gpu, fit.general.gpu_backend, fit.general.gpus
    )
    return _gpus is not None, (_gpus[0] if _gpus else None)


def _load_stacked_source():
    """Resolve the peak-grid source: FSTAT_GRID_DIR (standard) or
    FSTAT_PEAK_GRIDS (legacy per-box list, test-only).

    Returns ``(cache_dict, comb_path)`` where ``cache_dict`` maps the
    stacked-npz keys (legacy grids are stacked into the same layout;
    GMM fields absent when they need fitting on load), or ``(None, comb)``
    when no grid source is configured (pe mode).
    """
    grid_dir = os.environ.get("FSTAT_GRID_DIR", "").strip()
    peak_grids = [p for p in os.environ.get("FSTAT_PEAK_GRIDS", "").split(",")
                  if p.strip()]
    comb_cache = os.environ.get("FSTAT_COMB_CACHE", "").strip() or None

    if grid_dir:
        stacked_files = sorted(glob.glob(os.path.join(grid_dir,
                                                      "*_peaks_stacked.npz")))
        if len(stacked_files) != 1:
            sys.exit(f"FSTAT_GRID_DIR={grid_dir!r}: expected exactly one "
                     f"*_peaks_stacked.npz, found {len(stacked_files)}: "
                     f"{[os.path.basename(p) for p in stacked_files]}")
        d = dict(np.load(stacked_files[0]))
        print(f"[birth] stacked cache {os.path.basename(stacked_files[0])}: "
              f"K={d['logp_grids'].shape[0]} boxes, node shape "
              f"{d['logp_grids'].shape[1:]}", flush=True)
        if comb_cache is None:
            combs = sorted(glob.glob(os.path.join(grid_dir, "*_comb.npz")))
            if len(combs) == 1:
                comb_cache = combs[0]
        return d, comb_cache

    if peak_grids:
        # Legacy per-box caches (test-only): stack them into the same layout
        # (all boxes must share the global FSTAT_N_* node shape).
        grids, f0_los, f0_dxs = [], [], []
        mc_ax = alpha_ax = sd_ax = None
        for path in peak_grids:
            g = np.load(path)
            grids.append(np.asarray(g["logp_grid"], dtype=float))
            f0_ax = np.asarray(g["f0_ax"], dtype=float)
            f0_los.append(f0_ax[0])
            f0_dxs.append(f0_ax[1] - f0_ax[0])
            mc_ax = np.asarray(g["Mc_ax"], dtype=float)
            alpha_ax = np.asarray(g["alpha_ax"], dtype=float)
            sd_ax = np.asarray(g["sin_delta_ax"], dtype=float)
            print(f"[birth] legacy grid {os.path.basename(path)}  "
                  f"f0=[{f0_ax[0]:.5f}, {f0_ax[-1]:.5f}] mHz", flush=True)
        shapes = {g.shape for g in grids}
        if len(shapes) != 1:
            sys.exit(f"FSTAT_PEAK_GRIDS: mixed node shapes {shapes}; the "
                     "stacked layout needs one global FSTAT_N_* design.")
        d = dict(
            logp_grids=np.stack(grids), f0_los=np.asarray(f0_los),
            f0_dxs=np.asarray(f0_dxs), mc_ax=mc_ax, alpha_ax=alpha_ax,
            sin_delta_ax=sd_ax,
            peak_F=np.array([float(g.max()) for g in grids]),
        )
        return d, comb_cache

    return None, comb_cache


def build_birth_distribution(fit, floor_eps=0.1, comb_weight=0.0):
    """Stacked grids (+ optional comb) -> peak GMM -> floor -> container.

    ``comb_weight > 0`` adds a :class:`CombIntrinsicProposal` built from the
    comb cache with linear-in-F weighting -- proportional mass on EVERY comb
    peak, which is what successive births need after the loudest source is
    born and subtracted (the peak grids alone only cover their own boxes).
    Returns ``None`` when no grid source is configured.
    """
    from lisatools.sampling.fstat_proposal import (
        CombIntrinsicProposal,
        MixtureProposal,
        StackedFStatProposal4D,
        UniformFloorMixture,
        build_peak_gmm,
        fit_gmm_to_stacked,
        fstat_knob,
        make_gb_rj_birth_container,
        unpack_gmm_components,
    )

    cache, comb_cache = _load_stacked_source()
    if cache is None:
        return None

    use_cupy, gpu_index = _resolve_use_cupy(fit)
    mc_lims = list(fit.gb.m_chirp_lims) or [0.001, 1.0]

    # Peak-box mixture weights: w ~ F (default) or equal.
    weighting = os.environ.get("FSTAT_PEAK_WEIGHTING", "fstat").strip().lower()
    peak_F = np.clip(np.asarray(cache["peak_F"], dtype=float), 0.0, None)
    box_weights = None if weighting == "equal" else peak_F
    print(f"[birth] peak weighting: {weighting} "
          f"({'w ~ F' if box_weights is not None else 'equal per box'})",
          flush=True)

    # Per-band peak coverage (metadata present for grid-dir caches).
    if "band_idx" in cache and "band_edges" in cache:
        band_idx = np.asarray(cache["band_idx"], dtype=int)
        num_sub_bands = len(np.asarray(cache["band_edges"])) - 1
        counts = np.bincount(band_idx, minlength=num_sub_bands)
        print(f"[birth] per-interior-band peak counts: "
              f"{dict(enumerate(counts.tolist()))}", flush=True)
        zero = [k for k in range(1, num_sub_bands - 1) if counts[k] == 0]
        if zero:
            print(f"[birth] WARNING: interior sub-band(s) {zero} have ZERO "
                  "peak grids -- births there rely on the comb/floor only "
                  "(a from-zero production run expects the full per-band "
                  "complement).", flush=True)

    # The stacked grid proposal: production sampling layer by default
    # (and the reference / GMM-fit sample source on the gmm path). Lives
    # on-device for cupy runs -- on the run's OWN device: this container is
    # built before ``fit.build()``, i.e. before the run pins its main device,
    # so an unpinned upload would put the grids on device 0 while a
    # ``GPUS=2`` run samples on device 2.
    mem_mb = os.environ.get("FSTAT_GRID_MEM_MB", "").strip()
    stacked = StackedFStatProposal4D.from_cache(
        cache, weights=box_weights,
        mem_budget_mb=float(mem_mb) if mem_mb else None,
        use_cupy=use_cupy, device=gpu_index,
    )

    # Peak sampling layer. Default: the grid stack itself -- exact
    # rvs/logpdf by construction, no approximation and no fidelity gate.
    # FSTAT_PEAK_SAMPLING=gmm selects the batched-GMM alternative (memory
    # footprint ~MBs instead of the full grid stack; use with
    # FSTAT_GRID_MEM_MB pressure).
    sampling = str(fstat_knob("FSTAT_PEAK_SAMPLING", str)).strip().lower()
    if sampling == "gmm":
        # TODO(fstat-gmm-deprecation): the GMM sampling layer for the
        # F-stat birth proposal may be deprecated once the grid path proves
        # out at full-band GPU scale -- revisit after the GPU tests. (The
        # GMM machinery itself stays: the serial-search / refit container
        # path is a separate consumer.)
        if "gmm_ncomp" in cache:
            comps = unpack_gmm_components(cache)
            print(f"[birth] GMM components from cache "
                  f"({int(np.sum(cache['gmm_ncomp']))} Gaussians / "
                  f"{len(cache['gmm_ncomp'])} boxes)", flush=True)
        else:
            print("[birth] no GMM in cache; fitting FROM the grids on "
                  "load...", flush=True)
            t0 = time.time()
            comps = fit_gmm_to_stacked(
                stacked,
                n_samples_per_box=fstat_knob("FSTAT_GMM_SAMPLES", int),
                gpu=gpu_index,
                max_comp=fstat_knob("FSTAT_GMM_MAX_COMP", int),
            )
            print(f"[birth]   fitted in {time.time() - t0:.1f}s", flush=True)
        peak_component = build_peak_gmm(comps, box_weights=box_weights,
                                        use_cupy=use_cupy)
        # Fidelity gate (gmm path only): GMM logpdf vs the grid reference
        # on grid draws (host-side CPU copy so the gate runs identically
        # on GPU).
        try:
            gate_gmm = (peak_component if not use_cupy
                        else build_peak_gmm(comps, box_weights=box_weights,
                                            use_cupy=False))
            draws = np.asarray(stacked.rvs(size=(2000,)))
            lp_grid = np.asarray(stacked.logpdf(draws))
            lp_gmm = np.asarray(gate_gmm.logpdf(draws))
            finite = np.isfinite(lp_gmm) & np.isfinite(lp_grid)
            dl = lp_gmm[finite] - lp_grid[finite]
            oob = 1.0 - finite.mean()
            print(f"[gate] GMM vs grid logpdf on {finite.sum()} draws: "
                  f"median dlogp={np.median(dl):+.3f}  "
                  f"[5%, 95%]=[{np.percentile(dl, 5):+.3f}, "
                  f"{np.percentile(dl, 95):+.3f}]  "
                  f"outside-GMM-box frac={oob:.3f}", flush=True)
            if abs(np.median(dl)) > 1.0 or oob > 0.05:
                print("[gate] WARNING: GMM fidelity looks poor (|median "
                      "dlogp| > 1 or >5% of grid draws outside the GMM "
                      "boxes); consider more FSTAT_GMM_SAMPLES / "
                      "FSTAT_GMM_MAX_COMP, or the default grid sampling.",
                      flush=True)
        except Exception as e:  # noqa: BLE001 -- never block a run
            print(f"[gate] fidelity check failed (non-fatal): {e!r}",
                  flush=True)
    else:
        print("[birth] peak sampling: stacked grids (production; exact "
              "rvs/logpdf).", flush=True)
        peak_component = stacked

    comps_list, weights = [peak_component], [1.0]
    c = np.load(comb_cache) if comb_cache else None
    if c is not None and comb_weight > 0:
        comps_list.append(CombIntrinsicProposal(
            c["f0_nodes_mHz"], c["F_max"], mc_lims=mc_lims,
        ))
        weights = [1.0 - comb_weight, float(comb_weight)]
        print(f"[birth] comb component (linear-in-F) weight={comb_weight}",
              flush=True)
    base = comps_list[0] if len(comps_list) == 1 else MixtureProposal(
        comps_list, weights)

    # Floor box: f0 across the whole sampled band (f0_lims from the stacked
    # cache's band_edges when present, else the comb span / box extremes),
    # everything else the full prior ranges.
    if "band_edges" in cache:
        be = np.asarray(cache["band_edges"], dtype=float)
        f0_lo, f0_hi = float(be[1]) * 1e3, float(be[-2]) * 1e3
    elif c is not None:
        f0_lo, f0_hi = float(c["f0_nodes_mHz"][0]), float(c["f0_nodes_mHz"][-1])
    else:
        f0_los = np.asarray(cache["f0_los"], dtype=float)
        n0 = cache["logp_grids"].shape[1]
        f0_lo = float(f0_los.min())
        f0_hi = float((f0_los + (n0 - 1) * np.asarray(cache["f0_dxs"])).max())
    lo = [f0_lo, mc_lims[0], 0.0, -1.0]
    hi = [f0_hi, mc_lims[-1], 2.0 * np.pi, 1.0]
    mix = UniformFloorMixture(base, lo, hi, eps=float(floor_eps))
    print(f"[birth] floor box f0=[{f0_lo:.5f}, {f0_hi:.5f}] mHz  "
          f"eps={floor_eps}", flush=True)
    print(f"[birth] container use_cupy={use_cupy}", flush=True)
    # 9-column ratio basis (use_chirp_mass & use_astrophysical_f0_mc_prior):
    # the birth container must emit the fdot_astro_ratio column too, drawn
    # from its U[-M, M] prior (the F-stat intrinsics stay the 4-D f0/Mc/sky).
    ratio_max = (
        fit.gb.fdot_astro_ratio_max if getattr(fit.gb, "use_fdot_astro", False)
        else None
    )
    if ratio_max is not None:
        print(f"[birth] appending fdot_astro_ratio ~ U[-{ratio_max}, "
              f"{ratio_max}] (9-column basis)", flush=True)
    # Distance basis: slot 0 is dist (kpc) drawn uniform in dist_lims instead
    # of lnA (amplitude is derived from the sampled dist + Mc + f0).
    dist_lims = fit.gb.dist_lims if getattr(fit.gb, "use_distance", False) else None
    if dist_lims is not None:
        print(f"[birth] slot 0 = dist ~ U[{dist_lims[0]}, {dist_lims[1]}] kpc "
              "(distance basis)", flush=True)
    return make_gb_rj_birth_container(
        mix, fit.gb.A_lims, use_cupy=use_cupy, fdot_astro_ratio_max=ratio_max,
        dist_lims=dist_lims,
    )


def main():
    import matplotlib

    matplotlib.use("Agg")
    matplotlib.rcParams["text.usetex"] = False

    from lisatools.globalfit.stock import erebor
    from lisatools.sensitivity import MojitoNoiseEstimates
    from lisatools.globalfit.stock.erebor.noise import resolve_noise_file

    niter = int(os.environ.get("NITER", 20))
    nwalkers = int(os.environ.get("NWALKERS", 16))
    ntemps = int(os.environ.get("NTEMPS", 6))
    fit_dir = os.environ.get("FIT_DIR", "./gf_runs_fstat_rj/run/")

    highest_n = os.environ.get("GB_HIGHEST_FREQUENCIES")
    _band = (f"GB_HIGHEST_FREQUENCIES={highest_n} (N-th highest catalogue "
             "source)" if highest_n else "stock band knobs "
             "(GB_MIN_FREQ/GB_MAX_FREQ or GB_CENTER_FREQ/GB_N_LAYERS)")
    print(f"[main] band={_band}  niter={niter}  "
          f"nwalkers={nwalkers} ntemps={ntemps}", flush=True)

    fit = erebor.gb_no_fg(nwalkers=nwalkers, ntemps=ntemps)
    # search (default) = zero-leaf start; GB_MODE=pe = injection-seeded
    # leaves (debug-at-truth runs)
    fit.gb.mode = os.environ.get("GB_MODE", "search")
    fit.general.file_store_dir = fit_dir
    # GB_USE_CHIRP_MASS=0 -> legacy fdot basis (handles fdot<0 interacting
    # sources; see the fdot<0 TODO in recipe.setup_state_for_injection)
    fit.gb.use_chirp_mass = os.environ.get("GB_USE_CHIRP_MASS", "1") == "1"

    # Noise model: the empirical tabulated NOISE-brick PSD in mojito mode;
    # in synthetic mode (DATA_MODE=synthetic) there is no mojito data to
    # derive it from, so fall through to the stock ANALYTIC sensitivity
    # model (sensitivity_init_kwargs stays None -> the default backend).
    if fit.general.data_mode == "mojito":
        noise_file = resolve_noise_file(fit.general.mojito_data_path)
        fit.general.fixed_psd_kwargs = dict(psd_params=None, galfor_params=None)
        fit.general.sensitivity_init_kwargs = dict(
            tdi_generation=fit.general.tdi_gen,
            extra_components=[MojitoNoiseEstimates(noise_file, which="xyz")],
        )
    else:
        print(f"[noise] data_mode={fit.general.data_mode!r}: using the stock "
              "ANALYTIC sensitivity model (no mojito NOISE brick).", flush=True)

    # The birth proposal (from the cached stacked grids; no kernel sweep
    # needed). In GB_MODE=pe no grids are required -- leaves start at
    # truth and the seeded residual is measured before any RJ birth.
    from lisatools.sampling.fstat_proposal import fstat_knob

    # GB_FSTAT_FIT_IN_MOVE=1: the RJ birth move fits its OWN comb/peak grids
    # in setup(), against the live residual, and caches them under
    # GB_FSTAT_FIT_DIR -- so there is no offline npz to load and
    # FSTAT_GRID_DIR is not required. Everything else about this run
    # (MojitoNoiseEstimates sensitivity, fixed PSD, band knobs) is
    # unchanged, which is what keeps it a like-for-like of the offline-grid
    # runs. This is also the multi-GPU path: the in-move fit routes through
    # route_fstat_ll, while offline prep is single-device by design.
    if getattr(fit.gb, "fstat_fit_in_move", False):
        print("[main] GB_FSTAT_FIT_IN_MOVE=1: the RJ birth move fits its own "
              "F-stat grids in setup(); skipping the offline grid load.",
              flush=True)
    else:
        birth = build_birth_distribution(
            fit,
            floor_eps=fstat_knob("FSTAT_FLOOR_EPS", float),
            comb_weight=fstat_knob("FSTAT_COMB_WEIGHT", float),
        )
        _pe_mode = os.environ.get("GB_MODE", "search") == "pe"
        if birth is None and not _pe_mode:
            sys.exit("FSTAT_GRID_DIR (or the test-only FSTAT_PEAK_GRIDS) is "
                     "required for the search. With GB_FSTAT_FIT_IN_MOVE=1 no "
                     "offline grid is needed.")
        if birth is not None:
            fit.gb.rj_birth_distribution = birth

    fit.general.num_iterations = niter

    print("[main] fit.build() ...", flush=True)
    t0 = time.time()
    curr = fit.build()
    print(f"[main]   built in {time.time() - t0:.1f}s", flush=True)
    bp = curr.general_info.main_file_path
    if os.environ.get("FSTAT_RESUME", "0") == "1":
        # keep the backend: run.py resumes from its last stored sample (the
        # sampler shape env knobs must match the original run's)
        print(f"[main] resuming from existing backend {bp}", flush=True)
    elif bp and os.path.exists(bp):
        os.remove(bp)
        print(f"[main] removed stale backend {bp}", flush=True)

    t0 = time.time()
    fit.run()
    print(f"[main] run finished in {time.time() - t0:.1f}s "
          f"(backend: {bp})", flush=True)


if __name__ == "__main__":
    main()
