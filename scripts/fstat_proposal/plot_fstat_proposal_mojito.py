"""F-stat proposal grid prep on real mojito data: per-sub-band peak grids
+ the GMM sampling layer, driven by the global fit's GB sub-band structure.

Reuses the ``erebor.gb_no_fg`` stock recipe to stand up the full WDM stack
(orbits, TDIConfig, WDMSettings, chunked-het ``GBWDMComputations``, and the
``AnalysisContainerArray`` from the mojito residual), then runs the FIT
pipeline of the F-stat RJ-birth proposal:

1. **Band** from the stock GB band knobs (``GB_MIN_FREQ``/``GB_MAX_FREQ``,
   ``GB_CENTER_FREQ``+``GB_N_LAYERS``, or ``GB_HIGHEST_FREQUENCIES=N``); the
   stock build derives ``band_edges`` -> sub-bands and ``f0_lims``.
2. **Comb scan**: ONE batched kernel sweep over (all f0 nodes at 1/(2*Tobs)
   spacing spanning every interior sub-band) x (sky points), Mc fixed.
3. **Peak selection** (vectorized, on-device): two-tier non-maximum
   suppression -> F floor -> per-sub-band cap; each peak tagged with its
   sub-band index.
4. **Local grids**: every peak's 4-D box (f0 CLAMPED to its sub-band) swept
   in ONE chunked kernel stream -> stacked ``(K, n_f0, n_Mc, n_alpha,
   n_sd)`` grids (``StackedFStatProposal4D``) -- the production sampling
   layer. f0 nodes default to ~1/Tobs cells (the peak width); Mc/sky stay
   anisotropic-coarse (3/8/8).
5. **Save** one ``*_peaks_stacked.npz`` (grids + metadata, + the
   ``*_comb.npz``). ``FSTAT_FIT_GMM=1`` additionally fits the batched
   per-box GMM (``vec_fit_gmm_min_bic``) into the cache -- the optional
   memory-light sampling layer.

Batching invariant: EVERY F-stat computation here runs as single chunked
``get_fstat_ll_wdm`` calls spanning all sub-bands being run -- per-band /
per-sky / per-peak Python loops around kernel calls are forbidden.

The data contains ONLY the GB injection (no noise realization); the fixed
whitening PSD is the *tabulated empirical* estimate from the mojito NOISE
brick (``MojitoNoiseEstimates``, extras-only fixed-PSD path). Diagnostics
reference the loudest in-band catalogue source (no hardcoded targets).

Environment knobs::

    GB_MIN_FREQ / GB_MAX_FREQ / GB_CENTER_FREQ / GB_N_LAYERS /
    GB_HIGHEST_FREQUENCIES
                       (stock band knobs -- SAME semantics as the runner /
                        stock fit, so one setting sizes prep + search
                        identically; GB_CENTER_FREQ_HZ is a deprecated
                        alias for GB_CENTER_FREQ)
    FSTAT_DESIGN       ("comb" [default]: dense-f0 scan + stacked peak fits;
                        "coarse": legacy 4-D tensor grid + zoom -- kept as
                        the narrow-peak negative control)
    FSTAT_F0_SPACING_MHZ / FSTAT_COMB_NSKY / FSTAT_COMB_MC
                       (comb scan: f0 node spacing [default 1/(2*Tobs)],
                        sky-point count [6], fixed Mc [mean(m_chirp_lims)])
    FSTAT_PEAK_MIN_SNR / FSTAT_PEAK_MIN_F
                       (absolute selection floor, F = SNR^2/2; default
                        SNR 5 [F 12.5]; explicit SNR wins over explicit F.
                        Gates BOTH tiers -- the tier-2 local-max rescue is
                        never relative to the window max)
    FSTAT_PEAKS_PER_BAND
                       (per-sub-band peak cap [35])
    FSTAT_PEAKS_TO_FIT / FSTAT_PEAK_HALF_MHZ
                       (global cap on fitted peaks [all]; f0 half-width of
                        the peak box [2.5e-3 mHz ~ Doppler envelope],
                        clamped to the peak's sub-band)
    FSTAT_MC_MIN       (Mc grid-box floor [0.01])
    FSTAT_N_F0         (f0 nodes per box; AUTO default = one cell per
                        ~1/Tobs, clamped [12, 96] -- ~40 at 90 d)
    FSTAT_N_MC / FSTAT_N_ALPHA / FSTAT_N_SINDELTA
                       (anisotropic defaults 3 / 8 / 8 -- Mc/sky are
                        per-band unmeasurable)
    FSTAT_N_PER_AXIS   (blanket override for all four axes when set)
    FSTAT_BATCH        (kernel rows per get_fstat_ll_wdm call [4096])
    FSTAT_GRID_MEM_MB  (K-axis memory budget for the stacked grids)
    FSTAT_FIT_GMM=1    (ALSO fit the optional per-box GMM sampling layer
                        into the cache [off; the grids are production])
    FSTAT_GMM_SAMPLES / FSTAT_GMM_MAX_COMP
                       (GMM fit: samples per box [4096]; max components
                        per box [12])
    FSTAT_GRID_CACHE   (.npz base path for the comb + stacked caches)
    FSTAT_COMB_CACHE_REUSE=1
                       (reuse an existing *_comb.npz; peak selection is
                        re-run with the CURRENT knobs from the cached sweep)
    FSTAT_SAVE_PER_BOX=1
                       (also write legacy per-box *_peak<i>.npz caches)
    FSTAT_PLOT_PEAKS   (corner plots for the top-N fitted peaks [2])
    FSTAT_OUT          (default /tmp/fstat_proposal_mojito.png)
    MOJITO_DATA_PATH   (default ~/.mojito_cache/brickmarket/mojito_light_v1_0_0/)

Outputs: ``*_comb.npz`` + ``*_peaks_stacked.npz`` (grids + GMM + metadata)
under FSTAT_GRID_CACHE, plus diagnostic figures next to FSTAT_OUT.
"""

from __future__ import annotations

import os
import sys
import time

import numpy as np


def _to_host(x):
    """cupy/numpy array -> host numpy. No-op on numpy (GPU-safety helper).

    ``get_fstat_ll_wdm`` and ``FStatProposal4D`` return cupy arrays on the
    CUDA backends; this script stores/plots/serializes with numpy, so every
    kernel/proposal output is funneled through here.
    """
    return x.get() if hasattr(x, "get") else np.asarray(x)


# The stock recipe defaults expect mojito.
os.environ.setdefault(
    "MOJITO_DATA_PATH",
    os.path.expanduser("~/.mojito_cache/brickmarket/mojito_light_v1_0_0/"),
)
os.environ.setdefault("DATA_PROCESSOR", "mojito")
os.environ.setdefault("MAKE_PLOTS", "0")
os.environ.setdefault("NWALKERS", "2")
os.environ.setdefault("NTEMPS", "1")
os.environ.setdefault("GB_USE_CHIRP_MASS", "1")

# NOTE on the two chirp masses (kept for the diagnostics): the wdwd catalogue
# is an *interacting* DWD population, so the injected fdot is not purely
# GW-driven. ``Mc_Msol`` is the catalogue's mass-based chirp mass;
# ``Mc_eff_Msol`` is the chirp mass implied by the injected ``(f0, fdot)``
# through the monochromatic-GB relation -- and since the proposal grid maps
# Mc -> fdot through exactly that relation, the F-stat must peak at
# ``Mc_eff``, not at the mass-based value. The diagnostic reference source is
# always resolved from the catalogue (loudest in band), never hardcoded.


def _resolve_matplotlib():
    """Force a non-interactive backend + disable usetex before any lisatools
    submodule flips it back on. Import after this returns."""
    import matplotlib

    matplotlib.use("Agg")
    matplotlib.rcParams["text.usetex"] = False
    import matplotlib.pyplot as plt  # noqa: F401 (side effect)
    return matplotlib


def _maybe_patch_slab_kernel_args(gb_wdm_comp, wdm_holder, f0_probe_hz):
    """Handle installed GBGPU bindings that predate the task-b per-band slab
    kernel args ``(band_slab_Nf, slab_min_f)``.

    ``chunked_het.py::_slab_kernel_args`` always appends the two trailing
    slab args on the C++ path; a GBGPU build compiled before task-b rejects
    them with an arg-count ``TypeError``. Probe with a single throwaway
    F-stat evaluation and only patch the dispatch when the installed binding
    actually lacks the slots (patching a current build would break it the
    other way). The FStatProposal queries the full active band, so slab-off
    ``(0, empty)`` vs absent is behaviourally identical.
    """
    params = np.zeros((1, 9))
    params[:, 0] = 1e-22
    params[:, 1] = float(f0_probe_hz)  # keep the probe inside the active band
    params = gb_wdm_comp.xp.asarray(params)   # -> device on the CUDA backends
    try:
        gb_wdm_comp.get_fstat_ll_wdm(params, wdm_holder)
        print("[patch] installed GBGPU binding accepts slab args; no patch",
              flush=True)
    except TypeError:
        from lisatools.chunked_het import WDMComputationsBase

        WDMComputationsBase._slab_kernel_args = lambda self, holder: ()
        gb_wdm_comp.get_fstat_ll_wdm(params, wdm_holder)  # re-probe: must pass
        print("[patch] WDMComputationsBase._slab_kernel_args -> () "
              "(pre-task-b GBGPU binding)", flush=True)


def build_gb_wdm_comp_and_holder():
    """Configure ``erebor.gb_no_fg`` on mojito, build it, run ``load_info`` +
    ``setup_acs`` to get the ``AnalysisContainerArray``, then manually
    construct ``GBWDMComputations`` from the pre-built settings.

    The GB analysis band comes from the SAME stock knobs the runner / stock
    fit honor (``GB_MIN_FREQ``/``GB_MAX_FREQ``, ``GB_CENTER_FREQ`` +
    ``GB_N_LAYERS``, ``GB_HIGHEST_FREQUENCIES`` -- consumed by the stock
    ``GBSettings`` env-backed defaults at construction), so one band setting
    sizes grid prep and search identically.

    Returns ``(gb_wdm_comp, wdm_holder, curr)`` where ``wdm_holder`` is the
    residual ``AnalysisContainerArray`` (what ``get_fstat_ll_wdm`` consumes).
    """
    from mpi4py import MPI
    from eryn.state import BranchSupplemental

    from lisatools.globalfit.run import GlobalFit
    from lisatools.globalfit.stock import erebor

    print("[build] configuring erebor.gb_no_fg...", flush=True)
    fit = erebor.gb_no_fg(nwalkers=2, ntemps=1)
    # Deprecated alias: this script's old private band knob. The stock
    # GB_CENTER_FREQ (+ GB_N_LAYERS) is the supported interface.
    if "GB_CENTER_FREQ_HZ" in os.environ and fit.gb.center_freq is None:
        fit.gb.center_freq = float(os.environ["GB_CENTER_FREQ_HZ"])
        print("[build] GB_CENTER_FREQ_HZ is DEPRECATED; use GB_CENTER_FREQ "
              "(same value, stock knob).", flush=True)
    fit.gb.use_chirp_mass = True

    # --- Empirical PSD from the mojito NOISE brick (tabulated estimates) ---
    # gb_no_fg mojito mode loads ONLY the GB injection (no noise realization
    # in the data); by default it whitens with scalar params least-squares
    # fitted to the NOISE brick. For these tests we want the *tabulated*
    # empirical PSD instead: psd_params=None routes the fixed-PSD build
    # through extra_components only (extras-only path).
    from lisatools.sensitivity import MojitoNoiseEstimates
    from lisatools.globalfit.stock.erebor.noise import resolve_noise_file

    noise_file = resolve_noise_file(fit.general.mojito_data_path)
    if noise_file is None:
        raise FileNotFoundError(
            "mojito NOISE brick not found under "
            f"{fit.general.mojito_data_path!r} -- the empirical PSD requires it."
        )
    fit.general.fixed_psd_kwargs = dict(psd_params=None, galfor_params=None)
    fit.general.sensitivity_init_kwargs = dict(
        tdi_generation=fit.general.tdi_gen,
        extra_components=[MojitoNoiseEstimates(noise_file, which="xyz")],
    )
    print(f"[build] empirical PSD: MojitoNoiseEstimates({os.path.basename(noise_file)})",
          flush=True)

    print(f"[build] fit.build() -- loads mojito data...", flush=True)
    t0 = time.time()
    curr = fit.build()
    print(f"[build]   done in {time.time() - t0:.1f}s", flush=True)

    gi = curr.general_info
    gb_info = curr.source_info["gb"]
    print(f"[build] GB analysis band: f0_lims={gb_info.f0_lims}  "
          f"n_layers={getattr(gb_info, 'n_layers', '?')}", flush=True)
    print(f"[build] data_t0={gi.data_t0:.4f}  Tobs={gi.Tobs:.4e}s  "
          f"({gi.Tobs / 86400:.1f} d)", flush=True)

    # --- Manually build GBWDMComputations (mirror gb_no_fg.setup_gb_moves) ---
    from lisatools.domains import WDMSettings
    from gbgpu.gbcomps import GBWDMComputations

    if not isinstance(gi.domain_settings, WDMSettings):
        raise RuntimeError(
            f"Expected WDMSettings but got {type(gi.domain_settings).__name__}. "
            "gb_no_fg should default to WDM."
        )
    _wdm = gi.domain_settings
    _orig_t0 = float(_wdm.t0)
    _wdm.t0 = float(getattr(gi, "data_t0", 0.0))
    tdi_gen = 2 if getattr(gb_info, "use_tdi2", True) else 1
    tdi_gen_str = f"{tdi_gen}{'nd' if tdi_gen == 2 else 'st'} generation"
    print(f"[build] constructing GBWDMComputations (Nf={_wdm.Nf}, Nt={_wdm.Nt}, "
          f"backend={gi.force_backend})...", flush=True)
    t0 = time.time()
    gb_wdm_comp = GBWDMComputations(
        _wdm,
        t_ref=gb_info.t0,
        Nt_sub=int(gb_info.nt_sub),
        n_pad=int(gb_info.n_pad),
        N_sparse=int(gb_info.n_sparse),
        N_cp_sig=int(gb_info.n_cp_sig),
        N_cp_orbit=int(gb_info.n_cp_orbit),
        orbits=gi.gpu_orbits,
        tdi_config=tdi_gen_str,
        force_backend=gi.force_backend,
        tdi_type="XYZ",
    )
    print(f"[build]   done in {time.time() - t0:.1f}s", flush=True)

    # --- Build the AnalysisContainerArray via GlobalFit.load_info + setup_acs ---
    print("[build] GlobalFit.load_info() + setup_acs()...", flush=True)
    comm = MPI.COMM_WORLD
    bp = gi.main_file_path
    if os.path.exists(bp):
        os.remove(bp)
    gf = GlobalFit(curr, comm)

    priors = {}
    for name in gf.engine_info.branch_names:
        si = curr.source_info.get(name)
        if si is None:
            continue
        p = si.get("priors") if isinstance(si, dict) else getattr(si, "priors", None)
        if p:
            priors.update(p)

    state = gf.load_info(priors)
    ntemps, nwalkers = gf.ntemps, gf.nwalkers
    state.supplemental = BranchSupplemental(
        {"walker_inds": np.tile(np.arange(nwalkers), (ntemps, 1))},
        base_shape=(ntemps, nwalkers), copy=True,
    )
    t0 = time.time()
    acs = gf.setup_acs(state, rebuild_residuals=False)
    print(f"[build]   done in {time.time() - t0:.1f}s "
          f"(N containers: {len(acs.flatten())})", flush=True)

    # The wdm_holder for get_fstat_ll_wdm is the AnalysisContainerArray itself.
    return gb_wdm_comp, acs, curr


def build_fstat_proposal(gb_wdm_comp, wdm_holder, src, gb_info,
                          n_per_axis: int = 24, box=None, stage="stage1"):
    """Build FStatProposal4D on a grid box.

    Default box: the analysis band (``gb_info.f0_lims``) in f0 and sensible
    global defaults on the other axes; pass ``box`` (dict of ``*_range``
    tuples) to zoom. Per-axis node counts come from ``FSTAT_N_F0`` /
    ``FSTAT_N_MC`` / ``FSTAT_N_ALPHA`` / ``FSTAT_N_SINDELTA`` (each
    defaulting to ``n_per_axis`` = ``FSTAT_N_PER_AXIS``) -- on CPU the
    ~0.2 s/eval kernel cost makes an anisotropic grid (dense in f0, coarse in
    Mc/sky) the difference between minutes and hours.
    """
    from lisatools.sampling.fstat_proposal import FStatProposal4D, GridSpec

    n_f0 = int(os.environ.get("FSTAT_N_F0", n_per_axis))
    n_Mc = int(os.environ.get("FSTAT_N_MC", n_per_axis))
    n_alpha = int(os.environ.get("FSTAT_N_ALPHA", n_per_axis))
    n_sin_delta = int(os.environ.get("FSTAT_N_SINDELTA", n_per_axis))

    if box is None:
        box = dict(
            f0_range=(float(gb_info.f0_lims[0]) * 1e3,
                      float(gb_info.f0_lims[-1]) * 1e3),
            Mc_range=_mc_grid_range(gb_info),
            alpha_range=(0.0, 2 * np.pi),
            sin_delta_range=(-1.0, 1.0),
        )
    print(f"[grid:{stage}] f0 range: [{box['f0_range'][0]:.4f}, "
          f"{box['f0_range'][1]:.4f}] mHz  (source at {src['f0_mHz']:.4f} mHz)",
          flush=True)

    grid = GridSpec(
        n_f0=n_f0, n_Mc=n_Mc, n_alpha=n_alpha, n_sin_delta=n_sin_delta,
        **box,
    )
    n_total = n_f0 * n_Mc * n_alpha * n_sin_delta
    print(f"[grid:{stage}] {n_f0}x{n_Mc}x{n_alpha}x{n_sin_delta} = {n_total} "
          f"F-stat evals...", flush=True)
    t0 = time.time()
    prop = FStatProposal4D(
        gb_wdm_comp=gb_wdm_comp,
        wdm_holder=wdm_holder,
        grid_spec=grid,
        beta=1.0,
    )
    print(f"[grid:{stage}]   done in {time.time() - t0:.1f}s", flush=True)
    return prop


def _zoom_box(prop, n_cells: float = 2.0):
    """Zoom box centred on the stage-1 argmax cell: +/- ``n_cells`` stage-1
    cell widths per axis, clipped to the stage-1 box."""
    g = _to_host(prop._logp_grid[:-1, :-1, :-1, :-1])
    idx = np.unravel_index(np.argmax(g), g.shape)
    names = ("f0_range", "Mc_range", "alpha_range", "sin_delta_range")
    box = {}
    for j, name in enumerate(names):
        ax = _to_host(prop._axes[j])
        centre = 0.5 * (ax[idx[j]] + ax[idx[j] + 1])
        half = n_cells * float(prop._dx[j])
        box[name] = (max(float(ax[0]), centre - half),
                     min(float(ax[-1]), centre + half))
    return box


def _band_catalogue_sources(f0_lo_mHz, f0_hi_mHz):
    """Full truth parameters of every catalogue GB inside the band (or None).

    Returns a dict of arrays in the SAMPLING basis: ``f0`` [mHz], ``amp``,
    ``Mc_eff`` (chirp mass implied by the injected (f0, fdot) -- nan for
    fdot <= 0 interacting systems), ``alpha`` [rad], ``sin_delta``.
    Sky is stored equatorial in the catalogue, so RA/Dec map directly.
    """
    try:
        import h5py

        from gbgpu.utils.utility import get_chirp_mass_from_f_fdot

        path = os.path.join(
            os.environ["MOJITO_DATA_PATH"],
            "catalogues", "wdwd_cat_mojito_lite_processed.hdf5",
        )
        with h5py.File(path, "r") as f:
            B = f["Binaries"]
            f0 = B["GW22FrequencySSBFrame"][:] * 1e3
            keep = (f0 >= f0_lo_mHz) & (f0 <= f0_hi_mHz)
            amp = B["Amplitude"][keep]
            fdot = B["GW22FrequencyDerivativeSourceFrame"][keep]
            ra = B["RightAscension"][keep]
            dec = B["Declination"][keep]
        f0 = f0[keep]
        with np.errstate(invalid="ignore"):
            mc_eff = np.where(
                fdot > 0,
                get_chirp_mass_from_f_fdot(f0 * 1e-3, np.clip(fdot, 0, None)),
                np.nan,
            )
        return dict(f0=f0, amp=amp, Mc_eff=mc_eff, alpha=ra,
                    sin_delta=np.sin(dec))
    except Exception as e:  # pragma: no cover - cosmetic overlay only
        print(f"[cat] catalogue overlay unavailable: {e}", flush=True)
        return None


def _mc_grid_range(gb_info):
    """Mc GRID-box range: ``[max(prior floor, FSTAT_MC_MIN), prior top]``.

    The proposal/prior Mc limits stay the full ``m_chirp_lims``; the grid
    box floors at ``FSTAT_MC_MIN`` (default 0.01) because below that fdot is
    indistinguishable from 0 over months of data -- grid nodes there waste
    kernel evals without adding proposal coverage (the uniform floor mixture
    covers the full prior box).
    """
    from lisatools.sampling.fstat_proposal import fstat_knob

    mc_lims = list(getattr(gb_info, "m_chirp_lims", None) or [0.001, 1.0])
    mc_min = fstat_knob("FSTAT_MC_MIN", float)
    return (max(float(mc_lims[0]), mc_min), float(mc_lims[-1]))


def _comb_mc_fix(gb_info):
    """Comb-scan fixed Mc: ``FSTAT_COMB_MC`` env else ``mean(m_chirp_lims)``."""
    mc_lims = list(getattr(gb_info, "m_chirp_lims", None) or [0.001, 1.0])
    return float(os.environ.get(
        "FSTAT_COMB_MC", 0.5 * (float(mc_lims[0]) + float(mc_lims[-1]))
    ))


def _windowed_max(a, w, xp):
    """Sliding max over ``[i - w, i + w]`` via O(log w) shifted-max folds
    (xp-generic; no scipy)."""
    n = a.shape[0]
    pad = xp.full(w, -xp.inf, dtype=a.dtype)
    b = xp.concatenate([pad, a, pad])
    width = 2 * w + 1
    r = b.copy()
    done = 1
    while done < width:
        step = min(done, width - done)
        r[: r.shape[0] - step] = xp.maximum(r[: r.shape[0] - step], r[step:])
        done += step
    return r[:n]


# --- sweep checkpointing (FSTAT_GRID_CACHE runs only) -----------------------
# Every expensive sweep funnels through _chunked_fstat_sweep /
# run_stacked_peak_sweep, so checkpointing the row cursor of those two
# loops makes the whole ~hours-scale grid prep restartable: rerunning the
# SAME command resumes each unfinished sweep from its last saved row (a
# fingerprint of the sweep's actual inputs guards against resuming across
# changed knobs), and finished stages short-circuit through their final
# npz caches (*_comb.npz via FSTAT_COMB_CACHE_REUSE=1). Progress files live
# in ``<FSTAT_GRID_CACHE base>_parts/`` and are deleted once the stage's
# final npz is written. Cadence: FSTAT_CKPT_SECS (default 300).

def _ckpt_fingerprint(*arrays, extra=""):
    import hashlib
    h = hashlib.md5(extra.encode())
    for a in arrays:
        a = np.ascontiguousarray(_to_host(a))
        step = max(1, a.size // 257)
        h.update(str(a.shape).encode())
        h.update(a.reshape(-1)[::step].tobytes())
    return h.hexdigest()


def _ckpt_load(ckpt, n, fingerprint):
    """Returns (host F array or None, completed row count)."""
    if not ckpt:
        return None, 0
    path = ckpt + ".progress.npz"
    if not os.path.exists(path):
        return None, 0
    try:
        d = np.load(path, allow_pickle=False)
        if str(d["fingerprint"]) != fingerprint or int(d["n"]) != n:
            print(f"[ckpt] {path}: inputs changed -> restarting this sweep",
                  flush=True)
            return None, 0
        done = int(d["done"])
        print(f"[ckpt] resuming {os.path.basename(ckpt)} at {done}/{n} rows",
              flush=True)
        return np.array(d["F"]), done
    except Exception as exc:  # unreadable/partial file -> recompute
        print(f"[ckpt] {path}: unreadable ({exc}) -> restarting this sweep",
              flush=True)
        return None, 0


def _ckpt_save(ckpt, F_host, done, n, fingerprint):
    if not ckpt:
        return
    path = ckpt + ".progress.npz"
    tmp = ckpt + ".progress.tmp.npz"
    os.makedirs(os.path.dirname(path), exist_ok=True)
    np.savez(tmp, F=F_host, done=done, n=n, fingerprint=fingerprint)
    os.replace(tmp, path)


def _ckpt_clear(parts_dir, prefix):
    if not parts_dir or not os.path.isdir(parts_dir):
        return
    for fn in os.listdir(parts_dir):
        if fn.startswith(prefix) and fn.endswith(".progress.npz"):
            try:
                os.remove(os.path.join(parts_dir, fn))
            except OSError:
                pass


def _ckpt_secs():
    return float(os.environ.get("FSTAT_CKPT_SECS", "300"))


def _chunked_fstat_sweep(gb_wdm_comp, wdm_holder, params, label="", ckpt=None):
    """ONE chunked kernel stream over a pre-assembled physical params array.

    The batching invariant: every F-stat computation in this script sweeps a
    single concatenated parameter set spanning ALL sub-bands / sky points /
    peak boxes at once, in ``FSTAT_BATCH``-row ``get_fstat_ll_wdm`` calls --
    never a per-band / per-sky / per-peak kernel loop. Returns the F values
    on the compute backend's array module.

    ``ckpt``: optional progress-file base path; the sweep then resumes from
    the last saved row on rerun (see the checkpointing block above).
    """
    from lisatools.sampling.fstat_proposal import compute_fstat, fstat_knob

    xp = gb_wdm_comp.xp
    fp = _ckpt_fingerprint(params, extra=label) if ckpt else ""
    params = xp.asarray(params)
    n = params.shape[0]
    batch = fstat_knob("FSTAT_BATCH", int)
    F = xp.empty(n, dtype=xp.float64)
    F_prev, start = _ckpt_load(ckpt, n, fp)
    if F_prev is not None and start > 0:
        F[:start] = xp.asarray(F_prev[:start])
    t0 = time.time()
    last = t0
    last_ckpt = t0
    for s in range(start, n, batch):
        e = min(s + batch, n)
        N_arr, M_up = gb_wdm_comp.get_fstat_ll_wdm(params[s:e], wdm_holder)
        F[s:e] = compute_fstat(xp.asarray(N_arr), xp.asarray(M_up))
        if time.time() - last > 30:
            last = time.time()
            print(f"[sweep{label}] {e}/{n} evals "
                  f"({time.time() - t0:.0f}s)", flush=True)
        if ckpt and e < n and time.time() - last_ckpt > _ckpt_secs():
            last_ckpt = time.time()
            _ckpt_save(ckpt, _to_host(F[:e]), e, n, fp)
    if ckpt and start < n:
        _ckpt_save(ckpt, _to_host(F), n, n, fp)
    print(f"[sweep{label}] {n} evals in one chunked stream "
          f"(batch={batch}, {time.time() - t0:.0f}s"
          f"{f', resumed at {start}' if start else ''})", flush=True)
    return F


def select_comb_peaks(f0_nodes, F_max, gb_info, spacing, xp):
    """Vectorized per-sub-band peak selection from the comb's ``F_max(f0)``.

    All-array pipeline over every sub-band at once (xp-generic: stays
    on-device on the CUDA backends; only the final small peak table moves to
    host). Two-tier candidate set -> absolute SNR floor + interior-band mask
    -> per-band cap:

    * **Tier 1** (window champion): a node survives iff it carries the max F
      within +-``min_sep`` nodes (the Doppler envelope ~3e-3 mHz).
    * **Tier 2** (local-max rescue): ALSO keep any strict 3-point local
      maximum. The rescue is ABSOLUTE -- gated only by the shared SNR floor
      below, never by the local window max -- so a quiet real source inside
      a loud neighbor's Doppler window still gets its own grid (an extra
      grid on a sky image is harmless redundancy bounded by the per-band
      cap; a missing grid on a real source is the bad failure).

    The floor is SNR-based (``F = SNR^2 / 2``): set it as
    ``FSTAT_PEAK_MIN_SNR`` (wins when set) or directly as
    ``FSTAT_PEAK_MIN_F`` (default 20 ~ SNR 6.3).

    Returns a host ``(N, 4)`` array of ``(f0_mHz, F, node_idx, band_idx)``
    sorted by F descending. band_idx indexes ``gb_info.band_edges`` cells;
    only interior sub-bands (1 .. num_sub_bands - 2, the ``f0_lims`` span)
    are kept -- the consumer's per-cell gate rejects births elsewhere.
    """
    from lisatools.sampling.fstat_proposal import fstat_knob, fstat_peak_min_F

    f0_nodes = xp.asarray(f0_nodes)
    F_max = xp.asarray(F_max)
    # band_edges is Hz; ALL script-side f0 values are mHz. Convert ONCE.
    band_edges_mHz = xp.asarray(np.asarray(gb_info.band_edges, dtype=float) * 1e3)
    num_sub_bands = int(len(gb_info.band_edges) - 1)
    band_of_node = xp.searchsorted(band_edges_mHz, f0_nodes, side="right") - 1

    min_sep = max(1, int(round(3e-3 / spacing)))
    wmax = _windowed_max(F_max, min_sep, xp)
    tier1 = F_max >= wmax
    local3 = xp.zeros(F_max.shape, dtype=bool)
    local3[1:-1] = (F_max[1:-1] > F_max[:-2]) & (F_max[1:-1] > F_max[2:])
    tier2 = local3

    min_F = fstat_peak_min_F()
    interior = (band_of_node >= 1) & (band_of_node <= num_sub_bands - 2)
    cand = (tier1 | tier2) & (F_max >= min_F) & interior

    idx = xp.where(cand)[0]
    if int(idx.shape[0]) == 0:
        print(f"[peaks] NO peaks above F >= {min_F} in the interior "
              "sub-bands.", flush=True)
        return np.empty((0, 4))
    b = band_of_node[idx]
    Fv = F_max[idx]

    # Vectorized per-band cap: sort by (band, -F), within-band rank via the
    # band-group start offsets, keep rank < cap.
    cap = fstat_knob("FSTAT_PEAKS_PER_BAND", int)
    order = xp.lexsort(xp.stack([-Fv, b.astype(xp.float64)]))
    b_sorted = b[order]
    rank = xp.arange(order.shape[0]) - xp.searchsorted(
        b_sorted, b_sorted, side="left"
    )
    idx_keep = idx[order[rank < cap]]

    f0_h = _to_host(f0_nodes[idx_keep])
    F_h = _to_host(F_max[idx_keep])
    b_h = _to_host(band_of_node[idx_keep])
    peaks = np.column_stack([f0_h, F_h, _to_host(idx_keep), b_h])
    peaks = peaks[np.argsort(peaks[:, 1])[::-1]]

    counts = np.bincount(b_h.astype(int), minlength=num_sub_bands)
    print(f"[peaks] {len(peaks)} peaks (F >= {min_F:.1f} ~ SNR "
          f"{np.sqrt(2 * min_F):.1f}, cap {cap}/band); "
          f"per-interior-band counts: "
          f"{dict(enumerate(counts.tolist()))}", flush=True)
    zero_bands = [k for k in range(1, num_sub_bands - 1) if counts[k] == 0]
    if zero_bands:
        print(f"[peaks] WARNING: interior sub-band(s) {zero_bands} have ZERO "
              "peaks -- births there fall to the comb/floor components only.",
              flush=True)
    return peaks


def run_comb_scan(gb_wdm_comp, wdm_holder, gb_info, general_info, src,
                  out_path, cat_sources, cache_path=None):
    """Dense-in-f0 F-stat comb scan across the sub-band.

    With months of data the F-stat f0 peaks are ~1/Tobs wide (~1e-4 mHz) --
    far too narrow for any feasible 4-D tensor grid to land on (the plan's
    "narrow-peak" regime). But over the same stretch ``fdot*Tobs << 1/Tobs``,
    so Mc is nearly unmeasurable per band and can be held FIXED, and sky only
    enters through the Doppler ridge (peak shifts up to ~f0*v/c ~ 2e-3 mHz).
    The right scan is therefore dense in f0 (spacing ~ 1/(2*Tobs)) x a small
    spread of sky points, maximized over sky per f0 node. The whole
    (sky x f0) design is assembled into ONE parameter set and swept in a
    single chunked kernel stream (the batching invariant -- no per-sky
    loop); peak selection is the vectorized per-sub-band pipeline of
    :func:`select_comb_peaks`.

    Returns ``(f0_nodes_mHz, F_max, peaks, extras)`` where ``peaks`` is the
    host ``(N, 4)`` array ``(f0_mHz, F, node_idx, band_idx)``.
    """
    from gbgpu.utils.utility import get_fdot

    Tobs = float(general_info.Tobs)
    f0_lo = float(gb_info.f0_lims[0]) * 1e3
    f0_hi = float(gb_info.f0_lims[-1]) * 1e3
    spacing = float(os.environ.get("FSTAT_F0_SPACING_MHZ", 0.5 / Tobs * 1e3))
    f0_nodes = np.arange(f0_lo, f0_hi + 0.5 * spacing, spacing)
    mc_fix = _comb_mc_fix(gb_info)
    n_nodes = len(f0_nodes)
    xp = gb_wdm_comp.xp

    # Per-node sky count. FIXED if FSTAT_COMB_NSKY is set (back-compat);
    # otherwise ADAPTIVE: the comb resolves a source through its Doppler ridge
    # (peak shift ~ f0*v/c), so the sky grid must scale as ~(f0*Tobs*v/c)^2.
    # Floored at FSTAT_COMB_NSKY_MIN (default 16, "something we know works"),
    # capped at FSTAT_COMB_NSKY_MAX, and snapped up to powers of two so the
    # sweep runs as a few rectangular (n_sky x n_nodes_at_level) batches
    # instead of one giant fixed-n_sky design that over-samples low f.
    vc = float(os.environ.get("FSTAT_COMB_SKY_VC", "1e-4"))
    nsky_min = int(os.environ.get("FSTAT_COMB_NSKY_MIN", "16"))
    nsky_max = int(os.environ.get("FSTAT_COMB_NSKY_MAX", "512"))
    _fixed = os.environ.get("FSTAT_COMB_NSKY", "").strip()
    if _fixed:
        nsky_per_node = np.full(n_nodes, int(_fixed), dtype=int)
    else:
        req = np.ceil((f0_nodes * 1e-3 * Tobs * vc) ** 2)
        lvl = (2 ** np.ceil(np.log2(np.clip(req, 1.0, None)))).astype(int)
        nsky_per_node = np.clip(lvl, nsky_min, nsky_max).astype(int)

    def _sky_grid(n):
        # golden-ratio spread over the sphere
        ks = np.arange(n)
        return ((2.0 * np.pi * ks * 0.6180339887) % (2.0 * np.pi),  # alpha
                -1.0 + 2.0 * (ks + 0.5) / n)                         # sin_delta

    levels = np.unique(nsky_per_node)
    print(f"[comb] {n_nodes} f0 nodes; sky "
          f"{'FIXED=' + str(int(levels[0])) if len(levels) == 1 else 'ADAPTIVE'} "
          f"[{int(nsky_per_node.min())}..{int(nsky_per_node.max())}] over "
          f"{len(levels)} level(s) (min={nsky_min}, ~(f0*Tobs*{vc:g})^2); "
          f"spacing {spacing:.3e} mHz = {spacing / (1e3 / Tobs):.2f}/Tobs; "
          f"Mc fixed {mc_fix:.3f}", flush=True)

    # Sweep each sky level as its own rectangular batch; fill the per-node max
    # F and the best-sky (alpha, sin_delta) per node (the sky row that won).
    parts_dir = cache_path.replace(".npz", "_parts") if cache_path else None
    F_max_host = np.zeros(n_nodes)
    best_al_host = np.zeros(n_nodes)
    best_sd_host = np.zeros(n_nodes)
    total_evals = 0
    for lv in levels:
        idx = np.where(nsky_per_node == lv)[0]
        nodes = f0_nodes[idx]
        nn = len(nodes)
        al, sd = _sky_grid(int(lv))
        al = np.asarray(al); sd = np.asarray(sd)
        params = np.zeros((int(lv) * nn, 9))
        params[:, 0] = 1e-22
        params[:, 1] = np.tile(nodes, int(lv)) * 1e-3
        params[:, 2] = get_fdot(f=params[:, 1], Mc=np.full(params.shape[0], mc_fix))
        params[:, 5] = 0.5 * np.pi
        params[:, 7] = np.repeat(al, nn)
        params[:, 8] = np.arcsin(np.repeat(sd, nn))
        Fd = _chunked_fstat_sweep(
            gb_wdm_comp, wdm_holder, params, label=f":comb.nsky{int(lv)}",
            ckpt=(os.path.join(parts_dir, f"comb_nsky{int(lv)}")
                  if parts_dir else None),
        ).reshape(int(lv), nn)
        _kb = _to_host(Fd.argmax(axis=0)).astype(int)
        F_max_host[idx] = _to_host(Fd.max(axis=0))
        best_al_host[idx] = al[_kb]
        best_sd_host[idx] = sd[_kb]
        total_evals += int(lv) * nn
    print(f"[comb] total {total_evals} F-stat evals across {len(levels)} sky "
          f"level(s)", flush=True)

    F_max_dev = xp.asarray(F_max_host)
    peaks = select_comb_peaks(f0_nodes, F_max_dev, gb_info, spacing, xp)
    print("[comb] top peaks (f0 [mHz], F, band):", flush=True)
    for f0p, Fp, _, bi in peaks[:10]:
        print(f"[comb]   {f0p:.5f}  {Fp:10.2f}  band {int(bi)}", flush=True)

    F_max = F_max_host

    # Persist the sweep BEFORE any plotting -- a cosmetic figure failure must
    # never lose the kernel work. peaks is (N, 3): (f0, F, band_idx) +
    # band_edges so the runner/diagnostics can bin without a rebuild.
    if cache_path:
        comb_cache = cache_path.replace(".npz", "_comb.npz")
        _al_top, _sd_top = _sky_grid(int(nsky_per_node.max()))
        np.savez(comb_cache, f0_nodes_mHz=f0_nodes, F_max=F_max,
                 peaks=peaks[:, (0, 1, 3)] if len(peaks) else np.empty((0, 3)),
                 band_edges=np.asarray(gb_info.band_edges, dtype=float),
                 F_all=F_max[None, :], sky_alpha=_al_top, sky_sin_delta=_sd_top,
                 best_alpha=best_al_host, best_sin_delta=best_sd_host,
                 nsky_per_node=nsky_per_node)
        print(f"[cache] wrote {comb_cache}", flush=True)
        # The comb npz is now the durable artifact; drop the per-level
        # progress files so a later knob change can't resurrect stale rows.
        _ckpt_clear(parts_dir, "comb_")

    # --- comb figure: F(f0) vs the catalogue comb + proposal draws ---
    try:
        import matplotlib.pyplot as plt

        # rvs draws from the two comb-implied f0 densities (cheap, numpy):
        # beta=1 (w ~ exp(F): the true birth proposal -- collapses onto the
        # loudest peak) and tempered linear-in-F (w ~ F: proportional mass
        # on every peak, the successive-birth weighting).
        rng_s = np.random.default_rng(11)

        def _draw_f0(w_cells, n):
            w = np.clip(w_cells, 0, None)
            cdf = np.cumsum(w)
            cdf /= cdf[-1]
            idx = np.searchsorted(cdf, rng_s.random(n), side="right")
            idx = np.clip(idx, 0, len(w) - 1)
            u = rng_s.random(n)
            return f0_nodes[idx] + u * (f0_nodes[idx + 1] - f0_nodes[idx])

        g = F_max - F_max.max()
        s_exp = _draw_f0(0.5 * (np.exp(g[:-1]) + np.exp(g[1:])), 3000)
        s_lin = _draw_f0(0.5 * (F_max[:-1] + F_max[1:]), 3000)
        y_exp = np.interp(s_exp, f0_nodes, F_max) * 10 ** rng_s.uniform(
            0.10, 0.45, s_exp.size)
        y_lin = np.interp(s_lin, f0_nodes, F_max) * 10 ** rng_s.uniform(
            0.10, 0.45, s_lin.size)

        fig, ax = plt.subplots(figsize=(12, 4.8))
        ax.semilogy(f0_nodes, np.clip(F_max, 1e-3, None), "-", lw=0.7,
                    color="C0", label="max-over-sky F-stat")
        if cat_sources is not None:
            cf0, camp = cat_sources["f0"], cat_sources["amp"]
            for i in range(len(cf0)):
                ax.axvline(cf0[i], color="0.6", lw=0.8,
                           alpha=float(min(1.0, 0.2 + 0.8 * camp[i] / camp.max())),
                           zorder=0, label="catalogue GBs" if i == 0 else None)
        ax.axvline(src["f0_mHz"], color="r", ls="--", lw=1.2, label="target GB")
        ax.scatter(s_lin, y_lin, s=4, alpha=0.15, color="darkorange", zorder=3,
                   label="3k rvs draws, tempered (w ∝ F)")
        ax.scatter(s_exp, y_exp, s=4, alpha=0.15, color="green", zorder=4,
                   label="3k rvs draws, β=1 (w ∝ e^F)")
        ax.set_xlabel("f0 [mHz]")
        ax.set_ylabel("F-stat (max over sky)")
        ax.set_ylim(max(1e-3, np.clip(F_max, 1e-3, None).min() * 0.5),
                    F_max.max() * 10 ** 0.7)
        ax.set_title(f"F-stat comb scan + proposal draws, {len(f0_nodes)} "
                     f"nodes, Tobs={Tobs / 86400:.0f} d (draws jittered "
                     "above the curve)")
        ax.legend(fontsize=8, loc="upper left")
        fig.tight_layout()
        comb_path = out_path.replace(".png", "_comb.png")
        fig.savefig(comb_path, dpi=140)
        plt.close(fig)
        print(f"[plot] wrote {comb_path}", flush=True)
    except Exception as e:  # pragma: no cover - figure is cosmetic
        print(f"[plot] comb figure failed (data cached): {e}", flush=True)

    _al_top, _sd_top = _sky_grid(int(nsky_per_node.max()))
    extras = dict(sky_alpha=_al_top, sky_sin_delta=_sd_top, F_all=F_max[None, :],
                  best_alpha=best_al_host, best_sin_delta=best_sd_host,
                  mc_fix=mc_fix, spacing=spacing, nsky_per_node=nsky_per_node)
    return f0_nodes, F_max, peaks, extras


def run_peak_profile(gb_wdm_comp, wdm_holder, general_info, src, peak_f0_mHz,
                     comb_extras, out_path, cat_sources, rank=0):
    """Ultra-dense 1-D f0 scan through one comb peak at its best-fit sky.

    Resolves the actual peak shape (spacing 1/(10*Tobs), +/- 25/Tobs) so the
    measured FWHM can be compared against the matched-filter prediction
    ~1/Tobs. This is the "the peak really is that tight" figure.
    """
    from gbgpu.utils.utility import get_fdot

    from lisatools.sampling.fstat_proposal import compute_fstat

    Tobs = float(general_info.Tobs)
    df_T = 1e3 / Tobs                     # 1/Tobs in mHz
    spacing = df_T / 10.0
    half = 25.0 * df_T
    f0_nodes = np.arange(peak_f0_mHz - half, peak_f0_mHz + half + 0.5 * spacing,
                         spacing)

    # Best sky point at the profiled peak: the sky row that produced the
    # comb's value at the global-max column (rank-0 peak; profile only runs
    # there by default).
    F_all = comb_extras["F_all"]
    best_col = int(np.argmax(np.asarray(F_all).max(axis=0)))
    if comb_extras.get("best_alpha") is not None:  # adaptive sky: per-node best
        alpha_best = float(np.asarray(comb_extras["best_alpha"])[best_col])
        sd_best = float(np.asarray(comb_extras["best_sin_delta"])[best_col])
    else:                                          # legacy fixed sky grid
        k_best = int(np.argmax(np.asarray(F_all)[:, best_col]))
        alpha_best = float(comb_extras["sky_alpha"][k_best])
        sd_best = float(comb_extras["sky_sin_delta"][k_best])
    mc_fix = float(comb_extras["mc_fix"])
    print(f"[profile] peak {rank}: {len(f0_nodes)} nodes, spacing "
          f"{spacing:.2e} mHz (= 1/(10 Tobs)), sky (alpha={alpha_best:.3f}, "
          f"sin_delta={sd_best:.3f}), Mc={mc_fix:.3f}", flush=True)

    xp = gb_wdm_comp.xp
    params = np.zeros((len(f0_nodes), 9))
    params[:, 0] = 1e-22
    params[:, 1] = f0_nodes * 1e-3
    params[:, 2] = get_fdot(f=params[:, 1], Mc=np.full(len(f0_nodes), mc_fix))
    params[:, 5] = 0.5 * np.pi
    params[:, 7] = alpha_best
    params[:, 8] = np.arcsin(sd_best)
    params = xp.asarray(params)   # -> device on the CUDA backends
    F_prof = np.zeros(len(f0_nodes))
    for s in range(0, len(f0_nodes), 4096):
        e = min(s + 4096, len(f0_nodes))
        N_arr, M_up = gb_wdm_comp.get_fstat_ll_wdm(params[s:e], wdm_holder)
        F_prof[s:e] = _to_host(compute_fstat(N_arr, M_up))

    # measured FWHM around the maximum
    i_max = int(np.argmax(F_prof))
    F_pk = float(F_prof[i_max])
    above = F_prof >= 0.5 * F_pk
    # contiguous run containing the max
    lo = i_max
    while lo > 0 and above[lo - 1]:
        lo -= 1
    hi = i_max
    while hi < len(above) - 1 and above[hi + 1]:
        hi += 1
    fwhm = (hi - lo + 1) * spacing
    snr_est = float(np.sqrt(2.0 * F_pk))
    print(f"[profile] F_peak={F_pk:.1f} (SNR~{snr_est:.1f}) at "
          f"f0={f0_nodes[i_max]:.6f} mHz; FWHM={fwhm:.3e} mHz "
          f"({fwhm / df_T:.2f}/Tobs)", flush=True)

    import matplotlib
    import matplotlib.pyplot as plt
    # A transitive import (globalfit.diagnosticplot) flips usetex back on;
    # kill it here so a missing latex install can't crash the profile plot.
    matplotlib.rcParams["text.usetex"] = False

    fig, ax = plt.subplots(figsize=(11, 4.2))
    ax.plot(f0_nodes, F_prof, "-", lw=1.0, color="C0",
            label="F-stat (best-fit sky, Mc fixed)")
    if cat_sources is not None:
        cf0 = cat_sources["f0"]
        keep = (cf0 >= f0_nodes[0]) & (cf0 <= f0_nodes[-1])
        for i, fv in enumerate(cf0[keep]):
            ax.axvline(fv, color="r", ls="--", lw=1.2,
                       label="catalogue GB" if i == 0 else None)
    # 1/Tobs width reference bar at half max
    ax.plot([f0_nodes[i_max] - 0.5 * df_T, f0_nodes[i_max] + 0.5 * df_T],
            [0.5 * F_pk] * 2, lw=3, color="k", alpha=0.7,
            label="1/Tobs reference width")
    ax.set_xlabel("f0 [mHz]")
    ax.set_ylabel("F-stat")
    ax.set_title(
        f"Peak profile: F_peak={F_pk:.0f} (SNR~{snr_est:.0f}), "
        f"FWHM={fwhm:.2e} mHz = {fwhm / df_T:.2f}/Tobs "
        f"(Tobs={Tobs / 86400:.0f} d)"
    )
    ax.legend(fontsize=8)
    fig.tight_layout()
    prof_path = out_path.replace(".png", f"_profile{rank}.png")
    fig.savefig(prof_path, dpi=140)
    plt.close(fig)
    print(f"[plot] wrote {prof_path}", flush=True)
    return f0_nodes, F_prof, dict(F_pk=F_pk, fwhm_mHz=fwhm, snr=snr_est,
                                  f0_max=float(f0_nodes[i_max]))


def _filter_cat(cat_sources, f0_lo_mHz, f0_hi_mHz):
    """Restrict a catalogue dict to sources with f0 inside [lo, hi] mHz."""
    if cat_sources is None:
        return None
    keep = ((cat_sources["f0"] >= f0_lo_mHz)
            & (cat_sources["f0"] <= f0_hi_mHz))
    if not keep.any():
        return None
    return {k: v[keep] for k, v in cat_sources.items()}


def _report(prop, src, stage):
    """Print the argmax cell + rvs/logpdf diagnostics for one stage."""
    g = _to_host(prop._logp_grid[:-1, :-1, :-1, :-1])
    i0, i1, i2, i3 = np.unravel_index(np.argmax(g), g.shape)
    axes = prop._axes
    peak = (float(axes[0][i0]), float(axes[1][i1]),
            float(axes[2][i2]), float(axes[3][i3]))
    print(f"[peak:{stage}] grid argmax logp={g.max():.3f} at "
          f"f0={peak[0]:.4f}mHz  Mc={peak[1]:.4f}  alpha={peak[2]:.4f}  "
          f"sin_delta={peak[3]:.4f}", flush=True)
    mc_eff = float(src.get("Mc_eff_Msol", src["Mc_Msol"]))
    print(f"[peak:{stage}] injection:       f0={src['f0_mHz']:.4f}mHz  "
          f"Mc_eff={mc_eff:.4f} (mass-based {src['Mc_Msol']:.4f})  "
          f"alpha={src['RA_rad']:.4f}  "
          f"sin_delta={np.sin(src['Dec_rad']):.4f}", flush=True)

    inj_sampling = np.array([[src["f0_mHz"], mc_eff, src["RA_rad"],
                              float(np.sin(src["Dec_rad"]))]])
    lp_at_injection = float(_to_host(prop.logpdf(inj_sampling))[0])
    s = prop.rvs(size=(5000,))
    lp = _to_host(prop.logpdf(s))
    s = _to_host(s)
    print(f"[diag:{stage}] logpdf @ injection: {lp_at_injection:.3f}", flush=True)
    print(f"[diag:{stage}] logpdf on samples: median={np.median(lp):.3f}  "
          f"5%={np.percentile(lp, 5):.3f}  95%={np.percentile(lp, 95):.3f}",
          flush=True)
    print(f"[diag:{stage}] sample mean: {s.mean(axis=0)}", flush=True)
    print(f"[diag:{stage}] injection:   {inj_sampling[0]}", flush=True)


def _save_grid_cache(prop, cache_path, stage):
    if not cache_path:
        return
    path = cache_path.replace(".npz", f"_{stage}.npz")
    try:
        np.savez(
            path,
            logp_grid=_to_host(prop._logp_grid),
            f0_ax=_to_host(prop._axes[0]),
            Mc_ax=_to_host(prop._axes[1]),
            alpha_ax=_to_host(prop._axes[2]),
            sin_delta_ax=_to_host(prop._axes[3]),
            log_norm=float(_to_host(prop._log_norm)),
        )
        print(f"[cache] wrote {path}", flush=True)
    except Exception as e:
        print(f"[cache] failed to write {path}: {e}", flush=True)


def plot_corner(prop, src, n_samples=20_000,
                out_path="/tmp/fstat_proposal_mojito.png", cat_sources=None,
                stage=""):
    import matplotlib
    import matplotlib.pyplot as plt

    # Kill usetex again -- some transitive import (globalfit.diagnosticplot)
    # flipped it back on during build_gb_wdm_comp_and_holder().
    matplotlib.rcParams["text.usetex"] = False
    matplotlib.rcParams["axes.formatter.use_mathtext"] = True

    names = ["f0 [mHz]", "Mc [Msol]", "alpha [rad]", "sin_delta"]
    axes = prop._axes
    # The F-stat peaks at the (f0, fdot)-implied chirp mass Mc_eff (the grid
    # maps Mc -> fdot through the monochromatic relation); the catalogue
    # mass-based Mc differs for interacting systems and is shown separately.
    mc_eff = float(src.get("Mc_eff_Msol", src["Mc_Msol"]))
    injection_mu = np.array([
        src["f0_mHz"], mc_eff, src["RA_rad"], float(np.sin(src["Dec_rad"])),
    ])

    g_cells = _to_host(prop._logp_grid[:-1, :-1, :-1, :-1])
    p_cells = np.exp(g_cells - g_cells.max())
    samples = _to_host(prop.rvs(size=(n_samples,)))
    cell_centres = [0.5 * (axes[i][:-1] + axes[i][1:]) for i in range(4)]

    # Truth values per sampling axis for every in-box catalogue source
    # (Mc_eff is nan for fdot<=0 interacting systems -- skipped per panel).
    cat_truth = None
    camp = None
    if cat_sources is not None:
        cat_truth = [cat_sources["f0"], cat_sources["Mc_eff"],
                     cat_sources["alpha"], cat_sources["sin_delta"]]
        camp = cat_sources["amp"]

    fig, ax_grid = plt.subplots(4, 4, figsize=(12, 12))
    plt.suptitle(
        f"FStatProposal4D on real mojito F-stat (ID {src['ID']}, "
        f"f0={src['f0_mHz']:.3f} mHz){' -- ' + stage if stage else ''}",
        y=0.995,
    )

    for i in range(4):
        for j in range(4):
            a = ax_grid[i, j]
            if i == j:
                marg_axes = tuple(k for k in range(4) if k != i)
                p1 = p_cells.sum(axis=marg_axes)
                p1n = p1 / p1.max()
                a.plot(cell_centres[i], p1n, "b-", lw=1.5, label="grid marginal")
                if cat_truth is not None:
                    for k in range(len(camp)):
                        tv = cat_truth[i][k]
                        if np.isfinite(tv):
                            a.axvline(
                                tv, color="0.5", lw=0.8,
                                alpha=float(min(1.0, 0.15 + 0.85 * (camp[k] / camp.max()))),
                                zorder=0,
                                label="catalogue GBs" if (i == 0 and k == 0)
                                else None,
                            )
                a.axvline(injection_mu[i], color="r", linestyle="--", lw=1.5,
                          label="mojito GB (Mc from f0,fdot)" if i == 1
                          else "mojito GB")
                if i == 1 and abs(src["Mc_Msol"] - injection_mu[1]) > 1e-4:
                    a.axvline(src["Mc_Msol"], color="orange", linestyle=":",
                              lw=1.5, label="catalogue Mc (mass-based)")
                    a.legend(fontsize=7, loc="upper right")
                counts, edges = np.histogram(samples[:, i], bins=60)
                centres = 0.5 * (edges[:-1] + edges[1:])
                a.step(centres, counts / max(1, counts.max()),
                       where="mid", color="green", lw=1.0, label="rvs samples")
                a.set_xlabel(names[i])
                a.set_ylabel("density (peak-normed)")
                a.set_xlim(cell_centres[i][0], cell_centres[i][-1])
                if i == 0:
                    a.legend(fontsize=7, loc="upper right")
            elif i > j:
                marg_axes = tuple(k for k in range(4) if k not in (i, j))
                p2 = p_cells.sum(axis=marg_axes)
                # Cells span node edges [ax[0], ax[-1]] -- using cell centres
                # here shifts the heat map by half a cell and leaves an
                # off-by-half gap strip at the axes edges.
                extent = [axes[j][0], axes[j][-1],
                          axes[i][0], axes[i][-1]]
                a.imshow(p2.T, origin="lower", extent=extent, aspect="auto",
                         cmap="viridis")
                a.scatter(samples[::200, j], samples[::200, i], s=1.5, c="w",
                          alpha=0.4)
                if cat_truth is not None:
                    a.scatter(cat_truth[j], cat_truth[i], s=45, marker="P",
                              c="w", edgecolors="k", linewidths=0.7, zorder=5,
                              label=None)
                a.plot(injection_mu[j], injection_mu[i], "r*", markersize=12,
                       mec="k", mew=0.5, zorder=6)
                a.set_xlabel(names[j])
                a.set_ylabel(names[i])
            else:
                a.axis("off")

    fig.subplots_adjust(left=0.07, right=0.98, top=0.94, bottom=0.06,
                        wspace=0.35, hspace=0.35)
    plt.savefig(out_path, dpi=140)
    plt.close(fig)
    print(f"[plot] wrote {out_path}", flush=True)


def _resolve_reference_src(cat_sources, gb_info):
    """Diagnostic reference = the loudest IN-BAND catalogue source.

    The reference is always resolved from the catalogue (no hardcoded
    per-source dicts). Guards: empty catalogue -> a minimal band-centre
    reference; loudest source with ``fdot <= 0`` (nan ``Mc_eff``,
    interacting DWD) -> substitute ``mean(m_chirp_lims)``.
    """
    mc_lims = list(getattr(gb_info, "m_chirp_lims", None) or [0.001, 1.0])
    mc_mean = 0.5 * (float(mc_lims[0]) + float(mc_lims[-1]))
    f0_lo = float(gb_info.f0_lims[0]) * 1e3
    f0_hi = float(gb_info.f0_lims[-1]) * 1e3
    if cat_sources is not None and len(cat_sources["f0"]):
        i = int(np.argmax(cat_sources["amp"]))
        mc = float(cat_sources["Mc_eff"][i])
        if not np.isfinite(mc):
            mc = mc_mean
        src = {
            "ID": "band-loudest",
            "f0_mHz": float(cat_sources["f0"][i]),
            "Mc_eff_Msol": mc, "Mc_Msol": mc,
            "RA_rad": float(cat_sources["alpha"][i]),
            "Dec_rad": float(np.arcsin(cat_sources["sin_delta"][i])),
        }
        print(f"[main] diagnostics reference the loudest in-band catalogue "
              f"GB: f0={src['f0_mHz']:.4f} mHz (Mc_eff={mc:.4f}).", flush=True)
        return src
    print("[main] no catalogue sources in band; diagnostics use a minimal "
          "band-centre reference.", flush=True)
    return {"ID": "band-centre", "f0_mHz": 0.5 * (f0_lo + f0_hi),
            "Mc_eff_Msol": mc_mean, "Mc_Msol": mc_mean,
            "RA_rad": 0.0, "Dec_rad": 0.0}


def run_stacked_peak_sweep(gb_wdm_comp, wdm_holder, f0_los, f0_dxs, mc_ax,
                           alpha_ax, sd_ax, node_shape, ckpt=None):
    """ONE chunked kernel stream over EVERY peak box of EVERY sub-band.

    ``node_shape`` is ``(K, n_f0, n_Mc, n_alpha, n_sd)``. Rows are assembled
    per chunk from index arithmetic (so the full ``n_total x 9`` array never
    materializes); the kernel sees a single ``FSTAT_BATCH``-chunked stream --
    never a per-peak loop. Returns the node-shaped ``F`` array on the
    compute backend's module.

    ``ckpt``: optional progress-file base path; the sweep then resumes from
    the last saved row on rerun. The fingerprint hashes the box axes, so a
    changed peak selection or grid-density knob restarts cleanly.
    """
    from gbgpu.utils.utility import get_fdot

    from lisatools.sampling.fstat_proposal import compute_fstat, fstat_knob

    xp = gb_wdm_comp.xp
    n_total = int(np.prod(node_shape))
    batch = fstat_knob("FSTAT_BATCH", int)
    fp = (_ckpt_fingerprint(np.asarray(f0_los), np.asarray(f0_dxs),
                            np.asarray(mc_ax), np.asarray(alpha_ax),
                            np.asarray(sd_ax), extra=str(node_shape))
          if ckpt else "")
    f0_los_d = xp.asarray(f0_los)
    f0_dxs_d = xp.asarray(f0_dxs)
    mc_d = xp.asarray(mc_ax)
    al_d = xp.asarray(alpha_ax)
    sd_d = xp.asarray(sd_ax)
    F_flat = xp.empty(n_total, dtype=xp.float64)
    F_prev, start = _ckpt_load(ckpt, n_total, fp)
    if F_prev is not None and start > 0:
        F_flat[:start] = xp.asarray(F_prev[:start])
    t0 = time.time()
    last = t0
    last_ckpt = t0
    for s in range(start, n_total, batch):
        e = min(s + batch, n_total)
        k, i0, i1, i2, i3 = xp.unravel_index(xp.arange(s, e), node_shape)
        pr = xp.zeros((e - s, 9), dtype=xp.float64)
        pr[:, 0] = 1e-22
        pr[:, 1] = (f0_los_d[k] + i0 * f0_dxs_d[k]) * 1e-3
        pr[:, 2] = xp.asarray(get_fdot(f=pr[:, 1], Mc=mc_d[i1]))
        pr[:, 5] = 0.5 * np.pi
        pr[:, 7] = al_d[i2]
        pr[:, 8] = xp.arcsin(sd_d[i3])
        N_arr, M_up = gb_wdm_comp.get_fstat_ll_wdm(pr, wdm_holder)
        F_flat[s:e] = compute_fstat(xp.asarray(N_arr), xp.asarray(M_up))
        if time.time() - last > 30:
            last = time.time()
            print(f"[stageB] {e}/{n_total} evals ({time.time() - t0:.0f}s)",
                  flush=True)
        if ckpt and e < n_total and time.time() - last_ckpt > _ckpt_secs():
            last_ckpt = time.time()
            _ckpt_save(ckpt, _to_host(F_flat[:e]), e, n_total, fp)
    if ckpt and start < n_total:
        _ckpt_save(ckpt, _to_host(F_flat), n_total, n_total, fp)
    print(f"[stageB] {n_total} evals in one chunked stream (batch={batch}, "
          f"{time.time() - t0:.0f}s"
          f"{f', resumed at {start}' if start else ''})", flush=True)
    return F_flat.reshape(node_shape)


def run_stacked_stage_b(gb_wdm_comp, wdm_holder, curr, gb_info, peaks, src,
                        out_path, cat_sources, cache_path):
    """Stage B: clamped boxes -> ONE batched sweep -> stacked grids.

    Assemble (host, cheap): every selected peak's 4-D box with its f0 range
    CLAMPED to its sub-band edges (unclamped boxes would propose f0 the
    consumer's per-cell gate rejects -- wasted birth attempts). Sweep: one
    chunked kernel stream over all boxes (:func:`run_stacked_peak_sweep`).
    Build: ONE :class:`StackedFStatProposal4D` from the ``(K, ...)`` stack
    -- the grids ARE the production sampling layer. Node shape: f0 auto
    ~1/Tobs cells (:func:`fstat_n_f0`), Mc/sky anisotropic-coarse
    (:func:`fstat_n_axis`). Optionally (``FSTAT_FIT_GMM=1``) fit the
    batched per-box GMM FROM the grids as the memory-light alternative
    sampling layer. Save: one ``*_peaks_stacked.npz`` (grids first, GMM
    appended -- a GMM failure never loses the sweep).
    """
    from lisatools.sampling.fstat_proposal import (
        FStatProposal4D,
        StackedFStatProposal4D,
        fit_gmm_to_stacked,
        fstat_knob,
        fstat_n_axis,
        fstat_n_f0,
        pack_gmm_components,
    )

    if len(peaks) == 0:
        print("[stageB] no peaks selected; nothing to fit.", flush=True)
        return None

    half_f0 = fstat_knob("FSTAT_PEAK_HALF_MHZ", float)
    n_f0 = fstat_n_f0(2.0 * half_f0, float(curr.general_info.Tobs))
    n_Mc = fstat_n_axis("FSTAT_N_MC")
    n_alpha = fstat_n_axis("FSTAT_N_ALPHA")
    n_sd = fstat_n_axis("FSTAT_N_SINDELTA")
    n_fit_env = os.environ.get("FSTAT_PEAKS_TO_FIT", "").strip()
    if n_fit_env:
        peaks = peaks[: int(n_fit_env)]
        print(f"[stageB] FSTAT_PEAKS_TO_FIT caps the fit to the top "
              f"{len(peaks)} peaks (of the full selection).", flush=True)

    # Clamped per-box f0 axes (assembly bookkeeping).
    band_edges_mHz = np.asarray(gb_info.band_edges, dtype=float) * 1e3
    f0_los, f0_dxs, keep = [], [], []
    for row in peaks:
        f0p, _F, _node, bi = float(row[0]), float(row[1]), row[2], int(row[3])
        lo = max(f0p - half_f0, band_edges_mHz[bi])
        hi = min(f0p + half_f0, band_edges_mHz[bi + 1])
        if not hi - lo > 0:
            print(f"[stageB] WARNING: degenerate clamped box at "
                  f"f0={f0p:.5f} mHz (band {bi}); skipped.", flush=True)
            continue
        f0_los.append(lo)
        f0_dxs.append((hi - lo) / (n_f0 - 1))
        keep.append(row)
    peaks = np.asarray(keep)
    K = len(peaks)
    f0_los = np.asarray(f0_los)
    f0_dxs = np.asarray(f0_dxs)
    band_idx = peaks[:, 3].astype(int)

    mc_range = _mc_grid_range(gb_info)
    mc_ax = np.linspace(mc_range[0], mc_range[1], n_Mc)
    alpha_ax = np.linspace(0.0, 2 * np.pi, n_alpha)
    sd_ax = np.linspace(-1.0, 1.0, n_sd)
    node_shape = (K, n_f0, n_Mc, n_alpha, n_sd)
    print(f"[stageB] {K} peak boxes x {n_f0}x{n_Mc}x{n_alpha}x{n_sd} nodes; "
          f"Mc box {mc_range}; f0 boxes clamped to their sub-bands.",
          flush=True)

    _parts = cache_path.replace(".npz", "_parts") if cache_path else None
    logp_grids = run_stacked_peak_sweep(
        gb_wdm_comp, wdm_holder, f0_los, f0_dxs, mc_ax, alpha_ax, sd_ax,
        node_shape,
        ckpt=os.path.join(_parts, "stageb") if _parts else None,
    )  # beta = 1: logp = F

    mem_mb = os.environ.get("FSTAT_GRID_MEM_MB", "").strip()
    stacked = StackedFStatProposal4D(
        logp_grids, f0_los, f0_dxs, mc_ax, alpha_ax, sd_ax,
        weights=np.clip(peaks[:, 1], 0.0, None),  # report-time w ~ F
        mem_budget_mb=float(mem_mb) if mem_mb else None,
    )

    # Save the grids FIRST -- the GMM fit must never lose the kernel sweep.
    stacked_path = None
    save_fields = None
    if cache_path:
        stacked_path = cache_path.replace(".npz", "_peaks_stacked.npz")
        save_fields = dict(
            logp_grids=_to_host(logp_grids), f0_los=f0_los, f0_dxs=f0_dxs,
            mc_ax=mc_ax, alpha_ax=alpha_ax, sin_delta_ax=sd_ax,
            peak_f0_mHz=peaks[:, 0], peak_F=peaks[:, 1], band_idx=band_idx,
            band_f0_lo=band_edges_mHz[band_idx],
            band_f0_hi=band_edges_mHz[band_idx + 1],
            band_edges=np.asarray(gb_info.band_edges, dtype=float),
        )
        np.savez(stacked_path, **save_fields)
        print(f"[cache] wrote {stacked_path} (grids; GMM appended next)",
              flush=True)
        # Stacked npz is now the durable artifact for stage B.
        _ckpt_clear(cache_path.replace(".npz", "_parts"), "stageb")
        if os.environ.get("FSTAT_SAVE_PER_BOX", "0") == "1":
            for i in range(K):
                f0_ax_i = f0_los[i] + f0_dxs[i] * np.arange(n_f0)
                np.savez(cache_path.replace(".npz", f"_peak{i}.npz"),
                         logp_grid=_to_host(logp_grids[i]), f0_ax=f0_ax_i,
                         Mc_ax=mc_ax, alpha_ax=alpha_ax, sin_delta_ax=sd_ax,
                         log_norm=0.0)
            print(f"[cache] wrote {K} legacy per-box *_peak<i>.npz caches",
                  flush=True)

    # OPT-IN (FSTAT_FIT_GMM=1): batched per-box GMM fit FROM the stacked
    # grids (reuses vec_fit_gmm_min_bic) -- the memory-light alternative
    # sampling layer. The grids are the production layer; the runner's
    # FSTAT_PEAK_SAMPLING=gmm mode fits on load when the cache carries no
    # components, so skipping here loses nothing.
    if fstat_knob("FSTAT_FIT_GMM", int):
        gpus = getattr(curr.general_info, "gpus", None)
        try:
            t0 = time.time()
            comps = fit_gmm_to_stacked(
                stacked,
                n_samples_per_box=fstat_knob("FSTAT_GMM_SAMPLES", int),
                gpu=(gpus[0] if gpus else None),
                max_comp=fstat_knob("FSTAT_GMM_MAX_COMP", int),
            )
            n_comp = [len(w) for w in comps[0]]
            print(f"[gmm] fitted {K} boxes in {time.time() - t0:.1f}s "
                  f"(components/box: min {min(n_comp)}, max {max(n_comp)})",
                  flush=True)
            if stacked_path:
                save_fields.update(pack_gmm_components(comps))
                np.savez(stacked_path, **save_fields)
                print(f"[cache] re-wrote {stacked_path} with GMM components",
                      flush=True)
        except Exception as e:
            print(f"[gmm] fit FAILED (grids still cached; the runner fits "
                  f"on load): {e!r}", flush=True)
    else:
        print("[gmm] prep-time GMM fit skipped (opt in with FSTAT_FIT_GMM=1;"
              " the grid stack is the production sampling layer).",
              flush=True)

    # Reports + corner plots for the top boxes (cosmetic, non-fatal;
    # from_grid rebuilds are plotting-only -- no kernel work).
    n_plot = fstat_knob("FSTAT_PLOT_PEAKS", int)
    for i in range(min(n_plot, K)):
        stage = f"peak{i}"
        try:
            f0_ax_i = f0_los[i] + f0_dxs[i] * np.arange(n_f0)
            prop_i = FStatProposal4D.from_grid(
                (f0_ax_i, mc_ax, alpha_ax, sd_ax), _to_host(logp_grids[i])
            )
            _report(prop_i, src, stage)
            pk_cat = _filter_cat(cat_sources, f0_ax_i[0], f0_ax_i[-1])
            plot_corner(prop_i, src, n_samples=20_000,
                        out_path=out_path.replace(".png", f"_{stage}.png"),
                        cat_sources=pk_cat,
                        stage=f"local proposal @ peak {i} "
                              f"(f0={peaks[i, 0]:.5f} mHz, "
                              f"F={peaks[i, 1]:.1f}, band {int(peaks[i, 3])})")
        except Exception as e:
            print(f"[{stage}] report/plot skipped (non-fatal, grids "
                  f"written): {e!r}", flush=True)
    return stacked


def main():
    _resolve_matplotlib()

    n_per_axis = int(os.environ.get("FSTAT_N_PER_AXIS", 24))
    out_path = os.environ.get("FSTAT_OUT", "/tmp/fstat_proposal_mojito.png")

    gb_wdm_comp, wdm_holder, curr = build_gb_wdm_comp_and_holder()
    gb_info = curr.source_info["gb"]
    center_freq_hz = float(gb_info.center_freq)

    _maybe_patch_slab_kernel_args(gb_wdm_comp, wdm_holder, center_freq_hz)

    # Time a small probe batch to project the full sweep cost.
    probe_n = 256
    rng = np.random.default_rng(0)
    probe = np.zeros((probe_n, 9))
    probe[:, 0] = 1e-22
    probe[:, 1] = center_freq_hz * (1 + 1e-3 * rng.standard_normal(probe_n))
    probe[:, 5] = 0.5 * np.pi
    probe[:, 7] = rng.uniform(0, 2 * np.pi, probe_n)
    probe[:, 8] = rng.uniform(-1, 1, probe_n)
    probe = gb_wdm_comp.xp.asarray(probe)   # -> device on the CUDA backends
    t0 = time.time()
    gb_wdm_comp.get_fstat_ll_wdm(probe, wdm_holder)
    per_eval = (time.time() - t0) / probe_n
    print(f"[probe] F-stat: {per_eval * 1e3:.3f} ms/eval", flush=True)

    # Optional cache base for the comb + stacked grids. Enable via
    # FSTAT_GRID_CACHE=/path/to/grid.npz.
    cache_path = os.environ.get("FSTAT_GRID_CACHE", "").strip()
    cat_sources = _band_catalogue_sources(
        float(gb_info.f0_lims[0]) * 1e3, float(gb_info.f0_lims[-1]) * 1e3
    )
    if cat_sources is not None:
        print(f"[cat] {len(cat_sources['f0'])} catalogue GBs in the analysis "
              "band", flush=True)
    src = _resolve_reference_src(cat_sources, gb_info)

    design = os.environ.get("FSTAT_DESIGN", "comb").strip().lower()

    if design == "comb":
        # --- Stage A: dense-in-f0 comb scan (the narrow-peak-correct locate) ---
        gi = curr.general_info
        comb_cache_file = (cache_path.replace(".npz", "_comb.npz")
                           if cache_path else "")
        if (os.environ.get("FSTAT_COMB_CACHE_REUSE", "0") == "1"
                and comb_cache_file and os.path.exists(comb_cache_file)):
            # Reuse a previous sweep; peak selection is RE-RUN with the
            # current knobs from the cached F_max, so old caches (any peaks
            # format, Nx2 or Nx3) feed the new per-sub-band pipeline.
            d = np.load(comb_cache_file)
            f0_nodes, F_max = d["f0_nodes_mHz"], d["F_max"]
            spacing = float(f0_nodes[1] - f0_nodes[0])
            extras = dict(sky_alpha=d["sky_alpha"],
                          sky_sin_delta=d["sky_sin_delta"], F_all=d["F_all"],
                          best_alpha=d["best_alpha"] if "best_alpha" in d else None,
                          best_sin_delta=d["best_sin_delta"] if "best_sin_delta" in d else None,
                          mc_fix=_comb_mc_fix(gb_info), spacing=spacing)
            peaks = select_comb_peaks(f0_nodes, F_max, gb_info, spacing,
                                      gb_wdm_comp.xp)
            print(f"[comb] reused cache {comb_cache_file} ({len(f0_nodes)} "
                  f"nodes; selection re-run -> {len(peaks)} peaks)",
                  flush=True)
        else:
            f0_nodes, F_max, peaks, extras = run_comb_scan(
                gb_wdm_comp, wdm_holder, gb_info, gi, src, out_path,
                cat_sources, cache_path=cache_path,
            )

        # Ultra-dense 1-D profile through the top peak: measures the actual
        # peak FWHM against the ~1/Tobs matched-filter prediction. (A single
        # already-chunked stream -- consistent with the batching invariant.)
        if len(peaks) and os.environ.get("FSTAT_PEAK_PROFILE", "1") not in (
                "0", "false", "False"):
            try:
                run_peak_profile(gb_wdm_comp, wdm_holder, gi, src,
                                 float(peaks[0][0]), extras, out_path,
                                 cat_sources, rank=0)
            except Exception as e:
                print(f"[profile] plot skipped (non-fatal, grids still "
                      f"written): {e!r}", flush=True)

        # --- Stage B: stacked local grids over ALL selected peaks ---
        run_stacked_stage_b(gb_wdm_comp, wdm_holder, curr, gb_info, peaks,
                            src, out_path, cat_sources, cache_path)
        return

    # --- Legacy coarse design (kept as the narrow-peak negative control) ---
    prop = build_fstat_proposal(gb_wdm_comp, wdm_holder, src, gb_info,
                                 n_per_axis=n_per_axis, stage="stage1")
    _save_grid_cache(prop, cache_path, "stage1")
    _report(prop, src, "stage1")
    plot_corner(prop, src, n_samples=20_000,
                out_path=out_path.replace(".png", "_stage1.png"),
                cat_sources=cat_sources, stage="stage 1 (band)")

    if os.environ.get("FSTAT_ZOOM", "1") not in ("0", "false", "False"):
        box = _zoom_box(prop, n_cells=2.0)
        prop_zoom = build_fstat_proposal(gb_wdm_comp, wdm_holder, src, gb_info,
                                          n_per_axis=n_per_axis, box=box,
                                          stage="zoom")
        _save_grid_cache(prop_zoom, cache_path, "zoom")
        _report(prop_zoom, src, "zoom")
        zoom_cat = _filter_cat(cat_sources, *box["f0_range"])
        plot_corner(prop_zoom, src, n_samples=20_000,
                    out_path=out_path.replace(".png", "_zoom.png"),
                    cat_sources=zoom_cat, stage="stage 2 (zoom)")


if __name__ == "__main__":
    main()
