"""Real-F-stat visualization: FStatProposal4D against ``get_fstat_ll_wdm``
on a mojito WDM residual centred on the highest-frequency GB.

Reuses the ``erebor.gb_no_fg`` stock recipe to stand up the full WDM stack
(orbits, TDIConfig, WDMSettings, chunked-het ``GBWDMComputations``, and the
``AnalysisContainerArray`` from the mojito residual), then invokes the
F-stat proposal on a grid around the source.

The mojito catalogue's highest-freq GB (from the L1 catalogue) is::

    ID 7725228, f0 = 20.380 mHz, Mc = 0.519 Msol,
    RA = 4.062 rad, Dec = -0.905 rad, fdot = 1.03e-13

The data contains ONLY the GB injection (no noise realization); the fixed
whitening PSD is the *tabulated empirical* estimate from the mojito NOISE
brick (``MojitoNoiseEstimates``, extras-only fixed-PSD path).

Environment knobs::

    FSTAT_TARGET       ("highest" [default] or "band75" = ~7.5 mHz band)
    FSTAT_DESIGN       ("comb" [default]: dense-f0 scan + local peak fits;
                        "coarse": legacy 4-D tensor grid + zoom -- kept as
                        the narrow-peak negative control, it cannot resolve
                        ~1/Tobs peaks)
    FSTAT_F0_SPACING_MHZ / FSTAT_COMB_NSKY / FSTAT_COMB_MC / FSTAT_TOP_K
                       (comb scan: f0 node spacing [default 1/(2*Tobs)],
                        sky-point count [6], fixed Mc [target's Mc_eff],
                        reported peak count [10])
    FSTAT_PEAKS_TO_FIT / FSTAT_PEAK_HALF_MHZ
                       (local 4-D proposals per comb peak [1]; f0 half-width
                        of the peak box [2.5e-3 mHz ~ Doppler envelope])
    GB_CENTER_FREQ_HZ  (default = target's f0; band75 keeps the stock band)
    GB_N_LAYERS        (default 12 -- WDM layer count in the analysis band)
    FSTAT_N_PER_AXIS   (default 24; FSTAT_N_F0/_MC/_ALPHA/_SINDELTA override
                        per axis)
    FSTAT_GRID_CACHE   (optional .npz path to dump the swept F-stat grids)
    FSTAT_OUT          (default /tmp/fstat_proposal_mojito_<target>.png)
    MOJITO_DATA_PATH   (default ~/.mojito_cache/brickmarket/mojito_light_v1_0_0/)

Output: corner plot at ``FSTAT_OUT``.
"""

from __future__ import annotations

import os
import sys
import time

import numpy as np

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

# Mojito's highest-frequency GB (from the L1 catalogue, verified interactively).
#
# NOTE on the two chirp masses: the wdwd catalogue is an *interacting* DWD
# population, so the injected fdot is not purely GW-driven. ``Mc_Msol`` is the
# catalogue's mass-based chirp mass; ``Mc_eff_Msol`` is the chirp mass implied
# by the injected ``(f0, fdot)`` through the monochromatic-GB relation -- and
# since the proposal grid maps Mc -> fdot through exactly that relation, the
# F-stat must peak at ``Mc_eff``, not at the mass-based value.
MOJITO_HIGHEST_GB = {
    "ID": 7725228,
    "f0_mHz": 20.380377,
    "Mc_Msol": 0.5192,       # catalogue mass-based (tides suppress fdot)
    "Mc_eff_Msol": 0.4658,   # from (f0, fdot): get_chirp_mass_from_f_fdot
    "RA_rad": 4.0617,
    "Dec_rad": -0.9049,
    "fdot": 1.0245e-13,
    "A": 1.371e-22,
}

# Loudest GB inside the stock gb_no_fg band [7.36, 7.78] mHz (~30 catalogue
# sources live there; ID 14399620 at 7.3545 mHz is louder but sits just below
# the band edge). Select with FSTAT_TARGET=band75; the band75 run keeps the
# variant's default analysis band instead of recentring on the source.
MOJITO_BAND75_GB = {
    "ID": 1229636,
    "f0_mHz": 7.580260,
    "Mc_Msol": 0.3356,
    "Mc_eff_Msol": 0.3355,
    "RA_rad": 4.9791,
    "Dec_rad": -0.0631,
    "fdot": 1.578e-15,
    "A": 9.072e-23,
}


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


def build_gb_wdm_comp_and_holder(center_freq_hz: float, n_layers: int):
    """Configure ``erebor.gb_no_fg`` on mojito with the GB analysis band centred
    on ``center_freq_hz`` (``n_layers`` layers wide), build it, run
    ``load_info`` + ``setup_acs`` to get the ``AnalysisContainerArray``, then
    manually construct ``GBWDMComputations`` from the pre-built settings.

    Returns ``(gb_wdm_comp, wdm_holder, curr)`` where ``wdm_holder`` is the
    residual ``AnalysisContainerArray`` (what ``get_fstat_ll_wdm`` consumes).
    """
    from mpi4py import MPI
    from eryn.state import BranchSupplemental

    from lisatools.globalfit.run import GlobalFit
    from lisatools.globalfit.stock import erebor

    print("[build] configuring erebor.gb_no_fg...", flush=True)
    fit = erebor.gb_no_fg(nwalkers=2, ntemps=1)
    if center_freq_hz is not None:
        fit.gb.center_freq = float(center_freq_hz)
        fit.gb.n_layers = int(n_layers)
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
            Mc_range=(0.1, 1.0),
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
    g = np.asarray(prop._logp_grid[:-1, :-1, :-1, :-1])
    idx = np.unravel_index(np.argmax(g), g.shape)
    names = ("f0_range", "Mc_range", "alpha_range", "sin_delta_range")
    box = {}
    for j, name in enumerate(names):
        ax = np.asarray(prop._axes[j])
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


def run_comb_scan(gb_wdm_comp, wdm_holder, gb_info, general_info, src,
                  out_path, cat_sources, cache_path=None):
    """Dense-in-f0 F-stat comb scan across the sub-band.

    With months of data the F-stat f0 peaks are ~1/Tobs wide (~1e-4 mHz) --
    far too narrow for any feasible 4-D tensor grid to land on (the plan's
    "narrow-peak" regime). But over the same stretch ``fdot*Tobs << 1/Tobs``,
    so Mc is nearly unmeasurable per band and can be held FIXED, and sky only
    enters through the Doppler ridge (peak shifts up to ~f0*v/c ~ 2e-3 mHz).
    The right scan is therefore dense in f0 (spacing ~ 1/(2*Tobs)) x a small
    spread of sky points, maximized over sky per f0 node.

    Returns ``(f0_nodes_mHz, F_max, peaks)`` where ``peaks`` is the
    greedily-separated list of top local maxima ``(f0_mHz, F)``.
    """
    from gbgpu.utils.utility import get_fdot

    from lisatools.sampling.fstat_proposal import compute_fstat

    Tobs = float(general_info.Tobs)
    f0_lo = float(gb_info.f0_lims[0]) * 1e3
    f0_hi = float(gb_info.f0_lims[-1]) * 1e3
    spacing = float(os.environ.get("FSTAT_F0_SPACING_MHZ", 0.5 / Tobs * 1e3))
    f0_nodes = np.arange(f0_lo, f0_hi + 0.5 * spacing, spacing)
    n_sky = int(os.environ.get("FSTAT_COMB_NSKY", 6))
    # golden-ratio spread over the sphere
    ks = np.arange(n_sky)
    sky_sd = -1.0 + 2.0 * (ks + 0.5) / n_sky
    sky_al = (2.0 * np.pi * ks * 0.6180339887) % (2.0 * np.pi)
    mc_fix = float(os.environ.get(
        "FSTAT_COMB_MC", src.get("Mc_eff_Msol", src.get("Mc_Msol", 0.3))
    ))

    print(f"[comb] {len(f0_nodes)} f0 nodes x {n_sky} sky points = "
          f"{len(f0_nodes) * n_sky} evals  (spacing {spacing:.3e} mHz = "
          f"{spacing / (1e3 / Tobs):.2f}/Tobs; Mc fixed at {mc_fix:.3f})",
          flush=True)

    F = np.zeros((n_sky, len(f0_nodes)))
    t0 = time.time()
    for k in range(n_sky):
        params = np.zeros((len(f0_nodes), 9))
        params[:, 0] = 1e-22
        params[:, 1] = f0_nodes * 1e-3
        params[:, 2] = get_fdot(
            f=params[:, 1], Mc=np.full(len(f0_nodes), mc_fix)
        )
        params[:, 5] = 0.5 * np.pi
        params[:, 7] = sky_al[k]
        params[:, 8] = np.arcsin(sky_sd[k])
        for s in range(0, len(f0_nodes), 4096):
            e = min(s + 4096, len(f0_nodes))
            N_arr, M_up = gb_wdm_comp.get_fstat_ll_wdm(params[s:e], wdm_holder)
            F[k, s:e] = np.asarray(
                compute_fstat(np.asarray(N_arr), np.asarray(M_up))
            )
        print(f"[comb] sky {k + 1}/{n_sky} done ({time.time() - t0:.0f}s)",
              flush=True)
    F_max = F.max(axis=0)

    # Greedy top-K local maxima with a Doppler-envelope minimum separation.
    min_sep = max(1, int(round(3e-3 / spacing)))
    order = np.argsort(F_max)[::-1]
    top_k = int(os.environ.get("FSTAT_TOP_K", 10))
    peaks = []
    for idx in order:
        if len(peaks) >= top_k:
            break
        if all(abs(int(idx) - p[2]) >= min_sep for p in peaks):
            peaks.append((float(f0_nodes[idx]), float(F_max[idx]), int(idx)))
    print("[comb] top peaks (f0 [mHz], F):", flush=True)
    for f0p, Fp, _ in peaks:
        print(f"[comb]   {f0p:.5f}  {Fp:10.2f}", flush=True)

    # Persist the sweep BEFORE any plotting -- a cosmetic figure failure must
    # never lose ~half an hour of kernel work.
    if cache_path:
        comb_cache = cache_path.replace(".npz", "_comb.npz")
        np.savez(comb_cache, f0_nodes_mHz=f0_nodes, F_max=F_max,
                 peaks=np.array([(p[0], p[1]) for p in peaks]),
                 F_all=F, sky_alpha=sky_al, sky_sin_delta=sky_sd)
        print(f"[cache] wrote {comb_cache}", flush=True)

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

    extras = dict(sky_alpha=sky_al, sky_sin_delta=sky_sd, F_all=F,
                  mc_fix=mc_fix, spacing=spacing)
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
    k_best = int(np.argmax(F_all[:, int(np.argmax(F_all.max(axis=0)))]))
    alpha_best = float(comb_extras["sky_alpha"][k_best])
    sd_best = float(comb_extras["sky_sin_delta"][k_best])
    mc_fix = float(comb_extras["mc_fix"])
    print(f"[profile] peak {rank}: {len(f0_nodes)} nodes, spacing "
          f"{spacing:.2e} mHz (= 1/(10 Tobs)), sky (alpha={alpha_best:.3f}, "
          f"sin_delta={sd_best:.3f}), Mc={mc_fix:.3f}", flush=True)

    params = np.zeros((len(f0_nodes), 9))
    params[:, 0] = 1e-22
    params[:, 1] = f0_nodes * 1e-3
    params[:, 2] = get_fdot(f=params[:, 1], Mc=np.full(len(f0_nodes), mc_fix))
    params[:, 5] = 0.5 * np.pi
    params[:, 7] = alpha_best
    params[:, 8] = np.arcsin(sd_best)
    F_prof = np.zeros(len(f0_nodes))
    for s in range(0, len(f0_nodes), 4096):
        e = min(s + 4096, len(f0_nodes))
        N_arr, M_up = gb_wdm_comp.get_fstat_ll_wdm(params[s:e], wdm_holder)
        F_prof[s:e] = np.asarray(
            compute_fstat(np.asarray(N_arr), np.asarray(M_up))
        )

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
    g = np.asarray(prop._logp_grid[:-1, :-1, :-1, :-1])
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
    lp_at_injection = float(np.asarray(prop.logpdf(inj_sampling))[0])
    s = prop.rvs(size=(5000,))
    lp = np.asarray(prop.logpdf(s))
    print(f"[diag:{stage}] logpdf @ injection: {lp_at_injection:.3f}", flush=True)
    print(f"[diag:{stage}] logpdf on samples: median={np.median(lp):.3f}  "
          f"5%={np.percentile(lp, 5):.3f}  95%={np.percentile(lp, 95):.3f}",
          flush=True)
    print(f"[diag:{stage}] sample mean: {np.asarray(s).mean(axis=0)}", flush=True)
    print(f"[diag:{stage}] injection:   {inj_sampling[0]}", flush=True)


def _save_grid_cache(prop, cache_path, stage):
    if not cache_path:
        return
    path = cache_path.replace(".npz", f"_{stage}.npz")
    try:
        np.savez(
            path,
            logp_grid=np.asarray(prop._logp_grid),
            f0_ax=np.asarray(prop._axes[0]),
            Mc_ax=np.asarray(prop._axes[1]),
            alpha_ax=np.asarray(prop._axes[2]),
            sin_delta_ax=np.asarray(prop._axes[3]),
            log_norm=float(prop._log_norm),
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

    g_cells = np.asarray(prop._logp_grid[:-1, :-1, :-1, :-1])
    p_cells = np.exp(g_cells - g_cells.max())
    samples = np.asarray(prop.rvs(size=(n_samples,)))
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


def main():
    _resolve_matplotlib()

    target = os.environ.get("FSTAT_TARGET", "highest").strip().lower()
    src = {"highest": MOJITO_HIGHEST_GB, "band75": MOJITO_BAND75_GB}[target]
    # band75 keeps the variant's stock analysis band ([7.36, 7.78] mHz);
    # other targets recentre a 12-layer band on the source.
    if "GB_CENTER_FREQ_HZ" in os.environ:
        center_freq_hz = float(os.environ["GB_CENTER_FREQ_HZ"])
    elif target == "band75":
        center_freq_hz = None
    else:
        center_freq_hz = src["f0_mHz"] * 1e-3
    n_layers = int(os.environ.get("GB_N_LAYERS", 12))
    n_per_axis = int(os.environ.get("FSTAT_N_PER_AXIS", 24))
    out_path = os.environ.get(
        "FSTAT_OUT", f"/tmp/fstat_proposal_mojito_{target}.png"
    )

    print(f"[main] Target ({target}): mojito GB ID {src['ID']}, "
          f"f0={src['f0_mHz']:.4f} mHz", flush=True)
    if center_freq_hz is None:
        print("[main] Analysis band: variant default (min_freq/max_freq)",
              flush=True)
    else:
        print(f"[main] Analysis band centre: {center_freq_hz * 1e3:.4f} mHz "
              f"({n_layers} layers)", flush=True)

    gb_wdm_comp, wdm_holder, curr = build_gb_wdm_comp_and_holder(
        center_freq_hz, n_layers,
    )
    gb_info = curr.source_info["gb"]
    if center_freq_hz is None:
        center_freq_hz = float(src["f0_mHz"]) * 1e-3

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
    t0 = time.time()
    gb_wdm_comp.get_fstat_ll_wdm(probe, wdm_holder)
    per_eval = (time.time() - t0) / probe_n
    print(f"[probe] F-stat: {per_eval * 1e3:.3f} ms/eval -> full "
          f"{n_per_axis}^4 sweep ~ {per_eval * n_per_axis**4 / 60:.1f} min",
          flush=True)

    # Optional cache for the F-stat grids (suffixes _stage1/_zoom). Enable
    # via FSTAT_GRID_CACHE=/path/to/grid.npz.
    cache_path = os.environ.get("FSTAT_GRID_CACHE", "").strip()
    cat_sources = _band_catalogue_sources(
        float(gb_info.f0_lims[0]) * 1e3, float(gb_info.f0_lims[-1]) * 1e3
    )
    if cat_sources is not None:
        print(f"[cat] {len(cat_sources['f0'])} catalogue GBs in the analysis "
              "band", flush=True)

    design = os.environ.get("FSTAT_DESIGN", "comb").strip().lower()

    if design == "comb":
        # --- Stage A: dense-in-f0 comb scan (the narrow-peak-correct locate) ---
        gi = curr.general_info
        comb_cache_file = (cache_path.replace(".npz", "_comb.npz")
                           if cache_path else "")
        if (os.environ.get("FSTAT_COMB_CACHE_REUSE", "0") == "1"
                and comb_cache_file and os.path.exists(comb_cache_file)):
            # Reuse a previous sweep (e.g. to fit more peaks without paying
            # the ~35-min comb again).
            d = np.load(comb_cache_file)
            f0_nodes, F_max = d["f0_nodes_mHz"], d["F_max"]
            extras = dict(sky_alpha=d["sky_alpha"],
                          sky_sin_delta=d["sky_sin_delta"], F_all=d["F_all"],
                          mc_fix=float(os.environ.get(
                              "FSTAT_COMB_MC",
                              src.get("Mc_eff_Msol", src.get("Mc_Msol", 0.3)))),
                          spacing=float(f0_nodes[1] - f0_nodes[0]))
            peaks = [(float(f0), float(F), 0) for f0, F in d["peaks"]]
            print(f"[comb] reused cache {comb_cache_file} "
                  f"({len(f0_nodes)} nodes, {len(peaks)} peaks)", flush=True)
        else:
            f0_nodes, F_max, peaks, extras = run_comb_scan(
                gb_wdm_comp, wdm_holder, gb_info, gi, src, out_path,
                cat_sources, cache_path=cache_path,
            )

        # Ultra-dense 1-D profile through the top peak: measures the actual
        # peak FWHM against the ~1/Tobs matched-filter prediction.
        if peaks and os.environ.get("FSTAT_PEAK_PROFILE", "1") not in (
                "0", "false", "False"):
            # Diagnostic only -- never let a plot failure block Stage B (the
            # peak-grid writing below).
            try:
                run_peak_profile(gb_wdm_comp, wdm_holder, gi, src, peaks[0][0],
                                 extras, out_path, cat_sources, rank=0)
            except Exception as e:
                print(f"[profile] plot skipped (non-fatal, grids still "
                      f"written): {e!r}", flush=True)

        # --- Stage B: local 4-D proposal around the top comb peak(s) ---
        n_fit = int(os.environ.get("FSTAT_PEAKS_TO_FIT", 1))
        half_f0 = float(os.environ.get("FSTAT_PEAK_HALF_MHZ", 2.5e-3))
        for rank, (f0_pk, F_pk, _) in enumerate(peaks[:n_fit]):
            box = dict(
                f0_range=(f0_pk - half_f0, f0_pk + half_f0),
                Mc_range=(0.1, 1.0),
                alpha_range=(0.0, 2 * np.pi),
                sin_delta_range=(-1.0, 1.0),
            )
            stage = f"peak{rank}"
            prop = build_fstat_proposal(gb_wdm_comp, wdm_holder, src, gb_info,
                                         n_per_axis=n_per_axis, box=box,
                                         stage=stage)
            _save_grid_cache(prop, cache_path, stage)
            _report(prop, src, stage)
            pk_cat = _filter_cat(cat_sources, *box["f0_range"])
            plot_corner(prop, src, n_samples=20_000,
                        out_path=out_path.replace(".png", f"_{stage}.png"),
                        cat_sources=pk_cat,
                        stage=f"local proposal @ comb peak {rank} "
                              f"(f0={f0_pk:.5f} mHz, F={F_pk:.1f})")
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
