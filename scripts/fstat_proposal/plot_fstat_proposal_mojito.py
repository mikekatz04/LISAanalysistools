"""Real-F-stat visualization: FStatProposal4D against ``get_fstat_ll_wdm``
on a mojito WDM residual centred on the highest-frequency GB.

Reuses the ``erebor.gb_no_fg`` stock recipe to stand up the full WDM stack
(orbits, TDIConfig, WDMSettings, chunked-het ``GBWDMComputations``, and the
``AnalysisContainerArray`` from the mojito residual), then invokes the
F-stat proposal on a grid around the source.

The mojito catalogue's highest-freq GB (from the L1 catalogue) is::

    ID 7725228, f0 = 20.380 mHz, Mc = 0.519 Msol,
    RA = 4.062 rad, Dec = -0.905 rad, fdot = 1.03e-13

Environment knobs::

    GB_CENTER_FREQ_HZ  (default 0.02038 = highest-freq GB)
    GB_N_LAYERS        (default 12 -- WDM layer count in the analysis band)
    FSTAT_N_PER_AXIS   (default 24 -- grid resolution per intrinsic axis)
    MOJITO_DATA_PATH   (default /Users/mlkatz/.mojito_cache/.../mojito_light_v1_0_0/)

Output: corner plot at ``/tmp/fstat_proposal_mojito.png``.
"""

from __future__ import annotations

import os
import sys
import time

import numpy as np

# The stock recipe defaults expect mojito.
os.environ.setdefault(
    "MOJITO_DATA_PATH",
    "/Users/mlkatz/.mojito_cache/brickmarket/mojito_light_v1_0_0/",
)
os.environ.setdefault("DATA_PROCESSOR", "mojito")
os.environ.setdefault("MAKE_PLOTS", "0")
os.environ.setdefault("NWALKERS", "2")
os.environ.setdefault("NTEMPS", "1")
os.environ.setdefault("GB_USE_CHIRP_MASS", "1")

# Mojito's highest-frequency GB (from the L1 catalogue, verified interactively).
MOJITO_HIGHEST_GB = {
    "ID": 7725228,
    "f0_mHz": 20.380377,
    "Mc_Msol": 0.5192,
    "RA_rad": 4.0617,
    "Dec_rad": -0.9049,
    "fdot": 1.025e-13,
    "A": 1.371e-22,
}


def _resolve_matplotlib():
    """Force a non-interactive backend + disable usetex before any lisatools
    submodule flips it back on. Import after this returns."""
    import matplotlib

    matplotlib.use("Agg")
    matplotlib.rcParams["text.usetex"] = False
    import matplotlib.pyplot as plt  # noqa: F401 (side effect)
    return matplotlib


def _patch_slab_kernel_args():
    """The installed GBGPU kernel binding
    (``binding_gbgpu.cxx::gb_wdm_het_get_fstat_ll``, ~34 args) predates the
    task-b per-band slab args that ``chunked_het.py::_slab_kernel_args``
    always appends on the C++ path. Force the CPU dispatch to append ``()``
    like the JAX one so the arg-count matches until GBGPU is rebuilt with
    the trailing ``(band_slab_Nf, slab_min_f_int32)`` slots.

    The FStatProposal doesn't use narrow per-band slabs -- it queries the
    full active band -- so this patch is behaviourally a no-op for our use.
    """
    from lisatools.chunked_het import WDMComputationsBase

    WDMComputationsBase._slab_kernel_args = lambda self, holder: ()
    print("[patch] WDMComputationsBase._slab_kernel_args -> ()", flush=True)


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
    fit.gb.center_freq = float(center_freq_hz)
    fit.gb.n_layers = int(n_layers)
    fit.gb.use_chirp_mass = True

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
                          n_per_axis: int = 24):
    """Build FStatProposal4D on a grid around the source location.

    The grid box is bounded by the analysis band (``gb_info.f0_lims``) and by
    sensible defaults on the other axes.
    """
    from lisatools.sampling.fstat_proposal import FStatProposal4D, GridSpec

    sin_delta_src = float(np.sin(src["Dec_rad"]))
    f0_mHz_lo = float(gb_info.f0_lims[0]) * 1e3
    f0_mHz_hi = float(gb_info.f0_lims[-1]) * 1e3
    print(f"[grid] f0 range: [{f0_mHz_lo:.4f}, {f0_mHz_hi:.4f}] mHz  "
          f"(source at {src['f0_mHz']:.4f} mHz)", flush=True)

    grid = GridSpec(
        f0_range=(f0_mHz_lo, f0_mHz_hi),
        Mc_range=(0.1, 1.0),
        alpha_range=(0.0, 2 * np.pi),
        sin_delta_range=(-1.0, 1.0),
        n_f0=n_per_axis,
        n_Mc=n_per_axis,
        n_alpha=n_per_axis,
        n_sin_delta=n_per_axis,
    )
    print(f"[grid] {n_per_axis}^4 = {n_per_axis**4} F-stat evals...", flush=True)
    t0 = time.time()
    prop = FStatProposal4D(
        gb_wdm_comp=gb_wdm_comp,
        wdm_holder=wdm_holder,
        grid_spec=grid,
        beta=1.0,
    )
    print(f"[grid]   done in {time.time() - t0:.1f}s", flush=True)
    return prop


def plot_corner(prop, src, n_samples=20_000, out_path="/tmp/fstat_proposal_mojito.png"):
    import matplotlib
    import matplotlib.pyplot as plt

    # Kill usetex again -- some transitive import (globalfit.diagnosticplot)
    # flipped it back on during build_gb_wdm_comp_and_holder().
    matplotlib.rcParams["text.usetex"] = False
    matplotlib.rcParams["axes.formatter.use_mathtext"] = True

    names = ["f0 [mHz]", "Mc [Msol]", "alpha [rad]", "sin_delta"]
    axes = prop._axes
    injection_mu = np.array([
        src["f0_mHz"], src["Mc_Msol"], src["RA_rad"], float(np.sin(src["Dec_rad"])),
    ])

    g_cells = np.asarray(prop._logp_grid[:-1, :-1, :-1, :-1])
    p_cells = np.exp(g_cells - g_cells.max())
    samples = np.asarray(prop.rvs(size=(n_samples,)))
    cell_centres = [0.5 * (axes[i][:-1] + axes[i][1:]) for i in range(4)]

    fig, ax_grid = plt.subplots(4, 4, figsize=(12, 12))
    plt.suptitle(
        f"FStatProposal4D on real mojito F-stat (ID {src['ID']}, "
        f"f0={src['f0_mHz']:.3f} mHz)",
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
                a.axvline(injection_mu[i], color="r", linestyle="--", lw=1.5,
                          label="mojito GB")
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
                extent = [cell_centres[j][0], cell_centres[j][-1],
                          cell_centres[i][0], cell_centres[i][-1]]
                a.imshow(p2.T, origin="lower", extent=extent, aspect="auto",
                         cmap="viridis")
                a.scatter(samples[::200, j], samples[::200, i], s=1.5, c="w",
                          alpha=0.4)
                a.plot(injection_mu[j], injection_mu[i], "r*", markersize=12,
                       mec="k", mew=0.5)
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
    _patch_slab_kernel_args()

    src = MOJITO_HIGHEST_GB
    center_freq_hz = float(os.environ.get("GB_CENTER_FREQ_HZ", src["f0_mHz"] * 1e-3))
    n_layers = int(os.environ.get("GB_N_LAYERS", 12))
    n_per_axis = int(os.environ.get("FSTAT_N_PER_AXIS", 24))

    print(f"[main] Target: mojito highest-freq GB (ID {src['ID']}, "
          f"f0={src['f0_mHz']:.4f} mHz)", flush=True)
    print(f"[main] Analysis band centre: {center_freq_hz * 1e3:.4f} mHz "
          f"({n_layers} layers)", flush=True)

    gb_wdm_comp, wdm_holder, curr = build_gb_wdm_comp_and_holder(
        center_freq_hz, n_layers,
    )
    gb_info = curr.source_info["gb"]

    # Optional cache for the F-stat grid (bypasses the expensive
    # ``get_fstat_ll_wdm`` sweep on re-runs). Enable via
    # FSTAT_GRID_CACHE=/path/to/grid.npz.
    cache_path = os.environ.get("FSTAT_GRID_CACHE", "").strip()
    prop = build_fstat_proposal(gb_wdm_comp, wdm_holder, src, gb_info,
                                 n_per_axis=n_per_axis)
    if cache_path:
        try:
            np.savez(
                cache_path,
                logp_grid=np.asarray(prop._logp_grid),
                f0_ax=np.asarray(prop._axes[0]),
                Mc_ax=np.asarray(prop._axes[1]),
                alpha_ax=np.asarray(prop._axes[2]),
                sin_delta_ax=np.asarray(prop._axes[3]),
                log_norm=float(prop._log_norm),
            )
            print(f"[cache] wrote {cache_path}", flush=True)
        except Exception as e:
            print(f"[cache] failed to write {cache_path}: {e}", flush=True)

    # Print the grid-max cell to see where the F-stat actually peaks.
    g = np.asarray(prop._logp_grid[:-1, :-1, :-1, :-1])
    i0, i1, i2, i3 = np.unravel_index(np.argmax(g), g.shape)
    axes = prop._axes
    peak = (float(axes[0][i0]), float(axes[1][i1]),
            float(axes[2][i2]), float(axes[3][i3]))
    print(f"[peak] grid argmax logp={g.max():.3f} at f0={peak[0]:.4f}mHz  "
          f"Mc={peak[1]:.4f}  alpha={peak[2]:.4f}  sin_delta={peak[3]:.4f}",
          flush=True)
    print(f"[peak] injection:       f0={src['f0_mHz']:.4f}mHz  "
          f"Mc={src['Mc_Msol']:.4f}  alpha={src['RA_rad']:.4f}  "
          f"sin_delta={np.sin(src['Dec_rad']):.4f}", flush=True)

    # Diagnostics
    inj_sampling = np.array([[src["f0_mHz"], src["Mc_Msol"], src["RA_rad"],
                              float(np.sin(src["Dec_rad"]))]])
    lp_at_injection = float(np.asarray(prop.logpdf(inj_sampling))[0])
    s = prop.rvs(size=(5000,))
    lp = np.asarray(prop.logpdf(s))
    print(f"[diag] logpdf @ injection: {lp_at_injection:.3f}", flush=True)
    print(f"[diag] logpdf on samples: median={np.median(lp):.3f}  "
          f"5%={np.percentile(lp, 5):.3f}  95%={np.percentile(lp, 95):.3f}",
          flush=True)
    print(f"[diag] sample mean: {np.asarray(s).mean(axis=0)}", flush=True)
    print(f"[diag] injection:   {inj_sampling[0]}", flush=True)

    plot_corner(prop, src, n_samples=20_000)


if __name__ == "__main__":
    main()
