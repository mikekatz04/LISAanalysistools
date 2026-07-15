"""Visualization: FStatProposal4D on a mock F-stat surface centered on the
highest-frequency GB in mojito.

Purpose: validate the *proposal machinery* (grid + inverse-CDF, sampling
basis, boundary handling) using a known-Gaussian F-stat surface at a
realistic sky/frequency point. The mojito GB itself is:

    ID 7725228, f0 = 20.380 mHz, Mc = 0.519 Msol, RA = 4.062 rad,
    Dec = -0.905 rad, fdot = 1.03e-13 (from catalogue)

A companion script (``plot_fstat_proposal_mojito.py``) will run the *real*
``get_fstat_ll_wdm`` against the mojito residual once the WDM stack is
wired end-to-end.

Output: a corner plot of the 4D density with the source location marked,
saved to ``/tmp/fstat_proposal_mock_highest_gb.png``.

Run:
    python -m scripts.fstat_proposal.plot_fstat_proposal_mock_highest_gb
"""

from __future__ import annotations

import numpy as np
import matplotlib

matplotlib.use("Agg")
import matplotlib.pyplot as plt

from lisatools.sampling.fstat_proposal import (
    FStatProposal4D,
    GridSpec,
    compute_fstat,
)

# ``lisatools.globalfit.diagnosticplot`` (transitively imported by anything
# under ``lisatools.globalfit``) flips ``text.usetex`` on at module load; force
# it off HERE, after all imports have settled.
matplotlib.rcParams["text.usetex"] = False


# Highest-frequency GB in mojito (from the L1 catalogue).
MOJITO_HIGHEST_GB = {
    "ID": 7725228,
    "f0_mHz": 20.380377,
    "Mc_Msol": 0.5192,
    "RA_rad": 4.0617,
    "Dec_rad": -0.9049,
    "fdot": 1.025e-13,
}


class MockGaussianFstat:
    """Mock ``GBWDMComputations`` whose F-stat is a 4-D Gaussian in the
    sampling basis at ``(mu, sigma)``.

    Encodes the target ``F(theta)`` into ``(N, M)`` so
    :func:`~lisatools.sampling.fstat_proposal.compute_fstat` recovers
    ``F = -0.5 * ||(theta - mu)/sigma||^2 + F_peak`` from them. Uses
    ``M = I`` (identity, upper-triangle = [1, 0, 0, 0, 1, 0, 0, 1, 0, 1])
    and ``N = sqrt(2 * F_shifted) * e_0`` with ``F_shifted = F + const``
    (chosen so it's always positive).
    """

    def __init__(self, mu_sampling_basis, sigma_sampling_basis, F_peak=100.0):
        self.mu = np.asarray(mu_sampling_basis, dtype=np.float64)
        self.sigma = np.asarray(sigma_sampling_basis, dtype=np.float64)
        self.F_peak = float(F_peak)

    def get_fstat_ll_wdm(self, params, wdm_holder):
        """Invert the base class's sampling-basis -> physical-basis packing to
        get back the mock's native sampling coords, score them.
        """
        from gbgpu.utils.utility import get_chirp_mass_from_f_fdot

        f0_Hz = params[:, 1]
        fdot = params[:, 2]
        lam = params[:, 7]
        beta_rad = params[:, 8]

        f0_mHz = f0_Hz * 1e3
        Mc = np.asarray(get_chirp_mass_from_f_fdot(f0_Hz, fdot), dtype=np.float64)
        sin_delta = np.sin(beta_rad)
        theta = np.stack([f0_mHz, Mc, lam, sin_delta], axis=-1)

        diff = (theta - self.mu) / self.sigma
        F = -0.5 * np.sum(diff ** 2, axis=-1) + self.F_peak
        F = np.clip(F, 1e-6, None)  # positive so we can encode into N

        n = params.shape[0]
        N = np.zeros((n, 4), dtype=np.float64)
        N[:, 0] = np.sqrt(2.0 * F)
        M_upper = np.zeros((n, 10), dtype=np.float64)
        M_upper[:, [0, 4, 7, 9]] = 1.0  # 4x4 identity in row-major upper tri
        return N, M_upper


def make_proposal_around(src, half_widths=None, n_per_axis=48):
    """Build an FStatProposal4D on a zoomed grid around a source.

    ``half_widths`` sets the box: ``{'f0_mHz', 'Mc_Msol', 'alpha_rad',
    'sin_delta'}`` -> half-width per axis. The peak sits at the box centre.
    """
    if half_widths is None:
        half_widths = {
            "f0_mHz": 2.0,
            "Mc_Msol": 0.2,
            "alpha_rad": 0.5,
            "sin_delta": 0.15,
        }
    sin_delta_src = float(np.sin(src["Dec_rad"]))
    mu = np.array([src["f0_mHz"], src["Mc_Msol"], src["RA_rad"], sin_delta_src])
    # Gaussian widths: give it a narrow peak per axis (a fraction of the box).
    sigma = np.array([
        0.05 * half_widths["f0_mHz"],
        0.2 * half_widths["Mc_Msol"],
        0.2 * half_widths["alpha_rad"],
        0.15 * half_widths["sin_delta"],
    ])
    mock = MockGaussianFstat(mu, sigma, F_peak=50.0)

    grid = GridSpec(
        f0_range=(max(1e-3, src["f0_mHz"] - half_widths["f0_mHz"]),
                  src["f0_mHz"] + half_widths["f0_mHz"]),
        Mc_range=(max(0.05, src["Mc_Msol"] - half_widths["Mc_Msol"]),
                  src["Mc_Msol"] + half_widths["Mc_Msol"]),
        alpha_range=(src["RA_rad"] - half_widths["alpha_rad"], src["RA_rad"] + half_widths["alpha_rad"]),
        sin_delta_range=(max(-1.0, sin_delta_src - half_widths["sin_delta"]),
                         min(1.0, sin_delta_src + half_widths["sin_delta"])),
        n_f0=n_per_axis, n_Mc=n_per_axis, n_alpha=n_per_axis, n_sin_delta=n_per_axis,
    )
    prop = FStatProposal4D(gb_wdm_comp=mock, wdm_holder=None, grid_spec=grid, beta=1.0)
    return prop, mu, sigma


def plot_corner(prop, injection_mu, injection_sigma, n_samples=20_000, title="", out_path=None):
    """Corner plot: 6 pairwise marginal densities + 1D marginals on the diagonal.

    Overlays: (a) the 2-D marginal density from the grid (imshow), (b) the
    injection point as a red star, (c) n_samples ``.rvs()`` draws as a scatter.
    """
    names = ["f0 [mHz]", "Mc [Msol]", "alpha [rad]", "sin_delta"]
    axes = prop._axes  # (f0_ax, Mc_ax, al_ax, sd_ax)

    # 2-D marginals of the grid density: sum out the other axes.
    # logp_grid shape (n_f0, n_Mc, n_al, n_sd). Restrict to the (N-1)^d "active" slab.
    g_cells = prop._logp_grid[:-1, :-1, :-1, :-1]
    p_cells = np.exp(g_cells - g_cells.max())

    # Sample from the proposal.
    samples = np.asarray(prop.rvs(size=(n_samples,)))

    fig, ax_grid = plt.subplots(4, 4, figsize=(12, 12))
    plt.suptitle(title, y=0.995)

    # Axis-node centres for imshow extents (piecewise-constant on cells).
    cell_centres = [
        0.5 * (axes[i][:-1] + axes[i][1:]) for i in range(4)
    ]

    for i in range(4):
        for j in range(4):
            a = ax_grid[i, j]
            if i == j:
                # 1-D marginal on the diagonal. Normalize BOTH the grid
                # marginal and the sample histogram to their peak so they
                # sit on the same y-scale (peak-normalized, easier to read
                # than density-normalized when the peaks are sharp).
                marg_axes = tuple(k for k in range(4) if k != i)
                p1 = p_cells.sum(axis=marg_axes)
                p1n = p1 / p1.max()
                a.plot(cell_centres[i], p1n, "b-", lw=1.5, label="grid marginal")
                a.axvline(injection_mu[i], color="r", linestyle="--", lw=1.5, label="injection")
                # Peak-normalized histogram counts (not density) matches the blue curve.
                counts, edges = np.histogram(samples[:, i], bins=60)
                centres = 0.5 * (edges[:-1] + edges[1:])
                a.step(
                    centres, counts / max(1, counts.max()),
                    where="mid", color="green", lw=1.0, label="rvs samples",
                )
                a.set_xlabel(names[i])
                a.set_ylabel("density (peak-normed)")
                a.set_xlim(cell_centres[i][0], cell_centres[i][-1])
                if i == 0:
                    a.legend(fontsize=7, loc="upper right")
            elif i > j:
                # 2-D marginal below the diagonal: (j on x, i on y).
                marg_axes = tuple(k for k in range(4) if k not in (i, j))
                p2 = p_cells.sum(axis=marg_axes)
                # Ensure orientation (j on x, i on y): if j < i, transpose so that
                # axis-0 of p2 is axis j.
                if j < i:
                    p2 = p2 if _p2_axis_order(p2.shape, (j, i)) else p2.T
                extent = [
                    cell_centres[j][0], cell_centres[j][-1],
                    cell_centres[i][0], cell_centres[i][-1],
                ]
                a.imshow(
                    p2.T, origin="lower", extent=extent, aspect="auto",
                    cmap="viridis",
                )
                a.scatter(samples[::200, j], samples[::200, i], s=1.5, c="w", alpha=0.4)
                a.plot(injection_mu[j], injection_mu[i], "r*", markersize=12, mec="k", mew=0.5)
                a.set_xlabel(names[j])
                a.set_ylabel(names[i])
            else:
                a.axis("off")

    plt.tight_layout()
    if out_path is None:
        out_path = "/tmp/fstat_proposal_mock_highest_gb.png"
    plt.savefig(out_path, dpi=140)
    plt.close(fig)
    print(f"[plot] wrote {out_path}", flush=True)
    return out_path


def _p2_axis_order(p2_shape, axes_kept):
    """Reasoning helper: axis 0 of ``p2`` should correspond to the lower index
    of ``axes_kept`` after ``sum``. Returns True if that's already the case.
    """
    return axes_kept[0] < axes_kept[1]


def main():
    src = MOJITO_HIGHEST_GB
    print(f"Target: mojito GB ID {src['ID']}  f0={src['f0_mHz']:.4f}mHz  "
          f"Mc={src['Mc_Msol']:.4f}Msol  RA={src['RA_rad']:.4f}  Dec={src['Dec_rad']:.4f}",
          flush=True)

    # --- Zoomed grid around the source ---
    prop, mu, sigma = make_proposal_around(src, n_per_axis=48)
    print(f"[zoom] mu={mu}  sigma={sigma}", flush=True)
    # Overplot the injection sigma too as a text.
    plot_corner(
        prop, mu, sigma,
        title=(f"FStatProposal4D on mock F-stat around mojito highest-freq GB "
               f"(ID {src['ID']}, f0={src['f0_mHz']:.3f} mHz)"),
        out_path="/tmp/fstat_proposal_mock_zoom.png",
    )

    # --- Wide grid: full-band mHz range, coarser resolution ---
    prop_wide, mu_wide, sigma_wide = make_proposal_around(
        src,
        half_widths={"f0_mHz": 15.0, "Mc_Msol": 0.6, "alpha_rad": np.pi,
                     "sin_delta": 0.95},
        n_per_axis=40,
    )
    plot_corner(
        prop_wide, mu_wide, sigma_wide,
        title=(f"FStatProposal4D on wide grid (mock F-stat), mojito GB {src['ID']}"),
        out_path="/tmp/fstat_proposal_mock_wide.png",
    )

    # --- Print quantitative diagnostics ---
    inj_sampling = np.array([[src["f0_mHz"], src["Mc_Msol"], src["RA_rad"],
                              float(np.sin(src["Dec_rad"]))]])
    lp_at_injection = float(prop.logpdf(inj_sampling)[0])
    # Get logpdf statistics on the box.
    s = prop.rvs(size=(5000,))
    lp = prop.logpdf(s)
    print(f"[diag] logpdf @ injection (zoom): {lp_at_injection:.3f}", flush=True)
    print(f"[diag] logpdf on samples: median={np.median(lp):.3f}  "
          f"5%={np.percentile(lp, 5):.3f}  95%={np.percentile(lp, 95):.3f}", flush=True)
    # rvs-mean check
    mean_samples = s.mean(axis=0)
    print(f"[diag] sample mean vs injection:  {mean_samples}   vs   {mu}", flush=True)


if __name__ == "__main__":
    main()
