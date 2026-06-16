"""WDM (wavelet) sensitivity example.

Builds two WDM ``CompositeSensitivityMatrix`` instances on the same
:class:`WDMSettings` grid: instrument + Galactic foreground, with and without
a per-channel time modulation on the foreground. The diagonal XX scalogram is
plotted for both, plus a time slice at fixed frequency that shows the
modulation cycling in/out over the observation.

Run from the LISAanalysistools repo root:

    python plot_tests/plot_sensitivity_wdm.py
"""

import os

import matplotlib.pyplot as plt
import numpy as np
from matplotlib.colors import LogNorm

from lisatools.domains import WDMSettings
from lisatools.sensitivity import (
    CompositeSensitivityMatrix,
    GalacticForeground,
    InstrumentNoise,
)


OUT_DIR = os.path.dirname(os.path.abspath(__file__))

# Sangria-tuned hyperbolic-tangent foreground parameters (amp, fk, alpha, s1, s2).
SANGRIA_FOREGROUND_PARAMS = (
    3.26651613e-44,
    2.09278117e-03,
    1.18300266e00,
    3.01430978e03,
    2.95774596e03,
)


def make_yearly_modulation(period_s: float):
    """Return a ``t_arr -> (3, 3, Ntime)`` modulation callable.

    Diagonal channels (XX, YY, ZZ) oscillate at ``period_s`` with a 1/3-cycle
    phase offset between them -- a cartoon of the LISA-orbit modulation that
    re-points each arm pair relative to the Galactic bulge. Off-diagonals are
    held at ``-1/2`` (the stationary equal-arm value).
    """

    def _mod(t_arr):
        t = np.asarray(t_arr)
        nt = t.shape[0]
        M = np.full((3, 3, nt), -0.5)
        for k in range(3):
            M[k, k] = 1.0 + 0.4 * np.sin(2.0 * np.pi * t / period_s + 2.0 * np.pi * k / 3.0)
        return M

    return _mod


def scalogram(ax, settings, arr, *, title, vmin=None, vmax=None):
    """``pcolormesh`` of ``|arr|`` on (time [days], frequency [mHz]) WDM edges."""
    a = np.abs(np.asarray(arr))  # (Nf, Nt)
    t_days = np.asarray(settings.t_arr_edges, dtype=float) / 86400.0
    f_mHz = np.asarray(settings.f_arr_edges, dtype=float) * 1e3
    # ``f=0`` edge: drop it so a log-scaled colormap doesn't blow up.
    f_lo = f_mHz[f_mHz > 0].min()
    f_mHz = f_mHz.copy()
    f_mHz[f_mHz <= 0.0] = f_lo * 1e-3
    im = ax.pcolormesh(
        t_days,
        f_mHz,
        a,
        norm=LogNorm(vmin=vmin, vmax=vmax),
        cmap="viridis",
        shading="flat",
    )
    ax.set_ylim(f_lo, float(f_mHz[-1]))
    ax.set_xlabel("time [days]")
    ax.set_ylabel("frequency [mHz]")
    ax.set_title(title)
    return im


def main():
    Nf = 128
    Nt = 1024
    dt = 10.0
    settings = WDMSettings(
        Nf=Nf,
        Nt=Nt,
        dt=dt,
        min_freq=3e-4,
        max_freq=1e-2,
        force_backend="cpu",
    )

    # Show two cycles of the (cartoon) modulation across the observation.
    modulation_period_s = settings.Tobs / 2.0
    modulation = make_yearly_modulation(modulation_period_s)

    instrument = InstrumentNoise(tdi_generation=2, model="sangria")
    foreground_stationary = GalacticForeground(
        foreground_params=SANGRIA_FOREGROUND_PARAMS,
        tdi_generation=2,
        modulation=None,
    )
    foreground_modulated = GalacticForeground(
        foreground_params=SANGRIA_FOREGROUND_PARAMS,
        tdi_generation=2,
        modulation=modulation,
    )

    sm_stationary = CompositeSensitivityMatrix(settings, [instrument, foreground_stationary])
    sm_modulated = CompositeSensitivityMatrix(settings, [instrument, foreground_modulated])

    XX_stat = np.asarray(sm_stationary.sens_mat[0, 0])  # (Nf, Nt)
    XX_mod = np.asarray(sm_modulated.sens_mat[0, 0])
    YY_mod = np.asarray(sm_modulated.sens_mat[1, 1])
    ZZ_mod = np.asarray(sm_modulated.sens_mat[2, 2])

    # Match the colour scales across the two scalograms so visual differences
    # come from the data, not from autoscaling. WDM transforms can produce
    # non-finite values in the boundary layer; filter them before reducing.
    def _finite_positive(arr):
        finite = arr[np.isfinite(arr) & (arr > 0)]
        return finite

    vmin = float(min(_finite_positive(XX_stat).min(), _finite_positive(XX_mod).min()))
    vmax = float(max(_finite_positive(XX_stat).max(), _finite_positive(XX_mod).max()))

    fig = plt.figure(figsize=(13, 8))
    gs = fig.add_gridspec(2, 2, height_ratios=[1.3, 1.0], hspace=0.4, wspace=0.3)

    ax_stat = fig.add_subplot(gs[0, 0])
    im_stat = scalogram(
        ax_stat, settings, XX_stat,
        title="XX scalogram: stationary foreground",
        vmin=vmin, vmax=vmax,
    )
    fig.colorbar(im_stat, ax=ax_stat, label=r"$|S_{XX}|$")

    ax_mod = fig.add_subplot(gs[0, 1])
    im_mod = scalogram(
        ax_mod, settings, XX_mod,
        title="XX scalogram: foreground with time modulation",
        vmin=vmin, vmax=vmax,
    )
    fig.colorbar(im_mod, ax=ax_mod, label=r"$|S_{XX}|$")

    # Time slice at the foreground peak (the layer closest to the knee
    # frequency ``fk``); shows the diagonal channels moving in/out of phase.
    f_arr = np.asarray(settings.f_arr)
    fk = SANGRIA_FOREGROUND_PARAMS[1]
    layer = int(np.argmin(np.abs(f_arr - fk)))
    t_days = np.asarray(settings.t_arr) / 86400.0

    ax_slice = fig.add_subplot(gs[1, :])
    ax_slice.plot(t_days, XX_stat[layer], "k--", label="XX stationary")
    ax_slice.plot(t_days, XX_mod[layer], color="C0", label="XX modulated")
    ax_slice.plot(t_days, YY_mod[layer], color="C1", label="YY modulated")
    ax_slice.plot(t_days, ZZ_mod[layer], color="C2", label="ZZ modulated")
    ax_slice.set_yscale("log")
    ax_slice.set_xlabel("time [days]")
    ax_slice.set_ylabel(r"$|S_{ii}(t)|$ in layer $f \approx %.2f$ mHz" % (f_arr[layer] * 1e3,))
    ax_slice.set_title(
        "Diagonal-channel time slices: stationary (dashed) vs modulated (solid)"
    )
    ax_slice.legend(loc="upper right", ncol=4)

    fig.suptitle(
        "WDM sensitivity: instrument + Galactic foreground, stationary vs time-modulated "
        "(TDI 2, sangria)"
    )

    out = os.path.join(OUT_DIR, "sensitivity_wdm_with_foreground.png")
    fig.savefig(out, dpi=140)
    print(f"wrote {out}")


if __name__ == "__main__":
    main()
