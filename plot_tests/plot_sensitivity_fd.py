"""Frequency-domain sensitivity example.

Builds two FD ``CompositeSensitivityMatrix`` instances on the same
:class:`FDSettings` grid -- one with the LISA instrument noise alone, one
adding the stationary Galactic-confusion foreground -- and plots the diagonal
elements (XX, YY, ZZ) plus the magnitude of an off-diagonal (XY) so the
foreground's effect on both auto- and cross-spectra is visible.

Run from the LISAanalysistools repo root:

    python plot_tests/plot_sensitivity_fd.py
"""

import os

import matplotlib.pyplot as plt
import numpy as np

from lisatools.domains import FDSettings
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


def main():
    settings = FDSettings(
        N=20001,
        df=5e-6,
        min_freq=1e-4,
        max_freq=1e-1,
        force_backend="cpu",
    )

    instrument_only = CompositeSensitivityMatrix(
        settings,
        [InstrumentNoise(tdi_generation=2, model="sangria")],
    )
    instrument_plus_foreground = CompositeSensitivityMatrix(
        settings,
        [
            InstrumentNoise(tdi_generation=2, model="sangria"),
            # ``modulation=None`` -> stationary isotropic limit: diag=1, off-diag=-1/2.
            GalacticForeground(
                foreground_params=SANGRIA_FOREGROUND_PARAMS,
                tdi_generation=2,
                modulation=None,
            ),
        ],
    )

    f = np.asarray(settings.f_arr)

    fig, axes = plt.subplots(1, 2, figsize=(11, 4.5), sharex=True)
    diag_ax, cross_ax = axes

    # Diagonal: XX, YY, ZZ (symmetric, so they sit on top of one another for
    # the stationary case -- plotting all three confirms the diagonal-symmetric
    # construction at the matrix level).
    channel_labels = ["XX", "YY", "ZZ"]
    for ch, label in enumerate(channel_labels):
        diag_ax.loglog(
            f,
            np.asarray(instrument_only.sens_mat[ch, ch]),
            color="C0",
            alpha=0.5 + 0.15 * ch,
            label=f"instrument {label}" if ch == 0 else None,
        )
        diag_ax.loglog(
            f,
            np.asarray(instrument_plus_foreground.sens_mat[ch, ch]),
            color="C3",
            alpha=0.5 + 0.15 * ch,
            label=f"instrument + foreground {label}" if ch == 0 else None,
        )
    diag_ax.set_xlabel("frequency [Hz]")
    diag_ax.set_ylabel(r"$S_{ii}(f)$ [1/Hz]")
    diag_ax.set_title("Diagonal PSDs")
    diag_ax.legend(loc="lower left")

    # Off-diagonal magnitude: |XY|. The foreground introduces correlated power
    # between channels, lifting the cross-PSD above the instrument-only level.
    cross_ax.loglog(
        f,
        np.abs(np.asarray(instrument_only.sens_mat[0, 1])),
        color="C0",
        label="instrument",
    )
    cross_ax.loglog(
        f,
        np.abs(np.asarray(instrument_plus_foreground.sens_mat[0, 1])),
        color="C3",
        label="instrument + foreground",
    )
    cross_ax.set_xlabel("frequency [Hz]")
    cross_ax.set_ylabel(r"$|S_{XY}(f)|$ [1/Hz]")
    cross_ax.set_title("Off-diagonal (XY) cross-PSD")
    cross_ax.legend(loc="lower left")

    fig.suptitle(
        "FD sensitivity: instrument vs. instrument + Galactic foreground "
        "(TDI 2, sangria)"
    )
    fig.tight_layout()

    out = os.path.join(OUT_DIR, "sensitivity_fd_with_foreground.png")
    fig.savefig(out, dpi=140)
    print(f"wrote {out}")


if __name__ == "__main__":
    main()
