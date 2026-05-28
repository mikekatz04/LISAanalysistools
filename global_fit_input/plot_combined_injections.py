"""Diagnostic plotter for the combined smoke-test injection.

Builds the same four components the
:class:`SangriaPlusInjectionsProcessingStep` data processor sums into the
smoke-test data:

* Sangria sky-only slice (GBs + galactic foreground, ``mbhb`` removed).
* Synthetic correlated FD instrument noise (default seed = 12345).
* Synthetic EMRI TD waveform from the shared response wrapper.
* Synthetic SOBBH TD waveform from the shared response wrapper.

Then plots two figures:

* ``combined_injections_td.png`` — five-panel TD strip chart of the X
  channel, one panel per component plus the combined sum.
* ``combined_injections_fd.png`` — log-log ASD of the X channel for all
  four components plus the combined sum (one panel, overlaid).

Run::

    python LISAanalysistools/global_fit_input/plot_combined_injections.py
"""

from __future__ import annotations

import argparse
import os
import sys

import matplotlib.pyplot as plt
import numpy as np

# Resolve sibling imports the same way ``run_global.py`` does — by adding
# the directory to ``sys.path`` before pulling from the smoke-test module.
_HERE = os.path.dirname(os.path.abspath(__file__))
if _HERE not in sys.path:
    sys.path.insert(0, _HERE)

from fastlisaresponse.tdiconfig import TDIConfig

from lisatools.globalfit.preprocessing import SangriaDataLoader

from combined_gb_psd_emri_sobbh_global_fit_settings import (  # noqa: E402
    DT,
    EMRI_INJECTIONS_FULL_BASIS,
    SOBBH_INJECTIONS_FULL_BASIS,
    T_START,
    TOBS,
    _generate_correlated_fd_noise,
    _pad_or_clip,
    get_emri_response_wrapper,
    get_sobbh_response_wrapper,
)


DEFAULT_SANGRIA_PATH = (
    "/Users/mkatz/Research/LISAanalysistools/LDC2_sangria_training_v2.h5"
)


def _load_sangria_gb_only(data_input_path: str, target_N: int, nchannels: int = 3) -> np.ndarray:
    """Sangria sky-source-only slice (no instrument noise, mbhb removed)."""
    loader = SangriaDataLoader(
        data_input_path=data_input_path,
        remove_from_data=["noise", "mbhb"],
    )
    _, _, data_xyz, _ = loader.load_data()
    return _pad_or_clip(np.atleast_2d(data_xyz)[:nchannels], target_N)


def _td_to_asd(td: np.ndarray, dt: float) -> tuple[np.ndarray, np.ndarray]:
    """Per-channel one-sided ASD using the LDC convention ``FD = dt * rfft(TD)``.

    Returns ``(freqs, asd)`` where ``asd`` has shape ``(nchannels, Nf)``.
    """
    fd = dt * np.fft.rfft(td, axis=-1)
    N = td.shape[-1]
    T = N * dt
    freqs = np.fft.rfftfreq(N, d=dt)
    # One-sided ASD: sqrt(2 * |FD|^2 / T). The k = 0 (DC) bin gets zeroed by
    # the irfft already, so leave it.
    asd = np.abs(fd) * np.sqrt(2.0 / T)
    return freqs, asd


def _td_chunk_for_plot(td: np.ndarray, max_points: int = 12000) -> np.ndarray:
    """Decimate a TD stream uniformly to keep matplotlib plotting tractable."""
    if td.shape[-1] <= max_points:
        return td
    step = int(np.ceil(td.shape[-1] / max_points))
    return td[..., ::step]


def main(
    out_dir: str,
    sangria_path: str,
    noise_seed: int,
    channel_index: int,
) -> None:
    os.makedirs(out_dir, exist_ok=True)
    target_N = int(round(TOBS / DT))
    ch_label = ["X", "Y", "Z"][channel_index]

    print(f"[plot] Loading Sangria sky slice from {sangria_path}")
    gb_td = _load_sangria_gb_only(sangria_path, target_N=target_N)

    print(f"[plot] Generating correlated FD noise (seed={noise_seed})")
    noise_td = _generate_correlated_fd_noise(
        N=target_N,
        dt=DT,
        Soms_d=15e-12,
        Sa_a=3e-15,
        tdi_generation=2,
        seed=noise_seed,
    )

    tdi_config = TDIConfig("2nd generation", force_backend="cpu")

    print(f"[plot] Generating {EMRI_INJECTIONS_FULL_BASIS.shape[0]} EMRI injection(s)")
    emri_wave_gen = get_emri_response_wrapper(
        Tobs=TOBS,
        dt=DT,
        t_start=T_START,
        tdi_config=tdi_config,
        tdi_chan="XYZ",
        role="injection",
        force_backend="cpu",
    )
    emri_tds = []
    for params in EMRI_INJECTIONS_FULL_BASIS:
        sig = np.asarray(emri_wave_gen(*params, convert_to_ra_dec=False))
        emri_tds.append(_pad_or_clip(np.atleast_2d(sig)[:3], target_N))
    emri_td_sum = np.sum(emri_tds, axis=0)

    print(f"[plot] Generating {SOBBH_INJECTIONS_FULL_BASIS.shape[0]} SOBBH injection(s)")
    sobbh_wave_gen = get_sobbh_response_wrapper(
        Tobs=TOBS,
        dt=DT,
        t_start=T_START,
        tdi_config=tdi_config,
        tdi_chan="XYZ",
        role="injection",
        force_backend="cpu",
    )
    sobbh_tds = []
    for params in SOBBH_INJECTIONS_FULL_BASIS:
        sig = np.asarray(sobbh_wave_gen(*params, convert_to_ra_dec=False))
        sobbh_tds.append(_pad_or_clip(np.atleast_2d(sig)[:3], target_N))
    sobbh_td_sum = np.sum(sobbh_tds, axis=0)

    combined_td = gb_td + noise_td + emri_td_sum + sobbh_td_sum

    # TD figure: one panel per coarse component (per-source EMRI / SOBBH
    # streams overlap so far at LISA noise level that summing is more
    # informative). FD plot below shows each source individually.
    components = [
        ("Sangria GBs (dgb+igb+vgb)", gb_td),
        ("FD instrument noise", noise_td),
        (f"Synthetic EMRI (sum of {len(emri_tds)})", emri_td_sum),
        (f"Synthetic SOBBH (sum of {len(sobbh_tds)})", sobbh_td_sum),
        ("Combined injection", combined_td),
    ]

    # --- TD figure -------------------------------------------------------
    times = np.arange(target_N) * DT + T_START
    times_days = times / 86400.0
    fig_td, axes_td = plt.subplots(
        len(components), 1, figsize=(11, 2.0 * len(components)), sharex=True
    )
    for ax, (label, td) in zip(axes_td, components):
        td_dec = _td_chunk_for_plot(td[channel_index])
        t_dec = _td_chunk_for_plot(times_days)
        ax.plot(t_dec, td_dec, lw=0.5, color="C0")
        ax.set_ylabel(f"{label}\n{ch_label} strain")
        ax.grid(True, alpha=0.3)
    axes_td[-1].set_xlabel("time [days]")
    fig_td.suptitle(
        f"Combined smoke-test injection — TD ({ch_label} channel, "
        f"Tobs={TOBS/86400:.0f} d, dt={DT:.0f} s)"
    )
    fig_td.tight_layout()
    td_path = os.path.join(out_dir, "combined_injections_td.png")
    fig_td.savefig(td_path, dpi=150, bbox_inches="tight")
    plt.close(fig_td)
    print(f"[plot] wrote {td_path}")

    # --- FD figure -------------------------------------------------------
    # Plot every individual EMRI / SOBBH source as its own line so the
    # different sky / phase / distance realisations are visible, plus the
    # other components and the combined sum.
    fd_lines = [
        ("Sangria GBs", gb_td, dict(color="C2", lw=0.6)),
        ("FD noise", noise_td, dict(color="C7", lw=0.6, alpha=0.7)),
    ]
    for i, td_i in enumerate(emri_tds):
        fd_lines.append((f"EMRI {i+1}", td_i, dict(color="C0", lw=0.6, alpha=0.7 + 0.1 * i)))
    for i, td_i in enumerate(sobbh_tds):
        fd_lines.append((f"SOBBH {i+1}", td_i, dict(color="C3", lw=0.6, alpha=0.7 + 0.1 * i)))
    fd_lines.append(("Combined", combined_td, dict(color="black", lw=1.0)))

    fig_fd, ax_fd = plt.subplots(figsize=(10, 6))
    for label, td, kw in fd_lines:
        freqs, asd = _td_to_asd(td[channel_index : channel_index + 1], dt=DT)
        mask = (freqs > 0) & (freqs <= 0.1)
        ax_fd.loglog(freqs[mask], asd[0, mask], label=label, **kw)
    ax_fd.set_xlabel("frequency [Hz]")
    ax_fd.set_ylabel(f"ASD {ch_label} [strain / sqrt(Hz)]")
    ax_fd.set_title(
        f"Combined smoke-test injection — FD ASD ({ch_label} channel)"
    )
    ax_fd.grid(True, which="both", alpha=0.3)
    ax_fd.legend(loc="upper left", fontsize=8, ncols=2)
    fig_fd.tight_layout()
    fd_path = os.path.join(out_dir, "combined_injections_fd.png")
    fig_fd.savefig(fd_path, dpi=150, bbox_inches="tight")
    plt.close(fig_fd)
    print(f"[plot] wrote {fd_path}")


if __name__ == "__main__":
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument(
        "--out-dir",
        default="./gf_output/combined_smoke_test_artifacts",
        help="Directory to write the PNGs into.",
    )
    parser.add_argument(
        "--sangria-path",
        default=DEFAULT_SANGRIA_PATH,
        help="Path to the Sangria HDF5 file.",
    )
    parser.add_argument(
        "--noise-seed",
        type=int,
        default=12345,
        help="RNG seed for the synthetic FD noise (match the processor for reproducibility).",
    )
    parser.add_argument(
        "--channel",
        type=int,
        choices=[0, 1, 2],
        default=0,
        help="Channel index to plot (0=X, 1=Y, 2=Z).",
    )
    args = parser.parse_args()

    main(
        out_dir=args.out_dir,
        sangria_path=args.sangria_path,
        noise_seed=args.noise_seed,
        channel_index=args.channel,
    )
