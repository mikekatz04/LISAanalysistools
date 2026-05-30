"""Diagnostic plotter for the combined smoke-test injection.

Builds the same components the
:class:`SangriaPlusInjectionsProcessingStep` data processor sums into the
smoke-test data:

* Sangria sky-only slice (GBs + galactic foreground, ``mbhb`` removed).
* Synthetic correlated FD instrument noise (default seed = 12345).
* Synthetic EMRI TD waveforms from the shared response wrapper.
* Synthetic SOBBH TD waveforms from the shared response wrapper.
* Synthetic MBH (phentax) TD waveforms from the shared response wrapper.

Then plots three figures:

* ``combined_injections_td.png`` — six-panel TD strip chart of the X
  channel, one panel per coarse component plus the combined sum.
* ``combined_injections_fd.png`` — log-log ASD of the X channel for every
  component (each EMRI / SOBBH / MBH source on its own line) plus GB,
  noise, and the combined sum.
* ``combined_injections_wdm.png`` — six-panel WDM scalogram of ``|coeff|``
  on the X channel for each coarse component, using the same WDM grid
  the engine consumes via ``DOMAIN_CHOICE``.

Run::

    python LISAanalysistools/global_fit_input/plot_combined_injections.py
"""

from __future__ import annotations

import argparse
import os
import subprocess
import sys
import tempfile

import matplotlib.pyplot as plt
import numpy as np

# Resolve sibling imports the same way ``run_global.py`` does — by adding
# the directory to ``sys.path`` before pulling from the smoke-test module.
_HERE = os.path.dirname(os.path.abspath(__file__))
if _HERE not in sys.path:
    sys.path.insert(0, _HERE)

# Heavy waveform packages (FEW for EMRI, fastlisaresponse+JAX for SOBBH,
# phentax+JAX for MBH) interact badly when all three are loaded into one
# process: in practice the first call after the third package initialises
# silently kills the interpreter (exit 0, no traceback). To avoid that we
# generate each waveform family in its own fresh subprocess via
# :func:`_generate_in_subprocess` below and only load lightweight things
# (numpy, matplotlib, the Sangria loader, the WDM transform) into the
# main process.
from lisatools.domains import TDSettings, TDSignal
from lisatools.globalfit.preprocessing import SangriaDataLoader

from combined_gb_psd_emri_sobbh_global_fit_settings import (  # noqa: E402
    DOMAIN_CHOICE,
    DT,
    EMRI_INJECTIONS_FULL_BASIS,
    MBH_INJECTIONS_FULL_BASIS,
    SOBBH_INJECTIONS_FULL_BASIS,
    T_START,
    TOBS,
    _generate_correlated_fd_noise,
    _pad_or_clip,
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


_SUBPROC_GEN_TEMPLATE = """\
import os, sys, numpy as np
sys.path.insert(0, {plot_dir!r})
from fastlisaresponse.tdiconfig import TDIConfig
from combined_gb_psd_emri_sobbh_global_fit_settings import (
    DT, TOBS, T_START, _pad_or_clip,
    EMRI_INJECTIONS_FULL_BASIS,
    SOBBH_INJECTIONS_FULL_BASIS,
    MBH_INJECTIONS_FULL_BASIS,
    get_emri_response_wrapper,
    get_sobbh_response_wrapper,
    get_mbh_phentax_response_wrapper,
)

KIND = {kind!r}
OUT = {out_path!r}
target_N = int(round(TOBS / DT))
tdi = TDIConfig('2nd generation', force_backend='cpu')

if KIND == 'emri':
    wg = get_emri_response_wrapper(Tobs=TOBS, dt=DT, t_start=T_START,
        tdi_config=tdi, tdi_chan='XYZ', role='injection', force_backend='cpu')
    rows = EMRI_INJECTIONS_FULL_BASIS
elif KIND == 'sobbh':
    wg = get_sobbh_response_wrapper(Tobs=TOBS, dt=DT, t_start=T_START,
        tdi_config=tdi, tdi_chan='XYZ', role='injection', force_backend='cpu')
    rows = SOBBH_INJECTIONS_FULL_BASIS
elif KIND == 'mbh':
    wg = get_mbh_phentax_response_wrapper(Tobs=TOBS, dt=DT, t_start=T_START,
        tdi_config=tdi, tdi_chan='XYZ', role='injection', force_backend='cpu')
    rows = MBH_INJECTIONS_FULL_BASIS
else:
    raise SystemExit(f'unknown KIND {{KIND!r}}')

out = np.zeros((len(rows), 3, target_N), dtype=np.float64)
for i, params in enumerate(rows):
    sig = np.asarray(wg(*params, convert_to_ra_dec=False))
    out[i] = _pad_or_clip(np.atleast_2d(sig)[:3], target_N)
    print(f'[subproc/{{KIND}}] source {{i+1}}/{{len(rows)}} done', flush=True)

np.save(OUT, out)
print(f'[subproc/{{KIND}}] wrote {{OUT}}', flush=True)
"""


def _generate_in_subprocess(kind: str, n_sources: int, target_N: int) -> list[np.ndarray]:
    """Run ``kind in {'emri', 'sobbh', 'mbh'}`` waveform generation in a
    fresh subprocess; load the resulting ``(n_sources, 3, target_N)`` array
    back and split into a list of per-source ``(3, target_N)`` arrays.

    Using a subprocess keeps the FEW / fastlisaresponse / phentax+JAX state
    isolated so that loading all three families in the same parent process
    (which has been observed to silently kill the interpreter on the first
    call after the third initialises) does not happen here.
    """
    with tempfile.NamedTemporaryFile(suffix=".npy", delete=False) as tmp:
        out_path = tmp.name
    try:
        plot_dir = os.path.dirname(os.path.abspath(__file__))
        code = _SUBPROC_GEN_TEMPLATE.format(
            plot_dir=plot_dir, kind=kind, out_path=out_path
        )
        print(f"[plot] launching subprocess for {kind} (3 sources)", flush=True)
        env = dict(os.environ, PYTHONUNBUFFERED="1")
        subprocess.run(
            [sys.executable, "-u", "-c", code], check=True, env=env
        )
        arr = np.load(out_path)
    finally:
        try:
            os.remove(out_path)
        except OSError:
            pass
    assert arr.shape == (n_sources, 3, target_N), (
        f"subprocess returned shape {arr.shape}, expected "
        f"({n_sources}, 3, {target_N})"
    )
    return [arr[i] for i in range(n_sources)]


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

    # Each waveform family runs in its own short-lived subprocess so the
    # FEW / fastlisaresponse / phentax+JAX initialisations don't pile up
    # in the parent (which silently crashes once all three are loaded).
    mbh_tds = _generate_in_subprocess(
        "mbh", MBH_INJECTIONS_FULL_BASIS.shape[0], target_N
    )
    mbh_td_sum = np.sum(mbh_tds, axis=0)

    emri_tds = _generate_in_subprocess(
        "emri", EMRI_INJECTIONS_FULL_BASIS.shape[0], target_N
    )
    emri_td_sum = np.sum(emri_tds, axis=0)

    sobbh_tds = _generate_in_subprocess(
        "sobbh", SOBBH_INJECTIONS_FULL_BASIS.shape[0], target_N
    )
    sobbh_td_sum = np.sum(sobbh_tds, axis=0)

    combined_td = gb_td + noise_td + emri_td_sum + sobbh_td_sum + mbh_td_sum

    # TD figure: one panel per coarse component (per-source EMRI / SOBBH /
    # MBH streams overlap so far at LISA noise level that summing is more
    # informative). FD plot below shows each source individually.
    components = [
        ("Sangria GBs (dgb+igb+vgb)", gb_td),
        ("FD instrument noise", noise_td),
        (f"Synthetic EMRI (sum of {len(emri_tds)})", emri_td_sum),
        (f"Synthetic SOBBH (sum of {len(sobbh_tds)})", sobbh_td_sum),
        (f"Synthetic MBH phentax (sum of {len(mbh_tds)})", mbh_td_sum),
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
    # Plot every individual EMRI / SOBBH / MBH source as its own line so
    # the different sky / phase / distance / merger-time realisations are
    # visible, plus the other components and the combined sum.
    fd_lines = [
        ("Sangria GBs", gb_td, dict(color="C2", lw=0.6)),
        ("FD noise", noise_td, dict(color="C7", lw=0.6, alpha=0.7)),
    ]
    for i, td_i in enumerate(emri_tds):
        fd_lines.append((f"EMRI {i+1}", td_i, dict(color="C0", lw=0.6, alpha=0.7 + 0.1 * i)))
    for i, td_i in enumerate(sobbh_tds):
        fd_lines.append((f"SOBBH {i+1}", td_i, dict(color="C3", lw=0.6, alpha=0.7 + 0.1 * i)))
    for i, td_i in enumerate(mbh_tds):
        fd_lines.append((f"MBH {i+1}", td_i, dict(color="C4", lw=0.6, alpha=0.7 + 0.1 * i)))
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

    # --- WDM figure ------------------------------------------------------
    # Project each TD component through the run's WDM grid (resolved via
    # the same ``DOMAIN_CHOICE`` factory the engine consumes) and render
    # a scalogram per coarse component, X channel only. Same six panels
    # as the TD figure (Sangria GBs, FD noise, EMRI sum, SOBBH sum, MBH
    # sum, combined) so the views line up visually.
    print("[plot] Projecting components to WDM domain", flush=True)
    wdm_settings = DOMAIN_CHOICE(times=times, dt=DT, force_backend="cpu")
    td_settings = TDSettings(N=target_N, dt=DT, force_backend="cpu")

    def _wdm_project(td: np.ndarray) -> np.ndarray:
        """Return the ``(Nf_active, Nt_active)`` X-channel WDM coefficients."""
        sig = TDSignal(td, td_settings).transform(wdm_settings, window=None)
        # WDMSignal.arr has shape (nchannels, Nf_active, Nt_active)
        return np.asarray(sig.arr[channel_index])

    wdm_components = []
    for label, td in components:
        print(f"[plot]   WDM transform: {label}", flush=True)
        wdm_components.append((label, _wdm_project(td)))

    # Common log-color scale across panels: pick (vmin, vmax) from the
    # combined-injection panel so individual sources stay legible against
    # the overall dynamic range. Floor at 1e-30 to avoid log(0).
    combined_panel_idx = next(
        i for i, (lbl, _) in enumerate(wdm_components)
        if lbl.startswith("Combined")
    )
    combined_abs = np.abs(wdm_components[combined_panel_idx][1])
    vmax = float(combined_abs.max()) if combined_abs.size else 1.0
    vmin = max(vmax * 1e-6, 1e-30)

    t_edges_days = np.asarray(wdm_settings.t_arr_edges, dtype=float) / 86400.0
    f_edges_mHz = np.asarray(wdm_settings.f_arr_edges, dtype=float) * 1e3
    # Nudge the f=0 edge so log y-scaling stays well-defined; we
    # explicitly ylim away from it below anyway.
    f_lo = float(f_edges_mHz[f_edges_mHz > 0].min()) if np.any(f_edges_mHz > 0) else 1e-3
    f_edges_mHz_plot = f_edges_mHz.copy()
    f_edges_mHz_plot[f_edges_mHz_plot <= 0.0] = f_lo * 1e-3

    from matplotlib.colors import LogNorm

    fig_wdm, axes_wdm = plt.subplots(
        len(wdm_components), 1,
        figsize=(11, 2.4 * len(wdm_components)),
        sharex=True, sharey=True,
    )
    for ax, (label, coeffs) in zip(axes_wdm, wdm_components):
        z = np.abs(coeffs)
        # pcolormesh expects (Ny, Nx) = (Nf, Nt); ``coeffs.shape`` is
        # already (Nf_active, Nt_active).
        im = ax.pcolormesh(
            t_edges_days, f_edges_mHz_plot, z,
            norm=LogNorm(vmin=vmin, vmax=vmax),
            cmap="viridis", shading="flat",
        )
        ax.set_yscale("log")
        ax.set_ylim(f_lo, float(f_edges_mHz_plot[-1]))
        ax.set_ylabel(f"{label}\nf [mHz]")
        plt.colorbar(im, ax=ax, fraction=0.02, pad=0.01)
    axes_wdm[-1].set_xlabel("time [days]")
    fig_wdm.suptitle(
        f"Combined smoke-test injection — WDM |coeff| ({ch_label} channel, "
        f"Nf={wdm_settings.Nf}, Nt={wdm_settings.Nt}, dt={DT:.0f} s)"
    )
    fig_wdm.tight_layout()
    wdm_path = os.path.join(out_dir, "combined_injections_wdm.png")
    fig_wdm.savefig(wdm_path, dpi=150, bbox_inches="tight")
    plt.close(fig_wdm)
    print(f"[plot] wrote {wdm_path}")


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
