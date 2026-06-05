#!/usr/bin/env python
"""
Heterodyned frequency-domain TDI on the fly for Galactic binaries (prototype).

Strategy
--------
A GB time-domain TDI signal x(t) = Re[z(t)] where the complex carrier
z(t) = A(t) * exp(-1j * (tdi_phase(t) + phase_ref(t)))
has its positive-frequency content at f ~ +f0.  Taking the conjugate
zhat(t) = A * exp(+1j * (tdi_phase + phase_ref))  flips the dominant
support to f ~ +f0 of zhat. Heterodyning by the on-grid carrier f0_grid:

    s(tau) = zhat(tau) * exp(-2*pi*i * f0_grid * tau)
           = A(tau) * exp(+1j * (tdi_phase(tau) + phase_ref(tau)
                                 - 2*pi*f0_grid*tau))

is a *slowly-varying* function of local time tau = t - t_start.  Only the
small LISA response modulations (yearly Doppler, 25-day rotation) and the
fdot / (f0 - f0_grid) drift remain in the phase, so s(tau) can be sampled
sparsely (~256 pts/year) and FFT'd cheaply.

Then for f > 0 near f0:
    X(f) ~ (1/2) * S(f - f0_grid)
where the (1/2) comes from x = (z + z*)/2 and the term involving the
conjugate Z(f) is negligible far from -f0.  When f0_grid = k_f0 * df is on
the dense rfft grid, the sparse FFT bins map one-to-one onto dense bins
k_f0 + m, with the only frequency-bin offset being f0 - f0_grid (kept
inside `s` as a slow ramp).

We compare against the lisatools convention:
    X_ref(f_k) = rfft(x_dense)[k] * dt,   f_k = k / (N * dt)
on a dense time grid that uses the same t0 and the same Tobs = N * dt.

Run:
    python gb_heterodyne_fd.py
"""

from __future__ import annotations

import numpy as np
import matplotlib.pyplot as plt

from lisatools.detector import EqualArmlengthOrbits
from lisatools.utils.constants import YRSID_SI
from lisatools.domains import TDSettings, TDSignal, FDSettings, FDSignal

from fastlisaresponse.tdiconfig import TDIConfig
from fastlisaresponse.tdionfly import GBTDIonTheFly


BACKEND = "cpu"
xp = np


def make_dense_reference(params, t_arr_dense, Tobs, t_ref, dt, gb_kwargs):
    """Generate dense-time TDI and FFT via lisatools TDSignal.fft."""
    gb_dense = GBTDIonTheFly(
        t_arr_dense, Tobs, t_ref, 1.0 / dt, 1, **gb_kwargs
    )
    out = gb_dense(*params, convert_to_ra_dec=False, return_spline=False)
    x_dense = np.stack(
        [np.asarray(out.X[0]).real,
         np.asarray(out.Y[0]).real,
         np.asarray(out.Z[0]).real],
        axis=0,
    )  # (3, N_dense)

    td_set = TDSettings(x_dense.shape[-1], dt, t0=t_arr_dense[0], force_backend=BACKEND)
    fd = TDSignal(x_dense, td_set).fft()  # rfft * dt; no window
    return x_dense, fd.arr, fd.f_arr


def make_heterodyne_fd(params, t_arr_sparse, t_start, Tobs, t_ref, dt, df,
                      gb_kwargs):
    """
    Compute heterodyned FD TDI on a sparse time grid.

    Returns
    -------
    f_het_arr : (N_sparse,) absolute frequencies (Hz) on the dense rfft grid
    X_het     : (3, N_sparse) complex heterodyne FD prediction at those bins
    f0_grid   : the on-grid carrier used for heterodyning (Hz)
    k_f0      : the dense rfft bin closest to f0
    """
    f0 = float(params[1][0])  # params is [amp, f0, fdot, fddot, phi0, inc, psi, lam, beta]

    gb_sparse = GBTDIonTheFly(
        t_arr_sparse, Tobs, t_ref, 1.0 / dt, 1, **gb_kwargs
    )
    out_sparse = gb_sparse(*params, convert_to_ra_dec=False, return_spline=False)

    tdi_amp = np.asarray(out_sparse.tdi_amp[0])      # (3, N_sparse)
    tdi_phase = np.asarray(out_sparse.tdi_phase[0])  # (3, N_sparse)
    phase_ref = np.asarray(out_sparse.phase_ref[0])  # (N_sparse,)

    # snap f0 to the dense rfft grid: f0_grid = k_f0 * df
    k_f0 = int(round(f0 / df))
    f0_grid = k_f0 * df

    # local time tau = t_abs - t_start
    tau = np.asarray(t_arr_sparse) - t_start
    dt_sparse = Tobs / len(tau)

    # slow positive-freq complex signal
    # s(tau) = A * exp(+1j * (tdi_phase + phase_ref - 2*pi*f0_grid*tau))
    carrier = 2.0 * np.pi * f0_grid * tau
    s_slow = tdi_amp * np.exp(
        +1j * (tdi_phase + phase_ref[None, :] - carrier[None, :])
    )

    # ∫ s(tau) exp(-2*pi*i*g*tau) dtau  ~  dt_sparse * fft(s)[m] at g = m / Tobs
    S = np.fft.fft(s_slow, axis=-1) * dt_sparse  # (3, N_sparse)

    # bin offsets m in [-N/2, N/2-1] for each sparse FFT bin
    m_arr = np.fft.fftfreq(len(tau), d=1.0 / len(tau)).astype(int)

    # X_het(f_k) ~ (1/2) * S(f_k - f0_grid)
    X_het = 0.5 * S

    # absolute frequency bins on the dense grid: f = (k_f0 + m) * df
    f_het_arr = (k_f0 + m_arr) * df

    return f_het_arr, X_het, f0_grid, k_f0


def heterodyne_to_dense_rfft(X_het, f_het, df, N_dense):
    """
    Pack the sparse heterodyne FD values onto the full dense rfft grid
    (length N_dense // 2 + 1) and return that array. Bins outside the
    sparse support are zero.
    """
    n_rfft = N_dense // 2 + 1
    X_full = np.zeros((X_het.shape[0], n_rfft), dtype=complex)
    k_het = np.round(f_het / df).astype(int)
    mask = (k_het >= 0) & (k_het < n_rfft)
    X_full[:, k_het[mask]] = X_het[:, mask]
    return X_full


def compare_band(f_ref, X_ref, f_het, X_het, f0, half_bw_hz):
    """Restrict to |f - f0| <= half_bw_hz and compute mismatch metrics."""
    df = f_ref[1] - f_ref[0]

    # band on the dense ref grid
    mask_ref = np.abs(f_ref - f0) <= half_bw_hz
    f_band = f_ref[mask_ref]
    X_ref_band = X_ref[:, mask_ref]

    # pull X_het at exactly the same dense bins
    # f_het = (k_f0 + m)*df.  Build a lookup from f to value.
    # round to nearest int bin to match.
    k_het = np.round(f_het / df).astype(int)
    k_band = np.round(f_band / df).astype(int)
    # build map from k -> column of X_het
    order = np.argsort(k_het)
    k_het_sorted = k_het[order]
    # for every k_band, find its location in k_het_sorted
    pos = np.searchsorted(k_het_sorted, k_band)
    # safety: any out-of-range gets clipped to last; we'll then filter by equality
    pos = np.clip(pos, 0, len(k_het_sorted) - 1)
    valid = (k_het_sorted[pos] == k_band)
    if not np.all(valid):
        # restrict to bins available on the sparse grid
        f_band = f_band[valid]
        X_ref_band = X_ref_band[:, valid]
        pos = pos[valid]
        k_band = k_band[valid]
    X_het_band = X_het[:, order][:, pos]

    err = X_het_band - X_ref_band
    max_abs = np.max(np.abs(err))
    peak_ref = np.max(np.abs(X_ref_band))
    rel = max_abs / peak_ref

    # overlap-like metric per channel (no PSD weighting; flat metric)
    num = np.sum(X_het_band * np.conj(X_ref_band), axis=-1)
    den_a = np.sqrt(np.sum(np.abs(X_het_band) ** 2, axis=-1))
    den_b = np.sqrt(np.sum(np.abs(X_ref_band) ** 2, axis=-1))
    overlap = np.real(num) / (den_a * den_b + 1e-300)
    mismatch = 1.0 - overlap

    return dict(
        f_band=f_band, X_ref_band=X_ref_band, X_het_band=X_het_band,
        max_abs=max_abs, peak_ref=peak_ref, rel_max=rel,
        overlap=overlap, mismatch=mismatch,
    )


def main():
    import argparse
    parser = argparse.ArgumentParser()
    parser.add_argument("--pts-per-year", type=int, default=256)
    parser.add_argument("--sweep", action="store_true",
                        help="run a convergence sweep over pts/year")
    parser.add_argument("--dt", type=float, default=15.0)
    parser.add_argument("--f0-mhz", type=float, default=20.0)
    args = parser.parse_args()

    # ---- time grid ------------------------------------------------------
    dt = args.dt  # s
    # 1 year observation, snap N so Tobs is an integer multiple of dt
    Tobs_nom = 1.0 * YRSID_SI
    N_dense = int(round(Tobs_nom / dt))
    Tobs = N_dense * dt
    df = 1.0 / Tobs

    # start 6 months into the orbit so we are well away from orbit endpoints
    t_start = int(0.5 * YRSID_SI / dt) * dt
    t_ref = t_start  # GB phase referenced to start of observation
    t_arr_dense = t_start + np.arange(N_dense) * dt

    # sparse grid
    pts_per_year = args.pts_per_year
    N_sparse = int(round(pts_per_year * Tobs / YRSID_SI))
    # use linspace_endpoint-free arange to keep dt_sparse = Tobs/N_sparse
    dt_sparse = Tobs / N_sparse
    t_arr_sparse = t_start + np.arange(N_sparse) * dt_sparse

    # ---- TDI / orbits ---------------------------------------------------
    tdi_config = TDIConfig("2nd generation")
    orbits = EqualArmlengthOrbits(force_backend=BACKEND)
    gb_kwargs = dict(
        tdi_config=tdi_config, orbits=orbits, tdi_chan="XYZ",
        force_backend=BACKEND,
    )

    # ---- GB parameters --------------------------------------------------
    amp = np.array([8.0e-23])
    f0 = np.array([args.f0_mhz * 1e-3])
    fdot = np.array([1.0e-14])
    fddot = np.array([0.0])
    phi0 = np.array([2.09802430298])
    inc = np.array([0.23984234])
    psi = np.array([1.234019814])
    lam = np.array([4.09808143])
    beta = np.array([0.090])
    params = (amp, f0, fdot, fddot, phi0, inc, psi, lam, beta)

    print(f"dt={dt}s, Tobs={Tobs:.3e}s ({Tobs/YRSID_SI:.3f} yr), df={df:.3e} Hz")
    print(f"N_dense={N_dense}, N_sparse={N_sparse} ({pts_per_year} pts/year)")
    print(f"f0={f0[0]:.6e} Hz, dt_sparse={dt_sparse:.2f}s, "
          f"sparse Nyquist={0.5/dt_sparse:.3e} Hz")

    # ---- dense reference ------------------------------------------------
    import time as _t
    t0 = _t.time()
    x_dense, X_ref, f_ref = make_dense_reference(
        params, t_arr_dense, Tobs, t_ref, dt, gb_kwargs
    )
    print(f"dense  TDI: shape={x_dense.shape}, took {_t.time()-t0:.2f}s")

    # ---- optional convergence sweep ------------------------------------
    if args.sweep:
        print()
        print("convergence sweep: TD-mismatch vs sparse points/year")
        print(f"{'pts/yr':>8} {'N_sparse':>8} {'time[s]':>8}  "
              f"{'TD mm[X]':>10} {'TD mm[Y]':>10} {'TD mm[Z]':>10}")
        for pts in [64, 128, 256, 512, 1024, 2048]:
            N_s = int(round(pts * Tobs / YRSID_SI))
            dt_s = Tobs / N_s
            t_s = t_start + np.arange(N_s) * dt_s
            tt = _t.time()
            f_h, X_h, f0g, kf0 = make_heterodyne_fd(
                params, t_s, t_start, Tobs, t_ref, dt, df, gb_kwargs
            )
            X_full_s = heterodyne_to_dense_rfft(X_h, f_h, df, N_dense)
            x_rec_s = np.fft.irfft(X_full_s, n=N_dense, axis=-1) / dt
            i0 = int(0.05 * N_dense)
            i1 = N_dense - i0
            mm = []
            for c in range(3):
                num_l2 = np.sum(x_rec_s[c, i0:i1] * x_dense[c, i0:i1])
                da = np.sqrt(np.sum(x_rec_s[c, i0:i1] ** 2))
                db = np.sqrt(np.sum(x_dense[c, i0:i1] ** 2))
                mm.append(1.0 - num_l2 / (da * db + 1e-300))
            tt = _t.time() - tt
            print(f"{pts:>8d} {N_s:>8d} {tt:>8.2f}  "
                  f"{mm[0]:>10.3e} {mm[1]:>10.3e} {mm[2]:>10.3e}")
        return

    # ---- heterodyne FD --------------------------------------------------
    t0 = _t.time()
    f_het, X_het, f0_grid, k_f0 = make_heterodyne_fd(
        params, t_arr_sparse, t_start, Tobs, t_ref, dt, df, gb_kwargs
    )
    print(f"sparse TDI: shape={X_het.shape}, took {_t.time()-t0:.2f}s")
    print(f"f0_grid={f0_grid:.6e} Hz   (df={df:.3e}, f0-f0_grid={f0[0]-f0_grid:.3e})")
    print(f"k_f0={k_f0}, sparse bins available around f0_grid")

    # ---- compare in a band around f0 -----------------------------------
    half_bw = 0.4 / dt_sparse  # 80 % of sparse Nyquist
    res = compare_band(f_ref, X_ref, f_het, X_het, f0[0], half_bw)
    print()
    print(f"comparison band: |f - f0| <= {half_bw:.3e} Hz "
          f"({res['f_band'].size} bins)")
    print(f"  max |X_het - X_ref|             = {res['max_abs']:.3e}")
    print(f"  peak |X_ref|                    = {res['peak_ref']:.3e}")
    print(f"  relative max error              = {res['rel_max']:.3e}")
    for chan, name in enumerate("XYZ"):
        print(f"  overlap[{name}] = {res['overlap'][chan]:.10f}, "
              f"mismatch[{name}] = {res['mismatch'][chan]:.3e}")

    # ---- direct time-domain comparison ----------------------------------
    # Pack heterodyne FD onto the dense rfft grid and irfft -> time series.
    # Bins outside the sparse support are zero, so the reconstruction keeps
    # only the narrow band around f0 -- exactly what a GB occupies.
    X_het_full = heterodyne_to_dense_rfft(X_het, f_het, df, N_dense)
    # lisatools fft: X = rfft(x)*dt  =>  x = irfft(X)/dt with n=N_dense
    x_recon = np.fft.irfft(X_het_full, n=N_dense, axis=-1) / dt  # (3, N_dense)

    # Compare in a central window to avoid first/last few samples where
    # the dense reference can have any TDIonTheFly edge effects.
    edge_frac = 0.05
    i0 = int(edge_frac * N_dense)
    i1 = N_dense - i0
    diff_td = x_recon[:, i0:i1] - x_dense[:, i0:i1]
    peak_td = np.max(np.abs(x_dense[:, i0:i1]))
    rel_td = np.max(np.abs(diff_td)) / peak_td
    # L2 mismatch per channel over the windowed range
    num_l2 = np.sum(x_recon[:, i0:i1] * x_dense[:, i0:i1], axis=-1)
    den_a = np.sqrt(np.sum(x_recon[:, i0:i1] ** 2, axis=-1))
    den_b = np.sqrt(np.sum(x_dense[:, i0:i1] ** 2, axis=-1))
    overlap_td = num_l2 / (den_a * den_b + 1e-300)
    mismatch_td = 1.0 - overlap_td

    print()
    print(f"direct time-domain comparison "
          f"(central {100*(1-2*edge_frac):.0f}% of samples):")
    print(f"  max |x_recon - x_dense|  = {np.max(np.abs(diff_td)):.3e}")
    print(f"  peak |x_dense|           = {peak_td:.3e}")
    print(f"  relative max error       = {rel_td:.3e}")
    for chan, name in enumerate("XYZ"):
        print(f"  TD overlap[{name}] = {overlap_td[chan]:.10f}, "
              f"mismatch[{name}] = {mismatch_td[chan]:.3e}")

    # ---- plots ----------------------------------------------------------
    fig, axes = plt.subplots(3, 1, figsize=(8, 9), sharex=True)
    for chan, name in enumerate("XYZ"):
        ax = axes[chan]
        ax.semilogy(res["f_band"], np.abs(res["X_ref_band"][chan]),
                    label=f"{name}_ref (dense rfft)", lw=1.5)
        ax.semilogy(res["f_band"], np.abs(res["X_het_band"][chan]),
                    label=f"{name}_het (sparse 256/yr)", lw=1.0, ls="--")
        ax.semilogy(
            res["f_band"],
            np.abs(res["X_het_band"][chan] - res["X_ref_band"][chan]) + 1e-300,
            label="abs error", lw=0.8, alpha=0.7,
        )
        ax.set_ylabel(f"|{name}(f)|")
        ax.legend(fontsize=8, loc="upper right")
        ax.grid(True, which="both", alpha=0.3)
    axes[-1].set_xlabel("f [Hz]")
    fig.suptitle(
        f"Heterodyne FD vs dense rfft   "
        f"(N_sparse={N_sparse}, N_dense={N_dense}, f0={f0[0]*1e3:.2f} mHz)"
    )
    fig.tight_layout()
    out_path = "/Users/mkatz/Research/lisa_sprint_2026/gb_heterodyne_fd.png"
    fig.savefig(out_path, dpi=120)
    print(f"\nsaved FD plot to {out_path}")

    # time-domain plot: small chunk in the middle of the observation
    mid = N_dense // 2
    nshow = min(2000, N_dense - mid)
    t_show = (t_arr_dense[mid:mid + nshow] - t_arr_dense[mid]) / 3600.0  # hours

    fig2, axes2 = plt.subplots(3, 1, figsize=(8, 9), sharex=True)
    for chan, name in enumerate("XYZ"):
        ax = axes2[chan]
        ax.plot(t_show, x_dense[chan, mid:mid + nshow],
                label=f"{name} dense (lisatools)", lw=1.4)
        ax.plot(t_show, x_recon[chan, mid:mid + nshow],
                label=f"{name} from heterodyne FD", lw=0.9, ls="--")
        ax.plot(
            t_show,
            (x_recon[chan, mid:mid + nshow] - x_dense[chan, mid:mid + nshow]),
            label="residual", lw=0.7, alpha=0.7,
        )
        ax.set_ylabel(f"{name}(t)")
        ax.legend(fontsize=8, loc="upper right")
        ax.grid(True, alpha=0.3)
    axes2[-1].set_xlabel("t - t_mid [hours]")
    fig2.suptitle(
        f"Time domain: dense lisatools vs heterodyne FD -> irfft "
        f"(N_sparse={N_sparse}, f0={f0[0]*1e3:.2f} mHz)"
    )
    fig2.tight_layout()
    out_path_td = "/Users/mkatz/Research/lisa_sprint_2026/gb_heterodyne_td.png"
    fig2.savefig(out_path_td, dpi=120)
    print(f"saved TD plot to {out_path_td}")


if __name__ == "__main__":
    main()
