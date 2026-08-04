#!/usr/bin/env python
"""What IS the T-channel residual? Amplitude error, timing error, or new structure?

Established so far:
  * mojito's stored A2/E2/T2 == AET(X2,Y2,Z2) to 1e-13 -> the basis is not at fault;
  * both MBH responses match the data in A/E at ~2e-4 relative amplitude but only
    ~1e-2 in T, and T dominates <r|r> (94-99.9% of the likelihood cost);
  * the XYZ residual is NOT common-mode (it is >99% differential in raw power) --
    T is amplified purely by its ~1e-5 PSD.

T is the near-null combination: the leading GW response cancels, so what survives is
set by the arm-length asymmetry and the exact TDI-2 delay chain. This script asks what
functional form the T residual takes.

Model tested (least squares, in band):

    r_T(t)  ~  alpha * T_data(t)  +  beta * dT_data/dt(t)

  alpha -> fractional AMPLITUDE error in the modelled T response
  beta  -> effective TIME offset (since h(t+b) ~ h + b*dh/dt)

Reported:
  1. alpha, beta and the fraction of r_T variance the 2-parameter model explains.
     A high explained fraction means T is mis-modelled by a simple scale/timing
     error; a low one means genuinely different structure.
  2. Frequency accumulation of <r|r>_T -- the cumulative fraction vs f, which
     separates a low-frequency (orbit/arm-length) origin from a broadband one.
  3. Time localisation of r_T (merger vs inspiral), per response.
  4. Correlation between the LEGACY and ON-THE-FLY T residuals. High correlation =>
     a shared component from the common TDI/orbit machinery rather than the
     response-specific waveform path.

Usage::

    MBHB_ID=19 python mbh_t_channel_forensics.py
"""

from __future__ import annotations

import os

import numpy as np

os.environ.setdefault("MPLBACKEND", "Agg")

MBHB_ID = int(os.environ.get("MBHB_ID", "19"))
SENS_MODEL = os.environ.get("SENS_MODEL", "scirdv1")
TUKEY_ALPHA = float(os.environ.get("TUKEY_ALPHA", "0.05"))
F_MIN, F_MAX = 1e-4, 2.5e-2
SRC = os.environ.get("TD_NPZ", f"/tmp/mbh_three_way_td_id{MBHB_ID}.npz")
OUT = os.environ.get("PLOT", f"/tmp/mbh_t_forensics_id{MBHB_ID}.png")


def banner(s):
    print("\n" + "=" * 78 + f"\n {s}\n" + "=" * 78, flush=True)


def main():
    import matplotlib.pyplot as plt
    from scipy.signal.windows import tukey
    from lisatools.domains import FDSettings, TDSettings, TDSignal
    from lisatools.sensitivity import T2TDISens, SensitivityMatrix
    from lisatools.utils.utility import AET

    z = np.load(SRC)
    Dx, Ax, Bx = z["D"], z["A"], z["B"]
    dt = float(z["dt"])
    n = Dx.shape[-1]
    merger_idx = int(round((float(z["abs_merger"]) - float(z["window_t0"])) / dt))
    to_aet = lambda a: np.asarray(AET(a[0], a[1], a[2]))
    Dt = to_aet(Dx)[2]
    resid = {"legacy": to_aet(Ax)[2] - Dt, "on-the-fly": to_aet(Bx)[2] - Dt}

    print(f"id={MBHB_ID}  N={n}  dt={dt}s  merger idx={merger_idx} "
          f"({100*merger_idx/n:.1f}%)  rms(T_data)={np.sqrt(np.mean(Dt**2)):.6e}",
          flush=True)

    # in-band mask for the regression (band-limit before fitting so out-of-band
    # numerical junk does not drive the fit)
    f_all = np.fft.rfftfreq(n, dt)
    keep = (f_all >= F_MIN) & (f_all <= F_MAX)

    def bandlimit(x):
        F = np.fft.rfft(x)
        F[~keep] = 0.0
        return np.fft.irfft(F, n=n)

    Dt_b = bandlimit(Dt)
    dDt_b = np.gradient(Dt_b, dt)

    win = tukey(n, TUKEY_ALPHA)
    td = TDSettings(N=n, dt=dt, t0=float(z["window_t0"]), force_backend="cpu")
    fd = FDSettings(N=n // 2 + 1, df=1.0 / (n * dt), min_freq=F_MIN, max_freq=F_MAX,
                    force_backend="cpu")
    sm = SensitivityMatrix(fd, [T2TDISens], model=SENS_MODEL)
    psd = np.asarray(sm.sens_mat).squeeze()
    f_fd = np.asarray(fd.f_arr)

    cum = {}
    for label, r in resid.items():
        banner(f"{label}: T-channel residual forensics")
        rb = bandlimit(r)

        # --- 1. amplitude / timing regression
        M = np.vstack([Dt_b, dDt_b]).T
        coef, *_ = np.linalg.lstsq(M, rb, rcond=None)
        alpha, beta = float(coef[0]), float(coef[1])
        pred = M @ coef
        expl = 1.0 - float(np.sum((rb - pred) ** 2) / np.sum(rb ** 2))
        print(f"  alpha (fractional amplitude error) = {alpha:+.6e}", flush=True)
        print(f"  beta  (effective time offset)      = {beta:+.6e} s", flush=True)
        print(f"  variance of r_T explained by [alpha*T + beta*dT/dt] = "
              f"{100*expl:7.3f}%", flush=True)
        # each alone
        for nm, col in (("alpha only (scale)", Dt_b), ("beta only (timing)", dDt_b)):
            c = float(np.dot(col, rb) / np.dot(col, col))
            e = 1.0 - float(np.sum((rb - c * col) ** 2) / np.sum(rb ** 2))
            print(f"    {nm:<20} explains {100*e:7.3f}%  (coef={c:+.6e})", flush=True)

        # --- 2. frequency accumulation of <r|r>_T
        rsig = TDSignal(np.ascontiguousarray(r[None, :]), td).transform(fd, window=win)
        R = np.asarray(rsig.arr).squeeze()
        integ = np.abs(R) ** 2 / psd
        c = np.cumsum(integ)
        c = c / c[-1]
        cum[label] = c
        for target in (0.1, 0.5, 0.9, 0.99):
            i = int(np.searchsorted(c, target))
            i = min(i, len(f_fd) - 1)
            print(f"    {100*target:5.1f}% of <r|r>_T accumulated below "
                  f"{f_fd[i]:.5e} Hz", flush=True)

        # --- 3. time localisation
        w = int(round(86400.0 / dt))
        lo, hi = max(0, merger_idx - w), min(n, merger_idx + w)
        fr = float(np.sum(rb[lo:hi] ** 2) / np.sum(rb ** 2))
        print(f"  r_T power within +/-1 d of merger = {100*fr:.4f}%", flush=True)

    # --- 4. legacy vs on-the-fly T residual correlation
    ra, rb_ = bandlimit(resid["legacy"]), bandlimit(resid["on-the-fly"])
    print(f"\n  corr(legacy r_T, on-the-fly r_T) = "
          f"{float(np.corrcoef(ra, rb_)[0,1]):+.6f}", flush=True)
    print(f"  rms ratio on-the-fly/legacy      = "
          f"{float(np.sqrt(np.mean(rb_**2)/np.mean(ra**2))):.6f}", flush=True)

    fig, ax = plt.subplots(1, 2, figsize=(14, 4.5))
    tdays = np.arange(n) * dt / 86400.0
    for label, r in resid.items():
        ax[0].plot(tdays, bandlimit(r), lw=0.6, label=f"r_T {label}")
    ax[0].plot(tdays, Dt_b, lw=0.6, color="k", alpha=0.35, label="T data")
    ax[0].axvline(merger_idx * dt / 86400.0, color="k", ls=":", alpha=0.5)
    ax[0].set_xlabel("days into window"); ax[0].set_title("T-channel residual, TD")
    ax[0].legend(fontsize=8)
    for label, c in cum.items():
        ax[1].semilogx(f_fd, c, lw=1.0, label=label)
    ax[1].set_xlabel("f [Hz]"); ax[1].set_ylabel("cumulative fraction of <r|r>_T")
    ax[1].set_title("where the T cost accumulates"); ax[1].legend(fontsize=8)
    ax[1].grid(alpha=0.3)
    fig.suptitle(f"MBHB id={MBHB_ID}: T-channel forensics")
    fig.tight_layout(rect=[0, 0, 1, 0.95])
    fig.savefig(OUT, dpi=105)
    plt.close(fig)
    print(f"\nDONE.  plot -> {OUT}", flush=True)


if __name__ == "__main__":
    main()
