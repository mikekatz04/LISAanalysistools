#!/usr/bin/env python
"""Back out the XYZ structure behind the AET result: is the MBH residual common-mode?

Context: ``mbh_aet_channel_split.py`` showed both MBH responses match mojito in A and
E at 1e-8..1e-6 while T is 1e-5..3e-2, for every source tested. Since

    A = (Z - X)/sqrt2      E = (X - 2Y + Z)/sqrt6      T = (X + Y + Z)/sqrt3

A and E are DIFFERENCES and T is the symmetric SUM. A residual that cancels in the
differences but survives in the sum must be COMMON-MODE: r_X ~ r_Y ~ r_Z. This script
tests that directly and characterises the common part.

Note the AET conversion is applied identically to data and template, so the T
discrepancy cannot be an artifact of our AET convention -- it is a real difference in
(X+Y+Z) between template and data. A channel PERMUTATION is also excluded a priori:
T is invariant under permuting X/Y/Z, so a permutation error would corrupt A/E (which
are clean) and leave T alone (which is not what we see).

Decomposition, per residual r_c = template_c - data_c:

    common_c  = mean_c(r)            (identical in all three channels)
    diff_c    = r_c - common         (sums to zero across channels)

    power fractions:  3|common|^2 / sum_c |r_c|^2   and its complement

Outputs
  1. per-channel residual RMS and the common/differential power split
  2. the 3x3 correlation matrix of (r_X, r_Y, r_Z) -- common-mode => all ~ +1
  3. spectra of the common vs differential parts
  4. time-domain localisation of the common part (is it at t_c, or everywhere?)
  5. ABSOLUTE noise-weighted <r|r> per AET channel -- i.e. what T actually COSTS in
     log-likelihood, as opposed to its normalised mismatch
  6. |data| per AET channel, so the T residual can be judged against T's own content

Usage::

    MBHB_ID=19 python mbh_xyz_residual_structure.py
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
OUT = os.environ.get("PLOT", f"/tmp/mbh_xyz_structure_id{MBHB_ID}.png")


def banner(s):
    print("\n" + "=" * 78 + f"\n {s}\n" + "=" * 78, flush=True)


def main():
    import matplotlib.pyplot as plt
    from scipy.signal.windows import tukey
    from lisatools.analysiscontainer import AnalysisContainer
    from lisatools.domains import FDSettings, TDSettings, TDSignal
    from lisatools.sensitivity import (
        A2TDISens, E2TDISens, T2TDISens, SensitivityMatrix,
    )
    from lisatools.utils.utility import AET

    z = np.load(SRC)
    Dx, Ax, Bx = z["D"], z["A"], z["B"]
    dt = float(z["dt"])
    n = Dx.shape[-1]
    merger_idx = int(round((float(z["abs_merger"]) - float(z["window_t0"])) / dt))
    print(f"id={MBHB_ID}  N={n}  dt={dt}s  window={n*dt/86400:.2f} d  "
          f"merger idx={merger_idx} ({100*merger_idx/n:.1f}%)  src={SRC}", flush=True)

    win = tukey(n, TUKEY_ALPHA)
    td = TDSettings(N=n, dt=dt, t0=float(z["window_t0"]), force_backend="cpu")
    f_all = np.fft.rfftfreq(n, dt)
    band = (f_all >= F_MIN) & (f_all <= F_MAX)

    results = {}
    for label, T_ in (("legacy", Ax), ("on-the-fly", Bx)):
        r = T_ - Dx                                   # (3, N) XYZ residual
        common = r.mean(axis=0)                       # (N,)
        diff = r - common[None, :]

        banner(f"{label}: XYZ residual decomposition")
        for i, c in enumerate("XYZ"):
            print(f"  rms(r_{c}) = {np.sqrt(np.mean(r[i]**2)):.6e}", flush=True)
        p_tot = float(np.sum(r ** 2))
        p_com = float(3.0 * np.sum(common ** 2))
        p_dif = float(np.sum(diff ** 2))
        print(f"  common-mode power fraction  = {100*p_com/p_tot:8.4f}%", flush=True)
        print(f"  differential power fraction = {100*p_dif/p_tot:8.4f}%", flush=True)
        print(f"  rms(common) = {np.sqrt(np.mean(common**2)):.6e}   "
              f"rms(diff_X) = {np.sqrt(np.mean(diff[0]**2)):.6e}", flush=True)

        C = np.corrcoef(r)
        print("  correlation matrix of (r_X, r_Y, r_Z)   [+1 everywhere => common-mode]:",
              flush=True)
        for i, c in enumerate("XYZ"):
            print("    " + c + "  " + "  ".join(f"{C[i, j]:+.5f}" for j in range(3)),
                  flush=True)

        # spectra
        Fc = np.abs(np.fft.rfft(common * win)) * dt
        Fd = np.abs(np.fft.rfft(diff[0] * win)) * dt
        Fdat = np.abs(np.fft.rfft(Dx[0] * win)) * dt

        # time localisation of the common part: fraction of |common|^2 within
        # +/-1 day of the merger
        w = int(round(86400.0 / dt))
        lo, hi = max(0, merger_idx - w), min(n, merger_idx + w)
        frac_merger = float(np.sum(common[lo:hi] ** 2) / np.sum(common ** 2))
        print(f"  common-mode power within +/-1 d of merger = {100*frac_merger:.4f}%",
              flush=True)

        # absolute noise-weighted <r|r> per AET channel (the logL cost)
        to_aet = lambda a: np.asarray(AET(a[0], a[1], a[2]))
        Daet, Taet = to_aet(Dx), to_aet(T_)
        fd = FDSettings(N=n // 2 + 1, df=1.0 / (n * dt), min_freq=F_MIN,
                        max_freq=F_MAX, force_backend="cpu")
        print("  absolute noise-weighted inner products (full band):", flush=True)
        tot_rr = 0.0
        for ci, (cn, sens) in enumerate((("A", A2TDISens), ("E", E2TDISens),
                                         ("T", T2TDISens))):
            dsig = TDSignal(np.ascontiguousarray(Daet[[ci]]), td).transform(fd, window=win)
            tsig = TDSignal(np.ascontiguousarray(Taet[[ci]]), td).transform(fd, window=win)
            sm = SensitivityMatrix(fd, [sens], model=SENS_MODEL)
            ac = AnalysisContainer(dsig, sm)
            d_d = float(np.ravel(np.asarray(ac.inner_product(complex=False)).real)[0])
            opt, _ = ac.template_snr(tsig)
            h_h = float(opt) ** 2
            d_h = complex(ac.non_marg_d_h).real
            rr = d_d + h_h - 2.0 * d_h
            tot_rr += rr
            print(f"    {cn}:  <d|d>={d_d:.6e}  <h|h>={h_h:.6e}  <r|r>={rr:.6e}   "
                  f"SNR_d={np.sqrt(max(d_d,0)):.3f}", flush=True)
        print(f"    total <r|r> = {tot_rr:.6e}   ->  Delta logL = {-0.5*tot_rr:.6e}",
              flush=True)
        results[label] = (common, diff, Fc, Fd, Fdat)

    # ---- plot: common vs differential, TD and FD
    fig, ax = plt.subplots(2, 2, figsize=(15, 8))
    tdays = np.arange(n) * dt / 86400.0
    for k, (label, (common, diff, Fc, Fd, Fdat)) in enumerate(results.items()):
        ax[k, 0].plot(tdays, common, lw=0.6, label="common-mode (X+Y+Z)/3")
        ax[k, 0].plot(tdays, diff[0], lw=0.6, alpha=0.7, label="differential (X part)")
        ax[k, 0].axvline(merger_idx * dt / 86400.0, color="k", ls=":", alpha=0.5)
        ax[k, 0].set_title(f"{label}: residual, TD")
        ax[k, 0].set_xlabel("days into window")
        ax[k, 0].legend(fontsize=8)

        s = band
        ax[k, 1].loglog(f_all[s], Fdat[s], color="k", alpha=0.35, lw=0.8, label="|data X|")
        ax[k, 1].loglog(f_all[s], Fc[s], lw=0.8, label="|common-mode|")
        ax[k, 1].loglog(f_all[s], Fd[s], lw=0.8, alpha=0.8, label="|differential|")
        ax[k, 1].set_title(f"{label}: residual, FD")
        ax[k, 1].set_xlabel("f [Hz]")
        ax[k, 1].legend(fontsize=8)

    fig.suptitle(f"MBHB id={MBHB_ID}: XYZ residual common-mode vs differential")
    fig.tight_layout(rect=[0, 0, 1, 0.97])
    fig.savefig(OUT, dpi=105)
    plt.close(fig)
    print(f"\nDONE.  plot -> {OUT}", flush=True)


if __name__ == "__main__":
    main()
