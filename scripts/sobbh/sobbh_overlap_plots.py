"""Overlap plots for the two SOBHBs: LAT template (exact anchor, flip_hx=False)
vs mojito L1 data, with the noise-weighted (SciRD v1) overlap numbers annotated.
Reuses the cached data window + templates from sobbh_mojito_match_debug.py.
"""
import os
os.environ.setdefault("OMP_NUM_THREADS", "8")
import numpy as np
import h5py
import matplotlib
matplotlib.use("Agg")
import matplotlib.pyplot as plt
from scipy.signal.windows import tukey

from lisatools.domains import TDSettings, FDSettings, WDMSettings, TDSignal
from lisatools.analysiscontainer import AnalysisContainer
from lisatools.sensitivity import XYZ2SensitivityMatrix

N_WIN = 262144
DT = 10.0
F_MIN, F_MAX = 0.010, 0.030
SENS = "scirdv1"
CAT = "/Users/mkatz/.mojito_cache/brickmarket/mojito_light_v1_0_0/catalogues/sobhb_cat_mojito_lite_processed_MT.hdf5"


def wdm_mm5(data_td, tmpl, td_set, f0):
    """Noise-weighted narrowband WDM mm5 around f0 (mirror of the debug script)."""
    Nf = 512
    ws = WDMSettings(Nf=Nf, Nt=N_WIN // Nf, dt=DT, min_freq=F_MIN, max_freq=F_MAX,
                     force_backend="cpu")
    layer_df = 1.0 / (2 * Nf * DT)
    mf = int(np.floor(f0 / layer_df))
    lo, hi = (mf - 3) * layer_df, (mf + 2) * layer_df
    ws_b = WDMSettings(Nf=Nf, Nt=N_WIN // Nf, dt=DT, min_freq=lo, max_freq=hi,
                       force_backend="cpu")
    win = tukey(N_WIN, 0.1)
    d = TDSignal(data_td, td_set).transform(ws_b, window=win)
    t = TDSignal(tmpl, td_set).transform(ws_b, window=win)
    ac = AnalysisContainer(d, XYZ2SensitivityMatrix(ws_b, model=SENS))
    O = ac.template_inner_product(t, normalize=True, complex=True)
    return 1 - O.real


def main():
    win = tukey(N_WIN, 0.1)
    fig, axes = plt.subplots(2, 2, figsize=(15, 10.5))
    with h5py.File(CAT, "r") as f:
        b = f["Binaries"]
        pars = {s: dict(m1=float(b["PrimaryMassSSBFrame"][s]),
                        m2=float(b["SecondaryMassSSBFrame"][s]),
                        f0=float(b["GW22FrequencySSBFrame"][s])) for s in (0, 1)}

    for row, SRC in enumerate([0, 1]):
        z = np.load(f"/tmp/sobbh_mojito_data_src{SRC}.npz", allow_pickle=True)
        data_td = np.asarray(z["data_td"])
        tmpl = np.load(f"/tmp/sobbh_mojito_tmpl_src{SRC}_fhx_false.npy")

        td_set = TDSettings(N_WIN, DT, t0=0.0, force_backend="cpu")
        fd_set = FDSettings(N=N_WIN // 2 + 1, df=1.0 / (N_WIN * DT),
                            min_freq=F_MIN, max_freq=F_MAX, force_backend="cpu")
        data_fd = TDSignal(data_td, td_set).transform(fd_set, window=win)
        ac = AnalysisContainer(data_fd, XYZ2SensitivityMatrix(fd_set, model=SENS))
        sig = TDSignal(tmpl, td_set).transform(fd_set, window=win)

        O = ac.template_inner_product(sig, normalize=True, complex=True)
        opt, det = ac.template_snr(sig)
        snr_d = float(np.sqrt(ac.inner_product().real))
        mm5 = wdm_mm5(data_td, tmpl, td_set, pars[SRC]["f0"])

        f = np.asarray(fd_set.f_arr)
        Xd = np.asarray(data_fd.arr)[0]
        Xt = np.asarray(sig.arr)[0]
        p = pars[SRC]

        # ---- time domain (zoomed, valid region) ----
        ax = axes[row, 0]
        t = np.arange(N_WIN) * DT
        mid = N_WIN // 2
        sl = slice(mid, mid + 300)  # 3000 s
        ax.plot(t[sl], data_td[0][sl] * 1e21, "o", ms=4.5, color="0.25",
                label="mojito data  X")
        ax.plot(t[sl], tmpl[0][sl] * 1e21, "-", lw=1.4, color="crimson",
                label="LAT template  X")
        ax.set_xlabel("time [s]"); ax.set_ylabel(r"X  [$10^{-21}$]")
        ax.set_title(f"SOBHB src{SRC}  —  time domain", fontsize=11)
        ax.text(0.03, 0.05, f"$m_1$={p['m1']:.0f} $m_2$={p['m2']:.0f} $M_\\odot$,  "
                f"$f_0$={p['f0']*1e3:.2f} mHz", transform=ax.transAxes, fontsize=9,
                bbox=dict(boxstyle="round", fc="white", ec="0.6", alpha=0.8))
        ax.legend(loc="upper right", fontsize=9); ax.grid(alpha=0.3)

        # ---- frequency domain (overlap + residual) ----
        ax = axes[row, 1]
        ax.loglog(f * 1e3, np.abs(Xd), "-", lw=2.2, color="0.25", label="mojito |X(f)|")
        ax.loglog(f * 1e3, np.abs(Xt), "--", lw=1.3, color="crimson", label="LAT |X(f)|")
        ax.loglog(f * 1e3, np.abs(Xd - Xt), ":", lw=1.3, color="seagreen",
                  label="|data − template|")
        ax.set_xlabel("frequency [mHz]"); ax.set_ylabel("|X(f)|")
        ax.set_title(f"src{SRC}  —  frequency domain (SciRD v1 weighted)", fontsize=11)
        ax.legend(loc="lower left", fontsize=9); ax.grid(alpha=0.3, which="both")
        txt = (f"$|O|$ = {abs(O):.5f}\n"
               f"FD mm = $1-\\mathrm{{Re}}(O)$ = {1-O.real:.2e}\n"
               f"WDM mm5 = {mm5:.2e}\n"
               f"det SNR = {det:.3f}\n"
               f"data SNR = {snr_d:.3f}\n"
               f"(det/data = {det/snr_d:.5f})")
        ax.text(0.97, 0.97, txt, transform=ax.transAxes, ha="right", va="top",
                fontsize=10, family="monospace",
                bbox=dict(boxstyle="round", fc="lightyellow", ec="0.5"))

    fig.suptitle("SOBHB template vs mojito L1 data — NON-phase-maximized overlap "
                 "(A-flip + exact reference-epoch phase anchor, flip_hx=False)",
                 fontsize=12, y=0.995)
    fig.tight_layout(rect=[0, 0, 1, 0.95])
    fig.subplots_adjust(hspace=0.32, wspace=0.22)
    out = os.path.join(os.path.dirname(os.path.abspath(__file__)), "sobbh_overlap_plots.png")
    fig.savefig(out, dpi=130)
    print(f"saved {out}")
    for SRC in (0, 1):
        pass


if __name__ == "__main__":
    main()
