"""FINAL injection-parameter summary for SOBHB sources (src 0 and 1) vs mojito.

At the injection parameters (tc-consistent waveform, flip_hx=False, reference
epoch = TimeReferenceSSBFrame), legacy pyResponse (A) vs TDI-on-the-fly (B),
scirdv1 + mrdv1, full band [10,30] mHz:

  overlap:  1-Re(O) at injection (NON-phase-marg) | 1-|O| at injection (phase-marg)
            | 1-|O| phase+time-max (fitting factor)
  logL:     -1/2 <d-h|d-h> at the injection phase

Saves per-source 3-panel plot + the summary values (for the combined plot).
Uses the 30.3 d canonical window (/tmp/sobbh_mojito_data_src{SRC}.npz).
"""
import os, sys, gc, time, threading, resource
os.environ.setdefault("OMP_NUM_THREADS", "8")
import numpy as np
import matplotlib; matplotlib.use("Agg"); import matplotlib.pyplot as plt
MEM_CAP_GB = float(os.environ.get("SOBBH_MEM_CAP_GB", "7.5")); _IS_MAC = sys.platform == "darwin"
def rss_gb():
    r = resource.getrusage(resource.RUSAGE_SELF).ru_maxrss
    return r / 1e9 if _IS_MAC else r / 1e6
def _wd():
    while True:
        if rss_gb() > MEM_CAP_GB: os._exit(42)
        time.sleep(0.3)

from lisatools.detector import L1Orbits
from lisatools.globalfit.preprocessing import find_file
from lisatools.response.directresponse import ResponseWrapper
from lisatools.response.tdiconfig import TDIConfig
from lisatools.sources.sobbh.waveform import SOBBHWaveform
from lisatools.domains import TDSettings, FDSettings, TDSignal
from lisatools.analysiscontainer import AnalysisContainer
from lisatools.sensitivity import XYZ2SensitivityMatrix
from lisatools.utils.constants import YRSID_SI
from bbhx.sobbhtdionfly import SOBBHTDIonFly

REF = 97729089.327664
PATH = "/Users/mkatz/.mojito_cache/brickmarket/mojito_light_v1_0_0/"
SOBHB_L1 = os.path.join(PATH, "data", "SOBHB", "L1")
BACKEND = "cpu"; DT = 10.0; TDI_GEN = "2nd generation"; NCH = 3
NF, NT = 512, 512; N_WIN = NF * NT; TOBS = N_WIN * DT
F_MIN, F_MAX = 0.010, 0.030; ORDER = 40; T_BUFFER = 3.0e4
TUKEY_ALPHA = float(os.environ.get("SOBBH_TUKEY_ALPHA", "0.05"))
SENS = ["scirdv1", "mrdv1"]


def banner(s): print("\n" + "=" * 102 + f"\n {s}\n" + "=" * 102, flush=True)
def tukey(N, a):
    from scipy.signal.windows import tukey as _t
    return _t(N, a)


def build_AB(SRC, cat, data_t0):
    g = lambda k: float(cat[k])
    pf = np.array([g("PrimaryMassSSBFrame"), g("SecondaryMassSSBFrame"),
                   g("PrimarySpinCompZ"), g("SecondarySpinCompZ"),
                   g("LuminosityDistance") / 1e3, g("InclinationAngle"),
                   g("GW22FrequencySSBFrame"), g("RightAscension") % (2 * np.pi),
                   g("Declination"), g("PolarisationAngle") % np.pi, g("TrueAnomaly")])
    m1, m2, s1, s2, dist, inc, f0, ra, dec, psi, phi0 = pf
    orb = L1Orbits(find_file(SOBHB_L1, "SOBHB", SRC), force_backend=BACKEND, frame="icrs")
    pad = 1.0e5; lo = max(data_t0 - pad, float(orb.sc_t0)); hi = min(data_t0 + TOBS + pad, float(orb._sc_t_base[-1]))
    lt = np.asarray(orb.ltt_t); mk = (lt >= lo) & (lt <= hi)
    orb.ltt = np.asarray(orb.ltt)[mk].copy(); orb.ltt_t = lt[mk].copy(); orb.ltt_t0 = float(orb.ltt_t[0])
    del lt; gc.collect(); orb.configure(linear_interp_setup=True)
    gen = SOBBHWaveform(Tobs=TOBS, dt=DT, t0=data_t0, reference_time=REF, force_backend=BACKEND)
    legacy = ResponseWrapper(gen, orbits=orb, t0=data_t0, Tobs=TOBS / YRSID_SI, dt=DT,
                             index_lambda=7, index_beta=8, flip_hx=False,
                             tdi=TDIConfig(TDI_GEN, force_backend=BACKEND), tdi_chan="XYZ", order=ORDER,
                             remove_garbage="zero", is_ecliptic_latitude=True, t_buffer=T_BUFFER,
                             force_backend=BACKEND)
    A = np.atleast_2d(np.asarray(legacy(*pf, convert_to_ra_dec=False)))[:NCH]
    A = np.pad(A, ((0, 0), (0, N_WIN - A.shape[-1]))) if A.shape[-1] < N_WIN else A[:, :N_WIN]
    fly = SOBBHTDIonFly(SOBBHWaveform(Tobs=TOBS, dt=DT, t0=data_t0, reference_time=REF, force_backend=BACKEND),
                        orb, TDIConfig(TDI_GEN, force_backend=BACKEND), DT, TOBS, t0=data_t0, force_backend=BACKEND)
    grid_t = np.arange(N_WIN) * DT + data_t0
    B = np.asarray(fly(m1, m2, s1, s2, dist, f0, phi0, inc, ra, dec, psi,
                       upsample_t_arr=grid_t, combine=True))[:NCH]
    return A, B, f0, inc


def metrics(D, T, win, td, sens):
    fd = FDSettings(N=N_WIN // 2 + 1, df=1.0 / (N_WIN * DT), min_freq=F_MIN, max_freq=F_MAX, force_backend=BACKEND)
    f = np.asarray(fd.f_arr)
    ac = AnalysisContainer(TDSignal(D, td).transform(fd, window=win), XYZ2SensitivityMatrix(fd, model=sens))
    sig = TDSignal(T, td).transform(fd, window=win); t0 = np.asarray(sig.arr).copy()
    O0 = ac.template_inner_product(sig, normalize=True, complex=True)
    logL0 = float(np.real(ac.template_likelihood(sig)))
    def mO(tau):
        sig.arr[:] = t0 * np.exp(-2j * np.pi * f * tau)[None, :]
        return ac.template_inner_product(sig, normalize=True, complex=True)
    c = max(((abs(mO(x)), x) for x in np.linspace(-3, 3, 121)))
    ft = max(((abs(mO(x)), x) for x in c[1] + np.linspace(-0.02, 0.02, 81)))
    sig.arr[:] = t0
    return dict(reO_inj=1 - O0.real, mm_inj=1 - abs(O0), logL_inj=logL0,
                mm_pt=1 - ft[0], taustar=ft[1])


def main():
    threading.Thread(target=_wd, daemon=True).start()
    summary = {}
    for SRC in (0, 1):
        cache = f"/tmp/sobbh_mojito_data_src{SRC}.npz"
        z = np.load(cache, allow_pickle=True)
        D = np.asarray(z["data_td"])[:NCH]; data_t0 = float(z["data_t0"]); cat = z["cat"].item()
        A, B, f0, inc = build_AB(SRC, cat, data_t0)
        win = tukey(N_WIN, TUKEY_ALPHA); td = TDSettings(N_WIN, DT, t0=0.0, force_backend=BACKEND)
        banner(f"SOBHB src {SRC}   f0={f0*1e3:.3f} mHz  inc={inc:.3f}   (30.3 d, full band, tc-consistent)")
        print(f"  {'sens':>8} {'tmpl':>10} {'1-Re(O)inj':>12} {'1-|O|inj':>12} {'logL_inj':>12} "
              f"{'1-|O|ptmax':>12} {'tau*':>8}", flush=True)
        for sens in SENS:
            for nm, T in [("legacy", A), ("on-fly", B)]:
                m = metrics(D, T, win, td, sens)
                summary[(SRC, sens, nm)] = m
                print(f"  {sens:>8} {nm:>10} {m['reO_inj']:12.4e} {m['mm_inj']:12.4e} "
                      f"{m['logL_inj']:12.4e} {m['mm_pt']:12.4e} {m['taustar']:+8.3f}", flush=True)
        # plot (scirdv1 FD |X| + residuals)
        f = np.fft.rfftfreq(N_WIN, DT); sel = (f >= F_MIN) & (f <= F_MAX)
        FD = np.fft.rfft(D[0] * win) * DT; FA = np.fft.rfft(A[0] * win) * DT; FB = np.fft.rfft(B[0] * win) * DT
        fig, ax = plt.subplots(2, 1, figsize=(10, 9))
        ax[0].loglog(f[sel], np.abs(FD[sel]), label="mojito data", lw=1.0)
        ax[0].loglog(f[sel], np.abs(FA[sel]), "--", label="legacy pyResponse", lw=.9, alpha=.8)
        ax[0].loglog(f[sel], np.abs(FB[sel]), ":", label="TDI-on-the-fly", lw=1.2, alpha=.8)
        ax[0].axvline(f0, color="k", ls=":", alpha=.4); ax[0].set_xlim(F_MIN, F_MAX)
        ax[0].legend(); ax[0].set_title(f"SOBHB src{SRC}: FD |X|"); ax[0].set_xlabel("f [Hz]")
        ax[1].loglog(f[sel], np.abs((FA - FD)[sel]), label="|legacy - data|", lw=.9)
        ax[1].loglog(f[sel], np.abs((FB - FD)[sel]), label="|on-fly - data|", lw=.9)
        ax[1].axvline(f0, color="k", ls=":", alpha=.4); ax[1].set_xlim(F_MIN, F_MAX)
        ax[1].legend(); ax[1].set_title("FD |X| residual vs data"); ax[1].set_xlabel("f [Hz]")
        fig.tight_layout(); out = f"/tmp/sobbh_injection_final_src{SRC}.png"; fig.savefig(out, dpi=110); plt.close(fig)
        print(f"  plot -> {out}", flush=True)
        del A, B; gc.collect()
    np.savez("/tmp/sobbh_injection_summary.npz", summary=summary)
    print("\nDONE.  values -> /tmp/sobbh_injection_summary.npz", flush=True)


if __name__ == "__main__":
    main()
