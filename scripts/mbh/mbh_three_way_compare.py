"""Three-way comparison at the INJECTION params (no phase/time max headline):
  D = mojito data (cached window)
  A = pyTDResponse (PhenomTHMTDIWaveform)
  B = TDI-on-the-fly (MBHTDIonFly, t_ref-fixed)

For each pair (A-D, B-D, A-B) and band, reports:
  1-Re(O)  : NO phase/time max  (tests that injected phase+time are correct)
  1-|O|    : phase-maximized
  arg(O)   : residual phase [deg]
  tau*     : best time shift [s] (diagnostic)
Then plots all three: FD |X|, TD zoom @ merger, FD residuals.
"""
import os, sys, gc, time, threading, resource
os.environ.setdefault("OMP_NUM_THREADS", "8")
import numpy as np
import matplotlib; matplotlib.use("Agg"); import matplotlib.pyplot as plt
MEM_CAP_GB = 6.5; _IS_MAC = sys.platform == "darwin"
def rss_gb():
    r = resource.getrusage(resource.RUSAGE_SELF).ru_maxrss
    return r / 1e9 if _IS_MAC else r / 1e6
def _wd():
    while True:
        if rss_gb() > MEM_CAP_GB: os._exit(42)
        time.sleep(0.3)
def mark(m): print(f"[RSS {rss_gb():5.2f} GB] {m}", flush=True)
from lisatools.detector import L1Orbits
from lisatools.globalfit.preprocessing import find_file
from lisatools.globalfit.recipe_steps import mbh_catalogue_to_sampling_basis
from lisatools.globalfit.stock.erebor import make_mbh_transform_container
from lisatools.response.tdiconfig import TDIConfig
from lisatools.sources.bbh.waveform import PhenomTHMTDIWaveform
from lisatools.domains import TDSettings, FDSettings, TDSignal, place_td_signal_on_grid
from lisatools.analysiscontainer import AnalysisContainer
from lisatools.sensitivity import XYZ2SensitivityMatrix
from bbhx.mbhtdionfly import MBHTDIonFly
import phentax.waveform as pw

REF = 97729089.327664
PATH = "/Users/mkatz/.mojito_cache/brickmarket/mojito_light_v1_0_0/"
MBHB_L1 = os.path.join(PATH, "data", "MBHB", "L1")
MBHB_ID = 0; BACKEND = "cpu"; SENS_MODEL = "scirdv1"; DT = 10.0
TDI_GEN_STR = "2nd generation"; TDI_CHAN = "XYZ"; NCH = 3
F_MIN, F_MAX = 1e-4, 2.5e-2
HMS = (21, 33, 44); TOL = 1e-12; ORDER = 30; BUFFER = 15_000.0; START_FREQ = 7e-5
TUKEY_ALPHA = 0.05; POS_DT = 300.0; DTMIN = 0.1
MBH_TRANSFORM = make_mbh_transform_container()
DATA_CACHE = f"/tmp/mbh_mojito_data_id{MBHB_ID}.npz"


def banner(s): print("\n" + "=" * 78 + f"\n {s}\n" + "=" * 78, flush=True)
def tukey(N, a):
    from scipy.signal.windows import tukey as _t
    return _t(N, a)


def main():
    threading.Thread(target=_wd, daemon=True).start()
    banner("THREE-WAY compare: data D / pyTDResponse A / on-the-fly B (injection params)")
    z = np.load(DATA_CACHE, allow_pickle=True)
    D = z["data_td"]; window_t0 = float(z["window_t0"]); cat = z["cat"].item()
    abs_merger = float(z["abs_merger"])
    N_WIN = D.shape[1]; TOBS = N_WIN * DT
    wf = np.asarray(MBH_TRANSFORM.both_transforms(
        np.asarray(mbh_catalogue_to_sampling_basis(cat), float)), float)
    m1, m2, s1z, s2z, dist, phi_ref, inc, psi, ra, dec, t_plunge = wf
    dur_s = ((abs_merger - window_t0) / 86400.0 + 6.0) * 86400.0
    print(f"  N_WIN={N_WIN} Tobs={TOBS/86400:.1f}d inc={inc:.4f} dur={dur_s/86400:.1f}d", flush=True)

    orb = L1Orbits(find_file(MBHB_L1, "MBHB", MBHB_ID), force_backend=BACKEND, frame="icrs")
    pad = 1.0e5; lo = max(window_t0 - pad, float(orb.sc_t0)); hi = min(window_t0 + TOBS + pad, float(orb._sc_t_base[-1]))
    lt = np.asarray(orb.ltt_t); mm_ = (lt >= lo) & (lt <= hi)
    orb.ltt = np.asarray(orb.ltt)[mm_].copy(); orb.ltt_t = lt[mm_].copy(); orb.ltt_t0 = float(orb.ltt_t[0])
    del lt; gc.collect(); orb.configure(linear_interp_setup=True, dt=POS_DT); mark("orbit ready")

    grid = TDSettings(N=N_WIN, dt=DT, t0=window_t0, force_backend=BACKEND)
    genA = PhenomTHMTDIWaveform(
        waveform_kwargs=dict(higher_modes=list(HMS), include_negative_modes=True,
                             t_low_fit=True, coarse_grain=False, atol=TOL, rtol=TOL),
        Tobs=dur_s, start_freq=START_FREQ, use_reference_time=True, waveform_t0=REF,
        data_td_settings=grid, tdi_generation=TDI_GEN_STR, tdi_channels=TDI_CHAN,
        sampling_frequency=1.0 / DT, orbits=orb, order=ORDER, tukey_alpha=TUKEY_ALPHA,
        stft_dt=None, freq_min=F_MIN, freq_max=F_MAX, fft_batch_size=2, buffer_time=BUFFER,
        output_domain_settings=None, force_backend=BACKEND)
    tA, chA = genA.compute_tdi_channels(*wf)
    A = np.asarray(place_td_signal_on_grid(np.atleast_2d(chA)[:NCH], grid, times=tA).arr); mark("A built")
    wave_gen = pw.IMRPhenomTHM(T=dur_s, higher_modes=list(HMS), include_negative_modes=True,
                               t_low_fit=True, coarse_grain=True, atol=TOL, rtol=TOL)
    mbh = MBHTDIonFly(wave_gen, orb, TDIConfig(TDI_GEN_STR, force_backend=BACKEND), DT, dur_s,
                      t0=REF, dt_min=DTMIN, waveform_duration=dur_s, force_backend=BACKEND)
    grid_t = np.arange(N_WIN) * DT + window_t0
    B = np.asarray(mbh(m1, m2, s1z, s2z, dist, phi_ref, inc, ra, dec, psi, t_plunge,
                       upsample_t_arr=grid_t, combine=True))[:NCH]; mark("B built")

    win = tukey(N_WIN, TUKEY_ALPHA); td = TDSettings(N_WIN, DT, t0=window_t0, force_backend=BACKEND)

    def fd_of(arr, lo, hi):
        fd = FDSettings(N=N_WIN // 2 + 1, df=1.0 / (N_WIN * DT), min_freq=lo, max_freq=hi, force_backend=BACKEND)
        return fd, TDSignal(arr, td).transform(fd, window=win)

    def compare(ref, tmpl, lo, hi):
        fd, rsig = fd_of(ref, lo, hi); _, tsig = fd_of(tmpl, lo, hi)
        ac = AnalysisContainer(rsig, XYZ2SensitivityMatrix(fd, model=SENS_MODEL))
        O = ac.template_inner_product(tsig, normalize=True, complex=True)
        f = np.asarray(fd.f_arr); t0 = np.asarray(tsig.arr).copy()
        def mmt(tau):
            tsig.arr[:] = t0 * np.exp(-2j * np.pi * f * tau)[None, :]
            return 1 - abs(ac.template_inner_product(tsig, normalize=True, complex=True))
        bc = min(((mmt(x), x) for x in np.linspace(-20, 20, 161)))
        bf = min(((mmt(x), x) for x in bc[1] + np.linspace(-0.5, 0.5, 201)))
        tsig.arr[:] = t0
        return 1 - O.real, 1 - abs(O), np.degrees(np.angle(O)), bf[1], bf[0]

    bands = [("full", F_MIN, F_MAX), (">5e-4", 5e-4, F_MAX), (">1mHz", 1e-3, F_MAX), (">2mHz", 2e-3, F_MAX)]
    for pair, ref, tmpl in [("A=pyResp vs D", D, A), ("B=on-fly vs D", D, B), ("A vs B", A, B)]:
        banner(pair + "   [1-Re(O)=NO max | 1-|O|=phasemax | arg | tau* | mm@tau*]")
        for tag, lo, hi in bands:
            r, a, ph, ts, mt = compare(ref, tmpl, lo, hi)
            print(f"  {tag:7s} 1-Re(O)={r:.4e}  1-|O|={a:.4e}  arg={ph:+7.2f}deg  "
                  f"tau*={ts:+.3f}s  mm@tau*={mt:.4e}", flush=True)

    # ---- plots ----
    f = np.fft.rfftfreq(N_WIN, DT)
    FD = np.fft.rfft(D[0] * win) * DT; FA = np.fft.rfft(A[0] * win) * DT; FB = np.fft.rfft(B[0] * win) * DT
    sel = (f >= F_MIN) & (f <= F_MAX)
    merger_idx = int(round((abs_merger - window_t0) / DT))
    fig, ax = plt.subplots(3, 1, figsize=(11, 13))
    ax[0].loglog(f[sel], np.abs(FD[sel]), label="mojito data", lw=1.0)
    ax[0].loglog(f[sel], np.abs(FA[sel]), "--", label="pyTDResponse", lw=0.9, alpha=.8)
    ax[0].loglog(f[sel], np.abs(FB[sel]), ":", label="TDI-on-the-fly", lw=1.1, alpha=.8)
    ax[0].set_xlim(F_MIN, F_MAX); ax[0].legend(); ax[0].set_title("FD |X|"); ax[0].set_xlabel("f [Hz]")
    n0 = max(0, merger_idx - 1500); sl = slice(n0, min(N_WIN, n0 + 3500))
    tt = (np.arange(N_WIN)[sl] - merger_idx) * DT / 3600.0
    ax[1].plot(tt, D[0][sl], label="mojito data", lw=1.4)
    ax[1].plot(tt, A[0][sl], "--", label="pyTDResponse", lw=1.0)
    ax[1].plot(tt, B[0][sl], ":", label="TDI-on-the-fly", lw=1.0)
    ax[1].axvline(0, color="k", ls=":", alpha=.4); ax[1].legend()
    ax[1].set_title("TD X @ merger"); ax[1].set_xlabel("hours from merger")
    ax[2].loglog(f[sel], np.abs((FA - FD)[sel]), label="|pyResp - data|", lw=0.9)
    ax[2].loglog(f[sel], np.abs((FB - FD)[sel]), label="|on-fly - data|", lw=0.9)
    ax[2].loglog(f[sel], np.abs((FA - FB)[sel]), ":", color="purple", label="|pyResp - on-fly|", lw=1.0)
    ax[2].set_xlim(F_MIN, F_MAX); ax[2].legend(); ax[2].set_title("FD |X| residuals"); ax[2].set_xlabel("f [Hz]")
    fig.suptitle(f"MBHB id=0 injection: data vs pyTDResponse vs TDI-on-the-fly (inc={inc:.3f})")
    fig.tight_layout(rect=[0, 0, 1, 0.985])
    out = f"/tmp/mbh_three_way_id{MBHB_ID}.png"; fig.savefig(out, dpi=110); plt.close(fig)
    print(f"\nDONE.  plot -> {out}", flush=True)


if __name__ == "__main__":
    main()
