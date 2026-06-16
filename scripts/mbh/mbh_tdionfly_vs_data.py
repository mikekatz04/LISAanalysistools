"""Optimize the on-the-fly (MBHTDIonFly) overlap with the MOJITO DATA at the
injection (catalogue) parameters, using PHASE + TIME maximization.

Uses the cached mojito data window (/tmp/mbh_mojito_data_id0.npz from
mbh_mojito_match_debug.py), the catalogue injection params, the t_ref-fixed real
MBHTDIonFly, and reports mismatch (full + bands) phase-maximized and then
phase+time-maximized -- the time scan absorbs/reveals any residual reference
time offset.  Compare against pyResponse's 3.4e-5 (>1mHz).
"""
import os, sys, gc, time, threading, resource
os.environ.setdefault("OMP_NUM_THREADS", "8")
import numpy as np
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
from lisatools.domains import TDSettings, FDSettings, TDSignal
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
HMS = (21, 33, 44); TOL = 1e-12; DTMIN = 0.1
TUKEY_ALPHA = 0.05; POS_DT = 300.0
MBH_TRANSFORM = make_mbh_transform_container()
DATA_CACHE = f"/tmp/mbh_mojito_data_id{MBHB_ID}.npz"


def banner(s): print("\n" + "=" * 78 + f"\n {s}\n" + "=" * 78, flush=True)
def tukey(N, a):
    from scipy.signal.windows import tukey as _t
    return _t(N, a)


def main():
    threading.Thread(target=_wd, daemon=True).start()
    banner("MBHTDIonFly vs MOJITO DATA (injection params, phase+time max)")
    z = np.load(DATA_CACHE, allow_pickle=True)
    data_td = z["data_td"]; window_t0 = float(z["window_t0"]); cat = z["cat"].item()
    abs_merger = float(z["abs_merger"])
    N_WIN = data_td.shape[1]; TOBS = N_WIN * DT
    samp = mbh_catalogue_to_sampling_basis(cat)
    wf = np.asarray(MBH_TRANSFORM.both_transforms(np.asarray(samp, float)), float)
    m1, m2, s1z, s2z, dist, phi_ref, inc, psi, ra, dec, t_plunge = wf
    merger_in_win = (abs_merger - window_t0) / 86400.0
    dur_days = merger_in_win + 6.0   # cover the pre-merger window + margin
    dur_s = dur_days * 86400.0
    print(f"  N_WIN={N_WIN} Tobs={TOBS/86400:.1f}d  merger@{merger_in_win:.2f}d  "
          f"waveform_duration={dur_days:.1f}d  inc={inc:.4f}", flush=True)

    # orbit sliced to the window
    orb = L1Orbits(find_file(MBHB_L1, "MBHB", MBHB_ID), force_backend=BACKEND, frame="icrs")
    pad = 1.0e5
    lo = max(window_t0 - pad, float(orb.sc_t0)); hi = min(window_t0 + TOBS + pad, float(orb._sc_t_base[-1]))
    lt = np.asarray(orb.ltt_t); m = (lt >= lo) & (lt <= hi)
    orb.ltt = np.asarray(orb.ltt)[m].copy(); orb.ltt_t = lt[m].copy(); orb.ltt_t0 = float(orb.ltt_t[0])
    del lt; gc.collect(); orb.configure(linear_interp_setup=True, dt=POS_DT)
    mark("orbit ready")

    # on-the-fly template (real MBHTDIonFly, t_ref fix inside)
    wave_gen = pw.IMRPhenomTHM(T=dur_s, higher_modes=list(HMS), include_negative_modes=True,
                               t_low_fit=True, coarse_grain=True, atol=TOL, rtol=TOL)
    tdi_config = TDIConfig(TDI_GEN_STR, force_backend=BACKEND)
    mbh = MBHTDIonFly(wave_gen, orb, tdi_config, DT, dur_s, t0=REF, dt_min=DTMIN,
                      waveform_duration=dur_s, force_backend=BACKEND)
    grid_t = np.arange(N_WIN) * DT + window_t0
    B = np.asarray(mbh(m1, m2, s1z, s2z, dist, phi_ref, inc, ra, dec, psi, t_plunge,
                       upsample_t_arr=grid_t, combine=True))[:NCH]
    mark("on-the-fly template built")

    win = tukey(N_WIN, TUKEY_ALPHA); td_set = TDSettings(N_WIN, DT, t0=window_t0, force_backend=BACKEND)

    def setup(lo, hi):
        fd = FDSettings(N=N_WIN // 2 + 1, df=1.0 / (N_WIN * DT), min_freq=lo, max_freq=hi, force_backend=BACKEND)
        d = TDSignal(data_td, td_set).transform(fd, window=win)
        ac = AnalysisContainer(d, XYZ2SensitivityMatrix(fd, model=SENS_MODEL))
        t = TDSignal(B, td_set).transform(fd, window=win)
        return ac, t, np.asarray(fd.f_arr), np.asarray(t.arr).copy()

    def report(lo, hi, tag):
        ac, t, f, t0 = setup(lo, hi)
        O = ac.template_inner_product(t, normalize=True, complex=True)
        opt, det = ac.template_snr(t); dd = ac.inner_product()
        # time scan (phase already maximized inside |O|)
        def mmt(tau):
            t.arr[:] = t0 * np.exp(-2j * np.pi * f * tau)[None, :]
            return 1 - abs(ac.template_inner_product(t, normalize=True, complex=True))
        bc = min(((mmt(x), x) for x in np.linspace(-20, 20, 161)))
        bf = min(((mmt(x), x) for x in bc[1] + np.linspace(-0.5, 0.5, 401)))
        t.arr[:] = t0
        print(f"  {tag:14s} mm(pmax)={1-abs(O):.4e}  mm(p+t max)={bf[0]:.4e} @tau*={bf[1]:+.3f}s  "
              f"SNR d={np.sqrt(dd.real):.1f} h={opt:.1f} det={det:+.2f}", flush=True)

    banner("on-the-fly template vs mojito DATA  (phase-max and phase+time-max)")
    report(F_MIN, F_MAX, "full")
    report(5e-4, F_MAX, ">5e-4")
    report(1e-3, F_MAX, ">1mHz")
    report(2e-3, F_MAX, ">2mHz")
    print("\nDONE.", flush=True)


if __name__ == "__main__":
    main()
