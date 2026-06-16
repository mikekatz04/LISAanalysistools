"""Chase the two residuals after the t_ref fix (real MBHTDIonFly vs pyResponse):
  (1) the ~3.3 s high-band time offset -- sky-dependent (projection/de-Doppler)
      or fixed (grid/merger placement)?  Reports the best-fit tau* in the merger
      band at several sky positions + the k.x_sc1/c de-Doppler estimate.
  (2) the low-band [5e-4,1e-3] residual -- run with MBH_WIN_DAYS / MBH_DUR_DAYS
      large to test whether it is a truncation/edge effect.
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

REF = 97729089.327664; C = 299792458.0
PATH = "/Users/mkatz/.mojito_cache/brickmarket/mojito_light_v1_0_0/"
MBHB_L1 = os.path.join(PATH, "data", "MBHB", "L1")
MBHB_ID = 0; BACKEND = "cpu"; SENS_MODEL = "scirdv1"; DT = 10.0
TDI_GEN_STR = "2nd generation"; TDI_CHAN = "XYZ"; NCH = 3
F_MIN, F_MAX = 1e-4, 2.5e-2
HMS = (21, 33, 44); TOL = 1e-12; ORDER = 30; BUFFER = 15_000.0; START_FREQ = 7e-5
TUKEY_ALPHA = 0.05; POS_DT = 300.0
WIN_DAYS = float(os.environ.get("MBH_WIN_DAYS", "6.0"))
DUR_DAYS = float(os.environ.get("MBH_DUR_DAYS", "5.0"))
INC = float(os.environ.get("MBH_INC", str(np.pi / 2)))
MBH_TRANSFORM = make_mbh_transform_container()


def banner(s): print("\n" + "=" * 78 + f"\n {s}\n" + "=" * 78, flush=True)
def tukey(N, a):
    from scipy.signal.windows import tukey as _t
    return _t(N, a)


def build_orbit(window_t0, TOBS):
    orb = L1Orbits(find_file(MBHB_L1, "MBHB", MBHB_ID), force_backend=BACKEND, frame="icrs")
    pad = 1.0e5
    lo = max(window_t0 - pad, float(orb.sc_t0)); hi = min(window_t0 + TOBS + pad, float(orb._sc_t_base[-1]))
    lt = np.asarray(orb.ltt_t); m = (lt >= lo) & (lt <= hi)
    orb.ltt = np.asarray(orb.ltt)[m].copy(); orb.ltt_t = lt[m].copy(); orb.ltt_t0 = float(orb.ltt_t[0])
    del lt; gc.collect(); orb.configure(linear_interp_setup=True, dt=POS_DT); return orb


def gen_A(wf, window_t0, N_WIN, dur_s, orbit):
    grid = TDSettings(N=N_WIN, dt=DT, t0=window_t0, force_backend=BACKEND)
    g = PhenomTHMTDIWaveform(
        waveform_kwargs=dict(higher_modes=list(HMS), include_negative_modes=True,
                             t_low_fit=True, coarse_grain=False, atol=TOL, rtol=TOL),
        Tobs=dur_s, start_freq=START_FREQ, use_reference_time=True, waveform_t0=REF,
        data_td_settings=grid, tdi_generation=TDI_GEN_STR, tdi_channels=TDI_CHAN,
        sampling_frequency=1.0 / DT, orbits=orbit, order=ORDER, tukey_alpha=TUKEY_ALPHA,
        stft_dt=None, freq_min=F_MIN, freq_max=F_MAX, fft_batch_size=2, buffer_time=BUFFER,
        output_domain_settings=None, force_backend=BACKEND)
    t, ch = g.compute_tdi_channels(*wf)
    return np.asarray(place_td_signal_on_grid(np.atleast_2d(ch)[:NCH], grid, times=t).arr)


def gen_B(wave_gen, orbit, wf, window_t0, N_WIN, dur_s):
    m1, m2, s1z, s2z, dist, phi_ref, inc, psi, ra, dec, t_plunge = wf
    tdi_config = TDIConfig(TDI_GEN_STR, force_backend=BACKEND)
    mbh = MBHTDIonFly(wave_gen, orbit, tdi_config, DT, dur_s, t0=REF, dt_min=0.1,
                      waveform_duration=dur_s, force_backend=BACKEND)
    grid_t = np.arange(N_WIN) * DT + window_t0
    return np.asarray(mbh(m1, m2, s1z, s2z, dist, phi_ref, inc, ra, dec, psi, t_plunge,
                          upsample_t_arr=grid_t, combine=True))[:NCH]


def main():
    threading.Thread(target=_wd, daemon=True).start()
    banner(f"chase residuals (inc={INC:.3f}, win={WIN_DAYS}d, dur={DUR_DAYS}d)")
    z = np.load("/tmp/mbh_mojito_data_id0.npz", allow_pickle=True); cat = z["cat"].item()
    wf0 = np.asarray(MBH_TRANSFORM.both_transforms(
        np.asarray(mbh_catalogue_to_sampling_basis(cat), float)), float)
    wf0[6] = INC
    t_plunge = wf0[10]; abs_merger = REF + t_plunge
    N_WIN = int(round(WIN_DAYS * 86400 / DT)); window_t0 = abs_merger - 0.72 * N_WIN * DT
    TOBS = N_WIN * DT; dur_s = DUR_DAYS * 86400.0
    orbit = build_orbit(window_t0, TOBS)
    wave_gen = pw.IMRPhenomTHM(T=dur_s, higher_modes=list(HMS), include_negative_modes=True,
                               t_low_fit=True, coarse_grain=True, atol=TOL, rtol=TOL)
    win = tukey(N_WIN, TUKEY_ALPHA); td_set = TDSettings(N_WIN, DT, t0=window_t0, force_backend=BACKEND)
    # sc1 position at merger (for de-Doppler estimate)
    x1 = np.asarray(orbit.get_pos(np.array([abs_merger]), 1)).reshape(3)

    def tau_star_and_bands(wf):
        A = gen_A(wf, window_t0, N_WIN, dur_s, orbit)
        B = gen_B(wave_gen, orbit, wf, window_t0, N_WIN, dur_s)
        # high band time scan
        fd = FDSettings(N=N_WIN // 2 + 1, df=1.0 / (N_WIN * DT), min_freq=2e-3, max_freq=F_MAX, force_backend=BACKEND)
        ac = AnalysisContainer(TDSignal(A, td_set).transform(fd, window=win),
                               XYZ2SensitivityMatrix(fd, model=SENS_MODEL))
        Bf = TDSignal(B, td_set).transform(fd, window=win); f = np.asarray(fd.f_arr); B0 = np.asarray(Bf.arr).copy()
        def mmh(tau):
            Bf.arr[:] = B0 * np.exp(-2j * np.pi * f * tau)[None, :]
            return 1 - abs(ac.template_inner_product(Bf, normalize=True, complex=True))
        coarse = min(((mmh(t), t) for t in np.linspace(-15, 15, 121)))
        fine = min(((mmh(t), t) for t in coarse[1] + np.linspace(-0.5, 0.5, 401)))
        # low band mm (no time shift)
        def band(lo, hi):
            fdb = FDSettings(N=N_WIN // 2 + 1, df=1.0 / (N_WIN * DT), min_freq=lo, max_freq=hi, force_backend=BACKEND)
            a = AnalysisContainer(TDSignal(A, td_set).transform(fdb, window=win),
                                  XYZ2SensitivityMatrix(fdb, model=SENS_MODEL))
            return 1 - abs(a.template_inner_product(TSig := TDSignal(B, td_set).transform(fdb, window=win),
                                                    normalize=True, complex=True))
        lb = band(5e-4, 1e-3); fb = band(5e-4, F_MAX)
        del A, B; gc.collect()
        return fine[1], fine[0], lb, fb

    banner("(1) 3.3s offset: tau* in merger band vs SKY")
    print(f"  sc1@merger=({x1[0]:.3e},{x1[1]:.3e},{x1[2]:.3e}) m", flush=True)
    print(f"  {'sky':>22}  {'tau*[s]':>9}  {'mm@tau*':>10}  {'k.x1/c[s]':>10}", flush=True)
    for lbl, dra, ddec in [("natural", 0.0, 0.0), ("ra+1.0", 1.0, 0.0), ("ra-1.0", -1.0, 0.0),
                           ("dec+0.6", 0.0, 0.6)]:
        wf = wf0.copy(); wf[8] = wf0[8] + dra; wf[9] = wf0[9] + ddec
        ra, dec = wf[8], wf[9]
        # propagation direction k (from source toward SSB): -nhat(source)
        k = -np.array([np.cos(dec) * np.cos(ra), np.cos(dec) * np.sin(ra), np.sin(dec)])
        kx = float(k @ x1) / C
        ts, mm, lb, fb = tau_star_and_bands(wf)
        print(f"  {lbl:>22}  {ts:+9.3f}  {mm:10.3e}  {kx:+10.2f}", flush=True)
    banner(f"(2) low band [5e-4,1e-3] at win={WIN_DAYS}d (full={fb:.3e})")
    print(f"  low-band mm = {lb:.4e}  (run with MBH_WIN_DAYS=30 MBH_DUR_DAYS=28 to test truncation)", flush=True)
    print("\nDONE.", flush=True)


if __name__ == "__main__":
    main()
