"""Is the on-the-fly A-vs-B discrepancy just a TIME SHIFT / reference-phase offset?

The A-vs-B mismatch grows ~ with frequency (1% -> 29%), the signature of a
constant time shift (Dphi = 2*pi*f*tau).  My earlier A-vs-B was only
PHASE-maximized, not TIME-maximized -- so a missing reference-time/phase would
look exactly like this.  Uses the REAL MBHTDIonFly (no inline replica) for B and
PhenomTHMTDIWaveform for A, then scans a time shift tau on B: if mm collapses at
some tau*, the discrepancy is a time/reference misalignment, not a response error.
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

REF = 97729089.327664
PATH = "/Users/mkatz/.mojito_cache/brickmarket/mojito_light_v1_0_0/"
MBHB_L1 = os.path.join(PATH, "data", "MBHB", "L1")
MBHB_ID = int(os.environ.get("MBHB_ID", "0"))
BACKEND = "cpu"; SENS_MODEL = "scirdv1"; DT = 10.0
TDI_GEN_STR = "2nd generation"; TDI_CHAN = "XYZ"; NCH = 3
F_MIN, F_MAX = 1e-4, 2.5e-2
HMS = (21, 33, 44); TOL = 1e-12; ORDER = 30; BUFFER = 15_000.0; START_FREQ = 7e-5
TUKEY_ALPHA = 0.05; POS_DT = 300.0
WIN_DAYS = 6.0; DUR_DAYS = 5.0
INC = float(os.environ.get("MBH_INC", str(np.pi / 2)))
MBH_TRANSFORM = make_mbh_transform_container()
DATA_CACHE = f"/tmp/mbh_mojito_data_id{MBHB_ID}.npz"


def banner(s): print("\n" + "=" * 78 + f"\n {s}\n" + "=" * 78, flush=True)
def tukey(N, a):
    from scipy.signal.windows import tukey as _t
    return _t(N, a)


def build_orbit(window_t0, TOBS):
    orb = L1Orbits(find_file(MBHB_L1, "MBHB", MBHB_ID), force_backend=BACKEND, frame="icrs")
    pad = 1.0e5
    lo = max(window_t0 - pad, float(orb.sc_t0)); hi = min(window_t0 + TOBS + pad, float(orb._sc_t_base[-1]))
    ltt_t = np.asarray(orb.ltt_t); m = (ltt_t >= lo) & (ltt_t <= hi)
    orb.ltt = np.asarray(orb.ltt)[m].copy(); orb.ltt_t = ltt_t[m].copy(); orb.ltt_t0 = float(orb.ltt_t[0])
    del ltt_t; gc.collect()
    orb.configure(linear_interp_setup=True, dt=POS_DT)
    return orb


def main():
    threading.Thread(target=_wd, daemon=True).start()
    banner(f"A-vs-B TIME-SHIFT test (id={MBHB_ID}, inc={INC:.4f}) -- real MBHTDIonFly")
    z = np.load(DATA_CACHE, allow_pickle=True); cat = z["cat"].item()
    wf = np.asarray(MBH_TRANSFORM.both_transforms(
        np.asarray(mbh_catalogue_to_sampling_basis(cat), float)), float)
    wf[6] = INC
    m1, m2, s1z, s2z, dist, phi_ref, inc, psi, ra, dec, t_plunge = wf
    abs_merger = REF + t_plunge
    N_WIN = int(round(WIN_DAYS * 86400 / DT)); window_t0 = abs_merger - 0.72 * N_WIN * DT
    TOBS = N_WIN * DT; dur_s = DUR_DAYS * 86400.0
    orbit = build_orbit(window_t0, TOBS)
    grid = TDSettings(N=N_WIN, dt=DT, t0=window_t0, force_backend=BACKEND)

    # A: pyResponse
    genA = PhenomTHMTDIWaveform(
        waveform_kwargs=dict(higher_modes=list(HMS), include_negative_modes=True,
                             t_low_fit=True, coarse_grain=False, atol=TOL, rtol=TOL),
        Tobs=dur_s, start_freq=START_FREQ, use_reference_time=True, waveform_t0=REF,
        data_td_settings=grid, tdi_generation=TDI_GEN_STR, tdi_channels=TDI_CHAN,
        sampling_frequency=1.0 / DT, orbits=orbit, order=ORDER, tukey_alpha=TUKEY_ALPHA,
        stft_dt=None, freq_min=F_MIN, freq_max=F_MAX, fft_batch_size=2, buffer_time=BUFFER,
        output_domain_settings=None, force_backend=BACKEND)
    tA, chA = genA.compute_tdi_channels(*wf)
    A = np.asarray(place_td_signal_on_grid(np.atleast_2d(chA)[:NCH], grid, times=tA).arr)
    print(f"  A built  RSS={rss_gb():.2f}GB", flush=True)

    # B: REAL MBHTDIonFly
    wave_gen = pw.IMRPhenomTHM(T=dur_s, higher_modes=list(HMS), include_negative_modes=True,
                               t_low_fit=True, coarse_grain=True, atol=TOL, rtol=TOL)
    tdi_config = TDIConfig(TDI_GEN_STR, force_backend=BACKEND)
    mbh = MBHTDIonFly(wave_gen, orbit, tdi_config, DT, dur_s, t0=REF, dt_min=0.1,
                      waveform_duration=dur_s, force_backend=BACKEND)
    grid_t = np.arange(N_WIN) * DT + window_t0
    B = np.asarray(mbh(m1, m2, s1z, s2z, dist, phi_ref, inc, ra, dec, psi, t_plunge,
                       upsample_t_arr=grid_t, combine=True))[:NCH]
    print(f"  B (real MBHTDIonFly) built  RSS={rss_gb():.2f}GB", flush=True)

    win = tukey(N_WIN, TUKEY_ALPHA); td_set = TDSettings(N_WIN, DT, t0=window_t0, force_backend=BACKEND)
    fd = FDSettings(N=N_WIN // 2 + 1, df=1.0 / (N_WIN * DT), min_freq=5e-4, max_freq=F_MAX, force_backend=BACKEND)
    A_fd = TDSignal(A, td_set).transform(fd, window=win)
    B_fd = TDSignal(B, td_set).transform(fd, window=win)
    ac = AnalysisContainer(A_fd, XYZ2SensitivityMatrix(fd, model=SENS_MODEL))
    f_arr = np.asarray(fd.f_arr); B0 = np.asarray(B_fd.arr).copy()

    def mm_at(tau):
        B_fd.arr[:] = B0 * np.exp(-2j * np.pi * f_arr * tau)[None, :]
        return 1 - abs(ac.template_inner_product(B_fd, normalize=True, complex=True))

    banner("phase-max mismatch vs applied time shift tau (B -> B(t-tau))")
    mm0 = mm_at(0.0)
    # coarse then fine
    coarse = np.linspace(-60, 60, 241)
    bc = min(((mm_at(t), t) for t in coarse))
    fine = bc[1] + np.linspace(-1.0, 1.0, 401)
    bf = min(((mm_at(t), t) for t in fine))
    print(f"  mm(tau=0)            = {mm0:.4e}", flush=True)
    print(f"  coarse min           = {bc[0]:.4e} at tau={bc[1]:+.3f}s", flush=True)
    print(f"  fine   min           = {bf[0]:.4e} at tau*={bf[1]:+.4f}s", flush=True)
    print(f"  -> {'TIME SHIFT! (mm collapses)' if bf[0] < mm0/5 else 'not a pure time shift'}", flush=True)

    # band mm at tau=0 vs tau*
    def band_mm(lo, hi, tau):
        fdb = FDSettings(N=N_WIN // 2 + 1, df=1.0 / (N_WIN * DT), min_freq=lo, max_freq=hi, force_backend=BACKEND)
        a = AnalysisContainer(TDSignal(A, td_set).transform(fdb, window=win),
                              XYZ2SensitivityMatrix(fdb, model=SENS_MODEL))
        tsig = TDSignal(B, td_set).transform(fdb, window=win)
        fb = np.asarray(fdb.f_arr)
        tsig.arr[:] = np.asarray(tsig.arr) * np.exp(-2j * np.pi * fb * tau)[None, :]
        return 1 - abs(a.template_inner_product(tsig, normalize=True, complex=True))

    banner(f"band mm: tau=0  vs  tau*={bf[1]:+.4f}s")
    for lo, hi in [(5e-4, 1e-3), (1e-3, 3e-3), (3e-3, 8e-3), (8e-3, F_MAX)]:
        print(f"  [{lo:.1e},{hi:.1e}]  tau0={band_mm(lo,hi,0.0):.4e}   tau*={band_mm(lo,hi,bf[1]):.4e}", flush=True)
    print("\nDONE.", flush=True)


if __name__ == "__main__":
    main()
