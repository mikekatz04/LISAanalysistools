"""Wire + localize the ~1s reference-time offset between our SOBBH templates and
the mojito data (revealed by the time-max at 0.6yr: tau*=+0.975s, shared by
legacy and on-the-fly).

Rebuilds A (legacy, via SOBBHWaveform.reference_time) and B (on-the-fly, via the
SOBBHTDIonFly t_shift arg) with the shift applied, and reports the phase-max mm
vs data WITHOUT time-max. If the source-side shift recovers the time-max floor
(B -> ~2.6e-10, A -> ~2.2e-6), the offset is a source/epoch convention (not the
response), and t_shift is the fix.

Reuses /tmp/sobbh_mojito_data_219d_src0.npz. scirdv1, Tukey 0.05, full band.
"""
import os, sys, gc, time, threading, resource
os.environ.setdefault("OMP_NUM_THREADS", "8")
import numpy as np
MEM_CAP_GB = float(os.environ.get("SOBBH_MEM_CAP_GB", "7.8")); _IS_MAC = sys.platform == "darwin"
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
F_MIN, F_MAX = 0.010, 0.030; ORDER = 40; T_BUFFER = 3.0e4; TUKEY_ALPHA = 0.05
SRC = int(os.environ.get("SOBBH_SRC", "0")); N_DAYS = 219.0
N_WIN = int(round(N_DAYS * 86400 / DT)); TOBS = N_WIN * DT
TSHIFT = float(os.environ.get("SOBBH_TSHIFT", "0.975"))
DATA_CACHE = f"/tmp/sobbh_mojito_data_{int(N_DAYS)}d_src{SRC}.npz"


def banner(s): print("\n" + "=" * 86 + f"\n {s}\n" + "=" * 86, flush=True)
def tukey(N, a):
    from scipy.signal.windows import tukey as _t
    return _t(N, a)


def main():
    threading.Thread(target=_wd, daemon=True).start()
    banner(f"SOBBH ~1s t_shift fix  (apply +{TSHIFT}s)  0.6yr src {SRC}")
    z = np.load(DATA_CACHE, allow_pickle=True)
    D = np.asarray(z["data_td"])[:NCH]; data_t0 = float(z["data_t0"]); cat = z["cat"].item()
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
    del lt; gc.collect(); orb.configure(linear_interp_setup=True); mark("orbit ready")

    win = tukey(N_WIN, TUKEY_ALPHA); td = TDSettings(N_WIN, DT, t0=0.0, force_backend=BACKEND)
    fd = FDSettings(N=N_WIN // 2 + 1, df=1.0 / (N_WIN * DT), min_freq=F_MIN, max_freq=F_MAX, force_backend=BACKEND)
    f = np.asarray(fd.f_arr)
    ac = AnalysisContainer(TDSignal(D, td).transform(fd, window=win), XYZ2SensitivityMatrix(fd, model="scirdv1"))

    def mm_pm_tm(T, label):
        sig = TDSignal(T, td).transform(fd, window=win); t0 = np.asarray(sig.arr).copy()
        pm = 1 - abs(ac.template_inner_product(sig, normalize=True, complex=True))
        def mm(tau):
            sig.arr[:] = t0 * np.exp(-2j * np.pi * f * tau)[None, :]
            return 1 - abs(ac.template_inner_product(sig, normalize=True, complex=True))
        c = min(((mm(x), x) for x in np.linspace(-3, 3, 121)))
        fine = min(((mm(x), x) for x in c[1] + np.linspace(-0.05, 0.05, 101)))
        sig.arr[:] = t0
        print(f"  {label:28s}  phase-max mm={pm:.4e}   +time-max mm={fine[0]:.4e} @ tau*={fine[1]:+.4f}s", flush=True)

    grid_t = np.arange(N_WIN) * DT + data_t0

    def build_B(tshift):
        fly = SOBBHTDIonFly(
            SOBBHWaveform(Tobs=TOBS, dt=DT, t0=data_t0, reference_time=REF, force_backend=BACKEND),
            orb, TDIConfig(TDI_GEN, force_backend=BACKEND), DT, TOBS, t0=data_t0, force_backend=BACKEND)
        return np.asarray(fly(m1, m2, s1, s2, dist, f0, phi0, inc, ra, dec, psi,
                              t_shift=tshift, upsample_t_arr=grid_t, combine=True))[:NCH]

    def build_A(ref_time):
        gen = SOBBHWaveform(Tobs=TOBS, dt=DT, t0=data_t0, reference_time=ref_time, force_backend=BACKEND)
        lw = ResponseWrapper(gen, orbits=orb, t0=data_t0, Tobs=TOBS / YRSID_SI, dt=DT,
                             index_lambda=7, index_beta=8, flip_hx=False,
                             tdi=TDIConfig(TDI_GEN, force_backend=BACKEND), tdi_chan="XYZ", order=ORDER,
                             remove_garbage="zero", is_ecliptic_latitude=True, t_buffer=T_BUFFER,
                             force_backend=BACKEND)
        a = np.atleast_2d(np.asarray(lw(*pf, convert_to_ra_dec=False)))[:NCH]
        return np.pad(a, ((0, 0), (0, N_WIN - a.shape[-1]))) if a.shape[-1] < N_WIN else a[:, :N_WIN]

    banner("on-the-fly B  (SOBBHTDIonFly t_shift)")
    mm_pm_tm(build_B(0.0), "B  t_shift=0 (baseline)"); mark("B0")
    mm_pm_tm(build_B(+TSHIFT), f"B  t_shift=+{TSHIFT}"); mark("B+")
    # reference_time shift is equivalent to t_shift via pn_times = t - ref - tshift
    banner("legacy A  (SOBBHWaveform reference_time +/- shift)")
    mm_pm_tm(build_A(REF), "A  ref=REF (baseline)"); mark("A0")
    mm_pm_tm(build_A(REF + TSHIFT), f"A  ref=REF+{TSHIFT}"); mark("A+ref")
    print("\nDONE.", flush=True)


if __name__ == "__main__":
    main()
