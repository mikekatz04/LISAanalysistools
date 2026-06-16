"""Pin the on-the-fly mismatch floor vs the amp/phase node count n_grid at a
configurable baseline (SOBBH_DAYS). Build the legacy template ONCE, then sweep
SOBBHTDIonFly's n_grid and measure mm + logL vs the mojito data.

The SOBBH GW phase 2*Phi is smooth but accumulates ~2*pi*f0*Tobs over the
baseline; with a FIXED n_grid the node spacing grows with Tobs, so a too-coarse
grid stops resolving the phase and the mm rises. This sweep finds the n_grid
where the on-the-fly mm plateaus at its true floor for the given baseline.

Reuses the data cache from sobbh_logL_window.py
(/tmp/sobbh_mojito_data_{DAYS}d_src{SRC}.npz). scirdv1, Tukey alpha=0.05, full
band. Reports node spacing so the result is baseline-independent.
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
F_MIN, F_MAX = 0.010, 0.030
ORDER = 40; T_BUFFER = 3.0e4; TUKEY_ALPHA = 0.05; SENS = "scirdv1"
SRC = int(os.environ.get("SOBBH_SRC", "0"))
N_DAYS = float(os.environ.get("SOBBH_DAYS", "365"))
N_WIN = int(round(N_DAYS * 86400 / DT)); TOBS = N_WIN * DT
GRIDS = [int(x) for x in os.environ.get("SOBBH_GRIDS", "2048,8192,32768,131072").split(",")]
DATA_CACHE = f"/tmp/sobbh_mojito_data_{int(N_DAYS)}d_src{SRC}.npz"


def banner(s): print("\n" + "=" * 84 + f"\n {s}\n" + "=" * 84, flush=True)
def tukey(N, a):
    from scipy.signal.windows import tukey as _t
    return _t(N, a)


def main():
    threading.Thread(target=_wd, daemon=True).start()
    banner(f"SOBBH {N_DAYS:.0f} d n_grid sweep  (Tukey {TUKEY_ALPHA}, {SENS})  grids={GRIDS}")
    if not os.path.exists(DATA_CACHE):
        raise SystemExit(f"missing {DATA_CACHE}; run sobbh_logL_window.py SOBBH_DAYS={int(N_DAYS)} first.")
    z = np.load(DATA_CACHE, allow_pickle=True)
    D = np.asarray(z["data_td"])[:NCH]; data_t0 = float(z["data_t0"]); cat = z["cat"].item()
    g = lambda k: float(cat[k])
    pf = np.array([g("PrimaryMassSSBFrame"), g("SecondaryMassSSBFrame"),
                   g("PrimarySpinCompZ"), g("SecondarySpinCompZ"),
                   g("LuminosityDistance") / 1e3, g("InclinationAngle"),
                   g("GW22FrequencySSBFrame"), g("RightAscension") % (2 * np.pi),
                   g("Declination"), g("PolarisationAngle") % np.pi, g("TrueAnomaly")])
    m1, m2, s1, s2, dist, inc, f0, ra, dec, psi, phi0 = pf
    print(f"  N_WIN={N_WIN} ({N_DAYS:.0f} d)  f0={f0*1e3:.4f} mHz", flush=True)

    orb = L1Orbits(find_file(SOBHB_L1, "SOBHB", SRC), force_backend=BACKEND, frame="icrs")
    pad = 1.0e5; lo = max(data_t0 - pad, float(orb.sc_t0)); hi = min(data_t0 + TOBS + pad, float(orb._sc_t_base[-1]))
    lt = np.asarray(orb.ltt_t); mk = (lt >= lo) & (lt <= hi)
    orb.ltt = np.asarray(orb.ltt)[mk].copy(); orb.ltt_t = lt[mk].copy(); orb.ltt_t0 = float(orb.ltt_t[0])
    del lt; gc.collect(); orb.configure(linear_interp_setup=True); mark("orbit ready")

    win = tukey(N_WIN, TUKEY_ALPHA); td = TDSettings(N_WIN, DT, t0=0.0, force_backend=BACKEND)
    fd = FDSettings(N=N_WIN // 2 + 1, df=1.0 / (N_WIN * DT), min_freq=F_MIN, max_freq=F_MAX, force_backend=BACKEND)
    d = TDSignal(D, td).transform(fd, window=win)
    ac = AnalysisContainer(d, XYZ2SensitivityMatrix(fd, model=SENS))

    def report(tag, T):
        t = TDSignal(T, td).transform(fd, window=win)
        O = ac.template_inner_product(t, normalize=True, complex=True)
        logL = float(np.real(ac.template_likelihood(t)))
        print(f"  {tag:24s}  mm={1-abs(O):.4e}  logL={logL:+.4e}", flush=True)

    sobbh_gen = SOBBHWaveform(Tobs=TOBS, dt=DT, t0=data_t0, reference_time=REF, force_backend=BACKEND)
    legacy = ResponseWrapper(
        sobbh_gen, orbits=orb, t0=data_t0, Tobs=TOBS / YRSID_SI, dt=DT,
        index_lambda=7, index_beta=8, flip_hx=False, tdi=TDIConfig(TDI_GEN, force_backend=BACKEND),
        tdi_chan="XYZ", order=ORDER, remove_garbage="zero", is_ecliptic_latitude=True,
        t_buffer=T_BUFFER, force_backend=BACKEND)
    A = np.atleast_2d(np.asarray(legacy(*pf, convert_to_ra_dec=False)))[:NCH]
    A = (np.pad(A, ((0, 0), (0, N_WIN - A.shape[-1]))) if A.shape[-1] < N_WIN else A[:, :N_WIN]); mark("A built")
    banner("legacy pyResponse (reference)")
    report("legacy", A)

    grid_t = np.arange(N_WIN) * DT + data_t0
    banner("on-the-fly vs n_grid  (node spacing = Tobs / n_grid)")
    for ng in GRIDS:
        fly = SOBBHTDIonFly(
            SOBBHWaveform(Tobs=TOBS, dt=DT, t0=data_t0, reference_time=REF, force_backend=BACKEND),
            orb, TDIConfig(TDI_GEN, force_backend=BACKEND), DT, TOBS, t0=data_t0,
            n_grid=ng, force_backend=BACKEND)
        B = np.asarray(fly(m1, m2, s1, s2, dist, f0, phi0, inc, ra, dec, psi,
                           upsample_t_arr=grid_t, combine=True))[:NCH]
        report(f"n_grid={ng:>7} ({TOBS/ng/3600:5.2f} h/node)", B)
        del fly, B; gc.collect()
    print("\nDONE.", flush=True)


if __name__ == "__main__":
    main()
