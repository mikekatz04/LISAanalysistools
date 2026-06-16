"""Non-phase-marginalized match of the SOBBH templates vs mojito data, allowing
ONLY a best time shift (injection parameters otherwise fixed).

Reports, per template (legacy A / on-the-fly B), sensitivity, at the time shift
that maximizes Re<d|h>/sqrt(<h|h><d|d>):
  1-Re(O)  : NON-phase-marginalized mismatch (uses the injection phase)
  arg(O)   : residual constant phase [deg]  (what phase-marg would remove)
  1-|O|    : phase-marginalized mismatch (reference)
  logL     : -1/2 <d-h|d-h> at the injection phase
both at tau=0 (no shift) and at the best time shift tau*.

Uses the tc-consistent waveform (post-fix). 0.6yr window, Tukey 0.05.
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
SRC = int(os.environ.get("SOBBH_SRC", "0")); N_DAYS = float(os.environ.get("SOBBH_DAYS", "219"))
N_WIN = int(round(N_DAYS * 86400 / DT)); TOBS = N_WIN * DT
SENS = ["scirdv1", "mrdv1"]
DATA_CACHE = f"/tmp/sobbh_mojito_data_{int(N_DAYS)}d_src{SRC}.npz"


def banner(s): print("\n" + "=" * 100 + f"\n {s}\n" + "=" * 100, flush=True)
def tukey(N, a):
    from scipy.signal.windows import tukey as _t
    return _t(N, a)


def main():
    threading.Thread(target=_wd, daemon=True).start()
    z = np.load(DATA_CACHE, allow_pickle=True)
    D = np.asarray(z["data_td"])[:NCH]; data_t0 = float(z["data_t0"]); cat = z["cat"].item()
    g = lambda k: float(cat[k])
    pf = np.array([g("PrimaryMassSSBFrame"), g("SecondaryMassSSBFrame"),
                   g("PrimarySpinCompZ"), g("SecondarySpinCompZ"),
                   g("LuminosityDistance") / 1e3, g("InclinationAngle"),
                   g("GW22FrequencySSBFrame"), g("RightAscension") % (2 * np.pi),
                   g("Declination"), g("PolarisationAngle") % np.pi, g("TrueAnomaly")])
    m1, m2, s1, s2, dist, inc, f0, ra, dec, psi, phi0 = pf
    banner(f"NON-phase-marg match vs mojito  (src {SRC}, {N_DAYS:.0f}d, tc-consistent)")

    orb = L1Orbits(find_file(SOBHB_L1, "SOBHB", SRC), force_backend=BACKEND, frame="icrs")
    pad = 1.0e5; lo = max(data_t0 - pad, float(orb.sc_t0)); hi = min(data_t0 + TOBS + pad, float(orb._sc_t_base[-1]))
    lt = np.asarray(orb.ltt_t); mk = (lt >= lo) & (lt <= hi)
    orb.ltt = np.asarray(orb.ltt)[mk].copy(); orb.ltt_t = lt[mk].copy(); orb.ltt_t0 = float(orb.ltt_t[0])
    del lt; gc.collect(); orb.configure(linear_interp_setup=True)

    sobbh_gen = SOBBHWaveform(Tobs=TOBS, dt=DT, t0=data_t0, reference_time=REF, force_backend=BACKEND)
    legacy = ResponseWrapper(sobbh_gen, orbits=orb, t0=data_t0, Tobs=TOBS / YRSID_SI, dt=DT,
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

    win = tukey(N_WIN, TUKEY_ALPHA); td = TDSettings(N_WIN, DT, t0=0.0, force_backend=BACKEND)

    for name, T in [("A = legacy pyResponse", A), ("B = TDI-on-the-fly", B)]:
        banner(name)
        print(f"  {'sens':>8} {'shift':>10} {'1-Re(O)':>11} {'arg(O)deg':>10} {'1-|O|':>11} "
              f"{'tau*':>8} {'logL':>12}", flush=True)
        for sens in SENS:
            fd = FDSettings(N=N_WIN // 2 + 1, df=1.0 / (N_WIN * DT), min_freq=F_MIN, max_freq=F_MAX, force_backend=BACKEND)
            f = np.asarray(fd.f_arr)
            dsig = TDSignal(D, td).transform(fd, window=win)
            ac = AnalysisContainer(dsig, XYZ2SensitivityMatrix(fd, model=sens))
            tsig = TDSignal(T, td).transform(fd, window=win); t0arr = np.asarray(tsig.arr).copy()

            def at(tau):
                tsig.arr[:] = t0arr * np.exp(-2j * np.pi * f * tau)[None, :]
                O = ac.template_inner_product(tsig, normalize=True, complex=True)
                logL = float(np.real(ac.template_likelihood(tsig)))
                return O, logL
            # best time shift = max Re(O)  (non-phase-marg alignment)
            coarse = max(((np.real(at(x)[0]), x) for x in np.linspace(-3, 3, 241)))
            fine = max(((np.real(at(x)[0]), x) for x in coarse[1] + np.linspace(-0.02, 0.02, 81)))
            taustar = fine[1]
            for tag, tau in [("tau=0", 0.0), ("best", taustar)]:
                O, logL = at(tau)
                print(f"  {sens:>8} {tag:>10} {1-O.real:11.4e} {np.degrees(np.angle(O)):10.3f} "
                      f"{1-abs(O):11.4e} {tau:+8.4f} {logL:12.4e}", flush=True)
            tsig.arr[:] = t0arr
    print("\nDONE.", flush=True)


if __name__ == "__main__":
    main()
