"""AnalysisContainer likelihood / inner-product of the SOBBH templates vs the
mojito SOBHB data, scirdv1 + mrdv1.

  A = legacy pyResponse  : ResponseWrapper(SOBBHWaveform)  (flip_hx=False, REF)
  B = TDI-on-the-fly      : SOBBHTDIonFly

Reports per (template, sensitivity, band):  <d|d>, <d|h>, <h|h>, SNR_d, SNR_opt,
SNR_det, logL = -1/2 <d-h|d-h>, mm = 1-|O|.
Mirrors scripts/mbh/mbh_likelihood_compare.py.
"""
import os, sys, gc, time, threading, resource
os.environ.setdefault("OMP_NUM_THREADS", "8")
import numpy as np
MEM_CAP_GB = float(os.environ.get("SOBBH_MEM_CAP_GB", "7.5")); _IS_MAC = sys.platform == "darwin"
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
NF, NT = 512, 512; N_WIN = NF * NT; TOBS = N_WIN * DT
F_MIN, F_MAX = 0.010, 0.030
ORDER = 40; T_BUFFER = 3.0e4; TUKEY_ALPHA = 0.1
SRC = int(os.environ.get("SOBBH_SRC", "0"))
SENS = ["scirdv1", "mrdv1"]
DATA_CACHE = f"/tmp/sobbh_mojito_data_src{SRC}.npz"


def banner(s): print("\n" + "=" * 96 + f"\n {s}\n" + "=" * 96, flush=True)
def tukey(N, a):
    from scipy.signal.windows import tukey as _t
    return _t(N, a)


def main():
    threading.Thread(target=_wd, daemon=True).start()
    banner(f"SOBBH likelihood / inner-product vs mojito data (src {SRC}, scirdv1 + mrdv1)")
    z = np.load(DATA_CACHE, allow_pickle=True)
    D = np.asarray(z["data_td"])[:NCH]; data_t0 = float(z["data_t0"]); cat = z["cat"].item()
    g = lambda k: float(cat[k])
    pf = np.array([
        g("PrimaryMassSSBFrame"), g("SecondaryMassSSBFrame"),
        g("PrimarySpinCompZ"), g("SecondarySpinCompZ"),
        g("LuminosityDistance") / 1e3, g("InclinationAngle"),
        g("GW22FrequencySSBFrame"),
        g("RightAscension") % (2 * np.pi), g("Declination"),
        g("PolarisationAngle") % np.pi, g("TrueAnomaly"),
    ])
    m1, m2, s1, s2, dist, inc, f0, ra, dec, psi, phi0 = pf
    print(f"  f0={f0*1e3:.4f} mHz  inc={inc:.4f}", flush=True)

    orb = L1Orbits(find_file(SOBHB_L1, "SOBHB", SRC), force_backend=BACKEND, frame="icrs")
    pad = 1.0e5; lo = max(data_t0 - pad, float(orb.sc_t0)); hi = min(data_t0 + TOBS + pad, float(orb._sc_t_base[-1]))
    lt = np.asarray(orb.ltt_t); mk = (lt >= lo) & (lt <= hi)
    orb.ltt = np.asarray(orb.ltt)[mk].copy(); orb.ltt_t = lt[mk].copy(); orb.ltt_t0 = float(orb.ltt_t[0])
    del lt; gc.collect(); orb.configure(linear_interp_setup=True); mark("orbit ready")

    sobbh_gen = SOBBHWaveform(Tobs=TOBS, dt=DT, t0=data_t0, reference_time=REF, force_backend=BACKEND)
    legacy = ResponseWrapper(
        sobbh_gen, orbits=orb, t0=data_t0, Tobs=TOBS / YRSID_SI, dt=DT,
        index_lambda=7, index_beta=8, flip_hx=False, tdi=TDIConfig(TDI_GEN, force_backend=BACKEND),
        tdi_chan="XYZ", order=ORDER, remove_garbage="zero", is_ecliptic_latitude=True,
        t_buffer=T_BUFFER, force_backend=BACKEND)
    A = np.atleast_2d(np.asarray(legacy(*pf, convert_to_ra_dec=False)))[:NCH]
    A = (np.pad(A, ((0, 0), (0, N_WIN - A.shape[-1]))) if A.shape[-1] < N_WIN else A[:, :N_WIN]); mark("A built")

    fly = SOBBHTDIonFly(
        SOBBHWaveform(Tobs=TOBS, dt=DT, t0=data_t0, reference_time=REF, force_backend=BACKEND),
        orb, TDIConfig(TDI_GEN, force_backend=BACKEND), DT, TOBS, t0=data_t0, force_backend=BACKEND)
    grid_t = np.arange(N_WIN) * DT + data_t0
    B = np.asarray(fly(m1, m2, s1, s2, dist, f0, phi0, inc, ra, dec, psi,
                       upsample_t_arr=grid_t, combine=True))[:NCH]; mark("B built")

    # Persist the three TD-domain TDI arrays so the likelihood can be recomputed
    # by a process that imports ONLY the lisatools likelihood machinery (no
    # waveform/response generators) -- see sobbh_likelihood_pure.py.
    arr_cache = f"/tmp/sobbh_ll_arrays_src{SRC}.npz"
    np.savez(arr_cache, D=D, A=A, B=B, dt=DT, f0=f0, data_t0=data_t0,
             band_lo=np.array([F_MIN, f0 - 2e-3]), band_hi=np.array([F_MAX, f0 + 2e-3]),
             band_label=np.array(["full[10,30]mHz", "carrier+-2mHz"]))
    print(f"  saved TD arrays -> {arr_cache}", flush=True)

    win = tukey(N_WIN, TUKEY_ALPHA); td = TDSettings(N_WIN, DT, t0=0.0, force_backend=BACKEND)
    bands = [("full[10,30]mHz", F_MIN, F_MAX), ("carrier+-2mHz", f0 - 2e-3, f0 + 2e-3)]

    for name, T in [("A = legacy pyResponse", A), ("B = TDI-on-the-fly", B)]:
        banner(name + "   vs mojito data")
        print(f"  {'sens':>8} {'band':>16} {'<d|d>':>11} {'<d|h>':>11} {'<h|h>':>11} "
              f"{'SNRd':>7} {'SNRopt':>7} {'SNRdet':>8} {'logL':>12} {'mm':>10}", flush=True)
        for sens in SENS:
            for tag, flo, fhi in bands:
                fd = FDSettings(N=N_WIN // 2 + 1, df=1.0 / (N_WIN * DT), min_freq=flo, max_freq=fhi, force_backend=BACKEND)
                d = TDSignal(D, td).transform(fd, window=win)
                t = TDSignal(T, td).transform(fd, window=win)
                ac = AnalysisContainer(d, XYZ2SensitivityMatrix(fd, model=sens))
                dd = ac.inner_product().real
                dh = float(np.real(ac.template_inner_product(t, normalize=False, complex=False)))
                opt, det = ac.template_snr(t)
                hh = float(opt) ** 2
                logL = float(np.real(ac.template_likelihood(t)))
                O = ac.template_inner_product(t, normalize=True, complex=True)
                print(f"  {sens:>8} {tag:>16} {dd:11.3e} {dh:11.3e} {hh:11.3e} "
                      f"{np.sqrt(dd):7.1f} {float(opt):7.1f} {float(det):+8.2f} {logL:12.4e} {1-abs(O):10.3e}",
                      flush=True)
    print("\nDONE.", flush=True)


if __name__ == "__main__":
    main()
