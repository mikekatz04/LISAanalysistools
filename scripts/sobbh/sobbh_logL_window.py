"""SOBBH logL vs mojito over a configurable (default 3-month) Tukey-windowed
window. Legacy pyResponse (A) vs TDI-on-the-fly (B), scirdv1 + mrdv1.

Windowing: Tukey(N, alpha=0.1) applied to data and both templates identically
at the TDSignal->FD transform (same as the 30-day comparison).

SOBBH is a continuous near-monochromatic source, so a longer baseline grows the
accumulated SNR^2 -- and with it the logL gap between the legacy response and
the on-the-fly response. Set SOBBH_DAYS to change the baseline.

logL = -1/2 <d-h|d-h> via AnalysisContainer (pure lisatools). The three TD
arrays are also dumped to /tmp/sobbh_ll_arrays_3mo_src{SRC}.npz for an
independent pure recompute (scripts/sobbh/likelihood_pure.py).
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
from lisatools.globalfit.preprocessing import L1ProcessingStep, find_file
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
ORDER = 40; T_BUFFER = 3.0e4
TUKEY_ALPHA = float(os.environ.get("SOBBH_TUKEY_ALPHA", "0.05"))
SRC = int(os.environ.get("SOBBH_SRC", "0"))
N_GRID = int(os.environ.get("SOBBH_NGRID", "2048"))
N_DAYS = float(os.environ.get("SOBBH_DAYS", "90"))
N_WIN = int(round(N_DAYS * 86400 / DT)); TOBS = N_WIN * DT
SENS = ["scirdv1", "mrdv1"]
DATA_CACHE = f"/tmp/sobbh_mojito_data_{int(N_DAYS)}d_src{SRC}.npz"


def banner(s): print("\n" + "=" * 96 + f"\n {s}\n" + "=" * 96, flush=True)
def tukey(N, a):
    from scipy.signal.windows import tukey as _t
    return _t(N, a)


def load_data():
    if os.path.exists(DATA_CACHE):
        z = np.load(DATA_CACHE, allow_pickle=True)
        print(f"[cache] data window from {DATA_CACHE}", flush=True)
        return np.asarray(z["data_td"]), float(z["data_t0"]), z["cat"].item()
    print(f"[cache] MISS -> reading mojito for {N_DAYS:.0f} d ...", flush=True)
    loader = L1ProcessingStep(
        L1_folder=PATH, source_types=["sobhb"], source_ids={"sobhb": SRC},
        orbits_class=L1Orbits, orbits_kwargs=dict(force_backend=BACKEND, frame="icrs"),
        verbose=True)
    times = np.asarray(loader.times); data_full = np.asarray(loader.data)
    dt_native = float(loader.dt); data_t0 = float(times[0])
    cat = {k: float(np.asarray(v)) for k, v in loader.catalogue["SOBHB"][SRC].items()
           if np.asarray(v).dtype.kind in "fi"}
    deci = int(round(DT / dt_native))
    data_td = data_full[:, : N_WIN * deci : deci][:, :N_WIN].copy()
    del data_full, loader; gc.collect()
    np.savez(DATA_CACHE, data_td=data_td, data_t0=data_t0, cat=cat)
    print(f"[cache] wrote {DATA_CACHE}", flush=True)
    return data_td, data_t0, cat


def main():
    threading.Thread(target=_wd, daemon=True).start()
    banner(f"SOBBH logL over {N_DAYS:.0f} d  (Tukey alpha={TUKEY_ALPHA})  src {SRC}")
    D, data_t0, cat = load_data(); D = D[:NCH]
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
    print(f"  N_WIN={N_WIN} ({N_WIN*DT/86400:.1f} d)  f0={f0*1e3:.4f} mHz  inc={inc:.4f}", flush=True)

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
        orb, TDIConfig(TDI_GEN, force_backend=BACKEND), DT, TOBS, t0=data_t0,
        n_grid=N_GRID, force_backend=BACKEND)
    grid_t = np.arange(N_WIN) * DT + data_t0
    B = np.asarray(fly(m1, m2, s1, s2, dist, f0, phi0, inc, ra, dec, psi,
                       upsample_t_arr=grid_t, combine=True))[:NCH]; mark("B built")

    bands = [("full[10,30]mHz", F_MIN, F_MAX), ("carrier+-2mHz", f0 - 2e-3, f0 + 2e-3)]
    np.savez(f"/tmp/sobbh_ll_arrays_3mo_src{SRC}.npz", D=D, A=A, B=B, dt=DT,
             band_lo=np.array([b[1] for b in bands]), band_hi=np.array([b[2] for b in bands]),
             band_label=np.array([b[0] for b in bands]))

    win = tukey(N_WIN, TUKEY_ALPHA); td = TDSettings(N_WIN, DT, t0=0.0, force_backend=BACKEND)
    for name, T in [("A = legacy pyResponse", A), ("B = TDI-on-the-fly", B)]:
        banner(f"{name}   vs mojito data  ({N_DAYS:.0f} d, Tukey {TUKEY_ALPHA})")
        print(f"  {'sens':>8} {'band':>16} {'<d|d>':>11} {'<h|h>':>11} {'SNRopt':>7} "
              f"{'SNRdet':>8} {'logL':>12} {'mm':>10}", flush=True)
        for sens in SENS:
            for tag, flo, fhi in bands:
                fd = FDSettings(N=N_WIN // 2 + 1, df=1.0 / (N_WIN * DT), min_freq=flo, max_freq=fhi, force_backend=BACKEND)
                d = TDSignal(D, td).transform(fd, window=win)
                t = TDSignal(T, td).transform(fd, window=win)
                ac = AnalysisContainer(d, XYZ2SensitivityMatrix(fd, model=sens))
                dd = ac.inner_product().real
                opt, det = ac.template_snr(t)
                logL = float(np.real(ac.template_likelihood(t)))
                O = ac.template_inner_product(t, normalize=True, complex=True)
                print(f"  {sens:>8} {tag:>16} {dd:11.3e} {float(opt)**2:11.3e} {float(opt):7.1f} "
                      f"{float(det):+8.2f} {logL:12.4e} {1-abs(O):10.3e}", flush=True)
    print("\nDONE.", flush=True)


if __name__ == "__main__":
    main()
