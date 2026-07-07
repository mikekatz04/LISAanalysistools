"""SOBBH fidelity vs mojito data at the SAME metric as vgb_mojito_match.py.

Apples-to-apples companion to scripts/gb/vgb_mojito_match.py: single-source
SOBHB L1 stream, SOBBHTDIonFly (validated TD path) template, tukey 0.05,
scirdv1, mm = 1-|O| over a carrier band sized to hold the source's frequency
drift, at BOTH 30 d and 90 d. Purpose: determine whether the ~3e-6 (ZTF) /
2.4e-5 (HM Cnc) VGB floor is GB-specific or a shared response-level floor
that simply wasn't visible in the earlier 30-d / tukey-0.1 SOBBH checks.
"""
import os, sys, gc, time, threading, resource
os.environ.setdefault("OMP_NUM_THREADS", "8")
import numpy as np
MEM_CAP_GB = float(os.environ.get("GB_MEM_CAP_GB", "8.2")); _IS_MAC = sys.platform == "darwin"
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
from lisatools.response.tdiconfig import TDIConfig
from lisatools.sources.sobbh.waveform import SOBBHWaveform
from lisatools.domains import TDSettings, FDSettings, TDSignal
from lisatools.analysiscontainer import AnalysisContainer
from lisatools.sensitivity import XYZ2SensitivityMatrix
from bbhx.sobbhtdionfly import SOBBHTDIonFly

REF = 97729089.327664
PATH = "/Users/mkatz/.mojito_cache/brickmarket/mojito_light_v1_0_0/"
SOBHB_L1 = os.path.join(PATH, "data", "SOBHB", "L1")
BACKEND = "cpu"; DT = 10.0; TDI_GEN = "2nd generation"; NCH = 3; SENS = "scirdv1"
N_DAYS_MAX = 90.0; N_WIN_MAX = int(round(N_DAYS_MAX * 86400 / DT))
TUKEY_ALPHA = float(os.environ.get("GB_TUKEY_ALPHA", "0.05"))
SRC = int(os.environ.get("SOBBH_SRC", "0"))
DATA_CACHE = f"/tmp/sobbh_same_metric_src{SRC}_90d.npz"


def tukey(N, a):
    from scipy.signal.windows import tukey as _t
    return _t(N, a)


def load_cached():
    if os.path.exists(DATA_CACHE):
        z = np.load(DATA_CACHE, allow_pickle=True)
        print(f"[cache] {DATA_CACHE}", flush=True)
        return np.asarray(z["data_td"]), float(z["data_t0"]), z["cat"].item()
    print("[cache] MISS -> reading SOBHB L1 (one time)...", flush=True)
    loader = L1ProcessingStep(L1_folder=PATH, source_types=["sobhb"], source_ids={"sobhb": SRC},
                              orbits_class=L1Orbits, orbits_kwargs=dict(force_backend=BACKEND, frame="icrs"),
                              verbose=True)
    times = np.asarray(loader.times); data_full = np.asarray(loader.data)
    if data_full.shape[0] != 3: data_full = data_full.T
    dt_native = float(loader.dt); data_t0 = float(times[0]); deci = int(round(DT / dt_native))
    data_td = data_full[:, : N_WIN_MAX * deci : deci][:, :N_WIN_MAX].copy()
    cat = {k: float(np.asarray(v)) for k, v in loader.catalogue["SOBHB"][SRC].items()
           if np.asarray(v).dtype.kind in "fi"}
    del data_full, loader; gc.collect()
    np.savez(DATA_CACHE, data_td=data_td, data_t0=data_t0, cat=cat)
    print(f"[cache] wrote {DATA_CACHE}", flush=True)
    return data_td, data_t0, cat


def main():
    threading.Thread(target=_wd, daemon=True).start()
    D90, data_t0, cat = load_cached(); D90 = D90[:NCH]; mark("data loaded")
    g = lambda k: float(cat[k])
    m1, m2 = g("PrimaryMassSSBFrame"), g("SecondaryMassSSBFrame")
    s1 = g("PrimarySpinCompZ") if "PrimarySpinCompZ" in cat else 0.0
    s2 = g("SecondarySpinCompZ") if "SecondarySpinCompZ" in cat else 0.0
    dist = g("LuminosityDistance") / 1e3
    inc, f0 = g("InclinationAngle"), g("GW22FrequencySSBFrame")
    ra, dec = g("RightAscension") % (2 * np.pi), g("Declination")
    psi, phi0 = g("PolarisationAngle") % np.pi, g("TrueAnomaly")
    fdot_keys = [k for k in cat if "FrequencyDerivative" in k]
    fdot = g(fdot_keys[0]) if fdot_keys else 0.0
    print(f"  src{SRC}: f0={f0*1e3:.4f} mHz  fdot={fdot:.3e}  m1={m1:.1f} m2={m2:.1f}  "
          f"data_t0-REF={data_t0-REF:.2f}s", flush=True)

    orb = L1Orbits(find_file(SOBHB_L1, "SOBHB", SRC), force_backend=BACKEND, frame="icrs")
    orb.configure(linear_interp_setup=True); mark("orbit ready")
    tdi_config = TDIConfig(TDI_GEN, force_backend=BACKEND)

    print(f"\n  {'days':>5} {'band':>22} {'1-Re(O)':>11} {'1-|O|':>11} {'SNRdet':>8}", flush=True)
    for n_days in (30.0, 90.0):
        n_win = int(round(n_days * 86400 / DT)); tobs = n_win * DT
        D = D90[:, :n_win]
        fly = SOBBHTDIonFly(
            SOBBHWaveform(Tobs=tobs, dt=DT, t0=data_t0, reference_time=REF, force_backend=BACKEND),
            orb, tdi_config, DT, tobs, t0=data_t0, force_backend=BACKEND)
        grid_t = np.arange(n_win) * DT + data_t0
        B = np.asarray(fly(m1, m2, s1, s2, dist, f0, phi0, inc, ra, dec, psi,
                           upsample_t_arr=grid_t, combine=True))[:NCH]
        win = tukey(n_win, TUKEY_ALPHA); td = TDSettings(n_win, DT, t0=0.0, force_backend=BACKEND)
        drift = abs(fdot) * tobs
        bands = [("carrier+drift+-50uHz", f0 - 5e-5, f0 + drift + 5e-5),
                 ("carrier+-2mHz", f0 - 2e-3, f0 + 2e-3)]
        for tag, flo, fhi in bands:
            fd = FDSettings(N=n_win // 2 + 1, df=1.0 / (n_win * DT), min_freq=flo, max_freq=fhi,
                            force_backend=BACKEND)
            d = TDSignal(D, td).transform(fd, window=win)
            t = TDSignal(B, td).transform(fd, window=win)
            ac = AnalysisContainer(d, XYZ2SensitivityMatrix(fd, model=SENS))
            O = ac.template_inner_product(t, normalize=True, complex=True)
            _, det = ac.template_snr(t)
            print(f"  {n_days:5.0f} {tag:>22} {1-O.real:>+11.3e} {1-abs(O):>11.3e} {float(det):>+8.2f}",
                  flush=True)
        del B, fly; gc.collect()
    print("\nDONE.", flush=True)


if __name__ == "__main__":
    main()
