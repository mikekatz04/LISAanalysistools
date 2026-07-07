"""Is the VGB mismatch floor a decimation artifact? HM Cnc + ZTFs at native
2.5 s vs naive ::4 decimation to 10 s, same 90-d window, same metric as
vgb_mojito_match.py."""
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

import gbgpu  # noqa: F401
from lisatools.detector import L1Orbits
from lisatools.globalfit.preprocessing import L1ProcessingStep, find_file
from lisatools.response.tdiconfig import TDIConfig
from lisatools.response.tdionfly import GBTDIonTheFly
from lisatools.domains import TDSettings, FDSettings, TDSignal
from lisatools.analysiscontainer import AnalysisContainer
from lisatools.sensitivity import XYZ2SensitivityMatrix

REF = 97729089.327664
PATH = "/Users/mkatz/.mojito_cache/brickmarket/mojito_light_v1_0_0/"
VGB_L1 = os.path.join(PATH, "data", "VGB", "L1")
BACKEND = "cpu"; TDI_GEN = "2nd generation"; NCH = 3; SENS = "scirdv1"
N_DAYS = 90.0
TUKEY_ALPHA = 0.05
BAND_UHZ = 5.0
NATIVE_CACHE = "/tmp/vgb_mojito_native_90d.npz"
FIELDS = ["Amplitude", "GW22FrequencySSBFrame", "GW22FrequencyDerivativeSourceFrame",
          "TrueAnomaly", "InclinationAngle", "PolarisationAngle", "RightAscension", "Declination"]


def tukey(N, a):
    from scipy.signal.windows import tukey as _t
    return _t(N, a)


def load_native():
    if os.path.exists(NATIVE_CACHE):
        z = np.load(NATIVE_CACHE)
        print(f"[cache] {NATIVE_CACHE}", flush=True)
        return (np.asarray(z["data_td"]), float(z["data_t0"]), float(z["dt"]),
                np.asarray(z["top_params"]))
    loader = L1ProcessingStep(L1_folder=PATH, source_types=["vgb"], source_ids=None,
                              orbits_class=L1Orbits, orbits_kwargs=dict(force_backend=BACKEND, frame="icrs"),
                              verbose=True)
    times = np.asarray(loader.times)
    data_full = np.asarray(loader.data)
    if data_full.shape[0] != 3: data_full = data_full.T
    dt_native = float(loader.dt); data_t0 = float(times[0])
    n_native = int(round(N_DAYS * 86400 / dt_native))
    data_td = data_full[:, :n_native].copy()
    cat0 = loader.catalogue["VGB"][0]
    f = np.asarray(cat0["GW22FrequencySSBFrame"], float)
    order = np.argsort(f)[::-1][:3]
    top_params = np.array([[float(np.asarray(cat0[k])[i]) for k in FIELDS] for i in order])
    del data_full, loader, cat0; gc.collect()
    np.savez(NATIVE_CACHE, data_td=data_td, data_t0=data_t0, dt=dt_native, top_params=top_params)
    print(f"[cache] wrote {NATIVE_CACHE}", flush=True)
    return data_td, data_t0, dt_native, top_params


def mm_for(D, dt, data_t0, params8, n_win):
    A0, f0, fdot, phi0, inc, psi, ra, dec = params8
    tobs = n_win * dt
    orb = L1Orbits(find_file(VGB_L1, "VGB", 0), force_backend=BACKEND, frame="icrs")
    pad = 1.0e5; lo = max(data_t0 - pad, float(orb.sc_t0)); hi = min(data_t0 + tobs + pad, float(orb._sc_t_base[-1]))
    lt = np.asarray(orb.ltt_t); mk = (lt >= lo) & (lt <= hi)
    orb.ltt = np.asarray(orb.ltt)[mk].copy(); orb.ltt_t = lt[mk].copy(); orb.ltt_t0 = float(orb.ltt_t[0])
    orb.configure(linear_interp_setup=True)
    tdi_config = TDIConfig(TDI_GEN, force_backend=BACKEND)
    grid = np.arange(n_win) * dt + data_t0
    t_nodes = np.linspace(grid[0], grid[-1], 16384)
    gb_gen = GBTDIonTheFly(t_nodes, tobs, REF, 1.0 / dt, 1,
                           tdi_config=tdi_config, orbits=orb, tdi_chan="XYZ", force_backend=BACKEND)
    params9 = np.array([A0, f0, fdot, 0.0, phi0, inc, psi, ra, dec]).reshape(9, 1)
    out = gb_gen(*params9, convert_to_ra_dec=False, return_spline=True)
    B = np.asarray(out.eval_tdi(grid)); B = (B[0] if B.ndim == 3 else B)[:NCH]
    win = tukey(n_win, TUKEY_ALPHA); td = TDSettings(n_win, dt, t0=0.0, force_backend=BACKEND)
    delta = BAND_UHZ * 1e-6
    fd = FDSettings(N=n_win // 2 + 1, df=1.0 / (n_win * dt), min_freq=f0 - delta, max_freq=f0 + delta,
                    force_backend=BACKEND)
    d = TDSignal(D[:, :n_win], td).transform(fd, window=win)
    t = TDSignal(B, td).transform(fd, window=win)
    ac = AnalysisContainer(d, XYZ2SensitivityMatrix(fd, model=SENS))
    O = ac.template_inner_product(t, normalize=True, complex=True)
    del B, out, gb_gen, orb; gc.collect()
    return 1 - O.real, 1 - abs(O)


def main():
    threading.Thread(target=_wd, daemon=True).start()
    Dn, data_t0, dt_native, top_params = load_native(); Dn = Dn[:NCH]; mark("native data loaded")
    deci = int(round(10.0 / dt_native))
    print(f"  dt_native={dt_native}  deci={deci}", flush=True)
    print(f"\n  {'rank':>4} {'f0(mHz)':>10} {'variant':>16} {'1-Re(O)':>12} {'1-|O|':>12}", flush=True)
    for rank in range(3):
        p8 = top_params[rank]; f0 = p8[1]
        n10 = int(round(N_DAYS * 86400 / 10.0))
        r10 = mm_for(Dn[:, ::deci].copy(), 10.0, data_t0, p8, n10)
        print(f"  {rank:>4} {f0*1e3:>10.5f} {'naive 10s':>16} {r10[0]:>+12.3e} {r10[1]:>12.3e}", flush=True)
        n_nat = int(round(N_DAYS * 86400 / dt_native))
        rn = mm_for(Dn, dt_native, data_t0, p8, n_nat)
        print(f"  {rank:>4} {f0*1e3:>10.5f} {'native 2.5s':>16} {rn[0]:>+12.3e} {rn[1]:>12.3e}", flush=True)
    print("\nDONE.", flush=True)


if __name__ == "__main__":
    main()
