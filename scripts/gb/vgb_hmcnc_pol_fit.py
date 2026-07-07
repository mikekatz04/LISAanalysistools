"""HM Cnc polarization-content fit.

The response is linear in (hSp, hSc):
    hSp = -amp (1+cos^2 i) cos(phase),  hSc = -amp (2 cos i) sin(phase).
Build two unit basis TDI templates with the SAME amp/phase/psi/sky:
    T_P = T(inc=pi/2)              -> response to (-amp cos(phase), 0)
    T_C = (T(inc=0) - 2 T_P) / 2   -> response to (0, -amp sin(phase))
and LSQ-fit complex coefficients (a, b):  d ~ a T_P + b T_C  in the whitened
FD band. Catalogue prediction: a = 1+cos^2(i) (= 1.621 for HM Cnc),
b = 2 cos(i) (= 1.576), both real and positive relative to the baseline.
Reports fitted a, b, |b/a| vs predicted, relative phase, and residual mm
after the 2-basis fit (if ~ZTF floor, the whole anomaly is polarization
content).
"""
import os, gc, threading, time, resource, sys
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

import gbgpu  # noqa: F401
from lisatools.detector import L1Orbits
from lisatools.globalfit.preprocessing import find_file
from lisatools.response.tdiconfig import TDIConfig
from lisatools.response.tdionfly import GBTDIonTheFly
from lisatools.domains import TDSettings, FDSettings, TDSignal
from lisatools.analysiscontainer import AnalysisContainer
from lisatools.sensitivity import XYZ2SensitivityMatrix

REF = 97729089.327664
PATH = "/Users/mkatz/.mojito_cache/brickmarket/mojito_light_v1_0_0/"
VGB_L1 = os.path.join(PATH, "data", "VGB", "L1")
BACKEND = "cpu"; DT = 10.0; TDI_GEN = "2nd generation"; NCH = 3; SENS = "scirdv1"
N_DAYS = float(os.environ.get("GB_DAYS", "30")); N_WIN = int(round(N_DAYS * 86400 / DT)); TOBS = N_WIN * DT
TUKEY_ALPHA = 0.05; BAND_UHZ = 5.0
RANK = int(os.environ.get("VGB_RANK", "0"))
DATA_CACHE = f"/tmp/vgb_mojito_data_{int(N_DAYS)}d.npz"


def tukey(N, a):
    from scipy.signal.windows import tukey as _t
    return _t(N, a)


def main():
    threading.Thread(target=_wd, daemon=True).start()
    z = np.load(DATA_CACHE)
    D = np.asarray(z["data_td"])[:NCH]; data_t0 = float(z["data_t0"])
    A0, f0, fdot, phi0, inc, psi, ra, dec = np.asarray(z["top_params"])[RANK]
    print(f"rank {RANK}: f0={f0*1e3:.5f}  inc={inc:.4f}  a_cat={1+np.cos(inc)**2:.4f}  "
          f"b_cat={2*np.cos(inc):.4f}  ratio_cat={2*np.cos(inc)/(1+np.cos(inc)**2):.5f}", flush=True)

    orb = L1Orbits(find_file(VGB_L1, "VGB", 0), force_backend=BACKEND, frame="icrs")
    pad = 1.0e5; lo = max(data_t0 - pad, float(orb.sc_t0)); hi = min(data_t0 + TOBS + pad, float(orb._sc_t_base[-1]))
    lt = np.asarray(orb.ltt_t); mk = (lt >= lo) & (lt <= hi)
    orb.ltt = np.asarray(orb.ltt)[mk].copy(); orb.ltt_t = lt[mk].copy(); orb.ltt_t0 = float(orb.ltt_t[0])
    del lt; gc.collect(); orb.configure(linear_interp_setup=True)

    tdi_config = TDIConfig(TDI_GEN, force_backend=BACKEND)
    grid = np.arange(N_WIN) * DT + data_t0
    t_nodes = np.linspace(grid[0], grid[-1], 16384)
    gb_gen = GBTDIonTheFly(t_nodes, TOBS, REF, 1.0 / DT, 1,
                           tdi_config=tdi_config, orbits=orb, tdi_chan="XYZ", force_backend=BACKEND)
    win = tukey(N_WIN, TUKEY_ALPHA); td = TDSettings(N_WIN, DT, t0=0.0, force_backend=BACKEND)
    delta = BAND_UHZ * 1e-6
    fd = FDSettings(N=N_WIN // 2 + 1, df=1.0 / (N_WIN * DT), min_freq=f0 - delta, max_freq=f0 + delta,
                    force_backend=BACKEND)
    d = TDSignal(D, td).transform(fd, window=win)
    ac = AnalysisContainer(d, XYZ2SensitivityMatrix(fd, model=SENS))

    def tmpl(inc_val):
        p9 = np.array([A0, f0, fdot, 0.0, phi0, inc_val, psi, ra, dec]).reshape(9, 1)
        out = gb_gen(*p9, convert_to_ra_dec=False, return_spline=True)
        B = np.asarray(out.eval_tdi(grid)); B = (B[0] if B.ndim == 3 else B)[:NCH]
        return TDSignal(B, td).transform(fd, window=win)

    T_base = tmpl(inc)
    T_P = tmpl(np.pi / 2)                      # (1+cos^2 i)=1, cos i = 0
    T_0 = tmpl(0.0)                            # 2*T_Punit + 2*T_Cunit
    # unit-basis arrays
    TP = T_P
    TC_arr = (np.asarray(T_0.arr) - 2.0 * np.asarray(T_P.arr)) / 2.0
    import copy
    TC = copy.deepcopy(T_P); TC.arr[:] = TC_arr

    def ip(x, y):
        # complex inner product <x|y> via the container (template slot), no normalize
        return complex(ac.template_inner_product(y, normalize=False, complex=True,
                                                 data_arr_update=x.arr)) if False else None

    # Use direct whitened complex inner products via the sens matrix:
    # AnalysisContainer.template_inner_product(t, complex=True) = <d|t>.
    dh_P = complex(ac.template_inner_product(TP, normalize=False, complex=True))
    dh_C = complex(ac.template_inner_product(TC, normalize=False, complex=True))
    # Gram matrix: use a container whose 'data' is a template
    acP = AnalysisContainer(TP, XYZ2SensitivityMatrix(fd, model=SENS))
    acC = AnalysisContainer(TC, XYZ2SensitivityMatrix(fd, model=SENS))
    PP = complex(acP.template_inner_product(TP, normalize=False, complex=True))
    CC = complex(acC.template_inner_product(TC, normalize=False, complex=True))
    PC = complex(acP.template_inner_product(TC, normalize=False, complex=True))
    dd = float(ac.inner_product().real)

    G = np.array([[PP, PC], [np.conj(PC), CC]])
    v = np.array([dh_P, dh_C])
    coef = np.linalg.solve(G, v)
    a_fit, b_fit = coef
    # residual power: dd - 2 Re(coef^H v) + coef^H G coef
    res = dd - 2 * np.real(np.conj(coef) @ v) + np.real(np.conj(coef) @ G @ coef)
    mm_res = res / dd

    a_cat, b_cat = 1 + np.cos(inc) ** 2, 2 * np.cos(inc)
    print(f"\n  fitted a = {a_fit:.6f}   (cat {a_cat:.6f})", flush=True)
    print(f"  fitted b = {b_fit:.6f}   (cat {b_cat:.6f})", flush=True)
    print(f"  |b/a| fit = {abs(b_fit / a_fit):.6f}  vs cat {b_cat / a_cat:.6f}", flush=True)
    print(f"  arg(a) = {np.angle(a_fit):+.5f} rad   arg(b) = {np.angle(b_fit):+.5f} rad   "
          f"arg(b/a) = {np.angle(b_fit / a_fit):+.5f} rad (cat 0)", flush=True)
    print(f"  implied inc from |b/a|: {np.arccos(np.roots([abs(b_fit/a_fit), -2, abs(b_fit/a_fit)])[1] if False else 0):.4f}") if False else None
    r = abs(b_fit / a_fit)
    # solve r = 2c/(1+c^2) => c = (1 - sqrt(1-r^2))/r  (taking |c|<=1 root)
    c_imp = (1 - np.sqrt(max(0.0, 1 - r * r))) / r
    print(f"  implied cos(inc) = {c_imp:.5f}  (cat {np.cos(inc):.5f})  -> inc {np.arccos(c_imp):.5f} (cat {inc:.5f})", flush=True)
    print(f"  residual fractional power after 2-basis fit: {mm_res:.4e}  "
          f"(baseline mm was ~2.5e-5)", flush=True)

    # baseline mm for reference
    O = ac.template_inner_product(T_base, normalize=True, complex=True)
    print(f"  baseline 1-|O| = {1-abs(O):.4e}", flush=True)
    print("\nDONE.", flush=True)


if __name__ == "__main__":
    main()
