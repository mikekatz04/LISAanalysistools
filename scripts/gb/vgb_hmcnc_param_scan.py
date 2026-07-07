"""HM Cnc (face-on VGB) mismatch floor: 1-D parameter scans.

HM Cnc shows a constant-in-T mm ~ 2.5e-5 vs the noiseless VGB stream while
the near-edge-on ZTFs sit at ~1e-6 (T-growing). Face-on = h-cross dominated,
so scan each parameter around the catalogue value and report the offset that
minimizes mm -- a nonzero best-fit offset in (inc, psi) or (ra, dec) points
at a specific convention/factor difference vs the mojito generator.

Uses the 30-d cache from vgb_mojito_match.py (floor is duration-independent).
"""
import os, sys, gc, threading, time, resource
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
RANK = int(os.environ.get("VGB_RANK", "0"))  # 0 = HM Cnc
DATA_CACHE = f"/tmp/vgb_mojito_data_{int(N_DAYS)}d.npz"


def tukey(N, a):
    from scipy.signal.windows import tukey as _t
    return _t(N, a)


def main():
    threading.Thread(target=_wd, daemon=True).start()
    z = np.load(DATA_CACHE)
    D = np.asarray(z["data_td"])[:NCH]; data_t0 = float(z["data_t0"])
    p8 = np.asarray(z["top_params"])[RANK]
    A0, f0, fdot, phi0, inc, psi, ra, dec = p8
    print(f"rank {RANK}: f0={f0*1e3:.5f} mHz  inc={inc:.4f} (cos={np.cos(inc):.3f})  "
          f"psi={psi:.4f}  ra={ra:.4f} dec={dec:.4f}  ({N_DAYS:.0f} d)", flush=True)

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

    def mm(params8):
        a0, f_, fd_, ph, ic, ps, r_, de = params8
        p9 = np.array([a0, f_, fd_, 0.0, ph, ic, ps, r_, de]).reshape(9, 1)
        out = gb_gen(*p9, convert_to_ra_dec=False, return_spline=True)
        B = np.asarray(out.eval_tdi(grid)); B = (B[0] if B.ndim == 3 else B)[:NCH]
        t = TDSignal(B, td).transform(fd, window=win)
        return 1 - abs(ac.template_inner_product(t, normalize=True, complex=True))

    base = mm(p8)
    print(f"baseline mm = {base:.4e}\n", flush=True)

    # (name, index, offsets) -- additive unless noted
    scans = [
        ("inc",  4, np.linspace(-0.03, 0.03, 13)),
        ("psi",  5, np.linspace(-0.03, 0.03, 13)),
        ("ra",   6, np.linspace(-0.006, 0.006, 13)),
        ("dec",  7, np.linspace(-0.006, 0.006, 13)),
        ("f0",   1, f0 * np.linspace(-2e-9, 2e-9, 13)),
        ("fdot", 2, fdot * np.linspace(-0.5, 0.5, 13)),
        ("amp",  0, A0 * np.linspace(-0.01, 0.01, 13)),
        ("phi0", 3, np.linspace(-0.06, 0.06, 13)),
    ]
    print(f"{'param':>6} {'best_offset':>13} {'mm_best':>11} {'gain':>7}", flush=True)
    for name, idx, offs in scans:
        vals = []
        for o in offs:
            q = p8.copy(); q[idx] = q[idx] + o
            vals.append(mm(q))
        j = int(np.argmin(vals))
        print(f"{name:>6} {offs[j]:>13.4e} {vals[j]:>11.4e} {base/max(vals[j],1e-16):>7.2f}", flush=True)
    print("\nDONE.", flush=True)


if __name__ == "__main__":
    main()
