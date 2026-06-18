"""Log-like CHAIN vs the mojito data for the top high-frequency GB:

  (1) FULL lisatools: slow TD (GBTDIonTheFly) -> WDM -> AnalysisContainer
      template_likelihood  -- the dense ground-truth logL against the data.
  (2) chunked-het get_ll   -- GBWDMComputations.get_ll_wdm.
  (3) signal-het get_ll    -- GBSignalHetComputations.get_ll (REAL projection).

Evaluated at the injection AND at candidates perturbed in f0 so the dense logL
lands near 0, -5, -50, -500 -- so we compare the three get_ll's both at the peak
and as the candidate walks away from it. Reuses the cached mojito GB data and the
band/edge conventions from gb_getll_3check.
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
from lisatools.domains import TDSettings, TDSignal, WDMSettings, WDMSignal
from lisatools.datacontainer import DataResidualArray
from lisatools.analysiscontainer import AnalysisContainer
from lisatools.sensitivity import XYZ2SensitivityMatrix
from gbgpu.gbcomps import GBWDMComputations
from scipy.signal.windows import tukey as _tukey
sys.path.insert(0, os.path.join(os.path.dirname(os.path.abspath(__file__)), "..", "gb_chunked_het"))
from gbsignalhetcomputations import (  # noqa: E402
    GBSignalHetComputations, tukey_taper_layers, recommended_edge_cut)

REF = 97729089.327664
PATH = "/Users/mkatz/.mojito_cache/brickmarket/mojito_light_v1_0_0/"
GB_L1 = os.path.join(PATH, "data", "GB", "L1")
BACKEND = "cpu"; DT = 10.0; NCH = 3; SENS = "scirdv1"
NF = int(os.environ.get("GB_NF", "1460")); NT = int(os.environ.get("GB_NT", "512"))
N_WIN = NF * NT
RANK = int(os.environ.get("GB_RANK", "0"))
BAND_LAYERS = 15; TUK = 0.05; M_HALF = int(os.environ.get("GB_MHALF", "2"))
NT_SUB = 256; N_SPARSE = 256
DATA_CACHE = "/tmp/gb_mojito_data_365d.npz"
TARGETS = [0.0, -5.0, -50.0, -500.0]


def main():
    threading.Thread(target=_wd, daemon=True).start()
    z = np.load(DATA_CACHE)
    D = np.asarray(z["data_td"])[:NCH, :N_WIN]; data_t0 = float(z["data_t0"])
    A0, f0, fdot, phi0, inc, psi, ra, dec = np.asarray(z["top_params"])[RANK]
    p9 = np.array([A0, f0, fdot, 0.0, phi0, inc, psi, ra, dec])
    TOBS = N_WIN * DT
    taper = tukey_taper_layers(NT, TUK)
    EC_CH = recommended_edge_cut(NT, TUK, "chunked")
    EC_SH = recommended_edge_cut(NT, TUK, "sighet")
    layer_df = 1.0 / (2.0 * NF * DT); m_floor = int(f0 / layer_df)
    lof = (m_floor - BAND_LAYERS) * layer_df; hif = (m_floor + BAND_LAYERS + 1) * layer_df
    print(f"rank {RANK}: f0={f0*1e3:.4f} mHz  NT={NT} taper~{taper}  EC_chunk={EC_CH} EC_sighet={EC_SH} m_half={M_HALF}", flush=True)

    orb = L1Orbits(find_file(GB_L1, "GB", 0), force_backend=BACKEND, frame="icrs")
    pad = 1.0e5; lo = max(data_t0 - pad, float(orb.sc_t0)); hi = min(data_t0 + TOBS + pad, float(orb._sc_t_base[-1]))
    lt = np.asarray(orb.ltt_t); mk = (lt >= lo) & (lt <= hi)
    orb.ltt = np.asarray(orb.ltt)[mk].copy(); orb.ltt_t = lt[mk].copy(); orb.ltt_t0 = float(orb.ltt_t[0])
    del lt; gc.collect(); orb.configure(linear_interp_setup=True)
    tdi_config = TDIConfig("2nd generation", force_backend=BACKEND)
    grid = np.arange(N_WIN) * DT + data_t0
    t_nodes = np.linspace(grid[0], grid[-1], 16384)
    gb_dense = GBTDIonTheFly(t_nodes, TOBS, REF, 1.0 / DT, 1, tdi_config=tdi_config,
                             orbits=orb, tdi_chan="XYZ", force_backend=BACKEND)
    td_set = TDSettings(N_WIN, DT, t0=data_t0, force_backend=BACKEND)
    win = _tukey(N_WIN, TUK)

    # chunked uses EC_chunk (taper excluded); dense reference uses the same grid
    wdm_ch = WDMSettings(NF, NT, DT, t0=data_t0, min_freq=lof, max_freq=hif,
                         min_time=EC_CH * NF * DT, max_time=(NT - EC_CH) * NF * DT, force_backend=BACKEND)
    data_ch = TDSignal(D, td_set).transform(wdm_ch, window=win)
    sens_ch = XYZ2SensitivityMatrix(wdm_ch, model=SENS)
    an_ch = AnalysisContainer(DataResidualArray(data_ch), sens_ch)
    d_d = float(np.real(an_ch.inner_product()))
    invC = np.asarray(sens_ch.invC); invC = np.where(np.isfinite(invC), invC, 0.0)

    class _Holder:
        def __init__(s, data, iC):
            s.linear_data_arr = [np.ascontiguousarray(data).ravel()]
            s.linear_psd_arr = [np.ascontiguousarray(iC).ravel()]
        def __len__(s): return 1
    holder = _Holder(np.asarray(data_ch.arr), invC)
    comp = GBWDMComputations(wdm_ch, t_ref=REF, Nt_sub=NT_SUB, n_pad=NT_SUB // 8, N_sparse=N_SPARSE,
                             N_cp_sig=0, N_cp_orbit=0, orbits=orb, tdi_config="2nd generation",
                             force_backend=BACKEND, d_d=0.0, tdi_type="XYZ")
    # Same edge-cut as the dense/chunked so all three cover the IDENTICAL active
    # time region (the real-projection fix removed the sig-het's need for a small
    # edge-cut, so the per-method edge split is no longer required for accuracy).
    EC_SH_USE = EC_CH if int(os.environ.get("GB_SH_MATCH_EC", "1")) else EC_SH
    sighet = GBSignalHetComputations(D, p9, Nf=NF, Nt=NT, dt=DT, t0=data_t0, t_ref=REF,
                                     orbits=orb, tdi_config="2nd generation", min_freq=lof, max_freq=hif,
                                     tukey_alpha=TUK, edge_cut=EC_SH_USE, m_active_half_width=M_HALF,
                                     force_backend=BACKEND)

    def dense_ll(p):
        out = gb_dense(*p.reshape(9, 1), convert_to_ra_dec=False, return_spline=True)
        td = np.asarray(out.eval_tdi(grid)); td = (td[0] if td.ndim == 3 else td)[:NCH]
        return float(np.real(an_ch.template_likelihood(TDSignal(td, td_set).transform(wdm_ch, window=win))))

    def chunk_ll(p):
        return float(np.asarray(comp.get_ll_wdm(p.reshape(1, 9), holder, convert_to_ra_dec=False,
                                                use_layer_groups=True, group_band_layers=5, margin_layers=0))[0]) - 0.5 * d_d

    def sighet_ll(p):
        return float(np.asarray(sighet.get_ll(p))[0])

    # find Df0 (fraction of layer_df) that lands dense logL near each target
    def find_df0(target):
        if target == 0.0: return 0.0
        lo_f, hi_f = 0.0, 0.5
        for _ in range(18):
            mid = 0.5 * (lo_f + hi_f)
            ll = dense_ll(p9 + np.array([0, mid * layer_df, 0, 0, 0, 0, 0, 0, 0]))
            if ll < target: hi_f = mid
            else: lo_f = mid
        return 0.5 * (lo_f + hi_f)

    print(f"\n  {'target':>8} {'Df0/df':>9} | {'(1)dense':>13} {'(2)chunked':>13} {'(3)sighet':>13} | "
          f"{'(2)-(1)':>11} {'(3)-(1)':>11}", flush=True)
    for tgt in TARGETS:
        df0 = find_df0(tgt)
        p = p9 + np.array([0, df0 * layer_df, 0, 0, 0, 0, 0, 0, 0])
        l1 = dense_ll(p); l2 = chunk_ll(p); l3 = sighet_ll(p)
        print(f"  {tgt:>8.0f} {df0:>9.4f} | {l1:>13.5e} {l2:>13.5e} {l3:>13.5e} | "
              f"{l2-l1:>+11.3e} {l3-l1:>+11.3e}", flush=True)
    print("\nDONE.", flush=True)


if __name__ == "__main__":
    main()
