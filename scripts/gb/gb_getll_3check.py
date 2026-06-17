"""GB likelihood 3-check on the top-N mojito high-frequency sources, band-passed
in isolation. For each source, three numbers that must agree (no ll_scale; every
likelihood comes from an EXISTING get_ll / template_likelihood call):

  (1) lisatools dense  : GBTDIonTheFly TD -> WDM -> analysis.template_likelihood
  (2) chunked TEMPLATE : GBWDMComputations.fill_global_wdm WDM template
                          -> analysis.template_likelihood / template_inner_product
  (3) chunked GET_LL    : GBWDMComputations.get_ll_wdm(params, holder) - 0.5*d_d
                          (the exact call from compare_signalhet_vs_chunked_mcmc.py)

Stage 2 (separate) adds the signal-het get_ll. Uses the cached mojito GB data +
top-N params from gb_mojito_match.py (/tmp/gb_mojito_data_365d.npz).
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

import gbgpu  # noqa: F401 -- registers gbgpu_<flavor> backend
from lisatools.detector import L1Orbits
from lisatools.globalfit.preprocessing import find_file
from lisatools.response.tdiconfig import TDIConfig
from lisatools.response.tdionfly import GBTDIonTheFly
from lisatools.domains import TDSettings, TDSignal, WDMSettings, WDMSignal
from lisatools.datacontainer import DataResidualArray
from lisatools.analysiscontainer import AnalysisContainer
from lisatools.sensitivity import XYZ2SensitivityMatrix
from gbgpu.gbcomps import GBWDMComputations
sys.path.insert(0, os.path.join(os.path.dirname(os.path.abspath(__file__)), "..", "gb_chunked_het"))
from gbsignalhetcomputations import GBSignalHetComputations  # noqa: E402

REF = 97729089.327664
PATH = "/Users/mkatz/.mojito_cache/brickmarket/mojito_light_v1_0_0/"
GB_L1 = os.path.join(PATH, "data", "GB", "L1")
BACKEND = "cpu"; DT = 10.0; NCH = 3; SENS = "scirdv1"
N_DAYS = 365.0  # cache window; we use the first NF*NT samples below
NF = int(os.environ.get("GB_NF", "1460"))
NT = int(os.environ.get("GB_NT", "512"))  # Nt divisible by sig-het Nt_layer (64); env-tunable
# Larger NT -> finer WDM time resolution -> smaller sig-het heterodyne floor.
# Capped by the cached data length (365 d @ dt=10 = 3.15M samples) so NF*NT <= ~3.15M.
N_WIN = NF * NT              # layer_df = 1/(2*NF*dt) (Nf-only) unchanged -> band isolation preserved
TOPN = int(os.environ.get("GB_TOPN", "3"))
BAND_LAYERS = int(os.environ.get("GB_BAND_LAYERS", "15"))  # +/- layers around f0
TUKEY_ALPHA = float(os.environ.get("GB_TUKEY_ALPHA", "0.05"))  # 0.01-0.05, GB chunked/sig-het regime
# Edge-cut SCALES WITH DURATION. The global Tukey taper covers alpha*Nt/2 WDM
# time layers at each end. The chunked-het only mirrors the GLOBAL taper outside
# its per-chunk stitching window, so the active region must exclude the taper:
# edge_cut >= alpha*Nt/2. A fixed edge-cut (e.g. 20) is fine at small Nt (taper
# hidden) but lets the taper leak into the active region at large Nt -> chunked
# mm jumps from ~1e-9 to ~1e-3. Override with GB_EDGE to force a fixed value.
import math as _math
_edge_taper = int(_math.ceil(0.5 * TUKEY_ALPHA * NT)) + 8  # + margin for WDM window spread
EDGE = int(os.environ["GB_EDGE"]) if "GB_EDGE" in os.environ else max(20, _edge_taper)
NT_SUB = int(os.environ.get("NT_SUB", "256")); N_SPARSE = int(os.environ.get("N_SPARSE", "256"))
SKIP_SIGHET = os.environ.get("SKIP_SIGHET", "0") == "1"  # skip check (4) for fast chunked-param sweeps
DATA_CACHE = f"/tmp/gb_mojito_data_{int(N_DAYS)}d.npz"


def banner(s): print("\n" + "=" * 92 + f"\n {s}\n" + "=" * 92, flush=True)
def tukey(N, a):
    from scipy.signal.windows import tukey as _t
    return _t(N, a)


class _FullGridWDMHolder:
    """duck-type for GBWDMComputations.get_ll_wdm (from gb_chunked scripts)."""
    def __init__(self, data, invC):
        self.linear_data_arr = [np.ascontiguousarray(data).ravel()]
        self.linear_psd_arr = [np.ascontiguousarray(invC).ravel()]
    def __len__(self): return 1


class GBChunkedWaveWrap:
    """params(9,) -> chunked-het WDMSignal on the active band (good_gb_chunked_test_script)."""
    def __init__(self, comp, wdm_set, Nf, Nt):
        self.comp, self.wdm_set, self.Nf, self.Nt = comp, wdm_set, Nf, Nt
    def __call__(self, *params):
        full = np.zeros((3, self.Nf, self.Nt), dtype=float)
        self.comp.fill_global_wdm(np.asarray(params, float).reshape(1, 9), full,
                                  convert_to_ra_dec=False, factors=None)
        tpl = full[:, self.wdm_set.ind_min_f: self.wdm_set.ind_max_f + 1, :]
        if self.wdm_set.Nt_active != self.wdm_set.Nt:
            tpl = tpl[:, :, self.wdm_set.active_slice_t]
        return WDMSignal(tpl, self.wdm_set)


def main():
    threading.Thread(target=_wd, daemon=True).start()
    z = np.load(DATA_CACHE)
    D = np.asarray(z["data_td"])[:NCH, :N_WIN]; data_t0 = float(z["data_t0"])
    top_params = np.asarray(z["top_params"]); top_freqs = np.asarray(z["top_freqs"])
    banner(f"GB likelihood 3-check, top {TOPN} mojito hi-f, +/-{BAND_LAYERS} layers")
    TOBS = N_WIN * DT
    orb = L1Orbits(find_file(GB_L1, "GB", 0), force_backend=BACKEND, frame="icrs")
    pad = 1.0e5; lo = max(data_t0 - pad, float(orb.sc_t0)); hi = min(data_t0 + TOBS + pad, float(orb._sc_t_base[-1]))
    lt = np.asarray(orb.ltt_t); mk = (lt >= lo) & (lt <= hi)
    orb.ltt = np.asarray(orb.ltt)[mk].copy(); orb.ltt_t = lt[mk].copy(); orb.ltt_t0 = float(orb.ltt_t[0])
    del lt; gc.collect(); orb.configure(linear_interp_setup=True)
    tdi_config = TDIConfig("2nd generation", force_backend=BACKEND)

    grid = np.arange(N_WIN) * DT + data_t0
    t_nodes = np.linspace(grid[0], grid[-1], 16384)
    gb_dense = GBTDIonTheFly(t_nodes, TOBS, REF, 1.0 / DT, 1,
                             tdi_config=tdi_config, orbits=orb, tdi_chan="XYZ", force_backend=BACKEND)
    td_set = TDSettings(N_WIN, DT, t0=data_t0, force_backend=BACKEND)
    wd0 = WDMSettings(NF, NT, DT, t0=data_t0, min_freq=1e-4, max_freq=35e-3, force_backend=BACKEND)
    layer_df = wd0.layer_df
    print(f"  layer_df={layer_df*1e6:.3f} uHz  N_WIN={N_WIN}=NF*NT  "
          f"NT={NT} tukey={TUKEY_ALPHA} EDGE={EDGE} (taper~{0.5*TUKEY_ALPHA*NT:.0f} layers)", flush=True)

    print(f"\n  {'rk':>2} {'f0(mHz)':>9} {'SNR':>7} | {'(1)dense':>12} {'(2)chk_tmpl':>12} {'(3)chk_getll':>12} {'(4)sighet':>12} "
          f"| {'(3)-(1)':>10} {'(4)-(1)':>10} {'mm_chunk':>10}", flush=True)
    for rank in range(TOPN):
        A0, f0, fdot, phi0, inc, psi, ra, dec = top_params[rank]
        p9 = np.array([A0, f0, fdot, 0.0, phi0, inc, psi, ra, dec])
        m_floor = int(f0 / layer_df)
        lof = (m_floor - BAND_LAYERS) * layer_df; hif = (m_floor + BAND_LAYERS + 1) * layer_df
        wdm_set = WDMSettings(NF, NT, DT, t0=data_t0, min_freq=lof, max_freq=hif,
                              min_time=EDGE * NF * DT, max_time=(NT - EDGE) * NF * DT, force_backend=BACKEND)

        # band-passed mojito WDM data + analysis (same Tukey window as sig-het)
        win = tukey(N_WIN, TUKEY_ALPHA)
        data_wdm = TDSignal(D, td_set).transform(wdm_set, window=win)
        sens = XYZ2SensitivityMatrix(wdm_set, model=SENS)
        analysis = AnalysisContainer(DataResidualArray(data_wdm), sens)
        d_d = float(np.real(analysis.inner_product()))
        snr = float(analysis.snr()) if hasattr(analysis, "snr") else np.sqrt(abs(d_d))

        # (1) lisatools dense TD -> WDM
        out = gb_dense(*p9.reshape(9, 1), convert_to_ra_dec=False, return_spline=True)
        td_dense = np.asarray(out.eval_tdi(grid)); td_dense = (td_dense[0] if td_dense.ndim == 3 else td_dense)[:NCH]
        dense_wdm = TDSignal(td_dense, td_set).transform(wdm_set, window=win)
        ll1 = float(np.real(analysis.template_likelihood(dense_wdm)))

        # chunked-het computations (orbits MUST match dense)
        comp = GBWDMComputations(wdm_set, t_ref=REF, Nt_sub=NT_SUB, n_pad=NT_SUB // 8, N_sparse=N_SPARSE,
                                 N_cp_sig=0, N_cp_orbit=0, orbits=orb, tdi_config="2nd generation",
                                 force_backend=BACKEND, d_d=0.0, tdi_type="XYZ")
        # (2) chunked TEMPLATE -> lisatools template_likelihood
        wrap = GBChunkedWaveWrap(comp, wdm_set, NF, NT)
        chunk_wdm = wrap(*p9)
        ll2 = float(np.real(analysis.template_likelihood(chunk_wdm)))
        mm_chunk = float(1 - abs(analysis.template_inner_product(chunk_wdm, normalize=True)))
        if rank == 0:
            print(f"  [chunk geom] Nt_sub={comp.Nt_sub} N_sparse={comp.N_sparse} "
                  f"n_pad={comp.n_pad} n_chunks={comp.n_chunks} T_chunk={comp.T_chunk:.3e}s", flush=True)

        # (3) chunked DIRECT get_ll (exact existing call) - 0.5*d_d
        invC = np.asarray(sens.invC); invC = np.where(np.isfinite(invC), invC, 0.0)
        holder = _FullGridWDMHolder(np.asarray(data_wdm.arr), invC)
        ll3 = float(np.asarray(comp.get_ll_wdm(p9.reshape(1, 9), holder, convert_to_ra_dec=False,
                                               use_layer_groups=True, group_band_layers=5, margin_layers=0))[0]) - 0.5 * d_d

        # (4) signal-het DIRECT get_ll (GBSignalHetComputations.__init__ builds the
        #     heterodyne coeffs under the hood; same Tukey on data + FD sparse window)
        if SKIP_SIGHET:
            ll4 = float("nan"); sighet = None
        else:
            sighet = GBSignalHetComputations(D, p9, Nf=NF, Nt=NT, dt=DT, t0=data_t0, t_ref=REF,
                                             orbits=orb, tdi_config="2nd generation",
                                             min_freq=lof, max_freq=hif, tukey_alpha=TUKEY_ALPHA,
                                             edge_cut=EDGE, force_backend=BACKEND)
            ll4 = float(np.asarray(sighet.get_ll(p9))[0])

        print(f"  {rank:>2} {f0*1e3:>9.4f} {snr:>7.1f} | {ll1:>12.4e} {ll2:>12.4e} {ll3:>12.4e} {ll4:>12.4e} "
              f"| {ll3-ll1:>+10.3e} {ll4-ll1:>+10.3e} {mm_chunk:>10.3e}", flush=True)
        del comp, sighet; gc.collect()
    print("\nDONE.", flush=True)


if __name__ == "__main__":
    main()
