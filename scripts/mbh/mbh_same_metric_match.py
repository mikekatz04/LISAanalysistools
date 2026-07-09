"""MBH fidelity vs mojito data at the SAME metric as sobbh_same_metric_match.py.

Post-orbit-fix (LAT dev 7fc7c93: L1Orbits linear-interp grid capped at 3600 s,
was the file's native ~5.8 d) companion to scripts/gb/sobbh_same_metric_match.py:
single-source MBHB L1 stream, MBHTDIonFly (t_ref fix + coarse_graining_scale_factor
=48) template at the catalogue injection parameters, tukey 0.05, scirdv1,
mm = 1-|O| via AnalysisContainer.template_inner_product(normalize=True,
complex=True).  Purpose: learn the post-fix MBH floor and WHERE the residual
mismatch lives in frequency (response errors grow with f across the band;
waveform/PN/merger model differences concentrate near the merger frequencies).

Pre-fix recorded reference: mm ~ 6.1e-7..6.3e-7 in the FD >1 mHz band
(scripts/mbh/mbh_likelihood_compare.py, MBHTDIonFly B-template, scirdv1).
Post-fix GB/SOBBH floors for scale: GB wdwd 3.9e-11, VGB HM Cnc 1.8e-12,
SOBBH src0 2.4e-10 (90 d carrier bands).
"""
import os, sys, gc, time, threading, resource
os.environ.setdefault("OMP_NUM_THREADS", "8")
import numpy as np
MEM_CAP_GB = float(os.environ.get("MBH_MEM_CAP_GB", "8.2")); _IS_MAC = sys.platform == "darwin"
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
from lisatools.globalfit.recipe import mbh_catalogue_to_sampling_basis
from lisatools.globalfit.stock.erebor import make_mbh_transform_container
from lisatools.response.tdiconfig import TDIConfig
from lisatools.domains import TDSettings, FDSettings, TDSignal
from lisatools.analysiscontainer import AnalysisContainer
from lisatools.sensitivity import XYZ2SensitivityMatrix
from bbhx.mbhtdionfly import MBHTDIonFly
import phentax.waveform as pw

REF = 97729089.327664
PATH = "/Users/mkatz/.mojito_cache/brickmarket/mojito_light_v1_0_0/"
MBHB_L1 = os.path.join(PATH, "data", "MBHB", "L1")
MBHB_ID = int(os.environ.get("MBHB_ID", "0"))
BACKEND = "cpu"; DT = 10.0; TDI_GEN = "2nd generation"; NCH = 3; SENS = "scirdv1"
F_MIN, F_MAX = 1e-4, 2.5e-2
HMS = (21, 33, 44); TOL = 1e-12; DTMIN = 0.1; SCALE = 48.0
TUKEY_ALPHA = float(os.environ.get("MBH_TUKEY_ALPHA", "0.05"))
# Window carved AROUND the merger (MBHBs are loud transients): merger placed at
# MERGER_FRAC of the window so the pre-merger inspiral sits inside (same
# placement as mbh_mojito_match_debug.py / mbh_likelihood_compare.py).
WINDOW_DAYS = float(os.environ.get("MBH_WINDOW_DAYS", "48.0")); MERGER_FRAC = 0.72
N_WIN = int(round(WINDOW_DAYS * 86400 / DT)); TOBS = N_WIN * DT
PRE_FIX_MM_1MHZ = 6.1e-7          # recorded pre-orbit-fix >1mHz floor
DATA_CACHE = f"/tmp/mbh_same_metric_src{MBHB_ID}.npz"
TMPL_CACHE = f"/tmp/mbh_same_metric_tmpl_src{MBHB_ID}.npz"
MBH_TRANSFORM = make_mbh_transform_container()


def tukey(N, a):
    from scipy.signal.windows import tukey as _t
    return _t(N, a)


def load_cached():
    if os.path.exists(DATA_CACHE):
        z = np.load(DATA_CACHE, allow_pickle=True)
        if tuple(z["data_td"].shape) == (NCH, N_WIN):
            print(f"[cache] {DATA_CACHE}", flush=True)
            return (np.asarray(z["data_td"]), float(z["window_t0"]),
                    float(z["data_t0"]), z["cat"].item(), float(z["abs_merger"]))
        print(f"[cache] STALE shape {z['data_td'].shape} != ({NCH},{N_WIN}) -> re-read",
              flush=True)
    print("[cache] MISS -> reading MBHB L1 via L1ProcessingStep (one time)...", flush=True)
    loader = L1ProcessingStep(L1_folder=PATH, source_types=["mbhb"],
                              source_ids={"mbhb": MBHB_ID}, orbits_class=L1Orbits,
                              orbits_kwargs=dict(force_backend=BACKEND, frame="icrs"),
                              verbose=True)
    times = np.asarray(loader.times); data_full = np.asarray(loader.data)
    if data_full.shape[0] != 3: data_full = data_full.T
    dt_native = float(loader.dt); data_t0 = float(times[0])
    cat = {k: float(np.asarray(v)) for k, v in loader.catalogue["MBHB"][MBHB_ID].items()
           if np.asarray(v).dtype.kind in "fi"}
    t_plunge = cat["TimeCoalescencePhenomTPHMSSBFrame"]
    abs_merger = REF + t_plunge                # convention: REF + t_plunge
    deci = int(round(DT / dt_native)); n_full = data_full.shape[1]
    merger_idx_full = int(round((abs_merger - data_t0) / dt_native))
    start_full = merger_idx_full - int(round(MERGER_FRAC * N_WIN)) * deci
    start_full = max(0, min(start_full, n_full - N_WIN * deci))
    data_td = data_full[:, start_full: start_full + N_WIN * deci: deci][:, :N_WIN].copy()
    window_t0 = data_t0 + start_full * dt_native
    del data_full, times, loader; gc.collect()
    np.savez(DATA_CACHE, data_td=data_td, window_t0=window_t0, data_t0=data_t0,
             cat=cat, abs_merger=abs_merger)
    print(f"[cache] wrote {DATA_CACHE}  shape={data_td.shape}", flush=True)
    return data_td, window_t0, data_t0, cat, abs_merger


def build_template(window_t0, abs_merger, wf):
    """MBHTDIonFly template at injection params -- exact conventions from
    mbh_likelihood_compare.py (t_ref = -t_merge fix inside MBHTDIonFly,
    coarse_graining_scale_factor=48, dist in Mpc, t0=REF)."""
    if os.path.exists(TMPL_CACHE):
        z = np.load(TMPL_CACHE)
        if (z["B"].shape == (NCH, N_WIN) and abs(float(z["window_t0"]) - window_t0) < 1e-6):
            print(f"[cache] template from {TMPL_CACHE}", flush=True)
            return np.asarray(z["B"])
        print("[cache] template STALE -> rebuild", flush=True)
    m1, m2, s1z, s2z, dist, phi_ref, inc, psi, ra, dec, t_plunge = wf
    dur_s = ((abs_merger - window_t0) / 86400.0 + 6.0) * 86400.0
    print(f"  waveform_duration={dur_s/86400:.1f} d  scale_factor={SCALE}  dt_min={DTMIN}",
          flush=True)
    # Orbit: ltt sliced to the window (memory guard); positions get the
    # post-7fc7c93 linear-interp grid (3600 s cap) via configure's default.
    orb = L1Orbits(find_file(MBHB_L1, "MBHB", MBHB_ID), force_backend=BACKEND, frame="icrs")
    pad = 1.0e5
    lo = max(window_t0 - pad, float(orb.sc_t0)); hi = min(window_t0 + TOBS + pad,
                                                          float(orb._sc_t_base[-1]))
    lt = np.asarray(orb.ltt_t); mk = (lt >= lo) & (lt <= hi)
    orb.ltt = np.asarray(orb.ltt)[mk].copy(); orb.ltt_t = lt[mk].copy()
    orb.ltt_t0 = float(orb.ltt_t[0]); del lt; gc.collect()
    orb.configure(linear_interp_setup=True); mark("orbit ready (3600s pos grid)")
    wg = pw.IMRPhenomTHM(T=dur_s, higher_modes=list(HMS), include_negative_modes=True,
                         t_low_fit=True, coarse_grain=True, atol=TOL, rtol=TOL,
                         coarse_graining_scale_factor=SCALE)
    mbh = MBHTDIonFly(wg, orb, TDIConfig(TDI_GEN, force_backend=BACKEND), DT, dur_s,
                      t0=REF, dt_min=DTMIN, waveform_duration=dur_s, force_backend=BACKEND)
    grid_t = np.arange(N_WIN) * DT + window_t0
    B = np.asarray(mbh(m1, m2, s1z, s2z, dist, phi_ref, inc, ra, dec, psi, t_plunge,
                       upsample_t_arr=grid_t, combine=True))[:NCH]
    del mbh, wg, orb; gc.collect(); mark("template built")
    np.savez(TMPL_CACHE, B=B, window_t0=window_t0)
    return B


def main():
    threading.Thread(target=_wd, daemon=True).start()
    D, window_t0, data_t0, cat, abs_merger = load_cached(); D = D[:NCH]; mark("data loaded")
    samp = mbh_catalogue_to_sampling_basis(cat)
    wf = np.asarray(MBH_TRANSFORM.both_transforms(np.asarray(samp, float)), float)
    m1, m2, s1z, s2z, dist, phi_ref, inc, psi, ra, dec, t_plunge = wf
    print(f"  src{MBHB_ID}: m1={m1:.3e} m2={m2:.3e} dist={dist:.1f} Mpc  inc={inc:.4f}  "
          f"SNRest={cat.get('EstimatedSNR', float('nan')):.0f}", flush=True)
    print(f"  data_t0-REF={data_t0-REF:+.1f}s  merger @ {(abs_merger-data_t0)/86400:.2f} d "
          f"after data start  ({(abs_merger-window_t0)/86400:.2f} d into the "
          f"{WINDOW_DAYS:.0f}-d window, frac={MERGER_FRAC})", flush=True)

    B = build_template(window_t0, abs_merger, wf)

    win = tukey(N_WIN, TUKEY_ALPHA)
    td = TDSettings(N_WIN, DT, t0=window_t0, force_backend=BACKEND)
    bands = [("full[1e-4,2.5e-2]", F_MIN, F_MAX),
             ("low[1e-4,1e-3]", F_MIN, 1e-3),
             (">1mHz[1e-3,2.5e-2]", 1e-3, F_MAX),
             ("[2e-3,1e-2]", 2e-3, 1e-2),
             (">5mHz[5e-3,2.5e-2]", 5e-3, F_MAX)]
    print(f"\n  {'band':>20} {'1-Re(O)':>11} {'1-|O|':>11} {'SNRopt':>8} {'SNRdet':>8}",
          flush=True)
    mm = {}
    for tag, flo, fhi in bands:
        fd = FDSettings(N=N_WIN // 2 + 1, df=1.0 / (N_WIN * DT), min_freq=flo,
                        max_freq=fhi, force_backend=BACKEND)
        d = TDSignal(D, td).transform(fd, window=win)
        t = TDSignal(B, td).transform(fd, window=win)
        ac = AnalysisContainer(d, XYZ2SensitivityMatrix(fd, model=SENS))
        O = ac.template_inner_product(t, normalize=True, complex=True)
        opt, det = ac.template_snr(t)
        mm[tag] = 1 - abs(O)
        print(f"  {tag:>20} {1-O.real:>+11.3e} {1-abs(O):>11.3e} "
              f"{float(opt):>8.1f} {float(det):>+8.2f}", flush=True)

    # ---- learnings ------------------------------------------------------
    lo_mm = mm["low[1e-4,1e-3]"]; hi_mm = mm[">5mHz[5e-3,2.5e-2]"]
    mid_mm = mm["[2e-3,1e-2]"]; n1_mm = mm[">1mHz[1e-3,2.5e-2]"]
    print("\n" + "-" * 72, flush=True)
    print("LEARNINGS (MBHB id%d, %g d window, tukey %.2f, %s):" %
          (MBHB_ID, WINDOW_DAYS, TUKEY_ALPHA, SENS), flush=True)
    print(f"  * post-orbit-fix >1mHz mm = {n1_mm:.3e}  vs pre-fix recorded "
          f"{PRE_FIX_MM_1MHZ:.1e}  (x{PRE_FIX_MM_1MHZ/max(n1_mm,1e-30):.1f} better)", flush=True)
    print(f"  * frequency dependence: low[<1mHz]={lo_mm:.2e}  mid[2-10mHz]={mid_mm:.2e}  "
          f"high[>5mHz]={hi_mm:.2e}", flush=True)
    if hi_mm > 3 * lo_mm:
        print("  * mm CONCENTRATED at the merger/high-f band -> waveform-level "
              "(phentax IMRPhenomTHM merger/ringdown vs mojito MBH generator), "
              "NOT response (post-fix response floor is ~1e-12..1e-10).", flush=True)
    elif lo_mm > 3 * hi_mm:
        print("  * mm CONCENTRATED at low f -> early-inspiral (PN/start-freq/window "
              "truncation) territory, not merger physics.", flush=True)
    else:
        print("  * mm roughly flat in f -> no single-band culprit; could be a "
              "broadband convention (phase/time reference) residual.", flush=True)
    print(f"  * GB/SOBBH post-fix floors for scale: 3.9e-11 / 2.4e-10 -- MBH sits "
          f"{'well above (waveform-limited)' if n1_mm > 1e-8 else 'at'} the response floor.",
          flush=True)
    print("\nDONE.", flush=True)


if __name__ == "__main__":
    main()
