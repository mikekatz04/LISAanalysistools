"""Increase the TDI-on-the-fly sparse spline grid (via coarse_graining_scale_factor,
which sets the eval/spline grid the TD path uses) and measure whether the
legacy<->on-the-fly mismatch + the low-freq floor shrink.  Full data window,
injection params.  A=pyResponse (built once), B=on-the-fly at each scale factor;
also B-vs-data.
"""
import os, sys, gc, time, threading, resource
os.environ.setdefault("OMP_NUM_THREADS", "8")
import numpy as np
MEM_CAP_GB = float(os.environ.get("MBH_MEM_CAP_GB", "7.8")); _IS_MAC = sys.platform == "darwin"
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
from lisatools.globalfit.recipe_steps import mbh_catalogue_to_sampling_basis
from lisatools.globalfit.stock.erebor import make_mbh_transform_container
from lisatools.response.tdiconfig import TDIConfig
from lisatools.sources.bbh.waveform import PhenomTHMTDIWaveform
from lisatools.domains import TDSettings, FDSettings, TDSignal, place_td_signal_on_grid
from lisatools.analysiscontainer import AnalysisContainer
from lisatools.sensitivity import XYZ2SensitivityMatrix
from bbhx.mbhtdionfly import MBHTDIonFly
import phentax.waveform as pw

REF = 97729089.327664
PATH = "/Users/mkatz/.mojito_cache/brickmarket/mojito_light_v1_0_0/"
MBHB_L1 = os.path.join(PATH, "data", "MBHB", "L1")
MBHB_ID = 0; BACKEND = "cpu"; SENS_MODEL = "scirdv1"; DT = 10.0
TDI_GEN_STR = "2nd generation"; TDI_CHAN = "XYZ"; NCH = 3
F_MIN, F_MAX = 1e-4, 2.5e-2
HMS = (21, 33, 44); TOL = 1e-12; ORDER = 30; BUFFER = 15_000.0; START_FREQ = 7e-5
TUKEY_ALPHA = 0.05; POS_DT = 300.0; DTMIN = 0.1
SCALES = [float(x) for x in os.environ.get("MBH_SCALES", "12,48,120").split(",")]
MBH_TRANSFORM = make_mbh_transform_container()
DATA_CACHE = f"/tmp/mbh_mojito_data_id{MBHB_ID}.npz"


def banner(s): print("\n" + "=" * 78 + f"\n {s}\n" + "=" * 78, flush=True)
def tukey(N, a):
    from scipy.signal.windows import tukey as _t
    return _t(N, a)


def main():
    threading.Thread(target=_wd, daemon=True).start()
    banner("increase TDI-on-the-fly sparse grid (coarse_graining_scale_factor)")
    z = np.load(DATA_CACHE, allow_pickle=True)
    D = z["data_td"]; window_t0 = float(z["window_t0"]); cat = z["cat"].item(); abs_merger = float(z["abs_merger"])
    N_WIN = D.shape[1]; TOBS = N_WIN * DT
    wf = np.asarray(MBH_TRANSFORM.both_transforms(np.asarray(mbh_catalogue_to_sampling_basis(cat), float)), float)
    m1, m2, s1z, s2z, dist, phi_ref, inc, psi, ra, dec, t_plunge = wf
    dur_s = ((abs_merger - window_t0) / 86400.0 + 6.0) * 86400.0
    print(f"  N_WIN={N_WIN} inc={inc:.4f} dur={dur_s/86400:.1f}d  scales={SCALES}", flush=True)

    orb = L1Orbits(find_file(MBHB_L1, "MBHB", MBHB_ID), force_backend=BACKEND, frame="icrs")
    pad = 1.0e5; lo = max(window_t0 - pad, float(orb.sc_t0)); hi = min(window_t0 + TOBS + pad, float(orb._sc_t_base[-1]))
    lt = np.asarray(orb.ltt_t); mk = (lt >= lo) & (lt <= hi)
    orb.ltt = np.asarray(orb.ltt)[mk].copy(); orb.ltt_t = lt[mk].copy(); orb.ltt_t0 = float(orb.ltt_t[0])
    del lt; gc.collect(); orb.configure(linear_interp_setup=True, dt=POS_DT); mark("orbit ready")

    grid = TDSettings(N=N_WIN, dt=DT, t0=window_t0, force_backend=BACKEND)
    gA = PhenomTHMTDIWaveform(
        waveform_kwargs=dict(higher_modes=list(HMS), include_negative_modes=True,
                             t_low_fit=True, coarse_grain=False, atol=TOL, rtol=TOL),
        Tobs=dur_s, start_freq=START_FREQ, use_reference_time=True, waveform_t0=REF,
        data_td_settings=grid, tdi_generation=TDI_GEN_STR, tdi_channels=TDI_CHAN,
        sampling_frequency=1.0 / DT, orbits=orb, order=ORDER, tukey_alpha=TUKEY_ALPHA,
        stft_dt=None, freq_min=F_MIN, freq_max=F_MAX, fft_batch_size=2, buffer_time=BUFFER,
        output_domain_settings=None, force_backend=BACKEND)
    tA, chA = gA.compute_tdi_channels(*wf)
    A = np.asarray(place_td_signal_on_grid(np.atleast_2d(chA)[:NCH], grid, times=tA).arr); del gA; gc.collect(); mark("A built")

    win = tukey(N_WIN, TUKEY_ALPHA); td = TDSettings(N_WIN, DT, t0=window_t0, force_backend=BACKEND)
    def mm(ref, tmpl, lo, hi):
        fd = FDSettings(N=N_WIN // 2 + 1, df=1.0 / (N_WIN * DT), min_freq=lo, max_freq=hi, force_backend=BACKEND)
        a = AnalysisContainer(TDSignal(ref, td).transform(fd, window=win), XYZ2SensitivityMatrix(fd, model=SENS_MODEL))
        return 1 - abs(a.template_inner_product(TDSignal(tmpl, td).transform(fd, window=win), normalize=True, complex=True))

    grid_t = np.arange(N_WIN) * DT + window_t0
    bands = [("full", F_MIN, F_MAX), (">5e-4", 5e-4, F_MAX), (">1mHz", 1e-3, F_MAX), (">2mHz", 2e-3, F_MAX)]
    for sf in SCALES:
        wave_gen = pw.IMRPhenomTHM(T=dur_s, higher_modes=list(HMS), include_negative_modes=True,
                                   t_low_fit=True, coarse_grain=True, atol=TOL, rtol=TOL,
                                   coarse_graining_scale_factor=sf)
        mbh = MBHTDIonFly(wave_gen, orb, TDIConfig(TDI_GEN_STR, force_backend=BACKEND), DT, dur_s,
                          t0=REF, dt_min=DTMIN, waveform_duration=dur_s, force_backend=BACKEND)
        B = np.asarray(mbh(m1, m2, s1z, s2z, dist, phi_ref, inc, ra, dec, psi, t_plunge,
                           upsample_t_arr=grid_t, combine=True))[:NCH]
        banner(f"scale_factor={sf}   (sparse-grid density)")
        for tag, l, h in bands:
            print(f"  {tag:7s}  A-vs-B={mm(A, B, l, h):.4e}   B-vs-data={mm(D, B, l, h):.4e}", flush=True)
        mark(f"scale_factor={sf} done")
        del wave_gen, mbh, B; gc.collect()
    print("\n(reference: A=pyResp vs data >1mHz=3.4e-5, >2mHz=2.5e-6)", flush=True)
    print("DONE.", flush=True)


if __name__ == "__main__":
    main()
