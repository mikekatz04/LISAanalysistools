"""Sweep MBH PhenomTHMTDIWaveform / response settings vs overlap with mojito.

Reuses the cached data window from mbh_mojito_match_debug.py
(/tmp/mbh_mojito_data_id{ID}.npz).  For each setting combo it builds the
orbit + wave_gen, generates the template, and reports the PHASE-MAXIMIZED
mismatch in three bands:
  full   = [1e-4, 2.5e-2]  (includes the < 5e-4 Hz mojito low-freq excess)
  >5e-4  = [5e-4, 2.5e-2]
  >1mHz  = [1e-3, 2.5e-2]  (merger-dominated; the cleanest signal band)

Knobs swept (one at a time around the baseline):
  POS_DT      orbit-position interp grid (s)        [coarser = less RAM]
  order       pyResponseTDI Lagrange order
  start_freq  waveform start frequency (Hz)
  coarse_grain is FORCED False (pyResponseTDI needs an equispaced grid; the
              PhenomTHMTDIWaveform constructor raises on True), so it is not
              swept -- it is reported for the record.
"""
import os, sys, gc, time, threading, resource
os.environ.setdefault("OMP_NUM_THREADS", "8")
import numpy as np

MEM_CAP_GB = float(os.environ.get("MBH_MEM_CAP_GB", "6.5"))
_IS_MAC = sys.platform == "darwin"
def rss_gb():
    r = resource.getrusage(resource.RUSAGE_SELF).ru_maxrss
    return r / 1e9 if _IS_MAC else r / 1e6
def _watchdog():
    while True:
        if rss_gb() > MEM_CAP_GB:
            sys.stderr.write(f"\n[WATCHDOG] RSS {rss_gb():.2f}GB > {MEM_CAP_GB}GB -> abort\n"); sys.stderr.flush()
            os._exit(42)
        time.sleep(0.3)

from lisatools.detector import L1Orbits
from lisatools.globalfit.preprocessing import find_file
from lisatools.globalfit.recipe_steps import mbh_catalogue_to_sampling_basis
from lisatools.globalfit.stock.erebor import make_mbh_transform_container
from lisatools.sources.bbh.waveform import PhenomTHMTDIWaveform
from lisatools.domains import TDSettings, FDSettings, TDSignal, place_td_signal_on_grid
from lisatools.analysiscontainer import AnalysisContainer
from lisatools.sensitivity import XYZ2SensitivityMatrix

MOJITO_REFERENCE_TIME = 97729089.327664
PATH = "/Users/mkatz/.mojito_cache/brickmarket/mojito_light_v1_0_0/"
MBHB_L1 = os.path.join(PATH, "data", "MBHB", "L1")
MBHB_ID = int(os.environ.get("MBHB_ID", "0"))
BACKEND = "cpu"; SENS_MODEL = "scirdv1"; DT = 10.0
TDI_GEN_STR = "2nd generation"; TDI_CHAN = "XYZ"; NCHANNELS = 3
F_MIN, F_MAX = 1e-4, 2.5e-2
MBH_HIGHER_MODES = (21, 33, 44); MBH_PHENOM_TOL = 1e-12
MBH_RESPONSE_ORDER = 30; MBH_BUFFER_TIME = 15_000.0
TUKEY_ALPHA = 0.05
MBH_TRANSFORM = make_mbh_transform_container()

DATA_CACHE = f"/tmp/mbh_mojito_data_id{MBHB_ID}.npz"
_ORBITS = {}   # keyed by pos_dt


def banner(s):
    print("\n" + "=" * 78 + f"\n {s}\n" + "=" * 78, flush=True)


def tukey(N, a):
    from scipy.signal.windows import tukey as _t
    return _t(N, a)


def load_cached():
    z = np.load(DATA_CACHE, allow_pickle=True)
    return (z["data_td"], float(z["window_t0"]), z["cat"].item())


def get_orbit(window_t0, TOBS, pos_dt):
    if pos_dt not in _ORBITS:
        orb = L1Orbits(find_file(MBHB_L1, "MBHB", MBHB_ID), force_backend=BACKEND, frame="icrs")
        pad = 1.0e5
        lo = max(window_t0 - pad, float(orb.sc_t0))
        hi = min(window_t0 + TOBS + pad, float(orb._sc_t_base[-1]))
        ltt_t = np.asarray(orb.ltt_t); m = (ltt_t >= lo) & (ltt_t <= hi)
        orb.ltt = np.asarray(orb.ltt)[m].copy(); orb.ltt_t = ltt_t[m].copy()
        orb.ltt_t0 = float(orb.ltt_t[0]); del ltt_t; gc.collect()
        orb.configure(linear_interp_setup=True, dt=pos_dt)
        _ORBITS[pos_dt] = orb
        print(f"  [orbit] built pos_dt={pos_dt}s  RSS={rss_gb():.2f}GB", flush=True)
    return _ORBITS[pos_dt]


def make_template(samp, window_t0, N_WIN, TOBS, *, pos_dt, order, start_freq,
                  dur_days, coarse_grain=False):
    data_td_settings = TDSettings(N=N_WIN, dt=DT, t0=window_t0, force_backend=BACKEND)
    gen = PhenomTHMTDIWaveform(
        waveform_kwargs=dict(higher_modes=list(MBH_HIGHER_MODES), include_negative_modes=True,
                             t_low_fit=True, coarse_grain=coarse_grain,
                             atol=MBH_PHENOM_TOL, rtol=MBH_PHENOM_TOL),
        Tobs=dur_days * 86400.0, start_freq=start_freq, use_reference_time=True,
        waveform_t0=MOJITO_REFERENCE_TIME, data_td_settings=data_td_settings,
        tdi_generation=TDI_GEN_STR, tdi_channels=TDI_CHAN, sampling_frequency=1.0 / DT,
        orbits=get_orbit(window_t0, TOBS, pos_dt), order=order, tukey_alpha=TUKEY_ALPHA,
        stft_dt=None, freq_min=F_MIN, freq_max=F_MAX, fft_batch_size=2,
        buffer_time=MBH_BUFFER_TIME, output_domain_settings=None, force_backend=BACKEND,
    )
    params_in = MBH_TRANSFORM.both_transforms(np.asarray(samp, dtype=float))
    times, channels = gen.compute_tdi_channels(*params_in)
    grid = TDSettings(N=N_WIN, dt=DT, t0=window_t0, force_backend=BACKEND)
    arr = np.asarray(place_td_signal_on_grid(np.atleast_2d(channels)[:NCHANNELS], grid, times=times).arr)
    del gen; gc.collect()
    return arr


def main():
    threading.Thread(target=_watchdog, daemon=True).start()
    banner(f"MBH settings sweep vs mojito id={MBHB_ID}")
    data_td, window_t0, cat = load_cached()
    N_WIN = data_td.shape[1]; TOBS = N_WIN * DT
    samp = mbh_catalogue_to_sampling_basis(cat)
    win = tukey(N_WIN, TUKEY_ALPHA)
    td_set = TDSettings(N_WIN, DT, t0=window_t0, force_backend=BACKEND)

    # per-band data analysis containers (data-side fixed across settings)
    bands = {"full": (F_MIN, F_MAX), ">5e-4": (5e-4, F_MAX), ">1mHz": (1e-3, F_MAX)}
    acs = {}
    for tag, (lo, hi) in bands.items():
        fd = FDSettings(N=N_WIN // 2 + 1, df=1.0 / (N_WIN * DT), min_freq=lo, max_freq=hi,
                        force_backend=BACKEND)
        d = TDSignal(data_td, td_set).transform(fd, window=win)
        acs[tag] = (AnalysisContainer(d, XYZ2SensitivityMatrix(fd, model=SENS_MODEL)), fd)

    def mm_row(label, tmpl):
        out = {}
        for tag, (ac, fd) in acs.items():
            t = TDSignal(tmpl, td_set).transform(fd, window=win)
            O = ac.template_inner_product(t, normalize=True, complex=True)
            out[tag] = 1 - abs(O)
        print(f"  {label:34s} | full {out['full']:.4e} | >5e-4 {out['>5e-4']:.4e} "
              f"| >1mHz {out['>1mHz']:.4e} | RSS {rss_gb():.2f}GB", flush=True)

    # baseline + one-at-a-time variations (dur covers the pre-merger window)
    DUR = 40.0
    configs = [
        ("baseline pos300 ord30 sf7e-5", dict(pos_dt=300.0, order=30, start_freq=7e-5)),
        ("pos_dt=50  (finer orbit)",      dict(pos_dt=50.0,  order=30, start_freq=7e-5)),
        ("order=40   (higher Lagrange)",  dict(pos_dt=300.0, order=40, start_freq=7e-5)),
        ("order=20   (lower Lagrange)",   dict(pos_dt=300.0, order=20, start_freq=7e-5)),
        ("start_freq=2e-5 (lower)",       dict(pos_dt=300.0, order=30, start_freq=2e-5)),
    ]
    banner("mismatch (phase-maximized) per setting")
    for label, kw in configs:
        try:
            tmpl = make_template(samp, window_t0, N_WIN, TOBS, dur_days=DUR, **kw)
            mm_row(label, tmpl)
            del tmpl; gc.collect()
        except Exception as e:
            print(f"  {label:34s} | FAILED: {e!r}", flush=True)
        # free orbits except the baseline grid to bound RAM
        for k in list(_ORBITS):
            if k != 300.0:
                del _ORBITS[k]; gc.collect()
    print("\nDONE.", flush=True)


if __name__ == "__main__":
    main()
