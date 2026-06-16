"""Compare the LISA response path for the MBH template vs mojito: pyResponseTDI
(Lagrange-interpolated strain) vs TDI-on-the-fly (analytic delays on splined
mode amp/phase).  SAME waveform model (phentax IMRPhenomTHM, coarse_grain=False),
SAME params / orbit / data window -- only the response function differs.

  A = PhenomTHMTDIWaveform  -> compute_tdi_channels (pyResponseTDI, order-N
      Lagrange interpolation of the oscillatory h(t) to apply TDI delays).
  B = MBHTDIonFly           -> TDTDIonTheFly (delays applied analytically to
      the cubic-splined, MONOTONIC mode phase Phi(t); no interpolation of the
      oscillatory signal).

Reports phase-maximized mismatch of each template vs the mojito data, and A-vs-B
directly, in three bands.  If B's mismatch is lower, the pyResponseTDI
interpolation was the residual error source.

Reuses the cached data window from mbh_mojito_match_debug.py.
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
            sys.stderr.write(f"\n[WATCHDOG] RSS {rss_gb():.2f}GB -> abort\n"); sys.stderr.flush(); os._exit(42)
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

MOJITO_REFERENCE_TIME = 97729089.327664
PATH = "/Users/mkatz/.mojito_cache/brickmarket/mojito_light_v1_0_0/"
MBHB_L1 = os.path.join(PATH, "data", "MBHB", "L1")
MBHB_ID = int(os.environ.get("MBHB_ID", "0"))
BACKEND = "cpu"; SENS_MODEL = "scirdv1"; DT = 10.0
TDI_GEN_STR = "2nd generation"; TDI_CHAN = "XYZ"; NCHANNELS = 3
F_MIN, F_MAX = 1e-4, 2.5e-2
HMS = (21, 33, 44); TOL = 1e-12
ORDER = 30; BUFFER_TIME = 15_000.0; START_FREQ = 7e-5
TUKEY_ALPHA = 0.05; POS_DT = float(os.environ.get("MBH_POS_DT", "300.0"))
# TDI-on-the-fly mode grid.  Native mode is an ADAPTIVE (coarse-grained) grid
# whose densest step is dt_min (0.1 s near merger, sparse during inspiral) --
# this is what the validated MBHTDIonFly test uses.  A coarse_grain=False
# uniform grid at dt_min badly under-resolves the merger -> garbage template.
CG_B = os.environ.get("MBH_CG", "1") == "1"          # coarse_grain for path B
DTMIN = float(os.environ.get("MBH_DTMIN", "0.1"))    # densest adaptive step (s)
MBH_TRANSFORM = make_mbh_transform_container()
DATA_CACHE = f"/tmp/mbh_mojito_data_id{MBHB_ID}.npz"


def tukey(N, a):
    from scipy.signal.windows import tukey as _t
    return _t(N, a)


def banner(s): print("\n" + "=" * 78 + f"\n {s}\n" + "=" * 78, flush=True)


def build_orbit(window_t0, TOBS):
    orb = L1Orbits(find_file(MBHB_L1, "MBHB", MBHB_ID), force_backend=BACKEND, frame="icrs")
    pad = 1.0e5
    lo = max(window_t0 - pad, float(orb.sc_t0)); hi = min(window_t0 + TOBS + pad, float(orb._sc_t_base[-1]))
    ltt_t = np.asarray(orb.ltt_t); m = (ltt_t >= lo) & (ltt_t <= hi)
    orb.ltt = np.asarray(orb.ltt)[m].copy(); orb.ltt_t = ltt_t[m].copy(); orb.ltt_t0 = float(orb.ltt_t[0])
    del ltt_t; gc.collect()
    orb.configure(linear_interp_setup=True, dt=POS_DT)
    mark(f"orbit ready (pos_dt={POS_DT}s)")
    return orb


def make_imrphenomthm(dur_s, coarse_grain):
    return pw.IMRPhenomTHM(T=dur_s, higher_modes=list(HMS), include_negative_modes=True,
                           t_low_fit=True, coarse_grain=coarse_grain, atol=TOL, rtol=TOL)


def template_pyresponse(wf, window_t0, N_WIN, dur_s, orbit):
    """A: PhenomTHMTDIWaveform (pyResponseTDI Lagrange path)."""
    grid = TDSettings(N=N_WIN, dt=DT, t0=window_t0, force_backend=BACKEND)
    gen = PhenomTHMTDIWaveform(
        waveform_kwargs=dict(higher_modes=list(HMS), include_negative_modes=True,
                             t_low_fit=True, coarse_grain=False, atol=TOL, rtol=TOL),
        Tobs=dur_s, start_freq=START_FREQ, use_reference_time=True,
        waveform_t0=MOJITO_REFERENCE_TIME, data_td_settings=grid, tdi_generation=TDI_GEN_STR,
        tdi_channels=TDI_CHAN, sampling_frequency=1.0 / DT, orbits=orbit, order=ORDER,
        tukey_alpha=TUKEY_ALPHA, stft_dt=None, freq_min=F_MIN, freq_max=F_MAX,
        fft_batch_size=2, buffer_time=BUFFER_TIME, output_domain_settings=None,
        force_backend=BACKEND,
    )
    # waveform basis: [m1,m2,s1z,s2z,dist,phi_ref,inc,psi,ra,dec,t_plunge]
    times, channels = gen.compute_tdi_channels(*wf)
    arr = np.asarray(place_td_signal_on_grid(np.atleast_2d(channels)[:NCHANNELS], grid, times=times).arr)
    del gen; gc.collect()
    mark("template A (pyResponseTDI) built")
    return arr


def template_tdionfly_dtmin(wf, window_t0, N_WIN, dur_s, orbit, wave_gen, dt_min):
    """B: MBHTDIonFly (TDI-on-the-fly).  Same IMRPhenomTHM (wave_gen)."""
    tdi_config = TDIConfig(TDI_GEN_STR, force_backend=BACKEND)
    gen = MBHTDIonFly(wave_gen, orbit, tdi_config, DT, dur_s, t0=MOJITO_REFERENCE_TIME,
                      dt_min=dt_min, waveform_duration=dur_s, force_backend=BACKEND)
    grid_t = np.arange(N_WIN) * DT + window_t0
    m1, m2, s1z, s2z, dist, phi_ref, inc, psi, ra, dec, t_plunge = wf
    out = gen(m1, m2, s1z, s2z, dist, phi_ref, inc, ra, dec, psi, t_plunge,
              upsample_t_arr=grid_t, combine=True)
    arr = np.asarray(out)[:NCHANNELS]
    del gen; gc.collect()
    mark(f"template B (TDI-on-the-fly, dt_min={dt_min}s) built")
    return arr


def main():
    threading.Thread(target=_watchdog, daemon=True).start()
    banner(f"MBH response comparison vs mojito id={MBHB_ID}  (pyResponseTDI vs TDI-on-the-fly)")
    z = np.load(DATA_CACHE, allow_pickle=True)
    data_td = z["data_td"]; window_t0 = float(z["window_t0"]); cat = z["cat"].item()
    N_WIN = data_td.shape[1]; TOBS = N_WIN * DT
    samp = mbh_catalogue_to_sampling_basis(cat)
    wf = np.asarray(MBH_TRANSFORM.both_transforms(np.asarray(samp, float)), float)
    MERGER_FRAC = 0.72
    dur_s = (MERGER_FRAC * (TOBS / 86400.0) + 6.0) * 86400.0   # = match-script duration
    print(f"  N_WIN={N_WIN} Tobs={TOBS/86400:.1f}d dur={dur_s/86400:.1f}d  "
          f"B: coarse_grain={CG_B} dt_min={DTMIN}s  pos_dt={POS_DT}s", flush=True)

    orbit = build_orbit(window_t0, TOBS)
    A = template_pyresponse(wf, window_t0, N_WIN, dur_s, orbit)
    win = tukey(N_WIN, TUKEY_ALPHA)
    td_set = TDSettings(N_WIN, DT, t0=window_t0, force_backend=BACKEND)
    bands = {"full[1e-4,2.5e-2]": (F_MIN, F_MAX), ">5e-4": (5e-4, F_MAX), ">1mHz": (1e-3, F_MAX)}

    def mm(ref_td, tmpl_td, lo, hi):
        fd = FDSettings(N=N_WIN // 2 + 1, df=1.0 / (N_WIN * DT), min_freq=lo, max_freq=hi, force_backend=BACKEND)
        d = TDSignal(ref_td, td_set).transform(fd, window=win)
        t = TDSignal(tmpl_td, td_set).transform(fd, window=win)
        ac = AnalysisContainer(d, XYZ2SensitivityMatrix(fd, model=SENS_MODEL))
        O = ac.template_inner_product(t, normalize=True, complex=True)
        return 1 - abs(O)

    banner("A = pyResponseTDI  vs  mojito data")
    for tag, (lo, hi) in bands.items():
        print(f"  {tag:20s} mm = {mm(data_td, A, lo, hi):.4e}", flush=True)

    # dt_min convergence of the on-the-fly response B (coarse_grain=True).
    # If B->A as dt_min->0, the responses AGREE (B was merger-under-resolved, so
    # pyResponse interpolation is NOT a real error).  If B plateaus ~1.5% from A,
    # the response METHODS genuinely differ at that level.
    banner("B = TDI-on-the-fly  (coarse_grain=True) dt_min convergence")
    print(f"  {'dt_min[s]':>10s} {'B vs data >5e-4':>16s} {'B vs data >1mHz':>16s} "
          f"{'A vs B >1mHz':>14s}", flush=True)
    dtmins = [float(x) for x in os.environ.get("MBH_DTMINS", "0.1,0.05,0.02").split(",")]
    B = None
    for dm in dtmins:
        wg = make_imrphenomthm(dur_s, coarse_grain=True)
        Bm = template_tdionfly_dtmin(wf, window_t0, N_WIN, dur_s, orbit, wg, dm)
        print(f"  {dm:10.3f} {mm(data_td, Bm, 5e-4, F_MAX):16.4e} "
              f"{mm(data_td, Bm, 1e-3, F_MAX):16.4e} {mm(A, Bm, 1e-3, F_MAX):14.4e}", flush=True)
        if B is None:
            B = Bm   # keep the first (dt_min=0.1) for the plot
        del Bm, wg; gc.collect()

    # quick FD plot of both vs data (A and the dt_min=0.1 B)
    import matplotlib; matplotlib.use("Agg"); import matplotlib.pyplot as plt
    f = np.fft.rfftfreq(N_WIN, DT)
    Fd = np.fft.rfft(data_td[0] * win) * DT
    Fa = np.fft.rfft(A[0] * win) * DT
    Fb = np.fft.rfft(B[0] * win) * DT
    bsel = (f >= F_MIN) & (f <= F_MAX)
    fig, ax = plt.subplots(1, 2, figsize=(14, 5))
    ax[0].loglog(f[bsel], np.abs(Fd[bsel]), label="data", lw=1.0)
    ax[0].loglog(f[bsel], np.abs(Fa[bsel]), label="A pyResponse", lw=0.9, alpha=.8)
    ax[0].loglog(f[bsel], np.abs(Fb[bsel]), label="B on-the-fly", lw=0.9, alpha=.8, ls="--")
    ax[0].legend(); ax[0].set_title("FD |X| data vs both responses"); ax[0].set_xlabel("f [Hz]")
    ax[1].loglog(f[bsel], np.abs(Fa[bsel] - Fb[bsel]), color="purple", lw=0.9, label="|A - B|")
    ax[1].loglog(f[bsel], np.abs(Fd[bsel]), color="C0", lw=0.7, alpha=.4, label="data")
    ax[1].legend(); ax[1].set_title("FD |X| response difference A-B"); ax[1].set_xlabel("f [Hz]")
    fig.tight_layout(); fig.savefig(f"/tmp/mbh_mojito_response_compare_id{MBHB_ID}.png", dpi=110); plt.close(fig)
    print(f"DONE.  plot -> /tmp/mbh_mojito_response_compare_id{MBHB_ID}.png", flush=True)


if __name__ == "__main__":
    main()
