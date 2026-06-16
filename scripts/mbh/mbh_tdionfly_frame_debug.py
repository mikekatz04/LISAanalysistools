"""Frame / convention debug for the MBH TDI-on-the-fly response (MBHTDIonFly)
vs pyResponseTDI (PhenomTHMTDIWaveform).

Both paths consume sky in the ORBIT frame (orbits frame='icrs') with the SAME
ICRS (ra, dec, psi) -- verified: TDTDIonTheFly's base __call__(inc, psi, lam,
beta) takes lam=ra, beta=dec directly (no convert_to_ra_dec), pyResponse passes
lam=ra, beta=dec to get_projections.  So the residual A-vs-B (~1.5%, flat in
dt_min) is NOT a sky-frame (icrs/ecliptic) issue -- it is a POLARIZATION
convention difference: pyResponse bakes psi into h+,hx in the waveform; the
on-the-fly applies psi in the response.

This sweeps psi / sky sign+offset conventions on B (ALL within ICRS) and reports
the phase-maximized A-vs-B mismatch in the merger band (>1 mHz).  The variant
that drives A-vs-B -> 0 identifies the convention to fix in MBHTDIonFly.

Small window around the merger -> fast (each B gen ~seconds).  No mojito data
needed for A-vs-B (pure template-vs-template).
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
MBHB_ID = int(os.environ.get("MBHB_ID", "0"))
BACKEND = "cpu"; SENS_MODEL = "scirdv1"; DT = 10.0
TDI_GEN_STR = "2nd generation"; TDI_CHAN = "XYZ"; NCH = 3
F_MIN, F_MAX = 1e-4, 2.5e-2
HMS = (21, 33, 44); TOL = 1e-12; ORDER = 30; BUFFER = 15_000.0; START_FREQ = 7e-5
TUKEY_ALPHA = 0.05; POS_DT = 300.0
WIN_DAYS = float(os.environ.get("MBH_WIN_DAYS", "6.0"))     # small window around merger
DUR_DAYS = float(os.environ.get("MBH_DUR_DAYS", "5.0"))     # waveform pre-merger span
DTMIN = 0.1
MBH_TRANSFORM = make_mbh_transform_container()
DATA_CACHE = f"/tmp/mbh_mojito_data_id{MBHB_ID}.npz"


def banner(s): print("\n" + "=" * 78 + f"\n {s}\n" + "=" * 78, flush=True)
def tukey(N, a):
    from scipy.signal.windows import tukey as _t
    return _t(N, a)


def build_orbit(window_t0, TOBS):
    orb = L1Orbits(find_file(MBHB_L1, "MBHB", MBHB_ID), force_backend=BACKEND, frame="icrs")
    pad = 1.0e5
    lo = max(window_t0 - pad, float(orb.sc_t0)); hi = min(window_t0 + TOBS + pad, float(orb._sc_t_base[-1]))
    ltt_t = np.asarray(orb.ltt_t); m = (ltt_t >= lo) & (ltt_t <= hi)
    orb.ltt = np.asarray(orb.ltt)[m].copy(); orb.ltt_t = ltt_t[m].copy(); orb.ltt_t0 = float(orb.ltt_t[0])
    del ltt_t; gc.collect()
    orb.configure(linear_interp_setup=True, dt=POS_DT)
    return orb


def gen_A(wf, window_t0, N_WIN, dur_s, orbit):
    grid = TDSettings(N=N_WIN, dt=DT, t0=window_t0, force_backend=BACKEND)
    gen = PhenomTHMTDIWaveform(
        waveform_kwargs=dict(higher_modes=list(HMS), include_negative_modes=True,
                             t_low_fit=True, coarse_grain=False, atol=TOL, rtol=TOL),
        Tobs=dur_s, start_freq=START_FREQ, use_reference_time=True, waveform_t0=REF,
        data_td_settings=grid, tdi_generation=TDI_GEN_STR, tdi_channels=TDI_CHAN,
        sampling_frequency=1.0 / DT, orbits=orbit, order=ORDER, tukey_alpha=TUKEY_ALPHA,
        stft_dt=None, freq_min=F_MIN, freq_max=F_MAX, fft_batch_size=2, buffer_time=BUFFER,
        output_domain_settings=None, force_backend=BACKEND)
    times, ch = gen.compute_tdi_channels(*wf)
    arr = np.asarray(place_td_signal_on_grid(np.atleast_2d(ch)[:NCH], grid, times=times).arr)
    del gen; gc.collect()
    return arr


def gen_B(m1, m2, s1z, s2z, dist, phi_ref, inc, ra, dec, psi, t_plunge,
          window_t0, N_WIN, dur_s, orbit, wave_gen):
    tdi_config = TDIConfig(TDI_GEN_STR, force_backend=BACKEND)
    gen = MBHTDIonFly(wave_gen, orbit, tdi_config, DT, dur_s, t0=REF, dt_min=DTMIN,
                      waveform_duration=dur_s, force_backend=BACKEND)
    grid_t = np.arange(N_WIN) * DT + window_t0
    out = gen(m1, m2, s1z, s2z, dist, phi_ref, inc, ra, dec, psi, t_plunge,
              upsample_t_arr=grid_t, combine=True)
    arr = np.asarray(out)[:NCH]
    del gen; gc.collect()
    return arr


def main():
    threading.Thread(target=_watchdog, daemon=True).start()
    banner(f"MBH TDI-on-the-fly FRAME debug (id={MBHB_ID})  --  all sky in ICRS")
    z = np.load(DATA_CACHE, allow_pickle=True)
    cat = z["cat"].item()
    samp = mbh_catalogue_to_sampling_basis(cat)
    wf = np.asarray(MBH_TRANSFORM.both_transforms(np.asarray(samp, float)), float)
    m1, m2, s1z, s2z, dist, phi_ref, inc, psi, ra, dec, t_plunge = wf
    inc = float(os.environ.get("MBH_INC", str(inc)))   # override (e.g. pi/2 = edge-on, where psi matters)
    wf = wf.copy(); wf[6] = inc                         # A and B both use the (possibly overridden) inc
    abs_merger = REF + t_plunge

    # small window centered on merger
    N_WIN = int(round(WIN_DAYS * 86400 / DT))
    window_t0 = abs_merger - 0.72 * N_WIN * DT
    TOBS = N_WIN * DT
    dur_s = DUR_DAYS * 86400.0
    print(f"  N_WIN={N_WIN} win={WIN_DAYS}d dur={DUR_DAYS}d  ra={ra:.4f} dec={dec:.4f} psi={psi:.4f} inc={inc:.4f}", flush=True)

    orbit = build_orbit(window_t0, TOBS)
    A = gen_A(wf, window_t0, N_WIN, dur_s, orbit)
    print(f"  [A pyResponse built] RSS={rss_gb():.2f}GB", flush=True)
    wave_gen = pw.IMRPhenomTHM(T=dur_s, higher_modes=list(HMS), include_negative_modes=True,
                               t_low_fit=True, coarse_grain=True, atol=TOL, rtol=TOL)

    win = tukey(N_WIN, TUKEY_ALPHA)
    td_set = TDSettings(N_WIN, DT, t0=window_t0, force_backend=BACKEND)
    fd = FDSettings(N=N_WIN // 2 + 1, df=1.0 / (N_WIN * DT), min_freq=5e-4, max_freq=F_MAX, force_backend=BACKEND)
    A_fd = TDSignal(A, td_set).transform(fd, window=win)
    ac = AnalysisContainer(A_fd, XYZ2SensitivityMatrix(fd, model=SENS_MODEL))
    def AvB(B):
        t = TDSignal(B, td_set).transform(fd, window=win)
        return 1 - abs(ac.template_inner_product(t, normalize=True, complex=True))

    # inclination / polarization / sky convention variants -- ALL within ICRS.
    # flip_hx (cross-pol handedness) ~ (inc->pi-inc & psi->-psi).
    PI = np.pi
    variants = [   # (label, inc, ra, dec, psi)
        ("baseline",                       inc,      ra,  dec,  psi),
        ("inc->pi-inc",                    PI - inc, ra,  dec,  psi),
        ("psi->-psi",                      inc,      ra,  dec, -psi),
        ("inc->pi-inc, psi->-psi  [flipHx]", PI-inc,  ra,  dec, -psi),
        ("inc->-inc",                     -inc,      ra,  dec,  psi),
        ("inc->-inc, psi->-psi",          -inc,      ra,  dec, -psi),
        ("dec->-dec, psi->-psi",           inc,      ra, -dec, -psi),
    ]
    banner("A-vs-B (phase-max, >5e-4 Hz) per convention variant on B  [want ~0]")
    results = []
    for label, inc_v, ra_v, dec_v, psi_v in variants:
        B = gen_B(m1, m2, s1z, s2z, dist, phi_ref, inc_v, ra_v, dec_v, psi_v, t_plunge,
                  window_t0, N_WIN, dur_s, orbit, wave_gen)
        mm = AvB(B)
        results.append((mm, label))
        print(f"  {label:28s}  A-vs-B = {mm:.4e}", flush=True)
        del B; gc.collect()
    results.sort()
    banner("BEST variants (smallest A-vs-B)")
    for mm, label in results[:3]:
        print(f"  {label:28s}  {mm:.4e}", flush=True)
    print("\nDONE.", flush=True)


if __name__ == "__main__":
    main()
