"""A-vs-B (pyResponse vs TDI-on-the-fly) as a function of WAVEFORM inclination.

inc=0 vs inc=pi flips handedness (positive- vs negative-m mode dominance);
inc=+pi/2 vs -pi/2 flips sign(sin inc).  Mapping A-vs-B across these isolates
whether the on-the-fly discrepancy tracks the cross-pol / negative-mode parity.
Pure template-vs-template (no mojito data needed).
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
            sys.stderr.write("\n[WATCHDOG] abort\n"); sys.stderr.flush(); os._exit(42)
        time.sleep(0.3)

from lisatools.detector import L1Orbits
from lisatools.globalfit.preprocessing import find_file
from lisatools.globalfit.recipe_steps import mbh_catalogue_to_sampling_basis
from lisatools.globalfit.stock.erebor import make_mbh_transform_container
from lisatools.response.tdiconfig import TDIConfig
from lisatools.response.tdionfly import TDTDIonTheFly
from lisatools.sources.bbh.waveform import PhenomTHMTDIWaveform
from lisatools.domains import TDSettings, FDSettings, TDSignal, place_td_signal_on_grid
from lisatools.analysiscontainer import AnalysisContainer
from lisatools.sensitivity import XYZ2SensitivityMatrix
import phentax.waveform as pw

REF = 97729089.327664
PATH = "/Users/mkatz/.mojito_cache/brickmarket/mojito_light_v1_0_0/"
MBHB_L1 = os.path.join(PATH, "data", "MBHB", "L1")
MBHB_ID = int(os.environ.get("MBHB_ID", "0"))
BACKEND = "cpu"; SENS_MODEL = "scirdv1"; DT = 10.0
TDI_GEN_STR = "2nd generation"; TDI_CHAN = "XYZ"; NCH = 3
F_MIN, F_MAX = 1e-4, 2.5e-2
HMS = (21, 33, 44); TOL = 1e-12; ORDER = 30; BUFFER = 15_000.0; START_FREQ = 7e-5
TUKEY_ALPHA = 0.05; POS_DT = 300.0; DTMIN = 0.1
WIN_DAYS = 6.0; DUR_DAYS = 5.0
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


def gen_B(wave_gen, orbit, wf, window_t0, N_WIN, dur_s):
    m1, m2, s1z, s2z, dist, phi_ref, inc, psi, ra, dec, t_plunge = wf
    tdi_config = TDIConfig(TDI_GEN_STR, force_backend=BACKEND)
    nt, nm, sc_amp, sc_phase = wave_gen.compute_strain_components_amp_phase(
        m1, m2, s1z, s2z, dist, phi_ref, inc, psi, delta_t=DTMIN, t_min=-dur_s, t_ref=0.0)
    mode_amp = np.asarray(sc_amp) / 2.0
    mode_phase = np.pi - np.asarray(sc_phase)
    nmodes = wave_gen.num_modes
    _nt = np.asarray(nt[nm] + t_plunge + REF)
    nta = np.repeat(_nt[None, :], nmodes, axis=0)
    amp = np.asarray(mode_amp[0][:, nm[0]]); phase = np.asarray(mode_phase[0][:, nm[0]])
    tb = int(1000 / DT)
    eval_t = nta[:, tb:-tb]
    g = TDTDIonTheFly(eval_t, amp, phase, sampling_frequency=1.0 / DT, num_sub=nmodes,
                      t_input=nta, tdi_config=tdi_config, orbits=orbit, force_backend=BACKEND)
    out = g(np.full(nmodes, 0.0), np.full(nmodes, psi), np.full(nmodes, ra), np.full(nmodes, dec),
            return_spline=True)
    grid_t = np.arange(N_WIN) * DT + window_t0
    ntdi = np.zeros((out.t_arr.shape[0], 3, N_WIN))
    keep = (grid_t >= out.t_arr.min().item()) & (grid_t <= out.t_arr.max().item())
    ntdi[:, :, keep] = out.eval_tdi(grid_t[keep])
    arr = ntdi.sum(axis=0)[:NCH]
    del g, out; gc.collect()
    return arr


def main():
    threading.Thread(target=_watchdog, daemon=True).start()
    banner(f"A-vs-B vs WAVEFORM inclination (id={MBHB_ID})")
    z = np.load(DATA_CACHE, allow_pickle=True); cat = z["cat"].item()
    samp = mbh_catalogue_to_sampling_basis(cat)
    wf0 = np.asarray(MBH_TRANSFORM.both_transforms(np.asarray(samp, float)), float)
    t_plunge = wf0[10]; abs_merger = REF + t_plunge
    N_WIN = int(round(WIN_DAYS * 86400 / DT)); window_t0 = abs_merger - 0.72 * N_WIN * DT
    TOBS = N_WIN * DT; dur_s = DUR_DAYS * 86400.0
    orbit = build_orbit(window_t0, TOBS)
    wave_gen = pw.IMRPhenomTHM(T=dur_s, higher_modes=list(HMS), include_negative_modes=True,
                               t_low_fit=True, coarse_grain=True, atol=TOL, rtol=TOL)
    win = tukey(N_WIN, TUKEY_ALPHA); td_set = TDSettings(N_WIN, DT, t0=window_t0, force_backend=BACKEND)
    fd = FDSettings(N=N_WIN // 2 + 1, df=1.0 / (N_WIN * DT), min_freq=5e-4, max_freq=F_MAX, force_backend=BACKEND)

    def mm(Atd, Btd):
        a = AnalysisContainer(TDSignal(Atd, td_set).transform(fd, window=win),
                              XYZ2SensitivityMatrix(fd, model=SENS_MODEL))
        t = TDSignal(Btd, td_set).transform(fd, window=win)
        return 1 - abs(a.template_inner_product(t, normalize=True, complex=True))

    PI = np.pi
    incs = [("nat 0.42", float(wf0[6])), ("0", 0.0), ("pi/4", PI/4), ("pi/2", PI/2),
            ("3pi/4", 3*PI/4), ("pi", PI), ("-pi/4", -PI/4), ("-pi/2", -PI/2)]
    banner("A-vs-B (phase-max, >5e-4 Hz) vs inclination")
    print(f"  {'inc':>10} {'value':>9}   {'A-vs-B':>12}", flush=True)
    for lbl, inc_val in incs:
        wf = wf0.copy(); wf[6] = inc_val
        A = gen_A(wf, window_t0, N_WIN, dur_s, orbit)
        B = gen_B(wave_gen, orbit, wf, window_t0, N_WIN, dur_s)
        print(f"  {lbl:>10} {inc_val:9.4f}   {mm(A, B):12.4e}", flush=True)
        del A, B; gc.collect()
    print("\nDONE.", flush=True)


if __name__ == "__main__":
    main()
