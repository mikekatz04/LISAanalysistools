"""Find the correct MBHTDIonFly mode-phase convention by matching pyResponse.

MBHTDIonFly feeds the TDI-on-the-fly response mode_phase = pi - sc_phase, i.e.
the response sees amp*exp(i(pi - sc_phase)) = -conj(strain_component).  phentax's
own compute_polarizations builds h+ = Re(sum sc), h- = -Im(sum sc) directly (no
conjugation), so the conjugation flips the h+/hx relative sign -> an
inclination-dependent cross-pol error (small face-on, ~11% edge-on).

This replicates the MBHTDIonFly.__call__ internals with the mode_phase transform
PARAMETERIZED, and sweeps it at EDGE-ON (inc=pi/2, where the error is largest)
to find which convention drives A-vs-B -> 0.  The winner is the fix to apply in
bbhx/src/bbhx/mbhtdionfly.py.
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
            sys.stderr.write(f"\n[WATCHDOG] abort\n"); sys.stderr.flush(); os._exit(42)
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
WIN_DAYS = float(os.environ.get("MBH_WIN_DAYS", "6.0"))
DUR_DAYS = float(os.environ.get("MBH_DUR_DAYS", "5.0"))
INC_OVERRIDE = os.environ.get("MBH_INC", "1.5708")   # edge-on by default
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


# phase-transform conventions to test (response sees amp*exp(i*mode_phase))
PHASE_FNS = {
    "pi - phi  (current)": lambda p: np.pi - p,
    "phi":                 lambda p: p,
    "-phi":                lambda p: -p,
    "pi + phi":            lambda p: np.pi + p,
}


def gen_B(wave_gen, orbit, wf, window_t0, N_WIN, dur_s, phase_fn, inc_resp=0.0):
    """Replicates MBHTDIonFly.__call__ with mode_phase = phase_fn(sc_phase) and
    the inclination handed to the response (inc_resp; current code uses 0.0)."""
    m1, m2, s1z, s2z, dist, phi_ref, inc, psi, ra, dec, t_plunge = wf
    tdi_config = TDIConfig(TDI_GEN_STR, force_backend=BACKEND)
    new_times, new_mask, sc_amp, sc_phase = wave_gen.compute_strain_components_amp_phase(
        m1, m2, s1z, s2z, dist, phi_ref, inc, psi, delta_t=DTMIN, t_min=-dur_s, t_ref=0.0)
    mode_amp = np.asarray(sc_amp) / 2.0
    mode_phase = phase_fn(np.asarray(sc_phase))
    nmodes = wave_gen.num_modes
    _new_times = np.asarray(new_times[new_mask] + t_plunge + REF)
    new_times_arr = np.repeat(_new_times[None, :], nmodes, axis=0)
    amp = np.asarray(mode_amp[0][:, new_mask[0]])
    phase = np.asarray(mode_phase[0][:, new_mask[0]])
    tdi_buffer = int(1000 / DT)
    eval_t_arr = new_times_arr[:, tdi_buffer:-tdi_buffer]
    tdi_gen = TDTDIonTheFly(eval_t_arr, amp, phase, sampling_frequency=1.0 / DT,
                            num_sub=nmodes, t_input=new_times_arr, tdi_config=tdi_config,
                            orbits=orbit, force_backend=BACKEND)
    inc_arr = np.full(nmodes, inc_resp); pol = np.full(nmodes, psi)
    ra_arr = np.full(nmodes, ra); dec_arr = np.full(nmodes, dec)
    output = tdi_gen(inc_arr, pol, ra_arr, dec_arr, return_spline=True)
    grid_t = np.arange(N_WIN) * DT + window_t0
    new_tdi = np.zeros((output.t_arr.shape[0], 3, N_WIN))
    keep = (grid_t >= output.t_arr.min().item()) & (grid_t <= output.t_arr.max().item())
    new_tdi[:, :, keep] = output.eval_tdi(grid_t[keep])
    arr = new_tdi.sum(axis=0)[:NCH]
    del tdi_gen, output; gc.collect()
    return arr


def main():
    threading.Thread(target=_watchdog, daemon=True).start()
    banner(f"MBH on-the-fly mode-phase convention fix (id={MBHB_ID})")
    z = np.load(DATA_CACHE, allow_pickle=True); cat = z["cat"].item()
    samp = mbh_catalogue_to_sampling_basis(cat)
    wf = np.asarray(MBH_TRANSFORM.both_transforms(np.asarray(samp, float)), float)
    wf[6] = float(INC_OVERRIDE)                  # inc (edge-on by default)
    t_plunge = wf[10]; abs_merger = REF + t_plunge
    N_WIN = int(round(WIN_DAYS * 86400 / DT))
    window_t0 = abs_merger - 0.72 * N_WIN * DT; TOBS = N_WIN * DT
    dur_s = DUR_DAYS * 86400.0
    print(f"  inc={wf[6]:.4f} (edge-on if ~1.57)  N_WIN={N_WIN}", flush=True)

    orbit = build_orbit(window_t0, TOBS)
    A = gen_A(wf, window_t0, N_WIN, dur_s, orbit)
    wave_gen = pw.IMRPhenomTHM(T=dur_s, higher_modes=list(HMS), include_negative_modes=True,
                               t_low_fit=True, coarse_grain=True, atol=TOL, rtol=TOL)
    win = tukey(N_WIN, TUKEY_ALPHA); td_set = TDSettings(N_WIN, DT, t0=window_t0, force_backend=BACKEND)
    fd = FDSettings(N=N_WIN // 2 + 1, df=1.0 / (N_WIN * DT), min_freq=5e-4, max_freq=F_MAX, force_backend=BACKEND)
    ac = AnalysisContainer(TSig := TDSignal(A, td_set).transform(fd, window=win),
                           XYZ2SensitivityMatrix(fd, model=SENS_MODEL))
    def AvB(B):
        t = TDSignal(B, td_set).transform(fd, window=win)
        return 1 - abs(ac.template_inner_product(t, normalize=True, complex=True))

    cur = lambda p: np.pi - p   # current MBHTDIonFly mode_phase convention
    banner("PROBE: does the response 'inc' param even change B?")
    B0 = gen_B(wave_gen, orbit, wf, window_t0, N_WIN, dur_s, cur, inc_resp=0.0)
    Bpi = gen_B(wave_gen, orbit, wf, window_t0, N_WIN, dur_s, cur, inc_resp=np.pi)
    dmax = np.abs(B0 - Bpi).max(); ref = np.abs(B0).max()
    print(f"  max|B(inc=0) - B(inc=pi)| = {dmax:.3e}   (max|B0| = {ref:.3e}, "
          f"rel = {dmax/ref:.3e})", flush=True)
    print(f"  -> inc param is {'INERT (not reaching get_hp_hc!)' if dmax/ref < 1e-10 else 'ACTIVE'}", flush=True)
    print(f"  A-vs-B0(inc=0)  = {AvB(B0):.4e}", flush=True)
    print(f"  A-vs-Bpi(inc=pi)= {AvB(Bpi):.4e}", flush=True)
    print(f"  B0-vs-Bpi       = {AvB(Bpi if False else B0) if False else 0:.4e}", flush=True)
    # direct template-template mismatch B0 vs Bpi
    t0 = TDSignal(B0, td_set).transform(fd, window=win)
    tpi = TDSignal(Bpi, td_set).transform(fd, window=win)
    acB = AnalysisContainer(t0, XYZ2SensitivityMatrix(fd, model=SENS_MODEL))
    print(f"  mm(B0, Bpi)     = {1-abs(acB.template_inner_product(tpi, normalize=True, complex=True)):.4e}", flush=True)
    print("\nDONE.", flush=True)


if __name__ == "__main__":
    main()
