"""FINAL injection-parameter summary for MBHB id0 vs mojito.

Legacy pyResponse (A = PhenomTHMTDIWaveform) vs TDI-on-the-fly
(B = MBHTDIonFly, t_ref-fixed + coarse_graining_scale_factor=48), scirdv1 +
mrdv1, bands full[1e-4,2.5e-2] and >1mHz:

  overlap: 1-Re(O) at injection (NON-phase-marg) | 1-|O| at injection (phase-marg)
           | 1-|O| phase+time-max (fitting factor)
  logL:    -1/2 <d-h|d-h> at the injection phase

Saves a 2-panel plot + the summary values (for the combined plot).
Only MBHB source0 is present in the mojito light dataset (no id1).
"""
import os, sys, gc, time, threading, resource
os.environ.setdefault("OMP_NUM_THREADS", "8")
import numpy as np
import matplotlib; matplotlib.use("Agg"); import matplotlib.pyplot as plt
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
MBHB_ID = 0; BACKEND = "cpu"; DT = 10.0
TDI_GEN = "2nd generation"; NCH = 3
F_MIN, F_MAX = 1e-4, 2.5e-2
HMS = (21, 33, 44); TOL = 1e-12; ORDER = 30; BUFFER = 15_000.0; START_FREQ = 7e-5
TUKEY_ALPHA = 0.05; POS_DT = 300.0; DTMIN = 0.1; SCALE = 48.0
SENS = ["scirdv1", "mrdv1"]
MBH_TRANSFORM = make_mbh_transform_container()
DATA_CACHE = f"/tmp/mbh_mojito_data_id{MBHB_ID}.npz"


def banner(s): print("\n" + "=" * 102 + f"\n {s}\n" + "=" * 102, flush=True)
def tukey(N, a):
    from scipy.signal.windows import tukey as _t
    return _t(N, a)


def metrics(ac_data, T, td, win, fd, f):
    sig = TDSignal(T, td).transform(fd, window=win); t0 = np.asarray(sig.arr).copy()
    O0 = ac_data.template_inner_product(sig, normalize=True, complex=True)
    logL0 = float(np.real(ac_data.template_likelihood(sig)))
    def mO(tau):
        sig.arr[:] = t0 * np.exp(-2j * np.pi * f * tau)[None, :]
        return ac_data.template_inner_product(sig, normalize=True, complex=True)
    c = max(((abs(mO(x)), x) for x in np.linspace(-25, 25, 201)))
    ft = max(((abs(mO(x)), x) for x in c[1] + np.linspace(-0.3, 0.3, 121)))
    sig.arr[:] = t0
    return dict(reO_inj=1 - O0.real, mm_inj=1 - abs(O0), logL_inj=logL0,
                mm_pt=1 - ft[0], taustar=ft[1])


def main():
    threading.Thread(target=_wd, daemon=True).start()
    z = np.load(DATA_CACHE, allow_pickle=True)
    D = np.asarray(z["data_td"])[:NCH]; window_t0 = float(z["window_t0"]); cat = z["cat"].item()
    abs_merger = float(z["abs_merger"]); N_WIN = D.shape[1]; TOBS = N_WIN * DT
    wf = np.asarray(MBH_TRANSFORM.both_transforms(
        np.asarray(mbh_catalogue_to_sampling_basis(cat), float)), float)
    m1, m2, s1z, s2z, dist, phi_ref, inc, psi, ra, dec, t_plunge = wf
    dur_s = ((abs_merger - window_t0) / 86400.0 + 6.0) * 86400.0

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
        data_td_settings=grid, tdi_generation=TDI_GEN, tdi_channels="XYZ",
        sampling_frequency=1.0 / DT, orbits=orb, order=ORDER, tukey_alpha=TUKEY_ALPHA,
        stft_dt=None, freq_min=F_MIN, freq_max=F_MAX, fft_batch_size=2, buffer_time=BUFFER,
        output_domain_settings=None, force_backend=BACKEND)
    tA, chA = gA.compute_tdi_channels(*wf)
    A = np.asarray(place_td_signal_on_grid(np.atleast_2d(chA)[:NCH], grid, times=tA).arr); del gA; gc.collect(); mark("A built")
    wg = pw.IMRPhenomTHM(T=dur_s, higher_modes=list(HMS), include_negative_modes=True,
                         t_low_fit=True, coarse_grain=True, atol=TOL, rtol=TOL, coarse_graining_scale_factor=SCALE)
    mbh = MBHTDIonFly(wg, orb, TDIConfig(TDI_GEN, force_backend=BACKEND), DT, dur_s,
                      t0=REF, dt_min=DTMIN, waveform_duration=dur_s, force_backend=BACKEND)
    grid_t = np.arange(N_WIN) * DT + window_t0
    B = np.asarray(mbh(m1, m2, s1z, s2z, dist, phi_ref, inc, ra, dec, psi, t_plunge,
                       upsample_t_arr=grid_t, combine=True))[:NCH]; mark("B built")

    win = tukey(N_WIN, TUKEY_ALPHA); td = TDSettings(N_WIN, DT, t0=window_t0, force_backend=BACKEND)
    bands = [("full[1e-4,2.5e-2]", F_MIN, F_MAX), (">1mHz", 1e-3, F_MAX)]
    summary = {}
    banner(f"MBHB id{MBHB_ID}   inc={inc:.3f}   ({TOBS/86400:.0f} d window, merger-centered)")
    print(f"  {'sens':>8} {'band':>17} {'tmpl':>10} {'1-Re(O)inj':>12} {'1-|O|inj':>12} "
          f"{'logL_inj':>12} {'1-|O|ptmax':>12} {'tau*':>8}", flush=True)
    for sens in SENS:
        for tag, flo, fhi in bands:
            fd = FDSettings(N=N_WIN // 2 + 1, df=1.0 / (N_WIN * DT), min_freq=flo, max_freq=fhi, force_backend=BACKEND)
            f = np.asarray(fd.f_arr)
            ac = AnalysisContainer(TDSignal(D, td).transform(fd, window=win), XYZ2SensitivityMatrix(fd, model=sens))
            for nm, T in [("legacy", A), ("on-fly", B)]:
                m = metrics(ac, T, td, win, fd, f)
                summary[(sens, tag, nm)] = m
                print(f"  {sens:>8} {tag:>17} {nm:>10} {m['reO_inj']:12.4e} {m['mm_inj']:12.4e} "
                      f"{m['logL_inj']:12.4e} {m['mm_pt']:12.4e} {m['taustar']:+8.3f}", flush=True)

    # plot
    f = np.fft.rfftfreq(N_WIN, DT); sel = (f >= F_MIN) & (f <= F_MAX)
    FD = np.fft.rfft(D[0] * win) * DT; FA = np.fft.rfft(A[0] * win) * DT; FB = np.fft.rfft(B[0] * win) * DT
    fig, ax = plt.subplots(2, 1, figsize=(10, 9))
    ax[0].loglog(f[sel], np.abs(FD[sel]), label="mojito data", lw=1.0)
    ax[0].loglog(f[sel], np.abs(FA[sel]), "--", label="legacy pyResponse", lw=.9, alpha=.8)
    ax[0].loglog(f[sel], np.abs(FB[sel]), ":", label="TDI-on-the-fly", lw=1.2, alpha=.8)
    ax[0].set_xlim(F_MIN, F_MAX); ax[0].legend(); ax[0].set_title(f"MBHB id{MBHB_ID}: FD |X|"); ax[0].set_xlabel("f [Hz]")
    ax[1].loglog(f[sel], np.abs((FA - FD)[sel]), label="|legacy - data|", lw=.9)
    ax[1].loglog(f[sel], np.abs((FB - FD)[sel]), label="|on-fly - data|", lw=.9)
    ax[1].set_xlim(F_MIN, F_MAX); ax[1].legend(); ax[1].set_title("FD |X| residual vs data"); ax[1].set_xlabel("f [Hz]")
    fig.tight_layout(); out = f"/tmp/mbh_injection_final_id{MBHB_ID}.png"; fig.savefig(out, dpi=110); plt.close(fig)
    print(f"  plot -> {out}", flush=True)
    np.savez("/tmp/mbh_injection_summary.npz", summary=summary)
    print("\nDONE.  values -> /tmp/mbh_injection_summary.npz", flush=True)


if __name__ == "__main__":
    main()
