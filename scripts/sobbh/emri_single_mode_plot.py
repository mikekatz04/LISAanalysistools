"""Single-mode (2,2,0,0) TD + FD plot: package EMRITDIonFly vs legacy pyResponse on a
shared ecliptic orbit. Shows the amplitude (opt/data) and the AMPLITUDE-NORMALIZED
shape match (|O|), with a TD overlay scaled to the same RMS so the shape is visible.
"""
import os, time, threading, resource, gc
os.environ.setdefault("OMP_NUM_THREADS", "8")
import numpy as np, h5py
import matplotlib
matplotlib.use("Agg")
import matplotlib.pyplot as plt
from scipy.signal.windows import tukey
from mojito import MojitoL1File
from lisatools.detector import L1Orbits
from lisatools.globalfit.preprocessing import find_file
from lisatools.response.directresponse import ResponseWrapper
from lisatools.response.tdiconfig import TDIConfig
from lisatools.utils.constants import YRSID_SI
from lisatools.sources.utils import icrs_to_ecliptic
from lisatools.domains import TDSettings, FDSettings, TDSignal
from lisatools.analysiscontainer import AnalysisContainer
from lisatools.sensitivity import XYZ2SensitivityMatrix
from lisatools.sources.emri import EMRITDIonFly
from few.waveform import FastKerrEccentricEquatorialFlux, GenerateEMRIWaveform

PATH = "/Users/mkatz/.mojito_cache/brickmarket/mojito_light_v1_0_0/"
EMRI_L1 = os.path.join(PATH, "data", "EMRI", "L1")
REF = 97729089.327664; SRC = 1
DT = 20.0; N_WIN = 16384; TOBS_S = N_WIN * DT
T_BUF = 1000.0; N_PTS = 4096; DELAY = 800.0
MODE = [(2, 2, 0, 0)]


def wd():
    while True:
        if resource.getrusage(resource.RUSAGE_SELF).ru_maxrss / 1e9 > 6.0:
            os._exit(42)
        time.sleep(0.2)


class FixedModeGen:
    def __init__(self, fg, mode_sel):
        self.fg = fg; self.mode_sel = mode_sel
    def __call__(self, *args, **kwargs):
        kwargs.setdefault("mode_selection", self.mode_sel)
        kwargs.setdefault("include_minus_mkn", True)
        return self.fg(*args, **kwargs)
    def __getattr__(self, name):
        return getattr(self.fg, name)


def main():
    threading.Thread(target=wd, daemon=True).start()
    cat = os.path.join(PATH, "catalogues", "emri_cat_mojito_lite_processed_MT.hdf5")
    with h5py.File(cat, "r") as f:
        b = f["Binaries"]; g = lambda k: float(b[k][SRC])
        M, mu, a = g("PrimaryMassSSBFrame"), g("SecondaryMassSSBFrame"), g("PrimarySpinParameter")
        p0, e0, dist = g("SemiLatusRectum"), g("Eccentricity"), g("LuminosityDistance") / 1e3
        ra, dec = g("RightAscension") % (2 * np.pi), g("Declination")
        qK, phiK = g("PolarAnglePrimarySpin"), g("AzimuthalAnglePrimarySpin")
        Pp, Pt, Pr = g("AzimuthalPhase"), g("PolarPhase"), g("RadialPhase")
    lam_S, beta_S = icrs_to_ecliptic(float(ra), float(dec))
    qS_e, phiS_e = float(np.pi / 2 - beta_S), float(lam_S) % (2 * np.pi)
    lam_K, beta_K = icrs_to_ecliptic(float(phiK) % (2 * np.pi), float(np.pi / 2 - qK))
    qK_e, phiK_e = float(np.pi / 2 - beta_K), float(lam_K) % (2 * np.pi)
    inj = [M, mu, a, p0, e0, 1.0, dist, qS_e, phiS_e, qK_e, phiK_e, Pp, Pt, Pr]

    fp = find_file(EMRI_L1, "EMRI", SRC)
    ts = MojitoL1File(fp).tdis.time_sampling
    data_t0 = float(ts.t0); deci = int(round(DT / ts.dt))
    with h5py.File(fp, "r") as f:
        lf = float(f.attrs["laser_frequency"])
        dXYZ = np.stack([np.asarray(f["tdis"][c][: N_WIN * deci])[::deci][:N_WIN] / lf for c in ("X2", "Y2", "Z2")])

    print("  building ECLIPTIC orbit...", flush=True)
    orb = L1Orbits(fp, force_backend="cpu", frame="ecliptic")
    pad = 1e5; lo = max(REF - pad, float(orb.sc_t0)); hi = min(data_t0 + TOBS_S + pad, float(orb._sc_t_base[-1]))
    lt = np.asarray(orb.ltt_t); m = (lt >= lo) & (lt <= hi)
    orb.ltt = np.asarray(orb.ltt)[m].copy(); orb.ltt_t = lt[m].copy(); orb.ltt_t0 = float(orb.ltt_t[0]); gc.collect()
    orb.configure(linear_interp_setup=True)

    off = data_t0 - REF
    new_t = np.linspace(off + T_BUF, off + TOBS_S - T_BUF, N_PTS)
    wg = FastKerrEccentricEquatorialFlux(force_backend="cpu",
        inspiral_kwargs={"DENSE_STEPPING": 0, "max_init_len": int(1e4),
                         "upsample": True, "fix_t": True, "new_t": new_t}, sum_kwargs={"pad_output": True})
    tdi_config = TDIConfig("2nd generation", force_backend="cpu")

    print("  EMRITDIonFly single mode...", flush=True)
    fly = EMRITDIonFly(wg, orb, tdi_config, DT, TOBS_S, REF)
    out = fly(*inj, include_minus_mkn=True, mode_selection=MODE)
    tgrid = data_t0 + np.arange(N_WIN) * DT
    tg = np.asarray(out.x); t_lo, t_hi = float(np.max(tg[:, 0])) + DELAY, float(np.min(tg[:, -1])) - DELAY
    inside = (tgrid > t_lo) & (tgrid < t_hi)
    _tdi = out.eval_tdi(tgrid[inside]); tof = np.zeros((3, N_WIN)); tof[:, inside] = np.real(np.sum(np.asarray(_tdi), axis=0))

    print("  legacy single mode...", flush=True)
    fg = GenerateEMRIWaveform("FastKerrEccentricEquatorialFlux", return_list=False,
        inspiral_kwargs={"DENSE_STEPPING": 0, "max_init_len": int(1e4), "force_backend": "cpu"},
        sum_kwargs={"pad_output": True}, frame="detector", force_backend="cpu")
    wl = ResponseWrapper(FixedModeGen(fg, MODE), orbits=orb, t0=data_t0, Tobs=TOBS_S / YRSID_SI, dt=DT,
        index_lambda=8, index_beta=7, flip_hx=True, tdi=TDIConfig("2nd generation", force_backend="cpu"),
        tdi_chan="XYZ", order=40, remove_garbage="zero", t_buffer=3e4, force_backend="cpu")
    leg = np.atleast_2d(np.asarray(wl(*inj, convert_to_ra_dec=False)))[:3]
    leg = (np.pad(leg, ((0, 0), (0, N_WIN - leg.shape[-1]))) if leg.shape[-1] < N_WIN else leg[:, :N_WIN])

    # --- metrics ---
    win = tukey(N_WIN, 0.1)
    td_set = TDSettings(N_WIN, DT, t0=0.0, force_backend="cpu")
    fd_set = FDSettings(N=N_WIN // 2 + 1, df=1.0 / (N_WIN * DT), min_freq=1e-4, max_freq=1e-2, force_backend="cpu")
    data_fd = TDSignal(dXYZ, td_set).transform(fd_set, window=win)
    ac = AnalysisContainer(data_fd, XYZ2SensitivityMatrix(fd_set, model="scirdv1"))
    dd = float(ac.inner_product().real)
    optT = float(ac.template_snr(TDSignal(tof, td_set).transform(fd_set, window=win))[0])
    optL = float(ac.template_snr(TDSignal(leg, td_set).transform(fd_set, window=win))[0])
    f = np.fft.rfftfreq(N_WIN, d=DT)
    A = np.fft.rfft(tof[0] * win); B = np.fft.rfft(leg[0] * win)
    # true lag via FFT cross-correlation (any offset), then |O| there
    xc = np.fft.irfft(A * np.conj(B), n=N_WIN)
    k = int(np.argmax(np.abs(xc))); best_tau = (k if k <= N_WIN // 2 else k - N_WIN) * DT
    nrm = np.sqrt((np.abs(A) ** 2).sum() * (np.abs(B) ** 2).sum())
    O = abs(np.sum(np.conj(B) * A * np.exp(2j * np.pi * f * best_tau))) / nrm
    ampr = np.sqrt(np.mean(tof[0, inside]**2)) / np.sqrt(np.mean(leg[0, inside]**2))
    print(f"  opt/data: TOF={optT/np.sqrt(dd):.3f} legacy={optL/np.sqrt(dd):.3f}  "
          f"|O|(TOF,leg)={O:.4f}  RMS ratio={ampr:.3f}  best tau={best_tau:+.0f}s", flush=True)

    # align legacy to TOF by best tau for the TD overlay
    leg_al = np.fft.irfft(np.fft.rfft(leg[0]) * np.exp(2j * np.pi * f * best_tau), n=N_WIN)
    f0 = f[np.argmax(np.abs(B) * ((f > 1e-4) & (f < 1e-2)))]
    nper = int(12 / f0 / DT); c0 = int(0.5 * N_WIN); sl = slice(c0, c0 + nper)
    tt = (np.arange(N_WIN)[sl] - c0) * DT
    sel = (f >= max(1e-4, f0 - 3e-3)) & (f <= f0 + 3e-3)

    fig, ax = plt.subplots(3, 1, figsize=(11, 12))
    ax[0].loglog(f[sel], np.abs(A[sel]) * DT, label=f"EMRITDIonFly (opt/data {optT/np.sqrt(dd):.2f})", lw=1.4)
    ax[0].loglog(f[sel], np.abs(B[sel]) * DT, "--", label=f"legacy (opt/data {optL/np.sqrt(dd):.2f})", lw=1.2)
    ax[0].axvline(f0, color="k", ls=":", alpha=.4); ax[0].set_title(f"FD |X| single mode {MODE[0]}  (amp ratio {ampr:.2f})")
    ax[0].set_xlabel("f [Hz]"); ax[0].legend()
    ax[1].plot(tt, tof[0, sl], label="EMRITDIonFly", lw=1.4)
    ax[1].plot(tt, leg_al[sl], "--", label="legacy (aligned)", lw=1.2)
    ax[1].set_title("TD X  (raw amplitudes)"); ax[1].set_xlabel("s"); ax[1].legend()
    ax[2].plot(tt, tof[0, sl] / ampr, label="EMRITDIonFly / amp-ratio", lw=1.4)
    ax[2].plot(tt, leg_al[sl], "--", label="legacy (aligned)", lw=1.2)
    ax[2].set_title(f"TD X  amplitude-normalized  ->  |O|={O:.4f}"); ax[2].set_xlabel("s"); ax[2].legend()
    fig.suptitle(f"EMRI single mode {MODE[0]}: TDIonFly vs legacy  (f0={f0*1e3:.3f} mHz)")
    fig.tight_layout(rect=[0, 0, 1, 0.98])
    out_png = "/tmp/emri_single_mode.png"; fig.savefig(out_png, dpi=110); plt.close(fig)
    print(f"  saved {out_png}", flush=True)


if __name__ == "__main__":
    main()
