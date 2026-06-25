"""EMRI MIXED-FRAME check vs mojito (~1 month Tobs).

User-specified configuration:
  * FEW intrinsic waveform gets:  ECLIPTIC sky (qS_e POLAR angle, phiS_e azimuth) + FILE
    spin (qK, phiK raw from catalogue) -> produces h+/hx.
  * Response ONLY: orbits in the "icrs" frame, sky entered as raw catalogue (ra, dec),
    projecting the h+/hx from above.

Mechanics (see directresponse.py:875-905): ResponseWrapper reads the RESPONSE sky from
args[index_beta/lambda] (index_beta is a POLAR angle -> beta=pi/2-args[index_beta]) BEFORE
calling the waveform gen, which receives the SAME args. So we pass the ICRS sky in the
param vector (args[7]=pi/2-dec, args[8]=ra) for the response, and a thin wrapper rewrites
args[7]->qS_e, args[8]->phiS_e for FEW (keeping file spin args[9,10]). Reference: the
known data-reproducing pure-ecliptic config (ecliptic orbit + ecliptic sky + ecliptic spin).
"""
import os, time, threading, resource, gc
os.environ.setdefault("OMP_NUM_THREADS", "8")
import numpy as np, h5py
import matplotlib; matplotlib.use("Agg")
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
from few.waveform import GenerateEMRIWaveform

PATH = "/Users/mkatz/.mojito_cache/brickmarket/mojito_light_v1_0_0/"
EMRI_L1 = os.path.join(PATH, "data", "EMRI", "L1")
REF = 97729089.327664; SRC = 1
DT = 20.0
N_WIN = int(os.environ.get("N_WIN", str(2 ** 17))); TOBS_S = N_WIN * DT   # 2^17 -> ~30.3 d
ORDER = int(os.environ.get("ORDER", "25"))
T_BUFFER = float(os.environ.get("T_BUFFER", "18000.0"))
THRESH = float(os.environ.get("THRESH", "1e-5"))
PNG = "/tmp/emri_mixed_frame_check.png"


def wd():
    while True:
        if resource.getrusage(resource.RUSAGE_SELF).ru_maxrss / 1e9 > 6.0:
            os._exit(42)
        time.sleep(0.3)


class OverrideSky:
    """Make FEW use a FIXED (qS,phiS) regardless of the sky in the param vector (which
    carries the RESPONSE sky). Keeps spin args[9,10] and everything else untouched."""
    def __init__(self, fg, qS, phiS):
        self.fg = fg; self.qS = qS; self.phiS = phiS
    def __call__(self, *args, **kwargs):
        args = list(args); args[7] = self.qS; args[8] = self.phiS
        return self.fg(*args, **kwargs)
    def __getattr__(self, n):
        return getattr(self.fg, n)


def make_orbit(fp, frame, data_t0):
    orb = L1Orbits(fp, force_backend="cpu", frame=frame)
    pad = 1e5; lo = max(REF - pad, float(orb.sc_t0))
    hi = min(data_t0 + TOBS_S + pad, float(orb._sc_t_base[-1]))
    lt = np.asarray(orb.ltt_t); m = (lt >= lo) & (lt <= hi)
    orb.ltt = np.asarray(orb.ltt)[m].copy(); orb.ltt_t = lt[m].copy()
    orb.ltt_t0 = float(orb.ltt_t[0]); gc.collect()
    orb.configure(linear_interp_setup=True)
    return orb


def make_fg():
    return GenerateEMRIWaveform("FastKerrEccentricEquatorialFlux", return_list=False,
        inspiral_kwargs={"DENSE_STEPPING": 0, "max_init_len": int(1e4), "force_backend": "cpu"},
        sum_kwargs={"pad_output": True},
        mode_selector_kwargs={"mode_selection_threshold": THRESH},
        frame="detector", force_backend="cpu")


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
    INTR = [M, mu, a, p0, e0, 1.0, dist]; PHASES = [Pp, Pt, Pr]

    # ecliptic sky (POLAR angle, not latitude); ecliptic spin (for the reference only)
    lam_S, beta_S = icrs_to_ecliptic(float(ra), float(dec))
    qS_e, phiS_e = float(np.pi / 2 - beta_S), float(lam_S) % (2 * np.pi)
    lam_K, beta_K = icrs_to_ecliptic(float(phiK) % (2 * np.pi), float(np.pi / 2 - qK))
    qK_e, phiK_e = float(np.pi / 2 - beta_K), float(lam_K) % (2 * np.pi)
    print(f"  sky: ra={ra:.4f} dec={dec:.4f} | ecliptic qS_e(polar)={qS_e:.4f} phiS_e={phiS_e:.4f}", flush=True)
    print(f"  spin file(raw): qK={qK:.4f} phiK={phiK:.4f} | ecliptic qK_e={qK_e:.4f} phiK_e={phiK_e:.4f}", flush=True)

    fp = find_file(EMRI_L1, "EMRI", SRC)
    ts = MojitoL1File(fp).tdis.time_sampling
    data_t0 = float(ts.t0); deci = int(round(DT / ts.dt))
    with h5py.File(fp, "r") as f:
        lf = float(f.attrs["laser_frequency"])
        dXYZ = np.stack([np.asarray(f["tdis"][c][: N_WIN * deci])[::deci][:N_WIN] / lf
                         for c in ("X2", "Y2", "Z2")])

    # The intrinsic FEW h+/hx are referenced at the CATALOGUE epoch REF, so t0_arr = REF for
    # ALL of the response (the waveform's first sample sits at REF). The data grid is anchored
    # at data_t0 = REF + off, which splits into an INTEGER-sample part (offset_int, removed by
    # slicing the response output) and a sub-sample remainder T0_SHIFT = t0_shift_to_data
    # (NON-zero; handled inside the response). Response output sample (offset_int + k) lands on
    # data sample k: REF + T0_SHIFT + (offset_int+k)*DT = data_t0 + k*DT.
    off = data_t0 - REF
    offset_int = int(round(off / DT))
    T0_SHIFT = off - offset_int * DT            # sub-sample data-grid alignment, |.| < DT
    N_RESP = N_WIN + offset_int
    print(f"  REF={REF:.6f}  data_t0-REF={off:.3f}s -> offset_int={offset_int} samp + "
          f"t0_shift_to_data={T0_SHIFT:+.4f}s (NON-zero)", flush=True)

    n_buf = int(T_BUFFER / DT) + 4 * ORDER + 50
    win = np.zeros(N_WIN); win[n_buf:N_WIN - n_buf] = tukey(N_WIN - 2 * n_buf, 0.05)
    td_set = TDSettings(N_WIN, DT, t0=0.0, force_backend="cpu")
    fd_set = FDSettings(N=N_WIN // 2 + 1, df=1.0 / (N_WIN * DT),
                        min_freq=1e-4, max_freq=1e-2, force_backend="cpu")
    data_fd = TDSignal(dXYZ, td_set).transform(fd_set, window=win)
    ac = AnalysisContainer(data_fd, XYZ2SensitivityMatrix(fd_set, model="scirdv1"))
    data_snr = np.sqrt(float(ac.inner_product().real))
    fb = np.asarray(fd_set.f_arr); taus = np.linspace(-2000.0, 2000.0, 81)
    print(f"  N_WIN={N_WIN} (Tobs={TOBS_S/86400:.1f} d)  data SNR={data_snr:.3f}", flush=True)

    def metrics(leg):
        """noise-weighted overlaps. O0 = FIXED-alignment (tau=0) overlap -- this is the one
        the t0_shift_to_data fix moves, since a sub-sample misalignment is a pure time shift
        that the tau-MAX overlap (O) would otherwise absorb."""
        tmpl = TDSignal(leg, td_set).transform(fd_set, window=win)
        base = np.asarray(tmpl.arr).copy()
        opt, det0 = ac.template_snr(tmpl)        # det0 at tau=0 (true alignment)
        O0 = det0 / data_snr
        best_det, best_tau = -1.0, 0.0
        for t in taus:
            tmpl.arr[:] = base * np.exp(2j * np.pi * fb * t)[None, :]
            _, det = ac.template_snr(tmpl)
            if np.isfinite(det) and det > best_det:
                best_det, best_tau = float(det), float(t)
        O = best_det / data_snr
        # per-channel time-max |O| (unwhitened, for the plot context)
        Och = []
        ff = np.fft.rfftfreq(N_WIN, d=DT); ts2 = np.linspace(-2000, 2000, 201)
        for ci in range(3):
            A = np.fft.rfft(leg[ci] * win); B = np.fft.rfft(dXYZ[ci] * win)
            nm = np.sqrt((np.abs(A) ** 2).sum() * (np.abs(B) ** 2).sum()) + 1e-300
            Och.append(max(abs(np.sum(np.conj(B) * A * np.exp(2j * np.pi * ff * t))) / nm for t in ts2))
        return dict(O=O, O0=O0, mm=1 - O, mm0=1 - O0, opt=float(opt), tau=best_tau,
                    optdata=float(opt) / data_snr, Och=Och)

    def gen(wl, vec):
        # response output is on grid REF + T0_SHIFT + i*DT (length ~N_RESP); slice off the
        # integer-sample offset so output[offset_int + k] lands on data sample k.
        leg = np.atleast_2d(np.asarray(wl(*vec, convert_to_ra_dec=False)))[:3]
        leg = leg[:, offset_int:offset_int + N_WIN]
        return leg if leg.shape[-1] == N_WIN else np.pad(leg, ((0, 0), (0, N_WIN - leg.shape[-1])))

    # ---------- MIXED config: FEW ecliptic-sky+file-spin ; response ICRS orbit + (ra,dec) ----------
    print("\n  [MIXED] building ICRS orbit + FEW(ecliptic sky, file spin)...", flush=True)
    orb_icrs = make_orbit(fp, "icrs", data_t0)
    wl_mix = ResponseWrapper(OverrideSky(make_fg(), qS_e, phiS_e), orbits=orb_icrs, t0=REF,
        t0_shift_to_data=T0_SHIFT, Tobs=N_RESP * DT / YRSID_SI, dt=DT, index_lambda=8, index_beta=7,
        flip_hx=True, tdi=TDIConfig("2nd generation", force_backend="cpu"), tdi_chan="XYZ",
        order=ORDER, remove_garbage="zero", t_buffer=T_BUFFER, force_backend="cpu")
    # param vector: response reads args[7]=pi/2-dec (->beta=dec), args[8]=ra; FEW spin=args[9,10]=file
    vec_mix = INTR + [float(np.pi / 2 - dec), float(ra), float(qK), float(phiK)] + PHASES
    t0 = time.time(); leg_mix = gen(wl_mix, vec_mix); mmix = metrics(leg_mix)
    print(f"  [MIXED] mm0(tau=0)={mmix['mm0']:.4e} |O0|={mmix['O0']:.4f} | mm(tau-max)={mmix['mm']:.4e} "
          f"|O|={mmix['O']:.4f}  opt/data={mmix['optdata']:.3f}  tau={mmix['tau']:+.0f}s  "
          f"|O|ch={[f'{x:.3f}' for x in mmix['Och']]}  [{time.time()-t0:.1f}s]", flush=True)
    del wl_mix, orb_icrs; gc.collect()

    # ---------- REFERENCE: pure ecliptic (data-reproducing) ----------
    print("\n  [REF ecliptic] building ecliptic orbit + pure-ecliptic config...", flush=True)
    orb_ecl = make_orbit(fp, "ecliptic", data_t0)
    wl_ref = ResponseWrapper(make_fg(), orbits=orb_ecl, t0=REF, t0_shift_to_data=T0_SHIFT,
        Tobs=N_RESP * DT / YRSID_SI, dt=DT, index_lambda=8, index_beta=7, flip_hx=True,
        tdi=TDIConfig("2nd generation", force_backend="cpu"),
        tdi_chan="XYZ", order=ORDER, remove_garbage="zero", t_buffer=T_BUFFER, force_backend="cpu")
    vec_ref = INTR + [qS_e, phiS_e, qK_e, phiK_e] + PHASES
    leg_ref = gen(wl_ref, vec_ref); mref = metrics(leg_ref)
    print(f"  [REF ecliptic] mm0(tau=0)={mref['mm0']:.4e} |O0|={mref['O0']:.4f} | mm(tau-max)={mref['mm']:.4e} "
          f"|O|={mref['O']:.4f}  opt/data={mref['optdata']:.3f}  tau={mref['tau']:+.0f}s  "
          f"|O|ch={[f'{x:.3f}' for x in mref['Och']]}", flush=True)
    del wl_ref, orb_ecl; gc.collect()

    # ---------------- plot MIXED template vs mojito data ----------------
    sh = int(round(mmix['tau'] / DT))
    leg_plot = np.roll(leg_mix, sh, axis=1)        # time-align to data for the overlay
    tg = (data_t0 + np.arange(N_WIN) * DT - data_t0) / 86400.0
    ff = np.fft.rfftfreq(N_WIN, d=DT)
    mid = N_WIN // 2; w = slice(mid - 1500, mid + 1500)   # ~60 ks TD zoom
    fig, axes = plt.subplots(2, 3, figsize=(16, 8))
    for ci, ch in enumerate("XYZ"):
        A = np.abs(np.fft.rfft(leg_plot[ci] * win)); B = np.abs(np.fft.rfft(dXYZ[ci] * win))
        band = (ff > 5e-4) & (ff < 6e-3)
        axes[0, ci].loglog(ff[band], B[band], "k-", lw=0.9, label="mojito data")
        axes[0, ci].loglog(ff[band], A[band], "r-", lw=0.9, alpha=0.8, label="mixed template")
        axes[0, ci].set_title(f"{ch}  FD |.|   |O|_ch={mmix['Och'][ci]:.3f}"); axes[0, ci].set_xlabel("f [Hz]")
        axes[0, ci].legend(fontsize=8)
        axes[1, ci].plot(tg[w], dXYZ[ci][w], "k-", lw=0.8, label="mojito data")
        axes[1, ci].plot(tg[w], leg_plot[ci][w], "r-", lw=0.8, alpha=0.8, label="mixed template")
        axes[1, ci].set_title(f"{ch}  TD (zoom, tau-aligned)"); axes[1, ci].set_xlabel("t [day]")
        axes[1, ci].legend(fontsize=8)
    fig.suptitle(f"EMRI MIXED frame (FEW ecliptic-sky+file-spin -> ICRS-orbit response @ ra/dec) "
                 f"vs mojito, Tobs={TOBS_S/86400:.1f}d\n"
                 f"MIXED: mismatch={mmix['mm']:.3e} |O|={mmix['O']:.4f} opt/data={mmix['optdata']:.3f}   |   "
                 f"REF ecliptic: mismatch={mref['mm']:.3e} |O|={mref['O']:.4f} opt/data={mref['optdata']:.3f}",
                 fontsize=11)
    fig.tight_layout(); fig.savefig(PNG, dpi=120)
    print(f"\n  saved {PNG}", flush=True)


if __name__ == "__main__":
    main()
