"""Frame-consistency test via an angle sweep with the LEGACY response (NOT TDI-on-fly).

Hold all EMRI params at the mojito catalogue injection EXCEPT the sky+spin angles
(qS, phiS, qK, phiK). Sweep them and compute the noise-weighted, time-maximized FD
overlap with the mojito EMRI data. Done for orbits in BOTH frames:
  - ICRS   : feed the sky/spin angles in ICRS (qS=pi/2-dec, phiS=ra; qK,phiK raw cat)
  - ECLIPTIC: feed the ecliptic-converted angles (the known data-reproducing config)

If the frame handling is self-consistent the overlap MAX sits AT the injection point;
a frame bug shows up as the peak drifting off the injection. Two 2D planes per frame
(sky qS-phiS with spin@inj; spin qK-phiK with sky@inj) -> 4 heatmaps. Small Tobs.
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
N_WIN = int(os.environ.get("N_WIN", "8192")); TOBS_S = N_WIN * DT   # ~1.9 day
N_GRID = int(os.environ.get("N_GRID", "9"))
ORDER = int(os.environ.get("ORDER", "25"))
# t_buffer is zeroed off BOTH ends by remove_garbage="zero". The response also enforces a
# HARD FLOOR: t_buffer/dt - 2*(int(100*fs)+4*order) >= int(|x_sc|*C_inv)+4*order. With
# |x_sc|~1AU -> |x|*C_inv~500, fs=1/dt=0.05, order=25 -> t_buffer >= ~16200s. Use 18000s.
T_BUFFER = float(os.environ.get("T_BUFFER", "18000.0"))
PNG = "/tmp/emri_angle_sweep.png"


def wd():
    while True:
        if resource.getrusage(resource.RUSAGE_SELF).ru_maxrss / 1e9 > 6.5:
            os._exit(42)
        time.sleep(0.3)


def make_orbit(fp, frame, data_t0):
    orb = L1Orbits(fp, force_backend="cpu", frame=frame)
    pad = 1e5; lo = max(REF - pad, float(orb.sc_t0))
    hi = min(data_t0 + TOBS_S + pad, float(orb._sc_t_base[-1]))
    lt = np.asarray(orb.ltt_t); m = (lt >= lo) & (lt <= hi)
    orb.ltt = np.asarray(orb.ltt)[m].copy(); orb.ltt_t = lt[m].copy()
    orb.ltt_t0 = float(orb.ltt_t[0]); gc.collect()
    orb.configure(linear_interp_setup=True)
    return orb


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

    # injection sky/spin in each frame (same physical config, two representations)
    qS_i, phiS_i = float(np.pi / 2 - dec), float(ra)                       # ICRS
    qK_i, phiK_i = float(qK), float(phiK)
    lam_S, beta_S = icrs_to_ecliptic(float(ra), float(dec))
    qS_e, phiS_e = float(np.pi / 2 - beta_S), float(lam_S) % (2 * np.pi)   # ECLIPTIC
    lam_K, beta_K = icrs_to_ecliptic(float(phiK) % (2 * np.pi), float(np.pi / 2 - qK))
    qK_e, phiK_e = float(np.pi / 2 - beta_K), float(lam_K) % (2 * np.pi)

    fp = find_file(EMRI_L1, "EMRI", SRC)
    ts = MojitoL1File(fp).tdis.time_sampling
    data_t0 = float(ts.t0); deci = int(round(DT / ts.dt))
    with h5py.File(fp, "r") as f:
        lf = float(f.attrs["laser_frequency"])
        dXYZ = np.stack([np.asarray(f["tdis"][c][: N_WIN * deci])[::deci][:N_WIN] / lf
                         for c in ("X2", "Y2", "Z2")])

    # analysis window: ZERO over the response's buffer-zeroed edges (first/last t_buffer),
    # tukey over the usable middle -> data AND template share the same support so the
    # overlap isn't artificially capped by the template's zeroed edges.
    n_buf = int(T_BUFFER / DT) + 4 * ORDER + 50
    win = np.zeros(N_WIN); win[n_buf:N_WIN - n_buf] = tukey(N_WIN - 2 * n_buf, 0.2)
    print(f"  t_buffer={T_BUFFER:.0f}s n_buf={n_buf}  usable={N_WIN-2*n_buf}/{N_WIN} pts "
          f"({(N_WIN-2*n_buf)/N_WIN:.2f})", flush=True)
    td_set = TDSettings(N_WIN, DT, t0=0.0, force_backend="cpu")
    fd_set = FDSettings(N=N_WIN // 2 + 1, df=1.0 / (N_WIN * DT),
                        min_freq=1e-4, max_freq=1e-2, force_backend="cpu")
    data_fd = TDSignal(dXYZ, td_set).transform(fd_set, window=win)
    ac = AnalysisContainer(data_fd, XYZ2SensitivityMatrix(fd_set, model="scirdv1"))
    data_snr = np.sqrt(float(ac.inner_product().real))
    fb = np.asarray(fd_set.f_arr); taus = np.linspace(-1500.0, 1500.0, 41)
    print(f"  N_WIN={N_WIN} (Tobs={TOBS_S/86400:.2f} d)  N_GRID={N_GRID}  data SNR={data_snr:.3f}", flush=True)

    def overlap_max(leg):
        try:
            tmpl = TDSignal(leg, td_set).transform(fd_set, window=win)
            base = np.asarray(tmpl.arr).copy()
            opt, _ = ac.template_snr(tmpl)
            if not np.isfinite(opt) or opt <= 0:
                return 0.0
            best = 0.0
            for t in taus:
                tmpl.arr[:] = base * np.exp(2j * np.pi * fb * t)[None, :]
                _, det = ac.template_snr(tmpl)
                if np.isfinite(det) and det > best:
                    best = float(det)
            return best / data_snr
        except Exception:
            return 0.0

    def make_leg(wl, qs, ps, qk, pk):
        inj = INTR + [qs, ps, qk, pk] + PHASES
        try:
            leg = np.atleast_2d(np.asarray(wl(*inj, convert_to_ra_dec=False)))[:3]
        except Exception as e:
            print(f"    waveform fail at ({qs:.2f},{ps:.2f},{qk:.2f},{pk:.2f}): "
                  f"{type(e).__name__}", flush=True)
            return np.zeros((3, N_WIN))
        if leg.shape[-1] < N_WIN:
            leg = np.pad(leg, ((0, 0), (0, N_WIN - leg.shape[-1])))
        return leg[:, :N_WIN]

    qg = np.linspace(0.08, np.pi - 0.08, N_GRID)
    pg = np.linspace(0.0, 2 * np.pi, N_GRID)

    results = {}
    for fname, frame, (qS0, phiS0, qK0, phiK0) in [
        ("ICRS", "icrs", (qS_i, phiS_i, qK_i, phiK_i)),
        ("ECLIPTIC", "ecliptic", (qS_e, phiS_e, qK_e, phiK_e)),
    ]:
        print(f"\n=== {fname} orbit; injection (qS,phiS,qK,phiK)="
              f"({qS0:.3f},{phiS0:.3f},{qK0:.3f},{phiK0:.3f}) ===", flush=True)
        orb = make_orbit(fp, frame, data_t0)
        fg = GenerateEMRIWaveform("FastKerrEccentricEquatorialFlux", return_list=False,
            inspiral_kwargs={"DENSE_STEPPING": 0, "max_init_len": int(1e4), "force_backend": "cpu"},
            sum_kwargs={"pad_output": True}, frame="detector", force_backend="cpu")
        wl = ResponseWrapper(fg, orbits=orb, t0=data_t0, Tobs=TOBS_S / YRSID_SI, dt=DT,
            index_lambda=8, index_beta=7, flip_hx=True,
            tdi=TDIConfig("2nd generation", force_backend="cpu"), tdi_chan="XYZ",
            order=ORDER, remove_garbage="zero", t_buffer=T_BUFFER, force_backend="cpu")

        inj_ov = overlap_max(make_leg(wl, qS0, phiS0, qK0, phiK0))
        print(f"  overlap AT injection = {inj_ov:.4f}", flush=True)

        sky = np.zeros((N_GRID, N_GRID))
        for i, qs in enumerate(qg):
            for j, ps in enumerate(pg):
                sky[i, j] = overlap_max(make_leg(wl, qs, ps, qK0, phiK0))
            print(f"  sky  qS={qs:.2f}: max_row={sky[i].max():.3f}", flush=True)
        spin = np.zeros((N_GRID, N_GRID))
        for i, qk in enumerate(qg):
            for j, pk in enumerate(pg):
                spin[i, j] = overlap_max(make_leg(wl, qS0, phiS0, qk, pk))
            print(f"  spin qK={qk:.2f}: max_row={spin[i].max():.3f}", flush=True)

        si, sj = np.unravel_index(np.argmax(sky), sky.shape)
        ki, kj = np.unravel_index(np.argmax(spin), spin.shape)
        print(f"  SKY  max |O|={sky.max():.4f} @ qS={qg[si]:.3f} phiS={pg[sj]:.3f} "
              f"(inj qS={qS0:.3f} phiS={phiS0:.3f})", flush=True)
        print(f"  SPIN max |O|={spin.max():.4f} @ qK={qg[ki]:.3f} phiK={pg[kj]:.3f} "
              f"(inj qK={qK0:.3f} phiK={phiK0:.3f})", flush=True)
        results[fname] = dict(sky=sky, spin=spin, inj=(qS0, phiS0, qK0, phiK0), inj_ov=inj_ov)
        del orb, wl, fg; gc.collect()

    # ---- plot ----
    fig, axes = plt.subplots(2, 2, figsize=(13, 11))
    P, Q = np.meshgrid(pg, qg)
    for r, fname in enumerate(("ICRS", "ECLIPTIC")):
        R = results[fname]; qS0, phiS0, qK0, phiK0 = R["inj"]
        for c, (plane, q0, p0, ql) in enumerate([
            ("sky", qS0, phiS0, "qS / phiS"), ("spin", qK0, phiK0, "qK / phiK")]):
            ax = axes[r, c]; Z = R[plane]
            pc = ax.pcolormesh(P, Q, Z, shading="auto", cmap="viridis")
            ax.plot([p0], [q0], "*", ms=20, mfc="red", mec="white", mew=1.2, label="injection")
            mi, mj = np.unravel_index(np.argmax(Z), Z.shape)
            ax.plot([pg[mj]], [qg[mi]], "X", ms=13, mfc="none", mec="white", mew=2, label="max")
            ax.set_title(f"{fname} | {plane}: inj |O|={R['inj_ov']:.3f}  "
                         f"grid max={Z.max():.3f}", fontsize=11)
            ax.set_xlabel(f"phi ({ql.split('/')[1].strip()})")
            ax.set_ylabel(f"theta ({ql.split('/')[0].strip()})")
            ax.legend(loc="upper right", fontsize=8)
            fig.colorbar(pc, ax=ax, label="time-max noise-wtd |O|")
    fig.suptitle(f"EMRI legacy-response overlap vs sky/spin angles (Tobs={TOBS_S/86400:.2f} d, "
                 f"all other params @ injection)", fontsize=12)
    fig.tight_layout()
    fig.savefig(PNG, dpi=120)
    print(f"\n  saved {PNG}", flush=True)


if __name__ == "__main__":
    main()
