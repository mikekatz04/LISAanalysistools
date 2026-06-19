"""4-month EMRI SPIN-angle sweep (ecliptic frame, sky fixed at injection).

Sky (qS,phiS) FIXED at the ecliptic injection (polar angle, NOT latitude). Fine sweep
over the spin angles (qK,phiK) -> noise-weighted time-max FD overlap vs the mojito EMRI
data, 4-month Tobs, ecliptic orbit, legacy response, all other params at injection.
Then re-evaluate the injection-spin AND the best-spin with MORE modes
(mode_selection_threshold THRESH -> THRESH2=1e-5) to see if the match improves.

8 GB box -> watch RSS; the sweep + plot are saved BEFORE the heavier 1e-5 checks so a
memory kill there does not lose the sweep.
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
N_WIN = int(os.environ.get("N_WIN", str(2 ** 19))); TOBS_S = N_WIN * DT   # 2^19 -> ~3.99 months
N_GRID = int(os.environ.get("N_GRID", "15"))
ORDER = int(os.environ.get("ORDER", "25"))
T_BUFFER = float(os.environ.get("T_BUFFER", "18000.0"))
THRESH = float(os.environ.get("THRESH", "1e-3"))     # sweep modes (fewer -> fast/low-mem)
THRESH2 = float(os.environ.get("THRESH2", "1e-5"))   # "more modes" improvement check
WD_GB = float(os.environ.get("WD_GB", "6.2"))
PNG = "/tmp/emri_spin_sweep.png"


def rss_gb():
    return resource.getrusage(resource.RUSAGE_SELF).ru_maxrss / 1e9


def wd():
    while True:
        if rss_gb() > WD_GB:
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


def build_wl(orb, data_t0, thresh):
    fg = GenerateEMRIWaveform("FastKerrEccentricEquatorialFlux", return_list=False,
        inspiral_kwargs={"DENSE_STEPPING": 0, "max_init_len": int(1e4), "force_backend": "cpu"},
        sum_kwargs={"pad_output": True},
        mode_selector_kwargs={"mode_selection_threshold": thresh},
        frame="detector", force_backend="cpu")
    return ResponseWrapper(fg, orbits=orb, t0=data_t0, Tobs=TOBS_S / YRSID_SI, dt=DT,
        index_lambda=8, index_beta=7, flip_hx=True,
        tdi=TDIConfig("2nd generation", force_backend="cpu"), tdi_chan="XYZ",
        order=ORDER, remove_garbage="zero", t_buffer=T_BUFFER, force_backend="cpu")


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

    # ECLIPTIC injection: sky FIXED here (polar angle qS_e, NOT latitude); spin is swept
    lam_S, beta_S = icrs_to_ecliptic(float(ra), float(dec))
    qS_e, phiS_e = float(np.pi / 2 - beta_S), float(lam_S) % (2 * np.pi)
    lam_K, beta_K = icrs_to_ecliptic(float(phiK) % (2 * np.pi), float(np.pi / 2 - qK))
    qK_e, phiK_e = float(np.pi / 2 - beta_K), float(lam_K) % (2 * np.pi)

    fp = find_file(EMRI_L1, "EMRI", SRC)
    ts = MojitoL1File(fp).tdis.time_sampling
    data_t0 = float(ts.t0); deci = int(round(DT / ts.dt))
    with h5py.File(fp, "r") as f:
        lf = float(f.attrs["laser_frequency"])
        dXYZ = np.stack([np.asarray(f["tdis"][c][: N_WIN * deci])[::deci][:N_WIN] / lf
                         for c in ("X2", "Y2", "Z2")])

    n_buf = int(T_BUFFER / DT) + 4 * ORDER + 50
    win = np.zeros(N_WIN); win[n_buf:N_WIN - n_buf] = tukey(N_WIN - 2 * n_buf, 0.05)
    td_set = TDSettings(N_WIN, DT, t0=0.0, force_backend="cpu")
    fd_set = FDSettings(N=N_WIN // 2 + 1, df=1.0 / (N_WIN * DT),
                        min_freq=1e-4, max_freq=1e-2, force_backend="cpu")
    data_fd = TDSignal(dXYZ, td_set).transform(fd_set, window=win)
    ac = AnalysisContainer(data_fd, XYZ2SensitivityMatrix(fd_set, model="scirdv1"))
    data_snr = np.sqrt(float(ac.inner_product().real))
    fb = np.asarray(fd_set.f_arr); taus = np.linspace(-2000.0, 2000.0, 81)
    print(f"  N_WIN={N_WIN} (Tobs={TOBS_S/86400:.1f} d ~{TOBS_S/86400/30.44:.2f} mo)  "
          f"N_GRID={N_GRID}  data SNR={data_snr:.2f}  usable={N_WIN-2*n_buf}/{N_WIN}  "
          f"RSS={rss_gb():.2f}GB", flush=True)
    print(f"  sky FIXED (ecliptic): qS={qS_e:.4f} phiS={phiS_e:.4f}  "
          f"spin inj: qK={qK_e:.4f} phiK={phiK_e:.4f}", flush=True)

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

    def make_leg(wl, qk, pk):
        inj = INTR + [qS_e, phiS_e, qk, pk] + PHASES
        try:
            leg = np.atleast_2d(np.asarray(wl(*inj, convert_to_ra_dec=False)))[:3]
        except Exception as e:
            print(f"    wf fail ({qk:.2f},{pk:.2f}): {type(e).__name__}: {str(e)[:60]}", flush=True)
            return np.zeros((3, N_WIN))
        if leg.shape[-1] < N_WIN:
            leg = np.pad(leg, ((0, 0), (0, N_WIN - leg.shape[-1])))
        return leg[:, :N_WIN]

    print(f"\n  building ecliptic orbit + sweep generator (THRESH={THRESH:g})...", flush=True)
    orb = make_orbit(fp, "ecliptic", data_t0)
    wl = build_wl(orb, data_t0, THRESH)

    t0 = time.time(); inj_leg = make_leg(wl, qK_e, phiK_e)
    inj_ov = overlap_max(inj_leg)
    print(f"  injection-spin overlap (THRESH)={inj_ov:.4f}   [{time.time()-t0:.1f}s/eval, "
          f"RSS={rss_gb():.2f}GB]", flush=True)

    qg = np.linspace(0.06, np.pi - 0.06, N_GRID)
    pg = np.linspace(0.0, 2 * np.pi, N_GRID)
    spin = np.zeros((N_GRID, N_GRID))
    t_start = time.time()
    for i, qk in enumerate(qg):
        for j, pk in enumerate(pg):
            spin[i, j] = overlap_max(make_leg(wl, qk, pk))
        gc.collect()
        print(f"  qK={qk:.3f}: max_row={spin[i].max():.4f}  "
              f"[{time.time()-t_start:.0f}s, RSS={rss_gb():.2f}GB]", flush=True)
    ki, kj = np.unravel_index(np.argmax(spin), spin.shape)
    qk_best, pk_best = float(qg[ki]), float(pg[kj])
    print(f"\n  SPIN sweep max |O|={spin.max():.4f} @ qK={qk_best:.4f} phiK={pk_best:.4f}", flush=True)
    print(f"  injection spin   |O|={inj_ov:.4f} @ qK={qK_e:.4f} phiK={phiK_e:.4f}", flush=True)
    print(f"  d(qK)={np.degrees(qk_best-qK_e):+.1f}deg  d(phiK)="
          f"{np.degrees((pk_best-phiK_e+np.pi)%(2*np.pi)-np.pi):+.1f}deg", flush=True)

    # ---- plot + save BEFORE the heavier 1e-5 checks ----
    fig, ax = plt.subplots(figsize=(8.5, 6.5))
    P, Q = np.meshgrid(pg, qg)
    pc = ax.pcolormesh(P, Q, spin, shading="auto", cmap="viridis")
    ax.plot([phiK_e], [qK_e], "*", ms=22, mfc="red", mec="white", mew=1.4, label="injection")
    ax.plot([pk_best], [qk_best], "X", ms=15, mfc="none", mec="white", mew=2.5, label="grid max")
    ax.set_xlabel("phiK (spin azimuth)"); ax.set_ylabel("qK (spin polar)")
    ax.set_title(f"EMRI legacy resp | SPIN sweep, sky@inj, Tobs={TOBS_S/86400/30.44:.2f}mo, "
                 f"THRESH={THRESH:g}\ninj |O|={inj_ov:.4f}  grid max={spin.max():.4f}", fontsize=11)
    ax.legend(loc="upper right"); fig.colorbar(pc, ax=ax, label="time-max noise-wtd |O|")
    fig.tight_layout(); fig.savefig(PNG, dpi=120)
    print(f"  saved {PNG}", flush=True)
    np.savez("/tmp/emri_spin_sweep.npz", spin=spin, qg=qg, pg=pg,
             inj=(qS_e, phiS_e, qK_e, phiK_e), inj_ov=inj_ov, best=(qk_best, pk_best))

    # ---- MORE MODES (THRESH2=1e-5) at injection-spin AND best-spin ----
    print(f"\n  --- more modes (THRESH2={THRESH2:g}) ---  RSS={rss_gb():.2f}GB", flush=True)
    del wl; gc.collect()
    wl2 = build_wl(orb, data_t0, THRESH2)
    t0 = time.time(); ov_inj2 = overlap_max(make_leg(wl2, qK_e, phiK_e))
    print(f"  injection-spin: |O|({THRESH:g})={inj_ov:.4f} -> |O|({THRESH2:g})={ov_inj2:.4f}  "
          f"(d={ov_inj2-inj_ov:+.4f})  [{time.time()-t0:.1f}s, RSS={rss_gb():.2f}GB]", flush=True)
    ov_best2 = overlap_max(make_leg(wl2, qk_best, pk_best))
    print(f"  best-spin     : |O|({THRESH:g})={spin.max():.4f} -> |O|({THRESH2:g})={ov_best2:.4f}  "
          f"(d={ov_best2-spin.max():+.4f})", flush=True)


if __name__ == "__main__":
    main()
