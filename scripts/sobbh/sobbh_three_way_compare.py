"""Three-way comparison of the SOBBH templates against mojito SOBHB data.

  D = mojito data (cached window, /tmp/sobbh_mojito_data_src{SRC}.npz)
  A = legacy pyResponse  : ResponseWrapper(SOBBHWaveform)  (Lagrange interp)
  B = TDI-on-the-fly      : SOBBHTDIonFly(SOBBHWaveform)    (analytic delays)

A and B share the identical 3.5PN core (waveform_generate_h_plus_cross vs
waveform_generate_amp_phase) -- only the response differs -- so A-vs-B isolates
the projection, exactly like the MBH dig.

Legacy convention is the validated mojito-matching one (2026-06-14):
  flip_hx=False  +  reference_time = MOJITO_REFERENCE_TIME.
SOBBHTDIonFly reproduces flip_hx=False naturally (its get_hp_hc feed gives
(hp_rot, +hx_rot)) and shares the same reference_time, so B should track A.

For each pair (A-D, B-D, A-B) and band, reports:
  1-Re(O)  : NO phase/time max  (tests injected phase/time)
  1-|O|    : phase-maximized
  arg(O)   : residual phase [deg]
  tau*     : best time shift [s]
Plus WDM mm5 / mm2 narrowband, and the 3-panel plot.
"""
import os, sys, gc, time, threading, resource
os.environ.setdefault("OMP_NUM_THREADS", "8")
import numpy as np
import matplotlib; matplotlib.use("Agg"); import matplotlib.pyplot as plt

MEM_CAP_GB = float(os.environ.get("SOBBH_MEM_CAP_GB", "7.5")); _IS_MAC = sys.platform == "darwin"
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
from lisatools.response.directresponse import ResponseWrapper
from lisatools.response.tdiconfig import TDIConfig
from lisatools.sources.sobbh.waveform import SOBBHWaveform
from lisatools.domains import TDSettings, FDSettings, WDMSettings, TDSignal
from lisatools.analysiscontainer import AnalysisContainer
from lisatools.sensitivity import XYZ2SensitivityMatrix
from lisatools.utils.constants import YRSID_SI
from bbhx.sobbhtdionfly import SOBBHTDIonFly

REF = 97729089.327664
PATH = "/Users/mkatz/.mojito_cache/brickmarket/mojito_light_v1_0_0/"
SOBHB_L1 = os.path.join(PATH, "data", "SOBHB", "L1")
BACKEND = "cpu"; SENS_MODEL = os.environ.get("SOBBH_SENS", "scirdv1")
DT = 10.0; TDI_GEN = "2nd generation"; NCH = 3
NF, NT = 512, 512; N_WIN = NF * NT; TOBS = N_WIN * DT
F_MIN, F_MAX = 0.010, 0.030
ORDER = 40; T_BUFFER = 3.0e4; TUKEY_ALPHA = 0.1
N_GRID = int(os.environ.get("SOBBH_NGRID", "2048"))
SRC = int(os.environ.get("SOBBH_SRC", "0"))
DATA_CACHE = f"/tmp/sobbh_mojito_data_src{SRC}.npz"


def banner(s): print("\n" + "=" * 80 + f"\n {s}\n" + "=" * 80, flush=True)
def tukey(N, a):
    from scipy.signal.windows import tukey as _t
    return _t(N, a)


def main():
    threading.Thread(target=_wd, daemon=True).start()
    banner(f"SOBBH THREE-WAY: data D / legacy A / on-the-fly B  (src {SRC}, N_grid={N_GRID})")
    if not os.path.exists(DATA_CACHE):
        raise SystemExit(f"missing {DATA_CACHE}; run sobbh_mojito_match_debug.py first to cache it.")
    z = np.load(DATA_CACHE, allow_pickle=True)
    D = np.asarray(z["data_td"])[:NCH]; data_t0 = float(z["data_t0"]); cat = z["cat"].item()
    g = lambda k: float(cat[k])
    params_full = np.array([
        g("PrimaryMassSSBFrame"), g("SecondaryMassSSBFrame"),
        g("PrimarySpinCompZ"), g("SecondarySpinCompZ"),
        g("LuminosityDistance") / 1e3, g("InclinationAngle"),
        g("GW22FrequencySSBFrame"),
        g("RightAscension") % (2 * np.pi), g("Declination"),
        g("PolarisationAngle") % np.pi, g("TrueAnomaly"),
    ])
    m1, m2, s1, s2, dist, inc, f0, ra, dec, psi, phi0 = params_full
    print(f"  N_WIN={N_WIN} Tobs={TOBS/86400:.1f}d  f0={f0*1e3:.4f} mHz  inc={inc:.4f}  "
          f"data_t0-REF={data_t0-REF:.2f}s", flush=True)

    orb = L1Orbits(find_file(SOBHB_L1, "SOBHB", SRC), force_backend=BACKEND, frame="icrs")
    pad = 1.0e5; lo = max(data_t0 - pad, float(orb.sc_t0)); hi = min(data_t0 + TOBS + pad, float(orb._sc_t_base[-1]))
    lt = np.asarray(orb.ltt_t); mk = (lt >= lo) & (lt <= hi)
    orb.ltt = np.asarray(orb.ltt)[mk].copy(); orb.ltt_t = lt[mk].copy(); orb.ltt_t0 = float(orb.ltt_t[0])
    del lt; gc.collect(); orb.configure(linear_interp_setup=True); mark("orbit ready")

    # --- A: legacy pyResponse (validated mojito convention) ----------------
    sobbh_gen = SOBBHWaveform(Tobs=TOBS, dt=DT, t0=data_t0, reference_time=REF, force_backend=BACKEND)
    legacy = ResponseWrapper(
        sobbh_gen, orbits=orb, t0=data_t0, Tobs=TOBS / YRSID_SI, dt=DT,
        index_lambda=7, index_beta=8, flip_hx=False, tdi=TDIConfig(TDI_GEN, force_backend=BACKEND),
        tdi_chan="XYZ", order=ORDER, remove_garbage="zero", is_ecliptic_latitude=True,
        t_buffer=T_BUFFER, force_backend=BACKEND)
    A = np.atleast_2d(np.asarray(legacy(*params_full, convert_to_ra_dec=False)))[:NCH]
    A = (np.pad(A, ((0, 0), (0, N_WIN - A.shape[-1]))) if A.shape[-1] < N_WIN else A[:, :N_WIN])
    mark("A built")

    # --- B: TDI-on-the-fly -------------------------------------------------
    fly = SOBBHTDIonFly(
        SOBBHWaveform(Tobs=TOBS, dt=DT, t0=data_t0, reference_time=REF, force_backend=BACKEND),
        orb, TDIConfig(TDI_GEN, force_backend=BACKEND), DT, TOBS, t0=data_t0,
        n_grid=N_GRID, force_backend=BACKEND)
    grid_t = np.arange(N_WIN) * DT + data_t0
    B = np.asarray(fly(m1, m2, s1, s2, dist, f0, phi0, inc, ra, dec, psi,
                       upsample_t_arr=grid_t, combine=True))[:NCH]
    mark("B built")
    print(f"  |X|max  D={np.max(np.abs(D[0])):.3e}  A={np.max(np.abs(A[0])):.3e}  "
          f"B={np.max(np.abs(B[0])):.3e}  finite B={np.isfinite(B).all()}", flush=True)

    win = tukey(N_WIN, TUKEY_ALPHA); td = TDSettings(N_WIN, DT, t0=0.0, force_backend=BACKEND)

    def fd_of(arr, lo, hi):
        fd = FDSettings(N=N_WIN // 2 + 1, df=1.0 / (N_WIN * DT), min_freq=lo, max_freq=hi, force_backend=BACKEND)
        return fd, TDSignal(arr, td).transform(fd, window=win)

    def compare(ref, tmpl, lo, hi):
        fd, rsig = fd_of(ref, lo, hi); _, tsig = fd_of(tmpl, lo, hi)
        ac = AnalysisContainer(rsig, XYZ2SensitivityMatrix(fd, model=SENS_MODEL))
        O = ac.template_inner_product(tsig, normalize=True, complex=True)
        f = np.asarray(fd.f_arr); t0 = np.asarray(tsig.arr).copy()
        def mmt(tau):
            tsig.arr[:] = t0 * np.exp(-2j * np.pi * f * tau)[None, :]
            return 1 - abs(ac.template_inner_product(tsig, normalize=True, complex=True))
        period = 1.0 / f0
        bc = min(((mmt(x), x) for x in np.linspace(-1.5 * period, 1.5 * period, 241)))
        bf = min(((mmt(x), x) for x in bc[1] + np.linspace(-1.0, 1.0, 201)))
        tsig.arr[:] = t0
        return 1 - O.real, 1 - abs(O), np.degrees(np.angle(O)), bf[1], bf[0]

    bands = [("full[10,30]mHz", F_MIN, F_MAX),
             ("carrier +-2mHz", max(F_MIN, f0 - 2e-3), min(F_MAX, f0 + 2e-3)),
             ("carrier +-0.5mHz", max(F_MIN, f0 - 5e-4), min(F_MAX, f0 + 5e-4)),
             ("below f0", F_MIN, f0), ("above f0", f0, F_MAX)]
    for pair, ref, tmpl in [("A=legacy vs D", D, A), ("B=on-fly vs D", D, B), ("A vs B (response only)", A, B)]:
        banner(pair + "   [1-Re(O)=NOmax | 1-|O|=pmax | arg | tau* | mm@tau*]")
        for tag, l, h in bands:
            r, a, ph, ts, mt = compare(ref, tmpl, l, h)
            print(f"  {tag:18s} 1-Re(O)={r:+.4e}  1-|O|={a:.4e}  arg={ph:+7.2f}deg  "
                  f"tau*={ts:+.3f}s  mm@tau*={mt:.4e}", flush=True)

    # --- WDM mm5 / mm2 narrowband -----------------------------------------
    banner("WDM narrowband mm5 / mm2  (phase-max |O| and no-max Re(O))")
    ws = WDMSettings(NF, NT, DT, t0=data_t0, min_freq=F_MIN, max_freq=F_MAX, force_backend=BACKEND)
    ldf = ws.layer_df; m_floor = int(f0 / ldf)
    wdm_bands = [("full", F_MIN, F_MAX),
                 ("mm5", f0 - 3 * ldf, f0 + 2 * ldf),
                 ("mm2", (m_floor - 0.5) * ldf, (m_floor + 1.5) * ldf)]
    def wdm_mm(ref, tmpl, lo, hi):
        wsb = WDMSettings(NF, NT, DT, t0=data_t0, min_freq=lo, max_freq=hi, force_backend=BACKEND)
        d = TDSignal(ref, td).transform(wsb, window=win); t = TDSignal(tmpl, td).transform(wsb, window=win)
        ac = AnalysisContainer(d, XYZ2SensitivityMatrix(wsb, model=SENS_MODEL))
        O = ac.template_inner_product(t, normalize=True, complex=True)
        return 1 - O.real, 1 - abs(O)
    for pair, ref, tmpl in [("A vs D", D, A), ("B vs D", D, B), ("A vs B", A, B)]:
        for tag, l, h in wdm_bands:
            r, a = wdm_mm(ref, tmpl, l, h)
            print(f"  {pair:8s} {tag:5s}  1-Re(O)={r:+.4e}  mm(pmax)={a:.4e}", flush=True)

    # ---- plots ----
    f = np.fft.rfftfreq(N_WIN, DT)
    FD = np.fft.rfft(D[0] * win) * DT; FA = np.fft.rfft(A[0] * win) * DT; FB = np.fft.rfft(B[0] * win) * DT
    sel = (f >= F_MIN) & (f <= F_MAX)
    fig, ax = plt.subplots(3, 1, figsize=(11, 13))
    ax[0].loglog(f[sel], np.abs(FD[sel]), label="mojito data", lw=1.0)
    ax[0].loglog(f[sel], np.abs(FA[sel]), "--", label="legacy pyResponse", lw=0.9, alpha=.8)
    ax[0].loglog(f[sel], np.abs(FB[sel]), ":", label="TDI-on-the-fly", lw=1.2, alpha=.8)
    ax[0].axvline(f0, color="k", ls=":", alpha=.4)
    ax[0].set_xlim(F_MIN, F_MAX); ax[0].legend(); ax[0].set_title("FD |X|"); ax[0].set_xlabel("f [Hz]")
    n0 = int(0.3 * N_WIN); sl = slice(n0, n0 + int(8 / f0 / DT))
    tt = np.arange(N_WIN)[sl] * DT / 3600.0
    ax[1].plot(tt, D[0][sl], label="mojito data", lw=1.4)
    ax[1].plot(tt, A[0][sl], "--", label="legacy pyResponse", lw=1.0)
    ax[1].plot(tt, B[0][sl], ":", label="TDI-on-the-fly", lw=1.0)
    ax[1].legend(); ax[1].set_title("TD X (zoom ~8 periods)"); ax[1].set_xlabel("hours")
    ax[2].loglog(f[sel], np.abs((FA - FD)[sel]), label="|legacy - data|", lw=0.9)
    ax[2].loglog(f[sel], np.abs((FB - FD)[sel]), label="|on-fly - data|", lw=0.9)
    ax[2].loglog(f[sel], np.abs((FA - FB)[sel]), ":", color="purple", label="|legacy - on-fly|", lw=1.0)
    ax[2].axvline(f0, color="k", ls=":", alpha=.4)
    ax[2].set_xlim(F_MIN, F_MAX); ax[2].legend(); ax[2].set_title("FD |X| residuals"); ax[2].set_xlabel("f [Hz]")
    fig.suptitle(f"SOBHB src={SRC}: data vs legacy vs TDI-on-the-fly (f0={f0*1e3:.3f} mHz, inc={inc:.3f})")
    fig.tight_layout(rect=[0, 0, 1, 0.985])
    out = f"/tmp/sobbh_three_way_src{SRC}.png"; fig.savefig(out, dpi=110); plt.close(fig)
    print(f"\nDONE.  plot -> {out}", flush=True)


if __name__ == "__main__":
    main()
