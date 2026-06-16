"""Debug: match the LAT EMRI (FEW) model against Mojito EMRI source 1.

Same approach as sobbh_mojito_match_debug.py, EMRI flavour. Cached so the
25M read + L1 orbit build + each FEW template run happen ONCE
(/tmp/emri_mojito_*).

EMRI is broadband (many modes) + chirps faster than the SOBBH, so we use a
SHORT 0.5-month window from the data start (early inspiral, far from the
day-347 plunge) and full-band FD/WDM overlaps (no narrowband mm5/mm2).

Template path = canonical get_emri_response_wrapper:
  GenerateEMRIWaveform("FastKerrEccentricEquatorialFlux", frame="detector")
  + ResponseWrapper(index_lambda=8, index_beta=7, flip_hx=..., order=40,
    remove_garbage="zero")  with the DATA's L1Orbits (frame="icrs").

FEW 14-param basis (full_year_combined_global_fit_settings.py:973):
  [M, mu, a, p0, e0, xI0=cos(Incl), dist(Gpc),
   qS, phiS, qK, phiK, Phi_phi0, Phi_theta0, Phi_r0]

CONVENTIONS UNDER TEST (sky/spin in ICRS per direction):
  * qS: catalogue code passes Declination directly, but ResponseWrapper
    (is_ecliptic_latitude=False) does beta=pi/2-param[7] AND FEW treats
    param[7] as a POLAR angle -> correct qS = pi/2 - Dec.   [qS bug]
  * flip_hx: get_emri_response_wrapper hardcodes True (same suspect as SOBBH).
  * absolute phase / IC epoch: catalogue phases + p0/e0 are at
    MOJITO_REFERENCE_TIME; the waveform starts at data_t0 (=REF+850.5s) ->
    residual phase (FEW has no reference_time hook).
"""
import os
import sys
import gc
import time
import threading
import resource
os.environ.setdefault("OMP_NUM_THREADS", "8")
import numpy as np

# ---- memory watchdog: abort the process before it can swap the machine ----
MEM_CAP_GB = 6.0
_IS_MAC = sys.platform == "darwin"


def rss_gb():
    r = resource.getrusage(resource.RUSAGE_SELF).ru_maxrss
    return r / 1e9 if _IS_MAC else r / 1e6   # mac: bytes, linux: KB


def _watchdog():
    while True:
        if rss_gb() > MEM_CAP_GB:
            sys.stderr.write(f"\n[WATCHDOG] RSS {rss_gb():.2f}GB > {MEM_CAP_GB}GB -> abort\n")
            sys.stderr.flush()
            os._exit(42)
        time.sleep(0.3)


def mark(msg):
    print(f"[RSS {rss_gb():5.2f} GB] {msg}", flush=True)
import matplotlib
matplotlib.use("Agg")
import matplotlib.pyplot as plt

from lisatools.detector import L1Orbits
from lisatools.globalfit.preprocessing import L1ProcessingStep, find_file
from lisatools.response.directresponse import ResponseWrapper
from lisatools.response.tdiconfig import TDIConfig
from lisatools.domains import TDSettings, FDSettings, WDMSettings, TDSignal
from lisatools.analysiscontainer import AnalysisContainer
from lisatools.sensitivity import XYZ2SensitivityMatrix
from lisatools.utils.constants import YRSID_SI
from few.waveform import GenerateEMRIWaveform
from lisatools.sources.utils import icrs_to_ecliptic


class _FewEclipticSky:
    """Feed FEW's INTRINSIC waveform the ecliptic-polar sky (qS, phiS) while the
    LISA response still reads the ICRS sky from index_beta/index_lambda.
    Optionally also rotate the spin direction (qK, phiK) to ecliptic."""

    def __init__(self, few, qS_ecl, phiS_ecl, qK_ecl=None, phiK_ecl=None):
        self._few = few
        self.qS_ecl = float(qS_ecl)
        self.phiS_ecl = float(phiS_ecl)
        self.qK_ecl = None if qK_ecl is None else float(qK_ecl)
        self.phiK_ecl = None if phiK_ecl is None else float(phiK_ecl)

    def __call__(self, *args, **kwargs):
        a = list(args)
        a[7] = self.qS_ecl     # ecliptic polar angle (colatitude)
        a[8] = self.phiS_ecl   # ecliptic longitude
        if self.qK_ecl is not None:
            a[9] = self.qK_ecl     # spin polar (ecliptic)
            a[10] = self.phiK_ecl  # spin azimuth (ecliptic)
        return self._few(*a, **kwargs)

MOJITO_REFERENCE_TIME = 97729089.327664
PATH = "/Users/mkatz/.mojito_cache/brickmarket/mojito_light_v1_0_0/"
EMRI_L1 = os.path.join(PATH, "data", "EMRI", "L1")
BACKEND = "cpu"
SENS_MODEL = "scirdv1"

# dt=20 (Nyquist 25 mHz >> ~4.5 mHz signal) keeps FEW mode arrays small.
DT = 20.0
NF, NT = 256, 256
N_WIN = NF * NT          # 65536, Tobs ~ 15.16 d (~0.5 month)
TOBS = N_WIN * DT
TDI_GEN = "2nd generation"
F_MIN, F_MAX = 1e-3, 2e-2

EMRI_INSPIRAL_KWARGS = {"DENSE_STEPPING": 0, "max_init_len": int(1e4), "force_backend": BACKEND}
EMRI_SUM_KWARGS = {"pad_output": True}
# mode_selection_threshold env-overridable (EMRI_MODE_THRESH); lower => more modes.
EMRI_MODE_SELECTOR_KWARGS = {
    "mode_selection_threshold": float(os.environ.get("EMRI_MODE_THRESH", "1e-2"))
}

DATA_CACHE = "/tmp/emri_mojito_data.npz"
TMPL_CACHE = "/tmp/emri_mojito_tmpl_{tag}.npy"
_ORBITS = [None]
_FEW = [None]


def banner(s):
    print("\n" + "=" * 78 + f"\n {s}\n" + "=" * 78, flush=True)


def tukey(N, alpha):
    from scipy.signal.windows import tukey as _t
    return _t(N, alpha)


def load_data_cached():
    """Read the EMRI source via L1ProcessingStep (data + catalogue), then keep
    only the decimated 0.5-month window.

    NB: the orbit is NOT taken from the loader -- it is rebuilt and ltt-sliced
    in get_orbits() so ResponseWrapper's orbit deepcopy doesn't blow up RAM on
    the full-mission (1.2 GB) ltt arrays.  L1ProcessingStep does build a
    full-mission orbit internally during the read (~2 GB transient, GC'd before
    get_orbits runs), well under the 6 GB watchdog."""
    if os.path.exists(DATA_CACHE):
        z = np.load(DATA_CACHE, allow_pickle=True)
        if tuple(z["data_td"].shape) == (3, N_WIN):
            print(f"[cache] data window from {DATA_CACHE}", flush=True)
            return z["data_td"], float(z["data_t0"]), z["cat"].item()
        print(f"[cache] STALE ({z['data_td'].shape} != (3,{N_WIN})) -> re-read", flush=True)
    print("[cache] MISS -> reading mojito EMRI via L1ProcessingStep "
          "(one time)...", flush=True)
    loader = L1ProcessingStep(
        L1_folder=PATH, source_types=["emri"], source_ids={"emri": 1},
        orbits_class=L1Orbits,
        orbits_kwargs=dict(force_backend=BACKEND, frame="icrs"),
        verbose=True,
    )
    times = np.asarray(loader.times)
    data_full = np.asarray(loader.data)
    dt_native = float(loader.dt)
    data_t0 = float(times[0])
    cat = {k: float(np.asarray(v)) for k, v in loader.catalogue["EMRI"][1].items()
           if np.asarray(v).dtype.kind in "fi"}
    deci = int(round(DT / dt_native))
    data_td = data_full[:, : N_WIN * deci : deci][:, :N_WIN].copy()   # (3, N_WIN)
    del data_full
    np.savez(DATA_CACHE, data_td=data_td, data_t0=data_t0, cat=cat)
    print(f"[cache] wrote {DATA_CACHE}  shape={data_td.shape}", flush=True)
    gc.collect()
    return data_td, data_t0, cat


def get_orbits(t_start):
    """Configure the L1 orbit over ONLY the analysis window (~30K points)
    instead of the full 731-day mission (1.26M points) -- the full-mission
    linear_interp_setup transient is ~3.2 GB and was the OOM cause."""
    if _ORBITS[0] is None:
        print("[orbits] building L1Orbits over window...", flush=True)
        _frame = "ecliptic" if os.environ.get("EMRI_ALL_ECL") else "icrs"
        orb = L1Orbits(find_file(EMRI_L1, "EMRI", 1), force_backend=BACKEND, frame=_frame)
        pad = 1.0e5   # covers t_buffer (3e4) + light-travel delays
        lo = max(t_start - pad, float(orb.sc_t0))
        hi = min(t_start + TOBS + pad, float(orb._sc_t_base[-1]))

        # --- SLICE the native ltt arrays to the window (NO spline) ---
        # ltts are native fine-spaced (2.5s, 25M pts, ~1.2 GB); ResponseWrapper
        # deepcopies the orbit x2, so the full ltts blow up RAM. The C++ indexes
        # ltts purely by (ltt_t0, ltt_dt) so a contiguous truncation preserves
        # the uniform spacing exactly -- no interpolation needed.
        ltt_t = np.asarray(orb.ltt_t)
        m = (ltt_t >= lo) & (ltt_t <= hi)
        orb.ltt = np.asarray(orb.ltt)[m].copy()
        orb.ltt_t = ltt_t[m].copy()
        orb.ltt_t0 = float(orb.ltt_t[0])
        del ltt_t
        gc.collect()

        # positions: full-mission linear_interp_setup (the only finite path;
        # the window t_arr path yields a NaN response even with sc_t0 fixed).
        # ltt slice above is the real memory win; positions are ~0.36 GB.
        orb.configure(linear_interp_setup=True)
        _ORBITS[0] = orb
        print(f"[orbits] ready (ltt sliced {int(m.sum())} pts; positions full-mission)",
              flush=True)
    return _ORBITS[0]


def few_gen():
    if _FEW[0] is None:
        _FEW[0] = GenerateEMRIWaveform(
            "FastKerrEccentricEquatorialFlux", return_list=False,
            inspiral_kwargs=EMRI_INSPIRAL_KWARGS, sum_kwargs=EMRI_SUM_KWARGS,
            frame="detector", mode_selector_kwargs=EMRI_MODE_SELECTOR_KWARGS,
            force_backend=BACKEND,
        )
    return _FEW[0]


def build_template(cat, t_start, *, qS_mode, flip_hx, nchan=3):
    # env overrides probe the parameter mismatch via the lisatools overlap
    # (m1 sets the frequency range; default = catalogue SSB mass).
    M = float(os.environ.get("EMRI_M", cat["PrimaryMassSSBFrame"]))
    p0 = float(os.environ.get("EMRI_P0", cat["SemiLatusRectum"]))
    e0 = float(os.environ.get("EMRI_E0", cat["Eccentricity"]))
    mu = cat["SecondaryMassSSBFrame"]
    a = cat["PrimarySpinParameter"]; xI0 = np.cos(cat["InclinationAngle"])
    dist = cat["LuminosityDistance"] / 1e3
    dec = cat["Declination"]; ra = cat["RightAscension"] % (2 * np.pi)
    qS = (np.pi / 2 - dec) if qS_mode == "colat" else dec
    qK = cat["PolarAnglePrimarySpin"]; phiK = cat["AzimuthalAnglePrimarySpin"]
    phiS = ra
    # ALL-ECLIPTIC (the validated frame fix): FEW assumes ecliptic, so convert
    # the catalogue ICRS sky AND spin -> ecliptic; the response then reads the
    # ecliptic sky from params and get_orbits() loads the ecliptic orbit. NO
    # frame mixing (unlike EMRI_ECL_SKY which kept ICRS response/orbit).
    if os.environ.get("EMRI_ALL_ECL"):
        lam_ecl, beta_ecl = icrs_to_ecliptic(float(ra), float(dec))
        qS = np.pi / 2.0 - float(beta_ecl); phiS = float(lam_ecl) % (2 * np.pi)
        lamK, betaK = icrs_to_ecliptic(float(phiK) % (2 * np.pi), np.pi / 2.0 - float(qK))
        qK = np.pi / 2.0 - float(betaK); phiK = float(lamK) % (2 * np.pi)
    params = [M, mu, a, p0, e0, xI0, dist, qS, phiS, qK, phiK,
              cat["AzimuthalPhase"], cat["PolarPhase"], cat["RadialPhase"]]

    _LBL = ["M", "mu", "a", "p0", "e0", "xI0", "dist(Gpc)", "qS", "phiS",
            "qK", "phiK", "Phi_phi0", "Phi_theta0", "Phi_r0"]
    print("    [FEW params] " + ", ".join(f"{l}={v:.6g}" for l, v in zip(_LBL, params)),
          flush=True)

    tdi_config = TDIConfig(TDI_GEN, force_backend=BACKEND)
    fg = few_gen(); mark("few_gen ready")
    # DEFAULT: ICRS everywhere -- FEW gets the same ICRS-derived sky
    # (qS=pi/2-Dec, phiS=RA) the response uses. Set EMRI_ECL_SKY=1 to instead
    # feed FEW the ecliptic-polar sky (intrinsic-frame) while the response
    # keeps ICRS.
    if os.environ.get("EMRI_ALL_ECL"):
        print(f"    [sky] ALL-ECLIPTIC: FEW+response+orbit ecliptic; qS={qS:.4f} "
              f"phiS={phiS:.4f} qK={qK:.4f} phiK={phiK:.4f}", flush=True)
    elif os.environ.get("EMRI_ECL_SKY"):
        lam_ecl, beta_ecl = icrs_to_ecliptic(float(ra), float(dec))
        qS_ecl = np.pi / 2.0 - float(beta_ecl)
        phiS_ecl = float(lam_ecl) % (2 * np.pi)
        qK_ecl = phiK_ecl = None
        if os.environ.get("EMRI_QK_ECL"):
            lamK, betaK = icrs_to_ecliptic(float(phiK), np.pi / 2.0 - float(qK))
            qK_ecl = np.pi / 2.0 - float(betaK)
            phiK_ecl = float(lamK) % (2 * np.pi)
        print(f"    [sky] FEW ecliptic qS={qS_ecl:.4f}, phiS={phiS_ecl:.4f}; "
              f"response ICRS RA/Dec", flush=True)
        fg = _FewEclipticSky(fg, qS_ecl, phiS_ecl, qK_ecl, phiK_ecl)
    else:
        print(f"    [sky] ICRS everywhere: FEW + response use qS={qS:.4f}, "
              f"phiS={ra:.4f} (=pi/2-Dec, RA)", flush=True)
    orb = get_orbits(t_start); mark("orbits ready")
    # FEW's initial conditions (p0, e0, Phi_phi0/theta0/r0) are defined at the
    # catalogue REFERENCE time, not the data-window start. So START the response
    # at MOJITO_REFERENCE_TIME and slice it forward onto the data window
    # (data_t0 = REF + 850.5 s ~ 1.5 orbital cycles of phase/orbit evolution).
    n_off = int(round((t_start - MOJITO_REFERENCE_TIME) / DT))
    N_gen = N_WIN + n_off
    wave_gen = ResponseWrapper(
        fg, orbits=orb, t0=MOJITO_REFERENCE_TIME,
        Tobs=(N_gen * DT) / YRSID_SI, dt=DT,
        index_lambda=8, index_beta=7, flip_hx=flip_hx,
        tdi=tdi_config, tdi_chan="XYZ", order=40,
        remove_garbage="zero", t_buffer=3e4, force_backend=BACKEND,
    )
    mark(f"ResponseWrapper built (t0=REF, n_off={n_off}); generating "
         f"qS={qS_mode} flip={flip_hx}...")
    arr = np.atleast_2d(np.asarray(wave_gen(*params)))[:nchan]
    mark("template generated")
    # slice the REF-anchored response onto the data window [data_t0, +TOBS]
    if arr.shape[-1] < n_off + N_WIN:
        arr = np.pad(arr, ((0, 0), (0, n_off + N_WIN - arr.shape[-1])))
    arr = arr[:, n_off:n_off + N_WIN]
    # n_off is rounded, so the sliced grid (REF + n_off*DT) lands `resid` seconds
    # off the true data start; correct that sub-sample offset with an FD phase
    # ramp so the broadband EMRI harmonics align exactly (resid up to DT/2 ~ 10s).
    resid = (t_start - MOJITO_REFERENCE_TIME) - n_off * DT
    if abs(resid) > 1e-9:
        f = np.fft.rfftfreq(N_WIN, d=DT)
        A = np.fft.rfft(arr, axis=-1) * np.exp(2j * np.pi * f * resid)[None, :]
        arr = np.fft.irfft(A, n=N_WIN, axis=-1)
    return arr


def get_template_cached(tag, cat, t_start, *, qS_mode, flip_hx):
    p = TMPL_CACHE.format(tag=tag)
    if os.path.exists(p):
        print(f"[cache] template '{tag}' from {p}", flush=True)
        return np.load(p)
    arr = build_template(cat, t_start, qS_mode=qS_mode, flip_hx=flip_hx)
    if not os.environ.get("EMRI_NO_CACHE"):
        try:
            np.save(p, arr)
            print(f"[cache] wrote template '{tag}'", flush=True)
        except OSError as e:
            print(f"[cache] skip write ({e})", flush=True)
    return arr


def fd_setup(data_td, win):
    td_set = TDSettings(N_WIN, DT, t0=0.0, force_backend=BACKEND)
    fd_set = FDSettings(N=N_WIN // 2 + 1, df=1.0 / (N_WIN * DT),
                        min_freq=F_MIN, max_freq=F_MAX, force_backend=BACKEND)
    data_fd = TDSignal(data_td, td_set).transform(fd_set, window=win)
    sens = XYZ2SensitivityMatrix(fd_set, model=SENS_MODEL)
    return AnalysisContainer(data_fd, sens), td_set, fd_set


def fd_signal(td, td_set, fd_set, win):
    return TDSignal(td, td_set).transform(fd_set, window=win)


def report(ac, sig, label):
    O = ac.template_inner_product(sig, normalize=True, complex=True)
    opt, det = ac.template_snr(sig)
    dd = ac.inner_product()
    print(f"  [{label:22s}] |O|={abs(O):.5f}  Re(O)={O.real:+.5f}  "
          f"ph={np.degrees(np.angle(O)):+7.1f}  mm(pmax)={1-abs(O):.4e}  "
          f"SNR d={np.sqrt(dd.real):.2f} h={opt:.2f} det={det:+.2f}", flush=True)
    return O


def time_scan(ac, base_sig, fd_set, taus):
    f = np.asarray(fd_set.f_arr)
    base = np.asarray(base_sig.arr).copy()
    best = (-1.0, 0.0)
    for tau in taus:
        base_sig.arr[:] = base * np.exp(-2j * np.pi * f * tau)[None, :]
        O = ac.template_inner_product(base_sig, normalize=True, complex=True)
        if abs(O) > best[0]:
            best = (abs(O), tau)
    base_sig.arr[:] = base
    return best


def main():
    threading.Thread(target=_watchdog, daemon=True).start()
    mark("main start")
    banner("STAGE 1: load (cached) + anchors")
    data_td, data_t0, cat = load_data_cached()
    print(f"  data_t0={data_t0:.4f}  REF={MOJITO_REFERENCE_TIME:.4f}  "
          f"offset={data_t0-MOJITO_REFERENCE_TIME:.3f}s")
    for k in ["PrimaryMassSSBFrame", "SecondaryMassSSBFrame", "PrimarySpinParameter",
              "SemiLatusRectum", "Eccentricity", "InclinationAngle", "LuminosityDistance",
              "Declination", "RightAscension", "PolarAnglePrimarySpin",
              "AzimuthalAnglePrimarySpin", "AzimuthalPhase", "PolarPhase", "RadialPhase",
              "TimeCoalescenceSSBFrame", "EstimatedSNR"]:
        print(f"    {k:28s}= {cat[k]:.6g}")

    # data spectral content
    xf = np.abs(np.fft.rfft(data_td[0] * np.hanning(N_WIN)))
    ff = np.fft.rfftfreq(N_WIN, d=DT)
    bb = (ff > F_MIN) & (ff < F_MAX)
    top = ff[bb][np.argsort(xf[bb])[-6:][::-1]]
    print(f"  data top-6 freqs in band [Hz]: {np.array2string(np.sort(top), precision=5)}")

    win = tukey(N_WIN, alpha=0.1)
    ac, td_set, fd_set = fd_setup(data_td, win)

    banner("STAGE 2: sky-convention test (flip_hx=True; FEW builds h+ - i hx)")
    # flip_hx=True is correct for FEW per construction; the open question is the
    # qS colatitude bug: catalogue code passes Dec, but ResponseWrapper
    # (is_ecliptic_latitude=False) + FEW both want a polar angle = pi/2 - Dec.
    grid = [
        ("qS=pi/2-Dec (fix)", "colat", True),
        ("qS=Dec     (current code)", "lat", True),
    ]
    if os.environ.get("EMRI_ONE"):
        grid = grid[:1]   # measured single-variant run first
    taus = np.linspace(-2000.0, 2000.0, 161)   # wide: absorb 850.5s IC offset etc.
    best_variant = None
    for label, qmode, flip in grid:
        tag = f"q{qmode}_f{int(flip)}"
        td = get_template_cached(tag, cat, data_t0, qS_mode=qmode, flip_hx=flip)
        sig = fd_signal(td, td_set, fd_set, win)
        O = report(ac, sig, label)
        b = time_scan(ac, sig, fd_set, taus)
        print(f"      -> after time-scan: |O|max={b[0]:.5f} (tau*={b[1]:+.1f}s) "
              f"mm={1-b[0]:.4e}", flush=True)
        if best_variant is None or b[0] > best_variant[0]:
            best_variant = (b[0], b[1], label, qmode, flip, tag)
        del td, sig
        gc.collect()

    banner("STAGE 3: best variant detail + absolute-phase / IC-epoch")
    _, tau_b, label, qmode, flip, tag = best_variant
    print(f"  BEST: {label}  (|O|max={best_variant[0]:.5f}, tau*={tau_b:+.1f}s)")
    td_b = get_template_cached(tag, cat, data_t0, qS_mode=qmode, flip_hx=flip)
    sig_b = fd_signal(td_b, td_set, fd_set, win)
    O_b = ac.template_inner_product(sig_b, normalize=True, complex=True)
    print(f"  raw (no shift): Re(O)={O_b.real:+.5f}  phase={np.degrees(np.angle(O_b)):+.1f} deg")
    print(f"  -> residual constant phase = phase-reference/IC-epoch (FEW IC at data_t0,"
          f" catalogue IC at REF, {data_t0-MOJITO_REFERENCE_TIME:.1f}s earlier).")

    banner("STAGE 4: WDM full-band overlap (best variant, phase-maximized)")
    wdm_overlap(data_td, td_b, data_t0, win)

    make_plots(data_td, td_b, td_set, fd_set, win)
    print("\nDONE.  plots -> /tmp/emri_mojito_*.png", flush=True)


def wdm_overlap(data_td, tmpl_td, t_start, win):
    td_set = TDSettings(N_WIN, DT, t0=0.0, force_backend=BACKEND)
    ws = WDMSettings(NF, NT, DT, t0=t_start, min_freq=F_MIN, max_freq=F_MAX,
                     force_backend=BACKEND)
    d = TDSignal(data_td, td_set).transform(ws, window=win)
    t = TDSignal(tmpl_td, td_set).transform(ws, window=win)
    ac = AnalysisContainer(d, XYZ2SensitivityMatrix(ws, model=SENS_MODEL))
    O = ac.template_inner_product(t, normalize=True, complex=True)
    opt, det = ac.template_snr(t)
    dd = ac.inner_product()
    print(f"  WDM full-band: |O|={abs(O):.5f}  Re(O)={O.real:+.5f}  "
          f"mm={1-O.real:.4e}  mm(pmax)={1-abs(O):.4e}")
    print(f"  WDM SNR data={np.sqrt(dd.real):.2f} tmpl={opt:.2f} det={det:+.2f}")


def make_plots(data_td, td_b, td_set, fd_set, win):
    """TD-tracking diagnostic: align template to data (best time-shift + phase),
    overlay at START / MIDDLE / END of the window to check phase coherence."""
    from scipy.signal import hilbert
    f = np.asarray(fd_set.f_arr)
    d_arr = np.asarray(fd_signal(data_td, td_set, fd_set, win).arr)
    t_arr = np.asarray(fd_signal(td_b, td_set, fd_set, win).arr)

    # best constant time-shift + phase from the X-channel cross-spectrum
    prod = d_arr[0] * np.conj(t_arr[0])
    taus = np.linspace(-3000, 3000, 601)
    vals = np.array([abs(np.sum(prod * np.exp(2j * np.pi * f * tau))) for tau in taus])
    tau_best = float(taus[int(vals.argmax())])
    dphi = -np.angle(np.sum(prod * np.exp(2j * np.pi * f * tau_best)))
    tau_samp = int(round(tau_best / DT))
    aligned = np.real(hilbert(np.roll(td_b, tau_samp, axis=-1), axis=-1) * np.exp(1j * dphi))

    fig, ax = plt.subplots(2, 2, figsize=(13, 8))
    # FD overview
    ax[0, 0].loglog(f, np.abs(d_arr[0]), label="data X", lw=1.2)
    ax[0, 0].loglog(f, np.abs(t_arr[0]), label="template X", lw=1.0, alpha=.85)
    ax[0, 0].set_xlim(F_MIN, F_MAX); ax[0, 0].legend(); ax[0, 0].set_xlabel("Hz")
    ax[0, 0].set_title("FD |X|")
    # TD tracking at start / middle / end (aligned)
    nshow = 350   # ~7000 s ~ 7 periastron bursts -> per-burst phase visible
    for axn, frac, name in [((0, 1), 0.08, "START"), ((1, 0), 0.5, "MIDDLE"),
                            ((1, 1), 0.90, "END")]:
        n0 = int(frac * N_WIN); sl = slice(n0, n0 + nshow)
        tt = (np.arange(N_WIN)[sl]) * DT
        ax[axn].plot(tt, data_td[0][sl], label="data", lw=1.3)
        ax[axn].plot(tt, aligned[0][sl], "--", label="template (aligned)", lw=1.1)
        ax[axn].set_title(f"TD X {name}"); ax[axn].legend(fontsize=8); ax[axn].set_xlabel("s")
    fig.suptitle(f"EMRI TD tracking (aligned tau*={tau_best:.0f}s, "
                 f"dphi={np.degrees(dphi):.0f}deg) -- ICRS injection")
    fig.tight_layout(); fig.savefig("/tmp/emri_mojito_track.png", dpi=110)
    plt.close(fig)
    print(f"  TD-track plot -> /tmp/emri_mojito_track.png "
          f"(tau*={tau_best:.0f}s, dphi={np.degrees(dphi):.0f}deg)", flush=True)


if __name__ == "__main__":
    main()
