"""Debug: match the LAT MBH (phentax) global-fit model against Mojito MBHB data.

Same approach as ``sobbh_mojito_match_debug.py`` / ``emri_mojito_match_debug.py``
(load mojito source data -> build the canonical template -> FD/WDM overlap +
offset hunting), but the source machinery is swapped for the **MBH global-fit
setup**:

  * Template = the engine-side ``signal_gen`` path
    (``MBH_TRANSFORM.both_transforms`` -> ``PhenomTHMTDIWaveform``), the exact
    generator the full-year combined global fit registers for the MBH branch
    (see ``get_mbh_phenom_multi_erebor_settings`` /
    ``_get_mbh_phenom_wave_gen`` in
    ``global_fit_input/full_year_combined_global_fit_settings.py``).
  * Catalogue -> sampling basis via the stock
    ``lisatools.globalfit.recipe.mbh_catalogue_to_sampling_basis``.
  * Sky/polarization in **ICRS** (orbits loaded with ``frame="icrs"``), matching
    the 2026-06 run-frame directive.

MBHBs are loud, transient mergers (SNR ~ 130..3000 for these sources), so unlike
the near-monochromatic SOBBH or the early-inspiral EMRI window this script:

  1. Locates the merger and carves a window AROUND it (merger placed at
     ``MERGER_FRAC`` of the window) instead of starting from the data start.
  2. Actively MAXIMIZES the overlap: a coarse->fine merger-time (tau) scan plus
     phase maximization, because at SNR ~ 1000 even a 1e-4 mismatch leaves a
     detectable residual (det SNR ~ sqrt(2 * mm) * SNR).

Merger-time convention (the key unknown this script pins down)
-------------------------------------------------------------
The global fit places the absolute merger at ``waveform_t0 + merger_time`` with
``waveform_t0 = MBH_WAVEFORM_T0 = MOJITO_REFERENCE_TIME`` and
``merger_time = t_plunge`` (the sampling-basis slot, fed raw from the catalogue
``TimeCoalescencePhenomTPHMSSBFrame``).  With the data window starting at
``data_t0 ~ REF`` this lands the merger at grid offset ~ ``t_plunge`` seconds into
the file.  The script reports the convention-predicted merger location next to
the empirical (max-power) one, then scans tau to absorb any residual offset.

MBH sampling basis (11) -- ``mbh_catalogue_to_sampling_basis`` order:
  [logM, Q=m1/m2, s1z, s2z, dist(Gpc), phi_ref, cos_iota, psi, ra, sin_dec, t_plunge]
MBH waveform basis (11) -- after ``MBH_TRANSFORM`` (PhenomTHMTDIWaveform call order):
  [m1, m2, s1z, s2z, dist(Mpc), phi_ref, inc, psi, ra, dec, t_plunge]

Heavily cached so the slow steps run ONCE:
  * the windowed mojito read     -> /tmp/mbh_mojito_data_id{ID}.npz
  * each PhenomTHM/response template -> /tmp/mbh_mojito_tmpl_{tag}.npy

PREPARED, NOT YET RUN.  The MBHB L1 source streams are not downloaded on this
machine (only the catalogue is); the first run downloads the chosen source.
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
# This box has 8.6 GB RAM; the L1ProcessingStep full read + full-mission orbit
# build is the peak.  The orbit is ltt-sliced in get_orbits() (mirrors the EMRI
# guard) so ResponseWrapper's x2 deepcopy stays cheap.
MEM_CAP_GB = float(os.environ.get("MBH_MEM_CAP_GB", "6.0"))
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
from lisatools.globalfit.recipe import mbh_catalogue_to_sampling_basis
from lisatools.globalfit.stock.erebor import make_mbh_transform_container
from lisatools.response.tdiconfig import TDIConfig
from lisatools.sources.bbh.waveform import PhenomTHMTDIWaveform
from lisatools.domains import (
    TDSettings, FDSettings, WDMSettings, TDSignal, place_td_signal_on_grid,
)
from lisatools.analysiscontainer import AnalysisContainer
from lisatools.sensitivity import XYZ2SensitivityMatrix
from lisatools.utils.constants import YRSID_SI

# ----------------------------------------------------------------------------
# Anchors / config (mirrors full_year_combined_global_fit_settings.py)
# ----------------------------------------------------------------------------
MOJITO_REFERENCE_TIME = 97729089.327664
MBH_WAVEFORM_T0 = MOJITO_REFERENCE_TIME          # waveform_t0 in mojito mode
PATH = "/Users/mkatz/.mojito_cache/brickmarket/mojito_light_v1_0_0/"
MBHB_L1 = os.path.join(PATH, "data", "MBHB", "L1")
BACKEND = "cpu"
SENS_MODEL = "scirdv1"

# Source to match (EstimatedSNR in the catalogue).  Start with id 0 (SNR ~245,
# m1~4.6e5 + m2~1.05e5, f22~1.3e-4, merger ~day 300 of the file).  Other loud
# options: 17 -> 2969,  10 -> 1963,  19 -> 1925,  7 -> 1367,  13 -> 1088, ...
MBHB_ID = int(os.environ.get("MBHB_ID", "0"))

# Analysis sampling. DEFAULTS TO THE MOJITO NATIVE RATE -- a mojito comparison must be
# pinned to mojito's own settings, never to a convenience value. The window is built by
# NAIVE SUBSAMPLING (data_full[..., ::deci]) with no anti-alias filter while the template
# is generated natively at this DT, so any DT > native silently changes the physics being
# measured: at DT=10 the in-band corruption of the DATA is A 3.1e-5 / E 3.9e-5 but
# T 1.8e-2 -- LARGER than the response error under investigation, because T carries power
# above the DT=10 Nyquist and its in-band content is near-null. L/c ~ 8.3 s is also
# sub-sample at DT=10, so the TDI delay chain interpolates below one sample.
# Decimation is therefore OPT-IN (MBH_ALLOW_DECIMATE=1) and is checked against the file
# at load time; T-channel conclusions from a decimated run are invalid.
MOJITO_NATIVE_DT = 2.5         # asserted against loader.dt below
DT = float(os.environ.get("MBH_DT", str(MOJITO_NATIVE_DT)))
TDI_GEN_STR = "2nd generation"
TDI_CHAN = "XYZ"
NCHANNELS = 3

# Analysis band (global-fit MIN_FREQ / MAX_FREQ).
F_MIN, F_MAX = 1e-4, 2.5e-2

# PhenomTHM / response config (global-fit MBH_* constants).
MBH_HIGHER_MODES = (21, 33, 44)
MBH_PHENOM_TOL = 1e-12
MBH_START_FREQ = 7e-5
MBH_RESPONSE_ORDER = 30
MBH_BUFFER_TIME = 15_000.0

# Orbit-position interpolation grid (s).  The default LINEAR_INTERP_TIMESTEP is
# 50 s -> ~1.2 GB of x/v/n over the full mission, which pyResponseTDI deepcopies
# x2 -> OOM on an 8.6 GB box.  LISA spacecraft positions are smooth on the
# orbital (year) timescale, so a coarser grid is negligible at the inner-product
# level (the code's own prior default was 600 s); 300 s keeps the deepcopies
# small.  Tunable via env if an orbit-resolution-limited mismatch is suspected.
MBH_POS_DT = float(os.environ.get("MBH_POS_DT", "300.0"))

# Window: target span (days) carved AROUND the merger, half-day wavelets (the
# global-fit WDM resolution).  Merger placed at MERGER_FRAC of the window so the
# pre-merger inspiral (which carries most of the SNR) sits inside.
TARGET_WINDOW_DAYS = float(os.environ.get("MBH_WINDOW_DAYS", "48.0"))
MERGER_FRAC = 0.72
TUKEY_ALPHA = 0.05

# Template inspiral length (phentax T).  MUST cover the pre-merger span of the
# window (~MERGER_FRAC * TARGET_WINDOW_DAYS) so the early, LOW-FREQUENCY inspiral
# present in the mojito data is not truncated.  A too-short template (e.g. the
# global fit's YRSID/12 ~ 30.4 d) drops < ~3e-4 Hz content and inflates the
# mismatch: with the 30.4 d default, mm = 4.1e-2 entirely below 3e-4 Hz, while
# mm ~ 3e-4 above it.  Pre-merger window + 6 d margin (the extra inspiral before
# the window start is clipped onto the grid).
MBH_WAVEFORM_DURATION = float(os.environ.get(
    "MBH_WAVEFORM_DAYS", str(MERGER_FRAC * TARGET_WINDOW_DAYS + 6.0))) * 24 * 3600.0

NF, NT, WAVELET_DURATION = WDMSettings.adjust_to_even_bins(
    0.5 * 24 * 3600.0, 0.75 * 24 * 3600.0, DT, TARGET_WINDOW_DAYS * 24 * 3600.0
)
N_WIN = NF * NT
TOBS = N_WIN * DT

DATA_CACHE = f"/tmp/mbh_mojito_data_id{MBHB_ID}.npz"
# .npz: the cache now carries its grid key (n_win/dt/window_t0/id) alongside the
# array and is validated on load. The old bare-.npy caches are not readable here
# and are regenerated.
TMPL_CACHE = "/tmp/mbh_mojito_tmpl_id{id}_{tag}.npz"

MBH_TRANSFORM = make_mbh_transform_container()
_ORBITS = [None]            # memoized L1Orbits (freed once the wave_gen is built)
_WAVE_GEN = [None]          # the ONE PhenomTHMTDIWaveform (full-band WDM output)


def banner(s):
    print("\n" + "=" * 78 + f"\n {s}\n" + "=" * 78, flush=True)


def tukey(N, alpha):
    from scipy.signal.windows import tukey as _t
    return _t(N, alpha)


# ----------------------------------------------------------------------------
# cached data load -- L1ProcessingStep (data + orbits + catalogue), windowed
# around the merger.  The full-stream read runs at most once per source ever;
# subsequent runs reuse the cached window.
# ----------------------------------------------------------------------------
def load_data_cached():
    if os.path.exists(DATA_CACHE):
        z = np.load(DATA_CACHE, allow_pickle=True)
        # Shape ALONE is not a valid key: 48 d @ dt=10 and 12 d @ dt=2.5 are BOTH
        # (3, 414720), so a shape-only check silently accepts a window whose time
        # axis is wrong by the decimation factor (observed: merger at 34.56 d by
        # convention vs 8.62 d empirically). dt must match too, and a window with
        # no dt recorded predates this check and cannot be validated -> re-read.
        shape_ok = tuple(z["data_td"].shape) == (NCHANNELS, N_WIN)
        dt_ok = "dt" in z.files and abs(float(z["dt"]) - DT) < 1e-9
        if shape_ok and dt_ok:
            print(f"[cache] data window from {DATA_CACHE}  (dt={float(z['dt'])})",
                  flush=True)
            return (z["data_td"], float(z["window_t0"]), float(z["data_t0"]),
                    z["cat"].item(), float(z["abs_merger"]))
        why = (f"shape {z['data_td'].shape} != ({NCHANNELS},{N_WIN})" if not shape_ok
               else (f"dt {float(z['dt'])} != {DT}" if "dt" in z.files
                     else "no dt recorded (pre-dates the dt check)"))
        print(f"[cache] STALE ({why}) -> re-read", flush=True)

    print("[cache] MISS -> reading mojito MBHB via L1ProcessingStep "
          "(one time)...", flush=True)
    loader = L1ProcessingStep(
        L1_folder=PATH, source_types=["mbhb"], source_ids={"mbhb": MBHB_ID},
        orbits_class=L1Orbits,
        orbits_kwargs=dict(force_backend=BACKEND, frame="icrs"),
        verbose=True,
    )
    times = np.asarray(loader.times)
    data_full = np.asarray(loader.data)
    dt_native = float(loader.dt)
    data_t0 = float(times[0])
    cat = {k: float(np.asarray(v)) for k, v in loader.catalogue["MBHB"][MBHB_ID].items()
           if np.asarray(v).dtype.kind in "fi"}

    t_plunge = cat["TimeCoalescencePhenomTPHMSSBFrame"]
    abs_merger = MBH_WAVEFORM_T0 + t_plunge          # convention: REF + t_plunge

    # --- pin the analysis settings to the mojito file's own settings ---
    if abs(dt_native - MOJITO_NATIVE_DT) > 1e-9:
        print(f"[mojito] NOTE file dt={dt_native} differs from the assumed native "
              f"{MOJITO_NATIVE_DT}; trusting the FILE.", flush=True)
    deci = int(round(DT / dt_native))
    if deci != 1:
        if os.environ.get("MBH_ALLOW_DECIMATE", "0") not in ("1", "true", "True"):
            raise SystemExit(
                f"REFUSING to decimate mojito data {deci}x (MBH_DT={DT} vs file "
                f"dt={dt_native}).\nNaive subsampling with no anti-alias filter "
                f"corrupts the near-null T channel in-band by ~2e-2 (A/E ~3e-5), "
                f"which exceeds the response error under study; L/c~8.3 s also goes "
                f"sub-sample above dt=8.3.\nRe-run with MBH_DT={dt_native} (native), "
                f"or set MBH_ALLOW_DECIMATE=1 if you accept that ALL T-channel "
                f"results from this run are invalid.")
        print("#" * 78 + f"\n!! DECIMATING mojito data {deci}x "
              f"(dt {dt_native} -> {DT}) with NO anti-alias filter.\n"
              f"!! A/E remain usable; ANY T-CHANNEL RESULT FROM THIS RUN IS INVALID.\n"
              + "#" * 78, flush=True)
    n_full = data_full.shape[1]
    merger_idx_full = int(round((abs_merger - data_t0) / dt_native))
    start_full = merger_idx_full - int(round(MERGER_FRAC * N_WIN)) * deci
    start_full = max(0, min(start_full, n_full - N_WIN * deci))

    data_td = data_full[:, start_full: start_full + N_WIN * deci: deci][:, :N_WIN].copy()
    window_t0 = data_t0 + start_full * dt_native

    # dt/dt_native travel WITH the window so downstream scripts cannot silently
    # assume a different sampling than the one the data was built at.
    np.savez(DATA_CACHE, data_td=data_td, window_t0=window_t0, data_t0=data_t0,
             cat=cat, abs_merger=abs_merger, dt=DT, dt_native=dt_native,
             deci=deci)
    print(f"[cache] wrote {DATA_CACHE}  shape={data_td.shape}", flush=True)
    # Free the full-stream data AND the loader's full-mission orbit (~1.6 GB)
    # before get_orbits() builds the ltt-sliced one (8.6 GB box).
    del data_full, times, loader
    gc.collect()
    return data_td, window_t0, data_t0, cat, abs_merger


def get_orbits(window_t0):
    """Build the L1 orbit with its native ltt arrays SLICED to the analysis
    window, so PhenomTHMTDIWaveform's internal pyResponseTDI (which deepcopies
    the orbit x2) doesn't duplicate the full-mission ~1.2 GB ltt arrays.
    Mirrors the EMRI memory guard."""
    if _ORBITS[0] is None:
        print("[orbits] building L1Orbits over window...", flush=True)
        orb = L1Orbits(find_file(MBHB_L1, "MBHB", MBHB_ID),
                       force_backend=BACKEND, frame="icrs")
        pad = 1.0e5   # covers buffer_time (1.5e4) + light-travel delays
        lo = max(window_t0 - pad, float(orb.sc_t0))
        hi = min(window_t0 + TOBS + pad, float(orb._sc_t_base[-1]))

        # SLICE the native ltt arrays to the window. The C++ indexes ltts purely
        # by (ltt_t0, ltt_dt) so a contiguous truncation preserves the uniform
        # spacing exactly -- no interpolation needed.
        ltt_t = np.asarray(orb.ltt_t)
        m = (ltt_t >= lo) & (ltt_t <= hi)
        orb.ltt = np.asarray(orb.ltt)[m].copy()
        orb.ltt_t = ltt_t[m].copy()
        orb.ltt_t0 = float(orb.ltt_t[0])
        del ltt_t
        gc.collect()

        # positions: full-mission linear_interp_setup on a COARSE grid
        # (MBH_POS_DT) so the x/v/n arrays the response deepcopies stay small.
        orb.configure(linear_interp_setup=True, dt=MBH_POS_DT)
        _ORBITS[0] = orb
        mark(f"[orbits] ready (ltt sliced {int(m.sum())} pts; pos dt={MBH_POS_DT}s)")
    return _ORBITS[0]


# ----------------------------------------------------------------------------
# template: the global-fit signal_gen path (PhenomTHMTDIWaveform model)
# ----------------------------------------------------------------------------
def get_wave_gen(window_t0, wdm_full):
    """Build the ONE MBH PhenomTHMTDIWaveform (exactly as the global fit's
    ``_get_mbh_phenom_wave_gen``), bound to the full-band WDM output set.

    A SINGLE instance is reused for every template (TD via
    ``compute_tdi_channels`` + full-band WDM via ``get_signals_for_residuals``):
    each instance's internal ``pyResponseTDI`` deepcopies the orbit x2, so on an
    8.6 GB box we must not cache several.  The shared base orbit is dropped once
    the wave_gen owns its copies."""
    if _WAVE_GEN[0] is None:
        data_td_settings = TDSettings(N=N_WIN, dt=DT, t0=window_t0, force_backend=BACKEND)
        _WAVE_GEN[0] = PhenomTHMTDIWaveform(
            waveform_kwargs=dict(
                higher_modes=list(MBH_HIGHER_MODES),
                include_negative_modes=True,
                t_low_fit=True,
                coarse_grain=False,        # pyResponseTDI needs equispaced grids
                atol=MBH_PHENOM_TOL,
                rtol=MBH_PHENOM_TOL,
            ),
            Tobs=MBH_WAVEFORM_DURATION,    # waveform-generation window (phentax T)
            start_freq=MBH_START_FREQ,
            use_reference_time=True,
            waveform_t0=MBH_WAVEFORM_T0,
            data_td_settings=data_td_settings,
            tdi_generation=TDI_GEN_STR,
            tdi_channels=TDI_CHAN,
            sampling_frequency=1.0 / DT,
            orbits=get_orbits(window_t0),
            order=MBH_RESPONSE_ORDER,
            tukey_alpha=TUKEY_ALPHA,
            stft_dt=None,
            freq_min=F_MIN,
            freq_max=F_MAX,
            fft_batch_size=2,
            buffer_time=MBH_BUFFER_TIME,
            output_domain_settings=wdm_full,   # full-band WDM (signal_gen output)
            force_backend=BACKEND,
        )
        _ORBITS[0] = None              # wave_gen owns its orbit copies now
        gc.collect()
        mark("wave_gen built (orbit deepcopied x2 by pyResponseTDI)")
    return _WAVE_GEN[0]


def signal_gen(sampling_params, window_t0, wdm_full):
    """Engine-side MBH template generator (the run's ``_mbh_signal_gen``): one
    sampling-basis row -> one full-band WDM DomainBase via
    ``PhenomTHMTDIWaveform.get_signals_for_residuals``."""
    wave_gen = get_wave_gen(window_t0, wdm_full)
    params_in = MBH_TRANSFORM.both_transforms(np.asarray(sampling_params, dtype=float))
    return wave_gen.get_signals_for_residuals(*params_in)


def template_td_cached(tag, sampling_params, window_t0, wdm_full):
    """Raw TD TDI channels (same PhenomTHMTDIWaveform), placed on the window
    grid -- used for the fast FD tau/phase scan + narrowband WDM.  Cached."""
    # The cache MUST be keyed on the grid it was built for. Keying on (id, tag)
    # alone is silently wrong whenever the window changes: e.g. 48 d @ dt=10 and
    # 12 d @ dt=2.5 are BOTH 414720 samples, so a stale template loads with a
    # matching shape and yields plausible garbage. Validate against the grid and
    # regenerate on any mismatch.
    p = TMPL_CACHE.format(id=MBHB_ID, tag=tag)
    key = dict(n_win=int(N_WIN), dt=float(DT), window_t0=float(window_t0),
               mbhb_id=int(MBHB_ID))
    if os.path.exists(p):
        try:
            with np.load(p, allow_pickle=True) as zc:
                ok = (int(zc["n_win"]) == key["n_win"]
                      and abs(float(zc["dt"]) - key["dt"]) < 1e-9
                      and abs(float(zc["window_t0"]) - key["window_t0"]) < 1e-6
                      and int(zc["mbhb_id"]) == key["mbhb_id"])
                if ok:
                    print(f"[cache] template '{tag}' from {p}", flush=True)
                    return zc["arr"]
                print(f"[cache] STALE template '{tag}' ({p}): built for "
                      f"n_win={int(zc['n_win'])} dt={float(zc['dt'])} "
                      f"t0={float(zc['window_t0']):.3f}, need "
                      f"n_win={key['n_win']} dt={key['dt']} "
                      f"t0={key['window_t0']:.3f} -> regenerating", flush=True)
        except Exception as e:                       # legacy bare-.npy caches
            print(f"[cache] unusable template cache {p} ({e}) -> regenerating",
                  flush=True)
    wave_gen = get_wave_gen(window_t0, wdm_full)
    params_in = MBH_TRANSFORM.both_transforms(np.asarray(sampling_params, dtype=float))
    times, channels = wave_gen.compute_tdi_channels(*params_in)
    grid = TDSettings(N=N_WIN, dt=DT, t0=window_t0, force_backend=BACKEND)
    arr = np.asarray(
        place_td_signal_on_grid(np.atleast_2d(channels)[:NCHANNELS], grid, times=times).arr
    )
    np.savez(p, arr=arr, **key)
    print(f"[cache] wrote template '{tag}' -> {p}  "
          f"(n_win={key['n_win']} dt={key['dt']})", flush=True)
    return arr


# ----------------------------------------------------------------------------
# overlap helpers (FD)
# ----------------------------------------------------------------------------
def fd_setup(data_td, window_t0, win):
    td_set = TDSettings(N_WIN, DT, t0=window_t0, force_backend=BACKEND)
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
    print(f"  [{label:24s}] |O|={abs(O):.6f}  Re(O)={O.real:+.6f}  "
          f"ph={np.degrees(np.angle(O)):+7.1f}  mm(pmax)={1-abs(O):.4e}  "
          f"SNR d={np.sqrt(dd.real):.1f} h={opt:.1f} det={det:+.2f}", flush=True)
    return O


def time_scan(ac, base_sig, fd_set, taus):
    """Phase-maximized |O| vs time shift.  Returns (best|O|, best tau)."""
    f = np.asarray(fd_set.f_arr)
    base = np.asarray(base_sig.arr).copy()
    best = (-1.0, 0.0)
    for tau in taus:
        base_sig.arr[:] = base * np.exp(-2j * np.pi * f * tau)[None, :]
        O = ac.template_inner_product(base_sig, normalize=True, complex=True)
        if abs(O) > best[0]:
            best = (abs(O), float(tau))
    base_sig.arr[:] = base
    return best


# ----------------------------------------------------------------------------
# WDM overlap (the global-fit-native metric, via the signal_gen path)
# ----------------------------------------------------------------------------
def _wdm_report(ws, data_td, template, td_set, win, tag):
    d = TDSignal(data_td, td_set).transform(ws, window=win)
    t = (template if not isinstance(template, np.ndarray)
         else TDSignal(template, td_set).transform(ws, window=win))
    ac = AnalysisContainer(d, XYZ2SensitivityMatrix(ws, model=SENS_MODEL))
    O = ac.template_inner_product(t, normalize=True, complex=True)
    opt, det = ac.template_snr(t)
    dd = ac.inner_product()
    print(f"  WDM {tag:22s} |O|={abs(O):.6f}  Re(O)={O.real:+.6f}  "
          f"mm={1-O.real:.4e}  mm(pmax)={1-abs(O):.4e}  "
          f"SNR d={np.sqrt(dd.real):.1f} h={opt:.1f} det={det:+.2f}", flush=True)
    return O


def wdm_overlap(data_td, window_t0, sampling_params, win, wdm_full, tmpl_td):
    td_set = TDSettings(N_WIN, DT, t0=window_t0, force_backend=BACKEND)
    # full-band: native signal_gen (PhenomTHMTDIWaveform.get_signals_for_residuals,
    # bound to wdm_full) -- the global-fit-native metric.
    t_full = signal_gen(sampling_params, window_t0, wdm_full)
    _wdm_report(wdm_full, data_td, t_full, td_set, win, "full-band(signal_gen)")
    # split bands (same TD template): the low band carries the truncation-
    # sensitive early inspiral; the high band is merger-dominated.
    FSPLIT = 5e-4
    for lo, hi, tag in [(F_MIN, FSPLIT, "low(<5e-4Hz)"),
                        (FSPLIT, F_MAX, "merger(>5e-4Hz)")]:
        ws = WDMSettings(NF, NT, DT, t0=window_t0, min_freq=lo, max_freq=hi,
                         force_backend=BACKEND)
        _wdm_report(ws, data_td, tmpl_td, td_set, win, tag)


def make_plots(data_td, tmpl_td, td_set, fd_set, win, window_t0, abs_merger):
    f = np.asarray(fd_set.f_arr)
    d_fd = np.asarray(fd_signal(data_td, td_set, fd_set, win).arr)
    t_fd = np.asarray(fd_signal(tmpl_td, td_set, fd_set, win).arr)
    fig, ax = plt.subplots(2, 1, figsize=(9, 8))
    ax[0].loglog(f, np.abs(d_fd[0]), label="data X")
    ax[0].loglog(f, np.abs(t_fd[0]), label="template X (best)", alpha=.8)
    ax[0].set_xlim(F_MIN, F_MAX); ax[0].legend(); ax[0].set_title("MBH FD |X|")
    # TD zoom around the merger.
    merger_idx = int(round((abs_merger - window_t0) / DT))
    n0 = max(0, merger_idx - N_WIN // 12)
    sl = slice(n0, min(N_WIN, n0 + N_WIN // 6))
    tt = (np.arange(N_WIN)[sl]) * DT / 86400.0
    ax[1].plot(tt, data_td[0][sl], label="data X", lw=1.2)
    ax[1].plot(tt, tmpl_td[0][sl], "--", label="template X (best)", lw=1.0)
    ax[1].axvline(merger_idx * DT / 86400.0, color="k", ls=":", label="merger")
    ax[1].legend(); ax[1].set_title("MBH TD X (zoom @ merger)")
    ax[1].set_xlabel("days into window")
    fig.tight_layout()
    fig.savefig(f"/tmp/mbh_mojito_fd_td_id{MBHB_ID}.png", dpi=120)
    plt.close(fig)


# ----------------------------------------------------------------------------
def main():
    threading.Thread(target=_watchdog, daemon=True).start()
    mark("main start")
    rebuild = "--rebuild" in sys.argv
    if rebuild and os.path.exists(DATA_CACHE):
        os.remove(DATA_CACHE)

    banner(f"STAGE 1: load MBHB id={MBHB_ID} (cached) + anchors")
    print(f"  WDM grid: NF={NF} NT={NT}  wavelet={WAVELET_DURATION/3600:.2f} h  "
          f"N_WIN={N_WIN}  Tobs={TOBS/86400:.2f} d  dt={DT}", flush=True)
    data_td, window_t0, data_t0, cat, abs_merger = load_data_cached()
    mark("data loaded")
    samp = mbh_catalogue_to_sampling_basis(cat)              # sampling basis (11)
    wf = MBH_TRANSFORM.both_transforms(samp.copy())          # waveform basis (11)
    t_plunge = samp[-1]

    print(f"  data_t0={data_t0:.3f}  REF={MOJITO_REFERENCE_TIME:.3f}  "
          f"trim={data_t0-MOJITO_REFERENCE_TIME:.1f}s", flush=True)
    print(f"  window_t0={window_t0:.3f}  ({(window_t0-data_t0)/86400:.2f} d into file)",
          flush=True)
    print(f"  EstimatedSNR={cat.get('EstimatedSNR', float('nan')):.1f}  "
          f"m1={cat['PrimaryMassSSBFrame']:.3e}  m2={cat['SecondaryMassSSBFrame']:.3e}  "
          f"f22={cat.get('GW22FrequencySSBFrame', float('nan')):.3e}", flush=True)
    print(f"  sampling basis: {np.array2string(samp, precision=5)}", flush=True)
    print(f"  waveform basis: {np.array2string(np.atleast_1d(wf), precision=5)}", flush=True)

    # convention-predicted vs empirical (max rolling power) merger location.
    merger_idx_conv = int(round((abs_merger - window_t0) / DT))
    env = np.sqrt(np.mean(data_td ** 2, axis=0))
    box = max(1, N_WIN // 200)
    power = np.convolve(env ** 2, np.ones(box) / box, mode="same")
    merger_idx_emp = int(np.argmax(power))
    print(f"  merger (convention REF+t_plunge): idx={merger_idx_conv} "
          f"= {merger_idx_conv*DT/86400:.2f} d into window", flush=True)
    print(f"  merger (empirical max-power):     idx={merger_idx_emp} "
          f"= {merger_idx_emp*DT/86400:.2f} d into window  "
          f"-> dt_emp-conv={(merger_idx_emp-merger_idx_conv)*DT:+.1f}s", flush=True)
    f_merger = float(cat.get("GW22FrequencySSBFrame", 1e-3)) or 1e-3

    win = tukey(N_WIN, alpha=TUKEY_ALPHA)
    ac, td_set, fd_set = fd_setup(data_td, window_t0, win)

    # The ONE full-band WDM set: the wave_gen's output target AND the data-side
    # WDM grid (sprint rule: domain communicated by a settings object).
    wdm_full = WDMSettings(NF, NT, DT, t0=window_t0, min_freq=F_MIN, max_freq=F_MAX,
                           force_backend=BACKEND)

    banner("STAGE 2: FD overlap (raw, convention placement)")
    t_raw = template_td_cached("raw", samp, window_t0, wdm_full)
    mark("template (raw) built")
    s_raw = fd_signal(t_raw, td_set, fd_set, win)
    O_raw = report(ac, s_raw, "convention placement")

    banner("STAGE 3: MAXIMIZE overlap -- coarse->fine tau scan + phase")
    period = 1.0 / f_merger
    # coarse: absorb a convention/merger-time offset up to +-1 wavelet duration.
    coarse = np.linspace(-WAVELET_DURATION, WAVELET_DURATION, 161)
    b_coarse = time_scan(ac, s_raw, fd_set, coarse)
    print(f"  coarse: |O|max={b_coarse[0]:.6f} at tau={b_coarse[1]:+.1f}s "
          f"(grid {coarse[1]-coarse[0]:.1f}s)", flush=True)
    fine = b_coarse[1] + np.linspace(-2 * period, 2 * period, 241)
    b_fine = time_scan(ac, s_raw, fd_set, fine)
    print(f"  fine:   |O|max={b_fine[0]:.6f} at tau={b_fine[1]:+.3f}s "
          f"(period={period:.1f}s)", flush=True)
    tau_star = b_fine[1]
    print(f"  => best tau* = {tau_star:+.3f}s  -> shift t_plunge by this amount.",
          flush=True)
    print(f"     residual phase arg<d|h> = {np.degrees(np.angle(O_raw)):+.1f} deg "
          f"(phi_ref / phase-reference convention).", flush=True)

    banner("STAGE 4: regenerate at best tau via signal_gen -> WDM mismatch")
    samp_fix = samp.copy()
    samp_fix[-1] = t_plunge + tau_star                       # t_plunge += tau*
    # FD check of the regenerated (non-FD-shifted) template.
    t_fix = template_td_cached("fix", samp_fix, window_t0, wdm_full)
    s_fix = fd_signal(t_fix, td_set, fd_set, win)
    report(ac, s_fix, "FD @ best tau")
    wdm_overlap(data_td, window_t0, samp_fix, win, wdm_full, t_fix)

    make_plots(data_td, t_fix, td_set, fd_set, win, window_t0, abs_merger)
    print(f"\nDONE.  plots -> /tmp/mbh_mojito_fd_td_id{MBHB_ID}.png", flush=True)


if __name__ == "__main__":
    main()
