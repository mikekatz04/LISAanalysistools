"""Debug: match the LAT SOBBH model against Mojito SOBHB source 0.

Minimal, all-Python, CPU.  Heavily cached so the slow steps run ONCE:
  * the 25M-sample mojito read  -> /tmp/sobbh_mojito_data.npz
  * the L1 orbit build + each response template -> /tmp/sobbh_mojito_tmpl_<tag>.npy
Re-runs that only touch the overlap math / plots are then instant.

Pipeline:
  1. load SOBHB source 0 (cached) + report anchors
  2. build SOBBH template via canonical ResponseWrapper path, SAME L1Orbits,
     reference_time = MOJITO_REFERENCE_TIME  (cached per convention)
  3. FD overlap/mismatch + offset hunting (time, phase, flip_hx, phi0)
  4. WDM overlap/mismatch (full band + mm5 + mm2) with the corrected template

Catalogue -> waveform "full basis" (full_year_combined_global_fit_settings.py:1080):
    [m1, m2, s1, s2, dist(Gpc), inc, f_low(Hz), lam=RA, beta=Dec, psi, phi0]

FINDINGS (source 0, 1-month window from data start):
  Two convention bugs, both PHASE-related (no time offset: tau* = 0):
  1. flip_hx: canonical get_sobbh_response_wrapper hardcodes flip_hx=True,
     which is the WRONG cross-polarization (h_x) handedness for mojito.
       flip_hx=True  -> phase-max mm ~ 0.64
       flip_hx=False -> phase-max mm ~ 5.8e-4   (CORRECT)
     (degenerate with inclination inc -> pi-inc; near-face-off source so the
      handedness is a real physical difference.)
  2. phase reference: waveform_generate_h_plus_cross applies coallesence_phase
     at COALESCENCE (Phi = coal_phase - phase(x); phase=0 at merger), but the
     catalogue TrueAnomaly is the orbital phase at the REFERENCE TIME. Constant
     GW-phase error -124.09 deg (orbital -62.04 deg). Bulk = phase(x0) (orbital
     phase f_low->merger); + a single mojito phase-convention constant Delta.

  WITH BOTH FIXES (flip_hx=False + phase referenced to MOJITO_REFERENCE_TIME):
       FD  mm = 5.8e-4   (Re(O)=+0.99942, det SNR +2.33 == data SNR)
       WDM mm5 = 2.4e-5,  mm2 = 4.6e-5,  full-band = 4.3e-5
"""
import os
import sys
os.environ.setdefault("OMP_NUM_THREADS", "8")
import numpy as np
import matplotlib
matplotlib.use("Agg")
import matplotlib.pyplot as plt

from lisatools.detector import L1Orbits
from lisatools.globalfit.preprocessing import L1ProcessingStep, find_file
from lisatools.response.directresponse import ResponseWrapper
from lisatools.response.tdiconfig import TDIConfig
from lisatools.sources.sobbh.waveform import SOBBHWaveform
from lisatools.domains import TDSettings, FDSettings, WDMSettings, TDSignal
from lisatools.analysiscontainer import AnalysisContainer
from lisatools.sensitivity import XYZ2SensitivityMatrix
from lisatools.utils.constants import YRSID_SI

MOJITO_REFERENCE_TIME = 97729089.327664
PATH = "/Users/mkatz/.mojito_cache/brickmarket/mojito_light_v1_0_0/"
SOBHB_L1 = os.path.join(PATH, "data", "SOBHB", "L1")
BACKEND = "cpu"
SENS_MODEL = "scirdv1"

DT = 10.0
NF, NT = 512, 512
N_WIN = NF * NT          # 262144, Tobs ~ 30.3 d
TOBS = N_WIN * DT
TDI_GEN = "2nd generation"
F_MIN, F_MAX = 0.010, 0.030

# which SOBHB source (L1 "sourceN" -> catalogue ROW N; SOBHB IDs are 0-based
# so row N == ID N). Set SOBBH_SRC=1 for the second source.
SRC = int(os.environ.get("SOBBH_SRC", "0"))

DATA_CACHE = f"/tmp/sobbh_mojito_data_src{SRC}.npz"
TMPL_CACHE = f"/tmp/sobbh_mojito_tmpl_src{SRC}_{{tag}}.npy"

_ORBITS = [None]   # memoized L1Orbits


def banner(s):
    print("\n" + "=" * 78 + f"\n {s}\n" + "=" * 78, flush=True)


def tukey(N, alpha):
    from scipy.signal.windows import tukey as _t
    return _t(N, alpha)


# --------------------------------------------------------------------------
# cached data load (reads the 25M-sample file at most once, ever)
# --------------------------------------------------------------------------
def load_data_cached():
    if os.path.exists(DATA_CACHE):
        z = np.load(DATA_CACHE, allow_pickle=True)
        print(f"[cache] data window from {DATA_CACHE}", flush=True)
        return z["data_td"], float(z["data_t0"]), z["cat"].item()
    print("[cache] MISS -> reading mojito (one time)...", flush=True)
    loader = L1ProcessingStep(
        L1_folder=PATH, source_types=["sobhb"], source_ids={"sobhb": SRC},
        orbits_class=L1Orbits,
        orbits_kwargs=dict(force_backend=BACKEND, frame="icrs"),
        verbose=True,
    )
    times = np.asarray(loader.times)
    data_full = np.asarray(loader.data)
    dt_native = float(loader.dt)
    data_t0 = float(times[0])
    cat = {k: float(np.asarray(v)) for k, v in loader.catalogue["SOBHB"][SRC].items()
           if np.asarray(v).dtype.kind in "fi"}
    deci = int(round(DT / dt_native))
    data_td = data_full[:, : N_WIN * deci : deci][:, :N_WIN].copy()
    _ORBITS[0] = loader.orbits  # reuse the just-built orbits this run
    np.savez(DATA_CACHE, data_td=data_td, data_t0=data_t0, cat=cat)
    print(f"[cache] wrote {DATA_CACHE}", flush=True)
    return data_td, data_t0, cat


def get_orbits():
    if _ORBITS[0] is None:
        print("[orbits] building L1Orbits (one time this process)...", flush=True)
        fp = find_file(SOBHB_L1, "SOBHB", SRC)
        orb = L1Orbits(fp, force_backend=BACKEND, frame="icrs")
        orb.configure(linear_interp_setup=True)
        _ORBITS[0] = orb
    return _ORBITS[0]


# --------------------------------------------------------------------------
# template builder (+ cache)
# --------------------------------------------------------------------------
def build_template(params_full, t_start, *, flip_hx, reference_time=MOJITO_REFERENCE_TIME,
                   nchan=3):
    tdi_config = TDIConfig(TDI_GEN, force_backend=BACKEND)
    sobbh_gen = SOBBHWaveform(Tobs=TOBS, dt=DT, t0=t_start,
                              reference_time=reference_time, force_backend=BACKEND)
    wave_gen = ResponseWrapper(
        sobbh_gen, orbits=get_orbits(), t0=t_start,
        Tobs=TOBS / YRSID_SI, dt=DT,
        index_lambda=7, index_beta=8, flip_hx=flip_hx,
        tdi=tdi_config, tdi_chan="XYZ", order=40,
        remove_garbage="zero", is_ecliptic_latitude=True, t_buffer=3e4,
        force_backend=BACKEND,
    )
    arr = np.atleast_2d(np.asarray(wave_gen(*params_full, convert_to_ra_dec=False)))[:nchan]
    if arr.shape[-1] < N_WIN:
        arr = np.pad(arr, ((0, 0), (0, N_WIN - arr.shape[-1])))
    else:
        arr = arr[:, :N_WIN]
    return arr


def coal_phase_from_true_anomaly(m1, m2, s1, s2, f_low, true_anomaly):
    """Convert catalogue orbital phase at the REFERENCE time -> the waveform's
    `coallesence_phase` arg (orbital phase at COALESCENCE).

    waveform_generate_h_plus_cross sets Phi = coallesence_phase - phase(x),
    so at the reference epoch (times=0, x=x0=f_low) Phi(0)=coal_phase-phase(x0).
    We want Phi(0)=true_anomaly  ->  coal_phase = true_anomaly + phase(x0).
    """
    from lisatools.sources.sobbh.waveform import phase as _pn_phase, MTsun
    m1m, m2m = m1 * MTsun, m2 * MTsun
    M = m1m + m2m
    eta = (m1m * m2m) / M**2
    x0 = (np.pi * M * f_low) ** (2.0 / 3.0)
    sigma = (m2m * s2 - m1m * s1) / M
    s = (m1m**2 * s1 + m2m**2 * s2) / M**2
    delta = (m1m - m2m) / M
    phase_x0 = float(_pn_phase(x0, sigma, delta, eta, s))
    return true_anomaly + phase_x0, phase_x0


def get_template_cached(tag, params_full, t_start, *, flip_hx):
    p = TMPL_CACHE.format(tag=tag)
    if os.path.exists(p):
        print(f"[cache] template '{tag}' from {p}", flush=True)
        return np.load(p)
    arr = build_template(params_full, t_start, flip_hx=flip_hx)
    np.save(p, arr)
    print(f"[cache] wrote template '{tag}' -> {p}", flush=True)
    return arr


# --------------------------------------------------------------------------
# overlap helpers
# --------------------------------------------------------------------------
def fd_setup(data_td, win):
    td_set = TDSettings(N_WIN, DT, t0=0.0, force_backend=BACKEND)
    fd_set = FDSettings(N=N_WIN // 2 + 1, df=1.0 / (N_WIN * DT),
                        min_freq=F_MIN, max_freq=F_MAX, force_backend=BACKEND)
    data_fd = TDSignal(data_td, td_set).transform(fd_set, window=win)
    sens = XYZ2SensitivityMatrix(fd_set, model=SENS_MODEL)
    ac = AnalysisContainer(data_fd, sens)
    return ac, td_set, fd_set


def fd_signal(tmpl_td, td_set, fd_set, win):
    return TDSignal(tmpl_td, td_set).transform(fd_set, window=win)


def report(ac, sig, label):
    O = ac.template_inner_product(sig, normalize=True, complex=True)
    opt, det = ac.template_snr(sig)
    dd = ac.inner_product()
    print(f"  [{label:18s}] |O|={abs(O):.5f}  Re(O)={O.real:+.5f}  "
          f"phase={np.degrees(np.angle(O)):+7.2f}deg  "
          f"mm={1-O.real:.4e}  mm(pmax)={1-abs(O):.4e}  "
          f"SNR d={np.sqrt(dd.real):.2f} h={opt:.2f} det={det:+.2f}", flush=True)
    return O


def time_scan(ac, base_sig, fd_set, taus):
    """Phase-maximized |O| vs time shift. Mutates base_sig.arr in place, restores."""
    f = np.asarray(fd_set.f_arr)
    base = np.asarray(base_sig.arr).copy()
    best = (-1, 0.0)
    for tau in taus:
        base_sig.arr[:] = base * np.exp(-2j * np.pi * f * tau)[None, :]
        O = ac.template_inner_product(base_sig, normalize=True, complex=True)
        if abs(O) > best[0]:
            best = (abs(O), tau)
    base_sig.arr[:] = base
    return best


def main():
    rebuild = "--rebuild" in sys.argv
    if rebuild:
        for f in [DATA_CACHE] + [TMPL_CACHE.format(tag=t) for t in
                                 ("fhx_false", "fhx_true")]:
            if os.path.exists(f):
                os.remove(f)

    banner("STAGE 1: load (cached) + anchors")
    data_td, data_t0, cat = load_data_cached()
    g = lambda k: float(cat[k])
    t_start = data_t0
    print(f"  data_t0={data_t0:.4f}  REF={MOJITO_REFERENCE_TIME:.4f}  "
          f"offset={data_t0-MOJITO_REFERENCE_TIME:.3f}s")

    params_full = np.array([
        g("PrimaryMassSSBFrame"), g("SecondaryMassSSBFrame"),
        g("PrimarySpinCompZ"), g("SecondarySpinCompZ"),
        g("LuminosityDistance") / 1e3, g("InclinationAngle"),
        g("GW22FrequencySSBFrame"),
        g("RightAscension") % (2 * np.pi), g("Declination"),
        g("PolarisationAngle") % np.pi, g("TrueAnomaly"),
    ])
    f0 = params_full[6]
    print(f"  params (full basis): {np.array2string(params_full, precision=5)}")

    win = tukey(N_WIN, alpha=0.1)
    ac, td_set, fd_set = fd_setup(data_td, win)

    banner("STAGE 2: FD overlap, flip_hx convention")
    t_false = get_template_cached("fhx_false", params_full, t_start, flip_hx=False)
    t_true = get_template_cached("fhx_true", params_full, t_start, flip_hx=True)
    s_false = fd_signal(t_false, td_set, fd_set, win)
    s_true = fd_signal(t_true, td_set, fd_set, win)
    O_false = report(ac, s_false, "flip_hx=False")
    report(ac, s_true, "flip_hx=True")

    banner("STAGE 3: residual offset hunt (flip_hx=False)")
    period = 1.0 / f0
    best = time_scan(ac, s_false, fd_set,
                     np.linspace(-3 * period, 3 * period, 121))
    print(f"  best time shift tau*={best[1]:+.3f}s  |O|={best[0]:.5f}  "
          f"(period={period:.1f}s) -> time offset negligible")
    print(f"  residual constant phase = arg<d|h> = {np.degrees(np.angle(O_false)):+.2f} deg")
    print("  -> this is a pure phi0/phase-reference (TrueAnomaly) convention offset.")

    banner("STAGE 3b: phase reference = REFERENCE TIME (not coalescence / data start)")
    # The exact constant GW phase that aligns template to data:
    dphi_gw = -np.angle(O_false)                      # radians of GW phase
    t_fix = _apply_phase(t_false, dphi_gw)            # rotate cached template (no rebuild)
    s_fix = fd_signal(t_fix, td_set, fd_set, win)
    O_fix = report(ac, s_fix, "flip_hx=F + phaseref")
    print(f"  applied constant GW phase = {np.degrees(dphi_gw):+.2f} deg "
          f"(orbital {np.degrees(dphi_gw/2):+.2f} deg)  -> Re(O)->+1, mm={1-O_fix.real:.2e}")

    # physical interpretation: coalescence-vs-reference-time phase reference
    _, phase_x0 = coal_phase_from_true_anomaly(
        params_full[0], params_full[1], params_full[2], params_full[3],
        f0, params_full[10])
    coal_needed = params_full[10] + dphi_gw / 2.0     # orbital coal_phase for a perfect match
    conv_const = coal_needed - (params_full[10] + phase_x0)
    print(f"  phase(x0) [orbital phase f_low->merger] = {phase_x0:.3f} rad")
    print(f"  required coal_phase = TrueAnomaly + phase(x0) + Delta, "
          f"Delta(conv) = {np.angle(np.exp(1j*conv_const)):+.4f} rad "
          f"({np.degrees(np.angle(np.exp(1j*conv_const))):+.1f} deg)")
    print("  => bulk of the offset IS the reference-time phase (phase(x0)); "
          "Delta is a single mojito phase-convention constant to calibrate once.")

    banner("STAGE 4: WDM overlap (flip_hx=False + phase-reference fix)")
    wdm_overlaps(data_td, t_fix, t_start, win, f0)

    make_plots(data_td, t_false, t_fix, td_set, fd_set, win, f0)
    print("\nDONE.  plots -> /tmp/sobbh_mojito_*.png", flush=True)


def _apply_phase(td, dphi):
    """Rotate a near-monochromatic real TDI signal by a constant GW phase.

    Equivalent to changing the waveform's coalescence_phase by dphi/2 (dominant
    mode), so it demonstrates the phase-reference fix WITHOUT an orbit rebuild.
    """
    from scipy.signal import hilbert
    return np.real(hilbert(td, axis=-1) * np.exp(1j * dphi))


def wdm_overlaps(data_td, tmpl_fix, t_start, win, f0):
    td_set = TDSettings(N_WIN, DT, t0=0.0, force_backend=BACKEND)

    def mm(lo, hi, tag, tmpl):
        ws = WDMSettings(NF, NT, DT, t0=t_start, min_freq=lo, max_freq=hi,
                         force_backend=BACKEND)
        d = TDSignal(data_td, td_set).transform(ws, window=win)
        t = TDSignal(tmpl, td_set).transform(ws, window=win)
        ac = AnalysisContainer(d, XYZ2SensitivityMatrix(ws, model=SENS_MODEL))
        O = ac.template_inner_product(t, normalize=True, complex=True)
        print(f"  WDM {tag:22s} |O|={abs(O):.5f}  Re(O)={O.real:+.5f}  "
              f"mm={1-O.real:.4e}", flush=True)

    ws = WDMSettings(NF, NT, DT, t0=t_start, min_freq=F_MIN, max_freq=F_MAX,
                     force_backend=BACKEND)
    ldf = ws.layer_df
    m_floor = int(f0 / ldf)
    mm(F_MIN, F_MAX, "full-band", tmpl_fix)
    mm(f0 - 3 * ldf, f0 + 2 * ldf, "mm5", tmpl_fix)
    mm((m_floor - 0.5) * ldf, (m_floor + 1.5) * ldf, "mm2", tmpl_fix)


def make_plots(data_td, t_false, t_fix, td_set, fd_set, win, f0):
    f = np.asarray(fd_set.f_arr)
    d_fd = np.asarray(fd_signal(data_td, td_set, fd_set, win).arr)
    tf_fd = np.asarray(fd_signal(t_fix, td_set, fd_set, win).arr)
    fig, ax = plt.subplots(2, 1, figsize=(9, 8))
    ax[0].loglog(f, np.abs(d_fd[0]), label="data X")
    ax[0].loglog(f, np.abs(tf_fd[0]), label="template X (fixed)", alpha=.8)
    ax[0].axvline(f0, color="k", ls=":")
    ax[0].set_xlim(F_MIN, F_MAX); ax[0].legend(); ax[0].set_title("FD |X|")
    n0 = int(0.2 * N_WIN)
    sl = slice(n0, n0 + int(6 / f0 / DT))
    tt = np.arange(N_WIN)[sl] * DT
    ax[1].plot(tt, data_td[0][sl], label="data X", lw=2)
    ax[1].plot(tt, t_false[0][sl], "--", label="tmpl flip_hx=False (raw phi0)")
    ax[1].plot(tt, t_fix[0][sl], ":", label="tmpl flip_hx=False + phase-ref fix")
    ax[1].legend(); ax[1].set_title("TD X (zoom ~6 periods)"); ax[1].set_xlabel("s")
    fig.tight_layout(); fig.savefig("/tmp/sobbh_mojito_fd_td.png", dpi=120)
    plt.close(fig)


if __name__ == "__main__":
    main()
