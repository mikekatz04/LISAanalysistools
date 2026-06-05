"""
Compare the Chebyshev t(f) basis against the full FEW EMRI waveform for three
parameter sets.

For each (M1, M2, a, e0, p0) + (alpha_0..alpha_4) pair we:
  * generate the full single-mode (2,2,0,0)+(2,-2,0,0) EMRI projected through
    1st-generation TDI in both AET and XYZ;
  * pull the FEW trajectory to get the truth f(t) and phase(t) of the m=2 mode;
  * scan the empirical Chebyshev ``t_scale`` for the best 1-channel matched
    filter (max-tau, phase-max) against the A channel;
  * compute lisatools-based inner products / overlaps per channel
    (A, E, T, X, Y, Z) using AnalysisContainer.template_snr;
  * plot f(t), phase(t), and spectrograms for the EMRI A channel and the
    Chebyshev template.

Run with the ``deving`` conda env:

    conda activate deving
    python emri_chebyshev_comparison.py
"""

import os
import sys

# Self-detach from the launching shell's process group so a parent SIGKILL
# (e.g. when a harnessed Bash task terminates) doesn't take us down.
try:
    os.setsid()
except OSError:
    pass

import numpy as np
import matplotlib.pyplot as plt

from scipy.interpolate import CubicSpline
from scipy.integrate import cumulative_simpson
from scipy.signal import spectrogram
from scipy.optimize import minimize

import lisatools
from lisatools.detector import EqualArmlengthOrbits
from lisatools.utils.constants import YRSID_SI
from lisatools.utils.utility import get_array_module
from lisatools.datacontainer import DataResidualArray
from lisatools.analysiscontainer import AnalysisContainer
from lisatools.sensitivity import (
    AET1SensitivityMatrix,
    XYZ1SensitivityMatrix,
    SensitivityMatrix,
    A1TDISens,
    E1TDISens,
    T1TDISens,
    X1TDISens,
    Y1TDISens,
    Z1TDISens,
)
from lisatools.domains import TDSettings, TDSignal, FDSettings, FDSignal
from lisatools.sources.emri.chebyshevwave import ChebyshevWave

from fastlisaresponse import ResponseWrapper
from fastlisaresponse.tdiconfig import TDIConfig

from few.waveform import GenerateEMRIWaveform
from few.trajectory.ode import KerrEccEqFlux
from few.trajectory.inspiral import EMRIInspiral
from few.utils.geodesic import get_fundamental_frequencies
from few.utils.constants import MTSUN_SI


# -----------------------------------------------------------------------------
# Backend + global config
# -----------------------------------------------------------------------------
backend = "cpu"
backend_obj = lisatools.get_backend(backend)
xp = backend_obj.xp


def _to_cpu(arr):
    if hasattr(arr, "get") and not isinstance(arr, np.ndarray):
        return arr.get()
    return np.asarray(arr)


dt      = 15.0
Tobs    = 1.0          # years
t_start = 0.5 * YRSID_SI

orbits     = EqualArmlengthOrbits(force_backend=backend)
tdi_config = TDIConfig("1st generation")

emri_gen = GenerateEMRIWaveform(
    "FastKerrEccentricEquatorialFlux",
    return_list=False,
    inspiral_kwargs={"DENSE_STEPPING": 0, "max_init_len": int(1e4),
                     "force_backend": backend},
    sum_kwargs={"pad_output": True},
    frame="detector",
    mode_selector_kwargs={"mode_selection_threshold": 1e-5},
    force_backend=backend,
)

response_kwargs = dict(
    Tobs=Tobs,
    dt=dt,
    index_lambda=8,
    index_beta=7,
    flip_hx=True,
    force_backend=backend,
    tdi=tdi_config,
    order=20,
    remove_garbage="zero",
    is_ecliptic_latitude=False,
    t_buffer=3e4,
    t0=t_start,
)

response_AET = ResponseWrapper(emri_gen, orbits=orbits, tdi_chan="AET",
                               **response_kwargs)
response_XYZ = ResponseWrapper(emri_gen, orbits=orbits, tdi_chan="XYZ",
                               **response_kwargs)

traj_model = EMRIInspiral(func=KerrEccEqFlux)

# Auxiliary parameters shared across all 3 sets — matches
# emri_snr_correlation_test.py / dev_emri_search.py.
xI0           = -1.0
dist          = 1.06696103277        # Gpc
beta, lam     = 0.50015698, 4.707806421
qS, phiS      = np.pi / 2 - beta, lam
qK, phiK      = 0.7247331791, 4.039822998
Phi_phi0      = 2.15498605
Phi_theta0    = 0.4543828072
Phi_r0        = 1.124806044

# Single mode (2,2,0,0) only — no (2,-2,0,0) partner.
runtime_kwargs = dict(
    mode_selection=[(2, 2, 0, 0)],
    include_minus_mkn=False,
)

# Chebyshev grid (matches dev_emri_search.py).
CHEB_FMIN_REQ = 2.0e-3
CHEB_FMAX_REQ = 5.0e-3
CHEB_NPTS     = 32001

# t_scale scan — dev_emri_search.py hardcodes 600.0. We scan a wider range
# per set with two refinement passes around the maximum.
T_SCALE_COARSE  = np.linspace(50.0, 3000.0, 30)
T_SCALE_REFINE  = 21        # number of refine points around each coarse winner
T_SCALE_REFINE2 = 21        # 2nd refinement (finer grid)

# DE refit bounds — match dev_emri_search.py alpha priors. t_scale is broad.
DE_ALPHA_BOUNDS = [
    (10.6,    11.7),
    (-0.97,  -0.66),
    (-0.052, -0.005),
    (-0.0041, +0.0018),
    (-0.0010, +0.0003),
]
DE_T_SCALE_BOUNDS = (50.0, 2000.0)
NM_MAXFEV      = 200      # bounded eval count for Nelder-Mead refit
NM_PRINT_EVERY = 20


PARAM_SETS = [
    dict(label="set1",
         m1=6.87546892e+05, m2=2.01552295e+01, a=8.97791738e-01,
         e0=4.67444363e-01, p0=1.02313998e+01,
         alpha=np.array([ 1.10514090e+01, -7.36439420e-01, -3.00210231e-02,
                          1.25810088e-03,  1.77557571e-04])),
    dict(label="set2",
         m1=8.72649039e+05, m2=2.93269804e+01, a=8.15537316e-01,
         e0=3.44697545e-01, p0=1.02332324e+01,
         alpha=np.array([ 1.06387775e+01, -8.74561782e-01, -2.96890115e-02,
                         -1.53399469e-03, -7.95718643e-04])),
    dict(label="set3",
         m1=3.65672332e+05, m2=1.18495243e+01, a=4.92348924e-01,
         e0=4.06363143e-01, p0=1.31143998e+01,
         alpha=np.array([ 1.16960459e+01, -7.48812539e-01, -3.89720098e-02,
                         -6.28163526e-04,  5.25844530e-05])),
]


# -----------------------------------------------------------------------------
# Helpers — mirrored from emri_snr_correlation_test.py section 8
# -----------------------------------------------------------------------------
def make_chebyshev_template(alpha, N_out, dt, cheb_wave, t_scale):
    """Return (template, t_full, f_of_t, phase_evol) or (None,)*4 on failure.

    The Chebyshev t(f) is multiplied by ``t_scale``, anchored to t=0, truncated
    to fit in the data window, and resampled at ``dt``. The output template is
    ``env(t) * sin(phase_evol(t))`` with phase = 2*pi * f(t) * t (the
    dev_emri_search.py heuristic).
    """
    t_cheb, f_cheb = cheb_wave(*alpha, return_phase=False)
    t_cheb = t_cheb * t_scale
    if not np.all(np.diff(t_cheb) > 0):
        return None, None, None, None
    t_cheb = t_cheb - t_cheb[0]
    t_end_data = (N_out - 1) * dt
    inside = t_cheb < t_end_data
    if inside.sum() < 16:
        return None, None, None, None
    t_cheb = t_cheb[inside]
    f_cheb = f_cheb[inside]
    t_full = np.arange(0.0, t_cheb[-1], dt)
    if len(t_full) > N_out:
        t_full = t_full[:N_out]
    if len(t_full) < 16:
        return None, None, None, None
    f_of_t = CubicSpline(t_cheb, f_cheb)(t_full)
    phase_evol = 2 * np.pi * f_of_t * t_full
    env = np.abs(t_full[-1] - t_full) ** -0.25
    env[-1] = env[-2]
    template = np.zeros(N_out)
    template[: len(t_full)] = env * np.sin(phase_evol)
    return template, t_full, f_of_t, phase_evol


def build_full_invC(sens_active, fd_set, N):
    """Build a length-(N//2+1), NaN-cleaned, out-of-band-zeroed invC array on
    whatever backend ``sens_active`` lives on.

    ``sens_active`` may be a 1-D array (single channel) or 2-D (AET, shape
    (3, Nf_active)). Returns the same rank with the leading channel axis
    preserved.
    """
    _xp = get_array_module(sens_active)
    Nf_full = N // 2 + 1
    if sens_active.ndim == 1:
        out = _xp.zeros(Nf_full, dtype=sens_active.dtype)
        out[fd_set.ind_min : fd_set.ind_max + 1] = _xp.where(
            _xp.isfinite(sens_active), sens_active, 0.0
        )
        return out
    if sens_active.ndim == 2:
        nch = sens_active.shape[0]
        out = _xp.zeros((nch, Nf_full), dtype=sens_active.dtype)
        out[:, fd_set.ind_min : fd_set.ind_max + 1] = _xp.where(
            _xp.isfinite(sens_active), sens_active, 0.0
        )
        return out
    raise ValueError(f"unexpected sens_active.ndim = {sens_active.ndim}")


def normalize_template(template, invC_full_1ch, dt, df, N):
    """Scale ``template`` so <h|h> = 1 against a single-channel invC array."""
    _xp = get_array_module(invC_full_1ch)
    tpl = _xp.asarray(template)
    H = _xp.fft.rfft(tpl) * dt
    integ = _xp.where(_xp.isfinite(invC_full_1ch),
                      _xp.abs(H) ** 2 * invC_full_1ch, 0.0)
    auto = 0.5 * 4.0 * df * N * _xp.fft.irfft(integ, n=N)
    h_h = float(auto[0])
    if not np.isfinite(h_h) or h_h <= 0:
        return None, None
    return float(np.sqrt(h_h)), tpl / np.sqrt(h_h)


def matched_filter_tau(template_n, D_ch_xp, invC_full_1ch, dt, df, N):
    """|<d|h>|(tau) via FFT-IFFT with the same (1/2)*4*df*N convention."""
    _xp = get_array_module(invC_full_1ch)
    H = _xp.fft.rfft(template_n) * dt
    re = _xp.where(_xp.isfinite(invC_full_1ch),
                   _xp.conj(D_ch_xp) * H * invC_full_1ch, 0.0)
    im = _xp.where(_xp.isfinite(invC_full_1ch),
                   _xp.conj(D_ch_xp) * (-1j * H) * invC_full_1ch, 0.0)
    cr = 0.5 * 4.0 * df * N * _xp.fft.irfft(re, n=N)
    ci = 0.5 * 4.0 * df * N * _xp.fft.irfft(im, n=N)
    return _xp.abs(cr + 1j * ci)


def scan_t_scale(alpha, N, dt, df, cheb_wave, D_A_xp, invC_A_full, t_scales):
    """Return (best_t_scale, best_snr, [(t_scale, peak_snr), ...]).

    ``peak_snr`` is the max-over-tau matched filter |<d_A|h_n>|(tau) on the A
    channel for the normalised template at that t_scale.
    """
    history = []
    best_snr = -np.inf
    best_t   = float(t_scales[0])
    for t_s in t_scales:
        tpl, *_ = make_chebyshev_template(alpha, N, dt, cheb_wave,
                                          t_scale=float(t_s))
        if tpl is None:
            history.append((float(t_s), np.nan))
            continue
        _, tpl_n = normalize_template(tpl, invC_A_full, dt, df, N)
        if tpl_n is None:
            history.append((float(t_s), np.nan))
            continue
        snr_tau = matched_filter_tau(tpl_n, D_A_xp, invC_A_full, dt, df, N)
        peak = float(snr_tau.max())
        history.append((float(t_s), peak))
        if peak > best_snr:
            best_snr = peak
            best_t   = float(t_s)
    return best_t, best_snr, history


def nm_refit(alpha_seed, t_scale_seed, N, dt, df, cheb_wave,
             D_A_xp, invC_A_full,
             alpha_bounds, t_scale_bounds,
             maxfev=NM_MAXFEV):
    """Nelder-Mead refit over (alpha_0..4, t_scale) seeded from
    (``alpha_seed``, ``t_scale_seed``).

    Bounds are enforced by *clamping* the parameter vector inside the
    objective. The simplex is still free to wander outside, but each evaluated
    template uses clamped values — so the objective is continuous and bounded
    everywhere (no penalty discontinuity to confuse NM). The best-ever vertex
    is tracked explicitly so we don't lose it if NM finishes at a non-best
    simplex vertex.

    Returns (best_alpha length-5 ndarray, best_t_scale, best_snr).
    """
    bounds = list(alpha_bounds) + [t_scale_bounds]
    lo = np.array([b[0] for b in bounds])
    hi = np.array([b[1] for b in bounds])
    eval_count   = [0]
    best_peak    = [-np.inf]
    best_alpha_v = [np.asarray(alpha_seed).copy()]
    best_t_v     = [float(t_scale_seed)]

    def neg_max_tau(params):
        eval_count[0] += 1
        clamped = np.minimum(np.maximum(params, lo), hi)
        alpha_v = clamped[:5]
        t_s     = float(clamped[5])
        tpl, *_ = make_chebyshev_template(alpha_v, N, dt, cheb_wave,
                                          t_scale=t_s)
        if tpl is None:
            return 1.0
        _, tpl_n = normalize_template(tpl, invC_A_full, dt, df, N)
        if tpl_n is None:
            return 1.0
        snr_tau = matched_filter_tau(tpl_n, D_A_xp, invC_A_full, dt, df, N)
        peak = float(snr_tau.max())
        if not np.isfinite(peak):
            return 1.0
        if peak > best_peak[0]:
            best_peak[0]    = peak
            best_alpha_v[0] = alpha_v.copy()
            best_t_v[0]     = t_s
        if eval_count[0] % NM_PRINT_EVERY == 0:
            print(f"     [NM eval {eval_count[0]:4d}] "
                  f"best peak so far = {best_peak[0]:.4f}, "
                  f"current = {peak:.4f}", flush=True)
        return -peak

    x0 = np.concatenate([np.asarray(alpha_seed), [t_scale_seed]])
    res = minimize(
        neg_max_tau, x0=x0, method="Nelder-Mead",
        options=dict(maxiter=maxfev * 5, maxfev=maxfev,
                     xatol=1e-4, fatol=1e-5,
                     adaptive=True, disp=False),
    )
    print(f"     [NM done] {eval_count[0]} evals, "
          f"best peak = {best_peak[0]:.4f}, "
          f"final res.fun = {-float(res.fun):.4f}", flush=True)
    return best_alpha_v[0], best_t_v[0], best_peak[0]


def _ana_snr(channels_list, label):
    """Look up the lisatools optimal SNR for a specific single-channel
    AnalysisContainer in a list of (label, ana, sens) tuples."""
    for lbl, ana, _s in channels_list:
        if lbl == label:
            return float(ana.snr())
    return float("nan")


def lisatools_template_inner_products(template_n_xp, td_set, fd_set, channels):
    """For each (label, ana_container) in ``channels``, project the 1-channel
    Chebyshev ``template_n_xp`` into FD and return a dict of
    ``label -> (opt_snr, det_snr, overlap, complex_ip)``.

    ``overlap = det_snr / opt_snr`` (template is normalised, but we recompute
    opt_snr through lisatools for the check).
    """
    tpl_td  = TDSignal(template_n_xp[None, :], settings=td_set)
    tpl_fd  = tpl_td.fft(settings=fd_set)
    tpl_dra = DataResidualArray(tpl_fd)
    out = {}
    for label, ana in channels:
        opt_snr, det_snr = ana.template_snr(tpl_dra, phase_maximize=True)
        opt_snr = float(opt_snr)
        det_snr = float(det_snr)
        ip      = ana.template_inner_product(tpl_dra, complex=True)
        overlap = det_snr / opt_snr if opt_snr > 0 else 0.0
        out[label] = (opt_snr, det_snr, overlap, complex(ip))
    return out


# -----------------------------------------------------------------------------
# Main loop
# -----------------------------------------------------------------------------
out_dir = os.path.dirname(os.path.abspath(__file__))
summary_rows = []

for ps in PARAM_SETS:
    print("=" * 78)
    print(f"Parameter {ps['label']}:")
    print(f"  M1={ps['m1']:.4e}  M2={ps['m2']:.4e}  a={ps['a']:.4f}  "
          f"e0={ps['e0']:.4f}  p0={ps['p0']:.4f}")
    print(f"  alpha = {ps['alpha']}")

    m1, m2, a = ps["m1"], ps["m2"], ps["a"]
    p0, e0    = ps["p0"], ps["e0"]
    alpha     = ps["alpha"]

    inj_params = np.array([
        m1, m2, a, p0, e0, xI0,
        dist, qS, phiS, qK, phiK,
        Phi_phi0, Phi_theta0, Phi_r0,
    ])

    # ------------------------------------------------------------------
    # (a) Generate full-EMRI TDI injection in AET and XYZ
    # ------------------------------------------------------------------
    print("  Generating EMRI + LISA response (AET)...")
    hAET = xp.asarray(response_AET(*inj_params, convert_to_ra_dec=False,
                                   **runtime_kwargs))
    print(f"    AET shape: {hAET.shape}")

    print("  Generating EMRI + LISA response (XYZ)...")
    hXYZ = xp.asarray(response_XYZ(*inj_params, convert_to_ra_dec=False,
                                   **runtime_kwargs))
    print(f"    XYZ shape: {hXYZ.shape}")

    N = hAET.shape[-1]
    td_set = TDSettings(N, dt, force_backend=backend)

    # AET multi-channel analysis
    dra_AET  = DataResidualArray(hAET, input_signal_domain=td_set)
    sens_AET = AET1SensitivityMatrix(dra_AET.data_res_arr.settings,
                                     model="scirdv1")
    ana_AET  = AnalysisContainer(dra_AET, sens_AET)
    snr_AET  = float(ana_AET.snr())

    # XYZ multi-channel analysis
    dra_XYZ  = DataResidualArray(hXYZ, input_signal_domain=td_set)
    sens_XYZ = XYZ1SensitivityMatrix(dra_XYZ.data_res_arr.settings,
                                     model="scirdv1")
    ana_XYZ  = AnalysisContainer(dra_XYZ, sens_XYZ)
    snr_XYZ  = float(ana_XYZ.snr())

    fd_set = dra_AET.data_res_arr.settings
    df     = fd_set.df
    print(f"  lisatools optimal SNR — AET multi-channel: {snr_AET:.4f}")
    print(f"  lisatools optimal SNR — XYZ multi-channel: {snr_XYZ:.4f}")

    # Per-channel single-component AnalysisContainers — used both for the
    # t_scale scan (A channel) and for the lisatools per-channel overlap.
    per_channel = []
    sens_classes_AET = [("A", A1TDISens, 0),
                        ("E", E1TDISens, 1),
                        ("T", T1TDISens, 2)]
    sens_classes_XYZ = [("X", X1TDISens, 0),
                        ("Y", Y1TDISens, 1),
                        ("Z", Z1TDISens, 2)]

    def _make_single_channel_ana(parent_dra, sens_cls, idx, fd_set):
        sig    = FDSignal(parent_dra.data_res_arr[idx:idx + 1], settings=fd_set)
        dra1   = DataResidualArray(sig)
        sens1  = SensitivityMatrix(fd_set, [sens_cls], model="scirdv1")
        return AnalysisContainer(dra1, sens1), sens1

    channels_AET = []   # list of (label, ana_container, sens_matrix)
    for label, cls, idx in sens_classes_AET:
        ana_ch, sens_ch = _make_single_channel_ana(dra_AET, cls, idx, fd_set)
        channels_AET.append((label, ana_ch, sens_ch))
    channels_XYZ = []
    for label, cls, idx in sens_classes_XYZ:
        ana_ch, sens_ch = _make_single_channel_ana(dra_XYZ, cls, idx, fd_set)
        channels_XYZ.append((label, ana_ch, sens_ch))

    # ------------------------------------------------------------------
    # (b) FEW trajectory — truth f(t), phase(t) for the m=2 mode
    # ------------------------------------------------------------------
    print("  Getting FEW trajectory...")
    t_traj, p_traj, e_traj, x_traj, Phi_phi_t, Phi_theta_t, Phi_r_t = traj_model(
        m1, m2, a, p0, e0, xI0, T=Tobs,
    )
    Msec   = (m1 + m2) * MTSUN_SI
    Om_p, Om_t, Om_r = get_fundamental_frequencies(a, p_traj, e_traj, x_traj)
    f_phi      = Om_p / (2 * np.pi * Msec)
    f_emri_22  = 2.0 * f_phi
    phase_emri = 2.0 * Phi_phi_t

    # ------------------------------------------------------------------
    # (c) Chebyshev: snap fmin/fmax to the rfft grid, build the template,
    #     scan t_scale over A channel, refine around the winner.
    # ------------------------------------------------------------------
    fd_full  = np.fft.rfftfreq(N, dt)
    ind_min  = np.where(fd_full > CHEB_FMIN_REQ)[0][0]
    ind_max  = np.where(fd_full < CHEB_FMAX_REQ)[0][-1]
    fmin     = float(fd_full[ind_min])
    fmax     = float(fd_full[ind_max])
    cheb_wave = ChebyshevWave(CHEB_NPTS, fmin, fmax)

    # invC_A on the full rfft grid (NaN-cleaned, out-of-band masked)
    invC_A_full = build_full_invC(channels_AET[0][2].invC[0], fd_set, N)
    D_A_xp      = xp.fft.rfft(hAET[0]) * dt

    print(f"  Scanning Chebyshev t_scale over "
          f"[{T_SCALE_COARSE[0]:.0f}, {T_SCALE_COARSE[-1]:.0f}] ...")
    best_t_coarse, best_snr_coarse, _ = scan_t_scale(
        alpha, N, dt, df, cheb_wave, D_A_xp, invC_A_full, T_SCALE_COARSE,
    )
    # 1st refine around coarse winner
    step1 = T_SCALE_COARSE[1] - T_SCALE_COARSE[0]
    t_refine1 = np.linspace(max(10.0, best_t_coarse - step1),
                            best_t_coarse + step1, T_SCALE_REFINE)
    best_t1, best_snr1, _ = scan_t_scale(
        alpha, N, dt, df, cheb_wave, D_A_xp, invC_A_full, t_refine1,
    )
    # 2nd refine around 1st-refine winner
    step2 = t_refine1[1] - t_refine1[0]
    t_refine2 = np.linspace(max(10.0, best_t1 - step2),
                            best_t1 + step2, T_SCALE_REFINE2)
    best_t, best_snr, _ = scan_t_scale(
        alpha, N, dt, df, cheb_wave, D_A_xp, invC_A_full, t_refine2,
    )
    print(f"  Best t_scale  = {best_t:.2f}   "
          f"(max-tau |<d_A|h>| = {best_snr:.4f})")

    # ------------------------------------------------------------------
    # Inner closure: build template at (alpha, t_scale), compute per-channel
    # lisatools inner products + max-tau matched filter, and save a plot
    # with the Chebyshev curves shifted by tau* (A-channel matched-filter
    # peak) so they overlay where the matched filter wants them.
    # ------------------------------------------------------------------
    def evaluate_and_plot(alpha_in, t_scale_in, label_suffix, pass_label,
                          best_snr_hint=None):
        template_td, t_full_c, f_full_c, phase_full_c = make_chebyshev_template(
            alpha_in, N, dt, cheb_wave, t_scale=t_scale_in,
        )
        if template_td is None:
            print(f"  [{pass_label}] template invalid, skipping plot")
            return None
        h_norm, template_n = normalize_template(template_td, invC_A_full,
                                                dt, df, N)
        if template_n is None:
            print(f"  [{pass_label}] non-finite <h|h>, skipping plot")
            return None
        print(f"  [{pass_label}] <h|h>_A^0.5 = {h_norm:.4e}")

        # Raw (t, f, proper phase) for the t-f / phase overlay
        t_raw_full, f_raw_full = cheb_wave(*alpha_in, return_phase=False)
        t_raw_full = t_raw_full * t_scale_in
        t_raw_full = t_raw_full - t_raw_full[0]
        phase_proper = 2 * np.pi * cumulative_simpson(
            f_raw_full, x=t_raw_full, initial=0.0,
        )

        # lisatools per-channel inner products (tau=0)
        chan_aet_for_ips = [(lbl, ana) for (lbl, ana, _s) in channels_AET]
        chan_xyz_for_ips = [(lbl, ana) for (lbl, ana, _s) in channels_XYZ]
        ip_AET = lisatools_template_inner_products(template_n, td_set, fd_set,
                                                   chan_aet_for_ips)
        ip_XYZ = lisatools_template_inner_products(template_n, td_set, fd_set,
                                                   chan_xyz_for_ips)

        # Per-channel matched filter (max over tau)
        snr_tau_per_channel = {}
        snr_tau_peaks       = {}
        for label, ana_ch, sens_ch in channels_AET + channels_XYZ:
            invC_ch = build_full_invC(sens_ch.invC[0], fd_set, N)
            if label in ("A", "E", "T"):
                idx_ = sens_classes_AET[["A", "E", "T"].index(label)][2]
                data_td = hAET[idx_]
            else:
                idx_ = sens_classes_XYZ[["X", "Y", "Z"].index(label)][2]
                data_td = hXYZ[idx_]
            D_xp = xp.fft.rfft(data_td) * dt
            _, tpl_n_ch = normalize_template(template_td, invC_ch, dt, df, N)
            if tpl_n_ch is None:
                snr_tau_peaks[label] = (np.nan, np.nan, np.nan)
                continue
            snr_tau_ch = matched_filter_tau(tpl_n_ch, D_xp, invC_ch,
                                            dt, df, N)
            opt_data = ana_ch.snr()
            peak    = float(snr_tau_ch.max())
            i_peak  = int(snr_tau_ch.argmax())
            tau_pk  = (i_peak if i_peak <= N // 2 else i_peak - N) * dt
            overlap_max = peak / float(opt_data) if opt_data > 0 else 0.0
            snr_tau_per_channel[label] = snr_tau_ch
            snr_tau_peaks[label] = (peak, tau_pk, overlap_max)

        print(f"  [{pass_label}] Per-channel matched filter "
              "(max over tau, lisatools template_snr at tau=0):")
        print("    label  |   SNR_d (inj)   max|<d|h>|     tau*     "
              "overlap_max   lt_det_snr  lt_overlap0")
        for label in ["A", "E", "T", "X", "Y", "Z"]:
            if label in ("A", "E", "T"):
                ip_dict = ip_AET
                snr_d_  = _ana_snr(channels_AET, label)
            else:
                ip_dict = ip_XYZ
                snr_d_  = _ana_snr(channels_XYZ, label)
            opt_snr_ch, det_snr_ch, overlap0, _ = ip_dict[label]
            peak, tau_pk, overlap_max = snr_tau_peaks.get(
                label, (np.nan, np.nan, np.nan)
            )
            print(f"     {label}     |   {snr_d_:8.4f}   {peak:8.4f}   "
                  f"{tau_pk:11.1f}   {overlap_max:8.4f}   "
                  f"{det_snr_ch:8.4f}   {overlap0:8.4f}")
            summary_rows.append(dict(
                label=ps["label"], pass_label=pass_label, channel=label,
                snr_d=snr_d_, peak=peak, tau_pk=tau_pk,
                overlap_max=overlap_max,
                lt_opt_snr=opt_snr_ch, lt_det_snr=det_snr_ch,
                lt_overlap0=overlap0,
            ))

        # A-channel best-fit time shift — used to overlay the Chebyshev plot
        tau_shift_A = (snr_tau_peaks["A"][1]
                       if ("A" in snr_tau_peaks
                           and np.isfinite(snr_tau_peaks["A"][1]))
                       else 0.0)

        # Phase offset: subtract the Chebyshev's phase at the data-frame t=0
        # so the curve isn't artificially offset by a huge constant when the
        # shift is large. Outside the chebyshev's natural range np.interp
        # uses the boundary values, which is OK for visualisation.
        phase_offset_cheb = float(np.interp(
            -tau_shift_A, t_raw_full, phase_proper,
            left=phase_proper[0], right=phase_proper[-1],
        ))
        phase_offset_heur = float(np.interp(
            -tau_shift_A, t_full_c, phase_full_c,
            left=phase_full_c[0], right=phase_full_c[-1],
        ))

        # Plot config
        t_emri_end   = float(t_traj[-1])
        t_cheb_end   = float(t_raw_full[-1])
        t_xlim_lo    = min(0.0, tau_shift_A) * 1.05
        t_xlim_hi    = max(t_emri_end * 1.4, tau_shift_A + t_cheb_end * 1.05)

        nperseg, noverlap = 1024, 768
        f_s_inj, t_s_inj, S_inj = spectrogram(
            _to_cpu(hAET[0]), fs=1.0 / dt,
            nperseg=nperseg, noverlap=noverlap,
        )
        f_s_tpl, t_s_tpl, S_tpl = spectrogram(
            _to_cpu(template_td), fs=1.0 / dt,
            nperseg=nperseg, noverlap=noverlap,
        )

        # Apply tau* shift to Chebyshev's t-axis
        t_cheb_plot   = t_raw_full + tau_shift_A
        t_full_c_plot = t_full_c + tau_shift_A
        t_s_tpl_plot  = t_s_tpl + tau_shift_A

        fig = plt.figure(figsize=(15.5, 11.5))
        gs = fig.add_gridspec(3, 2, hspace=0.36, wspace=0.24)

        # (1) f(t) overlay (Chebyshev shifted to tau*)
        ax = fig.add_subplot(gs[0, 0])
        ax.plot(t_traj, f_emri_22, lw=2.0, color="C0",
                label="EMRI 2·f_phi(t) [FEW traj]")
        ax.plot(t_cheb_plot, f_raw_full, ls="--", lw=1.4, color="C1",
                label=(f"Chebyshev t(f) + tau*  "
                       f"(t_scale={t_scale_in:.2f}, "
                       f"tau*={tau_shift_A:.2e}s)"))
        ax.axhline(CHEB_FMIN_REQ, color="gray", ls=":", lw=0.6, alpha=0.6)
        ax.axhline(CHEB_FMAX_REQ, color="gray", ls=":", lw=0.6, alpha=0.6)
        ax.axvline(0.0, color="k", lw=0.4, alpha=0.3)
        ax.set_xlabel("t [s]"); ax.set_ylabel("f [Hz]")
        ax.set_title("f(t) — EMRI (2,2,0,0) vs Chebyshev (shifted)")
        ax.set_xlim(t_xlim_lo, t_xlim_hi)
        ax.set_ylim(0.0, max(float(f_emri_22.max()),
                             float(f_raw_full.max())) * 1.15)
        ax.legend(fontsize=8); ax.grid(alpha=0.3)

        # (2) phase(t) overlay
        ax = fig.add_subplot(gs[0, 1])
        ax.plot(t_traj, phase_emri - phase_emri[0], lw=2.0, color="C0",
                label="EMRI 2·Phi_phi(t)")
        ax.plot(t_cheb_plot, phase_proper - phase_offset_cheb,
                lw=1.4, ls="--", color="C1",
                label="Chebyshev cumulative-integral phase (shifted)")
        ax.plot(t_full_c_plot, phase_full_c - phase_offset_heur,
                lw=0.9, ls=":", color="C3", alpha=0.8,
                label="Chebyshev 2·pi·f(t)·t (heuristic, shifted)")
        ax.axvline(0.0, color="k", lw=0.4, alpha=0.3)
        ax.set_xlabel("t [s]"); ax.set_ylabel("phase [rad]")
        ax.set_title("phase(t) — EMRI vs Chebyshev")
        ax.set_xlim(t_xlim_lo, t_xlim_hi)
        ax.legend(fontsize=8); ax.grid(alpha=0.3)

        # (3) Spectrogram of EMRI A channel
        ax = fig.add_subplot(gs[1, 0])
        pcm = ax.pcolormesh(t_s_inj, f_s_inj,
                            10 * np.log10(S_inj + 1e-300),
                            shading="auto", cmap="viridis")
        ax.plot(t_traj, f_emri_22, color="white", lw=0.7, alpha=0.6,
                label="2·f_phi(t) overlay")
        ax.set_xlim(t_xlim_lo, t_xlim_hi)
        ax.set_ylim(0.0, max(float(f_emri_22.max()),
                             float(f_raw_full.max())) * 1.15)
        ax.set_xlabel("t [s]"); ax.set_ylabel("f [Hz]")
        ax.set_title("Spectrogram — EMRI A channel  [dB]")
        ax.legend(fontsize=8, loc="upper left")
        plt.colorbar(pcm, ax=ax)

        # (4) Spectrogram of Chebyshev template (also tau-shifted)
        ax = fig.add_subplot(gs[1, 1])
        pcm = ax.pcolormesh(t_s_tpl_plot, f_s_tpl,
                            10 * np.log10(S_tpl + 1e-300),
                            shading="auto", cmap="viridis")
        ax.plot(t_cheb_plot, f_raw_full, color="white", lw=0.7, alpha=0.6,
                label="Chebyshev t(f) overlay (shifted)")
        ax.set_xlim(t_xlim_lo, t_xlim_hi)
        ax.set_ylim(0.0, max(float(f_emri_22.max()),
                             float(f_raw_full.max())) * 1.15)
        ax.set_xlabel("t [s]"); ax.set_ylabel("f [Hz]")
        ax.set_title("Spectrogram — Chebyshev template (shifted)  [dB]")
        ax.legend(fontsize=8, loc="upper left")
        plt.colorbar(pcm, ax=ax)

        # (5/6) Matched-filter SNR(tau) for AET and XYZ
        lags_full_np = (np.arange(N) - N // 2) * dt
        ax = fig.add_subplot(gs[2, 0])
        for lbl in ("A", "E", "T"):
            if lbl in snr_tau_per_channel:
                ax.plot(lags_full_np,
                        _to_cpu(xp.fft.fftshift(snr_tau_per_channel[lbl])),
                        lw=0.8, label=lbl)
        ax.set_xlabel("time shift tau [s]"); ax.set_ylabel("|<d|h_n>|(tau)")
        ax.set_title("Matched filter — AET channels")
        ax.legend(); ax.grid(alpha=0.3)

        ax = fig.add_subplot(gs[2, 1])
        for lbl in ("X", "Y", "Z"):
            if lbl in snr_tau_per_channel:
                ax.plot(lags_full_np,
                        _to_cpu(xp.fft.fftshift(snr_tau_per_channel[lbl])),
                        lw=0.8, label=lbl)
        ax.set_xlabel("time shift tau [s]"); ax.set_ylabel("|<d|h_n>|(tau)")
        ax.set_title("Matched filter — XYZ channels")
        ax.legend(); ax.grid(alpha=0.3)

        title_snr = best_snr_hint if best_snr_hint is not None else \
                    snr_tau_peaks.get("A", (np.nan,))[0]
        fig.suptitle(
            f"{ps['label']} [{pass_label}]:  M1={m1:.3e}  M2={m2:.3e}  "
            f"a={a:.3f}  e0={e0:.3f}  p0={p0:.3f}\n"
            f"SNR_AET={snr_AET:.2f}   SNR_XYZ={snr_XYZ:.2f}   "
            f"t_scale={t_scale_in:.2f}   "
            f"max|<d_A|h>|={title_snr:.3f}   tau*={tau_shift_A:.2e}s",
            fontsize=10,
        )
        suffix = f"_{label_suffix}" if label_suffix else ""
        out_path = os.path.join(
            out_dir, f"compare_emri_chebyshev_{ps['label']}{suffix}.png",
        )
        fig.savefig(out_path, dpi=130, bbox_inches="tight")
        plt.close(fig)
        print(f"  [{pass_label}] Saved plot -> {out_path}")

        return dict(
            alpha=alpha_in, t_scale=t_scale_in,
            snr_tau_peaks=snr_tau_peaks,
        )

    # ------------------------------------------------------------------
    # (d) First pass — user's alphas at the best t_scale from the scan
    # ------------------------------------------------------------------
    evaluate_and_plot(alpha, best_t, "user", "user",
                      best_snr_hint=best_snr)

    # ------------------------------------------------------------------
    # (e) Second pass — DE refit over (alpha_0..4, t_scale), then re-plot
    # ------------------------------------------------------------------
    print("  Running Nelder-Mead refit over (alpha, t_scale) "
          "seeded at user values ...")
    de_alpha, de_t, de_snr = nm_refit(
        alpha, best_t, N, dt, df, cheb_wave, D_A_xp, invC_A_full,
        alpha_bounds=DE_ALPHA_BOUNDS,
        t_scale_bounds=DE_T_SCALE_BOUNDS,
        maxiter=NM_MAXITER,
    )
    print(f"  NM result:")
    print(f"     alpha   = {de_alpha}")
    print(f"     t_scale = {de_t:.4f}")
    print(f"     max-tau |<d_A|h>| = {de_snr:.4f}")
    evaluate_and_plot(de_alpha, de_t, "refit", "refit",
                      best_snr_hint=de_snr)

# -----------------------------------------------------------------------------
# Final summary
# -----------------------------------------------------------------------------
print("=" * 78)
print("Summary (lisatools per-channel inner products + max-tau matched filter)")
print("=" * 78)
header = ("set      pass     ch    SNR_d    lt_opt    lt_det    overlap0   "
          "mf_peak     tau*[s]      overlap_max")
print(header)
for row in summary_rows:
    print(f"{row['label']:6s}  {row['pass_label']:6s}  {row['channel']:2s}   "
          f"{row['snr_d']:8.4f}  {row['lt_opt_snr']:8.4f}  "
          f"{row['lt_det_snr']:8.4f}  {row['lt_overlap0']:9.4f}  "
          f"{row['peak']:8.4f}  {row['tau_pk']:11.1f}   "
          f"{row['overlap_max']:9.4f}")

print("Done.")
