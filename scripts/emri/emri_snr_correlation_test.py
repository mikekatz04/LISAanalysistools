"""
Cross-check the lisatools optimal SNR against the FFT/IFFT time-slide
matched-filter SNR for an EMRI signal projected through the LISA response,
then fit a Chebyshev t(f) template (lisatools.sources.emri.chebyshevwave)
to that injection by maximising the time-marginalised matched filter.

Setup:
  * Single-mode EMRI: only (l, m, k, n) = (2, 2, 0, 0) (with the m -> -m
    partner so the time-domain waveform is real).
  * Tobs ~ 1 yr, dt = 15 s, 1st-generation TDI via fastlisaresponse.
  * Two TDI channel choices: AET (diagonal noise) and XYZ (3x3 CSD).
  * Same template = data, so the matched filter SNR(tau) peaks at tau = 0
    and equals the lisatools optimal SNR there.

For each TDI choice we:
  (1) compute SNR via lisatools.AnalysisContainer.snr(),
  (2) compute SNR(tau) via 4 * df * N * irfft( H^* invC H ) / sqrt<h|h>,
      and report the value at tau = 0 plus the peak.

Then, against the A channel of the AET injection:
  (3) build a 1-D Chebyshev sin(phase(t)) * envelope template, normalise it
      to <h|h>_A = 1, and verify that lisatools.template_snr and the
      FFT/IFFT cross-correlation agree at tau = 0;
  (4) run scipy.optimize.differential_evolution over the 5 alpha
      coefficients with the inner objective being
      -max_tau |<d_A, h_norm>(tau)|, i.e. the time-marginalised matched
      filter SNR.
"""

import os
import numpy as np
import matplotlib.pyplot as plt
plt.rcParams['text.usetex'] = False

from scipy.interpolate import CubicSpline
from scipy.optimize import differential_evolution

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
)
from lisatools.domains import TDSettings, TDSignal, FDSettings, FDSignal
from lisatools.sources.emri.chebyshevwave import ChebyshevWave

from fastlisaresponse import ResponseWrapper
from fastlisaresponse.tdiconfig import TDIConfig

from few.waveform import GenerateEMRIWaveform


# -----------------------------------------------------------------------------
# 1. Time grid + orbits + TDI generation config
# -----------------------------------------------------------------------------
# Toggle CPU vs GPU here. Valid values include "cpu", "cuda", "gpu",
# "cuda11x", "cuda12x", "cuda13x". Everything downstream — orbits, EMRI
# generator, response wrapper, lisatools containers, and the chebyshev
# matched-filter math — picks up this choice via ``force_backend=backend``
# or the resolved ``xp`` module below.
backend = "cuda12x"

backend_obj = lisatools.get_backend(backend)
xp = backend_obj.xp  # numpy on CPU, cupy on a CUDA backend


def _to_cpu(arr):
    """Return a numpy view of an array regardless of backend (no-op on CPU)."""
    if hasattr(arr, "get") and not isinstance(arr, np.ndarray):
        return arr.get()
    return np.asarray(arr)


dt      = 15.0
Tobs    = 6.0 / 12.0      # ~ 2 months in years -- short for speed
t_start = 0.5 * YRSID_SI  # arbitrary epoch into the orbits

orbits     = EqualArmlengthOrbits(force_backend=backend)
tdi_config = TDIConfig("1st generation", force_backend=backend)

# -----------------------------------------------------------------------------
# 2. EMRI generator (FastKerrEccentricEquatorialFlux), single-mode at call time
# -----------------------------------------------------------------------------
emri_gen = GenerateEMRIWaveform(
    "FastKerrEccentricEquatorialFlux",
    return_list=False,
    inspiral_kwargs={"DENSE_STEPPING": 0, "max_init_len": int(1e4),
                     "force_backend": "cpu"},
    sum_kwargs={"pad_output": True},
    frame="detector",
    mode_selector_kwargs={"mode_selection_threshold": 1e-5},
    force_backend=backend,
)

# Common ResponseWrapper kwargs
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

# -----------------------------------------------------------------------------
# 3. Source parameters (matching the existing test script in this workspace)
# -----------------------------------------------------------------------------
m1, m2, a    = 534673.46222, 9.508330389, 0.2
p0, e0, xI0  = 8.57389267, 0.015584839, 1.0
dist         = 1.06696103277                                   # Gpc
beta, lam    = 0.50015698, 4.707806421         # ecliptic
qS, phiS     = np.pi / 2 - beta, lam                 # polar / azimuth
qK, phiK     = 0.7247331791, 4.039822998
Phi_phi0     = 2.15498605
Phi_theta0   = 0.4543828072
Phi_r0       = 1.124806044

inj_params = np.array([
    m1, m2, a, p0, e0, xI0,
    dist, qS, phiS, qK, phiK,
    Phi_phi0, Phi_theta0, Phi_r0,
])

# Restrict to a SINGLE harmonic mode
runtime_kwargs = dict(
    mode_selection=[(2, 2, 0, 0)],
    include_minus_mkn=True,
)

# -----------------------------------------------------------------------------
# 4. Generate the TDI signals
# -----------------------------------------------------------------------------
print(f"Generating EMRI + LISA response (AET, single mode) on backend={backend!r}...")
hAET = xp.asarray(response_AET(*inj_params, convert_to_ra_dec=False,
                               **runtime_kwargs))
print(f"  AET shape: {hAET.shape}")

print(f"Generating EMRI + LISA response (XYZ, single mode) on backend={backend!r}...")
hXYZ = xp.asarray(response_XYZ(*inj_params, convert_to_ra_dec=False,
                               **runtime_kwargs))
print(f"  XYZ shape: {hXYZ.shape}")

assert hAET.shape == hXYZ.shape, "AET and XYZ outputs must have the same shape"
N = hAET.shape[-1]

# -----------------------------------------------------------------------------
# 5. lisatools SNR via AnalysisContainer (auto TD -> FD)
# -----------------------------------------------------------------------------
td_set = TDSettings(N, dt, force_backend=backend)

dra_AET   = DataResidualArray(hAET, input_signal_domain=td_set)
sens_AET  = AET1SensitivityMatrix(dra_AET.data_res_arr.settings,
                                  model="scirdv1")
ana_AET   = AnalysisContainer(dra_AET, sens_AET)
snr_AET_lt = ana_AET.snr()

dra_XYZ   = DataResidualArray(hXYZ, input_signal_domain=td_set)
sens_XYZ  = XYZ1SensitivityMatrix(dra_XYZ.data_res_arr.settings,
                                  model="scirdv1")
ana_XYZ   = AnalysisContainer(dra_XYZ, sens_XYZ)
snr_XYZ_lt = ana_XYZ.snr()

print(f"\nlisatools optimal SNR:")
print(f"  AET:  {snr_AET_lt:.6f}")
print(f"  XYZ:  {snr_XYZ_lt:.6f}")

# -----------------------------------------------------------------------------
# 6. FFT / IFFT time-slide matched-filter SNR(tau)
# -----------------------------------------------------------------------------
def correlation_snr_curve(dra, sens, snr_lt, N):
    """Return (lags, SNR(tau)) using the same H, invC, df as lisatools.

    Backend-agnostic: ``H`` and ``invC`` come straight off the lisatools
    containers, so they live on the backend chosen by ``td_set`` /
    ``sens_*``. All FFT math is done in that same backend's ``xp``.
    """
    fd_settings = dra.data_res_arr.settings
    df          = fd_settings.df
    H           = dra.data_res_arr.arr               # (nchannels, Nf_active)
    invC        = sens.invC                          # (...,)+ (Nf_active,)

    _xp = get_array_module(H)

    # Build the cross-spectral integrand on the *full* one-sided rfft grid
    # (length N//2 + 1) so that irfft gives back length-N time samples.
    Nf_full   = N // 2 + 1
    f0        = fd_settings.ind_min                  # active band start bin
    f1        = fd_settings.ind_max + 1              # active band stop bin

    # Match lisatools' first-NaN-drop behaviour: zero out a bin whose invC
    # is NaN (this is how the LISA sensitivity diverges at f=0).
    if invC.ndim == 2:                               # AET (nchannels, Nf)
        # collapse channel sum first: integrand_active(f) = sum_c H_c* H_c invC_c
        integrand_active = _xp.sum(_xp.conj(H) * H * invC, axis=0)
        bad = _xp.isnan(integrand_active) | ~_xp.isfinite(integrand_active)
        integrand_active = _xp.where(bad, 0.0, integrand_active)
    elif invC.ndim == 3:                             # XYZ (3, 3, Nf)
        integrand_active = _xp.einsum("if,ijf,jf->f",
                                      _xp.conj(H), invC, H)
        bad = _xp.isnan(integrand_active) | ~_xp.isfinite(integrand_active)
        integrand_active = _xp.where(bad, 0.0, integrand_active)
    else:
        raise ValueError(f"unexpected invC.ndim = {invC.ndim}")

    # Pad zeros outside the FDSettings active band so we can irfft length N
    integrand = _xp.zeros(Nf_full, dtype=integrand_active.dtype)
    integrand[f0:f1] = integrand_active

    # Time-slide via inverse rDFT.
    # corr[n] = (h|h)(tau_n) for tau_n = n * dt
    # factor of 2 difference with lisatools so adding 1/2 factor
    corr = (1. / 2.) * 4.0 * df * N * _xp.fft.irfft(integrand, n=N)

    snr_tau = corr / snr_lt   # divide by sqrt<h|h>
    lags    = (_xp.arange(N) - N // 2) * dt
    return lags, _xp.fft.fftshift(snr_tau)

lags_AET, snr_AET_tau = correlation_snr_curve(dra_AET, sens_AET, snr_AET_lt, N)
lags_XYZ, snr_XYZ_tau = correlation_snr_curve(dra_XYZ, sens_XYZ, snr_XYZ_lt, N)

# tau = 0 sits at index N // 2 after fftshift
i0 = N // 2

print(f"\nFFT-IFFT cross-correlation SNR(tau = 0):")
print(f"  AET:  {float(snr_AET_tau[i0]):.6f}    (peak |SNR| = "
      f"{float(xp.max(xp.abs(snr_AET_tau))):.6f})")
print(f"  XYZ:  {float(snr_XYZ_tau[i0]):.6f}    (peak |SNR| = "
      f"{float(xp.max(xp.abs(snr_XYZ_tau))):.6f})")

print(f"\nDifference SNR(tau=0) - lisatools SNR:")
print(f"  AET:  {float(snr_AET_tau[i0]) - snr_AET_lt: .3e}")
print(f"  XYZ:  {float(snr_XYZ_tau[i0]) - snr_XYZ_lt: .3e}")

# -----------------------------------------------------------------------------
# 7. Plot
# -----------------------------------------------------------------------------
# Matplotlib only takes CPU arrays; pull whatever the backend produced
# back to numpy for plotting.
lags_AET_np    = _to_cpu(lags_AET)
snr_AET_tau_np = _to_cpu(snr_AET_tau)
lags_XYZ_np    = _to_cpu(lags_XYZ)
snr_XYZ_tau_np = _to_cpu(snr_XYZ_tau)
plt.rcParams['text.usetex'] = False
fig, ax = plt.subplots(2, 1, figsize=(9, 6), sharex=True)
ax[0].plot(lags_AET_np, snr_AET_tau_np, lw=0.7)
ax[0].axhline( snr_AET_lt, color="gray", ls=":", label=f"lisatools = {snr_AET_lt:.3f}")
ax[0].axhline(-snr_AET_lt, color="gray", ls=":")
ax[0].axvline(0.0, color="k", ls="--", alpha=0.3)
ax[0].set_ylabel("SNR(tau)")
ax[0].set_title("AET (diagonal noise)")
ax[0].legend(loc="upper right")

ax[1].plot(lags_XYZ_np, snr_XYZ_tau_np, lw=0.7, color="C1")
ax[1].axhline( snr_XYZ_lt, color="gray", ls=":", label=f"lisatools = {snr_XYZ_lt:.3f}")
ax[1].axhline(-snr_XYZ_lt, color="gray", ls=":")
ax[1].axvline(0.0, color="k", ls="--", alpha=0.3)
ax[1].set_xlabel("time shift tau [s]")
ax[1].set_ylabel("SNR(tau)")
ax[1].set_title("XYZ (full 3x3 CSD)")
ax[1].legend(loc="upper right")

plt.tight_layout()
out_png = os.path.join(os.path.dirname(__file__), "emri_snr_correlation.png")
plt.savefig(out_png, dpi=130)
print(f"\nSaved plot -> {out_png}")
plt.close(fig)

# =============================================================================
# 8. Chebyshev t(f) template — fit it to the A channel of the AET injection
# =============================================================================
#
# The Chebyshev parametrisation (lisatools.sources.emri.chebyshevwave) gives
# t(f) on a log-f grid in [fmin, fmax] from 5 coefficients alpha_0..alpha_4.
# It's normalised so t = 0 at the geometric mean fmid; downstream we shift it
# so the signal starts at t = 0 in the data window.  Following dev_emri_search,
# the natural-unit output is rescaled by ``CHEB_T_SCALE`` seconds and the
# amplitude envelope is the chirp-like (t_end - t)^(-1/4).
#
# We fit *one* template (a 1-D real time series, not TDI-projected) against
# the A channel of the AET injection — see dev_emri_search.py for the same
# per-channel matched-filter approach against X, Y, Z.
# -----------------------------------------------------------------------------
print("\n" + "=" * 78)
print("Chebyshev template fit (A channel of AET injection)")
print("=" * 78)

# --- Chebyshev config ---------------------------------------------------------
CHEB_FMIN   = 2.0e-3        # Hz — start of log-f grid (matches dev_emri_search)
CHEB_FMAX   = 5.0e-3        # Hz — end of log-f grid
CHEB_NPTS   = 32001         # number of points on the log-f grid

# Snap fmin/fmax to the rfft grid (matches the dev_emri_search convention)
fd_full = np.fft.rfftfreq(N, dt)
_ind_min = np.where(fd_full > CHEB_FMIN)[0][0]
_ind_max = np.where(fd_full < CHEB_FMAX)[0][-1]
CHEB_FMIN = float(fd_full[_ind_min])
CHEB_FMAX = float(fd_full[_ind_max])

cheb_wave = ChebyshevWave(CHEB_NPTS, CHEB_FMIN, CHEB_FMAX)

# 6-parameter search box: 5 chebyshev alphas + 1 overall t-scale on the
# chebyshev t output. Alpha ranges copied from dev_emri_search.py; the
# t_scale bracket is centred on dev_emri_search's hardcoded 600.0.
ALPHA_BOUNDS = [
    (10.6,    11.7),
    (-0.97,  -0.66),
    (-0.052, -0.005),
    (-0.0041, +0.0018),
    (-0.0010, +0.0003),
]
T_SCALE_BOUND  = (200.0, 2000.0)
PARAM_BOUNDS   = ALPHA_BOUNDS + [T_SCALE_BOUND]
N_ALPHA        = len(ALPHA_BOUNDS)
T_SCALE_DEFAULT = 600.0


def make_chebyshev_template(alpha, N_out, dt, cheb_wave, t_scale=T_SCALE_DEFAULT):
    """Return a length-``N_out`` real time-domain template ``sin(phase) * env``.

    ``alpha`` is the 5-element Chebyshev coefficient vector. ``t_scale`` is an
    overall multiplier on the chebyshev t(f) output (the empirical 600.0 from
    dev_emri_search.py is the default).

    Returns ``None`` if the supplied ``(alpha, t_scale)`` produces a
    non-monotonic t(f) or a track that does not fit in the data window.
    """
    t_cheb, f_cheb = cheb_wave(*alpha, return_phase=False)
    t_cheb = t_cheb * t_scale

    # require monotonic t(f) along the f-grid
    if not np.all(np.diff(t_cheb) > 0):
        return None

    # cheb_wave puts t = 0 at fmid, so t_cheb spans [-T_pre, +T_post] roughly.
    # Anchor the chirp to t = 0 in the data window; the matched-filter time
    # shift will then place it at the right epoch in the data.
    t_cheb = t_cheb - t_cheb[0]

    # Truncate to the data window (chirps longer than the data are useless)
    t_end_data = (N_out - 1) * dt
    inside = t_cheb < t_end_data
    if inside.sum() < 16:
        return None
    t_cheb = t_cheb[inside]
    f_cheb = f_cheb[inside]

    # Even time grid (starts at 0, ends just before t_cheb[-1])
    t_full = np.arange(0.0, t_cheb[-1], dt)
    if len(t_full) < 16:
        return None
    # Defensive: never exceed the data window (float arange edge cases)
    if len(t_full) > N_out:
        t_full = t_full[:N_out]

    f_of_t = CubicSpline(t_cheb, f_cheb)(t_full)

    # Match dev_emri_search.py: phase = 2*pi * f(t) * (t - t_start)
    phase_evol = 2 * np.pi * f_of_t * t_full

    # Chirp-like (t_end - t)^(-1/4) envelope, with the endpoint regularised
    env = np.abs(t_full[-1] - t_full) ** (-0.25)
    env[-1] = env[-2]

    template = np.zeros(N_out)
    L = len(t_full)
    template[:L] = env * np.sin(phase_evol)
    return template


def normalize_template(template, invC_A_full_grid, dt, df, N):
    """Scale ``template`` so <h|h>_A = 1 against the supplied (full-rfft-grid)
    inverse-PSD array, using the same factor-of-(1/2)*4*df*N convention as
    section 6 above.

    ``template`` may be either a numpy or backend (``xp``) array; the work
    is done on whichever backend ``invC_A_full_grid`` lives on, and the
    returned ``template_normed`` is on that same backend.

    Returns (snr_h_h_sqrt as Python float, template_normed on xp).
    """
    _xp = get_array_module(invC_A_full_grid)
    template_xp = _xp.asarray(template)
    H = _xp.fft.rfft(template_xp) * dt
    integrand_auto = _xp.where(_xp.isfinite(invC_A_full_grid),
                               _xp.abs(H) ** 2 * invC_A_full_grid, 0.0)
    auto = (1.0 / 2.0) * 4.0 * df * N * _xp.fft.irfft(integrand_auto, n=N)
    h_h = float(auto[0])
    if not np.isfinite(h_h) or h_h <= 0:
        return None, None
    snr = np.sqrt(h_h)
    return snr, template_xp / snr


def cross_correlation_snr_tau(template_normed, D_xp, invC_A_full_grid,
                              dt, df, N):
    """Phase-maximised SNR(tau) of a *normalised* template against a single
    time-domain data channel, using the same convention as section 6 above.

    ``D_xp`` is the precomputed ``rfft(h_data_td) * dt`` on the same backend
    as ``invC_A_full_grid`` (precomputing once avoids redoing the length-N
    data FFT every objective call).

    Returns the *unshifted* irfft-natural ordering (i.e. tau = 0 at index 0,
    tau increasing then wrapping to negative); the array lives on the same
    backend as ``invC_A_full_grid``.
    """
    _xp = get_array_module(invC_A_full_grid)
    template_xp = _xp.asarray(template_normed)
    H = _xp.fft.rfft(template_xp) * dt
    integrand_re = _xp.where(_xp.isfinite(invC_A_full_grid),
                             _xp.conj(D_xp) * H * invC_A_full_grid, 0.0)
    integrand_im = _xp.where(_xp.isfinite(invC_A_full_grid),
                             _xp.conj(D_xp) * (-1j * H) * invC_A_full_grid, 0.0)
    corr_re = (1.0 / 2.0) * 4.0 * df * N * _xp.fft.irfft(integrand_re, n=N)
    corr_im = (1.0 / 2.0) * 4.0 * df * N * _xp.fft.irfft(integrand_im, n=N)
    return _xp.abs(corr_re + 1j * corr_im)


# -----------------------------------------------------------------------------
# 8a. Build the A-channel-only analysis container we'll fit against
# -----------------------------------------------------------------------------
hA_td      = hAET[0]                     # the A channel of the injection (xp)
fd_set     = dra_AET.data_res_arr.settings  # FDSettings from sec. 5
df         = fd_set.df

# invC on the *full* one-sided rfft grid, on the active backend
# (NaN-cleaned, out-of-band masked). sens_AET.invC has shape
# (3, Nf_active) keyed to fd_set's active band; it lives on whatever
# backend fd_set was created with.
Nf_full          = N // 2 + 1
invC_A_active    = sens_AET.invC[0]                          # xp.array, (Nf_active,)
invC_A_full      = xp.zeros(Nf_full, dtype=invC_A_active.dtype)
invC_A_full[fd_set.ind_min:fd_set.ind_max + 1] = xp.where(
    xp.isfinite(invC_A_active), invC_A_active, 0.0
)

# Precompute the data FFT once; this is the dominant per-iteration FFT we
# can amortise across every objective call.
D_A_xp = xp.fft.rfft(hA_td) * dt        # length Nf_full, on xp

# Single-channel A analysis container — for the lisatools cross-check.
# dra_AET.data_res_arr is an FDSignal (already trimmed to the active band);
# slicing returns a raw xp array so we re-wrap it before constructing the
# DataResidualArray.
_a_signal = FDSignal(dra_AET.data_res_arr[0:1], settings=fd_set)
dra_A    = DataResidualArray(_a_signal)
sens_A   = SensitivityMatrix(fd_set, [A1TDISens], model="scirdv1")
ana_A    = AnalysisContainer(dra_A, sens_A)
snr_A_lt = float(ana_A.snr())
print(f"lisatools SNR of A-channel injection:           {snr_A_lt:.6f}")

# -----------------------------------------------------------------------------
# 8b. Pick a sample alpha and verify lisatools vs FFT-IFFT on the chebyshev
# -----------------------------------------------------------------------------
alpha_sample = np.array([
    11.3,
    -0.82,
    -0.025,
    -0.001,
    -0.0003,
])
t_scale_sample = T_SCALE_DEFAULT
print(f"\nSample alpha for cross-check: {alpha_sample}, t_scale={t_scale_sample}")

template_raw = make_chebyshev_template(alpha_sample, N, dt, cheb_wave,
                                       t_scale=t_scale_sample)

template_raw = hXYZ[0].copy()
assert template_raw is not None, "Sample alpha gave an invalid template."

snr_norm, template_n = normalize_template(template_raw, invC_A_full, dt, df, N)
print(f"  <h|h>_A^0.5 of raw template (FFT-IFFT):       {snr_norm:.6e}")

# (i) lisatools template_snr after normalisation: expect (1.0, det_snr).
# ``template_n`` is on the active backend, so wrapping with TDSignal
# (settings=td_set with force_backend=backend) keeps everything aligned.
template_td    = TDSignal(template_n[None, :], settings=td_set)
template_fd    = template_td.fft(settings=fd_set)
template_dra   = DataResidualArray(template_fd)
opt_snr_lt, det_snr_lt = ana_A.template_snr(template_dra, phase_maximize=True)
opt_snr_lt = float(opt_snr_lt)
det_snr_lt = float(det_snr_lt)
print(f"  lisatools opt_snr of normalised template:     {opt_snr_lt:.6f}  (== 1?)")
print(f"  lisatools |<d_A|h>| (phase-maxed, tau=0):     {det_snr_lt:.6f}")

# (ii) FFT-IFFT cross-correlation (uses the cached data FFT D_A_xp)
snr_tau_fft = cross_correlation_snr_tau(template_n, D_A_xp, invC_A_full,
                                        dt, df, N)
fft_at_zero = float(snr_tau_fft[0])
fft_peak    = float(snr_tau_fft.max())
i_peak      = int(snr_tau_fft.argmax())
tau_peak    = (i_peak if i_peak <= N // 2 else i_peak - N) * dt
print(f"  FFT-IFFT |<d_A|h>|(tau=0):                    {fft_at_zero:.6f}")
print(f"  FFT-IFFT max_tau |<d_A|h>|:                   {fft_peak:.6f}"
      f"  at tau = {tau_peak:.1f} s")
print(f"  lisatools vs FFT-IFFT (tau=0) difference:     "
      f"{(det_snr_lt - fft_at_zero): .3e}")

# -----------------------------------------------------------------------------
# 8c. Differential-evolution fit of (alpha_0..alpha_4, t_scale), marginalised
#     over time shift
# -----------------------------------------------------------------------------
def split_params(params):
    return params[:N_ALPHA], float(params[N_ALPHA])


def objective(params, return_snr_max=False):
    """Return -max_tau |<d_A|h_norm>(tau)|  (we minimise).

    ``params`` arrives from scipy as a CPU numpy array. The chebyshev
    template construction is done on CPU (it's small + scipy CubicSpline
    is CPU-only); everything from ``normalize_template`` onward runs on
    ``xp`` (numpy or cupy) and the FFTs/IFFTs happen on the active backend.
    """
    alpha_vec, t_scale = split_params(params)
    template_1 = make_chebyshev_template(alpha_vec, N, dt, cheb_wave,
                                       t_scale=t_scale)
    
    if template_1 is None:
        return 1.0  # invalid template -> bad fitness
    snr0, template_n1 = normalize_template(template_1, invC_A_full, dt, df, N)
    if template_n1 is None:
        return 1.0
    snr_tau = cross_correlation_snr_tau(template_n1, D_A_xp, invC_A_full,
                                        dt, df, N)
    tmp2 = (tmp1 := (snr_tau ** 2 - 1.0))  # - np.mean(tmp1)
    cum_sum_snr = xp.cumsum(tmp2)
    max_val = -np.inf
    for delta in [1, 4, 16, 64, 256, 1024, 4096]:
        tmp_arr = (_cum_sum_diff := (cum_sum_snr[delta:] - cum_sum_snr[:-delta])) / _cum_sum_diff.shape[0] ** (1/2) 
        if max_val < tmp_arr.max().item():
            max_val = tmp_arr.max().item()
    if return_snr_max:
        return -float(max_val), snr_tau.max().item()
    return -float(max_val)
    #return -float(tmp2.max() ** 2)

lags_full_np    = (np.arange(N) - N // 2) * dt
snr_tau_shifted_np = _to_cpu(xp.fft.fftshift(snr_tau_fft))
fig_cheb, axc   = plt.subplots(figsize=(9, 3.5))
axc.plot(lags_full_np, snr_tau_shifted_np, lw=0.7)
# axc.axhline(snr_best, color="gray", ls=":",
#             label=f"max = {snr_best:.3f}")
# axc.axvline(tau_peak_best, color="r", ls="--", alpha=0.5,
#             label=f"tau* = {tau_peak_best:.0f} s")
axc.set_xlabel("time shift tau [s]")
axc.set_ylabel("|<d_A|h_norm>|(tau)")
axc.set_title("Best-fit Chebyshev template — SNR(tau)")
axc.legend(loc="upper right")
plt.tight_layout()
out_png_cheb = os.path.join(os.path.dirname(__file__),
                          "emri_snr_correlation_chebyshev.png")
plt.savefig(out_png_cheb)
plt.show()
#.png

print("\nRunning differential_evolution over (alpha_0..alpha_4, t_scale) ...")
result = differential_evolution(
    objective,
    bounds=PARAM_BOUNDS,
    maxiter=30,
    popsize=12,
    tol=1e-3,
    seed=0,
    polish=True,
    workers=1,      # keep at 1 — cupy + multiprocessing don't play nicely
    updating="deferred",
    disp=True,
)
alpha_best, t_scale_best = split_params(result.x)
snr_best                 = -result.fun

# Recover the peak time-shift at the best (alpha, t_scale)
tpl_best       = make_chebyshev_template(alpha_best, N, dt, cheb_wave,
                                         t_scale=t_scale_best)
_, tpl_best_n  = normalize_template(tpl_best, invC_A_full, dt, df, N)
snr_tau_best   = cross_correlation_snr_tau(tpl_best_n, D_A_xp, invC_A_full,
                                           dt, df, N)

snr_tau_best   = cross_correlation_snr_tau(template_n, D_A_xp, invC_A_full,
                                           dt, df, N)
i_peak_best    = int(snr_tau_best.argmax())
# Signed lag index in the unshifted irfft array
k_signed       = i_peak_best if i_peak_best <= N // 2 else i_peak_best - N
# Physical lag for the data: see the alignment derivation in section 8d.
# The irfft index n maps to physical lag tau = -k_signed * dt under the
# conj(D)*H/S convention we use above.
tau_peak_best  = -k_signed * dt

print(f"\nBest-fit Chebyshev parameters:")
for k, a_k in enumerate(alpha_best):
    lo, hi = ALPHA_BOUNDS[k]
    print(f"  alpha_{k}  = {a_k:+.6f}     [{lo:+.4f}, {hi:+.4f}]")
print(f"  t_scale  = {t_scale_best:+.3f}     "
      f"[{T_SCALE_BOUND[0]:.1f}, {T_SCALE_BOUND[1]:.1f}]")

print(f"\nMax-over-tau matched-filter SNR (FFT-IFFT):      {snr_best:.6f}")
print(f"Best time shift tau* (data delay):               {tau_peak_best:.1f} s")
print(f"For reference, lisatools optimal A-channel SNR:  {snr_A_lt:.6f}")
print(f"(Chebyshev recovery fraction vs A-channel SNR:   "
      f"{snr_best / snr_A_lt:.3f})")

# -----------------------------------------------------------------------------
# 8d. Time-shift the best template by tau* and recompute the inner product
#     via the lisatools template_snr — cross-check against the FFT-IFFT peak.
# -----------------------------------------------------------------------------
#
# Alignment convention (matches np.roll's "positive = shift to later time"):
#
#   If d(t) ~= h(t - tau*), then aligning the template to the data means
#   producing h_aligned(t) = h(t - tau*), i.e.
#       tpl_aligned[i] = tpl[i - n_shift],     n_shift = int(round(tau*/dt))
#                     == np.roll(tpl, n_shift)
#   (positive n_shift -> roll right -> signal appears later in the array).
#
#   The irfft we use stores corr[n] at the wrap-around index n; the *physical*
#   lag is tau = -k_signed * dt with k_signed = n_unshifted % N reflected to
#   the signed range [-N/2, N/2). So roll-amount n_shift = +tau*/dt
#   = -k_signed.
n_shift            = -k_signed
tpl_best_aligned   = xp.roll(tpl_best_n, n_shift)        # stays on backend

tpl_aligned_td     = TDSignal(tpl_best_aligned[None, :], settings=td_set)
tpl_aligned_fd     = tpl_aligned_td.fft(settings=fd_set)
tpl_aligned_dra    = DataResidualArray(tpl_aligned_fd)
opt_snr_shift_lt, det_snr_shift_lt = ana_A.template_snr(
    tpl_aligned_dra, phase_maximize=True
)
opt_snr_shift_lt = float(opt_snr_shift_lt)
det_snr_shift_lt = float(det_snr_shift_lt)
print(f"\n--- Lisatools inner product on time-shifted best template ---")
print(f"  np.roll amount (samples):                       {n_shift}")
print(f"  Equivalent time shift applied:                  "
      f"{n_shift * dt:.1f} s")
print(f"  lisatools opt_snr of aligned template:          "
      f"{opt_snr_shift_lt:.6f}  (== 1 means norm preserved)")
print(f"  lisatools |<d_A|h_aligned>| (phase-maxed):      "
      f"{det_snr_shift_lt:.6f}")
print(f"  FFT-IFFT peak max_tau |<d_A|h>|:                "
      f"{snr_best:.6f}")
print(f"  Lisatools vs FFT-IFFT (after time shift):       "
      f"{(det_snr_shift_lt - snr_best): .3e}")

# -----------------------------------------------------------------------------
# 8d. Plot SNR(tau) at the best-fit alpha
# -----------------------------------------------------------------------------
lags_full_np    = (np.arange(N) - N // 2) * dt
snr_tau_shifted_np = _to_cpu(xp.fft.fftshift(snr_tau_best))
fig_cheb, axc   = plt.subplots(figsize=(9, 3.5))
axc.plot(lags_full_np, snr_tau_shifted_np, lw=0.7)
axc.axhline(snr_best, color="gray", ls=":",
            label=f"max = {snr_best:.3f}")
axc.axvline(tau_peak_best, color="r", ls="--", alpha=0.5,
            label=f"tau* = {tau_peak_best:.0f} s")
axc.set_xlabel("time shift tau [s]")
axc.set_ylabel("|<d_A|h_norm>|(tau)")
axc.set_title("Best-fit Chebyshev template — SNR(tau)")
axc.legend(loc="upper right")
plt.tight_layout()
out_png_cheb = os.path.join(os.path.dirname(__file__),
                            "emri_snr_correlation_chebyshev.png")
plt.savefig(out_png_cheb, dpi=130)
print(f"\nSaved plot -> {out_png_cheb}")
plt.close(fig_cheb)

from eryn.prior import ProbDistContainer, uniform_dist
alphas = ProbDistContainer({
        0: uniform_dist(10.6,    11.7),
        1: uniform_dist(-0.97,  -0.66),
        2: uniform_dist(-0.052, -0.005),
        3: uniform_dist(-0.0041, +0.0018),
        4: uniform_dist(-0.0010, +0.0003),
        5: uniform_dist(2.0, 2000.0),
})
print("STARTING!!!!!!!")
max_val = -np.inf
for draw in alphas.rvs(100000):
    tmp = objective(draw, return_snr_max=True)[-1]
    if max_val < tmp:
         max_val = tmp
         print("max_val raised:", max_val)
breakpoint()
