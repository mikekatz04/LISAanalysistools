"""
JAX reimplementation of GBGPU build_single_waveform.

A pure-JAX version of the single-source galactic-binary waveform in
SharedMemoryGBGPU.cu::build_single_waveform, exposing the same 9 input
parameters and producing the same frequency-domain TDI A,E (or A,E,T,
or X,Y,Z) output up to the round-off introduced by JAX's FFT vs cuFFTDx.

The point of this module is *differentiability*: ``jax.jacfwd`` or
``jax.jacrev`` over ``gb_jax_waveform`` gives the parameter Jacobian
that is consumed by the chain-rule kernels in gb_chain_rule_grad.py.

The internal computation matches the C kernel step by step:

  1. analytic Keplerian spacecraft positions (LISA.h constants)
  2. arm unit vectors, kdotr, kdotP, xi, fi, fonfs
  3. polarisation tensors eplus, ecross, polarisation amplitudes DP, DC
  4. Aij sums over j and k
  5. six Gs slow-part transfer functions for links (12,23,31,21,32,13)
  6. slow-part time-domain X, Y, Z with the fctr2 prefactor
  7. FFT, multiply by amp, fftshift, multiply by 0.5 T / N
  8. XYZ -> AET rotation as in AET_from_XYZ_swap

The integer start_ind = round(f0 * T) - N // 2 is returned alongside;
it is non-differentiable (the round() and the data-index lookup are
constant on small neighbourhoods of the parameter), exactly as the
C waveform also treats q = rint(f0 * T) as constant.
"""

from __future__ import annotations

import jax
import jax.numpy as jnp

# Match double precision in the C kernel.
jax.config.update("jax_enable_x64", True)


# ---------------------------------------------------------------------------
#  LISA constants (from src/gbgpu/cutils/LISA.h)
# ---------------------------------------------------------------------------

LARM = 2.5e9                      # mean arm length (m)
EC = 0.0048241852                 # orbital eccentricity
KAPPA = 0.0                       # initial azimuthal phase
LAMBDA0 = 0.0                     # initial constellation orientation
FM = 3.168753575e-8               # modulation frequency
FSTAR = 0.01908538063694777       # transfer frequency, c / (2 pi L)
CLIGHT = 299792458.0
AU = 149597870700.0
SQ3 = 1.73205080757


_INV_SQRT2 = 1.0 / 2.0 ** 0.5
_INV_SQRT3 = 1.0 / 3.0 ** 0.5
_INV_SQRT6 = 1.0 / 6.0 ** 0.5


def _spacecraft(t):
    """Analytic Keplerian LISA spacecraft positions.

    Parameters
    ----------
    t : (N,) array

    Returns
    -------
    P1, P2, P3 : (N, 3) arrays
    """
    alpha = 2.0 * jnp.pi * FM * t + KAPPA
    sa = jnp.sin(alpha)
    ca = jnp.cos(alpha)

    def _one(beta_const):
        sb = jnp.sin(beta_const)
        cb = jnp.cos(beta_const)
        x = AU * ca + AU * EC * (sa * ca * sb - (1.0 + sa * sa) * cb)
        y = AU * sa + AU * EC * (sa * ca * cb - (1.0 + ca * ca) * sb)
        z = -SQ3 * AU * EC * (ca * cb + sa * sb)
        return jnp.stack([x, y, z], axis=-1)

    return (
        _one(0.0 + LAMBDA0),
        _one(2.0 * jnp.pi / 3.0 + LAMBDA0),
        _one(4.0 * jnp.pi / 3.0 + LAMBDA0),
    )


def _safe_sinc(x):
    """Return sin(x)/x, smooth at x=0 (limit is 1)."""
    eps = 1e-12
    small = jnp.abs(x) < eps
    safe_x = jnp.where(small, 1.0, x)
    return jnp.where(small, 1.0 - x * x / 6.0, jnp.sin(safe_x) / safe_x)


def gb_jax_waveform(
    amp, f0, fdot, fddot, phi0, iota, psi, lam, beta,
    T, N, tdi_channel_setup="AE",
):
    """Single-source GB waveform in FD TDI channels, JAX-differentiable.

    Parameters
    ----------
    amp, f0, fdot, fddot, phi0, iota, psi, lam, beta : scalar (jax tracable)
        Galactic-binary parameters; ``beta`` is ecliptic latitude.  Same
        conventions as :py:meth:`gbgpu.gbgpu.GBGPUBase.run_wave`.
    T : float
        Observation time in seconds (must equal ``N * dt`` for the
        run_wave grid).
    N : int (static)
        Number of slow-part samples.
    tdi_channel_setup : "AE", "AET", or "XYZ".

    Returns
    -------
    wave : (nchannels, N) complex jnp.ndarray
        Frequency-domain TDI waveform, fft-shifted and scaled by 0.5 T / N,
        matching the layout used by ``GBGPUBase.run_wave(..., use_c_implementation=True)``.
    start_ind : int jnp scalar
        Global frequency index of ``wave[:, 0]``,
        ``= int(round(f0 * T)) - N // 2``.
    """
    theta_pol = jnp.pi / 2.0 - beta

    # polarisation amplitudes
    cosiota = jnp.cos(iota)
    cosps = jnp.cos(2.0 * psi)
    sinps = jnp.sin(2.0 * psi)
    Aplus = amp * (1.0 + cosiota * cosiota)
    Across = -2.0 * amp * cosiota

    # sky basis vectors
    sinth = jnp.sin(theta_pol); costh = jnp.cos(theta_pol)
    sinph = jnp.sin(lam); cosph = jnp.cos(lam)

    u = jnp.stack([costh * cosph, costh * sinph, -sinth])
    v = jnp.stack([sinph, -cosph, jnp.zeros_like(sinph)])
    k = jnp.stack([-sinth * cosph, -sinth * sinph, -costh])

    eplus = jnp.outer(v, v) - jnp.outer(u, u)              # (3,3)
    ecross = jnp.outer(u, v) + jnp.outer(v, u)             # (3,3)

    DP = Aplus * cosps - 1j * Across * sinps              # complex scalar
    DC = -Aplus * sinps - 1j * Across * cosps

    # slow-part time grid
    delta_t_slow = T / N
    t = jnp.arange(N, dtype=jnp.float64) * delta_t_slow      # (N,)

    P1, P2, P3 = _spacecraft(t)                              # each (N, 3)
    r12 = (P2 - P1) / LARM
    r13 = (P3 - P1) / LARM
    r23 = (P3 - P2) / LARM
    r31 = -r13

    # k dot r and k dot P
    kdotr_12 = jnp.einsum("i,ti->t", k, r12)
    kdotr_23 = jnp.einsum("i,ti->t", k, r23)
    kdotr_31 = jnp.einsum("i,ti->t", k, r31)

    kdotP1 = jnp.einsum("i,ti->t", k, P1) / CLIGHT
    kdotP2 = jnp.einsum("i,ti->t", k, P2) / CLIGHT
    kdotP3 = jnp.einsum("i,ti->t", k, P3) / CLIGHT

    # delayed time at spacecraft and instantaneous frequency
    xi1 = t - kdotP1
    xi2 = t - kdotP2
    xi3 = t - kdotP3

    fi1 = f0 + fdot * xi1 + 0.5 * fddot * xi1 ** 2
    fi2 = f0 + fdot * xi2 + 0.5 * fddot * xi2 ** 2
    fi3 = f0 + fdot * xi3 + 0.5 * fddot * xi3 ** 2

    fonfs1 = fi1 / FSTAR
    fonfs2 = fi2 / FSTAR
    fonfs3 = fi3 / FSTAR

    # Aij[ij] = sum_j [ (eplus . r_ij)_j r_ij_j * DP + (ecross . r_ij)_j r_ij_j * DC ]
    def _aij(rij):
        tmp_p = jnp.einsum("jk,tk->tj", eplus, rij)
        tmp_c = jnp.einsum("jk,tk->tj", ecross, rij)
        return jnp.sum(tmp_p * rij, axis=-1) * DP + jnp.sum(tmp_c * rij, axis=-1) * DC

    A12 = _aij(r12)
    A23 = _aij(r23)
    A31 = _aij(r31)

    # nearest Fourier bin -- piecewise constant in f0; matches C ``rint(f0*T)``.
    q = jnp.round(f0 * T)
    df_phase = 2.0 * jnp.pi * q / T
    om = 2.0 * jnp.pi * f0

    def _argS(xi):
        return (
            phi0
            + (om - df_phase) * t
            + jnp.pi * fdot * xi * xi
            + (jnp.pi / 3.0) * fddot * xi * xi * xi
        )

    arg_phasing1 = om * kdotP1 - _argS(xi1)
    arg_phasing2 = om * kdotP2 - _argS(xi2)
    arg_phasing3 = om * kdotP3 - _argS(xi3)

    # Gs[ij] for the six link/spacecraft pairs.
    # Order: (12=0,23=1,31=2,21=3,32=4,13=5)
    # s_all = [0, 1, 2, 1, 2, 0]   -> arg_phasing index
    # arm_index = [12,23,31,12,23,31]  -> Aij index
    # kdotr sign: 12 -> +kdotr_12 ; 21 -> -kdotr_12 (etc.)
    def _G(fonfs_s, kdotr_lk, arg_phase_s, A_ij):
        arg = 0.5 * fonfs_s * (1.0 + kdotr_lk)
        return 0.25 * _safe_sinc(arg) * jnp.exp(-1j * (arg + arg_phase_s)) * A_ij

    G_12 = _G(fonfs1,  kdotr_12, arg_phasing1, A12)
    G_23 = _G(fonfs2,  kdotr_23, arg_phasing2, A23)
    G_31 = _G(fonfs3,  kdotr_31, arg_phasing3, A31)
    G_21 = _G(fonfs2, -kdotr_12, arg_phasing2, A12)
    G_32 = _G(fonfs3, -kdotr_23, arg_phasing3, A23)
    G_13 = _G(fonfs1, -kdotr_31, arg_phasing1, A31)

    # frequency drift transfer factor
    f_evol = f0 + fdot * t + 0.5 * fddot * t * t
    omL = f_evol / FSTAR
    SomL = jnp.sin(omL)
    fctr_xfer = jnp.exp(-1j * omL)
    fctr2 = 4.0 * omL * SomL * fctr_xfer / amp     # divided by amp for dyn-range; restored after FFT

    # slow-part X, Y, Z (matches gbgpu.py:400-402)
    X_slow = fctr2 * (G_21 - G_31 + (G_12 - G_13) * fctr_xfer)
    Y_slow = fctr2 * (G_32 - G_12 + (G_23 - G_21) * fctr_xfer)
    Z_slow = fctr2 * (G_13 - G_23 + (G_31 - G_32) * fctr_xfer)

    # FFT, restore amp, fftshift, multiply by 0.5 T / N
    X_fd = jnp.fft.fft(X_slow) * amp
    Y_fd = jnp.fft.fft(Y_slow) * amp
    Z_fd = jnp.fft.fft(Z_slow) * amp

    fctr3 = 0.5 * T / N
    X_fd = jnp.fft.fftshift(X_fd) * fctr3
    Y_fd = jnp.fft.fftshift(Y_fd) * fctr3
    Z_fd = jnp.fft.fftshift(Z_fd) * fctr3

    if tdi_channel_setup == "AE":
        A_fd = (Z_fd - X_fd) * _INV_SQRT2
        E_fd = (X_fd - 2.0 * Y_fd + Z_fd) * _INV_SQRT6
        wave = jnp.stack([A_fd, E_fd], axis=0)
    elif tdi_channel_setup == "AET":
        A_fd = (Z_fd - X_fd) * _INV_SQRT2
        E_fd = (X_fd - 2.0 * Y_fd + Z_fd) * _INV_SQRT6
        T_fd = (X_fd + Y_fd + Z_fd) * _INV_SQRT3
        wave = jnp.stack([A_fd, E_fd, T_fd], axis=0)
    elif tdi_channel_setup == "XYZ":
        wave = jnp.stack([X_fd, Y_fd, Z_fd], axis=0)
    else:
        raise ValueError(f"unknown tdi_channel_setup {tdi_channel_setup!r}")

    start_ind = (jnp.round(f0 * T).astype(jnp.int64) - (N // 2))
    return wave, start_ind


# ---------------------------------------------------------------------------
#  Convenience: parameter Jacobian via JAX autodiff
# ---------------------------------------------------------------------------

PARAM_NAMES = ("amp", "f0", "fdot", "fddot", "phi0", "iota", "psi", "lam", "beta")
N_PARAMS = len(PARAM_NAMES)


def gb_jax_template_and_jacobian(theta, T, N, tdi_channel_setup="AE"):
    """Compute the FD template *and* its parameter Jacobian.

    Parameters
    ----------
    theta : (9,) array of (amp, f0, fdot, fddot, phi0, iota, psi, lam, beta).
    T, N, tdi_channel_setup : passed straight through to :func:`gb_jax_waveform`.

    Returns
    -------
    wave : (nchannels, N) complex
    jac : (9, nchannels, N) complex,
        ``jac[k] = d wave / d theta_k``.
    start_ind : int scalar (same as :func:`gb_jax_waveform`).
    """
    def _wave(th):
        return gb_jax_waveform(*th, T=T, N=N, tdi_channel_setup=tdi_channel_setup)[0]

    wave, start_ind = gb_jax_waveform(*theta, T=T, N=N, tdi_channel_setup=tdi_channel_setup)
    jac = jax.jacfwd(_wave)(theta)                  # (nchannels, N, 9) -> transpose
    jac = jnp.moveaxis(jac, -1, 0)                  # (9, nchannels, N)
    return wave, jac, start_ind
