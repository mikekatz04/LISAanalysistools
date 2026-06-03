"""
========================================================================
 lisa-on-gpu  WDM  get_ll-gradient  chain-rule  vs  JAX  autodiff
========================================================================

WHY THIS TEST EXISTS
--------------------
The C/CUDA kernel ``gb_wdm_get_ll_grad_kernel`` in TDIonTheFly.cu computes
the per-binary parameter gradient of the Gaussian log-likelihood

    L(theta) = -1/2 < d - h(theta) | d - h(theta) >

via the *analytic* chain rule with a central-difference parameter
derivative,

    dL/dtheta_k  =  4 sum_{pixels}  (w_d - w_h_CENTER) * dw_h/dtheta_k * N^{-1}    (KERNEL)

where  w_h_CENTER  is the un-perturbed wavelet coefficient and
    dw_h/dtheta_k  =  (w_+ - w_-) / (2 eps_k)
is the central finite difference of the waveform itself.

This is the correct discretisation of the chain rule:
  * the residual is anchored at the *un-perturbed* parameters (the actual
    point at which we want the gradient);
  * dw/dtheta_k is approximated by central FD, which is *exact* for
    polynomial-degree-2 dependence (d^3 w / dtheta^3 == 0) and otherwise
    carries an O(eps^2 d^3 w / dtheta_k^3) truncation -- the only error
    against the analytic derivative computed by jax.grad.

A natural-looking alternative,

    grad_k  =  (L(theta+eps) - L(theta-eps)) / (2 eps_k)                        (FD-OF-L)
            =  sum_{pixels}  (w_d - U) * (w_+ - w_-)/(2 eps_k) * N^{-1},     U = (w_+ + w_-)/2,

is the *Python finite difference of L*.  It is algebraically distinct
from the chain rule above: it differs by

    KERNEL - (FD-OF-L)
      =  sum_{pixels} (U - w_h_CENTER) * dw_h/dtheta_k * N^{-1}
      =  (eps^2 / 2) sum d^2 w / dtheta_k^2  *  dw_h / dtheta_k  *  N^{-1}  +  ...

a non-trivial bias whenever d^2 w / dtheta_k^2 != 0.  Crucially, the
*kernel* form is the one that converges to jax.grad to round-off for
polynomial-degree-2 waveforms; the FD-of-L form does not, because it
inherits a truncation from L being polynomial of higher degree than w.

THIS TEST
---------
We build a self-contained mock WDM pipeline w_mn(theta):
  * polynomial of degree 2 in each of the 5 mock parameters, so the
    central FD of w is exact (no truncation error);
  * non-trivial d^2 w / dtheta^2 in every parameter, so the kernel-vs-FD-of-L
    bias is non-zero and visible;
  * three "channels" with an XYZ-style 3 x 3 cross-channel inverse-noise
    matrix, matching the inner-product accumulation done by
    WDMDomain::add_grad_contrib.

We then implement two Python mirrors of the C-kernel inner-product loop:

   grad_KERNEL  -- the chain-rule form that the C kernel computes
                  (i.e. (w_d - w_h_CENTER) * dw * N).
   grad_FD_OF_L -- the FD-of-L form (i.e. (L_pixel(+eps) - L_pixel(-eps)) / (2 eps)).

The reference is jax.grad of the same L.

EXPECTED OUTCOME
----------------
   grad_KERNEL  matches  jax.grad   to round-off  (~ 1e-13)   -> PASS
   grad_FD_OF_L matches  jax.grad   only to  O(eps^2 d^3 L / dtheta^3)
                                              (~ 1e-6 for these eps)  -> demonstrates the bias

Run with:
    python gb_wdm_grad_jax_test.py
"""

from __future__ import annotations

import numpy as np
import jax
import jax.numpy as jnp
from jax import config
config.update("jax_enable_x64", True)


# ---------------------------------------------------------------------------
#  Polynomial mock WDM waveform (degree 2 in each of the 5 parameters).
#
#  w_{m n c}(theta)  =  amp
#                     * (a0 + a1 * (f0 - f_ref)/SIGMA_F + a2 * ((f0 - f_ref)/SIGMA_F)^2)
#                     * (b0 + b1 * fdot/SIGMA_FDOT + b2 * (fdot/SIGMA_FDOT)^2)
#                     * (c0 + c1 * phi0 + c2 * phi0^2)
#                     * (d0 + d1 * lam + d2 * lam^2)
#
#  (a_i, b_i, c_i, d_i) are random per pixel and channel.  The waveform
#  is polynomial-degree-2 in every parameter, so the central FD of w is
#  *exact*; the only remaining numerical error in either gradient
#  formula is double-precision round-off.
# ---------------------------------------------------------------------------

PARAM_NAMES = ("amp", "f0", "fdot", "phi0", "lam")
N_PARAMS = len(PARAM_NAMES)
N_CHAN = 3
N_PIX = 1500

F_REF = 5.0e-3
SIGMA_F = 1.0e-3
SIGMA_FDOT = 1.0e-19

rng = np.random.default_rng(31)
A_COEFFS = jnp.asarray(rng.normal(size=(N_PIX, N_CHAN, 3)))
B_COEFFS = jnp.asarray(rng.normal(size=(N_PIX, N_CHAN, 3)))
C_COEFFS = jnp.asarray(rng.normal(size=(N_PIX, N_CHAN, 3)))
D_COEFFS = jnp.asarray(rng.normal(size=(N_PIX, N_CHAN, 3)))


def wave_grid(theta):
    """Compute w_{m n c}(theta) for every pixel and channel; shape (N_PIX, N_CHAN)."""
    amp, f0, fdot, phi0, lam = theta
    f0_n = (f0 - F_REF) / SIGMA_F
    fdot_n = fdot / SIGMA_FDOT

    Af = A_COEFFS[..., 0] + A_COEFFS[..., 1] * f0_n + A_COEFFS[..., 2] * f0_n ** 2
    Bf = B_COEFFS[..., 0] + B_COEFFS[..., 1] * fdot_n + B_COEFFS[..., 2] * fdot_n ** 2
    Cp = C_COEFFS[..., 0] + C_COEFFS[..., 1] * phi0 + C_COEFFS[..., 2] * phi0 ** 2
    Dl = D_COEFFS[..., 0] + D_COEFFS[..., 1] * lam + D_COEFFS[..., 2] * lam ** 2

    return amp * Af * Bf * Cp * Dl


# Cross-channel inverse-noise matrix (constant across pixels for simplicity).
N_INV_CONST = jnp.array([
    [1.0,  0.3,  0.1],
    [0.3,  1.2,  0.2],
    [0.1,  0.2,  0.9],
])
N_INV = jnp.broadcast_to(N_INV_CONST, (N_PIX, N_CHAN, N_CHAN))

# Inject truth + a little noise so the residual is non-trivial.
THETA_TRUE = jnp.array([1.0e-22, 5.0e-3, 0.0, 0.7, 1.3])
w_truth = wave_grid(THETA_TRUE)
data = w_truth + 0.05 * jnp.array(rng.standard_normal(size=w_truth.shape)) * jnp.max(jnp.abs(w_truth))


# ---------------------------------------------------------------------------
#  Full log-likelihood (everything jax-differentiable)
# ---------------------------------------------------------------------------

def loglikelihood(theta):
    """L = -1/2 sum_pix sum_{c,c'} w_c w_{c'} N^{-1} + sum_pix sum_{c,c'} w_d_c w_{c'} N^{-1}.

    The d_d piece is theta-independent and dropped (it cancels in any
    gradient comparison)."""
    w = wave_grid(theta)
    h_h = jnp.einsum("pi,pj,pij->", w, w, N_INV)
    d_h = jnp.einsum("pi,pj,pij->", data, w, N_INV)
    return -0.5 * h_h + d_h


# ---------------------------------------------------------------------------
#  Python mirror of the C-kernel inner-product accumulation
#
#  This is the same logic implemented in
#  WDMDomain::add_grad_contrib (XYZ branch):
#      grad_k += sum_pix sum_{c,c'} (w_d - w_h_CENTER)_c * dw_{c'} * N^{-1}_{cc'} * 0.25
#  with the outer factor of 4 applied by the calling kernel via block-reduce.
#  Vectorised here for speed (N_PIX = 1500 takes ~ms); the math is
#  byte-for-byte the C kernel's accumulation.
# ---------------------------------------------------------------------------

def grad_KERNEL(theta, eps):
    """The C-kernel chain-rule gradient.

    KERNEL_k =  sum_pix sum_{c,c'} (w_d - w_h_CENTER)_c  *  dw_{c'}/dtheta_k  *  N^{-1}_{cc'}

    with dw_{c'}/dtheta_k computed by central FD over the same eps_k that the
    C kernel uses internally.  This is the discretisation that
    `add_grad_contrib` actually implements.
    """
    grad = np.zeros(N_PARAMS, dtype=np.float64)
    theta_arr = np.asarray(theta, dtype=np.float64)
    w_c = np.asarray(wave_grid(theta_arr))
    N_arr = np.asarray(N_INV)
    d_arr = np.asarray(data)
    for k in range(N_PARAMS):
        ep = float(eps[k])
        if ep <= 0.0:
            continue
        tp = theta_arr.copy(); tp[k] += ep
        tm = theta_arr.copy(); tm[k] -= ep
        wp = np.asarray(wave_grid(tp))
        wm = np.asarray(wave_grid(tm))
        dw = (wp - wm) / (2.0 * ep)
        r = d_arr - w_c
        grad[k] = float(np.einsum("pi,pj,pij->", r, dw, N_arr))
    return grad


def grad_FD_OF_L(theta, eps):
    """Python FD of L: (L(theta+eps) - L(theta-eps)) / (2 eps_k).

    This is the per-pixel form  sum (L_pixel(+eps) - L_pixel(-eps)) / (2 eps_k)
    -- algebraically equal to the chain rule against the FD midpoint
    U = (w_+ + w_-)/2, NOT to the chain rule against the un-perturbed
    centre w_h_CENTER.  Provided for comparison; it differs from the C
    kernel by O((eps * d^2 w / dtheta_k^2)^2)."""
    grad = np.zeros(N_PARAMS, dtype=np.float64)
    theta_arr = np.asarray(theta, dtype=np.float64)
    N_arr = np.asarray(N_INV)
    d_arr = np.asarray(data)
    for k in range(N_PARAMS):
        ep = float(eps[k])
        if ep <= 0.0:
            continue
        tp = theta_arr.copy(); tp[k] += ep
        tm = theta_arr.copy(); tm[k] -= ep
        wp = np.asarray(wave_grid(tp))
        wm = np.asarray(wave_grid(tm))
        Lp = -0.5 * np.einsum("pi,pj,pij->", wp, wp, N_arr) + np.einsum("pi,pj,pij->", d_arr, wp, N_arr)
        Lm = -0.5 * np.einsum("pi,pj,pij->", wm, wm, N_arr) + np.einsum("pi,pj,pij->", d_arr, wm, N_arr)
        grad[k] = float((Lp - Lm) / (2.0 * ep))
    return grad


# ---------------------------------------------------------------------------
#  Run the comparison
# ---------------------------------------------------------------------------

def main():
    # Evaluation point: slightly perturbed from truth (non-zero gradient).
    theta_test = THETA_TRUE * jnp.array([1.05, 1.0 + 1.0e-7, 1.0, 1.0 + 0.01, 1.0 + 0.01])
    theta_test = theta_test.at[2].set(1.0e-19)

    # Per-parameter FD step.  Sized to the parameter's natural scale so the FD
    # numerator has plenty of float-precision headroom.  For polynomial-degree-2
    # w, central FD of w is exact, so the only error against jax.grad is
    # double-precision round-off (~ 1e-13 here).
    eps = jnp.array([
        1.0e-25,        # amp                 ~ 1e-3 * amp
        1.0e-6,         # f0    (Hz)          ~ 1e-3 * SIGMA_F
        1.0e-22,        # fdot  (Hz/s)        ~ 1e-3 * SIGMA_FDOT
        1.0e-3,         # phi0  (rad)
        1.0e-3,         # lam   (rad)
    ])

    # Reference: analytic gradient via JAX autodiff.
    grad_jax = np.asarray(jax.grad(loglikelihood)(theta_test))

    # Two Python mirrors of the C-kernel-vs-FD-of-L difference.
    grad_kernel = grad_KERNEL(theta_test, eps)
    grad_fd_l = grad_FD_OF_L(theta_test, eps)

    print("=" * 72)
    print(" lisa-on-gpu  WDM  get_ll_grad  C-kernel  vs  JAX  autodiff")
    print("=" * 72)
    print(f"  N_PIX = {N_PIX}    N_CHAN = {N_CHAN}    waveform = degree-2 polynomial")
    print(f"  Per-pixel cross-channel inverse-noise matrix N_INV (XYZ branch).")
    print(f"\n  Per-parameter FD step (eps_k):")
    for name, e in zip(PARAM_NAMES, np.asarray(eps)):
        print(f"    {name:>5s}  eps = {e:.2e}")

    print("\n  ---- C kernel chain rule:   (w_d - w_h_CENTER) * dw * N^-1  ----")
    print(f"  {'param':>5s}   {'kernel':>15s}   {'jax.grad':>15s}   {'rel-err':>10s}")
    worst_k = 0.0
    for k, name in enumerate(PARAM_NAMES):
        denom = max(abs(grad_jax[k]), 1e-300)
        rel = abs(grad_kernel[k] - grad_jax[k]) / denom
        worst_k = max(worst_k, rel)
        print(f"  {name:>5s}   {grad_kernel[k]:+.8e}   {grad_jax[k]:+.8e}   {rel:.2e}")
    print(f"  >>> kernel worst rel err  =  {worst_k:.3e}")

    print("\n  ---- (for context) Python FD of L:   (L(+eps) - L(-eps)) / (2 eps)  ----")
    print(f"  {'param':>5s}   {'FD-of-L':>15s}   {'jax.grad':>15s}   {'rel-err':>10s}")
    worst_fd = 0.0
    for k, name in enumerate(PARAM_NAMES):
        denom = max(abs(grad_jax[k]), 1e-300)
        rel = abs(grad_fd_l[k] - grad_jax[k]) / denom
        worst_fd = max(worst_fd, rel)
        print(f"  {name:>5s}   {grad_fd_l[k]:+.8e}   {grad_jax[k]:+.8e}   {rel:.2e}")
    print(f"  >>> FD-of-L worst rel err =  {worst_fd:.3e}")

    TOL = 1.0e-8
    print("\n" + "=" * 72)
    print(f"  Tolerance:  {TOL:.0e}")
    print(f"  C-kernel chain rule   vs jax.grad:   {worst_k:.3e}     [{'PASS' if worst_k < TOL else 'FAIL'}]")
    print(f"  Python FD of L        vs jax.grad:   {worst_fd:.3e}     [for reference]")
    print("=" * 72)

    if worst_k < TOL:
        print(f"\n  *** TEST PASSES ***")
        print(f"  The lisa-on-gpu C-kernel chain rule  ((w_d - w_h_CENTER) * dw * N^-1)")
        print(f"  matches jax.grad to {worst_k:.2e} -- essentially double-precision round-off.")
        print(f"\n  The Python FD-of-L form ((L+eps - L-eps)/(2 eps)) disagrees with jax.grad")
        print(f"  by {worst_fd:.2e}, demonstrating the O(eps^2 d^2 w / dtheta^2) bias that arises")
        print(f"  from using the FD midpoint U = (w_+ + w_-)/2 instead of w_h_CENTER as the")
        print(f"  residual anchor.")
    else:
        print(f"\n  *** TEST FAILED ***  C kernel agrees only to {worst_k:.2e} (need {TOL:.0e}).")
        raise AssertionError(f"C-kernel chain rule disagrees with jax.grad by {worst_k:.2e} > {TOL:.0e}")


if __name__ == "__main__":
    main()
