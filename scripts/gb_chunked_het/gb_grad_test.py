"""
Cross-check the chain-rule gradients of get_ll / swap_likelihood_difference
against full JAX autodiff of the same Gaussian log-likelihood, using the
JAX reimplementation of the GBGPU waveform from gb_jax_waveform.py.

Pipeline
--------
  Method A (chain rule)
    1. h(theta), dh(theta)/dtheta  <-- gb_jax_waveform + jacfwd
    2. grad L  = get_ll_grad_kernel_vectorized(h, dh, d, S)

  Method B (full autodiff)
    1. L(theta) = -1/2 < d - h(theta) | d - h(theta) >, with h built by
       gb_jax_waveform inside a jax.grad call
    2. grad L = jax.grad(L)(theta)

Both should agree to ~1e-8 in double precision.  The same comparison is
then repeated for swap_likelihood_difference, whose gradient breaks into
``grad_add`` and ``grad_remove`` pieces.
"""

from __future__ import annotations

import numpy as np
import jax
import jax.numpy as jnp

from gb_jax_waveform import (
    gb_jax_waveform,
    gb_jax_template_and_jacobian,
    PARAM_NAMES,
    N_PARAMS,
)
from gb_chain_rule_grad import (
    get_ll_grad_kernel_vectorized,
    swap_ll_grad_kernel_vectorized,
)


YRSID_SI = 31558149.763545603


# ---------------------------------------------------------------------------
#  Test configuration
# ---------------------------------------------------------------------------

# observation window
DT = 15.0
T_OBS = 0.25 * YRSID_SI           # 3 months -- keep small for speed
N_OBS = int(T_OBS / DT)
T_OBS = N_OBS * DT
DF = 1.0 / T_OBS
FMAX = 1.0 / (2.0 * DT)
DATA_LENGTH = int(np.floor(FMAX / DF)) + 1     # length of one positive-freq data stream

# slow-part length used by the C waveform / our JAX waveform.
# Must be a power of two; pick large enough for the chosen source.
N_SLOW = 1024

# galactic-binary injection (truth)
THETA_INJ = jnp.array([
    1.0e-22,     # amp
    2.5e-3,      # f0  (Hz)
    1.0e-17,     # fdot
    0.0,         # fddot
    0.7,         # phi0
    0.9,         # iota
    1.2,         # psi
    3.4,         # lam (ecliptic longitude)
    -0.4,        # beta (ecliptic latitude)
])

# slightly different "template" parameters for grad check
THETA_TPL = THETA_INJ * jnp.array([
    1.1,   # amp +10%
    1.0 + 1e-6,
    1.0,
    1.0,
    1.0 + 0.05,
    1.0 - 0.02,
    1.0 + 0.03,
    1.0 - 0.01,
    1.0 + 0.04,
])

# "remove" / "add" templates for the swap-likelihood gradient
THETA_REMOVE = THETA_INJ
THETA_ADD = THETA_INJ + jnp.array([
    0.0, 5.0e-8, 0.0, 0.0, 0.05, 0.0, 0.0, 0.0, 0.0
])    # offset f0 by a few Fourier bins


# ---------------------------------------------------------------------------
#  Helpers
# ---------------------------------------------------------------------------

def place_template_in_data(h_local, start_ind, data_length):
    """Pad a length-N template starting at ``start_ind`` into a length-``data_length`` array."""
    nch, N = h_local.shape
    out = jnp.zeros((nch, data_length), dtype=h_local.dtype)
    idx = start_ind + jnp.arange(N)
    out = out.at[:, idx].set(h_local)
    return out


def build_data_and_psd(theta_inj, T, N, data_length, tdi="AE", noise_amp=2e-41):
    """Inject a single binary into a flat-PSD dataset."""
    h_inj, start_inj = gb_jax_waveform(*theta_inj, T=T, N=N, tdi_channel_setup=tdi)
    d = place_template_in_data(h_inj, int(start_inj), data_length)
    # flat PSD -- easy to read off, real test of the math, not the noise model.
    psd = jnp.full((d.shape[0], data_length), noise_amp, dtype=jnp.float64)
    return d, psd, h_inj, int(start_inj)


def inner_product_complex(a, b, psd, df):
    """4 df sum_i conj(a_i) * b_i / S_i, summed over all channels and bins."""
    return 4.0 * df * jnp.sum(jnp.conj(a) * b / psd)


def gaussian_loglike(d, h, psd, df):
    """L = -1/2 Re < d - h | d - h > over all channels."""
    r = d - h
    return -0.5 * jnp.real(inner_product_complex(r, r, psd, df))


# ---------------------------------------------------------------------------
#  Full-autodiff likelihood as a function of theta
# ---------------------------------------------------------------------------

def make_full_ll(d, psd, start_ind_inj, T, N, tdi="AE"):
    """Build L(theta) at fixed data, treating start_ind of the template as fixed.

    NOTE: ``start_ind`` for the template depends on f0 through round(),
    which is non-differentiable.  We freeze it at start_ind_inj (the
    nearest-Fourier-bin index of the data) -- the chain-rule kernel does
    the same.  For small perturbations off the injection this is exact.
    """
    def _ll(theta):
        h_local, _ = gb_jax_waveform(*theta, T=T, N=N, tdi_channel_setup=tdi)
        h_padded = place_template_in_data(h_local, start_ind_inj, d.shape[-1])
        return gaussian_loglike(d, h_padded, psd, 1.0 / T)
    return _ll


# ---------------------------------------------------------------------------
#  Main: build data, evaluate gradients, compare
# ---------------------------------------------------------------------------

def main():
    print("=" * 72)
    print("Test config:  T = {:.3e} s  dt = {:.1f} s  N_slow = {:d}  DATA_LENGTH = {:d}".format(
        float(T_OBS), float(DT), N_SLOW, DATA_LENGTH
    ))
    print("=" * 72)

    # ---------------------------------------------------------------------
    # Build "data" by injecting the truth into a flat-PSD stream
    # ---------------------------------------------------------------------
    d_full, psd_full, h_inj, start_inj = build_data_and_psd(
        THETA_INJ, T_OBS, N_SLOW, DATA_LENGTH, tdi="AE",
    )
    print(f"\n[injection] start_ind = {start_inj},  q_inj = {start_inj + N_SLOW // 2}")

    # =====================================================================
    #   PART 1   --   get_ll gradient
    # =====================================================================
    print("\n" + "-" * 72)
    print("PART 1  --  get_ll gradient")
    print("-" * 72)

    # template at THETA_TPL (so residual is non-zero)
    h_tpl, jac_tpl, start_tpl = gb_jax_template_and_jacobian(
        THETA_TPL, T=T_OBS, N=N_SLOW, tdi_channel_setup="AE",
    )
    # match start_ind for the chain-rule kernel: use the template's own start_ind
    # but place into a stream with data still at start_inj.
    print(f"[template]  start_ind = {int(start_tpl)},  q_tpl = {int(start_tpl) + N_SLOW // 2}")

    # ---- Method A : chain rule ----
    # data_minus_template here means "data minus all *other* templates";
    # here there's only one active binary so that's just the data itself.
    A_template = np.asarray(h_tpl[0])[:, None]                      # (M, 1)
    E_template = np.asarray(h_tpl[1])[:, None]
    dA_template = np.asarray(jac_tpl[:, 0, :])[:, :, None]          # (n_params, M, 1)
    dE_template = np.asarray(jac_tpl[:, 1, :])[:, :, None]
    A_data = np.asarray(d_full[0])
    E_data = np.asarray(d_full[1])
    A_psd = np.asarray(psd_full[0])
    E_psd = np.asarray(psd_full[1])
    start_ind_all = np.array([int(start_tpl)], dtype=np.int64)
    data_index = np.zeros(1, dtype=np.int32)
    noise_index = np.zeros(1, dtype=np.int32)

    grad_chain = get_ll_grad_kernel_vectorized(
        A_template, E_template, dA_template, dE_template,
        A_data, E_data, A_psd, E_psd, DF, start_ind_all,
        data_index, noise_index, DATA_LENGTH,
    )                                                             # (n_params, 1)
    grad_chain = np.asarray(grad_chain[:, 0])

    # ---- Method B : full autodiff ----
    full_ll = make_full_ll(d_full, psd_full, int(start_tpl),
                           T_OBS, N_SLOW, tdi="AE")
    grad_full = np.asarray(jax.grad(full_ll)(THETA_TPL))

    # ---- compare ----
    print("\n{:>8s}   {:>15s}   {:>15s}   {:>10s}".format(
        "param", "chain-rule", "jax.grad", "rel-err"))
    for k, name in enumerate(PARAM_NAMES):
        rel = abs(grad_chain[k] - grad_full[k]) / (abs(grad_full[k]) + 1e-300)
        print("{:>8s}   {:+.8e}   {:+.8e}   {:.2e}".format(name, grad_chain[k], grad_full[k], rel))

    abs_err = np.max(np.abs(grad_chain - grad_full))
    rel_err = abs_err / (np.max(np.abs(grad_full)) + 1e-300)
    print(f"\n[get_ll]  max abs err = {abs_err:.3e},  max rel err = {rel_err:.3e}")
    assert rel_err < 1e-6, "chain-rule gradient disagrees with autodiff"

    # =====================================================================
    #   PART 2   --   swap_likelihood_difference gradient
    # =====================================================================
    print("\n" + "-" * 72)
    print("PART 2  --  swap_likelihood_difference gradient")
    print("-" * 72)

    # waveforms + jacobians at the add and remove parameters
    h_rem, jac_rem, start_rem = gb_jax_template_and_jacobian(
        THETA_REMOVE, T=T_OBS, N=N_SLOW, tdi_channel_setup="AE",
    )
    h_add, jac_add, start_add = gb_jax_template_and_jacobian(
        THETA_ADD,    T=T_OBS, N=N_SLOW, tdi_channel_setup="AE",
    )
    print(f"[remove]  start_ind = {int(start_rem)},  q = {int(start_rem) + N_SLOW // 2}")
    print(f"[add]     start_ind = {int(start_add)},  q = {int(start_add) + N_SLOW // 2}")

    # Pre-state residual: data with the remove template subtracted (other templates = 0).
    h_rem_padded = place_template_in_data(h_rem, int(start_rem), DATA_LENGTH)
    d_passed = d_full - h_rem_padded

    # ---- Method A : chain rule ----
    A_rem = np.asarray(h_rem[0])[:, None]
    E_rem = np.asarray(h_rem[1])[:, None]
    A_add_arr = np.asarray(h_add[0])[:, None]
    E_add_arr = np.asarray(h_add[1])[:, None]
    dA_rem = np.asarray(jac_rem[:, 0, :])[:, :, None]
    dE_rem = np.asarray(jac_rem[:, 1, :])[:, :, None]
    dA_add = np.asarray(jac_add[:, 0, :])[:, :, None]
    dE_add = np.asarray(jac_add[:, 1, :])[:, :, None]

    grad_add_chain, grad_rem_chain = swap_ll_grad_kernel_vectorized(
        A_rem, E_rem, np.array([int(start_rem)], dtype=np.int64),
        A_add_arr, E_add_arr, np.array([int(start_add)], dtype=np.int64),
        dA_rem, dE_rem, dA_add, dE_add,
        np.asarray(d_passed[0]), np.asarray(d_passed[1]),
        A_psd, E_psd, DF,
        data_index, noise_index, DATA_LENGTH,
    )
    grad_add_chain = np.asarray(grad_add_chain[:, 0])
    grad_rem_chain = np.asarray(grad_rem_chain[:, 0])

    # ---- Method B : full autodiff of L_after - L_before ----
    def _ll_diff(theta_add, theta_remove):
        h_a, _ = gb_jax_waveform(*theta_add,    T=T_OBS, N=N_SLOW, tdi_channel_setup="AE")
        h_r, _ = gb_jax_waveform(*theta_remove, T=T_OBS, N=N_SLOW, tdi_channel_setup="AE")
        h_a_p = place_template_in_data(h_a, int(start_add), DATA_LENGTH)
        h_r_p = place_template_in_data(h_r, int(start_rem), DATA_LENGTH)
        r_before = d_passed
        r_after = d_passed - h_a_p + h_r_p
        L_before = -0.5 * jnp.real(inner_product_complex(r_before, r_before, psd_full, DF))
        L_after = -0.5 * jnp.real(inner_product_complex(r_after, r_after, psd_full, DF))
        return L_after - L_before

    grad_add_full = np.asarray(jax.grad(_ll_diff, argnums=0)(THETA_ADD, THETA_REMOVE))
    grad_rem_full = np.asarray(jax.grad(_ll_diff, argnums=1)(THETA_ADD, THETA_REMOVE))

    print("\n  >>> theta_add gradient")
    print("{:>8s}   {:>15s}   {:>15s}   {:>10s}".format(
        "param", "chain-rule", "jax.grad", "rel-err"))
    for k, name in enumerate(PARAM_NAMES):
        rel = abs(grad_add_chain[k] - grad_add_full[k]) / (abs(grad_add_full[k]) + 1e-300)
        print("{:>8s}   {:+.8e}   {:+.8e}   {:.2e}".format(name, grad_add_chain[k], grad_add_full[k], rel))

    print("\n  >>> theta_remove gradient")
    print("{:>8s}   {:>15s}   {:>15s}   {:>10s}".format(
        "param", "chain-rule", "jax.grad", "rel-err"))
    for k, name in enumerate(PARAM_NAMES):
        rel = abs(grad_rem_chain[k] - grad_rem_full[k]) / (abs(grad_rem_full[k]) + 1e-300)
        print("{:>8s}   {:+.8e}   {:+.8e}   {:.2e}".format(name, grad_rem_chain[k], grad_rem_full[k], rel))

    abs_err_a = np.max(np.abs(grad_add_chain - grad_add_full))
    rel_err_a = abs_err_a / (np.max(np.abs(grad_add_full)) + 1e-300)
    abs_err_r = np.max(np.abs(grad_rem_chain - grad_rem_full))
    rel_err_r = abs_err_r / (np.max(np.abs(grad_rem_full)) + 1e-300)
    print(f"\n[swap add]    max abs err = {abs_err_a:.3e},  max rel err = {rel_err_a:.3e}")
    print(f"[swap remove] max abs err = {abs_err_r:.3e},  max rel err = {rel_err_r:.3e}")
    assert rel_err_a < 1e-6 and rel_err_r < 1e-6, "swap chain-rule disagrees with autodiff"

    print("\nALL CHECKS PASSED")


if __name__ == "__main__":
    main()
