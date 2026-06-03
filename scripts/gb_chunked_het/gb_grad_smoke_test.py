"""
Pure-NumPy smoke check for the chain-rule kernels in gb_chain_rule_grad.py.

We do *not* need a physical GB waveform to validate the chain-rule
arithmetic -- any vector-valued function h(theta) will do.  Here we
take a small analytic mock template

    h(theta)_i = sum_k a_{k,i} * exp(1j * b_{k,i} * theta_k)

evaluated on M=128 frequency bins in 2 channels.  Its Jacobian is
trivial to compute analytically.  We then drop both into the kernels,
recompute L = -1/2 < d - h | d - h > directly, and finite-difference
to confirm the kernels match.

If this smoke check passes, the chain-rule kernels are correct and
gb_grad_test.py (which requires JAX) is comparing physics-level
waveform Jacobians against the same arithmetic.
"""

from __future__ import annotations

import numpy as np

from gb_chain_rule_grad import (
    get_ll_grad_kernel,
    get_ll_grad_kernel_vectorized,
    swap_ll_grad_kernel,
    swap_ll_grad_kernel_vectorized,
)


# ---------------------------------------------------------------------------
#  Mock waveform: h_i = sum_k a_{k,i} * exp(1j * b_{k,i} * theta_k)
# ---------------------------------------------------------------------------

def mock_template_and_jac(theta, M, n_channels, rng):
    """Return (h, dh/dtheta) for the analytic mock waveform.

    h.shape = (n_channels, M),
    dh.shape = (len(theta), n_channels, M).
    """
    n_params = len(theta)
    # fixed random coefficients (passed via rng) -- caller is responsible
    # for reusing the same rng state to keep coefficients consistent.
    a = rng.normal(size=(n_params, n_channels, M)) + 1j * rng.normal(size=(n_params, n_channels, M))
    b = rng.normal(size=(n_params, n_channels, M))
    # h = sum_k a_k * exp(1j * b_k * theta_k)
    phases = np.exp(1j * b * theta[:, None, None])      # (n_params, n_channels, M)
    h = (a * phases).sum(axis=0)                        # (n_channels, M)
    # dh / dtheta_k = a_k * 1j * b_k * exp(1j * b_k * theta_k)
    dh = a * 1j * b * phases                            # (n_params, n_channels, M)
    return h, dh


# ---------------------------------------------------------------------------
#  Direct log-likelihood (no kernel) and finite-difference gradient
# ---------------------------------------------------------------------------

def direct_loglike(theta, M, n_channels, A_data, E_data, A_psd, E_psd, df, start_ind, rng_seed):
    rng = np.random.default_rng(rng_seed)
    h, _ = mock_template_and_jac(theta, M, n_channels, rng)
    # h has shape (n_channels, M).  Pad into the data length via start_ind.
    data_length = A_data.shape[0]
    h_pad_A = np.zeros(data_length, dtype=complex)
    h_pad_E = np.zeros(data_length, dtype=complex)
    h_pad_A[start_ind:start_ind + M] = h[0]
    h_pad_E[start_ind:start_ind + M] = h[1]
    r_A = A_data - h_pad_A
    r_E = E_data - h_pad_E
    inner = 4.0 * df * (
        np.sum(np.conj(r_A) * r_A / A_psd) + np.sum(np.conj(r_E) * r_E / E_psd)
    )
    return -0.5 * inner.real


def finite_diff_grad(theta, eps, *args, **kwargs):
    g = np.zeros_like(theta)
    for k in range(len(theta)):
        tp = theta.copy(); tp[k] += eps
        tm = theta.copy(); tm[k] -= eps
        g[k] = (direct_loglike(tp, *args, **kwargs) - direct_loglike(tm, *args, **kwargs)) / (2 * eps)
    return g


# ---------------------------------------------------------------------------
#  Drivers
# ---------------------------------------------------------------------------

def run_get_ll_check():
    print("=" * 72)
    print(" get_ll_grad   chain-rule  vs   finite-difference")
    print("=" * 72)

    rng_master = np.random.default_rng(0)
    n_params = 9
    M = 64
    n_channels = 2
    data_length = 4096
    df = 1.0 / 1024.0
    start_ind = 1000

    theta = np.linspace(-0.2, 0.2, n_params)

    # build data = h_inj  +  small noise so residual is non-trivial
    rng_inj = np.random.default_rng(123)
    h_inj, _ = mock_template_and_jac(theta + 0.05, M, n_channels, rng_inj)
    A_data = np.zeros(data_length, dtype=complex)
    E_data = np.zeros(data_length, dtype=complex)
    A_data[start_ind:start_ind + M] = h_inj[0]
    E_data[start_ind:start_ind + M] = h_inj[1]

    A_psd = np.full(data_length, 1e-40)
    E_psd = np.full(data_length, 1e-40)

    # template + jacobian at theta
    rng_tpl = np.random.default_rng(123)
    h, dh = mock_template_and_jac(theta, M, n_channels, rng_tpl)
    A_template = h[0][:, None]                             # (M, 1)
    E_template = h[1][:, None]
    dA_template = dh[:, 0, :][:, :, None]                  # (n_params, M, 1)
    dE_template = dh[:, 1, :][:, :, None]

    # chain-rule kernels
    start_ind_all = np.array([start_ind], dtype=np.int64)
    data_index = np.zeros(1, dtype=np.int32)
    noise_index = np.zeros(1, dtype=np.int32)

    grad_out_loop = np.zeros((n_params, 1))
    get_ll_grad_kernel(
        grad_out_loop, A_template, E_template, dA_template, dE_template,
        A_data, E_data, A_psd, E_psd, df, start_ind_all, M, 1,
        data_index, noise_index, data_length,
    )
    grad_loop = grad_out_loop[:, 0]

    grad_vec = get_ll_grad_kernel_vectorized(
        A_template, E_template, dA_template, dE_template,
        A_data, E_data, A_psd, E_psd, df, start_ind_all,
        data_index, noise_index, data_length,
    )[:, 0]

    # finite-difference reference
    grad_fd = finite_diff_grad(
        theta, 1e-5,
        M, n_channels, A_data, E_data, A_psd, E_psd, df, start_ind, 123,
    )

    print("{:>3s}   {:>15s}   {:>15s}   {:>15s}   {:>10s}".format(
        "k", "kernel(loop)", "kernel(vec)", "finite-diff", "rel"))
    for k in range(n_params):
        rel = abs(grad_loop[k] - grad_fd[k]) / (abs(grad_fd[k]) + 1e-300)
        print("{:>3d}   {:+.6e}   {:+.6e}   {:+.6e}   {:.2e}".format(
            k, grad_loop[k], grad_vec[k], grad_fd[k], rel))

    err_loop = np.max(np.abs(grad_loop - grad_fd)) / (np.max(np.abs(grad_fd)) + 1e-300)
    err_vec = np.max(np.abs(grad_vec - grad_fd)) / (np.max(np.abs(grad_fd)) + 1e-300)
    print(f"\n  max rel err loop = {err_loop:.3e},  vectorized = {err_vec:.3e}")
    assert err_loop < 1e-5 and err_vec < 1e-5
    print("  get_ll PASS")


def direct_swap_ll_diff(
    theta_add, theta_remove, M, n_channels,
    A_data, E_data, A_psd, E_psd, df,
    start_ind_remove, start_ind_add,
    rng_seed_add, rng_seed_remove,
):
    """Compute  ll_diff = L(after swap) - L(before swap).

    Here ``A_data`` / ``E_data`` already have h_remove subtracted, i.e.
    they are the residual *at the current state*.  Cf. the convention
    in GBGPUBase.swap_likelihood_difference.
    """
    data_length = A_data.shape[0]
    rng_a = np.random.default_rng(rng_seed_add)
    rng_r = np.random.default_rng(rng_seed_remove)
    h_a, _ = mock_template_and_jac(theta_add, M, n_channels, rng_a)
    h_r, _ = mock_template_and_jac(theta_remove, M, n_channels, rng_r)

    h_a_pad = np.zeros((n_channels, data_length), dtype=complex)
    h_r_pad = np.zeros((n_channels, data_length), dtype=complex)
    h_a_pad[:, start_ind_add:start_ind_add + M] = h_a
    h_r_pad[:, start_ind_remove:start_ind_remove + M] = h_r

    r_before_A = A_data
    r_before_E = E_data
    r_after_A = A_data - h_a_pad[0] + h_r_pad[0]
    r_after_E = E_data - h_a_pad[1] + h_r_pad[1]

    def _norm(rA, rE):
        return 4.0 * df * (
            np.sum(np.conj(rA) * rA / A_psd) + np.sum(np.conj(rE) * rE / E_psd)
        )

    L_before = -0.5 * _norm(r_before_A, r_before_E).real
    L_after = -0.5 * _norm(r_after_A, r_after_E).real
    return L_after - L_before


def run_swap_ll_check():
    print("\n" + "=" * 72)
    print(" swap_ll_grad   chain-rule  vs   finite-difference")
    print("=" * 72)

    n_params = 9
    M = 64
    n_channels = 2
    data_length = 4096
    df = 1.0 / 1024.0
    start_ind_remove = 1000
    start_ind_add = 1020          # partial overlap with remove (intentional)

    theta_remove = np.linspace(-0.2, 0.2, n_params)
    theta_add = theta_remove + 0.07

    # data after subtracting h_remove (i.e. d_passed in the swap convention)
    rng = np.random.default_rng(7)
    d_passed_A = rng.normal(size=data_length) + 1j * rng.normal(size=data_length)
    d_passed_E = rng.normal(size=data_length) + 1j * rng.normal(size=data_length)
    A_psd = np.full(data_length, 1e-40)
    E_psd = np.full(data_length, 1e-40)

    # template + jacobian at the two parameter sets
    rng_a = np.random.default_rng(123)
    rng_r = np.random.default_rng(456)
    h_add, dh_add = mock_template_and_jac(theta_add, M, n_channels, rng_a)
    h_rem, dh_rem = mock_template_and_jac(theta_remove, M, n_channels, rng_r)

    A_add = h_add[0][:, None]; E_add = h_add[1][:, None]
    A_rem = h_rem[0][:, None]; E_rem = h_rem[1][:, None]
    dA_add = dh_add[:, 0, :][:, :, None]; dE_add = dh_add[:, 1, :][:, :, None]
    dA_rem = dh_rem[:, 0, :][:, :, None]; dE_rem = dh_rem[:, 1, :][:, :, None]

    data_index = np.zeros(1, dtype=np.int32)
    noise_index = np.zeros(1, dtype=np.int32)
    start_rem_arr = np.array([start_ind_remove], dtype=np.int64)
    start_add_arr = np.array([start_ind_add], dtype=np.int64)

    # ---- kernel (loop) ----
    grad_add_loop = np.zeros((n_params, 1)); grad_rem_loop = np.zeros((n_params, 1))
    swap_ll_grad_kernel(
        grad_add_loop, grad_rem_loop,
        A_rem, E_rem, start_rem_arr,
        A_add, E_add, start_add_arr,
        dA_rem, dE_rem, dA_add, dE_add,
        d_passed_A, d_passed_E, A_psd, E_psd,
        df, M, 1, data_index, noise_index, data_length,
    )

    # ---- kernel (vec) ----
    grad_add_vec, grad_rem_vec = swap_ll_grad_kernel_vectorized(
        A_rem, E_rem, start_rem_arr,
        A_add, E_add, start_add_arr,
        dA_rem, dE_rem, dA_add, dE_add,
        d_passed_A, d_passed_E, A_psd, E_psd, df,
        data_index, noise_index, data_length,
    )

    # ---- finite-difference reference ----
    eps = 1e-5
    grad_add_fd = np.zeros(n_params); grad_rem_fd = np.zeros(n_params)
    for k in range(n_params):
        tp = theta_add.copy(); tp[k] += eps
        tm = theta_add.copy(); tm[k] -= eps
        grad_add_fd[k] = (
            direct_swap_ll_diff(tp, theta_remove, M, n_channels, d_passed_A, d_passed_E, A_psd, E_psd, df, start_ind_remove, start_ind_add, 123, 456)
            - direct_swap_ll_diff(tm, theta_remove, M, n_channels, d_passed_A, d_passed_E, A_psd, E_psd, df, start_ind_remove, start_ind_add, 123, 456)
        ) / (2 * eps)
        tp = theta_remove.copy(); tp[k] += eps
        tm = theta_remove.copy(); tm[k] -= eps
        grad_rem_fd[k] = (
            direct_swap_ll_diff(theta_add, tp, M, n_channels, d_passed_A, d_passed_E, A_psd, E_psd, df, start_ind_remove, start_ind_add, 123, 456)
            - direct_swap_ll_diff(theta_add, tm, M, n_channels, d_passed_A, d_passed_E, A_psd, E_psd, df, start_ind_remove, start_ind_add, 123, 456)
        ) / (2 * eps)

    print("\n  >>> theta_add gradient")
    print("{:>3s}   {:>15s}   {:>15s}   {:>15s}   {:>10s}".format("k", "kernel(loop)", "kernel(vec)", "finite-diff", "rel"))
    for k in range(n_params):
        rel = abs(grad_add_loop[k, 0] - grad_add_fd[k]) / (abs(grad_add_fd[k]) + 1e-300)
        print("{:>3d}   {:+.6e}   {:+.6e}   {:+.6e}   {:.2e}".format(
            k, grad_add_loop[k, 0], grad_add_vec[k, 0], grad_add_fd[k], rel))

    print("\n  >>> theta_remove gradient")
    print("{:>3s}   {:>15s}   {:>15s}   {:>15s}   {:>10s}".format("k", "kernel(loop)", "kernel(vec)", "finite-diff", "rel"))
    for k in range(n_params):
        rel = abs(grad_rem_loop[k, 0] - grad_rem_fd[k]) / (abs(grad_rem_fd[k]) + 1e-300)
        print("{:>3d}   {:+.6e}   {:+.6e}   {:+.6e}   {:.2e}".format(
            k, grad_rem_loop[k, 0], grad_rem_vec[k, 0], grad_rem_fd[k], rel))

    err_add_loop = np.max(np.abs(grad_add_loop[:, 0] - grad_add_fd)) / (np.max(np.abs(grad_add_fd)) + 1e-300)
    err_add_vec = np.max(np.abs(grad_add_vec[:, 0] - grad_add_fd)) / (np.max(np.abs(grad_add_fd)) + 1e-300)
    err_rem_loop = np.max(np.abs(grad_rem_loop[:, 0] - grad_rem_fd)) / (np.max(np.abs(grad_rem_fd)) + 1e-300)
    err_rem_vec = np.max(np.abs(grad_rem_vec[:, 0] - grad_rem_fd)) / (np.max(np.abs(grad_rem_fd)) + 1e-300)

    print(f"\n  add:    loop={err_add_loop:.3e}   vec={err_add_vec:.3e}")
    print(f"  remove: loop={err_rem_loop:.3e}   vec={err_rem_vec:.3e}")
    assert err_add_loop < 1e-5 and err_rem_loop < 1e-5
    assert err_add_vec < 1e-5 and err_rem_vec < 1e-5
    print("  swap_ll PASS")


if __name__ == "__main__":
    run_get_ll_check()
    run_swap_ll_check()
    print("\nALL CHAIN-RULE KERNEL CHECKS PASSED")
