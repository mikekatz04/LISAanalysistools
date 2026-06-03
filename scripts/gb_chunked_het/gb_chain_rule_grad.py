"""
Chain-rule gradients of the GBGPU likelihood kernels with respect to
the 9 galactic-binary parameters

    theta = (amp, f0, fdot, fddot, phi0, iota, psi, lam, beta).

For a Gaussian log-likelihood

    L(theta) = -1/2 < d - h(theta) | d - h(theta) >,    < a|b > = 4 df sum_i a_i^* b_i / S_i,

the gradient with respect to a real-valued parameter is

    dL/dtheta_k = 4 df sum_i Re[ (d - h)_i^* * (dh_i/dtheta_k) ] / S_i,

summed across TDI channels.  This module exposes two kernels mirroring
the existing GBGPU C kernels:

    get_ll_grad_kernel(...)      -- gradient of get_ll (single template per binary)
    swap_ll_grad_kernel(...)     -- gradient of swap_ll_diff (paired add/remove templates)

The kernels are pure NumPy / CuPy and follow the exact memory layout
used by the C versions in src/gbgpu/cutils/gbgpu_utils.cu:

    A_template / E_template : shape (M, num_bin), C-contiguous so
        A_template[i, bin_i] corresponds to A_template[i*num_bin + bin_i].

The waveform parameter-Jacobian must be precomputed and passed in:

    dA_template / dE_template : shape (n_params, M, num_bin) complex,
        with dA_template[k, i, bin_i] = d h_A(i, bin_i) / d theta_k.

This Jacobian is most easily produced by autodiff of the JAX
reimplementation of the C waveform in gb_jax_waveform.py.

In addition to the kernels, two thin Python wrappers are provided
that follow the call signature of ``GBGPUBase.get_ll`` and
``GBGPUBase.swap_likelihood_difference``, accepting a stacked
``params`` array and a Jacobian tensor and returning the per-binary
parameter gradient.
"""

from __future__ import annotations

import numpy as np


N_PARAMS = 9
_PARAM_NAMES = ("amp", "f0", "fdot", "fddot", "phi0", "iota", "psi", "lam", "beta")


# ---------------------------------------------------------------------------
#  get_ll gradient
# ---------------------------------------------------------------------------

def get_ll_grad_kernel(
    grad_out,
    A_template,
    E_template,
    dA_template,
    dE_template,
    A_data,
    E_data,
    A_psd,
    E_psd,
    df,
    start_ind_all,
    M,
    num_bin,
    data_index,
    noise_index,
    data_length,
):
    """Per-binary parameter gradient of ``get_ll``.

    Direct translation of the ``get_ll`` C kernel with the inner accumulator
    replaced by the chain-rule expression

        grad_k += Re[ conj(d - h) * dh/dtheta_k ] / S

    summed across A and E channels and the M-point support of each
    template, then multiplied by 4 df.

    Parameters
    ----------
    grad_out : (n_params, num_bin) real array, modified in place.
    A_template, E_template : (M, num_bin) complex.
    dA_template, dE_template : (n_params, M, num_bin) complex,
        parameter-Jacobian of the template.
    A_data, E_data : (num_data * data_length,) complex, layout matches
        the C kernel: bin uses ``data_index[bin_i] * data_length + j``.
    A_psd, E_psd : (num_psd * data_length,) real, same layout.
    df : Fourier bin spacing (1/T).
    start_ind_all : (num_bin,) int, global frequency index of the
        first sample of each template.
    M : template length (== N from run_wave).
    num_bin : number of binaries.
    data_index, noise_index : (num_bin,) int32.
    data_length : length of one data realization (per channel).
    """
    n_params = dA_template.shape[0]
    assert dE_template.shape[0] == n_params

    for bin_i in range(num_bin):
        start_ind = int(start_ind_all[bin_i])
        d_idx = int(data_index[bin_i])
        n_idx = int(noise_index[bin_i])

        # initialize accumulator for this binary
        acc = np.zeros(n_params, dtype=np.float64)

        for i in range(M):
            j = start_ind + i
            # (no bounds check -- matches the C get_ll kernel)
            A_noise = A_psd[n_idx * data_length + j]
            E_noise = E_psd[n_idx * data_length + j]

            h_A = A_template[i, bin_i]
            h_E = E_template[i, bin_i]
            d_A = A_data[d_idx * data_length + j]
            d_E = E_data[d_idx * data_length + j]

            # residual d - h at the current parameters
            r_A = d_A - h_A
            r_E = d_E - h_E

            for k in range(n_params):
                dh_A = dA_template[k, i, bin_i]
                dh_E = dE_template[k, i, bin_i]
                # Re[conj(r) * dh] = r.real * dh.real + r.imag * dh.imag
                acc[k] += (r_A.real * dh_A.real + r_A.imag * dh_A.imag) / A_noise
                acc[k] += (r_E.real * dh_E.real + r_E.imag * dh_E.imag) / E_noise

        grad_out[:, bin_i] = 4.0 * df * acc


def get_ll_grad_kernel_vectorized(
    A_template,
    E_template,
    dA_template,
    dE_template,
    A_data,
    E_data,
    A_psd,
    E_psd,
    df,
    start_ind_all,
    data_index,
    noise_index,
    data_length,
    xp=np,
):
    """Vectorized version of :func:`get_ll_grad_kernel` for one shared M.

    Returns ``grad_out`` of shape ``(n_params, num_bin)``.

    Assumes a single common template length ``M = A_template.shape[0]``;
    if templates of different lengths are needed, call the per-binary
    kernel in a loop (the C wrapper already handles this by grouping
    binaries with equal ``N``).
    """
    M, num_bin = A_template.shape
    n_params = dA_template.shape[0]

    # build (M, num_bin) index of global frequency bin j = start_ind + i
    i_arr = xp.arange(M, dtype=xp.int64)[:, None]
    j_arr = i_arr + xp.asarray(start_ind_all, dtype=xp.int64)[None, :]   # (M, num_bin)

    # gather data / psd at these indices, channel by channel
    d_off = xp.asarray(data_index, dtype=xp.int64)[None, :] * data_length
    n_off = xp.asarray(noise_index, dtype=xp.int64)[None, :] * data_length

    d_A = A_data[d_off + j_arr]                  # (M, num_bin) complex
    d_E = E_data[d_off + j_arr]
    S_A = A_psd[n_off + j_arr]                   # (M, num_bin) real
    S_E = E_psd[n_off + j_arr]

    r_A = d_A - A_template                       # (M, num_bin)
    r_E = d_E - E_template

    # Re[ conj(r) * dh ] summed over M
    # broadcast: (1, M, num_bin) * (n_params, M, num_bin) -> sum over axis=1
    term_A = (r_A.conj()[None] * dA_template).real / S_A[None]
    term_E = (r_E.conj()[None] * dE_template).real / S_E[None]

    grad_out = 4.0 * df * (term_A + term_E).sum(axis=1)    # (n_params, num_bin)
    return grad_out


# ---------------------------------------------------------------------------
#  swap_ll_diff gradient
# ---------------------------------------------------------------------------

def swap_ll_grad_kernel(
    grad_out_add,
    grad_out_remove,
    A_remove,
    E_remove,
    start_ind_all_remove,
    A_add,
    E_add,
    start_ind_all_add,
    dA_remove,
    dE_remove,
    dA_add,
    dE_add,
    A_data,
    E_data,
    A_psd,
    E_psd,
    df,
    M,
    num_bin,
    data_index,
    noise_index,
    data_length,
):
    """Per-binary parameter gradient of ``swap_likelihood_difference``.

    The C kernel computes

        ll_diff = -1/2 Re[ -2 d_h_add + 2 d_h_remove
                           - 2 add_remove + add_add + remove_remove ]

    which is the log-likelihood ratio  L(after) - L(before)  when the
    "remove" template at the current state is swapped out for the
    "add" template.  With  d  the residual at the *current* state
    (data minus every active template including ``h_remove``), the
    post-swap residual is

        r_after = d - h_add + h_remove,

    and the gradients are

        d(ll_diff)/d(theta_add)    = +4 df  Re < r_after | dh_add / dtheta_add >
        d(ll_diff)/d(theta_remove) = -4 df  Re < r_after | dh_remove / dtheta_remove >.

    Both inner products are restricted to the union of the support of
    ``h_add`` and ``h_remove``, exactly as in the C kernel.

    Parameters
    ----------
    grad_out_add, grad_out_remove : (n_params, num_bin) real, modified in place.
    A_remove / E_remove : (M, num_bin) complex.
    A_add / E_add : (M, num_bin) complex.
    dA_remove / dE_remove : (n_params, M, num_bin) complex, Jacobian of
        the remove template w.r.t. its 9 parameters.
    dA_add / dE_add : (n_params, M, num_bin) complex, Jacobian of the
        add template w.r.t. its 9 parameters.
    A_data, E_data, A_psd, E_psd, df, data_index, noise_index, data_length
        as in :func:`get_ll_grad_kernel`.
    M, num_bin, start_ind_all_* as in the swap_ll_diff C kernel.
    """
    n_params = dA_add.shape[0]

    for bin_i in range(num_bin):
        s_rem = int(start_ind_all_remove[bin_i])
        s_add = int(start_ind_all_add[bin_i])
        d_idx = int(data_index[bin_i])
        n_idx = int(noise_index[bin_i])

        # window covering both add and remove supports
        lo = min(s_rem, s_add)
        hi = max(s_rem + M, s_add + M)

        acc_add = np.zeros(n_params, dtype=np.float64)
        acc_rem = np.zeros(n_params, dtype=np.float64)

        for j in range(lo, hi):
            A_noise = A_psd[n_idx * data_length + j]
            E_noise = E_psd[n_idx * data_length + j]
            d_A = A_data[d_idx * data_length + j]
            d_E = E_data[d_idx * data_length + j]

            in_add = (s_add <= j) and (j < s_add + M)
            in_rem = (s_rem <= j) and (j < s_rem + M)

            h_A_add = A_add[j - s_add, bin_i] if in_add else 0.0
            h_E_add = E_add[j - s_add, bin_i] if in_add else 0.0
            h_A_rem = A_remove[j - s_rem, bin_i] if in_rem else 0.0
            h_E_rem = E_remove[j - s_rem, bin_i] if in_rem else 0.0

            # post-swap residual
            r_A = d_A - h_A_add + h_A_rem
            r_E = d_E - h_E_add + h_E_rem

            for k in range(n_params):
                if in_add:
                    dh_A = dA_add[k, j - s_add, bin_i]
                    dh_E = dE_add[k, j - s_add, bin_i]
                    acc_add[k] += (r_A.real * dh_A.real + r_A.imag * dh_A.imag) / A_noise
                    acc_add[k] += (r_E.real * dh_E.real + r_E.imag * dh_E.imag) / E_noise
                if in_rem:
                    dh_A = dA_remove[k, j - s_rem, bin_i]
                    dh_E = dE_remove[k, j - s_rem, bin_i]
                    acc_rem[k] += (r_A.real * dh_A.real + r_A.imag * dh_A.imag) / A_noise
                    acc_rem[k] += (r_E.real * dh_E.real + r_E.imag * dh_E.imag) / E_noise

        grad_out_add[:, bin_i] = +4.0 * df * acc_add
        grad_out_remove[:, bin_i] = -4.0 * df * acc_rem


def swap_ll_grad_kernel_vectorized(
    A_remove,
    E_remove,
    start_ind_all_remove,
    A_add,
    E_add,
    start_ind_all_add,
    dA_remove,
    dE_remove,
    dA_add,
    dE_add,
    A_data,
    E_data,
    A_psd,
    E_psd,
    df,
    data_index,
    noise_index,
    data_length,
    xp=np,
):
    """Vectorized version of :func:`swap_ll_grad_kernel`.

    Returns ``(grad_out_add, grad_out_remove)`` each shaped ``(n_params, num_bin)``.

    The implementation evaluates each template on its own M-point grid,
    gathers the data / PSD samples at the matching global indices,
    forms ``r_after`` correctly *in the overlap* by adding the
    cross-template sample, and finally contracts with the Jacobians.
    """
    M, num_bin = A_add.shape
    n_params = dA_add.shape[0]
    assert A_remove.shape == (M, num_bin)
    assert dA_remove.shape == (n_params, M, num_bin)

    i_arr = xp.arange(M, dtype=xp.int64)[:, None]
    j_add = i_arr + xp.asarray(start_ind_all_add, dtype=xp.int64)[None, :]      # (M, num_bin)
    j_rem = i_arr + xp.asarray(start_ind_all_remove, dtype=xp.int64)[None, :]    # (M, num_bin)
    d_off = xp.asarray(data_index, dtype=xp.int64)[None, :] * data_length
    n_off = xp.asarray(noise_index, dtype=xp.int64)[None, :] * data_length

    # gather data/psd on each template's grid
    d_A_add = A_data[d_off + j_add]; d_E_add = E_data[d_off + j_add]
    S_A_add = A_psd[n_off + j_add];  S_E_add = E_psd[n_off + j_add]
    d_A_rem = A_data[d_off + j_rem]; d_E_rem = E_data[d_off + j_rem]
    S_A_rem = A_psd[n_off + j_rem];  S_E_rem = E_psd[n_off + j_rem]

    # contribution of the "other" template at each grid (zero where supports do not overlap).
    # For each sample on the add-grid at global index j = s_add + i, the remove sample is
    #   h_remove(j) = A_remove[j - s_rem, bin]   if j in [s_rem, s_rem+M)   else 0.
    s_rem = xp.asarray(start_ind_all_remove, dtype=xp.int64)[None, :]
    s_add = xp.asarray(start_ind_all_add, dtype=xp.int64)[None, :]

    # on the add grid:
    rel_add_to_rem = j_add - s_rem                                    # (M, num_bin)
    mask_add_in_rem = (rel_add_to_rem >= 0) & (rel_add_to_rem < M)
    safe_rel = xp.where(mask_add_in_rem, rel_add_to_rem, 0).astype(xp.int64)
    # gather along axis 0 with per-bin index
    bin_idx = xp.arange(num_bin, dtype=xp.int64)[None, :]
    h_A_rem_on_add = xp.where(mask_add_in_rem, A_remove[safe_rel, bin_idx], 0.0)
    h_E_rem_on_add = xp.where(mask_add_in_rem, E_remove[safe_rel, bin_idx], 0.0)

    # on the remove grid:
    rel_rem_to_add = j_rem - s_add
    mask_rem_in_add = (rel_rem_to_add >= 0) & (rel_rem_to_add < M)
    safe_rel2 = xp.where(mask_rem_in_add, rel_rem_to_add, 0).astype(xp.int64)
    h_A_add_on_rem = xp.where(mask_rem_in_add, A_add[safe_rel2, bin_idx], 0.0)
    h_E_add_on_rem = xp.where(mask_rem_in_add, E_add[safe_rel2, bin_idx], 0.0)

    # post-swap residual r_after = d - h_add + h_remove on each grid
    r_A_on_add = d_A_add - A_add        + h_A_rem_on_add
    r_E_on_add = d_E_add - E_add        + h_E_rem_on_add
    r_A_on_rem = d_A_rem - h_A_add_on_rem + A_remove
    r_E_on_rem = d_E_rem - h_E_add_on_rem + E_remove

    add_term_A = (r_A_on_add.conj()[None] * dA_add).real / S_A_add[None]
    add_term_E = (r_E_on_add.conj()[None] * dE_add).real / S_E_add[None]
    grad_add = +4.0 * df * (add_term_A + add_term_E).sum(axis=1)        # (n_params, num_bin)

    rem_term_A = (r_A_on_rem.conj()[None] * dA_remove).real / S_A_rem[None]
    rem_term_E = (r_E_on_rem.conj()[None] * dE_remove).real / S_E_rem[None]
    grad_remove = -4.0 * df * (rem_term_A + rem_term_E).sum(axis=1)

    return grad_add, grad_remove


# ---------------------------------------------------------------------------
#  Python-API wrappers mirroring GBGPUBase.get_ll / swap_likelihood_difference
# ---------------------------------------------------------------------------

def get_ll_grad(
    params,
    A_template,
    E_template,
    dA_template,
    dE_template,
    data_minus_template,
    psd,
    start_freq_ind,
    df,
    data_index=None,
    noise_index=None,
    data_length=None,
    xp=np,
):
    """Python wrapper for the get_ll-style gradient.

    ``data_minus_template`` here means data minus all active templates
    *except* the one being differentiated (same convention as in
    :func:`GBGPUBase.get_ll`).  ``A_template``, ``E_template`` are the
    current binary's template (so the residual at the current params is
    ``data_minus_template - h_current``).

    Returns
    -------
    grad : (n_params, num_bin) real array.
    """
    num_bin = params.shape[0]
    M = A_template.shape[0]
    if data_index is None:
        data_index = xp.zeros(num_bin, dtype=xp.int32)
    if noise_index is None:
        noise_index = xp.zeros(num_bin, dtype=xp.int32)

    if isinstance(data_minus_template, (list, tuple)):
        A_data, E_data = data_minus_template
    else:
        # 2 channels stacked
        A_data, E_data = data_minus_template[0], data_minus_template[1]

    if isinstance(psd, (list, tuple)):
        A_psd, E_psd = psd
    else:
        A_psd, E_psd = psd[0], psd[1]

    A_data = A_data.ravel()
    E_data = E_data.ravel()
    A_psd = A_psd.ravel()
    E_psd = E_psd.ravel()
    if data_length is None:
        data_length = A_data.shape[0]

    start_ind_all = start_freq_ind + xp.zeros(num_bin, dtype=xp.int64)
    if hasattr(start_freq_ind, "__len__"):
        start_ind_all = xp.asarray(start_freq_ind, dtype=xp.int64)

    return get_ll_grad_kernel_vectorized(
        A_template, E_template, dA_template, dE_template,
        A_data, E_data, A_psd, E_psd, df, start_ind_all,
        data_index, noise_index, data_length, xp=xp,
    )


def swap_ll_grad(
    params_remove,
    params_add,
    A_remove, E_remove, start_ind_all_remove,
    A_add,    E_add,    start_ind_all_add,
    dA_remove, dE_remove,
    dA_add,    dE_add,
    data_minus_template,
    psd,
    df,
    data_index=None,
    noise_index=None,
    data_length=None,
    xp=np,
):
    """Python wrapper for the swap_likelihood_difference-style gradient.

    Returns
    -------
    grad_add, grad_remove : (n_params, num_bin) each, real.

    The convention follows ``GBGPUBase.swap_likelihood_difference``:
    ``data_minus_template`` is the data residual at the *current* state
    (i.e. all templates subtracted including the one at ``params_remove``).
    """
    num_bin = params_add.shape[0]

    if isinstance(data_minus_template, (list, tuple)):
        A_data, E_data = data_minus_template
    else:
        A_data, E_data = data_minus_template[0], data_minus_template[1]

    if isinstance(psd, (list, tuple)):
        A_psd, E_psd = psd
    else:
        A_psd, E_psd = psd[0], psd[1]

    A_data = A_data.ravel(); E_data = E_data.ravel()
    A_psd = A_psd.ravel();   E_psd = E_psd.ravel()
    if data_length is None:
        data_length = A_data.shape[0]
    if data_index is None:
        data_index = xp.zeros(num_bin, dtype=xp.int32)
    if noise_index is None:
        noise_index = xp.zeros(num_bin, dtype=xp.int32)

    return swap_ll_grad_kernel_vectorized(
        A_remove, E_remove, start_ind_all_remove,
        A_add,    E_add,    start_ind_all_add,
        dA_remove, dE_remove, dA_add, dE_add,
        A_data, E_data, A_psd, E_psd, df,
        data_index, noise_index, data_length, xp=xp,
    )
