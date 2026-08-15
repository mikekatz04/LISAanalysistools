"""Inner products, likelihood terms, and information-matrix / covariance diagnostics."""

from __future__ import annotations

import os
import warnings
from typing import Any, List, Optional, Tuple

import matplotlib.pyplot as plt
import numpy as np
from eryn.utils import TransformContainer

try:
    import cupy as cp

except (ModuleNotFoundError, ImportError):
    import numpy as cp

    pass

from . import domains
from .datacontainer import DataResidualArray
from .sensitivity import SensitivityMatrix, SensitivityMatrixBase, get_sensitivity
from .utils.utility import get_array_module

# ``inner_product`` folds ``sig1.conj() * sig2 * invC`` into one reusable buffer
# instead of allocating two temporaries per channel pair. The operations and
# their order are unchanged, so the two spellings are bit-identical -- this
# switch exists so a test can run the plain-expression spelling side by side and
# MEASURE that, rather than take it on trust. Leave it True in production.
_USE_INNER_PRODUCT_BUFFER = True


def inner_product(
    sig1: np.ndarray | list | DataResidualArray,
    sig2: np.ndarray | list | DataResidualArray,
    basis_settings: domains.DomainSettingsBase = None,
    psd: Optional[str | None | np.ndarray | SensitivityMatrix] = "LISASens",
    psd_args: Optional[tuple] = (),
    psd_kwargs: Optional[dict] = {},
    normalize: Optional[bool | str] = False,
    complex: Optional[bool] = False,
) -> float | complex:
    r"""Compute the inner product between two signals weighted by a psd.

    The inner product between time series :math:`a(t)` and :math:`b(t)` is

    .. math::

        \langle a | b \\rangle = 2\int_{f_\\text{min}}^{f_\\text{max}} \\frac{\\tilde{a}(f)^*\\tilde{b}(f) + \\tilde{a}(f)\\tilde{b}(f)^*}{S_n(f)} df\ \ ,

    where :math:`\\tilde{a}(f)` is the Fourier transform of :math:`a(t)` and :math:`S_n(f)` is the one-sided Power Spectral Density of the noise.

    The inner product can be left complex using the ``complex`` kwarg.

    **GPU Capability**: Pass CuPy arrays rather than NumPy arrays.

    Args:
        sig1: First signal to use for the inner product.
            Can be time-domain or frequency-domain.
            Must be 1D ``np.ndarray``, list of 1D ``np.ndarray``s, or 2D ``np.ndarray``
            across channels with shape ``(nchannels, data length)``.
        sig2: Second signal to use for the inner product.
            Can be time-domain or frequency-domain.
            Must be 1D ``np.ndarray``, list of 1D ``np.ndarray``s, or 2D ``np.ndarray``
            across channels with shape ``(nchannels, data length)``.
        basis_settings: :class:`~lisatools.domains.DomainSettingsBase` describing the
            domain of ``sig1`` / ``sig2`` (only consulted when raw arrays are passed
            instead of :class:`~lisatools.datacontainer.DataResidualArray` objects).
        psd: Indicator of what psd to use. If a ``str``, this will be passed as the ``sens_fn`` kwarg to :func:`get_sensitivity`.
            If ``None``, it will be an array of ones. Or, you can pass a 1D ``np.ndarray`` of psd values that must be the same length
            as the frequency domain signals.
        psd_args: Arguments to pass to the psd function if ``type(psd) == str``.
        psd_kwargs: Keyword arguments to pass to the psd function if ``type(psd) == str``.
        normalize: Normalize the inner product. If ``True``, it will normalize the square root of the product of individual signal inner products.
            You can also pass ``"sig1"`` or ``"sig2"`` to normalize with respect to one signal.
        complex: If ``True``, return the complex value of the inner product rather than just its real-valued part.

    Returns:
        Inner product value.

    """
    # initial input checks and setup -- accept DomainBase directly to avoid
    # the DataResidualArray deprecation chatter; only wrap raw arrays.
    def _coerce(sig):
        if isinstance(sig, domains.DomainBase):
            return sig
        if isinstance(sig, DataResidualArray):
            return sig.data_res_arr
        if basis_settings is None:
            raise ValueError(
                "inner_product: raw-array inputs require ``basis_settings``."
            )
        if isinstance(sig, (list, tuple)):
            # A per-channel list/tuple of 1D arrays (a documented input form,
            # produced e.g. by ``info_matrix`` when it stacks the waveform
            # derivatives channel-by-channel). Resolve the array module from
            # the first channel and stack into a ``(nchannels, N)`` array.
            xp = get_array_module(sig[0])
            arr = xp.atleast_2d(xp.asarray(sig))
        else:
            xp = get_array_module(sig)
            arr = xp.atleast_2d(sig)
        return basis_settings.associated_class(arr, basis_settings)

    sig1 = _coerce(sig1)
    sig2 = _coerce(sig2)

    if sig1.nchannels != sig2.nchannels:
        raise ValueError(
            f"Signal 1 has {sig1.nchannels} channels. Signal 2 has {sig2.nchannels} channels. Must be the same."
        )
    
    nchannels = sig1.nchannels

    # Batch support. Either signal may carry a leading source axis; the common
    # case is an unbatched data stream against a batched template stack, which
    # broadcasts to a (nbatch,) vector of inner products. ``psd`` is never
    # batched. Channels live on the axis just above the basis axes, so a batched
    # signal is indexed ``arr[:, i]`` and an unbatched one ``arr[i]`` -- the
    # plain ``sig[i]`` used before silently addressed the SOURCE axis once a
    # batch was present.
    batched = sig1.is_batched or sig2.is_batched

    def _channel(sig, i):
        return sig.arr[:, i] if sig.is_batched else sig.arr[i]

    def _drop_leading_basis(sig, arr, ind_start):
        """Slice ``ind_start`` off the first BASIS axis, whatever the layout."""
        if ind_start == 0:
            return arr
        return arr[:, ind_start:] if sig.is_batched else arr[ind_start:]

    xp = get_array_module(_channel(sig1, 0))

    # checks
    for i in range(sig1.nchannels):
        if not type(_channel(sig1, 0)) == type(_channel(sig1, i)) and type(
            _channel(sig1, 0)
        ) == type(_channel(sig2, i)):
            raise ValueError(
                "Array in sig1, index 0 sets array module. Not all arrays match that module type (Numpy or Cupy)"
            )

    if sig1.data_shape != sig2.data_shape:
        raise ValueError(
            "The two signals are two different lengths. Must be the same length."
        )

    basis = sig1.data_res_arr.settings

    # get psd weighting. ``XYZSensitivityBackend`` and friends are
    # ``SensitivityMatrixBase`` subclasses but not ``SensitivityMatrix``
    # — accept anything in the base hierarchy so callers can pass a
    # backend-built matrix directly.
    if not isinstance(psd, SensitivityMatrixBase):
        psd = SensitivityMatrix(basis, [psd], *psd_args, **psd_kwargs)

    else:
        if psd.basis_settings != basis:
            raise ValueError("PSD basis is not equivalent to signal basis.")
        
        for i in list(psd.channel_shape):
            if i != nchannels:
                raise ValueError("Number of channels in PSD not equal to number of channels in signal.")

    operational_sets = []

    if len(psd.channel_shape) == 2:
        assert psd.shape[0] == psd.shape[1] == sig1.nchannels == sig2.nchannels

        # this avoids 9 inner products for 6 (with symmetry)
        for i in range(psd.shape[0]):
            # for j in range(i, psd.shape[1]):
            #     factor = 1.0 if i == j else 2.0
            #     operational_sets.append(
            #         dict(factor=factor, sig1_ind=i, sig2_ind=j, psd_ind=(i, j))
            #     )
            # TODO: this could be faster?
            for j in range(psd.shape[1]):  # i, psd.shape[1]):
                factor = 1.0  #  if i == j else -1.0  # 2.0
                operational_sets.append(
                    dict(factor=factor, sig1_ind=i, sig2_ind=j, psd_ind=(i, j))
                )

    elif len(psd.channel_shape) == 1:
        assert psd.shape[0] == sig1.nchannels == sig2.nchannels
        for i in range(psd.shape[0]):
            operational_sets.append(dict(factor=1.0, sig1_ind=i, sig2_ind=i, psd_ind=i))

    else:
        raise ValueError("# TODO")

    if complex:
        func = lambda x: x
    else:
        func = xp.real

    # initialize
    out = 0.0
    # x = freqs

    # ``sig1.conj() * sig2 * invC`` allocates TWO full-size temporaries per
    # channel pair, and a cross-spectral PSD makes nine pairs -- eighteen fresh
    # arrays the size of the whole (active) basis box per inner product. On the
    # windowed WDM box that allocation traffic, not the arithmetic, was the cost:
    # folding the same three factors into one reusable buffer measured 2.2x on
    # both the half-year and the two-year MBHB grids. The operations and their
    # order are unchanged, so the result is bit-identical.
    #
    # Only for the array modules whose ufuncs take ``out=``; JAX arrays (this
    # function is differentiated through by ``info_matrix``) keep the plain
    # expression.
    # ``_buf`` holds ``a.conj() * b``; ``_buf_out`` is only allocated when
    # multiplying by ``invC`` actually promotes (mixed precision) -- see the
    # two-stage promotion note below.
    _buf = None
    _buf_out = None
    _can_buffer = _USE_INNER_PRODUCT_BUFFER and getattr(xp, "__name__", "") in (
        "numpy", "cupy"
    )

    tmp = []
    # account for hp and hx if included in time domain signal
    for op_set in operational_sets:
        factor = op_set["factor"]
        temp1 = _channel(sig1, op_set["sig1_ind"])
        temp2 = _channel(sig2, op_set["sig2_ind"])
        inv_psd_tmp = psd.invC[op_set["psd_ind"]]

        if hasattr(sig1.data_res_arr, "apply_frequency_layer_mask") or hasattr(sig2.data_res_arr, "apply_frequency_layer_mask"):
            if hasattr(sig1.data_res_arr, "apply_frequency_layer_mask") and hasattr(sig2.data_res_arr, "apply_frequency_layer_mask"):
                if sig1.data_res_arr.frequency_layer_mask is not None and sig2.data_res_arr.frequency_layer_mask is not None:
                    if not np.array_equal(sig1.data_res_arr.frequency_layer_mask, sig2.data_res_arr.frequency_layer_mask):
                        raise ValueError("If both signals have a frequency layer mask, they must be the same.")
                func_apply = sig1.data_res_arr.apply_frequency_layer_mask
            elif hasattr(sig1.data_res_arr, "apply_frequency_layer_mask") and sig1.data_res_arr.frequency_layer_mask is not None:
                func_apply = sig1.data_res_arr.apply_frequency_layer_mask
            elif hasattr(sig2.data_res_arr, "apply_frequency_layer_mask") and sig2.data_res_arr.frequency_layer_mask is not None:
                func_apply = sig2.data_res_arr.apply_frequency_layer_mask

            temp1 = func_apply(temp1)
            temp2 = func_apply(temp2)
            inv_psd_tmp = func_apply(inv_psd_tmp)  # should be the same for sig1 and sig2 if they have the method
        
        # fix nan in first spot if it is there
        if True:  # inv_psd_tmp.ndim == 1 or :
            ind_start = 1 if np.any(np.isnan(inv_psd_tmp[0])) else 0
            sig_component_1 = _drop_leading_basis(sig1, temp1, ind_start)
            sig_component_2 = _drop_leading_basis(sig2, temp2, ind_start)
            inv_psd_component = inv_psd_tmp[ind_start:]

        # elif inv_psd_tmp.ndim == 2:
        #     ind_start = 1 if np.isnan(inv_psd_tmp[0, 0]) else 0
        #     sig_component_1 = temp1[ind_start:]
        #     sig_component_2 = temp2[ind_start:]
        #     inv_psd_component = inv_psd_tmp[ind_start:]

        else:
            raise ValueError(f"Component PSDs must be 1D or 2D. This has ndim {inv_psd_component.ndim}.")

        if _can_buffer:
            # ``a.conj() * b * c`` is evaluated LEFT TO RIGHT, so it promotes in
            # two steps: the first product lands in ``result_type(a, b)`` and only
            # that intermediate is promoted against ``invC``. Sizing one buffer at
            # the three-way ``result_type(a, b, c)`` silently widens the FIRST
            # multiply -- ``conjugate(a_c64, out=c128_buf)`` upcasts before
            # ``*= b``, so a complex64 signal pair against a float64 invC is then
            # multiplied in complex128 where the plain expression multiplies in
            # complex64. Same value to ~1e-7, different bits. Mirror the pairwise
            # promotion instead: stage 1 at ``result_type(a, b)``, stage 2 at
            # ``result_type(stage1, invC)``.
            _shape_ab = np.broadcast_shapes(
                sig_component_1.shape, sig_component_2.shape
            )
            _dtype_ab = xp.result_type(sig_component_1, sig_component_2)
            _shape_out = np.broadcast_shapes(_shape_ab, inv_psd_component.shape)
            _dtype_out = xp.result_type(_dtype_ab, inv_psd_component)

            if _buf is None or _buf.shape != _shape_ab or _buf.dtype != _dtype_ab:
                _buf = xp.empty(_shape_ab, dtype=_dtype_ab)
            if xp.iscomplexobj(sig_component_1):
                xp.conjugate(sig_component_1, out=_buf)
                _buf *= sig_component_2
            else:
                # ``.conj()`` on a real array is the identity (NumPy returns the
                # array itself), so skipping it changes nothing.
                xp.multiply(sig_component_1, sig_component_2, out=_buf)

            if _dtype_out == _dtype_ab and _shape_out == _shape_ab:
                # Uniform precision (the production path): the second product
                # neither widens nor broadcasts, so it stays in place and one
                # buffer serves the whole loop, exactly as before.
                _buf *= inv_psd_component
                y = func(_buf)
            else:
                # Mixed precision: the promotion is real, so it needs its own
                # destination. Still a bounded, loop-invariant pair of buffers --
                # two arrays for the whole call rather than two per channel pair.
                if (
                    _buf_out is None
                    or _buf_out.shape != _shape_out
                    or _buf_out.dtype != _dtype_out
                ):
                    _buf_out = xp.empty(_shape_out, dtype=_dtype_out)
                xp.multiply(_buf, inv_psd_component, out=_buf_out)
                y = func(_buf_out)
        else:
            y = (
                func(sig_component_1.conj() * sig_component_2 * inv_psd_component)
            )  # assumes right summation rule

        # switching to summation for comp to other domains
        # Batched: reduce the basis (and channel-broadcast) axes only, so the
        # leading source axis survives as one inner product per source.
        _sum_axes = tuple(range(1, y.ndim)) if batched else None
        tmp_out = factor * 4 * xp.sum(y, axis=_sum_axes) * psd.differential_component
        # y = (
        #     func((sig_component_1.conj() * sig_component_2) + (sig_component_2.conj() * sig_component_1)) * inv_psd_component
        # )  # assumes right summation rule
        # # switching to summation for comp to other domains
        # # I CHANGED THE 4 to a 2 and put in the complex components above for CSD issue (# TODO: check this)
        # tmp_out = factor * 2 * xp.sum(y) * psd.differential_component
        tmp.append(tmp_out)
        out += tmp_out

    tmp = xp.asarray(tmp)
    # normalize the inner produce
    normalization_value = 1.0
    if normalize is True:
        norm1 = inner_product(
            sig1,
            sig1,
            psd=psd,
            normalize=False,
        )
        norm2 = inner_product(
            sig2,
            sig2,
            psd=psd,
            normalize=False,
        )

        normalization_value = np.sqrt(norm1 * norm2)

    elif isinstance(normalize, str):
        if normalize == "sig1":
            sig_to_normalize = sig1

        elif normalize == "sig2":
            sig_to_normalize = sig2

        else:
            raise ValueError(
                "If normalizing with respect to sig1 or sig2, normalize kwarg must either be 'sig1' or 'sig2'."
            )

        normalization_value = inner_product(
            sig_to_normalize,
            sig_to_normalize,
            psd=psd,
            normalize=False,
        )

    elif normalize is not False:
        raise ValueError("Normalize must be True, False, 'sig1', or 'sig2'.")

    out /= normalization_value

    # remove from cupy if needed -- but skip the concretization for jax
    # arrays / tracers (calling ``.item()`` inside ``jax.grad`` raises a
    # ``ConcretizationTypeError`` and would drop the trace).
    try:
        import jax
        _is_jax = isinstance(out, (jax.Array, jax.core.Tracer))
    except (ImportError, ModuleNotFoundError):
        _is_jax = False
    # A batched call returns one value per source, so it stays an array on
    # whatever backend it was computed on -- only the scalar (unbatched) case
    # is concretized to a Python/NumPy number as before.
    if not _is_jax and not batched:
        try:
            out = out.item()
        except AttributeError:
            pass

    # add copy function to complex value for compatibility
    if complex and not _is_jax and not batched:
        out = np.complex128(out)

    return out


def residual_source_likelihood_term(
    data_res_arr: DataResidualArray, **kwargs: dict
) -> float | complex:
    """Calculate the source term in the Likelihood for a data residual (d - h).

    The source term in the likelihood is given by,

    .. math::

        \\log{\\mathcal{L}}_\\text{src} = -\\frac{1}{2}\\langle \\vec{d} - \\vec{h} | \\vec{d} - \\vec{h}\\rangle.

    Args:
        data_res_arr: Data residual.
        **kwargs: Keyword arguments to pass to :func:`inner_product`.

    Returns:
        Source term Likelihood value.

    """
    kwargs["normalize"] = False
    ip_val = inner_product(data_res_arr, data_res_arr, **kwargs)
    return -1 / 2.0 * ip_val


def noise_likelihood_term(psd: SensitivityMatrixBase) -> float:
    """Calculate the noise term in the Likelihood.

    The noise term in the likelihood is given by,

    .. math::

        \\log{\\mathcal{L}}_n = -\\kappa \\sum \\log{\\vec{S}_n},

    where :math:`\\kappa` is the domain's
    :attr:`~lisatools.domains.DomainSettingsBase.logdet_factor`: ``1.0`` in the
    one-sided complex (FD / Whittle) convention and ``0.5`` for real-basis
    domains (TD, WDM), where each real coefficient contributes
    :math:`-\\frac{1}{2}\\log{\\det{C}}` to the Gaussian density.

    Args:
        psd: Sensitivity information.

    Returns:
        Noise term Likelihood value.

    """
    # Backend-generic (numpy / cupy): the detC / psd arrays follow the domain's
    # array module, so resolve xp from them rather than hardcoding numpy.
    xp = get_array_module(psd.detC)

    fix = xp.isnan(psd[:]) | xp.isinf(psd[:])

    # assert np.sum(fix) == np.prod(psd.shape[:len(psd.basis_settings.basis_shape)]) or np.sum(fix) == 0, f"sum fix: {np.sum(fix)}; psd shape: {psd.shape}; basis shape: {psd.basis_settings.basis_shape}" #todo fix this
    assert (
        int(xp.sum(fix)) == int(np.prod(psd.shape[:-1])) or int(xp.sum(fix)) == 0
    ), f"sum fix: {int(xp.sum(fix))}; psd shape: {psd.shape}; basis shape: {psd.basis_settings.basis_shape}"
    # TODO: check on this / add warning
    detC = psd.detC
    keep = (detC != 0.0) & (~xp.isinf(detC)) & (~xp.isnan(detC))

    factor = getattr(psd.basis_settings, "logdet_factor", 1.0)
    nl_val = -factor * xp.sum(xp.log(xp.abs(detC[keep])))
    return nl_val


def residual_full_source_and_noise_likelihood(
    data_res_arr: DataResidualArray,
    psd: str | None | np.ndarray | SensitivityMatrixBase,
    **kwargs: dict,
) -> float | complex:
    """Calculate the full Likelihood including noise and source terms.

    The noise term is calculated with :func:`noise_likelihood_term`.

    The source term is calcualted with :func:`residual_source_likelihood_term`.

    Args:
        data_res_arr: Data residual.
        psd: Sensitivity information.
        **kwargs: Keyword arguments to pass to :func:`inner_product`.

    Returns:
       Full Likelihood value.

    """
    if not isinstance(psd, SensitivityMatrixBase):
        # TODO: maybe adjust so it can take a list just like Sensitivity matrix
        basis = data_res_arr.data_res_arr.settings
        psd = SensitivityMatrix(basis, [psd], **kwargs)

    # remove key
    for key in "psd", "psd_args", "psd_kwargs":
        if key in kwargs:
            kwargs.pop(key)

    rslt = residual_source_likelihood_term(data_res_arr, psd=psd, **kwargs)

    nlt = noise_likelihood_term(psd)
    return nlt + rslt


def psd_debug_checks_enabled() -> bool:
    """Whether the batched-noise-likelihood debug validation is switched on.

    Controlled by the ``PSD_DEBUG_CHECKS`` environment variable, default
    ``"0"`` (off). Set ``PSD_DEBUG_CHECKS=1`` to restore the all-or-nothing
    non-finite-covariance validation in
    :func:`batched_residual_full_source_and_noise_likelihoods` (see there for
    why it is off by default). Read per call so it can be flipped at runtime.
    """
    return bool(int(os.environ.get("PSD_DEBUG_CHECKS", "0")))


def batched_residual_full_source_and_noise_likelihoods(
    res_batch: np.ndarray,
    sens_mat_batch: np.ndarray | None,
    invC_batch: np.ndarray,
    detC_batch: np.ndarray,
    settings,
) -> np.ndarray:
    """Per-walker full (source + noise) log-likelihoods for a walker batch.

    Batched twin of :func:`residual_full_source_and_noise_likelihood` for a
    stack of independent walkers sharing one domain: the source term mirrors
    :func:`inner_product`'s per-channel-pair accumulation
    (``out += 4 * sum(real(conj(d_i) d_j invC_ij)) * differential_component``
    over all ``nch x nch`` pairs, then ``-1/2``), and the noise term mirrors
    :func:`noise_likelihood_term` (the all-or-nothing non-finite covariance
    guard -- a debug check, gated behind ``PSD_DEBUG_CHECKS`` and skipped by
    default, see below -- the ``detC`` keep mask, and
    ``-logdet_factor * sum(log|detC|)``) — evaluated for every walker in one
    array operation instead of a per-walker Python loop. Intended for
    already-sanitized matrices (non-finite ``invC`` zeroed, non-finite
    ``detC`` set to 1, as :meth:`SensitivityMatrixBase._setup_det_and_inv`
    guarantees), so the per-pair NaN ``ind_start`` skip in
    :func:`inner_product` is a no-op on both routes.

    Args:
        res_batch: Residuals, shape ``(nw, nch, *basis)``.
        sens_mat_batch: Covariance stack, shape ``(nch, nch, nw, *basis)``
            (channel axes leading, matching :func:`_mat3x3_det_inv`'s
            calling convention with the walker axis folded into
            ``data_shape``). Used **only** by the ``PSD_DEBUG_CHECKS``
            validation; pass ``None`` when the checks are off so the caller
            can release the covariance stack before this call.
        invC_batch: Inverse covariance, shape ``(nch, nch, nw, *basis)``.
        detC_batch: Determinant, shape ``(nw, *basis)``.
        settings: Domain settings providing ``differential_component`` and
            (optionally) ``logdet_factor``.

    Returns:
        ``(nw,)`` per-walker log-likelihoods (same array module as the
        inputs).
    """
    xp = get_array_module(detC_batch)
    nw, nch = res_batch.shape[:2]

    # --- source term: same pair loop / op order as inner_product --------
    diff = settings.differential_component
    src_sum = None
    for i in range(nch):
        for j in range(nch):
            y = xp.real(
                xp.conj(res_batch[:, i]) * res_batch[:, j] * invC_batch[i, j]
            )
            contrib = 4 * y.reshape(nw, -1).sum(axis=-1) * diff
            src_sum = contrib if src_sum is None else src_sum + contrib
    src = -1 / 2.0 * src_sum

    # --- noise term: same guards as noise_likelihood_term ---------------
    # The all-or-nothing non-finite covariance validation is a DEBUG check:
    # two full passes over the (nch, nch, nw, *basis) covariance stack plus a
    # device->host pull of the per-walker counts, i.e. a hard sync, on every
    # scoring call (~1300 per sampler iteration on the stock noise ladder --
    # which also serialises the threaded multi-GPU splits). It validates an
    # invariant the producer (`SensitivityMatrixBase._setup_det_and_inv` /
    # `_mat3x3_det_inv`) already guarantees, so it is off by default; set
    # PSD_DEBUG_CHECKS=1 to restore it.
    if sens_mat_batch is not None and psd_debug_checks_enabled():
        fix = xp.isnan(sens_mat_batch) | xp.isinf(sens_mat_batch)
        walker_axis = 2
        fix_counts = fix.sum(
            axis=tuple(a for a in range(fix.ndim) if a != walker_axis)
        )
        per_mat_shape = sens_mat_batch.shape[:2] + sens_mat_batch.shape[3:]
        allowed = int(np.prod(per_mat_shape[:-1]))
        fix_counts = np.asarray(
            fix_counts.get() if hasattr(fix_counts, "get") else fix_counts
        )
        assert np.all(
            (fix_counts == 0) | (fix_counts == allowed)
        ), (
            f"per-walker non-finite counts {fix_counts.tolist()}; "
            f"matrix shape {per_mat_shape}"
        )

    keep = (detC_batch != 0.0) & (~xp.isinf(detC_batch)) & (~xp.isnan(detC_batch))
    factor = getattr(settings, "logdet_factor", 1.0)
    # masked full-grid sum == the per-walker sum over detC[keep] (dropped
    # pixels contribute log(1) = 0)
    safe = xp.where(keep, xp.abs(detC_batch), 1.0)
    nl = -factor * xp.log(safe).reshape(nw, -1).sum(axis=-1)

    return nl + src


def data_signal_source_likelihood_term(
    data_arr: DataResidualArray, sig_arr: DataResidualArray, **kwargs: dict
) -> float | complex:
    r"""Calculate the source term in the Likelihood for separate signal and data.

    The source term in the likelihood is given by,

    .. math::

        \\log{\\mathcal{L}}_\\text{src} = -\\frac{1}{2}\\left(\\langle \\vec{d} | \\vec{d}\\rangle + \\langle \\vec{h} | \\vec{h}\\rangle - 2\\langle \\vec{d} | \\vec{h}\\rangle \\right)\ \ .

    Args:
        data_arr: Data.
        sig_arr: Signal.
        **kwargs: Keyword arguments to pass to :func:`inner_product`.

    Returns:
        Source term Likelihood value.

    """
    kwargs["normalize"] = False
    d_h = inner_product(data_arr, sig_arr, **kwargs)
    h_h = inner_product(sig_arr, sig_arr, **kwargs)
    d_d = inner_product(data_arr, data_arr, **kwargs)
    return -1 / 2.0 * (d_d + h_h - 2 * d_h)


def data_signal_full_source_and_noise_likelihood(
    data_arr: DataResidualArray,
    sig_arr: DataResidualArray,
    psd: str | None | np.ndarray | SensitivityMatrixBase,
    **kwargs: dict,
) -> float | complex:
    """Calculate the full Likelihood including noise and source terms.

    Here, the signal is treated separate from the data.

    The noise term is calculated with :func:`noise_likelihood_term`.

    The source term is calcualted with :func:`data_signal_source_likelihood_term`.

    Args:
        data_arr: Data.
        sig_arr: Signal.
        psd: Sensitivity information.
        **kwargs: Keyword arguments to pass to :func:`inner_product`.

    Returns:
       Full Likelihood value.

    """
    if not isinstance(psd, SensitivityMatrixBase):
        # TODO: maybe adjust so it can take a list just like Sensitivity matrix
        basis = data_arr.data_res_arr.settings
        psd = SensitivityMatrix(basis, [psd], **kwargs)

    # remove key
    for key in "psd", "psd_args", "psd_kwargs":
        if key in kwargs:
            kwargs.pop(key)

    rslt = data_signal_source_likelihood_term(data_arr, sig_arr, psd=psd, **kwargs)

    nlt = noise_likelihood_term(psd)

    return nlt + rslt


def snr(
    sig1: np.ndarray | list | DataResidualArray,
    *args: Any,
    data: Optional[np.ndarray | list | DataResidualArray] = None,
    **kwargs: Any,
) -> float:
    """Compute the snr between two signals weighted by a psd.

    The signal-to-noise ratio of a signal is :math:`\\sqrt{\\langle a|a\\rangle}`.

    This will be the optimal SNR if ``data==None``. If a data array is given, it will be the observed
    SNR: :math:`\\langle d|a\\rangle/\\sqrt{\\langle a|a\\rangle}`.

    **GPU Capability**: Pass CuPy arrays rather than NumPy arrays.

    Args:
        sig1: Signal to use as the templatefor the SNR.
            Can be time-domain or frequency-domain.
            Must be 1D ``np.ndarray``, list of 1D ``np.ndarray``s, or 2D ``np.ndarray``
            across channels with shape ``(nchannels, data length)``.
        *args: Arguments to pass to :func:`inner_product`.
        data: Data becomes the ``sig2`` argument to :func:`inner_product`.
        **kwargs: Keyword arguments to pass to :func:`inner_product`.

    Returns:
        Optimal or detected SNR value (depending on ``data`` kwarg).

    """

    # get optimal SNR
    opt_snr = np.sqrt(inner_product(sig1, sig1, *args, **kwargs).real)
    if data is None:
        return opt_snr

    else:
        # if inputed data, calculate detected SNR
        det_snr = inner_product(sig1, data, *args, **kwargs).real / opt_snr
        return det_snr


def h_var_p_eps(
    step: float,
    waveform_model: callable,
    params: np.ndarray | list,
    index: int,
    parameter_transforms: Optional[TransformContainer] = None,
    waveform_args: Optional[tuple] = (),
    waveform_kwargs: Optional[dict] = {},
) -> np.ndarray:  # TODO: check this
    """Calculate the waveform with a perturbation step of the variable V[i].

    Args:
        waveform_model: Callable function to the waveform generator with signature ``(*params, **waveform_kwargs)``.
        params: Source parameters that are over derivatives (not in fill dict of parameter transforms)
        step: Absolute step size for variable of interest.
        index: Index to parameter of interest.
        parameter_transforms: `TransformContainer <https://lisa-analysis-tools.github.io/Eryn/user/utils.html#eryn.utils.TransformContainer>`_ object to transform from the derivative parameter basis
            to the waveform parameter basis. This class can also fill in fixed parameters where the derivatives are not being taken.
        waveform_args: args (beyond parameters) for the waveform generator.
        waveform_kwargs: kwargs for the waveform generation.

    Returns:
        Perturbation to the waveform in the given parameter. Will always be 2D array with shape ``(num channels, data length)``

    """
    params_p_eps = params.copy()
    params_p_eps[index] += step

    if parameter_transforms is not None:
        # transform
        params_p_eps = parameter_transforms.transform_base_parameters(params_p_eps[None, :])[0]

    args_in = tuple(params_p_eps) + tuple(waveform_args)
    dh = waveform_model(*args_in, **waveform_kwargs)

    # adjust output based on waveform model output
    # needs to be 2D array
    if (isinstance(dh, np.ndarray) or isinstance(dh, cp.ndarray)) and dh.ndim == 1:
        xp = get_array_module(dh)
        dh = xp.atleast_2d(dh)

    elif isinstance(dh, list):
        xp = get_array_module(dh[0])
        dh = xp.asarray(dh)

    return dh


def dh_dlambda(
    eps: float,
    *args: tuple,
    more_accurate: Optional[bool] = True,
    **kwargs: dict,
) -> np.ndarray:
    """Derivative of the waveform.

    Calculate the derivative of the waveform with precision of order (step^4)
    with respect to the variable V in the i direction.

    Args:
        eps: Absolute **derivative** step size for variable of interest.
        *args: Arguments passed to :func:`h_var_p_eps`.
        more_accurate: If ``True``, run a more accurate derivate requiring 2x more waveform generations.
        **kwargs: Keyword arguments passed to :func:`h_var_p_eps`.

    Returns:
        Numerical derivative of the waveform with respect to a varibale of interest. Will be 2D array
        with shape ``(num channels, data length)``.

    """
    if more_accurate:
        # Derivative of the Waveform
        # up
        h_I_up_2eps = h_var_p_eps(2 * eps, *args, **kwargs)
        h_I_up_eps = h_var_p_eps(eps, *args, **kwargs)
        # down
        h_I_down_2eps = h_var_p_eps(-2 * eps, *args, **kwargs)
        h_I_down_eps = h_var_p_eps(-eps, *args, **kwargs)

        # make sure they are all the same length
        ind_max = np.min(
            [
                h_I_up_2eps.shape[1],
                h_I_up_eps.shape[1],
                h_I_down_2eps.shape[1],
                h_I_down_eps.shape[1],
            ]
        )

        # error scales as eps^4
        dh_I = (
            -h_I_up_2eps[:, :ind_max]
            + h_I_down_2eps[:, :ind_max]
            + 8 * (h_I_up_eps[:, :ind_max] - h_I_down_eps[:, :ind_max])
        ) / (12 * eps)

    else:
        # Derivative of the Waveform
        # up
        h_I_up_eps = h_var_p_eps(eps, *args, **kwargs)
        # down
        h_I_down_eps = h_var_p_eps(-eps, *args, **kwargs)

        ind_max = np.min([h_I_up_eps.shape[1], h_I_down_eps.shape[1]])

        # TODO: check what error scales as.
        dh_I = (h_I_up_eps[:, :ind_max] - h_I_down_eps[:, :ind_max]) / (2 * eps)
        # Time thta it takes for one variable: approx 5 minutes

    return dh_I


def info_matrix(
    eps: float | np.ndarray,
    waveform_model: callable,
    params: np.ndarray | list,
    deriv_inds: Optional[np.ndarray | list] = None,
    inner_product_kwargs: Optional[dict] = {},
    return_derivs: Optional[bool] = False,
    **kwargs: dict,
) -> np.ndarray | Tuple[np.ndarray, list]:
    """Calculate Information Matrix.

    This calculates the information matrix for a given waveform model at a given set of parameters.
    The inverse of the information matrix gives the covariance matrix.

    This is also referred to as the Fisher information matrix, but @MichaelKatz has chosen to leave out the name because of `this <https://www.newstatesman.com/long-reads/2020/07/ra-fisher-and-science-hatred>`_. That is why this codebase says "information matrix" throughout.

    The info matrix is given by:

    .. math::

        M_{ij} = \\langle h_i | h_j \\rangle \\text{ with } h_i = \\frac{\\partial h}{\\partial \\lambda_i}.

    Args:
        eps: Absolute **derivative** step size for variable of interest. Can be provided as a ``float`` value that applies
            to all variables or an array, one for each parameter being evaluated in the information matrix.
        waveform_model: Callable function to the waveform generator with signature ``(*params, **waveform_kwargs)``.
        params: Source parameters.
        deriv_inds: Subset of parameters of interest for which to calculate the information matrix, by index.
            If ``None``, it will be ``np.arange(len(params))``.
        inner_product_kwargs: Keyword arguments for the inner product function.
        return_derivs: If ``True``, also returns computed numerical derivatives.
        **kwargs: Keyword arguments passed to :func:`dh_dlambda`.

    Returns:
        If ``not return_derivs``, this will be the information matrix as a numpy array. If ``return_derivs is True``,
        it will be a tuple with the first entry as the information matrix and the second entry as the partial derivatives.

    """

    # setup initial information
    num_params = len(params)

    if deriv_inds is None:
        deriv_inds = np.arange(num_params)

    num_info_params = len(deriv_inds)

    if isinstance(eps, float):
        eps = np.full_like(params, eps)

    # collect derivatives over the variables of interest
    dh = []
    for i, eps_i in zip(deriv_inds, eps):
        # derivative up
        temp = dh_dlambda(eps_i, waveform_model, params, i, **kwargs)
        dh.append(temp)

    # calculate the components of the symmetric matrix
    info = np.zeros((num_info_params, num_info_params))
    for i in range(num_info_params):
        for j in range(i, num_info_params):
            info[i][j] = inner_product(
                [dh[i][k] for k in range(len(dh[i]))],
                [dh[j][k] for k in range(len(dh[i]))],
                **inner_product_kwargs,
            )
            info[j][i] = info[i][j]

    if return_derivs:
        return info, dh
    else:
        return info


def covariance(
    *info_mat_args: tuple,
    info_mat: Optional[np.ndarray] = None,
    diagonalize: Optional[bool] = False,
    return_info_mat: Optional[bool] = False,
    precision: Optional[bool] = False,
    **info_mat_kwargs: dict,
) -> np.ndarray | list:
    """Calculate covariance matrix for a set of EMRI parameters, computing the information matrix if not supplied.

    Args:
        *info_mat_args: Set of arguments to pass to :func:`info_matrix`. Not required if ``info_mat`` is not ``None``.
        info_mat: Pre-computed information matrix. If supplied, this matrix will be inverted.
        diagonalize: If ``True``, diagonalizes the covariance matrix.
        return_info_mat: If ``True``, also returns the computed information matrix.
        precision: If ``True``, uses 500-dps precision to compute the information matrix inverse (requires `mpmath <https://mpmath.org>`_). This is typically a good idea as the information matrix can be highly ill-conditioned.
        **info_mat_kwargs: Keyword arguments to pass to :func:`info_matrix`.

    Returns:
        Covariance matrix. If ``return_info_mat is True``. A list will be returned with the covariance as the first
        entry and the information matrix as the second entry. If ``return_derivs is True`` (keyword argument to :func:`info_matrix`),
        then another entry will be added to the list for the derivatives.

    """

    if info_mat is None:
        info_mat = info_matrix(*info_mat_args, **info_mat_kwargs)

    # parse output properly
    if "return_derivs" in info_mat_kwargs and info_mat_kwargs["return_derivs"]:
        return_derivs = True
        info_mat, dh = info_mat

    else:
        return_derivs = False

    # attempt to import and setup mpmath if precision required
    if precision:
        try:
            import mpmath as mp

            mp.mp.dps = 500
        except ModuleNotFoundError:
            print("mpmath module not installed. Defaulting to low precision...")
            precision = False

    if precision:
        hp_info_mat = mp.matrix(info_mat.tolist())
        U, S, V = mp.svd_r(hp_info_mat)  # singular value decomposition
        temp = mp.diag([val ** (-1) for val in S])  # get S**-1
        temp2 = V.T * temp * U.T  # construct pseudo-inverse
        cov = np.array(temp2.tolist(), dtype=np.float64)
    else:
        cov = np.linalg.pinv(info_mat)

    if diagonalize:
        # get eigeninformation
        eig_vals, eig_vecs = get_eigeninfo(cov, high_precision=precision)

        # diagonal cov now
        cov = np.dot(np.dot(eig_vecs.T, cov), eig_vecs)

    # just requesting covariance
    if True not in [return_info_mat, return_derivs]:
        return cov

    # if desiring more information, create a list capable of variable size
    returns = [cov]

    # add information matrix
    if return_info_mat:
        returns.append(info_mat)

    # add derivatives
    if return_derivs:
        returns.append(dh)

    return returns


def plot_covariance_corner(
    params: np.ndarray,
    cov: np.ndarray,
    nsamp: Optional[int] = 25000,
    fig: Optional[plt.Figure] = None,
    **kwargs: dict,
) -> plt.Figure:
    """Construct a corner plot for a given covariance matrix.

    The `corner <https://corner.readthedocs.io/en/latest/>`_ module is required for this.

    Args:
        params: The set of parameters used for the event (the mean vector of the covariance matrix).
        cov: Covariance matrix from which to construct the corner plot.
        nsamp: Number of samples to draw from the multivariate distribution.
        fig: Matplotlib :class:`plt.Figure` object. Use this if passing an existing corner plot figure.
        **kwargs: Keyword arguments for the corner plot - see the module documentation for more info.

    Returns:
        The corner plot figure.

    """

    # TODO: add capability for ChainConsumer?
    try:
        import corner
    except ModuleNotFoundError:
        raise ValueError("Attempting to plot using the corner module, but it is not installed.")

    # generate fake samples from the covariance distribution
    samp = np.random.multivariate_normal(params, cov, size=nsamp)

    # make corner plot
    fig = corner.corner(samp, **kwargs)
    return fig


def plot_covariance_contour(
    params: np.ndarray,
    cov: np.ndarray,
    horizontal_index: int,
    vertical_index: int,
    nsamp: Optional[int] = 25000,
    ax: Optional[plt.Axes] = None,
    **kwargs: dict,
) -> plt.Axes | Tuple[plt.Figure, plt.Axes]:
    """Construct a contour plot for a given covariance matrix on a single axis object.

    The `corner <https://corner.readthedocs.io/en/latest/>`_ module is required for this.

    Args:
        params: The set of parameters used for the event (the mean vector of the covariance matrix).
        cov: Covariance matrix from which to construct the corner plot.
        horizontal_index: Parameter index to plot along the horizontal axis of the contour plot.
        vertical_index: Parameter index to plot along the vertical axis of the contour plot.
        nsamp: Number of samples to draw from the multivariate distribution.
        fig: Matplotlib :class:`plt.Figure` object. Use this if passing an existing corner plot figure.
        **kwargs: Keyword arguments for the corner plot - see the module documentation for more info.

    Returns:
        If ``ax`` is provided, the return will be that ax object. If it is not provided, then a
        Matplotlib Figure and Axes obejct is created and returned as a tuple: ``(plt.Figure, plt.Axes)``.

    """

    # TODO: add capability for ChainConsumer?
    try:
        import corner
    except ModuleNotFoundError:
        raise ModuleNotFoundError(
            "Attempting to plot using the corner module, but it is not installed."
        )

    if ax is None:
        fig, ax = plt.subplots(1, 1)
    else:
        fig = None

    # generate fake samples from the covariance distribution
    samp = np.random.multivariate_normal(params, cov, size=nsamp)

    x = samp[:, horizontal_index]
    y = samp[:, vertical_index]

    # make corner plot
    corner.hist2d(x, y, ax=ax, **kwargs)

    if fig is None:
        return ax
    else:
        return (fig, ax)


def get_eigeninfo(
    arr: np.ndarray, high_precision: Optional[bool] = False
) -> Tuple[np.ndarray, np.ndarray]:
    """Performs eigenvalue decomposition and returns the eigenvalues and right-eigenvectors for the supplied information/covariance matrix.

    Args:
        arr: Input matrix for which to perform eigenvalue decomposition.
        high_precision: If ``True``, use 500-dps precision to ensure accurate eigenvalue decomposition
            (requires `mpmath <https://mpmath.org>`_ to be installed). Defaults to False.

    Returns:
        Tuple containing Eigenvalues and right-Eigenvectors for the supplied array, constructed such that evects[:,k] corresponds to evals[k].

    """

    if high_precision:
        try:
            import mpmath as mp

            mp.mp.dps = 500
        except ModuleNotFoundError:
            print("mpmath is not installed - using low-precision eigen decomposition.")
            high_precision = False

    if high_precision:
        hp_arr = mp.matrix(arr.tolist())
        # get eigenvectors
        E, EL, ER = mp.eig(hp_arr, left=True, right=True)

        # convert back
        evals = np.array(E, dtype=np.float64)
        evects = np.array(ER.tolist(), dtype=np.float64)

        return evals, evects

    else:
        evals, evects = np.linalg.eig(arr)
        return evals, evects


def cutler_vallisneri_bias(
    waveform_model_true: callable,
    waveform_model_approx: callable,
    params: np.ndarray,
    eps: float | np.ndarray,
    input_diagnostics: Optional[dict] = None,
    info_mat: Optional[np.ndarray] = None,
    deriv_inds: Optional[np.ndarray | list] = None,
    return_derivs: Optional[bool] = False,
    return_cov: Optional[bool] = False,
    parameter_transforms: Optional[TransformContainer] = None,
    waveform_true_args: Optional[tuple] = (),
    waveform_true_kwargs: Optional[dict] = {},
    waveform_approx_args: Optional[tuple] = (),
    waveform_approx_kwargs: Optional[dict] = {},
    inner_product_kwargs: Optional[dict] = {},
) -> list:
    """Calculate the Cutler-Vallisneri bias.

    # TODO: add basic math

    Args:
        waveform_model_true: Callable function to the **true** waveform generator with signature ``(*params, **waveform_kwargs)``.
        waveform_model_approx: Callable function to the **approximate** waveform generator with signature ``(*params, **waveform_kwargs)``.
        params: Source parameters.
        eps: Absolute **derivative** step size. See :func:`info_matrix`.
        input_diagnostics: Dictionary including the diagnostic information if it is precomputed. Dictionary must include
            keys ``"cov"`` (covariance matrix, output of :func:`covariance`), ``"h_true"`` (the **true** waveform),
            and ``"dh"`` (derivatives of the waveforms, list of outputs from :func:`dh_dlambda`).
        info_mat: Pre-computed information matrix. If supplied, this matrix will be inverted to find the covariance.
        deriv_inds: Subset of parameters of interest. See :func:`info_matrix`.
        return_derivs: If ``True``, also returns computed numerical derivatives.
        return_cov: If ``True``, also returns computed covariance matrix.
        parameter_transforms: `TransformContainer <https://lisa-analysis-tools.github.io/Eryn/user/utils.html#eryn.utils.TransformContainer>`_ object. See :func:`info_matrix`.
        waveform_true_args: Arguments for the **true** waveform generator.
        waveform_true_kwargs: Keyword arguments for the **true** waveform generator.
        waveform_approx_args: Arguments for the **approximate** waveform generator.
        waveform_approx_kwargs: Keyword arguments for the **approximate** waveform generator.
        inner_product_kwargs: Keyword arguments for the inner product function.

    Returns:
        List of return information. By default, it is ``[systematic error, bias]``.
        If ``return_derivs`` or ``return_cov`` are ``True``, they will be added to the list with derivs added before covs.

    """

    if deriv_inds is None:
        deriv_inds = np.arange(len(params))

    if info_mat is not None and input_diagnostics is not None:
        warnings.warn("Provided info_mat and input_diagnostics kwargs. Ignoring info_mat.")

    # adjust parameters to waveform basis
    params_in = parameter_transforms.transform_base_parameters(params.copy())

    if input_diagnostics is None:
        # get true waveform
        h_true = waveform_model_true(
            *(tuple(params_in) + tuple(waveform_true_args)), **waveform_true_kwargs
        )

        # get covariance info and waveform derivatives
        cov, dh = covariance(
            eps,
            waveform_model_true,
            params,
            return_derivs=True,
            deriv_inds=deriv_inds,
            info_mat=info_mat,
            parameter_transforms=parameter_transforms,
            waveform_args=waveform_true_args,
            waveform_kwargs=waveform_true_kwargs,
            inner_product_kwargs=inner_product_kwargs,
        )

    else:
        # pre-computed info
        cov = input_diagnostics["cov"]
        h_true = input_diagnostics["h_true"]
        dh = input_diagnostics["dh"]

    # get approximate waveform
    h_approx = waveform_model_approx(
        *(tuple(params_in) + tuple(waveform_approx_args)), **waveform_approx_kwargs
    )

    # adjust/check waveform outputs
    if isinstance(h_true, np.ndarray) and h_true.ndim == 1:
        h_true = [h_true]
    elif isinstance(h_true, np.ndarray) and h_true.ndim == 2:
        h_true = list(h_true)

    if isinstance(h_approx, np.ndarray) and h_approx.ndim == 1:
        h_approx = [h_approx]
    elif isinstance(h_approx, np.ndarray) and h_approx.ndim == 2:
        h_approx = list(h_approx)

    assert len(h_approx) == len(h_true)
    assert np.all(np.asarray([len(h_approx[i]) == len(h_true[i]) for i in range(len(h_true))]))

    # difference in the waveforms
    diff = [h_true[i] - h_approx[i] for i in range(len(h_approx))]

    # systematic err
    syst_vec = np.array(
        [
            inner_product(
                dh[k],
                diff,
                **inner_product_kwargs,
            )
            for k in range(len(deriv_inds))
        ]
    )

    bias = np.dot(cov, syst_vec)

    # return list
    returns = [syst_vec, bias]

    # add anything requested
    if return_cov:
        returns.append(cov)

    if return_derivs:
        returns.append(dh)

    return returns


def scale_to_snr(
    target_snr: float,
    sig: np.ndarray | list,
    *snr_args: tuple,
    return_orig_snr: Optional[bool] = False,
    **snr_kwargs: dict,
) -> np.ndarray | list | Tuple[np.ndarray | list, float]:
    """Calculate the SNR and scale a signal.

    Args:
        target_snr: Desired SNR value for the injected signal.
        sig: Signal to adjust. A copy will be made.
            Can be time-domain or frequency-domain.
            Must be 1D ``np.ndarray``, list of 1D ``np.ndarray``s, or 2D ``np.ndarray``
            across channels with shape ``(nchannels, data length)``.
        *snr_args: Arguments to pass to :func:`snr`.
        return_orig_snr: If ``True``, return the original SNR in addition to the adjusted data.
        **snr_kwargs: Keyword arguments to pass to :func:`snr`.

    Returns:
        Returns the copied input signal adjusted to the target SNR. If ``return_orig_snr is True``, the original
        SNR is added as the second entry of a tuple with the adjusted signal (as the first entry in the tuple).

    """
    # get the snr and adjustment factor
    snr_out = snr(sig, *snr_args, **snr_kwargs)
    factor = target_snr / snr_out

    # any changes back to the original signal type
    back_single = False
    back_2d_array = False

    if isinstance(sig, list) is False and sig.ndim == 1:
        sig = [sig]
        back_single = True

    elif isinstance(sig, list) is False and sig.ndim == 2:
        sig = list(sig)
        back_2d_array = True

    # adjust
    out = [sig_i * factor for sig_i in sig]

    # adjust type back to the input type
    if back_2d_array:
        out = np.asarray(out)

    if back_single:
        out = out[0]

    if return_orig_snr:
        return (out, snr_out)

    return out
