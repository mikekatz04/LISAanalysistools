"""General-purpose array, window, noise, and indexing helpers shared across the package."""

from __future__ import annotations

import typing
from typing import Tuple

import numpy as np

try:
    import cupy as cp

except (ModuleNotFoundError, ImportError):
    import numpy as cp

    pass


# Generic window function
def windowfun(winType, N, alpha=0.01, xp=None):
    """Generate a window function of length ``N``.

    The ``alpha`` parameter denotes the additional smoothness parameter
    necessary for some window types (e.g. Tukey, Planck, Kaiser).

    Available window types: ``nuttall``, ``nuttall3``, ``nuttall3a``,
    ``nuttall3b``, ``nuttall4``, ``nuttall4a``, ``nuttall4b``, ``nuttall4c``,
    ``blackman-harris``, ``tukey``, ``planck``, ``welch``, ``rectangular``,
    ``hamming``, ``hanning``, ``kaiser``.

    For more details about the window types see
    http://hdl.handle.net/11858/00-001M-0000-0013-557A-5.

    Args:
        winType: Name of the desired window (case-insensitive).
        N: Length of the window in samples.
        alpha: Smoothness parameter used by Tukey, Planck, and Kaiser windows.
        xp: Array module to allocate the window with (``numpy`` or ``cupy``).
            Defaults to ``numpy`` when ``None``.

    Returns:
        Tuple ``(winvals, winvals_norm)`` where ``winvals`` are the raw window
        values and ``winvals_norm`` is ``winvals`` divided by
        :math:`\\sqrt{\\sum w^2}` so that the squared sum of the normalized
        window is unity.

    Raises:
        NotImplementedError: If ``winType`` is not one of the supported window
            names.
    """
    if xp is None:
        xp = np

    # Define the more common z
    z       = (xp.arange(0,N,1)/N) * 2.0 * xp.pi
    N       = int(N)
    winType = winType.lower() # Be a little more robust on the inputs

    # Tukey window
    def tukey(N, alpha):
        # alpha -- parameter the defines the shape of the window
        if alpha <= 0.0:
            # alpha = 0 is the rectangular limit; the general expression below
            # divides by alpha, so return the rectangular window directly.
            return xp.ones(N, dtype=float)
        w         = xp.zeros(N, dtype=float)
        i         = xp.arange(0,N,1)
        r         = (2.0*i)/(alpha*(N-1))
        l1        = int(xp.rint((alpha*(N-1))/2.0))
        l2        = int(xp.rint((N-1)*(1.0-alpha/2.0)))
        w[0:l1]   = 0.5*(1.0 + xp.cos(xp.pi*(r[0:l1] - 1.0)))
        w[l1:l2]  = 1.0
        w[l2:N-1] = 0.5*(1+xp.cos(xp.pi*(r[l2:N-1] - 2.0/alpha + 1.0)))
        return w

    # planck window
    def planck(N, epsilon):
        # alpha -- parameter the defines the shape of the window
        w         = xp.zeros(N, dtype=float)
        i         = xp.arange(0,N,1)
        Z_plus    = 2.*epsilon*( 1./(1. + ((2.*i)/(N-1) - 1)) + 1./(1. - 2*epsilon + ((2.*i)/(N-1) - 1)) )
        Z_minus   = 2.*epsilon*( 1./(1. - ((2.*i)/(N-1) - 1)) + 1./(1. - 2*epsilon - ((2.*i)/(N-1) - 1)) )
        l1        = int(xp.rint( epsilon * (N-1) ))
        l2        = int(xp.rint( (1-epsilon) * (N-1) ))
        w[0:l1]   = 1./(xp.exp(Z_plus[0:l1]) + 1)
        w[l1:l2]  = 1.0
        w[l2:N-1] = 1./(xp.exp(Z_minus[l2:N-1]) + 1)
        return w

    # kaiser
    def kaiser(N, alpha):
        w  = sg.kaiser(N, beta=alpha)
        return w

    # nuttall
    def nuttall(N, z, type = 'nuttal4'):

        if type == 'nuttall':
            w = sg.get_window('nuttall', N)
        if type == 'nuttall3':
            w = 0.375 - 0.5 * xp.cos (z) + 0.125 * xp.cos (2 * z)
        if type == 'nuttall3a':
            w = 0.40897 - 0.5 * xp.cos (z) + 0.09103 * xp.cos (2 * z)
        elif type == 'nuttall3b':
            w = 0.4243801 - 0.4973406 * xp.cos (z) + 0.0782793 * xp.cos (2 * z)
        elif type == 'nuttall4':
            w = 0.3125 - 0.46875 * xp.cos(z) + 0.1875 * xp.cos (2 * z) - 0.03125 * xp.cos (3 * z)
        elif type == 'nuttall4a':
            w = 0.338946 - 0.481973 * xp.cos (z) + 0.161054 * xp.cos (2 * z) - 0.018027 * xp.cos (3 * z)
        elif type == 'nuttall4b':
            w = 0.355768 - 0.487396 * xp.cos (z) + 0.144232 * xp.cos (2 * z) - 0.012604 * xp.cos (3 * z)
        elif type == 'nuttall4c':
            w = 0.3635819 - 0.4891775 * xp.cos (z) + 0.1365995 * xp.cos (2 * z) - 0.0106411 * xp.cos (3 * z)
        return w

    # welch
    def welch(z):
        w = 1 - (2 * z - 1)**2
        return w

    # Blackman-Harris window
    def bh(z):
        w = 0.35875 - 0.48829 * xp.cos (z) + 0.14128 * xp.cos (2 * z) - 0.01168 * xp.cos (3 * z)
        return w

    # Rectangular window
    def rectangular(N):
        w = xp.ones(N)
        return w

    # Hamming window
    def hamm(N):
        w = xp.hamming(N)
        return w

    # Hanning window
    def hann(N):
        w = xp.hanning(N)
        return w

    # Choose the window
    if winType == 'blackman-harris' or winType == 'bh92':
        winvals = bh(z)
    elif winType == 'tukey':
        winvals = tukey(N,alpha)
    elif winType == 'planck':
        winvals = planck(N, alpha)
    elif winType == 'hamming':
        winvals = hamm(N)
    elif winType == 'hanning':
        winvals = hann(N)
    elif winType == 'welch':
        winvals = welch(z)
    elif winType == 'nuttall':
        winvals = nuttall(N,z,type = 'nuttall')
    elif winType == 'nuttall3':
        winvals = nuttall(N,z,type = 'nuttall3')
    elif winType == 'nuttall3a':
        winvals = nuttall(N,z,type = 'nuttall3a')
    elif winType == 'nuttall3b':
        winvals = nuttall(N,z,type = 'nuttall3b')
    elif winType == 'nuttall4':
        winvals = nuttall(N,z, type = 'nuttall4')
    elif winType == 'nuttall4a':
        winvals = nuttall(N,z, type = 'nuttall4a')
    elif winType == 'nuttall4b':
        winvals = nuttall(N,z, type = 'nuttall4b')
    elif winType == 'nuttall4c':
        winvals = nuttall(N,z, type = 'nuttall4c')
    elif winType == 'rectangular':
        winvals = rectangular(N)
    elif winType == 'kaiser':
        winvals = kaiser(N, alpha=alpha)
    else:
        raise NotImplementedError('Unknown window type ')

    K            = xp.sum(winvals*winvals)
    winvals_norm = winvals/(xp.sqrt(K)) # Normalise the window

    return winvals, winvals_norm

def tukey(N, alpha, xp=None):
    """Build a Tukey (cosine-tapered) window of length ``N``.

    Args:
        N: Length of the window in samples.
        alpha: Fraction of the window taper, in ``(0, 1)``. ``alpha`` near 0
            approaches a rectangular window; ``alpha = 1`` approaches a Hann
            window.
        xp: Array module to allocate the window with (``numpy`` or ``cupy``).
            Defaults to ``numpy`` when ``None``.

    Returns:
        Window values as a 1D array of length ``N``.
    """
    if xp is None:
        xp = np

    assert 0.0 < alpha < 1.0
    n = xp.arange(int(xp.ceil(N / 2.0)))
    in_window_edge = n < alpha * N / 2

    tmp = (in_window_edge) * 1.0 / 2.0 * (1.0 - xp.cos(2.0 * np.pi * n / (alpha * N))) + (
        ~in_window_edge
    ) * 1.0
    tukey_out = xp.zeros(N, dtype=float)
    # TODO: check this works for odd functions
    tukey_out[: tmp.shape[0]] = tmp
    tukey_out[-tmp.shape[0] :] = tmp[::-1]

    return tukey_out


def detrend(t, y):
    """Remove a linear trend and the residual mean from a time series.

    Fits ``y`` against ``t`` with a degree-1 polynomial, subtracts the fitted
    line, then subtracts the residual mean.

    Args:
        t: 1D array of sample times.
        y: 1D array of values matching ``t``.

    Returns:
        Detrended, mean-removed copy of ``y``.
    """
    # @Nikos data setup
    m, b = np.polyfit(t, y, 1)
    ytmp = y - (m * t + b)
    ydetrend = ytmp - np.mean(ytmp)
    return ydetrend


def get_array_module(arr: np.ndarray | cp.ndarray) -> object:
    """Return array library of an array (np/cp/jnp).

    Args:
        arr: Numpy, Cupy, or JAX array (including ``jax.Array`` /
            ``jax.core.Tracer`` inside a ``jax.jit`` / ``jax.grad`` trace).

    """
    if isinstance(arr, np.ndarray):
        return np
    if isinstance(arr, cp.ndarray):
        return cp
    # JAX arrays / tracers — gated so importing this module never requires jax.
    try:
        import jax
        import jax.numpy as jnp
    except (ImportError, ModuleNotFoundError):
        jax = None
    if jax is not None and isinstance(arr, (jax.Array, jax.core.Tracer)):
        return jnp
    raise ValueError("arr must be a numpy, cupy, or jax array.")


def asnumpy(arr) -> np.ndarray:
    """Return ``arr`` as a numpy array, copying off-device when needed.

    Mirrors ``cupy.asnumpy`` but is safe to call on inputs that are
    already numpy (or any array-like): cupy arrays are pulled to the
    host via ``.get()``; numpy / array-like inputs go through
    ``np.asarray`` without a copy when possible.
    """
    if isinstance(arr, np.ndarray):
        return arr
    if hasattr(arr, "get"):  # cupy.ndarray, multi-GPU views, etc.
        return arr.get()
    return np.asarray(arr)


def generate_noise_fd(
    N: int,
    df: float,
    *sensitivity_args: typing.Any,
    func: typing.Optional[typing.Callable] = None,
    **sensitivity_kwargs: typing.Any,
) -> np.ndarray[float]:
    """Generate noise directly in the Frequency Domain.

    Args:
        N: Number of points in frequency domain waveform assuming frequencies greater than
            or equal to zero.
        df: Frequency domain bin size (``1 / (N * dt)``).
        sensitivity_args: Arguments for ``func``.
        func: Function for generating the sensitivity curve as a function of frequency. (default: :func:`lisatools.sensitivity.get_sensivity`)
        sensitivity_kwargs: Keyword arguments for ``func``.

    Returns:
        An instance of generated noise in the frequency domain.


    """
    if not isinstance(N, int):
        raise ValueError(f"N must be an integer. See documentation for more information.")

    if not isinstance(df, float):
        raise ValueError(f"N must be an integer. See documentation for more information.")

    if func is None:
        # TODO: make this better
        from .sensitivity import get_sensitivity

        func = get_sensitivity

    freqs = np.arange(N) * df

    norm = 0.5 * (1.0 / df) ** 0.5
    psd = func(freqs, *sensitivity_args, **sensitivity_kwargs)
    noise_realization = psd ** (1 / 2) * (
        np.random.normal(0, norm, len(freqs)) + 1j * np.random.normal(0, norm, len(freqs))
    )
    return noise_realization


def AET(
    X: float | np.ndarray, Y: float | np.ndarray, Z: float | np.ndarray
) -> Tuple[float | np.ndarray, float | np.ndarray, float | np.ndarray]:
    """Transform to AET from XYZ

    .. math::

        A = (Z - X) / \\sqrt(2)

    .. math::

        E = (X - 2Y + Z) / \\sqrt(6)

    .. math::

        T = (X + Y + Z) / \\sqrt(3)

    Args:
        X: X-channel information.
        Y: Y-channel information.
        Z: Z-channel information.

    Returns:
        A, E, T Channels.

    """
    return (
        (Z - X) / np.sqrt(2.0),
        (X - 2.0 * Y + Z) / np.sqrt(6.0),
        (X + Y + Z) / np.sqrt(3.0),
    )


def searchsorted2d_vec(a, b, xp=None, gpu=None, **kwargs):
    """Row-wise vectorized :func:`numpy.searchsorted` over a 2D sorted array.

    For each row ``i``, returns the insertion indices of ``b[i]`` into ``a[i]``.
    The implementation offsets each row by a large multiple of the value range
    so a single 1D ``searchsorted`` over the flattened arrays produces the
    correct per-row result, avoiding a Python-level loop.

    Args:
        a: 2D array of shape ``(m, n)``, sorted along axis 1.
        b: 2D array of shape ``(m, k)`` containing the values to insert.
        xp: Array module (``numpy`` or ``cupy``). Defaults to ``numpy``.
        gpu: Optional CUDA device index to switch to before running on GPU.
            Ignored for the NumPy backend.
        **kwargs: Forwarded to ``xp.searchsorted`` (e.g. ``side``).

    Returns:
        2D ``int`` array of shape ``(m, k)`` of insertion indices, one row per
        row of ``a``.
    """
    if xp is None:
        xp = np
    else:
        try:
            xp.cuda.runtime.setDevice(gpu)
        except AttributeError:
            pass

    m, n = a.shape
    max_num = xp.maximum(a.max() - a.min(), b.max() - b.min()) + 1
    r = max_num * xp.arange(a.shape[0])[:, None]
    p = xp.searchsorted((a + r).ravel(), (b + r).ravel(), **kwargs).reshape(m, -1)

    out = p - n * (xp.arange(m)[:, None])
    try:
        xp.cuda.runtime.deviceSynchronize()
    except AttributeError:
        pass

    return out


def get_groups_from_band_structure(
    f0, band_edges, f0_2=None, xp=None, num_groups_base=3, fix_f_test=None, inds=None
):
    """Assign per-source group labels based on a frequency band structure.

    Each source frequency in ``f0`` is bucketed into a band defined by
    ``band_edges``. Sources sharing a temperature, walker, and band index are
    coalesced into mutually-exclusive groups, alternated modulo
    ``num_groups_base`` so that proposals operating on different groups do not
    collide on the same band. Sources with ``f0`` outside ``[band_edges.min(),
    band_edges.max()]`` are marked with group ``-1``; sources removed by the
    optional ``f0_2`` consistency check are marked with group ``-2``.

    Args:
        f0: Source frequencies, shape ``(ntemps, nwalkers, nleaves)``.
        band_edges: 1D sorted array of band-edge frequencies.
        f0_2: Optional secondary frequencies (e.g. proposed values) with the
            same shape as ``f0``. When provided, sources whose ``f0_2`` lands
            more than one band away from ``f0`` are excluded.
        xp: Array module (``numpy`` or ``cupy``). Defaults to ``numpy``.
        num_groups_base: Number of alternating group families (2, 3, or 4).
            Three is typical and ensures non-overlapping proposals across
            adjacent bands.
        fix_f_test: Optional boolean array, same shape as ``f0``. Entries set to
            ``True`` are forcibly removed from the kept set (treated like the
            ``f0_2`` rejection above).
        inds: Currently unused; retained for backwards compatibility.

    Returns:
        Integer array with the same shape as ``f0`` containing group labels
        per source. Special values are ``-1`` (out-of-range) and ``-2`` (not
        assigned to any kept group).

    Raises:
        ValueError: If ``num_groups_base`` is not in ``{2, 3, 4}``.
        TypeError: If ``f0`` or ``band_edges`` is not an ``xp.ndarray``.
    """
    if num_groups_base not in [2, 3, 4]:
        raise ValueError("num_groups_base must be 2 or 3 or 4.")
    if xp is None:
        xp = np

    else:
        try:
            xp.cuda.runtime.setDevice(xp.cuda.runtime.getDevice())

        except AttributeError:
            # it is numpy
            pass

    if not isinstance(f0, xp.ndarray) or not isinstance(band_edges, xp.ndarray):
        raise TypeError(
            "f0 and band_edges must be xp.ndarray with xp as numpy or cupy as given by the xp kwarg."
        )

    shape = f0.shape

    # remove any above or below bands
    bad = (f0 < band_edges.min()) | (f0 > band_edges.max())

    band_indices = xp.searchsorted(band_edges, f0.flatten()).reshape(shape) - 1

    # sort the bands in, but keep places with inds_band_indices
    band_indices_sorted = xp.sort(band_indices, axis=-1)
    inds_band_indices = xp.argsort(band_indices, axis=-1)

    if f0_2 is not None:
        assert f0_2.shape == f0.shape
        band_indices_2 = xp.searchsorted(band_edges, f0_2.flatten()).reshape(shape) - 1
        band_indices_2_sorted = xp.take_along_axis(band_indices_2, inds_band_indices, axis=-1)

        # very important: ensures the proposed new point is not further than 1 band away.
        diff = 1 if num_groups_base > 2 else 0
        keep = np.abs(band_indices_2_sorted.flatten() - band_indices_sorted.flatten()) <= diff
        if fix_f_test is not None:
            keep[fix_f_test.flatten()] = False
        remove = ~keep

    else:
        keep = np.ones(np.prod(band_indices_sorted.shape), dtype=bool)

    # temperature index associated with each band
    temp_inds = xp.repeat(
        xp.arange(band_indices_sorted.shape[0]), np.prod(band_indices_sorted.shape[1:])
    )[
        keep
    ]  # .reshape(shape)

    # walker index associated with each band
    walker_inds = (
        xp.tile(
            xp.arange(band_indices_sorted.shape[1]),
            (band_indices_sorted.shape[0], band_indices_sorted.shape[2], 1),
        )
        .transpose((0, 2, 1))
        .flatten()[keep]
    )

    if f0_2 is not None:
        temp_inds_remove = xp.repeat(
            xp.arange(band_indices_sorted.shape[0]),
            np.prod(band_indices_sorted.shape[1:]),
        )[
            remove
        ]  # .reshape(shape)

        # walker index associated with each band
        walker_inds_remove = (
            xp.tile(
                xp.arange(band_indices_sorted.shape[1]),
                (band_indices_sorted.shape[0], band_indices_sorted.shape[2], 1),
            )
            .transpose((0, 2, 1))
            .flatten()[remove]
        )
        inds_band_indices_remove = inds_band_indices.flatten()[remove]

    # special indexing method
    band_indices_sorted_special = (
        band_indices_sorted.flatten()[keep] + int(1e12) * temp_inds + int(1e6) * walker_inds
    )

    # get the unique special indicators
    (
        unique_special,
        unique_special_start_inds,
        unique_special_reverse,
        unique_special_counts,
    ) = np.unique(
        band_indices_sorted_special,
        return_index=True,
        return_inverse=True,
        return_counts=True,
    )

    # this basically makes mini arange setups for each band
    # the added_contribution for the first unique band index is removed
    added_contribution = xp.arange(band_indices_sorted_special.shape[0])

    # gets the groups
    combined = band_indices_sorted_special + added_contribution
    groups = (
        combined - (combined[unique_special_start_inds])[unique_special_reverse]
    )  # .reshape(shape)

    groups_even_odd_tmp = xp.asarray(
        [
            (num_groups_base * groups + i)
            * (band_indices_sorted.flatten()[keep] % num_groups_base == i)
            for i in range(num_groups_base)
        ]
    )
    groups_even_odd = xp.sum(groups_even_odd_tmp, axis=0)

    groups_out = -2 * xp.ones_like(f0, dtype=int)
    groups_out[(temp_inds, walker_inds, inds_band_indices.flatten()[keep])] = groups_even_odd

    groups_out[bad] = -1

    """if f0_2 is not None and not np.all(keep):
        fix = (temp_inds_remove, walker_inds_remove, inds_band_indices_remove)
        fix_2 = band_indices_2[fix]
        fix_1 = band_indices[fix]"""

    return groups_out


autodoc_type_aliases = {
    "Iterable": "Iterable",
    "ArrayLike": "ArrayLike",
}
