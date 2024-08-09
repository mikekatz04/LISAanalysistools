from __future__ import annotations
from typing import Tuple
import numpy as np

# from ..sensitivity import get_sensitivity

try:
    import cupy as cp

except (ModuleNotFoundError, ImportError):
    import numpy as cp

    pass


def get_array_module(arr: np.ndarray | cp.ndarray) -> object:
    """Return array library of an array (np/cp).

    Args:
        arr: Numpy or Cupy array.

    """
    if isinstance(arr, np.ndarray):
        return np
    elif isinstance(arr, cp.ndarray):
        return cp
    else:
        raise ValueError("arr must be a numpy or cupy array.")


def generate_noise_fd(freqs, df, *sensitivity_args, func=None, **sensitivity_kwargs):
    if func is None:
        func = get_sensitivity

    norm = 0.5 * (1.0 / df) ** 0.5
    psd = func(freqs, *sensitivity_args, **sensitivity_kwargs)
    noise_to_add = psd ** (1 / 2) * (
        np.random.normal(0, norm, len(freqs))
        + 1j * np.random.normal(0, norm, len(freqs))
    )
    return noise_to_add


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
        band_indices_2_sorted = xp.take_along_axis(
            band_indices_2, inds_band_indices, axis=-1
        )

        # very important: ensures the proposed new point is not further than 1 band away.
        diff = 1 if num_groups_base > 2 else 0
        keep = (
            np.abs(band_indices_2_sorted.flatten() - band_indices_sorted.flatten())
            <= diff
        )
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
        band_indices_sorted.flatten()[keep]
        + int(1e12) * temp_inds
        + int(1e6) * walker_inds
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
    groups_out[(temp_inds, walker_inds, inds_band_indices.flatten()[keep])] = (
        groups_even_odd
    )

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
