from multiprocessing.sharedctypes import Value
import numpy as np
from lisatools.sensitivity import get_sensitivity


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


def AET(X, Y, Z):
    return (
        (Z - X) / np.sqrt(2.0),
        (X - 2.0 * Y + Z) / np.sqrt(6.0),
        (X + Y + Z) / np.sqrt(3.0),
    )


def searchsorted2d_vec(a,b, xp=None, gpu=None, **kwargs):
    if xp is None:
        xp = np
    else:
        try:
            xp.cuda.runtime.setDevice(gpu)
        except AttributeError:
            pass

    m,n = a.shape
    max_num = xp.maximum(a.max() - a.min(), b.max() - b.min()) + 1
    r = max_num*xp.arange(a.shape[0])[:,None]
    p = xp.searchsorted( (a+r).ravel(), (b+r).ravel(), **kwargs).reshape(m,-1)

    out = p - n*(xp.arange(m)[:,None])
    try:
        xp.cuda.runtime.deviceSynchronize()
    except AttributeError:
        pass

    return out

"""
def get_groups_from_band_structure(f0_1, band_edges, f0_2=None, xp=None):
    if xp is None:
        xp = np

    else:
        try:
            xp.cuda.runtime.setDevice(xp.cuda.runtime.getDevice())

        except AttributeError:
            # it is numpy
            pass

    freqs = [f0_1] if f0_2 is None else [f0_1, f0_2]
    band_indices_special_arr = [None for _ in freqs]
    temp_inds_arr = [None for _ in freqs]
    walkers_inds_arr = [None for _ in freqs]
    for i, f0 in enumerate(freqs):
        if not isinstance(f0, xp.ndarray) or not isinstance(band_edges, xp.ndarray):
            raise TypeError("f0 and band_edges must be xp.ndarray with xp as numpy or cupy as given by the xp kwarg.")

        shape = f0.shape

        # remove any above or below bands
        bad = (f0 < band_edges.min()) | (f0 > band_edges.max())
        band_indices = xp.searchsorted(band_edges, f0.flatten()).reshape(shape) - 1

        # temperature index associated with each band
        temp_inds = xp.repeat(xp.arange(band_indices.shape[0]), np.prod(band_indices.shape[1:])).reshape(shape)

        # walker index associated with each band
        walker_inds = xp.tile(xp.arange(band_indices.shape[1]), (band_indices.shape[0], band_indices.shape[2], 1)).transpose((0, 2, 1))

        # special indexing method
        band_indices_special = (band_indices + int(1e12) * temp_inds + int(1e6) * walker_inds)

        band_indices_special_arr[i] = band_indices_special
        walkers_inds_arr[i] = walker_inds
        temp_inds_arr[i] = temp_inds

    if len(band_indices_special_arr) == 1:
        band_indices_special_arr.append(band_indices_special_arr[0].copy())

    band_indices_special_arr = xp.asarray(band_indices_special_arr).transpose((1, 2, 3, 0))
    walkers_inds_arr = xp.asarray(walkers_inds_arr).transpose((1, 2, 3, 0))
    temp_inds_arr = xp.asarray(temp_inds_arr).transpose((1, 2, 3, 0))

    # sort the bands in, but keep places with inds_band_indices
    inds_band_indices = xp.argsort(band_indices_special_arr[:, :, :, 0], axis=-1)

    band_indices_sorted_special = xp.take_along_axis(band_indices_special_arr, inds_band_indices[:, :, :, None], axis=2)

    band_indices_sorted_special_0 = band_indices_sorted_special[:, :, :, 0]

    # get the unique special indicators
    unique_special, unique_special_start_inds, unique_special_reverse, unique_special_counts = np.unique(band_indices_sorted_special_0, return_index=True, return_inverse=True, return_counts=True)

    # this basically makes mini arange setups for each band
    # the added_contribution for the first unique band index is removed
    added_contribution = xp.arange(np.prod(band_indices_sorted_special_0.shape)).reshape(band_indices_sorted_special_0.shape)

    # gets the groups
    combined = band_indices_sorted_special_0 + added_contribution
    groups = combined - ((combined.flatten()[unique_special_start_inds])[unique_special_reverse]).reshape(band_indices_sorted_special_0.shape)

    groups_even_odd = (2 * groups) * (band_indices_sorted_special_0 % 2 == 0) + (2 * groups + 1) * (band_indices_sorted_special_0 % 2 == 1)

    groups_even_odd[bad] = -1
    
    groups_out = groups_even_odd.copy()
    groups_out[(temp_inds.flatten(), walker_inds.flatten(), inds_band_indices.flatten())] = groups_even_odd.flatten()
    breakpoint()
    return groups_out

"""
def get_groups_from_band_structure(f0, band_edges, f0_2=None, xp=None, num_groups_base=3, fix_f_test=None, inds=None):

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
        raise TypeError("f0 and band_edges must be xp.ndarray with xp as numpy or cupy as given by the xp kwarg.")

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
        keep = (np.abs(band_indices_2_sorted.flatten() - band_indices_sorted.flatten()) <= diff)
        if fix_f_test is not None:
            keep[fix_f_test.flatten()] = False
        remove = ~keep
        
    else:
        keep = np.ones(np.prod(band_indices_sorted.shape), dtype=bool)

    # temperature index associated with each band
    temp_inds = xp.repeat(xp.arange(band_indices_sorted.shape[0]), np.prod(band_indices_sorted.shape[1:]))[keep]  # .reshape(shape)

    # walker index associated with each band
    walker_inds = xp.tile(xp.arange(band_indices_sorted.shape[1]), (band_indices_sorted.shape[0], band_indices_sorted.shape[2], 1)).transpose((0, 2, 1)).flatten()[keep]

    if f0_2 is not None:
        temp_inds_remove = xp.repeat(xp.arange(band_indices_sorted.shape[0]), np.prod(band_indices_sorted.shape[1:]))[remove]  # .reshape(shape)

        # walker index associated with each band
        walker_inds_remove = xp.tile(xp.arange(band_indices_sorted.shape[1]), (band_indices_sorted.shape[0], band_indices_sorted.shape[2], 1)).transpose((0, 2, 1)).flatten()[remove]
        inds_band_indices_remove = inds_band_indices.flatten()[remove]

    # special indexing method
    band_indices_sorted_special = (band_indices_sorted.flatten()[keep] + int(1e12) * temp_inds + int(1e6) * walker_inds)

    # get the unique special indicators
    unique_special, unique_special_start_inds, unique_special_reverse, unique_special_counts = np.unique(band_indices_sorted_special, return_index=True, return_inverse=True, return_counts=True)

    # this basically makes mini arange setups for each band
    # the added_contribution for the first unique band index is removed
    added_contribution = xp.arange(band_indices_sorted_special.shape[0])

    # gets the groups
    combined = band_indices_sorted_special + added_contribution
    groups = (combined - (combined[unique_special_start_inds])[unique_special_reverse])  # .reshape(shape)

    groups_even_odd_tmp = xp.asarray([(num_groups_base * groups + i) * (band_indices_sorted.flatten()[keep] % num_groups_base == i) for i in range(num_groups_base)])
    groups_even_odd = xp.sum(groups_even_odd_tmp, axis=0)
    
    groups_out = -2 * xp.ones_like(f0, dtype=int)
    groups_out[(temp_inds, walker_inds, inds_band_indices.flatten()[keep])] = groups_even_odd

    groups_out[bad] = -1

    """if f0_2 is not None and not np.all(keep):
        fix = (temp_inds_remove, walker_inds_remove, inds_band_indices_remove)
        fix_2 = band_indices_2[fix]
        fix_1 = band_indices[fix]"""

    return groups_out