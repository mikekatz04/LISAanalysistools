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

def get_groups_from_band_structure(f0, band_edges, xp=None):
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

    bad = (f0 < band_edges.min()) | (f0 > band_edges.max())
    band_indices = xp.searchsorted(band_edges, f0.flatten()).reshape(shape) - 1

    band_indices_sorted = xp.sort(band_indices, axis=-1)
    inds_band_indices = xp.argsort(band_indices, axis=-1)

    temp_inds = xp.repeat(xp.arange(band_indices_sorted.shape[0]), np.prod(band_indices_sorted.shape[1:])).reshape(shape)

    walker_inds = xp.tile(xp.arange(band_indices_sorted.shape[1]), (band_indices_sorted.shape[0], band_indices_sorted.shape[2], 1)).transpose((0, 2, 1))

    band_indices_sorted_special = (band_indices_sorted + int(1e12) * temp_inds + int(1e6) * walker_inds).flatten()

    unique_special, unique_special_start_inds, unique_special_reverse, unique_special_counts = np.unique(band_indices_sorted_special, return_index=True, return_inverse=True, return_counts=True)

    added_contribution = xp.arange(band_indices_sorted_special.shape[0])

    combined = band_indices_sorted_special + added_contribution
    groups = (combined - (combined[unique_special_start_inds])[unique_special_reverse]).reshape(shape)

    groups_even_odd = (2 * groups) * (band_indices_sorted % 2 == 0) + (2 * groups + 1) * (band_indices_sorted % 2 == 1)

    groups_even_odd[bad] = -1
    
    groups_out = groups_even_odd.copy()
    groups_out[(temp_inds.flatten(), walker_inds.flatten(), inds_band_indices.flatten())] = groups_even_odd.flatten()

    return groups_out