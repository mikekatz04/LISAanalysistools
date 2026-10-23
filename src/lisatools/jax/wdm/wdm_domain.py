"""JAX port of ``WDMDomain`` / ``WDMDomainWrap``.

Passive container (mirroring the C++ ``WDMDomainWrap`` of
``lisatools/cutils/binding_wdm_domain.hpp``) for the observed-data WDM
coefficients and the inverse-noise array. The actual inner-product
accumulators (``add_ip_contrib`` / ``add_ip_swap_contrib``) are expressed as
JAX ``jnp.sum`` reductions inside the GBGPU JAX kernels
(``gbgpu.jax.wdm.kernels``, ``gbgpu.jax.wdm.computation_group``; formerly
``fastlisaresponse.jax.wdm.computation_group``); this class only holds the
data and the grid scalars.
"""
from __future__ import annotations

import jax.numpy as jnp
import numpy as np


class WDMDomainWrapJAX:
    """Mirror of the C++ ``WDMDomainWrap`` data container.

    Constructor signature matches the C++ wrapper exactly. The two arrays are
    moved to JAX through the host (``np.asarray``, so pass numpy or JAX
    arrays, not cupy); the scalars are stored as in
    :class:`~lisatools.jax.wdm.wdm_settings.WDMSettingsWrapJAX` (including
    ``Nf_active`` / ``Nt_active``). No layout or length check is made here.

    Args:
        wdm_data (array-like): Flattened observed-data WDM coefficients
            (``num_data`` data streams).
        wdm_noise (array-like): Flattened inverse-noise array (``num_noise``
            noise realisations).
        layer_df: WDM frequency-layer width [Hz].
        layer_dt: WDM time-pixel width [s].
        Nf: Number of frequency layers.
        Nt: Number of time pixels.
        num_channel: Number of TDI channels.
        ind_min_t: First time pixel of the active band.
        ind_max_t: Last time pixel of the active band (inclusive).
        ind_min_f: First frequency layer of the active band.
        ind_max_f: Last frequency layer of the active band (inclusive).
        num_data: Number of data streams in ``wdm_data``.
        num_noise: Number of noise arrays in ``wdm_noise``.
    """

    def __init__(self,
                 wdm_data, wdm_noise,
                 layer_df: float, layer_dt: float,
                 Nf: int, Nt: int, num_channel: int,
                 ind_min_t: int, ind_max_t: int,
                 ind_min_f: int, ind_max_f: int,
                 num_data: int, num_noise: int):
        self.wdm_data = jnp.asarray(np.asarray(wdm_data))
        self.wdm_noise = jnp.asarray(np.asarray(wdm_noise))
        self.layer_df = float(layer_df)
        self.layer_dt = float(layer_dt)
        self.Nf = int(Nf)
        self.Nt = int(Nt)
        self.num_channel = int(num_channel)
        self.ind_min_t = int(ind_min_t)
        self.ind_max_t = int(ind_max_t)
        self.ind_min_f = int(ind_min_f)
        self.ind_max_f = int(ind_max_f)
        self.Nf_active = self.ind_max_f - self.ind_min_f + 1
        self.Nt_active = self.ind_max_t - self.ind_min_t + 1
        self.num_data = int(num_data)
        self.num_noise = int(num_noise)
