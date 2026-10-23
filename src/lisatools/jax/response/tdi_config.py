"""JAX-native mirror of ``TDIConfigWrap`` -- a frozen data container.

The C++ ``TDIConfig`` is a passive struct: the projection kernel reads
its arrays (``unit_starts``, ``unit_lengths``, ``tdi_base_link``,
``tdi_link_combinations``, ``tdi_signs_in``, ``channels``,
``num_units``, ``num_channels``) without ever modifying them. We mirror
that 1:1; the arrays stay as numpy ``int32`` / ``float64`` since the
projection iterates over them at trace time and uses their values to
build the JAX-traced computation. Build it from a
:class:`~lisatools.response.tdiconfig.TDIConfig` with
``TDIConfigWrapJAX(*tdi_config.pytdiconfig_args)``; the consumers are
:mod:`lisatools.jax.response.projection` and the GBGPU JAX kernels.
"""
from __future__ import annotations

import numpy as np


class TDIConfigWrapJAX:
    """Pure-Python analog of ``TDIConfigWrap``.

    Accepts the same 8 arguments as the C++ wrapper (the order of
    :attr:`lisatools.response.tdiconfig.TDIConfig.pytdiconfig_args`); every
    argument is stored under its own name, the arrays converted to numpy (on
    the host; cupy arrays are not accepted by ``np.asarray``).

    Args:
        unit_starts (array-like): int32 start of each unit's delay list.
        unit_lengths (array-like): int32 delay-list length per unit.
        tdi_base_links (array-like): int32 projected link per unit.
        tdi_link_combinations (array-like): int32 delay links, units
            concatenated (``-11`` = no delay).
        tdi_signs (array-like): float64 sign per unit.
        channels (array-like): int32 output channel per unit.
        num_units: Number of units.
        num_channels: Number of output channels (3).

    Example:
        .. code-block:: python

            from lisatools.jax.response.tdi_config import TDIConfigWrapJAX
            from lisatools.response.tdiconfig import TDIConfig

            tdi = TDIConfig("2nd generation", force_backend="cpu")
            wrap = TDIConfigWrapJAX(*tdi.pytdiconfig_args)
            wrap.num_units, wrap.num_channels   # (48, 3)
    """

    def __init__(
        self,
        unit_starts,
        unit_lengths,
        tdi_base_links,
        tdi_link_combinations,
        tdi_signs,
        channels,
        num_units: int,
        num_channels: int,
    ):
        self.unit_starts = np.asarray(unit_starts, dtype=np.int32)
        self.unit_lengths = np.asarray(unit_lengths, dtype=np.int32)
        self.tdi_base_links = np.asarray(tdi_base_links, dtype=np.int32)
        self.tdi_link_combinations = np.asarray(
            tdi_link_combinations, dtype=np.int32
        )
        self.tdi_signs = np.asarray(tdi_signs, dtype=np.float64)
        self.channels = np.asarray(channels, dtype=np.int32)
        self.num_units = int(num_units)
        self.num_channels = int(num_channels)
