"""Adapters that let a time-domain waveform generator drive a batched likelihood.

WHY AN ADAPTER IS NEEDED
    :meth:`~lisatools.sources.waveformbase.TDWaveformBase.__call__` returns a
    ``(signal, start_freqs)`` tuple -- the raw transform output -- which
    :class:`~lisatools.analysiscontainer.AnalysisContainer` cannot consume as a
    template. The sanctioned per-source conversion is
    ``_td_to_output_domain``, which :meth:`get_signals_for_residuals` already
    uses; it refuses a 2-D time array outright ("treat different sources
    separately").

    So a batch has to be split for the DOMAIN TRANSFORM even though it stays
    whole for the RESPONSE. That is the right split: one
    ``compute_tdi_channels`` call carries the whole batch through the
    expensive response evaluation, and only the per-source transform loops.

    ``__call__`` is deliberately left alone. Changing what it returns would
    alter every existing caller, and the point here is to add a capability,
    not to move the floor under working code.
"""

from __future__ import annotations

from typing import Any

import numpy as np

from ..domains import DomainBase, DomainBaseArray
from ..utils.utility import get_array_module

__all__ = ["BatchedDomainSignalGen"]


class BatchedDomainSignalGen:
    """Wrap a TD waveform generator so it can be an ``AnalysisContainer.signal_gen``.

    Returns a :class:`~lisatools.domains.DomainBase` in the generator's
    analysis domain: unbatched for scalar parameters, and carrying a leading
    SOURCE axis when handed arrays. That leading axis is what
    :func:`~lisatools.diagnostic.inner_product` reduces around to produce one
    inner product per source, so a batched template yields a vector of
    likelihoods rather than a combined one.

    Args:
        wave_gen: A :class:`~lisatools.sources.waveformbase.TDWaveformBase`
            (or anything exposing ``compute_tdi_channels`` and
            ``_td_to_output_domain``).

    Attributes:
        supports_batch: Mirrors the wrapped generator's own declaration --
            NEVER hardcoded True. Wrapping something that cannot guarantee a
            shared sub-sample alignment must not manufacture the capability;
            the wrapper only forwards a promise the generator already made.
    """

    def __init__(self, wave_gen: Any):
        self.wave_gen = wave_gen

    @property
    def supports_batch(self) -> bool:
        """Forward the wrapped generator's declaration, defaulting to False."""
        return bool(getattr(self.wave_gen, "supports_batch", False))

    def __repr__(self) -> str:
        return f"{type(self).__name__}({self.wave_gen!r})"

    def __call__(self, *params: Any, **kwargs: Any) -> DomainBase:
        times, channels = self.wave_gen.compute_tdi_channels(*params, **kwargs)

        if getattr(times, "ndim", 1) == 1:
            return self.wave_gen._td_to_output_domain(
                times_in=times, signal_in=channels
            )

        doms = [
            self.wave_gen._td_to_output_domain(
                times_in=times[i], signal_in=channels[i]
            )
            for i in range(channels.shape[0])
        ]
        return self._stack(doms)

    @staticmethod
    def _stack(doms: list) -> DomainBase:
        """One :class:`DomainBase` with a leading source axis.

        :class:`DomainBaseArray` is a LIST of per-source domains, which the
        inner products would have to loop over -- the very loop being removed.
        Stacking gives the single batched object whose leading axis
        ``inner_product`` already understands.
        """
        if not doms:
            raise ValueError("cannot stack an empty batch of signals")

        settings = doms[0].settings
        for i, d in enumerate(doms[1:], start=1):
            if d.settings is not settings and d.arr.shape != doms[0].arr.shape:
                raise ValueError(
                    f"source {i} produced shape {d.arr.shape} against "
                    f"{doms[0].arr.shape} for source 0; a batch can only be "
                    f"stacked when every source lands on the same basis. This "
                    f"usually means the sources were placed on different grids "
                    f"upstream."
                )

        xp = get_array_module(doms[0].arr)
        stacked = xp.stack([d.arr for d in doms], axis=0)
        return settings.associated_class(stacked, settings)
