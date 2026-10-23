"""Adapters that let a time-domain waveform generator drive a batched likelihood.

:class:`BatchedDomainSignalGen` turns a
:class:`~lisatools.sources.waveformbase.TDWaveformBase` into an
:class:`~lisatools.analysiscontainer.AnalysisContainer` ``signal_gen`` that returns one domain
object with a leading source axis. :class:`MBHWindowedWDMSignalGen` is the production MBH
variant (``MBH_LIKELIHOOD=batched``): a per-leaf segment WDM transform around the merger,
built by ``globalfit.stock.erebor.source_runtime.get_mbh_windowed_gen`` and driven by
:class:`~lisatools.globalfit.moves.mbhbatchedmove.MBHBatchedLikeMove`.

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

from ..domains import (
    DomainBase,
    DomainBaseArray,
    TDSettings,
    WDMSettings,
    WDMSignal,
    place_td_signal_on_grid,
)
from ..utils.utility import get_array_module, tukey

__all__ = ["BatchedDomainSignalGen", "MBHWindowedWDMSignalGen"]


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

    Example:
        .. code-block:: python

            from lisatools.analysiscontainer import AnalysisContainer
            from lisatools.sources.batching import BatchedDomainSignalGen
            from lisatools.sources.bbh import GridAlignedPhenomTHMTDIWaveform

            gen = BatchedDomainSignalGen(GridAlignedPhenomTHMTDIWaveform(**run_kwargs))
            ac = AnalysisContainer(data, sens, signal_gen=gen)
            # 1-D parameter arrays -> one DomainBase with a leading source axis
            batch_signal = gen(*rows.T)
    """

    def __init__(self, wave_gen: Any):
        self.wave_gen = wave_gen

    @property
    def supports_batch(self) -> bool:
        """Forward the wrapped generator's declaration, defaulting to False.

        Mirrors the wrapped generator's own declaration -- NEVER hardcoded
        True. Wrapping something that cannot guarantee a shared sub-sample
        alignment must not manufacture the capability; the wrapper only
        forwards a promise the generator already made.
        """
        return bool(getattr(self.wave_gen, "supports_batch", False))

    def __repr__(self) -> str:
        return f"{type(self).__name__}({self.wave_gen!r})"

    def __call__(self, *params: Any, **kwargs: Any) -> DomainBase:
        """Generate the template(s) in the generator's analysis domain.

        One ``compute_tdi_channels`` call for the whole batch, then
        ``wave_gen._td_to_output_domain`` per source and :meth:`_stack`.

        Args:
            *params: Waveform parameters, scalars (one source) or 1-D arrays (a batch).
            **kwargs: Forwarded to ``wave_gen.compute_tdi_channels``.

        Returns:
            DomainBase: Unbatched for 1-D ``times``; otherwise one domain object whose array
            has a leading source axis (also for a one-row batch).

        Raises:
            BatchNotLaunchable: Propagated from the generator when the batch cannot share one
                response call.
            ValueError: If the per-source signals land on different bases.
        """
        times, channels = self.wave_gen.compute_tdi_channels(*params, **kwargs)

        if getattr(times, "ndim", 1) == 1:
            return self.wave_gen._td_to_output_domain(
                times_in=times, signal_in=channels
            )

        # THE SOURCE COUNT COMES FROM ``times``, NOT FROM ``channels``.
        # pyResponseTDI squeezes its batch axis when batch_size == 1
        # (``return raw[0] if self.batch_size == 1 else raw``), while
        # _apply_response keeps a (1, N) time grid because its
        # ``single_source = isinstance(ra, float)`` test is False for a
        # length-1 ARRAY. So a one-row batch arrives as
        # times (1, N) + channels (nchannels, N) -- the two disagree about
        # whether a leading axis exists.
        #
        # Looping ``range(channels.shape[0])`` therefore ran nchannels times
        # for a single source and indexed times[1]: IndexError, which is not
        # a BatchNotLaunchable and so was NOT caught by the container's
        # fallback -- it killed the sampler call outright. A one-row chunk is
        # not exotic: any ``batch_max_size`` that does not divide the walker
        # count produces one as the remainder.
        n_src = int(times.shape[0])
        squeezed = channels.ndim == times.ndim

        doms = [
            self.wave_gen._td_to_output_domain(
                times_in=times[i],
                signal_in=channels if squeezed else channels[i],
            )
            for i in range(n_src)
        ]
        # _stack even for n_src == 1: the caller asked for a batch and the
        # leading axis is what makes the inner products return a vector of
        # one rather than a scalar.
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


class MBHWindowedWDMSignalGen(BatchedDomainSignalGen):
    """Batched ``signal_gen`` whose per-row domain step is a SEGMENT transform.

    Each row's TD channels are placed on a segment of the data lattice that
    is ``n_pad`` WDM layers wider than the kept box on each side, transformed
    on a segment ``WDMSettings`` (same ``Nf``, ``dt``; ``Nt_seg`` layers), and
    the ``n_pad`` edge layers are discarded. The result is a ``WDMSignal`` on
    the RUN's settings with ``active_slice_t = [n_start, n_start + Nt_keep)``,
    so containers slice their residual and PSD to it (``_slice_to_template``)
    and fills add it into the full residual (``add_signal``) -- the batched
    move first re-labels it onto the containers' own settings, whose ``t0``
    need not match these settings' (see ``t0_abs`` below). The template is
    zero at both segment ends (onset ramp inside the box; ringdown dead), so
    the periodic segment transform's wrap-around touches only the discarded
    pad layers -- measured on a toy grid: kept-layer relative error 1.2e-5 at
    8 pad layers, 9e-7 at 16 (tests/test_mbh_windowed_signal_gen.py).

    Args:
        wave_gen (WindowedGridAlignedMBHWaveform): generator exposing
            ``compute_tdi_channels`` (times on the data lattice, ABSOLUTE seconds). If it also
            exposes ``set_window(t_seg_abs, n_seg)`` it is told the segment (production:
            :class:`~lisatools.sources.bbh.gridaligned.WindowedGridAlignedMBHWaveform`).
        wdm_settings (WDMSettings): the run's :class:`~lisatools.domains.WDMSettings`
            (device-local). Only its GRID is used (``Nf``, ``Nt``, ``dt``, active box,
            window); its ``t0`` is not trusted as a time.
        nchannels: TDI channels the data carries (leading channels kept). Default ``3``.
        tukey_alpha: the run's full-length data window alpha; the matching
            slice of that window multiplies the segment (0 -> none). Default ``0.0``.
        t0_abs: ABSOLUTE time of layer 0 of the grid (the data start,
            ``general_info.data_t0``); the segment starts at ``t0_abs + s0 *
            layer_dt``. Defaults to ``wdm_settings.t0`` (read at
            :meth:`set_window`) for callers whose settings carry the data
            start. A stock erebor build's settings do NOT (the WDM factory
            builds ``t0 = 0``; a GB comp build later sets it to the data start
            in place), so the global fit passes it explicitly.
        decimate: lattice decimation factor (``MBH_WINDOW_DECIMATE``). The
            segment is sampled at ``decimate * dt`` and transformed on ``Nf / decimate``
            layers -- the same ``layer_dt``, hence the same pixels: for content
            below the coarse Nyquist the WDM coefficients equal the ``dt`` ones
            (measured 7e-12 at decimation 2, 4e-10 at 4 on a synthetic chirp sum).
            The generator must then produce its channels at ``decimate * dt`` too
            (it is told ``Nf / decimate * Nt_seg`` samples). ``Nf`` must be
            divisible by ``decimate``
            and the coarse grid must still hold the run's active band. Default ``1``.

    Attributes:
        geometry (dict | None): The last :meth:`set_window` layout (``n_start``, ``Nt_keep``,
            ``n_pad``, ``n_pad_lo``, ``n_pad_hi``, ``n_pad_hi_req``, ``s0``, ``Nt_seg``,
            ``t_seg``, ``decimate``); ``None`` before the first call.
        waveform_t0 (float): Set by the stock getter after construction (the snapped epoch).
        t_plunge_snap (float): Set by the stock getter (the epoch snap, subtracted from the
            rows' ``t_plunge`` by the move).

    Raises:
        ValueError: If ``decimate`` is not a positive divisor of ``wdm_settings.Nf``.

    Example:
        .. code-block:: python

            adapter = MBHWindowedWDMSignalGen(
                windowed_gen, wdm_settings, nchannels=3, tukey_alpha=window_alpha,
                t0_abs=data_t0, decimate=2,
            )
            adapter.set_window(n_start, Nt_keep, n_pad)
            wdm_batch = adapter(*rows.T)   # WDMSignal, leading source axis, kept box only
    """

    def __init__(
        self, wave_gen, wdm_settings, nchannels: int = 3, tukey_alpha: float = 0.0,
        t0_abs: float | None = None, decimate: int = 1,
    ):
        super().__init__(wave_gen)
        self.wdm = wdm_settings
        self.nchannels = int(nchannels)
        self.tukey_alpha = float(tukey_alpha or 0.0)
        self._t0_abs = None if t0_abs is None else float(t0_abs)
        self.decimate = int(decimate)
        if self.decimate < 1 or int(self.wdm.Nf) % self.decimate:
            raise ValueError(
                f"decimate={decimate}: must be a positive divisor of the grid's Nf={self.wdm.Nf}"
            )
        self.geometry = None
        self._seg_td = None
        self._seg_wdm = None
        self._box = None
        self._win_seg = None

    @property
    def t0_abs(self) -> float:
        """ABSOLUTE time of WDM layer 0 (``wdm.t0`` when none was given)."""
        return float(self.wdm.t0) if self._t0_abs is None else self._t0_abs

    @property
    def window_key(self):
        """``(n_start, Nt_keep, n_pad, n_pad_hi)`` as last passed to :meth:`set_window`."""
        g = self.geometry
        return None if g is None else (
            g["n_start"], g["Nt_keep"], g["n_pad"], g["n_pad_hi_req"]
        )

    def set_window(
        self, n_start: int, Nt_keep: int, n_pad: int, n_pad_hi: int | None = None
    ) -> None:
        """Set the kept box and the padded segment; tells ``wave_gen`` the segment.

        Kept layers ``[n_start, n_start + Nt_keep)`` (absolute WDM time-layer indices); segment
        pads ``n_pad`` below and ``n_pad_hi`` (default ``n_pad``) above. An edge-clamped
        window (:func:`~lisatools.globalfit.moves.mbhbatchedmove.mbh_window_layers`)
        passes a 0 pad on the side where the segment meets the grid edge. The low pad grows by
        one layer when needed to start the segment on an even absolute layer (the real WDM
        basis alternates with the parity of ``n + m``), and the high pad by one to make the
        segment's layer count even. Builds the segment ``TDSettings`` / ``WDMSettings`` (at
        ``decimate * dt``, ``Nf / decimate`` layers), the kept box (``wdm.get_slice``), the
        matching slice of the full-length Tukey window, and :attr:`geometry`.

        Args:
            n_start: First kept time layer (absolute index on the run grid).
            Nt_keep: Number of kept layers (``>= 1``).
            n_pad: Pad layers below the kept box (``>= 0``).
            n_pad_hi: Pad layers above; ``None`` means ``n_pad``.

        Raises:
            ValueError: For invalid counts, a segment outside ``[0, Nt)``, or kept layers
                outside the data's active time box.
            RuntimeError: If the segment settings do not reproduce the run's active frequency
                layers (e.g. ``Nf / decimate`` cannot hold the band).
        """
        n_start, Nt_keep, n_pad = int(n_start), int(Nt_keep), int(n_pad)
        n_pad_hi_req = n_pad if n_pad_hi is None else int(n_pad_hi)
        if Nt_keep < 1 or n_pad < 0 or n_pad_hi_req < 0:
            raise ValueError(
                f"set_window: Nt_keep={Nt_keep} n_pad={n_pad} n_pad_hi={n_pad_hi_req}"
            )
        # The real WDM basis alternates with the parity of the ABSOLUTE layer
        # index (n + m), so a segment whose layer 0 is an odd run layer
        # transforms in the opposite phase convention: measured kept-layer
        # error 1.36 at s0 odd vs 2.9e-5 at s0 even. Grow the low pad by one
        # layer to put the segment start on an even layer (an odd start is
        # >= 1, so there is always room; a start at layer 0 is even already).
        n_pad_lo = n_pad + ((n_start - n_pad) % 2)
        n_pad_hi = n_pad_hi_req
        Nt_seg = Nt_keep + n_pad_lo + n_pad_hi
        if Nt_seg % 2:  # WDMSettings needs an even layer count
            Nt_seg += 1
            n_pad_hi += 1
        s0 = n_start - n_pad_lo
        Nt = int(self.wdm.Nt)
        if s0 < 0 or s0 + Nt_seg > Nt:
            raise ValueError(
                f"segment layers [{s0}, {s0 + Nt_seg}) fall outside the WDM grid [0, {Nt})"
            )
        rel_t0 = n_start - int(self.wdm.ind_min_t)
        if rel_t0 < 0 or rel_t0 + Nt_keep > int(self.wdm.Nt_active):
            raise ValueError(
                f"kept layers [{n_start}, {n_start + Nt_keep}) fall outside the data's "
                f"active box [{self.wdm.ind_min_t}, {self.wdm.ind_max_t + 1})"
            )
        decimation = self.decimate
        Nf = int(self.wdm.Nf)
        Nf_seg = Nf // decimation
        dt = float(self.wdm.data_dt) * decimation
        layer_dt = float(self.wdm.layer_dt)
        t_seg = self.t0_abs + s0 * layer_dt
        backend = self.wdm.backend
        self._seg_td = TDSettings(Nf_seg * Nt_seg, dt, t0=t_seg, force_backend=backend)
        self._seg_wdm = WDMSettings(
            Nf_seg, Nt_seg, dt, t0=t_seg, oversample=self.wdm.oversample,
            min_freq=self.wdm.min_freq, max_freq=self.wdm.max_freq,
            is_complex=self.wdm.is_complex, force_backend=backend,
        )
        if (
            int(self._seg_wdm.Nf_active) != int(self.wdm.Nf_active)
            or int(self._seg_wdm.ind_min_f) != int(self.wdm.ind_min_f)
        ):
            raise RuntimeError(
                "segment WDM settings do not reproduce the run's active frequency layers"
                + (f" (decimate={decimation}: Nf/{decimation} = {Nf_seg} layers cannot hold "
                   f"the run's band up to {self.wdm.max_freq} Hz)" if decimation > 1 else "")
            )
        self._box = self.wdm.get_slice(
            (slice(0, int(self.wdm.Nf_active)), slice(rel_t0, rel_t0 + Nt_keep))
        )
        if self.tukey_alpha > 0.0:
            w = tukey(int(self.wdm.N), self.tukey_alpha, xp=np)
            self._win_seg = self.wdm.xp.asarray(w[s0 * Nf:(s0 + Nt_seg) * Nf:decimation])
        else:
            self._win_seg = None
        self.geometry = dict(
            n_start=n_start, Nt_keep=Nt_keep, n_pad=n_pad, n_pad_lo=n_pad_lo,
            n_pad_hi=n_pad_hi, n_pad_hi_req=n_pad_hi_req,
            s0=s0, Nt_seg=Nt_seg, t_seg=t_seg, decimate=decimation,
        )
        if hasattr(self.wave_gen, "set_window"):
            self.wave_gen.set_window(t_seg, Nf_seg * Nt_seg)

    def _to_domain(self, times, channels):
        """One row: place the leading ``nchannels`` on the segment lattice, segment WDM
        transform (window slice), keep layers ``[n_pad_lo, n_pad_lo + Nt_keep)``; returns a
        :class:`~lisatools.domains.WDMSignal` on the kept box."""
        # No guard here: this is an internal helper, only ever reached via
        # __call__, which checks self.geometry before touching wave_gen at
        # all -- a second check here would just be dead code (see the
        # mutation rule: every behaviour must have a test that fails if the
        # line is deleted, and no test can distinguish a raise here from the
        # one in __call__ since __call__ always fires first).
        g = self.geometry
        placed = place_td_signal_on_grid(
            channels[: self.nchannels], self._seg_td, times=times
        )
        seg = placed.transform(self._seg_wdm, window=self._win_seg)
        kept = seg.arr[..., g["n_pad_lo"]: g["n_pad_lo"] + g["Nt_keep"]]
        return WDMSignal(kept, self._box)

    def __call__(self, *params, **kwargs):
        """Generate the windowed WDM template(s) of one source or a batch.

        Args:
            *params: Waveform-basis parameters (scalars or 1-D arrays), forwarded to
                ``wave_gen.compute_tdi_channels``.
            **kwargs: Forwarded to ``wave_gen.compute_tdi_channels``.

        Returns:
            WDMSignal: On the kept box ``(nch, Nf_active, Nt_keep)``; a batch has a leading
            source axis.

        Raises:
            RuntimeError: If :meth:`set_window` was never called.
        """
        if self.geometry is None:
            raise RuntimeError(
                "MBHWindowedWDMSignalGen: call set_window() before generating"
            )
        times, channels = self.wave_gen.compute_tdi_channels(*params, **kwargs)
        if getattr(times, "ndim", 1) == 1:
            return self._to_domain(times, channels)
        n_src = int(times.shape[0])
        squeezed = channels.ndim == times.ndim
        return self._stack([
            self._to_domain(times[i], channels if squeezed else channels[i])
            for i in range(n_src)
        ])
