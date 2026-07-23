"""Galactic-binary TDI waveform generators built on :class:`gbgpu.gbgpu.GBGPU`.

Two stock classes:

* :class:`GBXYZTDIWaveform` — **the primary, global-fit-aligned GB waveform.**
  It fills the same rfft-length FD template buffer, TDI convention, orbits and
  oversample as the stock ``all_sources`` GB injection/template path
  (``lisatools.globalfit.stock.erebor.injections.SyntheticGBProcessingStep``):
  ``GBGPU.generate_global_template`` with ``tdi2=True``, ``XYZ`` channels,
  :class:`~lisatools.detector.EqualArmlengthOrbits`, ``oversample=1``. This is
  the class the tutorials teach and the one to reach for by default.
* :class:`GBAETWaveform` — the historical thin ``GBGPU.run_wave`` wrapper
  returning per-source ``(A, E, T)`` sparse frequency arrays. Kept unchanged
  for back-compat only; prefer :class:`GBXYZTDIWaveform`.
"""

from __future__ import annotations

from typing import Any, Optional, Tuple

import numpy as np

from gbgpu.gbgpu import GBGPU

from ..waveformbase import AETTDIWaveform


class GBXYZTDIWaveform:
    """Primary global-fit-aligned GB XYZ TDI-2 frequency-domain template.

    Reproduces the stock ``all_sources`` GB convention exactly (see
    ``lisatools.globalfit.stock.erebor.injections.SyntheticGBProcessingStep``):
    :meth:`GBGPU.generate_global_template` fills an rfft-length FD buffer using
    TDI 2nd generation, :class:`~lisatools.detector.EqualArmlengthOrbits`, and
    ``oversample=1``. The returned buffer is the GBGPU LDC-convention FD template
    (``FD = dt * rfft(td)``) — the same array ``SyntheticGBProcessingStep``
    inverse-FFTs onto the data grid. ``XYZ`` is the default and aligned channel
    set; ``AE`` is available but non-stock.

    Args:
        tdi_chan: ``"XYZ"`` (3 channels, the stock/global-fit default) or
            ``"AE"`` (2 channels).
        use_tdi2: Use the TDI 2nd-generation kernel (``True`` — global-fit
            default).
        oversample: GBGPU internal oversampling factor (``1`` — global-fit
            default; plenty for a template).
        orbits: :class:`~lisatools.detector.Orbits` carrying the phase-reference
            ``t0``. ``None`` builds a stock
            :class:`~lisatools.detector.EqualArmlengthOrbits` on
            ``force_backend`` (consistent with the ecliptic ``lam``/``beta``
            basis).
        force_backend: Backend selection at construction (``"cpu"`` /
            ``"cuda12x"`` / ...); never passed as a per-call kwarg.
    """

    def __init__(
        self,
        tdi_chan: str = "XYZ",
        use_tdi2: bool = True,
        oversample: int = 1,
        orbits: Optional[Any] = None,
        force_backend: str = "cpu",
    ) -> None:
        from ...detector import EqualArmlengthOrbits

        if tdi_chan not in ("XYZ", "AE"):
            raise ValueError(f"tdi_chan must be 'XYZ' or 'AE'; got {tdi_chan!r}.")

        self.tdi_chan = tdi_chan
        self.use_tdi2 = use_tdi2
        self.oversample = oversample
        self.nchannels_full = 3 if tdi_chan == "XYZ" else 2

        if orbits is None:
            orbits = EqualArmlengthOrbits(force_backend=force_backend)
        # NOTE(phi0 convention): GBGPU without flip_ref_phase carries
        # e^{+i*phi0}; the analysis side (on-the-fly template, GB sampling
        # transform, mojito data) is LDC/JaxGB e^{-i*phi0}. Any consumer
        # comparing THIS legacy-FD path against those must reconcile the
        # sign (cf. the 2026-07-23 synthetic-injection fix in
        # stock/erebor/injections.py).
        self.wave_gen = GBGPU(force_backend=force_backend, orbits=orbits)
        # ``generate_global_template`` reads ``self.gpus`` directly (set
        # elsewhere in the move setup); a stand-alone template needs it set.
        self.wave_gen.gpus = None if force_backend == "cpu" else [0]

    def __call__(
        self,
        params: np.ndarray,
        Tobs: float,
        dt: float,
        group_index: Optional[np.ndarray] = None,
        start_freq_ind: int = 0,
    ) -> np.ndarray:
        """Build the XYZ TDI-2 FD template buffer.

        Args:
            params: ``(num_sources, 9)`` array in the ``GBGPU`` global-template
                basis ``[A, f0, fdot, fddot, phi0, iota, psi, lam, beta]``.
            Tobs: Observation time in seconds.
            dt: Time-domain sample step in seconds.
            group_index: Optional ``(num_sources,)`` int32 group assignment
                (``None`` sums every source into group 0).
            start_freq_ind: Start frequency index of the FD buffer.

        Returns:
            ``(nchannels_full, Tobs//dt // 2 + 1)`` complex FD template in the
            GBGPU LDC convention.
        """
        xp = self.wave_gen.xp
        params = xp.atleast_2d(xp.asarray(params, dtype=xp.float64))
        if params.shape[1] != 9:
            raise ValueError(
                f"params must have shape (num_sources, 9); got {params.shape}."
            )
        num_sources = params.shape[0]

        target_N = int(round(Tobs / dt))
        data_length = target_N // 2 + 1

        if group_index is None:
            group_index = xp.zeros(num_sources, dtype=xp.int32)
        else:
            group_index = xp.asarray(group_index, dtype=xp.int32)

        # A 1D buffer is written in place (the 2D/3D path internally
        # ``.flatten()``s to a copy, dropping the kernel writes).
        template_flat = xp.zeros(
            self.nchannels_full * data_length, dtype=xp.complex128
        )
        self.wave_gen.generate_global_template(
            params,
            group_index,
            template_flat,
            start_freq_ind=start_freq_ind,
            T=Tobs,
            dt=dt,
            tdi_channel_setup=self.tdi_chan,
            tdi2=self.use_tdi2,
            oversample=self.oversample,
            data_length=data_length,
        )
        return template_flat.reshape(self.nchannels_full, data_length)


class GBAETWaveform(AETTDIWaveform):
    """Legacy AET waveform for Galactic Binaries (back-compat only).

    Thin wrapper over :meth:`GBGPU.run_wave` returning per-source ``(A, E, T)``
    sparse frequency-domain arrays. Retained unchanged for back-compat; the
    stock / global-fit-aligned GB waveform is :class:`GBXYZTDIWaveform` (XYZ,
    TDI 2nd generation) — prefer it for new code and tutorials.

    Args:
        *args: Arguments for :class:`GBGPU`.
        **kwargs: Keyword arguments for :class:`GBGPU`.

    """

    def __init__(self, *args: Any, **kwargs: Any) -> None:
        # wave generating class
        self.wave_gen = GBGPU(*args, **kwargs)

    @property
    def f_arr(self) -> np.ndarray:
        """Frequency array."""
        return self._f_arr

    @f_arr.setter
    def f_arr(self, f_arr: np.ndarray) -> None:
        """Set the frequency array."""
        self._f_arr = f_arr

    def __call__(
        self, *params: Any, return_array: Optional[bool] = False, **kwargs: Any
    ) -> Tuple[np.ndarray, np.ndarray, np.ndarray] | np.ndarray:
        """Generate waveform.

        Args:
            *params: Parameters going into :meth:`GBGPU.run_wave`.
            return_array: If ``True``, return ``array([A, E, T])``.
                If ``False``, return ``(A, E, T)``.
            **kwargs: Keyword arguments going into :meth:`GBGPU.run_wave`.

        Returns:
            Output waveform.

        """
        self.wave_gen.run_wave(
            *params,
            **kwargs,
        )

        # prepare outputs
        self.f_arr = self.wave_gen.freqs[0]
        A = self.wave_gen.A[0]
        E = self.wave_gen.E[0]
        T = self.wave_gen.X[0]

        if return_array:
            return np.array([A, E, T])
        else:
            return (A, E, T)
