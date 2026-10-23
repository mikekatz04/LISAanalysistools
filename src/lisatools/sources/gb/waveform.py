"""Galactic-binary TDI waveform generators built on :class:`gbgpu.gbgpu.GBGPU`.

Two stand-alone classes (the global fit's GB likelihood does not use them; it runs the GBGPU
signal-heterodyne / WDM lookup computations directly):

* :class:`GBXYZTDIWaveform` — the primary stand-alone GB waveform: an rfft-length XYZ
  TDI-2 FD template buffer from the fastGB kernel ``GBGPU.generate_global_template``
  (``tdi2=True``, ``oversample=1``; default orbits
  :class:`~lisatools.detector.EqualArmlengthOrbits`). This is the class the tutorials teach
  and the one to reach for by default.
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
    r"""Primary stand-alone GB XYZ TDI-2 frequency-domain template.

    :meth:`gbgpu.gbgpu.GBGPU.generate_global_template` fills an rfft-length FD buffer with TDI
    2nd generation and ``oversample=1``. The returned buffer is the GBGPU LDC-convention FD
    template (``FD = dt * rfft(td)``). ``XYZ`` is the default channel set; ``AE`` is available.

    Note:
        This is the legacy fastGB kernel without ``flip_ref_phase``, so it carries
        :math:`e^{+i\phi_0}`, while the analysis side (the on-the-fly / heterodyne templates,
        the GB sampling transform, mojito data) is LDC / JaxGB :math:`e^{-i\phi_0}`.
        ``SyntheticGBProcessingStep`` (the synthetic GB injection) no longer uses this kernel
        (it switched to the chunked-heterodyne FD kernel ``gb_fd_fill_global`` in 2026-08).
        The default :class:`~lisatools.detector.EqualArmlengthOrbits` are analytic equal-arm
        orbits for stand-alone use; the production model is unequal-arm
        (:class:`~lisatools.detector.L1Orbits`).

    Args:
        tdi_chan: ``"XYZ"`` (3 channels, default) or ``"AE"`` (2 channels).
        use_tdi2: Use the TDI 2nd-generation kernel (default ``True``).
        oversample: GBGPU internal oversampling factor (default ``1``; plenty for a template).
        orbits: :class:`~lisatools.detector.Orbits` carrying the phase-reference ``t0``.
            ``None`` builds an :class:`~lisatools.detector.EqualArmlengthOrbits` on
            ``force_backend`` (whose positions are SSB-ecliptic, matching the ecliptic
            ``lam`` / ``beta`` basis).
        force_backend: Backend selection at construction (``"cpu"`` /
            ``"cuda12x"`` / ...); never passed as a per-call kwarg.

    Raises:
        ValueError: If ``tdi_chan`` is not ``"XYZ"`` or ``"AE"``.

    Example:
        .. code-block:: python

            import numpy as np
            from lisatools.sources.gb import GBXYZTDIWaveform

            gb = GBXYZTDIWaveform(force_backend="cpu")
            dt, N = 10.0, 256 * 2048
            # [A, f0 (Hz), fdot (Hz/s), fddot, phi0, iota, psi, lam, beta]
            params = np.array([2e-22, 3.0e-3, 1.0e-17, 0.0, 0.3, 0.7, 1.2, 2.0, 0.5])
            tmpl = gb(params, N * dt, dt)   # (3, N // 2 + 1) XYZ FD template
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

        Every source is summed into one buffer (all into group 0 unless ``group_index`` says
        otherwise).

        Args:
            params: ``(num_sources, 9)`` (or ``(9,)``) array in the ``GBGPU`` global-template
                basis ``[A, f0 (Hz), fdot (Hz/s), fddot (Hz/s^2), phi0, iota, psi, lam, beta]``
                (angles in radians; ``lam`` / ``beta`` in the orbits' frame).
            Tobs: Observation time in seconds.
            dt: Time-domain sample step in seconds.
            group_index: Optional ``(num_sources,)`` int32 group assignment
                (``None`` sums every source into group 0).
            start_freq_ind: Start frequency index of the FD buffer. Default ``0``.

        Returns:
            xp.ndarray: ``(nchannels_full, N // 2 + 1)`` complex128 FD template in the GBGPU
            LDC convention, ``N = round(Tobs / dt)`` (backend array module).

        Raises:
            ValueError: If ``params`` does not have 9 columns.
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

    Thin wrapper over :meth:`gbgpu.gbgpu.GBGPU.run_wave` returning the sparse
    frequency-domain arrays of the FIRST source. Retained unchanged for back-compat; prefer
    :class:`GBXYZTDIWaveform` (XYZ, TDI 2nd generation) for new code and tutorials.

    Args:
        *args: Arguments for :class:`gbgpu.gbgpu.GBGPU`.
        **kwargs: Keyword arguments for :class:`gbgpu.gbgpu.GBGPU`.

    Attributes:
        wave_gen (GBGPU): The GBGPU generator.
    """

    def __init__(self, *args: Any, **kwargs: Any) -> None:
        # wave generating class
        self.wave_gen = GBGPU(*args, **kwargs)

    @property
    def f_arr(self) -> np.ndarray:
        """Frequencies [Hz] of the last call's first source (``xp.ndarray``)."""
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
            *params: Parameters going into :meth:`gbgpu.gbgpu.GBGPU.run_wave` (``A, f0, fdot,
                fddot, phi0, iota, psi, lam, beta``).
            return_array: If ``True``, return ``np.array([A, E, T])``.
                If ``False``, return ``(A, E, T)``.
            **kwargs: Keyword arguments going into :meth:`gbgpu.gbgpu.GBGPU.run_wave`
                (``T``, ``dt``, ``tdi2``, ...).

        Returns:
            np.ndarray | tuple: The first source's three channels. The third entry is read
            from the generator's ``X`` property (the TDI X channel), not ``T``.
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
