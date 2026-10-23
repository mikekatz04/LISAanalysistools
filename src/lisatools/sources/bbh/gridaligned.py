"""Grid-aligned Phentax MBHB generation (the batched MBH scoring path).

:class:`GridAlignedPhenomTHMTDIWaveform` evaluates every row of a walker batch on the DATA
lattice so ``pyResponseTDI`` can run the batch in one launch;
:class:`WindowedGridAlignedMBHWaveform` does the same on a per-leaf window around the merger
instead of the whole analysis window. With ``MBH_LIKELIHOOD=batched`` (the 9mo / 12mo runs)
the stock getter ``globalfit.stock.erebor.source_runtime.get_mbh_windowed_gen`` builds the
windowed class (``waveform_t0`` snapped onto the ``MBH_WINDOW_DECIMATE * dt`` lattice) inside
:class:`~lisatools.sources.batching.MBHWindowedWDMSignalGen`, which
:class:`~lisatools.globalfit.moves.mbhbatchedmove.MBHBatchedLikeMove` scores with.

Isolated from :mod:`lisatools.sources.bbh.waveform` because
:meth:`GridAlignedPhenomTHMTDIWaveform._aligned_polarizations` reaches into
phentax internals -- ``initial_processing``, ``_compute_strain_single``,
``rotate_by_polarization_angle`` -- and that coupling is easier to keep
honest with a module boundary around it than buried among the stock classes.
"""

from __future__ import annotations

import numpy as np

from ...utils.constants import *
from .waveform import PhenomTHMTDIWaveform, jax, jnp
from ...utils.exceptions import BatchNotLaunchable
from ...utils.utility import get_array_module

__all__ = ["GridAlignedPhenomTHMTDIWaveform", "WindowedGridAlignedMBHWaveform"]


class GridAlignedPhenomTHMTDIWaveform(PhenomTHMTDIWaveform):
    """:class:`PhenomTHMTDIWaveform` that evaluates on the DATA lattice.

    A drop-in replacement whose only difference from the stock class is the
    time grid it evaluates on. That difference is what makes a batch of
    independent parameter sets launchable at all.

    Why this exists: ``pyResponseTDI`` shares ONE relative evaluation grid across a batch, so
    ``t0_shift_to_data`` -- the sub-sample offset between a source's own grid
    and the data grid -- must be identical for every row, and it refuses a
    batch whose offsets differ by more than 1e-12 s.

    Both parameters an MCMC walker actually moves break that:

    * ``t_merger`` is added straight onto the evaluation grid.
    * ``mT`` does too, less obviously: phentax builds its time grid BACKWARDS
      from ``tmax`` in geometric units, so the anchor moves by
      ``500 * MTSUN_SI`` ~ 2.5e-3 s per solar mass.

    So a walker batch was rejected outright. Evaluating every source on the
    data lattice makes each offset EXACTLY zero -- not merely inside the
    tolerance -- and the batch launches.

    The merger time is split into a lattice part and a sub-sample part. The
    grid carries the lattice part; the waveform is evaluated at
    ``t_arr - m_frac`` so the merger still lands at the requested time. The
    sub-sample part is spent inside the waveform rather than against the data
    grid, which is precisely what the response cannot absorb per-source.

    Set :attr:`grid_align` to False for stock behaviour in-process (that is
    how the A/B comparison is taken); the class is otherwise interchangeable.

    The shared lattice spans the analysis window (``domain_settings.N`` samples) plus a
    ``tdi_buffer_time`` lead and tail (:meth:`_common_grid_spec`); the output is cropped back
    to precisely the data grid. Each row's first ``buffer_time`` of TDI output after its own
    onset is zeroed, as on the stock path (:meth:`_zero_onset_warmup`).

    Args:
        *args: Positional arguments of
            :class:`~lisatools.sources.bbh.waveform.PhenomTHMTDIWaveform`.
        **kwargs: Keyword arguments of
            :class:`~lisatools.sources.bbh.waveform.PhenomTHMTDIWaveform` (same defaults).
            ``waveform_t0`` must sit on the data lattice ``data_t0 + k * dt`` (checked at
            generation time) and ``coarse_grain`` must be off.

    Raises:
        ValueError: At generation time, if ``waveform_t0 - data_t0`` is not a multiple of
            ``dt`` (to 1e-9 s) or the phentax model is coarse-grained.
        ImportError: Without jax / phentax (already at construction, from
            :class:`~lisatools.sources.bbh.waveform.PhenomTHMWaveformBase`; the generation-time
            check in :meth:`_check_alignable` is a second guard).
    """

    #: Per-instance escape hatch; see the class docstring. Class attribute, ``True``;
    #: set ``False`` on an instance for the stock grid (then :attr:`supports_batch` is
    #: ``False``).
    grid_align: bool = True

    @property
    def supports_batch(self) -> bool:
        """True only while alignment is actually ON.

        ONE decision in ONE place. A class-level ``supports_batch = True``
        beside a separate ``grid_align`` flag lets the two disagree: with
        ``grid_align = False`` the generator would still advertise batching,
        the container would still try, and ``pyResponseTDI`` would refuse --
        a guaranteed failed launch per call, reported as a fallback warning.
        """
        return bool(self.grid_align)

    # -- preconditions -----------------------------------------------------
    def _check_alignable(self) -> None:
        """Raise unless the lattice alignment can be delivered (jax present, no coarse grain,
        ``waveform_t0`` on the data lattice); called by :meth:`_aligned_polarizations`."""
        if jax is None:  # pragma: no cover - exercised only without jax
            raise ImportError(
                "grid-aligned generation needs jax (and phentax); this class "
                "imports cleanly without them so that lisatools.sources.bbh "
                "stays importable, but it cannot generate."
            )
        """Refuse to claim an alignment we cannot actually deliver.

        Both of these are silent-wrongness risks rather than crashes, so they
        are checked rather than assumed.
        """
        if getattr(self.waveform, "coarse_grain", False):
            raise ValueError(
                "grid-aligned generation requires coarse_grain=False: the "
                "coarse-grained phentax grid is non-uniform, so 'the next "
                "sample is dt later' -- which the lattice construction "
                "assumes -- does not hold. The legacy response path already "
                "forces it off; set coarse_grain=False or use the stock "
                "PhenomTHMTDIWaveform."
            )
        dt = float(self.dt)
        data_t0 = float(self.data_t0)
        # The lattice that matters is the DATA lattice -- data_t0 + k*dt -- which
        # is what this class evaluates on (see the class docstring). Testing each
        # t0 against ABSOLUTE zero additionally demands that the dataset's own
        # time origin be a multiple of dt, which is a property of an arbitrary
        # epoch choice rather than of alignability. Mojito CD1-L is the case in
        # point: its L1 stream is a perfectly uniform dt=2.5 s lattice, but the
        # epoch sits 0.172 s off absolute zero, and its reference time REF sits
        # 0.2 samples (0.5 s) off its own data grid -- so no shift can put both
        # on the absolute lattice, and the absolute test is unsatisfiable for a
        # dataset that is in fact perfectly alignable.
        #
        # Measuring the offset relative to data_t0 also SHRINKS the floating
        # point concern rather than enlarging it: the quantity differenced is
        # O(window span) ~ 1e3-1e7 s instead of O(1e8), so ulp is <= 1e-9 s
        # rather than 1.5e-8 s. data_t0 is the lattice origin, hence exactly on
        # it by construction and no longer something to test.
        offset = float(self.waveform_t0) - data_t0
        residual = offset - np.rint(offset / dt) * dt
        if abs(residual) > 1e-9:
            raise ValueError(
                f"grid-aligned generation requires waveform_t0 to sit on the "
                f"DATA lattice (data_t0 + k*dt); got waveform_t0 - data_t0 = "
                f"{offset!r} with dt = {dt!r}, residual {residual:.6e} s. "
                f"The alignment is exact only because the waveform and data "
                f"grids cancel exactly; with a non-lattice offset the "
                f"per-source spread reappears and EXCEEDS the 1e-12 tolerance "
                f"in directresponse.py, re-breaking the batch with a message "
                f"pointing at the waveform rather than here. Remedy: snap "
                f"waveform_t0 to data_t0 + rint(offset/dt)*dt and subtract the "
                f"same shift from t_plunge, which leaves the absolute merger "
                f"time unchanged."
            )

    # -- the grid ----------------------------------------------------------
    def _split_merger_time(self, merger_time):
        """``merger_time -> (m_int*dt, m_frac)``, ``m_frac`` in ``(-dt/2, dt/2]``."""
        mt = np.atleast_1d(np.asarray(merger_time, dtype=np.float64))
        m_grid = np.rint(mt / self.dt) * self.dt
        return m_grid, mt - m_grid

    def _common_grid_spec(self, T):
        """``(k0, n_grid)`` for the shared absolute lattice.

        PARAMETER-INDEPENDENT BY CONSTRUCTION -- a function of ``data_t0``,
        ``dt``, ``tdi_buffer_time`` and ``T`` only, never of the batch. That is
        the whole point, for three measured reasons:

        * ``phentax._compute_strain_single`` is ``@jax.jit``. A batch-derived
          length recompiles XLA on every likelihood call: 3.2 s against 0.040 s
          cached, an 80x tax that would silently eat the batching win.
        * If the span depended on the batch, a row's column offset would depend
          on WHICH OTHER ROWS shared its call, and a walker's likelihood would
          change with batch membership -- fatal for detailed balance. On a fixed
          lattice each row's columns are a function of its own parameters alone,
          so composition invariance is structural rather than hoped for.
        * A union-of-rows span grows with the walker cloud (+43% samples for a
          one-day merger-time spread) and would need its own refusal path. This
          one never grows and never refuses.

        ``n_lead * dt >= tdi_buffer_time`` is required: it places the ``_lead``
        crop point (``data_t0 - tdi_buffer_time``) strictly inside the grid, so
        the crop lands at the same absolute time it does on the serial path and
        the leading zeroed region is identical.

        The span is the ANALYSIS window, ``domain_settings.N`` -- NOT phentax's
        generation window ``T``. The two differ, and sizing on ``T`` produces a
        grid longer than the data grid, which the FD transform rejects outright
        (``Signal length (262985) != target FFT length (197238)``). Sizing on
        the analysis window also makes ``_apply_response``'s ``start_ind`` crop
        remove exactly the lead margin and leave precisely the data grid.

        Plus a TAIL of :meth:`_tail_samples` past the data end, which the
        dispatch crops off again after the response (:meth:`_crop_tail`), so
        the output is still precisely the data grid. The response reads the
        strain up to ~500 s AHEAD of each output sample (the SSB ->
        spacecraft delay) and ``_apply_response`` zero-pads the strain after
        the lattice: for a source still inspiralling at the data end (a
        merger after it, e.g. a truncated epoch) a lattice stopping there is
        a strain STEP the stock waveform does not have, and the last few
        hundred seconds of output carry its response (measured on the
        windowed subclass: 2.1e3 x the local stock signal; mojito id 16
        leaked 3.4e-3 of noise-weighted ||delta|| into the WDM box).
        """
        dt = float(self.dt)
        n_lead = int(np.ceil(self.tdi_buffer_time / dt)) + 1
        k_data = int(np.rint((self.data_t0 - self.waveform_t0) / dt))
        return k_data - n_lead, n_lead + int(self.domain_settings.N) + self._tail_samples()

    def _tail_samples(self) -> int:
        """Lattice samples past the end of the kept output: ``tdi_buffer_time``
        (>= the response's forward read) plus one sample."""
        return int(np.ceil(float(self.tdi_buffer_time) / float(self.dt))) + 1

    def _output_tail_samples(self) -> int:
        """Trailing output samples :meth:`_crop_tail` drops: the lattice tail
        (a stubbed lattice without one overrides this to 0)."""
        return self._tail_samples()

    def _crop_tail(self, times, channels):
        """Drop the lattice tail from the response output (see
        :meth:`_common_grid_spec`): the output ends at the data end (parent)
        or the segment end (windowed subclass) again -- the parent's is
        precisely the data grid, as FD consumers require."""
        n = int(self._output_tail_samples())
        if n <= 0:
            return times, channels
        return times[..., :-n], channels[..., :-n]

    def _aligned_polarizations(
        self, m1, m2, s1z, s2z, distance, phi_ref, inclination, psi,
        merger_time, start_freq=None, ref_freq=None, T=None,
        onset_ramp=True, synchronize=False,
    ):
        """Batched polarizations on a grid of exact multiples of ``dt``.

        Returns ``(times, h_plus, h_cross, merger_time_on_grid, onset_abs)``.
        The fourth item is what must reach :meth:`_apply_response` in place of
        the requested ``merger_time``: the sub-sample part has already been
        spent inside the waveform. The fifth is each row's ABSOLUTE onset
        label (its first valid lattice sample, clamped to the lattice start),
        consumed by :meth:`_zero_onset_warmup`.
        """
        self._check_alignable()
        xp = self.xp
        dt = float(self.dt)
        wf = self.waveform

        ref_kw = self.get_reference_quantities(
            merger_time=merger_time, start_freq=start_freq, ref_freq=ref_freq)

        args = [self._to_jax(np.atleast_1d(np.asarray(v, dtype=np.float64)))
                for v in (m1, m2, s1z, s2z, distance, phi_ref,
                          inclination, psi)]

        # 1. Everything phentax needs, on ITS grid. This is the public entry
        #    point and exactly what compute_polarizations_at_once calls first.
        #
        #    ``**ref_kw`` IS FORWARDED WHOLE, BY KEYWORD, exactly as the stock
        #    ``wave_gen_batch`` does. Unpacking only the keys this method
        #    happens to name and filling ``initial_processing``'s positionals
        #    by hand silently dropped ``t_min``.
        #
        #    ``get_reference_quantities`` adds ``t_min = -T`` whenever
        #    ``time_bounded_start`` is set -- which is the DEFAULT -- and
        #    phentax derives the start from ``f_min`` only when ``t_min`` is
        #    NaN. Hardcoding NaN therefore un-bounded the template in time and
        #    shortened it badly at high total mass: measured 57,789 valid
        #    samples against the stock 525,970 at m1 = 1e7, m2 = 8e6 Msun, i.e.
        #    11% of the analysis window, for a reason that has nothing to do
        #    with grid alignment. A walker proposing high mT would have taken
        #    a likelihood hit attributable to this alone.
        wf_params, times_mass, mask, amp22, ph22 = wf.initial_processing(
            *args,
            delta_t=dt,
            T=T if T is not None else wf.T,
            **ref_kw,
        )

        M_sec = np.asarray(wf_params.total_mass) * MTSUN_SI          # (B,)

        # ``mask`` is used ONLY for its per-row VALID COUNT. It must not be
        # reused, padded or broadcast onto the shared lattice: where it is
        # False, ``times_mass`` holds the constant ``Mt_min`` repeated rather
        # than real earlier times, and outside a row's own range the model
        # returns a smooth FULL-AMPLITUDE inspiral -- no NaN, no decay, no
        # self-limiting (measured at 26% of in-band peak across 466,627
        # samples). Under lisatools' default ``time_bounded_start=True`` the
        # returned mask is entirely True, so reusing it looks perfectly correct
        # in the default configuration and is silently wrong the moment a
        # high-mass or f_min-started row appears. It is rebuilt below.
        n_valid = np.asarray(mask.sum(axis=1)).astype(np.int64)      # (B,)
        t_last_sec = np.asarray(times_mass[:, -1]) * M_sec           # (B,)

        m_grid, m_frac = self._split_merger_time(merger_time)
        if m_grid.size == 1 and M_sec.size > 1:
            m_grid = np.repeat(m_grid, M_sec.size)
            m_frac = np.repeat(m_frac, M_sec.size)

        k0, n_grid = self._common_grid_spec(T if T is not None else wf.T)
        k_merge = np.rint(m_grid / dt).astype(np.int64)              # exact: m_grid is n*dt

        # Per-row window as INTEGER COLUMN INDICES on the shared lattice.
        e_idx = np.rint((t_last_sec + m_frac) / dt).astype(np.int64) + k_merge
        j_end = e_idx - k0
        j_start = j_end - (n_valid - 1)      # may be < 0: inspiral opening before
                                             # the grid is intended, not an error

        # Evaluation times, by INTEGER arithmetic. Inside a row's own window the
        # columns therefore carry bit-identical values to the per-row grid this
        # replaces -- the equivalence was checked directly (max|diff| = 0.0).
        jj = jnp.arange(n_grid, dtype=jnp.int64)
        n_col = (k0 + jj)[None, :] - jnp.asarray(k_merge)[:, None]
        eval_sec = n_col.astype(jnp.float64) * dt - jnp.asarray(m_frac)[:, None]
        times_new = eval_sec / jnp.asarray(M_sec)[:, None]

        mask_new = (jj[None, :] >= jnp.asarray(j_start)[:, None]) & (
            jj[None, :] <= jnp.asarray(j_end)[:, None]
        )

        strain = jax.vmap(wf._compute_strain_single)(
            times_new, mask_new, wf_params, amp22, ph22,
            wf_params.inclination, wf_params.phi_ref,
        )
        h_plus = jnp.real(strain)
        h_cross = -jnp.imag(strain)
        h_plus, h_cross = wf.rotate_by_polarization_angle(
            h_plus, h_cross, wf_params.psi)

        h_plus = self._from_jax(h_plus, do_synchronize=synchronize)
        h_cross = self._from_jax(h_cross, do_synchronize=synchronize)

        # ONE array of times, shared by every row. ``trim_and_shift_times`` is
        # deliberately NOT called: its premise is that a row's valid samples are
        # a right-aligned SUFFIX, which this grid breaks by design -- each row's
        # block ends at its own merger and is interior. Measured on a 4-walker
        # cloud, ``times[:, -max_valid:]`` silently drops 1557-1599 valid
        # leading samples from 3 of 4 rows. The exact n*dt lattice supplies the
        # one guarantee that method actually provided downstream: strictly
        # increasing, exactly dt-spaced times.
        times_1d = xp.arange(n_grid, dtype=xp.float64) * dt + (k0 * dt)
        times_out = xp.broadcast_to(times_1d[None, :], (M_sec.size, n_grid))

        if onset_ramp:
            # ``num_pad`` is each row's OWN onset column, so the taper is
            # anchored where that row actually begins rather than at the array
            # start. That is what makes the ramp correct on a shared grid.
            ramp = self._leading_onset_ramp(
                num_points=n_grid,
                num_pad=np.maximum(j_start, 0),
                taper_length=int(self.tdi_buffer_time * 5 / dt),
                xp=xp,
            )
            h_plus = h_plus * ramp
            h_cross = h_cross * ramp

        # Column j_start is the lattice point NEAREST the row's first valid
        # phentax time (e_idx rounds, and j_start = e_idx - (n_valid - 1)):
        # exactly where the stock response array begins after its
        # t0_shift_to_data re-alignment, so the warm-up zeroing below is
        # anchored on the same sample as the stock's.
        onset_abs = float(self.waveform_t0) + (k0 + np.maximum(j_start, 0)) * dt

        # The merger lattice offset is already INSIDE the grid, so
        # ``_apply_response`` must shift by zero. It uses merger_time only for
        # shifted_t_arr; every other reference is logging.
        return times_out, h_plus, h_cross, np.zeros_like(m_grid), onset_abs

    def _zero_onset_warmup(self, times, channels, onset_abs):
        """Zero each row's first ``buffer_time`` of TDI output after ITS onset.

        STOCK PARITY. In the stock (non-grid-aligned) path the response
        array starts at the waveform's own first sample -- unless that
        sample precedes ``data_t0 - tdi_buffer_time``, in which case
        ``TDWaveformBase._apply_response``'s ``_lead`` crop re-bases the
        array start there -- and the response zeros the first
        ``buffer_time / dt`` output samples of that array: the turn-on
        transient of a waveform that starts abruptly (or through a short
        ramp) mid-data. On the shared lattice the response array starts at
        the LATTICE head instead, so the same zeroing lands there and a row
        whose onset is interior kept its transient: mojito MBHB id 17 carried
        a 7e-24 TDI spike, 200x the local signal, for ~5000 s after onset --
        2.4 nats of logL at truth against the stock template (2026-09-30).
        This method reproduces only the INTERIOR-onset case. Rows whose onset
        precedes the lattice are clamped to the lattice start, where the
        zeroing coincides with the lattice-head zeroing ``_apply_response``
        already did (itself subject to the same ``_lead`` re-basing near
        ``data_t0``) -- nothing extra is removed there.
        """
        n_buf = int(self.buffer_time / self.dt)
        if n_buf <= 0:
            return times, channels
        xp = get_array_module(channels)
        t = xp.atleast_2d(xp.asarray(times))
        cut = xp.asarray(np.atleast_1d(onset_abs), dtype=xp.float64) + (n_buf - 0.5) * float(self.dt)
        keep = t >= cut[:, None]                       # (B, N)
        # In place: ``channels`` is the fresh ``xp.array`` copy
        # ``_apply_response`` builds from the response output (or a crop view
        # of it), owned by this call -- no second (B, C, N) array (~2 GB at
        # B = 24 on GPU).
        if channels.ndim == 3:
            channels *= keep[:, None, :]
        else:                                          # single row: (C, N)
            channels *= keep[0][None, :]
        return times, channels

    # -- dispatch ----------------------------------------------------------
    # These exist so the SPLIT merger time reaches ``_apply_response``.
    # Nothing in waveformbase changes.
    #: Container-level flags that :class:`~lisatools.analysiscontainer.AnalysisContainer`
    #: injects into a generator's kwargs. They are NOT waveform parameters, and
    #: ``_aligned_polarizations`` has a strict signature, so they are dropped
    #: here rather than forwarded. The stock ``wave_gen_batch`` tolerates them
    #: only because it happens to carry ``**kwargs``; being explicit says which
    #: names are deliberately ignored instead of swallowing every typo.
    #:
    #: ``apply_transform`` asks a generator to apply its own internal
    #: parameter transform. These take waveform parameters directly and have
    #: none, so the flag is a no-op -- but raising TypeError on it made the
    #: class unusable from the standard container path.
    _CONTAINER_FLAGS = ("apply_transform", "per_model_per_signal")

    def _drop_container_flags(self, kwargs):
        """Pop :attr:`_CONTAINER_FLAGS` from ``kwargs`` in place and return it."""
        for _f in self._CONTAINER_FLAGS:
            kwargs.pop(_f, None)
        return kwargs

    def _call_batched(self, *args, ra, dec, merger_time, **kwargs):
        """Batch on the shared lattice: aligned polarizations, ONE response call with the
        lattice merger offset (zero), onset warm-up zeroing, tail crop. ``ref_freq`` and the
        container flags are dropped. With ``grid_align=False`` defers to the stock path."""
        if not self.grid_align:
            return super()._call_batched(
                *args, ra=ra, dec=dec, merger_time=merger_time, **kwargs)
        kwargs.pop("ref_freq", None)
        self._drop_container_flags(kwargs)
        t, hp, hc, m_grid, onset_abs = self._aligned_polarizations(
            *args, merger_time=merger_time, **kwargs)
        times, channels = self._apply_response(t, hp, hc, ra, dec, m_grid)
        return self._crop_tail(*self._zero_onset_warmup(times, channels, onset_abs))

    def _call_single(self, *args, ra, dec, merger_time, **kwargs):
        """One source on the same lattice and with the same onset ramp as
        :meth:`_call_batched` (so serial and batched evaluation agree)."""
        if not self.grid_align:
            return super()._call_single(
                *args, ra=ra, dec=dec, merger_time=merger_time, **kwargs)
        kwargs.pop("ref_freq", None)
        self._drop_container_flags(kwargs)
        # SAME ramp setting as _call_batched. These disagreed before -- single
        # used onset_ramp=False, batched used True -- which is a ~3000 s taper
        # of difference between serial and batched for the identical row.
        t, hp, hc, m_grid, onset_abs = self._aligned_polarizations(
            *args, merger_time=merger_time, **kwargs)
        times, channels = self._apply_response(
            t[0], hp[0], hc[0], float(ra), float(dec), float(m_grid[0]))
        return self._crop_tail(*self._zero_onset_warmup(times, channels, onset_abs))


class WindowedGridAlignedMBHWaveform(GridAlignedPhenomTHMTDIWaveform):
    """Grid-aligned generation on a PER-LEAF window instead of the analysis window.

    The parent's shared lattice spans the whole analysis window
    (``domain_settings.N``) -- six months of samples for a source that lives
    for 90 days. The global fit sets a window around each leaf's merger
    (user ruling 2026-09-29: 90 days before to 10 days after, plus pads) and
    this class evaluates the batch on THAT lattice. Everything else -- the
    exact integer lattice, the split merger time, ``merger_time = 0`` handed
    to ``_apply_response`` -- is the parent's.

    ``set_window`` takes the ABSOLUTE segment start (a data-lattice time that
    is also a WDM layer boundary) and the segment's sample count; the lattice
    is prepended with ``n_lead`` samples so the invalid head lands outside
    the segment, where the placement clips it, and appended with
    ``tdi_buffer_time`` of tail so the response's forward read (the SSB ->
    spacecraft delay, <= ~500 s) sees the real strain at the segment end even
    when the source merges after it. Two heads are invalid:

    * the response's retarded reads reach up to ``tdi_buffer_time`` (600 s)
      before a sample, so the first ``tdi_buffer_time`` of output is
      warm-up;
    * ``_apply_response`` then ZEROS the first ``buffer_time / dt`` output
      samples of the lattice outright (``buffer_time`` defaults to 15000 s,
      ``MBH_PHENOM_DEFAULT_BUFFER_TIME``).

    The lead is therefore sized on ``max(tdi_buffer_time, buffer_time)``.
    Sizing it on ``tdi_buffer_time`` alone (610 s at dt = 10 s) let the
    zeroed head run ~14.4 ks INTO the segment, silently deleting any signal
    there. Generating without a window is refused: silently falling back
    to the analysis-window lattice is exactly the 6-month generation this
    class exists to avoid.

    Exception near the data start: the lead keeps the zeroed head outside the
    segment only while the lattice start lies after ``data_t0 -
    tdi_buffer_time``. For a segment starting within ~``buffer_time``
    (~4 h) of the run's ``data_t0`` (an edge-clamped window, see
    ``mbh_window_layers``), ``_apply_response``'s ``_lead`` crop re-bases
    the array start to ``data_t0 - tdi_buffer_time`` and the
    ``buffer_time`` zeroing then removes the first ~4 h of the segment. The
    stock path does exactly the same there (identical crop and zeroing), so
    windowed and stock templates still agree; those hours sit inside the
    data window's taper.

    The constructor is that of :class:`GridAlignedPhenomTHMTDIWaveform`; the production build
    (``get_mbh_windowed_gen``) passes a decimated ``data_td_settings`` at
    ``MBH_WINDOW_DECIMATE * dt`` and the run's WDM settings as ``output_domain_settings``.
    :class:`~lisatools.sources.batching.MBHWindowedWDMSignalGen` calls :meth:`set_window` before
    each batch.

    Raises:
        RuntimeError: At generation time, if :meth:`set_window` was never called.
    """

    _window_spec = None

    def set_window(self, t_seg_abs: float, n_seg: int) -> None:
        """Set the lattice to a segment: ``n_seg`` samples from the absolute time ``t_seg_abs``.

        The lattice gets ``n_lead = ceil(max(tdi_buffer_time, buffer_time) / dt) + 1`` samples
        before the segment and :meth:`_tail_samples` after it; the result is stored as
        :attr:`window_spec` ``(k0, n_grid)`` in lattice samples from ``waveform_t0``.

        Args:
            t_seg_abs: Absolute segment start [s]; must lie on ``waveform_t0 + k * dt``.
            n_seg: Segment length in samples.

        Raises:
            ValueError: If ``t_seg_abs`` is off the waveform lattice by more than 1e-6 samples.
        """
        dt = float(self.dt)
        rel = (float(t_seg_abs) - float(self.waveform_t0)) / dt
        k_seg = int(np.rint(rel))
        if abs(rel - k_seg) > 1e-6:
            raise ValueError(
                f"segment start {t_seg_abs!r} is not on the waveform lattice "
                f"(waveform_t0 + k*dt): residual {(rel - k_seg) * dt:.3e} s"
            )
        # Cover BOTH invalid heads (see the class docstring): the retarded-read
        # warm-up and the ``buffer_time`` zeroing in ``_apply_response``.
        lead_time = max(float(self.tdi_buffer_time), float(getattr(self, "buffer_time", 0.0) or 0.0))
        n_lead = int(np.ceil(lead_time / dt)) + 1
        # ...and run PAST the segment end by ``tdi_buffer_time``: the response
        # reads the strain up to ~500 s AHEAD of each output sample (the SSB
        # -> spacecraft delay), and ``_apply_response`` zero-pads the strain
        # after the lattice. For a source still inspiralling at the segment
        # end (merger after the data end, edge-clamped box) a lattice stopping
        # there is a strain STEP the stock waveform does not have: mojito
        # MBHB id 16 merging 1 d after the grid end carried a 2.6e5 x spike in
        # the segment's last ~300 s, leaking 3.4e-3 of noise-weighted
        # ||delta|| into the active box through the WDM transform
        # (2026-09-30). The dispatch crops the tail off after the response
        # (``_crop_tail``), so the output ends at the segment end.
        self._window_spec = (k_seg - n_lead, n_lead + int(n_seg) + self._tail_samples())

    @property
    def window_spec(self):
        """tuple[int, int] | None: ``(k0, n_grid)`` of the last :meth:`set_window`, or ``None``."""
        return self._window_spec

    def _common_grid_spec(self, T):
        """The window's ``(k0, n_grid)``; ``T`` is ignored. Raises ``RuntimeError`` before
        :meth:`set_window` (no fall back to the analysis-window lattice)."""
        if self._window_spec is None:
            raise RuntimeError(
                "WindowedGridAlignedMBHWaveform.set_window() was never called; "
                "refusing to fall back to the full analysis-window lattice."
            )
        return self._window_spec
