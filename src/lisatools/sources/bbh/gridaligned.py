"""Grid-aligned Phentax MBHB generation.

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

__all__ = ["GridAlignedPhenomTHMTDIWaveform"]


class GridAlignedPhenomTHMTDIWaveform(PhenomTHMTDIWaveform):
    """:class:`PhenomTHMTDIWaveform` that evaluates on the DATA lattice.

    A drop-in replacement whose only difference from the stock class is the
    time grid it evaluates on. That difference is what makes a batch of
    independent parameter sets launchable at all.

    WHY THIS EXISTS
    ---------------
    ``pyResponseTDI`` shares ONE relative evaluation grid across a batch, so
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
    """

    #: Per-instance escape hatch; see the class docstring.
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
        for name in ("waveform_t0", "data_t0"):
            value = float(getattr(self, name))
            residual = value - np.rint(value / dt) * dt
            if abs(residual) > 1e-9:
                raise ValueError(
                    f"grid-aligned generation requires {name} to sit on the "
                    f"dt lattice; got {name} = {value!r} with dt = {dt!r}, "
                    f"residual {residual:.6e} s. The alignment is exact only "
                    f"because waveform_t0 and data_t0 cancel exactly; with a "
                    f"non-lattice value the per-source spread reappears at "
                    f"O(ulp(1e7)) ~ 2e-9 s, which EXCEEDS the 1e-12 tolerance "
                    f"in directresponse.py and would re-break the batch with "
                    f"a message pointing at the waveform rather than here."
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
        """
        dt = float(self.dt)
        n_lead = int(np.ceil(self.tdi_buffer_time / dt)) + 1
        k_data = int(np.rint((self.data_t0 - self.waveform_t0) / dt))
        return k_data - n_lead, n_lead + int(self.domain_settings.N)

    def _aligned_polarizations(
        self, m1, m2, s1z, s2z, distance, phi_ref, inclination, psi,
        merger_time, start_freq=None, ref_freq=None, T=None,
        onset_ramp=True, synchronize=False,
    ):
        """Batched polarizations on a grid of exact multiples of ``dt``.

        Returns ``(times, h_plus, h_cross, merger_time_on_grid)``. The fourth
        item is what must reach :meth:`_apply_response` in place of the
        requested ``merger_time``: the sub-sample part has already been spent
        inside the waveform.
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

        # The merger lattice offset is already INSIDE the grid, so
        # ``_apply_response`` must shift by zero. It uses merger_time only for
        # shifted_t_arr; every other reference is logging.
        return times_out, h_plus, h_cross, np.zeros_like(m_grid)

    # -- dispatch ----------------------------------------------------------
    # These exist so the SPLIT merger time reaches ``_apply_response``.
    # Nothing in waveformbase changes.
    def _call_batched(self, *args, ra, dec, merger_time, **kwargs):
        if not self.grid_align:
            return super()._call_batched(
                *args, ra=ra, dec=dec, merger_time=merger_time, **kwargs)
        kwargs.pop("ref_freq", None)
        t, hp, hc, m_grid = self._aligned_polarizations(
            *args, merger_time=merger_time, **kwargs)
        return self._apply_response(t, hp, hc, ra, dec, m_grid)

    def _call_single(self, *args, ra, dec, merger_time, **kwargs):
        if not self.grid_align:
            return super()._call_single(
                *args, ra=ra, dec=dec, merger_time=merger_time, **kwargs)
        kwargs.pop("ref_freq", None)
        # SAME ramp setting as _call_batched. These disagreed before -- single
        # used onset_ramp=False, batched used True -- which is a ~3000 s taper
        # of difference between serial and batched for the identical row.
        t, hp, hc, m_grid = self._aligned_polarizations(
            *args, merger_time=merger_time, **kwargs)
        return self._apply_response(
            t[0], hp[0], hc[0], float(ra), float(dec), float(m_grid[0]))
