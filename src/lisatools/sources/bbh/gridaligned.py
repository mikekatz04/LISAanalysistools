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
        n_times = int(times_mass.shape[-1])

        # 2. The lattice we want. ``times_mass`` is in geometric units and
        #    anchored at Mt_end, so its LAST sample sits t_last_sec after the
        #    peak -- that anchor, not t_arr[0], is what sets the alignment.
        t_last_sec = np.asarray(times_mass[:, -1]) * M_sec           # (B,)
        m_grid, m_frac = self._split_merger_time(merger_time)
        if m_grid.size == 1 and M_sec.size > 1:
            m_grid = np.repeat(m_grid, M_sec.size)
            m_frac = np.repeat(m_frac, M_sec.size)

        # t_arr[j] is an integer multiple of dt, so t_arr + m_grid +
        # waveform_t0 lands on the data lattice; the waveform is EVALUATED at
        # t_arr[j] - m_frac so the merger still sits at the requested time.
        # Built from integers rather than by adding a float offset, so
        # "integer multiple of dt" is exact and not merely close.
        n_last = np.rint((t_last_sec + m_frac) / dt)                 # (B,)
        j = np.arange(n_times, dtype=np.float64)
        n_grid = n_last[:, None] - (n_times - 1.0 - j[None, :])
        t_arr_sec = n_grid * dt
        eval_sec = t_arr_sec - m_frac[:, None]

        # 3. Exact re-evaluation of the model at those times.
        times_new = self._to_jax(eval_sec / M_sec[:, None])
        strain = jax.vmap(wf._compute_strain_single)(
            times_new, mask, wf_params, amp22, ph22,
            wf_params.inclination, wf_params.phi_ref,
        )
        h_plus = jnp.real(strain)
        h_cross = -jnp.imag(strain)
        h_plus, h_cross = wf.rotate_by_polarization_angle(
            h_plus, h_cross, wf_params.psi)

        h_plus = self._from_jax(h_plus, do_synchronize=synchronize)
        h_cross = self._from_jax(h_cross, do_synchronize=synchronize)
        mask_x = self._from_jax(mask, do_synchronize=synchronize)
        times_x = xp.asarray(t_arr_sec)

        # 4. The stock trim / onset-ramp tail, verbatim, so the two paths
        #    differ ONLY in the grid.
        times_out = self.trim_and_shift_times(times_x, mask_x, xp=xp, dt=dt)
        num_keep = times_out.shape[-1]
        num_pad = num_keep - mask_x.sum(axis=1).astype(int)

        if not onset_ramp:
            if int(xp.max(num_pad)) > 0:
                raise ValueError(
                    "onset_ramp=False requires every source in the batch to "
                    "produce the same number of valid samples (num_pad == 0 "
                    f"for all); got max num_pad = {int(xp.max(num_pad))}. "
                    "This is NOT a grid-alignment failure -- it is the same "
                    "equal-length requirement the stock batched path has, and "
                    "it depends on the f_min-determined inspiral length "
                    "versus the window."
                )
            return times_out, h_plus[:, -num_keep:], h_cross[:, -num_keep:], m_grid

        taper_length = int(self.tdi_buffer_time * 5 / dt)
        ramp = self._leading_onset_ramp(
            num_points=num_keep, num_pad=num_pad,
            taper_length=taper_length, xp=xp)
        return (times_out, h_plus[:, -num_keep:] * ramp,
                h_cross[:, -num_keep:] * ramp, m_grid)

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
        # B == 1 => num_pad == 0 by construction, so this reproduces the stock
        # mask-and-drop path rather than the ramp.
        kwargs.setdefault("onset_ramp", False)
        t, hp, hc, m_grid = self._aligned_polarizations(
            *args, merger_time=merger_time, **kwargs)
        return self._apply_response(
            t[0], hp[0], hc[0], float(ra), float(dec), float(m_grid[0]))
