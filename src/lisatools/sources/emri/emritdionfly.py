"""Time-domain EMRI TDI-on-the-fly waveform generator.

The EMRI analogue of :class:`bbhx.sobbhtdionfly.SOBBHTDIonFly` /
:class:`bbhx.mbhtdionfly.MBHTDIonFly`: take a FEW sparse-mode holder, reconstruct
per-``(l, m, k, n)``-mode amplitude/phase, and feed each mode (plus its ``-m``
partner) to the generic :class:`~lisatools.response.tdionfly.TDTDIonTheFly`
response. On the run path :class:`~lisatools.sources.emri.wdm_direct.EMRIDirectWDM` builds an
:class:`EMRITDIonFly` with ``frame="icrs_special"`` and uses its sky conversion
(:meth:`EMRITDIonFly.sky`), its ``AMP_FACTOR`` and :func:`host_holder`; with the production
``response="dense"`` the response itself is
:class:`~lisatools.response.tdionfly.TDDenseTDIonTheFly`, not :meth:`EMRITDIonFly.run_response`.

**Frame convention.** FEW takes its sky/spin angles ``(qS, phiS, qK, phiK)`` as **ecliptic**
polar angles, so the viewing angle and polarization are produced in the **ecliptic** frame.
With ``frame="ecliptic"`` (the class default) the sky is handed to the response in that frame
too, so the ``orbits`` MUST be loaded with ``frame="ecliptic"`` (the sky/polarization frame and
the orbits frame must agree; the previous version converted only the sky to ICRS while leaving
``psi`` ecliptic, a mixed frame that did not match the orbits). With
``frame="icrs_special"`` (the validated mojito recipe, used by the direct-to-WDM template) the
sky is converted ecliptic -> ICRS for ``frame="icrs"`` orbits while ``psi`` stays the FEW one,
matching the production ``ResponseWrapper`` path.

**Per-mode inclination.** Each mode is fed with ``inc = 0``; the response kernel's
``(1 + cos^2 i)`` / ``2 cos i`` inclination factors then reduce to ``2``, which
``AMP_FACTOR = 1/2`` cancels. The viewing-angle dependence is already baked into the FEW
spin-weighted harmonics ``Y_{lm}(theta)``, which differ for ``+m`` vs ``-m`` -- hence the two
are fed as separate subs.
"""
from typing import Optional

import numpy as np

from few.utils.utility import get_polarization_angle, get_viewing_angles

from lisatools.response.directresponse import ecliptic_to_icrs
from lisatools.response.tdionfly import TDTDIonTheFly
from lisatools.utils.constants import YRSID_SI

from .domain import few_domain_guard


class _SkipFew(Exception):
    """Internal: a precomputed holder replaces the FEW call."""


def host_holder(H):
    """Host (numpy) copy of a FEW SparseInfoHolder.

    On a GPU FEW generator the holder's arrays are cupy; the TOF feed, the harmonic tracks
    and the mode bookkeeping are small host-side computations (the response itself runs on
    the TOF backend), so everything downstream reads this copy.

    Args:
        H: FEW ``SparseInfoHolder`` (``return_sparse_holder=True`` output).

    Returns:
        types.SimpleNamespace: ``t_arr``, ``teuk_modes``, ``phases``, ``freqs``, ``ylms``,
        ``ls``, ``ms``, ``ks``, ``ns`` as numpy arrays (``phases`` / ``freqs`` ``None`` when
        absent) and ``integrate_backwards`` as a bool.
    """
    import types

    def _h(x):
        return x.get() if hasattr(x, "get") else (np.asarray(x) if x is not None else None)

    return types.SimpleNamespace(
        t_arr=_h(H.t_arr), teuk_modes=_h(H.teuk_modes), phases=_h(getattr(H, "phases", None)),
        freqs=_h(getattr(H, "freqs", None)), ylms=_h(H.ylms), ls=_h(H.ls), ms=_h(H.ms), ks=_h(H.ks),
        ns=_h(H.ns), integrate_backwards=bool(getattr(H, "integrate_backwards", False)),
    )


class EMRITDIonFly:
    """Build the TDI response of a FEW EMRI waveform mode-by-mode on the fly.

    Args:
        wave_gen: a FEW waveform generator returning a sparse-mode holder when
            called with ``return_sparse_holder=True`` (e.g.
            :class:`few.waveform.FastKerrEccentricEquatorialFlux`).
        orbits: LISA orbits whose frame matches ``frame``: **loaded with**
            ``frame="ecliptic"`` for ``"ecliptic"``, ``frame="icrs"`` for ``"icrs_special"``
            (see the frame convention in the module docstring).
        tdi_config: a :class:`~lisatools.response.tdiconfig.TDIConfig`.
        dt: sampling cadence [s].
        Tobs: observation time [s] (converted to years for FEW, whose ``T`` is in years).
        t0: absolute epoch [s] of the trajectory start (the FEW initial
            conditions are defined at ``t0``).
        delay_margin: seconds to trim the TDI evaluation grid inside the
            waveform spline at each end, so every delayed waveform query
            (``t - k.x`` with the SSB projection ``|k.x| <~ 1 AU / c ~ 500 s``,
            plus the TDI arm delays) stays inside the spline. The default 600 s
            covers the LISA geometry; a 1-sample trim is NOT enough.
        frame: ``"ecliptic"`` (default, the historical all-ecliptic feed: sky,
            polarization and orbits all ecliptic) or ``"icrs_special"`` (the
            validated mojito recipe: FEW viewing angle and ``psi`` from the
            ecliptic-polar sky + raw catalogue spin, the sky handed to the
            response converted ecliptic -> ICRS, orbits loaded with
            ``frame="icrs"``). mojito's polarization basis is ICRS, so only
            ``icrs_special`` matches it; ``ecliptic`` differs by the
            parallactic angle (~8% mismatch, LAT 3057a3e4).
        n_fine: when set, feed the response ``n_fine`` trajectory points
            evaluated from the integrator's dense (8th-order) output via FEW's
            ``upsample``/``new_t`` path instead of the sparse adaptive knots
            (tens of points over the inspiral, which a cubic amp/phase spline
            cannot hold). ``None`` keeps the sparse feed. Must be ``>= 16``.
        t_fine_window: optional absolute ``(t_lo, t_hi)`` [s]: place the fine
            points over this window only (padded outward by ``delay_margin``
            plus two point spacings, so the delay trim still covers it).
            Default: the whole ``[t0, t0 + Tobs]``.
        t_fine: optional explicit fine times [s] relative to ``t0`` (FEW clock), possibly
            non-uniform; sorted and de-duplicated, at least 16 points. Overrides ``n_fine``
            (set to its size) and ``t_fine_window``.

    Attributes:
        AMP_FACTOR (float): ``1/2``, cancels the ``inc = 0`` kernel factor ``1 + cos^2 0``.
        FRAMES (tuple): The accepted ``frame`` values.
        last_holder (types.SimpleNamespace): Host holder of the last :meth:`prepare_feed`.
        tdi_gen (TDTDIonTheFly): Response object of the last :meth:`run_response`.

    Raises:
        ValueError: For an unknown ``frame``, ``n_fine < 16`` or fewer than 16 ``t_fine``
            points.

    Example:
        .. code-block:: python

            from lisatools.response.tdiconfig import TDIConfig
            from lisatools.sources.emri import EMRITDIonFly

            fly = EMRITDIonFly(few_gen, icrs_orbits, TDIConfig("2nd generation"), dt, Tobs,
                               t0=ref_epoch, frame="icrs_special", n_fine=4096)
            out = fly(*few_params)        # TDTDIOutput, one row per (l, m, k, n) sub
            xyz = out.eval_tdi(t_eval)    # per-sub channels; sum over subs for the template
    """

    FRAMES = ("ecliptic", "icrs_special")

    def __init__(self, wave_gen, orbits, tdi_config, dt, Tobs, t0, delay_margin=600.0,
                 frame="ecliptic", n_fine=None, t_fine_window=None, t_fine=None):
        if frame not in self.FRAMES:
            raise ValueError(f"frame must be one of {self.FRAMES}, got {frame!r}")
        if n_fine is not None and int(n_fine) < 16:
            raise ValueError("n_fine must be >= 16")
        self.wave_gen = wave_gen
        self.orbits = orbits
        self.tdi_config = tdi_config
        self.dt = dt
        self.T = Tobs
        self.t0 = t0
        self.delay_margin = delay_margin
        self.frame = frame
        self.n_fine = None if n_fine is None else int(n_fine)
        self.t_fine_window = t_fine_window
        # explicit (possibly NON-uniform) fine times relative to t0: the input splines
        # auto-detect general spacing and the response is evaluated point by point
        self.t_fine = None
        if t_fine is not None:
            self.t_fine = np.unique(np.asarray(t_fine, dtype=float))
            if self.t_fine.size < 16:
                raise ValueError("t_fine needs >= 16 points")
            self.n_fine = int(self.t_fine.size)

    # cancels the inc=0 kernel factor (1 + cos^2 0) = 2 of the TDI-on-the-fly response
    AMP_FACTOR = 1 / 2.0

    @staticmethod
    def mode_amp_phase(K, include_minus_mkn=True, amp_factor=1.0):
        """Per-sub (amplitude, phase) with h = sum_sub amp * exp(-1j * phase).

        Rows: the holder's m >= 0 modes, then (if ``include_minus_mkn``) the
        -m partners of the m != 0 modes. The phase is ``m Phi_phi + k Phi_theta + n Phi_r``
        minus the phases of ``Y_lm`` and of the unwrapped Teukolsky amplitude (the -m rows use
        FEW's ``(-1)^l Y_{l,-m} conj(A) e^{+i Phi}`` term).

        Args:
            K: Host FEW holder (:func:`host_holder`), with ``phases`` at its times.
            include_minus_mkn (bool, optional): Append the -m partners. Default ``True``.
            amp_factor (float, optional): Multiplies every amplitude. Default ``1.0``.

        Returns:
            tuple[np.ndarray, np.ndarray]: ``(amp, phase)``, each ``(num_sub, n_times)``.
        """
        mode_amp_phase = np.unwrap(np.angle(K.teuk_modes), axis=0)
        mode_amp_amp = np.abs(K.teuk_modes)
        ylm_phase = np.angle(K.ylms)
        ylm_amp = np.abs(K.ylms)
        nm = K.ms.shape[0]
        _mode_phase = (
            K.ms[None, :] * K.phases[:, 0][:, None]
            + K.ks[None, :] * K.phases[:, 1][:, None]
            + K.ns[None, :] * K.phases[:, 2][:, None]
        )
        phase_plus = _mode_phase - ylm_phase[:nm] - mode_amp_phase
        amp_plus = amp_factor * mode_amp_amp * ylm_amp[:nm]
        if not include_minus_mkn:
            return amp_plus.T, phase_plus.T
        # FEW's -m term (summation/directmodesum.py): (-1)^l Y_{l,-m} conj(A) e^{+i Phi}
        #   = |Y_{l,-m}||A| exp(-i [-Phi - arg Y_{l,-m} + arg A - l*pi]).
        # NOT -phase_plus: that assumes arg Y_{l,-m} = -arg Y_{l,m} - l*pi, which the
        # real SWSH prefactors violate by pi for some modes (2% strain error at 1e-7).
        keep_minus_m = K.ms != 0
        l_pi = np.pi * np.asarray(K.ls, dtype=float)[None, :]
        phase_minus = (
            -_mode_phase - ylm_phase[nm:][None, :] + mode_amp_phase - l_pi
        )[:, keep_minus_m]
        amp_minus = (amp_factor * mode_amp_amp * ylm_amp[nm:])[:, keep_minus_m]
        return (
            np.concatenate([amp_plus, amp_minus], axis=-1).T,
            np.concatenate([phase_plus, phase_minus], axis=-1).T,
        )

    def _fine_times(self) -> np.ndarray:
        """Fine trajectory times, relative to ``t0`` (FEW's clock)."""
        if self.t_fine is not None:
            return self.t_fine
        if self.t_fine_window is None:
            lo, hi = 0.0, float(self.T)
        else:
            lo = float(self.t_fine_window[0]) - self.t0
            hi = float(self.t_fine_window[1]) - self.t0
        pad = self.delay_margin + 2.0 * (hi - lo) / (self.n_fine - 1)
        lo, hi = max(0.0, lo - pad), min(float(self.T), hi + pad)
        return np.linspace(lo, hi, self.n_fine)

    @property
    def dt(self) -> float:
        """Sampling cadence [s]."""
        return self._dt

    @dt.setter
    def dt(self, dt: float):
        self._dt = dt
        self.sampling_frequency = 1 / dt

    def __call__(
        self,
        m1: float,
        m2: float,
        a: float,
        p0: float,
        e0: float,
        x0: float,
        dist: float,
        qS: float,
        phiS: float,
        qK: float,
        phiK: float,
        Phi_phi0: float,
        Phi_theta0: float,
        Phi_r0: float,
        *add_args: Optional[tuple],
        include_minus_mkn: bool = True,
        holder=None,
        **kwargs: Optional[dict],
    ):
        """Run FEW (or take ``holder``) and return the TDI-on-the-fly response of its modes.

        Args:
            m1: Primary (MBH) mass [Msun].
            m2: Secondary (compact object) mass [Msun].
            a: Dimensionless spin.
            p0: Initial semi-latus rectum.
            e0: Initial eccentricity.
            x0: Inclination cosine (``+1`` / ``-1``).
            dist: Luminosity distance [Gpc].
            qS: Sky polar angle, ecliptic [rad].
            phiS: Sky azimuth, ecliptic [rad].
            qK: Spin polar angle [rad].
            phiK: Spin azimuth [rad].
            Phi_phi0: Initial azimuthal phase [rad].
            Phi_theta0: Initial polar phase [rad].
            Phi_r0: Initial radial phase [rad].
            *add_args: Ignored.
            include_minus_mkn: Feed the -m partners as separate subs. Default ``True``.
            holder (optional): A precomputed FEW sparse holder on this instance's fine grid
                (needs ``n_fine`` / ``t_fine``); skips the FEW call. Default ``None``.
            **kwargs: Forwarded to the FEW generator (``inspiral_kwargs`` is owned by this
                class when ``n_fine`` is set). The generator's own ``inspiral_kwargs`` are
                restored afterwards.

        Returns:
            TDTDIOutput: The response output with splines (``return_spline=True``), one row
            per sub.

        Raises:
            WaveformDomainError: When FEW refuses the parameters.
            ValueError: If ``inspiral_kwargs`` is passed with ``n_fine`` set, ``holder`` is
                given without a fine grid, or the delay trim leaves fewer than 4 points.
        """
        # (qS, phiS, qK, phiK) are ECLIPTIC polar angles -> the FEW viewing
        # angle and psi always come from them. frame="ecliptic": the sky also
        # stays ecliptic (matching ecliptic orbits). frame="icrs_special": the
        # sky is converted ecliptic -> ICRS for the response, which runs against
        # ICRS orbits (the validated mojito recipe).
        theta, phi = get_viewing_angles(qS, phiS, qK, phiK)
        psi = get_polarization_angle(qS, phiS, qK, phiK)
        if self.frame == "icrs_special":
            lam, beta = ecliptic_to_icrs(phiS, np.pi / 2 - qS)
        else:
            lam = phiS
            beta = np.pi / 2 - qS

        # FEW's T is in YEARS (few/waveform/base.py:173); self.T is seconds.
        T_years = float(self.T) / YRSID_SI
        if self.n_fine is not None:
            if "inspiral_kwargs" in kwargs:
                raise ValueError("EMRITDIonFly(n_fine=...) owns inspiral_kwargs; do not pass them")
            new_t = self._fine_times()
            T_years = max(float(self.T), float(new_t[-1])) / YRSID_SI
            kwargs = dict(kwargs)
            kwargs["inspiral_kwargs"] = {"upsample": True, "fix_t": True, "new_t": new_t}

        # FEW merges call-time inspiral_kwargs into the generator PERMANENTLY
        # (few/waveform/base.py:236); snapshot and restore so a shared generator
        # (e.g. the cached legacy ResponseWrapper's) never inherits new_t/upsample.
        _ik = getattr(self.wave_gen, "inspiral_kwargs", None)
        _ik_saved = dict(_ik) if isinstance(_ik, dict) else None

        # Out-of-domain (a, p0, e0) raises bare ValueError/AssertionError in
        # FEW; re-raise typed so the sampler can score the point at -1e300.
        if holder is not None:
            # a precomputed FEW sparse holder on THIS fine grid (EMRIDirectWDM's one FEW
            # call): no second FEW call
            if self.n_fine is None:
                raise ValueError("EMRITDIonFly(holder=...) needs the fine grid it was made on (t_fine/n_fine)")
            Kerr_wave = holder
        try:
            if holder is not None:
                raise _SkipFew()
            with few_domain_guard():
                Kerr_wave = self.wave_gen(
                    m1,
                    m2,
                    a,
                    p0,
                    e0,
                    x0,
                    theta,
                    phi,
                    dist=dist,
                    Phi_phi0=Phi_phi0,
                    Phi_theta0=Phi_theta0,
                    Phi_r0=Phi_r0,
                    T=T_years,
                    dt=self.dt,
                    return_sparse_holder=True,
                    include_minus_mkn=include_minus_mkn,
                    **kwargs,
                )
        except _SkipFew:
            pass
        finally:
            if _ik_saved is not None:
                _ik.clear()
                _ik.update(_ik_saved)

        t_arr_in, mode_amp, mode_phase, t_arr_tdi = self.prepare_feed(Kerr_wave, include_minus_mkn)
        return self.run_response([(t_arr_in, mode_amp, mode_phase, t_arr_tdi, psi, lam, beta)])

    def sky(self, qS, phiS, qK, phiK):
        """Viewing angles for FEW and the response's sky / polarization, in this instance's frame.

        Args:
            qS (float): Sky polar angle, ecliptic [rad].
            phiS (float): Sky azimuth, ecliptic [rad].
            qK (float): Spin polar angle [rad].
            phiK (float): Spin azimuth [rad].

        Returns:
            tuple: ``(theta, phi, psi, lam, beta)``: FEW's viewing angles and polarization
            (from FEW ``get_viewing_angles`` / ``get_polarization_angle``), and the sky
            longitude / latitude: ecliptic ``(phiS, pi/2 - qS)`` for ``frame="ecliptic"``,
            converted to ICRS (RA, Dec) for ``frame="icrs_special"``.
        """
        theta, phi = get_viewing_angles(qS, phiS, qK, phiK)
        psi = get_polarization_angle(qS, phiS, qK, phiK)
        if self.frame == "icrs_special":
            lam, beta = ecliptic_to_icrs(phiS, np.pi / 2 - qS)
        else:
            lam, beta = phiS, np.pi / 2 - qS
        return theta, phi, psi, lam, beta

    def prepare_feed(self, Kerr_wave, include_minus_mkn=True):
        """FEW holder -> ``(t_arr_in, mode_amp, mode_phase, t_arr_tdi)`` for the response.

        Copies the holder to host (stored as :attr:`last_holder`), builds the per-sub
        amplitude / phase (:meth:`mode_amp_phase` with :attr:`AMP_FACTOR`) and hands them to
        :meth:`prepare_feed_arrays`.

        Args:
            Kerr_wave: FEW sparse holder (host or device).
            include_minus_mkn (bool, optional): Include the -m partners. Default ``True``.

        Returns:
            tuple: See :meth:`prepare_feed_arrays`.
        """
        Kerr_wave = host_holder(Kerr_wave)   # GPU generator: cupy holder -> host copy
        self.last_holder = Kerr_wave   # consumers (EMRIDirectWDM) need the same trajectory's modes
        mode_amp, mode_phase = self.mode_amp_phase(
            Kerr_wave, include_minus_mkn=include_minus_mkn, amp_factor=self.AMP_FACTOR
        )
        return self.prepare_feed_arrays(np.asarray(Kerr_wave.t_arr, dtype=float), mode_amp, mode_phase)

    def prepare_feed_arrays(self, t_src, mode_amp, mode_phase):
        """Response feed arrays from per-sub amplitude / phase on the source times.

        With a fine grid, a trajectory that stops before the requested end is continued past
        the stop with ZERO amplitude and linearly continued phase (two delay margins plus two
        steps). The TDI evaluation grid is then trimmed by ``delay_margin`` at each end (by
        sample count on a uniform feed, by time on a non-uniform one).

        Args:
            t_src (np.ndarray): Source times [s], FEW clock (relative to ``t0``), ``(n,)``.
            mode_amp (np.ndarray): Per-sub amplitudes, ``(S, n)``.
            mode_phase (np.ndarray): Per-sub phases, ``(S, n)``.

        Returns:
            tuple: ``(t_arr_in, mode_amp, mode_phase, t_arr_tdi)``: absolute input times
            ``(S, n')``, the (possibly extended) amplitude and phase, and the trimmed absolute
            evaluation times ``(S, n_tdi)``.

        Raises:
            ValueError: If the trim leaves fewer than 4 evaluation times.
        """
        t_src = np.asarray(t_src, dtype=float)
        if self.n_fine is not None and t_src.size > 2:
            # A plunge inside the requested window: FEW's fix_t cut the fine grid at the
            # trajectory end, and the delay trim below would then drop the last
            # ~delay_margin of signal plus the response's tail after the stop. Continue
            # the feed past the end with ZERO amplitude (the production waveform is
            # zero-padded after the plunge); the phase continues linearly.
            requested_end = float(self._fine_times()[-1])
            sp = float(t_src[-1] - t_src[-2])
            if t_src[-1] < requested_end - 0.5 * sp:
                # one delay_margin is eaten by the trim, the second covers the response's
                # tail after the stop (SSB projection |k.x| <~ 500 s)
                n_ext = int(np.ceil((2.0 * self.delay_margin + 2.0 * sp) / sp)) + 2
                steps = np.arange(1, n_ext + 1, dtype=float)
                t_src = np.concatenate([t_src, t_src[-1] + sp * steps])
                dphi = (mode_phase[:, -1] - mode_phase[:, -2])[:, None]
                mode_phase = np.concatenate([mode_phase, mode_phase[:, -1:] + dphi * steps[None, :]], axis=1)
                mode_amp = np.concatenate([mode_amp, np.zeros((mode_amp.shape[0], n_ext))], axis=1)
        t_arr_in = self.t0 + np.repeat(t_src[:, None], mode_phase.shape[0], axis=-1).T
        # Trim the TDI grid inside the waveform spline by the max response delay so
        # the delayed waveform queries (t - k.x) never fall outside the spline.
        dt_traj = float(t_arr_in[0, 1] - t_arr_in[0, 0]) if t_arr_in.shape[1] > 1 else self.dt
        steps = np.diff(t_src)
        if steps.size and np.allclose(steps, steps[0], rtol=1e-9, atol=0.0):
            n_trim = max(1, int(np.ceil(self.delay_margin / dt_traj)) + 1)
            t_arr_tdi = t_arr_in[:, n_trim:-n_trim]
        else:   # non-uniform feed: trim by TIME (one local step beyond the delay margin)
            keep = ((t_src >= t_src[0] + self.delay_margin + steps[0])
                    & (t_src <= t_src[-1] - self.delay_margin - steps[-1]))
            t_arr_tdi = t_arr_in[:, keep]
        if t_arr_tdi.shape[1] < 4:
            raise ValueError(
                f"EMRITDIonFly: only {t_arr_in.shape[1]} trajectory points; the {self.delay_margin:g} s "
                "delay trim leaves too few for the response grid. Pass n_fine (e.g. span/80 s) "
                "to feed a fine trajectory."
            )
        return t_arr_in, mode_amp, mode_phase, t_arr_tdi

    def run_response(self, feeds):
        """ONE TDI-on-the-fly response for every sub of every feed.

        Builds one :class:`~lisatools.response.tdionfly.TDTDIonTheFly` (stored as
        :attr:`tdi_gen`) with ``inc = 0`` for every sub and the feed's ``psi`` / ``lam`` /
        ``beta`` repeated over its subs. Many templates in one call fill the GPU (the kernel
        launches one block per sub). The response object is built without ``force_backend``
        (its default backend resolution applies).

        Args:
            feeds (list): ``(t_arr_in, mode_amp, mode_phase, t_arr_tdi, psi, lam, beta)`` per
                template (e.g. :meth:`prepare_feed` plus :meth:`sky`); all must share the
                fine-grid lengths.

        Returns:
            TDTDIOutput: The response with splines; its sub axis is the concatenation of the
            feeds' subs in feed order.
        """
        t_in = np.concatenate([f[0] for f in feeds], axis=0)
        amp = np.concatenate([f[1] for f in feeds], axis=0)
        ph = np.concatenate([f[2] for f in feeds], axis=0)
        t_tdi = np.concatenate([f[3] for f in feeds], axis=0)
        num_sub = amp.shape[0]
        self.tdi_gen = TDTDIonTheFly(
            t_tdi,
            amp,
            ph,
            self.dt,
            num_sub,
            t_input=t_in,
            tdi_config=self.tdi_config,
            orbits=self.orbits,
        )
        inc = np.zeros(num_sub)
        psi_in = np.concatenate([np.full(f[1].shape[0], f[4]) for f in feeds])
        lam_in = np.concatenate([np.full(f[1].shape[0], f[5]) for f in feeds])
        beta_in = np.concatenate([np.full(f[1].shape[0], f[6]) for f in feeds])
        return self.tdi_gen(inc, psi_in, lam_in, beta_in, return_spline=True)
