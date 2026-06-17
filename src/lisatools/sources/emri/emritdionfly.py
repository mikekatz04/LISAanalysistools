"""Time-domain EMRI TDI-on-the-fly waveform generator.

The EMRI analogue of :class:`bbhx.sobbhtdionfly.SOBBHTDIonFly` /
:class:`bbhx.mbhtdionfly.MBHTDIonFly`: take a FEW sparse-mode holder, reconstruct
per-``(l, m, k, n)``-mode amplitude/phase, and feed each mode (plus its ``-m``
partner) to the generic :class:`~lisatools.response.tdionfly.TDTDIonTheFly`
response.

Frame convention
----------------
FEW takes its sky/spin angles ``(qS, phiS, qK, phiK)`` as **ecliptic** polar
angles, so the viewing angle, polarization and sky position are all produced in
the **ecliptic** frame and handed to the response in that frame -- meaning the
``orbits`` MUST be loaded with ``frame="ecliptic"``.  This mirrors the way
``SOBBHTDIonFly`` / ``MBHTDIonFly`` consume ``(ra, dec)`` with ``frame="icrs"``:
the sky/polarization frame and the orbits frame must agree.  (The previous
version converted only the sky to ICRS while leaving ``psi`` ecliptic, a mixed
frame that did not match the orbits.)

Per-mode inclination
--------------------
Each mode is fed with ``inc = 0``; the response kernel's ``(1 + cos^2 i)`` /
``2 cos i`` inclination factors then reduce to ``2``, which ``AMP_FACTOR = 1/2``
cancels.  The viewing-angle dependence is already baked into the FEW
spin-weighted harmonics ``Y_{lm}(theta)``, which differ for ``+m`` vs ``-m`` --
hence the two are fed as separate subs.
"""
from typing import Optional

import numpy as np

from few.utils.utility import get_polarization_angle, get_viewing_angles

from lisatools.response.tdionfly import TDTDIonTheFly


class EMRITDIonFly:
    """Build the TDI response of a FEW EMRI waveform mode-by-mode on the fly.

    Args:
        wave_gen: a FEW waveform generator returning a sparse-mode holder when
            called with ``return_sparse_holder=True`` (e.g.
            :class:`few.waveform.FastKerrEccentricEquatorialFlux`).
        orbits: LISA orbits **loaded with** ``frame="ecliptic"`` (see the frame
            convention in the module docstring).
        tdi_config: a :class:`~lisatools.response.tdiconfig.TDIConfig`.
        dt: sampling cadence [s].
        Tobs: observation time [s].
        t0: absolute epoch [s] of the trajectory start (the FEW initial
            conditions are defined at ``t0``).
        delay_margin: seconds to trim the TDI evaluation grid inside the
            waveform spline at each end, so every delayed waveform query
            (``t - k.x`` with the SSB projection ``|k.x| <~ 1 AU / c ~ 500 s``,
            plus the TDI arm delays) stays inside the spline. The default 600 s
            covers the LISA geometry; a 1-sample trim is NOT enough.
    """

    def __init__(self, wave_gen, orbits, tdi_config, dt, Tobs, t0, delay_margin=600.0):
        self.wave_gen = wave_gen
        self.orbits = orbits
        self.tdi_config = tdi_config
        self.dt = dt
        self.T = Tobs
        self.t0 = t0
        self.delay_margin = delay_margin

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
        **kwargs: Optional[dict],
    ):
        # (qS, phiS, qK, phiK) are ECLIPTIC polar angles -> viewing angle,
        # polarization and sky position all stay in the ecliptic frame, matching
        # the (ecliptic) orbits.  No sky->ICRS conversion (that mixed frames).
        theta, phi = get_viewing_angles(qS, phiS, qK, phiK)
        psi = get_polarization_angle(qS, phiS, qK, phiK)
        lam = phiS
        beta = np.pi / 2 - qS

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
            T=self.T,
            dt=self.dt,
            return_sparse_holder=True,
            include_minus_mkn=include_minus_mkn,
            **kwargs,
        )

        mode_amp_phase = np.unwrap(np.angle(Kerr_wave.teuk_modes), axis=0)
        mode_amp_amp = np.abs(Kerr_wave.teuk_modes)

        ylm_phase = np.angle(Kerr_wave.ylms)
        ylm_amp = np.abs(Kerr_wave.ylms)
        _mode_phase = (
            Kerr_wave.ms[None, :] * Kerr_wave.phases[:, 0][:, None]
            + Kerr_wave.ks[None, :] * Kerr_wave.phases[:, 1][:, None]
            + Kerr_wave.ns[None, :] * Kerr_wave.phases[:, 2][:, None]
        )

        AMP_FACTOR = 1 / 2.0  # cancels the inc=0 kernel factor (1 + cos^2 0) = 2
        if include_minus_mkn:
            # m >= 0 fed directly; the -m partner fed with the negated phase and
            # its own Y_{l,-m} amplitude (different SWSH from +m).
            keep_minus_m = Kerr_wave.ms != 0
            phase_m_zero_and_above = _mode_phase - ylm_phase[: Kerr_wave.ms.shape[0]] - mode_amp_phase
            phase_m_below_zero = -phase_m_zero_and_above[:, keep_minus_m]
            mode_phase = np.concatenate([phase_m_zero_and_above, phase_m_below_zero], axis=-1).T

            amp_m_zero_and_above = AMP_FACTOR * mode_amp_amp * ylm_amp[: _mode_phase.shape[1]]
            amp_m_below_zero = (AMP_FACTOR * mode_amp_amp * ylm_amp[_mode_phase.shape[1]:])[:, keep_minus_m]
            mode_amp = np.concatenate([amp_m_zero_and_above, amp_m_below_zero], axis=-1).T
        else:
            mode_phase = (_mode_phase - ylm_phase[: Kerr_wave.ms.shape[0]] - mode_amp_phase).T
            mode_amp = AMP_FACTOR * (mode_amp_amp * ylm_amp[: Kerr_wave.ms.shape[0]]).T

        t_arr_in = self.t0 + np.repeat(Kerr_wave.t_arr[:, None], mode_phase.shape[0], axis=-1).T
        # Trim the TDI grid inside the waveform spline by the max response delay so
        # the delayed waveform queries (t - k.x) never fall outside the spline.
        dt_traj = float(t_arr_in[0, 1] - t_arr_in[0, 0]) if t_arr_in.shape[1] > 1 else self.dt
        n_trim = max(1, int(np.ceil(self.delay_margin / dt_traj)) + 1)
        t_arr_tdi = t_arr_in[:, n_trim:-n_trim]
        num_sub = mode_amp.shape[0]

        self.tdi_gen = TDTDIonTheFly(
            t_arr_tdi,
            mode_amp,
            mode_phase,
            1.0,
            num_sub,
            t_input=t_arr_in,
            tdi_config=self.tdi_config,
            orbits=self.orbits,
        )

        inc = np.zeros(num_sub)
        psi_in = np.full(num_sub, psi)
        lam_in = np.full(num_sub, lam)
        beta_in = np.full(num_sub, beta)
        # sky + polarization in the ECLIPTIC frame, matching frame="ecliptic" orbits.
        return self.tdi_gen(inc, psi_in, lam_in, beta_in, return_spline=True)
