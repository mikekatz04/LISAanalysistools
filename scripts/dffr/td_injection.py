"""Time-domain TDI of a galactic binary through LAT's full response, for the DFFR validation.

The barycentric strain follows ``GBDFFRSource`` and ``LISATDIonTheFly::get_hp_hc``: in the source
frame ``h+ = -A (1 + cos^2 iota) cos(Phi)`` and ``hx = -2 A cos(iota) sin(Phi)``, rotated by twice
the polarisation angle, with the phase and amplitude of ``gb_physics.hh``. ``pyResponseTDI``
projects it on the six links at the retarded times (Lagrange interpolation of order 25) and forms
the TDI combinations with the delays of the same orbits. ``stft_accuracy.py`` and the likelihood
surfaces of step 6 build their time-domain data with it; ``docs/response/dffr/tests.md`` records
what it is checked against.
"""

from __future__ import annotations

import numpy as np

from lisatools.response.directresponse import pyResponseTDI
from lisatools.response.tdiconfig import TDIConfig

PARAMETER_NAMES = ("amp", "f0", "fdot", "fddot", "phi0", "iota", "psi", "lam", "beta")
#: Lagrange order of pyResponseTDI's fractional delays, its default.
LAGRANGE_ORDER = 25
# * Simulated before and after the data: pyResponseTDI needs about 4200 s at 5 s sampling before
# * its first valid sample, and the strain at a spacecraft can lead the barycentre by 500 s.
DEFAULT_MARGIN = 10000.0


def gb_strain(params, times, t_ref, xp=np):
    """Barycentric ``h+ + i hx`` of one galactic binary at ``times`` (seconds, orbit frame).

    ``params`` holds the nine parameters of ``PARAMETER_NAMES``; ``t_ref`` is their epoch.
    """
    amp, f0, fdot, fddot, phi0, iota, psi = (float(value) for value in params[:7])
    t_diff = times - t_ref
    # * The formulas of gb_phase and gb_amplitude_ratio, in their order of operations.
    phase = -phi0 + 2 * np.pi * (
        f0 * t_diff + 0.5 * fdot * t_diff * t_diff + (1.0 / 6.0) * fddot * t_diff * t_diff * t_diff
    )
    amplitude = amp * (1.0 + (2.0 / 3.0) * fdot / f0 * t_diff)
    cos_iota = np.cos(iota)
    plus_source = -amplitude * (1.0 + cos_iota * cos_iota) * xp.cos(phase)
    cross_source = -2.0 * amplitude * cos_iota * xp.sin(phase)
    cos_two_psi, sin_two_psi = np.cos(2.0 * psi), np.sin(2.0 * psi)
    h_plus = plus_source * cos_two_psi - cross_source * sin_two_psi
    h_cross = plus_source * sin_two_psi + cross_source * cos_two_psi
    return h_plus + 1j * h_cross


class TimeDomainTDI:
    """X, Y, Z of one galactic binary at a time, on ``num_samples`` samples from ``data_start``.

    Args:
        orbits: LAT orbits, on the backend of ``force_backend``; times count in their frame.
        tdi_generation: ``"1st generation"`` or ``"2nd generation"``.
        data_start: time of the first output sample [s].
        num_samples: number of output samples.
        sample_interval: sampling interval [s].
        margin: time simulated before and after the output [s], a whole number of samples.
        force_backend: ``"cpu"`` or a CUDA backend name.
    """

    def __init__(
        self,
        orbits,
        tdi_generation,
        data_start,
        num_samples,
        sample_interval,
        margin=DEFAULT_MARGIN,
        force_backend="cpu",
    ):
        self.margin_samples = int(round(margin / sample_interval))
        if abs(self.margin_samples * sample_interval - margin) > 1e-9 * margin:
            raise ValueError("margin must be a whole number of samples")
        self.data_start = float(data_start)
        self.num_samples = int(num_samples)
        self.sample_interval = float(sample_interval)
        self.first_time = self.data_start - self.margin_samples * self.sample_interval
        total = self.num_samples + 2 * self.margin_samples
        self.response = pyResponseTDI(
            1.0 / self.sample_interval,
            total,
            order=LAGRANGE_ORDER,
            tdi=TDIConfig(tdi_generation, force_backend=force_backend),
            orbits=orbits,
            tdi_chan="XYZ",
            force_backend=force_backend,
        )
        if self.response.num_pts != total:
            raise ValueError("the simulated span runs past the end of the orbit file")
        self.xp = self.response.xp
        self.times = self.first_time + self.xp.arange(total) * self.sample_interval

    def __call__(self, params, t_ref):
        """Array (3, num_samples) of X, Y, Z for the nine parameters ``params``."""
        strain = gb_strain(params, self.times, t_ref, self.xp)
        self.response.get_projections(
            strain,
            float(params[7]),
            float(params[8]),
            t0=self.first_time,
            t_buffer=self.margin_samples * self.sample_interval,
        )
        channels = self.response.get_tdi_delays()
        first = self.margin_samples
        return self.xp.stack([channel[first : first + self.num_samples] for channel in channels])
