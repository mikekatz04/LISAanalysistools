"""Tests of the source-independent DFFR stages of cutils/lat_dffr_kernels.hh.

Three diagnostic methods of the backend reach them: ``STFTFresnelWrap.compute_dffr_sub_intervals``
and ``compute_dffr_windowed_kernels``, and ``KinematicsTableWrap.compute_dffr_link_coefficients``.

* The bounded sub-interval integral and its first moment against adaptive quadrature, on the cases
  and with the bounds of the standalone DFFR package's ``tests/test_fresnel.py``.
* The windowed transform and its first moment against quadrature of the window times the chirp,
  for a boxcar, the production Tukey taper and a wide taper.
* The closed-form TDI coefficients of both generations against the units of ``TDIConfig``, and
  ``TDIConfig.generation``.
* Argument checks, and the same tests on the CUDA backend with the CUDA values against the CPU
  values (skipped without a usable GPU).
"""

from __future__ import annotations

import functools
import unittest
import warnings

import numpy as np
import scipy.integrate

import lisatools
from lisatools.domaincomputation import KINEMATICS_TABLE_FIELDS
from lisatools.response.tdiconfig import TDI_SECOND_GENERATION, TDIConfig

#: Link labels in the order of every lisatools orbit class; link ij is received by i, sent by j.
LINKS = (12, 23, 31, 13, 32, 21)
EPS = np.finfo(np.float64).eps

#: The segment of DFFR's tests: one day about its midpoint, with one-day frequency bins.
HALF_DURATION = 43200.0
SEGMENT_DURATION = 2.0 * HALF_DURATION
FREQUENCY_BIN = 1.0 / SEGMENT_DURATION
#: The sign-preserving floor on the chirp rate, DFFR_CHIRP_RATE_FLOOR [Hz/s].
CHIRP_RATE_FLOOR = 1.0e-22

#: (chirp rate [Hz/s], detuning [Hz]): Doppler-dominated rates of both signs, a small rate and the
#: chirp-rate floor; on the carrier, within a bin of it and far from it.
SUB_INTERVAL_CASES = tuple(
    (chirp_rate, detuning)
    for chirp_rate in (4.0e-13, -4.0e-13, 1.0e-20, -1.0e-22)
    for detuning in (0.0, 0.3 * FREQUENCY_BIN, 7.0 * FREQUENCY_BIN, -18.5 * FREQUENCY_BIN)
)
# ! DFFR's bounds, on int exp(i psi) dtau as a fraction of 2 HALF_DURATION and on its moment as a
# ! fraction of HALF_DURATION^2. The rational fits of lat_fresnel.hh (about 2e-10) set them; the
# ! moment reaches 2.2e-8 within a quarter bin of the carrier at |alpha| = 4e-13.
KERNEL_BOUND = 2.0e-10
MOMENT_BOUND = 5.0e-8

#: Window tapers: a boxcar, the production taper of one-day segments and a wide taper.
WINDOW_TAPERS = (0.0, 0.1157, 0.5)
#: Carrier and chirp rates of the windowed cases: Doppler-dominated rates of both signs, a small
#: rate and the floor, with the carrier 0.37 bins off the grid as in DFFR's test.
WINDOWED_CARRIER = 0.02 + 0.37 * FREQUENCY_BIN
WINDOWED_CHIRP_RATES = (4.0e-13, -2.5e-13, 1.0e-20, -1.0e-22)
#: Bins of the windowed cases relative to the grid bin at 0.02 Hz: the band of production and two
#: bins outside it on either side.
WINDOWED_BINS = (-15, -12, -7, -2, -1, 0, 1, 2, 7, 12, 15)
# ? The windowed transform sums six sub-intervals with window weights of total size 2 (0.5, 0.5
# ? and four of 0.25), so its bounds are twice DFFR's bounds of one sub-interval, in the same
# ? normalisations.
WINDOWED_KERNEL_BOUND = 2.0 * KERNEL_BOUND
WINDOWED_MOMENT_BOUND = 2.0 * MOMENT_BOUND

# ? Absolute bound on the closed-form coefficients against the unit sums of TDIConfig. Each unit is
# ? a product of up to seven unit phasors (about 13 roundings, phasors from different libm
# ? implementations included) and a coefficient sums up to eight units of either form: about
# ? 3e-14 together, for coefficients of order one to four.
LINK_COEFFICIENT_BOUND = 1.0e-13

# * CUDA against CPU, for the transforms (of the normalisations above) and the coefficients: ten
# * times the 1.0e-15 that an A100 gave (2026-10-19; the transforms at most 3.8e-16).
GPU_AGAINST_CPU_BOUND = 1.0e-14


def _cuda_backend_name():
    """Name of a usable CUDA backend, or None."""
    try:
        import cupy

        if cupy.cuda.runtime.getDeviceCount() < 1:
            return None
        for name in ("cuda13x", "cuda12x", "cuda11x"):
            if lisatools.has_backend(name):
                cupy.zeros(1)  # * a device that cannot allocate is not usable
                return name
    except Exception:
        return None
    return None


def _to_host(array):
    return array.get() if hasattr(array, "get") else np.asarray(array)


def _fresnel_wrap(backend, window_taper: float = 0.0, segment_duration: float = SEGMENT_DURATION):
    """An STFTFresnelWrap of one segment; the diagnostics read its window and segment only."""
    return backend.STFTFresnelWrap(
        1,
        1,
        3,
        0.0,
        0.0,
        1.0,
        segment_duration,
        FREQUENCY_BIN,
        window_alpha=window_taper,
    )


# ---------------------------------------------------------------------------------------------
# Evaluations on a backend
# ---------------------------------------------------------------------------------------------


def sub_intervals(backend, cases=SUB_INTERVAL_CASES):
    """Kernel and moment of each case over [-HALF_DURATION, HALF_DURATION], host arrays."""
    xp = backend.xp
    num = len(cases)
    kernel = xp.zeros(num, dtype=np.complex128)
    moment = xp.zeros(num, dtype=np.complex128)
    _fresnel_wrap(backend).compute_dffr_sub_intervals(
        kernel,
        moment,
        xp.full(num, -HALF_DURATION),
        xp.full(num, HALF_DURATION),
        xp.asarray([detuning for _, detuning in cases]),
        xp.asarray([chirp_rate for chirp_rate, _ in cases]),
        CHIRP_RATE_FLOOR,
        num,
    )
    return _to_host(kernel), _to_host(moment)


def windowed_cases():
    """(taper, chirp rate, frequency) of every windowed case."""
    frequencies = [0.02 + bin_offset * FREQUENCY_BIN for bin_offset in WINDOWED_BINS]
    return tuple(
        (window_taper, chirp_rate, frequency)
        for window_taper in WINDOW_TAPERS
        for chirp_rate in WINDOWED_CHIRP_RATES
        for frequency in frequencies
    )


def windowed_kernels(backend):
    """Kernel and moment of every windowed case, host arrays in the order of windowed_cases."""
    xp = backend.xp
    kernels, moments = [], []
    for window_taper in WINDOW_TAPERS:
        cases = [case for case in windowed_cases() if case[0] == window_taper]
        num = len(cases)
        kernel = xp.zeros(num, dtype=np.complex128)
        moment = xp.zeros(num, dtype=np.complex128)
        _fresnel_wrap(backend, window_taper).compute_dffr_windowed_kernels(
            kernel,
            moment,
            xp.full(num, WINDOWED_CARRIER),
            xp.asarray([chirp_rate for _, chirp_rate, _ in cases]),
            xp.asarray([frequency for _, _, frequency in cases]),
            CHIRP_RATE_FLOOR,
            num,
        )
        kernels.append(_to_host(kernel))
        moments.append(_to_host(moment))
    return np.concatenate(kernels), np.concatenate(moments)


def light_travel_times(num_segments: int) -> np.ndarray:
    """(6, NT) light travel times of ESA-like arms, different per link and segment [s]."""
    rng = np.random.default_rng(23)
    return 8.33 + 0.03 * rng.uniform(-1.0, 1.0, size=(6, num_segments))


def carrier_frequencies(num_segments: int) -> np.ndarray:
    """Carriers from 0.1 to 50 mHz, where the delays turn the round trips by up to 5 rad."""
    return np.geomspace(1.0e-4, 5.0e-2, num_segments)


def link_coefficients(backend, ltt: np.ndarray, carriers: np.ndarray, generation: int):
    """(NT, 3, 6) closed-form coefficients of DFFR at the given delays and carriers."""
    xp = backend.xp
    num_segments = ltt.shape[1]
    arrays = {
        name: xp.zeros(int(np.prod(shape)) * num_segments)
        for name, shape in KINEMATICS_TABLE_FIELDS
    }
    arrays["light_travel_time"] = xp.ascontiguousarray(xp.asarray(ltt.reshape(-1)))
    table = backend.KinematicsTableWrap(
        HALF_DURATION, SEGMENT_DURATION, num_segments, *arrays.values()
    )
    coefficients = xp.zeros(num_segments * 18, dtype=np.complex128)
    table.compute_dffr_link_coefficients(coefficients, xp.asarray(carriers), generation)
    return _to_host(coefficients).reshape(num_segments, 3, 6)


# ---------------------------------------------------------------------------------------------
# Reference values
# ---------------------------------------------------------------------------------------------


def quadrature(lower_bound, upper_bound, detuning, chirp_rate, power, weight=None):
    """int tau^power w(tau) exp(i [2 pi detuning tau + pi alpha tau^2]) by adaptive quadrature.

    ! quad warns that rounding may limit the requested 1e-13 of the larger integrals; its own
    ! error estimates stay far below the bounds, so the warning is muted.
    """
    if weight is None:
        weight = lambda tau: 1.0  # noqa: E731

    def phase(tau):
        return 2.0 * np.pi * detuning * tau + np.pi * chirp_rate * tau * tau

    options = dict(limit=4000, epsabs=0.0, epsrel=1.0e-13)
    with warnings.catch_warnings():
        warnings.simplefilter("ignore", scipy.integrate.IntegrationWarning)
        real = scipy.integrate.quad(
            lambda tau: tau**power * weight(tau) * np.cos(phase(tau)),
            lower_bound,
            upper_bound,
            **options,
        )[0]
        imaginary = scipy.integrate.quad(
            lambda tau: tau**power * weight(tau) * np.sin(phase(tau)),
            lower_bound,
            upper_bound,
            **options,
        )[0]
    return real + 1j * imaginary


def windowed_quadrature(window_taper, chirp_rate, frequency, power):
    """The windowed transform of the case by quadrature, piece by piece of the window.

    The periodic Tukey window is one on the plateau |tau| <= tau_0 = (1 - rho) T / 2 and
    (1 + cos(2 pi (|tau| - tau_0) / (rho T))) / 2 on the tapers; the Fourier origin at the segment
    start contributes exp(-i pi f T).
    """
    detuning = WINDOWED_CARRIER - frequency
    chirp_rate = np.sign(chirp_rate) * max(abs(chirp_rate), CHIRP_RATE_FLOOR)
    if window_taper <= 0.0:
        integral = quadrature(-HALF_DURATION, HALF_DURATION, detuning, chirp_rate, power)
    else:
        plateau = HALF_DURATION * (1.0 - window_taper)
        taper_frequency = 1.0 / (window_taper * SEGMENT_DURATION)

        def taper(tau):
            return 0.5 * (1.0 + np.cos(2.0 * np.pi * taper_frequency * (abs(tau) - plateau)))

        integral = (
            quadrature(-HALF_DURATION, -plateau, detuning, chirp_rate, power, taper)
            + quadrature(-plateau, plateau, detuning, chirp_rate, power)
            + quadrature(plateau, HALF_DURATION, detuning, chirp_rate, power, taper)
        )
    return np.exp(-1j * np.pi * frequency * SEGMENT_DURATION) * integral


@functools.lru_cache(maxsize=None)
def sub_interval_reference():
    kernels, moments = [], []
    for chirp_rate, detuning in SUB_INTERVAL_CASES:
        floored = np.sign(chirp_rate) * max(abs(chirp_rate), CHIRP_RATE_FLOOR)
        kernels.append(quadrature(-HALF_DURATION, HALF_DURATION, detuning, floored, 0))
        moments.append(quadrature(-HALF_DURATION, HALF_DURATION, detuning, floored, 1))
    return np.array(kernels), np.array(moments)


@functools.lru_cache(maxsize=None)
def windowed_reference():
    cases = windowed_cases()
    kernels = np.array([windowed_quadrature(*case, power=0) for case in cases])
    moments = np.array([windowed_quadrature(*case, power=1) for case in cases])
    return kernels, moments


def unit_sum_coefficients(generation: int, ltt: np.ndarray, carriers: np.ndarray) -> np.ndarray:
    """(NT, 3, 6) coefficients from the units of TDIConfig, one delay phasor per link label.

    Each unit adds sign * prod_d exp(-2 pi i f L_d) to its channel at its base link, with the
    delays and the cyclic permutations to Y and Z exactly as TDIConfig sets them up.
    """
    config = TDIConfig(f"{'1st' if generation == 1 else '2nd'} generation", force_backend="cpu")
    base_links = _to_host(config.tdi_base_links)
    delays = _to_host(config.tdi_link_combinations)
    signs = _to_host(config.tdi_signs)
    channels = _to_host(config.channels)
    operation_index = _to_host(config.tdi_operation_index)

    # * The phase rounds as in the kernel, (-2 pi f) L; cos and sin may differ by an ulp.
    phase = -2.0 * np.pi * carriers[None, :] * ltt
    phasors = np.cos(phase) + 1j * np.sin(phase)  # (6, NT)
    link_position = {label: index for index, label in enumerate(LINKS)}
    coefficients = np.zeros((ltt.shape[1], 3, 6), dtype=np.complex128)
    for unit in range(base_links.size):
        product = np.full(ltt.shape[1], float(signs[unit]), dtype=np.complex128)
        for delay in delays[operation_index == unit]:
            if delay != -11:  # ? -11 marks a unit without delays
                product = product * phasors[link_position[int(delay)]]
        coefficients[:, int(channels[unit]), link_position[int(base_links[unit])]] += product
    return coefficients


# ---------------------------------------------------------------------------------------------
# Measured errors, for the tests and a numbers script
# ---------------------------------------------------------------------------------------------


def sub_interval_errors(backend) -> list:
    """Per case: (chirp rate, detuning, kernel error, moment error) in DFFR's normalisations."""
    kernel, moment = sub_intervals(backend)
    reference_kernel, reference_moment = sub_interval_reference()
    rows = []
    for index, (chirp_rate, detuning) in enumerate(SUB_INTERVAL_CASES):
        kernel_error = abs(kernel[index] - reference_kernel[index]) / SEGMENT_DURATION
        moment_error = abs(moment[index] - reference_moment[index]) / HALF_DURATION**2
        rows.append((chirp_rate, detuning, float(kernel_error), float(moment_error)))
    return rows


def windowed_kernel_errors(backend) -> dict:
    """{taper: (largest kernel error, largest moment error)} in the same normalisations."""
    kernel, moment = windowed_kernels(backend)
    reference_kernel, reference_moment = windowed_reference()
    tapers = np.array([case[0] for case in windowed_cases()])
    errors = {}
    for window_taper in WINDOW_TAPERS:
        mask = tapers == window_taper
        kernel_error = np.abs(kernel[mask] - reference_kernel[mask]).max() / SEGMENT_DURATION
        moment_error = np.abs(moment[mask] - reference_moment[mask]).max() / HALF_DURATION**2
        errors[window_taper] = (float(kernel_error), float(moment_error))
    return errors


def link_coefficient_errors(backend, num_segments: int = 64) -> dict:
    """{generation: largest absolute difference from the unit sums of TDIConfig}."""
    ltt = light_travel_times(num_segments)
    carriers = carrier_frequencies(num_segments)
    errors = {}
    for generation in (1, 2):
        closed_form = link_coefficients(backend, ltt, carriers, generation)
        reference = unit_sum_coefficients(generation, ltt, carriers)
        errors[generation] = float(np.abs(closed_form - reference).max())
    return errors


def gpu_against_cpu_differences(gpu_backend) -> dict:
    """{name: largest difference}, the transforms in the normalisations of the bounds."""
    cpu_backend = lisatools.get_backend("cpu")
    differences = {}
    gpu_kernel, gpu_moment = sub_intervals(gpu_backend)
    cpu_kernel, cpu_moment = sub_intervals(cpu_backend)
    differences["sub_interval_kernel"] = float(
        np.abs(gpu_kernel - cpu_kernel).max() / SEGMENT_DURATION
    )
    differences["sub_interval_moment"] = float(
        np.abs(gpu_moment - cpu_moment).max() / HALF_DURATION**2
    )
    gpu_kernel, gpu_moment = windowed_kernels(gpu_backend)
    cpu_kernel, cpu_moment = windowed_kernels(cpu_backend)
    differences["windowed_kernel"] = float(np.abs(gpu_kernel - cpu_kernel).max() / SEGMENT_DURATION)
    differences["windowed_moment"] = float(np.abs(gpu_moment - cpu_moment).max() / HALF_DURATION**2)
    ltt = light_travel_times(64)
    carriers = carrier_frequencies(64)
    for generation in (1, 2):
        differences[f"link_coefficients_{generation}"] = float(
            np.abs(
                link_coefficients(gpu_backend, ltt, carriers, generation)
                - link_coefficients(cpu_backend, ltt, carriers, generation)
            ).max()
        )
    return differences


# ---------------------------------------------------------------------------------------------
# Tests
# ---------------------------------------------------------------------------------------------


class DFFRStagesTest(unittest.TestCase):
    """The source-independent DFFR stages against reference values, CPU backend."""

    backend_name = "cpu"

    @classmethod
    def setUpClass(cls):
        cls.backend = lisatools.get_backend(cls.backend_name)

    def test_sub_interval_matches_quadrature(self):
        """The bounded integral and its moment on DFFR's cases, with DFFR's bounds."""
        for chirp_rate, detuning, kernel_error, moment_error in sub_interval_errors(self.backend):
            with self.subTest(chirp_rate=chirp_rate, detuning_bins=detuning / FREQUENCY_BIN):
                self.assertLess(kernel_error, KERNEL_BOUND, msg=f"kernel: {kernel_error:.3e}")
                self.assertLess(moment_error, MOMENT_BOUND, msg=f"moment: {moment_error:.3e}")

    def test_windowed_transform_matches_quadrature(self):
        """The windowed transform and its moment for three tapers, against quadrature."""
        for window_taper, (kernel_error, moment_error) in windowed_kernel_errors(
            self.backend
        ).items():
            with self.subTest(window_taper=window_taper):
                self.assertLess(
                    kernel_error, WINDOWED_KERNEL_BOUND, msg=f"kernel: {kernel_error:.3e}"
                )
                self.assertLess(
                    moment_error, WINDOWED_MOMENT_BOUND, msg=f"moment: {moment_error:.3e}"
                )

    def test_closed_forms_match_tdiconfig(self):
        """Both closed forms against the unit sums of TDIConfig, to rounding."""
        for generation, error in link_coefficient_errors(self.backend).items():
            with self.subTest(generation=generation):
                self.assertLessEqual(error, LINK_COEFFICIENT_BOUND, msg=f"{error:.3e}")

    def test_arguments_are_checked(self):
        """Wrong generations, floors, segments and lengths raise ValueError."""
        xp = self.backend.xp
        ltt = light_travel_times(4)
        carriers = carrier_frequencies(4)
        with self.assertRaises(ValueError):
            link_coefficients(self.backend, ltt, carriers, 3)
        arrays = {
            name: xp.zeros(int(np.prod(shape)) * 4) for name, shape in KINEMATICS_TABLE_FIELDS
        }
        table = self.backend.KinematicsTableWrap(
            HALF_DURATION, SEGMENT_DURATION, 4, *arrays.values()
        )
        with self.assertRaises(ValueError):
            table.compute_dffr_link_coefficients(
                xp.zeros(17 * 4, dtype=np.complex128), xp.asarray(carriers), 2
            )
        one = xp.zeros(1, dtype=np.complex128)
        with self.assertRaises(ValueError):
            _fresnel_wrap(self.backend).compute_dffr_sub_intervals(
                one, one, xp.zeros(1), xp.ones(1), xp.zeros(1), xp.ones(1), 0.0, 1
            )
        # ! df * dt must be one: a segment of two days with one-day bins is refused.
        with self.assertRaises(ValueError):
            _fresnel_wrap(
                self.backend, segment_duration=2.0 * SEGMENT_DURATION
            ).compute_dffr_windowed_kernels(
                one, one, xp.zeros(1), xp.ones(1), xp.zeros(1), CHIRP_RATE_FLOOR, 1
            )


class TDIGenerationTest(unittest.TestCase):
    """TDIConfig.generation names the stock combinations and nothing else."""

    def test_stock_and_custom_combinations(self):
        self.assertEqual(TDIConfig("1st generation", force_backend="cpu").generation, 1)
        self.assertEqual(TDIConfig("2nd generation", force_backend="cpu").generation, 2)
        # * The order of the units does not change the combination, nor does an explicit type.
        reordered = [dict(unit, type="delay") for unit in reversed(TDI_SECOND_GENERATION)]
        self.assertEqual(TDIConfig(reordered, force_backend="cpu").generation, 2)
        self.assertIsNone(TDIConfig(TDI_SECOND_GENERATION[:-1], force_backend="cpu").generation)


@unittest.skipUnless(_cuda_backend_name(), "requires a usable CUDA backend")
class DFFRStagesGPUTest(DFFRStagesTest):
    """The same tests on the CUDA backend, and the CUDA values against the CPU values."""

    backend_name = _cuda_backend_name() or "cpu"

    def test_gpu_matches_cpu(self):
        """The transforms and the coefficients agree within GPU_AGAINST_CPU_BOUND."""
        for name, difference in gpu_against_cpu_differences(self.backend).items():
            with self.subTest(quantity=name):
                self.assertLessEqual(difference, GPU_AGAINST_CPU_BOUND, msg=f"{difference:.3e}")


if __name__ == "__main__":
    unittest.main()
