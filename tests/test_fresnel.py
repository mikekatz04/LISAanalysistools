"""Tests of the Fresnel functions of cutils/lat_fresnel.hh, shared by STFTFresnel and DFFR.

Two diagnostic methods of the backend's ``STFTFresnelWrap`` reach the functions:
``compute_fresnel_functions`` returns C, S, the auxiliary functions f and g and the derivatives of
f and g, through the two STFTFresnel members that delegate to the header, and
``compute_phase_kernel_moments`` returns the single-interval phase kernel and its first moment,
which call the series and the auxiliary functions of the header directly.

* C and S against ``scipy.special.fresnel`` for |x| <= 1000, one bound per branch, and their odd
  symmetry.
* f, g and their derivatives against quadrature of the integral representations of f and g
  (DLMF 7.7), from the seam |x| = 1.6 to 1e8; the phase kernels reach about 3e7 at the chirp-rate
  floor of 1e-22 Hz/s.
* The phase kernel and its moment against adaptive quadrature, on the cases and with the bounds of
  the standalone DFFR package's ``tests/test_fresnel.py``.
* The same tests on the CUDA backend, and the CUDA backend against the CPU backend (skipped
  without a usable GPU).
"""

from __future__ import annotations

import functools
import unittest
import warnings

import numpy as np
import scipy.integrate
import scipy.special

import lisatools

#: Seams of lat_fresnel.hh: the series below the first, the rational fits up to the second.
SERIES_SEAM = 1.6
ASYMPTOTIC_SEAM = 8.0
EPS = np.finfo(np.float64).eps

#: Outputs of compute_fresnel_functions, in the order of its arguments.
FUNCTION_NAMES = ("cosine", "sine", "aux_f", "aux_g", "slope_f", "slope_g")
AUXILIARY_NAMES = ("aux_f", "aux_g", "slope_f", "slope_g")

# ? Bounds on the absolute errors of (C, S) per branch. Series: a few roundings of values below
# ? one (about 4e-16 measured). Rational fits: their error in C and S, 1.6e-10 and 7.0e-11, at the
# ? seam. Asymptotic series: rounding of the phase pi x^2 / 2, about |x| eps / 2 in C and S, which
# ? is 1.1e-13 at |x| = 1000.
INTEGRAL_BOUNDS = {
    "series": (1e-15, 1e-15),
    "rational": (2e-10, 1e-10),
    "asymptotic": (2e-13, 2e-13),
}
# ? Bounds on f, g and their derivatives per branch. Rational fits: their relative errors at the
# ? seam, 8.6e-10 for f and 2.2e-9 for g and f'; g' falls to -7.5e-5 near |x| = 8 while the fit
# ? error does not, so its bound there is absolute, above the 8.3e-10 at the seam. Asymptotic
# ? series: the truncation at |x| = 8, 6e-15 for f, 1.3e-13 for g and f' and 9.6e-13 for g'.
AUXILIARY_BOUNDS = {
    ("rational", "aux_f"): (1e-9, "relative"),
    ("rational", "aux_g"): (2.5e-9, "relative"),
    ("rational", "slope_f"): (2.5e-9, "relative"),
    ("rational", "slope_g"): (1e-9, "absolute"),
    ("asymptotic", "aux_f"): (1e-14, "relative"),
    ("asymptotic", "aux_g"): (2.5e-13, "relative"),
    ("asymptotic", "slope_f"): (2.5e-13, "relative"),
    ("asymptotic", "slope_g"): (2e-12, "relative"),
}

#: The segment of DFFR's test: one day about its midpoint, with one-day frequency bins.
HALF_DURATION = 43200.0
FREQUENCY_BIN = 1.0 / 86400.0
#: (chirp rate [Hz/s], detuning [Hz]): Doppler-dominated rates of both signs, a small rate and the
#: chirp-rate floor; on the carrier, within a bin of it and far from it.
PHASE_KERNEL_CASES = tuple(
    (chirp_rate, detuning)
    for chirp_rate in (4.0e-13, -4.0e-13, 1.0e-20, -1.0e-22)
    for detuning in (0.0, 0.3 * FREQUENCY_BIN, 7.0 * FREQUENCY_BIN, -18.5 * FREQUENCY_BIN)
)
# ! DFFR's bounds, on int exp(i psi) dtau as a fraction of 2 HALF_DURATION and on its moment as a
# ! fraction of HALF_DURATION^2. The rational fits (about 2e-10) set them; the moment reaches 2.2e-8
# ! within a quarter bin of the carrier at |alpha| = 4e-13.
KERNEL_BOUND = 2.0e-10
MOMENT_BOUND = 5.0e-8

# * CUDA against CPU: eight roundings, absolute for C, S and g' (values below one) and relative for
# * f, g and f'. The phase kernel and its moment get 1e-14 of the DFFR normalisations, ten times
# * the 3.0e-16 that an A100 gave (2026-10-19); the functions measured at most 2.2e-16 there.
GPU_FUNCTION_BOUNDS = {
    "cosine": (8 * EPS, "absolute"),
    "sine": (8 * EPS, "absolute"),
    "aux_f": (8 * EPS, "relative"),
    "aux_g": (8 * EPS, "relative"),
    "slope_f": (8 * EPS, "relative"),
    "slope_g": (8 * EPS, "absolute"),
}
GPU_PHASE_KERNEL_BOUND = 1e-14


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


def _fresnel_wrap(backend):
    """An STFTFresnelWrap; the two diagnostics read none of its grid or window settings."""
    return backend.STFTFresnelWrap(1, 1, 3, 0.0, 0.0, 1.0, 2.0 * HALF_DURATION, FREQUENCY_BIN)


# ---------------------------------------------------------------------------------------------
# Evaluations on a backend
# ---------------------------------------------------------------------------------------------


def fresnel_functions(backend, arguments: np.ndarray) -> dict:
    """C, S, f, g and the derivatives of f and g at each argument, as host arrays.

    f, g and their derivatives are taken at |x| and hold above the seam only.
    """
    xp = backend.xp
    outputs = {name: xp.zeros(arguments.size) for name in FUNCTION_NAMES}
    _fresnel_wrap(backend).compute_fresnel_functions(
        *outputs.values(), xp.asarray(arguments, dtype=np.float64), arguments.size
    )
    return {name: _to_host(value) for name, value in outputs.items()}


def phase_kernel_moments(backend, cases=PHASE_KERNEL_CASES):
    """Phase kernel and moment of each case over [-HALF_DURATION, HALF_DURATION], host arrays.

    kernel = sqrt(2 |alpha|) int exp(i psi) dtau and moment = the same with a tau weight, where
    psi = pi alpha tau^2 + 2 pi detuning tau.
    """
    xp = backend.xp
    num = len(cases)
    chirp_rates = np.array([chirp_rate for chirp_rate, _ in cases])
    detunings = np.array([detuning for _, detuning in cases])
    kernel = xp.zeros(num, dtype=np.complex128)
    moment = xp.zeros(num, dtype=np.complex128)
    # * f_eff = 0 and f0 = detuning make f0 - f_eff the detuning exactly, and t_ref = t_origin = 0
    # * removes the phase that re-references the Fourier transform.
    _fresnel_wrap(backend).compute_phase_kernel_moments(
        kernel,
        moment,
        xp.zeros(num),
        xp.zeros(num),
        xp.asarray(detunings),
        xp.asarray(chirp_rates),
        xp.full(num, -HALF_DURATION),
        xp.full(num, HALF_DURATION),
        xp.zeros(num),
        num,
    )
    return _to_host(kernel), _to_host(moment)


# ---------------------------------------------------------------------------------------------
# Arguments and reference values
# ---------------------------------------------------------------------------------------------


def _rational_arguments(num: int) -> np.ndarray:
    """The rational branch, including the doubles next to both of its seams."""
    inner = np.linspace(SERIES_SEAM, ASYMPTOTIC_SEAM, num)[1:-1]
    edges = [np.nextafter(SERIES_SEAM, np.inf), np.nextafter(ASYMPTOTIC_SEAM, 0.0)]
    return np.sort(np.concatenate([inner, edges]))


def positive_integral_arguments() -> np.ndarray:
    """0 <= x <= 1000: every branch, with the seam itself in the series branch."""
    series = np.linspace(0.0, SERIES_SEAM, 4001)
    asymptotic = np.geomspace(ASYMPTOTIC_SEAM, 1000.0, 2001)
    return np.concatenate([series, _rational_arguments(6401), asymptotic])


def integral_arguments() -> np.ndarray:
    positive = positive_integral_arguments()
    return np.concatenate([-positive[::-1], positive])


def auxiliary_arguments() -> np.ndarray:
    """1.6 < x <= 1e8, where the phase kernels use f and g."""
    return np.concatenate([_rational_arguments(641), np.geomspace(ASYMPTOTIC_SEAM, 1.0e8, 241)])


def branch_masks(arguments: np.ndarray) -> dict:
    magnitude = np.abs(arguments)
    return {
        "series": magnitude <= SERIES_SEAM,
        "rational": (magnitude > SERIES_SEAM) & (magnitude < ASYMPTOTIC_SEAM),
        "asymptotic": magnitude >= ASYMPTOTIC_SEAM,
    }


def reference_auxiliary(argument: float) -> tuple:
    """f, g, f' and g' at x > 0 by quadrature of smooth, positive integrands (DLMF 7.7.10-11).

    With a = pi x^2 / 2 and I_p = int_0^inf exp(-s^2) s^p / (1 + s^4 / a^2) ds:
    f = (sqrt 2 / pi) I_0 / sqrt(a), g = (sqrt 2 / pi) I_2 / a^(3/2), f' = -pi x g and
    g' = -sqrt(2) x I_4 / a^(5/2), each integrand without cancellation.
    """
    scale = 0.5 * np.pi * argument * argument

    def moment_integral(power: int) -> float:
        # * exp(-144) ends the integrand at s = 12, far below the rounding of the result.
        return scipy.integrate.quad(
            lambda variable: np.exp(-variable * variable)
            * variable**power
            / (1.0 + variable**4 / (scale * scale)),
            0.0,
            12.0,
            epsabs=0.0,
            epsrel=2e-14,
            limit=400,
        )[0]

    aux_f = np.sqrt(2.0) / np.pi * moment_integral(0) / np.sqrt(scale)
    aux_g = np.sqrt(2.0) / np.pi * moment_integral(2) / scale**1.5
    slope_g = -np.sqrt(2.0) * argument * moment_integral(4) / scale**2.5
    return aux_f, aux_g, -np.pi * argument * aux_g, slope_g


def quadrature(lower_bound, upper_bound, detuning, chirp_rate, power):
    """int tau^power exp(i [2 pi detuning tau + pi alpha tau^2]) by adaptive quadrature.

    ! quad warns that rounding may limit the requested 1e-13; its own error estimates stay below
    ! 1.1e-14 of the normalisations in every case, far below the bounds, so the warning is muted.
    """

    def phase(tau):
        return 2.0 * np.pi * detuning * tau + np.pi * chirp_rate * tau * tau

    options = dict(limit=4000, epsabs=0.0, epsrel=1.0e-13)
    with warnings.catch_warnings():
        warnings.simplefilter("ignore", scipy.integrate.IntegrationWarning)
        real = scipy.integrate.quad(
            lambda tau: tau**power * np.cos(phase(tau)), lower_bound, upper_bound, **options
        )[0]
        imaginary = scipy.integrate.quad(
            lambda tau: tau**power * np.sin(phase(tau)), lower_bound, upper_bound, **options
        )[0]
    return real + 1j * imaginary


@functools.lru_cache(maxsize=None)
def integral_reference():
    arguments = integral_arguments()
    sine, cosine = scipy.special.fresnel(arguments)
    return arguments, cosine, sine


@functools.lru_cache(maxsize=None)
def auxiliary_reference():
    arguments = auxiliary_arguments()
    values = np.array([reference_auxiliary(argument) for argument in arguments])
    return arguments, {name: values[:, index] for index, name in enumerate(AUXILIARY_NAMES)}


@functools.lru_cache(maxsize=None)
def phase_kernel_reference():
    """DFFR's references: the integral and its moment, without the factor sqrt(2 |alpha|)."""
    kernels, moments = [], []
    for chirp_rate, detuning in PHASE_KERNEL_CASES:
        kernels.append(quadrature(-HALF_DURATION, HALF_DURATION, detuning, chirp_rate, 0))
        moments.append(quadrature(-HALF_DURATION, HALF_DURATION, detuning, chirp_rate, 1))
    return np.array(kernels), np.array(moments)


# ---------------------------------------------------------------------------------------------
# Measured errors, shared with scripts/dffr/fresnel_unit_numbers.py
# ---------------------------------------------------------------------------------------------


def integral_errors(backend) -> dict:
    """{branch: (largest |C - C_ref|, largest |S - S_ref|)}."""
    arguments, cosine_reference, sine_reference = integral_reference()
    values = fresnel_functions(backend, arguments)
    errors = {}
    for branch, mask in branch_masks(arguments).items():
        errors[branch] = (
            float(np.abs(values["cosine"][mask] - cosine_reference[mask]).max()),
            float(np.abs(values["sine"][mask] - sine_reference[mask]).max()),
        )
    return errors


def auxiliary_errors(backend) -> dict:
    """{(branch, function): largest error}, relative or absolute as AUXILIARY_BOUNDS says."""
    arguments, reference = auxiliary_reference()
    values = fresnel_functions(backend, arguments)
    masks = branch_masks(arguments)
    errors = {}
    for (branch, name), (_, kind) in AUXILIARY_BOUNDS.items():
        mask = masks[branch]
        difference = np.abs(values[name][mask] - reference[name][mask])
        if kind == "relative":
            difference = difference / np.abs(reference[name][mask])
        errors[(branch, name)] = float(difference.max())
    return errors


def phase_kernel_errors(backend) -> list:
    """Per case: (chirp rate, detuning, kernel error, moment error) in DFFR's normalisation.

    The kernel is divided by sqrt(2 |alpha|) back to the integral of DFFR's test; its error is then
    taken as a fraction of 2 HALF_DURATION, the moment's as a fraction of HALF_DURATION^2.
    """
    kernel, moment = phase_kernel_moments(backend)
    reference_kernel, reference_moment = phase_kernel_reference()
    rows = []
    for index, (chirp_rate, detuning) in enumerate(PHASE_KERNEL_CASES):
        root = np.sqrt(2.0 * abs(chirp_rate))
        kernel_error = abs(kernel[index] / root - reference_kernel[index]) / (2.0 * HALF_DURATION)
        moment_error = abs(moment[index] / root - reference_moment[index]) / HALF_DURATION**2
        rows.append((chirp_rate, detuning, float(kernel_error), float(moment_error)))
    return rows


def gpu_against_cpu_differences(gpu_backend) -> dict:
    """{name: largest difference}, the function differences as GPU_FUNCTION_BOUNDS says."""
    cpu_backend = lisatools.get_backend("cpu")
    arguments = np.concatenate([integral_arguments(), auxiliary_arguments()])
    gpu_values = fresnel_functions(gpu_backend, arguments)
    cpu_values = fresnel_functions(cpu_backend, arguments)
    above_seam = np.abs(arguments) > SERIES_SEAM
    differences = {}
    for name, (_, kind) in GPU_FUNCTION_BOUNDS.items():
        mask = above_seam if name in AUXILIARY_NAMES else np.ones(arguments.size, dtype=bool)
        difference = np.abs(gpu_values[name][mask] - cpu_values[name][mask])
        if kind == "relative":
            difference = difference / np.abs(cpu_values[name][mask])
        differences[name] = float(difference.max())
    gpu_kernel, gpu_moment = phase_kernel_moments(gpu_backend)
    cpu_kernel, cpu_moment = phase_kernel_moments(cpu_backend)
    roots = np.sqrt(2.0 * np.abs([chirp_rate for chirp_rate, _ in PHASE_KERNEL_CASES]))
    differences["phase_kernel"] = float(
        (np.abs(gpu_kernel - cpu_kernel) / roots).max() / (2.0 * HALF_DURATION)
    )
    differences["phase_moment"] = float(
        (np.abs(gpu_moment - cpu_moment) / roots).max() / HALF_DURATION**2
    )
    return differences


# ---------------------------------------------------------------------------------------------
# Tests
# ---------------------------------------------------------------------------------------------


class FresnelFunctionsTest(unittest.TestCase):
    """The shared Fresnel functions against reference values, CPU backend."""

    backend_name = "cpu"

    @classmethod
    def setUpClass(cls):
        cls.backend = lisatools.get_backend(cls.backend_name)

    def test_integrals_match_scipy(self):
        """C and S against scipy.special.fresnel, one bound per branch."""
        for branch, (cosine_error, sine_error) in integral_errors(self.backend).items():
            cosine_bound, sine_bound = INTEGRAL_BOUNDS[branch]
            with self.subTest(branch=branch):
                self.assertLessEqual(cosine_error, cosine_bound, msg=f"C: {cosine_error:.3e}")
                self.assertLessEqual(sine_error, sine_bound, msg=f"S: {sine_error:.3e}")

    def test_integrals_are_odd(self):
        """C(-x) = -C(x) and S(-x) = -S(x), bit for bit."""
        positive = positive_integral_arguments()
        values = fresnel_functions(self.backend, positive)
        mirrored = fresnel_functions(self.backend, -positive)
        np.testing.assert_array_equal(mirrored["cosine"], -values["cosine"])
        np.testing.assert_array_equal(mirrored["sine"], -values["sine"])

    def test_auxiliary_functions_match_their_integrals(self):
        """f, g and their derivatives against quadrature, one bound per branch and function."""
        errors = auxiliary_errors(self.backend)
        for key, (bound, kind) in AUXILIARY_BOUNDS.items():
            with self.subTest(branch=key[0], function=key[1]):
                self.assertLessEqual(errors[key], bound, msg=f"{kind} error {errors[key]:.3e}")

    def test_phase_kernel_matches_quadrature(self):
        """The single-interval kernel and its moment on DFFR's cases, with DFFR's bounds."""
        for chirp_rate, detuning, kernel_error, moment_error in phase_kernel_errors(self.backend):
            with self.subTest(chirp_rate=chirp_rate, detuning_bins=detuning / FREQUENCY_BIN):
                self.assertLess(kernel_error, KERNEL_BOUND, msg=f"kernel: {kernel_error:.3e}")
                self.assertLess(moment_error, MOMENT_BOUND, msg=f"moment: {moment_error:.3e}")


@unittest.skipUnless(_cuda_backend_name(), "requires a usable CUDA backend")
class FresnelFunctionsGPUTest(FresnelFunctionsTest):
    """The same tests on the CUDA backend, and the CUDA values against the CPU values."""

    backend_name = _cuda_backend_name() or "cpu"

    def test_gpu_matches_cpu(self):
        """Every function and the phase kernel agree within their rounding bounds."""
        differences = gpu_against_cpu_differences(self.backend)
        for name, (bound, kind) in GPU_FUNCTION_BOUNDS.items():
            with self.subTest(function=name):
                self.assertLessEqual(
                    differences[name], bound, msg=f"{kind}: {differences[name]:.3e}"
                )
        for name in ("phase_kernel", "phase_moment"):
            with self.subTest(function=name):
                self.assertLessEqual(differences[name], GPU_PHASE_KERNEL_BOUND)


if __name__ == "__main__":
    unittest.main()
