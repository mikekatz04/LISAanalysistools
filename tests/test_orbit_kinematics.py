"""Tests of the five-point orbit derivatives and the kinematics table at STFT segment midpoints.

The derivative methods of the C++ ``Orbits`` class (``get_velocity``, ``get_acceleration``,
``get_ltt_rate``, ``get_ltt_acceleration``) have no Python API; they are reached through the
kinematics table, which evaluates them at the segment midpoints (receivers, light travel times)
and at the emission times of the links (senders).

* Quadratic positions and light travel times on two offset grids: the five-point derivatives of
  the linearly interpolated tables are exact up to rounding when the step is a whole number of
  grid cells.
* Orbit-like sinusoids against their analytic derivatives, with tolerances set by the
  interpolation and rounding errors.
* The table against a NumPy mirror of the same interpolation, stencils and link vectors, within
  a rounding bound per field.
* ``ESAOrbits``: receiver velocities against the velocities of the orbit file.
* Coverage errors, argument checks, the cache of ``STFTComputationGroup.cpp_kinematics``, and the
  CUDA backend against the CPU backend (skipped without a usable GPU).
"""

from __future__ import annotations

import unittest

import numpy as np

import lisatools
from lisatools.domaincomputation import KINEMATICS_TABLE_FIELDS

#: Link labels in the order of every lisatools orbit class; link ij is received by i, sent by j.
LINKS = np.array([12, 23, 31, 13, 32, 21], dtype=np.int32)
RECEIVERS = np.array([1, 2, 3, 1, 3, 2], dtype=np.int32)
SENDERS = np.array([2, 3, 1, 3, 2, 1], dtype=np.int32)

#: Target step of the derivatives in Detector.hpp (ORBIT_STENCIL_STEP) [s].
STENCIL_STEP = 2000.0
#: Bound on the light travel time used by the coverage check (Detector.cu) [s].
MAX_LIGHT_TRAVEL_TIME = 20.0
EPS = np.finfo(np.float64).eps

OMEGA = 2.0 * np.pi / 3.15581498e7  # ? annual angular frequency [rad/s]
ORBIT_RADIUS = 1.496e11  # [m]
# * Phase offsets that put the three spacecraft about 2.5e9 m apart on the circle.
SPACECRAFT_PHASES = np.array([0.0, 0.0167, 0.0334])

# * Two grids with different starts and spacings, as in L1Orbits, at absolute times near 2e9 s.
# * The light travel times start later and end later than the positions, so each grid limits
# * one end of the coverage.
POSITION_GRID = dict(t0=2.0e9 + 0.25, dt=500.0, num=2400)
LTT_GRID = dict(t0=2.0e9 + 0.25 + 850.5, dt=2.5, num=484_000)
STFT_GRID = dict(first_start=2.0e9 + 0.25 + 1.0e5, segment_time=86400.0, num_segments=10)

# ? Relative error budgets of the annual circle: (dt^2 / 8) Omega^2 = 1.2e-9 from interpolating
# ? node values on the 500 s grid; rounding of 1 AU over h^2 gives 7e-9 per unit of rounding in
# ? the acceleration and 3e-6 in the light-travel-time acceleration (8.3 s over 2.4e-15 s/s^2).
CIRCLE_RELATIVE_BUDGET = {
    "position_receiver": 2e-9,
    "position_sender": 2e-9,
    "velocity_receiver": 1e-8,
    "velocity_sender": 1e-8,
    "acceleration_receiver": 5e-8,
    "acceleration_sender": 5e-8,
    "light_travel_time": 1e-15,
    "light_travel_time_rate": 1e-9,
    "light_travel_time_acceleration": 2e-5,
    "unit_direction": 5e-8,
    "unit_direction_rate": 5e-8,
}
# ? (Omega Delta)^3 / 24 = 1.4e-6: derivative error of a cubic spline through the ESA file's
# ? position samples, Delta = 1.87 days apart.
ESA_VELOCITY_RELATIVE_BOUND = 2e-6


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


def _stencil_step(dt: float) -> float:
    """Mirror of orbit_stencil_step in Detector.cu."""
    cells = max(1.0, np.floor(STENCIL_STEP / dt + 0.5))
    return cells * dt


def _grid_times(grid: dict) -> np.ndarray:
    return grid["t0"] + grid["dt"] * np.arange(grid["num"])


def _midpoints(first_start: float, segment_time: float, num_segments: int) -> np.ndarray:
    first_midpoint = first_start + 0.5 * segment_time
    return first_midpoint + np.arange(num_segments, dtype=np.float64) * segment_time


# ---------------------------------------------------------------------------------------------
# Synthetic orbits
# ---------------------------------------------------------------------------------------------


class QuadraticModel:
    """Positions and light travel times that are quadratic in time."""

    def __init__(self, reference_time: float):
        rng = np.random.default_rng(11)
        self.reference_time = reference_time
        # * Orbit-like magnitudes: 1 AU, 30 km/s, 6e-3 m/s^2.
        self.offset = ORBIT_RADIUS * (1.0 + 0.01 * rng.standard_normal((3, 3)))
        self.rate = 3.0e4 * rng.standard_normal((3, 3))
        self.curvature = 3.0e-3 * rng.standard_normal((3, 3))
        self.ltt_offset = 8.3 + 0.01 * rng.standard_normal(6)
        self.ltt_rate = 1.0e-8 * rng.standard_normal(6)
        self.ltt_curvature = 1.0e-15 * rng.standard_normal(6)

    def position(self, time, spacecraft_index):
        tau = np.asarray(time) - self.reference_time
        return (
            self.offset[spacecraft_index]
            + self.rate[spacecraft_index] * tau[..., None]
            + self.curvature[spacecraft_index] * tau[..., None] ** 2
        )

    def velocity(self, time, spacecraft_index):
        tau = np.asarray(time) - self.reference_time
        return self.rate[spacecraft_index] + 2.0 * self.curvature[spacecraft_index] * tau[..., None]

    def acceleration(self, time, spacecraft_index):
        tau = np.asarray(time) - self.reference_time
        return np.broadcast_to(2.0 * self.curvature[spacecraft_index], tau.shape + (3,))

    def ltt(self, time, link_index):
        tau = np.asarray(time) - self.reference_time
        return (
            self.ltt_offset[link_index]
            + self.ltt_rate[link_index] * tau
            + self.ltt_curvature[link_index] * tau**2
        )

    def ltt_derivative(self, time, link_index):
        tau = np.asarray(time) - self.reference_time
        return self.ltt_rate[link_index] + 2.0 * self.ltt_curvature[link_index] * tau

    def ltt_second_derivative(self, time, link_index):
        return np.full(np.shape(time), 2.0 * self.ltt_curvature[link_index])


class CircularModel:
    """Three spacecraft on the annual circle, and light travel times that breathe yearly."""

    def __init__(self, reference_time: float):
        self.reference_time = reference_time
        self.ltt_phases = np.linspace(0.0, 2.0, 6)

    def _angle(self, time, spacecraft_index):
        return (
            OMEGA * (np.asarray(time) - self.reference_time) + SPACECRAFT_PHASES[spacecraft_index]
        )

    def position(self, time, spacecraft_index):
        angle = self._angle(time, spacecraft_index)
        return ORBIT_RADIUS * np.stack([np.cos(angle), np.sin(angle), 1e-3 * np.cos(angle)], -1)

    def velocity(self, time, spacecraft_index):
        angle = self._angle(time, spacecraft_index)
        return (
            ORBIT_RADIUS
            * OMEGA
            * np.stack([-np.sin(angle), np.cos(angle), -1e-3 * np.sin(angle)], -1)
        )

    def acceleration(self, time, spacecraft_index):
        return -(OMEGA**2) * self.position(time, spacecraft_index)

    def ltt(self, time, link_index):
        angle = OMEGA * (np.asarray(time) - self.reference_time) + self.ltt_phases[link_index]
        return 8.3 + 0.03 * np.sin(angle)

    def ltt_derivative(self, time, link_index):
        angle = OMEGA * (np.asarray(time) - self.reference_time) + self.ltt_phases[link_index]
        return 0.03 * OMEGA * np.cos(angle)

    def ltt_second_derivative(self, time, link_index):
        angle = OMEGA * (np.asarray(time) - self.reference_time) + self.ltt_phases[link_index]
        return -0.03 * OMEGA**2 * np.sin(angle)


def sample_tables(model, position_grid=POSITION_GRID, ltt_grid=LTT_GRID) -> dict:
    """Flat C++ table arrays of ``model`` on the two grids (normal vectors are unused: zeros)."""
    sc_times = _grid_times(position_grid)
    ltt_times = _grid_times(ltt_grid)
    positions = np.stack([model.position(sc_times, index) for index in range(3)], axis=1)
    light_travel_times = np.stack([model.ltt(ltt_times, index) for index in range(6)], axis=1)
    return dict(
        x_arr=np.ascontiguousarray(positions.reshape(-1)),
        ltt_arr=np.ascontiguousarray(light_travel_times.reshape(-1)),
        n_arr=np.zeros(position_grid["num"] * 18),
    )


def build_orbits_wrap(backend, tables: dict, position_grid=POSITION_GRID, ltt_grid=LTT_GRID):
    """An OrbitsWrap on ``backend`` plus the device arrays it points to (keep both alive)."""
    xp = backend.xp
    arrays = {name: xp.asarray(value) for name, value in tables.items()}
    arrays.update(links=xp.asarray(LINKS), sc_r=xp.asarray(RECEIVERS), sc_e=xp.asarray(SENDERS))
    wrap = backend.OrbitsWrap(
        position_grid["t0"],
        position_grid["dt"],
        position_grid["num"],
        ltt_grid["t0"],
        ltt_grid["dt"],
        ltt_grid["num"],
        arrays["n_arr"],
        arrays["ltt_arr"],
        arrays["x_arr"],
        arrays["links"],
        arrays["sc_r"],
        arrays["sc_e"],
        2.5e9,
    )
    return wrap, arrays


def build_table(backend, orbits_wrap, first_start, segment_time, num_segments) -> dict:
    """Fill a kinematics table directly through KinematicsTableWrap; host arrays (..., NT)."""
    xp = backend.xp
    arrays = {
        name: xp.zeros(int(np.prod(shape)) * num_segments)
        for name, shape in KINEMATICS_TABLE_FIELDS
    }
    table = backend.KinematicsTableWrap(
        first_start + 0.5 * segment_time, segment_time, num_segments, *arrays.values()
    )
    table.build(orbits_wrap)
    return {
        name: _to_host(arrays[name]).reshape(shape + (num_segments,))
        for name, shape in KINEMATICS_TABLE_FIELDS
    }


# ---------------------------------------------------------------------------------------------
# NumPy mirror of Detector.cu
# ---------------------------------------------------------------------------------------------


class NumpyOrbits:
    """The interpolation and five-point stencils of the C++ Orbits class, in NumPy."""

    def __init__(self, tables: dict, position_grid=POSITION_GRID, ltt_grid=LTT_GRID):
        self.positions = tables["x_arr"].reshape(position_grid["num"], 3, 3)
        self.light_travel_times = tables["ltt_arr"].reshape(ltt_grid["num"], 6)
        self.position_grid = position_grid
        self.ltt_grid = ltt_grid

    @staticmethod
    def _interpolate(table, time, grid):
        # ! Same operations, in the same order, as Orbits::get_window and Orbits::interpolate.
        window = ((time - grid["t0"]) / grid["dt"]).astype(np.int64)
        assert np.all((window >= 0) & (window < grid["num"] - 1)), "outside the table"
        down, up = table[window], table[window + 1]
        offset = time - (grid["t0"] + grid["dt"] * window)
        if table.ndim > 1:
            offset = offset[:, None]
        return ((up - down) / grid["dt"]) * offset + down

    @staticmethod
    def _centre(time, grid, step):
        lower = grid["t0"] + 2.0 * step
        upper = grid["t0"] + float(grid["num"] - 2) * grid["dt"] - 2.0 * step
        return np.clip(time, lower, upper)

    def position(self, time, spacecraft_index):
        return self._interpolate(self.positions[:, spacecraft_index], time, self.position_grid)

    def _position_samples(self, time, spacecraft_index):
        step = _stencil_step(self.position_grid["dt"])
        centre = self._centre(time, self.position_grid, step)
        samples = [
            self.position(centre + shift * step, spacecraft_index)
            for shift in (2.0, 1.0, 0.0, -1.0, -2.0)
        ]
        return step, samples

    def velocity(self, time, spacecraft_index):
        step, (plus_two, plus_one, _, minus_one, minus_two) = self._position_samples(
            time, spacecraft_index
        )
        rate_scale = 1.0 / (12.0 * step)
        return (-plus_two + 8.0 * plus_one - 8.0 * minus_one + minus_two) * rate_scale

    def acceleration(self, time, spacecraft_index):
        step, (plus_two, plus_one, at_centre, minus_one, minus_two) = self._position_samples(
            time, spacecraft_index
        )
        acceleration_scale = 1.0 / (12.0 * step * step)
        return (
            -plus_two + 16.0 * plus_one - 30.0 * at_centre + 16.0 * minus_one - minus_two
        ) * acceleration_scale

    def ltt(self, time, link_index):
        return self._interpolate(self.light_travel_times[:, link_index], time, self.ltt_grid)

    def _ltt_samples(self, time, link_index):
        step = _stencil_step(self.ltt_grid["dt"])
        centre = self._centre(time, self.ltt_grid, step)
        samples = [
            self.ltt(centre + shift * step, link_index) for shift in (2.0, 1.0, 0.0, -1.0, -2.0)
        ]
        return step, samples

    def ltt_rate(self, time, link_index):
        step, (plus_two, plus_one, _, minus_one, minus_two) = self._ltt_samples(time, link_index)
        return (-plus_two + 8.0 * plus_one - 8.0 * minus_one + minus_two) / (12.0 * step)

    def ltt_acceleration(self, time, link_index):
        step, (plus_two, plus_one, at_centre, minus_one, minus_two) = self._ltt_samples(
            time, link_index
        )
        return (-plus_two + 16.0 * plus_one - 30.0 * at_centre + 16.0 * minus_one - minus_two) / (
            12.0 * step * step
        )


def kinematics_table(model, midpoints) -> dict:
    """Kinematics table fields of ``model`` (analytic model or NumpyOrbits) at ``midpoints``.

    Analytic models give the exact table; NumpyOrbits mirrors the three C++ stages.
    """
    is_mirror = isinstance(model, NumpyOrbits)
    ltt = np.stack([model.ltt(midpoints, link) for link in range(6)])
    if is_mirror:
        ltt_rate = np.stack([model.ltt_rate(midpoints, link) for link in range(6)])
        ltt_acceleration = np.stack([model.ltt_acceleration(midpoints, link) for link in range(6)])
    else:
        ltt_rate = np.stack([model.ltt_derivative(midpoints, link) for link in range(6)])
        ltt_acceleration = np.stack(
            [model.ltt_second_derivative(midpoints, link) for link in range(6)]
        )

    def state(times, spacecraft_index):
        return (
            model.position(times, spacecraft_index),
            model.velocity(times, spacecraft_index),
            model.acceleration(times, spacecraft_index),
        )

    receiver = [state(midpoints, index) for index in range(3)]
    sender = [state(midpoints - ltt[link], SENDERS[link] - 1) for link in range(6)]
    fields = {
        "position_receiver": np.stack([item[0].T for item in receiver]),
        "velocity_receiver": np.stack([item[1].T for item in receiver]),
        "acceleration_receiver": np.stack([item[2].T for item in receiver]),
        "position_sender": np.stack([item[0].T for item in sender]),
        "velocity_sender": np.stack([item[1].T for item in sender]),
        "acceleration_sender": np.stack([item[2].T for item in sender]),
        "light_travel_time": ltt,
        "light_travel_time_rate": ltt_rate,
        "light_travel_time_acceleration": ltt_acceleration,
    }
    # * Stage three, in the operation order of kinematics_unit_direction_kernel.
    receiver_rows = RECEIVERS - 1
    separation = fields["position_receiver"][receiver_rows] - fields["position_sender"]
    separation_rate = (
        fields["velocity_receiver"][receiver_rows]
        - fields["velocity_sender"] * (1.0 - ltt_rate)[:, None, :]
    )
    distance = np.sqrt(
        separation[:, 0] * separation[:, 0]
        + separation[:, 1] * separation[:, 1]
        + separation[:, 2] * separation[:, 2]
    )[:, None, :]
    inverse_distance = 1.0 / distance
    unit_direction = separation * inverse_distance
    radial_rate = (
        unit_direction[:, 0] * separation_rate[:, 0]
        + unit_direction[:, 1] * separation_rate[:, 1]
        + unit_direction[:, 2] * separation_rate[:, 2]
    )[:, None, :]
    fields["unit_direction"] = unit_direction
    fields["unit_direction_rate"] = (
        separation_rate - unit_direction * radial_rate
    ) * inverse_distance
    return fields


def rounding_bounds(tables: dict, link_distance: float) -> dict:
    """Bound per field on the difference between two roundings of the same table.

    Each interpolated value carries a few units of rounding of the table entries; a stencil
    scales them by the sum of its coefficients over 12 h (rate) or 12 h^2 (acceleration).
    """
    ulps = 8.0  # ? a few roundings per value, and FMA contraction on either side
    position_scale = np.abs(tables["x_arr"]).max()
    ltt_scale = np.abs(tables["ltt_arr"]).max()
    step_sc = _stencil_step(POSITION_GRID["dt"])
    step_ltt = _stencil_step(LTT_GRID["dt"])
    position = ulps * EPS * position_scale
    velocity = position * 18.0 / (12.0 * step_sc)
    acceleration = position * 64.0 / (12.0 * step_sc**2)
    return {
        "position_receiver": position,
        "velocity_receiver": velocity,
        "acceleration_receiver": acceleration,
        "position_sender": position,
        "velocity_sender": velocity,
        "acceleration_sender": acceleration,
        "light_travel_time": ulps * EPS * ltt_scale,
        "light_travel_time_rate": ulps * EPS * ltt_scale * 18.0 / (12.0 * step_ltt),
        "light_travel_time_acceleration": ulps * EPS * ltt_scale * 64.0 / (12.0 * step_ltt**2),
        "unit_direction": 4.0 * position / link_distance,
        "unit_direction_rate": 4.0 * velocity / link_distance,
    }


# ---------------------------------------------------------------------------------------------
# Tests
# ---------------------------------------------------------------------------------------------


class KinematicsTableTest(unittest.TestCase):
    """The kinematics table and the derivative methods on synthetic orbits, CPU backend."""

    backend_name = "cpu"

    @classmethod
    def setUpClass(cls):
        cls.backend = lisatools.get_backend(cls.backend_name)
        cls.reference_time = POSITION_GRID["t0"] + 6.0e5
        cls.midpoints = _midpoints(**STFT_GRID)

    def _table_for(self, model):
        tables = sample_tables(model)
        wrap, arrays = build_orbits_wrap(self.backend, tables)
        table = build_table(self.backend, wrap, **STFT_GRID)
        return tables, table

    def test_quadratic_orbits_are_differentiated_exactly(self):
        """Quadratics: velocities, accelerations and light-travel-time rates exact to rounding."""
        model = QuadraticModel(self.reference_time)
        tables, table = self._table_for(model)
        expected = kinematics_table(model, self.midpoints)
        bounds = rounding_bounds(tables, link_distance=2.0e9)
        for name in (
            "velocity_receiver",
            "acceleration_receiver",
            "light_travel_time_rate",
            "light_travel_time_acceleration",
        ):
            error = np.abs(table[name] - expected[name]).max()
            self.assertLessEqual(error, bounds[name], msg=f"{name}: {error:.3e}")
        # * Senders sit at t - L(t) with the table's own L, which is interpolated, not exact.
        emission = self.midpoints[None, :] - table["light_travel_time"]
        for link in range(6):
            for name, derivative in (
                ("velocity_sender", model.velocity),
                ("acceleration_sender", model.acceleration),
            ):
                expected_sender = derivative(emission[link], SENDERS[link] - 1).T
                error = np.abs(table[name][link] - expected_sender).max()
                self.assertLessEqual(error, bounds[name], msg=f"{name} {link}: {error:.3e}")

    def test_circular_orbits_against_analytic_derivatives(self):
        """Annual circle: every field within its interpolation and rounding budget."""
        model = CircularModel(self.reference_time)
        _, table = self._table_for(model)
        expected = kinematics_table(model, self.midpoints)
        for name, budget in CIRCLE_RELATIVE_BUDGET.items():
            scale = np.abs(expected[name]).max()
            error = np.abs(table[name] - expected[name]).max() / scale
            self.assertLess(error, budget, msg=f"{name}: relative error {error:.3e}")

    def test_table_matches_numpy_mirror(self):
        """The C++ table against a NumPy copy of the same arithmetic, within a rounding bound."""
        model = CircularModel(self.reference_time)
        tables, table = self._table_for(model)
        mirror = kinematics_table(NumpyOrbits(tables), self.midpoints)
        bounds = rounding_bounds(tables, link_distance=2.0e9)
        for name, _ in KINEMATICS_TABLE_FIELDS:
            error = np.abs(table[name] - mirror[name]).max()
            self.assertLessEqual(error, bounds[name], msg=f"{name}: {error:.3e}")

    def test_coverage_errors_name_the_grid(self):
        """Midpoints whose stencils leave a grid raise ValueError naming that grid."""
        tables = sample_tables(QuadraticModel(self.reference_time))
        wrap, arrays = build_orbits_wrap(self.backend, tables)
        margin_sc = 2.0 * _stencil_step(POSITION_GRID["dt"])
        margin_ltt = 2.0 * _stencil_step(LTT_GRID["dt"])

        # * The earliest grid that fits; the light travel times start 850.5 s after the positions.
        # ? 1 ms of slack: at 2e9 s, adding and removing half a day can move a time by one ulp.
        earliest_start = (
            max(
                POSITION_GRID["t0"] + MAX_LIGHT_TRAVEL_TIME + margin_sc, LTT_GRID["t0"] + margin_ltt
            )
            - 0.5 * 86400.0
        )
        build_table(self.backend, wrap, earliest_start + 1e-3, 86400.0, 2)
        with self.assertRaisesRegex(ValueError, "light travel time grid"):
            build_table(self.backend, wrap, earliest_start - 1.0, 86400.0, 2)

        # * The latest grid that fits; the positions end before the light travel times.
        sc_end = POSITION_GRID["t0"] + (POSITION_GRID["num"] - 2) * POSITION_GRID["dt"]
        latest_start = sc_end - margin_sc - 0.5 * 86400.0
        build_table(self.backend, wrap, latest_start - 1e-3, 86400.0, 1)
        with self.assertRaisesRegex(ValueError, "position grid"):
            build_table(self.backend, wrap, latest_start + 1.0, 86400.0, 1)

    def test_wrap_rejects_bad_arrays(self):
        """Wrong lengths raise ValueError; arrays that would need a converted copy raise TypeError."""
        xp = self.backend.xp
        num = 3
        arrays = [xp.zeros(int(np.prod(shape)) * num) for _, shape in KINEMATICS_TABLE_FIELDS]
        self.backend.KinematicsTableWrap(1.0, 86400.0, num, *arrays)
        short = list(arrays)
        short[0] = xp.zeros(9 * num - 1)
        with self.assertRaisesRegex(ValueError, "position_receiver"):
            self.backend.KinematicsTableWrap(1.0, 86400.0, num, *short)
        single = list(arrays)
        single[0] = xp.zeros(9 * num, dtype=xp.float32)
        with self.assertRaises(TypeError):
            self.backend.KinematicsTableWrap(1.0, 86400.0, num, *single)
        strided = list(arrays)
        strided[0] = xp.zeros(18 * num)[::2]
        with self.assertRaises(TypeError):
            self.backend.KinematicsTableWrap(1.0, 86400.0, num, *strided)


@unittest.skipUnless(_cuda_backend_name(), "requires a usable CUDA backend")
class KinematicsTableGPUTest(KinematicsTableTest):
    """The same tests on the CUDA backend, and the CUDA table against the CPU table."""

    backend_name = _cuda_backend_name() or "cpu"

    def test_gpu_matches_cpu(self):
        """The CUDA and CPU tables agree within the rounding bound of each field."""
        model = CircularModel(self.reference_time)
        tables = sample_tables(model)
        cpu_backend = lisatools.get_backend("cpu")
        cpu_wrap, cpu_arrays = build_orbits_wrap(cpu_backend, tables)
        cpu_table = build_table(cpu_backend, cpu_wrap, **STFT_GRID)
        gpu_wrap, gpu_arrays = build_orbits_wrap(self.backend, tables)
        gpu_table = build_table(self.backend, gpu_wrap, **STFT_GRID)
        bounds = rounding_bounds(tables, link_distance=2.0e9)
        for name, _ in KINEMATICS_TABLE_FIELDS:
            error = np.abs(gpu_table[name] - cpu_table[name]).max()
            self.assertLessEqual(error, bounds[name], msg=f"{name}: {error:.3e}")


class ESAOrbitsKinematicsTest(unittest.TestCase):
    """The table on the ESA trailing orbits that lisatools ships, CPU backend."""

    def test_receiver_velocities_match_the_orbit_file(self):
        """Five-point velocities of the interpolated positions against the file velocities.

        Both come from cubic splines through the file samples, 1.87 days apart: one of the
        positions, differentiated, and one of the velocities. They differ by the spline error
        of a derivative, about (Omega dt_file)^3 / 24 = 1.4e-6 of the orbital speed.
        """
        from lisatools.detector import ESAOrbits

        backend = lisatools.get_backend("cpu")
        orbits = ESAOrbits(force_backend="cpu")
        cpp_orbits = orbits.pycppdetector
        first_start, segment_time, num_segments = 10.0 * 86400.0, 86400.0, 30
        table = build_table(backend, cpp_orbits, first_start, segment_time, num_segments)
        midpoints = _midpoints(first_start, segment_time, num_segments)
        for index in range(3):
            file_velocity = _to_host(orbits.get_vel(midpoints, index + 1)).T
            error = np.abs(table["velocity_receiver"][index] - file_velocity).max()
            relative_error = error / np.abs(file_velocity).max()
            self.assertLess(relative_error, ESA_VELOCITY_RELATIVE_BOUND, msg=f"sc {index + 1}")


class STFTComputationGroupKinematicsTest(unittest.TestCase):
    """STFTComputationGroup.cpp_kinematics builds once per orbits object and keeps the table."""

    def test_cache_and_fields(self):
        from lisatools.analysiscontainer import AnalysisContainer, AnalysisContainerArray
        from lisatools.detector import EqualArmlengthOrbits
        from lisatools.domains import STFTSettings
        from lisatools.sensitivity import XYZSensitivityBackend

        segment_time, num_times = 86400.0, 4
        settings = STFTSettings(
            t0=10.0 * segment_time,
            dt=segment_time,
            df=1.0 / segment_time,
            NT=num_times,
            NF=64,
            min_freq=20.0 / segment_time,
            max_freq=40.0 / segment_time,
            force_backend="cpu",
        )
        orbits = EqualArmlengthOrbits(force_backend="cpu")
        shape = (3, num_times, settings.NF_active)
        sensitivity = XYZSensitivityBackend(orbits=orbits, settings=settings, force_backend="cpu")
        sensitivity.sens_mat = np.zeros((3,) + shape, dtype=np.complex128)
        sensitivity.invC = np.ones((3,) + shape, dtype=np.complex128)
        sensitivity.detC = np.ones(shape[1:])
        sensitivity.channel_shape = (3, 3)
        container = AnalysisContainer(
            settings.associated_class(np.zeros(shape, dtype=np.complex128), settings), sensitivity
        )
        group = AnalysisContainerArray(
            [container], domain_group_kwargs=dict(tdi_type="XYZ")
        ).cpp_splits[0]

        cpp_orbits = orbits.pycppdetector
        table = group.cpp_kinematics(cpp_orbits)
        self.assertIs(group.cpp_kinematics(cpp_orbits), table)
        self.assertEqual(table.num_time_segments, num_times)
        self.assertEqual(table.first_segment_midpoint, 10.5 * segment_time)
        fields = group.kinematics_fields(cpp_orbits)
        direct = build_table(
            lisatools.get_backend("cpu"), cpp_orbits, 10.0 * segment_time, segment_time, num_times
        )
        for name, field_shape in KINEMATICS_TABLE_FIELDS:
            self.assertEqual(fields[name].shape, field_shape + (num_times,))
            np.testing.assert_array_equal(fields[name], direct[name], err_msg=name)

        # * Another orbits object, even with the same content, gets its own table.
        other = orbits.pycppdetector
        self.assertIsNot(group.cpp_kinematics(other), table)


if __name__ == "__main__":
    unittest.main()
