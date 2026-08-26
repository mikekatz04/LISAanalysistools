"""Focused tests for the native Phentax TDI-on-the-fly wrapper."""

from __future__ import annotations

import numpy as np

from lisatools.sources.bbh.waveform import PhenomTHMTDIOnFlyWaveform
from lisatools.sources.waveformbase import TDTDIOnFlyWaveformBase


class _FakePhentax:
    def __init__(self, times, mask, amplitude, phase):
        self.outputs = (times, mask, amplitude, phase)
        self.kwargs = None

    def compute_strain_components_amp_phase(self, *args, **kwargs):
        self.kwargs = kwargs
        return self.outputs


class _AmpPhaseHarness:
    xp = np
    dt = 2.5

    def __init__(self, waveform):
        self.waveform = waveform
        self.trim_dt = None

    @staticmethod
    def _to_jax(value):
        return np.asarray(value)

    @staticmethod
    def _from_jax(value, do_synchronize=False):
        return np.asarray(value)

    @staticmethod
    def get_reference_quantities(**kwargs):
        return {"t_ref": np.array([-10.0]), "f_min": 7.0e-5}

    def trim_and_shift_times(self, times, mask, *, xp, dt):
        self.trim_dt = dt
        return np.asarray(times)


def test_native_phentax_forwards_runtime_grid_and_does_not_ramp_modes():
    times = np.array([[-3.0, -2.0, -1.0, 0.0]])
    mask = np.ones_like(times, dtype=bool)
    amplitude = np.arange(1.0, 9.0).reshape(1, 2, 4)
    phase = np.arange(11.0, 19.0).reshape(1, 2, 4)
    fake = _FakePhentax(times, mask, amplitude, phase)
    harness = _AmpPhaseHarness(fake)

    out_times, out_amplitude, out_phase = (
        PhenomTHMTDIOnFlyWaveform.get_amp_phase(
            harness,
            1.0,
            2.0,
            0.1,
            -0.2,
            100.0,
            0.3,
            0.4,
            0.5,
            merger_time=10.0,
            delta_t=0.1,
            t_min=-123456.0,
        )
    )

    assert fake.kwargs["delta_t"] == 0.1
    assert fake.kwargs["t_min"] == -123456.0
    assert harness.trim_dt == 0.1
    np.testing.assert_array_equal(out_times, times)
    np.testing.assert_array_equal(out_amplitude, amplitude)
    np.testing.assert_array_equal(out_phase, phase)


class _EvaluationHarness:
    xp = np

    @staticmethod
    def get_tdi_buffers(delta_t):
        return 1, 1, float(delta_t[:, 0].min()), float(delta_t[:, -1].min())


def test_native_phentax_retains_full_unbuffered_adaptive_grid():
    # More than 2,000 samples after t=0 reproduces the case where the generic
    # tail-only policy discarded the merger completely.
    input_times = np.arange(-2.0, 2502.0)[None, :]
    result = PhenomTHMTDIOnFlyWaveform.get_evaluation_times(
        _EvaluationHarness(), input_times
    )

    np.testing.assert_array_equal(result, input_times[:, 1:])
    assert result.shape[-1] > 2_000
    assert np.any(result == 0.0)


def test_native_phentax_keeps_every_node_above_the_leading_buffer():
    """The trailing buffer must NOT be trimmed.

    ``get_tdi_buffers`` derives ``end_buffer`` from the FINAL node spacing,
    which on a coarse-grained MBHB grid is the dense post-merger spacing.  The
    trim therefore ran to ~1500 nodes and cut the evaluation grid 600 s below
    the last node, discarding live ringdown -- worth a full-band mismatch of
    1.65e-05 on mojito MBHB 0.  ``pad`` supplies the spline headroom the trim
    was meant to guarantee, so the grid must reach the last node.
    """

    # Coarse inspiral spacing, dense ringdown spacing: the asymmetry is what
    # made the trailing trim so much larger than the leading one.
    input_times = np.concatenate(
        [np.arange(-10_000.0, 0.0, 100.0), np.arange(0.0, 500.0, 0.5)]
    )[None, :]

    result = PhenomTHMTDIOnFlyWaveform.get_evaluation_times(
        _EvaluationHarness(), input_times
    )

    assert result[0, -1] == input_times[0, -1]
    assert result.shape[-1] == input_times.shape[-1] - 1


class _PaddingHarness:
    xp = np
    tdi_buffer_time = 600.0
    get_tdi_buffers = TDTDIOnFlyWaveformBase.get_tdi_buffers


def test_tdi_spline_padding_rounds_up_to_required_physical_support():
    # Neither edge spacing divides 600 s.  Flooring would provide only 370 s
    # on the left and 410 s on the right.
    times = np.array([[0.0, 370.0, 780.0]])
    amplitude = np.array([[1.0, 2.0, 3.0]])
    phase = np.array([[4.0, 5.0, 6.0]])

    padded_times, padded_amplitude, padded_phase = TDTDIOnFlyWaveformBase.pad(
        _PaddingHarness(), times, amplitude, phase
    )

    assert times[0, 0] - padded_times[0, 0] >= 600.0
    assert padded_times[0, -1] - times[0, -1] >= 600.0
    start = int(np.ceil(600.0 / 370.0))
    np.testing.assert_array_equal(
        padded_times[:, start : start + times.shape[-1]], times
    )
    np.testing.assert_array_equal(
        padded_amplitude[:, start : start + amplitude.shape[-1]], amplitude
    )
    np.testing.assert_array_equal(
        padded_phase[:, start : start + phase.shape[-1]], phase
    )
