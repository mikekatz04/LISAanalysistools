"""``GB_ORTHO_CHECK``: the premise check must actually RUN on a GPU.

WHAT WENT WRONG. Across the whole 2026-08/09 v7 production run the check
logged 3,350 lines of

    [GB_ORTHO ...] premise check skipped: TypeError('Implicit conversion
    to a NumPy array is not allowed...')

and produced ZERO orthogonality measurements. The guard around it
downgrades any internal failure to a warning -- correct for a diagnostic,
which must never break the sampler -- but it discarded the TRACEBACK, so
the log said what happened and never where. That is what turned a
one-line bug into a run's worth of lost measurement.

HOW THIS TEST WORKS. The bug is unreachable on CPU: on numpy everything
converts implicitly and the check passes. So this file supplies an ``xp``
whose arrays behave like CuPy's in the one way that matters -- they raise
on implicit conversion to numpy and expose ``.get()`` -- and asserts the
check COMPLETES. Same technique as the fake-``xp`` ``factors`` test in
``test_gb_observable_basis_wiring``: reproduce the device semantics that
break, without a device.
"""
import os
import types
import unittest
from unittest import mock

import numpy as np

from lisatools.globalfit.moves.gbspecialstretch import (
    GBSpecialStretchMove,
    _ortho_boundary_pairs,
)

_CUPY_MSG = ("Implicit conversion to a NumPy array is not allowed. "
             "Please use `.get()` to construct a NumPy array explicitly.")


class _DeviceArray(np.ndarray):
    """A numpy array that refuses implicit numpy conversion, like CuPy.

    Two behaviours are what matter, and they are exactly the two that
    made the production failure invisible on CPU:

    * ``np.asarray(x)`` / ``float(x)`` / ``host[x]`` raise TypeError;
    * ``.get()`` returns a real host array.
    """

    def __array__(self, *a, **k):
        raise TypeError(_CUPY_MSG)

    def get(self):
        return np.asarray(self).view(np.ndarray) if False else \
            np.ndarray.__array__(self)


def _dev(a, dtype=None):
    return np.asarray(a, dtype=dtype).view(_DeviceArray)


class _XP:
    """numpy namespace whose constructors return ``_DeviceArray``."""

    def __getattr__(self, k):
        v = getattr(np, k)
        # Do NOT wrap types: np.int32 & friends are callable but are used
        # as DTYPES (``.astype(xp.int32)``), and wrapping them makes numpy
        # report "Cannot interpret <function> as a data type" -- a fake
        # bug that would masquerade as the real one.
        if isinstance(v, type) or not callable(v):
            return v

        def wrapped(*a, **kw):
            out = v(*a, **kw)
            return out.view(_DeviceArray) if isinstance(out, np.ndarray) else out
        return wrapped


class _Res:
    def __init__(self, n):
        self.hh_add = _dev(np.full(n, 4.0))
        self.hh_remove = _dev(np.full(n, 9.0))
        self.hh_cross = _dev(np.full(n, 0.6))


class _Engine:
    def __init__(self):
        self.calls = 0

    def get_swap_ll(self, *a, **kw):
        self.calls += 1
        params = a[1]
        return _Res(int(params.shape[0]))


NSRC = 40


def _sorter():
    """A stub BandSorter with DEVICE-resident arrays throughout."""
    rng = np.random.default_rng(0)
    f0 = np.sort(rng.uniform(6.0e-3, 6.4e-3, NSRC))
    band = np.repeat(np.arange(NSRC // 4), 4)          # 4 sources per band
    s = types.SimpleNamespace(
        inds=_dev(np.ones(NSRC, dtype=bool)),
        temp_inds=_dev(np.zeros(NSRC, dtype=np.int32)),
        walker_inds=_dev(np.zeros(NSRC, dtype=np.int32)),
        band_inds=_dev(band.astype(np.int32)),
        coords_in=_dev(np.column_stack(
            [np.full(NSRC, 1e-22), f0] + [np.zeros(NSRC)] * 7)),
        N_vals=_dev(np.full(NSRC, 128, dtype=np.int32)),
    )
    return s


def _move(engine=None):
    s = types.SimpleNamespace(
        xp=_XP(),
        name="rj_fstat_search",
        waveform_kwargs={},
        _likelihood_engine=engine if engine is not None else _Engine(),
    )
    return s


LOGGER = "lisatools.globalfit.moves.gbspecialstretch"
CHECK = GBSpecialStretchMove._run_ortho_premise_check


class BoundaryPairsTest(unittest.TestCase):
    """The helper already pulls to host; pin it so it stays that way."""

    def test_accepts_device_arrays(self):
        s = _sorter()
        i, j = _ortho_boundary_pairs(
            s.coords_in[:, 1], s.walker_inds, s.band_inds, s.inds,
            units=2, remainder=0, max_pairs=8)
        self.assertGreater(i.size, 0)
        self.assertEqual(i.dtype.kind, "i")


class PremiseCheckRunsTest(unittest.TestCase):
    """THE regression. It must MEASURE, not skip."""

    def _run(self):
        mv, sorter = _move(), _sorter()
        model = types.SimpleNamespace(analysis_container_arr=object())
        with mock.patch.dict(os.environ, {"GB_ORTHO_CHECK": "1",
                                          "GB_ORTHO_MAX_PAIRS": "8"}):
            with self.assertLogs(LOGGER, "INFO") as cm:
                CHECK(mv, model, sorter, 2, 0)
        return mv, "\n".join(cm.output)

    def test_it_does_not_skip_on_device_arrays(self):
        _, msg = self._run()
        self.assertNotIn("premise check skipped", msg,
                         "the check must RUN on device arrays, not skip")

    def test_it_reports_a_measured_overlap(self):
        mv, msg = self._run()
        self.assertIn("normalized overlap", msg)
        # hh_cross 0.6 / sqrt(4 * 9) = 0.1 by construction
        self.assertIn("1.000e-01", msg)
        self.assertEqual(mv._likelihood_engine.calls, 1)

    def test_it_warns_when_the_overlap_exceeds_the_tolerance(self):
        mv, sorter = _move(), _sorter()
        model = types.SimpleNamespace(analysis_container_arr=object())
        with mock.patch.dict(os.environ, {"GB_ORTHO_CHECK": "1",
                                          "GB_ORTHO_TOL": "1e-3"}):
            with self.assertLogs(LOGGER, "WARNING") as cm:
                CHECK(mv, model, sorter, 2, 0)
        self.assertIn("orthogonality premise is weak", "\n".join(cm.output))

    def test_disarmed_is_a_no_op(self):
        mv, sorter = _move(), _sorter()
        model = types.SimpleNamespace(analysis_container_arr=object())
        with mock.patch.dict(os.environ, {"GB_ORTHO_CHECK": "0"}):
            CHECK(mv, model, sorter, 2, 0)
        self.assertEqual(mv._likelihood_engine.calls, 0)


class FailureIsDiagnosableTest(unittest.TestCase):
    """A guard that hides its traceback costs runs, not minutes.

    The v7 log said WHAT failed and never WHERE. The check must stay
    non-fatal -- it is a diagnostic -- but the first failure has to carry
    a traceback.
    """

    def _boom_engine(self):
        class Boom:
            calls = 0

            def get_swap_ll(self, *a, **kw):
                raise RuntimeError("kaboom")
        return Boom()

    def test_a_failure_is_still_non_fatal(self):
        mv, sorter = _move(self._boom_engine()), _sorter()
        model = types.SimpleNamespace(analysis_container_arr=object())
        with mock.patch.dict(os.environ, {"GB_ORTHO_CHECK": "1"}):
            with self.assertLogs(LOGGER, "WARNING"):
                CHECK(mv, model, sorter, 2, 0)   # must not raise

    def test_the_first_failure_carries_a_traceback(self):
        mv, sorter = _move(self._boom_engine()), _sorter()
        model = types.SimpleNamespace(analysis_container_arr=object())
        with mock.patch.dict(os.environ, {"GB_ORTHO_CHECK": "1"}):
            with self.assertLogs(LOGGER, "WARNING") as cm:
                CHECK(mv, model, sorter, 2, 0)
        msg = "\n".join(cm.output)
        self.assertIn("Traceback", msg)
        self.assertIn("get_swap_ll", msg,
                      "the traceback must name the failing frame")

    def test_repeat_failures_do_not_spam_the_traceback(self):
        """3,350 tracebacks would be its own problem."""
        mv, sorter = _move(self._boom_engine()), _sorter()
        model = types.SimpleNamespace(analysis_container_arr=object())
        with mock.patch.dict(os.environ, {"GB_ORTHO_CHECK": "1"}):
            with self.assertLogs(LOGGER, "WARNING") as cm:
                for _ in range(4):
                    CHECK(mv, model, sorter, 2, 0)
        self.assertEqual("\n".join(cm.output).count("Traceback"), 1)


if __name__ == "__main__":
    unittest.main()
