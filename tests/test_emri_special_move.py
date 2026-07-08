"""Equivalence tests for :class:`EMRISpecialMove`'s per-source likelihood path.

EMRI waveforms are per-source / scalar-float, so :class:`EMRISpecialMove`
overrides :meth:`_compute_like_chunk` to generate sources one at a time (numpy
floats) and concatenate the single-source templates into the batched array the
C++ likelihood consumes. These tests drive that production override against the
**real C++ FD / STFT kernels** (via the ``_ACSHost`` fixture from
``test_aca_cpp_likelihood_backend``) and assert it reproduces the existing
**batched** forwarder (``cpp_template_likelihood``) elementwise.

A fake per-source generator returns, for a scalar coordinate ``c == batch row b``,
the single-source template ``templ[b][None]`` (+ start_freqs / start_times),
mimicking the ``(1, ch, ...)`` leading-axis output of a real single-source
``__call__``. Encoding the global batch-row index in the coordinate lets the
per-source path land each template at the correct flat position regardless of
split routing or ``batch_size_per_gpu`` chunking.
"""

from __future__ import annotations

import unittest

import numpy as np

from lisatools.globalfit.moves.emrispecialmove import EMRISpecialMove

# Import the kernel-fixture module as a namespace (NOT `from ... import Test...`):
# that keeps its TestCase classes out of this module's namespace so unittest does
# not re-collect/re-run them here. Robust to both `unittest discover` (tests.<mod>)
# and in-directory (<mod>) layouts.
try:
    import tests.test_aca_cpp_likelihood_backend as _kernels
except ImportError:
    import test_aca_cpp_likelihood_backend as _kernels


class _FakeEMRIGen:
    """Per-source generator: ``__call__(scalar c) -> single-source template for row int(c)``."""

    def __init__(self, templ, sf, st=None):
        self.templ = templ  # (nb, ch, ...)
        self.sf = sf        # (nb,)
        self.st = st        # (nb,) or None
        self.xp = np

    def __call__(self, c, **kw):
        b = int(round(float(c)))
        sig = self.templ[b][None]  # (1, ch, ...) — single-source leading axis
        sfb = np.asarray([self.sf[b]], dtype=np.float64)
        if self.st is not None:
            return sig, sfb, np.asarray([self.st[b]], dtype=np.float64)
        return sig, sfb


def _make_move(host, gen, batch_size_per_gpu, run_threaded):
    """Production EMRISpecialMove bound to the host fixture (attributes set directly).

    Bypasses ``__init__`` because ``_ACSHost`` carries its own ``.acs`` attribute
    that the constructor's DCGA-unwrap would misread; we only need the handful of
    attributes the per-source likelihood path touches.
    """
    m = EMRISpecialMove.__new__(EMRISpecialMove)
    m.acs = host
    m._waveform_generators = [gen for _ in range(host.num_splits)]
    m.waveform_like_method = "__call__"
    m.waveform_like_kwargs = {}
    m._run_async = False
    m._run_threaded = run_threaded
    m.batch_size_per_gpu = batch_size_per_gpu
    return m


class TestEMRISpecialMoveFD(unittest.TestCase):
    """Per-source EMRISpecialMove == batched forwarder over the real FD kernel."""

    def test_matches_batched_forwarder(self):
        fixt = _kernels.TestFDForwarderRealKernel()
        host = fixt._make_host(num_acs=6, num_splits=3)
        data_index, templ, sf = fixt._batch(6, nb=20)  # tiled walker-ids (prev_logl)
        ref = host.cpp_template_likelihood(data_index, templ, sf, start_times=None)

        gen = _FakeEMRIGen(templ, sf, st=None)
        coords_in = np.arange(len(data_index), dtype=float).reshape(-1, 1)

        for B in (None, 1, 2, 7, 1000):
            for threaded in (False, True):
                with self.subTest(batch_size_per_gpu=B, run_threaded=threaded):
                    move = _make_move(host, gen, B, threaded)
                    out = move.compute_like(coords_in, data_index)
                    np.testing.assert_allclose(out, ref, rtol=1e-10, atol=1e-10)


class TestEMRISpecialMoveSTFT(unittest.TestCase):
    """Per-source EMRISpecialMove == batched forwarder over the real STFT kernel."""

    def test_matches_batched_forwarder(self):
        fixt = _kernels.TestSTFTForwarderRealKernel()
        host = fixt._make_host(num_acs=4, num_splits=2)
        data_index, templ, sf, st = fixt._batch(4, nb=14)
        ref = host.cpp_template_likelihood(data_index, templ, sf, start_times=st)

        gen = _FakeEMRIGen(templ, sf, st=st)
        coords_in = np.arange(len(data_index), dtype=float).reshape(-1, 1)

        for B in (None, 1, 2, 3, 1000):
            for threaded in (False, True):
                with self.subTest(batch_size_per_gpu=B, run_threaded=threaded):
                    move = _make_move(host, gen, B, threaded)
                    out = move.compute_like(coords_in, data_index)
                    np.testing.assert_allclose(out, ref, rtol=1e-10, atol=1e-10)


if __name__ == "__main__":
    unittest.main()
