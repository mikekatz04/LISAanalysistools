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


class _FakeBookkeepingGen:
    """Deterministic per-source generator that logs which replica ran each source."""

    def __init__(self, replica_id, log):
        self.replica_id = replica_id
        self.log = log

    def __call__(self, c, **kw):
        self.log.append((int(round(float(c))), self.replica_id))
        return np.full(4, float(c) + 1.0)


class _FakeBookkeepingHost:
    """Minimal ACS stand-in for the cold-chain bookkeeping path.

    Contiguous entry -> split map (like ``AnalysisContainerArray.gpu_splits``),
    per-entry residual buffers, and an ``_loop_operation`` that mirrors the
    real one (per-split dispatch, optional threading, empty splits skipped).
    """

    def __init__(self, num_entries, num_splits):
        self.gpus = None  # exercise the CPU branch (no device contexts / frees)
        self.num_splits = num_splits
        self._splits = np.array_split(np.arange(num_entries), num_splits)
        self.residuals = np.zeros((num_entries, 4))

    def unpack_indices(self, data_index):
        data_index = np.asarray(data_index)
        positions = [np.where(np.isin(data_index, s))[0] for s in self._splits]
        intra = [data_index[p] for p in positions]
        return positions, intra, None

    def _loop_operation(
        self, operation, operation_args_per_split, positions_per_split=None,
        run_threaded=False, **kw,
    ):
        import concurrent.futures as cf

        live = [i for i in range(self.num_splits) if len(positions_per_split[i])]
        if run_threaded:
            with cf.ThreadPoolExecutor(max_workers=max(1, len(live))) as ex:
                futures = {i: ex.submit(operation, *operation_args_per_split[i]) for i in live}
                return [futures[i].result() if i in futures else None
                        for i in range(self.num_splits)]
        return [operation(*operation_args_per_split[i]) if i in live else None
                for i in range(self.num_splits)]

    def signal_operation(self, sign, templates, data_index=None):
        for t, di in zip(templates, np.asarray(data_index)):
            self.residuals[int(di)] += sign * t


class TestEMRIColdChainBookkeeping(unittest.TestCase):
    """Split-parallel `_apply_cold_chain_sources` (on `MultiGPUResidualAddRemoveMove`,
    inherited by the EMRI/MBH moves): correct routing, values, replica ownership."""

    def _run(self, run_threaded, move_cls=EMRISpecialMove, num_entries=7, num_splits=3):
        host = _FakeBookkeepingHost(num_entries, num_splits)
        log = []
        move = move_cls.__new__(move_cls)
        move.acs = host
        move._waveform_generators = [
            _FakeBookkeepingGen(i, log) for i in range(num_splits)
        ]
        move.waveform_gen_method = "__call__"
        move.waveform_gen_kwargs = {}
        move._run_threaded = run_threaded
        coords = np.arange(num_entries, dtype=float).reshape(-1, 1)

        move._apply_cold_chain_sources(coords, sign=+1)
        expected = np.stack([np.full(4, i + 1.0) for i in range(num_entries)])
        np.testing.assert_allclose(host.residuals, expected, rtol=0, atol=0)

        # every source generated exactly once, on the replica owning its split
        self.assertEqual(sorted(s for s, _ in log), list(range(num_entries)))
        owner = {int(e): k for k, s in enumerate(host._splits) for e in s}
        for source, replica in log:
            self.assertEqual(replica, owner[source])

        # subtracting the same sources restores a zero residual
        move._apply_cold_chain_sources(coords, sign=-1)
        np.testing.assert_allclose(host.residuals, 0.0, rtol=0, atol=0)

    def test_serial(self):
        self._run(run_threaded=False)

    def test_threaded(self):
        self._run(run_threaded=True)

    def test_parent_class(self):
        from lisatools.globalfit.moves.addremovemove import (
            MultiGPUResidualAddRemoveMove,
        )

        for threaded in (False, True):
            with self.subTest(run_threaded=threaded):
                self._run(run_threaded=threaded, move_cls=MultiGPUResidualAddRemoveMove)


if __name__ == "__main__":
    unittest.main()
