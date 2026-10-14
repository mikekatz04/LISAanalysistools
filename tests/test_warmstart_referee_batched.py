"""Batched referee merge test (warmstart/match_referee.py, 2026-10-13).

The merge test used to regenerate BOTH centroid templates for every
same-island pair, one ``run_wave`` call per pair: 231,737 pairs -> 463,915
CPU waveforms, 38 of the 45 minutes of the 6mo -> 9mo warm-start build. The
batched path generates each component's template once per template length
and computes the phase-maximised overlaps in vectorized groups on either
backend. These tests pin it to the scalar reference ``Matcher.match`` and to
the per-pair template-length rule (a pair's templates share
``N = max(N_i, N_j)``, as the per-pair call did), and pin the leaf-row
window the member re-extraction reads.
"""
import os
import sys
import tempfile
import types
import unittest
from unittest import mock

import h5py
import numpy as np

from lisatools.globalfit.warmstart import match_referee as mr


class FakeGB:
    """Stands in for ``GBGPU`` in the referee: ``run_wave`` fills
    ``XYZf`` (m, 3, N) complex and ``start_inds`` (m,) int, one template
    per source, each depending on that source's parameters AND on ``N``
    (as FastGB's slow-part sampling does), independent of the batch."""

    xp = np

    def __init__(self):
        self.generated = 0

    def run_wave(self, amp, f0, fdot, fddot, phi0, iota, psi, lam, beta,
                 N=None, T=None, dt=None):
        amp, f0, phi0, psi = (np.atleast_1d(np.asarray(a, float))
                              for a in (amp, f0, phi0, psi))
        n = np.arange(N)[None, None, :]
        ch = np.arange(3)[None, :, None]
        self.XYZf = (amp[:, None, None] * 1e23) * np.exp(
            1j * (phi0[:, None, None] + ch * psi[:, None, None]
                  + n * (f0[:, None, None] * 1e3 + 1.0 / N)))
        # small integer offsets so cross-band pairs still overlap
        self.start_inds = 100 + (np.round(f0 * 1e7).astype(int) % 3)
        self.generated += len(amp)


class IdentityTransform:
    """The rows below are already physical (amp, f0 [Hz], fdot, fddot,
    phi0, iota, psi, lam, beta)."""

    def both_transforms(self, rows):
        return np.atleast_2d(np.asarray(rows, float)).copy()


def _rows():
    """12 sources, 4 per template-length band at Tobs = 180 d, oversample 2
    (gbgpu get_N: f0 < 1 mHz -> 64, 1-10 mHz -> 128, 10-30 mHz -> 512)."""
    f0 = np.array([0.40e-3, 0.41e-3, 0.42e-3, 0.43e-3,
                   3.00e-3, 3.01e-3, 3.02e-3, 3.03e-3,
                   15.0e-3, 15.1e-3, 15.2e-3, 15.3e-3])
    rng = np.random.default_rng(3)
    rows = np.zeros((12, 9))
    rows[:, 0] = 1e-24 * (1.0 + rng.random(12))
    rows[:, 1] = f0
    rows[:, 4] = rng.uniform(0, 2 * np.pi, 12)
    rows[:, 5] = rng.uniform(0.1, 3.0, 12)
    rows[:, 6] = rng.uniform(0, np.pi, 12)
    rows[:, 7] = rng.uniform(0, 2 * np.pi, 12)
    rows[:, 8] = rng.uniform(-1.0, 1.0, 12)
    return rows


TOBS = 15552000.0


def _matcher(chunk_bytes=None):
    kw = {} if chunk_bytes is None else {"chunk_bytes": chunk_bytes}
    mm = mr.Matcher(FakeGB(), TOBS, 10.0, 2, **kw)
    mm.tc = IdentityTransform()
    return mm


class BatchedOverlapTest(unittest.TestCase):
    """``batched_overlaps`` = ``Matcher.match`` for every pair."""

    def test_hand_checked_overlaps(self):
        T = np.zeros((4, 3, 4), complex)
        T[0, :, :] = [1, 1, 0, 0]
        T[1, :, :] = [1, 0, 0, 0]
        T[2, :, :] = [1, 1, 0, 0]
        kmin = np.array([10, 11, 10, 10])
        pairs = np.array([[0, 1], [1, 0], [0, 2], [0, 3]])
        got = mr.batched_overlaps(T, kmin, pairs)
        # [0,1]: shared bins 11..13 -> a [1,0,0], b [1,0,0] on 3 channels:
        # |num| 3, |a|^2 6, |b|^2 3 -> 3/sqrt(18); [1,0] the same pair
        # reversed; [0,2] identical -> 1; [0,3] a zero template -> 0.
        np.testing.assert_allclose(
            got, [3 / np.sqrt(18), 3 / np.sqrt(18), 1.0, 0.0], rtol=1e-14)

    def test_templates_without_shared_bins_match_zero(self):
        T = np.ones((3, 3, 4), complex)
        kmin = np.array([0, 4, 3])
        got = mr.batched_overlaps(T, kmin, np.array([[0, 1], [0, 2]]))
        # [0,1] disjoint -> 0; [0,2] share bin 3 only: |num| 3, norms 12/12
        np.testing.assert_allclose(got, [0.0, 3 / 12], rtol=1e-14)

    def test_equals_the_scalar_reference_for_every_pair(self):
        rng = np.random.default_rng(7)
        m, N = 40, 16
        T = rng.normal(size=(m, 3, N)) + 1j * rng.normal(size=(m, 3, N))
        kmin = rng.integers(100, 107, size=m)
        kmin[:3] = [0, 40, 1000]                      # far / disjoint ones
        ii, jj = np.triu_indices(m, 1)
        pairs = np.stack([ii, jj], axis=1)
        pairs = np.concatenate([pairs, pairs[::7, ::-1]])  # some reversed
        want = np.array([mr.Matcher.match(T[i], kmin[i], T[j], kmin[j])
                         for i, j in pairs])
        for max_pairs in (None, 3):
            with self.subTest(max_pairs=max_pairs):
                kw = {} if max_pairs is None else {"max_pairs": max_pairs}
                got = mr.batched_overlaps(T, kmin, pairs, **kw)
                np.testing.assert_allclose(got, want, rtol=1e-12, atol=1e-15)


class PairChunkTest(unittest.TestCase):

    def test_chunks_cover_each_pair_once_in_order_under_the_template_cap(self):
        pairs = np.array([[0, 1], [0, 2], [1, 2], [3, 4], [4, 5], [5, 6],
                          [2, 7]])
        chunks = mr.pair_chunks(pairs, max_templates=3)
        self.assertEqual(np.concatenate(chunks).tolist(), list(range(7)))
        for c in chunks:
            self.assertLessEqual(len(np.unique(pairs[c])), 3)
        # greedy over the unique templates: {0,1,2} | {3,4,5} | {5,6} | {2,7}
        self.assertEqual([c.tolist() for c in chunks],
                         [[0, 1, 2], [3, 4], [5], [6]])


class PairMatchesTest(unittest.TestCase):
    """``Matcher.pair_matches`` = the old per-pair ``waves`` + ``match``."""

    def setUp(self):
        from gbgpu.utils.utility import get_N

        self.rows = _rows()
        # precondition: three template lengths, as the docstring of _rows
        self.assertEqual(
            get_N(self.rows[:, 0], self.rows[:, 1], TOBS, oversample=2)
            .tolist(), [64] * 4 + [128] * 4 + [512] * 4)
        ii, jj = np.triu_indices(12, 1)
        self.pairs = np.stack([ii, jj], axis=1)
        legacy = _matcher()
        self.want = np.empty(len(self.pairs))
        for n, (i, j) in enumerate(self.pairs):
            wf, k = legacy.waves(self.rows[[i, j]])
            self.want[n] = mr.Matcher.match(wf[0], k[0], wf[1], k[1])

    def test_matches_the_per_pair_path(self):
        got = _matcher().pair_matches(self.rows, self.pairs)
        np.testing.assert_allclose(got, self.want, rtol=1e-12, atol=1e-15)

    def test_each_template_is_generated_once_per_template_length(self):
        mm = _matcher()
        mm.pair_matches(self.rows, self.pairs)
        # pairs inside the 64 band need its 4 rows at 64; pairs whose max is
        # 128 need the 8 rows of the two lower bands at 128; every pair with
        # a 512 source needs all 12 rows at 512 -> 4 + 8 + 12 templates (the
        # per-pair path generates 2 x 66 = 132)
        self.assertEqual(mm.gb.generated, 24)
        self.assertEqual(mm.n_wf, 24)

    def test_a_tiny_memory_budget_changes_nothing(self):
        mm = _matcher(chunk_bytes=1)            # every chunk at its minimum
        got = mm.pair_matches(self.rows, self.pairs)
        np.testing.assert_allclose(got, self.want, rtol=1e-12, atol=1e-15)


class LeafRowsWindowTest(unittest.TestCase):
    """The member re-extraction must read the rows the fit read."""

    def test_reads_the_last_k_written_rows_past_row_471(self):
        nit, nw, nl = 480, 2, 3
        with tempfile.TemporaryDirectory() as d:
            path = os.path.join(d, "store.h5")
            with h5py.File(path, "w") as f:
                g = f.create_group("global_fit")
                chain = np.zeros((nit, 1, 1, nw, nl, 9))
                chain[..., 1] = 1.0 + np.arange(nit)[:, None, None, None, None]
                g.create_dataset("chain/gb", data=chain)
                inds = np.zeros((nit, 1, 1, nw, nl), bool)
                inds[..., :2] = True
                g.create_dataset("inds/gb", data=inds)
                g.create_dataset("log_like", data=-np.ones((nit, 1, 1, nw)))
            X, sid = mr.load_leaf_rows(path, last_k=5)
        self.assertEqual(np.unique(sid // nw).tolist(),
                         [475, 476, 477, 478, 479])
        self.assertEqual(len(X), 5 * nw * 2)


class ResolveBackendTest(unittest.TestCase):

    def _with_cupy(self, module):
        return mock.patch.dict(sys.modules, {"cupy": module})

    def test_auto_uses_cuda_when_cupy_sees_a_device(self):
        cp = types.SimpleNamespace(cuda=types.SimpleNamespace(
            runtime=types.SimpleNamespace(getDeviceCount=lambda: 2)))
        with self._with_cupy(cp):
            self.assertEqual(mr.resolve_backend("auto"), "cuda")

    def test_auto_falls_back_to_cpu_without_a_device(self):
        cp = types.SimpleNamespace(cuda=types.SimpleNamespace(
            runtime=types.SimpleNamespace(getDeviceCount=lambda: 0)))
        with self._with_cupy(cp):
            self.assertEqual(mr.resolve_backend("auto"), "cpu")
        with self._with_cupy(None):             # import cupy -> ImportError
            self.assertEqual(mr.resolve_backend("auto"), "cpu")

    def test_explicit_names_pass_through(self):
        self.assertEqual(mr.resolve_backend("CPU"), "cpu")
        self.assertEqual(mr.resolve_backend("cuda12x"), "cuda12x")


if __name__ == "__main__":
    unittest.main()
