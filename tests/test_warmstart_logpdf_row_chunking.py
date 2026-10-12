"""The windowed-mixture logpdf is chunked over ROWS as well as pairs.

9mo job 757 (2026-10-12) died in full_pe's first ``rj_warm_pe``:

    rj_warm_pe -> BandSorter.__init__ -> rj_prop.logpdf
      -> _mixture_logpdf -> t["win_hi"][kk]
    OutOfMemoryError: allocating 4,773,120,000 bytes
                      (allocated so far: 92,007,001,088)

The pair stage had been chunked since 2026-09-19, but the candidate gather
before it still materialised several ``(n, D)`` arrays in one go: ``n`` is
every GB slot of the walker (20,000 -- BandSorter scores them all, the
empty ones filled with fresh draws) and ``D`` the candidate-window overlap
depth of the production warm-start set (29,832 here), so each array was
4.77 GB. In the search stages the card had room; after the PE F-stat
refit it held ~70 GB and it did not.

Each row's density depends on that row alone, so slicing the coordinate
block is float-exact. That is what is tested here, on the REAL method:
bit identity at every chunk size including one row, plus a mutation guard
that counts the blocks (an unchunked logpdf gives the same numbers, so the
numbers alone cannot catch it).
"""

import unittest
from unittest import mock

import numpy as np

from lisatools.globalfit.warmstart import proposal as mod
from lisatools.globalfit.warmstart.proposal import WarmStartComponents

TOBS = 7776000.0  # 3 months [s]


def _make_cov(sigmas, corr_off=0.15):
    d = len(sigmas)
    C = (1.0 - corr_off) * np.eye(d) + corr_off * np.ones((d, d))
    D = np.diag(np.asarray(sigmas, dtype=float))
    return D @ C @ D


def _components(n_close=12, seed=0):
    """``n_close`` components inside ONE candidate window (so the overlap
    depth is > 1) plus two isolated ones."""
    rng = np.random.default_rng(seed)
    base = np.array([8.0, 2.0000, 0.60, 3.00, 0.30, 1.20, 4.00, 0.20, 0.05])
    means = []
    for k in range(n_close):
        m = base.copy()
        m[0] = 5.0 + k
        m[1] = 2.0 + 1.0e-4 * k          # 1e-4 mHz apart: all in one window
        m[3] = (m[3] + 0.5 * k) % (2.0 * np.pi)
        means.append(m)
    means.append(np.array([5.0, 5.0000, 0.70, 0.05, 0.00, 0.50, 5.50, 0.60, 0.20]))
    means.append(np.array([20.0, 8.0000, 0.40, 5.00, 0.70, 2.50, 0.30, -0.80, 0.00]))
    means = np.array(means)
    sig = [0.5, 2.0e-5, 0.01, 0.30, 0.05, 0.15, 0.05, 0.05, 0.02]
    covs = np.stack([_make_cov(sig) for _ in range(len(means))])
    p = rng.uniform(0.1, 0.9, size=len(means))
    return means, covs, p


class RowChunkingIsExactTest(unittest.TestCase):
    def setUp(self):
        means, covs, p = _components()
        self.ws = WarmStartComponents(means, covs, p, new_tobs=TOBS, seed=3)
        self.assertGreater(self.ws._overlap_depth, 1)
        rng = np.random.default_rng(11)
        draws = np.asarray(self.ws.rvs(size=150))
        far = rng.normal(size=(20, 9))
        far[:, 1] = 12.0                   # no candidate window: density 0
        self.x = np.vstack([draws, far])
        self.n_draw = draws.shape[0]

    def _budget_for(self, rows):
        return int(rows) * mod._LOGPDF_BYTES_PER_CANDIDATE * self.ws._overlap_depth

    def _logpdf_with_budget(self, budget):
        with mock.patch.object(mod, "_LOGPDF_MAX_BYTES", int(budget)):
            return np.asarray(self.ws.logpdf(self.x))

    def test_every_row_chunk_size_is_bit_identical_to_one_shot(self):
        ref = self._logpdf_with_budget(1 << 40)
        self.assertTrue(np.isfinite(ref[:self.n_draw]).all())
        self.assertTrue(np.isneginf(ref[self.n_draw:]).all())
        n = self.x.shape[0]
        for rows in (1, 2, 7, 64, n - 1, n, n + 1):
            got = self._logpdf_with_budget(self._budget_for(rows))
            np.testing.assert_array_equal(
                got, ref, err_msg=f"rows per chunk {rows} changed the result")

    def test_the_block_function_runs_once_per_chunk(self):
        """Mutation guard: count the blocks, since the numbers cannot tell."""
        orig = WarmStartComponents._mixture_logpdf_rows
        n = self.x.shape[0]
        for rows in (1, 7, 64, n - 1, n, n + 1):
            calls = []

            def _spy(self_, *a, **k):
                calls.append(1)
                return orig(self_, *a, **k)

            with mock.patch.object(mod, "_LOGPDF_MAX_BYTES", self._budget_for(rows)), \
                    mock.patch.object(WarmStartComponents, "_mixture_logpdf_rows", _spy):
                self.ws.logpdf(self.x)
            self.assertEqual(len(calls), -(-n // rows), f"rows per chunk {rows}")

    def test_single_row_input_still_works(self):
        with mock.patch.object(mod, "_LOGPDF_MAX_BYTES", self._budget_for(1)):
            one = np.asarray(self.ws.logpdf(self.x[:1]))
        self.assertEqual(one.shape, (1,))
        self.assertTrue(np.isfinite(one[0]))


class RowsPerChunkTest(unittest.TestCase):
    def test_the_job_757_allocation_is_chunked_within_budget(self):
        n, nbytes = 20_000, 4_773_120_000
        depth = nbytes // 8 // n                      # 29,832
        rows = WarmStartComponents._logpdf_rows_per_chunk(depth)
        self.assertGreaterEqual(rows, 1)
        self.assertLess(rows, n)                      # it really does chunk
        self.assertLessEqual(rows * mod._LOGPDF_BYTES_PER_CANDIDATE * depth,
                             mod._LOGPDF_MAX_BYTES)

    def test_at_least_one_row_however_small_the_budget_or_deep_the_set(self):
        for budget in (0, 1, 55, 56):
            self.assertEqual(
                WarmStartComponents._logpdf_rows_per_chunk(10 ** 9, budget=budget), 1)
        self.assertEqual(WarmStartComponents._logpdf_rows_per_chunk(0, budget=56), 1)

    def test_budget_scales_the_rows(self):
        per_row = mod._LOGPDF_BYTES_PER_CANDIDATE * 1000
        self.assertEqual(WarmStartComponents._logpdf_rows_per_chunk(1000, budget=8 * per_row), 8)
        self.assertEqual(WarmStartComponents._logpdf_rows_per_chunk(1000, budget=16 * per_row), 16)
        # a budget just short of a whole row count floors, never rounds up
        self.assertEqual(WarmStartComponents._logpdf_rows_per_chunk(1000, budget=8 * per_row - 1), 7)


if __name__ == "__main__":
    unittest.main()
